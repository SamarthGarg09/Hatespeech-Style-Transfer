import os
import time
import torch
from torch import nn
import numpy as np
from utils import tensor2text, detokenize
from evaluator import Evaluator

def get_length(tokens, eos_token):
    lengths = torch.cumsum((tokens==eos_token), dim=1) # after eos token (including it) every idx will be 1
    lengths = (lengths==0).long().sum(-1) + 1 # +1 for eos token
    return lengths

def concat_neg_and_pos_ids(batch, eos_idx, pad_idx, reverse=False):
    '''
    Batch is processed such that the negative and positive samples are concatenated
    and the style labels are also concatenated. for multi batch inputs, the shape 
    would be (batch_size*2, seq_len).
    '''
    batch_neg, batch_pos = batch['pos_input_ids'], batch['neg_input_ids']
    diff = batch_pos.size(1) - batch_neg.size(1)
    
    if diff<0:
        pad = torch.full_like(batch_neg[:, :-diff], pad_idx) 
        batch_pos = torch.cat((batch_pos, pad), dim=1)             
    else:
        pad = torch.full_like(batch_pos[:, :diff], pad_idx)
        batch_neg = torch.cat((batch_neg, pad), dim=1)

    pos_style = torch.ones_like(batch_pos[:, 0])
    neg_style = torch.zeros_like(batch_neg[:, 0])

    if reverse:
        batch_pos, batch_neg = batch_neg, batch_pos
        pos_style, neg_style = neg_style, pos_style

    tokens = torch.cat((batch_pos, batch_neg), dim=0) 
    styles = torch.cat((pos_style, neg_style), dim=0)
    inp_length = get_length(tokens, eos_idx)

    return tokens, styles, inp_length

def word_drop(x, l, drop_prob, pad_idx):
    '''
    All the dropped tokens are placed at the end of the sequence.
    '''
    if not drop_prob:
        return x
    
    rand = torch.rand(x.shape, dtype=torch.float32).to(x.device)
    pos_idx = torch.arange(x.shape[1]).unsqueeze(0).expand_as(x).to(x.device)
    token_mask = pos_idx < (l.unsqueeze(1) - 1) # -1 since its 0 indexed
    drop_mask = (rand<drop_prob) & token_mask
    x2 = x.clone()
    pos_idx.masked_fill_(drop_mask, x.size(1)-1) # replace with eos token 
    pos_idx = pos_idx.sort(1)[0]
    x2 = x2.gather(1, pos_idx)
    
    return x2

def f_step(
        config, vocab, tokenizer,
        style_model, disc_model, 
        style_optim,
        batch, word_drop_ratio, temperature,
        enable_cycle_reconstruction=True
):
    disc_model.eval()
    pad_idx, eos_idx = tokenizer.pad_token_id, tokenizer.eos_token_id
    batch_size = batch['pos_input_ids'].size(0)
    loss_fn = nn.NLLLoss(reduction='none')
    input_ids, style_ids, input_length = concat_neg_and_pos_ids(batch, eos_idx, pad_idx, reverse=False)
    reverse_style_ids = 1 - style_ids
    token_mask = (input_ids != pad_idx).float()
    style_optim.zero_grad()
    
    noise_inp_ids = word_drop(
        input_ids, input_length, config['inp_drop_prob']*word_drop_ratio, pad_idx
    )
    # Since there can be multiple tokens dropped and placed at the end of the sequence,
    # we can't add 1 instead we have to add no of tokens dropped and replaced with eos token.
    noise_inp_length = get_length(noise_inp_ids, eos_idx)


    self_log_probs = style_model(
        input_ids, input_ids, noise_inp_length, style_ids, 
        generate=False, temperature=temperature, differentiable_decode=False
    )
    # self reconstruction loss
    self_reconstruction_loss = loss_fn(self_log_probs.transpose(1, 2), input_ids) * token_mask
    self_reconstruction_loss = self_reconstruction_loss.sum()/batch_size
    self_reconstruction_loss *= config['self_reconstruction_loss_weight']

    self_reconstruction_loss.backward() #calculate gradients for reconstruction

    if not enable_cycle_reconstruction:
        style_optim.step()
        disc_model.train()
        # style_optim.zero_grad()
        return self_reconstruction_loss.item(), 0, 0
    
    gen_log_probs = style_model(
        input_ids,
        None,
        input_length,
        reverse_style_ids,
        temperature,
        generate=True,
        differentiable_decode=True
    ) #pos_sent + neg_style -> neg_sent; 0-pos_style, 1->neg_style

    gen_soft_tokens = gen_log_probs.exp()
    gen_soft_token_len = get_length(gen_soft_tokens.argmax(-1), eos_idx)

    cyclic_rec_log_probs = style_model(
        gen_soft_tokens,    
        input_ids, 
        gen_soft_token_len,
        style_ids,
        generate=False,
        temperature = temperature,
        differentiable_decode=False
    )

    cyclic_rec_loss = loss_fn(cyclic_rec_log_probs.transpose(1, 2), input_ids) * token_mask
    cyclic_rec_loss = cyclic_rec_loss.sum()/batch_size
    cyclic_rec_loss *= config['cyclic_rec_loss_weight']
    
    # adversarial loss
    adv_log_probs = disc_model(gen_soft_tokens, gen_soft_token_len, reverse_style_ids)
    if config['discriminator_method'] == 'Multi':
        adv_labels = reverse_style_ids + 1
    else:
        adv_labels = torch.ones_like(reverse_style_ids)
    adv_loss = loss_fn(adv_log_probs, adv_labels)
    adv_loss = adv_loss.sum() / batch_size
    adv_loss *= config['adv_factor']

    (cyclic_rec_loss + adv_loss).backward() # calc

    torch.nn.utils.clip_grad.clip_grad_norm_(style_model.parameters(), 5)
    style_optim.step()
    style_model.train()
    return self_reconstruction_loss.item(), cyclic_rec_loss.item(), adv_loss.item()

def disc_step(config, vocab, tokenizer, style_model, disc_model, batch, disc_optimizer, temperature):
    pad_idx = tokenizer.pad_token_id
    eos_idx = tokenizer.eos_token_id
    style_model.eval()
    loss_fn = nn.NLLLoss(reduction='none')

    input_ids, style_ids, input_length = concat_neg_and_pos_ids(batch, eos_idx, pad_idx)
    rev_style_ids = 1 - style_ids
    batch_size = input_ids.size(0)

    with torch.no_grad():
        raw_style_gen_log_probs = style_model(
            input_ids, 
            None,
            input_length,
            style_ids, 
            generate=True,
            temperature=temperature,
            differentiable_decode=True
        )
        rev_style_gen_log_probs = style_model(
            input_ids, 
            None,
            input_length, 
            rev_style_ids,
            generate=True,
            temperature=temperature,
            differentiable_decode=True
        )
    gen_soft_probs = raw_style_gen_log_probs.exp()
    raw_gen_length = get_length(gen_soft_probs.argmax(-1), eos_idx)

    rev_gen_soft_tokens = rev_style_gen_log_probs.exp()
    rev_style_length = get_length(rev_gen_soft_tokens.argmax(-1), eos_idx)

    if config['discriminator_method'] == 'conditional':
        real_log_probs = disc_model(input_ids, input_length, style_ids)
        fake_log_probs = disc_model(input_ids, input_length, rev_style_ids)
        log_probs = torch.cat((real_log_probs, fake_log_probs), 0)
        real_labels = torch.ones_like(style_ids)
        fake_labels = torch.zeros_like(rev_style_ids)
        labels = torch.cat((real_labels, fake_labels), 0)

        real_gen_log_probs = disc_model(gen_soft_probs, raw_gen_length, style_ids)
        fake_gen_log_probs = disc_model(rev_gen_soft_tokens, rev_style_length, rev_style_ids)
        gen_log_probs = torch.cat((real_gen_log_probs, fake_gen_log_probs), 0)
        real_gen_labels = torch.ones_like(style_ids)
        fake_gen_labels = torch.zeros_like(rev_style_ids)
        gen_labels = torch.cat((real_gen_labels, fake_gen_labels), 0)

    elif config['discriminator_method'] == "Multi":
        log_probs = disc_model(input_ids, input_length)
        labels = style_ids + 1
        raw_gen_log_probs = disc_model(gen_soft_probs, raw_gen_length)
        rev_gen_log_probs = disc_model(rev_gen_soft_tokens, rev_style_length)
        gen_log_probs = torch.cat((raw_gen_log_probs, rev_gen_log_probs), 0)
        raw_gen_labels = style_ids + 1
        rev_gen_labels = torch.zeros_like(rev_style_ids)
        gen_labels = torch.cat((raw_gen_labels, rev_gen_labels), 0)


    adv_log_probs = torch.cat((log_probs, gen_log_probs), 0)
    adv_labels = torch.cat((labels, gen_labels), 0)
    adv_loss = (loss_fn(adv_log_probs, adv_labels).sum()/batch_size)
    adv_loss.backward()

    disc_optimizer.zero_grad()
    torch.nn.utils.clip_grad.clip_grad_norm_(disc_model.parameters(), max_norm=5)
    disc_optimizer.step()
    # different from original position
    style_model.train()
    
    return adv_loss.item()
    
def train(config, vocab, tokenizer, style_model, disc_model, train_loader, val_loader, test_loader):
    style_optimizer = torch.optim.Adam(style_model.parameters(), lr=config['lr_style_model'], weight_decay=config['weight_decay'])
    disc_optimizer = torch.optim.Adam(disc_model.parameters(), lr=config['lr_disc_model'], weight_decay=config['weight_decay'])

    his_g_slf_loss, his_g_cyclic_loss, his_g_adv_loss, his_d_adv_loss = [], [], [], []
    
    global_step = 0
    style_model.train()
    disc_model.train()

    config['save_folder'] = os.path.join(config['save_path'], str(time.strftime('%b%d%H%M%S', time.localtime())))
    os.makedirs(config['save_folder'])
    os.makedirs(os.path.join(config['save_folder'], 'ckpts'))
    print('Save Path:', config['save_folder'])

    print('Style Model Pretraining......')
    for i, batch in enumerate(train_loader):
        batch = {k:v.to(eval(config['device'])) for k, v in batch.items()}
        if i>= config['F_pretrain_iter']:
            break
        slf_loss, cyc_loss, _ = f_step(
            config, vocab, tokenizer, style_model, 
            disc_model, style_optimizer, batch, 1.0, 1.0, False
        )
        his_g_slf_loss.append(slf_loss)
        his_g_cyclic_loss.append(cyc_loss)

        if (i+1) % 10 == 0:
            avg_g_slf_loss = np.mean(his_g_slf_loss)
            avg_g_cyclic_loss = np.mean(his_g_cyclic_loss)
            his_g_slf_loss, his_g_cyclic_loss = [], []
            print('[iter: {}] slf_loss:{:.4f}, rec_loss:{:.4f}'.format(i, avg_g_slf_loss, avg_g_cyclic_loss))

    print('Training start.....')
    # temp
    def calculate_temperature(temperature_config, step):
        temperature_config = eval(temperature_config)
        num = len(temperature_config)
        for i in range(num):
            t_a, s_a = temperature_config[i]
            if i == num - 1:
                return t_a
            t_b, s_b = temperature_config[i + 1]
            if s_a <= step < s_b:
                k = (step - s_a) / (s_b - s_a)
                temperature = (1 - k) * t_a + k * t_b
                return temperature
    
    batch_iters = iter(train_loader)
    while True:
        drop_decay = calculate_temperature(config['drop_rate_config'], global_step)
        temperature = calculate_temperature(config['temperature_config'], global_step)
        try:
            batch = next(batch_iters)
        except StopIteration:
            batch_iters = iter(train_loader)
            batch = next(batch_iters)

        for _ in range(config['iter_D']):
            try:
                batch = next(batch_iters)
            except StopIteration:
                batch_iters = iter(train_loader)
                batch = next(batch_iters)
            batch = {k:v.to(eval(config['device'])) for k, v in batch.items()}
            disc_adv_loss = disc_step(
                config, vocab, tokenizer, style_model, disc_model, batch, disc_optimizer, temperature
            )            
            his_d_adv_loss.append(disc_adv_loss)

        for _ in range(config['iter_F']):
            try:
                batch = next(batch_iters)
            except StopIteration:
                batch_iters = iter(train_loader)
                batch = next(batch_iters)
            batch = {k:v.to(eval(config['device'])) for k, v in batch.items()}
            gen_slf_loss, gen_cyc_loss, gen_adv_loss = \
            f_step(
                config, vocab, tokenizer, 
                style_model, disc_model,
                style_optimizer, batch, drop_decay, 
                temperature, True 
            )

            his_g_slf_loss.append(gen_slf_loss)
            his_g_cyclic_loss.append(gen_cyc_loss)
            his_g_adv_loss.append(gen_adv_loss)
        
        global_step += 1

        if global_step % config['log_steps'] == 0:
            avg_g_slf_loss = np.mean(his_g_slf_loss)
            avg_g_cyc_loss = np.mean(his_g_cyclic_loss)
            avg_g_adv_loss = np.mean(his_g_adv_loss)
            avg_d_adv_loss = np.mean(his_d_adv_loss)
            log_str = '[iter {}] Gen Avg self loss: {} | \
                Gen Avg Cyclic Loss: {} \
                Gen Avg Adv Loss: {}\
                Disc Avg Adv Loss: {}'
            print(log_str.format(
                global_step, avg_g_slf_loss, avg_g_cyc_loss, 
                avg_g_adv_loss, avg_d_adv_loss
            ))

        if global_step % config['eval_steps'] == 0:
            his_g_slf_loss, his_g_cyclic_loss, his_g_adv_loss, his_d_adv_loss = [], [], [], []
            
            # save model
            save_dir = os.path.join(config['save_folder'], 'ckpts')
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
            torch.save(style_model.state_dict(), os.path.join(save_dir, f'{global_step}_style_model.pth'))
            torch.save(disc_model.state_dict(), os.path.join(save_dir, f'{global_step}_disc_model.pth'))
            evaluate(config, vocab, tokenizer, style_model, test_loader, global_step, temperature)

def evaluate(config, vocab, tokenizer, style_model, test_loader, global_step, temperature):
    style_model.eval()
    vocab_size = len(vocab)
    eos_idx = tokenizer.eos_token_id

    def test(data_iter, pos_style, neg_style):
        pos_actual_text, pos_predicted_output, pos_rev_output = [], [], []
        neg_actual_text, neg_predicted_output, neg_rev_output = [], [], []
        device = eval(config['device'])
        for batch in data_iter:
            batch = {k:v.to(eval(config['device'])) for k, v in batch.items()}
            pos_input_ids = batch['pos_input_ids'].to(device)
            neg_input_ids = batch['neg_input_ids'].to(device)

            pos_input_length = get_length(pos_input_ids, eos_idx)
            neg_input_length = get_length(neg_input_ids, eos_idx)

            pos_styles = torch.full_like(pos_input_ids[:, 0], pos_style).to(device)
            neg_styles = torch.full_like(pos_input_ids[:, 0], neg_style).to(device)
            rev_style_pos = 1 - pos_styles
            rev_style_neg = 1 - neg_styles

            with torch.no_grad():
                pos_gen_log_probs = style_model(
                    pos_input_ids, None, pos_input_length, pos_styles, generate=True, temperature=temperature, differentiable_decode=True
                )
                neg_gen_log_probs = style_model(
                    neg_input_ids, None, neg_input_length, neg_styles, generate=True, temperature=temperature, differentiable_decode=True
                )
                pos_rev_gen_log_probs = style_model(
                    pos_input_ids, None, pos_input_length, rev_style_pos, generate=True, temperature=temperature, differentiable_decode=True
                )
                neg_rev_gen_log_probs = style_model(
                    neg_input_ids, None, neg_input_length, rev_style_neg, generate=True, temperature=temperature, differentiable_decode=True
                )

            pos_actual_text += detokenize(tokenizer, vocab, pos_input_ids.cpu())
            neg_actual_text += detokenize(tokenizer, vocab, neg_input_ids.cpu())

            pos_predicted_output += detokenize(tokenizer, vocab, pos_gen_log_probs.argmax(-1).cpu())
            neg_predicted_output += detokenize(tokenizer, vocab, neg_gen_log_probs.argmax(-1).cpu())

            pos_rev_output += detokenize(tokenizer, vocab, pos_rev_gen_log_probs.argmax(-1).cpu())
            neg_rev_output += detokenize(tokenizer, vocab, neg_rev_gen_log_probs.argmax(-1).cpu())

        return zip((pos_actual_text, pos_predicted_output, pos_rev_output), (neg_actual_text, neg_predicted_output, neg_rev_output))

    actual_output, predicted_output, rev_output = test(test_loader, 1, 0)

    evaluator = Evaluator()
    ref_text = evaluator.yelp_ref

    acc_neg = evaluator.yelp_acc_0(rev_output[1])
    acc_pos = evaluator.yelp_acc_1(rev_output[0])
    bleu_neg = evaluator.yelp_ref_bleu_0(rev_output[1])
    bleu_pos = evaluator.yelp_ref_bleu_1(rev_output[0])
    # ppl_neg = evaluator.yelp_ppl(rev_output[1])
    # ppl_pos = evaluator.yelp_ppl(rev_output[0])

    for k in range(5):
        idx = np.random.randint(len(rev_output[0]))
        print('*' * 20, 'neg sample', '*' * 20)
        print('[gold]', actual_output[0][idx])
        print('[raw ]', predicted_output[0][idx])
        print('[rev ]', rev_output[0][idx])
        print('[ref ]', ref_text[0][idx])

    print('*' * 20, '********', '*' * 20)
    

    for k in range(5):
        idx = np.random.randint(len(rev_output[1]))
        print('*' * 20, 'pos sample', '*' * 20)
        print('[gold]', actual_output[1][idx])
        print('[raw ]', predicted_output[1][idx])
        print('[rev ]', rev_output[1][idx])
        print('[ref ]', ref_text[1][idx])

    print('*' * 20, '********', '*' * 20)

    print(('[auto_eval] acc_pos: {:.4f} acc_neg: {:.4f} ' + \
          'bleu_pos: {:.4f} bleu_neg: {:.4f} ').format(
              acc_pos, acc_neg, bleu_pos, bleu_neg,
    ))
    # save output
    save_file = config['save_folder'] + '/' + str(global_step) + '.txt'
    eval_log_file = config['save_folder'] + '/eval_log.txt'
    with open(eval_log_file, 'a') as fl:
        print(('iter{:5d}:  acc_pos: {:.4f} acc_neg: {:.4f} ' + \
               'bleu_pos: {:.4f} bleu_neg: {:.4f} ').format(
            global_step, acc_pos, acc_neg, bleu_pos, bleu_neg,
        ), file=fl)
    with open(save_file, 'w') as fw:
        print(('[auto_eval] acc_pos: {:.4f} acc_neg: {:.4f} ' + \
               'bleu_pos: {:.4f} bleu_neg: {:.4f} ').format(
            acc_pos, acc_neg, bleu_pos, bleu_neg,
        ), file=fw)

        for idx in range(len(rev_output[0])):
            print('*' * 20, 'neg sample', '*' * 20, file=fw)
            print('[gold]', actual_output[0][idx], file=fw)
            print('[raw ]', predicted_output[0][idx], file=fw)
            print('[rev ]', rev_output[0][idx], file=fw)
            print('[ref ]', ref_text[0][idx], file=fw)

        print('*' * 20, '********', '*' * 20, file=fw)

        for idx in range(len(rev_output[1])):
            print('*' * 20, 'pos sample', '*' * 20, file=fw)
            print('[gold]', actual_output[1][idx], file=fw)
            print('[raw ]', predicted_output[1][idx], file=fw)
            print('[rev ]', rev_output[1][idx], file=fw)
            print('[ref ]', ref_text[1][idx], file=fw)

        print('*' * 20, '********', '*' * 20, file=fw)
        
    style_model.train()