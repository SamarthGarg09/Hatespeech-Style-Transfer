import yaml
import torch
from data import load_data, get_tokenizer
from _transformers.models import StyleTransformer, Discriminator
from train import train
with open('/Data/deeksha/disha/code_p/style_transformer_repl/config.yml', 'r') as file:
    config = yaml.safe_load(file)

def load_model(config):
    style_model = StyleTransformer(
        len(vocab), 
        config['num_style_embeds'],
        config['num_enc_layers'],
        config['num_dec_layers'],
        config['d_model'],
        config['num_heads'],
        config['dropout'],
        config['max_length'],
        config['padding_idx']
    ).to(eval(config['device']))
    
    disc_model = Discriminator(
        len(vocab),
        config['num_style_embeds'],
        config['num_enc_layers'],
        config['d_model'],
        config['num_heads'],
        config['dropout'],
        config['max_length'],
        eval(config['num_classes']),
        config['padding_idx']
    ).to(eval(config['device']))
    # if config['load_from'] is not None:
    #     print("Loading models from checkpoint")
    #     style_model.load_state_dict(torch.load(config['style_model_ckpt']))
    #     disc_model.load_state_dict(torch.load(config['disc_model_ckpt']))
    return style_model, disc_model
        
if __name__ == "__main__":
    config = config['Config']
    tokenizer = get_tokenizer()
    vocab = tokenizer.get_vocab()
    train_dl, val_dl, test_dl = load_data(config)
    style_model, disc_model = load_model(config)
    train(
        config, vocab, tokenizer, style_model, disc_model, train_dl, val_dl, test_dl
    )
