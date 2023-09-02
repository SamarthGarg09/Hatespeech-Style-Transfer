def tensor2text(tokenizer, vocab, tensor):
    tensor = tensor.cpu().numpy()
    text = []
    index2word = {v:k for k,v in vocab.items()}
    eos_idx = tokenizer.eos_token_id
    unk_idx = tokenizer.unk_token_id
    stop_idxs = [vocab['!'], vocab['.'], vocab['?']]
    for sample in tensor:
        sample_filtered = []
        prev_token = None
        for idx in list(sample):
            if prev_token in stop_idxs:
                break
            if idx == unk_idx or idx == prev_token or idx == eos_idx:
                continue
            prev_token = idx
            sample_filtered.append(index2word[idx])
            
        sample = ' '.join(sample_filtered)
        text.append(sample)

    return text

def detokenize(tokenizer, vocab, tensor):
    # 
    decoded_texts = tokenizer.batch_decode(tensor, skip_special_tokens=True)
    return decoded_texts