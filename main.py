import yaml
import torch
from data import load_data, get_tokenizer
from _transformers.models import StyleTransformer, Discriminator
from train import train
# Open the YAML file
with open('config.yml', 'r') as file:
    config = yaml.safe_load(file)

if __name__ == "__main__":
    config = config['Config']
    tokenizer = get_tokenizer()
    vocab = tokenizer.get_vocab()
    train_dl, val_dl, test_dl = load_data(config)
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
    train(
        config, vocab, tokenizer, style_model, disc_model, train_dl, val_dl, test_dl
    )
