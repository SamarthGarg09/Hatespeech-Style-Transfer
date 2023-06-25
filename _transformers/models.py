import torch
from torch import nn
from utils import embedding, EmbeddingLayer
from decoder import Decoder
from encoder import Encoder

class StyleTransformer(nn.Module):
    def __init__(self, vocab_size, embed_dim, num_style_embeds, num_enc_layers, num_dec_layers, model_dim, num_heads, dropout, max_allowed_length, temperature, padding_idx=99):
        super().__init__()
        self.allowed_seq_len = max_allowed_length
        self.style_embeds = embedding(num_style_embeds, model_dim)
        self.embed = EmbeddingLayer(vocab_size, model_dim, max_allowed_length, padding_idx=padding_idx)
        self.encoder = Encoder(
            num_enc_layers=num_enc_layers,
            hidden_dim=model_dim,
            num_heads=num_heads,
            dropout_ratio=dropout
        )
        self.decoder = Decoder(
            vocab_size=vocab_size,
            hidden_dim=model_dim,
            num_heads=num_heads,
            num_layers=num_dec_layers,
            dropout_ratio=dropout,
            temperature=temperature
        )
        self.sos_token = torch.nn.Parameter(torch.randn(model_dim))
        self.temperature = temperature

    def forward(self, src, tgt, src_mask, style, generate=False):
        batch_size = src_mask.shape[0]
        max_seq_len = src_mask.shape[1]
        
        pos_idx = torch.arange(self.allowed_seq_len).unsqueeze(0).expand((batch_size, -1))
        pos_idx = pos_idx.to(src.device)

        src_mask = (src_mask == 0).short()
        src_mask = torch.cat((torch.zeros_like(src_mask[:, :1]), src_mask), dim=-1).view(batch_size, 1, 1, max_seq_len+1)


        style_embedding = self.style_embeds(style).view(batch_size, 1, -1)
        enc = torch.cat((style_embedding, self.embed(src, pos_idx[:, :max_seq_len])), dim=1)
        memory=self.encoder(enc, src_mask)

        sos_token = self.sos_token.view(1, 1, -1).expand((batch_size, -1, -1))

        max_dec_length = tgt.shape[1]
        tgt_mask = torch.ones((self.allowed_seq_len, self.allowed_seq_len)).to(src_mask.device)
        tgt_mask = (tgt_mask.tril() == 0).view(1, 1, self.allowed_seq_len, self.allowed_seq_len)
        
        if not generate:
            dec_input = tgt[:, :-1]
            decoder_embeds = torch.cat((sos_token, self.embed(dec_input, pos_idx[:, :max_dec_length])), dim=1)

            logits = self.decoder(
                decoder_embeds, 
                memory, 
                tgt_mask, 
                src_mask
            )
        else:
            logits = []
            next_token = sos_token
            prev_states = None
            for i in range(self.allowed_seq_len):
                logit, prev_states = self.decoder.regressive_generate(
                    next_token,
                    memory,
                    src_mask,
                    tgt_mask[:, :, i:i+1, :i+1],
                    self.temperature,
                    prev_states
                )
                logits.append(logit)
                # next_token = torch.cat((next_token, self.embed(logit.argmax(-1), pos_idx[:, i:i+1])), dim=1)
            logits = torch.cat(logits, 1)
        return logits


if __name__ == "__main__":
    # dummy data
    model = StyleTransformer(400, 300, 2, 4, 4, 128, 4, 0.1, 10, 1.0, 0)
    # dec2 = Decoder2(4, 128, 100, 4, 0.1)
    src = torch.randint(0, 100, (2, 10))
    tgt = torch.randint(0, 100, (2, 10))
    src_mask = torch.randint(0, 2, (2, 10)).float()
    tgt_mask = torch.randint(0, 2, (2, 1)).float()
    gen_text = model(
        src, 
        tgt, 
        src_mask, 
        torch.tensor([0, 1]), 
        generate=True
    )
    print(gen_text)