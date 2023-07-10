import torch
from torch import nn
import torch.nn.functional as F
from _transformers.utils import embedding, EmbeddingLayer, Linear
from _transformers.decoder import Decoder
from _transformers.encoder import Encoder

class StyleTransformer(nn.Module):
    def __init__(self, vocab_size, num_style_embeds, num_enc_layers, num_dec_layers, model_dim, num_heads, dropout, max_allowed_length, padding_idx=99):
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
        )
        self.sos_token = torch.nn.Parameter(torch.randn(model_dim))

    def forward(self, src, tgt, inp_lengths, style, temperature, generate=False):
        batch_size = src.shape[0]
        max_seq_len = src.shape[1]
        
        assert max_seq_len <= self.allowed_seq_len, f"Max sequence length {max_seq_len} is greater than allowed sequence length {self.allowed_seq_len}"

        pos_idx = torch.arange(self.allowed_seq_len).unsqueeze(0).expand((batch_size, -1))
        pos_idx = pos_idx.to(src.device)

        src_mask = pos_idx[:, :max_seq_len] >= inp_lengths.unsqueeze(-1)
        src_mask = torch.cat((torch.zeros_like(src_mask[:, :1]), src_mask), dim=1).view(batch_size, 1, 1, max_seq_len+1)


        style_embedding = self.style_embeds(style).view(batch_size, 1, -1)
        enc = torch.cat((style_embedding, self.embed(src, pos_idx[:, :max_seq_len])), dim=1)
        memory=self.encoder(enc, src_mask)

        sos_token = self.sos_token.view(1, 1, -1).expand((batch_size, -1, -1))

        tgt_mask = torch.ones((self.allowed_seq_len, self.allowed_seq_len)).to(src_mask.device)
        tgt_mask = (tgt_mask.tril() == 0).view(1, 1, self.allowed_seq_len, self.allowed_seq_len)
        
        if not generate:
            max_dec_length = tgt.shape[1]
            dec_input = tgt[:, :-1]
            decoder_embeds = torch.cat((sos_token, self.embed(dec_input, pos_idx[:, :max_dec_length - 1])), dim=1)

            logits = self.decoder(
                decoder_embeds, 
                memory, 
                tgt_mask[:, :, :max_dec_length, :max_dec_length], 
                src_mask,
                temperature
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
                    temperature,
                    prev_states
                )
                logits.append(logit)
                next_token = self.embed(logit.argmax(-1), pos_idx[:, i:i+1])
                # next_token = torch.cat((next_token, self.embed(logit.argmax(-1), pos_idx[:, i:i+1])), dim=1)
            logits = torch.cat(logits, 1)
        return logits

class Discriminator(nn.Module):
    def __init__(
            self, 
        vocab_size, num_style_embeds, num_enc_layers, 
        model_dim, num_heads, dropout, max_allowed_length, num_classes, 
        padding_idx=99
    ):
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
        self.cls_token = nn.Parameter(torch.rand(model_dim))
        self.cls_head = Linear(model_dim, num_classes)

    def forward(self, src, input_lengths, style_ids=None):
        batch_size = src.shape[0]
        max_seq_len = src.shape[1]
        num_extra_tokens = 1 if style_ids is None else 2
        pos_ids = torch.arange(max_seq_len).unsqueeze(0).expand((batch_size, -1)).to(src.device)

        src_mask = pos_ids>=input_lengths.unsqueeze(1)

        for _ in range(num_extra_tokens):
            src_mask = torch.cat((torch.zeros_like(src_mask[:, :1]), src_mask), dim=1)
        src_mask = src_mask.view(batch_size, 1, 1, max_seq_len+num_extra_tokens)

        cls_token = self.cls_token.unsqueeze(0).expand((batch_size, 1, -1))

        if style_ids is not None:
            style_emb = self.style_embeds(style_ids).unsqueeze(1)
            enc = torch.cat((cls_token, style_emb), dim=1)
        enc = torch.cat((enc, self.embed(src, pos_ids[:, :max_seq_len])), dim=1)
        enc = self.encoder(enc, src_mask)
        logits = self.cls_head(enc[:, 0])

        return F.log_softmax(logits, -1)


# if __name__ == '__main__':
#     disc = Discriminator(100, 2, 4, 128, 4, 0.1, 300, 2)
#     src = torch.randint(0, 100, (2, 10))
#     inp_tokens = torch.randint(0, 200, size=(2, ))
#     style_ids = torch.tensor([0, 1])
#     a = disc(
#         src, inp_tokens, style_ids
#     )
#     print(a.shape)

# if __name__ == "__main__":
#     # dummy data
#     model = StyleTransformer(400, 300, 2, 4, 4, 128, 4, 0.1, 10, 1.0, 0)
#     # dec2 = Decoder2(4, 128, 100, 4, 0.1)
#     src = torch.randint(0, 100, (2, 10))
#     tgt = torch.randint(0, 100, (2, 10))
#     src_mask = torch.randint(0, 2, (2, 10)).float()
#     tgt_mask = torch.randint(0, 2, (2, 1)).float()
#     gen_text = model(
#         src, 
#         tgt, 
#         src_mask, 
#         torch.tensor([0, 1]), 
#         generate=True
#     )
#     print(gen_text)