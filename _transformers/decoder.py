import torch
from torch import nn
import torch.nn.functional as F
from _transformers.attention import MultiHeadAttention
from _transformers.encoder import FeedForwardLayer, Encoder
from _transformers.utils import layer_norm, NormAttnBlock, FeedForwardLayer, Linear

class Generator(nn.Module):
    def __init__(self, d_model, vocab_size):
        super(Generator, self).__init__()
        self.proj = Linear(d_model, vocab_size)

    def forward(self, x, temperature):
        return F.log_softmax(self.proj(x) / temperature, dim=-1)
    
class DecoderLayer(nn.Module):
    def __init__(self, hidden_dim, num_heads, dropout_ratio) -> None:
        super().__init__()
        self.masked_multi_head_attn = MultiHeadAttention(hidden_dim, num_heads)
        self.conditional_head_attn = MultiHeadAttention(hidden_dim, num_heads)
        self.ffn = FeedForwardLayer(hidden_dim, dropout_ratio)
        self.sublayers = nn.ModuleList([NormAttnBlock(hidden_dim, dropout_ratio) for _ in range(3)])

    def forward(self, x, conditional_encoding, tgt_mask, src_mask):
        # x.shape -> (1, 1, 128)
        y = self.sublayers[0](x, lambda x:self.masked_multi_head_attn(x, x, x, tgt_mask))
        y = self.sublayers[1](y, lambda y: self.conditional_head_attn(y, conditional_encoding, conditional_encoding, src_mask))
        y = self.sublayers[2](y, lambda y: self.ffn(y))
        return y
    
    def regressive_generate(self, x, memory, src_mask, tgt_mask, prev_states=None):
        new_states=[]
        m=memory

        x = torch.cat((prev_states[0], x), dim=1) if prev_states else x
        new_states.append(x)
        x = self.sublayers[0].regressive_generate(x, lambda x: self.masked_multi_head_attn(x[:, -1:], x, x, tgt_mask))

        # tgt_mask.shape?, x.shape?
        x = torch.cat((prev_states[1], x), dim=1) if prev_states else x
        new_states.append(x)
        x = self.sublayers[1].regressive_generate(x, lambda x: self.conditional_head_attn(x[:, -1:], m, m, src_mask))

        x = torch.cat((prev_states[2], x), dim=1) if prev_states else x
        new_states.append(x)
        x = self.sublayers[2].regressive_generate(x, lambda x:self.ffn(x[:, -1:]))

        return x, new_states

class Decoder(nn.Module):
    def __init__(self, vocab_size, hidden_dim, num_heads, num_layers, dropout_ratio, temperature=1.0) -> None:
        super().__init__()
        self.layers = nn.ModuleList([
            DecoderLayer(hidden_dim, num_heads, dropout_ratio)
            for _ in range(num_layers)
        ])
        self.norm = layer_norm(hidden_dim)
        self.generator = Generator(hidden_dim, vocab_size)

    def forward(self, x, conditional_encoding, tgt_mask, src_mask, temperature):
        for layer in self.layers:
            x = layer(x, conditional_encoding, tgt_mask, src_mask)
        x = self.norm(x)
        x = self.generator(x, temperature)
        return x
    
    def regressive_generate(self, x, memory, src_mask, tgt_mask, temperature, prev_states=None):
        y=x
        new_states=[]
        for i, layer in enumerate(self.layers):
            y, new_sub_state=layer.regressive_generate(y, memory, src_mask, tgt_mask, 
                                                    prev_states[i] if prev_states else None)
            new_states.append(new_sub_state)
        # y.shape?
        new_states.append(torch.cat((prev_states[-1], y), dim=1) if prev_states else y) 
        y = self.norm(new_states[-1])[:, -1:]
        return self.generator(y, temperature), new_states

if __name__ == "__main__":
    # dummy data
    enc = Encoder(100, 128, 4, 0.1)
    dec = Decoder(100, 128, 4, 4, 0.1)
    # dec2 = Decoder2(4, 128, 100, 4, 0.1)
    src = torch.randint(0, 100, (1, 10, 128)).float()
    tgt = torch.randint(0, 100, (1, 10, 128)).float()
    src_mask = torch.randint(0, 2, (1, 1)).float()
    tgt_mask = torch.randint(0, 2, (1, 1)).float()
    # memory = enc(src, src_mask)
    memory = torch.randn(1, 10, 128)
    for i in range(10):
        gen_dec = dec.regressive_generate(tgt, memory, src_mask, tgt_mask, 1.0)
        # gen_desc = dec2.incremental_forward(tgt, memory, src_mask, tgt_mask, 1.0)
        tgt = gen_dec[0]
        memory = torch.cat((memory, tgt), dim=1)

    print(gen_dec[0].shape)