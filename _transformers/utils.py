import torch
from torch import nn
import torch.nn.functional as F

def layer_norm(embed_dim, eps=1e-6):
    return nn.LayerNorm(embed_dim, eps=eps)

class NormAttnBlock(nn.Module):
    def __init__(self, d_model, dropout):
        super().__init__()
        self.layer_norm = layer_norm(d_model)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x, sublayer_fn):
        y = self.layer_norm(x)
        return x+self.dropout(sublayer_fn(y))
    
    def regressive_generate(self, x, sublayer_fn):
        y = sublayer_fn(self.layer_norm(x))
        return x[:, -1:] + self.dropout(y)
    
def embedding(num_embeds, embed_dim, pad_idx=0):
    embedding = nn.Embedding(num_embeds, embed_dim, padding_idx=pad_idx)
    nn.init.xavier_uniform_(embedding.weight)
    nn.init.constant_(embedding.weight[pad_idx], 0)
    return embedding

class EmbeddingLayer(nn.Module):
    def __init__(self, vocab_size, embed_dim=64, max_seq_length=512, padding_idx=99) -> None:
        super().__init__()
        self.token_embedding = embedding(vocab_size, embed_dim=embed_dim, pad_idx=padding_idx)
        self.position_embedding = embedding(max_seq_length, embed_dim=embed_dim, pad_idx=padding_idx)
        self.max_seq_length = max_seq_length
        
    def forward(self, x, pos):
        if x.ndim == 2:
            return self.token_embedding(x)+self.position_embedding(pos)
        else:
            # x->(B, T, E)  E.wei->(E, emb_dim) pos_emb->(B, T, emb_dim)
            return x@self.token_embedding.weight+self.position_embedding(pos)

class FeedForwardLayer(nn.Module):
    def __init__(self, hidden_dim, dropout_ratio) -> None:
        super().__init__()
        self.fc1 = Linear(hidden_dim, 4*hidden_dim)
        self.fc2 = Linear(4*hidden_dim, hidden_dim)
        self.drop = dropout_ratio

    def forward(self, x):
        x = F.dropout(F.gelu(self.fc1(x)), p=self.drop)
        x = self.fc2(x)
        return x
    
def Linear(in_features, out_features, bias=True, uniform=True):
    m = nn.Linear(in_features, out_features, bias)
    if uniform:
        nn.init.xavier_uniform_(m.weight)
    else:
        nn.init.xavier_normal_(m.weight)
    if bias:
        nn.init.constant_(m.bias, 0.)
    return m
