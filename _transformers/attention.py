import torch
from torch import nn
from _transformers.utils import Linear

class SelfAttention(nn.Module):
    def __init__(self, head_dim) -> None:
        super().__init__()
        self.head_dim = head_dim

    def forward(self, q, k, v, mask=None):
        # q, k, v -> (B, NH, T, HD)
        wei = (q @ k.transpose(-2, -1) / self.head_dim ** (0.5)).to(q.device)
        if mask is not None:
            wei = wei.masked_fill(mask, float('-inf'))
        # wei -> (B, NH, T, T)
        attention_scores = torch.nn.functional.softmax(wei, dim=-1)
        attention_feats = attention_scores @ v
        # attetnion_feats -> (B, NH, T, HD)
        return attention_scores, attention_feats
    
class MultiHeadAttention(nn.Module):
    def __init__(self, hidden_dim, num_heads) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        self.h = self.num_heads
        self.layers = nn.ModuleList([
            Linear(hidden_dim, hidden_dim)
            for _ in range(num_heads)
        ])
        self.scaled_attention = SelfAttention(self.head_dim)
        self.fc = Linear(hidden_dim, hidden_dim)

    def forward(self, q, k, v, mask=None):
        
        batch_size = q.shape[0]
        # seq_length = q.shape[1]

        q = self.layers[0](q).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.layers[1](k).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.layers[2](v).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)

        # inputs[0] -> (B, NH, T, HD)
        _, attn_feats = self.scaled_attention(q, k, v, mask)
        attn_concat = attn_feats.transpose(1, 2).contiguous().view(batch_size, -1, self.head_dim * self.h)
        
        return self.fc(attn_concat)
        
# if __name__ == "__main__":
#     a = torch.randn(8, 16, 64)
#     mask = torch.tril(torch.ones(16, 16)).unsqueeze(0).unsqueeze(0)

#     mha = MultiHeadAttention(64, 4)
#     print(mha(a, a, a, mask).shape)