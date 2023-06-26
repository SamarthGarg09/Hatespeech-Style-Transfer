import torch
from torch import nn
from attention import MultiHeadAttention
import torch.nn.functional as F
from utils import NormAttnBlock, layer_norm, FeedForwardLayer
    
class EncoderLayer(nn.Module):
    def __init__(self, hidden_dim, num_heads, dropout_ratio) -> None:
        super().__init__()
        self.sublayers = nn.ModuleList([NormAttnBlock(hidden_dim, dropout_ratio) for _ in range(2)])
        self.self_attention = MultiHeadAttention(hidden_dim, num_heads)
        self.ffn = FeedForwardLayer(hidden_dim, dropout_ratio)

    def forward(self, x, mask):
        x = self.sublayers[0](x, lambda x: self.self_attention(x, x, x, mask))
        x = self.sublayers[1](x, lambda x: self.ffn(x))
        return x

class Encoder(nn.Module):
    def __init__(self, num_enc_layers, hidden_dim, num_heads, dropout_ratio):
        super().__init__()
        self.layers = nn.ModuleList([
            EncoderLayer(hidden_dim, num_heads, dropout_ratio)
            for _ in range(num_enc_layers)
        ])
        self.norm = layer_norm(hidden_dim)

    def forward(self, x, mask):
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)

if __name__ == "__main__":
    a = torch.randn(8, 16, 64)
    mask = torch.tril(torch.ones(16, 16)).unsqueeze(0).unsqueeze(0)
    enc = Encoder(3, 64, 4, 0.1)
    print(enc(a, mask).shape)