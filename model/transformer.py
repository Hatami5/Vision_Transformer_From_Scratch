import torch.nn as nn
import torch.nn.functional as F

class MLP(nn.Module):
    def __init__(self, embed_dim, mlp_dim, drop):
        super().__init__()
        self.fc1 = nn.Linear(embed_dim, mlp_dim)
        self.fc2 = nn.Linear(mlp_dim, embed_dim)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.drop(F.gelu(self.fc1(x)))
        return self.drop(self.fc2(x))


class TransformerEncoder(nn.Module):
    def __init__(self, embed_dim, num_heads, mlp_dim, drop):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = nn.MultiheadAttention(
            embed_dim, num_heads, dropout=drop, batch_first=True
        )
        self.norm2 = nn.LayerNorm(embed_dim)
        self.mlp = MLP(embed_dim, mlp_dim, drop)

    def forward(self, x):
        attn_out, _ = self.attn(self.norm1(x), self.norm1(x), self.norm1(x))
        x = x + attn_out
        x = x + self.mlp(self.norm2(x))
        return x
