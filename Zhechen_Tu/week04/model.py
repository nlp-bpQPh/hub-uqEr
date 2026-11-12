"""A small decoder-only Transformer for next-character prediction."""
from dataclasses import dataclass
import torch
from torch import nn
from torch.nn import functional as F

@dataclass
class Config:
    vocab_size: int
    context: int = 128
    width: int = 128
    heads: int = 4
    layers: int = 2
    dropout: float = 0.1

class Attention(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.heads, self.dropout = cfg.heads, cfg.dropout
        self.qkv = nn.Linear(cfg.width, 3 * cfg.width)
        self.proj = nn.Linear(cfg.width, cfg.width)
        self.residual_dropout = nn.Dropout(cfg.dropout)

    def forward(self, x):
        b, t, c = x.shape
        q, k, v = self.qkv(x).chunk(3, dim=-1)
        q, k, v = [a.reshape(b, t, self.heads, c // self.heads).transpose(1, 2) for a in (q, k, v)]
        # Position t can attend only to positions <= t, never future targets.
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True,
                dropout_p=self.dropout if self.training else 0.0)
        y = y.transpose(1, 2).contiguous().reshape(b, t, c)
        return self.residual_dropout(self.proj(y))

class Block(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.ln1, self.ln2 = nn.LayerNorm(cfg.width), nn.LayerNorm(cfg.width)
        self.attn = Attention(cfg)
        self.ffn = nn.Sequential(nn.Linear(cfg.width, 4 * cfg.width), nn.GELU(),
                                 nn.Linear(4 * cfg.width, cfg.width), nn.Dropout(cfg.dropout))

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        return x + self.ffn(self.ln2(x))

class LanguageModel(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        if cfg.width % cfg.heads:
            raise ValueError('width must be divisible by heads')
        self.cfg = cfg
        self.token = nn.Embedding(cfg.vocab_size, cfg.width)
        self.position = nn.Embedding(cfg.context, cfg.width)
        self.blocks = nn.Sequential(*(Block(cfg) for _ in range(cfg.layers)))
        self.norm = nn.LayerNorm(cfg.width)
        self.head = nn.Linear(cfg.width, cfg.vocab_size, bias=False)
        self.apply(self.initialize)

    @staticmethod
    def initialize(module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, mean=0., std=.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                nn.init.zeros_(module.bias)

    def forward(self, ids):
        if ids.ndim != 2 or not 0 < ids.shape[1] <= self.cfg.context:
            raise ValueError('Expected [batch, time] with 1 <= time <= context')
        x = self.token(ids) + self.position(torch.arange(ids.shape[1], device=ids.device))
        return self.head(self.norm(self.blocks(x)))
