import torch 
import torch.nn as nn

import math

class OneHeadAttention(nn.Module):
    def __init__(self, d_model, n_heads):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.dk = d_model // n_heads
        #qkv 学习矩阵每个都是d_model*d_model,除以n_heads就是每个头
        self.wq = nn.Linear(d_model, self.dk)
        self.wk = nn.Linear(d_model, self.dk)
        self.wv = nn.Linear(d_model, self.dk)

    def forward(self,x):
        q = self.wq(x)
        k = self.wk(x)
        v = self.wv(x)
        qk = q @k.transpose(-2,-1)
        qk = qk / math.sqrt(self.dk)
        attn = nn.functional.softmax(qk,dim=-1)
        return attn @ v

class MultiHeadSelfAttention(nn.Module):
    def __init__(self,d_model,n_heads):
        super().__init__()
        self.heads = nn.ModuleList([OneHeadAttention(d_model,n_heads)  for _ in range(n_heads)])
        self.linear = nn.Linear(d_model,d_model)

    def forward(self,x):
        return self.linear(torch.cat([head(x) for head in self.heads],dim=-1))

class FFN(nn.Module):
    def __init__(self,d_model,hidden_size):
        '''
        ffn 的hidden_size 一般是d_model的4倍,d_model的维度就是emmbedding的输出维度
        '''
        super().__init__()
        self.linear1 = nn.Linear(d_model,hidden_size)
        self.linear2 = nn.Linear(hidden_size,d_model)
        self.relu = nn.ReLU()
    def forward(self,x):
        return self.linear2(self.relu(self.linear1(x)))


class TransformerBlock(nn.Module):
    def __init__(self,d_model,n_heads,hidden_size,dropout):
        super().__init__()
        self.multi_head_attention = MultiHeadSelfAttention(d_model=d_model,n_heads=n_heads)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = FFN(d_model,hidden_size)

    def forward(self,x):
        x = self.multi_head_attention(x) + x
        x = self.norm1(x)
        x = self.ffn(x) + x
        x = self.norm2(x)
        return x