#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
File: LM.py
Author: Miku
Date: 2026-09-03
Description: transformer + mask + 策略采样 实现文本续写
"""

import random
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

PAD_TOKEN = '<pad>'
UNK_TOKEN = '<unk>'

class CharDataset(Dataset):
    def __init__(self, text, vocab, seq_len) -> None:
        super().__init__()
        ids = [vocab.get(c, vocab[UNK_TOKEN]) for c in text]
        # 注意传入model的是文字的编码表张量！！
        self.data = torch.tensor(ids, dtype=torch.long)
        self.seq_len = seq_len

    def __len__(self):
        # 一共能找出多少个长度为seq_len的句子=字符总数-最后一个滑动窗口大小(即句长seq_len)
        return len(self.data)-self.seq_len

    def __getitem__(self, idx):
        # 错1位自训练
        x = self.data[idx  :idx+self.seq_len]
        y = self.data[idx+1:idx+1+self.seq_len]
        return x,y

class LM(nn.Module):
    def __init__(self, vocab_size, embed_dim, seq_len, n_head, fd_dim, n_layers, dropout):
        super().__init__()
        self.word_embd = nn.Embedding(vocab_size, embed_dim)
        self.pos_embd = nn.Embedding(seq_len, embed_dim)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=n_head,
            dim_feedforward=fd_dim,
            dropout=dropout,
            batch_first=True)
        self.transformerEncoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.linear = nn.Linear(embed_dim, vocab_size)  # 将特征维度转为最终需要的词表维度

    def forward(self, x):
        seq_len = x.size(1)  # [batch_size, seq_len, embd_dim]
        pos = torch.arange(seq_len, dtype=torch.long, device=x.device)  # ！！与x放在同一设备上，否则崩溃！！
        embd_res = self.word_embd(x) + self.pos_embd(pos)
        # MASK
        inf_I = torch.full((seq_len, seq_len), float('-inf'), device=x.device)  # 构造全-inf方阵
        mask = torch.triu(inf_I, diagonal=1)  
        transformer_res = self.transformerEncoder(embd_res, mask=mask)
        return self.linear(transformer_res)

def run_epoch(model, loader_data, criterion, optimizer, device, train_flag):
    model.train(train_flag)  # model.train(false)=model.eval()
    total_loss = 0
    total_tokens = 0

    for x, y in loader_data:  # x,y均是batch: x=[B, L, V]  y=[B,L]
        x, y = x.to(device), y.to(device)
        logits = model(x)
        loss = criterion(logits.reshape(-1,logits.size(-1)), y.reshape(-1))  # 展开成[B, L]维y_pred与1维y

        if train_flag:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        total_loss += loss.item() * y.numel()
        total_tokens += y.numel()

    avg_loss = total_loss / total_tokens
    return avg_loss

def main():
    # 超参数
    EPOCH = 6  # 训练轮数
    SEQ_LEN = 64  # 句长上限
    BATCH_SIZE = 128
    N_LAYERS = 2  # transformer层数  对应bert 12层
    EMBD_DIM = 32  # 对应bert的768
    N_HEAD = 1  # 多头个数
    FEEDFORWARD_DIM = 128  # 前馈层隐藏维度  对应bert 3072 (4×embd_dim)
    DROPOUT = 0.5
    LR = 0.001  # 学习率  
    TRAIN_RATIO = 0.95  # 训练与评估的划分比
    DEVICE = torch.device('cuda')  # GPU

    # 1.get vocab  从海量语料中学习，每个完整batch下每句话定长，无需pad
    # pad 的用途是把变长序列填成等长，好在一个 batch 里对齐。
    with open('corpus.txt', 'r', encoding="utf-8", errors="ignore") as f:
        text = f.read()
    chars = sorted(set(text))
    chars = [PAD_TOKEN, UNK_TOKEN] + chars 
    vocab = {c: idx for idx, c in enumerate(chars)}
    # 2.get random sentences str
    lines = text.splitlines()  # 基于text中的\n \r等分割
    # 防评估数据集永远来自某个文本的最后5%；实际需要的是全文本中随机的5%
    random.shuffle(lines)
    spt_idx = int( len(lines)*TRAIN_RATIO )
    train_txt = '\n'.join(lines[:spt_idx])  # 大长str中，仍需\n告诉模型一句话何处结束
    eval_txt = '\n'.join(lines[spt_idx:])  
    # 3.word2id 编码结果**张量**
    train_ds = CharDataset(train_txt, vocab, SEQ_LEN)
    eval_ds  = CharDataset(eval_txt,  vocab, SEQ_LEN)
    # 4.构建batch
    train_data = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)  # 把凑不满batch_size的最后一组丢掉，利于保持梯度稳定
    eval_data =  DataLoader(eval_ds,  batch_size=BATCH_SIZE, shuffle=False, drop_last=False)  # 验证集无需打乱数据，且不应丢弃任何数据
    # 5.模型设置 vocab_size, embed_dim, n_head, fd_dim, n_layers, dropout
    model = LM(len(vocab), EMBD_DIM, SEQ_LEN, N_HEAD, FEEDFORWARD_DIM, N_LAYERS, DROPOUT).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    criterion = nn.CrossEntropyLoss()
    # 6.训练
    for epoch in range(EPOCH):   
        train_avg_loss = run_epoch(model, train_data, criterion, optimizer, DEVICE, True)
        with torch.no_grad():
            eval_avg_loss = run_epoch(model, eval_data, criterion, optimizer, DEVICE, False)
        print(f"epoch: {epoch}, train_avg_loss={train_avg_loss:>10.4f}, eval_avg_loss={eval_avg_loss:>10.4f}")

    # 7.续写尝试
    generate_words = []
    txt='经济危机时，政府'
    ids = [vocab.get(c, vocab[UNK_TOKEN]) for c in txt]
    # unsqueeze(0):在第0维插入一个大小为 1 的维度 => 因为model的x要求是[B,L]，不插则只有[L] 
    x=torch.tensor(ids, dtype=torch.long, device=DEVICE).unsqueeze(0)  # model在cuda上，则x也必须在cuda
    model.eval()
    with torch.no_grad():
        for _ in range(20):
            logits = model(x)
            next_word_id = logits[0,-1].argmax().item()  # 贪心策略，取概率最高者的索引(标量)
            next_word = chars[next_word_id]  # 预测的下一个字
            generate_words.append(next_word)  

            next_word_tensor = torch.tensor([next_word_id], dtype=torch.long, device=DEVICE).unsqueeze(0)
            x = torch.cat([x, next_word_tensor], dim=1)  # dim=0是unsqueeze插入的句子条数维，dim=1是一句话里的字数维
    print(txt + "".join(generate_words))

if __name__ == "__main__":
    main()
