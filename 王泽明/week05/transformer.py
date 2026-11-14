#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
File: transformer.py
Author: Miku
Date: 2026-11-10
Description: transformer + mask 预训练
"""

import random

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

PAD_TOKEN = '<pad>'
UNK_TOKEN = '<unk>'
EOS_TOKEN = '<eos>'

# 模型结构超参数：预训练与 SFT 共用
SEQ_LEN = 128
EMBD_DIM = 128
N_HEAD = 4
FEEDFORWARD_DIM = 512
N_LAYERS = 4
DROPOUT = 0.3


def build_vocab(*texts):
    """合并多份文本的全部字符构建词表，三个特殊 token 固定在最前。"""
    vocab = {PAD_TOKEN: 0, UNK_TOKEN: 1, EOS_TOKEN: 2}
    for i, c in enumerate(sorted(set(''.join(texts)))):
        vocab[c] = 3 + i
    return vocab


def encode(text, vocab):
    """按词表把文本编码成 id 列表，词表外的字记为 <unk>。"""
    return [vocab.get(c, vocab[UNK_TOKEN]) for c in text]


def save_checkpoint(path, model, chars):
    """保存模型权重与词表字符列表，供后续 SFT 加载。"""
    torch.save({'state_dict': model.state_dict(), 'chars': chars}, path)


class CharDataset(Dataset):
    def __init__(self, text, vocab, seq_len) -> None:
        super().__init__()
        ids = encode(text, vocab)
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
    total_loss, total_tokens = 0.0, 0

    for x, y in loader_data:  # x,y均是batch: x=[B, L, V]  y=[B,L]
        x, y = x.to(device), y.to(device)
        logits = model(x)
        loss = criterion(logits.reshape(-1,logits.size(-1)), y.reshape(-1))  # 展开成[B, L]维y_pred与1维y

        if train_flag:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # SFT 的 y 含 -100 掩码位，只累计有效位置；预训练无 -100，等价于 y.numel()
        n_tokens = (y != -100).sum().item()
        total_loss += loss.item() * n_tokens
        total_tokens += n_tokens

    return total_loss / max(total_tokens, 1)

def main():
    # 超参数
    SEED = 42
    EPOCH = 10  # 训练轮数
    BATCH_SIZE = 128
    LR = 0.0003  # 学习率，模型变大后从 0.001 下调
    TRAIN_RATIO = 0.95  # 训练与评估的划分比
    PATIENCE = 2  # 验证集 loss 连续多少轮不降就早停
    DEVICE = torch.device('cuda')  # GPU

    # 固定随机性：random 管语料划分，torch 管权重初始化、DataLoader 打乱与 dropout
    random.seed(SEED)
    torch.manual_seed(SEED)

    # 1.get vocab  预训练语料与 SFT 数据的字符合并建表，两个阶段共用一张词表
    with open('corpus.txt', 'r', encoding="utf-8", errors="ignore") as f:
        text = f.read()
    with open('sft_train/sft_train.jsonl', 'r', encoding='utf-8') as f:
        sft_text = f.read()
    vocab = build_vocab(text, sft_text)
    chars = [c for c, _ in sorted(vocab.items(), key=lambda kv: kv[1])]  # id->字符 反查表
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
    # 6.训练：验证集 loss 最优时保存，连续 PATIENCE 轮不降则早停
    best_eval_loss, epochs_without_improve = float('inf'), 0
    for epoch in range(EPOCH):
        train_avg_loss = run_epoch(model, train_data, criterion, optimizer, DEVICE, True)
        with torch.no_grad():
            eval_avg_loss = run_epoch(model, eval_data, criterion, optimizer, DEVICE, False)
        save_note = ''
        if eval_avg_loss < best_eval_loss:
            best_eval_loss, epochs_without_improve = eval_avg_loss, 0
            save_checkpoint('pretrained.pt', model, chars)
            save_note = '  <- saved'
        else:
            epochs_without_improve += 1
        print(f"epoch: {epoch}, train_avg_loss={train_avg_loss:>10.4f}, eval_avg_loss={eval_avg_loss:>10.4f}{save_note}")
        if epochs_without_improve >= PATIENCE:
            print(f"early stop at epoch {epoch} (best eval_loss={best_eval_loss:.4f})")
            break


if __name__ == "__main__":
    main()