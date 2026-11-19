#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
File: sft.py
Author: Miku
Date: 2026-11-10
Description: sft
"""

import json

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from transformer import (
    LM, PAD_TOKEN, UNK_TOKEN, EOS_TOKEN,
    SEQ_LEN, EMBD_DIM, N_HEAD, FEEDFORWARD_DIM, N_LAYERS, DROPOUT,
    encode, save_checkpoint, run_epoch,
)

SEPARATOR = "\n"


def build_question(instruction):
    """问题末尾追加分隔符，训练与生成共用。"""
    return instruction + SEPARATOR


def load_jsonl(path):
    """逐行解析 jsonl，跳过空行。"""
    samples = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            samples.append(json.loads(line))
    return samples


class SFTDataset(Dataset):
    """
    y[i] = x[i+1]：位置 i 的预测目标是下一个字。
    问题部分和 pad 位填 -100，不参与损失计算；
    从问题的最后一位开始计算损失，该位预测答案的第一个字。
    """

    def __init__(self, data, char_to_id, max_length):
        self.data = []
        pad_id = char_to_id[PAD_TOKEN]
        eos_id = char_to_id[EOS_TOKEN]
        num_skipped = 0
        for item in data:
            question_ids = encode(build_question(item["instruction"]), char_to_id)
            answer_ids = encode(item["output"], char_to_id)
            answer_ids.append(eos_id)
            token_ids = question_ids + answer_ids
            if len(token_ids) > max_length:
                num_skipped += 1
                continue  # 放不下整条答案的样本跳过，避免截断后学不到 <eos>
            loss_start = len(question_ids) - 1  # 问题末位：该位预测答案的第一个字
            input_ids = token_ids + [pad_id] * (max_length - len(token_ids))
            labels = [token_ids[pos + 1] if loss_start <= pos < len(token_ids) - 1 else -100
                      for pos in range(max_length)]
            self.data.append((
                torch.tensor(input_ids, dtype=torch.long),
                torch.tensor(labels, dtype=torch.long),
            ))
        if num_skipped:
            print(f"跳过 {num_skipped} 条超过 {max_length} 字的样本")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


def generate(model, char_to_id, id_to_char, question_text, max_new_chars, device):
    """贪心逐字生成，遇到 <eos> 停止；序列总长不超过位置编码表容量。"""
    max_positions = model.pos_embd.weight.shape[0]
    question_ids = encode(question_text, char_to_id)
    if len(question_ids) > max_positions - 1:
        question_ids = question_ids[:max_positions - 2] + question_ids[-1:]  # 保留开头与末尾的分隔符，与训练侧一致
    if not question_ids:
        question_ids = [char_to_id[PAD_TOKEN]]
    input_ids = torch.tensor(question_ids, dtype=torch.long, device=device).unsqueeze(0)
    forbidden_ids = [char_to_id[PAD_TOKEN], char_to_id[UNK_TOKEN]]  # 特殊 token 不作为普通字输出
    model.eval()
    generated_chars = []
    with torch.no_grad():
        for _ in range(max_new_chars):
            if input_ids.size(1) >= max_positions:
                break  # 位置编码表已用满，再生成会索引越界
            logits = model(input_ids)
            logits[0, -1].index_fill_(
                0, torch.tensor(forbidden_ids, device=logits.device), float('-inf'))
            next_char_id = logits[0, -1].argmax().item()
            if next_char_id == char_to_id[EOS_TOKEN]:
                break
            generated_chars.append(id_to_char[next_char_id])
            input_ids = torch.cat(
                [input_ids, torch.tensor([[next_char_id]], dtype=torch.long,
                                         device=input_ids.device)], dim=1)
    return ''.join(generated_chars)


def main():
    # 超参数
    NUM_EPOCHS = 20
    PATIENCE = 2  # 验证集 loss 连续多少轮不降就早停
    BATCH_SIZE = 32
    LR = 3e-4
    MAX_LENGTH = SEQ_LEN  # 与预训练句长一致
    SEED = 42
    DEVICE = torch.device('cuda')  # GPU

    torch.manual_seed(SEED)

    # 1.先读词表定 vocab_size，再建模型并填入预训练权重
    checkpoint = torch.load('pretrained.pt', map_location='cpu')
    state_dict, vocab_chars = checkpoint['state_dict'], checkpoint['chars']
    char_to_id = {c: i for i, c in enumerate(vocab_chars)}
    model = LM(len(char_to_id), EMBD_DIM, MAX_LENGTH, N_HEAD,
               FEEDFORWARD_DIM, N_LAYERS, DROPOUT).to(DEVICE)
    model.load_state_dict(state_dict)

    # 2.数据
    train_raw = load_jsonl('sft_train/sft_train.jsonl')
    val_raw = load_jsonl('sft_train/sft_val.jsonl')
    train_dataset = SFTDataset(train_raw, char_to_id, MAX_LENGTH)
    val_dataset = SFTDataset(val_raw, char_to_id, MAX_LENGTH)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

    # 3.训练：验证集 loss 最优时保存，连续 PATIENCE 轮不降则早停
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    criterion = nn.CrossEntropyLoss()  # 默认 ignore_index=-100，跳过掩码位
    best_val_loss, epochs_without_improve = float('inf'), 0
    for epoch in range(NUM_EPOCHS):
        train_loss = run_epoch(model, train_loader, criterion, optimizer, DEVICE, True)
        with torch.no_grad():
            val_loss = run_epoch(model, val_loader, criterion, optimizer, DEVICE, False)
        save_note = ''
        if val_loss < best_val_loss:
            best_val_loss, epochs_without_improve = val_loss, 0
            save_checkpoint('sft_model.pt', model, vocab_chars)
            save_note = '  <- saved'
        else:
            epochs_without_improve += 1
        print(f"epoch {epoch}: train_loss={train_loss:.4f} val_loss={val_loss:.4f}{save_note}")
        if epochs_without_improve >= PATIENCE:
            print(f"early stop at epoch {epoch} (best val_loss={best_val_loss:.4f})")
            break

    # 4.手动测试
    best_state_dict = torch.load('sft_model.pt', map_location='cpu')['state_dict']
    model.load_state_dict(best_state_dict)
    id_to_char = {i: c for c, i in char_to_id.items()}
    ques = build_question('1,2,4,7,-3,0中的最小值是几')
    answer = generate(model, char_to_id, id_to_char, ques, MAX_LENGTH, DEVICE)
    print("Q:", ques)
    print("A:", answer)


if __name__ == "__main__":
    main()
