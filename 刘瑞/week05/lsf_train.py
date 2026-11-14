"""
第五周作业：基于预训练 TransformerLM 做 SFT（指令微调）

流程概览（和 train_langid_cls.py 一样按步骤往下看）：
  1. 读 SFT 数据，拼对话模板
  2. 加载预训练 ckpt，往词表里加特殊标记，扩展 embedding
  3. 编码成 input_ids / labels（只对 assistant 回复算 loss）
  4. 微调训练，保存 sft_best.pt
  5. 加载微调后的模型做问答
"""

import json
import os
import random
import re

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from transformer_lm import TransformerLM

HERE = os.path.dirname(os.path.abspath(__file__))

# 对话角色特殊标记：必须当作「一个 token」，不能按字符拆开
SPECIALS = ["<|user|>", "<|assistant|>", "<|end|>"]
SPECIAL_RE = re.compile("(" + "|".join(re.escape(s) for s in SPECIALS) + ")")


# 1. 准备 SFT 数据
# 1.1 读取 jsonl：每行是 {"instruction": 问题, "output": 回答}
def load_jsonl(path):
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            if "instruction" in obj and "output" in obj:
                rows.append(obj)
    return rows


# 1.2 拼对话模板（训练和推理必须用同一套格式）
#     <|user|>问题<|assistant|>回答<|end|>
def build_dialog(data):
    dialog = []
    for item in data:
        q = item["instruction"]
        a = item["output"]
        dialog.append("<|user|>" + q + "<|assistant|>" + a + "<|end|>")
    return dialog


# 2. 词表与编码
# 2.1 往预训练字表末尾追加特殊标记
def add_special_tokens(char2idx, idx2char, specials=None):
    if specials is None:
        specials = SPECIALS
    for tok in specials:
        if tok not in char2idx:
            i = len(char2idx)
            char2idx[tok] = i
            idx2char[i] = tok
    return char2idx, idx2char


# 2.2 编码：特殊标记整段一个 id，其余按字符查表
def encode_text(text, char2idx):
    ids = []
    unk = char2idx.get("<unk>", 0)
    for part in SPECIAL_RE.split(text):
        if not part:
            continue
        if part in SPECIALS:
            ids.append(char2idx[part])
        else:
            for c in part:
                ids.append(char2idx.get(c, unk))
    return ids


# 2.3 构造训练样本：
#     prompt（到 <|assistant|> 为止）labels = -100，不参与 loss
#     response（含 <|end|>）labels = token id，只在这里学「怎么回答」
def process_sft_data(dialog_list, char2idx, max_len):
    input_ids_list = []
    labels_list = []
    marker = "<|assistant|>"
    for text in dialog_list:
        split_idx = text.find(marker)
        if split_idx == -1:
            continue
        prompt_text = text[: split_idx + len(marker)]
        response_text = text[split_idx + len(marker) :]

        prompt_ids = encode_text(prompt_text, char2idx)
        response_ids = encode_text(response_text, char2idx)
        input_ids = prompt_ids + response_ids
        labels = [-100] * len(prompt_ids) + response_ids

        # 超长从左边截断，尽量保住回复尾部
        if len(input_ids) > max_len:
            overflow = len(input_ids) - max_len
            input_ids = input_ids[overflow:]
            labels = labels[overflow:]

        if all(t == -100 for t in labels):
            continue
        input_ids_list.append(input_ids)
        labels_list.append(labels)
    return input_ids_list, labels_list


# 3. 扩展模型参数（词表变大 / 序列变长时必须做）
# 3.1 扩展 token embedding，旧权重拷过去，新行随机初始化
def resize_token_embeddings(model, new_vocab_size):
    old = model.tok.weight.data
    old_v, d = old.shape
    if new_vocab_size == old_v:
        return
    new_emb = nn.Embedding(new_vocab_size, d)
    nn.init.normal_(new_emb.weight, std=0.02)
    new_emb.weight.data[:old_v] = old
    model.tok = new_emb
    model.head.weight = model.tok.weight  # 保持输入输出 weight tying


# 3.2 扩展位置编码（预训练 max_len=64，SFT 对话更长，扩到 128）
def resize_pos_embeddings(model, new_max_len):
    if model.pos is None:
        return
    old = model.pos.weight.data
    old_len, d = old.shape
    if new_max_len <= old_len:
        return
    new_pos = nn.Embedding(new_max_len, d)
    nn.init.normal_(new_pos.weight, std=0.02)
    new_pos.weight.data[:old_len] = old
    model.pos = new_pos


# 3.3 从预训练 ckpt 建模型，再按新词表/新长度 resize
def build_model_from_ckpt(ck, vocab_size, max_len, device):
    c = ck["args"]
    old_v = len(ck["char2idx"])
    old_max = c.get("seq_len", 64)
    model = TransformerLM(
        old_v, c["d_model"], c["heads"], c["layers"], old_max,
        c.get("dropout", 0.1),
        use_pos=not c.get("no_pos", False),
        causal=not c.get("no_mask", False),
    )
    model.load_state_dict(ck["model_state"])
    resize_token_embeddings(model, vocab_size)
    resize_pos_embeddings(model, max_len)
    return model.to(device)


# 4. Dataset / DataLoader
class SFTDataset(Dataset):
    def __init__(self, input_ids_list, labels_list):
        self.input_ids = input_ids_list
        self.labels = labels_list

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, i):
        return self.input_ids[i], self.labels[i]


# 右 padding：因果注意力下，末尾 pad 不会污染真实 token 的上下文
def collate_sft(batch):
    max_len = max(len(x) for x, _ in batch)
    bsz = len(batch)
    input_ids = torch.zeros(bsz, max_len, dtype=torch.long)
    labels = torch.full((bsz, max_len), -100, dtype=torch.long)
    for i, (ids, labs) in enumerate(batch):
        n = len(ids)
        input_ids[i, :n] = torch.tensor(ids, dtype=torch.long)
        labels[i, :n] = torch.tensor(labs, dtype=torch.long)
    return input_ids, labels


# 5. 损失与验证
# 标准因果 LM：位置 t 预测 t+1；ignore_index=-100 跳过 prompt
def sft_loss(logits, labels):
    shift_logits = logits[:, :-1, :].contiguous()
    shift_labels = labels[:, 1:].contiguous()
    return F.cross_entropy(
        shift_logits.reshape(-1, shift_logits.size(-1)),
        shift_labels.reshape(-1),
        ignore_index=-100,
    )


def evaluate(model, data, device):
    model.eval()
    watch_loss = []
    with torch.no_grad():
        for x, y in data:
            x, y = x.to(device), y.to(device)
            loss = sft_loss(model(x), y)
            watch_loss.append(loss.item())
    return float(np.mean(watch_loss)) if watch_loss else float("inf")


# 6. 问答推理
def load_sft_model(ckpt_path, device):
    """加载已经微调好的 sft_best.pt（词表里已经有特殊标记）。"""
    ck = torch.load(ckpt_path, map_location=device)
    c = ck["args"]
    char2idx = ck["char2idx"]
    idx2char = {int(k): v for k, v in ck["idx2char"].items()}
    max_len = c.get("sft_max_len") or c.get("seq_len", 128)
    model = TransformerLM(
        len(char2idx), c["d_model"], c["heads"], c["layers"], max_len,
        0.0,
        use_pos=not c.get("no_pos", False),
        causal=not c.get("no_mask", False),
    ).to(device)
    model.load_state_dict(ck["model_state"])
    model.eval()
    return model, char2idx, idx2char, max_len


def decode_ids(ids, idx2char):
    return "".join(idx2char.get(i, "") for i in ids)


def predict(question, model, char2idx, idx2char, max_len=128, max_new=80, temperature=0.8):
    """
    按训练同款格式提问：
        <|user|>问题<|assistant|>
    然后逐字生成，直到碰到 <|end|>。
    """
    device = next(model.parameters()).device
    prompt = "<|user|>" + question + "<|assistant|>"
    ids = encode_text(prompt, char2idx)
    end_id = char2idx["<|end|>"]
    out = []

    with torch.no_grad():
        for _ in range(max_new):
            ctx = ids[-max_len:]
            logits = model(torch.tensor([ctx], device=device))[0, -1]
            if temperature <= 0.01:
                nxt = int(logits.argmax())
            else:
                nxt = int(torch.multinomial(F.softmax(logits / temperature, dim=-1), 1))
            if nxt == end_id:
                break
            ids.append(nxt)
            out.append(nxt)

    return decode_ids(out, idx2char)


if __name__ == "__main__":
    # ---------- 超参数（和语言识别作业一样，集中写在入口） ----------
    batch_size = 32
    epoch_size = 3
    lr = 0.0003
    max_len = 128
    val_ratio = 0.05
    seed = 42

    ckpt_path = os.path.join(HERE, "tf_mask.pt")       # 第四周预训练权重
    data_path = os.path.join(HERE, "data", "sft_train.jsonl")
    save_path = os.path.join(HERE, "sft_best.pt")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(seed)
    random.seed(seed)

    # ---------- 1. 读数据、拼对话 ----------
    rows = load_jsonl(data_path)
    dialogs = build_dialog(rows)

    # ---------- 2. 加载预训练 + 加特殊标记 + 扩 embedding ----------
    # map_location="cpu"：在 GPU 上存的 ckpt，无 CUDA 时也能加载
    ck = torch.load(ckpt_path, map_location="cpu")
    char2idx = dict(ck["char2idx"])
    idx2char = {int(k): v for k, v in dict(ck["idx2char"]).items()}
    char2idx, idx2char = add_special_tokens(char2idx, idx2char)

    # ---------- 3. 编码成 input_ids / labels ----------
    input_ids, labels = process_sft_data(dialogs, char2idx, max_len)

    # 划分训练 / 验证
    idx = list(range(len(input_ids)))
    random.shuffle(idx)
    n_val = max(1, int(len(idx) * val_ratio))
    val_idx = set(idx[:n_val])
    tr_x = [input_ids[i] for i in idx if i not in val_idx]
    tr_y = [labels[i] for i in idx if i not in val_idx]
    va_x = [input_ids[i] for i in idx if i in val_idx]
    va_y = [labels[i] for i in idx if i in val_idx]

    train_data = DataLoader(
        SFTDataset(tr_x, tr_y),
        batch_size=batch_size, shuffle=True, collate_fn=collate_sft, drop_last=True,
    )
    dev_data = DataLoader(
        SFTDataset(va_x, va_y),
        batch_size=batch_size, shuffle=False, collate_fn=collate_sft,
    )

    model = build_model_from_ckpt(ck, len(char2idx), max_len, device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    print("device: %s  样本数: %d (train=%d val=%d)  词表: %d  max_len: %d"
          % (device, len(input_ids), len(tr_x), len(va_x), len(char2idx), max_len))

    # ---------- 4. 训练循环 ----------
    best = float("inf")
    for epoch in range(epoch_size):
        model.train()
        watch_loss = []
        for x, y in train_data:
            x, y = x.to(device), y.to(device)
            loss = sft_loss(model(x), y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            optimizer.zero_grad()
            watch_loss.append(loss.item())

        train_loss = float(np.mean(watch_loss))
        val_loss = evaluate(model, dev_data, device)
        print("=========\n第%d轮平均loss:%f  验证loss:%f"
              % (epoch + 1, train_loss, val_loss))

        # 验证更好就存一份（和语言模型作业存最优 ckpt 一样）
        if val_loss < best:
            best = val_loss
            sft_args = dict(ck["args"])
            sft_args.update({
                "seq_len": max_len,
                "sft": True,
                "sft_lr": lr,
                "sft_epochs": epoch_size,
                "sft_max_len": max_len,
            })
            torch.save({
                "model_state": model.state_dict(),
                "char2idx": char2idx,
                "idx2char": idx2char,
                "args": sft_args,
            }, save_path)
            print("  -> 已保存最优模型到 %s" % save_path)

    print("\n最佳验证 loss: %.4f  已保存 %s" % (best, save_path))

    # ---------- 5. 训练结束后，测几条问答（仿语言识别作业末尾 predict） ----------
    model, char2idx, idx2char, max_len = load_sft_model(save_path, device)
    test_questions = [
        "请解释以下缩写的含义：NLP。",
        "给定两个数字，请选择其中的较大数并将其打印出来。 数字1: 238 数字2: 564",
        "请问北京奥运会的举办时间和地点分别是什么？",
        "给出一些适合夏天穿的衣服款式。",
    ]
    print("\n======= 问答测试 =======")
    for q in test_questions:
        ans = predict(q, model, char2idx, idx2char, max_len=max_len, temperature=0.5)
        print("问：%s" % q)
        print("答：%s\n" % ans)
