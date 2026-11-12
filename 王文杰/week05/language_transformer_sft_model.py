"""
字符级 Transformer 语言模型的 SFT（监督微调）训练脚本。
在已预训练好的模型（best_Transformer_model.pt）基础上做微调，
支持 alpaca 风格 jsonl 指令-回答数据 + loss_mask（只对回答区算损失）。

用法:
    python language_transformer_sft_model.py \
        --train_file sft_train.jsonl \
        --val_file sft_val.jsonl \
        --pretrained best_Transformer_model.pt \
        --save best_Transformer_sft.pt
"""

import math
import argparse
import glob
import random
import json
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader


# ─────────────────────────── SFT 数据 ───────────────────────────

SFT_SEP_INSTR = "\n"   # 指令与回答之间的分隔（用换行，字符级天然单字符分隔）
SFT_END = "\n"          # 样本结束符


def load_sft_jsonl(path):
    """读取 alpaca 风格 jsonl：每行 {"instruction": ..., "output": ...}"""
    samples = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            instr = obj.get("instruction", "")
            out = obj.get("output", "")
            samples.append((instr, out))
    return samples


def build_sft_text(instr, out):
    """拼成 指令 + 分隔 + 回答 的单段文本，供字符级模型切样本。"""
    return instr + SFT_SEP_INSTR + out + SFT_END


# ─────────────────────────── 数据 ───────────────────────────

def load_corpus(pattern="*.txt"):
    texts = []
    for path in glob.glob(pattern):
        with open(path, encoding="utf-8", errors="ignore") as f:
            texts.append(f.read())
    return "".join(texts)


def build_vocab(text, max_vocab=None):
    """
    构建字表：按出现频率取前 max_vocab 个字（None = 全部字符）。
    语料里的生僻字超出字表后会被 CharDataset 过滤掉 —— 字表越小模型越小，
    但覆盖不足会丢信息，这是词表大小的取舍。
    """
    from collections import Counter
    counter = Counter(text)
    chars = [c for c, _ in counter.most_common(max_vocab)] if max_vocab else sorted(set(text))
    char2idx = {c: i for i, c in enumerate(chars)}
    idx2char = {i: c for c, i in char2idx.items()}
    return char2idx, idx2char


class CharDataset(Dataset):
    """
    字符级语言模型的训练样本：输入 x = 连续 seq_len 个字符，目标 y = 往后错一位。

    stride 控制切块间隔：
        stride=1  —— 每个字符位置都当一个样本（样本大量重叠，一期语料能切出上百万个样本）
        stride=seq_len —— 不重叠切块（标准做法，样本数 = 语料长度 / seq_len）
    语料一大，stride=1 会让一轮训练变得极慢，这不是模型慢，而是样本数被放大了 seq_len 倍。

    sft 模式额外返回 loss_mask：回答区为 1、指令区为 0（SFT 只对回答区算 loss）。
    """

    def __init__(self, text, char2idx, seq_len, stride=1, max_samples=None,
                 sep_instr=None, sft=False):
        self.seq_len = seq_len
        self.stride = max(1, stride)
        self.sft = sft
        self.sep_instr = sep_instr
        ids = [char2idx.get(c, -1) for c in text]
        # 过滤掉不在词表里的字符（SFT 模式下这些字模型无法处理）
        ids = [i for i in ids if i >= 0]
        self.data = torch.tensor(ids, dtype=torch.long)
        n = max(0, (len(self.data) - self.seq_len - 1) // self.stride + 1)
        # 样本上限：机器差时可以只用前 max_samples 个样本，快速跑完一轮
        self.n = min(n, max_samples) if max_samples else n

    def __len__(self):
        return self.n

    def __getitem__(self, idx):
        start = idx * self.stride
        x = self.data[start: start + self.seq_len]
        y = self.data[start + 1: start + self.seq_len + 1]
        if not self.sft:
            return x, y
        # SFT：构造 loss_mask，只在回答区（分隔符之后）算 loss
        mask = torch.zeros(self.seq_len, dtype=torch.float)
        if self.sep_instr is not None:
            # 找到本样本中分隔符出现的位置，其后（含回答）标记为 1
            # y 是第 2..seq_len+1 个字，分隔符若在 y 中，则从该位置起为回答区
            sep_id = self.sep_instr
            for j in range(self.seq_len):
                if y[j].item() == sep_id:
                    mask[j + 1:] = 1.0
                    break
        return x, y, mask


# ─────────────────────────── 模型 ───────────────────────────

class PositionalEncoding(nn.Module):
    """ sinusoid 位置编码：自注意力本身对顺序无感知，必须显式注入位置信息 """

    def __init__(self, d_model, max_len=512):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))   # (1, max_len, d_model)

    def forward(self, x):
        return x + self.pe[:, : x.size(1)]


class LM(nn.Module):
    """ decoder-only Transformer：embed_dim 必须能被 nhead 整除 """

    def __init__(self, vocab_size, embed_dim, hidden_dim, num_layers, dropout, nhead=2, max_len=512):
        super().__init__()
        d_model = embed_dim
        self.embed = nn.Embedding(vocab_size, d_model)
        self.pos_enc = PositionalEncoding(d_model, max_len)
        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=num_layers)
        self.drop = nn.Dropout(dropout)
        self.fc = nn.Linear(d_model, vocab_size)

    def forward(self, x):
        T = x.size(1)
        e = self.drop(self.pos_enc(self.embed(x)))
        # 因果掩码：屏蔽未来 token，否则训练时会"偷看答案"，PPL 虚低但无法推理
        causal_mask = nn.Transformer.generate_square_subsequent_mask(T).to(x.device)
        out = self.encoder(e, mask=causal_mask, is_causal=True)
        logits = self.fc(self.drop(out))   # (B, T, V)
        return logits


# ─────────────────────────── 训练 / 评估 ───────────────────────────

def run_epoch(model, loader, criterion, optimizer, device, train=True, log_every=50, sft=False):
    model.train(train)
    total_loss = 0.0
    total_tokens = 0

    for step, batch in enumerate(loader, 1):
        if sft:
            x, y, mask = batch
            x, y, mask = x.to(device), y.to(device), mask.to(device)
        else:
            x, y = batch
            x, y = x.to(device), y.to(device)

        logits = model(x)
        # 每个位置的 loss（不自动平均）
        loss_per = criterion(logits.reshape(-1, logits.size(-1)), y.reshape(-1))

        if sft:
            # 只对回答区（mask=1）的位置求平均
            mask_flat = mask.reshape(-1)
            loss = (loss_per * mask_flat).sum() / mask_flat.sum().clamp(min=1.0)
        else:
            loss = loss_per.mean()

        if train:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        total_loss += loss.item() * y.numel()
        total_tokens += y.numel()

        # 进度打印：避免 CPU 上跑大 epoch 时看起来像卡死
        if train and step % log_every == 0:
            print(f"    [{step}/{len(loader)} batch]  loss={loss.item():.4f}", flush=True)

    avg_loss = total_loss / total_tokens
    ppl = math.exp(avg_loss)
    return avg_loss, ppl


# ─────────────────────────── 主函数 ───────────────────────────

def  main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_file",  default="sft_train.jsonl", help="SFT 训练集 jsonl")
    parser.add_argument("--val_file",    default="sft_val.jsonl", help="SFT 验证集 jsonl")
    parser.add_argument("--pretrained",  default="best_Transformer_model.pt", help="预训练 checkpoint 路径（默认 best_Transformer_model.pt，SFT 在其基础上微调）")
    parser.add_argument("--epochs",     type=int,   default=10)
    parser.add_argument("--seq_len",    type=int,   default=64)
    parser.add_argument("--stride",     type=int,   default=64, help="切块间隔：1=密集重叠，seq_len=不重叠")
    parser.add_argument("--max_samples", type=int,  default=10000, help="训练样本上限，0=不限；机器差调小可快速跑完", )
    parser.add_argument("--batch_size", type=int,   default=32)
    parser.add_argument("--dropout",    type=float, default=0.3)
    parser.add_argument("--lr",         type=float, default=1e-3)
    parser.add_argument("--seed",       type=int,   default=42)
    parser.add_argument("--save",       default="best_Transformer_sft.pt")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    random.seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}  model: TRANSFORMER  mode: SFT")

    # ── 复用预训练的词表与超参（SFT 不重建词表）──
    ckpt_pre = torch.load(args.pretrained, map_location=device, weights_only=False)
    char2idx = ckpt_pre["char2idx"]
    idx2char = ckpt_pre["idx2char"]
    cfg_pre = ckpt_pre["args"]
    args.embed_dim = cfg_pre["embed_dim"]
    args.hidden_dim = cfg_pre["hidden_dim"]
    args.num_layers = cfg_pre["num_layers"]
    args.nhead = cfg_pre.get("nhead", 4)
    vocab_size = len(char2idx)
    print(f"复用预训练词表: {args.pretrained}  (字表 {vocab_size} 字)")

    # ── 准备 SFT 训练/验证文本与数据集 ──
    train_samples = load_sft_jsonl(args.train_file)
    val_samples = load_sft_jsonl(args.val_file)
    train_text = "".join(build_sft_text(i, o) for i, o in train_samples)
    val_text = "".join(build_sft_text(i, o) for i, o in val_samples)
    sep_id = char2idx.get(SFT_SEP_INSTR, None)
    train_ds = CharDataset(train_text, char2idx, args.seq_len, args.stride,
                           max_samples=args.max_samples, sep_instr=sep_id, sft=True)
    val_ds = CharDataset(val_text, char2idx, args.seq_len, args.stride,
                         max_samples=args.max_samples or 2000, sep_instr=sep_id, sft=True)

    print(f"训练样本数: {len(train_ds):,}  验证样本数: {len(val_ds):,}")

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=True, drop_last=True)

    # ── 模型 ──
    model = LM(
        vocab_size=vocab_size,
        embed_dim=args.embed_dim,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        dropout=args.dropout,
        nhead=args.nhead,
    ).to(device)

    # ── 关键：SFT 载入预训练权重（覆盖随机初始化）──
    model.load_state_dict(ckpt_pre["model_state"])
    print(f"已载入预训练权重，在其基础上做 SFT")

    total_params = sum(p.numel() for p in model.parameters())
    print(f"模型参数量: {total_params:,}")

    criterion = nn.CrossEntropyLoss(reduction="none")
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    best_val_ppl = float("inf")

    print(f"\n{'Epoch':>6}  {'Train Loss':>10}  {'Train PPL':>10}  {'Val Loss':>10}  {'Val PPL':>10}")
    print("-" * 56)

    sft = True
    for epoch in range(1, args.epochs + 1):
        tr_loss, tr_ppl = run_epoch(model, train_loader, criterion, optimizer, device, train=True, sft=sft)
        with torch.no_grad():
            va_loss, va_ppl = run_epoch(model, val_loader, criterion, optimizer, device, train=False, sft=sft)

        marker = "  *" if va_ppl < best_val_ppl else ""
        if va_ppl < best_val_ppl:
            best_val_ppl = va_ppl
            torch.save({
                "model_state": model.state_dict(),
                "char2idx": char2idx,
                "idx2char": idx2char,
                "args": vars(args),
            }, args.save)

        print(f"{epoch:>6}  {tr_loss:>10.4f}  {tr_ppl:>10.2f}  {va_loss:>10.4f}  {va_ppl:>10.2f}{marker}")

    print(f"\n训练完成。最佳验证 PPL: {best_val_ppl:.2f}  已保存至 {args.save}")


if __name__ == "__main__":
    main()
