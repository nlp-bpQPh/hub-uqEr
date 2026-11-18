"""
字符级语言模型训练脚本（Transformer 版）。

和 language_model.py 相比，只把模型本身换掉：
    RNN / LSTM  →  Transformer（decoder-only，自回归）
其余流程（语料加载、词表、切块、PPL 评估、checkpoint 字段）保持原样。

关键点 —— mask
    语言模型预测第 t 个字时，只允许看到第 0..t 个字。
    Transformer 的自注意力默认"全局可见"，若不加 mask，第 t 个位置的
    logits 会直接看到第 t+1.. 个未来字：训练 loss 会低得不正常（因为答案
    就摆在输入里），但推理时没有未来字，模型立刻失效。所以必须在
    softmax 之前加上因果 mask（上三角 -inf）：

        scores = QKᵀ / √d_head
        scores = scores + mask        # mask[i, j] = -inf (j > i)
        attn   = softmax(scores, -1)  # 未来位置的权重恰好为 0

    --self_test 里的对照实验会证明：开 mask 时改掉未来字，前缀输出不变；
    关 mask 时改掉未来字，前缀输出跟着变（信息泄漏）。

用法:
    python language_model_transformer.py --epochs 20
    python language_model_transformer.py --d_model 256 --nhead 8 --num_layers 4 --ffn_dim 1024
    python language_model_transformer.py --epochs 20 --log_every 50 --log_file train.log
    python language_model_transformer.py --epochs 20 --resume best_model_transformer.pt
    python language_model_transformer.py --self_test      # 只跑 mask 自检，不训练

从任意目录启动都可以：--corpus 的相对路径先按当前工作目录查找，
找不到时自动回退到脚本所在目录（语料和脚本放在一起）。
"""

import math
import time
import argparse
import glob
import os
import random
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader


# ─────────────────────────── 数据 ───────────────────────────

def corpus_paths(pattern="*.txt"):
    """
    把 --corpus 展开成文件列表。

    相对路径先按**当前工作目录**解析；一个都找不到时，再回退到**脚本所在目录**。
    这样无论是在语料目录里直接运行，还是在别处用绝对路径启动脚本，
    都能正常读到语料。
    """
    if os.path.isabs(pattern):
        return glob.glob(pattern)
    hits = glob.glob(pattern)
    if hits:
        return hits
    here = os.path.dirname(os.path.abspath(__file__))
    return glob.glob(os.path.join(here, pattern))


def load_corpus(pattern="*.txt"):
    texts = []
    for path in corpus_paths(pattern):
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
        stride=1  —— 每个字符位置都当一个样本（样本大量重叠）
        stride=seq_len —— 不重叠切块（标准做法，样本数 = 语料长度 / seq_len）
    语料一大，stride=1 会让一轮训练变得极慢，这不是模型慢，而是样本数被放大了 seq_len 倍。
    """

    def __init__(self, text, char2idx, seq_len, stride=1):
        self.seq_len = seq_len
        self.stride = max(1, stride)
        ids = [char2idx[c] for c in text if c in char2idx]
        self.data = torch.tensor(ids, dtype=torch.long)

    def __len__(self):
        return max(0, (len(self.data) - self.seq_len - 1) // self.stride + 1)

    def __getitem__(self, idx):
        start = idx * self.stride
        x = self.data[start: start + self.seq_len]
        y = self.data[start + 1: start + self.seq_len + 1]
        return x, y


# ─────────────────────────── 模型（Transformer） ───────────────────────────

NEG_INF = float("-inf")


def build_causal_mask(seq_len, device=None):
    """
    加性因果 mask，形状 (T, T)：
        mask[i, j] = 0      当 j <= i  —— 第 i 个位置可以看到自己及其左侧
        mask[i, j] = -inf   当 j >  i  —— 右侧（未来）位置被彻底屏蔽

    用 -inf 而不是 0/1 开关：它可以直接加在 softmax 的输入上，
    exp(-inf) = 0，未来位置的注意力权重天然就是 0。
    注意对角线是 0（diagonal=1 表示从对角线上方一格开始屏蔽），
    因为预测当前位置时，当前位置的输入是允许看的 —— 这正是"错一位"的由来。
    """
    mask = torch.full((seq_len, seq_len), NEG_INF, device=device)
    return torch.triu(mask, diagonal=1)


class PositionalEncoding(nn.Module):
    """
    正弦位置编码（Attention Is All You Need 原文公式），无参数：
        PE[pos, 2i]   = sin(pos / 10000^(2i/d))
        PE[pos, 2i+1] = cos(pos / 10000^(2i/d))

    自注意力对顺序是无感知的（把输入的字打乱，输出也只是跟着打乱），
    位置编码负责把"这是第几个字"的信息注入进去。
    这里按需现算，推理时上下文比训练时更长也不会越界；
    换成可学习的 nn.Embedding(max_len, d_model) 也可以，但那样会有一个长度上限。
    """

    def __init__(self, d_model):
        super().__init__()
        assert d_model % 2 == 0, "d_model 取偶数，正弦/余弦才能成对排布"
        self.d_model = d_model

    def forward(self, x):
        # x: (B, T, d_model)
        T, d = x.size(-2), self.d_model
        device, dtype = x.device, x.dtype
        pos = torch.arange(T, device=device, dtype=torch.float32).unsqueeze(1)   # (T, 1)
        idx = torch.arange(0, d, 2, device=device, dtype=torch.float32)          # (d/2,)
        div = torch.exp(-math.log(10000.0) * idx / d)                            # (d/2,)
        pe = torch.zeros(T, d, device=device, dtype=torch.float32)
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        return x + pe.to(dtype)


class CausalSelfAttention(nn.Module):
    """多头自注意力，attn_mask 为加性 mask（因果 mask / padding mask 都在这里生效）。"""

    def __init__(self, d_model, nhead, dropout):
        super().__init__()
        assert d_model % nhead == 0, f"d_model({d_model}) 必须能被 nhead({nhead}) 整除"
        self.nhead = nhead
        self.head_dim = d_model // nhead
        self.scale = self.head_dim ** -0.5        # 1/√d_head，防止点积过大让 softmax 饱和
        self.qkv = nn.Linear(d_model, 3 * d_model)  # 一个矩阵一次算出 Q/K/V
        self.proj = nn.Linear(d_model, d_model)
        self.attn_drop = nn.Dropout(dropout)
        self.proj_drop = nn.Dropout(dropout)

    def forward(self, x, attn_mask=None):
        B, T, C = x.shape
        q, k, v = self.qkv(x).split(C, dim=-1)                     # 各 (B, T, C)
        # 拆多头：(B, T, H, Dh) → (B, H, T, Dh)
        q = q.reshape(B, T, self.nhead, self.head_dim).transpose(1, 2)
        k = k.reshape(B, T, self.nhead, self.head_dim).transpose(1, 2)
        v = v.reshape(B, T, self.nhead, self.head_dim).transpose(1, 2)

        scores = (q @ k.transpose(-2, -1)) * self.scale             # (B, H, T, T)
        if attn_mask is not None:
            scores = scores + attn_mask                             # ← mask 在这里生效
        attn = self.attn_drop(torch.softmax(scores, dim=-1))        # 未来位置权重 = 0
        out = attn @ v                                              # (B, H, T, Dh)
        out = out.transpose(1, 2).reshape(B, T, C)
        return self.proj_drop(self.proj(out))


class TransformerBlock(nn.Module):
    """
    一层 Transformer（Pre-LN 写法）：
        x = x + Attention(LN(x))
        x = x + FFN(LN(x))

    Pre-LN 把归一化放在子层之前，梯度更稳，小模型不用 warmup 也能训；
    原论文是 Post-LN（LN 放在残差相加之后），通常要配合 warmup 才收敛得动。
    """

    def __init__(self, d_model, nhead, ffn_dim, dropout):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.attn = CausalSelfAttention(d_model, nhead, dropout)
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, ffn_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_dim, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x, attn_mask=None):
        x = x + self.attn(self.norm1(x), attn_mask)
        x = x + self.ffn(self.norm2(x))
        return x


class LM(nn.Module):
    """字符级 Transformer 语言模型：Embedding → (PE) → N×Block → LayerNorm → Linear。"""

    def __init__(self, vocab_size, d_model, nhead, num_layers, ffn_dim, dropout):
        super().__init__()
        self.d_model = d_model
        self.embed = nn.Embedding(vocab_size, d_model)
        self.pos = PositionalEncoding(d_model)
        self.blocks = nn.ModuleList(
            TransformerBlock(d_model, nhead, ffn_dim, dropout)
            for _ in range(num_layers)
        )
        self.norm_f = nn.LayerNorm(d_model)     # 所有 block 之后再归一化一次
        self.drop = nn.Dropout(dropout)
        self.fc = nn.Linear(d_model, vocab_size)

    def forward(self, x, key_padding_mask=None, causal=True):
        """
        x: (B, T) 字符 id
        key_padding_mask: (B, T) bool，True 表示该位置是 padding、不参与注意力。
            本脚本的数据是定长切块，没有 padding，默认 None；保留这个参数是
            为了变长输入（例如输入法批量解码）也能直接套用。
            注意：右侧 padding 时因果 mask 已经顺带挡住了它，真正需要
            padding mask 的是左侧 padding，或者非因果的双向注意力。
        causal: 仅给 --self_test 做对照实验用，正常训练/推理恒为 True。
        返回: (B, T, vocab_size)，每个位置的"下一个字"打分
        """
        B, T = x.shape

        attn_mask = build_causal_mask(T, x.device) if causal else None   # (T, T)
        if key_padding_mask is not None:
            pad_mask = torch.zeros(B, 1, 1, T, device=x.device,
                                   dtype=self.embed.weight.dtype)
            pad_mask = pad_mask.masked_fill(key_padding_mask[:, None, None, :], NEG_INF)
            attn_mask = pad_mask if attn_mask is None else attn_mask + pad_mask

        # 乘 √d_model：让 embedding 的尺度与位置编码、残差同量级
        h = self.drop(self.pos(self.embed(x) * math.sqrt(self.d_model)))
        for block in self.blocks:
            h = block(h, attn_mask)
        return self.fc(self.norm_f(h))          # (B, T, V)


def load_checkpoint(path, device="cpu"):
    """
    从 checkpoint 恢复模型和词表，供输入法等下游脚本复用：
        from language_model_transformer import load_checkpoint
        model, char2idx, idx2char = load_checkpoint("best_model_transformer.pt")
    """
    ckpt = torch.load(path, map_location=device)
    cfg = ckpt["args"]
    model = LM(
        vocab_size=len(ckpt["char2idx"]),
        d_model=cfg["d_model"],
        nhead=cfg["nhead"],
        num_layers=cfg["num_layers"],
        ffn_dim=cfg["ffn_dim"],
        dropout=0.0,        # 推理阶段关闭 dropout
    ).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return model, ckpt["char2idx"], ckpt["idx2char"]


# ─────────────────────────── mask 自检 ───────────────────────────

def self_test():
    """
    不做训练，只验证 mask 是否真的生效 —— 判断依据是"改掉未来字，前缀输出变不变"。

    第 t 个位置的输出只应该依赖第 0..t 个输入：
      1. 开因果 mask：把 t 之后的字随便改，前缀 logits 必须完全不变；
      2. 关因果 mask：同样的改动会让前缀 logits 改变 —— 这就是信息泄漏，
         训练时 loss 会掉得很漂亮，推理时却拿不到未来字，模型直接废掉；
      3. padding mask：把被标记为 padding 的位置整段换掉，有效位置输出不应变。
    """
    torch.manual_seed(0)
    device = torch.device("cpu")
    V, B, T = 50, 4, 12
    model = LM(vocab_size=V, d_model=32, nhead=4, num_layers=2, ffn_dim=64,
               dropout=0.0).to(device)
    model.eval()

    x = torch.randint(0, V, (B, T))
    with torch.no_grad():
        base_causal = model(x, causal=True)
        base_leak = model(x, causal=False)
        default = model(x)      # 默认必须等价于 causal=True

    default_ok = torch.allclose(default, base_causal, atol=0.0)

    t = T // 2
    changed = x.clone()
    changed[:, t + 1:] = torch.randint(0, V, (B, T - t - 1))
    with torch.no_grad():
        new_causal = model(changed, causal=True)
        new_leak = model(changed, causal=False)

    prefix_same = torch.allclose(base_causal[:, :t + 1], new_causal[:, :t + 1], atol=1e-6)
    leak_happens = not torch.allclose(base_leak[:, :t + 1], new_leak[:, :t + 1], atol=1e-6)

    pad = torch.zeros(B, T, dtype=torch.bool)
    pad[:, -2:] = True                      # 右侧两个位置当作 padding
    x2 = x.clone()
    x2[:, -2:] = torch.randint(0, V, (B, 2))
    with torch.no_grad():
        # 用双向（causal=False）才能看出 padding mask 的作用：
        # 因果模式下，右侧 padding 本来就在"未来"，光靠因果 mask 就看不到了。
        p1 = model(x, key_padding_mask=pad, causal=False)
        p2 = model(x2, key_padding_mask=pad, causal=False)
    pad_ok = torch.allclose(p1[:, :-2], p2[:, :-2], atol=1e-6)

    print("Transformer mask 自检")
    print(f"  [{'PASS' if prefix_same else 'FAIL'}] 因果 mask 开启：改动未来字，前缀 logits 不变")
    print(f"  [{'PASS' if leak_happens else 'FAIL'}] 因果 mask 关闭：改动未来字，前缀 logits 变化（说明 mask 确实在起作用）")
    print(f"  [{'PASS' if pad_ok else 'FAIL'}] padding mask：被屏蔽位置的内容不影响有效位置")
    print(f"  [{'PASS' if default_ok else 'FAIL'}] 默认 forward 等价于 causal=True")

    if not (prefix_same and leak_happens and pad_ok and default_ok):
        raise SystemExit("mask 自检失败")
    print("全部通过。")


# ─────────────────────────── 训练 / 评估 ───────────────────────────

def run_epoch(model, loader, criterion, optimizer, device, train=True, grad_clip=0.0,
              log_every=0, log_file="", epoch=0, t0=None):
    model.train(train)
    total_loss = 0.0
    total_tokens = 0
    n_steps = len(loader)

    for step, (x, y) in enumerate(loader, 1):
        x, y = x.to(device), y.to(device)
        logits = model(x)
        loss = criterion(logits.reshape(-1, logits.size(-1)), y.reshape(-1))

        if train:
            optimizer.zero_grad()
            loss.backward()
            # Transformer 比 RNN 更容易出现梯度尖峰，裁剪一下更稳（0 = 关闭）
            if grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()

            # 每 log_every 步打一行进度，长训练时能看出它到底在不在走
            if log_every and step % log_every == 0:
                elapsed = (time.time() - t0) / 60 if t0 else 0.0
                msg = (f"  epoch {epoch}  step {step:>6}/{n_steps}  "
                       f"loss {loss.item():.4f}  elapsed {elapsed:5.1f} min")
                print(msg, flush=True)
                if log_file:
                    with open(log_file, "a", encoding="utf-8") as f:
                        f.write(msg + "\n")

        total_loss += loss.item() * y.numel()
        total_tokens += y.numel()

    avg_loss = total_loss / total_tokens
    ppl = math.exp(avg_loss)
    return avg_loss, ppl


# ─────────────────────────── 主函数 ───────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs",     type=int,   default=20)
    parser.add_argument("--seq_len",    type=int,   default=64)
    parser.add_argument("--stride",     type=int,   default=1, help="切块间隔：1=密集重叠，seq_len=不重叠")
    parser.add_argument("--batch_size", type=int,   default=128)
    # ── Transformer 结构超参 ──
    parser.add_argument("--d_model",    type=int,   default=256, help="模型维度（每个字的向量长度）")
    parser.add_argument("--nhead",      type=int,   default=8,   help="注意力头数，需能整除 d_model")
    parser.add_argument("--num_layers", type=int,   default=4,   help="Transformer block 层数")
    parser.add_argument("--ffn_dim",    type=int,   default=1024, help="前馈网络隐层维度，通常是 4*d_model")
    parser.add_argument("--dropout",    type=float, default=0.1)
    parser.add_argument("--lr",         type=float, default=3e-4)
    parser.add_argument("--grad_clip",  type=float, default=1.0,  help="梯度裁剪阈值，0=关闭")
    parser.add_argument("--val_ratio",  type=float, default=0.05)
    parser.add_argument("--max_vocab",  type=int,   default=4000, help="字表最大字数，0=不限")
    parser.add_argument("--seed",       type=int,   default=42)
    parser.add_argument("--corpus",     default="*.txt")
    parser.add_argument("--save",       default="best_model_transformer.pt")
    parser.add_argument("--resume",     default="", help="从 checkpoint 续跑，如 --resume best_model_transformer.pt")
    parser.add_argument("--log_every",  type=int,   default=0, help="每 N 个 batch 打印一次进度，0=关闭")
    parser.add_argument("--log_file",   default="", help="进度同时追加写入该文件")
    parser.add_argument("--self_test",  action="store_true", help="只跑 mask 自检，不训练")
    args = parser.parse_args()

    if args.self_test:
        self_test()
        return

    # 续跑：先读 checkpoint 里存的结构超参，保证重建的模型和权重对得上
    resume_ckpt = None
    if args.resume:
        resume_ckpt = torch.load(args.resume, map_location="cpu")
        saved = resume_ckpt.get("args", {})
        for key in ("d_model", "nhead", "num_layers", "ffn_dim", "seq_len", "max_vocab"):
            if key in saved:
                setattr(args, key, saved[key])
        print(f"续跑: {args.resume}  (已完成 {resume_ckpt.get('epoch', '?')} 轮, "
              f"best val PPL {resume_ckpt.get('best_val_ppl', float('inf')):.2f})")

    torch.manual_seed(args.seed)
    random.seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}  model: Transformer ({args.num_layers} 层, "
          f"d_model={args.d_model}, {args.nhead} 头, ffn_dim={args.ffn_dim})")
    print("因果 mask: 开启（第 t 个位置只能看到 0..t）")

    # 数据准备
    paths = corpus_paths(args.corpus)
    if not paths:
        raise FileNotFoundError(
            f"没找到匹配 {args.corpus!r} 的文本文件。\n"
            f"  当前目录: {os.getcwd()}\n"
            f"  脚本目录: {os.path.dirname(os.path.abspath(__file__))}\n"
            "  相对路径先按当前目录解析，找不到才回退到脚本目录；也可以直接传绝对路径。"
        )
    text = load_corpus(args.corpus)
    print(f"语料文件: {', '.join(os.path.basename(p) for p in paths)}")
    print(f"语料字符数: {len(text):,}")

    char2idx, idx2char = build_vocab(text, args.max_vocab or None)
    vocab_size = len(char2idx)
    print(f"词表大小: {vocab_size}")

    lines = text.splitlines()
    random.shuffle(lines)
    split = int(len(lines) * (1 - args.val_ratio))
    train_text = "\n".join(lines[:split])
    val_text   = "\n".join(lines[split:])

    train_ds = CharDataset(train_text, char2idx, args.seq_len, args.stride)
    val_ds   = CharDataset(val_text,   char2idx, args.seq_len, args.stride)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, drop_last=True)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size, shuffle=True, drop_last=True)

    # 模型
    model = LM(
        vocab_size=vocab_size,
        d_model=args.d_model,
        nhead=args.nhead,
        num_layers=args.num_layers,
        ffn_dim=args.ffn_dim,
        dropout=args.dropout,
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"模型参数量: {total_params:,}")

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    start_epoch = 1
    best_val_ppl = float("inf")
    if resume_ckpt is not None:
        model.load_state_dict(resume_ckpt["model_state"])
        if "optimizer_state" in resume_ckpt:
            optimizer.load_state_dict(resume_ckpt["optimizer_state"])
        start_epoch = int(resume_ckpt.get("epoch", 0)) + 1
        best_val_ppl = float(resume_ckpt.get("best_val_ppl", float("inf")))

    print(f"\n{'Epoch':>6}  {'Train Loss':>10}  {'Train PPL':>10}  {'Val Loss':>10}  {'Val PPL':>10}")
    print("-" * 56)

    t0 = time.time()
    for epoch in range(start_epoch, args.epochs + 1):
        tr_loss, tr_ppl = run_epoch(model, train_loader, criterion, optimizer, device,
                                    train=True, grad_clip=args.grad_clip,
                                    log_every=args.log_every, log_file=args.log_file,
                                    epoch=epoch, t0=t0)
        with torch.no_grad():
            va_loss, va_ppl = run_epoch(model, val_loader, criterion, optimizer, device,
                                        train=False, grad_clip=args.grad_clip)

        marker = "  *" if va_ppl < best_val_ppl else ""
        if va_ppl < best_val_ppl:
            best_val_ppl = va_ppl
            torch.save({
                "model_state": model.state_dict(),
                # optimizer / 当前轮次 / 最优 PPL 一并存下，--resume 才接得上
                "optimizer_state": optimizer.state_dict(),
                "epoch": epoch,
                "best_val_ppl": best_val_ppl,
                "char2idx": char2idx,
                "idx2char": idx2char,
                # 多存 arch/model 两个字段，方便下游区分 checkpoint 类型
                "args": dict(vars(args), model="transformer", arch="transformer"),
            }, args.save)

        print(f"{epoch:>6}  {tr_loss:>10.4f}  {tr_ppl:>10.2f}  {va_loss:>10.4f}  {va_ppl:>10.2f}{marker}")

    if start_epoch > args.epochs:
        print(f"\n没有需要训练的轮次（checkpoint 已完成 {start_epoch - 1} 轮，--epochs={args.epochs}）。")
    else:
        print(f"\n训练完成。最佳验证 PPL: {best_val_ppl:.2f}  已保存至 {args.save}  "
              f"（本次用时 {(time.time() - t0) / 60:.1f} 分钟）")


if __name__ == "__main__":
    main()
