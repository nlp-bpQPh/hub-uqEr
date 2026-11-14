"""
第四周作业：训练基于 Transformer 的语言模型（字符级、Decoder-only 自回归）。

与课程 RNN/LSTM 版 language_model.py 使用同一份语料、同一种切样本方式、
同一个 PPL 口径，可以直接和课程基准对比：
    1-gram 543.5 / 2-gram 88.7 / 3-gram 247.1 / RNN 40.83 / LSTM 38.32

模型结构（对应《语言模型》PPT 第 19-26 页）：
    Token Embedding + 可学习位置编码
    -> N x [ Pre-LN -> Masked Self-Attention -> 残差
             Pre-LN -> FFN(先升维4倍再降回)   -> 残差 ]
    -> 最终 LayerNorm -> Linear（输出每个字的 logits）

用法:
    python homework_transformer_lm.py --epochs 8
    python homework_transformer_lm.py --prompt "今天天气" --max_new_tokens 50
"""

import math
import argparse
import glob
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader


# ─────────────────────────── 数据（与课程脚本保持一致，保证可对比）───────────────────────────

def load_corpus(pattern="*.txt"):
    texts = []
    for path in glob.glob(pattern):
        with open(path, encoding="utf-8", errors="ignore") as f:
            texts.append(f.read())
    return "".join(texts)


def build_vocab(text, max_vocab=None):
    from collections import Counter
    counter = Counter(text)
    chars = [c for c, _ in counter.most_common(max_vocab)] if max_vocab else sorted(set(text))
    char2idx = {c: i for i, c in enumerate(chars)}
    idx2char = {i: c for c, i in char2idx.items()}
    return char2idx, idx2char


class CharDataset(Dataset):
    """输入 x = 连续 seq_len 个字符，目标 y = 往后错一位（预测下一个字）。"""

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


# ─────────────────────────── 手写 Transformer 组件 ───────────────────────────

class CausalSelfAttention(nn.Module):
    """
    多头因果自注意力（PPT 第 23-24 页）：
        Q,K,V = 三个线性投影      ——「注意力管看谁」
        score = QK^T / sqrt(d_k)  —— 除 sqrt(d_k)：点积方差随维度增大，不除 softmax 会饱和
        mask   = 上三角置 -inf    —— 因果掩码：只能看当前及之前的字，不能偷看答案
        out    = softmax(score) @ V，再拼回多头过输出投影
    """

    def __init__(self, d_model, n_head, dropout=0.1):
        super().__init__()
        assert d_model % n_head == 0, "d_model 必须能被 n_head 整除"
        self.n_head = n_head
        self.d_head = d_model // n_head
        # 三个独立的 W_Q/W_K/W_V 拼成一个大矩阵（多头不是省参数）
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.proj = nn.Linear(d_model, d_model)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        B, T, C = x.shape
        q, k, v = self.qkv(x).chunk(3, dim=-1)
        # (B, T, C) -> (B, n_head, T, d_head)：把多头拆开并行算
        q = q.view(B, T, self.n_head, self.d_head).transpose(1, 2)
        k = k.view(B, T, self.n_head, self.d_head).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.d_head).transpose(1, 2)

        # QK^T / sqrt(d_k)：(B, n_head, T, T)，每行是当前字对全部字的权重分布
        att = (q @ k.transpose(-2, -1)) / math.sqrt(self.d_head)
        # 因果掩码：位置 i 只允许看 <= i（Decoder-only 与 BERT 双向注意力的本质区别）
        mask = torch.triu(torch.ones(T, T, dtype=torch.bool, device=x.device), diagonal=1)
        att = att.masked_fill(mask, float("-inf"))
        att = F.softmax(att, dim=-1)          # 每行之和为 1
        att = self.drop(att)
        y = att @ v                           # (B, n_head, T, d_head)
        y = y.transpose(1, 2).contiguous().view(B, T, C)   # 拼回多头
        return self.proj(self.drop(y))


class Block(nn.Module):
    """
    Pre-LN 结构（PPT 第 26 页：BERT 用 Post-LN，后续很多模型改 Pre-LN 更稳定）：
        x = x + Attn(LN(x))    先归一化再进子层
        x = x + FFN(LN(x))
    FFN 先升维 4 倍再降回（BERT-base: 768 -> 3072 -> 768）。
    """

    def __init__(self, d_model, n_head, ffn_mult=4, dropout=0.1):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)      # NLP 固定用 LayerNorm，不用 BatchNorm
        self.attn = CausalSelfAttention(d_model, n_head, dropout)
        self.ln2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, ffn_mult * d_model),
            nn.GELU(),                         # BERT/现代 Transformer 的激活函数
            nn.Linear(ffn_mult * d_model, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.ffn(self.ln2(x))
        return x


class TransformerLM(nn.Module):
    """
    Decoder-only 语言模型。PPT 第 25 页的 BERT 输入是 Token+Position+Segment
    三种 Embedding 相加；生成式 LM 没有 句对 任务，只留 Token + Position。
    """

    def __init__(self, vocab_size, d_model=192, n_head=4, n_layer=4,
                 ffn_mult=4, seq_len=64, dropout=0.1):
        super().__init__()
        self.tok_emb = nn.Embedding(vocab_size, d_model)
        # 可学习位置编码：自注意力本身是集合运算没有顺序概念（PPT 第 22 页）
        self.pos_emb = nn.Embedding(seq_len, d_model)
        self.drop = nn.Dropout(dropout)
        self.blocks = nn.ModuleList([
            Block(d_model, n_head, ffn_mult, dropout) for _ in range(n_layer)
        ])
        self.ln_f = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size, bias=False)
        self.seq_len = seq_len
        # 权重绑定：输出层和词嵌入共用一张表，省 vocab_size x d_model 个参数
        self.head.weight = self.tok_emb.weight
        self.apply(self._init_weights)   # GPT 风格小方差初始化，见下方注释

    @staticmethod
    def _init_weights(m):
        # nn.Embedding 默认方差 1、Linear 默认 1/sqrt(fan_in)。配权重绑定后初始 logits
        # 尺度 ≈ sqrt(d_model)，softmax 会饱和、初始 loss 远大于 ln(V)（冒烟实测 50+）。
        # GPT 系模型统一用 std=0.02，让初始 logits 接近 0，训练从一个温和的起点开始。
        if isinstance(m, (nn.Linear, nn.Embedding)):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.LayerNorm):
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)

    def forward(self, x):
        B, T = x.shape
        pos = torch.arange(T, device=x.device)
        h = self.drop(self.tok_emb(x) + self.pos_emb(pos))
        for blk in self.blocks:
            h = blk(h)
        return self.head(self.ln_f(h))       # (B, T, V) 返回 logits，softmax 交给 CrossEntropyLoss


# ─────────────────────────── 训练 / 评估（与课程脚本同口径）───────────────────────────

def run_epoch(model, loader, criterion, optimizer, device, train=True, scheduler=None):
    model.train(train)
    total_loss, total_tokens = 0.0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        loss = criterion(logits.reshape(-1, logits.size(-1)), y.reshape(-1))
        if train:
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)  # 梯度裁剪：防止训练早期爆炸
            optimizer.step()
            if scheduler is not None:
                scheduler.step()   # 逐步调度：warmup 才有意义
        total_loss += loss.item() * y.numel()
        total_tokens += y.numel()
    avg_loss = total_loss / total_tokens
    return avg_loss, math.exp(avg_loss)      # PPL = exp(平均交叉熵)


@torch.no_grad()
def generate(model, char2idx, idx2char, prompt, max_new_tokens=50,
             temperature=0.8, top_k=10, device="cpu"):
    """自回归续写：每次用最后 seq_len 个字预测下一个字，温度+top-k 采样增加多样性。"""
    model.eval()
    ids = [char2idx[c] for c in prompt if c in char2idx]
    if not ids:
        return prompt + "（提示词里的字都不在字表中）"
    for _ in range(max_new_tokens):
        ctx = torch.tensor([ids[-model.seq_len:]], device=device)   # 超长就截最右 seq_len 个字
        logits = model(ctx)[0, -1] / temperature
        if top_k:
            v, _ = torch.topk(logits, top_k)
            logits[logits < v[-1]] = float("-inf")
        probs = F.softmax(logits, dim=-1)
        nxt = torch.multinomial(probs, 1).item()
        ids.append(nxt)
        ch = idx2char[nxt]
        if ch == "\n":
            break
    return "".join(idx2char[i] for i in ids)


# ─────────────────────────── 主函数 ───────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs",     type=int,   default=8)
    parser.add_argument("--seq_len",    type=int,   default=64)
    parser.add_argument("--stride",     type=int,   default=32, help="切块间隔：32=半重叠，兼顾样本量与训练时长")
    parser.add_argument("--batch_size", type=int,   default=128)
    parser.add_argument("--d_model",    type=int,   default=192)
    parser.add_argument("--n_head",     type=int,   default=4)
    parser.add_argument("--n_layer",    type=int,   default=4)
    parser.add_argument("--dropout",    type=float, default=0.1)
    parser.add_argument("--lr",         type=float, default=1e-3)
    parser.add_argument("--warmup",     type=int,   default=400, help="线性 warmup 步数，Transformer 训练稳定的关键")
    parser.add_argument("--val_ratio",  type=float, default=0.05)
    parser.add_argument("--max_vocab",  type=int,   default=4000)
    parser.add_argument("--seed",       type=int,   default=42)
    parser.add_argument("--corpus",     default="*.txt")
    parser.add_argument("--save",       default="best_model_transformer.pt")
    parser.add_argument("--prompt",     default=None, help="只续写不训练：python homework_transformer_lm.py --prompt 今天")
    parser.add_argument("--max_new_tokens", type=int, default=50)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ckpt = torch.load(args.save, map_location=device) if args.prompt else None
    if args.prompt and ckpt is None and not glob.glob(args.corpus):
        raise FileNotFoundError("续写模式需要已训练的模型文件（--save 指向）")

    if args.prompt and ckpt is not None:
        char2idx = ckpt["char2idx"]
        idx2char = {int(k): v for k, v in ckpt["idx2char"].items()}
        cfg = ckpt["args"]
        model = TransformerLM(
            vocab_size=len(char2idx), d_model=cfg["d_model"], n_head=cfg["n_head"],
            n_layer=cfg["n_layer"], dropout=0.0, seq_len=cfg["seq_len"],
        ).to(device)
        model.load_state_dict(ckpt["model_state"])
        print(generate(model, char2idx, idx2char, args.prompt, args.max_new_tokens, device=device))
        return

    print(f"device: {device}  model: TransformerLM(d_model={args.d_model}, n_head={args.n_head}, n_layer={args.n_layer})")

    # 数据准备（与课程 RNN/LSTM 版完全一致：按行打乱再切 5% 做验证）
    text = load_corpus(args.corpus)
    if not text:
        raise FileNotFoundError("未找到任何 .txt 文件，请确认路径正确。")
    print(f"语料字符数: {len(text):,}")

    char2idx, idx2char = build_vocab(text, args.max_vocab or None)
    print(f"词表大小: {len(char2idx)}")

    lines = text.splitlines()
    random.shuffle(lines)
    split = int(len(lines) * (1 - args.val_ratio))
    train_text = "\n".join(lines[:split])
    val_text   = "\n".join(lines[split:])

    train_ds = CharDataset(train_text, char2idx, args.seq_len, args.stride)
    val_ds   = CharDataset(val_text,   char2idx, args.seq_len, args.stride)
    print(f"训练样本数: {len(train_ds):,}  验证样本数: {len(val_ds):,}")

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, drop_last=True)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size, shuffle=True, drop_last=False)

    model = TransformerLM(
        vocab_size=len(char2idx), d_model=args.d_model, n_head=args.n_head,
        n_layer=args.n_layer, dropout=args.dropout, seq_len=args.seq_len,
    ).to(device)
    print(f"模型参数量: {sum(p.numel() for p in model.parameters()):,}")

    criterion = nn.CrossEntropyLoss()
    # Transformer 专用：线性 warmup + 余弦衰减。
    # 没有 warmup 时 Adam 前几十步会把注意力 logits 推爆（本作业冒烟时实测初期 loss 飙到 50+），
    # 这也是 PPT「学习率调度」里 Warmup 的真实用途。
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    steps_per_epoch = len(train_loader)
    total_steps = max(1, steps_per_epoch * args.epochs)
    warmup_steps = min(args.warmup, total_steps // 5)

    def lr_lambda(step):
        if step < warmup_steps:                       # 线性升温
            return (step + 1) / max(1, warmup_steps)
        # 余弦退火到 0：兼顾探索与收敛（Week2 讲过的两种调度这里一起用上）
        import math as _m
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1.0 + _m.cos(_m.pi * min(1.0, progress)))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    best_val_ppl = float("inf")
    print(f"\n{'Epoch':>6}  {'Train Loss':>10}  {'Train PPL':>10}  {'Val Loss':>10}  {'Val PPL':>10}")
    print("-" * 56)

    for epoch in range(1, args.epochs + 1):
        tr_loss, tr_ppl = run_epoch(model, train_loader, criterion, optimizer, device, train=True, scheduler=scheduler)
        with torch.no_grad():
            va_loss, va_ppl = run_epoch(model, val_loader, criterion, optimizer, device, train=False)

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
    print("课程基准对比: 1-gram 543.5 | 2-gram 88.7 | 3-gram 247.1 | RNN 40.83 | LSTM 38.32")

    # 用验证集开头的话做几个续写样例
    model.load_state_dict(torch.load(args.save, map_location=device)["model_state"])
    for p in ["为什么", "感冒了", "如何学好"]:
        print(f"\n[{p}] -> {generate(model, char2idx, idx2char, p, 40, device=device)}")


if __name__ == "__main__":
    main()
