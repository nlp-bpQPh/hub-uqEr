"""
字符级 Transformer 语言模型训练脚本（自包含版）。

与 RNN/LSTM 版相比的差异：
    1. 循环层换成 TransformerEncoder（自注意力 + 前馈），带因果掩码，保证只看左边。
    2. 补充正弦位置编码 —— 注意力本身对顺序不敏感，没有位置信息等于乱序看字。
    3. 输出层与词向量做权重绑定（tied weights），省掉约一半参数。
    4. 优化器改 AdamW + 线性 warmup + 余弦退火，Transformer 比 RNN 更吃学习率调度。
    5. 全部依赖只有 torch，语料与拼音表就在本目录下，整个文件夹可整体拷走。

用法:
    python language_model.py                        # 默认配置
    python language_model.py --epochs 5 --stride 64 # 快速跑几轮
    python language_model.py --max_chars 200000     # 只用前 20 万字快速试跑
"""

import math
import argparse
import glob
import os
import random

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader


# ─────────────────────────── 数据 ───────────────────────────

def resolve_pattern(pattern, script_dir=None):
    """
    把相对路径的语料 glob 解析到脚本所在目录，避免「换个目录运行就找不到 .txt」。
    绝对路径原样返回。
    """
    if os.path.isabs(pattern):
        return pattern
    base = script_dir or os.path.dirname(os.path.abspath(__file__))
    return os.path.join(base, pattern)


def load_corpus(pattern="*.txt"):
    texts = []
    for path in sorted(glob.glob(pattern)):
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

    stride 控制切块间隔（0 表示等于 seq_len）：
        stride=1      —— 每个字符位置都当一个样本（样本大量重叠，语料能切出上百万个样本）
        stride=seq_len —— 不重叠切块（标准做法，样本数 = 语料长度 / seq_len）
    语料一大，stride=1 会让一轮训练变得极慢，这不是模型慢，而是样本数被放大了 seq_len 倍。
    Transformer 每个位置都能并行看全段，样本重叠几乎不带来额外信息，因此默认不重叠。
    """

    def __init__(self, text, char2idx, seq_len, stride=0):
        self.seq_len = seq_len
        self.stride = max(1, stride or seq_len)
        ids = [char2idx[c] for c in text if c in char2idx]
        self.data = torch.tensor(ids, dtype=torch.long)

    def __len__(self):
        return max(0, (len(self.data) - self.seq_len - 1) // self.stride + 1)

    def __getitem__(self, idx):
        start = idx * self.stride
        x = self.data[start: start + self.seq_len]
        y = self.data[start + 1: start + self.seq_len + 1]
        return x, y


# ─────────────────────────── 模型 ───────────────────────────

class PositionalEncoding(nn.Module):
    """
    正弦位置编码：把「第几个字」编码进向量，注册为 buffer（不参与训练）。
    注意力机制本身对位置无感，缺了这一步模型只知道有哪些字，不知道谁在前。
    """

    def __init__(self, d_model, max_len=1024, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term[: pe[:, 1::2].shape[1]])
        self.register_buffer("pe", pe.unsqueeze(0), persistent=False)   # (1, max_len, d_model)

    def forward(self, x):
        return self.dropout(x + self.pe[:, : x.size(1)])


class LM(nn.Module):
    """
    字符级 Transformer 语言模型。

    接口约定与循环版完全一致：输入 (B, T) 的字符 id，输出 (B, T, V) 的 logits，
    这样训练脚本、束搜索解码都不需要感知内部到底是 RNN 还是 Transformer。
    """

    def __init__(self, vocab_size, embed_dim, hidden_dim, num_layers, model_type, dropout,
                 nhead=4, max_len=1024, tie_weights=1):
        super().__init__()
        assert embed_dim % nhead == 0, "embed_dim 必须能被 nhead 整除（每个头分到的维度要相等）"
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.max_len = max_len

        # 1) 字向量 + 位置编码
        self.embed = nn.Embedding(vocab_size, embed_dim)
        self.pos = PositionalEncoding(embed_dim, max_len=max_len, dropout=dropout)

        # 2) Transformer 编码器（norm_first 更稳，激活用 gelu）
        layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=nhead,
            dim_feedforward=hidden_dim,   # 注意：这里的 hidden_dim 是前馈层宽度，不是隐状态宽度
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        try:
            # 新版 PyTorch 支持关掉 nested tensor，避免带 mask 时的报警告
            self.encoder = nn.TransformerEncoder(layer, num_layers=num_layers,
                                                 enable_nested_tensor=False)
        except TypeError:
            self.encoder = nn.TransformerEncoder(layer, num_layers=num_layers)

        # 3) 收尾归一化 + 输出层
        self.norm = nn.LayerNorm(embed_dim)
        self.fc = nn.Linear(embed_dim, vocab_size)
        if tie_weights:
            # 权重绑定：输入查表和输出打分共用同一张向量表，直接省掉 V×d 个参数
            self.fc.weight = self.embed.weight
            # 绑定后 logits 的量级完全由词向量决定，默认 N(0,1) 会算出 ±60 的 logits，
            # 首轮 loss 直接飙到几十；按 GPT-2 的做法改成小方差初始化
            nn.init.normal_(self.embed.weight, mean=0.0, std=0.02)

        self._mask_cache = {}

    def _causal_mask(self, T, device):
        """上三角遮罩：第 t 个位置只能看到 0..t，防止训练时偷看后面的字。"""
        key = (T, device)
        mask = self._mask_cache.get(key)
        if mask is None:
            mask = torch.triu(
                torch.full((T, T), float("-inf"), device=device), diagonal=1
            )
            self._mask_cache[key] = mask
        return mask

    def forward(self, x):
        T = x.size(1)
        e = self.pos(self.embed(x))                            # (B, T, d)
        out = self.encoder(e, mask=self._causal_mask(T, x.device))
        logits = self.fc(self.norm(out))                       # (B, T, V)
        return logits


# ─────────────────────────── 训练 / 评估 ───────────────────────────

def run_epoch(model, loader, criterion, optimizer, device, train=True,
              clip_grad=1.0, scheduler=None, log_every=0, epoch=0):
    model.train(train)
    total_loss = 0.0
    total_tokens = 0

    for step, (x, y) in enumerate(loader, 1):
        x, y = x.to(device), y.to(device)
        logits = model(x)
        loss = criterion(logits.reshape(-1, logits.size(-1)), y.reshape(-1))

        if train:
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            # 梯度裁剪：深一点的网络偶发大梯度会把权重打飞，裁到 1.0 很常见
            if clip_grad:
                nn.utils.clip_grad_norm_(model.parameters(), clip_grad)
            optimizer.step()
            if scheduler is not None:
                scheduler.step()

        total_loss += loss.item() * y.numel()
        total_tokens += y.numel()

        if log_every and step % log_every == 0:
            print(f"  epoch {epoch} step {step}/{len(loader)}  "
                  f"loss={total_loss / total_tokens:.4f}", flush=True)

    if total_tokens == 0:
        raise ValueError(
            "该数据集切不出任何样本：语料太小，或 seq_len / val_ratio 设置不合适。"
            "请增大 --max_chars 或减小 --seq_len。"
        )

    avg_loss = total_loss / total_tokens
    ppl = math.exp(avg_loss)
    return avg_loss, ppl


def build_scheduler(optimizer, total_steps, warmup_steps):
    """线性 warmup + 余弦退火：Transformer 前期步长大容易崩，先热身再降速。"""
    warmup = max(1, warmup_steps)

    def lr_lambda(step):
        if step < warmup:
            return step / warmup
        progress = (step - warmup) / max(1, total_steps - warmup)
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


# ─────────────────────────── 主函数 ───────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs",     type=int,   default=10)
    parser.add_argument("--seq_len",    type=int,   default=128)
    parser.add_argument("--stride",     type=int,   default=0, help="切块间隔：0=等于seq_len（不重叠），1=密集重叠")
    parser.add_argument("--batch_size", type=int,   default=64)
    parser.add_argument("--embed_dim",  type=int,   default=128, help="d_model，必须能被 nhead 整除")
    parser.add_argument("--hidden_dim", type=int,   default=256, help="前馈层宽度（Transformer 里不是隐状态宽度）")
    parser.add_argument("--num_layers", type=int,   default=2)
    parser.add_argument("--nhead",      type=int,   default=4,   help="注意力头数")
    parser.add_argument("--dropout",    type=float, default=0.1)
    parser.add_argument("--lr",         type=float, default=3e-4)
    parser.add_argument("--warmup_ratio", type=float, default=0.05, help="warmup 占总步数比例")
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--clip_grad",  type=float, default=1.0, help="梯度裁剪阈值，0=不裁")
    parser.add_argument("--val_ratio",  type=float, default=0.05)
    parser.add_argument("--max_vocab",  type=int,   default=4000, help="字表最大字数，0=不限")
    parser.add_argument("--max_len",    type=int,   default=1024, help="位置编码支持的最大长度")
    parser.add_argument("--tie_weights", type=int,  default=1,   help="1=输出层与词向量权重绑定，0=分开")
    parser.add_argument("--max_chars",  type=int,   default=0,   help="只用语料前 N 个字符，0=全部")
    parser.add_argument("--seed",       type=int,   default=42)
    parser.add_argument("--corpus",     default="*.txt")
    parser.add_argument("--save",       default="best_model.pt")
    parser.add_argument("--log_every",  type=int,   default=100, help="每多少步打印一次进度，0=不打印")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    random.seed(args.seed)

    # 所有相对路径都相对脚本所在目录解析，方便整个文件夹搬走运行
    script_dir = os.path.dirname(os.path.abspath(__file__))
    args.corpus = resolve_pattern(args.corpus, script_dir)
    if not os.path.isabs(args.save):
        args.save = os.path.join(script_dir, args.save)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}  model: TRANSFORMER")

    # 数据准备
    text = load_corpus(args.corpus)
    if not text:
        raise FileNotFoundError(
            f"未能加载语料，请确认 glob 模式能匹配到 .txt 文件：{args.corpus}"
        )
    if args.max_chars and len(text) > args.max_chars:
        text = text[: args.max_chars]
    print(f"语料字符数: {len(text):,}")

    char2idx, idx2char = build_vocab(text, args.max_vocab or None)
    vocab_size = len(char2idx)
    print(f"词表大小: {vocab_size}")

    lines = text.splitlines()
    random.shuffle(lines)
    split = int(len(lines) * (1 - args.val_ratio))
    train_text = "\n".join(lines[:split])
    val_text   = "\n".join(lines[split:])

    stride = args.stride if args.stride > 0 else args.seq_len
    train_ds = CharDataset(train_text, char2idx, args.seq_len, stride)
    val_ds   = CharDataset(val_text,   char2idx, args.seq_len, stride)
    print(f"训练样本数: {len(train_ds):,}  验证样本数: {len(val_ds):,}（stride={stride}）")

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, drop_last=True)
    # 验证集不丢尾批：小语料时验证集可能不足一个 batch，drop_last 会让验证集变空
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size, shuffle=False, drop_last=False)

    # 模型
    model = LM(
        vocab_size=vocab_size,
        embed_dim=args.embed_dim,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        model_type="transformer",
        dropout=args.dropout,
        nhead=args.nhead,
        max_len=max(args.max_len, args.seq_len),
        tie_weights=args.tie_weights,
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"模型参数量: {total_params:,}"
          f"{'（已做权重绑定）' if args.tie_weights else ''}")

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    total_steps = max(1, len(train_loader) * args.epochs)
    scheduler = build_scheduler(optimizer, total_steps,
                                int(total_steps * args.warmup_ratio))

    best_val_ppl = float("inf")

    print(f"\n{'Epoch':>6}  {'Train Loss':>10}  {'Train PPL':>10}  "
          f"{'Val Loss':>10}  {'Val PPL':>10}  {'LR':>9}")
    print("-" * 68)

    for epoch in range(1, args.epochs + 1):
        cur_lr = optimizer.param_groups[0]["lr"]
        tr_loss, tr_ppl = run_epoch(
            model, train_loader, criterion, optimizer, device, train=True,
            clip_grad=args.clip_grad, scheduler=scheduler,
            log_every=args.log_every, epoch=epoch,
        )
        with torch.no_grad():
            va_loss, va_ppl = run_epoch(
                model, val_loader, criterion, optimizer, device, train=False
            )

        marker = "  *" if va_ppl < best_val_ppl else ""
        if va_ppl < best_val_ppl:
            best_val_ppl = va_ppl
            torch.save({
                "model_state": model.state_dict(),
                "char2idx": char2idx,
                "idx2char": idx2char,
                "args": vars(args),
            }, args.save)

        print(f"{epoch:>6}  {tr_loss:>10.4f}  {tr_ppl:>10.2f}  "
              f"{va_loss:>10.4f}  {va_ppl:>10.2f}  {cur_lr:>9.2e}{marker}")

    print(f"\n训练完成。最佳验证 PPL: {best_val_ppl:.2f}  已保存至 {args.save}")


if __name__ == "__main__":
    main()
