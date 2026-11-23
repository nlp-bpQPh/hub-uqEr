import argparse
import glob
import math
import os
import random
from collections import Counter

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter


# ---------------------------------------------------------------------
# 数据处理
# ---------------------------------------------------------------------

def load_corpus(pattern="*.txt"):
    """读取匹配 pattern 的所有文本文件。"""
    paths = sorted(glob.glob(pattern))
    if not paths:
        raise FileNotFoundError(
            f"没有找到语料文件：{pattern}。请在语料文件所在目录运行，或通过 --corpus 指定路径。"
        )

    texts = []
    for path in paths:
        if os.path.isfile(path):
            with open(path, "r", encoding="utf-8", errors="ignore") as f:
                texts.append(f.read())

    text = "\n".join(t for t in texts if t)
    if not text:
        raise ValueError("语料文件存在，但没有读取到有效文本。")
    print(f"读取语料文件数：{len(paths)}")
    return text


def build_vocab(text, max_vocab=None):
    """按频率构建字符词表；max_vocab=None 表示保留全部字符。"""
    counter = Counter(text)
    if max_vocab and max_vocab > 0:
        chars = [c for c, _ in counter.most_common(max_vocab)]
    else:
        chars = sorted(counter.keys())

    if not chars:
        raise ValueError("无法从空语料构建词表。")

    char2idx = {c: i for i, c in enumerate(chars)}
    idx2char = {i: c for c, i in char2idx.items()}
    return char2idx, idx2char


def split_corpus(text, val_ratio, seed):
    """优先按行切分，减少训练样本跨越文档边界的情况。"""
    lines = [line for line in text.splitlines() if line]
    rng = random.Random(seed)

    if len(lines) >= 2:
        rng.shuffle(lines)
        split_at = int(len(lines) * (1.0 - val_ratio))
        split_at = min(max(split_at, 1), len(lines) - 1)
        train_text = "\n".join(lines[:split_at])
        val_text = "\n".join(lines[split_at:])
    else:
        split_at = int(len(text) * (1.0 - val_ratio))
        split_at = min(max(split_at, 1), len(text) - 1)
        train_text = text[:split_at]
        val_text = text[split_at:]

    return train_text, val_text


class CharDataset(Dataset):
    """字符级 next-token 预测数据集。"""

    def __init__(self, text, char2idx, seq_len, stride):
        self.seq_len = seq_len
        self.stride = max(1, stride)
        ids = [char2idx[c] for c in text if c in char2idx]
        self.data = torch.tensor(ids, dtype=torch.long)

    def __len__(self):
        n = len(self.data)
        if n <= self.seq_len:
            return 0
        return (n - self.seq_len - 1) // self.stride + 1

    def __getitem__(self, index):
        start = index * self.stride
        x = self.data[start:start + self.seq_len]
        y = self.data[start + 1:start + self.seq_len + 1]
        return x, y


# ---------------------------------------------------------------------
# Decoder-only Transformer
# ---------------------------------------------------------------------

class PositionalEmbedding(nn.Module):
    def __init__(self, max_len, d_model):
        super().__init__()
        self.position_embedding = nn.Embedding(max_len, d_model)

    def forward(self, x):
        seq_len = x.size(1)
        positions = torch.arange(seq_len, device=x.device).unsqueeze(0)
        return x + self.position_embedding(positions)


class TransformerLM(nn.Module):
    def __init__(
            self,
            vocab_size,
            d_model=256,
            nhead=8,
            num_layers=4,
            dim_feedforward=1024,
            dropout=0.1,
            max_len=256,
    ):
        super().__init__()
        if d_model % nhead != 0:
            raise ValueError(f"d_model ({d_model}) 必须能被 nhead ({nhead}) 整除。")

        self.vocab_size = vocab_size
        self.d_model = d_model
        self.max_len = max_len

        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.position_embedding = PositionalEmbedding(max_len, d_model)
        self.embedding_dropout = nn.Dropout(dropout)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers,
            norm=nn.LayerNorm(d_model),
        )
        self.final_dropout = nn.Dropout(dropout)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)

        # 权重共享
        self.lm_head.weight = self.token_embedding.weight

    def forward(self, input_ids):
        if input_ids.dim() != 2:
            raise ValueError("input_ids 必须是形状为 (batch, seq_len) 的二维张量。")

        seq_len = input_ids.size(1)
        if seq_len > self.max_len:
            raise ValueError(f"输入序列长度 {seq_len} 超过模型最大长度 {self.max_len}。")

        x = self.token_embedding(input_ids) * math.sqrt(self.d_model)
        x = self.position_embedding(x)
        x = self.embedding_dropout(x)

        causal_mask = torch.triu(
            torch.ones(seq_len, seq_len, device=input_ids.device, dtype=torch.bool),
            diagonal=1,
        )
        hidden = self.transformer(x, mask=causal_mask)
        hidden = self.final_dropout(hidden)
        return self.lm_head(hidden)


LM = TransformerLM


# ---------------------------------------------------------------------
# 训练与验证
# ---------------------------------------------------------------------

def run_epoch(model, loader, criterion, optimizer, device, train=True, grad_clip=1.0):
    model.train(train)
    total_loss = 0.0
    total_tokens = 0

    context = torch.enable_grad() if train else torch.no_grad()
    with context:
        for x, y in loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

            logits = model(x)
            loss = criterion(
                logits.reshape(-1, logits.size(-1)),
                y.reshape(-1),
            )

            if train:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                if grad_clip and grad_clip > 0:
                    nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                optimizer.step()

            token_count = y.numel()
            total_loss += loss.item() * token_count
            total_tokens += token_count

    if total_tokens == 0:
        return float("inf"), float("inf")

    avg_loss = total_loss / total_tokens
    ppl = math.exp(min(avg_loss, 50.0))
    return avg_loss, ppl


# ---------------------------------------------------------------------
# --- [加载模型] 1. 新增推理生成函数 ---
# ---------------------------------------------------------------------
@torch.no_grad()
def generate_text(model, start_text, char2idx, idx2char, max_new_tokens=100, device="cpu"):
    """给定开头，自回归生成文本"""
    model.eval()
    # 过滤掉不在词表中的字符
    ids = [char2idx[c] for c in start_text if c in char2idx]
    if not ids:
        # 如果开头完全没有词表中的字，随便给个起始符（比如 id 0）
        ids = [0]

    x = torch.tensor([ids], dtype=torch.long).to(device)

    for _ in range(max_new_tokens):
        # 截断，防止超出模型定义的最大长度
        x_cond = x[:, -model.max_len:]
        logits = model(x_cond)

        # 取最后一个时间步的预测分布
        next_token_logits = logits[0, -1, :]

        # 选择概率最大的词（贪婪搜索）
        next_id = torch.argmax(next_token_logits).item()

        # 追加到序列中
        ids.append(next_id)
        x = torch.tensor([ids], dtype=torch.long).to(device)

    # 将 id 转回文字
    return "".join(idx2char[i] for i in ids)


def main():
    parser = argparse.ArgumentParser(description="使用 Decoder-only Transformer 训练字符级语言模型")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--seq_len", type=int, default=128)
    parser.add_argument("--stride", type=int, default=128)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--d_model", type=int, default=256)
    parser.add_argument("--nhead", type=int, default=8)
    parser.add_argument("--num_layers", type=int, default=4)
    parser.add_argument("--dim_feedforward", type=int, default=1024)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--val_ratio", type=float, default=0.05)
    parser.add_argument("--max_vocab", type=int, default=4000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--corpus", default="*.txt")
    parser.add_argument("--save", default="best_model_transformer.pt")
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--log_dir", type=str, default="runs/lm_experiment", help="TensorBoard日志保存目录")

    # --- [加载模型] 2. 增加推理模式参数 ---
    parser.add_argument("--load_model", type=str, default=None,
                        help="指定加载的模型路径，如果设置此项则跳过训练直接生成")
    parser.add_argument("--prompt", type=str, default="今天天气", help="推理时的起始文字")
    parser.add_argument("--gen_len", type=int, default=100, help="推理时生成的字符数")

    args = parser.parse_args()

    torch.manual_seed(args.seed)
    random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ==========================================
    # --- [加载模型] 3. 推理模式分支 ---
    # ==========================================
    if args.load_model:
        print(f"运行设备：{device}")
        print(f"【推理模式】正在加载模型：{args.load_model}")

        if not os.path.exists(args.load_model):
            raise FileNotFoundError(f"找不到模型文件：{args.load_model}")

        # 加载权重和词表
        checkpoint = torch.load(args.load_model, map_location=device)
        char2idx = checkpoint["char2idx"]
        idx2char = checkpoint["idx2char"]
        vocab_size = checkpoint["vocab_size"]

        # 从保存的参数中获取模型结构信息（优先使用原训练时的超参，防止维度不匹配）
        saved_args = checkpoint.get("args", {})

        # 重建模型
        model = TransformerLM(
            vocab_size=vocab_size,
            d_model=saved_args.get("d_model", args.d_model),
            nhead=saved_args.get("nhead", args.nhead),
            num_layers=saved_args.get("num_layers", args.num_layers),
            dim_feedforward=saved_args.get("dim_feedforward", args.dim_feedforward),
            dropout=0.0,  # 推理时关闭 dropout
            max_len=saved_args.get("seq_len", args.seq_len),
        ).to(device)

        model.load_state_dict(checkpoint["model_state"])
        print(f"模型加载成功！(词表大小: {vocab_size})")

        print(f"\n--- 开始生成 ---")
        print(f"提示词: {args.prompt}")
        print(f"生成中（最大长度 {args.gen_len}）...\n")

        generated_text = generate_text(
            model,
            start_text=args.prompt,
            char2idx=char2idx,
            idx2char=idx2char,
            max_new_tokens=args.gen_len,
            device=device
        )

        print("【生成结果】")
        print(generated_text)
        print("\n----------------")
        return  # 生成完毕，直接退出 main 函数，不执行后续训练代码
    # ==========================================
    # 训练逻辑
    # ==========================================

    print(f"运行设备：{device}")
    print("模型类型：Decoder-only Transformer")

    text = load_corpus(args.corpus)
    print(f"语料字符数：{len(text):,}")

    char2idx, idx2char = build_vocab(text, args.max_vocab or None)
    vocab_size = len(char2idx)
    print(f"词表大小：{vocab_size}")

    train_text, val_text = split_corpus(text, args.val_ratio, args.seed)
    train_ds = CharDataset(train_text, char2idx, args.seq_len, args.stride)
    val_ds = CharDataset(val_text, char2idx, args.seq_len, args.stride)

    if len(train_ds) == 0 or len(val_ds) == 0:
        raise ValueError("训练或验证文本不足，请减小 --seq_len 或检查语料。")

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        drop_last=False, num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False,
        drop_last=False, num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
    )
    print(f"训练样本数：{len(train_ds):,}；验证样本数：{len(val_ds):,}")

    model = TransformerLM(
        vocab_size=vocab_size,
        d_model=args.d_model,
        nhead=args.nhead,
        num_layers=args.num_layers,
        dim_feedforward=args.dim_feedforward,
        dropout=args.dropout,
        max_len=args.seq_len,
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"模型可训练参数量：{total_params:,}")

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, args.epochs))

    writer = SummaryWriter(log_dir=args.log_dir)
    print(f"TensorBoard 日志将保存到：{os.path.abspath(args.log_dir)}")
    print("启动 TensorBoard 命令： tensorboard --logdir=runs")

    best_val_ppl = float("inf")
    save_dir = os.path.dirname(os.path.abspath(args.save))
    os.makedirs(save_dir, exist_ok=True)

    print(f"\n{'Epoch':>6} {'Train Loss':>12} {'Train PPL':>12} {'Val Loss':>12} {'Val PPL':>12} {'LR':>10}")
    print("-" * 72)

    for epoch in range(1, args.epochs + 1):
        train_loss, train_ppl = run_epoch(
            model, train_loader, criterion, optimizer, device, train=True, grad_clip=args.grad_clip,
        )
        val_loss, val_ppl = run_epoch(
            model, val_loader, criterion, optimizer, device, train=False
        )
        scheduler.step()

        current_lr = optimizer.param_groups[0]["lr"]

        # --- [TensorBoard] 4. 实时记录指标 ---
        writer.add_scalars('Loss', {'Train': train_loss, 'Validation': val_loss}, epoch)
        writer.add_scalars('PPL', {'Train': train_ppl, 'Validation': val_ppl}, epoch)
        writer.add_scalar('Learning_Rate', current_lr, epoch)

        improved = val_ppl < best_val_ppl
        if improved:
            best_val_ppl = val_ppl
            checkpoint = {
                "model_state": model.state_dict(),
                "char2idx": char2idx,
                "idx2char": idx2char,
                "args": vars(args),
                "model_type": "transformer",
                "vocab_size": vocab_size,
            }
            torch.save(checkpoint, args.save)

        marker = " *" if improved else ""
        print(
            f"{epoch:6d} {train_loss:12.4f} {train_ppl:12.2f} {val_loss:12.4f} {val_ppl:12.2f} {current_lr:10.2e}{marker}")

    # --- [TensorBoard] 5. 训练结束后关闭 writer ---
    writer.close()

    print(f"\n训练完成。最佳验证 PPL：{best_val_ppl:.2f}")
    print(f"最佳模型已保存到：{os.path.abspath(args.save)}")


if __name__ == "__main__":
    main()
