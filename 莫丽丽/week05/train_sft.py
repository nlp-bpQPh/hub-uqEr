# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import csv
import math
import os
import random
import time
from typing import Tuple

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from dataset import CharTokenizer, SFTDataset, collate_sft_batch, dataset_summary, load_jsonl
from transformer_lm import TransformerLM

HERE = os.path.dirname(os.path.abspath(__file__))


def set_seed(seed: int) -> None:
    """固定 Python 与 PyTorch 随机种子，便于重复实验。"""
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def masked_loss_sum(logits: torch.Tensor, labels: torch.Tensor) -> Tuple[torch.Tensor, int]:
    """返回有效 Assistant Token 的交叉熵总和及数量。"""
    vocab_size = logits.size(-1)
    loss_sum = F.cross_entropy(
        logits.reshape(-1, vocab_size),
        labels.reshape(-1),
        ignore_index=-100,
        reduction="sum",
    )
    token_count = int(labels.ne(-100).sum().item())
    return loss_sum, token_count


@torch.no_grad()
def evaluate_loss(model: TransformerLM, loader: DataLoader, device: torch.device) -> Tuple[float, float]:
    """使用 Teacher Forcing 计算按有效 Token 加权的验证 Loss 和 PPL。"""
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    for batch in loader:
        input_ids = batch["input_ids"].to(device, non_blocking=True)
        labels = batch["labels"].to(device, non_blocking=True)
        logits = model(input_ids)
        loss_sum, token_count = masked_loss_sum(logits, labels)
        total_loss += float(loss_sum.item())
        total_tokens += token_count
    if total_tokens == 0:
        raise RuntimeError("验证集没有有效监督 Token，请检查 Loss Mask。")
    mean_loss = total_loss / total_tokens
    return mean_loss, math.exp(min(mean_loss, 50.0))


def build_scheduler(optimizer, total_steps: int, warmup_ratio: float, min_lr_ratio: float):
    """Linear warmup + cosine decay，并保留一个非零最低学习率比例。"""
    warmup_steps = max(1, int(total_steps * warmup_ratio)) if warmup_ratio > 0 else 0

    def lr_lambda(step: int) -> float:
        if warmup_steps and step < warmup_steps:
            return float(step + 1) / float(warmup_steps)
        decay_steps = max(1, total_steps - warmup_steps)
        progress = min(1.0, max(0.0, (step - warmup_steps) / decay_steps))
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return min_lr_ratio + (1.0 - min_lr_ratio) * cosine

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def save_checkpoint(
    path: str,
    model: TransformerLM,
    tokenizer: CharTokenizer,
    model_config: dict,
    train_config: dict,
    epoch: int,
    train_loss: float,
    val_loss: float,
    best_val_loss: float,
) -> None:
    """保存模型参数、字符表、模型配置和训练配置，供 inference.py 复现模型。"""
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    torch.save(
        {
            "model_state": model.state_dict(),
            "itos": tokenizer.itos,
            "model_config": model_config,
            "train_config": train_config,
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "best_val_loss": best_val_loss,
        },
        path,
    )


def train(args: argparse.Namespace) -> None:
    """执行从随机初始化开始的纯监督式指令训练。"""
    set_seed(args.seed)
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("指定了 --device cuda，但当前 PyTorch 无法使用 CUDA。")

    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    train_rows = load_jsonl(args.train_data)
    val_rows = load_jsonl(args.val_data)
    tokenizer = CharTokenizer.from_records(train_rows)
    train_dataset = SFTDataset(train_rows, tokenizer, args.max_seq_len, args.max_target_chars)
    val_dataset = SFTDataset(val_rows, tokenizer, args.max_seq_len, args.max_target_chars)

    train_stats = dataset_summary(train_rows, tokenizer)
    val_stats = dataset_summary(val_rows, tokenizer)

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=False,
        collate_fn=collate_sft_batch,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        drop_last=False,
        collate_fn=collate_sft_batch,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
    )

    model_config = {
        "vocab_size": tokenizer.vocab_size,
        "d_model": args.d_model,
        "n_head": args.heads,
        "n_layer": args.layers,
        "max_len": args.max_seq_len,
        "dropout": args.dropout,
        "pad_id": 0,
        "tie_weights": True,
    }
    model = TransformerLM(**model_config).to(device)
    parameter_count = sum(p.numel() for p in model.parameters())

    # 标准 AdamW：对 bias 和 LayerNorm 参数不施加 weight decay。
    decay_params, no_decay_params = [], []
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        if parameter.ndim < 2 or "norm" in name.lower():
            no_decay_params.append(parameter)
        else:
            decay_params.append(parameter)
    optimizer = torch.optim.AdamW(
        [
            {"params": decay_params, "weight_decay": args.weight_decay},
            {"params": no_decay_params, "weight_decay": 0.0},
        ],
        lr=args.lr,
        betas=(0.9, 0.95),
        eps=1e-8,
    )

    updates_per_epoch = math.ceil(len(train_loader) / args.grad_accum_steps)
    total_steps = max(1, updates_per_epoch * args.epochs)
    scheduler = build_scheduler(optimizer, total_steps, args.warmup_ratio, args.min_lr_ratio)

    os.makedirs(args.save_dir, exist_ok=True)
    os.makedirs(args.log_dir, exist_ok=True)
    best_path = os.path.join(args.save_dir, "best.pt")
    latest_path = os.path.join(args.save_dir, "latest.pt")
    metrics_path = os.path.join(args.log_dir, "metrics.csv")
    train_config = vars(args).copy()

    print(f"设备：{device}")
    print(f"训练样本：{len(train_dataset)}；验证样本：{len(val_dataset)}")
    print(f"字符词表：{tokenizer.vocab_size}；模型参数量：{parameter_count:,}")
    print(f"模型：{args.layers} 层 / d_model={args.d_model} / heads={args.heads} / dropout={args.dropout}")
    print(f"训练集数据检查：{train_stats}")
    print(f"验证集数据检查：{val_stats}")
    print("训练模板：纯 SFT；没有 RAG、检索上下文或自动答案拼接。")
    print("Loss Mask：只有 Assistant 标准答案字符及 EOS 参与 Cross Entropy；Prompt/PAD 标签=-100。")
    if train_dataset.truncated_targets or val_dataset.truncated_targets:
        print(f"警告：目标答案发生截断，train={train_dataset.truncated_targets}, val={val_dataset.truncated_targets}。")
    if train_dataset.truncated_prompts or val_dataset.truncated_prompts:
        print(f"提示：Prompt 发生截断，train={train_dataset.truncated_prompts}, val={val_dataset.truncated_prompts}。")
    if val_stats["unknown_chars"]:
        print(f"警告：验证集存在 {val_stats['unknown_chars']} 个训练词表外字符。")

    # 每次训练创建独立日志，避免不同超参数实验混在同一个 CSV。
    if os.path.exists(metrics_path):
        os.remove(metrics_path)

    best_val_loss = float("inf")
    no_improve_epochs = 0
    rows_for_csv = []
    optimizer.zero_grad(set_to_none=True)

    for epoch in range(1, args.epochs + 1):
        model.train()
        started = time.time()
        epoch_loss_sum = 0.0
        epoch_token_count = 0
        accumulated_tokens = 0
        accumulated_micro_batches = 0

        for step, batch in enumerate(train_loader, start=1):
            input_ids = batch["input_ids"].to(device, non_blocking=True)
            labels = batch["labels"].to(device, non_blocking=True)
            logits = model(input_ids)
            loss_sum, token_count = masked_loss_sum(logits, labels)
            if token_count == 0:
                continue

            # 梯度按 token loss 总和累计，在 optimizer.step 前除以实际有效 Token 数。
            loss_sum.backward()
            accumulated_tokens += token_count
            accumulated_micro_batches += 1
            epoch_loss_sum += float(loss_sum.detach().item())
            epoch_token_count += token_count

            update_now = (
                accumulated_micro_batches >= args.grad_accum_steps or step == len(train_loader)
            )
            if update_now and accumulated_tokens > 0:
                for parameter in model.parameters():
                    if parameter.grad is not None:
                        parameter.grad.div_(accumulated_tokens)
                grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)

                if args.log_steps > 0 and step % args.log_steps == 0:
                    print(
                        f"Epoch {epoch}/{args.epochs} step {step}/{len(train_loader)} "
                        f"loss={epoch_loss_sum / max(epoch_token_count, 1):.4f} "
                        f"lr={optimizer.param_groups[0]['lr']:.2e} grad_norm={float(grad_norm):.3f}"
                    )
                accumulated_tokens = 0
                accumulated_micro_batches = 0

        train_loss = epoch_loss_sum / max(epoch_token_count, 1)
        val_loss, val_ppl = evaluate_loss(model, val_loader, device)
        improved = val_loss < best_val_loss - args.min_delta
        duration = time.time() - started

        if improved:
            best_val_loss = val_loss
            no_improve_epochs = 0
            save_checkpoint(
                best_path, model, tokenizer, model_config, train_config,
                epoch, train_loss, val_loss, best_val_loss,
            )
        else:
            no_improve_epochs += 1

        save_checkpoint(
            latest_path, model, tokenizer, model_config, train_config,
            epoch, train_loss, val_loss, best_val_loss,
        )

        rows_for_csv.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "val_perplexity": val_ppl,
                "learning_rate": optimizer.param_groups[0]["lr"],
                "seconds": duration,
                "best": int(improved),
            }
        )
        with open(metrics_path, "w", newline="", encoding="utf-8-sig") as file:
            writer = csv.DictWriter(file, fieldnames=list(rows_for_csv[0].keys()))
            writer.writeheader()
            writer.writerows(rows_for_csv)

        print(
            f"Epoch {epoch:03d}/{args.epochs} | train_loss={train_loss:.4f} "
            f"| val_loss={val_loss:.4f} | val_ppl={val_ppl:.3f} "
            f"| lr={optimizer.param_groups[0]['lr']:.2e} | {duration:.1f}s"
            f"{' | BEST' if improved else ''}"
        )

        if args.patience > 0 and no_improve_epochs >= args.patience:
            print(f"验证 Loss 连续 {args.patience} 轮没有改善，触发 Early Stopping。")
            break

    print(f"\n最佳验证 Loss：{best_val_loss:.6f}")
    print(f"最佳模型：{best_path}")
    print(f"最近模型：{latest_path}")
    print(f"训练记录：{metrics_path}")
    print("请优先使用 best.pt，而不是默认使用最后一轮 latest.pt。")


def build_parser() -> argparse.ArgumentParser:
    """定义 SFT 训练参数"""
    parser = argparse.ArgumentParser(description="从零初始化的 Decoder-only Transformer 纯 SFT 训练。")
    parser.add_argument("--train-data", default=os.path.join(HERE, "data", "sft_train.jsonl"))
    parser.add_argument("--val-data", default=os.path.join(HERE, "data", "sft_val.jsonl"))
    parser.add_argument("--max-seq-len", type=int, default=256)
    parser.add_argument("--max-target-chars", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--grad-accum-steps", type=int, default=2)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--warmup-ratio", type=float, default=0.03)
    parser.add_argument("--min-lr-ratio", type=float, default=0.10)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--d-model", type=int, default=256)
    parser.add_argument("--heads", type=int, default=8)
    parser.add_argument("--layers", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.05)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--patience", type=int, default=10, help="验证 Loss 连续多少轮未改善后早停；0 表示关闭。")
    parser.add_argument("--min-delta", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="cuda")
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--log-steps", type=int, default=0)
    parser.add_argument("--save-dir", default=os.path.join(HERE, "checkpoints"))
    parser.add_argument("--log-dir", default=os.path.join(HERE, "logs"))
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.batch_size < 1 or args.grad_accum_steps < 1 or args.epochs < 1:
        raise ValueError("batch-size、grad-accum-steps 和 epochs 必须至少为 1。")
    if not 0.0 <= args.warmup_ratio < 1.0:
        raise ValueError("warmup-ratio 必须处于 [0,1) 范围内。")
    if not 0.0 <= args.min_lr_ratio <= 1.0:
        raise ValueError("min-lr-ratio 必须处于 [0,1] 范围内。")
    train(args)


if __name__ == "__main__":
    main()
