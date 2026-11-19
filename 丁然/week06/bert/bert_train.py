"""BERT 15 分类训练和验证入口。"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
import time
from pathlib import Path

WEEK_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(WEEK_DIR))

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from transformers import get_linear_schedule_with_warmup

from bert_config import (
    DEFAULT_MODEL_PATH,
    DEFAULT_MODEL_DIR,
    DEFAULT_OUTPUT_DIR,
    DEFAULT_TRAIN_PATH,
    DEFAULT_VAL_PATH,
    MAX_LENGTH,
    NUM_LABELS,
    get_device,
)
from bert_data import balanced_subset, encode_rows, load_rows, make_loader
from bert_fgm import FGM
from bert_losses import build_loss
from bert_model import BertClassifier, load_tokenizer
from labels import LABELS


TRAIN_EPOCHS = 1
LOG_EVERY = 20
FOCAL_GAMMA = 2.0
FGM_EPS = 0.5


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.backends.mps.is_available():
        torch.mps.manual_seed(seed)


def run_epoch(
    model: BertClassifier,
    loader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None = None,
    scheduler: torch.optim.lr_scheduler.LRScheduler | None = None,
    fgm: FGM | None = None,
    epoch: int = 1,
) -> dict[str, object]:
    """训练或验证一个 epoch。"""

    training = optimizer is not None
    model.train(training)
    total_loss = 0.0
    total_count = 0
    true_labels: list[int] = []
    predictions: list[int] = []
    start_time = time.time()

    context = torch.enable_grad() if training else torch.inference_mode()
    with context:
        for step, batch in enumerate(loader, start=1):
            batch = {name: value.to(device) for name, value in batch.items()}
            logits = model(batch["input_ids"], batch["attention_mask"]).logits
            loss = criterion(logits, batch["labels"])

            if training:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                if fgm is not None:
                    fgm.attack()
                    adversarial_logits = model(
                        batch["input_ids"],
                        batch["attention_mask"],
                    ).logits
                    adversarial_loss = criterion(
                        adversarial_logits,
                        batch["labels"],
                    )
                    adversarial_loss.backward()
                    fgm.restore()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()

            size = batch["labels"].size(0)
            total_loss += loss.item() * size
            total_count += size
            true_labels.extend(batch["labels"].cpu().tolist())
            predictions.extend(logits.argmax(dim=-1).detach().cpu().tolist())

            if step == 1 or step % LOG_EVERY == 0 or step == len(loader):
                elapsed = time.time() - start_time
                speed = total_count / max(elapsed, 1e-6)
                remaining = (len(loader) - step) * elapsed / step
                phase = "训练" if training else "验证"
                learning_rate = scheduler.get_last_lr()[0] if scheduler else 0.0
                print(
                    f"[{phase}] epoch={epoch}/{TRAIN_EPOCHS} "
                    f"step={step}/{len(loader)} "
                    f"loss={total_loss / total_count:.4f} "
                    f"speed={speed:.1f}条/s "
                    f"lr={learning_rate:.2e} "
                    f"剩余≈{remaining / 60:.1f}分钟",
                    flush=True,
                )

    return {
        "loss": total_loss / max(total_count, 1),
        "accuracy": accuracy_score(true_labels, predictions),
        "macro_f1": f1_score(
            true_labels,
            predictions,
            average="macro",
            zero_division=0,
        ),
        "weighted_f1": f1_score(
            true_labels,
            predictions,
            average="weighted",
            zero_division=0,
        ),
        "labels": true_labels,
        "predictions": predictions,
    }


def save_results(
    output_dir: Path,
    model_dir: Path,
    model: BertClassifier,
    tokenizer: object,
    args: argparse.Namespace,
    metrics: dict[str, object],
) -> None:
    """指标和模型分开保存，便于只保留一份最优权重。"""

    output_dir.mkdir(parents=True, exist_ok=True)
    model_dir.mkdir(parents=True, exist_ok=True)
    metadata = {
        "pooling": args.pooling,
        "max_length": MAX_LENGTH,
        "num_labels": NUM_LABELS,
        "model_path": args.model_path,
        "loss": args.loss,
        "fgm": args.fgm,
        "warmup_ratio": args.warmup_ratio,
    }
    meta_text = json.dumps(metadata, ensure_ascii=False, indent=2)
    (output_dir / "meta.json").write_text(meta_text, encoding="utf-8")
    (model_dir / "meta.json").write_text(meta_text, encoding="utf-8")

    tokenizer.save_pretrained(model_dir)
    model.encoder.config.save_pretrained(model_dir)
    torch.save(model.state_dict(), model_dir / "model.pt")
    (output_dir / "metrics.json").write_text(
        json.dumps(
            {key: value for key, value in metrics.items() if key not in {"labels", "predictions"}},
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    labels = metrics["labels"]
    predictions = metrics["predictions"]
    matrix = confusion_matrix(labels, predictions, labels=list(range(NUM_LABELS)))
    np.savetxt(output_dir / "confusion_matrix.csv", matrix, delimiter=",", fmt="%d")

    plt.rcParams["font.sans-serif"] = ["PingFang SC", "Arial Unicode MS"]
    plt.rcParams["axes.unicode_minus"] = False
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        matrix,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=LABELS,
        yticklabels=LABELS,
    )
    plt.xlabel("预测类别")
    plt.ylabel("真实类别")
    plt.title("BERT 验证集混淆矩阵")
    plt.tight_layout()
    plt.savefig(output_dir / "confusion_matrix.png", dpi=180)
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="BERT 中文新闻 15 分类")
    parser.add_argument("--model-path", default=DEFAULT_MODEL_PATH)
    parser.add_argument("--train-path", default=str(DEFAULT_TRAIN_PATH))
    parser.add_argument("--val-path", default=str(DEFAULT_VAL_PATH))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--model-dir", default=str(DEFAULT_MODEL_DIR))
    parser.add_argument("--pooling", choices=["cls", "mean", "max"], default="cls")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--eval-batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--warmup-ratio", type=float, default=0.0)
    parser.add_argument("--loss", choices=["ce", "focal"], default="ce")
    parser.add_argument("--fgm", action="store_true", help="启用 FGM 对抗训练")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--smoke", action="store_true", help="使用 60/30 条样本快速自检")
    args = parser.parse_args()

    os.environ.setdefault("MPLCONFIGDIR", str(Path(args.output_dir) / ".mplconfig"))
    set_seed(args.seed)
    device = get_device(args.device)
    tokenizer = load_tokenizer(args.model_path)

    train_rows = load_rows(args.train_path)
    val_rows = load_rows(args.val_path)
    if args.smoke:
        train_rows = balanced_subset(train_rows, 60, args.seed)
        val_rows = balanced_subset(val_rows, 30, args.seed + 1)

    train_dataset = encode_rows(tokenizer, train_rows, MAX_LENGTH)
    val_dataset = encode_rows(tokenizer, val_rows, MAX_LENGTH)
    train_loader = make_loader(train_dataset, args.batch_size, True, args.seed)
    val_loader = make_loader(val_dataset, args.eval_batch_size, False, args.seed)

    model = BertClassifier(
        model_path=args.model_path,
        num_labels=NUM_LABELS,
        pooling=args.pooling,
    ).to(device)
    criterion = build_loss(args.loss, FOCAL_GAMMA)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=0.01,
    )
    total_steps = len(train_loader) * TRAIN_EPOCHS
    warmup_steps = (
        math.ceil(total_steps * args.warmup_ratio)
        if args.warmup_ratio > 0
        else 0
    )
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        warmup_steps,
        total_steps,
    )
    fgm = FGM(model, FGM_EPS) if args.fgm else None

    print(
        f"设备={device} 训练={len(train_dataset)} 验证={len(val_dataset)} "
        f"batch={args.batch_size} epochs={TRAIN_EPOCHS} "
        f"每轮步数={len(train_loader)} pooling={args.pooling} "
        f"loss={args.loss} warmup={args.warmup_ratio:.2f} "
        f"warmup_steps={warmup_steps} fgm={args.fgm}",
        flush=True,
    )
    best_f1 = -1.0
    best_state = None
    best_metrics = None
    for epoch in range(1, TRAIN_EPOCHS + 1):
        start = time.time()
        print(f"开始训练 epoch={epoch}/{TRAIN_EPOCHS}", flush=True)
        train_metrics = run_epoch(
            model,
            train_loader,
            criterion,
            device,
            optimizer,
            scheduler,
            fgm=fgm,
            epoch=epoch,
        )
        print(f"开始验证 epoch={epoch}/{TRAIN_EPOCHS}", flush=True)
        val_metrics = run_epoch(
            model,
            val_loader,
            criterion,
            device,
            epoch=epoch,
        )
        print(
            f"epoch={epoch} loss={train_metrics['loss']:.4f} "
            f"acc={val_metrics['accuracy']:.4f} "
            f"macro_f1={val_metrics['macro_f1']:.4f} "
            f"time={time.time() - start:.1f}s"
        )
        if val_metrics["macro_f1"] > best_f1:
            best_f1 = val_metrics["macro_f1"]
            best_state = {
                name: value.detach().cpu().clone()
                for name, value in model.state_dict().items()
            }
            best_metrics = val_metrics

    model.load_state_dict(best_state)
    model.cpu()
    save_results(
        Path(args.output_dir),
        Path(args.model_dir),
        model,
        tokenizer,
        args,
        best_metrics,
    )
    print(f"实验结果已保存: {Path(args.output_dir).resolve()}")
    print(f"模型权重已保存: {Path(args.model_dir).resolve()}")


if __name__ == "__main__":
    main()
