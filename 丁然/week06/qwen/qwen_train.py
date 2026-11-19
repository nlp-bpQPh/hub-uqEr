"""训练一次 Qwen LoRA 实验。"""

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

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import seaborn as sns  # noqa: E402
import torch  # noqa: E402
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score  # noqa: E402
from transformers import get_linear_schedule_with_warmup  # noqa: E402

from labels import LABELS  # noqa: E402
from qwen_config import (  # noqa: E402
    DEFAULT_MODEL_PATH,
    DEFAULT_TRAIN_PATH,
    DEFAULT_VAL_PATH,
    EVAL_LIMIT,
    MAX_LENGTH,
    NUM_LABELS,
    get_device,
)
from qwen_data import (  # noqa: E402
    balanced_subset,
    encode_rows,
    load_rows,
    make_loader,
)
from qwen_model import QwenClassifier, build_lora_model, load_tokenizer  # noqa: E402


LOG_EVERY = 20


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.backends.mps.is_available():
        torch.mps.manual_seed(seed)


def run_epoch(
    model: QwenClassifier,
    loader: torch.utils.data.DataLoader,
    criterion: torch.nn.Module,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None = None,
    scheduler: torch.optim.lr_scheduler.LRScheduler | None = None,
    epoch: int = 1,
) -> dict[str, object]:
    training = optimizer is not None
    model.train(training)
    total_loss = 0.0
    total_count = 0
    labels: list[int] = []
    predictions: list[int] = []
    start = time.time()

    with torch.enable_grad() if training else torch.inference_mode():
        for step, batch in enumerate(loader, start=1):
            batch = {name: value.to(device) for name, value in batch.items()}
            logits = model(batch["input_ids"], batch["attention_mask"]).logits
            loss = criterion(logits, batch["labels"])
            if training:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()

            size = batch["labels"].size(0)
            total_loss += loss.item() * size
            total_count += size
            labels.extend(batch["labels"].cpu().tolist())
            predictions.extend(logits.argmax(dim=-1).detach().cpu().tolist())

            if step == 1 or step % LOG_EVERY == 0 or step == len(loader):
                elapsed = time.time() - start
                speed = total_count / max(elapsed, 1e-6)
                remaining = (len(loader) - step) * elapsed / step
                phase = "训练" if training else "验证"
                print(
                    f"[{phase}] epoch={epoch} step={step}/{len(loader)} "
                    f"loss={total_loss / total_count:.4f} "
                    f"speed={speed:.1f}条/s 剩余≈{remaining / 60:.1f}分钟",
                    flush=True,
                )

    return {
        "loss": total_loss / max(total_count, 1),
        "accuracy": accuracy_score(labels, predictions),
        "macro_f1": f1_score(labels, predictions, average="macro", zero_division=0),
        "weighted_f1": f1_score(
            labels,
            predictions,
            average="weighted",
            zero_division=0,
        ),
        "labels": labels,
        "predictions": predictions,
    }


def save_result(
    output_dir: Path,
    model_dir: Path,
    model: QwenClassifier,
    tokenizer: object,
    args: argparse.Namespace,
    metrics: dict[str, object],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    model_dir.mkdir(parents=True, exist_ok=True)
    meta = {
        "experiment": "lora",
        "num_labels": NUM_LABELS,
        "max_length": MAX_LENGTH,
        "eval_limit": args.eval_limit,
        "lora_rank": 8,
        "lora_alpha": 16,
        "lora_targets": ["q_proj", "v_proj"],
    }
    meta_text = json.dumps(meta, ensure_ascii=False, indent=2)
    (output_dir / "meta.json").write_text(meta_text, encoding="utf-8")
    (model_dir / "meta.json").write_text(meta_text, encoding="utf-8")
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
    plt.title("Qwen 全量 LoRA 混淆矩阵")
    plt.tight_layout()
    plt.savefig(output_dir / "confusion_matrix.png", dpi=180)
    plt.close()

    tokenizer.save_pretrained(model_dir)
    model.encoder.save_pretrained(model_dir)
    model.encoder.get_base_model().config.save_pretrained(model_dir / "base_config")
    torch.save(model.classifier.state_dict(), model_dir / "classifier_head.pt")


def main() -> None:
    parser = argparse.ArgumentParser(description="Qwen 全量 LoRA 微调")
    parser.add_argument("--model-path", default=DEFAULT_MODEL_PATH)
    parser.add_argument("--train-path", default=str(DEFAULT_TRAIN_PATH))
    parser.add_argument("--val-path", default=str(DEFAULT_VAL_PATH))
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--eval-limit", type=int, default=EVAL_LIMIT)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--eval-batch-size", type=int, default=16)
    parser.add_argument("--gradient-accumulation", type=int, default=2)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--warmup-ratio", type=float, default=0.03)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()

    os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/week06-qwen-mpl")
    set_seed(args.seed)
    device = get_device(args.device)
    tokenizer = load_tokenizer(args.model_path)
    train_rows = load_rows(args.train_path)
    val_rows = load_rows(args.val_path)

    if args.smoke:
        train_rows = balanced_subset(train_rows, 60, args.seed)
    if args.smoke:
        val_rows = balanced_subset(val_rows, 30, args.seed + 1)
    elif args.eval_limit > 0:
        val_rows = balanced_subset(val_rows, args.eval_limit, args.seed + 1)

    train_dataset = encode_rows(tokenizer, train_rows, MAX_LENGTH)
    val_dataset = encode_rows(tokenizer, val_rows, MAX_LENGTH)
    train_loader = make_loader(train_dataset, args.batch_size, True, args.seed)
    val_loader = make_loader(val_dataset, args.eval_batch_size, False, args.seed)
    model = build_lora_model(args.model_path, NUM_LABELS).to(device)
    criterion = torch.nn.CrossEntropyLoss()
    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=args.learning_rate, weight_decay=0.01)
    updates = math.ceil(len(train_loader) / args.gradient_accumulation)
    warmup_steps = math.ceil(updates * args.warmup_ratio)
    scheduler = get_linear_schedule_with_warmup(optimizer, warmup_steps, updates)

    print(
        f"模式=lora 设备={device} 训练={len(train_dataset)} "
        f"验证={len(val_dataset)} max_length={MAX_LENGTH} "
        f"warmup_steps={warmup_steps}",
        flush=True,
    )
    model.train()
    optimizer.zero_grad(set_to_none=True)
    for step, batch in enumerate(train_loader, start=1):
        batch = {name: value.to(device) for name, value in batch.items()}
        logits = model(batch["input_ids"], batch["attention_mask"]).logits
        loss = criterion(logits, batch["labels"])
        (loss / args.gradient_accumulation).backward()
        if step % args.gradient_accumulation == 0 or step == len(train_loader):
            torch.nn.utils.clip_grad_norm_(trainable, 1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)
        if step == 1 or step % LOG_EVERY == 0 or step == len(train_loader):
            print(
                f"[训练] step={step}/{len(train_loader)} loss={loss.item():.4f}",
                flush=True,
            )

    val_metrics = run_epoch(
        model,
        val_loader,
        criterion,
        device,
    )
    print(
        f"结果 accuracy={val_metrics['accuracy']:.4f} "
        f"macro_f1={val_metrics['macro_f1']:.4f}",
        flush=True,
    )
    model.cpu()
    save_result(
        Path(args.output_dir),
        Path(args.model_dir),
        model,
        tokenizer,
        args,
        val_metrics,
    )
    print(f"实验完成: {args.output_dir}", flush=True)


if __name__ == "__main__":
    main()
