"""训练第一阶段事件类型检测器。"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader
from transformers import AutoModel, AutoTokenizer
from transformers.utils import logging as transformers_logging

PIPELINE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(PIPELINE_DIR))

from common import (  # noqa: E402
    EventSchema,
    detector_metrics,
    ensure_article_split,
    filter_by_ids,
    load_jsonl,
    tune_threshold,
)
from config import (  # noqa: E402
    DEFAULT_BATCH_SIZE,
    DEFAULT_DETECT_PATH,
    DEFAULT_EPOCHS,
    DEFAULT_MAX_LENGTH,
    DEFAULT_MODEL_PATH,
    DEFAULT_OUTPUT_ROOT,
    DEFAULT_SCHEMA_PATH,
    DEFAULT_SPLIT_PATH,
    get_device,
)
from stage1_data import (  # noqa: E402
    DetectorDataset,
    collate_detector,
    print_detector_examples,
)
from stage1_model import DetectorModel  # noqa: E402


transformers_logging.set_verbosity_error()


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if torch.backends.mps.is_available():
        torch.mps.manual_seed(seed)


def freeze_lower_layers(model: DetectorModel, layer_count: int) -> None:
    """冻结底部 BERT 层，降低第一阶段训练开销。"""

    if layer_count <= 0:
        return
    for parameter in model.encoder.embeddings.parameters():
        parameter.requires_grad = False
    for layer in model.encoder.encoder.layer[:layer_count]:
        for parameter in layer.parameters():
            parameter.requires_grad = False


def limit_rows(
    rows: list[dict[str, Any]],
    max_count: int | None,
    seed: int,
) -> list[dict[str, Any]]:
    """smoke 模式下按文章 id 截取，避免同一文章被拆开。"""

    if max_count is None or max_count >= len(rows):
        return rows
    article_ids = sorted({row["id"] for row in rows})
    random.Random(seed).shuffle(article_ids)
    selected = set(article_ids[:max_count])
    return [row for row in rows if row["id"] in selected]


def run_epoch(
    model: DetectorModel,
    loader: DataLoader,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None = None,
    epoch: int = 1,
) -> tuple[float, list[list[float]], list[list[int]]]:
    """训练或验证一个 epoch，返回概率和真实多标签。"""

    training = optimizer is not None
    model.train(training)
    total_loss = 0.0
    total_batches = 0
    probabilities: list[list[float]] = []
    gold_labels: list[list[int]] = []
    start_time = time.time()
    context = torch.enable_grad() if training else torch.inference_mode()

    with context:
        for step, batch in enumerate(loader, start=1):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            logits = model(input_ids, attention_mask)
            loss = model.loss(logits, labels)

            if training:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

            probabilities.extend(
                torch.sigmoid(logits.detach()).cpu().tolist()
            )
            gold_labels.extend(
                labels.detach().long().cpu().tolist()
            )
            total_loss += float(loss.detach().cpu())
            total_batches += 1

            if step == 1 or step % 20 == 0 or step == len(loader):
                phase = "训练" if training else "验证"
                print(
                    f"[第一阶段-{phase}] epoch={epoch} "
                    f"step={step}/{len(loader)} "
                    f"loss={total_loss / total_batches:.4f} "
                    f"用时={time.time() - start_time:.1f}s",
                    flush=True,
                )

    return total_loss / max(total_batches, 1), probabilities, gold_labels


def save_outputs(
    output_dir: Path,
    model: DetectorModel,
    tokenizer: Any,
    schema: EventSchema,
    metadata: dict[str, Any],
    metrics: dict[str, float],
    rows: list[dict[str, Any]],
    probabilities: list[list[float]],
) -> None:
    """保存第一阶段模型、阈值、指标和预测概率。"""

    output_dir.mkdir(parents=True, exist_ok=True)
    tokenizer.save_pretrained(output_dir)
    model.encoder.config.save_pretrained(output_dir)
    torch.save(
        {
            name: value.detach().cpu()
            for name, value in model.state_dict().items()
        },
        output_dir / "model.pt",
    )
    (output_dir / "meta.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (output_dir / "schema.json").write_text(
        json.dumps(schema.to_dict(), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (output_dir / "metrics.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    threshold = metadata["threshold"]
    with (output_dir / "predictions.jsonl").open(
        "w",
        encoding="utf-8",
    ) as file:
        for row, row_probabilities in zip(rows, probabilities):
            predicted_types = [
                event_type
                for event_type, probability in zip(
                    schema.event_types,
                    row_probabilities,
                )
                if probability >= threshold
            ]
            item = {
                "id": row["id"],
                "sent_id": row["sent_id"],
                "text": row["text"],
                "types_gold": row["types"],
                "types_pred": predicted_types,
                "scores": {
                    event_type: round(probability, 6)
                    for event_type, probability in zip(
                        schema.event_types,
                        row_probabilities,
                    )
                },
            }
            file.write(json.dumps(item, ensure_ascii=False) + "\n")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="训练第一阶段事件检测器")
    parser.add_argument("--detect-path", default=str(DEFAULT_DETECT_PATH))
    parser.add_argument("--schema-path", default=str(DEFAULT_SCHEMA_PATH))
    parser.add_argument("--model-path", default=DEFAULT_MODEL_PATH)
    parser.add_argument("--split-path", default=str(DEFAULT_SPLIT_PATH))
    parser.add_argument(
        "--output-dir",
        default=str(DEFAULT_OUTPUT_ROOT / "stage1"),
    )
    parser.add_argument("--max-length", type=int, default=DEFAULT_MAX_LENGTH)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--eval-batch-size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--freeze-layers", type=int, default=8)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--max-articles", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--overwrite-split", action="store_true")
    parser.add_argument("--data-debug-count", type=int, default=2)
    parser.add_argument("--smoke", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    set_seed(args.seed)
    if args.smoke:
        args.epochs = 1
        args.max_articles = args.max_articles or 120
        args.max_length = min(args.max_length, 128)

    device = get_device(args.device)
    schema = EventSchema(args.schema_path)
    detect_rows = load_jsonl(args.detect_path)
    split = ensure_article_split(
        detect_rows,
        args.split_path,
        args.val_ratio,
        args.seed,
        overwrite=args.overwrite_split,
    )
    train_rows = filter_by_ids(detect_rows, set(split["train_ids"]))
    val_rows = filter_by_ids(detect_rows, set(split["val_ids"]))
    train_rows = limit_rows(train_rows, args.max_articles, args.seed + 1)
    val_rows = limit_rows(val_rows, args.max_articles, args.seed + 2)

    tokenizer = AutoTokenizer.from_pretrained(
        str(Path(args.model_path).expanduser().resolve()),
        local_files_only=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.unk_token

    train_dataset = DetectorDataset(
        train_rows,
        tokenizer,
        schema,
        args.max_length,
    )
    val_dataset = DetectorDataset(
        val_rows,
        tokenizer,
        schema,
        args.max_length,
    )
    if args.data_debug_count > 0:
        print_detector_examples(
            "训练集",
            train_dataset,
            tokenizer,
            schema,
            args.data_debug_count,
        )
        print_detector_examples(
            "验证集",
            val_dataset,
            tokenizer,
            schema,
            min(1, args.data_debug_count),
        )

    def collate(batch: list[Any]) -> dict[str, torch.Tensor]:
        return collate_detector(batch, tokenizer.pad_token_id)

    generator = torch.Generator().manual_seed(args.seed)
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        generator=generator,
        collate_fn=collate,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.eval_batch_size,
        shuffle=False,
        collate_fn=collate,
    )

    encoder = AutoModel.from_pretrained(
        str(Path(args.model_path).expanduser().resolve()),
        local_files_only=True,
        dtype=torch.float32,
    )
    model = DetectorModel(
        encoder,
        num_labels=len(schema.event_types),
    ).to(device)
    freeze_lower_layers(model, args.freeze_layers)
    optimizer = torch.optim.AdamW(
        [
            parameter
            for parameter in model.parameters()
            if parameter.requires_grad
        ],
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )

    print(
        f"stage1 device={device} train_rows={len(train_rows)} "
        f"val_rows={len(val_rows)} train_articles="
        f"{len({row['id'] for row in train_rows})} "
        f"val_articles={len({row['id'] for row in val_rows})}",
        flush=True,
    )

    best_loss = float("inf")
    best_state: dict[str, torch.Tensor] | None = None
    best_probabilities: list[list[float]] | None = None
    best_gold: list[list[int]] | None = None
    for epoch in range(1, args.epochs + 1):
        train_loss, _, _ = run_epoch(
            model,
            train_loader,
            device,
            optimizer=optimizer,
            epoch=epoch,
        )
        val_loss, probabilities, gold = run_epoch(
            model,
            val_loader,
            device,
            epoch=epoch,
        )
        print(
            f"stage1 epoch={epoch} train_loss={train_loss:.4f} "
            f"val_loss={val_loss:.4f}",
            flush=True,
        )
        if val_loss < best_loss:
            best_loss = val_loss
            best_state = {
                name: value.detach().cpu().clone()
                for name, value in model.state_dict().items()
            }
            best_probabilities = probabilities
            best_gold = gold

    assert best_state is not None
    assert best_probabilities is not None
    assert best_gold is not None
    model.load_state_dict(best_state)
    threshold, metrics = tune_threshold(
        best_probabilities,
        best_gold,
        schema.event_types,
    )
    metrics["loss"] = best_loss
    metrics["threshold"] = threshold

    metadata = {
        "stage": 1,
        "task": "multi_label_event_detection",
        "model_path": args.model_path,
        "max_length": args.max_length,
        "threshold": threshold,
        "event_types": schema.event_types,
        "train_rows": len(train_rows),
        "val_rows": len(val_rows),
        "train_articles": len({row["id"] for row in train_rows}),
        "val_articles": len({row["id"] for row in val_rows}),
    }
    model.cpu()
    save_outputs(
        Path(args.output_dir),
        model,
        tokenizer,
        schema,
        metadata,
        metrics,
        val_rows,
        best_probabilities,
    )
    print(
        f"第一阶段完成: {Path(args.output_dir).resolve()} "
        f"threshold={threshold:.2f}",
        flush=True,
    )


if __name__ == "__main__":
    main()

