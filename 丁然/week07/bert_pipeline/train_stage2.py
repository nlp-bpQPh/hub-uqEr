"""训练第二阶段事件类型条件抽取器。"""

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

from common import EventSchema, ensure_article_split, filter_by_ids, load_jsonl  # noqa: E402
from config import (  # noqa: E402
    DEFAULT_BATCH_SIZE,
    DEFAULT_DETECT_PATH,
    DEFAULT_EPOCHS,
    DEFAULT_EXTRACT_PATH,
    DEFAULT_MAX_LENGTH,
    DEFAULT_MODEL_PATH,
    DEFAULT_OUTPUT_ROOT,
    DEFAULT_SCHEMA_PATH,
    DEFAULT_SPLIT_PATH,
    get_device,
)
from stage2_data import (  # noqa: E402
    ExtractionDataset,
    ExtractionLabels,
    collate_extraction,
    extract_spans,
    print_extraction_examples,
)
from stage2_model import ConditionalExtractor  # noqa: E402


transformers_logging.set_verbosity_error()


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if torch.backends.mps.is_available():
        torch.mps.manual_seed(seed)


def freeze_lower_layers(
    model: ConditionalExtractor,
    layer_count: int,
) -> None:
    """冻结底层 BERT，保留后部编码层和类型 embedding 可训练。"""

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
    if max_count is None or max_count >= len(rows):
        return rows
    result = list(rows)
    random.Random(seed).shuffle(result)
    return result[:max_count]


def prf(
    prediction: list[set[tuple[Any, ...]]],
    gold: list[set[tuple[Any, ...]]],
) -> tuple[float, float, float]:
    """计算论元或触发词的精确位置 P/R/F1。"""

    tp = sum(len(p & g) for p, g in zip(prediction, gold))
    fp = sum(len(p - g) for p, g in zip(prediction, gold))
    fn = sum(len(g - p) for p, g in zip(prediction, gold))
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-12)
    return precision, recall, f1


def build_records(
    predictions: list[list[int]],
    offsets_batch: list[list[tuple[int, int]]],
    rows: list[dict[str, Any]],
    row_indices: torch.Tensor,
    labels: ExtractionLabels,
) -> list[dict[str, Any]]:
    """把预测 BIO 和原始字符标注转换成可比较的 span 集合。"""

    records: list[dict[str, Any]] = []
    for batch_index, pred_tags in enumerate(predictions):
        row = rows[int(row_indices[batch_index])]
        offsets = offsets_batch[batch_index]
        event = row["event"]
        event_type = event["type"]

        trigger_gold = {
            (
                "trigger",
                None,
                int(event["trigger"]["start"]),
                int(event["trigger"]["end"]),
            )
        }
        arg_gold = {
            (
                "arg",
                argument["role"],
                int(argument["start"]),
                int(argument["end"]),
            )
            for argument in event["args"]
        }

        spans = extract_spans(pred_tags, labels, event_type)
        trigger_pred = {
            (
                span.kind,
                span.role,
                offsets[span.start][0],
                offsets[span.end - 1][1],
            )
            for span in spans
            if span.kind == "trigger"
        }
        arg_pred = {
            (
                span.kind,
                span.role,
                offsets[span.start][0],
                offsets[span.end - 1][1],
            )
            for span in spans
            if span.kind == "arg"
        }
        records.append(
            {
                "event_type": event_type,
                "trigger_gold": trigger_gold,
                "trigger_pred": trigger_pred,
                "arg_gold": arg_gold,
                "arg_pred": arg_pred,
            }
        )
    return records


def compute_metrics(records: list[dict[str, Any]]) -> dict[str, float]:
    """计算第二阶段抽取指标。"""

    trigger_p, trigger_r, trigger_f1 = prf(
        [record["trigger_pred"] for record in records],
        [record["trigger_gold"] for record in records],
    )
    arg_p, arg_r, arg_f1 = prf(
        [record["arg_pred"] for record in records],
        [record["arg_gold"] for record in records],
    )
    trigger_exact = sum(
        record["trigger_pred"] == record["trigger_gold"]
        for record in records
    ) / max(len(records), 1)
    arg_exact = sum(
        record["arg_pred"] == record["arg_gold"]
        for record in records
    ) / max(len(records), 1)
    event_exact = sum(
        record["trigger_pred"] == record["trigger_gold"]
        and record["arg_pred"] == record["arg_gold"]
        for record in records
    ) / max(len(records), 1)
    return {
        "trigger_precision": trigger_p,
        "trigger_recall": trigger_r,
        "trigger_f1": trigger_f1,
        "trigger_exact_match": trigger_exact,
        "arg_precision": arg_p,
        "arg_recall": arg_r,
        "arg_f1": arg_f1,
        "arg_exact_match": arg_exact,
        "event_exact_match": event_exact,
        "selection_score": 0.3 * trigger_f1 + 0.7 * arg_f1,
    }


def run_epoch(
    model: ConditionalExtractor,
    loader: DataLoader,
    device: torch.device,
    labels: ExtractionLabels,
    optimizer: torch.optim.Optimizer | None = None,
    epoch: int = 1,
) -> tuple[float, dict[str, float], list[dict[str, Any]]]:
    """训练或验证第二阶段一个 epoch。"""

    training = optimizer is not None
    model.train(training)
    total_loss = 0.0
    total_batches = 0
    rows = loader.dataset.rows
    records: list[dict[str, Any]] = []
    start_time = time.time()
    context = torch.enable_grad() if training else torch.inference_mode()

    with context:
        for step, batch in enumerate(loader, start=1):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            event_type_ids = batch["event_type_ids"].to(device)
            gold_tags = batch["labels"].to(device)
            emissions = model(
                input_ids,
                attention_mask,
                event_type_ids,
            )
            loss = model.loss(emissions, gold_tags, attention_mask)

            if training:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

            predictions = model.decode(
                emissions.detach(),
                attention_mask,
                input_ids,
            )
            records.extend(
                build_records(
                    predictions,
                    batch["offsets"],
                    rows,
                    batch["row_indices"],
                    labels,
                )
            )
            total_loss += float(loss.detach().cpu())
            total_batches += 1

            if step == 1 or step % 20 == 0 or step == len(loader):
                phase = "训练" if training else "验证"
                print(
                    f"[第二阶段-{phase}] epoch={epoch} "
                    f"step={step}/{len(loader)} "
                    f"loss={total_loss / total_batches:.4f} "
                    f"用时={time.time() - start_time:.1f}s",
                    flush=True,
                )

    metrics = compute_metrics(records)
    metrics["loss"] = total_loss / max(total_batches, 1)
    return metrics["loss"], metrics, records


def save_outputs(
    output_dir: Path,
    model: ConditionalExtractor,
    tokenizer: Any,
    labels: ExtractionLabels,
    metadata: dict[str, Any],
    metrics: dict[str, float],
    records: list[dict[str, Any]],
    rows: list[dict[str, Any]],
) -> None:
    """保存第二阶段模型、指标和验证预测。"""

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
    (output_dir / "labels.json").write_text(
        json.dumps(labels.to_dict(), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (output_dir / "metrics.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    with (output_dir / "predictions.jsonl").open(
        "w",
        encoding="utf-8",
    ) as file:
        for row, record in zip(rows, records):
            item = {
                "id": row["id"],
                "sent_id": row["sent_id"],
                "text": row["text"],
                "event_type": record["event_type"],
                "trigger_gold": sorted(record["trigger_gold"], key=str),
                "trigger_pred": sorted(record["trigger_pred"], key=str),
                "arg_gold": sorted(record["arg_gold"], key=str),
                "arg_pred": sorted(record["arg_pred"], key=str),
            }
            file.write(json.dumps(item, ensure_ascii=False) + "\n")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="训练第二阶段条件抽取器")
    parser.add_argument("--decoder", choices=["linear", "crf"], default="linear")
    parser.add_argument("--detect-path", default=str(DEFAULT_DETECT_PATH))
    parser.add_argument("--extract-path", default=str(DEFAULT_EXTRACT_PATH))
    parser.add_argument("--schema-path", default=str(DEFAULT_SCHEMA_PATH))
    parser.add_argument("--model-path", default=DEFAULT_MODEL_PATH)
    parser.add_argument("--split-path", default=str(DEFAULT_SPLIT_PATH))
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--max-length", type=int, default=DEFAULT_MAX_LENGTH)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--eval-batch-size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--freeze-layers", type=int, default=8)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--max-train", type=int, default=None)
    parser.add_argument("--max-val", type=int, default=None)
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
        args.max_train = args.max_train or 300
        args.max_val = args.max_val or 120
        args.max_length = min(args.max_length, 128)
    if args.output_dir is None:
        args.output_dir = str(DEFAULT_OUTPUT_ROOT / "stage2" / args.decoder)

    device = get_device(args.device)
    schema = EventSchema(args.schema_path)
    labels = ExtractionLabels(schema)
    detect_rows = load_jsonl(args.detect_path)
    extract_rows = load_jsonl(args.extract_path)
    split = ensure_article_split(
        detect_rows,
        args.split_path,
        args.val_ratio,
        args.seed,
        overwrite=args.overwrite_split,
    )
    train_rows = limit_rows(
        filter_by_ids(extract_rows, set(split["train_ids"])),
        args.max_train,
        args.seed + 1,
    )
    val_rows = limit_rows(
        filter_by_ids(extract_rows, set(split["val_ids"])),
        args.max_val,
        args.seed + 2,
    )
    if not train_rows or not val_rows:
        raise ValueError("第二阶段训练集或验证集为空，请检查文章级切分")

    tokenizer = AutoTokenizer.from_pretrained(
        str(Path(args.model_path).expanduser().resolve()),
        local_files_only=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.unk_token
    train_dataset = ExtractionDataset(
        train_rows,
        tokenizer,
        labels,
        args.max_length,
    )
    val_dataset = ExtractionDataset(
        val_rows,
        tokenizer,
        labels,
        args.max_length,
    )
    if args.data_debug_count > 0:
        print_extraction_examples(
            "训练集",
            train_dataset,
            tokenizer,
            labels,
            args.data_debug_count,
        )
        print_extraction_examples(
            "验证集",
            val_dataset,
            tokenizer,
            labels,
            min(1, args.data_debug_count),
        )

    def collate(batch: list[Any]) -> dict[str, Any]:
        return collate_extraction(batch, tokenizer.pad_token_id)

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
    model = ConditionalExtractor(
        encoder,
        labels,
        num_event_types=len(schema.event_types),
        decoder=args.decoder,
        cls_token_id=tokenizer.cls_token_id,
        sep_token_id=tokenizer.sep_token_id,
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
        f"stage2 decoder={args.decoder} device={device} "
        f"train={len(train_rows)} val={len(val_rows)} "
        f"labels={labels.num_labels}",
        flush=True,
    )

    best_score = -1.0
    best_state: dict[str, torch.Tensor] | None = None
    best_metrics: dict[str, float] | None = None
    best_records: list[dict[str, Any]] | None = None
    for epoch in range(1, args.epochs + 1):
        _, train_metrics, _ = run_epoch(
            model,
            train_loader,
            device,
            labels,
            optimizer=optimizer,
            epoch=epoch,
        )
        _, val_metrics, val_records = run_epoch(
            model,
            val_loader,
            device,
            labels,
            epoch=epoch,
        )
        print(
            f"stage2 epoch={epoch} train_loss={train_metrics['loss']:.4f} "
            f"val_loss={val_metrics['loss']:.4f} "
            f"trigger_f1={val_metrics['trigger_f1']:.4f} "
            f"arg_exact={val_metrics['arg_exact_match']:.4f} "
            f"arg_f1={val_metrics['arg_f1']:.4f}",
            flush=True,
        )
        if val_metrics["selection_score"] > best_score:
            best_score = val_metrics["selection_score"]
            best_state = {
                name: value.detach().cpu().clone()
                for name, value in model.state_dict().items()
            }
            best_metrics = val_metrics
            best_records = val_records

    assert best_state is not None
    assert best_metrics is not None
    assert best_records is not None
    model.load_state_dict(best_state)
    metadata = {
        "stage": 2,
        "task": "conditioned_event_extraction",
        "decoder": args.decoder,
        "model_path": args.model_path,
        "max_length": args.max_length,
        "num_labels": labels.num_labels,
        "train_rows": len(train_rows),
        "val_rows": len(val_rows),
        "freeze_layers": args.freeze_layers,
    }
    model.cpu()
    save_outputs(
        Path(args.output_dir),
        model,
        tokenizer,
        labels,
        metadata,
        best_metrics,
        best_records,
        val_rows,
    )
    print(
        f"第二阶段完成: {Path(args.output_dir).resolve()}",
        flush=True,
    )


if __name__ == "__main__":
    main()

