"""评估第一阶段、第二阶段和端到端结果。"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Any

PIPELINE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(PIPELINE_DIR))

from common import ensure_article_split, filter_by_ids, load_jsonl  # noqa: E402
from config import (  # noqa: E402
    DEFAULT_DETECT_PATH,
    DEFAULT_EXTRACT_PATH,
    DEFAULT_SPLIT_PATH,
)
from pipeline_predict import (  # noqa: E402
    extract_for_type,
    load_pipeline,
    predict_types,
)


def event_to_spans(event: dict[str, Any] | None) -> tuple[
    set[tuple[Any, ...]],
    set[tuple[Any, ...]],
]:
    """把流水线 JSON 中的字符位置转成可比较的 span 集合。"""

    if event is None:
        return set(), set()
    trigger = set()
    if event["trigger"] is not None:
        trigger.add(
            (
                "trigger",
                None,
                event["trigger"]["start"],
                event["trigger"]["end"],
            )
        )
    args = {
        (
            "arg",
            argument["role"],
            argument["start"],
            argument["end"],
        )
        for argument in event["args"]
    }
    return trigger, args


def gold_spans(row: dict[str, Any]) -> tuple[
    set[tuple[Any, ...]],
    set[tuple[Any, ...]],
]:
    """读取原始标注中的触发词和论元字符区间。"""

    event = row["event"]
    trigger = {
        (
            "trigger",
            None,
            int(event["trigger"]["start"]),
            int(event["trigger"]["end"]),
        )
    }
    args = {
        (
            "arg",
            argument["role"],
            int(argument["start"]),
            int(argument["end"]),
        )
        for argument in event["args"]
    }
    return trigger, args


def prf(
    prediction: list[set[tuple[Any, ...]]],
    gold: list[set[tuple[Any, ...]]],
) -> tuple[float, float, float]:
    """汇总精确位置匹配的 P/R/F1。"""

    tp = sum(len(p & g) for p, g in zip(prediction, gold))
    fp = sum(len(p - g) for p, g in zip(prediction, gold))
    fn = sum(len(g - p) for p, g in zip(prediction, gold))
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-12)
    return precision, recall, f1


def span_metrics(
    records: list[tuple[
        set[tuple[Any, ...]],
        set[tuple[Any, ...]],
        set[tuple[Any, ...]],
        set[tuple[Any, ...]],
    ]],
) -> dict[str, float]:
    """根据触发词和论元记录计算抽取指标。"""

    trigger_p, trigger_r, trigger_f1 = prf(
        [record[0] for record in records],
        [record[1] for record in records],
    )
    arg_p, arg_r, arg_f1 = prf(
        [record[2] for record in records],
        [record[3] for record in records],
    )
    return {
        "trigger_precision": trigger_p,
        "trigger_recall": trigger_r,
        "trigger_f1": trigger_f1,
        "arg_precision": arg_p,
        "arg_recall": arg_r,
        "arg_f1": arg_f1,
        "event_exact_match": sum(
            trigger_pred == trigger_gold and arg_pred == arg_gold
            for (
                trigger_pred,
                trigger_gold,
                arg_pred,
                arg_gold,
            ) in records
        ) / max(len(records), 1),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="评估两阶段事件流水线")
    parser.add_argument("--stage1-dir", required=True)
    parser.add_argument("--stage2-dir", required=True)
    parser.add_argument("--detect-path", default=str(DEFAULT_DETECT_PATH))
    parser.add_argument("--extract-path", default=str(DEFAULT_EXTRACT_PATH))
    parser.add_argument("--split-path", default=str(DEFAULT_SPLIT_PATH))
    parser.add_argument("--output", default=None)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    bundle = load_pipeline(
        args.stage1_dir,
        args.stage2_dir,
        args.device,
    )
    detect_rows = load_jsonl(args.detect_path)
    extract_rows = load_jsonl(args.extract_path)
    split = ensure_article_split(
        detect_rows,
        args.split_path,
        0.1,
        args.seed,
    )
    rows = filter_by_ids(extract_rows, set(split["val_ids"]))
    random.Random(args.seed).shuffle(rows)
    if args.max_samples is not None:
        rows = rows[:args.max_samples]

    type_exact = 0
    type_hits = 0
    type_pred_count = 0
    oracle_records = []
    pipeline_records = []
    prediction_rows = []
    for row in rows:
        gold_type = row["event"]["type"]
        type_predictions = predict_types(bundle, row["text"])
        predicted_types = [
            item["event_type"] for item in type_predictions
        ]
        type_exact += predicted_types == [gold_type]
        type_hits += gold_type in predicted_types
        type_pred_count += len(predicted_types)

        trigger_gold, arg_gold = gold_spans(row)
        oracle_event = extract_for_type(
            bundle,
            row["text"],
            gold_type,
        )
        oracle_trigger, oracle_args = event_to_spans(oracle_event)
        oracle_records.append(
            (oracle_trigger, trigger_gold, oracle_args, arg_gold)
        )

        if gold_type in predicted_types:
            pipeline_event = extract_for_type(
                bundle,
                row["text"],
                gold_type,
            )
        else:
            pipeline_event = None
        pipeline_trigger, pipeline_args = event_to_spans(pipeline_event)
        pipeline_records.append(
            (
                pipeline_trigger,
                trigger_gold,
                pipeline_args,
                arg_gold,
            )
        )
        prediction_rows.append(
            {
                "id": row["id"],
                "sent_id": row["sent_id"],
                "text": row["text"],
                "gold_type": gold_type,
                "predicted_types": predicted_types,
                "oracle_extraction": oracle_event,
                "pipeline_event": pipeline_event,
            }
        )

    metrics = {
        "samples": len(rows),
        "stage1_type_subset_accuracy": type_exact / max(len(rows), 1),
        "stage1_gold_type_recall": type_hits / max(len(rows), 1),
        "stage1_average_predicted_types": type_pred_count / max(len(rows), 1),
        "stage2_gold_type": span_metrics(oracle_records),
        "pipeline_end_to_end": span_metrics(pipeline_records),
    }
    print(json.dumps(metrics, ensure_ascii=False, indent=2))

    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(
            json.dumps(metrics, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        prediction_path = output_path.with_name(
            output_path.stem + "_predictions.jsonl"
        )
        with prediction_path.open("w", encoding="utf-8") as file:
            for item in prediction_rows:
                file.write(json.dumps(item, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()

