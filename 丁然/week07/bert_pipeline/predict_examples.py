"""运行两阶段流水线示例并输出 JSON。"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PIPELINE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(PIPELINE_DIR))

from examples import EXAMPLES  # noqa: E402
from pipeline_predict import load_pipeline, predict_text  # noqa: E402


def expected_spans(example: dict) -> tuple[
    set[tuple],
    set[tuple],
]:
    """把示例 gold 标注转换成字符级 span 集合。"""

    event = example["event"]
    trigger = {
        (
            "trigger",
            None,
            event["trigger"]["start"],
            event["trigger"]["end"],
        )
    }
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


def predicted_spans(event: dict | None) -> tuple[
    set[tuple],
    set[tuple],
]:
    """从流水线输出中取出一个事件的 span 集合。"""

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


def main() -> None:
    parser = argparse.ArgumentParser(description="两阶段事件抽取示例")
    parser.add_argument("--stage1-dir", required=True)
    parser.add_argument("--stage2-dir", required=True)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    bundle = load_pipeline(
        args.stage1_dir,
        args.stage2_dir,
        args.device,
    )
    results = []
    for example in EXAMPLES:
        result = predict_text(bundle, example["text"])
        gold_type = example["event"]["type"]
        type_correct = result["event_types"] == [gold_type]
        matched_event = next(
            (
                event
                for event in result["events"]
                if event["event_type"] == gold_type
            ),
            None,
        )
        trigger_gold, args_gold = expected_spans(example)
        trigger_pred, args_pred = predicted_spans(matched_event)
        evaluation = {
            "type_correct": type_correct,
            "trigger_correct": trigger_pred == trigger_gold,
            "args_correct": args_pred == args_gold,
            "event_correct": (
                type_correct
                and trigger_pred == trigger_gold
                and args_pred == args_gold
            ),
        }
        results.append(
            {
                "id": example["id"],
                **result,
                "evaluation": evaluation,
            }
        )

    sample_count = len(results)
    summary = {
        "samples": sample_count,
        "type_accuracy": sum(
            result["evaluation"]["type_correct"]
            for result in results
        ) / sample_count,
        "trigger_exact_match": sum(
            result["evaluation"]["trigger_correct"]
            for result in results
        ) / sample_count,
        "arg_exact_match": sum(
            result["evaluation"]["args_correct"]
            for result in results
        ) / sample_count,
        "event_exact_match": sum(
            result["evaluation"]["event_correct"]
            for result in results
        ) / sample_count,
    }
    output = {
        "summary": summary,
        "results": results,
    }
    output_text = json.dumps(output, ensure_ascii=False, indent=2)
    print(output_text)
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(output_text + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()

