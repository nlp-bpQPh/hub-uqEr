"""运行 LLM few-shot 示例并计算示例级准确率。"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ICL_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(ICL_DIR))

from config import (  # noqa: E402
    DEFAULT_MODEL_PATH,
    DEFAULT_SCHEMA_PATH,
    DEFAULT_TRAIN_EXTRACT_PATH,
)
from examples import EXAMPLES  # noqa: E402
from model import LLMFewShot  # noqa: E402
from parser import normalize_output  # noqa: E402
from prompts import build_messages, select_demonstrations  # noqa: E402
from schema import EventSchema, load_jsonl  # noqa: E402


def expected_spans(example: dict) -> tuple[set[tuple], set[tuple]]:
    """读取示例 gold 的触发词和论元字符区间。"""

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


def predicted_spans(result: dict) -> tuple[set[tuple], set[tuple]]:
    """从 LLM 输出中取出 gold 类型的 span。"""

    if not result:
        return set(), set()
    trigger = set()
    if result.get("trigger") is not None:
        trigger.add(
            (
                "trigger",
                None,
                result["trigger"]["start"],
                result["trigger"]["end"],
            )
        )
    args = {
        (
            "arg",
            argument["role"],
            argument["start"],
            argument["end"],
        )
        for argument in result.get("args", [])
    }
    return trigger, args


def main() -> None:
    parser = argparse.ArgumentParser(description="LLM few-shot 示例评估")
    parser.add_argument("--model-path", default=DEFAULT_MODEL_PATH)
    parser.add_argument("--schema-path", default=str(DEFAULT_SCHEMA_PATH))
    parser.add_argument(
        "--train-extract-path",
        default=str(DEFAULT_TRAIN_EXTRACT_PATH),
    )
    parser.add_argument("--shots-per-type", type=int, default=1)
    parser.add_argument("--max-input-tokens", type=int, default=8192)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto")
    parser.add_argument(
        "--attn-implementation",
        choices=["eager", "sdpa"],
        default="eager",
    )
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    schema = EventSchema(args.schema_path)
    demonstrations = select_demonstrations(
        load_jsonl(args.train_extract_path),
        max(args.shots_per_type, 0),
        args.seed,
    )
    runtime = LLMFewShot(
        args.model_path,
        args.device,
        args.attn_implementation,
    )

    results = []
    examples = EXAMPLES[:args.limit] if args.limit else EXAMPLES
    for example in examples:
        raw_output = runtime.generate(
            build_messages(example["text"], demonstrations, schema),
            args.max_input_tokens,
            args.max_new_tokens,
        )
        normalized = normalize_output(raw_output, example["text"], schema)
        gold_type = example["event"]["type"]
        matched_event = next(
            (
                event
                for event in normalized["events"]
                if event["event_type"] == gold_type
            ),
            None,
        )
        trigger_gold, args_gold = expected_spans(example)
        trigger_pred, args_pred = predicted_spans(matched_event or {})
        type_correct = [event["event_type"] for event in normalized["events"]] == [
            gold_type
        ]
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
                **normalized,
                "evaluation": evaluation,
            }
        )

    sample_count = len(results)
    summary = {
        "samples": sample_count,
        "shots_per_type": args.shots_per_type,
        "demonstrations": len(demonstrations),
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

