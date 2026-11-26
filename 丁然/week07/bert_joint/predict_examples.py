"""批量预测示例文本，并输出格式化 JSON。"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_DIR))

from examples import EXAMPLES  # noqa: E402
from predict import load_model_bundle, predict  # noqa: E402


def expected_spans(example: dict) -> tuple[set[tuple], set[tuple]]:
    """把示例 gold 标注转换成字符级 span。"""

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
    """从单模型输出中提取字符级 span。"""

    trigger = set()
    if result["trigger"] is not None:
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
        for argument in result["args"]
    }
    return trigger, args


def main() -> None:
    parser = argparse.ArgumentParser(description="批量运行示例事件抽取")
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--device", default="auto")
    parser.add_argument(
        "--output",
        default=None,
        help="可选 JSON 输出路径；不传则只打印到终端",
    )
    args = parser.parse_args()

    # 1. 只加载一次模型，循环预测所有示例。
    bundle = load_model_bundle(args.model_dir, args.device)
    results = []
    for example in EXAMPLES:
        result = predict(
            bundle.model,
            bundle.tokenizer,
            bundle.label_space,
            example["text"],
            int(bundle.metadata["max_length"]),
            bundle.device,
        )
        trigger_gold, args_gold = expected_spans(example)
        trigger_pred, args_pred = predicted_spans(result)
        type_correct = result["event_type"] == example["event"]["type"]
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
            result["evaluation"]["type_correct"] for result in results
        ) / sample_count,
        "trigger_exact_match": sum(
            result["evaluation"]["trigger_correct"] for result in results
        ) / sample_count,
        "arg_exact_match": sum(
            result["evaluation"]["args_correct"] for result in results
        ) / sample_count,
        "event_exact_match": sum(
            result["evaluation"]["event_correct"] for result in results
        ) / sample_count,
    }
    output = {
        "summary": summary,
        "results": results,
    }
    output_text = json.dumps(output, ensure_ascii=False, indent=2)
    print(output_text)

    # 3. 可选写入文件，便于和本次训练结果一起保存。
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(output_text + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
