"""统计事件抽取数据的真实 BERT token 长度和截断损失。"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

from transformers import AutoTokenizer


DEFAULT_MODEL_PATH = "~/Downloads/模型/bert/bert-base-chinese"
DEFAULT_DATA_PATH = Path(__file__).resolve().parents[1] / "data" / "train_extract.jsonl"


def _quantile(sorted_values: list[int], ratio: float) -> int:
    # 使用简单下标分位数，样本量足够时与插值结果影响很小。
    index = min(len(sorted_values) - 1, int(len(sorted_values) * ratio))
    return sorted_values[index]


def main() -> None:
    parser = argparse.ArgumentParser(description="分析事件抽取数据长度")
    parser.add_argument("--data-path", default=str(DEFAULT_DATA_PATH))
    parser.add_argument("--model-path", default=DEFAULT_MODEL_PATH)
    args = parser.parse_args()

    # 必须使用实际训练模型的 tokenizer，字符数不能代替 token 数。
    tokenizer = AutoTokenizer.from_pretrained(
        str(Path(args.model_path).expanduser().resolve()),
        local_files_only=True,
    )
    # 读取全部正样本，不在这里做训练/验证切分。
    rows = [
        json.loads(line)
        for line in Path(args.data_path).open(encoding="utf-8")
        if line.strip()
    ]

    lengths: list[int] = []
    type_spans: list[tuple[str, list[tuple[str, int]]]] = []
    for row in rows:
        # 不截断编码，才能计算每个候选 max_length 的真实损失。
        encoded = tokenizer(
            row["text"],
            add_special_tokens=True,
            truncation=False,
            return_offsets_mapping=True,
        )
        offsets = encoded["offset_mapping"]
        lengths.append(len(encoded["input_ids"]))
        event = row["event"]
        # 触发词和论元都先转成原文字符区间。
        raw_spans = [
            (
                "trigger",
                int(event["trigger"]["start"]),
                int(event["trigger"]["end"]),
            )
        ]
        raw_spans.extend(
            ("arg", int(argument["start"]), int(argument["end"]))
            for argument in event["args"]
        )

        token_spans: list[tuple[str, int]] = []
        for kind, span_start, span_end in raw_spans:
            # 记录实体最后一个 token，判断它是否越过截断位置。
            indexes = [
                index
                for index, (token_start, token_end) in enumerate(offsets)
                if token_start < token_end
                and max(token_start, span_start)
                < min(token_end, span_end)
            ]
            token_spans.append(
                (kind, indexes[-1] if indexes else -1)
            )
        # 同时保留事件类型，后面统计不同类别受截断的影响。
        type_spans.append((event["type"], token_spans))

    sorted_lengths = sorted(lengths)
    print(
        "token_length "
        f"min={sorted_lengths[0]} "
        f"p50={_quantile(sorted_lengths, 0.50)} "
        f"p75={_quantile(sorted_lengths, 0.75)} "
        f"p90={_quantile(sorted_lengths, 0.90)} "
        f"p95={_quantile(sorted_lengths, 0.95)} "
        f"p99={_quantile(sorted_lengths, 0.99)} "
        f"max={sorted_lengths[-1]} "
        f"mean={sum(sorted_lengths) / len(sorted_lengths):.1f}"
    )

    # 对比多个候选上限，重点看标注是否丢失而不是只看文本长度。
    limits = [64, 96, 128, 160, 192, 224, 256, 288, 320, 352, 384]
    total_args = sum(
        1
        for _, spans in type_spans
        for kind, _ in spans
        if kind == "arg"
    )
    print(
        "limit rows_truncated event_loss trigger_loss "
        "arg_loss arg_loss_rate"
    )
    for limit in limits:
        rows_truncated = sum(length > limit for length in lengths)
        event_loss = trigger_loss = arg_loss = 0
        for _, spans in type_spans:
            event_loss += any(last >= limit for _, last in spans)
            trigger_loss += sum(
                kind == "trigger" and last >= limit
                for kind, last in spans
            )
            arg_loss += sum(
                kind == "arg" and last >= limit
                for kind, last in spans
            )
        print(
            f"{limit:5d} {rows_truncated:14d} {event_loss:10d} "
            f"{trigger_loss:12d} {arg_loss:8d} "
            f"{arg_loss / total_args:.4%}"
        )

    # 部分长尾事件更容易出现在长文本中，因此再按类型统计。
    by_type: dict[str, list[list[tuple[str, int]]]] = defaultdict(list)
    for event_type, spans in type_spans:
        by_type[event_type].append(spans)
    print("\n事件类型在关键 max_length 下的截断事件数")
    print("type count 128 192 256 320 384")
    for event_type in sorted(by_type):
        losses = [
            sum(
                any(last >= limit for _, last in spans)
                for spans in by_type[event_type]
            )
            for limit in [128, 192, 256, 320, 384]
        ]
        print(
            f"{event_type} {len(by_type[event_type])} "
            + " ".join(map(str, losses))
        )


if __name__ == "__main__":
    main()
