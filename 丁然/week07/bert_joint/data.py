"""数据读取、标签编码和批处理。"""

from __future__ import annotations

import json
import random
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import torch
from torch.utils.data import Dataset

from labels import LabelSpace


@dataclass(frozen=True)
class TokenSpan:
    """用 token 下标表示的触发词或论元。"""

    kind: str
    role: str | None
    start: int
    end: int

    def key(self) -> tuple[str, str | None, int, int]:
        return self.kind, self.role, self.start, self.end


@dataclass
class EncodedExample:
    input_ids: list[int]
    attention_mask: list[int]
    labels: list[int]
    offsets: list[tuple[int, int]]
    row_index: int


def load_rows(path: str | Path) -> list[dict[str, Any]]:
    """逐行读取 JSONL，并做最小字段检查。"""

    rows: list[dict[str, Any]] = []
    with Path(path).open(encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row.get("text"), str) or not row["text"]:
                raise ValueError(f"第 {line_number} 行缺少 text")
            event = row.get("event")
            if not isinstance(event, dict) or "trigger" not in event:
                raise ValueError(f"第 {line_number} 行缺少 event.trigger")
            rows.append(row)
    if not rows:
        raise ValueError(f"数据为空: {path}")
    return rows


def _balanced_limit(
    rows: list[dict[str, Any]],
    max_count: int | None,
    seed: int,
) -> list[dict[str, Any]]:
    """按事件类型轮流取样，避免小样本测试漏掉类别。"""

    if max_count is None or max_count >= len(rows):
        return rows

    buckets: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        buckets[row["event"]["type"]].append(row)

    rng = random.Random(seed)
    event_types = sorted(buckets)
    for event_type in event_types:
        rng.shuffle(buckets[event_type])

    offsets = {event_type: 0 for event_type in event_types}
    result: list[dict[str, Any]] = []
    while len(result) < max_count:
        added = False
        for event_type in event_types:
            index = offsets[event_type]
            if index < len(buckets[event_type]) and len(result) < max_count:
                result.append(buckets[event_type][index])
                offsets[event_type] += 1
                added = True
        if not added:
            break
    rng.shuffle(result)
    return result


def split_rows(
    rows: Iterable[dict[str, Any]],
    val_ratio: float,
    seed: int,
    max_train: int | None = None,
    max_val: int | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """按事件类型分层切分训练集和验证集。"""

    buckets: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        buckets[row["event"]["type"]].append(row)

    rng = random.Random(seed)
    train_rows: list[dict[str, Any]] = []
    val_rows: list[dict[str, Any]] = []
    for event_type in sorted(buckets):
        bucket = list(buckets[event_type])
        rng.shuffle(bucket)
        val_count = max(1, round(len(bucket) * val_ratio))
        val_count = min(val_count, max(1, len(bucket) - 1))
        val_rows.extend(bucket[:val_count])
        train_rows.extend(bucket[val_count:])

    train_rows = _balanced_limit(train_rows, max_train, seed + 1)
    val_rows = _balanced_limit(val_rows, max_val, seed + 2)
    rng.shuffle(train_rows)
    rng.shuffle(val_rows)
    return train_rows, val_rows


def _overlaps(
    token_start: int,
    token_end: int,
    span_start: int,
    span_end: int,
) -> bool:
    return max(token_start, span_start) < min(token_end, span_end)


class EventDataset(Dataset):
    """提前编码文本和统一标签，训练时只做动态 padding。"""

    def __init__(
        self,
        rows: list[dict[str, Any]],
        tokenizer: Any,
        label_space: LabelSpace,
        max_length: int,
    ) -> None:
        self.examples = [
            self._encode(row, index, tokenizer, label_space, max_length)
            for index, row in enumerate(rows)
        ]
        self.rows = rows

    @staticmethod
    def _encode(
        row: dict[str, Any],
        row_index: int,
        tokenizer: Any,
        label_space: LabelSpace,
        max_length: int,
    ) -> EncodedExample:
        # 1. 编码文本并取得每个 token 对应的原文字符区间。
        encoded = tokenizer(
            row["text"],
            add_special_tokens=True,
            truncation=True,
            max_length=max_length,
            return_offsets_mapping=True,
        )
        input_ids = list(encoded["input_ids"])
        attention_mask = [1] * len(input_ids)
        offsets = [tuple(map(int, pair)) for pair in encoded["offset_mapping"]]

        # 2. 初始化为忽略；只有正文 token 参与分类和 CRF。
        event = row["event"]
        event_type = event["type"]
        labels = [-100] * len(input_ids)
        for index, (start, end) in enumerate(offsets):
            if index > 0 and start < end:
                labels[index] = label_space.outside_id

        # 3. 汇总当前事件的全部触发词和论元标注。
        # BIO 一个位置只能放一个标签；重叠时触发词优先。
        candidates: list[list[tuple[int, int]]] = [
            [] for _ in input_ids
        ]
        spans: list[tuple[str, str | None, int, int, int]] = [
            (
                "trigger",
                None,
                int(event["trigger"]["start"]),
                int(event["trigger"]["end"]),
                0,
            )
        ]
        for argument in event["args"]:
            spans.append(
                (
                    "arg",
                    argument["role"],
                    int(argument["start"]),
                    int(argument["end"]),
                    1,
                )
            )

        # 4. 把原文字符区间映射到 token 区间。
        for kind, role, span_start, span_end, priority in spans:
            # 一个字符可能落在多个子词中，所有重叠子词都要带标签。
            token_indexes = [
                index
                for index, (token_start, token_end) in enumerate(offsets)
                if token_start < token_end
                and _overlaps(
                    token_start,
                    token_end,
                    span_start,
                    span_end,
                )
            ]
            # 5. 生成 BIO 标签并记录当前位置的所有候选标签。
            for piece_index, token_index in enumerate(token_indexes):
                # 每段实体只有第一个 token 是 B，其余是 I。
                if kind == "trigger":
                    begin_id, inside_id = (
                        label_space.event_trigger_tag_ids[event_type]
                    )
                    tag_id = begin_id if piece_index == 0 else inside_id
                else:
                    assert role is not None
                    begin_id, inside_id = label_space.role_tag_ids[role]
                    tag_id = begin_id if piece_index == 0 else inside_id
                candidates[token_index].append((priority, tag_id))

        # 6. 每个 token 只保留优先级最高的一个标签。
        for index, token_candidates in enumerate(candidates):
            # 元组先比较优先级，再比较标签编号，保证编码结果可复现。
            if token_candidates:
                labels[index] = min(token_candidates)[1]

        return EncodedExample(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            offsets=offsets,
            row_index=row_index,
        )

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, index: int) -> EncodedExample:
        return self.examples[index]


def collate_examples(
    examples: list[EncodedExample],
    pad_token_id: int,
) -> dict[str, Any]:
    """动态 padding；标签的 padding 仍保持为 -100。"""

    # 每个 batch 单独补到本批最长长度，不固定补到 max_length。
    max_length = max(len(example.input_ids) for example in examples)
    batch_size = len(examples)
    input_ids = torch.full(
        (batch_size, max_length),
        pad_token_id,
        dtype=torch.long,
    )
    attention_mask = torch.zeros(
        (batch_size, max_length),
        dtype=torch.long,
    )
    labels = torch.full(
        (batch_size, max_length),
        -100,
        dtype=torch.long,
    )
    for row_index, example in enumerate(examples):
        length = len(example.input_ids)
        input_ids[row_index, :length] = torch.tensor(
            example.input_ids,
            dtype=torch.long,
        )
        attention_mask[row_index, :length] = 1
        labels[row_index, :length] = torch.tensor(
            example.labels,
            dtype=torch.long,
        )

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
        "offsets": [example.offsets for example in examples],
        "row_indices": torch.tensor(
            [example.row_index for example in examples],
            dtype=torch.long,
        ),
    }


def extract_token_spans(
    labels: list[int],
    label_space: LabelSpace,
    event_type: str | None = None,
) -> list[TokenSpan]:
    """把 BIO 标签解码为触发词和论元 token 区间。"""

    spans: list[TokenSpan] = []
    active_kind: str | None = None
    active_role: str | None = None
    active_trigger_event: str | None = None
    active_start = 0

    def close_span(end: int) -> None:
        nonlocal active_kind, active_role, active_trigger_event
        if active_kind is None:
            return
        if active_kind == "arg" and event_type is not None:
            if active_role not in label_space.valid_roles(event_type):
                active_kind = None
                active_role = None
                return
        spans.append(
            TokenSpan(
                kind=active_kind,
                role=active_role,
                start=active_start,
                end=end,
            )
        )
        active_kind = None
        active_role = None
        active_trigger_event = None

    valid_roles = (
        label_space.valid_roles(event_type)
        if event_type is not None
        else None
    )
    for index, tag_id in enumerate(labels):
        # O 和 padding 会结束当前实体。
        if tag_id < 0 or tag_id == label_space.outside_id:
            close_span(index)
            continue
        trigger_event = label_space.trigger_tag_to_event.get(tag_id)
        if trigger_event is not None:
            is_begin = label_space.trigger_tag_is_begin[tag_id]
            if (
                is_begin
                or active_kind != "trigger"
                or active_trigger_event != trigger_event
            ):
                close_span(index)
                active_kind = "trigger"
                active_role = None
                active_trigger_event = trigger_event
                active_start = index
            continue

        role = label_space.tag_to_role.get(tag_id)
        if role is None:
            close_span(index)
            continue
        if valid_roles is not None and role not in valid_roles:
            close_span(index)
            continue

        # 孤立 I 也视为新实体起点，避免解码时整段丢失。
        is_begin = label_space.tag_is_begin[tag_id]
        if is_begin or active_kind != "arg" or active_role != role:
            close_span(index)
            active_kind = "arg"
            active_role = role
            active_start = index

    close_span(len(labels))
    return spans


def event_type_from_labels(
    labels: list[int],
    label_space: LabelSpace,
) -> str | None:
    """从触发词标签中提取事件类型。"""

    for tag_id in labels:
        event_type = label_space.trigger_tag_to_event.get(tag_id)
        if event_type is not None:
            return event_type
    return None


def print_data_examples(
    dataset_name: str,
    rows: list[dict[str, Any]],
    examples: list[EncodedExample],
    tokenizer: Any,
    label_space: LabelSpace,
    max_samples: int,
) -> None:
    """打印真实训练样本，核对文本、token 和标签是否对齐。"""

    if max_samples <= 0:
        return
    for sample_index in range(min(max_samples, len(examples))):
        row = rows[sample_index]
        example = examples[sample_index]
        event_type = row["event"]["type"]
        raw_tokens = tokenizer.convert_ids_to_tokens(example.input_ids)
        covered_end = max(
            (end for _, end in example.offsets),
            default=0,
        )

        print(
            f"\n[数据检查:{dataset_name}] 样本={sample_index + 1} "
            f"id={row.get('id')} 事件类型={event_type} "
            f"token数={len(example.input_ids)} "
            f"字符数={len(row['text'])} "
            f"截断={'是' if covered_end < len(row['text']) else '否'}",
            flush=True,
        )
        print("输入文本：", flush=True)
        print(row["text"].replace("\n", "\\n"), flush=True)

        # token 和标签分开打印，方便逐列核对训练输入。
        display_tokens: list[str] = []
        display_labels: list[str] = []
        for token, tag_id, (start, end) in zip(raw_tokens, example.labels, example.offsets):
            token_text = row["text"][start:end] if end > start else token
            display_tokens.append(token_text)
            display_labels.append(
                "IGNORE(-100)"
                if tag_id == -100
                else label_space.label_names[tag_id]
            )

        print("输入文本转token之后：", flush=True)
        for start in range(0, len(display_tokens), 12):
            chunk = display_tokens[start:start + 12]
            print("  " + " | ".join(chunk), flush=True)
        print("标签：", flush=True)
        for start in range(0, len(display_labels), 12):
            chunk = display_labels[start:start + 12]
            print("  " + " | ".join(chunk), flush=True)

        # 从生成的标签反解 span，检查是否与原始标注一致。
        spans = extract_token_spans(
            example.labels,
            label_space,
            event_type,
        )
        print("标签反解结果:", flush=True)
        for span in spans:
            char_start = example.offsets[span.start][0]
            char_end = example.offsets[span.end - 1][1]
            print(
                f"  {span.kind} role={span.role} "
                f"token=[{span.start},{span.end}) "
                f"char=[{char_start},{char_end}) "
                f"text={row['text'][char_start:char_end]!r}",
                flush=True,
            )

