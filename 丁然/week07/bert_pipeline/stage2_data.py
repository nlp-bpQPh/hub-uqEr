"""第二阶段条件抽取的标签定义和数据编码。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from torch.utils.data import Dataset

from common import EventSchema


class ExtractionLabels:
    """第二阶段标签：O、触发词 B/I 和 61 个角色 B/I，共 125 类。"""

    def __init__(self, schema: EventSchema) -> None:
        self.schema = schema
        self.outside_id = 0
        self.trigger_b_id = 1
        self.trigger_i_id = 2
        self.role_tag_ids: dict[str, tuple[int, int]] = {}
        self.tag_to_role: dict[int, str] = {}
        self.tag_is_begin: dict[int, bool] = {}
        self.label_names = ["O", "B-触发词", "I-触发词"]

        next_id = 3
        for role in schema.roles:
            begin_id = next_id
            inside_id = next_id + 1
            next_id += 2
            self.role_tag_ids[role] = (begin_id, inside_id)
            self.tag_to_role[begin_id] = role
            self.tag_to_role[inside_id] = role
            self.tag_is_begin[begin_id] = True
            self.tag_is_begin[inside_id] = False
            self.label_names.extend([f"B-{role}", f"I-{role}"])

        self.num_labels = next_id
        if self.num_labels != 125:
            raise ValueError(f"预期 125 个标签，实际为 {self.num_labels}")

    def to_dict(self) -> dict[str, Any]:
        return {
            "num_labels": self.num_labels,
            "outside_id": self.outside_id,
            "trigger_b_id": self.trigger_b_id,
            "trigger_i_id": self.trigger_i_id,
            "label_names": self.label_names,
            "role_tag_ids": self.role_tag_ids,
            "schema": self.schema.to_dict(),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ExtractionLabels":
        """从训练目录的 labels.json 恢复第二阶段标签空间。"""

        schema = EventSchema.from_dict(data["schema"])
        instance = cls(schema)
        if int(data["num_labels"]) != instance.num_labels:
            raise ValueError("保存的第二阶段标签数量不匹配")
        return instance


@dataclass(frozen=True)
class ExtractionSpan:
    kind: str
    role: str | None
    start: int
    end: int


@dataclass
class ExtractionExample:
    input_ids: list[int]
    attention_mask: list[int]
    labels: list[int]
    event_type_id: int
    offsets: list[tuple[int, int]]
    row_index: int


class ExtractionDataset(Dataset):
    """把文本、事件类型和 BIO 标签编码成第二阶段输入。"""

    def __init__(
        self,
        rows: list[dict[str, Any]],
        tokenizer: Any,
        labels: ExtractionLabels,
        max_length: int,
    ) -> None:
        self.rows = rows
        self.examples = [
            self._encode(row, index, tokenizer, labels, max_length)
            for index, row in enumerate(rows)
        ]

    @staticmethod
    def _encode(
        row: dict[str, Any],
        row_index: int,
        tokenizer: Any,
        labels: ExtractionLabels,
        max_length: int,
    ) -> ExtractionExample:
        encoded = tokenizer(
            row["text"],
            add_special_tokens=True,
            truncation=True,
            max_length=max_length,
            return_offsets_mapping=True,
        )
        input_ids = list(encoded["input_ids"])
        offsets = [
            tuple(map(int, pair))
            for pair in encoded["offset_mapping"]
        ]
        tags = [-100] * len(input_ids)
        for index, (start, end) in enumerate(offsets):
            if index > 0 and start < end:
                tags[index] = labels.outside_id

        event = row["event"]
        candidates: list[list[tuple[int, int]]] = [
            [] for _ in input_ids
        ]
        raw_spans = [
            (
                "trigger",
                None,
                int(event["trigger"]["start"]),
                int(event["trigger"]["end"]),
                0,
            )
        ]
        raw_spans.extend(
            (
                "arg",
                argument["role"],
                int(argument["start"]),
                int(argument["end"]),
                1,
            )
            for argument in event["args"]
        )

        for kind, role, span_start, span_end, priority in raw_spans:
            token_indexes = [
                index
                for index, (token_start, token_end) in enumerate(offsets)
                if token_start < token_end
                and max(token_start, span_start) < min(token_end, span_end)
            ]
            for piece_index, token_index in enumerate(token_indexes):
                if kind == "trigger":
                    tag_id = (
                        labels.trigger_b_id
                        if piece_index == 0
                        else labels.trigger_i_id
                    )
                else:
                    assert role is not None
                    begin_id, inside_id = labels.role_tag_ids[role]
                    tag_id = begin_id if piece_index == 0 else inside_id
                candidates[token_index].append((priority, tag_id))

        for index, token_candidates in enumerate(candidates):
            if token_candidates:
                tags[index] = min(token_candidates)[1]

        return ExtractionExample(
            input_ids=input_ids,
            attention_mask=[1] * len(input_ids),
            labels=tags,
            event_type_id=labels.schema.event_to_id[event["type"]],
            offsets=offsets,
            row_index=row_index,
        )

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, index: int) -> ExtractionExample:
        return self.examples[index]


def collate_extraction(
    examples: list[ExtractionExample],
    pad_token_id: int,
) -> dict[str, Any]:
    """动态 padding，同时保留事件类型和字符偏移。"""

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
    event_type_ids = torch.tensor(
        [example.event_type_id for example in examples],
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
        "event_type_ids": event_type_ids,
        "offsets": [example.offsets for example in examples],
        "row_indices": torch.tensor(
            [example.row_index for example in examples],
            dtype=torch.long,
        ),
    }


def extract_spans(
    tags: list[int],
    labels: ExtractionLabels,
    event_type: str | None = None,
) -> list[ExtractionSpan]:
    """把第二阶段 BIO 标签还原成触发词和论元区间。"""

    spans: list[ExtractionSpan] = []
    active_kind: str | None = None
    active_role: str | None = None
    active_start = 0

    def close_span(end: int) -> None:
        nonlocal active_kind, active_role
        if active_kind is None:
            return
        if (
            active_kind == "arg"
            and event_type is not None
            and active_role not in labels.schema.valid_roles(event_type)
        ):
            active_kind = None
            active_role = None
            return
        spans.append(
            ExtractionSpan(
                kind=active_kind,
                role=active_role,
                start=active_start,
                end=end,
            )
        )
        active_kind = None
        active_role = None

    for index, tag_id in enumerate(tags):
        if tag_id < 0 or tag_id == labels.outside_id:
            close_span(index)
            continue
        if tag_id == labels.trigger_b_id:
            close_span(index)
            active_kind = "trigger"
            active_role = None
            active_start = index
            continue
        if tag_id == labels.trigger_i_id:
            if active_kind != "trigger":
                close_span(index)
                active_kind = "trigger"
                active_role = None
                active_start = index
            continue

        role = labels.tag_to_role.get(tag_id)
        if role is None:
            close_span(index)
            continue
        is_begin = labels.tag_is_begin[tag_id]
        if is_begin or active_kind != "arg" or active_role != role:
            close_span(index)
            active_kind = "arg"
            active_role = role
            active_start = index

    close_span(len(tags))
    return spans


def print_extraction_examples(
    dataset_name: str,
    dataset: ExtractionDataset,
    tokenizer: Any,
    labels: ExtractionLabels,
    max_samples: int,
) -> None:
    """打印第二阶段文本、事件类型和 token 标签。"""

    for index in range(min(max_samples, len(dataset))):
        row = dataset.rows[index]
        example = dataset.examples[index]
        tokens = tokenizer.convert_ids_to_tokens(example.input_ids)
        label_names = [
            "IGNORE(-100)" if tag_id == -100 else labels.label_names[tag_id]
            for tag_id in example.labels
        ]
        event_type = labels.schema.id_to_event[example.event_type_id]
        print(
            f"\n[第二阶段数据检查:{dataset_name}] 样本={index + 1} "
            f"事件类型={event_type} token数={len(tokens)}",
            flush=True,
        )
        print("输入文本：", flush=True)
        print(row["text"].replace("\n", "\\n"), flush=True)
        print("输入文本转token之后：", flush=True)
        for start in range(0, len(tokens), 12):
            print("  " + " | ".join(tokens[start:start + 12]), flush=True)
        print("标签：", flush=True)
        for start in range(0, len(label_names), 12):
            print(
                "  " + " | ".join(label_names[start:start + 12]),
                flush=True,
            )

