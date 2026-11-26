"""第一阶段事件检测数据编码。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from torch.utils.data import Dataset

from common import EventSchema


@dataclass
class DetectorExample:
    input_ids: list[int]
    attention_mask: list[int]
    labels: list[float]
    row_index: int


class DetectorDataset(Dataset):
    """把文本编码成 BERT 输入和 13 维多标签。"""

    def __init__(
        self,
        rows: list[dict[str, Any]],
        tokenizer: Any,
        schema: EventSchema,
        max_length: int,
    ) -> None:
        self.rows = rows
        self.examples = [
            self._encode(row, index, tokenizer, schema, max_length)
            for index, row in enumerate(rows)
        ]

    @staticmethod
    def _encode(
        row: dict[str, Any],
        row_index: int,
        tokenizer: Any,
        schema: EventSchema,
        max_length: int,
    ) -> DetectorExample:
        # 第一阶段只需要文本和 13 维多标签，不需要字符位置。
        encoded = tokenizer(
            row["text"],
            add_special_tokens=True,
            truncation=True,
            max_length=max_length,
        )
        labels = [0.0] * len(schema.event_types)
        for event_type in row["types"]:
            labels[schema.event_to_id[event_type]] = 1.0
        input_ids = list(encoded["input_ids"])
        return DetectorExample(
            input_ids=input_ids,
            attention_mask=[1] * len(input_ids),
            labels=labels,
            row_index=row_index,
        )

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, index: int) -> DetectorExample:
        return self.examples[index]


def collate_detector(
    examples: list[DetectorExample],
    pad_token_id: int,
) -> dict[str, torch.Tensor]:
    """动态 padding，并构造多标签训练张量。"""

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
    labels = torch.tensor(
        [example.labels for example in examples],
        dtype=torch.float32,
    )
    for row_index, example in enumerate(examples):
        length = len(example.input_ids)
        input_ids[row_index, :length] = torch.tensor(
            example.input_ids,
            dtype=torch.long,
        )
        attention_mask[row_index, :length] = 1
    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
    }


def print_detector_examples(
    dataset_name: str,
    dataset: DetectorDataset,
    tokenizer: Any,
    schema: EventSchema,
    max_samples: int,
) -> None:
    """打印第一阶段输入和 multi-hot 标签。"""

    for index in range(min(max_samples, len(dataset))):
        row = dataset.rows[index]
        example = dataset.examples[index]
        tokens = tokenizer.convert_ids_to_tokens(example.input_ids)
        active_labels = [
            event_type
            for event_type, value in zip(schema.event_types, example.labels)
            if value == 1
        ]
        print(
            f"\n[第一阶段数据检查:{dataset_name}] "
            f"样本={index + 1} id={row['id']} "
            f"token数={len(tokens)}",
            flush=True,
        )
        print("输入文本：", flush=True)
        print(row["text"].replace("\n", "\\n"), flush=True)
        print("输入文本转token之后：", flush=True)
        for start in range(0, len(tokens), 12):
            print(
                "  " + " | ".join(tokens[start:start + 12]),
                flush=True,
            )
        print(f"事件类型标签：{active_labels or ['无事件']}", flush=True)

