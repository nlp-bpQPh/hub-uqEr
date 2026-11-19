"""原始新闻数据读取和 BERT 输入编码。"""

from __future__ import annotations

import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader, Dataset

from bert_config import NUM_LABELS


class NewsDataset(Dataset):
    """保存已经分词的输入、mask 和数字标签。"""

    def __init__(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: torch.Tensor,
    ) -> None:
        self.input_ids = input_ids
        self.attention_mask = attention_mask
        self.labels = labels

    def __len__(self) -> int:
        return self.labels.numel()

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        return {
            "input_ids": self.input_ids[index],
            "attention_mask": self.attention_mask[index],
            "labels": self.labels[index],
        }


def load_rows(path: str | Path) -> list[dict[str, Any]]:
    """读取并检查原始 JSON，不修改数据文件。"""

    rows = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(rows, list) or not rows:
        raise ValueError(f"数据文件为空或格式错误: {path}")
    for index, row in enumerate(rows):
        if not isinstance(row.get("sentence"), str) or not row["sentence"].strip():
            raise ValueError(f"第 {index} 条数据缺少 sentence")
        if not isinstance(row.get("label"), int) or not 0 <= row["label"] < 15:
            raise ValueError(f"第 {index} 条数据标签非法")
    return rows


def balanced_subset(
    rows: list[dict[str, Any]],
    max_samples: int | None,
    seed: int,
) -> list[dict[str, Any]]:
    """小数据测试时按类别轮流取样，避免漏掉某些类别。"""

    if max_samples is None or max_samples >= len(rows):
        return rows

    buckets: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        buckets[row["label"]].append(row)
    rng = random.Random(seed)
    for bucket in buckets.values():
        rng.shuffle(bucket)

    result: list[dict[str, Any]] = []
    offsets = {label: 0 for label in range(NUM_LABELS)}
    while len(result) < max_samples:
        added = False
        for label in range(NUM_LABELS):
            if offsets[label] < len(buckets[label]) and len(result) < max_samples:
                result.append(buckets[label][offsets[label]])
                offsets[label] += 1
                added = True
        if not added:
            break
    rng.shuffle(result)
    return result


def encode_rows(
    tokenizer: Any,
    rows: list[dict[str, Any]],
    max_length: int,
) -> NewsDataset:
    encoded = tokenizer(
        [row["sentence"] for row in rows],
        padding="max_length",
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
    )
    return NewsDataset(
        input_ids=encoded["input_ids"],
        attention_mask=encoded["attention_mask"],
        labels=torch.tensor([row["label"] for row in rows], dtype=torch.long),
    )


def make_loader(
    dataset: NewsDataset,
    batch_size: int,
    shuffle: bool,
    seed: int,
) -> DataLoader:
    generator = torch.Generator().manual_seed(seed)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        generator=generator,
    )
