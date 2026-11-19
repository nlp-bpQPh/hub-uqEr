"""数据管道：jsonl → Pydantic 模型 → chat template → 带掩码的张量。

关键点：prompt 是完整样本的严格前缀，所以 mask 位置 = prompt 的 token 数，
不需要做字符级对齐。
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader, Dataset

from schemas import SFTExample

IGNORE: int = -100

TokenIds = list[int]
Sample = tuple[TokenIds, TokenIds]          # (input_ids, labels)
Batch = dict[str, torch.Tensor]


def load_examples(path: Path) -> list[SFTExample]:
    """逐行读取 jsonl，交给 Pydantic 校验。"""
    examples: list[SFTExample] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                examples.append(SFTExample.model_validate_json(line))
    return examples


class SFTDataset(Dataset[Sample]):
    """把 (instruction, output) 变成 (input_ids, labels)。"""

    def __init__(self, examples: list[SFTExample], tok: Any, max_len: int) -> None:
        self.items: list[Sample] = []
        self.dropped: int = 0
        for ex in examples:
            messages = [
                {"role": "user", "content": ex.instruction},
                {"role": "assistant", "content": ex.output},
            ]
            full = tok.apply_chat_template(messages, tokenize=False)
            prompt = tok.apply_chat_template(
                messages[:1], tokenize=False, add_generation_prompt=True)

            prompt_ids: TokenIds = tok(prompt, add_special_tokens=False)["input_ids"]
            full_ids: TokenIds = tok(full, add_special_tokens=False)["input_ids"]

            if len(full_ids) > max_len:      # 超长样本直接丢，避免截断回答
                self.dropped += 1
                continue

            labels: TokenIds = [IGNORE] * len(prompt_ids) + full_ids[len(prompt_ids):]
            self.items.append((full_ids, labels))

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, index: int) -> Sample:
        return self.items[index]


def make_collate(pad_id: int):
    """右侧 padding；labels 用 -100 填充，不参与 loss。"""
    def collate(batch: list[Sample]) -> Batch:
        max_len: int = max(len(ids) for ids, _ in batch)
        input_ids = torch.full((len(batch), max_len), pad_id, dtype=torch.long)
        labels = torch.full((len(batch), max_len), IGNORE, dtype=torch.long)
        mask = torch.zeros((len(batch), max_len), dtype=torch.long)
        for row, (ids, lab) in enumerate(batch):
            input_ids[row, : len(ids)] = torch.tensor(ids)
            labels[row, : len(lab)] = torch.tensor(lab)
            mask[row, : len(ids)] = 1
        return {"input_ids": input_ids, "attention_mask": mask, "labels": labels}
    return collate


def build_loader(path: Path, tok: Any, max_len: int, batch_size: int,
                 shuffle: bool) -> tuple[SFTDataset, DataLoader]:
    """读文件 → 建数据集 → 包成 DataLoader。"""
    dataset = SFTDataset(load_examples(path), tok, max_len)
    loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=shuffle,
        collate_fn=make_collate(tok.pad_token_id), num_workers=0,
    )
    return dataset, loader
