"""数据层：读取 Peoples Daily NER 数据，做字符级 BIO 对齐，构造 DataLoader。

数据格式：{"tokens": ["海","钓",...], "ner_tags": ["O","B-LOC",...]}
标签集：O / B-PER / I-PER / B-ORG / I-ORG / B-LOC / I-LOC
"""
from __future__ import annotations

import json
from functools import partial
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Dataset


# ---------------------------------------------------------------- 原始数据
def load_split(path: str | Path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def load_label_names(path: str | Path) -> list[str]:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def build_label_maps(label_names: list[str]) -> tuple[dict[str, int], dict[int, str]]:
    label2id = {name: i for i, name in enumerate(label_names)}
    id2label = {i: name for name, i in label2id.items()}
    return label2id, id2label


# ---------------------------------------------------------------- Dataset
class NERDataset(Dataset):
    """把 (tokens, tags) 转成模型输入。

    对齐规则（关键）：
      - 用 tokenizer(..., is_split_into_words=True) 保持字符边界；
      - 一个字可能被切成多个 subword：只有第一个 subword 保留标签，其余 -100；
      - [CLS]/[SEP]/padding 全部 -100；
      - 超过 max_len 的部分直接截断（标签同步截断）。
    """

    def __init__(self, samples: list[dict], tokenizer, label2id: dict[str, int],
                 max_len: int = 128):
        self.samples = samples
        self.tokenizer = tokenizer
        self.label2id = label2id
        self.max_len = max_len

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict:
        item = self.samples[idx]
        tokens: list[str] = item["tokens"]
        tags: list[str] = item["ner_tags"]

        enc = self.tokenizer(
            tokens,
            is_split_into_words=True,
            truncation=True,
            max_length=self.max_len,
            padding=False,
            return_token_type_ids=True,
        )
        word_ids = enc.word_ids()
        labels: list[int] = []
        prev_word = None
        for wid in word_ids:
            if wid is None:                     # [CLS]/[SEP]
                labels.append(-100)
            elif wid != prev_word:              # 该字的第一个 subword
                labels.append(self.label2id[tags[wid]])
            else:                               # 同一字的后续 subword
                labels.append(-100)
            prev_word = wid

        return {
            "input_ids": enc["input_ids"],
            "attention_mask": enc["attention_mask"],
            "token_type_ids": enc.get("token_type_ids", [0] * len(enc["input_ids"])),
            "labels": labels,
            "length": len(item["tokens"]),
        }


def collate_fn(batch: list[dict], pad_token_id: int = 0):
    max_len = max(len(b["input_ids"]) for b in batch)
    input_ids, attention_mask, token_type_ids, labels = [], [], [], []
    for b in batch:
        pad = max_len - len(b["input_ids"])
        input_ids.append(b["input_ids"] + [pad_token_id] * pad)
        attention_mask.append(b["attention_mask"] + [0] * pad)
        token_type_ids.append(b["token_type_ids"] + [0] * pad)
        labels.append(b["labels"] + [-100] * pad)
    return {
        "input_ids": torch.tensor(input_ids, dtype=torch.long),
        "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
        "token_type_ids": torch.tensor(token_type_ids, dtype=torch.long),
        "labels": torch.tensor(labels, dtype=torch.long),
    }


def make_loader(samples: list[dict], tokenizer, label2id: dict[str, int],
                batch_size: int, max_len: int, shuffle: bool,
                num_workers: int = 0):
    ds = NERDataset(samples, tokenizer, label2id, max_len)
    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=partial(collate_fn, pad_token_id=tokenizer.pad_token_id or 0),
        pin_memory=torch.cuda.is_available(),
    )


def select_subset(samples: list[dict], limit: int | None, seed: int = 42) -> list[dict]:
    """按固定种子抽样，用于快速冒烟 / 数据量消融。"""
    if not limit or limit >= len(samples):
        return samples
    import random
    rng = random.Random(seed)
    return rng.sample(samples, limit)
