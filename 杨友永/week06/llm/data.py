"""LLM 的 SFT 数据：把 (商品名, 品类) 转成「提示 + 答案」的 token 序列。

labels 只在答案部分有值，提示部分为 -100 —— 这就是 SFT 与预训练的唯一区别。
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader, Dataset

from dataset import ProductDataset

SYSTEM = "你是一个电商商品分类助手，只回答品类名称。"


def build_prompt(name: str) -> str:
    return f"请判断以下商品名属于哪个品类。\n商品名：{name}\n品类："


def build_chat(tok: Any, name: str) -> str:
    extra = {}
    if "enable_thinking" in (tok.chat_template or ""):
        extra["enable_thinking"] = False
    return tok.apply_chat_template(
        [{"role": "system", "content": SYSTEM},
         {"role": "user", "content": build_prompt(name)}],
        tokenize=False, add_generation_prompt=True, **extra)


class SFTDataset(Dataset[dict]):
    def __init__(self, ds: ProductDataset, tok: Any, max_len: int = 128) -> None:
        self.items = []
        for x in ds.items:
            p_ids = tok(build_chat(tok, x.text), add_special_tokens=False)["input_ids"]
            a_ids = tok(x.category + tok.eos_token, add_special_tokens=False)["input_ids"]
            ids = (p_ids + a_ids)[:max_len]
            labels = ([-100] * len(p_ids) + a_ids)[:max_len]
            self.items.append({"input_ids": ids, "labels": labels})

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, i: int) -> dict:
        return self.items[i]


def make_collate(tok: Any):
    pad_id = tok.pad_token_id or tok.eos_token_id

    def collate(batch: list[dict]) -> dict[str, torch.Tensor]:
        max_len = max(len(b["input_ids"]) for b in batch)
        input_ids, labels, mask = [], [], []
        for b in batch:
            pad = max_len - len(b["input_ids"])
            input_ids.append(b["input_ids"] + [pad_id] * pad)
            labels.append(b["labels"] + [-100] * pad)
            mask.append([1] * len(b["input_ids"]) + [0] * pad)
        return {"input_ids": torch.tensor(input_ids, dtype=torch.long),
                "attention_mask": torch.tensor(mask, dtype=torch.long),
                "labels": torch.tensor(labels, dtype=torch.long)}
    return collate


def build_loader(ds: SFTDataset, tok: Any, batch_size: int,
                 shuffle: bool = False, seed: int = 42) -> DataLoader:
    generator = torch.Generator().manual_seed(seed)
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle,
                      collate_fn=make_collate(tok), num_workers=0,
                      generator=generator, drop_last=shuffle)


def label_token_ids(tok: Any, names: list[str]) -> list[int]:
    """每个类别的首 token id（用于「约束解码」式评测）。"""
    ids = []
    for n in names:
        enc = tok(n, add_special_tokens=False)["input_ids"]
        if not enc:
            raise ValueError(f"类别 {n!r} 分词为空")
        ids.append(enc[0])
    return ids
