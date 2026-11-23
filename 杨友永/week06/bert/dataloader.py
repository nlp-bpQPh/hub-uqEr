"""BERT 的 DataLoader：分词、padding、组 batch。"""
from __future__ import annotations

from typing import Any

import torch
from torch.utils.data import DataLoader

from dataset import ProductDataset

MAX_LEN = 32          # 商品名 P95 = 21，32 足够


def build_tokenizer(model_path) -> Any:
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained(str(model_path), local_files_only=True)


def make_collate(tok: Any, max_len: int = MAX_LEN):
    def collate(items) -> dict[str, torch.Tensor]:
        enc = tok([x.text for x in items], padding=True, truncation=True,
                  max_length=max_len, return_tensors="pt")
        out = {"input_ids": enc["input_ids"],
               "attention_mask": enc["attention_mask"],
               "labels": torch.tensor([x.label for x in items], dtype=torch.long)}
        if "token_type_ids" in enc:
            out["token_type_ids"] = enc["token_type_ids"]
        return out
    return collate


def build_loader(ds: ProductDataset, tok: Any, batch_size: int,
                 max_len: int = MAX_LEN, shuffle: bool = False,
                 seed: int = 42) -> DataLoader:
    generator = torch.Generator().manual_seed(seed)
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle,
                      collate_fn=make_collate(tok, max_len),
                      num_workers=0, generator=generator, drop_last=shuffle)


def build_all(tok: Any, batch_size: int, max_len: int = MAX_LEN
              ) -> tuple[DataLoader, DataLoader, DataLoader]:
    return (build_loader(ProductDataset("train"), tok, batch_size, max_len, shuffle=True),
            build_loader(ProductDataset("val"), tok, batch_size, max_len),
            build_loader(ProductDataset("test"), tok, batch_size, max_len))
