"""LLM 分类评测：在答案位置比较 10 个类别首 token 的 logits。

比「自由生成再解析」快得多，而且与 BERT 的 macro-F1 口径一致。
"""
from __future__ import annotations

from typing import Any

import torch

from dataset import ProductDataset
from llm.data import build_chat


@torch.no_grad()
def predict(model: Any, tok: Any, ds: ProductDataset, label_ids: list[int],
            device, batch_size: int = 16, max_len: int = 128) -> list[int]:
    model.eval()
    label_t = torch.tensor(label_ids, device=device)
    preds: list[int] = []
    texts = [x.text for x in ds.items]

    for i in range(0, len(texts), batch_size):
        chunk = texts[i:i + batch_size]
        enc = tok([build_chat(tok, t) for t in chunk], padding=True,
                  truncation=True, max_length=max_len, return_tensors="pt",
                  add_special_tokens=False)
        enc = {k: v.to(device) for k, v in enc.items()}
        logits = model(**enc).logits[:, -1, :]          # 左填充，最后一个位置是答案位
        scores = logits[:, label_t]                     # [B, 10]
        preds.extend(scores.argmax(-1).cpu().tolist())
    return preds
