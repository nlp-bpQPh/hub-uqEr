"""BERT 分类模型：四种池化方式。

    cls       [CLS] 向量（BERT 的默认做法，基线）
    mean      所有 token 的掩码均值
    max       所有 token 的掩码最大
    cls_mean  [CLS] 与 mean 拼接后接分类头
"""
from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn
from transformers import AutoModel

POOLINGS = ("cls", "mean", "max", "cls_mean")


class BertClassifier(nn.Module):
    def __init__(self, model_path: str | Path, num_classes: int,
                 pooling: str = "cls", dropout: float = 0.1) -> None:
        super().__init__()
        if pooling not in POOLINGS:
            raise ValueError(f"pooling 必须是 {POOLINGS} 之一，得到 {pooling!r}")
        self.pooling = pooling
        self.encoder = AutoModel.from_pretrained(
            str(model_path), local_files_only=True, return_dict=True)
        hidden = int(self.encoder.config.hidden_size)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden * (2 if pooling == "cls_mean" else 1),
                                    num_classes)

    def num_parameters(self) -> dict[str, float]:
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return {"total": total, "trainable": trainable,
                "trainable_pct": round(100 * trainable / total, 2)}

    def pool(self, hidden: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        if self.pooling == "cls":
            return hidden[:, 0]

        m = mask.unsqueeze(-1).to(hidden.dtype)
        if self.pooling == "mean":
            return (hidden * m).sum(1) / m.sum(1).clamp(min=1e-9)

        if self.pooling == "max":
            # PAD 位置必须屏蔽成 -inf，否则 max 会取到 PAD 的隐状态
            filled = hidden.masked_fill(~mask.bool().unsqueeze(-1),
                                        torch.finfo(hidden.dtype).min)
            return filled.max(dim=1).values

        # cls_mean
        cls = hidden[:, 0]
        mean = (hidden * m).sum(1) / m.sum(1).clamp(min=1e-9)
        return torch.cat([cls, mean], dim=-1)

    def forward(self, input_ids, attention_mask, token_type_ids=None) -> torch.Tensor:
        kwargs = {"input_ids": input_ids, "attention_mask": attention_mask}
        if token_type_ids is not None:
            kwargs["token_type_ids"] = token_type_ids
        out = self.encoder(**kwargs)
        return self.classifier(self.dropout(self.pool(out.last_hidden_state, attention_mask)))


def build_model(model_path, num_classes: int, pooling: str = "cls",
                device=None, dropout: float = 0.1) -> BertClassifier:
    model = BertClassifier(model_path, num_classes, pooling, dropout)
    return model.to(device) if device is not None else model
