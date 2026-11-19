"""BERT 编码器和分类头。"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any

import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer
from transformers.utils import logging as transformers_logging


# 关闭基础模型加载时较长的层匹配报告。
transformers_logging.set_verbosity_error()


class BertClassifier(nn.Module):
    """支持 CLS、Mean、Max 三种句向量池化的 BERT 分类器。"""

    def __init__(
        self,
        model_path: str | None = None,
        encoder: nn.Module | None = None,
        num_labels: int = 15,
        pooling: str = "cls",
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        if encoder is None:
            if model_path is None:
                raise ValueError("model_path 和 encoder 必须提供其中一个")
            encoder = AutoModel.from_pretrained(
                str(Path(model_path).expanduser().resolve()),
                local_files_only=True,
                dtype=torch.float32,
            )
        self.encoder = encoder
        self.pooling = pooling
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(self.encoder.config.hidden_size, num_labels)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        **_: Any,
    ) -> SimpleNamespace:
        outputs = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True,
        )
        hidden = outputs.last_hidden_state

        if self.pooling == "cls":
            pooled = hidden[:, 0]
        elif self.pooling == "mean":
            mask = attention_mask.unsqueeze(-1).to(hidden.dtype)
            pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
        elif self.pooling == "max":
            mask = attention_mask.unsqueeze(-1).bool()
            hidden = hidden.masked_fill(~mask, torch.finfo(hidden.dtype).min)
            pooled = hidden.max(dim=1).values
        else:
            raise ValueError(f"不支持的池化方式: {self.pooling}")

        return SimpleNamespace(logits=self.classifier(self.dropout(pooled)))

    def get_input_embeddings(self) -> nn.Module:
        """供 FGM 获取 BERT 词嵌入层。"""

        return self.encoder.get_input_embeddings()


def load_tokenizer(model_path: str) -> Any:
    path = str(Path(model_path).expanduser().resolve())
    tokenizer = AutoTokenizer.from_pretrained(path, local_files_only=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.unk_token
    return tokenizer
