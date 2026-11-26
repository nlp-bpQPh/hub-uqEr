"""第一阶段多标签事件检测模型。"""

from __future__ import annotations

import torch
import torch.nn as nn


class DetectorModel(nn.Module):
    """BERT + 13 维 sigmoid 事件类型输出。"""

    def __init__(
        self,
        encoder: nn.Module,
        num_labels: int,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.encoder = encoder
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(
            self.encoder.config.hidden_size,
            num_labels,
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        hidden = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True,
        ).last_hidden_state
        # [CLS] 表示整段文本，输出 13 个独立 logit。
        return self.classifier(self.dropout(hidden[:, 0]))

    def loss(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        # 多标签任务中 13 个事件类型分别使用二分类交叉熵。
        return nn.functional.binary_cross_entropy_with_logits(
            logits,
            labels,
        )

