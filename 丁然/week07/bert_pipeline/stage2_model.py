"""第二阶段事件类型条件抽取模型。"""

from __future__ import annotations

import torch
import torch.nn as nn

from crf import CRF


class ConditionalExtractor(nn.Module):
    """BERT 文本表示加事件类型 embedding，再输出 125 类序列标签。"""

    def __init__(
        self,
        encoder: nn.Module,
        labels: object,
        num_event_types: int,
        decoder: str = "linear",
        dropout: float = 0.1,
        cls_token_id: int = 101,
        sep_token_id: int = 102,
    ) -> None:
        super().__init__()
        self.encoder = encoder
        self.labels = labels
        self.decoder = decoder
        self.dropout = nn.Dropout(dropout)
        self.type_embedding = nn.Embedding(
            num_event_types,
            self.encoder.config.hidden_size,
        )
        self.classifier = nn.Linear(
            self.encoder.config.hidden_size,
            labels.num_labels,
        )
        self.crf = CRF(labels.num_labels) if decoder == "crf" else None
        self.cls_token_id = int(cls_token_id)
        self.sep_token_id = int(sep_token_id)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        event_type_ids: torch.Tensor,
    ) -> torch.Tensor:
        hidden = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True,
        ).last_hidden_state

        # 事件类型向量加到每个 token，显式告诉模型当前抽取哪类事件。
        type_vector = self.type_embedding(event_type_ids).unsqueeze(1)
        emissions = self.classifier(self.dropout(hidden + type_vector))
        is_cls = input_ids.eq(self.cls_token_id)
        emissions = emissions.masked_fill(
            is_cls.unsqueeze(-1),
            -1e4,
        )
        return emissions

    def loss(
        self,
        emissions: torch.Tensor,
        labels: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        if self.crf is not None:
            # 去掉 CLS，SEP 和 padding 通过 -100 屏蔽。
            return self.crf(
                emissions[:, 1:],
                labels[:, 1:],
                attention_mask[:, 1:],
            )
        return nn.functional.cross_entropy(
            emissions.reshape(-1, self.labels.num_labels),
            labels.reshape(-1),
            ignore_index=-100,
        )

    def decode(
        self,
        emissions: torch.Tensor,
        attention_mask: torch.Tensor,
        input_ids: torch.Tensor,
    ) -> list[list[int]]:
        if self.crf is not None:
            content_mask = (
                attention_mask[:, 1:].bool()
                & input_ids[:, 1:].ne(self.sep_token_id)
            )
            paths = self.crf.decode(emissions[:, 1:], content_mask)
            return [
                [-100]
                + path
                + [-100] * (emissions.shape[1] - len(path) - 1)
                for path in paths
            ]
        prediction = emissions.argmax(dim=-1)
        prediction = prediction.masked_fill(
            ~attention_mask.bool(),
            -100,
        )
        return prediction.tolist()

