"""BERT 事件抽取模型。"""

from __future__ import annotations

from types import SimpleNamespace

import torch
import torch.nn as nn

from crf import CRF


class EventTagger(nn.Module):
    """共享 BERT 编码器，每个 token 输出 149 类序列标签。"""

    def __init__(
        self,
        encoder: nn.Module,
        label_space: object,
        decoder: str = "linear",
        dropout: float = 0.1,
        cls_token_id: int = 101,
        sep_token_id: int = 102,
    ) -> None:
        super().__init__()
        if decoder not in {"linear", "crf"}:
            raise ValueError(f"不支持的解码方式: {decoder}")
        self.encoder = encoder
        self.label_space = label_space
        self.decoder = decoder
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(
            self.encoder.config.hidden_size,
            self.label_space.num_labels,
        )
        self.crf = (
            CRF(self.label_space.num_labels)
            if decoder == "crf"
            else None
        )

        text_ids = torch.tensor(
            self.label_space.text_label_ids,
            dtype=torch.long,
        )
        self.register_buffer("text_ids", text_ids)
        self.cls_token_id = int(cls_token_id)
        self.sep_token_id = int(sep_token_id)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> SimpleNamespace:
        # 1. 使用共享 BERT 编码整条文本。
        hidden = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True,
        ).last_hidden_state

        # 2. 每个位置输出 149 个序列标签分数。
        emissions = self.classifier(self.dropout(hidden))

        # 3. 根据当前位置是 CLS 还是正文，生成合法标签掩码。
        position_mask = self._build_position_mask(
            input_ids,
            attention_mask,
        )
        # 4. 非法标签压到极小值，linear 和 CRF 解码都会遵守该限制。
        emissions = emissions.masked_fill(
            ~position_mask,
            -1e4,
        )
        return SimpleNamespace(
            emissions=emissions,
            position_mask=position_mask,
        )

    def _build_position_mask(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        """[CLS] 只允许 O，其余有效位置允许 149 类序列标签。"""

        batch_size, sequence_length = input_ids.shape
        valid = torch.zeros(
            (
                batch_size,
                sequence_length,
                self.label_space.num_labels,
            ),
            dtype=torch.bool,
            device=input_ids.device,
        )
        active = attention_mask.bool()
        is_cls = input_ids.eq(self.cls_token_id)
        # [CLS] 不承载内容，只允许标成 O。
        valid[:, :, self.label_space.outside_id] = (
            active & is_cls
        )
        # 正文和 [SEP] 仅开放 O 与各种 BIO 标签。
        valid[:, :, self.text_ids] = (
            active & ~is_cls
        ).unsqueeze(-1)
        return valid

    def loss(
        self,
        emissions: torch.Tensor,
        labels: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        if self.crf is not None:
            # 去掉 CLS；SEP 由 -100 标签在 CRF 内部屏蔽。
            return self.crf(
                emissions[:, 1:],
                labels[:, 1:],
                attention_mask[:, 1:],
            )

        # 所有位置统一做序列标签分类。
        return nn.functional.cross_entropy(
            emissions.reshape(-1, self.label_space.num_labels),
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
            # CRF 只在正文 token 上解码，不把 CLS、SEP 当成标签位置。
            content_mask = (
                attention_mask[:, 1:].bool()
                & input_ids[:, 1:].ne(self.sep_token_id)
            )
            paths = self.crf.decode(
                emissions[:, 1:],
                content_mask,
            )
            # 补回 CLS、SEP 和 padding，保持序列位置与原输入对齐。
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

