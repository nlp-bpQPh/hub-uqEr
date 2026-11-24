"""BERT 序列标注模型：Linear 头 / Linear + CRF 头。

两者共享同一个 BERT 主干，唯一区别是解码方式：
  - linear：逐 token argmax，标签之间独立；
  - linear+crf：整句 Viterbi，显式建模标签转移（BIO 约束）。
"""
from __future__ import annotations

import torch
import torch.nn as nn
from transformers import BertModel

from bert.crf import CRF, pack_sequences

IGNORE_INDEX = -100


class BertTagger(nn.Module):
    """BERT + Dropout + Linear（逐 token 分类）。"""

    def __init__(self, model_path: str, num_labels: int, dropout: float = 0.1):
        super().__init__()
        self.bert = BertModel.from_pretrained(model_path)
        hidden = self.bert.config.hidden_size
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden, num_labels)
        self.num_labels = num_labels

    def forward(self, input_ids, attention_mask, token_type_ids=None, labels=None):
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            return_dict=True,
        )
        logits = self.classifier(self.dropout(outputs.last_hidden_state))
        loss = None
        if labels is not None:
            loss = nn.functional.cross_entropy(
                logits.view(-1, self.num_labels), labels.view(-1),
                ignore_index=IGNORE_INDEX,
            )
        return {"logits": logits, "loss": loss}

    def decode(self, logits, attention_mask) -> list[list[int]]:
        return logits.argmax(dim=-1).tolist()


class BertCRFTagger(nn.Module):
    """BERT + Dropout + Linear + CRF。"""

    def __init__(self, model_path: str, num_labels: int, dropout: float = 0.1):
        super().__init__()
        self.bert = BertModel.from_pretrained(model_path)
        hidden = self.bert.config.hidden_size
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden, num_labels)
        self.crf = CRF(num_labels)
        self.num_labels = num_labels

    def forward(self, input_ids, attention_mask, token_type_ids=None, labels=None,
                return_logits: bool = True):
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            return_dict=True,
        )
        logits = self.classifier(self.dropout(outputs.last_hidden_state))

        loss = None
        if labels is not None:
            # 有效位置 = attention_mask=1 且 labels≠-100（跳过 CLS/SEP、subword 续接与 padding）
            labels = labels.to(attention_mask.device)
            mask = attention_mask.bool() & (labels != IGNORE_INDEX)
            # 压紧到"有效 token 在前的定长序列"，CRF 的 start/end 才落在真实边界
            packed_em, packed_tags, packed_mask = pack_sequences(
                logits, labels.clamp(min=0), mask.long())
            loss = self.crf.nll(packed_em, packed_tags, packed_mask)

        if return_logits:
            return {"logits": logits, "loss": loss}
        return {"loss": loss}

    @torch.no_grad()
    def decode(self, logits, attention_mask, labels=None) -> list[list[int]]:
        if labels is not None:
            labels = labels.to(attention_mask.device)
            mask = attention_mask.bool() & (labels != IGNORE_INDEX)
        else:
            mask = attention_mask.bool()
        packed_em, _, packed_mask = pack_sequences(logits, None, mask.long())
        return self.crf.decode(packed_em, packed_mask)


def build_model(kind: str, model_path: str, num_labels: int, dropout: float = 0.1):
    if kind == "linear":
        return BertTagger(model_path, num_labels, dropout)
    if kind == "crf":
        return BertCRFTagger(model_path, num_labels, dropout)
    raise ValueError(f"未知模型类型: {kind}")
