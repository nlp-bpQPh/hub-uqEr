"""BERT/biLSTM taggers, reusing the course CRF implementation."""
from __future__ import annotations

import sys
from pathlib import Path

import torch
from torch import nn

_ROOT = Path(__file__).resolve().parents[1]
_COURSE_COMMON = _ROOT.parents[1] / "week7-信息抽取" / "code" / "common"
if _COURSE_COMMON.is_dir() and str(_COURSE_COMMON) not in sys.path:
    sys.path.insert(0, str(_COURSE_COMMON))
from crf import CRF  # noqa: E402

_TORCHVISION_STUB = None


def _load_auto_model_class():
    """Work around this machine's mismatched optional torchvision install.

    Transformers only needs the missing torchvision::nms registration during
    import for this text-only BERT task; no vision op is executed.
    """
    global _TORCHVISION_STUB
    try:
        torch.ops.torchvision.nms
    except (AttributeError, RuntimeError):
        _TORCHVISION_STUB = torch.library.Library("torchvision", "FRAGMENT")
        try:
            _TORCHVISION_STUB.define("nms(Tensor boxes, Tensor scores, float iou_threshold) -> Tensor")
        except RuntimeError:
            pass
    from transformers import AutoModel
    return AutoModel


class BiLSTMTagger(nn.Module):
    def __init__(self, vocab_size: int, num_tags: int, emb_dim: int = 96,
                 hidden: int = 128, dropout: float = 0.25, decoder: str = "crf"):
        super().__init__()
        self.decoder_type = decoder
        self.embedding = nn.Embedding(vocab_size, emb_dim, padding_idx=0)
        self.dropout = nn.Dropout(dropout)
        self.lstm = nn.LSTM(emb_dim, hidden, batch_first=True, bidirectional=True)
        self.classifier = nn.Linear(hidden * 2, num_tags)
        self.crf = CRF(num_tags) if decoder == "crf" else None

    def emissions(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        x = self.dropout(self.embedding(input_ids))
        x, _ = self.lstm(x)
        return self.classifier(self.dropout(x))


class BertTagger(nn.Module):
    def __init__(self, model_name: str, num_tags: int, decoder: str = "linear",
                 num_layers: int | None = None):
        super().__init__()
        AutoModel = _load_auto_model_class()
        self.encoder = AutoModel.from_pretrained(model_name)
        if num_layers is not None:
            layers = self.encoder.encoder.layer
            if not 1 <= num_layers <= len(layers):
                raise ValueError(f"num_layers must be between 1 and {len(layers)}")
            self.encoder.encoder.layer = nn.ModuleList(list(layers[:num_layers]))
            self.encoder.config.num_hidden_layers = num_layers
        hidden = int(self.encoder.config.hidden_size)
        self.dropout = nn.Dropout(0.1)
        self.classifier = nn.Linear(hidden, num_tags)
        self.decoder_type = decoder
        self.crf = CRF(num_tags) if decoder == "crf" else None

    def emissions(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        output = self.encoder(input_ids=input_ids, attention_mask=attention_mask.long())
        return self.classifier(self.dropout(output.last_hidden_state))


def hard_bio_constraints(num_tags: int) -> tuple[torch.Tensor, torch.Tensor]:
    """BIO transition mask for the course CRF's hard-decoding ablation."""
    from xfund_data import TAGS
    if num_tags != len(TAGS):
        raise ValueError("Unexpected tag inventory")
    allowed = torch.zeros((num_tags, num_tags), dtype=torch.bool)
    for i, prev in enumerate(TAGS):
        for j, cur in enumerate(TAGS):
            if cur == "O" or cur.startswith("B-"):
                allowed[i, j] = True
            elif cur.startswith("I-"):
                cls = cur[2:]
                allowed[i, j] = prev in (f"B-{cls}", f"I-{cls}")
    start = torch.zeros(num_tags, dtype=torch.bool)
    start[0] = True  # The sequence starts with [CLS], supervised as O.
    return allowed, start
