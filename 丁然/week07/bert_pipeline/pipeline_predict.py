"""加载两个阶段，对输入文本执行端到端事件抽取。"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from transformers import AutoConfig, AutoModel, AutoTokenizer
from transformers.utils import logging as transformers_logging

PIPELINE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(PIPELINE_DIR))

from common import EventSchema  # noqa: E402
from config import get_device  # noqa: E402
from stage1_model import DetectorModel  # noqa: E402
from stage2_data import ExtractionLabels, extract_spans  # noqa: E402
from stage2_model import ConditionalExtractor  # noqa: E402


transformers_logging.set_verbosity_error()


@dataclass
class PipelineBundle:
    stage1_model: DetectorModel
    stage1_tokenizer: Any
    stage1_meta: dict[str, Any]
    schema: EventSchema
    stage2_model: ConditionalExtractor
    stage2_tokenizer: Any
    stage2_meta: dict[str, Any]
    labels: ExtractionLabels
    device: torch.device


def load_pipeline(
    stage1_dir: str | Path,
    stage2_dir: str | Path,
    device_name: str = "auto",
) -> PipelineBundle:
    """从两个训练目录恢复完整流水线。"""

    stage1_dir = Path(stage1_dir)
    stage2_dir = Path(stage2_dir)
    device = get_device(device_name)

    stage1_meta = json.loads(
        (stage1_dir / "meta.json").read_text(encoding="utf-8")
    )
    schema = EventSchema.from_dict(
        json.loads(
            (stage1_dir / "schema.json").read_text(encoding="utf-8")
        )
    )
    stage1_tokenizer = AutoTokenizer.from_pretrained(
        stage1_dir,
        local_files_only=True,
    )
    stage1_encoder = AutoModel.from_config(
        AutoConfig.from_pretrained(stage1_dir, local_files_only=True),
        dtype=torch.float32,
    )
    stage1_model = DetectorModel(
        stage1_encoder,
        num_labels=len(schema.event_types),
        dropout=0.0,
    )
    stage1_model.load_state_dict(
        torch.load(
            stage1_dir / "model.pt",
            map_location="cpu",
            weights_only=True,
        )
    )
    stage1_model.to(device).eval()

    stage2_meta = json.loads(
        (stage2_dir / "meta.json").read_text(encoding="utf-8")
    )
    labels = ExtractionLabels.from_dict(
        json.loads(
            (stage2_dir / "labels.json").read_text(encoding="utf-8")
        )
    )
    stage2_tokenizer = AutoTokenizer.from_pretrained(
        stage2_dir,
        local_files_only=True,
    )
    stage2_encoder = AutoModel.from_config(
        AutoConfig.from_pretrained(stage2_dir, local_files_only=True),
        dtype=torch.float32,
    )
    stage2_model = ConditionalExtractor(
        stage2_encoder,
        labels,
        num_event_types=len(schema.event_types),
        decoder=stage2_meta["decoder"],
        dropout=0.0,
        cls_token_id=stage2_tokenizer.cls_token_id,
        sep_token_id=stage2_tokenizer.sep_token_id,
    )
    stage2_model.load_state_dict(
        torch.load(
            stage2_dir / "model.pt",
            map_location="cpu",
            weights_only=True,
        )
    )
    stage2_model.to(device).eval()
    return PipelineBundle(
        stage1_model=stage1_model,
        stage1_tokenizer=stage1_tokenizer,
        stage1_meta=stage1_meta,
        schema=schema,
        stage2_model=stage2_model,
        stage2_tokenizer=stage2_tokenizer,
        stage2_meta=stage2_meta,
        labels=labels,
        device=device,
    )


def predict_types(
    bundle: PipelineBundle,
    text: str,
) -> list[dict[str, Any]]:
    """第一阶段输出事件类型和 sigmoid 分数。"""

    encoded = bundle.stage1_tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=int(bundle.stage1_meta["max_length"]),
    )
    input_ids = encoded["input_ids"].to(bundle.device)
    attention_mask = encoded["attention_mask"].to(bundle.device)
    with torch.inference_mode():
        logits = bundle.stage1_model(input_ids, attention_mask)
        probabilities = torch.sigmoid(logits)[0].cpu().tolist()

    threshold = float(bundle.stage1_meta["threshold"])
    return [
        {
            "event_type": event_type,
            "score": probability,
        }
        for event_type, probability in zip(
            bundle.schema.event_types,
            probabilities,
        )
        if probability >= threshold
    ]


def _span_to_dict(
    text: str,
    offsets: list[tuple[int, int]],
    span: Any,
) -> dict[str, Any]:
    """把 token span 转成原文字符区间。"""

    start = offsets[span.start][0]
    end = offsets[span.end - 1][1]
    return {
        "role": span.role,
        "text": text[start:end],
        "start": start,
        "end": end,
    }


def extract_for_type(
    bundle: PipelineBundle,
    text: str,
    event_type: str,
) -> dict[str, Any]:
    """第二阶段针对一个已知事件类型抽取触发词和论元。"""

    encoded = bundle.stage2_tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=int(bundle.stage2_meta["max_length"]),
        return_offsets_mapping=True,
    )
    offsets = [
        tuple(map(int, pair))
        for pair in encoded.pop("offset_mapping")[0].tolist()
    ]
    input_ids = encoded["input_ids"].to(bundle.device)
    attention_mask = encoded["attention_mask"].to(bundle.device)
    event_type_ids = torch.tensor(
        [bundle.schema.event_to_id[event_type]],
        dtype=torch.long,
        device=bundle.device,
    )
    with torch.inference_mode():
        emissions = bundle.stage2_model(
            input_ids,
            attention_mask,
            event_type_ids,
        )
        tags = bundle.stage2_model.decode(
            emissions,
            attention_mask,
            input_ids,
        )[0]

    spans = extract_spans(tags, bundle.labels, event_type)
    trigger_spans = [span for span in spans if span.kind == "trigger"]
    trigger = None
    if trigger_spans:
        trigger = _span_to_dict(text, offsets, trigger_spans[0])
        trigger.pop("role")
    return {
        "event_type": event_type,
        "trigger": trigger,
        "args": [
            _span_to_dict(text, offsets, span)
            for span in spans
            if span.kind == "arg"
        ],
    }


def predict_text(bundle: PipelineBundle, text: str) -> dict[str, Any]:
    """执行完整两阶段流程。"""

    type_predictions = predict_types(bundle, text)
    events = []
    for type_prediction in type_predictions:
        event = extract_for_type(
            bundle,
            text,
            type_prediction["event_type"],
        )
        event["score"] = type_prediction["score"]
        events.append(event)
    return {
        "text": text,
        "event_types": [
            item["event_type"] for item in type_predictions
        ],
        "events": events,
    }

