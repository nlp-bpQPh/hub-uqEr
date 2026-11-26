"""加载训练结果，对单条文本做事件抽取。"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from transformers import AutoConfig, AutoModel, AutoTokenizer
from transformers.utils import logging as transformers_logging

PROJECT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_DIR))

from config import get_device  # noqa: E402
from data import (  # noqa: E402
    TokenSpan,
    event_type_from_labels,
    extract_token_spans,
)
from labels import LabelSpace  # noqa: E402
from model import EventTagger  # noqa: E402


transformers_logging.set_verbosity_error()


@dataclass
class ModelBundle:
    """预测所需的模型、tokenizer、标签空间和元数据。"""

    model: EventTagger
    tokenizer: Any
    label_space: LabelSpace
    metadata: dict[str, Any]
    device: torch.device


def _span_to_dict(
    text: str,
    offsets: list[tuple[int, int]],
    span: TokenSpan,
) -> dict[str, Any]:
    # 第一个 token 的开头到最后一个 token 的结尾就是原文字符区间。
    char_start = offsets[span.start][0]
    char_end = offsets[span.end - 1][1]
    return {
        "role": span.role,
        "text": text[char_start:char_end],
        "start": char_start,
        "end": char_end,
    }


def load_model_bundle(
    model_dir: str | Path,
    device_name: str = "auto",
) -> ModelBundle:
    """从训练输出目录恢复一套完整预测环境。"""

    model_dir = Path(model_dir)
    metadata = json.loads(
        (model_dir / "meta.json").read_text(encoding="utf-8")
    )
    label_data = json.loads(
        (model_dir / "label_space.json").read_text(encoding="utf-8")
    )
    label_space = LabelSpace.from_dict(label_data)
    tokenizer = AutoTokenizer.from_pretrained(
        model_dir,
        local_files_only=True,
    )
    config = AutoConfig.from_pretrained(
        model_dir,
        local_files_only=True,
    )
    encoder = AutoModel.from_config(
        config,
        dtype=torch.float32,
    )
    model = EventTagger(
        encoder=encoder,
        label_space=label_space,
        decoder=metadata["decoder"],
        dropout=0.0,
        cls_token_id=tokenizer.cls_token_id,
        sep_token_id=tokenizer.sep_token_id,
    )
    state_dict = torch.load(
        model_dir / "model.pt",
        map_location="cpu",
        weights_only=True,
    )
    model.load_state_dict(state_dict)
    device = get_device(device_name)
    model.to(device).eval()
    return ModelBundle(
        model=model,
        tokenizer=tokenizer,
        label_space=label_space,
        metadata=metadata,
        device=device,
    )


def predict(
    model: EventTagger,
    tokenizer: Any,
    label_space: LabelSpace,
    text: str,
    max_length: int,
    device: torch.device,
) -> dict[str, Any]:
    # tokenizer 负责截断，offset_mapping 负责恢复原文字符位置。
    encoded = tokenizer(
        text,
        add_special_tokens=True,
        truncation=True,
        max_length=max_length,
        return_offsets_mapping=True,
        return_tensors="pt",
    )
    offsets = [
        tuple(map(int, pair))
        for pair in encoded.pop("offset_mapping")[0].tolist()
    ]
    input_ids = encoded["input_ids"].to(device)
    attention_mask = encoded["attention_mask"].to(device)
    with torch.inference_mode():
        output = model(input_ids, attention_mask)
        predictions = model.decode(
            output.emissions,
            attention_mask,
            input_ids,
        )[0]

    event_type = event_type_from_labels(predictions, label_space)
    # 按预测事件类型过滤 schema 中不允许的论元角色。
    spans = (
        extract_token_spans(predictions, label_space, event_type)
        if event_type is not None
        else []
    )
    trigger_spans = [
        span for span in spans if span.kind == "trigger"
    ]
    trigger = None
    if trigger_spans:
        trigger = _span_to_dict(text, offsets, trigger_spans[0])
        # 触发词没有角色字段，删除 None 可让输出更直观。
        trigger.pop("role")
    result = {
        "text": text,
        "event_type": event_type,
        "trigger": trigger,
        "args": [
            _span_to_dict(text, offsets, span)
            for span in spans
            if span.kind == "arg"
        ],
    }
    return result


def main() -> None:
    # 1. 读取模型目录和待预测文本。
    parser = argparse.ArgumentParser(description="单条文本事件抽取")
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--text", default=None)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    # 2. 恢复模型、tokenizer、标签映射和设备。
    bundle = load_model_bundle(args.model_dir, args.device)

    # 3. 执行预测并输出 JSON 结果。
    text = args.text or input("请输入新闻文本：")
    result = predict(
        bundle.model,
        bundle.tokenizer,
        bundle.label_space,
        text,
        int(bundle.metadata["max_length"]),
        bundle.device,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

