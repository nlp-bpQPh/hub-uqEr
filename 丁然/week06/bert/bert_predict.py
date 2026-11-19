"""使用训练完成的 BERT 模型预测单条新闻。"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

WEEK_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(WEEK_DIR))

import torch
from transformers import AutoConfig, AutoModel

from bert_config import get_device
from bert_model import BertClassifier, load_tokenizer
from labels import LABELS


def main() -> None:
    parser = argparse.ArgumentParser(description="BERT 新闻分类预测")
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("text", nargs="?", help="待分类新闻文本")
    parser.add_argument("--top-k", type=int, default=3)
    args = parser.parse_args()

    model_dir = Path(args.model_dir)
    meta = json.loads((model_dir / "meta.json").read_text(encoding="utf-8"))
    tokenizer = load_tokenizer(str(model_dir))
    device = get_device()

    encoder = AutoModel.from_config(
        AutoConfig.from_pretrained(model_dir),
        dtype=torch.float32,
    )
    model = BertClassifier(
        encoder=encoder,
        num_labels=meta["num_labels"],
        pooling=meta["pooling"],
        dropout=0.0,
    )
    model.load_state_dict(
        torch.load(model_dir / "model.pt", map_location="cpu", weights_only=True)
    )
    model.to(device).eval()

    text = args.text or input("请输入新闻文本：")
    encoded = tokenizer(
        text,
        padding="max_length",
        truncation=True,
        max_length=meta["max_length"],
        return_tensors="pt",
    )
    with torch.inference_mode():
        logits = model(
            encoded["input_ids"].to(device),
            encoded["attention_mask"].to(device),
        ).logits
    probabilities = logits.softmax(dim=-1)[0].cpu()
    values, indices = probabilities.topk(min(args.top_k, len(LABELS)))
    for probability, index in zip(values.tolist(), indices.tolist()):
        print(f"{LABELS[index]}: {probability:.4%}")


if __name__ == "__main__":
    main()
