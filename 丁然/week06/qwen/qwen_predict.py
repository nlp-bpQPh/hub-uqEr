"""加载最优 Qwen LoRA 模型并预测单条新闻。"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


WEEK_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(WEEK_DIR))

import torch  # noqa: E402

from labels import LABELS  # noqa: E402
from qwen_model import load_lora_model  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Qwen LoRA 新闻分类预测")
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("text", nargs="?")
    parser.add_argument("--top-k", type=int, default=3)
    args = parser.parse_args()

    model, tokenizer, meta, device = load_lora_model(args.model_dir)
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
    values, indices = logits.softmax(dim=-1)[0].cpu().topk(
        min(args.top_k, len(LABELS))
    )
    for probability, index in zip(values.tolist(), indices.tolist()):
        print(f"{LABELS[index]}: {probability:.4%}")


if __name__ == "__main__":
    main()
