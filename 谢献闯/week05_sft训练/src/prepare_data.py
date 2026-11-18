# -*- coding: utf-8 -*-
"""
第 1 步：把 (instruction, output) 转成带 Loss Mask 的训练样本。

关键点（课件第五部分讲的）：
  1. 套 ChatML 模板：<|im_start|>user ... <|im_end|> <|im_start|>assistant ... <|im_end|>
  2. Loss Mask：prompt 段（用户问题）的 label 置 -100，只有回答段参与 loss
  3. 结果存成 .pt，训练时直接读，省掉每次重复分词

用法：python src/prepare_data.py
"""
import json
import sys
from pathlib import Path

import torch
from transformers import AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass


def encode(tokenizer, instruction: str, output: str):
    """返回 (input_ids, labels)，prompt 段 labels 全为 -100。"""
    prompt = config.build_prompt(instruction)
    target = config.build_target(output)

    prompt_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
    target_ids = tokenizer(target, add_special_tokens=False)["input_ids"]

    input_ids = (prompt_ids + target_ids)[: config.MAX_LEN]
    labels = ([-100] * len(prompt_ids) + target_ids)[: config.MAX_LEN]
    return input_ids, labels


def process(split: str, tokenizer):
    path = config.DATA_DIR / f"sft_{split}.jsonl"
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]

    data, dropped = [], 0
    for row in rows:
        ids, labels = encode(tokenizer, row["instruction"], row["output"])
        if all(l == -100 for l in labels):     # 截断后回答全没了，丢掉
            dropped += 1
            continue
        data.append({"input_ids": ids, "labels": labels})

    out = config.PROCESSED_DIR / f"{split}.pt"
    torch.save(data, out)

    total = sum(len(d["input_ids"]) for d in data)
    prompt_len = sum(sum(1 for l in d["labels"] if l == -100) for d in data)
    target_len = total - prompt_len
    stats = {
        "split": split, "samples": len(data), "dropped": dropped,
        "avg_total_tokens": round(total / len(data), 1),
        "avg_prompt_tokens": round(prompt_len / len(data), 1),
        "avg_target_tokens": round(target_len / len(data), 1),
        "loss_ratio": round(target_len / total, 3),
        "max_tokens": max(len(d["input_ids"]) for d in data),
    }
    print(f"[{split}] {len(data)} 条 -> {out.name} | "
          f"平均 {stats['avg_total_tokens']} tok（提问 {stats['avg_prompt_tokens']} + 回答 {stats['avg_target_tokens']}）"
          f"| loss 覆盖率 {stats['loss_ratio']:.0%} | 截断丢弃 {dropped}")
    return stats


def show_one(tokenizer):
    """打印一条样本，直观看到哪些 token 被 mask。"""
    row = json.loads((config.DATA_DIR / "sft_train.jsonl").read_text(encoding="utf-8").splitlines()[0])
    ids, labels = encode(tokenizer, row["instruction"], row["output"])
    n_prompt = sum(1 for l in labels if l == -100)
    print("\n示例（前 80 个 token）：")
    print("  原文 =", tokenizer.decode(ids[:80]).replace("\n", "\\n"))
    print(f"  前 {n_prompt} 个 token 的 label=-100（不参与 loss），其余 {len(ids) - n_prompt} 个参与")
    print("  参与 loss 的部分 =", tokenizer.decode(ids[n_prompt:]))
    print("  第 1 个回答 token 的 label =", labels[n_prompt] if n_prompt < len(labels) else None)


def main():
    config.ensure_dirs()
    tokenizer = AutoTokenizer.from_pretrained(config.MODEL_DIR)
    print("tokenizer:", config.MODEL_DIR.name, "| vocab", len(tokenizer))

    stats = [process("train", tokenizer), process("val", tokenizer)]
    show_one(tokenizer)

    (config.OUTPUT_DIR / "data_stats.json").write_text(
        json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print("\n完成。下一步：python src/train_sft.py")


if __name__ == "__main__":
    main()
