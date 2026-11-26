# -*- coding: utf-8 -*-
"""在含正负样本的 detect 集上评估模型能否判断一句话是否含事件触发词。"""
import argparse
import json
import os
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "common"))

from eval_utils import CKPT_DIR
from event_data import BioDataset, DATA_DIR, collate_bio, load_jsonl
from model_paths import resolve_checkpoint_model_path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--data_file", default="dev_detect_clean.jsonl")
    ap.add_argument("--data_dir", default="")
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--device", default="")
    ap.add_argument("--output", default="")
    args = ap.parse_args()

    from transformers import AutoTokenizer
    from models import build_model, predict_char_tags

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    ckpt_path = Path(args.ckpt)
    if not ckpt_path.exists():
        ckpt_path = CKPT_DIR / args.ckpt
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    train_args = ckpt["args"]

    class Args: pass
    model_args = Args()
    for key, value in train_args.items():
        setattr(model_args, key, value)
    model_args.hard_constraint = bool(train_args.get("hard_constraint", False))
    model_args.bert_path = resolve_checkpoint_model_path(
        train_args.get("bert_path"), train_args.get("encoder_name", ""))
    model = build_model(model_args, ckpt["label_scheme"]).to(device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()

    tokenizer = AutoTokenizer.from_pretrained(
        model_args.bert_path, use_fast=True,
        local_files_only=(Path(model_args.bert_path).expanduser().exists()
                          or os.environ.get("HF_HUB_OFFLINE") == "1"
                          or os.environ.get("TRANSFORMERS_OFFLINE") == "1"),
    )
    if not getattr(tokenizer, "is_fast", False):
        raise RuntimeError("需要 fast tokenizer 才能使用 word_ids() 对齐字符标签。")
    data_dir = Path(args.data_dir) if args.data_dir else DATA_DIR
    rows = load_jsonl(data_dir / args.data_file)
    dataset = BioDataset(rows, tokenizer, ckpt["label_scheme"],
                         max_length=train_args.get("max_len", 256), with_labels=True)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False,
                        num_workers=0, collate_fn=collate_bio)
    tags, _ = predict_char_tags(model, loader, ckpt["id2label"], device)

    tp = fp = tn = fn = 0
    for row, seq in zip(rows, tags):
        pred = any(t.startswith(("B-TRG:", "I-TRG:", "B-TRG", "I-TRG")) for t in seq)
        gold = bool(row.get("has_event", False))
        if pred and gold: tp += 1
        elif pred and not gold: fp += 1
        elif not pred and not gold: tn += 1
        else: fn += 1
    n = max(len(rows), 1)
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    specificity = tn / (tn + fp) if tn + fp else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    result = {
        "model": train_args.get("encoder_name", "Transformer"),
        "checkpoint": str(ckpt_path), "data_file": args.data_file, "n": len(rows),
        "tp": tp, "fp": fp, "tn": tn, "fn": fn,
        "accuracy": (tp + tn) / n, "precision": precision, "recall": recall,
        "specificity": specificity, "f1": f1,
        "n_truncated": dataset.n_truncated,
        "definition": "预测序列中至少出现一个 TRG 标签即判为有事件；gold 来自 has_event。"
    }
    output = Path(args.output) if args.output else (
        ROOT / "outputs" / "metrics" / "disc" / f"{ckpt_path.stem}_presence.json")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    print(f"结果已写入：{output}")

if __name__ == "__main__":
    main()
