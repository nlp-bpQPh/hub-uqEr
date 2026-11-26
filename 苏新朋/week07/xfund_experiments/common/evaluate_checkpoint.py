from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "common"))
from metrics import save_report
from models import BertTagger, hard_bio_constraints, _load_auto_model_class
from sequence import FeatureDataset, make_bert_examples, make_collate, predictions_to_blocks
from xfund_data import TAG_TO_ID, load_xfund


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--val", default=str(ROOT / "data/source/zh.val.json"))
    p.add_argument("--out", required=True)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--max-chars", type=int, default=240)
    p.add_argument("--max-len", type=int, default=256)
    p.add_argument("--hard-bio", action="store_true")
    p.add_argument("--threads", type=int, default=8)
    args = p.parse_args()
    torch.set_num_threads(args.threads)
    _load_auto_model_class()
    from transformers import AutoTokenizer
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    model_name = checkpoint["model_name"]
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    docs = load_xfund(args.val)
    examples = make_bert_examples(docs, tokenizer, args.max_chars, args.max_len)
    model = BertTagger(model_name, len(TAG_TO_ID), decoder="crf",
                       num_layers=checkpoint.get("bert_layers", 4))
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    if args.hard_bio:
        transitions, starts = hard_bio_constraints(len(TAG_TO_ID))
        model.crf.constraint.copy_(transitions)
        model.crf.start_constraint.copy_(starts)
    loader = DataLoader(FeatureDataset(examples), batch_size=args.batch_size,
                        shuffle=False, num_workers=0,
                        collate_fn=make_collate(tokenizer.pad_token_id))
    by_index = {}
    device = torch.device("cpu")
    with torch.no_grad():
        for batch in loader:
            ids = batch["input_ids"].to(device)
            mask = batch["attention_mask"].to(device)
            emissions = model.emissions(ids, mask)
            paths = model.crf.decode(emissions, mask)
            for i, path in enumerate(paths):
                by_index[batch["example_indices"][i]] = [int(x) for x in path]
    paths = [by_index[i] for i in range(len(examples))]
    predictions = predictions_to_blocks(docs, examples, paths)
    report = save_report(args.out, docs, predictions, {
        "name": "bert_crf_hard_bio" if args.hard_bio else "bert_crf_checkpoint_eval",
        "checkpoint": str(args.checkpoint),
        "decoder_constraint": "hard BIO" if args.hard_bio else "soft CRF",
        "bert_layers": checkpoint.get("bert_layers", 4),
        "official_split": "zh.val",
        "training_reused": True,
        "note": "Same checkpoint; only decode-time BIO constraints change. Block labels scored; relation links not scored.",
    })
    print(json.dumps(report["metrics"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
