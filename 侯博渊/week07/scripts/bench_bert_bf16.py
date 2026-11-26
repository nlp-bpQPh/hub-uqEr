# -*- coding: utf-8 -*-
"""Benchmark BERT 4L CPU with bf16 autocast and thread counts."""
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
CODE = Path(__file__).resolve().parents[1] / "code"
sys.path.insert(0, str(CODE / "common"))

import torch
from torch.utils.data import DataLoader
from transformers import BertTokenizerFast

from event_data import BioDataset, build_joint_schema, collate_bio, load_extract_split, load_schema
from models import build_model


class A:
    pass


def run(layers, bs, threads, amp, n=128):
    torch.set_num_threads(threads)
    data_dir = CODE / "data" / "processed"
    schema = load_schema(data_dir / "event_schema.json")
    scheme = build_joint_schema(schema)
    tok = BertTokenizerFast.from_pretrained(str(CODE.parent / "pretrain_models" / "bert-base-chinese"))
    rows = load_extract_split("train", data_dir)[:n]
    ds = BioDataset(rows, tok, scheme, 128)
    loader = DataLoader(ds, batch_size=bs, shuffle=False, collate_fn=collate_bio)

    a = A()
    a.bert_path = str(CODE.parent / "pretrain_models" / "bert-base-chinese")
    a.bert_layers = layers
    a.decoder = "linear"
    a.dropout = 0.1
    a.hard_constraint = False
    model = build_model(a, scheme)
    model.train()
    opt = torch.optim.AdamW(model.parameters(), lr=2e-5)

    t0 = time.time()
    n_tok = 0
    for batch in loader:
        opt.zero_grad()
        if amp:
            with torch.autocast("cpu", dtype=torch.bfloat16):
                out = model(batch["input_ids"], batch["attention_mask"], batch["labels"],
                            batch.get("token_type_ids"))
        else:
            out = model(batch["input_ids"], batch["attention_mask"], batch["labels"],
                        batch.get("token_type_ids"))
        out[1].backward()
        opt.step()
        n_tok += int(batch["attention_mask"].sum())
    dt = time.time() - t0
    print(f"layers={layers} bs={bs} threads={threads} amp={amp}: {dt:.1f}s "
          f"{n_tok/dt:.0f} tok/s  ({n_tok/dt*3600/1e6:.2f}M tok/h)")


if __name__ == "__main__":
    print("threads default", torch.get_num_threads(), "bf16 supported?",
          torch.backends.mkldnn.is_available())
    run(4, 32, 4, False)
    run(4, 32, 8, False)
    run(4, 32, 8, True)
    run(4, 64, 8, True)
