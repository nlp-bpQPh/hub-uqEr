# -*- coding: utf-8 -*-
"""Measure CPU training/inference speed for BERT joint tagging (12L vs 4L, batch sizes)."""
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "8")
CODE = Path(__file__).resolve().parents[1] / "code"
sys.path.insert(0, str(CODE / "common"))

import torch
from torch.utils.data import DataLoader
from transformers import BertTokenizerFast

from event_data import BioDataset, build_joint_schema, collate_bio, load_extract_split, load_schema
from models import build_model, predict_char_tags


class A:
    pass


def bench(layers, batch_size, n=128, max_len=128):
    data_dir = CODE / "data" / "processed"
    schema = load_schema(data_dir / "event_schema.json")
    scheme = build_joint_schema(schema)
    tok = BertTokenizerFast.from_pretrained(str(CODE.parent / "pretrain_models" / "bert-base-chinese"))
    rows = load_extract_split("train", data_dir)[:n]
    ds = BioDataset(rows, tok, scheme, max_len)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, collate_fn=collate_bio)

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
    for i, batch in enumerate(loader):
        opt.zero_grad()
        out = model(batch["input_ids"], batch["attention_mask"], batch["labels"],
                    batch.get("token_type_ids"))
        loss = out[1]
        loss.backward()
        opt.step()
        n_tok += int(batch["attention_mask"].sum())
        if i == 2:
            pass
    dt = time.time() - t0
    print(f"layers={layers} bs={batch_size} n={n} steps={len(loader)} time={dt:.1f}s "
          f"{dt/len(loader):.2f}s/step {n_tok/dt:.0f} tok/s")
    model.eval()
    t0 = time.time()
    tags, _ = predict_char_tags(model, loader, scheme["id2label"], torch.device("cpu"))
    print(f"  predict {n} rows: {time.time()-t0:.1f}s")


if __name__ == "__main__":
    print("threads", torch.get_num_threads())
    bench(12, 32, n=128)
    bench(4, 32, n=128)
    bench(4, 64, n=128)
