# -*- coding: utf-8 -*-
"""Benchmark structbert-tiny (4L/256h) training speed vs BERT-base 4L on this CPU."""
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


def run(path, n=96, bs=32):
    torch.set_num_threads(8)
    data_dir = CODE / "data" / "processed"
    schema = load_schema(data_dir / "event_schema.json")
    scheme = build_joint_schema(schema)
    tok = BertTokenizerFast.from_pretrained(str(path))
    rows = load_extract_split("train", data_dir)[:n]
    ds = BioDataset(rows, tok, scheme, 160)
    loader = DataLoader(ds, batch_size=bs, shuffle=False, collate_fn=collate_bio)
    a = A()
    a.bert_path = str(path)
    a.bert_layers = 0
    a.decoder = "linear"
    a.dropout = 0.1
    a.hard_constraint = False
    model = build_model(a, scheme)
    model.train()
    opt = torch.optim.AdamW(model.parameters(), lr=3e-5)
    t0 = time.time()
    n_tok = 0
    for batch in loader:
        opt.zero_grad()
        out = model(batch["input_ids"], batch["attention_mask"], batch["labels"],
                    batch.get("token_type_ids"))
        out[1].backward()
        opt.step()
        n_tok += int(batch["attention_mask"].sum())
    dt = time.time() - t0
    print(f"{Path(path).name}: {n} rows, {dt:.1f}s, {n_tok/dt:.0f} tok/s, "
          f"{dt/len(loader):.2f}s/step")
    print("   params(M):", round(sum(p.numel() for p in model.parameters()) / 1e6, 2))


if __name__ == "__main__":
    run(CODE.parent / "pretrain_models" / "bert-base-chinese")
    run(Path(r"E:\python\第六周 分类任务\week6-数据接入与分流\_models\structbert-tiny"))
