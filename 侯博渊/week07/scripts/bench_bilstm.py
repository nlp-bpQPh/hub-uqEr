# -*- coding: utf-8 -*-
"""Time one BiLSTM-CRF training step to find the bottleneck."""
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
CODE = Path(__file__).resolve().parents[1] / "code"
sys.path.insert(0, str(CODE / "common"))

import torch
from torch.utils.data import DataLoader

from event_data import (CharDataset, build_joint_schema, collate_char, load_extract_split,
                        load_schema, make_vocab)
from crf import CRFClassifier

torch.set_num_threads(8)


class Enc(torch.nn.Module):
    def __init__(self, vocab_size, emb=128, hidden=256):
        super().__init__()
        self.emb = torch.nn.Embedding(vocab_size, emb, padding_idx=0)
        self.lstm = torch.nn.LSTM(emb, hidden, num_layers=1, bidirectional=True, batch_first=True)
        self.hidden_size = hidden * 2

    def forward(self, ids, mask, token_type_ids=None):
        x = self.emb(ids)
        out, _ = self.lstm(x)
        return out


def main():
    data_dir = CODE / "data" / "processed"
    schema = load_schema(data_dir / "event_schema.json")
    scheme = build_joint_schema(schema)
    rows = load_extract_split("train", data_dir)[:64]
    vocab = make_vocab(rows)
    ds = CharDataset(rows, vocab, scheme, max_length=160)
    loader = DataLoader(ds, batch_size=32, collate_fn=collate_char)
    model = CRFClassifier(Enc(len(vocab)), 512, len(scheme["labels"]), dropout=0.3, use_crf=True)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)

    for i, batch in enumerate(loader):
        t0 = time.time()
        out = model(batch["input_ids"], batch["attention_mask"], batch["labels"])
        t1 = time.time()
        loss = out[1]
        loss.backward()
        t2 = time.time()
        opt.step()
        opt.zero_grad()
        t3 = time.time()
        print(f"step {i}: fwd={t1-t0:.2f}s bwd={t2-t1:.2f}s opt={t3-t2:.2f}s "
              f"len={int(batch['attention_mask'].sum())}")
        if i >= 3:
            break
    # decode timing
    model.eval()
    batch = next(iter(loader))
    t0 = time.time()
    with torch.no_grad():
        paths = model.predict(batch["input_ids"], batch["attention_mask"])
    print("decode 32 seqs:", round(time.time() - t0, 2), "s")


if __name__ == "__main__":
    main()
