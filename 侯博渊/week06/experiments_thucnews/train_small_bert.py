"""
THUCNews 10 类新闻分类：StructBERT-tiny（4 层 / 256 维 / 8.77M）微调。

数据：train 4,000 / dev 1,000 / test 1,500，10 类平衡；文本是长新闻（中位数 630 字），
max_length=192（会截断），动态 padding。
"""
import argparse
import json
import os
import random
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CODE = ROOT / "code"
sys.path.insert(0, str(CODE))

_TMP = ROOT / "_tmp"
_TMP.mkdir(exist_ok=True)
os.environ["TMP"] = str(_TMP)
os.environ["TEMP"] = str(_TMP)
os.environ["TMPDIR"] = str(_TMP)
tempfile.tempdir = str(_TMP)

import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn as nn  # noqa: E402
from torch.utils.data import DataLoader, Dataset  # noqa: E402

from common.metrics import (  # noqa: E402
    classification_report,
    expected_calibration_error,
    pred_distribution,
)

DATA = ROOT / "_data" / "thucnews"
MODEL_DIR = ROOT / "_models" / "structbert-tiny"
OUT = ROOT / "experiments_thucnews" / "outputs"
NUM_LABELS = 10


class TinyClassifier(nn.Module):
    def __init__(self, bert, hidden=256, num_labels=NUM_LABELS, dropout=0.1):
        super().__init__()
        self.bert = bert
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden, num_labels)

    def forward(self, input_ids, attention_mask):
        out = self.bert(input_ids=input_ids, attention_mask=attention_mask, return_dict=True)
        return self.classifier(self.dropout(out.last_hidden_state[:, 0]))


def load_tiny_model():
    from transformers import BertConfig, BertModel

    cd = json.loads((MODEL_DIR / "config.json").read_text(encoding="utf-8"))
    cfg = BertConfig(
        vocab_size=cd["vocab_size"], hidden_size=cd["hidden_size"],
        num_hidden_layers=cd["num_hidden_layers"],
        num_attention_heads=cd["num_attention_heads"],
        intermediate_size=cd["intermediate_size"],
        hidden_act=cd.get("hidden_act", "gelu"),
        hidden_dropout_prob=cd.get("hidden_dropout_prob", 0.1),
        attention_probs_dropout_prob=cd.get("attention_probs_dropout_prob", 0.1),
        max_position_embeddings=cd["max_position_embeddings"],
        type_vocab_size=cd["type_vocab_size"],
    )
    model = BertModel(cfg)
    sd = torch.load(MODEL_DIR / "pytorch_model.bin", map_location="cpu", weights_only=False)
    new_sd = {k[len("bert."):]: v for k, v in sd.items() if k.startswith("bert.")}
    missing, _ = model.load_state_dict(new_sd, strict=False)
    if missing:
        raise RuntimeError(f"权重缺失: {missing[:5]}")
    print(f"[model] StructBERT-tiny {sum(p.numel() for p in model.parameters())/1e6:.2f}M", flush=True)
    return model, cfg


def load_json(split):
    return json.loads((DATA / f"{split}.json").read_text(encoding="utf-8"))


def encode(tok, texts, max_length, batch=1000):
    ids, masks = [], []
    for i in range(0, len(texts), batch):
        enc = tok(texts[i:i + batch], truncation=True, max_length=max_length, padding=False)
        ids.extend(enc["input_ids"])
        masks.extend(enc["attention_mask"])
    return ids, masks


class NewsDataset(Dataset):
    def __init__(self, ids, masks, labels):
        self.ids, self.masks, self.labels = ids, masks, labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, i):
        return (torch.tensor(self.ids[i]), torch.tensor(self.masks[i]),
                torch.tensor(self.labels[i]))


def collate(batch, pad_id=0):
    L = max(len(x[0]) for x in batch)
    ids = torch.full((len(batch), L), pad_id, dtype=torch.long)
    masks = torch.zeros((len(batch), L), dtype=torch.long)
    labels = torch.tensor([x[2] for x in batch], dtype=torch.long)
    for i, (x, m, _) in enumerate(batch):
        n = len(x)
        ids[i, :n] = x
        masks[i, :n] = m
    return ids, masks, labels


@torch.no_grad()
def predict(model, loader, device):
    model.eval()
    ys, preds, probs = [], [], []
    for ids, masks, y in loader:
        logits = model(ids.to(device), masks.to(device))
        p = torch.softmax(logits.float(), dim=-1).cpu()
        probs.append(p)
        preds.extend(p.argmax(-1).tolist())
        ys.extend(y.tolist())
    return np.array(ys), np.array(preds), torch.cat(probs).numpy()


def evaluate(model, loader, device):
    y_true, y_pred, probs = predict(model, loader, device)
    report = classification_report(y_true, y_pred, num_classes=NUM_LABELS)
    report["ece"] = expected_calibration_error(y_true, probs)
    report["pred_distribution"] = pred_distribution(y_pred, num_classes=NUM_LABELS)
    return report, y_true, y_pred, probs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument("--max_length", type=int, default=192)
    ap.add_argument("--max_steps", type=int, default=0)
    ap.add_argument("--no_save", action="store_true")
    args = ap.parse_args()

    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    torch.set_num_threads(8)
    device = "cpu"

    from transformers import BertTokenizerFast
    tok = BertTokenizerFast(vocab_file=str(MODEL_DIR / "vocab.txt"), do_lower_case=False)

    train, dev, test = load_json("train"), load_json("dev"), load_json("test")
    t0 = time.time()
    tr_ids, tr_masks = encode(tok, [d["text"] for d in train], args.max_length)
    dv_ids, dv_masks = encode(tok, [d["text"] for d in dev], args.max_length)
    te_ids, te_masks = encode(tok, [d["text"] for d in test], args.max_length)
    print(f"[data] train={len(train)} dev={len(dev)} test={len(test)} | 编码 {time.time()-t0:.1f}s",
          flush=True)

    tr_ds = NewsDataset(tr_ids, tr_masks, [d["label"] for d in train])
    dv_ds = NewsDataset(dv_ids, dv_masks, [d["label"] for d in dev])
    te_ds = NewsDataset(te_ids, te_masks, [d["label"] for d in test])
    tr_dl = DataLoader(tr_ds, batch_size=args.batch_size, shuffle=True,
                       collate_fn=lambda b: collate(b, tok.pad_token_id))
    dv_dl = DataLoader(dv_ds, batch_size=args.batch_size, shuffle=False,
                       collate_fn=lambda b: collate(b, tok.pad_token_id))
    te_dl = DataLoader(te_ds, batch_size=args.batch_size, shuffle=False,
                       collate_fn=lambda b: collate(b, tok.pad_token_id))

    bert, cfg = load_tiny_model()
    model = TinyClassifier(bert, hidden=cfg.hidden_size).to(device)

    from sklearn.utils.class_weight import compute_class_weight
    cw = compute_class_weight("balanced", classes=np.arange(NUM_LABELS),
                              y=[d["label"] for d in train])
    weight = torch.tensor(cw, dtype=torch.float, device=device)
    crit = nn.CrossEntropyLoss(weight=weight)

    head_params = list(model.classifier.parameters()) + list(model.dropout.parameters())
    bert_params = [p for n, p in model.named_parameters()
                   if not n.startswith("classifier") and not n.startswith("dropout")]
    opt = torch.optim.AdamW([{"params": bert_params, "lr": args.lr},
                             {"params": head_params, "lr": args.lr * 2}],
                            weight_decay=0.01)

    steps_per_epoch = min(len(tr_dl), args.max_steps) if args.max_steps else len(tr_dl)
    total_steps = steps_per_epoch * args.epochs
    warmup = int(total_steps * 0.1)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda step: step / max(1, warmup) if step < warmup
        else max(0.0, (total_steps - step) / max(1, total_steps - warmup)))

    print(f"[cfg] epochs={args.epochs} batch={args.batch_size} lr={args.lr} "
          f"max_length={args.max_length} steps/epoch={steps_per_epoch}", flush=True)

    best_f1, best_state, best_epoch = 0.0, None, 0
    log, t_train = [], time.time()
    for epoch in range(1, args.epochs + 1):
        model.train()
        t_epoch = time.time()
        running, n_step = 0.0, 0
        for i, (ids, masks, y) in enumerate(tr_dl):
            opt.zero_grad(set_to_none=True)
            loss = crit(model(ids.to(device), masks.to(device)), y.to(device))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            running += loss.item()
            n_step += 1
            if i % 20 == 0:
                print(f"  epoch {epoch} step {i}/{steps_per_epoch} loss={loss.item():.4f}",
                      flush=True)
            if args.max_steps and n_step >= args.max_steps:
                break

        dev_report, _, _, _ = evaluate(model, dv_dl, device)
        log.append({"epoch": epoch, "train_loss": round(running / max(1, n_step), 4),
                    "dev_acc": dev_report["acc"], "dev_macro_f1": dev_report["macro_f1"],
                    "elapsed_s": round(time.time() - t_epoch, 1)})
        print(f"epoch {epoch}: dev_acc={dev_report['acc']} dev_f1={dev_report['macro_f1']} "
              f"({log[-1]['elapsed_s']}s)", flush=True)
        if dev_report["macro_f1"] > best_f1:
            best_f1, best_epoch = dev_report["macro_f1"], epoch
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}

    assert best_state is not None
    model.load_state_dict(best_state)
    test_report, y_true, y_pred, probs = evaluate(model, te_dl, device)
    train_seconds = round(time.time() - t_train, 1)
    test_report.update({
        "method": "structbert_tiny_4l256_thucnews",
        "params_m": round(sum(p.numel() for p in model.parameters()) / 1e6, 2),
        "train_seconds": train_seconds,
        "best_epoch": best_epoch,
        "epochs": args.epochs,
        "max_length": args.max_length,
        "dynamic_padding": True,
    })
    print("\n" + json.dumps({k: test_report[k] for k in
                             ("acc", "macro_f1", "best_epoch", "train_seconds")},
                            ensure_ascii=False))
    print(f"best epoch {best_epoch} | dev_f1 {best_f1} | train {train_seconds}s")

    if args.no_save:
        return
    OUT.mkdir(parents=True, exist_ok=True)
    meta = {"n_train": len(train), "n_dev": len(dev), "n_test": len(test),
            "dataset": "THUCNews", "model": "iic/nlp_structbert_backbone_tiny_std",
            "params_m": test_report["params_m"], "train_seconds": train_seconds,
            "best_epoch": best_epoch, "seed": 42}
    preds = [{"text": d["text"], "true": int(t), "pred": int(p),
              "conf": round(float(c), 4)}
             for d, t, p, c in zip(test, y_true, y_pred, probs.max(axis=1))]
    (OUT / "small_bert_pred.json").write_text(
        json.dumps({"meta": meta, "report": test_report, "preds": preds},
                   ensure_ascii=False, indent=2), encoding="utf-8")
    (OUT / "small_bert_log.json").write_text(
        json.dumps(log, ensure_ascii=False, indent=2), encoding="utf-8")
    torch.save({"state_dict": best_state, "report": test_report, "meta": meta},
               OUT / "small_bert_ckpt.pt")
    print("→", OUT / "small_bert_pred.json")


if __name__ == "__main__":
    main()
