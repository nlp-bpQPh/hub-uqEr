"""Train course-style character taggers and evaluate on official XFUND zh.val."""
from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from torch.optim import AdamW
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "common"))
from metrics import evaluate_blocks, save_report
from models import BertTagger, BiLSTMTagger, _load_auto_model_class
from sequence import FeatureDataset, make_bilstm_examples, make_bert_examples, make_char_vocab, make_collate, predictions_to_blocks
from xfund_data import TAG_TO_ID, load_xfund


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)


def infer_tag_paths(model: torch.nn.Module, loader: DataLoader, device: torch.device,
                    decoder: str) -> list[list[int]]:
    model.eval()
    by_index: dict[int, list[int]] = {}
    with torch.no_grad():
        for batch in loader:
            input_ids = batch["input_ids"].to(device)
            mask = batch["attention_mask"].to(device)
            emissions = model.emissions(input_ids, mask)
            paths = model.crf.decode(emissions, mask) if decoder == "crf" else emissions.argmax(dim=-1).cpu().tolist()
            for i, path in enumerate(paths):
                length = int(mask[i].sum().item())
                by_index[batch["example_indices"][i]] = [int(x) for x in path[:length]]
    return [by_index[i] for i in range(len(loader.dataset))]


def block_predictions(model: torch.nn.Module, examples: list[dict[str, Any]], docs: list[dict[str, Any]],
                      collate, batch_size: int, device: torch.device,
                      decoder: str) -> dict[str, list[str]]:
    loader = DataLoader(FeatureDataset(examples), batch_size=batch_size, shuffle=False,
                        num_workers=0, collate_fn=collate)
    paths = infer_tag_paths(model, loader, device, decoder)
    return predictions_to_blocks(docs, examples, paths)


def loss_for_batch(model, batch, decoder: str, device: torch.device):
    input_ids = batch["input_ids"].to(device)
    mask = batch["attention_mask"].to(device)
    emissions = model.emissions(input_ids, mask)
    if decoder == "crf":
        return model.crf.neg_log_likelihood(emissions, batch["labels_crf"].to(device), mask)
    labels = batch["labels_ce"].to(device)
    return F.cross_entropy(emissions.reshape(-1, emissions.shape[-1]), labels.reshape(-1),
                           ignore_index=-100)


def train(args) -> None:
    set_seed(args.seed)
    torch.set_num_threads(args.threads)
    device = torch.device("cpu") if args.device == "cpu" or not torch.cuda.is_available() else torch.device("cuda")
    train_docs = load_xfund(args.train)
    val_docs = load_xfund(args.val)
    indexes = list(range(len(train_docs)))
    random.Random(args.seed).shuffle(indexes)
    n_inner = max(1, round(len(indexes) * args.inner_val))
    inner_indexes = set(indexes[:n_inner])
    fit_docs = [d for i, d in enumerate(train_docs) if i not in inner_indexes]
    inner_docs = [d for i, d in enumerate(train_docs) if i in inner_indexes]
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.method == "bilstm_crf":
        vocab = make_char_vocab(fit_docs)
        train_examples = make_bilstm_examples(fit_docs, vocab, args.max_chars)
        inner_examples = make_bilstm_examples(inner_docs, vocab, args.max_chars)
        val_examples = make_bilstm_examples(val_docs, vocab, args.max_chars)
        model = BiLSTMTagger(len(vocab), len(TAG_TO_ID), decoder="crf")
        pad_id = vocab["<PAD>"]
        model_name = "character BiLSTM initialized from scratch"
        (out_dir / "vocab.json").write_text(json.dumps(vocab, ensure_ascii=False), encoding="utf-8")
        optimizer = AdamW(model.parameters(), lr=args.lstm_lr, weight_decay=0.01)
    else:
        from transformers import AutoTokenizer
        _load_auto_model_class()  # register the optional torchvision workaround first
        tokenizer = AutoTokenizer.from_pretrained(args.model_name)
        train_examples = make_bert_examples(fit_docs, tokenizer, args.max_chars, args.max_len)
        inner_examples = make_bert_examples(inner_docs, tokenizer, args.max_chars, args.max_len)
        val_examples = make_bert_examples(val_docs, tokenizer, args.max_chars, args.max_len)
        decoder = "crf" if args.method == "bert_crf" else "linear"
        model = BertTagger(args.model_name, len(TAG_TO_ID), decoder=decoder,
                           num_layers=args.bert_layers)
        pad_id = tokenizer.pad_token_id
        model_name = args.model_name
        optimizer = AdamW(model.parameters(), lr=args.bert_lr, weight_decay=0.01)

    decoder = "crf" if args.method in ("bilstm_crf", "bert_crf") else "linear"
    collate = make_collate(pad_id)
    train_loader = DataLoader(FeatureDataset(train_examples), batch_size=args.batch_size,
                              shuffle=True, num_workers=0, collate_fn=collate)
    model.to(device)
    best_score = -1.0
    history = []
    best_path = out_dir / "best.pt"
    start_all = time.time()

    for epoch in range(1, args.epochs + 1):
        model.train()
        sum_loss = 0.0
        t0 = time.time()
        for batch in train_loader:
            optimizer.zero_grad(set_to_none=True)
            loss = loss_for_batch(model, batch, decoder, device)
            if not torch.isfinite(loss):
                raise RuntimeError(f"Non-finite training loss: {loss.item()}")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            sum_loss += float(loss.detach().cpu())

        inner_preds = block_predictions(model, inner_examples, inner_docs, collate,
                                        args.batch_size, device, decoder)
        inner_metrics = evaluate_blocks(inner_docs, inner_preds)
        row = {
            "epoch": epoch,
            "train_loss": sum_loss / max(1, len(train_loader)),
            "inner_macro_f1": inner_metrics["macro_f1_all_four_classes"],
            "inner_entity_micro_f1": inner_metrics["entity_micro"]["f1"],
            "epoch_seconds": time.time() - t0,
        }
        history.append(row)
        print(json.dumps(row, ensure_ascii=False))
        if row["inner_macro_f1"] > best_score:
            best_score = row["inner_macro_f1"]
            torch.save({
                "method": args.method,
                "state_dict": model.state_dict(),
                "vocab_size": len(vocab) if args.method == "bilstm_crf" else None,
                "model_name": model_name,
                "bert_layers": args.bert_layers if args.method != "bilstm_crf" else None,
                "seed": args.seed,
            }, best_path)

    (out_dir / "training_log.json").write_text(json.dumps({
        "method": args.method,
        "seed": args.seed,
        "device": str(device),
        "fit_documents": len(fit_docs),
        "inner_validation_documents": len(inner_docs),
        "official_validation_documents": len(val_docs),
        "epochs": history,
        "total_seconds": time.time() - start_all,
        "model_name": model_name,
        "bert_layers": args.bert_layers if args.method != "bilstm_crf" else None,
        "max_chars_per_chunk": args.max_chars,
        "max_length": args.max_len if args.method != "bilstm_crf" else None,
    }, ensure_ascii=False, indent=2), encoding="utf-8")

    checkpoint = torch.load(best_path, map_location=device, weights_only=True)
    model.load_state_dict(checkpoint["state_dict"])
    val_preds = block_predictions(model, val_examples, val_docs, collate,
                                  args.batch_size, device, decoder)
    report = save_report(out_dir, val_docs, val_preds, {
        "name": args.method,
        "model": model_name,
        "bert_layers": args.bert_layers if args.method != "bilstm_crf" else None,
        "decoder": decoder,
        "training_documents": len(fit_docs),
        "inner_validation_documents": len(inner_docs),
        "official_split": "zh.val",
        "seed": args.seed,
        "epochs_requested": args.epochs,
        "best_inner_macro_f1": best_score,
        "note": "Block-level labels are scored. XFUND key-answer relation links are retained but not scored.",
    })
    (out_dir / "inner_validation_metrics.json").write_text(
        json.dumps({"macro_f1": best_score}, ensure_ascii=False, indent=2), encoding="utf-8")
    print("OFFICIAL_VALIDATION")
    print(json.dumps(report["metrics"], ensure_ascii=False, indent=2))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--method", choices=["bilstm_crf", "bert_linear", "bert_crf"], required=True)
    p.add_argument("--train", default=str(ROOT / "data/source/zh.train.json"))
    p.add_argument("--val", default=str(ROOT / "data/source/zh.val.json"))
    p.add_argument("--out", required=True)
    local_model = ROOT / "models/bert-base-chinese"
    default_model = str(local_model) if (local_model / "config.json").exists() else "google-bert/bert-base-chinese"
    p.add_argument("--model-name", default=default_model)
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--max-chars", type=int, default=240)
    p.add_argument("--max-len", type=int, default=256)
    p.add_argument("--inner-val", type=float, default=0.20)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--threads", type=int, default=8)
    p.add_argument("--device", default="cpu")
    p.add_argument("--bert-lr", type=float, default=2e-5)
    p.add_argument("--bert-layers", type=int, default=4)
    p.add_argument("--lstm-lr", type=float, default=1e-3)
    train(p.parse_args())


if __name__ == "__main__":
    main()
