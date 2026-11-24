"""BERT 序列标注训练入口：--head linear | crf。

示例：
    python -m bert.train --head linear --epochs 3 --batch-size 32
    python -m bert.train --head crf    --epochs 3 --batch-size 16
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import torch
from torch.optim import AdamW
from transformers import BertTokenizer, get_linear_schedule_with_warmup

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from bert.model import build_model                       # noqa: E402
from common import metrics as M                          # noqa: E402
from common.config import BertConfig, DataConfig, OUTPUT_DIR, env_summary  # noqa: E402
from common.data import (build_label_maps, load_label_names, load_split,  # noqa: E402
                         make_loader, select_subset)
from common.utils import (Timer, describe_device, get_device, human_time,  # noqa: E402
                          save_json, set_seed, setup_console)


def parse_args():
    p = argparse.ArgumentParser(description="BERT 序列标注（linear / linear+CRF）")
    p.add_argument("--head", choices=["linear", "crf"], default="linear")
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--batch-size", type=int, default=None)
    p.add_argument("--lr", type=float, default=None)
    p.add_argument("--head-lr-mult", type=float, default=None)
    p.add_argument("--max-len", type=int, default=None)
    p.add_argument("--limit", type=int, default=None, help="训练集抽样条数（冒烟用）")
    p.add_argument("--eval-limit", type=int, default=None, help="验证/测试抽样条数")
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--device", default=None)
    p.add_argument("--num-workers", type=int, default=None)
    p.add_argument("--output-dir", default=None)
    return p.parse_args()


def build_optimizer(model, cfg: BertConfig):
    backbone, head = [], []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if name.startswith("bert."):
            backbone.append(param)
        else:
            head.append(param)
    return AdamW([
        {"params": backbone, "lr": cfg.lr},
        {"params": head, "lr": cfg.lr * cfg.head_lr_mult},
    ], weight_decay=cfg.weight_decay)


@torch.no_grad()
def evaluate(model, loader, device, head: str, id2label: dict[int, str]) -> dict:
    model.eval()
    gold_all, pred_all = [], []
    for batch in loader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        token_type_ids = batch["token_type_ids"].to(device)
        labels = batch["labels"]

        if head == "crf":
            out = model(input_ids, attention_mask, token_type_ids, labels=None)
            paths = model.decode(out["logits"], attention_mask, labels)
            for b in range(labels.size(0)):
                valid = labels[b] != -100
                gold_all.append([id2label[int(i)] for i in labels[b][valid]])
                pred_all.append([id2label[i] for i in paths[b]])
        else:
            logits = model(input_ids, attention_mask, token_type_ids)["logits"]
            preds = logits.argmax(dim=-1).cpu()
            for b in range(labels.size(0)):
                valid = labels[b] != -100
                gold_all.append([id2label[int(i)] for i in labels[b][valid]])
                pred_all.append([id2label[int(i)] for i in preds[b][valid]])
    return M.evaluate(gold_all, pred_all)


def train_one_epoch(model, loader, optimizer, scheduler, device, head, grad_clip):
    model.train()
    total_loss, steps = 0.0, 0
    for batch in loader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        token_type_ids = batch["token_type_ids"].to(device)
        labels = batch["labels"].to(device)

        out = model(input_ids, attention_mask, token_type_ids, labels=labels)
        loss = out["loss"]
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            [p for p in model.parameters() if p.requires_grad], grad_clip)
        optimizer.step()
        scheduler.step()
        total_loss += float(loss.detach())
        steps += 1
    return total_loss / max(steps, 1)


def main():
    setup_console()
    args = parse_args()
    cfg = BertConfig()
    if args.epochs is not None:
        cfg.epochs = args.epochs
    if args.batch_size is not None:
        cfg.batch_size = args.batch_size
    if args.lr is not None:
        cfg.lr = args.lr
    if args.head_lr_mult is not None:
        cfg.head_lr_mult = args.head_lr_mult
    if args.max_len is not None:
        cfg.max_len = args.max_len
    if args.seed is not None:
        cfg.seed = args.seed

    dcfg = DataConfig()
    num_workers = args.num_workers if args.num_workers is not None else dcfg.num_workers
    out_dir = Path(args.output_dir) if args.output_dir else OUTPUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    set_seed(cfg.seed)
    device = get_device(args.device)
    print("[env]", env_summary())
    print("[device]", describe_device(device))

    label_names = load_label_names(dcfg.label_path)
    label2id, id2label = build_label_maps(label_names)
    num_labels = len(label_names)

    train = select_subset(load_split(dcfg.train_path), args.limit, cfg.seed)
    val = select_subset(load_split(dcfg.val_path), args.eval_limit, cfg.seed)
    test = select_subset(load_split(dcfg.test_path), args.eval_limit, cfg.seed)

    tokenizer = BertTokenizer.from_pretrained(str(cfg.model_path))
    train_loader = make_loader(train, tokenizer, label2id, cfg.batch_size,
                               cfg.max_len, True, num_workers)
    val_loader = make_loader(val, tokenizer, label2id, cfg.batch_size * 2,
                             cfg.max_len, False, num_workers)
    test_loader = make_loader(test, tokenizer, label2id, cfg.batch_size * 2,
                              cfg.max_len, False, num_workers)

    model = build_model(args.head, str(cfg.model_path), num_labels, cfg.dropout).to(device)
    n_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"[model] bert+{args.head}  params={n_params:.1f}M  train={len(train)} "
          f"val={len(val)} test={len(test)} max_len={cfg.max_len}")

    optimizer = build_optimizer(model, cfg)
    total_steps = len(train_loader) * cfg.epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer, int(cfg.warmup_ratio * total_steps), total_steps)

    run_id = f"bert_{args.head}_n{len(train)}_ep{cfg.epochs}"
    run_dir = out_dir / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    best_f1, best_epoch, history = -1.0, 0, []
    t0 = time.time()
    for epoch in range(1, cfg.epochs + 1):
        t_ep = time.time()
        train_loss = train_one_epoch(model, train_loader, optimizer, scheduler,
                                     device, args.head, cfg.grad_clip)
        val_metrics = evaluate(model, val_loader, device, args.head, id2label)
        val_f1 = val_metrics["micro"]["f1"]
        history.append({
            "epoch": epoch,
            "train_loss": train_loss,
            "val": val_metrics,
            "sec": round(time.time() - t_ep, 1),
        })
        print(f"[epoch {epoch}/{cfg.epochs}] loss={train_loss:.4f} "
              f"val_F1={val_f1:.4f} P={val_metrics['micro']['precision']:.4f} "
              f"R={val_metrics['micro']['recall']:.4f} "
              f"({time.time() - t_ep:.0f}s)")
        if val_f1 > best_f1:
            best_f1, best_epoch = val_f1, epoch
            torch.save({"state_dict": model.state_dict(), "epoch": epoch,
                        "val_f1": val_f1, "head": args.head},
                       run_dir / "best.pt")
            print(f"  ✓ 保存最优模型（val_F1={val_f1:.4f}）")

    elapsed = time.time() - t0
    ckpt = torch.load(run_dir / "best.pt", map_location=device)
    model.load_state_dict(ckpt["state_dict"])
    test_metrics = evaluate(model, test_loader, device, args.head, id2label)

    print()
    print(M.format_report(test_metrics, f"{run_id} / test"))
    print(f"[done] 训练 {human_time(elapsed)}  最优 epoch={best_epoch} val_F1={best_f1:.4f}")

    result = {
        "run_id": run_id,
        "model": f"bert-base-chinese + {args.head}",
        "head": args.head,
        "train_n": len(train),
        "val_n": len(val),
        "test_n": len(test),
        "epochs": cfg.epochs,
        "batch_size": cfg.batch_size,
        "lr": cfg.lr,
        "head_lr_mult": cfg.head_lr_mult,
        "max_len": cfg.max_len,
        "params_m": round(n_params, 1),
        "best_epoch": best_epoch,
        "best_val_f1": best_f1,
        "train_seconds": round(elapsed, 1),
        "history": history,
        "test": test_metrics,
    }
    save_json(result, run_dir / "metrics.json")
    (run_dir / "log.txt").write_text(
        "\n".join([M.format_report(h["val"], f"epoch {h['epoch']} val") for h in history]
                  + ["", M.format_report(test_metrics, "test")]),
        encoding="utf-8",
    )
    print(f"[saved] {run_dir / 'metrics.json'}")


if __name__ == "__main__":
    main()
