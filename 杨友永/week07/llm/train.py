"""LLM（Qwen2.5-0.5B-Instruct + LoRA）序列标注训练入口。

与 BERT 方案同数据、同标签、同评估口径，便于横向对比。

示例：
    python -m llm.train --epochs 2 --batch-size 8 --grad-accum 4 --lr 2e-4
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from peft import PeftModel
from transformers import get_linear_schedule_with_warmup

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from common import metrics as M                          # noqa: E402
from common.config import DataConfig, LLMConfig, OUTPUT_DIR, env_summary  # noqa: E402
from common.data import (build_label_maps, load_label_names, load_split,  # noqa: E402
                         make_loader, select_subset)
from common.eval import evaluate_logits                  # noqa: E402
from common.utils import (describe_device, get_device, human_time,  # noqa: E402
                          save_json, set_seed, setup_console)
from llm.model import attach_lora, load_tokenizer_and_model  # noqa: E402


def parse_args():
    p = argparse.ArgumentParser(description="LLM + LoRA 序列标注")
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--batch-size", type=int, default=None)
    p.add_argument("--grad-accum", type=int, default=None)
    p.add_argument("--lr", type=float, default=None)
    p.add_argument("--max-len", type=int, default=None)
    p.add_argument("--lora-r", type=int, default=None)
    p.add_argument("--lora-alpha", type=int, default=None)
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--eval-limit", type=int, default=None)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--device", default=None)
    p.add_argument("--num-workers", type=int, default=None)
    p.add_argument("--output-dir", default=None)
    return p.parse_args()


def train_one_epoch(model, loader, optimizer, scheduler, device, grad_accum, grad_clip):
    model.train()
    total_loss, steps, micro = 0.0, 0, 0
    optimizer.zero_grad(set_to_none=True)
    for batch in loader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        logits = model(input_ids=input_ids, attention_mask=attention_mask).logits.float()
        loss = F.cross_entropy(logits.view(-1, logits.size(-1)), labels.view(-1),
                               ignore_index=-100)
        (loss / grad_accum).backward()

        micro += 1
        if micro % grad_accum == 0:
            torch.nn.utils.clip_grad_norm_(
                [p for p in model.parameters() if p.requires_grad], grad_clip)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)
        total_loss += float(loss.detach())
        steps += 1
    return total_loss / max(steps, 1)


def main():
    setup_console()
    args = parse_args()
    cfg = LLMConfig()
    for field in ("epochs", "batch_size", "grad_accum", "lr", "max_len",
                  "lora_r", "lora_alpha", "seed"):
        v = getattr(args, field, None)
        if v is not None:
            setattr(cfg, field, v)

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

    tokenizer, base = load_tokenizer_and_model(str(cfg.model_path), num_labels, device)
    model = attach_lora(base, cfg.lora_r, cfg.lora_alpha, cfg.lora_dropout).to(device)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"[model] {cfg.model_path.name} + LoRA r={cfg.lora_r} "
          f"trainable={trainable/1e6:.2f}M ({trainable/total*100:.2f}%)")

    train_loader = make_loader(train, tokenizer, label2id, cfg.batch_size,
                               cfg.max_len, True, num_workers)
    val_loader = make_loader(val, tokenizer, label2id, cfg.batch_size * 2,
                             cfg.max_len, False, num_workers)
    test_loader = make_loader(test, tokenizer, label2id, cfg.batch_size * 2,
                              cfg.max_len, False, num_workers)

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=cfg.lr, weight_decay=cfg.weight_decay)
    steps_per_epoch = max(1, len(train_loader) // cfg.grad_accum)
    total_steps = steps_per_epoch * cfg.epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer, int(cfg.warmup_ratio * total_steps), total_steps)

    run_id = f"llm_lora_n{len(train)}_ep{cfg.epochs}"
    run_dir = out_dir / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    best_f1, best_epoch, history = -1.0, 0, []
    t0 = time.time()
    for epoch in range(1, cfg.epochs + 1):
        t_ep = time.time()
        train_loss = train_one_epoch(model, train_loader, optimizer, scheduler,
                                     device, cfg.grad_accum, cfg.grad_clip)
        val_metrics = evaluate_logits(model, val_loader, device, id2label,
                                      take_token_type_ids=False)
        val_f1 = val_metrics["micro"]["f1"]
        history.append({"epoch": epoch, "train_loss": train_loss,
                        "val": val_metrics, "sec": round(time.time() - t_ep, 1)})
        print(f"[epoch {epoch}/{cfg.epochs}] loss={train_loss:.4f} "
              f"val_F1={val_f1:.4f} P={val_metrics['micro']['precision']:.4f} "
              f"R={val_metrics['micro']['recall']:.4f} ({time.time() - t_ep:.0f}s)")
        if val_f1 > best_f1:
            best_f1, best_epoch = val_f1, epoch
            model.save_pretrained(run_dir / "adapter")
            (run_dir / "adapter" / "label_map.json").write_text(
                __import__("json").dumps({"id2label": id2label}, ensure_ascii=False),
                encoding="utf-8")
            print(f"  ✓ 保存最优 adapter（val_F1={val_f1:.4f}）")

    elapsed = time.time() - t0
    # 用最优 adapter 重新加载做测试集评估
    base2 = type(base).from_pretrained(str(cfg.model_path), num_labels=num_labels,
                                       torch_dtype=base.dtype, local_files_only=True)
    best_model = PeftModel.from_pretrained(base2, run_dir / "adapter").to(device)
    test_metrics = evaluate_logits(best_model, test_loader, device, id2label,
                                   take_token_type_ids=False)

    print()
    print(M.format_report(test_metrics, f"{run_id} / test"))
    print(f"[done] 训练 {human_time(elapsed)}  最优 epoch={best_epoch} val_F1={best_f1:.4f}")

    result = {
        "run_id": run_id,
        "model": f"Qwen2.5-0.5B-Instruct + LoRA(r={cfg.lora_r}) + Linear",
        "head": "lora_linear",
        "train_n": len(train), "val_n": len(val), "test_n": len(test),
        "epochs": cfg.epochs, "batch_size": cfg.batch_size,
        "grad_accum": cfg.grad_accum, "lr": cfg.lr, "max_len": cfg.max_len,
        "lora_r": cfg.lora_r, "lora_alpha": cfg.lora_alpha,
        "trainable_m": round(trainable / 1e6, 2),
        "best_epoch": best_epoch, "best_val_f1": best_f1,
        "train_seconds": round(elapsed, 1),
        "history": history, "test": test_metrics,
    }
    save_json(result, run_dir / "metrics.json")
    (run_dir / "log.txt").write_text(
        "\n".join([M.format_report(h["val"], f"epoch {h['epoch']} val") for h in history]
                  + ["", M.format_report(test_metrics, "test")]),
        encoding="utf-8")
    print(f"[saved] {run_dir / 'metrics.json'}")


if __name__ == "__main__":
    main()
