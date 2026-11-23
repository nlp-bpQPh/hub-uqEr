"""LLM 训练：SFT（全参）与 LoRA。"""
from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch

from bert.train import record
from dataset import ProductDataset, build_label_map
from device import Device
from llm.data import SFTDataset, build_loader, label_token_ids
from llm.eval import predict
from llm.model import load_llm
from metrics import summarize

OUT = Path(__file__).resolve().parent.parent / "outputs"


@dataclass(frozen=True)
class LLMConfig:
    mode: str = "lora"            # lora / full
    lr: float = 2e-4              # LoRA 用 2e-4，全参用 2e-5
    epochs: int = 2
    batch_size: int = 8
    max_len: int = 128
    weight_decay: float = 0.0
    warmup_ratio: float = 0.03
    max_grad_norm: float = 1.0
    lora_r: int = 16
    lora_alpha: int = 32
    seed: int = 42
    eval_batch_size: int = 32
    limit: int | None = None


def train(cfg: LLMConfig, model_dir, device_str: str | None = None,
          verbose: bool = True) -> dict:
    torch.manual_seed(cfg.seed); np.random.seed(cfg.seed)
    dev = Device.detect(device_str)
    label_map = build_label_map()
    names = [n for n, _ in sorted(label_map.items(), key=lambda x: x[1])]

    tok, model, params = load_llm(model_dir, dev.target, lora=(cfg.mode == "lora"),
                                  r=cfg.lora_r, alpha=cfg.lora_alpha)
    label_ids = label_token_ids(tok, names)

    tr = ProductDataset("train", limit=cfg.limit)
    va = ProductDataset("val")
    train_ds = SFTDataset(tr, tok, cfg.max_len)
    train_loader = build_loader(train_ds, tok, cfg.batch_size, shuffle=True)

    run_id = f"llm_{cfg.mode}"
    if verbose:
        print(f"[device] {dev}\n[run] {run_id}  lr={cfg.lr}  epochs={cfg.epochs}"
              f"  train={len(tr)}")

    opt = torch.optim.AdamW(params, lr=cfg.lr, weight_decay=cfg.weight_decay)
    total = max(1, len(train_loader) * cfg.epochs)
    warmup = max(1, int(total * cfg.warmup_ratio))
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: s / warmup if s < warmup else max(0.0, (total - s) / (total - warmup)))

    best_f1, best_epoch, history = -1.0, 0, []
    OUT.mkdir(parents=True, exist_ok=True)
    started = time.time()

    for epoch in range(1, cfg.epochs + 1):
        model.train()
        total_loss, n = 0.0, 0
        for batch in train_loader:
            moved = {k: dev.to(v) for k, v in batch.items()}
            with dev.autocast():
                loss = model(**moved).loss
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, cfg.max_grad_norm)
            opt.step(); sched.step()
            total_loss += loss.item(); n += 1

        y_pred = predict(model, tok, va, label_ids, dev.target, cfg.eval_batch_size, cfg.max_len)
        y_true = [x.label for x in va.items]
        metrics = summarize(np.array(y_true), np.array(y_pred), len(names))
        history.append({"epoch": epoch, "train_loss": round(total_loss / max(1, n), 4),
                        **metrics})
        if metrics["macro_f1"] > best_f1:
            best_f1, best_epoch = metrics["macro_f1"], epoch
            state = ({k: v for k, v in model.state_dict().items()
                      if any(k.endswith(s) for s in (".A.weight", ".B.weight"))}
                     if cfg.mode == "lora" else model.state_dict())
            torch.save(state, OUT / f"{run_id}.pt")
        if verbose:
            print(f"  ep{epoch} loss={history[-1]['train_loss']:.4f} "
                  f"val macro-F1={metrics['macro_f1']:.4f} acc={metrics['acc']:.4f}")

    result = {"run_id": run_id, "family": "llm", "config": asdict(cfg),
              "loss": "sft-ce", "params": sum(p.numel() for p in model.parameters()),
              "trainable": sum(p.numel() for p in params),
              "best_epoch": best_epoch, "best_macro_f1": round(best_f1, 4),
              "history": history, "train_s": round(time.time() - started, 1),
              "device": dev.describe()}
    (OUT / f"{run_id}.json").write_text(json.dumps(result, ensure_ascii=False, indent=2),
                                        encoding="utf-8")
    record(result)
    if verbose:
        print(f"[done] {run_id}  {result['train_s']}s  best macro-F1={best_f1:.4f}")
    return result
