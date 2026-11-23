"""BERT 训练：配置、单轮、评测、落盘。"""
from __future__ import annotations

import csv
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch

from bert.dataloader import MAX_LEN, build_all, build_tokenizer
from bert.losses import FGM, build_criterion
from bert.model import build_model
from dataset import build_label_map
from device import Device
from metrics import summarize

OUT = Path(__file__).resolve().parent.parent / "outputs"
RESULTS = OUT / "results.csv"


@dataclass(frozen=True)
class BertConfig:
    pooling: str = "cls"
    lr: float = 3e-5
    epochs: int = 3
    batch_size: int = 32
    max_len: int = MAX_LEN
    weight_decay: float = 0.01
    warmup_ratio: float = 0.1
    max_grad_norm: float = 1.0
    dropout: float = 0.1
    label_smoothing: float = 0.0      # >0 时启用 LS
    focal_gamma: float = 0.0          # >0 时启用 Focal
    fgm_eps: float = 0.0              # >0 时启用 FGM
    seed: int = 42
    eval_batch_size: int = 128


def pick_loss(cfg: BertConfig) -> tuple[str, torch.nn.Module]:
    if cfg.focal_gamma > 0:
        return "focal", build_criterion("focal", focal_gamma=cfg.focal_gamma)
    if cfg.label_smoothing > 0:
        return "ls", build_criterion("ls", label_smoothing=cfg.label_smoothing)
    return "ce", build_criterion("ce")


@torch.no_grad()
def evaluate(model, loader, dev: Device, num_classes: int) -> tuple[dict, list, list]:
    model.eval()
    preds, labels = [], []
    for batch in loader:
        moved = {k: dev.to(v) for k, v in batch.items()}
        with dev.autocast():
            logits = model(moved["input_ids"], moved["attention_mask"],
                           moved.get("token_type_ids"))
        preds.append(logits.float().argmax(-1).cpu())
        labels.append(moved["labels"].cpu())
    y_pred = torch.cat(preds).numpy()
    y_true = torch.cat(labels).numpy()
    return summarize(y_true, y_pred, num_classes), y_true.tolist(), y_pred.tolist()


def train(cfg: BertConfig, run_id: str, model_path, device_str: str | None = None,
          verbose: bool = True) -> dict:
    torch.manual_seed(cfg.seed); np.random.seed(cfg.seed)
    dev = Device.detect(device_str)
    label_map = build_label_map()
    num_classes = len(label_map)

    tok = build_tokenizer(model_path)
    train_loader, val_loader, _ = build_all(tok, cfg.batch_size, cfg.max_len)
    model = build_model(model_path, num_classes, cfg.pooling, device=dev.target)
    loss_name, criterion = pick_loss(cfg)
    fgm = FGM(model, cfg.fgm_eps) if cfg.fgm_eps > 0 else None

    if verbose:
        print(f"[device] {dev}\n[run] {run_id}  pooling={cfg.pooling} loss={loss_name}"
              f" fgm={cfg.fgm_eps}")
        print(f"[model] {model.num_parameters()}")

    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=cfg.lr, weight_decay=cfg.weight_decay)
    total = max(1, len(train_loader) * cfg.epochs)
    warmup = max(1, int(total * cfg.warmup_ratio))
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: s / warmup if s < warmup else max(0.0, (total - s) / (total - warmup)))

    best_f1, best_epoch = -1.0, 0
    history = []
    OUT.mkdir(parents=True, exist_ok=True)
    started = time.time()

    for epoch in range(1, cfg.epochs + 1):
        model.train()
        total_loss, n = 0.0, 0
        for batch in train_loader:
            moved = {k: dev.to(v) for k, v in batch.items()}
            with dev.autocast():
                logits = model(moved["input_ids"], moved["attention_mask"],
                               moved.get("token_type_ids"))
                loss = criterion(logits.float(), moved["labels"])

            opt.zero_grad(set_to_none=True)
            loss.backward()

            if fgm is not None:
                fgm.attack()                       # 扰动 embedding
                with dev.autocast():
                    logits2 = model(moved["input_ids"], moved["attention_mask"],
                                    moved.get("token_type_ids"))
                    loss2 = criterion(logits2.float(), moved["labels"])
                loss2.backward()                   # 梯度累加
                fgm.restore()

            torch.nn.utils.clip_grad_norm_(params, cfg.max_grad_norm)
            opt.step(); sched.step()
            total_loss += loss.item(); n += 1

        metrics, _, _ = evaluate(model, val_loader, dev, num_classes)
        history.append({"epoch": epoch, "train_loss": round(total_loss / max(1, n), 4),
                        **metrics})
        if metrics["macro_f1"] > best_f1:
            best_f1, best_epoch = metrics["macro_f1"], epoch
            torch.save(model.state_dict(), OUT / f"{run_id}.pt")
        if verbose:
            print(f"  ep{epoch} loss={history[-1]['train_loss']:.4f} "
                  f"val macro-F1={metrics['macro_f1']:.4f} acc={metrics['acc']:.4f}")

    result = {"run_id": run_id, "family": "bert", "config": asdict(cfg),
              "loss": loss_name, "params": model.num_parameters()["total"],
              "best_epoch": best_epoch, "best_macro_f1": round(best_f1, 4),
              "history": history, "train_s": round(time.time() - started, 1),
              "device": dev.describe()}
    (OUT / f"{run_id}.json").write_text(json.dumps(result, ensure_ascii=False, indent=2),
                                        encoding="utf-8")
    record(result)
    if verbose:
        print(f"[done] {run_id}  {result['train_s']}s  best macro-F1={best_f1:.4f}")
    return result


def record(result: dict) -> None:
    """汇总到 outputs/results.csv，供对比报告使用。"""
    RESULTS.parent.mkdir(parents=True, exist_ok=True)
    exists = RESULTS.exists()
    with open(RESULTS, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["run_id", "family", "loss", "params",
                                          "best_macro_f1", "best_epoch", "train_s"])
        if not exists:
            w.writeheader()
        w.writerow({"run_id": result["run_id"], "family": result["family"],
                    "loss": result.get("loss", "-"), "params": result.get("params", "-"),
                    "best_macro_f1": result.get("best_macro_f1", 0),
                    "best_epoch": result.get("best_epoch", 0),
                    "train_s": result.get("train_s", 0)})
