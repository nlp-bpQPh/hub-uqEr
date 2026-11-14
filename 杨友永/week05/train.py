"""全参 SFT：所有参数参与更新，按验证 loss 保存最优。"""
from __future__ import annotations

import functools
import math
import time
from typing import Any

# torch < 2.4 的 MPS 后端缺 aten::isin（generate 会用到），
# 打开回退让个别算子走 CPU，其余仍在 GPU 上跑。
import os as _os
_os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import torch
from torch.utils.data import DataLoader

from config import PATH, TRAIN, TrainConfig
from device import describe, pick_device, pick_dtype
from data import Batch, build_loader
from schemas import EpochRecord, TrainResult


def load_base(path=PATH.model_dir) -> tuple[Any, Any]:
    """加载 tokenizer 与底座模型。"""
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(str(path), local_files_only=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    dtype = pick_dtype(pick_device())
    try:
        model = AutoModelForCausalLM.from_pretrained(
            str(path), local_files_only=True, dtype=dtype)
    except TypeError:                       # transformers < 5
        model = AutoModelForCausalLM.from_pretrained(
            str(path), local_files_only=True, torch_dtype=dtype)
    return tok, model


@torch.no_grad()
def evaluate(model: Any, loader: DataLoader, device: torch.device) -> float:
    """验证集平均 loss。"""
    model.eval()
    total, count = 0.0, 0
    for batch in loader:
        moved: Batch = {k: v.to(device) for k, v in batch.items()}
        total += model(**moved).loss.item()
        count += 1
    return total / max(1, count)


def _lr_lambda(step: int, total_steps: int, warmup: int) -> float:
    if step < warmup:
        return step / max(1, warmup)
    return max(0.0, (total_steps - step) / max(1, total_steps - warmup))


def train(cfg: TrainConfig = TRAIN, limit: int | None = None,
          lr: float | None = None, device_str: str | None = None) -> TrainResult:
    """训练主流程，返回并落盘训练结果。"""
    torch.manual_seed(cfg.seed)
    lr = cfg.lr if lr is None else lr
    PATH.out_dir.mkdir(parents=True, exist_ok=True)
    device = pick_device(device_str)
    print(f"[device] {describe(device)}")

    tok, model = load_base()
    model.to(device)
    if cfg.grad_ckpt:
        model.config.use_cache = False
        model.gradient_checkpointing_enable()

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"[params] trainable={trainable/1e6:.1f}M / total={total/1e6:.1f}M")

    train_ds, train_loader = build_loader(
        PATH.train_file, tok, cfg.max_len, cfg.batch_size, shuffle=True)
    val_ds, val_loader = build_loader(
        PATH.val_file, tok, cfg.max_len, cfg.batch_size, shuffle=False)

    if limit:
        subset = torch.utils.data.Subset(train_ds, range(min(limit, len(train_ds))))
        train_loader = DataLoader(subset, batch_size=cfg.batch_size, shuffle=True,
                                  collate_fn=train_loader.collate_fn)
    print(f"[data] train={len(train_ds)} val={len(val_ds)} dropped={train_ds.dropped}")

    params = list(model.parameters())
    optimizer = torch.optim.AdamW(params, lr=lr, weight_decay=cfg.weight_decay)
    steps_per_epoch = math.ceil(len(train_loader) / cfg.grad_accum)
    total_steps = max(1, int(steps_per_epoch * cfg.epochs))
    warmup = max(1, int(total_steps * cfg.warmup_ratio))
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, functools.partial(_lr_lambda, total_steps=total_steps, warmup=warmup))

    history: list[EpochRecord] = []
    best_val, best_epoch = float("inf"), 0
    use_amp = device.type == "cuda"
    step = 0
    started = time.time()

    for epoch in range(1, int(math.ceil(cfg.epochs)) + 1):
        model.train()
        running = 0.0
        for i, batch in enumerate(train_loader, 1):
            moved: Batch = {k: v.to(device) for k, v in batch.items()}
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=use_amp):
                loss = model(**moved).loss / cfg.grad_accum
            loss.backward()
            running += loss.item() * cfg.grad_accum

            if i % cfg.grad_accum == 0 or i == len(train_loader):
                torch.nn.utils.clip_grad_norm_(params, cfg.max_grad_norm)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                step += 1
                if step % 20 == 0:
                    print(f"  ep{epoch} step {step}/{total_steps} "
                          f"loss={running / i:.4f}")

        val_loss = evaluate(model, val_loader, device)
        history.append(EpochRecord(epoch=epoch,
                                   train_loss=round(running / max(1, len(train_loader)), 4),
                                   val_loss=round(val_loss, 4)))
        saved = ""
        if val_loss < best_val:                 # 只保存验证集最好的那一轮
            best_val, best_epoch = val_loss, epoch
            model.save_pretrained(PATH.ckpt_dir)
            # tokenizer 不保存：它不会因 SFT 改变，且跨 tokenizers 版本容易不兼容
            saved = "  ← 保存"
        print(f"[eval] epoch {epoch}  val_loss={val_loss:.4f}{saved}")

    result = TrainResult(
        train_n=len(train_ds), val_n=len(val_ds), dropped=train_ds.dropped,
        epochs=cfg.epochs, lr=lr, batch_size=cfg.batch_size,
        grad_accum=cfg.grad_accum, grad_ckpt=cfg.grad_ckpt, max_len=cfg.max_len,
        trainable=trainable, total=total, history=history,
        best_epoch=best_epoch, best_val_loss=round(best_val, 4),
        train_s=round(time.time() - started, 1), device=str(device),
    )
    PATH.result_file.parent.mkdir(parents=True, exist_ok=True)
    PATH.result_file.write_text(result.model_dump_json(indent=2), encoding="utf-8")
    print(f"\n[done] {result.train_s}s，best epoch={best_epoch} "
          f"val_loss={best_val:.4f} → {PATH.ckpt_dir}")
    return result
