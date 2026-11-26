"""
L2 主线：BERT 序列标注做事件抽取（一次前向同时抽「事件类型 + 触发词 + 论元」）

教学重点：
  1. **子词对齐（word_ids）**：token 数可能 ≠ 字数 —— bert-base-chinese 对中文基本一字一 token，
     但特殊符 / 空格（不产生 token）/ 数字英文（整句分词时会被切成 "005"+"##10"）/ 换 tokenizer
     都会让两者错位。只有首个子词背标签、其余设 -100；不这么做，同一个 span 的 loss 会被重复算。
  2. **一套脚本两种解码器**：`--decoder linear|crf`
       linear：每个 token 独立 softmax，简单快，但会喷出非法 BIO 序列
       crf   ：加转移矩阵做全局最优解码，非法序列显著减少（本项目手写，见 crf.py）
     两者共用同一份标签体系与解码代码，指标才可比。
  3. **非法 BIO 转移统计**：linear vs crf 的核心量化对比点（PPT 上那张柱状图）。
  4. **BIO 标签体系**：joint 体系 149 个标签（触发词按 13 类拆开、论元 61 个角色跨类型共享），
     这是「联合抽取」—— 一个模型端到端，但因为标签空间大、每类样本少，论元通常偏弱。
     两段式（train_pipeline.py）正是为了解决这个问题，两边的对比是本周的核心。
  5. **模型选择用内层验证集**：从 train 里切 10% 做 early stopping，**dev 只用来最终评测**。
     否则「按 dev 选 epoch」会把 dev 数字抬虚（选点本身就是一个隐藏超参）。

使用方式：
  python bert/train.py                            # 主线：12 层 BERT + CRF
  python bert/train.py --decoder linear           # 对照：BERT + Linear
  python bert/train.py --decoder linear --max_len 128 --run_name bert_linear_len128
  python bert/train.py --bert_layers 4            # 消融：只训 4 层（约 1/3 时间）
  python bert/train.py --decoder crf --hard_constraint   # 消融：把非法转移硬约束到 0
  python bert/train.py --mock                     # 用 data/_mock 的 20 条数据跑通流程

产物：
  outputs/checkpoints/<run_name>.pt        checkpoint（含 label2id / 超参 / 最优 epoch）
  outputs/logs/train_<run_name>.json       逐 epoch 曲线（train_loss / 内层val / dev 监控值）
  outputs/disc/pred_<run_name>.jsonl       预测（交给统一评测）
  outputs/metrics/disc/<run_name>.json     指标（含非法 BIO 统计、训练时长、参数量、延迟）
"""

import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import argparse
import json
import math
import random
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, get_linear_schedule_with_warmup

ROOT = Path(__file__).resolve().parents[1]   # 项目根目录（方法目录的上一级）
sys.path.insert(0, str(ROOT / "common"))     # 公共模块：event_data / crf / eval_utils / models / log_utils

from crf import CRFClassifier
from eval_utils import CKPT_DIR, LOG_DIR, Timer, evaluate_predictions
from log_utils import TrainLogger, peak_gpu_mb
from event_data import (BioDataset, DATA_DIR, ROOT, build_bio_constraint_mask, collate_bio,
                        build_joint_schema, build_start_constraint_mask, decode_bio_spans,
                        illegal_bio_stats, load_enums, load_extract_split, load_jsonl, load_schema,
                        make_pred_row, row_event, sample_negative_rows, spans_to_event)
from model_paths import BERT_MODEL_DIR, resolve_model_path

from models import BertEncoder, batch_to_device, build_model, predict_char_tags

MOCK_DIR = ROOT / "data" / "_mock"
IGNORE_INDEX = -100

# ─────────────────────────── 模型 ───────────────────────────

# ─────────────────────────── 训练 / 评估 ───────────────────────────

def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def span_f1_from_char_tags(rows: list[dict], char_tags: list[list[str]]) -> float:
    """
    训练中做模型选择用的轻量指标：**span 级精确匹配 F1**（角色+文本完全一致），
    口径和最终评测的 argument F1 接近，但不依赖 930 条 dev、也更快。
    """
    tp = fp = fn = 0
    for row, tags in zip(rows, char_tags):
        ge = row_event(row) or {}
        gold = {(a["role"], a["text"]) for a in ge.get("args") or []}
        gold.add(("__trigger__", (ge.get("trigger") or {}).get("text", "")))
        ev = spans_to_event(decode_bio_spans(tags), row["text"])
        pred = {(a["role"], a["text"]) for a in (ev or {}).get("args", [])}
        pred.add(("__trigger__", (ev or {}).get("trigger", {}).get("text", "")))
        pred.discard(("__trigger__", ""))
        gold.discard(("__trigger__", ""))
        tp += len(gold & pred)
        fp += len(pred - gold)
        fn += len(gold - pred)
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    return 2 * p * r / (p + r) if p + r else 0.0

def evaluate_loss_and_f1(model, loader, id2label: dict, device) -> tuple[float, float]:
    """在验证集上算平均 loss 与 span F1。"""
    model.eval()
    total, n = 0.0, 0
    with torch.no_grad():
        for batch in loader:
            batch = batch_to_device(batch, device)
            _, loss = model(batch["input_ids"], batch["attention_mask"],
                            batch["labels"], batch.get("token_type_ids"))
            total += float(loss.item())
            n += 1
    avg_loss = total / max(n, 1)
    tags, _ = predict_char_tags(model, loader, id2label, device)
    f1 = span_f1_from_char_tags(loader.dataset.rows, tags)
    return avg_loss, f1

def build_optimizer(model, args):
    """分层学习率：BERT 用小 lr（2e-5），新初始化的分类头 / CRF 用 args.head_lr_mult 倍。"""
    enc_params, head_params = [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        (enc_params if name.startswith("encoder.") else head_params).append(p)
    groups = [{"params": enc_params, "lr": args.lr}]
    if head_params:
        groups.append({"params": head_params, "lr": args.lr * args.head_lr_mult})
    return AdamW(groups, weight_decay=args.weight_decay)

def run_epoch(model, loader, optimizer, scheduler, device, args, epoch: int,
              logger=None) -> float:
    model.train()
    total, n = 0.0, 0
    optimizer.zero_grad()
    n_batches = len(loader)
    tail_size = n_batches % args.grad_accum
    tail_start = n_batches - tail_size if tail_size else n_batches
    for step, batch in enumerate(loader):
        batch = batch_to_device(batch, device)
        _, loss = model(batch["input_ids"], batch["attention_mask"],
                        batch["labels"], batch.get("token_type_ids"))
        # 最后一组不足 grad_accum 个 micro-batch 时，按实际组大小取平均，
        # 避免 3/4 组等尾部梯度被额外缩小为完整组的 3/4。
        group_size = tail_size if tail_size and step >= tail_start else args.grad_accum
        (loss / group_size).backward()
        total += float(loss.item())
        n += 1
        is_full_group_end = (step + 1) % args.grad_accum == 0
        is_epoch_end = (step + 1) == n_batches
        if is_full_group_end or is_epoch_end:
            # 梯度裁剪：CRF 在序列开头几步容易出大梯度，不裁剪偶发 loss 爆掉
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=args.max_grad_norm)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
        if logger is not None:
            logger.log_step(epoch=epoch, step=step, loss=float(loss.item()),
                            lr=scheduler.get_last_lr()[0])
        if (step + 1) % 50 == 0:
            print(f"  epoch {epoch} step {step + 1}/{len(loader)} loss={total / n:.4f}", flush=True)
    return total / max(n, 1)

# ─────────────────────────── 主流程 ───────────────────────────

def main():
    args = parse_args()
    set_seed(args.seed)
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    print(f"设备：{device}")
    if device.type == "cuda":
        print(f"显存占用（启动时）：{torch.cuda.memory_allocated() / 1024 ** 2:.0f} MiB")

    data_dir = (Path(args.data_dir) if args.data_dir
                else (MOCK_DIR if args.mock else DATA_DIR))
    schema = load_schema(data_dir / "event_schema.json")
    enums = load_enums(data_dir / "event_schema.json")
    label_scheme = build_joint_schema(schema)
    labels = label_scheme["labels"]
    id2label = label_scheme["id2label"]
    print(f"标签体系：joint，{len(labels)} 个标签（{schema_stats_str(schema)}）")

    model_path = Path(str(args.bert_path)).expanduser()
    tokenizer = AutoTokenizer.from_pretrained(
        str(args.bert_path), use_fast=True,
        local_files_only=(model_path.exists()
                          or os.environ.get("HF_HUB_OFFLINE") == "1"
                          or os.environ.get("TRANSFORMERS_OFFLINE") == "1"),
    )
    if not getattr(tokenizer, "is_fast", False):
        raise RuntimeError("本项目依赖 fast tokenizer 的 word_ids() 做字符对齐；请安装 fast tokenizer。")

    train_rows = load_extract_split("train", data_dir)
    positive_train_count = len(train_rows)
    negative_count = 0
    if args.negative_ratio > 0:
        detect_path = data_dir / "train_detect.jsonl"
        if not detect_path.exists():
            raise FileNotFoundError(f"开启负样本训练需要文件：{detect_path}")
        train_rows, negative_count = sample_negative_rows(
            train_rows, load_jsonl(detect_path), args.negative_ratio, args.seed)

    dev_path = data_dir / args.dev_file
    if not dev_path.exists() and args.dev_file == "dev_extract_clean.jsonl" and (data_dir / "dev_extract.jsonl").exists():
        # mock/旧数据目录没有 clean 文件时，明确回退到原 dev，并把真实文件名写入 checkpoint。
        dev_path = data_dir / "dev_extract.jsonl"
        args.dev_file = dev_path.name
    if not dev_path.exists():
        raise FileNotFoundError(f"评测文件不存在：{dev_path}")
    dev_rows = load_jsonl(dev_path)

    # 内层验证集：从 train 切 10%，只用于选 epoch；dev 完全不参与训练/选点
    rng = random.Random(args.seed)
    idxs = list(range(len(train_rows)))
    rng.shuffle(idxs)
    n_inner = max(1, int(len(train_rows) * args.inner_val))
    inner_idx = set(idxs[:n_inner])
    fit_rows = [r for i, r in enumerate(train_rows) if i not in inner_idx]
    val_rows = [r for i, r in enumerate(train_rows) if i in inner_idx]
    print(f"训练 {len(fit_rows)} 句 / 内层验证 {len(val_rows)} 句 / 评测 {args.dev_file} {len(dev_rows)} 句")
    print(f"训练样本构成：正样本 {positive_train_count}，新增负样本 {negative_count}（ratio={args.negative_ratio:g}）")

    fit_ds = BioDataset(fit_rows, tokenizer, label_scheme, args.max_len)
    val_ds = BioDataset(val_rows, tokenizer, label_scheme, args.max_len)
    dev_ds = BioDataset(dev_rows, tokenizer, label_scheme, args.max_len)
    fit_loader = DataLoader(fit_ds, batch_size=args.batch_size, shuffle=True, num_workers=0, collate_fn=collate_bio)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=0, collate_fn=collate_bio)
    dev_loader = DataLoader(dev_ds, batch_size=args.batch_size, shuffle=False, num_workers=0, collate_fn=collate_bio)

    model = build_model(args, label_scheme).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"模型参数：{n_params / 1e6:.1f}M（可训练 {n_trainable / 1e6:.1f}M），"
          f"编码器={args.encoder_name}，解码器={args.decoder}，层数={args.bert_layers or 'full(12)'}")

    optimizer = build_optimizer(model, args)
    total_steps = max(1, math.ceil(len(fit_loader) / max(args.grad_accum, 1)) * args.epochs)
    warmup_steps = int(total_steps * args.warmup_ratio)
    scheduler = get_linear_schedule_with_warmup(optimizer, warmup_steps, total_steps)
    print(f"训练步数 {total_steps}（warmup {warmup_steps} 步，{args.warmup_ratio:.0%}）")

    CKPT_DIR.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    ckpt_path = CKPT_DIR / f"{args.run_name}.pt"

    logger = TrainLogger(method=args.run_name, args=args, root=ROOT, params=n_params,
                         notes=f"{args.encoder_name} 序列标注（{args.decoder}）训练过程")

    log, best_f1, best_epoch = [], -1.0, 0
    train_start = time.perf_counter()

    for epoch in range(1, args.epochs + 1):
        t0 = time.perf_counter()
        train_loss = run_epoch(model, fit_loader, optimizer, scheduler, device, args, epoch,
                               logger=logger)
        val_loss, val_f1 = evaluate_loss_and_f1(model, val_loader, id2label, device)
        # dev 只做「监控曲线」，不参与选点（否则 dev 数字会偏乐观）
        _, dev_f1_monitor = evaluate_loss_and_f1(model, dev_loader, id2label, device)
        elapsed = time.perf_counter() - t0
        print(f"epoch {epoch}/{args.epochs} | train_loss={train_loss:.4f} | "
              f"inner_loss={val_loss:.4f} inner_spanF1={val_f1:.4f} | "
              f"dev_spanF1(监控)={dev_f1_monitor:.4f} | {elapsed:.0f}s", flush=True)
        logger.log_epoch(epoch=epoch, train_loss=train_loss, elapsed=elapsed,
                         val_metrics={"inner_loss": val_loss, "inner_span_f1": val_f1,
                                      "dev_span_f1_monitor": dev_f1_monitor},
                         extra={"best_so_far": round(best_f1, 6)})
        log.append({"epoch": epoch, "train_loss": round(train_loss, 6),
                    "inner_val_loss": round(val_loss, 6),
                    "inner_val_span_f1": round(val_f1, 6),
                    "dev_span_f1_monitor": round(dev_f1_monitor, 6),
                    "elapsed_s": round(elapsed, 1)})
        if val_f1 > best_f1:
            best_f1, best_epoch = val_f1, epoch
            torch.save({"state_dict": model.state_dict(),
                        "label_scheme": {k: label_scheme[k] for k in
                                         ("labels", "label2id", "id2label", "name")},
                        "id2label": id2label,
                        "args": vars(args),
                        "epoch": epoch,
                        "inner_val_span_f1": val_f1}, ckpt_path)
            print(f"  ★ 内层验证最优 → 保存 {ckpt_path.name}", flush=True)

    train_s = time.perf_counter() - train_start
    with open(LOG_DIR / f"train_{args.run_name}.json", "w", encoding="utf-8") as f:
        json.dump(log, f, ensure_ascii=False, indent=2)

    # ── 用最优 checkpoint 在 dev 上出最终预测 ──
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["state_dict"])
    print(f"加载最优 checkpoint（epoch {ckpt['epoch']}，内层验证 spanF1={best_f1:.4f}）")

    with Timer() as timer:
        dev_tags, dev_pred_info = predict_char_tags(model, dev_loader, id2label, device)
    dev_illegal = illegal_bio_stats(dev_tags)
    dev_token_illegal = dev_pred_info.get("illegal_token_bio", {})

    # 训练集上的非法统计也报一份（模型在见过的数据上表现如何，便于对照）
    with Timer() as timer_fit:
        fit_tags, fit_pred_info = predict_char_tags(model, fit_loader, id2label, device)
    fit_illegal = illegal_bio_stats(fit_tags)
    fit_token_illegal = fit_pred_info.get("illegal_token_bio", {})

    preds = []
    n_empty = 0
    for row, tags in zip(dev_rows, dev_tags):
        ev = spans_to_event(decode_bio_spans(tags), row["text"], enums=enums)
        if ev is None:
            n_empty += 1
        preds.append(make_pred_row(row, [ev] if ev else []))
    print(f"dev 解码：{len(preds)} 句，其中没抽到触发词 {n_empty} 句")

    lat = timer.summary(len(dev_rows))
    meta = {
        "display_name": f"{args.encoder_name}-{args.bert_layers or 12}L+{args.decoder.upper()}",
        "encoder_name": args.encoder_name,
        "pretrained_model": str(args.bert_path),
        "train_positive_count": positive_train_count,
        "train_negative_count": negative_count,
        "negative_ratio": args.negative_ratio,
        "dev_file": args.dev_file,
        "train_s": round(train_s, 1),
        "params_m": round(n_params / 1e6, 1),
        "ms_per_sentence": lat["ms_per_sentence"],
        "peak_gpu_mb": lat["peak_gpu_mb"],
        "decoder": args.decoder,
        "bert_layers": args.bert_layers or 12,
        "max_len": args.max_len,
        "epochs": args.epochs,
        "lr": args.lr,
        "head_lr_mult": args.head_lr_mult,
        "batch_size": args.batch_size,
        "grad_accum": args.grad_accum,
        "best_epoch": best_epoch,
        "inner_val_span_f1": round(best_f1, 4),
        "num_labels": len(labels),
        "illegal_dev": dev_illegal,  # 字符级：空格补 O 后的统计，保留兼容旧图表
        "illegal_token_dev": dev_token_illegal,  # token 级：真实模型解码序列
        "illegal_train_pred": fit_illegal,
        "illegal_token_train_pred": fit_token_illegal,
        "n_truncated": dev_ds.n_truncated,
        "hard_constraint": bool(args.hard_constraint),
        "seed": args.seed,
        "selection_note": "按 train 切出的 10% 内层验证集选 epoch；dev 未参与训练与选点",
    }
    res = evaluate_predictions(args.run_name, preds, meta=meta,
                              gold_path=dev_path)

    print("\n【非法 BIO 统计（dev，模型原始输出）】")
    print(f"  字符级：含非法序列 {dev_illegal['illegal_seqs']} 条，"
          f"非法开头 {dev_illegal['illegal_start']} 处、非法转移 {dev_illegal['illegal_transition']} 处")
    if dev_token_illegal:
        print(f"  token级：含非法序列 {dev_token_illegal['illegal_seqs']} 条，"
              f"非法开头 {dev_token_illegal['illegal_start']} 处、"
              f"非法转移 {dev_token_illegal['illegal_transition']} 处")
    print(f"  训练集对照：含非法序列 {fit_illegal['illegal_seqs']} 条"
          f"（{fit_illegal['illegal_seq_rate']:.1%}）")
    print(f"\n训练耗时 {train_s:.0f}s | 推理延迟 {lat['ms_per_sentence']} ms/句 | "
          f"峰值显存 {lat['peak_gpu_mb']:.0f} MiB")

    # 训练日志收尾（说明见 common/log_utils.py）：好让「训练过程」可被学生事后复看
    logger.finalize(
        total_s=train_s,
        best_epoch=best_epoch,
        best_inner_val_span_f1=round(best_f1, 6),
        dev_metrics={k: res[k]["f1"] for k in ("trigger", "argument", "event_exact")},
        inference_latency_ms=lat["ms_per_sentence"],
        peak_gpu_mb=peak_gpu_mb(),
        illegal_dev=dev_illegal,
        illegal_train_pred_sample=fit_illegal,
    )

def schema_stats_str(schema: dict) -> str:
    roles = {r for v in schema.values() for r in v}
    return f"{len(schema)} 类 / {sum(len(v) for v in schema.values())} 槽位 / {len(roles)} 去重角色"

def parse_args():
    ap = argparse.ArgumentParser(description="基于预训练 Transformer 的金融事件抽取（linear / crf）")
    ap.add_argument("--decoder", choices=["linear", "crf"], default="crf")
    ap.add_argument("--bert_path", "--model_path", dest="bert_path", default="",
                    help="预训练模型目录或 Hugging Face 模型 ID（BERT/RoBERTa 均可）")
    ap.add_argument("--encoder_name", default="BERT", help="实验展示名称，如 BERT / RoBERTa-WWM-Ext")
    ap.add_argument("--dev_file", default="dev_extract_clean.jsonl",
                    help="最终评测 JSONL；默认使用清理过文本重叠的 dev 文件")
    ap.add_argument("--negative_ratio", type=float, default=0.0,
                    help="负样本:正样本比例；0=关闭，1=约 1:1")
    ap.add_argument("--bert_layers", type=int, default=0, help="0=完整 12 层；4=消融")
    ap.add_argument("--max_len", type=int, default=256)
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--head_lr_mult", type=float, default=5.0)
    ap.add_argument("--warmup_ratio", type=float, default=0.1)
    ap.add_argument("--grad_accum", type=int, default=1)
    ap.add_argument("--max_grad_norm", type=float, default=1.0)
    ap.add_argument("--weight_decay", type=float, default=0.01)
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--inner_val", type=float, default=0.1, help="从 train 切多少做内层验证")
    ap.add_argument("--hard_constraint", action="store_true", help="CRF 硬约束（非法=0 的消融）")
    ap.add_argument("--device", default="", help="cuda / cpu；留空自动选（mock 冒烟建议 cpu，别抢卡）")
    ap.add_argument("--log_every", type=int, default=50,
                    help="每多少步往 steps.jsonl 写一行（供学生画 loss 曲线）")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--run_name", default="")
    ap.add_argument("--mock", action="store_true", help="用 data/_mock 的 20 条手造数据（离线冒烟）")
    ap.add_argument("--data_dir", default="", help="数据目录；留空=全量 data/processed（--mock 时用 data/_mock）")
    args = ap.parse_args()
    if args.negative_ratio < 0:
        ap.error("--negative_ratio 不能小于 0")
    if args.max_len < 8:
        ap.error("--max_len 至少为 8，必须为特殊 token 和正文留出空间")
    if args.batch_size < 1 or args.grad_accum < 1:
        ap.error("--batch_size 和 --grad_accum 必须为正整数")
    if not args.bert_path:
        args.bert_path = str(resolve_model_path(BERT_MODEL_DIR))
    if not args.run_name:
        tag = f"{args.encoder_name.lower()}{args.bert_layers}L" if args.bert_layers else args.encoder_name.lower()
        extra = "_hard" if args.hard_constraint else ""
        neg = f"_neg{args.negative_ratio:g}" if args.negative_ratio > 0 else ""
        args.run_name = f"{tag}_{args.decoder}_{args.max_len}{extra}{neg}" + ("_mock" if args.mock else "")
    return args

if __name__ == "__main__":
    main()
