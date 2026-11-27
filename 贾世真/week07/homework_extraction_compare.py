# -*- coding: utf-8 -*-
"""
第七周作业：事件抽取的预训练模型对比实验

实验目的：
  在 DuEE-Fin 金融公告事件抽取数据集上，固定数据、标签体系、解码方式和评测口径，
  只更换预训练编码器，比较 BERT(12 层) 与 RBT3(3 层) 的抽取效果与成本。

实验变量：
  唯一变量 = 预训练编码器
    - BERT: bert-base-chinese（12 层，本地权重）
    - RBT3: hfl/rbt3（3 层，中文 BERT 蒸馏版，已下载到本地 HF 缓存）
  其余保持一致：joint 149 标签、Linear 解码、子词对齐、max_len、epoch、batch、lr、seed、数据划分。

运行：
  cd ~/ai-course/workspace/hub-uqEr/贾世真/week07
  ~/ai-course/.venv/bin/python homework_extraction_compare.py \
      --train-size 2000 --eval-size 300 --epochs 2 --batch-size 8

  # 只跑冒烟：
  ~/ai-course/.venv/bin/python homework_extraction_compare.py \
      --train-size 160 --eval-size 40 --epochs 1 --batch-size 8 --output-dir results_smoke
"""

import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import argparse
import json
import random
import sys
import time
from collections import Counter
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import AutoModel, AutoTokenizer, get_linear_schedule_with_warmup

ROOT = Path(__file__).resolve().parent
COMMON_DIR = ROOT / "common"
if str(COMMON_DIR) not in sys.path:
    sys.path.insert(0, str(COMMON_DIR))

from event_data import (
    BioDataset, IGNORE_INDEX, build_joint_schema, collate_bio, decode_bio_spans,
    illegal_bio_stats, load_enums, load_extract_split, load_schema, make_pred_row,
    row_event, schema_stats, spans_to_event,
)
from evaluate_events import as_event_list, evaluate, row_key


BERT_PATH = "/Users/really/Downloads/八斗学院资料/week4-语言模型/bert-base-chinese"
RBT3_PATH = "hfl/rbt3"


# ─────────────────────────── 模型 ───────────────────────────

class LinearTagger(nn.Module):
    """预训练编码器 + Linear 序列标注头：每个 token 独立 softmax 预测 BIO 标签。"""

    def __init__(self, model_path: str, num_labels: int, dropout: float = 0.1,
                 local_files_only: bool = True):
        super().__init__()
        self.bert = AutoModel.from_pretrained(model_path, local_files_only=local_files_only)
        self.hidden_size = self.bert.config.hidden_size
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(self.hidden_size, num_labels)
        self.num_labels = num_labels

    def forward(self, input_ids, attention_mask, token_type_ids=None, labels=None):
        out = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            return_dict=True,
        )
        logits = self.classifier(self.dropout(out.last_hidden_state))
        loss = None
        if labels is not None:
            loss = F.cross_entropy(
                logits.view(-1, self.num_labels),
                labels.view(-1),
                ignore_index=IGNORE_INDEX,
            )
        return logits, loss


def batch_to_device(batch: dict, device) -> dict:
    return {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}


# ─────────────────────────── 训练 / 推理 ───────────────────────────

def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.mps.manual_seed(seed) if torch.backends.mps.is_available() else None


def select_subset(rows, size, seed):
    """按固定 seed 抽取前 size 条，保证两个模型用同一批样本。"""
    if size <= 0 or size >= len(rows):
        return list(rows)
    idx = list(range(len(rows)))
    random.Random(seed).shuffle(idx)
    idx = sorted(idx[:size])  # 排序还原原始顺序，便于阅读
    return [rows[i] for i in idx]


def span_f1_from_char_tags(rows, char_tags):
    """内层验证用的轻量 span F1：角色+文本完全匹配。"""
    tp = fp = fn = 0
    for row, tags in zip(rows, char_tags):
        ge = row_event(row) or {}
        gold = {("__trigger__", (ge.get("trigger") or {}).get("text", ""))}
        gold.update(("arg", a["role"], a["text"]) for a in (ge.get("args") or []))
        gold.discard(("__trigger__", ""))
        ev = spans_to_event(decode_bio_spans(tags), row["text"])
        pred = {("__trigger__", (ev or {}).get("trigger", {}).get("text", ""))}
        pred.update(("arg", a["role"], a["text"]) for a in (ev or {}).get("args", []))
        pred.discard(("__trigger__", ""))
        tp += len(gold & pred)
        fp += len(pred - gold)
        fn += len(gold - pred)
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    return 2 * p * r / (p + r) if p + r else 0.0


def predict_char_tags(model, loader, id2label, device):
    """子词级预测 → 字符级 BIO 标签序列，只取每个汉字的首个子词。"""
    model.eval()
    all_tags = []
    with torch.no_grad():
        for batch in loader:
            batch = batch_to_device(batch, device)
            logits, _ = model(
                batch["input_ids"], batch["attention_mask"],
                batch.get("token_type_ids"),
            )
            preds = logits.argmax(-1).cpu().tolist()
            word_ids = batch["word_ids"].cpu().tolist()
            for b in range(len(preds)):
                max_wid = max(word_ids[b]) if word_ids[b] else -1
                n_chars = max_wid + 1 if max_wid >= 0 else 0
                tags = ["O"] * n_chars
                prev_wid = None
                for j, wid in enumerate(word_ids[b]):
                    if wid < 0 or wid >= n_chars:
                        prev_wid = wid
                        continue
                    if wid != prev_wid and j < len(preds[b]):
                        tags[wid] = id2label.get(preds[b][j], "O")
                    prev_wid = wid
                all_tags.append(tags)
    return all_tags


def evaluate_loss_and_f1(model, loader, id2label, device):
    model.eval()
    total, n = 0.0, 0
    with torch.no_grad():
        for batch in loader:
            batch = batch_to_device(batch, device)
            _, loss = model(
                batch["input_ids"], batch["attention_mask"],
                batch.get("token_type_ids"), batch["labels"],
            )
            total += float(loss.item())
            n += 1
    tags = predict_char_tags(model, loader, id2label, device)
    return total / max(n, 1), span_f1_from_char_tags(loader.dataset.rows, tags)


def run_epoch(model, loader, optimizer, scheduler, device, grad_accum, max_grad_norm, epoch):
    model.train()
    total, n = 0.0, 0
    optimizer.zero_grad()
    for step, batch in enumerate(loader):
        batch = batch_to_device(batch, device)
        _, loss = model(
            batch["input_ids"], batch["attention_mask"],
            batch.get("token_type_ids"), batch["labels"],
        )
        (loss / grad_accum).backward()
        total += float(loss.item())
        n += 1
        if (step + 1) % grad_accum == 0:
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_grad_norm)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
        if (step + 1) % 50 == 0:
            print(f"  epoch {epoch} step {step+1}/{len(loader)} loss={total/n:.4f}", flush=True)
    if len(loader) % grad_accum != 0:
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_grad_norm)
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()
    return total / max(n, 1)


def build_optimizer(model, lr, head_lr_mult, weight_decay):
    enc, head = [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        (enc if name.startswith("bert.") else head).append(p)
    groups = [{"params": enc, "lr": lr}]
    if head:
        groups.append({"params": head, "lr": lr * head_lr_mult})
    return AdamW(groups, weight_decay=weight_decay)


def train_one_model(name, model_path, args, label_scheme, enums, data_dir, device):
    print(f"\n========== 训练 {name}：{model_path} ==========")
    set_seed(args.seed)

    id2label = label_scheme["id2label"]
    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)

    full_train = load_extract_split("train", data_dir)
    full_dev = load_extract_split("dev", data_dir)
    train_rows = select_subset(full_train, args.train_size, args.seed)
    dev_rows = select_subset(full_dev, args.eval_size, args.seed)

    # 内层验证集：从训练样本切 10%，只用于选 epoch；dev 不参与训练与选点
    rng = random.Random(args.seed)
    idxs = list(range(len(train_rows)))
    rng.shuffle(idxs)
    n_inner = max(1, int(len(train_rows) * args.inner_val))
    inner_idx = set(idxs[:n_inner])
    fit_rows = [r for i, r in enumerate(train_rows) if i not in inner_idx]
    val_rows = [r for i, r in enumerate(train_rows) if i in inner_idx]
    print(f"训练 {len(fit_rows)} 句 / 内层验证 {len(val_rows)} 句 / 评测 {len(dev_rows)} 句")

    fit_ds = BioDataset(fit_rows, tokenizer, label_scheme, args.max_len)
    val_ds = BioDataset(val_rows, tokenizer, label_scheme, args.max_len)
    dev_ds = BioDataset(dev_rows, tokenizer, label_scheme, args.max_len)
    fit_loader = DataLoader(fit_ds, batch_size=args.batch_size, shuffle=True,
                            num_workers=0, collate_fn=collate_bio)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                            num_workers=0, collate_fn=collate_bio)
    dev_loader = DataLoader(dev_ds, batch_size=args.batch_size, shuffle=False,
                            num_workers=0, collate_fn=collate_bio)

    model = LinearTagger(model_path, len(label_scheme["labels"]),
                         dropout=args.dropout, local_files_only=True).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"参数 {n_params/1e6:.1f}M（可训练 {n_trainable/1e6:.1f}M），"
          f"标签数 {len(label_scheme['labels'])}，max_len {args.max_len}")

    optimizer = build_optimizer(model, args.lr, args.head_lr_mult, args.weight_decay)
    total_steps = max(1, len(fit_loader) * args.epochs // max(args.grad_accum, 1))
    warmup_steps = int(total_steps * args.warmup_ratio)
    scheduler = get_linear_schedule_with_warmup(optimizer, warmup_steps, total_steps)

    log = []
    best_f1, best_epoch, best_state = -1.0, 0, None
    train_start = time.perf_counter()
    for epoch in range(1, args.epochs + 1):
        t0 = time.perf_counter()
        train_loss = run_epoch(model, fit_loader, optimizer, scheduler, device,
                               args.grad_accum, args.max_grad_norm, epoch)
        val_loss, val_f1 = evaluate_loss_and_f1(model, val_loader, id2label, device)
        elapsed = time.perf_counter() - t0
        print(f"epoch {epoch}/{args.epochs} | train_loss={train_loss:.4f} | "
              f"inner_loss={val_loss:.4f} inner_spanF1={val_f1:.4f} | {elapsed:.0f}s", flush=True)
        log.append({"epoch": epoch, "train_loss": round(train_loss, 6),
                    "inner_val_loss": round(val_loss, 6),
                    "inner_val_span_f1": round(val_f1, 6),
                    "elapsed_s": round(elapsed, 1)})
        if val_f1 > best_f1:
            best_f1, best_epoch = val_f1, epoch
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            print(f"  ★ 内层验证最优 → 保存 {name} epoch {epoch}", flush=True)
    train_s = time.perf_counter() - train_start

    if best_state is not None:
        model.load_state_dict(best_state)
    print(f"加载最优 checkpoint（epoch {best_epoch}，inner_spanF1={best_f1:.4f}）")

    infer_start = time.perf_counter()
    dev_tags = predict_char_tags(model, dev_loader, id2label, device)
    infer_s = time.perf_counter() - infer_start
    illegal = illegal_bio_stats(dev_tags)

    preds, badcases = [], []
    for row, tags in zip(dev_rows, dev_tags):
        ev = spans_to_event(decode_bio_spans(tags), row["text"], enums=enums)
        pred_row = make_pred_row(row, [ev] if ev else [])
        preds.append(pred_row)
        gold_events = as_event_list(row)
        pred_events = as_event_list(pred_row)
        # 完整事件不完全正确 → 记为 badcase
        if not _events_exact_equal(gold_events, pred_events):
            badcases.append({
                "id": row.get("id", ""), "sent_id": row.get("sent_id", ""),
                "text": row["text"][:120],
                "gold": _event_brief(gold_events),
                "pred": _event_brief(pred_events),
                "error_type": _classify_error(gold_events, pred_events),
            })

    metrics = evaluate(dev_rows, preds, verbose=True)
    metrics["method"] = name
    metrics["model_path"] = model_path
    metrics["meta"] = {
        "display_name": name,
        "train_s": round(train_s, 1),
        "infer_s": round(infer_s, 2),
        "ms_per_sentence": round(infer_s * 1000 / max(len(dev_rows), 1), 2),
        "params_m": round(n_params / 1e6, 1),
        "trainable_m": round(n_trainable / 1e6, 1),
        "best_epoch": best_epoch,
        "inner_val_span_f1": round(best_f1, 4),
        "num_labels": len(label_scheme["labels"]),
        "illegal_dev": illegal,
        "n_truncated": dev_ds.n_truncated,
        "train_size": len(fit_rows),
        "eval_size": len(dev_rows),
        "epochs": args.epochs,
        "lr": args.lr,
        "batch_size": args.batch_size,
        "max_len": args.max_len,
        "seed": args.seed,
    }
    return {
        "name": name,
        "model_path": model_path,
        "preds": preds,
        "metrics": metrics,
        "train_log": log,
        "badcases": badcases,
    }


# ─────────────────────────── badcase 工具 ───────────────────────────

def _events_exact_equal(gold_events, pred_events):
    g = Counter((e["type"], e["trigger"], tuple(sorted(e["args"]))) for e in gold_events)
    p = Counter((e["type"], e["trigger"], tuple(sorted(e["args"]))) for e in pred_events)
    return g == p


def _event_brief(events):
    if not events:
        return "（无）"
    parts = []
    for e in events:
        args = "; ".join(f'{a[0]}={a[1]}' for a in e["args"][:4])
        parts.append(f'[{e["type"]}] trig={e["trigger"]} {args}')
    return " | ".join(parts)


def _classify_error(gold_events, pred_events):
    if not pred_events:
        return "漏抽事件"
    if not gold_events:
        return "多抽事件"
    g = gold_events[0]
    p = pred_events[0]
    if g["type"] != p["type"]:
        return "事件类型错误"
    if g["trigger"] != p["trigger"]:
        return "触发词错误"
    if Counter(g["args"]) != Counter(p["args"]):
        if len(p["args"]) < len(g["args"]):
            return "论元漏抽"
        if len(p["args"]) > len(g["args"]):
            return "论元多抽"
        return "论元边界/角色错误"
    return "其他"


# ─────────────────────────── 汇总 / 输出 ───────────────────────────

def write_predictions(result, out_dir):
    path = out_dir / f"predictions_{result['name']}.jsonl"
    with open(path, "w", encoding="utf-8") as f:
        for r in result["preds"]:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    return path


def write_metrics(result, out_dir):
    path = out_dir / f"metrics_{result['name']}.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(result["metrics"], f, ensure_ascii=False, indent=2)
    return path


def write_train_log(result, out_dir):
    path = out_dir / f"train_log_{result['name']}.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(result["train_log"], f, ensure_ascii=False, indent=2)
    return path


def write_summary(results, args, out_dir):
    rows = []
    for r in results:
        m = r["metrics"]
        meta = m["meta"]
        rows.append({
            "method": r["name"],
            "model": r["model_path"],
            "trigger_f1": m["trigger"]["f1"],
            "argument_f1": m["argument"]["f1"],
            "event_f1": m["event_exact"]["f1"],
            "type_macro_f1": m.get("type_macro_f1", 0.0),
            "train_s": meta["train_s"],
            "params_m": meta["params_m"],
            "ms_per_sentence": meta["ms_per_sentence"],
            "illegal_seq_rate": meta["illegal_dev"]["illegal_seq_rate"],
            "best_epoch": meta["best_epoch"],
        })
    # csv
    csv_path = out_dir / "summary.csv"
    with open(csv_path, "w", encoding="utf-8") as f:
        f.write(",".join(rows[0].keys()) + "\n")
        for r in rows:
            f.write(",".join(str(v) for v in r.values()) + "\n")
    # json
    json_path = out_dir / "summary.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump({"args": vars(args), "results": rows,
                   "badcases": {r["name"]: r["badcases"][:10] for r in results}},
                  f, ensure_ascii=False, indent=2)
    return csv_path, json_path


def make_comparison_chart(results, out_dir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    names = [r["name"] for r in results]
    metrics = ["trigger_f1", "argument_f1", "event_f1"]
    labels = ["Trigger F1", "Argument F1", "Event F1"]
    x = np.arange(len(labels))
    width = 0.35
    fig, ax = plt.subplots(figsize=(8, 5))
    for i, name in enumerate(names):
        r = results[i]
        vals = [r["metrics"][k]["f1"] for k in ["trigger", "argument", "event_exact"]]
        offset = (i - (len(names) - 1) / 2) * width
        bars = ax.bar(x + offset, vals, width, label=name)
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.01,
                    f"{v:.3f}", ha="center", va="bottom", fontsize=10)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylim(0, max(0.8, max(r["metrics"]["argument"]["f1"] for r in results) + 0.1))
    ax.set_ylabel("F1")
    ax.set_title("Event Extraction: BERT vs RBT3 (same split / same metrics)")
    ax.legend()
    fig.tight_layout()
    path = out_dir / "comparison.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def write_results_txt(results, args, out_dir, chart_path):
    lines = []
    lines.append("第七周作业：事件抽取预训练模型对比实验")
    lines.append("=" * 60)
    lines.append("")
    lines.append("实验目的：固定数据、标签、解码与评测，只更换预训练编码器，比较 BERT 与 RBT3。")
    lines.append("")
    lines.append("实验变量")
    lines.append(f"- BERT: bert-base-chinese（12 层，本地权重）")
    lines.append(f"- RBT3: hfl/rbt3（3 层，中文 BERT 蒸馏版）")
    lines.append("")
    lines.append("固定条件")
    lines.append(f"- 数据集: DuEE-Fin（train_extract / dev_extract）")
    lines.append(f"- 训练样本: {args.train_size} 句（seed={args.seed} 抽样，两模型同批）")
    lines.append(f"- 评测样本: {args.eval_size} 句（seed={args.seed} 抽样，两模型同批）")
    lines.append(f"- 标签体系: joint 149 个 BIO 标签")
    lines.append(f"- 解码器: Linear（每 token 独立 softmax）")
    lines.append(f"- max_len={args.max_len}, epochs={args.epochs}, batch_size={args.batch_size}, lr={args.lr}")
    lines.append(f"- 内层验证: 从训练集切 {args.inner_val:.0%} 选 epoch；dev 不参与选点")
    lines.append("")
    lines.append("实验结果")
    lines.append(f"{'方法':<8}{'Trigger F1':>12}{'Argument F1':>14}{'Event F1':>11}"
                 f"{'训练(s)':>10}{'延迟(ms/句)':>14}{'参数(M)':>10}")
    for r in results:
        m = r["metrics"]
        meta = m["meta"]
        lines.append(f"{r['name']:<8}{m['trigger']['f1']:>12.4f}{m['argument']['f1']:>14.4f}"
                     f"{m['event_exact']['f1']:>11.4f}{meta['train_s']:>10.1f}"
                     f"{meta['ms_per_sentence']:>14.2f}{meta['params_m']:>10.1f}")
    lines.append("")
    lines.append("主要结论")
    best_arg = max(results, key=lambda r: r["metrics"]["argument"]["f1"])
    best_evt = max(results, key=lambda r: r["metrics"]["event_exact"]["f1"])
    faster = min(results, key=lambda r: r["metrics"]["meta"]["train_s"])
    lines.append(f"- 论元 F1 最高：{best_arg['name']}（{best_arg['metrics']['argument']['f1']:.4f}）")
    lines.append(f"- 完整事件 F1 最高：{best_evt['name']}（{best_evt['metrics']['event_exact']['f1']:.4f}）")
    lines.append(f"- 训练最快：{faster['name']}（{faster['metrics']['meta']['train_s']:.1f}s）")
    lines.append("- 两个模型使用完全相同的样本和评测脚本，差异主要来自预训练编码器。")
    lines.append("- 本实验为小样本对比，结论不等于全量 DuEE-Fin 上的最终表现。")
    lines.append("")
    lines.append("典型 badcase（BERT，前 10 条）")
    for r in results:
        if r["name"] != "BERT":
            continue
        for i, bc in enumerate(r["badcases"][:10], 1):
            lines.append(f"{i:2d}. [{bc['error_type']}] id={bc['id']} sent={bc['sent_id']}")
            lines.append(f"    原文: {bc['text']}")
            lines.append(f"    金标: {bc['gold']}")
            lines.append(f"    预测: {bc['pred']}")
    lines.append("")
    lines.append("复现命令")
    lines.append(f"cd ~/ai-course/workspace/hub-uqEr/贾世真/week07")
    lines.append(f"~/ai-course/.venv/bin/python homework_extraction_compare.py "
                 f"--train-size {args.train_size} --eval-size {args.eval_size} "
                 f"--epochs {args.epochs} --batch-size {args.batch_size} --output-dir {args.output_dir}")
    lines.append("")
    lines.append("注意事项")
    lines.append("- 课件已有数字仅供参考，本表数字为本机实际运行结果。")
    lines.append("- 预训练模型均为本地加载，不依赖 dl-env。")
    lines.append("- RBT3 已通过 hf-mirror 下载到本地 HF 缓存，离线可用。")
    path = out_dir / "RESULTS.txt"
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


# ─────────────────────────── 主流程 ───────────────────────────

def parse_args():
    ap = argparse.ArgumentParser(description="第七周作业：事件抽取预训练模型对比")
    ap.add_argument("--models", nargs="+", default=["BERT", "RBT3"],
                    choices=["BERT", "RBT3"], help="参与对比的模型")
    ap.add_argument("--train-size", type=int, default=2000, help="训练样本数，0=全量")
    ap.add_argument("--eval-size", type=int, default=300, help="评测样本数，0=全量 dev")
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--max-len", type=int, default=256)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--head-lr-mult", type=float, default=5.0)
    ap.add_argument("--warmup-ratio", type=float, default=0.1)
    ap.add_argument("--grad-accum", type=int, default=1)
    ap.add_argument("--max-grad-norm", type=float, default=1.0)
    ap.add_argument("--weight-decay", type=float, default=0.01)
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--inner-val", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", default="", help="mps / cpu；留空自动选")
    ap.add_argument("--output-dir", default="results")
    return ap.parse_args()


def main():
    args = parse_args()
    if args.device:
        device = torch.device(args.device)
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    print(f"设备：{device}")

    data_dir = ROOT / "data" / "processed"
    schema = load_schema(data_dir / "event_schema.json")
    enums = load_enums(data_dir / "event_schema.json")
    label_scheme = build_joint_schema(schema)
    print(f"标签体系：joint，{len(label_scheme['labels'])} 个标签（{schema_stats(schema)}）")

    model_paths = {"BERT": BERT_PATH, "RBT3": RBT3_PATH}
    out_dir = ROOT / args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    results = []
    for name in args.models:
        result = train_one_model(name, model_paths[name], args, label_scheme, enums,
                                 data_dir, device)
        write_predictions(result, out_dir)
        write_metrics(result, out_dir)
        write_train_log(result, out_dir)
        results.append(result)
        # 及时落盘，避免长训练中途丢失
        write_summary(results, args, out_dir)

    chart_path = make_comparison_chart(results, out_dir)
    csv_path, json_path = write_summary(results, args, out_dir)
    results_txt = write_results_txt(results, args, out_dir, chart_path)
    print("\n========== 交付文件 ==========")
    print(f"对比图: {chart_path}")
    print(f"指标表: {csv_path}")
    print(f"完整结果: {json_path}")
    print(f"实验说明: {results_txt}")
    for r in results:
        print(f"{r['name']} 预测: {out_dir / ('predictions_' + r['name'] + '.jsonl')}")
        print(f"{r['name']} 指标: {out_dir / ('metrics_' + r['name'] + '.json')}")
        print(f"{r['name']} 日志: {out_dir / ('train_log_' + r['name'] + '.json')}")


if __name__ == "__main__":
    main()
