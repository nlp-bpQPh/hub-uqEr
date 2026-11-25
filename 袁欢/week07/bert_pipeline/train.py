"""
L2 两段式流水线（pipeline）事件检测：类型分类 → 类型条件触发词专家

这是本周的**核心教学对比**：同样是 BERT，一次联合抽取（bert/train.py，joint 19 标签）
vs 拆成两段各管一件事，哪个更高？

两段的分工（2026-11-24 换装 CCED 起：论元链路已砍、原检测段因无负样本删除）：
  (a) 事件类型 9 分类  这句是 9 类里的哪一类？（[CLS] 分类头）
  (b) 类型条件触发词抽取 已知类型 t，抽「触发词 span」（B-TRG/I-TRG，每类 3 个标签）

为什么流水线常常更高：
  - (b) 的标签空间小一个数量级（每类 3 标签 vs joint 的 19），触发词边界学得准
  - 联合模型的 19 个标签里，长尾类型只有几十个样本，触发词边界容易学糊
为什么流水线也常常更低（课上要讲的反面）：
  - 误差会**累积**：(a) 类型错 → (b) 的专家一定全错；而且两段各跑一次，延迟约 2 倍
  - 训练要维护 10 个模型（1 类型 + 9 专家），工程复杂度高
所以本项目把两个数字都实测出来，让学生看到「不是越拆越好」。

这一版刻意用**线性头**而不是 CRF：(b) 段每类只有 3 个标签，非法序列问题本来就小，
加 CRF 的收益不抵训练/推理成本 —— 这个取舍也写进 NOTES。

使用方式：
  python bert_pipeline/train.py                        # 完整训练 + 组合评测
  python bert_pipeline/train.py --arg_epochs 2         # 少训一点
  python bert_pipeline/train.py --skip_train           # 复用已有 checkpoint，只重新组合评测
  python bert_pipeline/train.py --mock                 # 20 条手造数据跑通流程

产物：
  outputs/checkpoints/pipe_type.pt / pipe_arg_<类型>.pt
  outputs/metrics/disc/pipeline.json           端到端（dev_extract）
  outputs/metrics/disc/pipeline_oracle_type.json  类型给对时的上界（定位误差来源）
"""

import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import argparse
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import AutoModel, BertTokenizerFast, get_linear_schedule_with_warmup

ROOT = Path(__file__).resolve().parents[1]   # 项目根目录（方法目录的上一级）
sys.path.insert(0, str(ROOT / "common"))     # 公共模块：event_data / crf / eval_utils / models / log_utils

from log_utils import TrainLogger, peak_gpu_mb
from eval_utils import (CKPT_DIR, LOG_DIR, METRIC_DIR, Timer, evaluate_predictions,
                        evaluate_nested_subset)
from event_data import (BioDataset, ClsDataset, DATA_DIR, ROOT, build_typed_schema,
                        collate_bio, collate_cls, decode_bio_spans, illegal_bio_stats,
                        load_enums, load_extract_split,
                        load_schema, make_pred_row, spans_to_event)
from model_paths import BERT_MODEL_DIR, resolve_model_path

from models import BertCls, _Layers, _layers_of, build_arg_model

MOCK_DIR = ROOT / "data" / "_mock"

# ─────────────────────────── BERT 分类模型 ───────────────────────────

def train_classifier(rows, labels_of, num_labels, tokenizer, args, device,
                     ckpt_name: str, task: str, logger=None) -> tuple[BertCls, dict]:
    """通用的 BERT 分类训练（类型分类用），避免两套几乎一样的训练代码。"""
    ds = ClsDataset(rows, tokenizer, args.cls_max_len, task=task,
                    type2id=labels_of if isinstance(labels_of, dict) else None)
    loader = DataLoader(ds, batch_size=args.cls_batch_size, shuffle=True,
                        num_workers=0, collate_fn=collate_cls)
    model = BertCls(str(args.bert_path), num_labels, args.dropout).to(device)
    opt = AdamW(model.parameters(), lr=args.cls_lr, weight_decay=0.01)
    total = max(1, len(loader) * args.cls_epochs)
    sched = get_linear_schedule_with_warmup(opt, int(total * 0.1), total)

    t0 = time.perf_counter()
    for epoch in range(1, args.cls_epochs + 1):
        model.train()
        tot, n = 0.0, 0
        for step, batch in enumerate(loader):
            ids = batch["input_ids"].to(device)
            mask = batch["attention_mask"].to(device)
            tti = batch["token_type_ids"].to(device)
            lab = batch["labels"].to(device)
            _, loss = model(ids, mask, tti, lab)
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            tot += float(loss.item())
            n += 1
            if logger is not None:
                logger.log_step(epoch=epoch, step=step, loss=float(loss.item()),
                                lr=sched.get_last_lr()[0], extra={"stage": task})
        print(f"  [{ckpt_name}] epoch {epoch}/{args.cls_epochs} loss={tot / max(n, 1):.4f}",
              flush=True)
    train_s = time.perf_counter() - t0
    CKPT_DIR.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": model.state_dict(), "args": vars(args)}, CKPT_DIR / ckpt_name)
    return model, {"train_s": round(train_s, 1), "epochs": args.cls_epochs}

@torch.no_grad()
def predict_classifier(model, rows, tokenizer, args, device, task="type",
                       type2id=None, batch_size=64) -> list[int]:
    """批量推理分类模型，返回每行的预测 label id。"""
    ds = ClsDataset(rows, tokenizer, args.cls_max_len, task=task, type2id=type2id or {})
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False,
                        num_workers=0, collate_fn=collate_cls)
    model.eval()
    out = []
    for batch in loader:
        logits, _ = model(batch["input_ids"].to(device),
                          batch["attention_mask"].to(device),
                          batch["token_type_ids"].to(device))
        out.extend(logits.argmax(-1).tolist())
    return out

# ─────────────────────────── (b) 类型条件触发词抽取 ───────────────────────────

def train_arg_experts(extract_rows, schema, tokenizer, args, device,
                      logger=None) -> dict:
    """
    为每个事件类型训一个「该类型的触发词专家」序列标注模型。

    训练数据：只用该类型的句子（最少的「供应链攻击」只有 23 条）
    标签空间：TRG + roles(t)；换装 CCED 后 roles 为空 → 每类只剩 O/B-TRG/I-TRG 3 个标签
    关键：9 个模型是**依次训练**的，每个训完就释放显存，
         否则 9 × BERT 同时驻留显存，8GB 显卡直接爆。
    """
    by_type = defaultdict(list)
    for r in extract_rows:
        ev = r.get("event")
        if ev:
            by_type[ev["type"]].append(r)

    infos = {}
    t_all = time.perf_counter()
    for etype in sorted(schema):
        rows = by_type.get(etype, [])
        if not rows:
            print(f"  [arg:{etype}] 没有训练样本，跳过")
            continue
        scheme = build_typed_schema(schema, etype)
        ds = BioDataset(rows, tokenizer, scheme, args.max_len, typed=True)
        loader = DataLoader(ds, batch_size=args.arg_batch_size, shuffle=True,
                            num_workers=0, collate_fn=collate_bio)
        encoder_path = str(args.bert_path)

        model = build_arg_model(encoder_path, scheme, args).to(device)
        opt = AdamW(model.parameters(), lr=args.arg_lr, weight_decay=0.01)
        total = max(1, len(loader) * args.arg_epochs)
        sched = get_linear_schedule_with_warmup(opt, int(total * 0.1), total)

        t0 = time.perf_counter()
        for epoch in range(1, args.arg_epochs + 1):
            model.train()
            tot, n = 0.0, 0
            for step, batch in enumerate(loader):
                inp = batch["input_ids"].to(device)
                msk = batch["attention_mask"].to(device)
                tti = batch["token_type_ids"].to(device)
                lab = batch["labels"].to(device)
                _, loss = model(inp, msk, lab, tti)
                opt.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
                sched.step()
                tot += float(loss.item())
                n += 1
                if logger is not None:
                    logger.log_step(epoch=epoch, step=step, loss=float(loss.item()),
                                    lr=sched.get_last_lr()[0],
                                    extra={"stage": f"arg:{etype}"})
            if epoch == args.arg_epochs:
                print(f"  [arg:{etype}] {len(rows)} 句 × {args.arg_epochs} epoch "
                      f"loss={tot / max(n, 1):.4f}", flush=True)
        torch.save({"state_dict": model.state_dict(),
                    "labels": scheme["labels"], "etype": etype,
                    "args": vars(args)}, CKPT_DIR / f"pipe_arg_{etype}.pt")
        infos[etype] = {"n_rows": len(rows), "num_labels": len(scheme["labels"]),
                        "train_s": round(time.perf_counter() - t0, 1)}
        # 立刻释放：8GB 卡上绝不能同时留 9 个 BERT
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    infos["_total_train_s"] = round(time.perf_counter() - t_all, 1)
    return infos

# ─────────────────────────── 组合：类型 → 触发词专家 ───────────────────────────

def compose(dev_extract_rows, tokenizer, args, device,
            oracle_type: bool = False) -> tuple[list[dict], dict]:
    """
    把两段串起来，对 dev_extract 出端到端预测。

    oracle_type=True 时用**金标类型**代替类型分类器的输出 —— 这不是作弊，
    而是标准的「上界分析」：如果类型给对能到多少，就能判断端到端的主要损失
    是来自「类型判错」还是「触发词边界抽不准」，PPT 上定位瓶颈很有用。
    """
    from models import BertEncoder, build_arg_model, predict_char_tags

    schema = load_schema()
    enums = load_enums()

    # (a) 类型 9 分类
    types_sorted = sorted(schema)
    type2id = {t: i for i, t in enumerate(types_sorted)}
    id2type = {i: t for t, i in type2id.items()}
    type_model = BertCls(str(args.bert_path), len(types_sorted), args.dropout).to(device)
    ty_ckpt = torch.load(CKPT_DIR / "pipe_type.pt", map_location=device, weights_only=False)
    type_model.load_state_dict(ty_ckpt["state_dict"])
    type_pred = predict_classifier(type_model, dev_extract_rows, tokenizer, args,
                                   device, task="type", type2id=type2id)
    del type_model
    if device.type == "cuda":
        torch.cuda.empty_cache()

    # 决定每句用哪个类型
    final_types = []
    for row, tp in zip(dev_extract_rows, type_pred):
        if oracle_type:
            ev = row.get("event") or {}
            final_types.append(ev.get("type", id2type[tp]))
        else:
            final_types.append(id2type[tp])

    # (b) 按「预测类型」分组，每组的专家模型只加载一次
    preds: list[dict] = [None] * len(dev_extract_rows)
    idx_by_type = defaultdict(list)
    for i, t in enumerate(final_types):
        idx_by_type[t].append(i)

    n_arg_trunc = 0
    for etype, idxs in idx_by_type.items():
        ckpt_path = CKPT_DIR / f"pipe_arg_{etype}.pt"
        if not ckpt_path.exists():
            for i in idxs:
                preds[i] = make_pred_row(dev_extract_rows[i], [])
            continue
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        scheme = {"labels": ckpt["labels"],
                  "label2id": {l: k for k, l in enumerate(ckpt["labels"])},
                  "id2label": {k: l for k, l in enumerate(ckpt["labels"])}}
        model = build_arg_model(str(args.bert_path), scheme,
                                _layers_of(ckpt, args.dropout)).to(device)
        model.load_state_dict(ckpt["state_dict"])
        rows = [dev_extract_rows[i] for i in idxs]
        ds = BioDataset(rows, tokenizer, scheme, args.max_len, typed=True)
        n_arg_trunc += ds.n_truncated
        loader = DataLoader(ds, batch_size=args.arg_batch_size, shuffle=False,
                            num_workers=0, collate_fn=collate_bio)
        # 注意 predict_char_tags（train_bio 里）返回 (tags, info)，这里要用解包拿到 tags
        tags, _ = predict_char_tags(model, loader, scheme["id2label"], device)
        for i, row, tg in zip(idxs, rows, tags):
            ev = spans_to_event(decode_bio_spans(tg), row["text"], typed_type=etype,
                                enums=enums)
            preds[i] = make_pred_row(row, [ev] if ev else [])
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    # 保险：任何没被填上的行都写成空预测，绝不丢行（丢行会让评测分母变小、F1 虚高）
    for i, p in enumerate(preds):
        if p is None:
            preds[i] = make_pred_row(dev_extract_rows[i], [])

    # 统计两段各错多少，便于定位瓶颈
    n_type_wrong = sum(1 for row, t in zip(dev_extract_rows, final_types)
                       if t != (row.get("event") or {}).get("type"))
    diag = {
        "n_type_wrong": n_type_wrong,
        "type_accuracy_on_dev_extract": round(
            1 - n_type_wrong / max(len(dev_extract_rows), 1), 4),
        "n_arg_truncated": n_arg_trunc,
    }
    return preds, diag

# ─────────────────────────── 主流程 ───────────────────────────

def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    data_dir = (Path(args.data_dir) if args.data_dir
                else (MOCK_DIR if args.mock else DATA_DIR))
    suffix = "_mock" if args.mock else ""
    print(f"设备：{device} | 数据：{data_dir}")

    schema = load_schema(data_dir / "event_schema.json")
    types_sorted = sorted(schema)
    type2id = {t: i for i, t in enumerate(types_sorted)}
    tokenizer = BertTokenizerFast.from_pretrained(str(args.bert_path))

    train_extract = load_extract_split("train", data_dir)
    dev_extract = load_extract_split("dev", data_dir)
    print(f"数据：train_extract {len(train_extract)} / dev_extract {len(dev_extract)}")

    # 两段式是 10 个模型（1 类型分类 + 9 触发词专家）的组合。
    # 日志按「阶段」记录，学生能看出训练时间花在哪一段。
    logger = TrainLogger(method="bert_pipeline", args=args, root=ROOT,
                         notes="两段式：9 类型分类 + 9 个类型条件触发词专家")

    t_all = time.perf_counter()

    # --only_experts：只重训 9 个触发词专家、复用已训好的类型分类器。
    # 用途：改了专家侧标签/超参时没必要重训类型分类（省一半时间）。
    if args.only_experts:
        print("  [only_experts] 跳过类型分类训练，复用现有 checkpoint")

    # ── (a) 类型 9 分类 ──
    if not args.skip_train and not args.only_experts:
        print("\n(a) 训练事件类型 9 分类器...")
        train_classifier(train_extract, type2id, len(types_sorted), tokenizer, args, device,
                         "pipe_type.pt", task="type", logger=logger)

    # ── (b) 9 个类型条件触发词专家 ──
    arg_infos = {}
    _t_experts = time.perf_counter()
    if not args.skip_train:
        print("\n(b) 训练 9 个类型条件触发词专家（依次训练，训完即释放显存）...")
        arg_infos = train_arg_experts(train_extract, schema, tokenizer, args, device,
                                      logger=logger)
    else:
        print("  [skip_train] 跳过专家训练，直接用已有 checkpoint 组合评测")
    logger.log_epoch(epoch=args.arg_epochs, train_loss=0.0,
                     elapsed=time.perf_counter() - _t_experts,
                     val_metrics={k: v.get("train_s") for k, v in arg_infos.items()
                                  if isinstance(v, dict)},
                     extra={"stage": "9_trigger_experts"})

    # ── 组合评测 ──
    print("\n组合两段 → dev_extract 端到端评测...")
    with Timer() as timer:
        preds, diag = compose(dev_extract, tokenizer, args, device)
    lat = timer.summary(len(dev_extract))
    print(f"   诊断：类型判错 {diag['n_type_wrong']} 句 / "
          f"dev_extract 上类型准确率 {diag['type_accuracy_on_dev_extract']:.4f}")

    meta = {
        "display_name": "两段式流水线（类型→类型条件触发词）",
        "train_s": round(time.perf_counter() - t_all, 1),
        "ms_per_sentence": lat["ms_per_sentence"],
        "peak_gpu_mb": lat["peak_gpu_mb"],
        "max_len": args.max_len, "arg_epochs": args.arg_epochs,
        "pipeline_stages": 2,
        "num_models": 1 + len(arg_infos),
        "diag": diag,
        "arg_experts": arg_infos,
        "note": "两段各跑一次；延迟约是单模型的 2 倍（见 meta.ms_per_sentence）",
    }
    res = evaluate_predictions(f"pipeline{suffix}", preds, meta=meta,
                               gold_path=data_dir / "dev_extract.jsonl", with_detect=False)

    # ── 上界分析：类型给金标 ──
    print("\n上界分析：类型用金标（隔离出「类型判错」造成的损失）...")
    preds_o, diag_o = compose(dev_extract, tokenizer, args, device,
                              oracle_type=True)
    meta_o = dict(meta)
    meta_o["display_name"] = "两段式（类型给金标的上界）"
    meta_o["note"] = "类型用金标，只保留触发词专家一段的错误 → 用来定位瓶颈在类型还是触发词边界"
    meta_o["diag"] = diag_o
    res_o = evaluate_predictions(f"pipeline_oracle_type{suffix}", preds_o, meta=meta_o,
                                 gold_path=data_dir / "dev_extract.jsonl")

    print("\n" + "=" * 70)
    print(f"{'方法':<26}{'trigger F1':>12}{'event F1':>12}")
    print("-" * 70)
    for tag, r in (("流水线（端到端）", res), ("流水线（类型金标上界）", res_o)):
        print(f"{tag:<26}{r['trigger']['f1']:>12.4f}{r['event_exact']['f1']:>12.4f}")
    print("=" * 70)
    print(f"总耗时 {meta['train_s']:.0f}s，延迟 {lat['ms_per_sentence']} ms/句")
    logger.finalize(total_s=meta["train_s"], diag=diag,
                    dev_metrics={k: res[k]["f1"] for k in ("trigger", "event_exact")},
                    inference_latency_ms=lat["ms_per_sentence"], peak_gpu_mb=peak_gpu_mb(),
                    pipeline_stages=2, num_models=meta.get("num_models"))

def parse_args():
    ap = argparse.ArgumentParser(description="两段式流水线事件检测（类型分类 → 触发词专家）")
    ap.add_argument("--bert_path", default="")
    ap.add_argument("--cls_epochs", type=int, default=2)
    ap.add_argument("--cls_lr", type=float, default=2e-5)
    ap.add_argument("--cls_batch_size", type=int, default=64)
    ap.add_argument("--cls_max_len", type=int, default=192)
    ap.add_argument("--arg_epochs", type=int, default=3)
    ap.add_argument("--arg_lr", type=float, default=3e-5)
    ap.add_argument("--arg_batch_size", type=int, default=32)
    ap.add_argument("--max_len", type=int, default=256)
    ap.add_argument("--arg_layers", type=int, default=4,
                    help="触发词专家保留几层 BERT（默认 4：9 个模型省磁盘、快 3 倍）")
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--device", default="", help="cuda / cpu；留空自动选（mock 冒烟建议 cpu，别抢卡）")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--skip_train", action="store_true")
    ap.add_argument("--log_every", type=int, default=50,
                    help="每多少步往 steps.jsonl 写一行（供学生画 loss 曲线）")
    ap.add_argument("--only_experts", action="store_true",
                    help="只重训 9 个触发词专家，复用已有 type 分类器")
    ap.add_argument("--mock", action="store_true", help="用 data/_mock 的 20 条手造数据（离线冒烟）")
    ap.add_argument("--data_dir", default="", help="数据目录；留空=全量 data/processed（--mock 时用 data/_mock）")
    args = ap.parse_args()
    if not args.bert_path:
        args.bert_path = str(resolve_model_path(BERT_MODEL_DIR))
    return args

if __name__ == "__main__":
    main()
