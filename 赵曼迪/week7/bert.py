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
import random
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import BertConfig, BertModel, BertTokenizer, AutoTokenizer,get_linear_schedule_with_warmup

ROOT = Path(__file__).resolve().parents[0]  # 项目根目录（方法目录的上一级）
sys.path.insert(0, str(ROOT / "common"))     # 公共模块：event_data / crf / eval_utils / models / log_utils

from common.crf import CRFClassifier
from common.eval_utils import CKPT_DIR, LOG_DIR, Timer, evaluate_predictions
from common.log_utils import TrainLogger, peak_gpu_mb
from common.event_data import (BioDataset, DATA_DIR, ROOT, build_bio_constraint_mask, collate_bio,
                        build_joint_schema, build_start_constraint_mask, decode_bio_spans,
                        illegal_bio_stats, load_enums, load_extract_split, load_schema,
                        make_pred_row, row_event, spans_to_event)
from common.model_paths import BERT_MODEL_DIR, resolve_model_path

from common.models import BertEncoder, batch_to_device, build_model, predict_char_tags

from common.steps_loss import plot_steps_jsonl

MOCK_DIR = ROOT / "data" / "_mock"
IGNORE_INDEX = -100

# ─────────────────────────── 模型 ───────────────────────────

# ─────────────────────────── 训练 / 评估 ───────────────────────────

def set_seed(seed: int) -> None:
    random.seed(seed)  # 固定python内置random
    torch.manual_seed(seed)  # 固定CPU上torch随机数
    torch.cuda.manual_seed_all(seed)  # 固定所有GPU的torch随机数

def span_f1_from_char_tags(rows: list[dict], char_tags: list[list[str]]) -> float:
    """
    训练中做模型选择用的轻量指标：**span 级精确匹配 F1**（角色+文本完全一致），
    口径和最终评测的 argument F1 接近，但不依赖 930 条 dev、也更快。
    只对标注里的实体 / 论元 / 触发词做预测，精确率p/召回率r以整个span作为一个指标，不是单个字符一个指标计算，如果一个span中只有一个字符预测错了，哪也算整个span错误
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
            _, loss = model(batch["input_ids"],
                batch["attention_mask"],
                batch["labels"],
                batch.get("token_type_ids"),
                valid_tag_mask=batch.get("valid_tag_mask"))
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
    for step, batch in enumerate(loader):
        batch = batch_to_device(batch, device)
        _, loss = model(batch["input_ids"],
                batch["attention_mask"],
                batch["labels"],
                batch.get("token_type_ids"),
                valid_tag_mask=batch.get("valid_tag_mask")) ##传参见CRFClassifier函数
        (loss / args.grad_accum).backward()
        total += float(loss.item())
        n += 1
        if (step + 1) % args.grad_accum == 0:
            # 梯度裁剪：CRF 在序列开头几步容易出大梯度，不裁剪偶发 loss 爆掉
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=args.max_grad_norm) #拿到模型所有可训练参数，先算：所有参数梯度的二范数，把所有梯度拼成一个大向量，求它的 L2 范数
                                                                                      #如果这个范数 ≤ max_grad_norm：什么都不做，梯度原样保留，反之等比例缩放全部梯度，让最终范数刚好等于 max_norm
            optimizer.step()  #更新模型权重
            scheduler.step()  #调整学习率
            optimizer.zero_grad()
        if logger is not None:
            logger.log_step(epoch=epoch, step=step, loss=float(loss.item()),
                            lr=scheduler.get_last_lr()[0])  #logger.log_step会往 log.txt（TrainLogger 内部封装死的，文本流水账，所有信息拼成字符串，随时打开看） 追加内容
                                                            #存入tensorboard events 二进制文件，启动 tensorboard 命令才能可视化曲线：loss 下降曲线、lr 变化曲线。
                                                            #`scheduler.get_last_lr()` 返回一个列表，存当前优化器分组的学习率，大部分情况只有一组 lr，`[0]` 取第一个，拿到当前真实的学习率
        if (step + 1) % 50 == 0:
            print(f"  epoch {epoch} step {step + 1}/{len(loader)} loss={total / n:.4f}", flush=True)
    if len(loader) % args.grad_accum != 0:
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=args.max_grad_norm)
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()
    return total / max(n, 1)

def schema_stats_str(schema: dict) -> str:
    roles = {r for v in schema.values() for r in v}
    return f"{len(schema)} 类 / {sum(len(v) for v in schema.values())} 槽位 / {len(roles)} 去重角色"

# ─────────────────────────── 主流程 ───────────────────────────

def main():
    args = parse_args()
    set_seed(args.seed)
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    print(f"设备：{device}")
    if device.type == "cuda":
        print(f"显存占用（启动时）：{torch.cuda.memory_allocated() / 1024 ** 2:.0f} MiB")

    data_dir = (Path(args.data_dir) if args.data_dir
                else (MOCK_DIR if args.mock else DATA_DIR))  #找到数据文件夹:D:\AI\week7\data\processed
    schema = load_schema(data_dir / "event_schema.json")  #去掉event_schema文件中以下划线开头的键，把键值转换位列表，读取成python字典
    enums = load_enums(data_dir / "event_schema.json")  #专门取出 __enums__ 这一块：{"公司上市.环节": ["筹备上市", "暂停上市", "正式上市", "终止上市"]}
    label_scheme = build_joint_schema(schema)  #生成标签名：每个标签拆成 B-xxx、I-xxx，再加 O（非实体）；标签对应的数字等
    labels = label_scheme["labels"]
    id2label = label_scheme["id2label"]  #数字对应的标签
    print(f"标签体系：joint，{len(labels)} 个标签（{schema_stats_str(schema)}）")

    tokenizer = AutoTokenizer.from_pretrained(str(args.bert_path), use_fast=True)  #args.bert_path：模型路径，from_pretrained(路径)：从本地模型目录加载分词器
    #原BertAutoTokenizerTokenizer改为AutoTokenizer，支持后续.word_ids()
    train_rows = load_extract_split("train", data_dir)  #读取训练数据
    dev_rows = load_extract_split("dev", data_dir)  #读取验证数据

    # 内层验证集：从 train 切 10%，只用于选 epoch；dev 完全不参与训练/选点
    rng = random.Random(args.seed)  #单独创建一个固定随机种子的随机器
    idxs = list(range(len(train_rows)))  #生成下标列表：[0,1,2,3,...总训练样本数-1]
    rng.shuffle(idxs)  #把下标打乱顺序，随机抽样
    n_inner = max(1, int(len(train_rows) * args.inner_val)) #计算要切出来多少条做内层验证；max(1, ...)兜底，防止数据集特别小的时候切出来 0 条
    inner_idx = set(idxs[:n_inner])  #取打乱之后前 n_inner 个下标，放进集合，方便快速判断
    fit_rows = [r for i, r in enumerate(train_rows) if i not in inner_idx]  #真正用来更新参数的训练子集（去掉被切走的那 10%）
    val_rows = [r for i, r in enumerate(train_rows) if i in inner_idx]  #内层验证集，从原 train 切出来的，训练过程每轮在这上面算指标，用来监控、保存最优模型、早停
    print(f"训练 {len(fit_rows)} 句 / 内层验证 {len(val_rows)} 句 / 评测 dev {len(dev_rows)} 句")

    fit_ds = BioDataset(fit_rows, tokenizer, label_scheme, args.max_len)  #生成输入样本：input_ids、attention_mask、word_ids（每一个 token，对应输入的第几个 word，且[CLS]/[SEP]/[PAD]等特殊位置转成-1）、
                                                                          #token_type_ids、labels（和输入样本对其token的输出样本）、row_idx（样本在原始 json 列表里的下标）
    val_ds = BioDataset(val_rows, tokenizer, label_scheme, args.max_len)
    dev_ds = BioDataset(dev_rows, tokenizer, label_scheme, args.max_len)
    fit_loader = DataLoader(fit_ds, batch_size=args.batch_size, shuffle=True, num_workers=0, collate_fn=collate_bio)
    #shuffle=True 是在组装 batch 之前，先把整个数据集的样本顺序打乱，再按顺序切出一个个 batch
    #num_workers=0：只用主进程加载；Windows 环境、debug 阶段经常设 0，避免多进程报错、死锁
    # collate_fn=collate_bio：做padding 补齐到这个 batch 内最长长度（默认 collate：要求所有样本的 tensor shape 完全一样，直接堆叠，但现在样本长度不一样，需要手动写函数补齐）
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=0, collate_fn=collate_bio)
    dev_loader = DataLoader(dev_ds, batch_size=args.batch_size, shuffle=False, num_workers=0, collate_fn=collate_bio)

    model = build_model(args, label_scheme).to(device)  
    #这里根据传参控制：use_crf=True时同时传入constraint、start_constraint，训练阶段输出logits：BERT → dropout → Linear[B, L, num_tags],同时计算loss：[B, L, num_tags] + 转移矩阵 算 CRF loss
    #                                                                   推理 / 验证阶段（predict)采用维特比解码输出有实际含义（剔除PAD、CLS等输入位置）对应token的标签ID
    #                use_crf=False时（此时是否传入constraint、start_constraint都不会生效），输出logits：BERT → dropout → Linear[B, L, num_tags],同时计算loss：[B, L, num_tags] 算交叉熵 CE loss
    n_params = sum(p.numel() for p in model.parameters())
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"模型参数：{n_params / 1e6:.1f}M（可训练 {n_trainable / 1e6:.1f}M），"
          f"解码器={args.decoder}，BERT 层数={args.bert_layers or 'full(12)'}")

    optimizer = build_optimizer(model, args)  #采用分层学习率
    #AdamW([
    #     {"params": bert_params, "lr": args.lr},
    #     {"params": head_params, "lr": args.lr * args.head_lr_mult},
    # ], weight_decay=args.weight_decay)
    total_steps = max(1, len(fit_loader) * args.epochs // max(args.grad_accum, 1))
    warmup_steps = int(total_steps * args.warmup_ratio)
    scheduler = get_linear_schedule_with_warmup(optimizer, warmup_steps, total_steps)  #前0.1，系数变化0~1，后0.9，系数变化1~0
    print(f"训练步数 {total_steps}（warmup {warmup_steps} 步，{args.warmup_ratio:.0%}）")

    CKPT_DIR.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    ckpt_path = CKPT_DIR / f"{args.run_name}.pt"

    logger = TrainLogger(method=args.run_name, args=args, root=ROOT, params=n_params,
                         notes=f"BERT 序列标注（{args.decoder}）训练过程") #调用logger，存在root/method/文件名里，method、args、params、notes写入config.json

    log, best_f1, best_epoch = [], -1.0, 0
    train_start = time.perf_counter()  #time.perf_counter()是 Python 高精度计时器

    for epoch in range(1, args.epochs + 1):
        t0 = time.perf_counter()
        train_loss = run_epoch(model, fit_loader, optimizer, scheduler, device, args, epoch,
                               logger=logger)  #返回这一组样本的平均损失值
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
                         extra={"best_so_far": round(best_f1, 6)})     #往log.txt，tensorboard events文件里顺序往下写入
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
        dev_tags, _ = predict_char_tags(model, dev_loader, id2label, device)
    dev_illegal = illegal_bio_stats(dev_tags) #返回统计非法 BIO 序列等指标

    # 训练集上的非法统计也报一份（模型在见过的数据上表现如何，便于对照）
    with Timer() as timer_fit:
        fit_tags, _ = predict_char_tags(model, fit_loader, id2label, device)
    fit_illegal = illegal_bio_stats(fit_tags)

    preds = []
    n_empty = 0
    for row, tags in zip(dev_rows, dev_tags):
        ev = spans_to_event(decode_bio_spans(tags), row["text"], enums=enums)
        if ev is None:
            n_empty += 1
        preds.append(make_pred_row(row, [ev] if ev else []))
    print(f"dev 解码：{len(preds)} 句，其中没抽到触发词 {n_empty} 句")
    # 例：row 原始样本
    # row = {
    # "id": 1,
    # "text": "9月10日北京召开会议",
    # }
    # tags 字符BIO标签（dev_tags里的一条）
    # tags = ["B-时间", "I-时间", "I-时间", "O", "B-地点", "I-地点", "O", "B-触发词", "I-触发词"]
    #decode_bio_spans(tags) 输出 span 列表：
    # [
    # {"type": "时间", "start": 0, "end": 3, "text": "9月10日"},
    # {"type": "地点", "start": 4, "end": 6, "text": "北京"},
    # {"type": "触发词", "start": 7, "end": 9, "text": "召开会议"}
    # ]
    #spans_to_event(...) 得到 ev（事件结构）
    # ev = {
    # "trigger": {"role": "__trigger__", "text": "召开会议"},
    # "args": [
    #     {"role": "时间", "text": "9月10日"},
    #     {"role": "地点", "text": "北京"}
    # ]
    # }   
    #make_pred_row(row, [ev]) 打包成一条，放进 preds
#     {
#     "id": 1,
#     "text": "9月10日北京召开会议",
#     "events": [
#         {
#             "trigger": {"text": "召开会议"},
#             "args": [
#                 {"role": "时间", "text": "9月10日"},
#                 {"role": "地点", "text": "北京"}
#             ]
#         }
#     ]
# }

    lat = timer.summary(len(dev_rows))
    meta = {
        "display_name": f"BERT-{args.bert_layers or 12}L+{args.decoder.upper()}",
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
        "illegal_dev": dev_illegal,
        "illegal_train_pred": fit_illegal,
        "n_truncated": dev_ds.n_truncated,
        "hard_constraint": bool(args.hard_constraint),
        "seed": args.seed,
        "selection_note": "按 train 切出的 10% 内层验证集选 epoch；dev 未参与训练与选点",
    }
    res = evaluate_predictions(args.run_name, preds, meta=meta,
                              gold_path=data_dir / "dev_extract.jsonl")

    print("\n【非法 BIO 统计（dev，模型原始输出）】")
    print(f"  序列数 {dev_illegal['total_seqs']}："
          f"含非法序列 {dev_illegal['illegal_seqs']} 条"
          f"（{dev_illegal['illegal_seq_rate']:.1%}），"
          f"其中非法开头 {dev_illegal['illegal_start']} 处、"
          f"非法转移 {dev_illegal['illegal_transition']} 处")
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

    # ===== 自动画图（新增）=====
    try:
        plot_steps_jsonl(logger.steps_jsonl)
    except Exception as e:
        print(f"[auto plot] 画图失败，不影响训练结果：{e}")


def parse_args():
    ap = argparse.ArgumentParser(description="BERT 序列标注做事件抽取（linear / crf）")
    ap.add_argument("--decoder", choices=["linear", "crf"], default="crf")
    ap.add_argument("--bert_path", default="")
    ap.add_argument("--bert_layers", type=int, default=0, help="0=完整 12 层；4=消融")
    ap.add_argument("--max_len", type=int, default=256)
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--head_lr_mult", type=float, default=5.0) #10
    ap.add_argument("--warmup_ratio", type=float, default=0.1)
    ap.add_argument("--grad_accum", type=int, default=1) #4
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
    if not args.bert_path:
        args.bert_path = str(resolve_model_path(BERT_MODEL_DIR))
    if not args.run_name:
        tag = f"bert{args.bert_layers}L" if args.bert_layers else "bert"
        extra = "_hard" if args.hard_constraint else ""
        args.run_name = f"{tag}_{args.decoder}_{args.max_len}{extra}" + ("_mock" if args.mock else "")
    return args

if __name__ == "__main__":
    main()



"""
线性头（不加 CRF），grad_accum=4
python bert.py  --decoder linear --grad_accum 4 --epochs 12 --mock

CRF + 硬约束（hard_constraint），grad_accum=4
python bert.py --decoder crf --hard_constraint --grad_accum 4 --epochs 12 --mock
BERT 子词拆分产生序列中间‑100 标签；原始手写 CRF 不支持，粗暴替换‑100 为 0 引入错误监督信号，导致 CRF 训练效果差；
完整修复需要新增 valid_tag_mask 标记有效监督位置，改动数据集与 dataloader 链路。现已修改
"""