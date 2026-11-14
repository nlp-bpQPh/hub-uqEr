# -*- coding: utf-8 -*-
"""
第三步：在测试集上评估模型。

产出：
  outputs/metrics.json            总体指标 + 每种语言的 P/R/F1
  outputs/classification_report.txt  完整分类报告
  outputs/confusion_matrix.png    混淆矩阵热力图
  outputs/errors.csv              分错的样本（便于分析）

用法：python src/evaluate.py
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (accuracy_score, classification_report, confusion_matrix,
                             f1_score, precision_recall_fscore_support)
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, BertForSequenceClassification

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # 项目根目录（config.py）
_SRC = Path(__file__).resolve().parent
sys.path.insert(0, str(_SRC))  # 同目录（train.py）
import config
from train import LangDataset, evaluate, make_collate, pick_device

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass


def setup_cjk_font():
    """让 matplotlib 能正常显示中文标签。"""
    import matplotlib
    matplotlib.use("Agg")
    from matplotlib import font_manager
    for name in ("Microsoft YaHei", "SimHei", "SimSun", "Noto Sans CJK SC"):
        try:
            font_manager.findfont(name, fallback_to_default=False)
            matplotlib.rcParams["font.sans-serif"] = [name]
            matplotlib.rcParams["axes.unicode_minus"] = False
            return True
        except Exception:
            continue
    return False


def main():
    parser = argparse.ArgumentParser(description="在测试集上评估 12 国语言识别模型")
    parser.add_argument("--model_dir", default=str(config.SAVE_DIR))
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--max_len", type=int, default=config.MAX_LEN)
    parser.add_argument("--split", default="test", choices=["test", "dev"])
    args = parser.parse_args()

    device = pick_device()
    use_amp = device.type == "cuda"
    model_dir = Path(args.model_dir)
    if not (model_dir / "config.json").exists():
        raise SystemExit(f"找不到模型：{model_dir}\n请先运行 python src/train.py")

    print(f"加载模型：{model_dir}")
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = BertForSequenceClassification.from_pretrained(model_dir).to(device)

    df = pd.read_csv(config.DATA_PROCESSED / f"{args.split}.csv", encoding="utf-8-sig")
    ds = LangDataset(df, tokenizer, args.max_len)
    loader = DataLoader(
        ds, batch_size=args.batch_size, shuffle=False,
        collate_fn=make_collate(tokenizer.pad_token_id or 0),
        num_workers=0, pin_memory=(device.type == "cuda"),
    )

    metrics, preds, golds = evaluate(model, loader, device, use_amp)
    golds, preds = np.array(golds), np.array(preds)

    order = list(range(config.NUM_LABELS))
    zh_names = [config.LABEL_NAMES_ZH[c] for c in config.LABELS]

    acc = accuracy_score(golds, preds)
    macro_f1 = f1_score(golds, preds, average="macro")
    weighted_f1 = f1_score(golds, preds, average="weighted")
    p_macro, r_macro, _, _ = precision_recall_fscore_support(golds, preds, average="macro", zero_division=0)

    print("\n" + "=" * 62)
    print(f"测试集 {len(golds)} 条")
    print(f"准确率 Accuracy       : {acc:.4f}")
    print(f"宏平均 Macro-F1       : {macro_f1:.4f}")
    print(f"加权平均 Weighted-F1  : {weighted_f1:.4f}")
    print(f"宏平均 Precision      : {p_macro:.4f}")
    print(f"宏平均 Recall         : {r_macro:.4f}")
    print(f"测试集 Loss           : {metrics['loss']:.4f}")
    print("=" * 62)

    report_txt = classification_report(
        golds, preds, labels=order, target_names=zh_names, digits=4, zero_division=0
    )
    print(report_txt)

    # ---------- 混淆矩阵 ----------
    cm = confusion_matrix(golds, preds, labels=order)
    cm_norm = cm.astype(float) / np.maximum(cm.sum(axis=1, keepdims=True), 1)
    has_cjk = setup_cjk_font()
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(9.2, 7.6))
    im = ax.imshow(cm_norm, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(len(zh_names))); ax.set_yticks(range(len(zh_names)))
    ax.set_xticklabels(zh_names if has_cjk else config.LABELS, rotation=40, ha="right", fontsize=9)
    ax.set_yticklabels(zh_names if has_cjk else config.LABELS, fontsize=9)
    ax.set_xlabel("预测标签"); ax.set_ylabel("真实标签")
    ax.set_title(f"12 国语言识别混淆矩阵（行归一化，准确率 {acc:.2%}）")
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            if cm[i, j]:
                ax.text(j, i, f"{cm[i, j]}", ha="center", va="center", fontsize=8,
                        color="white" if cm_norm[i, j] > 0.5 else "#333333")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    cm_path = config.OUTPUT_DIR / "confusion_matrix.png"
    fig.savefig(cm_path, dpi=160)
    plt.close(fig)
    print(f"[写出] {cm_path}")

    # ---------- 每种语言 ----------
    p, r, f, s = precision_recall_fscore_support(golds, preds, labels=order, zero_division=0)
    per_lang = {
        code: {"name_zh": config.LABEL_NAMES_ZH[code], "precision": round(float(p[i]), 4),
               "recall": round(float(r[i]), 4), "f1": round(float(f[i]), 4), "support": int(s[i])}
        for i, code in enumerate(config.LABELS)
    }

    # ---------- 最易混淆的语言对 ----------
    pairs = []
    for i in range(len(order)):
        for j in range(len(order)):
            if i != j and cm[i, j] > 0:
                pairs.append({"true": config.LABELS[i], "pred": config.LABELS[j], "count": int(cm[i, j])})
    pairs.sort(key=lambda x: -x["count"])

    # ---------- 错误样例 ----------
    err = df.copy()
    err["pred_id"] = preds
    err["pred"] = [config.LABELS[i] for i in preds]
    err["pred_zh"] = err["pred"].map(config.LABEL_NAMES_ZH)
    err["pred_prob"] = np.nan
    err = err[err["label_id"] != err["pred_id"]]
    err[["text", "label", "label_zh", "pred", "pred_zh"]].to_csv(
        config.OUTPUT_DIR / "errors.csv", index=False, encoding="utf-8-sig", quoting=1
    )

    result = {
        "split": args.split,
        "n_samples": int(len(golds)),
        "accuracy": round(float(acc), 6),
        "macro_f1": round(float(macro_f1), 6),
        "weighted_f1": round(float(weighted_f1), 6),
        "macro_precision": round(float(p_macro), 6),
        "macro_recall": round(float(r_macro), 6),
        "loss": round(float(metrics["loss"]), 6),
        "per_language": per_lang,
        "confusion_pairs": pairs[:10],
        "n_errors": int(len(err)),
    }
    (config.OUTPUT_DIR / "metrics.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (config.OUTPUT_DIR / "classification_report.txt").write_text(
        f"测试集 {len(golds)} 条 | Accuracy={acc:.4f} | Macro-F1={macro_f1:.4f}\n\n" + report_txt,
        encoding="utf-8",
    )

    if pairs:
        print("最容易混淆的语言对（真实 -> 预测）：")
        for item in pairs[:5]:
            print(f"  {config.LABEL_NAMES_ZH[item['true']]:<5} -> "
                  f"{config.LABEL_NAMES_ZH[item['pred']]:<5} {item['count']} 条")
    print(f"\n共错分 {len(err)} 条，明细见 {config.OUTPUT_DIR / 'errors.csv'}")
    print(f"指标已保存：{config.OUTPUT_DIR / 'metrics.json'}")
    print("下一步：python src/predict.py \"要识别的文本\"")


if __name__ == "__main__":
    main()
