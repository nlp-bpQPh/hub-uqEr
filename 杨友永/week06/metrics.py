"""评估指标：混淆矩阵、macro-F1、分类报告。不依赖 sklearn。"""
from __future__ import annotations

import numpy as np


def confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, k: int) -> np.ndarray:
    cm = np.zeros((k, k), dtype=np.int64)
    for t, p in zip(y_true, y_pred):
        cm[int(t), int(p)] += 1
    return cm


def accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float((y_true == y_pred).mean())


def per_class_prf(cm: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """返回每类的 (precision, recall, f1)。"""
    tp = np.diag(cm).astype(float)
    pred_sum = cm.sum(axis=0).astype(float)      # 预测为该类的总数
    true_sum = cm.sum(axis=1).astype(float)      # 真实为该类的总数
    precision = np.divide(tp, pred_sum, out=np.zeros_like(tp), where=pred_sum > 0)
    recall = np.divide(tp, true_sum, out=np.zeros_like(tp), where=true_sum > 0)
    f1 = np.divide(2 * precision * recall, precision + recall,
                   out=np.zeros_like(tp), where=(precision + recall) > 0)
    return precision, recall, f1


def macro_f1(y_true: np.ndarray, y_pred: np.ndarray, k: int) -> float:
    _, _, f1 = per_class_prf(confusion_matrix(y_true, y_pred, k))
    return float(f1.mean())


def weighted_f1(y_true: np.ndarray, y_pred: np.ndarray, k: int) -> float:
    cm = confusion_matrix(y_true, y_pred, k)
    _, _, f1 = per_class_prf(cm)
    support = cm.sum(axis=1).astype(float)
    total = support.sum()
    return float((f1 * support).sum() / total) if total else 0.0


def summarize(y_true, y_pred, k: int) -> dict[str, float]:
    return {
        "acc": round(accuracy(y_true, y_pred), 4),
        "macro_f1": round(macro_f1(y_true, y_pred, k), 4),
        "weighted_f1": round(weighted_f1(y_true, y_pred, k), 4),
    }


def print_report(y_true, y_pred, id2name: dict[int, str]) -> None:
    k = len(id2name)
    cm = confusion_matrix(y_true, y_pred, k)
    p, r, f = per_class_prf(cm)
    print(f"{'类别':<14}{'precision':>10}{'recall':>9}{'f1':>8}{'support':>9}")
    for i in range(k):
        print(f"{id2name[i]:<14}{p[i]:>10.3f}{r[i]:>9.3f}{f[i]:>8.3f}{cm[i].sum():>9}")
    print(f"{'macro':<14}{'':>10}{'':>9}{f.mean():>8.3f}{cm.sum():>9}")


def top_confusions(y_true, y_pred, id2name: dict[int, str], n: int = 5) -> list[str]:
    """最容易混淆的类别对。"""
    cm = confusion_matrix(y_true, y_pred, len(id2name))
    pairs = []
    for i in range(len(id2name)):
        for j in range(len(id2name)):
            if i != j and cm[i, j] > 0:
                pairs.append((cm[i, j], id2name[i], id2name[j]))
    pairs.sort(reverse=True)
    return [f"{a} → {b}  {c} 次" for c, a, b in pairs[:n]]
