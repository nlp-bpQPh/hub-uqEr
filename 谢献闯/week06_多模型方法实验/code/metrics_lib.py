"""统一指标：Accuracy / Macro-F1 / per-class recall / 混淆矩阵 / ECE。

与课程 common/metrics.py 口径一致，只额外记录「解析失败」为 0（本作业全是判别式）。
"""

import numpy as np


def classification_report(y_true, y_pred, labels=None):
    from sklearn.metrics import accuracy_score, confusion_matrix, f1_score

    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    if labels is None:
        labels = sorted(set(y_true.tolist()) | set(y_pred.tolist()))

    report = {
        "n": int(len(y_true)),
        "acc": round(float(accuracy_score(y_true, y_pred)), 4),
        "macro_f1": round(float(f1_score(y_true, y_pred, average="macro", labels=labels)), 4),
        "weighted_f1": round(float(f1_score(y_true, y_pred, average="weighted", labels=labels)), 4),
    }
    per_class = {}
    for c in labels:
        mask = y_true == c
        r = float((y_pred[mask] == c).mean()) if mask.sum() else None
        per_class[str(c)] = None if r is None else round(r, 4)
    report["per_class_recall"] = per_class
    report["confusion_matrix"] = confusion_matrix(y_true, y_pred, labels=labels).tolist()
    report["pred_distribution"] = {
        "used_classes": int(len(set(y_pred.tolist()))),
        "counts": {str(k): int(v) for k, v in zip(*np.unique(y_pred, return_counts=True))},
    }
    return report


def expected_calibration_error(y_true, probs, n_bins=10):
    y_true = np.asarray(y_true)
    probs = np.asarray(probs)
    conf = probs.max(axis=1)
    pred = probs.argmax(axis=1)
    correct = (pred == y_true).astype(float)
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        lo, hi = bins[i], bins[i + 1]
        mask = (conf > lo) & (conf <= hi)
        if mask.sum() == 0:
            continue
        ece += abs(correct[mask].mean() - conf[mask].mean()) * mask.sum() / len(conf)
    return round(float(ece), 4)
