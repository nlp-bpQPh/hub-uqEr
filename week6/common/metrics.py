"""
统一评测指标（全项目共用）

教学重点：
  1. 分类任务只看 Accuracy 会被长尾类骗：头部类预测对就够高，尾部类全错也看不出来。
     所以主指标用 Macro-F1（每类等权），辅以 per-class recall 和混淆矩阵。
  2. Macro-F1 和 Accuracy 的差距本身就是诊断信号：
     两者差得多 → 尾部类有问题；差不多 → 各类表现均匀。
  3. 置信度校准（ECE）用来回答「模型说 0.9 时，实际命中率是不是 0.9」——
     这个数字决定了阈值能不能用来做「低置信转人工」。

为什么单独成文件：
  四路方法必须用**同一个函数**算指标，避免「这脚本用 weighted F1、那脚本用 binary F1」
  导致数字不可比（这类错在真实项目里非常常见）。
"""

import numpy as np


def classification_report(y_true, y_pred, num_classes=15):
    """
    返回统一口径的指标字典。

    注意 f1_score 必须显式写 average：sklearn 默认是 binary（二分类），
    直接 f1_score(y, p) 在多分类上会报错或给出误导结果。
    """
    from sklearn.metrics import accuracy_score, f1_score, confusion_matrix

    y_true = np.asarray(y_true) #转为数组
    y_pred = np.asarray(y_pred)
    valid = y_pred >= 0  #对成功解析的展示TRUE,否则展示FLASE的数组                     # LLM 可能解析失败，用 -1 标记
    unparsed = int((~valid).sum()) #取反，求和，即统计解析失败的样本数

    report = {
        "n": int(len(y_true)),
        "unparsed": unparsed,
        "unparsed_rate": round(unparsed / len(y_true), 4) if len(y_true) else 0.0,
    }
    if valid.sum() == 0:
        return report

    yt, yp = y_true[valid], y_pred[valid] #只拿出**解析成功**的样本的真实标签、预测标签，用来算常规 acc、f1
    report["acc"] = round(float(accuracy_score(yt, yp)), 4) #acc：仅在成功样本上算的准确率，准确率：正确预测的样本数量/全部样本数量
    # 解析失败的样本按「答错」计入总准确率：生成式方法常靠丢弃失败样本来虚高准确率，
    # 对外的口径必须把所有样本都算进分母（张榜/简历用这个数）
    n_correct = int((yt == yp).sum()) #成功样本里预测对的数量
    report["acc_incl_unparsed"] = round(n_correct / len(y_true), 4) if len(y_true) else 0.0  #成功样本里预测对的数量/全部样本数
    report["macro_f1"] = round(float(f1_score(yt, yp, average="macro")), 4)  #macro：每个类别的 F1 单独算，直接取平均，**类别权重一样，不受样本不均衡影响**，适合零样本
    report["weighted_f1"] = round(float(f1_score(yt, yp, average="weighted")), 4) #weighted：每个类别 F1 按该类样本数量加权平均；样本多的类别话语权更大

    # 每类 recall：定位「哪一类被吃掉了」，比总分布更有用
    per_class = {}
    labels = list(range(num_classes))
    recalls = []
    for c in labels:
        mask = yt == c  #mask:真实值是否是当前标签的布尔值数组
        r = float((yp[mask] == c).mean()) if mask.sum() else None  #yp[mask] == c：真实和预测都为c;.mean():求平均，即召回率：真实和预测都为c/全部为c的样本
        recalls.append(r)
        per_class[c] = None if r is None else round(r, 4) #字典：`{类别号: 召回率}`
    report["per_class_recall"] = per_class
    report["confusion_matrix"] = confusion_matrix(yt, yp, labels=labels).tolist()
    #混淆矩阵：**行 = 真实标签 yt，列 = 预测标签 yp**
    #`labels` 规定行列顺序，和循环算召回的 `labels` 完全对齐
    # yt = np.array([0,0,1,1])
    # yp = np.array([0,1,1,1])
    # labels = [0,1]
    # cm = confusion_matrix(yt, yp, labels=labels)\
    #print(cm)
    # `.tolist()` 之后变成：[[1, 1], [0, 2]]
    # 解读：
    # 第 0 行（真实 0）：1 个预测 0，1 个预测 1
    # 第 1 行（真实 1）：0 个预测 0，2 个预测 1

    return report


def expected_calibration_error(y_true, probs, n_bins=10):
    """
    ECE：把预测概率分桶，比较「平均置信度」和「实际准确率」的差。
    值越小越可信；>0.1 基本不能拿置信度当阈值用。

    probs: shape [N, C] 的 softmax 概率
    """
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
        gap = abs(correct[mask].mean() - conf[mask].mean())
        ece += gap * mask.sum() / len(conf)
    return round(float(ece), 4)


def pred_distribution(preds, num_classes=15):
    """预测分布：LLM zero-shot 塌缩到少数几类时，一眼就能看出来"""
    counts = {}
    for p in preds:
        counts[int(p)] = counts.get(int(p), 0) + 1
    used = sum(1 for k in counts if k >= 0)
    return {"used_classes": used, "counts": counts}


def to_markdown_table(results: dict):
    """
    把一个 {方法名: 指标字典} 的字典渲染成 Markdown 表，方便直接贴进文档。
    results 的每一项至少要有 acc / macro_f1。
    """
    lines = ["| 方法 | 准确率 | Macro-F1 | 解析失败 |", "|---|---|---|---|"]
    for name, r in results.items():
        acc = r.get("acc", "-")
        f1 = r.get("macro_f1", "-")
        up = r.get("unparsed", 0)
        lines.append(f"| {name} | {acc} | {f1} | {up} |")
    return "\n".join(lines)

