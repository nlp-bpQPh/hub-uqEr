"""CNews 规则与多数类基线，复用课程 rule 分类思路。"""

from collections import Counter


def majority_predict(train, n):
    """始终预测训练集中最多的类别，作为简单基线。"""
    majority = Counter(row["label"] for row in train).most_common(1)[0][0]
    return [majority] * n


def classify(text, labels, fallback):
    """命中类别名称时输出对应类，否则退化为训练集多数类。"""
    return next((label for label in labels if label in text), fallback)


def predict(train, texts, labels):
    fallback = Counter(row["label"] for row in train).most_common(1)[0][0]
    return [classify(text, labels, fallback) for text in texts]
