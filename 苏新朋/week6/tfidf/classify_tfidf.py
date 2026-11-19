"""CNews 字符级 TF-IDF + LinearSVC，沿用课程 TF-IDF 示例。"""

import time

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC

SEED = 42


def predict(train, test, ngram_range=(1, 2), c=1.0):
    """在训练数据拟合向量器与线性 SVM，并返回预测及耗时。"""
    vectorizer = TfidfVectorizer(
        analyzer="char",
        ngram_range=ngram_range,
        max_features=300000,
        sublinear_tf=True,
    )
    start = time.perf_counter()
    x_train = vectorizer.fit_transform([row["text"] for row in train])
    x_test = vectorizer.transform([row["text"] for row in test])
    model = LinearSVC(C=c, random_state=SEED)
    model.fit(x_train, [row["label"] for row in train])
    predictions = model.predict(x_test).tolist()
    return predictions, round(time.perf_counter() - start, 2)
