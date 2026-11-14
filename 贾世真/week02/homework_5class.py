# coding:utf8
"""
第二周作业：五维向量多分类任务
任务定义：对于一个 5 维向量，它的最大值在第几维，它就属于第几类（0~4，共 5 类）

对应课程知识点：
- 模型 = 带参数的公式        -> nn.Linear(5, 5)
- 前向传播                  -> 算出 5 个类别的打分(logits)
- 损失函数                  -> 交叉熵 CrossEntropyLoss（内部已含 Softmax）
- 反向传播 + 梯度下降        -> loss.backward() + optimizer.step()
- Epoch / Batch             -> 外层 epoch_num 轮，内层每 batch_size 个样本更新一次
"""

import torch
import torch.nn as nn
import numpy as np


class TorchModel(nn.Module):
    def __init__(self, input_size, class_num):
        super(TorchModel, self).__init__()
        # 一层线性层就够了：输入 5 维 -> 输出 5 个类别的打分
        self.linear = nn.Linear(input_size, class_num)

    def forward(self, x, y=None):
        logits = self.linear(x)          # (batch_size, 5) 未归一化的打分
        if y is None:
            return logits                # 预测：交给外面自己 argmax
        loss = nn.functional.cross_entropy(logits, y)   # 内部已做 log_softmax
        return loss


def build_sample(class_num=5):
    """生成一个样本：随机 5 维向量，最大值所在维度即类别"""
    x = np.random.random(class_num)
    y = int(np.argmax(x))
    return x, y


def build_dataset(total_sample_num, class_num=5):
    X, Y = [], []
    for _ in range(total_sample_num):
        x, y = build_sample(class_num)
        X.append(x)
        Y.append(y)
    # 交叉熵的 target 用 LongTensor，形状是 (n,)，不要写成 (n, 1)
    return torch.FloatTensor(X), torch.LongTensor(Y)


def evaluate(model, class_num=5, test_num=500):
    """评估：整体准确率 + 每个类别的召回率"""
    model.eval()
    x, y = build_dataset(test_num, class_num)
    with torch.no_grad():
        pred = torch.argmax(model(x), dim=1)
    correct = int((pred == y).sum())
    acc = correct / test_num

    per_class = {}
    for c in range(class_num):
        mask = (y == c)
        cnt = int(mask.sum())
        hit = int((pred[mask] == c).sum()) if cnt else 0
        per_class[c] = hit / cnt if cnt else 0.0
    return acc, per_class


def main():
    epoch_num = 20            # 训练轮数
    batch_size = 20           # 每批样本数
    train_sample = 1000       # 训练集样本总数
    input_size = 5            # 输入向量维度
    class_num = 5             # 类别数：最大值在第 0/1/2/3/4 维
    learning_rate = 0.01

    model = TorchModel(input_size, class_num)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    train_x, train_y = build_dataset(train_sample, class_num)
    log = []

    for epoch in range(epoch_num):
        model.train()
        watch_loss = []
        # 打乱数据，避免每轮 batch 顺序一样
        idx = torch.randperm(train_sample)
        for batch_index in range(train_sample // batch_size):
            sel = idx[batch_index * batch_size:(batch_index + 1) * batch_size]
            xb, yb = train_x[sel], train_y[sel]
            loss = model(xb, yb)        # 1. 前向 + 算 loss
            optimizer.zero_grad()       # 2. 梯度清零
            loss.backward()             # 3. 反向传播，算出所有参数的梯度
            optimizer.step()            # 4. 沿负梯度更新参数
            watch_loss.append(loss.item())

        avg_loss = float(np.mean(watch_loss))
        acc, per_class = evaluate(model, class_num)
        log.append([acc, avg_loss])
        print("第%02d轮  loss=%.4f  准确率=%.4f" % (epoch + 1, avg_loss, acc))

    print("\n各分类召回率：")
    for c in range(class_num):
        print("  第%d维最大 → %.2f%%" % (c, log[-1][0] and per_class[c] * 100))

    torch.save(model.state_dict(), "homework_model.bin")
    print("\n模型已保存为 homework_model.bin")

    # 手工构造几个样本验证一下
    test_vec = [
        [0.9, 0.1, 0.2, 0.3, 0.4],   # 第 0 维最大
        [0.1, 0.9, 0.2, 0.3, 0.4],   # 第 1 维最大
        [0.1, 0.2, 0.9, 0.3, 0.4],   # 第 2 维最大
        [0.1, 0.2, 0.3, 0.9, 0.4],   # 第 3 维最大
        [0.1, 0.2, 0.3, 0.4, 0.9],   # 第 4 维最大
    ]
    model.eval()
    with torch.no_grad():
        out = model(torch.FloatTensor(test_vec))
        pred = torch.argmax(out, dim=1)
    print("\n手工样本验证：")
    for v, p in zip(test_vec, pred):
        print("  输入 %s → 预测第 %d 类（实际最大值在第 %d 维）" % (v, p, int(np.argmax(v))))


if __name__ == "__main__":
    main()
