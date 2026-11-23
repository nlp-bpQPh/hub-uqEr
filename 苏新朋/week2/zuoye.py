
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import torch
import torch.nn as nn
import numpy as np

"""
====================================================
作业要求：
  自己设计多分类任务完成模型训练，如果不知道设计什么任务，就做以下任务:
  对于任意五维向量，它最大数字在第几维就属于第几类
====================================================

任务描述：
  给模型一个由 5 个数字组成的向量，比如 [0.9, 0.1, 0.3, 0.2, 0.5]。
  这 5 个数字里，第 0 个数字 0.9 最大，所以它的「类别」就是 0；
  如果第 2 个数字最大，类别就是 2，以此类推。

  所以一共有 5 个类别：0、1、2、3、4。
  模型要做的就是：看一眼这 5 个数字，猜出最大的那个在第几位。

三个关键知识点（复习 CrossEntropy.py）：
  1. 模型输出的是一组「分数」（logits），还不是概率；
  2. nn.CrossEntropyLoss 内部会自动先做 softmax、再算交叉熵，
     所以模型里我们不用手动写 softmax；
  3. 标签直接用类别编号（0~4），而不是 one-hot 向量。
"""


class TorchModel(nn.Module):
    """
    定义一个最简单的模型：只有一个线性层。

    nn.Linear(5, 5) 是什么意思？
      - 第一个 5：输入是 5 个数（5 维向量）
      - 第二个 5：输出是 5 个数（对应 5 个类别的「分数」）

    线性层做的事就是 y = x * W + b（矩阵乘法加偏置），
    这在 DNNforward.py 里已经手动实现过了。
    """
    def __init__(self):
        super(TorchModel, self).__init__()
        # 输入 5 维，输出 5 个类别的分数
        self.linear = nn.Linear(5, 5)

    def forward(self, x):
        # 输入 x 形状是 (batch_size, 5)
        # 输出形状是 (batch_size, 5)，每一行就是对应样本的 5 个类别分数
        return self.linear(x)


def build_sample():
    """
    生成一个训练样本（一条数据）。

    步骤：
      1. np.random.random(5) 生成 5 个 0~1 之间的随机小数，
         比如 [0.63, 0.11, 0.85, 0.42, 0.27]
      2. np.argmax(x) 找出最大的那个数在第几个位置（下标从 0 开始）。
         上面例子中 0.85 最大，它在第 2 个位置，所以标签 y = 2
    """
    x = np.random.random(5)   # 随机生成 5 个 0~1 之间的小数
    y = int(np.argmax(x))     # 最大值在第几位，类别就是几
    return x, y               # 返回 (特征, 标签)


def build_dataset(sample_num):
    """
    批量生成 sample_num 条样本，组成一个数据集。

    返回：
      X：所有特征，形状 (sample_num, 5)
      Y：所有标签，形状 (sample_num,)
    """
    X = []
    Y = []
    for i in range(sample_num):
        x, y = build_sample()
        X.append(x)
        Y.append(y)

    # 把 Python 列表转成 PyTorch 的 tensor（张量）
    # 特征用浮点型 FloatTensor，标签用整型 LongTensor
    # 注意：交叉熵要求标签必须是 LongTensor（长整型），这是常见的一个坑
    return torch.FloatTensor(np.array(X)), torch.LongTensor(np.array(Y))


def evaluate(model):
    """
    测试模型现在「猜得有多准」。

    做法：
      随机生成 200 条新数据（模型没见过），
      让模型预测，再和真实标签对比，算出准确率。
    """
    model.eval()   # 切到「测试模式」（养成习惯，虽然这里没用到 dropout）

    test_x, test_y = build_dataset(200)   # 生成 200 条测试数据

    correct = 0    # 记录猜对了几条
    with torch.no_grad():   # 测试阶段不需要算梯度，关掉它可以省内存
        logits = model(test_x)              # 前向传播，得到每条数据 5 个类别的分数
        pred = torch.argmax(logits, dim=1)  # 分数最高的那个类别，就是预测结果

        # 逐条对比预测结果和真实标签
        for i in range(len(pred)):
            if pred[i] == test_y[i]:
                correct += 1

    acc = correct / len(test_y)   # 准确率 = 猜对的条数 / 总条数
    return acc


def main():
    """
    训练的主流程：
      1. 准备模型、损失函数、优化器
      2. 反复训练很多轮，让 loss 不断下降
      3. 每轮结束后测一次准确率
    """
    # ---------- 1. 训练配置 ----------
    epoch_num = 50        # 训练多少轮（一轮 = 把训练集整个过一遍）
    batch_size = 20       # 每次拿多少条数据一起算（一个 batch）
    train_sample = 1000   # 训练集总共多少条数据
    learning_rate = 0.01  # 学习率：每次更新参数时迈的步子大小

    # ---------- 2. 准备模型、损失函数、优化器 ----------
    model = TorchModel()                  # 建立模型
    loss_fn = nn.CrossEntropyLoss()       # 交叉熵损失（内部自动做 softmax）
    optim = torch.optim.Adam(model.parameters(), lr=learning_rate)  # Adam 优化器

    # ---------- 3. 生成训练数据 ----------
    train_x, train_y = build_dataset(train_sample)

    # ---------- 4. 开始训练 ----------
    for epoch in range(epoch_num):
        model.train()   # 切到「训练模式」
        watch_loss = [] # 记录这一轮里每个 batch 的 loss，最后求平均

        # 把训练集切成很多个小 batch，一个 batch 一个 batch 地训练
        for batch_index in range(train_sample // batch_size):
            # 取出这一个 batch 的数据
            x = train_x[batch_index * batch_size : (batch_index + 1) * batch_size]
            y = train_y[batch_index * batch_size : (batch_index + 1) * batch_size]

            # ① 前向传播：把数据喂给模型，得到每个类别的分数
            logits = model(x)
            # ② 算损失：拿「分数」和「真实标签」算交叉熵，衡量预测有多差
            loss = loss_fn(logits, y)
            # ③ 反向传播：自动算出每个参数对 loss 的梯度
            loss.backward()
            # ④ 更新参数：让参数朝 loss 变小的方向走一步
            optim.step()
            # ⑤ 梯度清零：不清零的话，下一轮的梯度会累加上来（重要！）
            optim.zero_grad()

            watch_loss.append(loss.item())  # 记录这个 batch 的 loss

        # 每轮结束，打印平均 loss 和准确率
        avg_loss = np.mean(watch_loss)
        acc = evaluate(model)
        print("第 %d 轮：平均 loss = %.4f，准确率 = %.2f%%" % (epoch + 1, avg_loss, acc * 100))

    return model


def predict(model, input_vec):
    """
    用训练好的模型，对几条新数据做预测，看看效果。
    """
    model.eval()
    with torch.no_grad():
        logits = model(torch.FloatTensor(input_vec))   # 前向传播，得到分数
        probs = torch.softmax(logits, dim=1)           # softmax 把分数转成概率
        pred = torch.argmax(logits, dim=1)             # 分数最高的就是预测类别

    # 逐条打印结果
    for i in range(len(input_vec)):
        print("输入向量：", input_vec[i])
        print("  真实最大值在第 %d 维，模型预测类别 = %d" % (int(np.argmax(input_vec[i])), int(pred[i])))
        print("  每个类别的概率：", np.round(probs[i].numpy(), 4))
        print()


if __name__ == "__main__":
    model = main()

    print("\n========== 用新数据测试 ==========")
    # 故意构造最大值在不同位置的 4 条数据，方便观察
    test_vec = [
        [0.9, 0.1, 0.2, 0.3, 0.4],    # 第0维最大 -> 应该是类别 0
        [0.1, 0.2, 0.3, 0.8, 0.4],    # 第3维最大 -> 应该是类别 3
        [0.2, 0.9, 0.1, 0.3, 0.5],    # 第1维最大 -> 应该是类别 1
        [0.1, 0.2, 0.3, 0.4, 0.95],   # 第4维最大 -> 应该是类别 4
    ]
    predict(model, test_vec)
