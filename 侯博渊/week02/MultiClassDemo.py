# coding:utf8

# 解决 OpenMP 库冲突问题
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import torch
import torch.nn as nn
import numpy as np
import random
import json
import matplotlib.pyplot as plt

"""

基于pytorch框架编写模型训练
实现一个自行构造的多分类(5分类)任务
规律：x是一个5维向量，向量中哪个维度的数字最大，就属于第几类
     例如 x = [0.1, 0.7, 0.3, 0.2, 0.4]，第2维的数字最大 -> 属于第2类

和二分类版本的区别：
1. 输出从 1 个值变成 5 个值（5 个类别的得分 logits）
2. loss 从 mse_loss 换成 cross_entropy（多分类标准做法，内部自带 softmax）
3. 标签从 0/1 变成类别下标 0~4，且必须是 LongTensor
"""


class TorchModel(nn.Module):
    def __init__(self, input_size, class_num):
        super(TorchModel, self).__init__()
        self.linear = nn.Linear(input_size, class_num)  # 线性层，5维输入 -> 5个类别的得分
        self.loss = nn.functional.cross_entropy  # 多分类用交叉熵

    # 当输入真实标签，返回loss值；无真实标签，返回5个类别的概率
    def forward(self, x, y=None):
        logits = self.linear(x)  # (batch_size, 5) 每个类别的原始得分
        if y is not None:
            return self.loss(logits, y)  # y 是类别下标，shape (batch_size,)
        else:
            return torch.softmax(logits, dim=-1)  # 得分转成概率，5个概率相加等于1


# 生成一个样本
# 随机生成一个5维向量，最大值在第几维，就是第几类（下标从0开始，0~4）
def build_sample():
    x = np.random.random(5)
    y = int(np.argmax(x))  # 最大值的下标 = 类别
    return x, y


# 随机生成一批样本
# 5个类别均匀分布（每个维度成为最大值的概率都是1/5）
def build_dataset(total_sample_num):
    X = []
    Y = []
    for i in range(total_sample_num):
        x, y = build_sample()
        X.append(x)
        Y.append(y)
    # 标签要用 LongTensor（int64），交叉熵只认这个
    return torch.FloatTensor(np.array(X)), torch.LongTensor(np.array(Y))


# 测试代码
# 用来测试每轮模型的准确率
def evaluate(model):
    model.eval()
    test_sample_num = 100
    x, y = build_dataset(test_sample_num)
    correct = 0
    with torch.no_grad():
        y_pred = model(x)  # (100, 5) 每个样本属于5个类别的概率
        pred_class = torch.argmax(y_pred, dim=-1)  # 取概率最大的那个类别的下标
        correct = int((pred_class == y).sum())
    print("正确预测个数：%d, 正确率：%f" % (correct, correct / test_sample_num))
    return correct / test_sample_num


def main():
    # 配置参数
    epoch_num = 20  # 训练轮数
    batch_size = 20  # 每次训练样本个数
    train_sample = 5000  # 每轮训练总共训练的样本总数（5分类比2分类要多一些样本）
    input_size = 5  # 输入向量维度
    class_num = 5  # 类别个数
    learning_rate = 0.01  # 学习率
    # 建立模型
    model = TorchModel(input_size, class_num)
    # 选择优化器
    optim = torch.optim.Adam(model.parameters(), lr=learning_rate)
    log = []
    # 创建训练集，正常任务是读取训练集
    train_x, train_y = build_dataset(train_sample)
    # 训练过程
    for epoch in range(epoch_num):
        model.train()
        watch_loss = []
        for batch_index in range(train_sample // batch_size):
            # 取出一个batch数据作为输入   train_x[0:20]  train_y[0:20] train_x[20:40]  train_y[20:40]
            x = train_x[batch_index * batch_size : (batch_index + 1) * batch_size]
            y = train_y[batch_index * batch_size : (batch_index + 1) * batch_size]
            loss = model(x, y)  # 计算loss  model.forward(x,y)
            loss.backward()  # 计算梯度
            optim.step()  # 更新权重
            optim.zero_grad()  # 梯度归零
            watch_loss.append(loss.item())
        print("=========\n第%d轮平均loss:%f" % (epoch + 1, np.mean(watch_loss)))
        acc = evaluate(model)  # 测试本轮模型结果
        log.append([acc, float(np.mean(watch_loss))])
    # 保存模型
    torch.save(model.state_dict(), "model_multiclass.bin")
    # 画图
    print(log)
    plt.plot(range(len(log)), [l[0] for l in log], label="acc")  # 画acc曲线
    plt.plot(range(len(log)), [l[1] for l in log], label="loss")  # 画loss曲线
    plt.legend()
    plt.show()
    return


# 使用训练好的模型做预测
def predict(model_path, input_vec):
    input_size = 5
    class_num = 5
    model = TorchModel(input_size, class_num)
    model.load_state_dict(torch.load(model_path))  # 加载训练好的权重
    print(model.state_dict())

    model.eval()  # 测试模式
    with torch.no_grad():  # 不计算梯度
        result = model.forward(torch.FloatTensor(np.array(input_vec)))  # (n, 5) 概率
    for vec, res in zip(input_vec, result):
        # res 是5个类别的概率，取最大值的下标就是预测类别（+1 是为了显示成"第几维"）
        cls = int(torch.argmax(res))
        print("输入：%s, 预测类别：第%d维, 概率值：%f" % (vec, cls + 1, float(res[cls])))
        print("      各类别概率：%s" % ["%.3f" % float(p) for p in res])


if __name__ == "__main__":
    main()
    test_vec = [[0.81889086, 0.15229675, 0.31082123, 0.03504317, 0.88920843],  # 第5维最大 -> 第5类
                [0.94963533, 0.5524256,  0.95758807, 0.95520434, 0.84890681],  # 第3维最大 -> 第3类
                [0.90797868, 0.67482528, 0.13625847, 0.34675372, 0.19871392],  # 第1维最大 -> 第1类
                [0.39349776, 0.59416669, 0.92579291, 0.41567412, 0.8358894],   # 第3维最大 -> 第3类
                [0.1,        0.9,        0.3,        0.2,        0.4]]          # 第2维最大 -> 第2类
    predict("model_multiclass.bin", test_vec)
