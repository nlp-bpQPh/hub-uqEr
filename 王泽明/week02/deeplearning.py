#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
File: deeplearn.py
Author: Miku
Date: 2026-08-23
Description: 多分类任务的训练:一个5维随机向量，哪一维数字最大就属于第几类
"""
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"  # 防止numpy和PyTorch各自带的OpenMP运行时打架
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt


def get_category(array1):
    """
    获取数组所属种类，若含多个相同的最大值，返回首个的下标作为所属种类
    """
    return np.argmax(array1)

def create_dataset(num):
    """
    生成num个随机样本，创建数据集
    """
    x = []
    y = []
    for i in range(num):
        array1 = np.random.random(5)
        category1 = get_category(array1)
        x.append(array1)
        y.append(category1)
    return torch.FloatTensor(x), torch.LongTensor(y)  # 注意交叉熵的y表示类别，必须是整数！

class TorchModel(nn.Module):
    def __init__(self, dimension) -> None:
        super(TorchModel, self).__init__()
        self.linear = nn.Linear(dimension, 5)  # 线性层  5：输出在5个种类上各自的概率分布
        self.loss = nn.functional.cross_entropy  # 损失函数-交叉熵  自带softmax，
                                                 # 不可再手动指定self.activation = nn.Softmax

    def forward(self, x, y=None):
        y_pred = self.linear(x)  # 使用了交叉熵损失函数，则无需y_pred = self.activation(x)
        if y is not None:  # 训练模式，需计算loss算梯度
            return self.loss(y_pred, y)
        else:  # 仅评估模式，只返回预测结果
            return  y_pred

def evaluate(model):
    """
    测试模型准确率
    """
    model.eval()  # 将模型由默认的训练模式切换为评估模式
    # 生成200个随机样本
    x, y = create_dataset(200)
    right, wrong = 0, 0  #元组解包
    with torch.no_grad():  # 关闭梯度求导计算，节省显存+计算资源
        y_pred = model(x)  # 模型预测 model.forward(x)
        for y_p, y_t in zip(y_pred, y):
            category_p = int(torch.argmax(y_p))  # 获取max值的索引！！！
            if category_p == y_t:
                right += 1
            else:
                wrong += 1
    print(f"right: {right}, wrong: {wrong}")
    return right/(right+wrong)

def predict(model_path, exam_vector):
    """
    使用训练好的模型进行预测
    """
    model = TorchModel(5)
    model.load_state_dict(torch.load(model_path))
    print(model.state_dict())

    model.eval()  # 评估模式
    with torch.no_grad():
        result = model.forward(torch.FloatTensor(exam_vector))  # 模型预测
    for vector, res in zip(exam_vector, result):    
        category = torch.argmax(res)
        print(f"{vector}, 属于第{category+1}类")


def main():
    epoch = 100  # 训练轮数
    batch = 20  # 每次训练个数
    sample = 10000  # 训练总数
    dimension = 5  # 输入向量维度
    lr = 0.02  # 学习率

    model = TorchModel(dimension)  # 建立模型
    optim = torch.optim.Adam(model.parameters(), lr=lr)  # 优化器(梯队下降器)
    tran_x, tran_y = create_dataset(sample)  # 创建训练集
    plot_acc_loss = []
    for turn in range(epoch):
        model.train()
        batch_losses = []
        for batch_i in range(sample//batch):
            # 逐batch训练调参
            x = tran_x[batch_i*batch:(batch_i+1)*batch]
            y = tran_y[batch_i*batch:(batch_i+1)*batch]
            loss = model(x, y)  # 计算loss  model.forward(x,y)
            loss.backward()  # 计算梯度
            optim.step()  # 更新权重
            optim.zero_grad()  # 梯度归零
            batch_losses.append(loss.item())  # .item(): 指名道姓提取tensor标量值
        avg_loss = np.mean(batch_losses)  # 平均损失
        print(f"第{turn}轮平均loss: {avg_loss}")
        accuracy = evaluate(model)
        plot_acc_loss.append([accuracy, float(avg_loss)])

    print(plot_acc_loss)
    torch.save(model.state_dict(), 'model.txt')  # 保存模型    
    plt.plot([item[0] for item in plot_acc_loss], label='acc')
    plt.plot([item[1] for item in plot_acc_loss], label='loss')
    plt.legend()
    plt.show()
    return



if __name__ == "__main__":
    # main()

    test_vec = [[0.88889086,0.15229675,0.31082123,0.03504317,0.88920843],
                [0.94963533,0.5524256,0.95758807,0.95520434,0.84890681],
                [0.90797868,0.67482528,0.13625847,0.34675372,0.19871392],
                [0.99349776,0.59416669,0.92579291,0.41567412,0.1358894],
                [0.5,-1,-2,2,0]]
                
    predict("model.txt", test_vec)