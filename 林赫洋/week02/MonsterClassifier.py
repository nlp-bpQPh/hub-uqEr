# coding:utf8

# 解决 OpenMP 库冲突问题
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

# 脚本所在目录：模型权重一律按此定位，因此在任何文件夹下运行都能找到 monster_model.json
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

import torch
import torch.nn as nn
import numpy as np
import random
import json
import matplotlib.pyplot as plt

"""

任务：x是一个5维向量，依次为[血量, 攻击, 速度, 防御, 体型]
输出该怪物属于 史莱姆(0)、哥布林(1)、骷髅(2)、龙(3) 中的哪一类

四个类别的区间大幅交叠，存在无法区分的模糊样本

"""

# 各怪物类别的属性区间，每个类别内部顺序固定为 [hp, atk, spd, dfn, size]
# 四类区间大幅交叠：任意两类在五个维度上都有大面积重合，所以同样的数值可能属于多个类别
# 但仍然保留各自的倾向：史莱姆低血低攻、骷髅高速低防、龙高血高防大体型、哥布林各项居中
MONSTER_RANGES = [
    [(10, 450), (5, 110), (1, 55), (0, 70), (1, 5.5)],       # 0 史莱姆
    [(120, 680), (25, 145), (18, 82), (5, 110), (2, 7)],     # 1 哥布林
    [(90, 520), (35, 155), (38, 100), (0, 65), (1.5, 5)],    # 2 骷髅
    [(320, 1000), (75, 200), (15, 72), (38, 150), (4, 10)],  # 3 龙
]

CLASS_NAMES = ["史莱姆", "哥布林", "骷髅", "龙"]  # 标签 0/1/2/3 对应的怪物名称


class TorchModel(nn.Module):
    def __init__(self, input_size, class_num):
        super(TorchModel, self).__init__()
        self.bn = nn.BatchNorm1d(input_size)  # 输入是未归一化的原始数值，先批归一化稳定训练
        self.linear1 = nn.Linear(input_size, 32)  # 5 -> 32
        self.relu = torch.relu  # 激活函数，写法同原demo的 self.activation = torch.sigmoid
        self.linear2 = nn.Linear(32, class_num)  # 32 -> 4，输出四类logits
        self.activation = torch.softmax  # 归一化函数，只在推理时把logits转成四类概率
        self.loss = nn.CrossEntropyLoss()  # 交叉熵内部已含softmax，模型里不再手动加

    # 当输入真实标签，返回loss值；无真实标签，返回softmax后的四类概率
    def forward(self, x, y=None):
        x = self.bn(x)  # (batch_size, 5) -> (batch_size, 5)
        x = self.linear1(x)  # (batch_size, 5) -> (batch_size, 32)
        x = self.relu(x)  # (batch_size, 32) -> (batch_size, 32)
        x = self.linear2(x)  # (batch_size, 32) -> (batch_size, 4)
        if y is not None:
            return self.loss(x, y)  # 传入标签，用logits和真实标签算交叉熵
        else:
            return self.activation(x, dim=1)  # 不传标签，返回四类概率


# 生成一个样本：先生成类别标签（四类各占25%），再按该类别区间均匀生成5维特征
def build_sample():
    label = int(np.random.random() * 4)  # 只用np.random.random，落在0/1/2/3，四类等概率
    ranges = MONSTER_RANGES[label]  # 取出该类别的5个属性区间
    x = np.random.random(5)  # 先取5个[0,1)的随机数，和原demo写法完全一致
    for i in range(5):  # 再把每个随机数线性映射到对应属性的区间上
        low, high = ranges[i]
        x[i] = low + x[i] * (high - low)
    return x, label


# 随机生成一批样本，标签先均匀生成，因此四类天然各占25%
def build_dataset(total_sample_num):
    X = []
    Y = []
    for i in range(total_sample_num):
        x, y = build_sample()
        X.append(x)
        Y.append(y)  # 标签直接append标量得到(N,)，交叉熵要求这个形状，不能像原demo那样append([y])
    # 标签必须是long类型，CrossEntropyLoss要求类别索引是整数，不能是FloatTensor
    return torch.FloatTensor(X), torch.LongTensor(Y)


# 测试代码
# 用来测试每轮模型的准确率
def evaluate(model):
    model.eval()  # 评估模式：BatchNorm用训练期累积的均值和方差，不更新
    test_sample_num = 1000  # 每轮在1000个新生成的样本上评估
    x, y = build_dataset(test_sample_num)
    # 统计四类样本个数，沿用原demo打印正负样本分布的写法
    count = [0, 0, 0, 0]
    for y_t in y:
        count[int(y_t)] += 1
    print("本次预测集中共有%d个史莱姆，%d个哥布林，%d个骷髅，%d个龙"
          % (count[0], count[1], count[2], count[3]))
    correct, wrong = 0, 0
    with torch.no_grad():  # 不计算梯度
        y_pred = model(x)  # 模型预测 model.forward(x)，返回四类概率
        for y_p, y_t in zip(y_pred, y):  # 与真实标签进行对比
            p_list = [float(p) for p in y_p]  # 把四类概率转成普通list
            pred = p_list.index(max(p_list))  # 概率最大的那个下标就是预测类别
            if pred == int(y_t):
                correct += 1  # 预测类别和真实类别一致
            else:
                wrong += 1
    print("正确预测个数：%d, 正确率：%f" % (correct, correct / (correct + wrong)))
    return correct / (correct + wrong)


# 保存模型：把每层权重的类型、形状、数值都写成文本，保存后可以直接用记事本打开查看
def save_model(model, save_path):
    state = model.state_dict()
    dump = {}
    for name in state:
        v = state[name]  # 取出这一层的权重张量
        dump[name] = {"dtype": str(v.dtype).split(".")[-1],  # 例如 float32
                      "shape": list(v.shape),  # 例如 [32, 5]，这一项只是方便人看
                      "data": v.tolist()}  # 权重数值，转成普通list
    with open(save_path, "w", encoding="utf8") as f:
        json.dump(dump, f, indent=2)  # indent让文件逐层换行，方便阅读
    print("模型已保存为可读文本：%s" % save_path)


# 加载模型：从json文本还原每层权重，再装进同结构的模型
def load_model(model, load_path):
    with open(load_path, "r", encoding="utf8") as f:
        dump = json.load(f)
    state = {}
    for name in dump:
        item = dump[name]
        dtype = getattr(torch, item["dtype"])  # "float32" -> torch.float32
        state[name] = torch.tensor(item["data"], dtype=dtype)  # 嵌套list自动还原成对应形状
    model.load_state_dict(state)


def main():
    # 配置参数
    epoch_num = 20  # 训练轮数
    batch_size = 20  # 每次训练样本个数
    train_sample = 50000  # 每轮训练总共训练的样本总数
    input_size = 5  # 输入向量维度
    class_num = 4  # 怪物类别数
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
        model.train()  # 训练模式：BatchNorm会更新均值和方差，这一步不能省
        watch_loss = []
        for batch_index in range(train_sample // batch_size):
            # 取出一个batch数据作为输入  train_x[0:20] train_y[0:20] train_x[20:40] train_y[20:40]
            x = train_x[batch_index * batch_size: (batch_index + 1) * batch_size]
            y = train_y[batch_index * batch_size: (batch_index + 1) * batch_size]
            loss = model(x, y)  # 计算loss  model.forward(x,y)
            loss.backward()  # 计算梯度
            optim.step()  # 更新权重
            optim.zero_grad()  # 梯度归零
            watch_loss.append(loss.item())
        print("=========\n第%d轮平均loss:%f" % (epoch + 1, np.mean(watch_loss)))
        acc = evaluate(model)  # 测试本轮模型结果
        log.append([acc, float(np.mean(watch_loss))])
    # 保存模型，另存为monster_model.json（文本格式），不覆盖原model.bin
    save_model(model, os.path.join(BASE_DIR, "monster_model.json"))
    # 画图
    plt.plot(range(len(log)), [l[0] for l in log], label="acc")  # 画acc曲线
    plt.plot(range(len(log)), [l[1] for l in log], label="loss")  # 画loss曲线
    plt.legend()
    plt.show()
    return


# 使用训练好的模型做预测
def predict(model_path, input_vec):
    input_size = 5
    class_num = 4
    model = TorchModel(input_size, class_num)
    load_model(model, model_path)  # 从json文本还原权重
    print(model.state_dict())

    model.eval()  # 推理前必须切到测试模式，否则BatchNorm会用当前batch的统计量
    with torch.no_grad():  # 不计算梯度
        result = model.forward(torch.FloatTensor(input_vec))  # 模型预测，返回四类概率
    for vec, res in zip(input_vec, result):
        p_list = [float(p) for p in res]  # 把四类概率转成普通list
        pred = p_list.index(max(p_list))  # 概率最大的那个下标就是预测类别
        print("输入：%s, 预测类别：%d(%s), 四类概率：%s" % (vec, pred, CLASS_NAMES[pred], p_list))


if __name__ == "__main__":
    main()
    # 四条测试向量：每类一条，取各类最有代表性的数值，顺序为 [hp, atk, spd, dfn, size]
    test_vec = [[15.0, 8.0, 3.0, 2.0, 1.2],          # 史莱姆：低血低攻小体型
                [250.0, 90.0, 60.0, 60.0, 6.0],      # 哥布林：各项居中
                [150.0, 40.0, 98.0, 5.0, 2.0],       # 骷髅：高速低防小体型
                [980.0, 195.0, 70.0, 145.0, 9.5]]    # 龙：高血高防大体型
    # 用绝对路径加载权重，与当前工作目录无关
    predict(os.path.join(BASE_DIR, "monster_model.json"), test_vec)
