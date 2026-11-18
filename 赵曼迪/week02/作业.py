# 解决 OpenMP 库冲突问题
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random
import json
import matplotlib.pyplot as plt

"""

基于pytorch框架编写模型训练
实现一个自行构造的找规律(机器学习)任务
规律：x是一个5维向量，找到最大的数为1，其余的数均为0

选择Softmax激活函数搭配交叉熵损失：
以下只针对单层线性分类模型，复合后的损失函数是凸函数。
优化目标：寻找使预测值和真实标签差距最小的模型参数。
凸损失曲面只有唯一的全局极小值，不存在局部极大、鞍点、其他局部极小，
梯度为0的临界点就是全局最优点，梯度下降可以稳定收敛。

如果换成Softmax+MSE或者多层非线性网络，损失不再是凸函数。
本模型参数总量 5*5+5=30维，高维非凸曲面会出现鞍点：
同一个临界点，沿参数1方向看是极大值，沿参数2方向看是极小值。

"""
class TorchModule(nn.Module):
    def __init__(self,input_size):
        super().__init__()
        self.linear = nn.Linear(input_size,5)

    def forward(self,x,y=None):
        logits  = self.linear(x) #只有第一次随机生成一组w和b，之后如果这个模型类生成的实例有w和b，会自行带入
        """
        关于self.linear(x)：变量直接转为函数：
        当变量后接(),触发__call__魔法函数，调用父类Module的forward函数，
        但父类的forward函数：raise NotImplementedError,直接只调父类会抛出异常，
        所以新定义的TorchModule子类需要重写forward函数，这样__call__会识别到并开始读取logits  = self.linear(x)，
        由于self.linear在上面定义的是nn.Linear(input,output)，所以self.linear.forward(x)等价于nn.Linear.forward(x)，
        开始调用同属Module下的子类Linear的forward函数
        """
        if y is not None:
            return F.cross_entropy(logits,y) #计算损失值，训练时用
            #F.cross_entropy:包含Softmax激活函数和交叉熵损失
        else:
            return F.softmax(logits,dim=1) #输出预测值,例：[0.05, 0.05, 0.85, 0.03, 0.02]，测试时用

#生成随机的输入值，并生成对应正确的输出值
def build_sample():
    #随机生成一组数组（包含5个随机值）
    x=np.random.random(5)
    #取出最大值对应的索引
    max_index = np.argmax(x)
    return x , max_index

#通过build_sample函数随机生成一批样本
def build_dataset(total_sample_num):
    X = []
    Y = []
    for i in range(total_sample_num):
        x,y=build_sample()
        X.append(x)
        Y.append(y) #CrossEntropyLoss对应的标签（真实值）要求是一维张量，存放类别索引，这里样本的写法要和CrossEntropy的真实值保持一致
    return torch.FloatTensor(X),torch.LongTensor(Y)#转化成张量

#构造验证集函数
def evaluate(model):
    model.eval() # 推理模式，关闭dropout、bn的训练行为
    #生成100条验证样本
    x , y = build_dataset(100)
    correct, wrong = 0, 0
    #关闭计算梯度
    with torch.no_grad():
        #针对生成的实例model计算其预测值
        y_pred = model(x)
        #找到每一维张量中的最大值
        pred_index = torch.argmax(y_pred,dim=1)
        #按顺序依次比较每一对预测值和真实值
        for y_num, pred_index_num in zip(y,pred_index):
            if y_num == pred_index_num:
                correct += 1
            else:
                wrong += 1
    print(f"验证集的准确率为：{correct / (correct + wrong)}")
    return correct / (correct + wrong)


def main():
    """
    设定轮数（对同一批样本训练）和每轮的训练次数（按次数缓慢更新梯度）以及每次训练的样本个数（每次用不同的样本判断应该往哪走）
    将每一轮训练的损失值展示出来，查看每轮训练后参数对应的损失值
    训练完一轮后在验证集上用设定的参数验证，查看验证集的准确率
    将训练集的损失值和验证集的准确性画出来
    """
    # 配置参数
    epoch_num = 20 #训练轮数
    batch_size = 20 #每次训练样本个数
    train_sample = 5000 #每轮训练的样本个数
    input_size = 5 #输入向量维度
    learning_rate = 0.01 #学习率
    #建立模型
    model = TorchModule(input_size)
    #选择优化器
    optim = torch.optim.Adam(model.parameters(),lr = learning_rate) #model.parameters()继承父类的字典：将优化器训练的参数存入
    #每轮结束后，存储训练集的损失值和验证的准确率
    log = []
    #生成样本
    train_x,train_y =build_dataset(train_sample)
    #按轮数训练
    for epoch in range(epoch_num):
        model.train() #开启训练模式
        #存储每一轮训练的损失值
        watch_loss =[]
        #每轮样本分次训练，共5000个样本，每次20个，分250次训练
        for batch_index in range(train_sample // batch_size):
            X = train_x[batch_index * batch_size :(batch_index + 1)* batch_size]
            Y = train_y[batch_index * batch_size :(batch_index + 1)* batch_size]
            optim.zero_grad() # 清梯度
            loss = model(X,Y)
            #实例加()同上解释，将这个20张量的样本带入自定义类TorchModule的forward函数，logits是一个20张量的线性y=wx+b，输出的损失值是一个平均的损失标量并附带一些说明
            loss.backward() #计算梯度，带入到优化器中:w.grad += 本次算出的梯度
            optim.step() #(w = w - lr * w.grad（只是展示w的变化方向))：更新权重（参数），新更新的权重自动更新到model.parameters(),每次只保留最新参数

            watch_loss.append(loss.item()) #loss.item()只取出那个标量
        print(f"第{epoch}轮训练完成，损失值为{np.mean(watch_loss)}")

        acc = evaluate(model)
        log.append([acc,float(np.mean(watch_loss))])

    #只保存模型权重参数
    torch.save(model.state_dict(),"model.bin")
    #画图
    print(log)
    plt.plot(range(len(log)), [l[0]for l in log] , label = "acc") #验证集曲线:x轴，y轴，标签
    plt.plot(range(len(log)), [l[1]for l in log] , label = "loss") #训练集曲线:x轴，y轴，标签
    plt.legend() #将上面设定的标签展示在画布对应的图像上
    plt.show() #展示

    return

#测试模型
def predict(model_path, input_vec):
    #生成一个新的模型实例
    modle_c = TorchModule(5)
    #将保存的参数带入到新实例中
    modle_c.load_state_dict(torch.load(model_path))

    modle_c.eval()
    with torch.no_grad():
        result = modle_c(torch.FloatTensor(input_vec))
    for vec, res in zip (input_vec,result):
        pred_idx = torch.argmax(res)
        print(f"对数组：{vec},判断的最大值对应的索引：{pred_idx}")

 
if __name__ == "__main__":
    #main()
    test_vec = [[0.81889086,0.15229675,0.31082123,0.03504317,0.88920843],
                [0.94963533,0.5524256,0.95758807,0.95520434,0.84890681],
                [0.90797868,0.67482528,0.13625847,0.34675372,0.19871392],
                [0.39349776,0.59416669,0.92579291,0.41567412,0.8358894]]
    predict("model.bin", test_vec)
    
