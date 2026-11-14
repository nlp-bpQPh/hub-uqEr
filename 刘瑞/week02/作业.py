
import torch
import torch.nn as nn
import numpy as np

class TrainModel(nn.Module):
    def __init__(self,input_size):
        super(TrainModel, self).__init__()
        self.linear=nn.Linear(input_size,input_size) # 全连接层,设置线性函数f(x)=wx+b
        self.softmax = nn.Softmax(dim=1) # 设置激活函数
        self.loss=nn.CrossEntropyLoss() # 设置损失函数

    def forward(self,x,y=None):
        y_out=self.linear(x)
        y_softmax=self.softmax(y_out)
        if y is not None:
            loss=self.loss(y_out,y)
            return loss
        else:
            return y_softmax

def create_data(data_size,input_size):

    data_x=np.random.randn(data_size,input_size).astype(np.float32)
    data_y=np.argmax(data_x,axis=1).astype(np.int64)
    return data_x,data_y

def evaluate(model):
    model.eval() # 切换模式

    x,y=create_data(1000,5) # 生成测试集
    x=torch.from_numpy(x)
    y=torch.from_numpy(y)

    with torch.no_grad():  # 不计算梯度
        pre_y=model.forward(x) # 获得预测值
        real_y=torch.argmax(x,dim=1) # 或者确定值

        acc=(real_y==y).float().mean().item() # 比较预测值和确定值

        for vec, prob_vec, pred in zip(x, pre_y, real_y):
            print("输入：%s, 预测类别：%d, 概率值：%.4f"% (vec.numpy().round(4), pred.item(), prob_vec[pred].item()))
        return acc

def main():
    batch_size=32 # 每轮训练的数据量
    epoch_size=5000 # 训练多少轮
    train_data=20000 # 一共的训练数据
    lr=0.001 # 学习率

    model=TrainModel(5)

    ad=torch.optim.Adam(model.parameters(),lr=lr) # 优化器,用来迭代权重

    train_x,train_y=create_data(train_data,5) # 生成训练数据

    x=torch.from_numpy(train_x)
    y=torch.from_numpy(train_y)

    for epoch in range(epoch_size):

        # 每轮需要打乱一下训练集
        shuffle_data=torch.utils.data.TensorDataset(x,y)
        data=torch.utils.data.DataLoader(shuffle_data,batch_size=batch_size,shuffle=True)
        model.train()
        watch_loss=[]
        for x,y in data:

            loss=model.forward(x,y) # 获得损失值
            loss.backward() # 计算梯度
            ad.step() # 更新权重
            ad.zero_grad() # 梯度归0
            watch_loss.append(loss.item())
        print("=========\n第%d轮平均loss:%f" % (epoch + 1, np.mean(watch_loss)))
        acc=evaluate(model)
        print("第%d轮平均loss:%f  验证准确率:%.4f" % (epoch + 1, np.mean(watch_loss), acc))

if __name__ == '__main__':
    main()
