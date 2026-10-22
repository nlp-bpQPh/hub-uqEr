import torch
import torch.nn as nn
import numpy as np


class TorchModel(nn.Module):
    def __init__(self, input_size):
        super().__init__()
        self.linear = nn.Linear(in_features=input_size,out_features=5)
        self.loss = nn.functional.cross_entropy


    def forward(self,x,y = None):
        y_pred =self.linear(x)
        if y is not None:
            #训练过程
            return self.loss(y_pred,y)
        else:

            return torch.softmax(y_pred,dim=-1)
        

def generate_train_data():
    x = np.random.random(5)
    return (x, int(np.argmax(x)))
def generate_dataset(num):
    X=[]
    Y=[]
    for i in range(num):
        x,y = generate_train_data()
        X.append(x)
        Y.append(y)
    return torch.FloatTensor(X),torch.LongTensor(Y)


def evaluate(model):

    model.eval()
    val_x,val_y = generate_dataset(100)
    correct = 0
    with torch.no_grad():
        y_pred = model(val_x)
        pred_index = torch.argmax(y_pred,dim = 1)
        for p ,i in zip(pred_index,val_y):
            if p == i:
                correct +=1
    acc = correct/len(val_y)
    print("验证集 = 正确率：%f" % acc)
    return acc

def main():
    epochs =50
    batch_size = 32
    lr = 0.01
    train_sample = 5000
    input_size = 5

    train_x,train_y = generate_dataset(num=train_sample)

    model = TorchModel(input_size=input_size)

    optim = torch.optim.Adam(model.parameters(), lr=lr)

    # 【新增】早停机制相关参数
    best_acc = 0.0           # 记录历史最高准确率
    patience = 5             # 容忍连续多少轮不提升（早停阈值）
    patience_counter = 0     # 计数器
    best_model_path = "best_demo.bin"


    for epoch in range(epochs):

        model.train()

        watch_loss = []  
        for batch_index in range(train_sample//batch_size):
            x = train_x[batch_index*batch_size:(batch_index+1)*batch_size]
            y = train_y[batch_index*batch_size:(batch_index+1)*batch_size]
            loss = model(x,y)
            loss.backward()
            optim.step()
            optim.zero_grad()
            watch_loss.append(loss.item())
        val_acc = evaluate(model=model)

        # 【新增】计算平均 loss
        avg_loss = np.mean(watch_loss)

        # 【新增】打印训练日志
        print(f"Epoch [{epoch+1}/{epochs}]  Loss: {avg_loss:.4f}  Val Acc: {val_acc:.4f}")



        # 【新增】早停与保存最佳模型逻辑
        if val_acc > best_acc:
            best_acc = val_acc
            patience_counter = 0
            # 只有准确率突破历史最高，才保存权重
            torch.save(model.state_dict(), best_model_path)
            print(f"  --> 发现更好的模型，验证集准确率提升至 {best_acc:.4f}，已保存权重！")
       

    print(f"\n训练完成,  最高验证集准确率为: {best_acc:.4f}，最佳模型已保存至 {best_model_path}")


def predict(path,vec):
    model = TorchModel(5)
    model.load_state_dict(torch.load(path))
    model.eval()
    with torch.no_grad():
        #转张量
        y_pred = model(torch.FloatTensor(vec))
        max_index = torch.argmax(y_pred,dim=1)
    for i, (row, idx) in enumerate(zip(y_pred, max_index)):
        print("样本%d: 预测类别=%d, 概率=[%s]" % (i, idx, ", ".join(["%f" % float(x) for x in row])))



if __name__ == "__main__":
    main()

    # test_vec = [
    #     [0.9, 0.2, 0.3, 90, 0.4],   
    #     [0.1, 0.8, 10, 0.3, 0.4],   
    #     [0.2, 0.3, 0.7, 0.1, 0.5],  
    #     [0.1, 100, 0.3, 0.9, 0.4],  
    #     [0.1, 0.2, 0.3, 0.4, 0.8],  
    # ]
    # predict("best_demo.bin", test_vec)
    