import torch
import torch.nn as nn
import numpy as np
import random
import os
import csv
from torch.utils.data import DataLoader,Dataset
from dataset import TextDataset
from model import TorchModel
import json

ROOT = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(ROOT, "data", "v12")

LABELS = ["ar", "en", "es", "hi", "ja", "ko", "pt", "ru", "th", "tr", "vi", "zh"]
LABEL_TO_ID = {label: index for index, label in enumerate(LABELS)}

# 超参数
SEED = 42
MAX_LEN = 40
EMBEDDING_DIM = 128
HIDDEN_DIM = 128
DROPOUT = 0.3
BATCH_SIZE = 64
LR = 1e-3
EPOCHS = 10


def load_data(dir_path,file_name):
    path = os.path.join(dir_path,f"{file_name}.csv")
    rows= []
    with open(path,"r",encoding="utf-8") as file:
        for r in csv.DictReader(file):
            rows.append( (r["text"],r["label"]))
    return rows

def load_vocab(dir_path,file_name):
    path = os.path.join(dir_path,f"{file_name}.json")
    with open(path,"r",encoding="utf-8") as file:
        return json.load(file)


def calculate_metrics(y_true, y_pred, labels): 
    """
        - tp[“ar”]：真实是 ar，也预测成 ar 的数量
        - fp[“ar”]：真实不是 ar，却预测成 ar 的数量
        - fn[“ar”]：真实是 ar，却预测成其他语言的数量
    """  
    tp = {label: 0 for label in labels}  # 为每个类别初始化 TP 计数
    fp = {label: 0 for label in labels}  # 为每个类别初始化 FP 计数
    fn = {label: 0 for label in labels}  # 为每个类别初始化 FN 计数
    correct = 0  # 初始化整体预测正确的样本数

    for actual, predicted in zip(y_true, y_pred):  # 每次同时取出一条真实标签和预测标签
        if actual == predicted:  # 判断当前样本是否预测正确
            tp[actual] += 1  # 预测正确时，当前真实类别的 TP 加 1
            correct += 1  # 整体正确样本数加 1
        else:  # 当前样本预测错误时执行下面的逻辑
            fn[actual] += 1  # 真实类别被漏判一次，所以该类别 FN 加 1
            fp[predicted] += 1  # 预测类别被多判一次，所以该类别 FP 加 1

    results = {}  # 创建空字典，用于保存每个类别的最终指标

    for label in labels:  # 依次计算每个语言类别的指标
        true_positive = tp[label]  # 读取该类别被正确预测的数量
        false_positive = fp[label]  # 读取不属于该类别却预测成该类别的数量
        false_negative = fn[label]  # 读取属于该类别却没有被找到的数量

        class_accuracy = (  # 开始计算该类别准确率
            true_positive / (true_positive + false_negative)  # 用 TP 除以该类真实样本总数
            if true_positive + false_negative > 0  # 判断该类真实样本总数是否大于 0
            else 0.0  # 没有真实样本时把类别准确率设为 0
        )  # 结束类别准确率计算

        precision = (  # 开始计算该类别精确率
            true_positive / (true_positive + false_positive)  # 用 TP 除以所有预测成该类别的数量
            if true_positive + false_positive > 0  # 判断预测成该类别的总数是否大于 0
            else 0.0  # 该类别从未被预测时把精确率设为 0
        )  # 结束精确率计算

        f1 = (  # 开始计算该类别 F1
            2 * precision * class_accuracy / (precision + class_accuracy)  # 使用精确率和类别准确率计算 F1
            if precision + class_accuracy > 0  # 判断精确率与类别准确率之和是否大于 0
            else 0.0  # 精确率和类别准确率都为 0 时把 F1 设为 0
        )  # 结束 F1 计算

        results[label] = {  # 将该类别结果写入 results 字典
            "class_accuracy": class_accuracy,  # 保存类别准确率
            "precision": precision,  # 保存精确率
            "f1": f1,  # 保存 F1
            "support": true_positive + false_negative,  # 保存真实属于该类别的样本总数
        }  # 结束该类别结果字典

    accuracy = correct / len(y_true)  # 用正确样本数除以总样本数，计算整体准确率

    macro_f1 = sum(result["f1"] for result in results.values()) / len(labels)  # 对每个类别的 F1 求平均

    return accuracy, macro_f1, results  # 返回整体准确率、Macro-F1 和每个类别的结果


def evaluate(model,loader,need_f1 = False):
    model.eval()
    y_true = []
    y_pred = []

    with torch.no_grad():
        for X,y in loader:
            logits = model(X)
            pred_index = torch.argmax(logits,dim = 1)

            y_true.extend(LABELS[index] for index in y.tolist())
            y_pred.extend(LABELS[index] for index in pred_index.tolist())

    accuracy, macro_f1, results = calculate_metrics(y_true, y_pred, LABELS)
    if need_f1:
        print(f"  Accuracy: {accuracy:.4f}  Macro-F1: {macro_f1:.4f}")
        for label in LABELS:
            result = results[label]
            print(
                f"  {label:>2}  类别准确率: {result['class_accuracy']:.4f}"
                f"  精确率: {result['precision']:.4f}"
                f"  F1: {result['f1']:.4f}"
                f"  样本数: {result['support']}"
            )

    return accuracy


def main():
    random.seed(SEED)
    torch.manual_seed(SEED)
    #加载训练数据
    train_rows = load_data(DATA_DIR, "train")
    test_rows = load_data(DATA_DIR, "test")
    dev_rows = load_data(DATA_DIR, "dev")
    print(f"训练集大小: {len(train_rows)}")
    print(f"测试集大小: {len(test_rows)}")
    print(f"验证集大小: {len(dev_rows)}")
    print()

    vocab = load_vocab(os.path.join(ROOT, "data"),"vocab")


    train_loader = DataLoader(TextDataset(vocab,train_rows,MAX_LEN,LABEL_TO_ID),batch_size=BATCH_SIZE,shuffle=True)
    test_loader = DataLoader(TextDataset(vocab,test_rows,MAX_LEN,LABEL_TO_ID),batch_size=BATCH_SIZE,shuffle=True)
    dev_loader = DataLoader(TextDataset(vocab,dev_rows,MAX_LEN,LABEL_TO_ID),batch_size=BATCH_SIZE,shuffle=True)

    model = TorchModel(len(vocab),EMBEDDING_DIM,HIDDEN_DIM,DROPOUT,len(LABELS))
    print(f"模型参数量为{sum([p.numel() for p in model.parameters()])}\n")


    optim = torch.optim.Adam(model.parameters(),lr=LR)


    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0
        for X,y in train_loader:
            loss = model(X,y)
            loss.backward()
            optim.step()
            optim.zero_grad()
            total_loss += loss.item()
        #验证集
        evl_acc= evaluate(model,dev_loader)
        print(f"Epoch [{epoch+1}/{EPOCHS}]  Loss: {total_loss/len(train_loader):.4f}  Val Acc: {evl_acc:.4f}")

    test_acc = evaluate(model,test_loader,need_f1=True)


    print(f"测试集准确率为: {test_acc:.4f}")
    torch.save({"state_dict": model.state_dict(), "vocab": dict(vocab),
                        "labels": LABELS, "maxlen": MAX_LEN,
                        "embed_dim": EMBEDDING_DIM, "hidden_dim": HIDDEN_DIM,
                        "dropout": DROPOUT}, os.path.join(ROOT, "data","best_model.pt"))
    print(f"已保存模型至 {os.path.join(ROOT, 'data', 'best_model.pt')}")

if __name__ == '__main__':
    main()

