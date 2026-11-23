
import csv
from collections import Counter, OrderedDict

import torch
import torch.nn as nn
import unicodedata
import numpy as np


# 1.创建词表
# 1.1.读取文件获得句子，标签
def build_sentences_vocab(path):
    rows = []
    with open(path, encoding="utf-8") as f:
        for r in csv.DictReader(f):
            rows.append((r["text"], r["label"]))
    return rows
# 1.2 过滤特殊字符和标点
def normalize(text):
    t = unicodedata.normalize("NFKC", text).lower()   # ① NFKC + 小写
    return "".join(c for c in t
                   if unicodedata.category(c)[0] in ("L", "M"))  # ② 只保留字母和组合符号

# 1.3 构建词表
def build_char_vocab(rows, min_freq=1):
    freq = Counter()
    # 计数
    for text, _ in rows:
        freq.update(normalize(text))

    # 过滤低频 + 排序（频次降序，字符升序，保证可复现）
    items = sorted(((c, n) for c, n in freq.items() if n >= min_freq),
                   key=lambda kv: (-kv[1], kv[0]))

    # 构建词表
    vocab = {"<PAD>": 0, "<UNK>": 1}
    for ch, _ in items:
        vocab[ch] = len(vocab)

    return vocab


# 1.3 获取句子在词表中的索引
def build_sentence_idx(sentence,vocab,max_length=64):
    unk_id=vocab["<UNK>"]
    pad_id=vocab["<PAD>"]

    words=normalize(sentence)
    idx=[vocab.get(c,unk_id) for c in words]

    if len(idx)>max_length:
        idx=idx[:max_length]
    else:
        idx=idx+[pad_id]*(max_length-len(idx))
    return idx


# 2.创建训练模型
class LanguageModel(nn.Module):
    def __init__(self,vocab_size,label_size,input_size,hidden_size):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, input_size,padding_idx=0)
        self.rnn = nn.RNN(input_size=input_size,hidden_size=hidden_size,batch_first=True)
        self.layer_norm = nn.LayerNorm(hidden_size)
        self.dropout = nn.Dropout(p=0.3)
        self.fc = nn.Linear(hidden_size, label_size)

    def forward(self,x):
        """不明白这行什么意思"""
        mask = (x != 0).unsqueeze(-1)


        embedded = self.embedding(x)
        rnn,_= self.rnn(embedded)

        """不明白这行什么意思"""
        rnn = rnn.masked_fill(~mask, -1e9)


        pooled = rnn.max(dim=1)[0]
        norm = self.layer_norm(pooled)
        drop=self.dropout(norm)
        logits = self.fc(drop)
        return logits

def get_laebl(labels,k):
    for key,value in labels.items():
        if key==k:
            return value
    return None

def evaluate(model, data):
    model.eval()
    correct = 0
    total = 0
    printed = False
    with torch.no_grad():
        for x, y in data:
            pre_y = model(x)                          # [N, 12] logits
            prob  = torch.softmax(pre_y, dim=1)        # [N, 12] 概率
            pred_y = torch.argmax(pre_y, dim=1)        # [N]    预测类别 0~11

            correct += (pred_y == y).sum().item()
            total   += y.size(0)

            # 只打印第一个 batch 的前几条，避免刷屏
            if not printed:
                for vec, pv, pred in zip(x, prob, pred_y):
                    print("输入：%s, 预测类别：%d, 概率值：%.4f"
                          % (vec.numpy(), pred.item(), pv[pred].item()))
                printed = True

    return correct / total if total > 0 else 0.0

def predict(text, model, vocab, labels, max_length=40):
    model.eval()
    idx = build_sentence_idx(text, vocab, max_length)
    x = torch.tensor([idx], dtype=torch.long)   # [1, L]
    with torch.no_grad():
        logits = model(x)                        # [1, 12]
        prob = torch.softmax(logits, dim=1)
        pred_id = torch.argmax(logits, dim=1).item()
        conf = prob[0, pred_id].item()

    # id -> 语言标签（labels 是 {lang: id}，反转一下）
    id2label = {v: k for k, v in labels.items()}
    return id2label[pred_id], conf



if __name__ == '__main__':

    labels = {"ar": 0, "en": 1, "es": 2, "hi": 3, "ja": 4, "pt": 5,
              "ru": 6, "zh": 7, "ko": 8, "vi": 9, "th": 10, "tr": 11}
    batch_size = 32  # 每轮训练的数据量
    epoch_size = 50  # 训练多少轮
    lr = 0.001  # 学习率


    row=build_sentences_vocab("./data/train.csv")

    dev_row=build_sentences_vocab("./data/dev.csv")

    vocab=build_char_vocab(row,min_freq=1)
    X=[]
    Y=[]
    for text,label in row:
        index = build_sentence_idx(text,vocab,40)
        label_index=get_laebl(labels,label)
        X.append(index)
        Y.append(label_index)
    X = torch.tensor(X, dtype=torch.long)  # [N, L]
    Y = torch.tensor(Y,dtype=torch.long)


    dev_x=[]
    dev_y=[]
    for text, label in dev_row:
        index = build_sentence_idx(text, vocab, 40)
        label_index = get_laebl(labels, label)
        dev_x.append(index)
        dev_y.append(label_index)
    dev_x = torch.tensor(dev_x)  # [N, L]
    dev_y = torch.tensor(dev_y)

    model=LanguageModel(len(vocab),len(labels),64,128)
    optimizer = torch.optim.Adam(model.parameters(),lr=lr)
    loss_func = nn.CrossEntropyLoss()
    # 每轮需要打乱一下训练集
    shuffle_data = torch.utils.data.TensorDataset(X, Y)
    data = torch.utils.data.DataLoader(shuffle_data, batch_size=batch_size, shuffle=True)

    dev_shuffle_data = torch.utils.data.TensorDataset(dev_x, dev_y)
    dev_data = torch.utils.data.DataLoader(dev_shuffle_data, batch_size=batch_size, shuffle=False)
    for epoch in range(epoch_size):

        model.train()
        watch_loss = []
        for x, y in data:
            pre_y = model.forward(x)  # 获得损失值
            loss = loss_func(pre_y, y)
            loss.backward()
            optimizer.step()  # 更新权重
            optimizer.zero_grad()  # 梯度归0
            watch_loss.append(loss.item())
        print("=========\n第%d轮平均loss:%f" % (epoch + 1, np.mean(watch_loss)))
        acc = evaluate(model,dev_data)
        print("第%d轮平均loss:%f  验证准确率:%.4f" % (epoch + 1, np.mean(watch_loss), acc))

    # 训练循环结束后，测试几个句子
    test_sentences = [
        "Hello, how are you today?",  # en
        "你好，今天过得怎么样？",  # zh
        "Bonjour, comment allez-vous ?",  # fr(注意：训练集没有法语，会被判成最像的)
        "こんにちは、元気ですか？",  # ja
        "Привет, как дела?",  # ru
        "안녕하세요, 잘 지내세요?",  # ko
    ]

    for s in test_sentences:
        lang, conf = predict(s, model, vocab, labels)
        print(f"句子：{s!r:40s} => 预测语言：{lang}  置信度：{conf:.4f}")