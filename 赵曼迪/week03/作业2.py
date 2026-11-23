# -*- coding: utf-8 -*-
"""
任务：区别12中语言
模型：Enbeddind-RNN/LSTM/CNN-MaxPool/AvgPool(池化）-BatchNorm/LayerNorm(归一化)-Dropout-Linear-Softmax
########
Enbeddind：
处理文字，将文字转换为向量
分词，根据实际情况分词，将特定的字或词存在字典中，只显示对应的数字，由Enbeddind自动生成向量，
例：vocab = {"<pad>":0, "<unk>":1, "你":2, "好":3, "un":4},调用torch.nn.Embedding(VOCAB_SIZE, EMBED_DIM)，
EMBED_DIM:每个位置对应的向量维度，VOCAB_SIZE：词典里有多少个字词就是多少，pad：根据文字的矩阵长度补零，unk：词典没没有的词统一表示1
当训练中的文字包含对应的词后,取出词在字典中的位置，然后在Embedding中取出这个位置对应的向量，例：
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import torch
import torch.nn as nn

#词组
vocab = {"<pad>":0,"<unk>":1,"你":2, "好":3, "中":4, "国":5, "欢":6, "迎":7}

#设置参数
batch_size = len(vocab)
embed_dim = 6
embedding = nn.Embedding(batch_size,embed_dim,padding_idx=0)

#要传入的数据
sentences = ["中国欢迎你", "你好中国", "feda"]
#设置矩阵的维度
max_len = 5

#输出传入的数据在词组中对应的位置（值）
embedding_list = []
for sentence in sentences:
    dim = [vocab.get(i,vocab["<unk>"])for i in sentence][:5]
    dim += [vocab["<pad>"]] * (max_len - len(sentence))
    embedding_list.append(dim)
print(embedding_list)

#将对应的位置和值带入embedding.weight,在对应位置上生成向量
x = torch.LongTensor(embedding_list)
out = embedding(x)

print (out)

##########
RNN：
h(t)=tanh(w_h * h(t-1) + w_x * x(t) + b)
x为上层输出的向量，将t-1的输出值带入到下一个x中共同得到一个输出值
使用：torch.nn.RNN(embed_dim, hidden_dim, batch_first=True)
embed_dim:输入x的长度，hidden_dim：设置参数w_h的长度，也为输出h长度，batch_first=True:加了，默认传入的shape(B,L,H)即和embedding的输出shape一一对应

######
LSTM:
当x个数很多时，最早的x信息所占比例已经不多，因此：
1.像RNN一样生成当前的输出值h(t)
h(t)=tanh(w_H * H(t-1) + w_x * x(t) + b) ,也可以写为h(t)=tanh(w_H[h(t-1),x(t)] + b)
2.设置输入门：it=sigmoid(w_i[H(t-1),x(t)]+b_i)将上步得到的结果和这步的输入拼起来，判断各个数0~1的比例
      遗忘门：ft=sigmoid(w_f[H(t-1),x(t)]+b_f)同上
      输出门：ot=sigmoid(w_o[H(t-1),x(t)]+b_o)同上
3.得到新结果：c(t) = ft * c(t-1) + it * h(t) 注：这里的*为对位相乘
4.将得到的结果再做一次输出门：
  H(t) = ot * tanh(c(t))注：这里的*为对位相乘
使用：
lstm = nn.LSTM(
    input_size=64,      # 输入向量维度，就是embed_dim，每个token的向量长度，同RNN.embed_dim
    hidden_size=128,    # LSTM隐状态h、细胞状态c的维度（上面3，4步对应的结果）
    num_layers=1,       # 堆叠LSTM层数
    batch_first=True,   # 输入输出格式：(B, seq_len, dim)，同RNN
    bidirectional=False # 是否双向LSTM
)

#######
MaxPool/AvgPool(池化）:
利用找最大值/平均值降维，例：将5*3*10降为5*10，3和10具体降哪一维根据模型的实际情况而定
MaxPool：对每一个特征通道，在所有有效时间步中选取数值最大的特征作为输出。
优势：擅长捕捉局部强信号，只要序列中出现标志性特征（如语种独有的假名、字符），就可以被模型捕捉，对语种识别这类任务适配度高，计算简单高效。
缺陷：对噪声敏感，异常峰值容易主导结果，会忽略其余弱特征信息。
使用：pooled_max = h.max(dim=1)[0]
AvgPool：对每一个特征通道，对全部有效时间步特征求平均值。
优势：融合整条序列所有位置信息，结果更加平滑，抗噪性更好，适合依赖整体分布的文本任务。
缺陷：强特征会被均值稀释，标志性信号被弱化；掩码处理更复杂，不能直接调用 mean，需要按有效长度加权平均，否则 padding 占位 0 会干扰计算结果。
使用：pooled_avg = h.mean(dim=1)

########
BatchNorm/LayerNorm(归一化)
把范围很大的特征映射到均值为 0、方差接近 1 的稳定分布，缓解梯度消失 / 爆炸，加速收敛，提升训练稳定性，一定程度减少对初始化、学习率的敏感程度
BatchNorm：对同一个特征通道（同一个 H 下标），在整个 batch 所有样本、所有时间步上统计均值、方差，适用：CNN
LayerNorm：对单一样本、单个时间步内部的特征维度 H做归一化，使用：RNN、LSTM、Transformer（变长文本）

########
Dropout：
元素级随机随机关闭
dropout = nn.Dropout(dropout)，dropout：设定的参数，例：0.3，30%的上一层的输出层神元置为0 ，其余的神经元扩大1/(1-0.3)倍
防止过拟合，训练时用，验证时model.eval()关闭

########
损失：CrossEntropyLoss

########
优化器：Adam
维护两个一阶、二阶梯度滑动平均变量：
m(t)：梯度的一阶矩（动量，梯度的指数移动均值，用来加速收敛，平滑梯度震荡）
v(t)：梯度的二阶矩（梯度平方的指数移动均值，用来做自适应学习率）
对每个参数单独计算更新步长：梯度大的时候步长自动缩小，梯度小的时候步长自动放大；每个参数拥有独立的动态学习率
"""

import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
import csv
import time
import random
import argparse
import unicodedata
from collections import Counter, OrderedDict

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# ─── 超参数 ────────────────────────────────────────────────────────────
SEED = 42
MAX_LEN = 40          # 中文中位长度约 10 字符、德语约 39，统一截到 40
EMBED_DIM = 64
HIDDEN_DIM = 64
DROPOUT = 0.3
BATCH_SIZE = 64
LR = 1e-3
EPOCHS = 8
NUM_THREADS = 1       # 小模型单线程比 16 线程快 2.7 倍（实测），别改成 16

HERE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(HERE, "v12")
VOCAB_PATH = os.path.join(HERE,"vocab", "vocab_v12.txt")

# 课堂用的 8 个语种：模型的输出就是这 8 个格子
LABELS = ["ar", "en", "es", "hi", "ja", "ko", "pt", "ru", "th", "tr", "vi", "zh"]



# ─── 1. 文本归一化（必须和建数据时一致）────────────────────────────────
def normalize(text):
    """
    NFKC 归一化 + 转小写 + 只保留「字母(L)」与「组合符号(M)」。

    为什么保留 M：印地语、泰语的元音是组合标记（कि 里的 ि），
    只留字母会把印地语砍成裸辅音，模型就没法区分。
    """
    t = unicodedata.normalize("NFKC", text).lower()
    return "".join(c for c in t if unicodedata.category(c)[0] in ("L", "M"))


# ─── 2. 字符表构建 ─────────────────────────────────────────────────────
def build_char_vocab(sentences, min_freq=1):
    """
    字符表 = 训练集里所有出现过的字符，每个字符发一个整数编号。

    第 0 号 <PAD>：补齐用（Embedding 里 padding_idx=0，向量恒为 0、不参与训练）
    第 1 号 <UNK>：没见过的字符（字符级下几乎用不到）
    其余按出现次数从高到低编号，次数相同按字符编码排序（保证可复现）

    多语种的关键点：这张表不需要按语种分开建。汉字、拉丁字母、西里尔字母、
    阿拉伯字母、天城文、泰文全都排在同一张表里 —— 这就是"字符级天然多语种"。

    课堂主流程没有直接调用它（字符表是从 vocab_v8.txt 读的）；
    一旦换了训练集或加了语种，就要用它重新生成一张表，
    否则新语种的字符会全部落到 1 号 <UNK>，模型等于睁眼瞎。
    """
    freq = Counter()
    for s in sentences:
        freq.update(normalize(s))
    chars = sorted(((c, n) for c, n in freq.items() if n >= min_freq),
                   key=lambda kv: (-kv[1], kv[0]))
    vocab = OrderedDict([("<PAD>", 0), ("<UNK>", 1)])
    for ch, _ in chars:
        vocab[ch] = len(vocab)
    return vocab


def load_vocab_file(path):
    """
    读已经生成好的字符表（每行「字符 \\t id \\t 出现次数」，以 # 开头的是注释）。

    课堂用的就是 data/langid/vocab/vocab_v8.txt，它是从 v8 训练集统计出来的。
    → 换了训练集、加了语种，这张表就得重新生成（用上面的 build_char_vocab）。
    """
    vocab = OrderedDict()
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.startswith("#") or not line.strip():
                continue
            ch, idx, _freq = line.rstrip("\n").split("\t")
            vocab[ch] = int(idx)
    return vocab


def encode(text, vocab, maxlen=MAX_LEN):
    ids = [vocab.get(ch, 1) for ch in normalize(text)][:maxlen]
    ids += [0] * (maxlen - len(ids))
    return ids


# ─── 3. 读数据 ─────────────────────────────────────────────────────────
def load_split(data_dir, name):
    path = os.path.join(data_dir, f"{name}.csv")
    rows = []
    with open(path, encoding="utf-8") as f:
        for r in csv.DictReader(f):
            rows.append((r["text"], r["label"]))
    return rows


class TextDataset(Dataset):
    def __init__(self, rows, vocab, label2id):
        self.X = [encode(t, vocab) for t, _ in rows]
        self.y = [label2id[l] for _, l in rows]

    def __len__(self):
        return len(self.y)

    def __getitem__(self, i):
        return torch.tensor(self.X[i], dtype=torch.long), self.y[i]


# ─── 4. 模型 ───────────────────────────────────────────────────────────
class LangIDClassifier(nn.Module):
    """
    Embedding → RNN → 掩码 MaxPool → LayerNorm → Dropout → Linear

    MaxPool 前面那两行 mask 是为了让补齐位不参与池化：
    补齐位喂进去的是全 0 向量，RNN 照样会往前走，池化时就会把
    "读过一串 0" 的状态也算进来。masked_fill 成 -inf 后取最大值就干净了。
    """

    def __init__(self, vocab_size, num_classes,
                 embed_dim=EMBED_DIM, hidden_dim=HIDDEN_DIM, dropout=DROPOUT):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        #self.rnn = nn.RNN(embed_dim, hidden_dim, batch_first=True)
        #self.rnn = nn.LSTM(embed_dim, hidden_dim, batch_first=True)
        self.rnn = nn.LSTM(embed_dim, hidden_dim, batch_first=True,bidirectional=True)
        #self.norm = nn.LayerNorm(hidden_dim)
        self.norm = nn.LayerNorm(hidden_dim *2)
        self.dropout = nn.Dropout(dropout)
        #self.fc = nn.Linear(hidden_dim, num_classes)
        self.fc = nn.Linear(hidden_dim*2, num_classes)

    def forward(self, x):
        mask = (x != 0).unsqueeze(-1)              # (B, L, 1)  补齐位为 False
        h, _ = self.rnn(self.embedding(x))         # (B, L, H)
        h = h.masked_fill(~mask, -1e9)
        pooled = h.max(dim=1)[0]                   # (B, H)
        #pooled = self.dropout(self.norm(pooled))
        pooled = self.norm(pooled)
        return self.fc(pooled)                     # (B, C)  未过 softmax 的 logits


# ─── 5. 评测 ───────────────────────────────────────────────────────────
def predict(model, loader):
    model.eval()
    preds, golds = [], []
    with torch.no_grad():
        for X, y in loader:
            preds += model(X).argmax(dim=1).tolist()
            golds += y.tolist()
    return preds, golds


def macro_f1(preds, golds, num_classes):
    """每一类各自算 F1，再对类别取平均（少数类不会被大类淹掉）。"""
    f1s = []
    for c in range(num_classes):
        tp = sum(p == c and g == c for p, g in zip(preds, golds))
        fp = sum(p == c and g != c for p, g in zip(preds, golds))
        fn = sum(p != c and g == c for p, g in zip(preds, golds))
        if tp + fp + fn == 0:
            continue
        prec = tp / (tp + fp) if tp + fp else 0.0
        rec = tp / (tp + fn) if tp + fn else 0.0
        f1s.append(2 * prec * rec / (prec + rec) if prec + rec else 0.0)
    return sum(f1s) / len(f1s)


def confusion(preds, golds, labels):
    n = len(labels)
    mat = [[0] * n for _ in range(n)]
    for p, g in zip(preds, golds):
        mat[g][p] += 1
    return mat


def print_confusion(mat, labels, top=3):
    """只打印错的格子，按错误数排序 —— 课堂上看 zh/ja 那一格就够了。"""
    pairs = [(mat[g][p], labels[g], labels[p])
             for g in range(len(labels)) for p in range(len(labels)) if g != p]
    pairs.sort(reverse=True)
    print("  最常混淆的组合（真实 → 预测）：")
    for cnt, g, p in pairs[:top]:
        if cnt == 0:
            break
        print(f"    {g} → {p}: {cnt} 条")


# ─── 6. 训练 ───────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default=DATA_DIR)
    ap.add_argument("--vocab", default=VOCAB_PATH,
                    help="字符表文件（默认用课堂那张 vocab_v8.txt）")
    ap.add_argument("--epochs", type=int, default=EPOCHS)
    ap.add_argument("--per-class", type=int, default=0,
                    help="只用每类前 N 条训练，用来演示「数据少了会过拟合」")
    ap.add_argument("--threads", type=int, default=NUM_THREADS)
    ap.add_argument("--save", default="", help="保存模型到该路径，供 eval_langid.py 使用")
    args = ap.parse_args()

    torch.set_num_threads(args.threads)
    random.seed(SEED)
    torch.manual_seed(SEED)

    train_rows = load_split(args.data_dir, "train")
    dev_rows = load_split(args.data_dir, "dev")
    test_rows = load_split(args.data_dir, "test")

    # 数据里的标签必须和 LABELS 对得上，否则模型的输出格子根本不够用
    seen = sorted({l for _, l in train_rows})
    extra = [l for l in seen if l not in LABELS]
    if extra:
        raise SystemExit(
            f"数据里出现了 LABELS 之外的标签：{' '.join(extra)}\n"
            f"  当前 LABELS（{len(LABELS)} 类）：{' '.join(LABELS)}\n"
            f"  模型输出维度 = len(LABELS)，多出来的语种没有对应的输出格子。"
        )
    label2id = {l: i for i, l in enumerate(LABELS)}
    num_classes = len(LABELS)

    if args.per_class:
        by_label = {}
        for t, l in train_rows:
            by_label.setdefault(l, []).append(t)
        train_rows = [(t, l) for l, ts in by_label.items()
                      for t in random.Random(SEED).sample(ts, min(args.per_class, len(ts)))]
        random.Random(SEED).shuffle(train_rows)

    vocab = load_vocab_file(args.vocab)
    print(f"数据目录：{args.data_dir}")
    print(f"  训练 {len(train_rows)} 条 / 验证 {len(dev_rows)} 条 / 测试 {len(test_rows)} 条")
    print(f"  类别 {num_classes} 个：{' '.join(LABELS)}")
    print(f"  字符表 {len(vocab)} 个字符（来自 {os.path.basename(args.vocab)}）")
    print(f"  随机猜的准确率基准：{1/num_classes:.1%}")

    train_loader = DataLoader(TextDataset(train_rows, vocab, label2id),
                              batch_size=BATCH_SIZE, shuffle=True)
    dev_loader = DataLoader(TextDataset(dev_rows, vocab, label2id), batch_size=128)
    test_loader = DataLoader(TextDataset(test_rows, vocab, label2id), batch_size=128)

    model = LangIDClassifier(len(vocab), num_classes)
    print(f"  模型参数量：{sum(p.numel() for p in model.parameters()):,}\n")

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    t0 = time.time()
    for epoch in range(1, args.epochs + 1):
        model.train()
        total = 0.0
        for X, y in train_loader:
            logits = model(X)
            loss = criterion(logits, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total += loss.item()
        preds, golds = predict(model, dev_loader)
        acc = sum(p == g for p, g in zip(preds, golds)) / len(golds)
        print(f"Epoch {epoch:2d}/{args.epochs}  loss={total/len(train_loader):.4f}  "
              f"dev_acc={acc:.4f}")
    elapsed = time.time() - t0
    print(f"\n训练耗时：{elapsed:.0f} 秒（{args.epochs} 个 epoch，{args.threads} 线程）")

    # 测试集
    preds, golds = predict(model, test_loader)
    acc = sum(p == g for p, g in zip(preds, golds)) / len(golds)
    f1 = macro_f1(preds, golds, num_classes)
    print(f"\n测试集：accuracy = {acc:.4f}   macro-F1 = {f1:.4f}")
    print_confusion(confusion(preds, golds, LABELS), LABELS)

    # 逐类准确率
    print("\n  各语种准确率：")
    for c, lab in enumerate(LABELS):
        idx = [i for i, g in enumerate(golds) if g == c]
        hit = sum(preds[i] == c for i in idx)
        print(f"    {lab}: {hit}/{len(idx)} = {hit/max(1,len(idx)):.3f}")

    # 推理示例
    print("\n--- 推理示例 ---")
    model.eval()
    demos = [t for t, _ in test_rows[:4]]
    with torch.no_grad():
        for text in demos:
            logits = model(torch.tensor([encode(text, vocab)]))
            prob = torch.softmax(logits, dim=1)[0]
            pid = int(prob.argmax())
            print(f"  [{LABELS[pid]} {prob[pid]:.2f}]  {text[:50]}")

    if args.save:
        torch.save({"state_dict": model.state_dict(), "vocab": dict(vocab),
                    "labels": LABELS, "maxlen": MAX_LEN,
                    "embed_dim": EMBED_DIM, "hidden_dim": HIDDEN_DIM,
                    "dropout": DROPOUT}, args.save)
        print(f"\n模型已保存：{args.save}")


if __name__ == "__main__":
    main()


