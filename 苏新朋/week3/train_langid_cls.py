# -*- coding: utf-8 -*-
"""
train_langid_cls.py —— 多语种「语种识别」分类器（综合实战脚本）

任务：一句话 → 它是什么语言（分类问题，类别数 = len(LABELS)）
模型：Embedding → RNN → 掩码 MaxPool → LayerNorm → Dropout → Linear → Softmax
损失：CrossEntropyLoss      优化器：Adam
数据：data/langid/v12（作业版，12 个语种；课堂原版是 v8 的 8 个语种）
运行：CPU 即可。

    python train_langid_cls.py                           # 用默认数据训练
    python train_langid_cls.py --save langid_rnn_v12.pt  # 训练完把模型存下来

【本课想让学生看到的四件事】
  1. 同一份代码、同一张字符表，换数据就能处理多种语言文字
  2. 字符级不需要分词器：中文按字、英文按字母、印地语按字母+元音标记，一视同仁
  3. 组件是一个个加进去的：去掉 LayerNorm / Dropout / RNN 都能现场对比
  4. 错得最有意思的是 es ↔ pt（同一套拉丁字母），不是"模型没学会"

【语种列表和字符表都是写死的】
  · LABELS 直接列出语种，模型的输出维度 = len(LABELS)
  · 字符表不从训练集现算，而是读 data/langid/vocab/vocab_v12.txt
  要换数据集/加语种时，这两处都得跟着改（作业就是把 8 类扩成 12 类）。

【字符表是怎么来的】见 build_char_vocab()：
  把训练集每句话按字拆开 → 统计每个字的出现次数 → 按次数从高到低编号。
  文件每行「字符 \\t id \\t 出现次数」。

【整体流程一览】（先有全局图，再看细节）

    原始句子 "Hola, ¿cómo estás?"
        │  normalize()：NFKC 归一化、转小写、去掉标点空格数字
        ▼
    "holacómoestás"
        │  encode()：查字符表，每个字符 → 一个整数；截断/补 0 到固定长度
        ▼
    [13, 6, 10, 3, 15, ...,  0, 0, 0]        ← 形状 (L,)
        │  DataLoader 打包成一批
        ▼
    张量 X：(B, L)                            ← B 句话，每句 L 个字符编号
        │  模型 forward()
        ▼
    logits：(B, C)                            ← 每句话在 C 个语种上的打分
        │  CrossEntropyLoss 和真实标签比较 → 反向传播 → Adam 更新参数
        ▼
    重复若干个 epoch，模型越来越准
"""

import os
import csv
import time
import random
import argparse
import unicodedata                      # Python 自带的 Unicode 工具，用来做归一化和判断字符类别
from collections import Counter, OrderedDict

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader


# ═══════════════════════════════════════════════════════════════════════
# 0. 超参数 —— "训练前就要定好、训练中不会被学习"的那些数字
# ═══════════════════════════════════════════════════════════════════════
SEED = 42             # 随机种子：固定后每次运行结果一样，方便对比实验

MAX_LEN = 50          # 每句话最多取多少个字符。
                      # 为什么要固定长度？因为一批数据要拼成一个矩阵 (B, L)，每行必须一样长。
                      # 短的句子后面补 0（<PAD>），长的直接截断。
                      # 取多长合适：中文中位长度约 10 字符、德语约 39，太短会丢信息，太长浪费计算。

EMBED_DIM = 128       # 每个字符被表示成多少维的向量（字符的"身份证"有多长）
HIDDEN_DIM = 128      # RNN 隐藏状态的维度（模型"记忆"的容量）
DROPOUT = 0.3         # 训练时随机让 30% 的神经元"休息"，防止模型死记硬背（过拟合）
BATCH_SIZE = 64       # 一次喂给模型 64 句话，算一次平均梯度再更新
LR = 1e-3             # 学习率：每次更新参数迈多大步。Adam 的经典默认值就是 0.001
EPOCHS = 24           # 把整个训练集完整过多少遍

NUM_THREADS = 2       # CPU 线程数。小模型线程多了反而慢（线程调度开销 > 计算量），
                      # 实测单线程比 16 线程快 2.7 倍，别改成 16

# 路径：HERE 是本脚本所在目录，这样无论在哪个目录下运行脚本，都能找到数据
HERE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(HERE, "data", "langid", "v12")
VOCAB_PATH = os.path.join(HERE, "data", "langid", "vocab", "vocab_v12.txt")

# 语种标签列表：模型最后一层就输出这么多个"格子"，每个格子是一个语种的得分。
# 列表里的顺序 = 类别编号：ar 是 0 号，en 是 1 号……tr 是 11 号。
#   课堂原版 8 种：ar 阿拉伯语 / en 英语 / es 西班牙语 / hi 印地语
#                  ja 日语 / pt 葡萄牙语 / ru 俄语 / zh 中文
#   作业新增 4 种：ko 韩语 / th 泰语 / vi 越南语 / tr 土耳其语
LABELS = [ "ar" , "en" , "es" , "hi" , "ja" , "pt" , "ru" , "zh" , "ko" , "th" , "vi" , "tr" ]



# ═══════════════════════════════════════════════════════════════════════
# 1. 文本归一化 —— 把"长得不一样但意思一样"的字符统一掉
# ═══════════════════════════════════════════════════════════════════════
def normalize(text):
    """
    NFKC 归一化 + 转小写 + 只保留「字母(L)」与「组合符号(M)」。

    逐步拆解，以 "Ｈｅｌｌｏ, World! 123" 为例：
      ① NFKC：全角 "Ｈｅｌｌｏ" → 半角 "Hello"；兼容字符统一成标准形式
      ② lower()：→ "hello, world! 123"   （大小写对判断语种没帮助，合并能减少字符表大小）
      ③ 过滤：Unicode 给每个字符都标了类别，首字母表示大类：
             L = Letter 字母（含汉字、假名、阿拉伯字母……）
             M = Mark   组合标记（附着在字母上的符号，如元音标记）
             P = 标点、N = 数字、Z = 空格 …… 这些全扔掉
         → "helloworld"

    为什么扔掉标点和数字？—— 它们在各语言里基本通用，对判断语种没有区分度，反而是噪声。

    为什么一定要保留 M：印地语、泰语的元音是组合标记（कि 里的 ि），
    只留字母会把印地语砍成裸辅音，模型就没法区分。

    ⚠ 训练和预测必须用同一个 normalize，否则模型看到的字符分布对不上。
    """
    t = unicodedata.normalize("NFKC", text).lower()
    # unicodedata.category(c) 返回如 "Lu"(大写字母)、"Mn"(非间距标记)，取 [0] 就是大类
    return "".join(c for c in t if unicodedata.category(c)[0] in ("L", "M"))


# ═══════════════════════════════════════════════════════════════════════
# 2. 字符表 —— 神经网络只认数字，所以要给每个字符发一个编号
# ═══════════════════════════════════════════════════════════════════════
def build_char_vocab(sentences, min_freq=1):
    """
    字符表 = 训练集里所有出现过的字符，每个字符发一个整数编号。

    第 0 号 <PAD>：补齐用（Embedding 里 padding_idx=0，向量恒为 0、不参与训练）
    第 1 号 <UNK>：没见过的字符（字符级下几乎用不到）
    其余按出现次数从高到低编号，次数相同按字符编码排序（保证可复现）

    例：训练集只有 ["aab", "ba"] →
        频次 a:3, b:2 → 字符表 {<PAD>:0, <UNK>:1, a:2, b:3}

    多语种的关键点：这张表不需要按语种分开建。汉字、拉丁字母、西里尔字母、
    阿拉伯字母、天城文、泰文全都排在同一张表里 —— 这就是"字符级天然多语种"。

    主流程没有直接调用它（字符表是从 vocab 文件读的，见 build_vocab_v12.py）；
    一旦换了训练集或加了语种，就要用它重新生成一张表，
    否则新语种的字符会全部落到 1 号 <UNK>，模型等于睁眼瞎。
    """
    freq = Counter()
    for s in sentences:
        freq.update(normalize(s))            # 字符串本身可迭代，update 会逐字符计数
    # 排序 key：(-次数, 字符) → 次数降序；次数相同按字符升序，保证每次结果一致
    chars = sorted(((c, n) for c, n in freq.items() if n >= min_freq),
                   key=lambda kv: (-kv[1], kv[0]))
    vocab = OrderedDict([("<PAD>", 0), ("<UNK>", 1)])
    for ch, _ in chars:
        vocab[ch] = len(vocab)               # 新编号 = 当前表长度，即依次 2, 3, 4...
    return vocab


def load_vocab_file(path):
    """
    读已经生成好的字符表（每行「字符 \\t id \\t 出现次数」，以 # 开头的是注释）。

    为什么用现成文件而不是每次现算？
      · 训练和推理必须用完全一样的编号，存成文件最稳妥
      · 课堂上大家用同一张表，结果可对照

    → 换了训练集、加了语种，这张表就得重新生成（用上面的 build_char_vocab）。
    """
    vocab = OrderedDict()
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.startswith("#") or not line.strip():   # 跳过注释行和空行
                continue
            ch, idx, _freq = line.rstrip("\n").split("\t") # 出现次数这里用不到，用 _ 开头表示忽略
            vocab[ch] = int(idx)
    return vocab


def encode(text, vocab, maxlen=MAX_LEN):
    """
    一句话 → 固定长度的整数列表。

    例（假设 maxlen=6）："Hola" → normalize → "hola"
        → 查表 [13, 6, 10, 3] → 补 0 → [13, 6, 10, 3, 0, 0]
    """
    # vocab.get(ch, 1)：查不到的字符给 1 号 <UNK>；[:maxlen] 超长截断
    ids = [vocab.get(ch, 1) for ch in normalize(text)][:maxlen]
    ids += [0] * (maxlen - len(ids))         # 不够长就在末尾补 0 号 <PAD>
    return ids


# ═══════════════════════════════════════════════════════════════════════
# 3. 读数据 —— 从 CSV 到 PyTorch 能吃的 Dataset
# ═══════════════════════════════════════════════════════════════════════
def load_split(data_dir, name):
    """
    读 train.csv / dev.csv / test.csv，返回 [(句子, 语种), ...]

    三份数据的分工（机器学习的标准做法）：
      train 训练集：模型拿来学习、更新参数
      dev   验证集：每个 epoch 后看一眼效果，用来调超参数（模型不从它学习）
      test  测试集：最后只考一次，代表真实水平（调参时绝对不能偷看）
    """
    path = os.path.join(data_dir, f"{name}.csv")
    rows = []
    with open(path, encoding="utf-8") as f:
        for r in csv.DictReader(f):          # DictReader 按表头读，每行是 {"text":..., "label":...}
            rows.append((r["text"], r["label"]))
    return rows


class TextDataset(Dataset):
    """
    PyTorch 的 Dataset 只需要实现两个方法：
      __len__     ：一共多少条数据
      __getitem__ ：取第 i 条，返回 (输入, 标签)
    DataLoader 会自动调用它们，帮我们打乱、分批。

    这里在 __init__ 里一次性把所有句子编码好，训练时直接取，省得每个 epoch 重复编码。
    """
    def __init__(self, rows, vocab, label2id):
        self.X = [encode(t, vocab) for t, _ in rows]   # 句子 → 整数列表
        self.y = [label2id[l] for _, l in rows]        # "zh" → 7 这样的类别编号

    def __len__(self):
        return len(self.y)

    def __getitem__(self, i):
        # 输入必须是 long（整数）类型，因为 Embedding 要拿它当下标查表
        return torch.tensor(self.X[i], dtype=torch.long), self.y[i]


# ═══════════════════════════════════════════════════════════════════════
# 4. 模型 —— 本课核心
# ═══════════════════════════════════════════════════════════════════════
class LangIDClassifier(nn.Module):
    """
    Embedding → RNN → 掩码 MaxPool → LayerNorm → Dropout → Linear

    各层的作用和张量形状变化（B=批大小, L=句长, E=EMBED_DIM, H=HIDDEN_DIM, C=类别数）：

      输入 x            (B, L)      整数编号
        │ Embedding     查表：每个编号换成一个 E 维向量（这些向量是学出来的）
        ▼               (B, L, E)
        │ RNN           从左到右逐字读，每读一个字更新一次"记忆"
        ▼               (B, L, H)   每个位置都输出一个隐藏状态
        │ MaxPool       在 L 个位置上，每一维取最大值 → 把整句压成一个向量
        ▼               (B, H)      "这句话里最显著的特征"
        │ LayerNorm     把数值拉回均值 0、方差 1 附近，训练更稳定
        │ Dropout       训练时随机置零一部分，防过拟合
        │ Linear        H 维 → C 维，每一维是一个语种的得分
        ▼               (B, C)      logits

    为什么用 MaxPool 而不是只取 RNN 最后一个输出？
      最后一个位置可能是补齐的 <PAD>；而且简单 RNN 记性差，读到句尾前面的信息已经淡了。
      MaxPool 让句子里任何位置的关键特征（比如出现了一个假名"の"）都能被保留下来。

    MaxPool 前面那两行 mask 是为了让补齐位不参与池化：
    补齐位喂进去的是全 0 向量，RNN 照样会往前走，池化时就会把
    "读过一串 0" 的状态也算进来。masked_fill 成 -1e9 后取最大值就干净了。
    """

    def __init__(self, vocab_size, num_classes,
                 embed_dim=EMBED_DIM, hidden_dim=HIDDEN_DIM, dropout=DROPOUT):
        super().__init__()                   # 必须先调用父类初始化，PyTorch 才能注册下面的子层
        # padding_idx=0：0 号 <PAD> 的向量固定为全 0，且不接收梯度（补齐位不该学到东西）
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        # batch_first=True：输入输出的形状是 (B, L, ...)，否则默认是 (L, B, ...)
        self.rnn = nn.RNN(embed_dim, hidden_dim, batch_first=True)
        self.norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim, num_classes)   # fc = fully connected 全连接层

    def forward(self, x):
        # x != 0 → 真实字符为 True，补齐位为 False；unsqueeze(-1) 加一维，方便和 h 对齐广播
        mask = (x != 0).unsqueeze(-1)              # (B, L, 1)  补齐位为 False
        # nn.RNN 返回两个值：所有位置的输出 h，以及最后时刻的隐藏状态（这里不用，丢给 _）
        h, _ = self.rnn(self.embedding(x))         # (B, L, H)
        # ~mask 取反 → 补齐位为 True，把这些位置填成极小值，取 max 时它们永远不会被选中
        h = h.masked_fill(~mask, -1e9)
        # max 返回 (最大值, 下标)，只要 [0] 最大值
        pooled = h.max(dim=1)[0]                   # (B, H)
        pooled = self.dropout(self.norm(pooled))
        # 注意这里不做 softmax：CrossEntropyLoss 内部自带 softmax，再加一次就算重复了
        return self.fc(pooled)                     # (B, C)  未过 softmax 的 logits


# ═══════════════════════════════════════════════════════════════════════
# 5. 评测 —— 模型到底学得怎么样
# ═══════════════════════════════════════════════════════════════════════
def predict(model, loader):
    """在一份数据上跑一遍，返回 (预测类别列表, 真实类别列表)。"""
    model.eval()                     # 切到评估模式：关闭 Dropout（预测时要用上全部神经元）
    preds, golds = [], []
    with torch.no_grad():            # 不记录计算图、不算梯度 → 省内存、更快
        for X, y in loader:
            # argmax(dim=1)：在 C 个语种得分里挑最高的那个作为预测
            preds += model(X).argmax(dim=1).tolist()
            golds += y.tolist()
    return preds, golds


def macro_f1(preds, golds, num_classes):
    """
    每一类各自算 F1，再对类别取平均（少数类不会被大类淹掉）。

    针对某一类 c（比如 "zh"）：
      TP 真阳性：真的是 zh，也预测成 zh        ✔
      FP 假阳性：其实不是 zh，却预测成 zh      ✘ 误报
      FN 假阴性：真的是 zh，却预测成别的       ✘ 漏报
      精确率 precision = TP / (TP+FP)   "说是 zh 的里面，有多少真是 zh"
      召回率 recall    = TP / (TP+FN)   "所有 zh 里面，找出来了多少"
      F1 = 两者的调和平均，两个都高 F1 才高

    为什么不只看 accuracy？如果某语种数据特别多，模型只要猜它就能拿高准确率，
    macro-F1 给每个语种同等权重，能暴露出"小语种全错"的问题。
    """
    f1s = []
    for c in range(num_classes):
        tp = sum(p == c and g == c for p, g in zip(preds, golds))
        fp = sum(p == c and g != c for p, g in zip(preds, golds))
        fn = sum(p != c and g == c for p, g in zip(preds, golds))
        if tp + fp + fn == 0:        # 这一类既没出现也没被预测，跳过不计入平均
            continue
        prec = tp / (tp + fp) if tp + fp else 0.0
        rec = tp / (tp + fn) if tp + fn else 0.0
        f1s.append(2 * prec * rec / (prec + rec) if prec + rec else 0.0)
    return sum(f1s) / len(f1s)


def confusion(preds, golds, labels):
    """
    混淆矩阵：mat[真实][预测] = 条数。
    对角线 = 预测对的；非对角线 = 搞混的。看非对角线就知道模型把谁当成了谁。
    """
    n = len(labels)
    mat = [[0] * n for _ in range(n)]
    for p, g in zip(preds, golds):
        mat[g][p] += 1
    return mat


def print_confusion(mat, labels, top=3):
    """只打印错的格子，按错误数排序 —— 典型的是 es↔pt、zh↔ja 这几格。"""
    pairs = [(mat[g][p], labels[g], labels[p])
             for g in range(len(labels)) for p in range(len(labels)) if g != p]  # g != p：只要非对角线
    pairs.sort(reverse=True)         # 按错误条数从多到少
    print("  最常混淆的组合（真实 → 预测）：")
    for cnt, g, p in pairs[:top]:
        if cnt == 0:
            break
        print(f"    {g} → {p}: {cnt} 条")


# ═══════════════════════════════════════════════════════════════════════
# 6. 训练主流程 —— 把前面所有零件串起来
# ═══════════════════════════════════════════════════════════════════════
def main():
    # ── 6.1 命令行参数：不改代码就能换配置做实验 ──
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default=DATA_DIR)
    ap.add_argument("--vocab", default=VOCAB_PATH,
                    help="字符表文件（默认 vocab_v12.txt）")
    ap.add_argument("--epochs", type=int, default=EPOCHS)
    ap.add_argument("--per-class", type=int, default=0,
                    help="只用每类前 N 条训练，用来演示「数据少了会过拟合」")
    ap.add_argument("--threads", type=int, default=NUM_THREADS)
    ap.add_argument("--save", default="", help="保存模型到该路径，供 eval_langid.py 使用")
    args = ap.parse_args()

    # ── 6.2 固定随机性：线程数 + 两处随机种子 ──
    torch.set_num_threads(args.threads)
    random.seed(SEED)                # Python 自带的随机（用于抽样、打乱）
    torch.manual_seed(SEED)          # PyTorch 的随机（参数初始化、Dropout、DataLoader 打乱）

    # ── 6.3 读三份数据 ──
    train_rows = load_split(args.data_dir, "train")
    dev_rows = load_split(args.data_dir, "dev")
    test_rows = load_split(args.data_dir, "test")

    # 安全检查：数据里的标签必须都在 LABELS 里，否则模型的输出格子根本不够用
    # （作业里加了新语种却忘了改 LABELS，就会在这里报错，而不是莫名其妙崩在后面）
    seen = sorted({l for _, l in train_rows})
    extra = [l for l in seen if l not in LABELS]
    if extra:
        raise SystemExit(
            f"数据里出现了 LABELS 之外的标签：{' '.join(extra)}\n"
            f"  当前 LABELS（{len(LABELS)} 类）：{' '.join(LABELS)}\n"
            f"  模型输出维度 = len(LABELS)，多出来的语种没有对应的输出格子。"
        )
    label2id = {l: i for i, l in enumerate(LABELS)}   # {"ar":0, "en":1, ...}
    num_classes = len(LABELS)

    # ── 6.4 （可选）缩小训练集，演示过拟合 ──
    # 例：--per-class 50 → 每个语种只留 50 条训练，会看到 loss 很低但 dev_acc 上不去
    if args.per_class:
        by_label = {}
        for t, l in train_rows:
            by_label.setdefault(l, []).append(t)      # 按语种分组
        train_rows = [(t, l) for l, ts in by_label.items()
                      for t in random.Random(SEED).sample(ts, min(args.per_class, len(ts)))]
        random.Random(SEED).shuffle(train_rows)       # 打乱，避免一批里全是同一个语种

    # ── 6.5 字符表 + 数据概况 ──
    vocab = load_vocab_file(args.vocab)
    print(f"数据目录：{args.data_dir}")
    print(f"  训练 {len(train_rows)} 条 / 验证 {len(dev_rows)} 条 / 测试 {len(test_rows)} 条")
    print(f"  类别 {num_classes} 个：{' '.join(LABELS)}")
    print(f"  字符表 {len(vocab)} 个字符（来自 {os.path.basename(args.vocab)}）")
    # 瞎猜的准确率 = 1/类别数，模型至少要远超这个数才算学到了东西
    print(f"  随机猜的准确率基准：{1/num_classes:.1%}")

    # ── 6.6 DataLoader：自动分批 ──
    # 只有训练集 shuffle=True：每个 epoch 打乱顺序，防止模型记住"数据的排列顺序"
    # 验证/测试只做预测、不算梯度，内存压力小，所以 batch 可以开大到 128
    train_loader = DataLoader(TextDataset(train_rows, vocab, label2id),
                              batch_size=BATCH_SIZE, shuffle=True)
    dev_loader = DataLoader(TextDataset(dev_rows, vocab, label2id), batch_size=128)
    test_loader = DataLoader(TextDataset(test_rows, vocab, label2id), batch_size=128)

    # ── 6.7 建模型、损失函数、优化器 ──
    model = LangIDClassifier(len(vocab), num_classes)
    # 参数量里大头是 Embedding：字符表大小 × EMBED_DIM
    print(f"  模型参数量：{sum(p.numel() for p in model.parameters()):,}\n")

    # 交叉熵 = 多分类的标准损失：先 softmax 成概率，再取正确类别概率的 -log
    # 正确类别概率越接近 1，损失越接近 0
    criterion = nn.CrossEntropyLoss()
    # Adam：会给每个参数自适应调整步长，比普通 SGD 更省心
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    # ── 6.8 训练循环 ──
    t0 = time.time()
    for epoch in range(1, args.epochs + 1):
        model.train()                # 切到训练模式：开启 Dropout
        total = 0.0
        for X, y in train_loader:
            # 训练"五步曲"，几乎所有 PyTorch 训练代码都是这个骨架：
            logits = model(X)                # ① 前向：算出预测
            loss = criterion(logits, y)      # ② 算损失：预测和答案差多远
            optimizer.zero_grad()            # ③ 清空上一批残留的梯度（PyTorch 默认会累加）
            loss.backward()                  # ④ 反向传播：算出每个参数的梯度
            optimizer.step()                 # ⑤ 更新参数：沿梯度反方向走一小步
            total += loss.item()             # .item() 把单元素张量转成 Python 数字
        # 每个 epoch 结束在验证集上看一眼
        # 正常现象：loss 持续下降、dev_acc 持续上升；
        # 如果 loss 还在降但 dev_acc 不涨甚至下降 → 过拟合了
        preds, golds = predict(model, dev_loader)
        acc = sum(p == g for p, g in zip(preds, golds)) / len(golds)
        print(f"Epoch {epoch:2d}/{args.epochs}  loss={total/len(train_loader):.4f}  "
              f"dev_acc={acc:.4f}")
    elapsed = time.time() - t0
    print(f"\n训练耗时：{elapsed:.0f} 秒（{args.epochs} 个 epoch，{args.threads} 线程）")

    # ── 6.9 期末考试：测试集（只在最后考一次）──
    preds, golds = predict(model, test_loader)
    acc = sum(p == g for p, g in zip(preds, golds)) / len(golds)
    f1 = macro_f1(preds, golds, num_classes)
    print(f"\n测试集：accuracy = {acc:.4f}   macro-F1 = {f1:.4f}")
    print_confusion(confusion(preds, golds, LABELS), LABELS)

    # 逐类准确率：看哪个语种拖了后腿
    print("\n  各语种准确率：")
    for c, lab in enumerate(LABELS):
        idx = [i for i, g in enumerate(golds) if g == c]   # 真实标签是这一类的样本下标
        hit = sum(preds[i] == c for i in idx)              # 其中预测对了几条
        print(f"    {lab}: {hit}/{len(idx)} = {hit/max(1,len(idx)):.3f}")  # max(1,..) 防止除以 0

    # ── 6.10 推理示例：真正"用"模型的样子 ──
    print("\n--- 推理示例 ---")
    model.eval()
    demos = [t for t, _ in test_rows[:4]]
    with torch.no_grad():
        for text in demos:
            # encode 得到一维列表，外面再套一层 [] 变成 batch=1 的 (1, L)
            logits = model(torch.tensor([encode(text, vocab)]))
            # 推理时才手动 softmax，把得分变成 0~1 的概率，方便看模型有多"自信"
            prob = torch.softmax(logits, dim=1)[0]
            pid = int(prob.argmax())
            print(f"  [{LABELS[pid]} {prob[pid]:.2f}]  {text[:50]}")

    # ── 6.11 保存模型 ──
    # 不只存权重 state_dict，还把字符表、标签、超参数一起存进去：
    # 加载时才能用完全一样的配置重建模型、用完全一样的编号编码输入。
    # langid_rnn_v12.pt 就是这样存出来的（用 torch.load 读出来正是这 7 个键）。
    if args.save:
        torch.save({"state_dict": model.state_dict(), "vocab": dict(vocab),
                    "labels": LABELS, "maxlen": MAX_LEN,
                    "embed_dim": EMBED_DIM, "hidden_dim": HIDDEN_DIM,
                    "dropout": DROPOUT}, args.save)
        print(f"\n模型已保存：{args.save}")


# 只有直接运行本文件时才执行 main()；被别的脚本 import 时不会自动开始训练
if __name__ == "__main__":
    main()
