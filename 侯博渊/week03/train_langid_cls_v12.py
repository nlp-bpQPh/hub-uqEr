# -*- coding: utf-8 -*-
"""
train_langid_cls_v12.py —— 多语种「语种识别」分类器（v12 作业版 · 12 语种 + 双向 LSTM）

任务：一句话 → 它是什么语言（12 分类）
模型：Embedding → 双向 LSTM → 掩码 MaxPool → LayerNorm → Dropout → Linear → Softmax
损失：CrossEntropyLoss      优化器：Adam
数据：data/langid/v12（课堂 8 语种 + ko / th / vi / tr）
运行：CPU 即可。

    python train_langid_cls_v12.py                          # 12 语种，首次运行自动建字符表
    python train_langid_cls_v12.py --save langid_bilstm_v12.pt
    python train_langid_cls_v12.py --rebuild-vocab          # 强制重算 vocab_v12.txt

【评测】训练脚本末尾自带 test 指标；也可以交给 eval_langid.py：
    python eval_langid.py --model langid_bilstm_v12.pt --data-dir data/langid/v12
    python eval_langid.py --model langid_bilstm_v12.pt --data-dir data/langid/v12 --split test_hard
    python eval_langid.py --model langid_bilstm_v12.pt --data-dir data/langid/v12 --openset

【和课堂版 train_langid_cls.py 的四处不同】
  1. LABELS 从 8 个扩到 12 个，模型输出维度 = len(LABELS)
  2. 数据目录换成 data/langid/v12
  3. 字符表 vocab_v12.txt 随课件没有发，脚本第一次运行会从 v12/train.csv 统计生成
  4. 模型构建模块把单向 RNN 换成双向 LSTM（nn.LSTM(bidirectional=True)）
     · 每个时间步输出 [正向 h_t ; 反向 h_t]，维度是 2 * hidden_dim
     · 因此 LayerNorm / Linear 的输入维度都要用 hidden_dim * 2
     · 用 pack_padded_sequence 让反向分支也只读真实字符，不读补齐的 0

【为什么语种识别适合双向】
  语种线索不只在句首：词尾、后缀、假名/谚文/泰文的位置分布都很有用。
  单向 RNN 只能从前往后看；双向 LSTM 同时从后往前看，再把两个方向拼起来，
  对短句和语序差异更稳。代价是参数量和计算量都大约翻倍。
"""

import os
import sys
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
HIDDEN_DIM = 64       # 每个方向 64；双向拼接后特征维度是 128
DROPOUT = 0.3
BATCH_SIZE = 64
LR = 1e-3
EPOCHS = 15
NUM_THREADS = 1       # 小模型单线程往往比多线程快，别一上来就开满

HERE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(HERE, "data", "langid", "v12")
VOCAB_PATH = os.path.join(HERE, "data", "langid", "vocab", "vocab_v12.txt")

# 12 个语种 = 课堂 8 个 + 作业新增 4 个；模型的输出就是这 12 个格子
LABELS = ["ar", "en", "es", "hi", "ja", "pt", "ru", "zh",
          "ko", "th", "vi", "tr"]


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

    v12 的字符表没有随课件发，所以本脚本第一次运行时会调用它，
    把统计结果写成和 vocab_v8.txt 一样的「字符 \\t id \\t 出现次数」文件。
    返回 (vocab, freq)：freq 是每个字符的原始出现次数，写文件时要用。
    """
    freq = Counter()
    for s in sentences:
        freq.update(normalize(s))
    chars = sorted(((c, n) for c, n in freq.items() if n >= min_freq),
                   key=lambda kv: (-kv[1], kv[0]))
    vocab = OrderedDict([("<PAD>", 0), ("<UNK>", 1)])
    for ch, _ in chars:
        vocab[ch] = len(vocab)
    return vocab, freq


def save_vocab(vocab, freq, path):
    """按 vocab_v8.txt 的格式写文件，方便人类查看、也方便下一次直接 load。"""
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write("# 字符\tid\t出现次数\n")
        for ch, idx in vocab.items():
            f.write(f"{ch}\t{idx}\t{freq.get(ch, 0)}\n")


def load_vocab_file(path):
    """读字符表（每行「字符 \\t id \\t 出现次数」，以 # 开头的是注释）。"""
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


# ─── 4. 模型（本页作业的核心改动：RNN → 双向 LSTM）────────────────────
class LangIDClassifier(nn.Module):
    """
    Embedding → 双向 LSTM → 掩码 MaxPool → LayerNorm → Dropout → Linear

    nn.LSTM(bidirectional=True) 的输出形状是 (B, L, 2H)：
    最后一维前半段是正向 LSTM 在 t 时刻的隐藏状态，后半段是反向 LSTM 在 t 时刻的隐藏状态。
    池化把这 L 个时间步压成一个 2H 向量，再送进分类头。

    为什么要 pack_padded_sequence：
    反向 LSTM 是从序列末尾往前读的。如果直接喂补齐到 40 的矩阵，
    反向分支第一步读到的就是补齐的 0，真实字符的状态会被这些 0 污染。
    pack 之后两个方向都只读每条样本的真实长度，pad_packed_sequence 再把结果补回矩阵。

    MaxPool 前面那行 mask 仍然要保留：pack 回来的矩阵里，补齐位是 0 向量，
    池化时不能把它们当成「特征 0」参与取最大值，要填成 -inf 排掉。
    """

    def __init__(self, vocab_size, num_classes,
                 embed_dim=EMBED_DIM, hidden_dim=HIDDEN_DIM, dropout=DROPOUT):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, batch_first=True,
                            bidirectional=True)
        self.norm = nn.LayerNorm(hidden_dim * 2)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim * 2, num_classes)

    def forward(self, x):
        mask = (x != 0).unsqueeze(-1)                        # (B, L, 1) 补齐位为 False
        lengths = mask.squeeze(-1).sum(dim=1).cpu()          # (B,) 每条样本的真实长度
        emb = self.embedding(x)                              # (B, L, E)
        packed = nn.utils.rnn.pack_padded_sequence(
            emb, lengths, batch_first=True, enforce_sorted=False)
        h, _ = self.lstm(packed)                             # (B, L, 2H)
        h, _ = nn.utils.rnn.pad_packed_sequence(
            h, batch_first=True, total_length=x.size(1))
        h = h.masked_fill(~mask, -1e9)
        pooled = h.max(dim=1)[0]                             # (B, 2H)
        pooled = self.dropout(self.norm(pooled))
        return self.fc(pooled)                               # (B, C) 未过 softmax 的 logits


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


def print_confusion(mat, labels, top=5):
    """只打印错的格子，按错误数排序 —— 12 类时看前 5 个就够了。"""
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
    # Windows 控制台/管道默认是 GBK，打印阿拉伯文、印地文、泰文示例会直接
    # UnicodeEncodeError；统一按 UTF-8 输出（控制台本身支持 UTF-8，管道也不会再炸）。
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default=DATA_DIR)
    ap.add_argument("--vocab", default=VOCAB_PATH,
                    help="字符表文件（默认 data/langid/vocab/vocab_v12.txt）")
    ap.add_argument("--rebuild-vocab", action="store_true",
                    help="强制从训练集重算字符表并覆盖 --vocab 指向的文件")
    ap.add_argument("--epochs", type=int, default=EPOCHS)
    ap.add_argument("--per-class", type=int, default=0,
                    help="只用每类前 N 条训练，用来演示「数据少了会过拟合」")
    ap.add_argument("--threads", type=int, default=NUM_THREADS)
    ap.add_argument("--save", default="", help="保存模型到该路径")
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
    missing = [l for l in LABELS if l not in seen]
    if missing:
        print(f"[警告] 训练集里没有这些标签的样本：{' '.join(missing)}（对应的输出格子学不到东西）")

    label2id = {l: i for i, l in enumerate(LABELS)}
    num_classes = len(LABELS)

    # ── 字符表：v12 的表没有随课件发，第一次运行时从 train.csv 现场统计 ──
    need_build = args.rebuild_vocab or not os.path.exists(args.vocab)
    if need_build:
        vocab, freq = build_char_vocab([t for t, _ in train_rows])
        save_vocab(vocab, freq, args.vocab)
        print(f"已根据训练集生成字符表：{args.vocab}（{len(vocab)} 个字符）")
    else:
        vocab = load_vocab_file(args.vocab)

    if args.per_class:
        by_label = {}
        for t, l in train_rows:
            by_label.setdefault(l, []).append(t)
        train_rows = [(t, l) for l, ts in by_label.items()
                      for t in random.Random(SEED).sample(ts, min(args.per_class, len(ts)))]
        random.Random(SEED).shuffle(train_rows)

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

    # 推理示例：每个语种抽一条，12 个格子都看一眼
    print("\n--- 推理示例（每语种一条）---")
    model.eval()
    demos = []
    for lab in LABELS:
        for text, gold in test_rows:
            if gold == lab:
                demos.append((lab, text))
                break
    with torch.no_grad():
        for lab, text in demos:
            logits = model(torch.tensor([encode(text, vocab)]))
            prob = torch.softmax(logits, dim=1)[0]
            pid = int(prob.argmax())
            mark = "命中" if LABELS[pid] == lab else "错判"
            print(f"  {mark}  真实={lab:<2} 预测=[{LABELS[pid]} {prob[pid]:.2f}]  {text[:45]}")

    if args.save:
        torch.save({"state_dict": model.state_dict(), "vocab": dict(vocab),
                    "labels": LABELS, "maxlen": MAX_LEN,
                    "embed_dim": EMBED_DIM, "hidden_dim": HIDDEN_DIM,
                    "dropout": DROPOUT, "arch": "bilstm",
                    "bidirectional": True}, args.save)
        print(f"\n模型已保存：{args.save}")


if __name__ == "__main__":
    main()
