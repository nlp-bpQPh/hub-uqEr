# -*- coding: utf-8 -*-
"""
train_transformer_langid.py —— 用 Transformer 识别 12 种语言

"""

import os
import csv
import copy
import math
import time
import random
import argparse
import unicodedata
from collections import Counter, OrderedDict

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# ─── 超参数（初学阶段先照着用，不要一上来就调）────────────────────────
SEED = 42
MAX_LEN = 40          # 每句话最多保留 40 个字符，短了补 0，长了截断
EMBED_DIM = 128       # 每个字符变成多少维的向量（必须能被 NUM_HEADS 整除）
NUM_HEADS = 4         # 注意力头数：4 个头 = 从 4 个角度看同一句话
NUM_LAYERS = 4        # Transformer 编码器堆几层（BERT-base 是 12 层）
FF_DIM = 256          # 前馈网络中间层维度，一般是 EMBED_DIM 的 2~4 倍
DROPOUT = 0.2         # 随机丢掉 20% 的神经元，防止过拟合
BATCH_SIZE = 64
LR = 1e-3
EPOCHS = 10
NUM_THREADS = 4       # CPU 线程数

# 12 个语种的编号顺序（模型输出的第 i 个格子对应 LABELS[i]）
LABELS = ["ar", "en", "es", "hi", "ja", "ko", "pt", "ru", "th", "tr", "vi", "zh"]

# 标签 -> 中文名，只用于打印，让结果好读
LANGUAGE_NAMES = {
    "ar": "阿拉伯语", "en": "英语", "es": "西班牙语", "hi": "印地语",
    "ja": "日语", "ko": "韩语", "pt": "葡萄牙语", "ru": "俄语",
    "th": "泰语", "tr": "土耳其语", "vi": "越南语", "zh": "中文",
}

# 数据路径：已从 week3 复制到本项目 data/langid/v12，整个项目可以单独拷走运行
HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_DATA_DIR = os.path.join(HERE, "data", "langid", "v12")

# 自己写几句 12 种语言的话，训练完直接喂给模型看效果
DEMO_TEXTS = [
    "今天天气真好，我们出去走走吧。",
    "I think this movie is really interesting.",
    "今日はとても良い天気ですね。",
    "오늘 날씨가 정말 좋네요.",
    "Сегодня очень хорошая погода.",
    "الطقس جميل جدا اليوم.",
    "आज मौसम बहुत अच्छा है।",
    "วันนี้อากาศดีมาก",
    "Hôm nay thời tiết rất đẹp.",
    "Bugün hava çok güzel.",
    "El niño come una manzana.",
    "Eu gosto muito de música brasileira.",
]


# ─── 1. 文本归一化 ────────────────────────────────────────────────────
def normalize(text):
    
    t = unicodedata.normalize("NFKC", text).lower()
    return "".join(c for c in t if unicodedata.category(c)[0] in ("L", "M"))


# ─── 2. 字符表：字符 -> 整数 id ──────────────────────────────────────
def build_char_vocab(sentences):
    
    freq = Counter()
    for s in sentences:
        freq.update(normalize(s))

    chars = sorted(freq.items(), key=lambda kv: (-kv[1], kv[0]))
    vocab = OrderedDict([("<PAD>", 0), ("<UNK>", 1)])
    for ch, _ in chars:
        vocab[ch] = len(vocab)
    return vocab


def encode(text, vocab, maxlen=MAX_LEN):
    """一句话 -> 40 个整数：逐字查表，截断，后面补 0。"""
    ids = [vocab.get(ch, 1) for ch in normalize(text)][:maxlen]
    ids += [0] * (maxlen - len(ids))
    return ids


# ─── 3. 读数据 ───────────────────────────────────────────────────────
def load_split(data_dir, name):
    """读 train.csv / dev.csv / test.csv，每行是 text,label 两列。"""
    path = os.path.join(data_dir, f"{name}.csv")
    rows = []
    with open(path, encoding="utf-8") as f:
        for r in csv.DictReader(f):
            rows.append((r["text"], r["label"]))
    return rows


class TextDataset(Dataset):
    """PyTorch 数据集：把文本提前编码成 id 序列，取的时候直接给张量。"""

    def __init__(self, rows, vocab, label2id):
        self.X = [encode(t, vocab) for t, _ in rows]
        self.y = [label2id[l] for _, l in rows]

    def __len__(self):
        return len(self.y)

    def __getitem__(self, i):
        return torch.tensor(self.X[i], dtype=torch.long), self.y[i]


# ─── 4. 位置编码 ─────────────────────────────────────────────────────
class PositionalEncoding(nn.Module):
    

    def __init__(self, d_model, max_len=MAX_LEN, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)

        pe = torch.zeros(max_len, d_model)                     # (max_len, d_model)
        pos = torch.arange(0, max_len).unsqueeze(1).float()    # (max_len, 1)
        div = torch.exp(torch.arange(0, d_model, 2).float()
                        * (-math.log(10000.0) / d_model))      # (d_model/2,)
        pe[:, 0::2] = torch.sin(pos * div)                     # 偶数维用 sin
        pe[:, 1::2] = torch.cos(pos * div)                     # 奇数维用 cos

        # register_buffer：跟着模型一起搬到 GPU/CPU，但不是可训练参数
        self.register_buffer("pe", pe.unsqueeze(0))            # (1, max_len, d_model)

    def forward(self, x):
        # x: (B, L, d_model)，只取前 L 个位置编码加上去
        return self.dropout(x + self.pe[:, :x.size(1)])


# ─── 5. 模型：Embedding + 位置编码 + Transformer + 池化 + 分类 ────────
class LangIDTransformer(nn.Module):
    

    def __init__(self, vocab_size, num_classes,
                 d_model=EMBED_DIM, nhead=NUM_HEADS, num_layers=NUM_LAYERS,
                 ff_dim=FF_DIM, dropout=DROPOUT, max_len=MAX_LEN):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=0)
        self.pos_enc = PositionalEncoding(d_model, max_len=max_len, dropout=dropout)

        # 一层编码器 = 多头自注意力 + 前馈网络（各自带残差和 LayerNorm）
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=ff_dim,
            dropout=dropout,
            batch_first=True,   # 输入形状用 (B, L, d_model)，和 Embedding 一致
            norm_first=True,    # Pre-LN：先归一化再算注意力，训练更稳
        )
        # enable_nested_tensor=False：初学阶段用普通张量更直观，也少一条警告
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers,
                                             enable_nested_tensor=False)

        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(d_model, num_classes)

    def forward(self, x):
        # x: (B, L) 的字符 id
        pad_mask = (x == 0)                       # True 表示这一位是补齐的 0

        h = self.embedding(x)                     # (B, L, d_model)
        h = self.pos_enc(h)                       # 加上位置信息

        # src_key_padding_mask：告诉注意力“补齐位不用看”，避免把 0 当成内容
        h = self.encoder(h, src_key_padding_mask=pad_mask)

        # 只对真实字符求平均，补齐位不参与
        keep = (~pad_mask).unsqueeze(-1)          # (B, L, 1)
        pooled = (h * keep).sum(dim=1) / keep.sum(dim=1).clamp(min=1)   # (B, d_model)

        pooled = self.dropout(self.norm(pooled))
        return self.fc(pooled)                    # (B, 12) 原始分数 logits


# ─── 6. 评测小工具 ───────────────────────────────────────────────────
def predict(model, loader):
    """在某个数据集上跑一遍，返回（预测结果, 正确答案）。"""
    model.eval()
    preds, golds = [], []
    with torch.no_grad():                         # 评测不需要算梯度
        for X, y in loader:
            preds += model(X).argmax(dim=1).tolist()
            golds += y.tolist()
    return preds, golds


def macro_f1(preds, golds, num_classes):
    """每个语种各算一个 F1，再取平均：小语种不会被大语种淹没。"""
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


def confusion_matrix(preds, golds, num_classes):
    """混淆矩阵：第 g 行第 p 列 = 真实是 g、却被猜成 p 的条数。"""
    mat = [[0] * num_classes for _ in range(num_classes)]
    for p, g in zip(preds, golds):
        mat[g][p] += 1
    return mat


def print_top_confusions(mat, labels, top=3):
    """只打印最容易混淆的几组，看起来不费劲。"""
    pairs = [(mat[g][p], labels[g], labels[p])
             for g in range(len(labels)) for p in range(len(labels)) if g != p]
    pairs.sort(reverse=True)
    print("  最容易混淆的组合（真实 → 预测）：")
    for cnt, g, p in pairs[:top]:
        if cnt == 0:
            break
        print(f"    {LANGUAGE_NAMES[g]}({g}) → {LANGUAGE_NAMES[p]}({p})：{cnt} 条")


# ─── 7. 训练 ─────────────────────────────────────────────────────────
def build_model(vocab_size, num_classes, cfg):
    """按配置造模型，训练和加载时共用，避免结构对不上。"""
    return LangIDTransformer(
        vocab_size, num_classes,
        d_model=cfg["embed_dim"], nhead=cfg["num_heads"], num_layers=cfg["num_layers"],
        ff_dim=cfg["ff_dim"], dropout=cfg["dropout"], max_len=cfg["max_len"],
    )


def run_predict(path, texts):
    """加载训练好的模型，对几句话做预测。"""
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    model = build_model(len(ckpt["vocab"]), len(ckpt["labels"]), ckpt["config"])
    model.load_state_dict(ckpt["state_dict"])
    model.eval()

    print(f"模型：{path}")
    for text in texts:
        ids = torch.tensor([encode(text, ckpt["vocab"], ckpt["config"]["max_len"])])
        with torch.no_grad():
            prob = torch.softmax(model(ids), dim=1)[0]
        pid = int(prob.argmax())
        label = ckpt["labels"][pid]
        print(f"  [{label} {LANGUAGE_NAMES[label]}  {prob[pid]:.2%}]  {text}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default=DEFAULT_DATA_DIR, help="v12 数据目录")
    ap.add_argument("--epochs", type=int, default=EPOCHS)
    ap.add_argument("--threads", type=int, default=NUM_THREADS)
    ap.add_argument("--save", default="langid_transformer_v12.pt", help="模型保存路径")
    ap.add_argument("--predict", default="", help="只做预测，例如 --predict \"今天天气真好\"")
    args = ap.parse_args()

    # 预测模式：不训练，直接加载已有模型
    if args.predict:
        run_predict(args.save, [args.predict])
        return

    torch.set_num_threads(args.threads)
    random.seed(SEED)
    torch.manual_seed(SEED)

    # ---------- 读数据 ----------
    train_rows = load_split(args.data_dir, "train")
    dev_rows = load_split(args.data_dir, "dev")
    test_rows = load_split(args.data_dir, "test")

    # 检查数据里的标签和 LABELS 是否完全一致，防止“输出格子不够”
    seen = sorted({l for _, l in train_rows})
    if seen != sorted(LABELS):
        raise SystemExit(f"数据标签 {seen} 与 LABELS {sorted(LABELS)} 不一致")

    label2id = {l: i for i, l in enumerate(LABELS)}
    num_classes = len(LABELS)

    # ---------- 字符表：只从训练集统计 ----------
    vocab = build_char_vocab([t for t, _ in train_rows])
    print(f"数据目录：{args.data_dir}")
    print(f"  训练 {len(train_rows)} 条 / 验证 {len(dev_rows)} 条 / 测试 {len(test_rows)} 条")
    print(f"  类别 {num_classes} 个：{' '.join(LABELS)}")
    print(f"  字符表 {len(vocab)} 个字符（只用训练集统计）")
    print(f"  随机猜的准确率基准：{1 / num_classes:.1%}")

    train_loader = DataLoader(TextDataset(train_rows, vocab, label2id),
                              batch_size=BATCH_SIZE, shuffle=True)
    dev_loader = DataLoader(TextDataset(dev_rows, vocab, label2id), batch_size=128)
    test_loader = DataLoader(TextDataset(test_rows, vocab, label2id), batch_size=128)

    # ---------- 造模型 ----------
    config = {
        "embed_dim": EMBED_DIM, "num_heads": NUM_HEADS, "num_layers": NUM_LAYERS,
        "ff_dim": FF_DIM, "dropout": DROPOUT, "max_len": MAX_LEN,
    }
    model = build_model(len(vocab), num_classes, config)
    print(f"  模型参数量：{sum(p.numel() for p in model.parameters()):,}\n")

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.01)

    best_acc, best_state = 0.0, None
    t0 = time.time()

    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0

        for X, y in train_loader:
            logits = model(X)                 # ① 前向：算出 12 个分数
            loss = criterion(logits, y)       # ② 算损失
            optimizer.zero_grad()             # ③ 清空上一轮梯度
            loss.backward()                   # ④ 反向传播
            # 梯度裁剪：Transformer 偶尔会有特别大的梯度，剪一下更稳
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()                  # ⑤ 更新参数
            total_loss += loss.item()

        # 每个 epoch 在验证集上看看效果，保存最好的一版
        preds, golds = predict(model, dev_loader)
        dev_acc = sum(p == g for p, g in zip(preds, golds)) / len(golds)
        if dev_acc > best_acc:
            best_acc = dev_acc
            best_state = copy.deepcopy(model.state_dict())
        print(f"Epoch {epoch:2d}/{args.epochs}  loss={total_loss / len(train_loader):.4f}  "
              f"dev_acc={dev_acc:.4f}")

    elapsed = time.time() - t0
    print(f"\n训练耗时：{elapsed:.0f} 秒（{args.epochs} 个 epoch，{args.threads} 线程）")

    # ---------- 用验证集上最好的那一版去测测试集 ----------
    if best_state is not None:
        model.load_state_dict(best_state)

    preds, golds = predict(model, test_loader)
    acc = sum(p == g for p, g in zip(preds, golds)) / len(golds)
    f1 = macro_f1(preds, golds, num_classes)
    print(f"\n测试集：accuracy = {acc:.4f}   macro-F1 = {f1:.4f}")

    print_top_confusions(confusion_matrix(preds, golds, num_classes), LABELS)

    print("\n  各语种准确率：")
    for c, lab in enumerate(LABELS):
        idx = [i for i, g in enumerate(golds) if g == c]
        hit = sum(preds[i] == c for i in idx)
        print(f"    {lab}({LANGUAGE_NAMES[lab]}): {hit}/{len(idx)} = {hit / max(1, len(idx)):.3f}")

    # ---------- 保存模型（字符表、标签、结构参数一起存）----------
    save_path = os.path.join(HERE, args.save) if not os.path.isabs(args.save) else args.save
    torch.save({
        "state_dict": model.state_dict(),
        "vocab": dict(vocab),
        "labels": LABELS,
        "config": config,
    }, save_path)
    print(f"\n模型已保存：{save_path}")

    # ---------- 手写 12 句话，现场看看效果 ----------
    print("\n--- 12 语种现场预测 ---")
    run_predict(save_path, DEMO_TEXTS)


if __name__ == "__main__":
    main()
