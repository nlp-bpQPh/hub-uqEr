# -*- coding: utf-8 -*-
"""
train_langid_cls.py —— 多语种「语种识别」分类器

任务：一句话 → 它是什么语言（ 12 分类）
模型结构可配置：Embedding → RNN/LSTM/BiLSTM → 掩码 MaxPool → 可选 LayerNorm/Dropout → Linear
损失：CrossEntropyLoss      优化器：Adam
默认数据：data/langid/v12（12 语种）

    python train_langid_cls.py
    python train_langid_cls.py --data-dir data/langid/v8
    python train_langid_cls.py --epochs 8 --save langid_rnn_v12.pt

模型变体：--model-type rnn|lstm|bilstm；--no-layernorm / --no-dropout 可单独消融模块。
支持标签：ar/en/es/hi/ja/ko/pt/ru/th/tr/vi/zh。类别顺序由标签表统一管理，

"""

import os
import csv
import time
import random
import argparse
import unicodedata
from collections import Counter, OrderedDict

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

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
DATA_DIR = os.path.join(HERE, "data", "langid", "v12")
VOCAB_DIR = os.path.join(HERE, "data", "langid", "vocab")

# 稳定的 12 类输出顺序。使用 v8 数据时，会自动只保留该数据实际包含的 8 类。
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

    训练时会优先加载匹配数据版本的字符表；字符表不存在时，自动从当前训练集生成，
    因此新增语种时应使用覆盖其字符的 v12 字符表，或重新构建字符表。
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
    """读取字符表文件；每行是「字符\tid\t出现次数」，以 # 开头的是注释。"""
    vocab = OrderedDict()
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.startswith("#") or not line.strip():
                continue
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 2:
                raise ValueError(f"字符表格式错误：{path}，行内容：{line!r}")
            ch, idx = parts[:2]
            vocab[ch] = int(idx)
    if vocab.get("<PAD>") != 0 or vocab.get("<UNK>") != 1:
        raise ValueError(f"字符表必须包含 <PAD>=0 和 <UNK>=1：{path}")
    return vocab


def save_vocab_file(vocab, sentences, path):
    """保存字符表和字符频次，格式与数据构建脚本保持一致。"""
    freq = Counter()
    for sentence in sentences:
        freq.update(normalize(sentence))
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write("# 字符\tid\t出现次数\n")
        for ch, idx in vocab.items():
            f.write(f"{ch}\t{idx}\t{freq.get(ch, 0)}\n")

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
    """可切换循环单元和消融 LayerNorm/Dropout 的语种分类器。

    model_type:
        rnn    —— 单向 vanilla RNN（基线）
        lstm   —— 单向 LSTM
        bilstm —— 双向 LSTM，前向/后向隐藏状态拼接后池化
    use_layernorm / use_dropout 用于独立消融实验。其余网络组件不变。

    MaxPool 前使用 mask，避免补齐位置参与池化；双向模型的输出通道数为
    hidden_dim * 2，因此分类层和 LayerNorm 的维度会自动相应调整。
    """

    def __init__(self, vocab_size, num_classes,
                 embed_dim=EMBED_DIM, hidden_dim=HIDDEN_DIM, dropout=DROPOUT,
                 model_type="rnn", use_layernorm=True, use_dropout=True):
        super().__init__()
        model_type = model_type.lower()
        if model_type not in {"rnn", "lstm", "bilstm"}:
            raise ValueError(f"不支持的 model_type: {model_type}，应为 rnn/lstm/bilstm")
        self.model_type = model_type
        self.use_layernorm = bool(use_layernorm)
        self.use_dropout = bool(use_dropout)
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        bidirectional = model_type == "bilstm"
        if model_type == "rnn":
            self.rnn = nn.RNN(embed_dim, hidden_dim, batch_first=True)
        else:
            self.rnn = nn.LSTM(embed_dim, hidden_dim, batch_first=True,
                               bidirectional=bidirectional)
        output_dim = hidden_dim * (2 if bidirectional else 1)
        self.norm = nn.LayerNorm(output_dim) if self.use_layernorm else nn.Identity()
        self.dropout = nn.Dropout(dropout) if self.use_dropout else nn.Identity()
        self.fc = nn.Linear(output_dim, num_classes)

    def forward(self, x):
        mask = (x != 0).unsqueeze(-1)              # (B, L, 1)  补齐位为 False
        embedded = self.embedding(x)
        if self.model_type == "bilstm":
            # 双向序列必须屏蔽右侧 padding，否则反向 LSTM 会先读入补齐位，
            # 再读真实字符，导致真实字符的反向状态受填充长度影响。
            lengths = (x != 0).sum(dim=1).clamp(min=1).cpu()
            packed = pack_padded_sequence(embedded, lengths, batch_first=True,
                                          enforce_sorted=False)
            packed_h, _ = self.rnn(packed)
            h, _ = pad_packed_sequence(packed_h, batch_first=True,
                                       total_length=x.size(1))
        else:
            h, _ = self.rnn(embedded)              # (B, L, H)
        h = h.masked_fill(~mask, -1e9)
        pooled = h.max(dim=1)[0]                   # (B, H) 或 (B, 2H)
        pooled = self.dropout(self.norm(pooled))
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
    ap.add_argument("--data-dir", default=DATA_DIR,
                    help="数据目录，默认 data/langid/v12")
    ap.add_argument("--vocab", default="",
                    help="可选字符表路径；默认按数据目录自动选择，缺失时从训练集生成")
    ap.add_argument("--epochs", type=int, default=EPOCHS)
    ap.add_argument("--seed", type=int, default=SEED,
                    help="随机种子；模型对比实验应让所有组使用相同种子")
    ap.add_argument("--model-type", choices=["rnn", "lstm", "bilstm"], default="rnn",
                    help="循环结构：rnn（基线）、lstm（单向）或 bilstm（双向 LSTM）")
    ap.add_argument("--no-layernorm", action="store_true",
                    help="消融实验：不使用 LayerNorm，其余设置保持不变")
    ap.add_argument("--no-dropout", action="store_true",
                    help="消融实验：不使用 Dropout，其余设置保持不变")
    ap.add_argument("--metrics-out", default="",
                    help="可选：将最终评测指标写入 JSON 文件，供对比脚本汇总")
    ap.add_argument("--per-class", type=int, default=0,
                    help="只用每类前 N 条训练，用来演示「数据少了会过拟合」")
    ap.add_argument("--threads", type=int, default=NUM_THREADS)
    ap.add_argument("--save", default="", help="模型保存路径；默认根据数据版本保存为 langid_rnn_v12.pt / langid_rnn_v8.pt")
    args = ap.parse_args()

    torch.set_num_threads(args.threads)
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    train_rows = load_split(args.data_dir, "train")
    dev_rows = load_split(args.data_dir, "dev")
    test_rows = load_split(args.data_dir, "test")
    if not train_rows:
        raise SystemExit(f"训练集为空，请检查数据目录：{args.data_dir}")

    # 输出类别按稳定顺序排列，但只启用所选数据集中真正存在的标签。
    seen = {l for _, l in train_rows}
    extra = sorted(seen - set(LABELS))
    if extra:
        raise SystemExit(
            f"训练数据包含项目未配置的标签：{' '.join(extra)}\n"
            f"当前支持的 12 类：{' '.join(LABELS)}"
        )
    labels = [label for label in LABELS if label in seen]
    label2id = {label: i for i, label in enumerate(labels)}
    num_classes = len(labels)

    for split_name, rows in (("dev", dev_rows), ("test", test_rows)):
        unknown = sorted({label for _, label in rows} - set(label2id))
        if unknown:
            raise SystemExit(
                f"{split_name}.csv 存在训练集中未出现的标签：{' '.join(unknown)}；"
                "请确保 train/dev/test 的类别一致。"
            )

    if args.per_class:
        by_label = {}
        for t, l in train_rows:
            by_label.setdefault(l, []).append(t)
        train_rows = [(t, l) for l, ts in by_label.items()
                      for t in random.Random(args.seed).sample(ts, min(args.per_class, len(ts)))]
        random.Random(args.seed).shuffle(train_rows)

    profile = os.path.basename(os.path.normpath(args.data_dir))
    if args.vocab:
        vocab_path = args.vocab
    elif profile in ("v8", "v12"):
        vocab_path = os.path.join(VOCAB_DIR, f"vocab_{profile}.txt")
    else:
        vocab_path = os.path.join(args.data_dir, "vocab.txt")

    if os.path.isfile(vocab_path):
        vocab = load_vocab_file(vocab_path)
    else:
        # 新数据或缺失字符表时，直接根据训练集建表，避免新语种字符全部变成 <UNK>。
        vocab = build_char_vocab([text for text, _ in train_rows])
        save_vocab_file(vocab, [text for text, _ in train_rows], vocab_path)
        print(f"字符表不存在，已根据训练集自动生成：{vocab_path}")

    print(f"数据目录：{args.data_dir}")
    print(f"  训练 {len(train_rows)} 条 / 验证 {len(dev_rows)} 条 / 测试 {len(test_rows)} 条")
    print(f"  类别 {num_classes} 个：{' '.join(labels)}")
    print(f"  字符表 {len(vocab)} 个字符（来自 {vocab_path}）")
    print(f"  随机猜的准确率基准：{1/num_classes:.1%}")

    # 独立的 DataLoader generator 确保不同模型从相同随机种子、相同批次顺序开始。
    loader_generator = torch.Generator().manual_seed(args.seed)
    train_loader = DataLoader(TextDataset(train_rows, vocab, label2id),
                              batch_size=BATCH_SIZE, shuffle=True,
                              generator=loader_generator)
    dev_loader = DataLoader(TextDataset(dev_rows, vocab, label2id), batch_size=128)
    test_loader = DataLoader(TextDataset(test_rows, vocab, label2id), batch_size=128)

    model = LangIDClassifier(
        len(vocab), num_classes, model_type=args.model_type,
        use_layernorm=not args.no_layernorm,
        use_dropout=not args.no_dropout,
    )
    parameter_count = sum(p.numel() for p in model.parameters())
    print(f"  模型结构：{args.model_type} / LayerNorm={not args.no_layernorm} / "
          f"Dropout={not args.no_dropout}")
    print(f"  模型参数量：{parameter_count:,}\n")

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

    # 保留最终 epoch 的验证与测试成绩；五组对比使用相同训练轮数和数据划分。
    dev_preds, dev_golds = predict(model, dev_loader)
    dev_acc = sum(p == g for p, g in zip(dev_preds, dev_golds)) / len(dev_golds)
    preds, golds = predict(model, test_loader)
    acc = sum(p == g for p, g in zip(preds, golds)) / len(golds)
    f1 = macro_f1(preds, golds, num_classes)
    print(f"\n验证集：accuracy = {dev_acc:.4f}")
    print(f"测试集：accuracy = {acc:.4f}   macro-F1 = {f1:.4f}")
    print_confusion(confusion(preds, golds, labels), labels,
                    top=min(5, max(1, num_classes - 1)))

    # 逐类准确率
    print("\n  各语种准确率：")
    for c, lab in enumerate(labels):
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
            print(f"  [{labels[pid]} {prob[pid]:.2f}]  {text[:50]}")

    suffix = args.model_type
    if args.no_layernorm:
        suffix += "_no_layernorm"
    if args.no_dropout:
        suffix += "_no_dropout"
    save_path = args.save or os.path.join(
        HERE, f"langid_{suffix}_{profile if profile in ('v8', 'v12') else 'custom'}.pt")
    os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
    torch.save({"state_dict": model.state_dict(), "vocab": dict(vocab),
                "labels": labels, "maxlen": MAX_LEN,
                "embed_dim": EMBED_DIM, "hidden_dim": HIDDEN_DIM,
                "dropout": DROPOUT, "model_type": args.model_type,
                "use_layernorm": not args.no_layernorm,
                "use_dropout": not args.no_dropout,
                "seed": args.seed}, save_path)
    print(f"\n模型已保存：{save_path}")

    if args.metrics_out:
        import json
        metrics_path = os.path.abspath(args.metrics_out)
        os.makedirs(os.path.dirname(metrics_path), exist_ok=True)
        class_accuracy = {}
        for c, lab in enumerate(labels):
            idx = [i for i, g in enumerate(golds) if g == c]
            class_accuracy[lab] = (sum(preds[i] == c for i in idx) / len(idx)) if idx else None
        metrics = {
            "model_type": args.model_type,
            "use_layernorm": not args.no_layernorm,
            "use_dropout": not args.no_dropout,
            "num_parameters": parameter_count,
            "train_seconds": round(elapsed, 4),
            "epochs": args.epochs,
            "seed": args.seed,
            "num_classes": num_classes,
            "train_size": len(train_rows),
            "dev_size": len(dev_rows),
            "test_size": len(test_rows),
            "dev_accuracy": dev_acc,
            "test_accuracy": acc,
            "macro_f1": f1,
            "per_class_accuracy": class_accuracy,
            "model_path": os.path.abspath(save_path),
        }
        with open(metrics_path, "w", encoding="utf-8") as f:
            json.dump(metrics, f, ensure_ascii=False, indent=2)
        print(f"指标已保存：{metrics_path}")


if __name__ == "__main__":
    main()
