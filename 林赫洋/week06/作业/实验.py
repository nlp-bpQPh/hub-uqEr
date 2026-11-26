# coding:utf8
"""
week06 作业：换预训练模型，做分类方法的对比实验

一个脚本跑完三路方法，在同一批验证样本上比 Macro-F1：
    ① 规则 / 关键词      —— 不训练，看到关键词就判类
    ② TF-IDF + SVM       —— 按词的统计特征分类
    ③ 微调预训练模型     —— BERT-base / MacBERT / 小模型，只换模型这一个东西

用法（复制这一行就能跑）：
    D:\\Miniconda3\\envs\\py312\\python.exe 实验.py

想跑快一点（1 分钟内）：
    D:\\Miniconda3\\envs\\py312\\python.exe 实验.py --quick

想用全量数据、跑 3 轮（约 15 分钟，出正式报告用这个）：
    D:\\Miniconda3\\envs\\py312\\python.exe 实验.py --full
"""

import argparse
import json
import random
import sys
import time
from pathlib import Path

BASE = Path(__file__).resolve().parent
DATA = BASE / "data"

# TNEWS 的 15 类（顺序必须和官方一致，否则标签全错）
LABELS = ["故事", "文化", "娱乐", "体育", "财经",
          "房产", "汽车", "教育", "科技", "军事",
          "旅游", "国际", "证券", "农业", "电竞"]
NAME2ID = {n: i for i, n in enumerate(LABELS)}

# 三个预训练模型：换模型 = 只换这里的一行
MODELS = {
    "BERT-base": {"path": "models/bert-base-chinese", "params": "102M"},  # 基线 12 层 768 维
    "MacBERT":   {"path": "models/macbert-base",      "params": "102M"},  # 换预训练任务，同尺寸
    "小模型":     {"path": "models/roberta-l4-h256",   "params": "11M"},   # 4 层 256 维
}


# ───────────────────── 一、读数据：先去重，再抽同一批样本 ─────────────────────

def load_data():
    train = json.load(open(DATA / "train.json", encoding="utf-8"))
    val = json.load(open(DATA / "val.json", encoding="utf-8"))
    return train, val


def build_eval_set(train, val, n=500, seed=42):
    """验证集里有一部分和训练集完全一样（数据泄漏），先剔掉，再固定 seed 抽样。

    为什么要去重：TF-IDF 靠「词面像不像」判断，如果验证集里混着训练集原句，
    它等于在背答案，指标会虚高。所有方法必须在同一批干净样本上比才公平。
    """
    seen = {d["sentence"] for d in train}
    clean = [d for d in val if d["sentence"] not in seen]
    print(f"验证集 {len(val)} 条，其中 {len(val) - len(clean)} 条和训练集重复（已剔除）")
    return random.Random(seed).sample(clean, min(n, len(clean)))


# ───────────────────── 二、指标：准确率 + Macro-F1 ─────────────────────

def score(y_true, y_pred):
    """准确率：答对的比例。Macro-F1：每类先算 F1 再平均，长尾类不会被头部淹没。"""
    from sklearn.metrics import f1_score
    acc = sum(1 for a, b in zip(y_true, y_pred) if a == b) / len(y_true)
    return round(acc, 4), round(f1_score(y_true, y_pred, average="macro"), 4)


# ───────────────────── 三、方法①：规则 / 关键词 ─────────────────────

# 每个类放几个「看到就基本能定」的词；越具体的类越靠前
KEYWORDS = {
    "证券": ["券商", "证券", "基金", "期货", "A股", "港股", "涨停", "科创板"],
    "财经": ["股", "营收", "净利", "财报", "业绩", "市值", "融资", "亿元"],
    "军事": ["军事", "军演", "导弹", "航母", "战机", "国防", "阅兵"],
    "国际": ["美国", "总统", "联合国", "外交", "欧盟", "俄罗斯", "日本"],
    "房产": ["楼市", "地产", "房价", "楼盘", "物业", "土地"],
    "科技": ["芯片", "5G", "AI", "人工智能", "手机", "华为", "互联网", "大模型"],
    "汽车": ["汽车", "车型", "特斯拉", "比亚迪", "发动机"],
    "教育": ["高考", "学校", "学生", "教育部", "考研", "招生"],
    "农业": ["农业", "种植", "农民", "水稻", "养殖", "生猪", "粮食"],
    "旅游": ["旅游", "景区", "酒店", "自驾", "游客", "度假"],
    "电竞": ["电竞", "战队", "选手", "英雄联盟", "王者荣耀", "主播"],
    "体育": ["比赛", "冠军", "奥运", "联赛", "球员", "NBA", "足球"],
    "娱乐": ["明星", "电影", "综艺", "演唱会", "票房", "导演"],
    "文化": ["文化", "书法", "文物", "非遗", "展览", "博物馆"],
    "故事": ["故事", "回忆", "老人", "乡村", "小说", "情感"],
}


def run_rule(train, samples):
    """规则：从前往后匹配关键词，命中就定类；全都没命中就猜训练集里最多的类。

    「猜最多的类」= 什么都不会时的兜底策略，正好可以当成下限。
    """
    counts = {}
    for d in train:
        counts[d["label"]] = counts.get(d["label"], 0) + 1
    default = max(counts, key=counts.get)

    preds, hit = [], 0
    for s in samples:
        label = default
        for name, words in KEYWORDS.items():
            if any(w in s["sentence"] for w in words):
                label = NAME2ID[name]
                hit += 1
                break
        preds.append(label)
    print(f"  （关键词命中 {hit}/{len(samples)} 条，其余按「最常见类」兜底）")
    return preds


# ───────────────────── 四、方法②：TF-IDF + SVM ─────────────────────

def run_tfidf(train, samples):
    """TF-IDF：把每句话变成「词的重要性向量」，再用线性 SVM 画分界线。

    词的重要性 = 在这句话里出现得多（TF）× 在别处出现得少（IDF）。
    不用预训练模型，秒级训练、CPU 就行 —— 「数据封闭、要便宜」场景的基线。
    """
    import jieba
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.pipeline import Pipeline
    from sklearn.svm import LinearSVC

    def cut(text):
        return [w for w in jieba.cut(text) if w.strip()]

    pipe = Pipeline([
        ("tfidf", TfidfVectorizer(tokenizer=cut, token_pattern=None,
                                  ngram_range=(1, 2), min_df=2, sublinear_tf=True)),
        ("clf", LinearSVC(C=1.0, class_weight="balanced")),
    ])
    pipe.fit([d["sentence"] for d in train], [d["label"] for d in train])
    return pipe.predict([s["sentence"] for s in samples]).tolist()


# ───────────────────── 五、方法③：微调预训练模型（只换模型） ─────────────────────

def run_finetune(train, samples, model_path, epochs, batch_size=64, max_len=64, lr=2e-5, seed=42):
    """微调：拿一个已经读过海量中文的模型，在你的 15 类数据上再训几轮。

    和 TF-IDF 的本质区别：TF-IDF 只认「这个词出现过没有」，
    预训练模型在预训练阶段已经学过「词和词的搭配、上下文」，
    所以标注数据不多时它起点更高 —— 这就是「换模型」值得试的原因。

    换模型不改一行训练代码，只换 model_path 指向的目录。
    """
    import numpy as np
    import torch
    from torch.utils.data import DataLoader, Dataset
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    tok = AutoTokenizer.from_pretrained(model_path)

    class DS(Dataset):
        """把一句话变成 token id；一次批量 tokenize 比逐条快很多。"""

        def __init__(self, rows):
            self.items = []
            for i in range(0, len(rows), 1000):
                chunk = rows[i:i + 1000]
                enc = tok([r["sentence"] for r in chunk], truncation=True, max_length=max_len,
                          padding="max_length", return_tensors="pt")
                for j in range(len(chunk)):
                    self.items.append((enc["input_ids"][j], enc["attention_mask"][j], chunk[j]["label"]))

        def __len__(self):
            return len(self.items)

        def __getitem__(self, i):
            ids, mask, y = self.items[i]
            return ids, mask, torch.tensor(y)

    model = AutoModelForSequenceClassification.from_pretrained(
        model_path, num_labels=len(LABELS)).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)

    # 类别加权：15 类样本数差 20 多倍，不加权时少见的类会被完全忽略
    from sklearn.utils.class_weight import compute_class_weight
    cw = compute_class_weight("balanced", classes=np.arange(len(LABELS)),
                              y=[d["label"] for d in train])
    loss_fn = torch.nn.CrossEntropyLoss(weight=torch.tensor(cw, dtype=torch.float, device=device))

    loader = DataLoader(DS(train), batch_size=batch_size, shuffle=True, drop_last=True)
    for epoch in range(1, epochs + 1):
        model.train()
        t0, total = time.time(), 0.0
        for i, (ids, mask, y) in enumerate(loader):
            opt.zero_grad(set_to_none=True)
            # return_dict=True：这个老 config 里 return_dict=False，不传会拿到 tuple
            logits = model(input_ids=ids.to(device), attention_mask=mask.to(device),
                           return_dict=True).logits
            loss = loss_fn(logits, y.to(device))
            loss.backward()
            opt.step()
            total += loss.item()
            if i % 100 == 0:
                print(f"    第 {epoch} 轮 step {i}/{len(loader)} loss={loss.item():.3f}", flush=True)
        print(f"    第 {epoch} 轮完成，平均 loss={total / max(1, len(loader)):.3f}"
              f"，用时 {time.time() - t0:.0f}s", flush=True)

    model.eval()
    preds = []
    with torch.no_grad():
        for i in range(0, len(samples), 128):
            chunk = samples[i:i + 128]
            enc = tok([s["sentence"] for s in chunk], truncation=True, max_length=max_len,
                      padding=True, return_tensors="pt").to(device)
            preds.extend(model(**enc, return_dict=True).logits.argmax(-1).cpu().tolist())
    return preds


# ───────────────────── 六、主流程：一路一路跑，最后打一张表 ─────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true", help="只用 1.2 万条训练、1 轮（默认就是这个）")
    ap.add_argument("--full", action="store_true", help="全量 5.3 万条、3 轮（正式报告用）")
    ap.add_argument("--n_eval", type=int, default=500, help="评测多少条")
    args = ap.parse_args()

    # 0) 环境自检：跑错 python 时直接告诉你该用哪个
    missing = []
    for m in ("sklearn", "jieba", "torch", "transformers"):
        try:
            __import__(m)
        except ImportError:
            missing.append(m)
    if missing:
        print(f"当前 python 缺 {missing}")
        print(r"请用：D:\Miniconda3\envs\py312\python.exe 实验.py")
        sys.exit(1)

    train, val = load_data()
    samples = build_eval_set(train, val, args.n_eval)
    y_true = [s["label"] for s in samples]

    if args.full:
        epochs = 3
    else:
        train, epochs = train[:12000], 1
    print(f"训练 {len(train)} 条 / 评测 {len(samples)} 条 / 每个模型训 {epochs} 轮\n")

    results = []
    t0 = time.time()

    print("[1/3] 规则 / 关键词")
    preds = run_rule(train, samples)
    acc, f1 = score(y_true, preds)
    results.append(("规则/关键词", "-", acc, f1, f"{time.time() - t0:.0f}s"))

    print("\n[2/3] TF-IDF + SVM")
    t1 = time.time()
    preds = run_tfidf(train, samples)
    acc, f1 = score(y_true, preds)
    results.append(("TF-IDF+SVM", "-", acc, f1, f"{time.time() - t1:.0f}s"))

    print("\n[3/3] 微调预训练模型（换模型只改一行路径）")
    for name, info in MODELS.items():
        print(f"  → {name}")
        t2 = time.time()
        try:
            preds = run_finetune(train, samples, str(BASE / info["path"]), epochs)
        except (FileNotFoundError, OSError):
            print(f"     跳过：没找到 {info['path']}")
            continue
        acc, f1 = score(y_true, preds)
        results.append((name, info["params"], acc, f1, f"{time.time() - t2:.0f}s"))

    print("\n" + "=" * 58)
    print(f"{'方法':<16}{'参数量':>8}{'准确率':>10}{'Macro-F1':>11}{'耗时':>9}")
    print("-" * 58)
    for name, param, acc, f1, sec in results:
        print(f"{name:<16}{param:>8}{acc:>10}{f1:>11}{sec:>9}")
    print("=" * 58)
    print("Macro-F1 越高越好。要写的结论就两句：换模型涨了几个点、小模型掉了几个点。")


if __name__ == "__main__":
    main()
