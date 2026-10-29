# -*- coding: utf-8 -*-
# build_vocab_v12.py —— 一次性生成 12 语种字符表（跑完可删）
import os
from collections import Counter
from train_langid_cls import build_char_vocab, load_split, normalize

HERE = os.path.dirname(os.path.abspath(__file__))

# 1. 读 v12 训练集（只用训练集建字符表，测试集不能"偷看"）
rows = load_split(os.path.join(HERE, "data", "langid", "v12"), "train")
sentences = [t for t, _ in rows]

# 2. 复用训练脚本的 build_char_vocab，保证归一化逻辑和训练时完全一致
vocab = build_char_vocab(sentences)   # OrderedDict: {字符: id}

# 3. 补一个频次统计，只为让第三列"出现次数"写出来好看（读表时会忽略它）
freq = Counter()
for s in sentences:
    freq.update(normalize(s))

# 4. 写成和 vocab_v8.txt 一样的格式：字符 \t id \t 出现次数
out = os.path.join(HERE, "data", "langid", "vocab", "vocab_v12.txt")
os.makedirs(os.path.dirname(out), exist_ok=True)
with open(out, "w", encoding="utf-8") as f:
    f.write("# 字符\tid\t出现次数\n")
    for ch, idx in vocab.items():
        f.write(f"{ch}\t{idx}\t{freq.get(ch, 0)}\n")

print(f"已生成 {out}，共 {len(vocab)} 个字符")