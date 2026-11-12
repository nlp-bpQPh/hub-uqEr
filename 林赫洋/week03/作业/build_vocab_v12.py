# -*- coding: utf-8 -*-
"""
build_vocab_v12.py —— 作业第一步：生成 12 类字符表

本目录自带一份 build_langid_dataset.py（与课堂脚本逐字节相同），
这里直接复用其中生成 vocab_v8.txt 的两个函数，不重写逻辑：
    build_char_vocab()  统计字符频次并编号（<PAD>=0、<UNK>=1，其余按频次降序）
    save_vocab()        写成「字符 \\t id \\t 出现次数」+ 首行注释的格式

脚本、数据、字符表全部在本目录内，week03 可以整体挪到任何地方直接跑。

运行（在哪都行）：
    python build_vocab_v12.py
    python week03/build_vocab_v12.py

产出：
    week03/data/langid/vocab/vocab_v12.txt
"""
import os
import csv
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)                     # 用本目录这份 build_langid_dataset.py

from build_langid_dataset import build_char_vocab, save_vocab  # noqa: E402

TRAIN_CSV = os.path.join(HERE, "data", "langid", "v12", "train.csv")
OUT_PATH = os.path.join(HERE, "data", "langid", "vocab", "vocab_v12.txt")


def main():
    with open(TRAIN_CSV, encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    texts = [r["text"] for r in rows]
    vocab, freq = build_char_vocab(texts)     # 与 vocab_v8.txt 同一套逻辑
    save_vocab(vocab, freq, OUT_PATH)

    print(f"训练集 {len(rows)} 条，标签 {sorted({r['label'] for r in rows})}")
    print(f"字符表 {len(vocab)} 个字符 -> {OUT_PATH}")


if __name__ == "__main__":
    main()
