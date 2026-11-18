# -*- coding: utf-8 -*-
"""
train_v12.py —— 作业第二步：12 类训练

本目录自带一份 train_langid_cls.py（与课堂脚本逐字节相同），它是「8 类写死版」，
文件开头注释写明了「要换数据集/加语种时，这两处都得跟着改」：LABELS 与字符表路径。

这里不复制、不改写模型代码，只在外层把这三个常量换成 12 类版，
然后直接调用课堂脚本的 main()：
    模型结构 / TextDataset / 训练循环 / 测试报告 / 推理示例 / --save 存档
全部沿用课堂代码，没有第二份实现。

脚本、数据、模型全部在本目录内，week03 可以整体挪到任何地方直接跑。
--epochs / --per-class / --threads / --save 等参数照旧可用。

运行（在哪都行）：
    python train_v12.py
    python week03/train_v12.py --epochs 8
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)                     # 用本目录这份 train_langid_cls.py

import train_langid_cls as base              # noqa: E402

# 作业只改这三处：数据集、字符表、语种列表（8 类 + ko th vi tr）
base.DATA_DIR = os.path.join(HERE, "data", "langid", "v12")
base.VOCAB_PATH = os.path.join(HERE, "data", "langid", "vocab", "vocab_v12.txt")
base.LABELS = ["ar", "en", "es", "hi", "ja", "ko", "pt", "ru", "th", "tr", "vi", "zh"]

if __name__ == "__main__":
    # Windows 控制台默认是 GBK，课堂脚本的「推理示例」会打印印地语/泰文句子，
    # 不改编码会 UnicodeEncodeError 中断（只影响输出，不影响训练与存档）。
    if sys.stdout.encoding and sys.stdout.encoding.lower().replace("-", "") != "utf8":
        sys.stdout.reconfigure(encoding="utf-8")

    # 课堂脚本的 --save 是相对当前工作目录的；这里默认存到本目录，
    # 于是从任何路径运行都落到 week03/ 里，不会因为 cwd 不同而存丢。
    if not any(a == "--save" or a.startswith("--save=") for a in sys.argv[1:]):
        sys.argv += ["--save", os.path.join(HERE, "langid_rnn_v12.pt")]
    base.main()
