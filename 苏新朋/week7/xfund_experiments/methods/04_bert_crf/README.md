# BERT + CRF

沿用课程的 BERT 序列标注结构和手写 CRF。使用 bert-base-chinese 前 4 层预训练权重；与 BERT + Linear 使用相同数据拆分、标签表、序列长度和训练轮数。

运行：`python run.py`。默认学习软转移分数；硬 BIO 约束在相邻方法目录中只改解码、不重训。
