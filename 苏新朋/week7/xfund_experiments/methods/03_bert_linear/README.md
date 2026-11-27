# BERT + Linear

沿用课程的 BERT 逐字 BIO 头与 `word_ids` 对齐思想，使用公开 `bert-base-chinese` 的前 4 层预训练权重（对应课件层数消融），Linear 分类头预测 XFUND 标签。CPU 环境统一跑 2 epoch、160 字符分块。

运行：`python run.py`。模型首次运行会从 Hugging Face 下载。训练、内层选模、官方验证集和预测文件均保存在 `outputs/`。
