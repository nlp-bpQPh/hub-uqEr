# 字符级 BiLSTM-CRF

复用课程手写 CRF，输入为 XFUND OCR 块中的字符序列，预测 `header/question/answer/other` 的 BIO 标签。训练在字符级进行，评测将字符预测聚合回 XFUND OCR 块。

运行：`python run.py`。默认使用 train 内部 20% 文档选 epoch，最终只在官方 `zh.val` 上报告。
