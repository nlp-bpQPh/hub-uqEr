# BERT + CRF 硬 BIO 解码

复用 `04_bert_crf/outputs/best.pt`，在相同官方验证集上只启用课程 CRF 的硬转移约束，不做第二次训练。

运行：先训练 `04_bert_crf`，再执行 `python run.py`。这是课程“CRF 软约束 vs 解码硬约束”的对照。
