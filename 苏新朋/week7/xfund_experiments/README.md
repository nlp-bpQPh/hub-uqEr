# XFUND 抽取方法验证

本实验用 XFUND 中文训练集与官方验证集，按课件方法验证 OCR 块级字段标签：`header`、`question`、`answer`、`other`。官方 train 有 149 份文档，官方 val 有 50 份文档。样本中的 `linking` 关系随预测保留，但本轮指标只评字段块分类，不报告键值关联分数。

数据是表单图片的 OCR 标注（含文本块和坐标），不是 GB 标准正文语料。结果只能说明这些方法在 XFUND 表单标签上的表现，不能直接推断 GB 标准字段抽取效果。

## 已实现的方法目录

- `methods/01_rules`：规则、正则、OCR 坐标启发式。
- `methods/02_bilstm_crf`：字符级 BiLSTM + 课程手写 CRF。
- `methods/03_bert_linear`：bert-base-chinese 前 4 层 + Linear（CPU 适配，复用预训练权重）。
- `methods/04_bert_crf`：bert-base-chinese 前 4 层 + 课程手写 CRF。
- `methods/05_bert_crf_hard_decode`：复用 CRF checkpoint，只启用 BIO 硬约束解码。
- `methods/06_llm_api_zero_shot`、`methods/07_llm_api_few_shot`：适配 XFUND 的课程 prompt 方法；需要 API 凭证才可运行。

每种方法的 checkpoint（如有）、`metrics.json` 和 `predictions.jsonl` 均留在各自的 `methods/<method>/outputs/`。

## 数据与指标口径

- train：`data/source/zh.train.json`；val：`data/source/zh.val.json`（官方发布标注）。
- train 内按文档随机留 20% 做选 epoch；官方 val 只用于最后报告。
- 主指标：`entity_micro.f1`（header/question/answer 三类实体块 micro-F1）；同时报告四类 macro-F1、块准确率和逐类分数。
- 所有模型使用相同文本块边界和官方 val。当前没有评估 question-answer link F1，也没有从原 PDF 图片重新跑 OCR。

## 环境注意

本机为 CPU-only。BiLSTM 可直接训练；BERT 使用公开 bert-base-chinese 权重。课程 CRF 文件通过相对路径复用 `../week7-信息抽取/code/common/crf.py`。API 方法不会保存或打印密钥。
