# BERT 两阶段事件流水线

## 第一阶段

使用 `train_detect.jsonl` 训练 13 维多标签事件检测：

```text
BERT → Linear(768, 13) → sigmoid
```

多个事件类型可以同时输出，无事件表现为 13 个概率全部低于阈值。

## 第二阶段

使用 `train_extract.jsonl` 训练事件类型条件抽取器：

```text
BERT 文本表示 + 事件类型 embedding
→ Linear(768, 125)
→ 触发词和论元 BIO 标签
```

125 个标签由以下部分构成：

```text
O                         1
触发词 B/I                 2
61 个论元角色 × B/I      122
```

推理时第一阶段输出事件类型，第二阶段按每个类型执行一次抽取。

## 数据切分

按文章 `id` 生成一次训练/验证切分，第一阶段和第二阶段共用，避免同一文章的
标题和正文同时出现在训练集与验证集。

## 训练

```bash
python train_stage1.py
python train_stage2.py --decoder linear
python train_stage2.py --decoder crf
```

一次训练全部组件：

```bash
python run_all.py
```

快速自检：

```bash
python run_all.py --smoke
```

## 预测

```bash
python predict.py \
  --stage1-dir ../outputs/pipeline/two_stage/stage1 \
  --stage2-dir ../outputs/pipeline/two_stage/stage2/linear \
  --text "公司股东将部分股份办理了质押。"
```

批量示例：

```bash
python predict_examples.py \
  --stage1-dir ../outputs/pipeline/two_stage/stage1 \
  --stage2-dir ../outputs/pipeline/two_stage/stage2/linear
```

