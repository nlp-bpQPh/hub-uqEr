# week07 事件抽取

本目录包含两套实现。

## `bert_joint`

一个模型直接输出事件类型、触发词和论元，支持 `linear` 和 `crf`。

```bash
cd bert_joint
python run_all.py
```

## `bert_pipeline`

两阶段流水线：先分类事件类型，再按类型抽取触发词和论元。

```text
第一阶段：train_detect.jsonl
第二阶段：train_extract.jsonl
```

```bash
cd bert_pipeline
python run_all.py
```

## `llm_fewshot`

直接使用本地大语言模型做 few-shot 事件抽取，不训练模型。

```bash
cd llm_fewshot
python predict_examples.py --shots-per-type 1
```

两套实现都使用本地 BERT：

```text
~/Downloads/模型/bert/bert-base-chinese
```
