# 事件抽取

模型使用一个共享 BERT 编码器和一个 149 类输出头。事件类型直接写入触发词标签，例如 `B-企业收购-触发词`，不再通过 `[CLS]` 单独分类。

标签组成：

```text
O                                      1
13 种事件类型 × B/I 触发词              26
61 个论元角色 × B/I                    122
合计                                   149
```

默认 `max_length=256`，并按 batch 动态 padding。可以先运行数据分析：

```bash
python analyze_data.py
```

训练开始时会默认打印 3 条训练样本和 2 条验证样本，包括：

- 原始文本
- token 与 BIO 标签
- 截断状态
- 从标签反解出的触发词和论元字符区间

关闭或调整日志数量：

```bash
python train.py --decoder linear --data-debug-count 0
python train.py --decoder linear --data-debug-count 5
```

## 训练

```bash
python train.py --decoder linear
python train.py --decoder crf
```

两个方案一起训练并生成对比：

```bash
python run_all.py
```

默认结果位于：

```text
../outputs/experiments/linear_crf/
├── linear/
├── crf/
├── logs/
├── comparison.json
├── comparison.csv
└── comparison.md
```

后续两段式模型只要提供相同格式的 `meta.json` 和 `metrics.json`，就可以追加到同一张对比表：

```bash
python run_all.py \
  --compare-only \
  --run-name linear_crf \
  --extra-result two_stage=../outputs/two_stage
```

快速自检：

```bash
python train.py --decoder linear --smoke
python train.py --decoder crf --smoke
```

组合脚本的快速自检：

```bash
python run_all.py --smoke
```

## 预测

```bash
python predict.py \
  --model-dir ../outputs/linear \
  --text "公司股东傅宇晨将部分股份办理了质押业务。"
```

批量运行内置示例：

```bash
python predict_examples.py \
  --model-dir ../outputs/experiments/linear_crf/linear
```

输出为格式化 JSON，包含：

```json
{
  "id": "example_01",
  "text": "原始文本",
  "event_type": "事件类型",
  "trigger": {
    "text": "触发词",
    "start": 10,
    "end": 12
  },
  "args": [
    {
      "role": "论元角色",
      "text": "论元文本",
      "start": 20,
      "end": 24
    }
  ]
}
```

`start` 和 `end` 是原文字符区间，采用左闭右开形式：`text[start:end]` 就是实体文本。

保存示例预测：

```bash
python predict_examples.py \
  --model-dir ../outputs/experiments/linear_crf/linear \
  --output ./examples_prediction.json
```

## 指标计算位置

- `model.py::EventTagger.decode`：逐位置 argmax 或调用 CRF 解码
- `crf.py::CRF.decode`：Viterbi 最优标签路径
- `data.py::extract_token_spans`：BIO 标签还原为触发词和论元 span
- `train.py::build_records`：把预测和标签转成可比较的 span 集合
- `train.py::_prf`：计算触发词和论元的 precision、recall、F1
- `train.py::compute_metrics`：计算类型准确率、类型 macro-F1、span F1 和完整事件匹配率

`argmax` 只负责把 token 分数解码成标签，不是准确率。正式指标先按原文字符位置还原
span，再计算完全匹配和 F1：

- `event_type_accuracy`：事件类型预测正确的样本比例
- `trigger_exact_match`：触发词位置完全一致的样本比例
- `arg_exact_match`：全部论元角色和字符位置完全一致的样本比例
- `event_exact_match`：事件类型、触发词和全部论元同时完全一致

模型默认指定以下权重目录：

```text
~/Downloads/模型/bert/bert-base-chinese
```
