# LLM Few-shot 事件抽取

直接加载本地原始 `Qwen2.5-0.5B-Instruct`，通过 few-shot in-context learning
抽取事件。不训练模型，不加载 LoRA 或其他微调权重。

默认模型：

```text
~/.cache/modelscope/models/Qwen--Qwen2.5-0.5B-Instruct/snapshots/master
```

## 单条预测

```bash
python predict.py \
  --text "万讯自控股东傅宇晨办理股份质押，质押2290万股。"
```

## 示例评估

```bash
python predict_examples.py --shots-per-type 1
```

保存结果：

```bash
python predict_examples.py \
  --shots-per-type 1 \
  --output ../outputs/llm_fewshot/examples.json
```

只运行前 2 个示例：

```bash
python predict_examples.py --shots-per-type 1 --limit 2
```

Qwen2 在 macOS MPS 上默认使用 SDPA，可能触发 GQA 张量维度错误。脚本默认使用
`eager` attention。仍然失败时可显式使用 CPU：

```bash
python predict_examples.py --shots-per-type 1 --device cpu
```

模型只生成事件类型、触发词文字和论元文字。脚本再根据文字回到原文中查找
字符位置，不让模型直接生成不可靠的 `start/end`。
