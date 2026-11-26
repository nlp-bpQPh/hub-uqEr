# Qwen 对比实验

本目录只比较三个配置：

1. `lora`：全部训练集，Qwen LoRA 微调 1 epoch。
2. `zero_shot_icl`：不训练，只提供任务说明和 15 个类别。
3. `few_shot_icl`：不训练，每类提供 1 个示例后直接分类。

LoRA 使用 Qwen2.5-0.5B-Instruct、rank 8、`q_proj/v_proj`、
最后有效 token 池化和交叉熵损失。

ICL 不自由生成文本，而是在提示词末尾只比较 15 个类别名称首 token
的概率，直接取最大概率类别。这样不会出现解释文本、重复感叹号或解析失败。

一条命令运行全部实验：

```bash
/opt/homebrew/anaconda3/envs/py312/bin/python qwen/run_experiments.py
```

默认使用 500 条分层验证样本，便于控制 ICL 推理时间，可用
`--eval-limit` 调整。LoRA 和 ICL 使用完全相同的验证子集。

结果分别保存到：

```text
outputs/qwen/lora/
outputs/qwen/zero_shot_icl/
outputs/qwen/few_shot_icl/
outputs/qwen/best_model/
outputs/qwen/summary.csv
```

ICL 不产生训练权重。`best_model` 保存唯一的 LoRA adapter 和分类头。

预测：

```bash
/opt/homebrew/anaconda3/envs/py312/bin/python qwen/qwen_predict.py \
  --model-dir outputs/qwen/best_model \
  "中国乒乓球队在世界杯决赛中夺得冠军"
```
