# 基于 Qwen3-0.6B 的 SFT 训练

> 用课件第五部分讲的 SFT 方法，把只会「续写」的基座模型调成「被问就答、答完就停」的助手。
> 数据用课程提供的 `sft_train.jsonl`（3000 条），参数高效微调用 LoRA。

---

## 一、作业要做什么

给定 `(instruction, output)` 指令数据，对预训练语言模型做有监督微调（SFT），让它学会：

1. **看指令**：看到用户问题就作答，而不是顺着往下续写；
2. **守格式**：按 ChatML 的 `<|im_start|>user / assistant / <|im_end|>` 说话；
3. **会收尾**：答完生成 `<|im_end|>` 停下来。

结构一个新层都不加，只改数据、loss 落点，和课件第 27~29 页说的一致。

## 二、方法

### 2.1 数据格式（ChatML + Loss Mask）

```
<|im_start|>user
给定两个数字，请选择其中的较大数并将其打印出来。 数字1: 238 数字2: 564<|im_end|>
<|im_start|>assistant
较大的数字是 564。<|im_end|>
└──────────────────┬──────────────────┘  └────────┬────────┘
        labels = -100（不参与 loss）                labels = 原文（参与 loss）
```

- prompt 段 label 置 `-100`，只有回答段算交叉熵 —— 让模型学「怎么答」，而不是背下问题；
- 回答末尾的 `<|im_end|>` 也参与 loss，模型才会自己学会停；
- 统计：平均 51.7 个 token（提问 31.1 + 回答 20.6），**loss 覆盖率约 40%**。

### 2.2 训练设置

| 项目 | 取值 |
|---|---|
| 基座模型 | `Qwen3-0.6B-Base`（28 层、hidden 1024、词表 15.2 万） |
| 微调方式 | LoRA，r=16，alpha=32，dropout=0.05 |
| LoRA 作用层 | q/k/v/o_proj + gate/up/down_proj |
| 可训练参数 | **10.09 M / 606.14 M ≈ 1.67%** |
| 学习率 | 2e-4，cosine 衰减，warmup 3% |
| Batch | 8（无梯度累积），共 4 个 epoch |
| 精度 | bfloat16 |
| 硬件 | RTX 3060 12GB |
| 模型选择 | 每轮抽 20 条验证样本真实生成，按**字符级 F1** 选最优（不是只看 val_loss） |

> 为什么按生成 F1 选模型：这批数据是开放问答，同一个意思有很多种说法，teacher-forcing 的
> val_loss 会随过拟合先降后升，但「答得像不像」才是我们真正关心的。

## 三、结果

<!--RESULTS-->

## 四、目录结构

```
第五周作业-SFT训练/
├── config.py                # ChatML 模板、超参、LoRA 配置
├── requirements.txt
├── run_all.ps1              # 一键跑完 数据→训练→评估
├── data/
│   ├── sft_train.jsonl      # 课程给的 3000 条训练数据
│   ├── sft_val.jsonl        # 200 条验证数据
│   └── processed/           # 分词 + loss mask 后的 train.pt / val.pt
├── src/
│   ├── prepare_data.py      # ChatML + loss mask，落成 .pt
│   ├── train_sft.py         # LoRA SFT（含每轮生成打分）
│   ├── evaluate.py          # SFT 前 vs 后：loss / PPL / token acc / EM / 字符 F1
│   └── chat.py              # 和 SFT 后的模型对话（支持 --base 看基座）
├── models/
│   ├── Qwen3-0.6B-Base/     # 基座权重（1.14 GB）
│   └── qwen3-0.6b-sft-lora/ # 训练出来的 LoRA 适配器（39 MB）
└── outputs/
    ├── metrics.json         # SFT 前后的全部指标
    ├── compare.md           # 抽样对比表
    ├── generations.json     # 40 条完整生成结果
    ├── train_log.csv        # 每轮训练/验证指标
    └── curves.png           # 训练曲线
```

## 五、怎么跑

```bash
# 0) 依赖（本机 py312 环境已装好）
pip install -r requirements.txt

# 1) 数据：套 ChatML 模板 + 打 loss mask
python src/prepare_data.py

# 2) 训练（约 10 分钟 / RTX 3060）
python src/train_sft.py

# 3) 评估：SFT 前 vs 后
python src/evaluate.py --num_samples 40

# 4) 对话
python src/chat.py "给定两个数字，选择较大数。 数字1: 238 数字2: 564"
python src/chat.py                 # 交互式
python src/chat.py --base          # 只看基座模型（对比用）
```

> 默认 `python` 是 Anaconda base（3.13，没装 torch），请用
> `D:\tool\anaconda3\envs\py312\python.exe`，或先 `conda activate py312`。

## 六、关键代码位置

| 课件概念 | 代码 |
|---|---|
| ChatML 模板 | `config.build_prompt()` / `build_target()` |
| Loss Mask（`-100`） | `src/prepare_data.py: encode()` |
| 只有回答段算 loss | `train_sft.py` 里 labels 传进模型，HF 内部按 `ignore_index=-100` 算交叉熵 |
| 参数高效（LoRA） | `src/train_sft.py: LoraConfig + get_peft_model` |
| 「预训练给知识，SFT 给形式」 | 看 `outputs/compare.md`：同一套权重，SFT 前只会复读问题，SFT 后会作答 |
