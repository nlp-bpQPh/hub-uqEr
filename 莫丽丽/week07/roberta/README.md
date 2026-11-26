# RoBERTa 中文金融事件抽取

本目录提供基于 `hfl/chinese-roberta-wwm-ext` 的训练入口。它与 `bert/` 共用完全相同的 BIO 标签、CRF 结构、损失函数、子词对齐与评估逻辑，尽量只改变预训练编码器，以便和 BERT 做可解释的对照。

> `hfl/chinese-roberta-wwm-ext` 是中文 RoBERTa-wwm-ext 常用权重。Hugging Face 当前配置以 BERT 兼容结构注册（`model_type=bert`），但权重来自 RoBERTa-wwm 预训练方案。因此本文中的“换 RoBERTa”指更换中文 RoBERTa-wwm-ext 预训练 checkpoint，而不是把模型结构完全换成 XLM-R。代码通过 `AutoModel/AutoTokenizer` 加载，可兼容其他合适的预训练编码器。

## 1. 安装与本地预训练模型（当前项目默认离线）

```bash
cd code
pip install -r requirements.txt
```

把已经准备好的模型权重目录放在**项目根目录**：

```text
roberta升级后的完整项目/
├─ pretrain_models/
│  └─ chinese-roberta-wwm-ext/
│     ├─ config.json
│     ├─ model.safetensors 或 pytorch_model.bin
│     ├─ vocab.txt
│     └─ tokenizer_config.json 等 tokenizer 文件
└─ 信息抽取/
   └─ code/
      ├─ roberta/train.py
      └─ ...
```

新版 `roberta/train.py` 会自动向上查找 `pretrain_models/chinese-roberta-wwm-ext`，并打开 `HF_HUB_OFFLINE=1`、`TRANSFORMERS_OFFLINE=1`。因此原命令中的 Hub ID 会被自动替换为本地绝对路径，**不会联网下载模型或 tokenizer**：

```bash
python roberta/train.py --bert_path hfl/chinese-roberta-wwm-ext --encoder_name RoBERTa-WWM-Ext --decoder crf --max_len 256 --epochs 3 --batch_size 32 --grad_accum 1 --negative_ratio 0 --dev_file dev_extract_clean.jsonl --seed 42 --run_name roberta_crf_256_clean
```

如果文件夹名不同，可以传入实际本地绝对路径。若本地目录缺少模型权重或 tokenizer 文件，程序会报错提示，不会悄悄回退联网。

## 2. 先做公平对照（最重要）

必须让 BERT 与 RoBERTa 使用相同数据文件、标签、解码器、最大长度、随机种子、epoch、batch size、学习率、负样本比例和评测指标。新的默认验证文件是清理了文本重叠的 `dev_extract_clean.jsonl`（920 条）；原始 `dev_extract.jsonl` 为 930 条。不要把新 clean-dev 数字直接与旧 full-dev 数字当成严格同口径比较。

```bash
# BERT 基线：相同 clean dev，关闭负样本
python bert/train.py --bert_path bert-base-chinese --encoder_name BERT --decoder crf --max_len 256 --epochs 3 --batch_size 32 --grad_accum 1 --negative_ratio 0 --dev_file dev_extract_clean.jsonl --seed 42 --run_name bert_crf_256_clean

# RoBERTa：只改变预训练模型，其余参数不变
python roberta/train.py --bert_path hfl/chinese-roberta-wwm-ext --encoder_name RoBERTa-WWM-Ext --decoder crf --max_len 256 --epochs 3 --batch_size 32 --grad_accum 1 --negative_ratio 0 --dev_file dev_extract_clean.jsonl --seed 42 --run_name roberta_crf_256_clean
```

若在离线环境使用本地模型目录，将 `--bert_path` 改为实际目录，例如 `pretrain_models/chinese-roberta-wwm-ext`。BERT 本地权重也应显式指定，以保证两次运行都使用同一套环境。

## 3. 准确性增强消融实验

### 3.1 负样本训练

原抽取训练集以正样本为主；检测集包含大量无事件句。可将同一训练 split 的 `train_detect.jsonl` 中 `has_event=false` 的句子作为全 O 序列标签，按文本去重后加入训练：

```bash
python roberta/train.py --bert_path hfl/chinese-roberta-wwm-ext --decoder crf --max_len 256 --epochs 3 --batch_size 32 --negative_ratio 1.0 --dev_file dev_extract_clean.jsonl --seed 42 --run_name roberta_crf_256_neg1
```

`negative_ratio=1.0` 意味着目标为每条正样本增加一条负样本；若可用负样本不足则加入全部可用负样本。默认 `0.0` 不改变旧训练集组成。

然后对完整检测集做“是否出现任意触发词”的二分类评估：

```bash
python roberta/evaluate_presence.py --ckpt outputs/checkpoints/roberta_crf_256_neg1.pt --data_file dev_detect_clean.jsonl
```

输出 accuracy、precision、recall、specificity、F1 及混淆矩阵。该指标与事件级 F1 不同，必须并列报告，不能互相替代。

### 3.2 长文本截断实验

原项目 `max_len=256` 会截断部分较长句子。可以单独验证增加上下文是否减少漏抽：

```bash
python roberta/train.py --bert_path hfl/chinese-roberta-wwm-ext --decoder crf --max_len 512 --epochs 3 --batch_size 8 --grad_accum 4 --negative_ratio 0 --dev_file dev_extract_clean.jsonl --seed 42 --run_name roberta_crf_512_clean
```

`batch_size=8, grad_accum=4` 让有效 batch size 约为 32，以便接近 256 长度实验的 batch 规模。仍需记录显存与耗时。该模型最大位置长度为 512，输入中的特殊 token 也占用位置。

### 3.3 CRF 硬约束解码

同一 checkpoint 做解码期硬约束复评，不需要重训：

```bash
python bert/evaluate_hard_constraint.py --ckpt outputs/checkpoints/roberta_crf_256_clean.pt --run_name roberta_crf_256_clean_hard --dev_file dev_extract_clean.jsonl
```

该实验保持模型权重不变，只在 CRF 解码时禁止非法 BIO 转移。它应与对应软约束 checkpoint 使用同一份验证集。

## 4. LLM 提示词 v2 消融（与 RoBERTa 训练分开看）

RoBERTa/BERT 是判别式序列标注模型，不使用自然语言 prompt；修改 prompt 不会改变它们的训练输入。原项目的 LLM zero/few-shot 线路则新增可选 `--prompt_version v2`：在旧提示词基础上增加事件边界判断、事件类型/论元角色闭集约束、避免常识补全和输出前逐字证据核验。原 `v1` 保留，便于同模型、同样本、同 zero/few-shot 条件比较。

```bash
# 原提示词基线
python llm_zero_shot/llm_extract.py --backend api --mode few --prompt_version v1 --n 200 --tag api_few_v1_n200
# 新提示词
python llm_zero_shot/llm_extract.py --backend api --mode few --prompt_version v2 --n 200 --tag api_few_v2_n200
# 只查看新提示词，不调用模型
python llm_zero_shot/llm_extract.py --print_prompt --prompt_version v2
```

该提示词实验应固定模型、数据样本 ID、temperature 等生成参数，仅比较 v1/v2；prompt v2 不意味着一定提高 F1，需观察 JSON 合法率、触发词/论元幻觉率和事件级 F1。

## 5. 常用命令

```bash
# 复评 checkpoint：自动读取训练时记录的 dev_file
python bert/evaluate.py --ckpt outputs/checkpoints/roberta_crf_256_clean.pt

# 对完整检测集评估是否含事件
python roberta/evaluate_presence.py --ckpt outputs/checkpoints/roberta_crf_256_clean.pt --data_file dev_detect_clean.jsonl

# 运行不依赖 transformers 和 GPU 的数据回归测试
python -m unittest tests/test_roberta_data.py -v
```

## 6. BERT / RoBERTa 同口径对照与图表生成

绘图脚本已扩展为自动读取 BERT 与 RoBERTa 的结果。为了让主图里的模型对比公平，建议先用**同一份 clean dev**复评已有 BERT checkpoint，再绘制图表（不需要重新训练 BERT）：

```bash
# 1) 用已有 BERT checkpoint 在 920 条 clean dev 上复评 soft CRF（不需要重新训练）
python bert/evaluate.py --ckpt outputs/checkpoints/bert_crf_256.pt --run_name bert_crf_256_clean --data_file dev_extract_clean.jsonl

# 2) 同一个 BERT checkpoint 在 clean dev 上做 CRF hard 复评（不需要重新训练）
python bert/evaluate_hard_constraint.py --ckpt outputs/checkpoints/bert_crf_256.pt --run_name bert_crf_256_clean_hard --dev_file dev_extract_clean.jsonl

# 3) 用 RoBERTa checkpoint 重新运行 hard 评测（会生成完整 JSON、预测文件和 token 级 BIO 统计）
python bert/evaluate_hard_constraint.py --ckpt outputs/checkpoints/roberta_crf_256_clean.pt --run_name roberta_crf_256_clean_hard --dev_file dev_extract_clean.jsonl

# 4) 一次性绘制全部图表并更新方法对比 Markdown 表
python eval/make_figures.py
python eval/compare_methods.py --out outputs/metrics/disc/summary.md
```

图表输出到 `outputs/figures/`：

| 图表 | 内容 |
|---|---|
| `fig_method_compare.png` | 规则、BiLSTM、BERT、RoBERTa、流水线的触发词/论元/事件级 F1；每个方法显示样本量 |
| `fig_bert_roberta_reference_compare.png` | 当前已有 BERT 历史结果与本次 RoBERTa 结果并排；明确标注验证集不同，只作参考 |
| `fig_bert_roberta_same_dev.png` | BERT 与 RoBERTa 的专门同验证集对照；只在 dev 文件与样本数一致时生成 |
| `fig_type_confusion_bert_roberta_compare.png` | 两个模型在同一 clean dev 上并排比较事件类型混淆矩阵；需完整预测文件和中文字体 |
| `fig_illegal_bio.png` | 真实 token 序列的 BIO 非法数对比，不把空格字符映射引起的统计混进来 |
| `fig_per_role_f1.png` | 单个模型各论元角色 F1（默认优先 RoBERTa hard） |
| `fig_per_role_compare.png` | clean dev 上 BERT 与 RoBERTa 的逐角色论元 F1 对比 |
| `fig_training_curves.png` | BERT / RoBERTa 训练损失与内层验证 span F1 曲线 |
| `fig_type_confusion_*.png` | BERT 与 RoBERTa 分别生成事件类型混淆矩阵，自动读取各自 `data_file` |
| `fig_latency_cost.png` | 样本量、延迟、参数量、训练时间和三级 F1 |

如果 BERT 与 RoBERTa 的 `data_file` 或样本数不同，严格同口径对照图会跳过并给出原因；脚本另外生成 `fig_bert_roberta_reference_compare.png`，但图中会明确警告验证集不同。你当前这次 930 条 `dev_extract.jsonl` 的 BERT 历史记录与 920 条 `dev_extract_clean.jsonl` 的 RoBERTa 结果，只能放在参考对照图，不能用于正式声称 RoBERTa 提升了多少。

### 字符级与 token 级非法 BIO 的区别

旧版 `illegal_dev` 是字符级统计。tokenizer 不会为普通空格产生 token，字符还原时该位置会补成 `O`，于是一个 token 级合法的 `B-x I-x` 片段，在字符序列里可能出现 `O -> I-x`。所以你之前看到的“硬约束后非法转移 183 处”不一定代表硬约束失效。新版在 `meta.illegal_token_dev` 中另存真实 token 级统计：硬约束真正需要检查的是 token 级非法开头、非法转移是否为 0；字符级统计仍保留作字符映射诊断。

## 7. 主要实现改动

- `common/models.py`：序列标注编码器使用 `AutoModel`；增加 token 级非法 BIO 统计，并保留字符级统计用于分析空格映射带来的伪转移。
- `common/model_paths.py`、`roberta/train.py`：新增 RoBERTa 本地权重路径解析；原 Hugging Face ID 自动指向项目根目录 `pretrain_models/chinese-roberta-wwm-ext`，训练入口强制离线。
- `bert/train.py`：统一改用 `AutoTokenizer(use_fast=True)`；保存预训练模型名、验证文件名、正/负样本数；支持 `--dev_file`、`--negative_ratio`；训练步数按梯度累积向上取整。
- `bert/evaluate.py` / `bert/evaluate_hard_constraint.py`：从 checkpoint 还原本地编码器目录；支持指定 `--dev_file`；同时保存 `illegal_token_dev`。
- `eval/make_figures.py`：将 RoBERTa 加入主图、非法 BIO、训练曲线、角色级 F1、混淆矩阵与延迟表；新增“不同验证集参考对照图”，并保留同 clean dev 的严格 BERT/RoBERTa 对照。事件类型混淆矩阵增加“未抽取/其他”列，避免漏抽样本在图中消失。
- `eval/compare_methods.py`：增加 RoBERTa 主结果和硬约束消融；仅在验证集一致时自动计算 BERT/RoBERTa 的 F1 差值。
- `common/event_data.py`：新增确定性负样本采样函数，按文本去重并将负样本自动编码为全 O。
- `roberta/train.py`：以 RoBERTa-wwm-ext 为默认预训练权重，复用同一训练主线。
- `roberta/evaluate_presence.py`：对包含正负样本的 detect 集输出事件有无的分类指标。
- `tests/test_roberta_data.py`：检查负样本采样、全 O 标签和 clean-dev 文件。

## 8. 注意事项

1. 不要只替换模型却同时更换数据、长度、负样本比例、epoch 或评测集；那样无法归因准确率变化。
2. 新实验必须重新训练 BERT clean-dev 基线，旧项目指标不是 clean-dev 口径。
3. 这次提供的是修改后的可运行代码与实验脚本；没有在交付环境中完成 RoBERTa 训练，因为原压缩包不带预训练权重，当前环境也没有安装 `transformers`/模型权重。不要把历史 BERT 指标误写成 RoBERTa 结果。
4. 负样本可提高对无事件句的识别能力，但若加入过多，模型可能更倾向于输出全 O；建议按 0、0.5、1.0 比例做消融，并同时看事件级 F1 和 presence F1。
