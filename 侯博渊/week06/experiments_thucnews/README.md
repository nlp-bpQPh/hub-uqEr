# THUCNews 10 类新闻分类实验

> 数据集换成 **THUCNews**（10 类中文新闻），比二分类情感任务更接近真实多分类场景。
> 目标：在 10 类上重新比较规则 / TF-IDF / StructBERT-tiny / BERT-base 线性探针。

## 数据

- 来源：`hf-mirror.com/datasets/spiritx2023/ThuCnews`（`cnews.val.txt` / `cnews.test.txt`）
- 类别：体育、娱乐、家居、房产、教育、时尚、时政、游戏、科技、财经
- 规模：
  - train 4,000（每类 400，从 val 的 5,000 条里分层抽）
  - dev 1,000（每类 100）
  - test 1,500（每类 150，从官方 test 里分层抽）
- 文本：中位数约 630 字，p95 约 2,440 字（长新闻）
- test 与 train 完全重复：0 / 1,500
- 本地位置：`_data/thucnews/{train,dev,test}.json` + `labels.json`

## 关键结果（test 1,500）

| 方法 | Macro-F1 |
|---|---|
| 规则 / 关键词 | 0.6676 |
| TF-IDF + LinearSVC | **0.9117** |
| TF-IDF + LogReg | 0.8789 |
| StructBERT-tiny | 0.8930 |
| BERT-base 线性探针 | 0.9109 |

> TF-IDF 与 BERT 探针只差 **0.08 个点**，基本并列；tiny 落后 TF-IDF 1.9 点。
> dev 比 test 高约 7 个点（TF-IDF 0.9860 / tiny 0.9675），**dev 只能选 epoch，不能当 test 估计**。

## 目录

```
experiments_thucnews/
├── prepare_data.py        # 下载 cnews.val/test.txt → 分层 JSON
├── run_baselines.py       # 规则关键词 / TF-IDF+LinearSVC / TF-IDF+LogReg
├── train_small_bert.py    # StructBERT-tiny 微调（3 epoch，max_len=192，动态 padding）
├── bert_linear_probe.py   # 冻结 BERT-base + 逻辑回归（2,000 条特征）
├── make_report.py         # 汇总 REPORT.md + comparison.png
├── REPORT.md
└── outputs/
    ├── rule_pred.json
    ├── tfidf_svc_pred.json
    ├── tfidf_lr_pred.json
    ├── small_bert_pred.json / small_bert_log.json / small_bert_ckpt.pt
    ├── bert_probe_pred.json
    ├── summary.md
    └── comparison.png
```

## 运行

```powershell
python experiments_thucnews/prepare_data.py
python experiments_thucnews/run_baselines.py
python experiments_thucnews/train_small_bert.py --epochs 3
python experiments_thucnews/bert_linear_probe.py --n_train 2000
python experiments_thucnews/make_report.py
```

冒烟：

```powershell
python experiments_thucnews/train_small_bert.py --epochs 1 --max_steps 3 --no_save
```

## 口径

| 项 | 说明 |
|---|---|
| 训练 / 验证 / 测试 | 4,000 / 1,000 / 1,500（10 类平衡） |
| 小模型 | 3 epoch、batch 64、max_len=192、动态 padding、lr 5e-5（头 ×2）、类别加权 |
| BERT 探针 | 2,000 条分层抽样，冻结 BERT 抽 [CLS] + 逻辑回归，max_len=128 |
| 指标 | Accuracy、Macro-F1、per-class recall、ECE |

## 限制

- 训练只用了 4,000 条（官方训练集 5 万条），是为了 CPU 可跑；数据量上去后 BERT 可能反超。
- 长新闻被截断（tiny 192 token / BERT 探针 128 token），会低估预训练模型的上限。
- BERT-base 只做线性探针，不是全量微调（本机无 GPU）。
- 单 seed；1,500 条测试集噪声约 ±1 个点。
