# malicious_phish.csv 换装完整修改方案

> 目标：把 G:\workspace\information\ai\code 的五路分类实验从 TNEWS（15 类中文新闻标题）换成
> malicious_phish.csv（字段 url / type，4 类：benign / phishing / malware / defacement，约 65 万条英文 URL）
> 做恶意 URL 检测。
> 执行环境约定：本机仅改代码与轻量验证；BERT / LoRA 训练在远程 GPU 机执行。

---

## 0. 总策略

原架构的教学主张「换领域改映射表，不改模型」，但本次跨越更大：中文→英文、15→4 类、官方切分→自切分。
方案围绕三条主线，让 8 个方法目录的改动压到最小：

1. **数据契约不变**：新增转换脚本把 `(url, type)` 转成与 TNEWS 完全相同的
   `{"idx","sentence","label"}` JSON（`sentence` 键承载 URL），`common/data.py` 的
   `load_splits / clean_val / sample_subset` **零代码改动**自动生效；
2. **标签口径中枢**：`common/labels.py` 重写 4 类英文标签 + 新业务分流规则；
3. **各方法只换「特征与模型配置」**：规则换 URL 词法特征、TF-IDF 调 n-gram、
   BERT 换英文底座、LLM 三路换 prompt 与解析。

前置两步（改代码之前）：

```bash
# ① 备份旧数据（转换脚本会覆盖 train.json / val.json）
cd G:/workspace/information/ai/code/data
cp train.json tnews_train.json.bak && cp val.json tnews_val.json.bak
```

② 数据集实测（2026-11-17，data/malicious_phish.csv，651,194 数据行）：
- 类别集合确认为 4 类：benign 428,103（65.7%）/ phishing 94,111（14.4%）/
  defacement 96,457（14.8%）/ malware 32,520（5.0%）；
- **脏行 3 条**（乱码/空 type）——转换脚本按「过滤丢弃」处理，不中断；
- **(url,type) 完全重复 10,132 条（1.6%）**，去重后 641,062 条；
- **同 URL 不同标签 6 条**（标签矛盾，按 url 去重可顺手清掉，量级可忽略）；
- URL 平均 60 字符、最长 2,175、超 200 字符 10,501 条（1.6%，与第 12 节截断取舍一致）。

依赖补充：`pip install pandas`（原 requirements 无 pandas；
国内镜像 `pip install pandas -i https://pypi.tuna.tsinghua.edu.cn/simple`）。

---

## 1. 新建 data/convert_phish.py（改动核心，完整代码）

```python
"""
数据转换：malicious_phish.csv → 内部契约 train.json / val.json

把 (url, type) 转成 {"idx","sentence","label"}，sentence 键继续承载 URL，
下游五路脚本的 d["sentence"] 全部零改动。

用法：
  python data/convert_phish.py                              # 数据集已在 data/ 下，无参即可
  python data/convert_phish.py --train_n 0                 # 全量训练（GPU 档）
"""
import argparse
import json
import random
import sys
from collections import Counter
from pathlib import Path

import pandas as pd

CODE_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(CODE_ROOT))
from common.labels import LABELS, LABEL2ID   # 转换与五路方法共用同一张标签表


def normalize(url: str) -> str:
    """统一小写 + 截 200 字符。截断的必要性见第 8 节：SFT 的 max_length=256，
    超长 URL 会把 chat 模板/答案 token 一起切掉，造成 loss 全被 mask 的废样本。
    实测仅 1.6%（10,501 条）超长 URL 受影响。"""
    return url.strip().lower()[:200]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default=str(Path(__file__).resolve().parent / "malicious_phish.csv"))
    ap.add_argument("--train_n", type=int, default=50000, help="训练集条数，0=全量")
    ap.add_argument("--val_n", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    df = pd.read_csv(args.src)
    df = df.dropna(subset=["url", "type"])

    # 实测有 3 条脏行（乱码/空 type）：过滤丢弃并打印，别让个别脏数据挡住整批转换
    bad = ~df["type"].isin(LABELS)
    if bad.any():
        print(f"[warn] 丢弃 {int(bad.sum())} 条 LABELS 外的脏行:", df.loc[bad, "type"].tolist()[:5])
        df = df[~bad]

    df["url"] = df["url"].map(normalize)
    df = df.drop_duplicates(subset=["url", "type"])          # 实测 10,132 条重复（1.6%）：train/val 泄漏从源头掐掉
    # 可选更干净的口径：按 url 单键去重（实测仅 6 条同 URL 不同标签的矛盾样本）
    # df = df.drop_duplicates(subset=["url"])

    total = len(df)
    train_rows, val_rows = [], []
    for label in LABELS:                                      # 分层切分：每类内部独立按比例分
        rows = df[df["type"] == label].sample(frac=1.0, random_state=args.seed).to_dict("records")
        n_val = max(1, round(len(rows) * args.val_n / total))
        val_rows.extend(rows[:n_val])
        train_rows.extend(rows[n_val:])
    random.Random(args.seed).shuffle(train_rows)
    random.Random(args.seed).shuffle(val_rows)

    if 0 < args.train_n < len(train_rows):                    # GPU 不在身边时的抽样档
        by_type = {}
        for r in train_rows:
            by_type.setdefault(r["type"], []).append(r)
        keep = []
        for rs in by_type.values():
            keep.extend(rs[:round(len(rs) * args.train_n / len(train_rows))])
        train_rows = keep

    def to_records(rows, start):
        return [{"idx": start + i, "sentence": r["url"], "label": LABEL2ID[r["type"]]}
                for i, r in enumerate(rows)]

    tr, va = to_records(train_rows, 0), to_records(val_rows, len(train_rows))
    (CODE_ROOT / "data" / "train.json").write_text(json.dumps(tr, ensure_ascii=False), encoding="utf-8")
    (CODE_ROOT / "data" / "val.json").write_text(json.dumps(va, ensure_ascii=False), encoding="utf-8")
    print(f"train={len(tr)} val={len(va)}")
    print("train 各类:", Counter(LABELS[r["label"]] for r in tr))
    print("val   各类:", Counter(LABELS[r["label"]] for r in va))
```

要点：
- **分层切分（stratify）+ 固定 seed=42**：malicious_phish 无官方切分，benign 实测占 65.7%，
  不 stratify 的话小 val 里 malware 可能只有几十条；
- **整表先去重再切**：实测 (url,type) 完全重复 10,132 条（1.6%），源头去重后 `n_dup` 会是 0，
  `load_clean_val()` 的 meta 机制照常工作（历史教学点变成「源头已清洗」的对照）；
- 训练规模三档：默认 5 万条 / `--train_n 0` 全量 64 万（远程 GPU 档）/ `--train_n 10000`（快速档）。

---

## 2. 重写 common/labels.py（完整代码）

```python
"""
标签体系与安全分流口径（全项目共用）——从「投研主题」换成「URL 安全」

四类来自 malicious_phish.csv 的 type 字段：benign / phishing / malware / defacement。
分流业务口径：良性放行、恶意拦截、低置信转人工——与原投研版本同构，
只是「相关」的定义从「金融/产业」换成「是否恶意」。
"""

LABELS = ["benign", "phishing", "malware", "defacement"]
LABEL2ID = {name: i for i, name in enumerate(LABELS)}

# 3 组主题：safe / phishing（凭证窃取）/ malicious（站点被黑或投毒）
GROUP_MAP = {
    "benign": "safe",
    "phishing": "phishing",
    "malware": "malicious",
    "defacement": "malicious",
}
GROUPS = ["safe", "phishing", "malicious"]

# 安全团队关心「要不要拦」：phishing 与 malicious 组都算「需要处理」
RELEVANT_GROUPS = {"phishing", "malicious"}


def to_group(label_id: int) -> str:
    return GROUP_MAP[LABELS[label_id]]


def is_relevant(label_id: int) -> bool:
    return to_group(label_id) in RELEVANT_GROUPS


def queue_name(label_id: int, conf: float, conf_threshold: float = 0.8) -> str:
    """良性→pass；疑似恶意但置信低→review；恶意且高置信→block"""
    if not is_relevant(label_id):
        return "pass"
    if conf < conf_threshold:
        return "review"
    return "block"
```

**注意队列名是破坏性变更**：`research/review/drop` → `pass/review/block`，
唯一引用队列名做分组统计的下游是 `service/demo_early_report.py:50-68`（见第 9 节）。

---

## 3. common/metrics.py（2 处）

| 位置 | 改动 |
|---|---|
| metrics.py:20 | `def classification_report(y_true, y_pred, num_classes=15)` → `num_classes=len(LABELS)`（文件头加 `from common.labels import LABELS`，无循环依赖） |
| metrics.py:90 | `def pred_distribution(preds, num_classes=15)` → 同上 |

其余（ECE、acc_incl_unparsed、per_class_recall、混淆矩阵）全部自动适配 4 类。

---

## 4. common/model_paths.py:17（1 行）

```python
BERT_DIR = "bert-base-uncased"    # 原为 bert-base-chinese；URL 是英文+符号，中文底座完全不适用
```

获取权重的两条路（按环境二选一）：

```bash
# 路线 A（推荐）：hf-mirror 下载到 pretrain_models 目录
set HF_ENDPOINT=https://hf-mirror.com
huggingface-cli download bert-base-uncased --local-dir G:/workspace/information/ai/pretrain_models/bert-base-uncased

# 路线 B：训练机上 from_pretrained("bert-base-uncased") 缓存后整目录拷回
python common/model_paths.py    # 自检脚本会打印两个模型解析到的路径，改完必跑
```

Qwen3-0.6B 不用动。

---

## 5. 重写 rule/classify_rule.py 的分类函数（完整代码）

「URL 含类别名」的旧规则在英文 URL 上完全不成立，按 URL 词法特征重写：

```python
import re

SUSPICIOUS_TLDS = {".tk", ".ml", ".ga", ".cf", ".gq", ".top", ".xyz", ".buzz", ".click", ".country", ".review"}
SUSPICIOUS_WORDS = ("login", "verify", "account", "secure", "banking", "confirm",
                    "webscr", "signin", "update", "password", "recover")
IP_HOST = re.compile(r"^https?://[^/@]+")


def url_rule_hits(url: str) -> list:
    """返回命中的规则名列表（可多条），供快筛与错误分析"""
    low = url.lower()
    m = IP_HOST.match(low)
    host = m.group(0)[8:].split("/")[0] if m else ""        # 提取 host 段
    hits = []
    if host and re.fullmatch(r"[0-9.]+", host.split(":")[0]):
        hits.append("ip_host")                             # 裸 IP 做 host
    if "@" in host:
        hits.append("at_in_host")                         # http://baidu.com@evil.com
    if any(host.endswith(t) for t in SUSPICIOUS_TLDS):
        hits.append("suspicious_tld")                     # 廉价/滥用高发 TLD
    if any(w in low for w in SUSPICIOUS_WORDS):
        hits.append("keyword")                            # 路径中的钓鱼话术
    if len(url) > 75:
        hits.append("very_long")                           # 超长编码/子域堆叠
    return hits


def classify(text: str, fallback: int) -> int:
    """L0 规则：命中的词法特征全部指向 phishing 档；一条不中 → majority 兜底（benign）。
    教学结论不变：规则能分「恶意/良性」，分不了「phishing/malware/defacement」，
    它仍是快筛兜底，不是分流主力。"""
    if url_rule_hits(text):
        return LABEL2ID["phishing"]
    return fallback
```

同时改两处统计口径（classify_rule.py:69 的「类别名命中率」打印 → 改为打印 `url_rule_hits`
命中率），majority 兜底逻辑（classify_rule.py:55）零改动（自动变成猜 benign）。

---

## 6. tfidf/classify_tfidf.py（1 个默认值）

| 位置 | 改动 |
|---|---|
| classify_tfidf.py:35 | `--ngram` 默认 `"1,2"` → `"2,5"` |

URL 上信号在「子串/token 片段」（`.tk`、`//login`、`-verify-`），char 2-5gram 恰好覆盖；
`analyzer="char"`（classify_tfidf.py:53）保持。去重后 64 万条全量训练时 `max_features=300000`
维度不变，训练时长仍在分钟级。`--leak` 开关的教学价值保留（源头已去重后它将展示 0 泄漏，
可以留作对照或注明已失效）。

---

## 7. BERT 两脚本（改 max_length，其余自动继承）

| 文件 | 位置 | 改动 |
|---|---|---|
| bert/train_bert.py:167 | `--max_length` 默认 64 | → `128` |
| bert/predict_bert.py:36 | `predict(..., max_length=64)` | → `128`（**必须与训练一致**，否则 padding 分布漂移拉低指标） |

自动继承、无需改的：`num_labels=len(LABELS)`（train_bert.py:191、predict_bert.py:62）、
类别加权（train_bert.py:207，新数据 66:15:12:5 的不均衡下更要开）、FGM/LS/Focal、AMP、
分层学习率全链路。

训练量预估（独显，按 4060 档推算）：5 万条 × 3 epoch ≈ 15 分钟；全量 64 万 × 1 epoch ≈ 60 分钟
（建议全量档用 `--epochs 1`，数据量大了深跑易过拟合）。`TnewsDataset` 类名可顺手改
`PhishDataset`（纯重命名，predict_bert.py:30 的 import 要同步）。

---

## 8. LLM 三路（SYSTEM prompt + parse_label + max_length）

三个文件共用同一套替换，给出成品代码：

**SYSTEM（英文 + 类别定义；phishing 与 malware 是 LLM 最易混的两个类，定义必须写进 prompt）：**

```python
SYSTEM = ("You are a URL security classifier. Classify the given URL into exactly one "
          "of these categories:\n"
          "- benign: a normal, legitimate website URL\n"
          "- phishing: a fake site impersonating a legitimate one, aimed at stealing credentials\n"
          "- malware: a URL that serves malicious software or leads to a malicious download\n"
          "- defacement: a hacked/vandalized legitimate website\n"
          "Answer with only the category name, nothing else.")
```

**parse_label（原中文「子串包含+最长匹配」在英文上有误伤，如 benign 命中 non-benign；
改精确等值优先）：**

```python
import re

def parse_label(text: str) -> int:
    t = text.strip().lower()
    for i, name in enumerate(LABELS):
        if t == name:                          # ① 精确等值：模型只吐标签名的常态
            return i
    hits = [n for n in LABELS if re.search(rf"\b{re.escape(n)}\b", t)]
    return LABELS.index(max(hits, key=len)) if hits else -1    # ② 词边界匹配容忍 "The category is: phishing."
```

逐文件改动点：

| 文件 | 位置 | 改动 |
|---|---|---|
| llm_zero_shot/classify_zero_shot.py:39-40 | SYSTEM | 上面的英文版 |
| llm_zero_shot/classify_zero_shot.py:43-55 | parse_label | 上面的精确版 |
| llm_api_zero_shot/classify_api_zero_shot.py:44-45 / :71-74 | SYSTEM / parse_label | 同上两处 |
| llm_lora/train_lora.py:40-41 | SYSTEM | 同上（**训练与 eval 的 SYSTEM 必须逐字符一致**，否则模板错位掉分） |
| llm_lora/train_lora.py:97 | `--max_length` 默认 128 | → `256`。**这是必改项**：SFTDataset:67-68 对 `input_ids` 截断，URL 长时模板+答案被切掉，labels 全 -100 变成零 loss 的废样本（转换脚本已把 URL 截 200 字符兜底，256 才放得下 system+URL+答案） |
| llm_lora/train_lora.py:88 | `--num_train` 默认 5000 | → `20000`（原 5000 是 5.3 万训练集的 9%；65 万数据下等比不现实，2 万条 GPU 约 3 小时是性价比点） |
| llm_lora/eval_lora.py:34-35 / :38-40 | SYSTEM / parse_label | 与 train_lora 逐字符一致 |
| llm_lora/eval_lora.py:47 | `--max_new_tokens` 默认 8 | 不用动（最长标签 defacement 约 2-3 token） |

`enable_thinking=False`、chat 模板、Loss Masking、collate、LoRA 配置（r=8, target_modules）
全部零改动。

---

## 9. service 三脚本

**FAST_RULES 重写（classify_news.py:36-39 与 classify_cli.py:52-55 两处，内容一致）：**

```python
# 规则快筛：URL 词法硬信号命中即判 phishing 档，不再过模型（省钱段不变）
FAST_RULES = [
    ("ip_host", "phishing"),        # 裸 IP 做 host
    ("at_in_host", "phishing"),     # host 段含 @（http://real.com@evil.com）
]
```

实现复用第 5 节的 `url_rule_hits`（把函数挪到 `rule/classify_rule.py` 后 service 侧
import，或复制同款逻辑到各自文件——原项目两文件本就是各自内嵌 FAST_RULES 的写法，
保持原结构即复制）。`_rule_hit` 里 `LABELS.index(label)`（classify_news.py:75）不变。

其余逐点：

| 文件 | 位置 | 改动 |
|---|---|---|
| service/classify_news.py:80 | `max_length=64` | → `128` |
| service/classify_news.py:31-32 | MODEL_NAME / MODEL_VERSION | `"bert-url-v1"` / 更新月份 |
| service/classify_cli.py:158 | `_calibrate_bert` 的 `max_length=64` | → `128` |
| service/classify_cli.py:223 | `_predict` BERT 分支 `max_length=64` | → `128` |
| service/classify_cli.py:185-186 | TF-IDF 后端 `ngram_range=(1,2)` | → `(2,5)`（与第 6 节保持同一配置） |
| service/demo_early_report.py:22-43 | MESSAGES 20 条中文标题 | → 换 15~20 条混合 URL 样例（benign 大站 + 各恶意档各几条，从 val.json 抽真实样本最有说服力） |
| service/demo_early_report.py:50-68 | `buckets = {"research","review","drop"}` 及三处 `show()` 标题 | → `pass/review/block`（labels.py 新队列名） |

温度校准（`_fit_temperature`/`_calibrate`）、后端降级链（auto→bert→tfidf→fallback）、
`--json` 输出契约字段名（topic/sub_topic/relevant/conf/queue/model/version）全部零改动，
变的只是字段取值语义。

---

## 10. 零改动清单（自动继承）

- `common/data.py`（契约不变：load_splits 读 train/val.json、clean_val 整串去重对 URL 有效、
  sample_subset 固定 seed 抽样）；
- `eval/eval_all.py`（SOURCES 结果文件名不变；建议顺手把 eval_all.py:35 的显示名
  `API zero-shot（deepseek-v4-flash）` 改成从 meta 读，非必需）；
- `outputs/` 旧结果文件全部作废重跑（建议先归档到 `outputs/tnews/` 备份）。

---

## 11. 执行顺序与验收清单（训练在远程 GPU 机）

```bash
# 1) 数据（本机可做）
python data/convert_phish.py                               # 数据集已在 data/ 下，无参即可

# 2) 模型自检（本机，确认 bert-base-uncased 已就位）
python common/model_paths.py

# 3) 秒级两路（本机）
python rule/classify_rule.py --n 500
python tfidf/classify_tfidf.py --n 500

# 4) BERT（远程 GPU：先冒烟再正式）
python bert/train_bert.py --max_steps 20 --max_eval 500 --no_save   # 流程验证
python bert/train_bert.py --epochs 3                                 # 5 万档 ≈15 分钟
python bert/predict_bert.py --n 500

# 5) LLM 路线（远程 GPU；API 路线本机可跑）
python llm_zero_shot/classify_zero_shot.py --n 500
python llm_lora/train_lora.py --num_train 20000 --epochs 2           # ≈3 小时
python llm_lora/eval_lora.py --n 500
python llm_api_zero_shot/classify_api_zero_shot.py --n 500            # 需 LLM_API_KEY

# 6) 汇总 + 服务（本机）
python eval/eval_all.py
python service/demo_early_report.py
python service/classify_cli.py --text "http://192.168.10.1/confirm/login" --json
```

逐项验收标准：

| 检查项 | 通过标准 |
|---|---|
| 转换脚本 | 输出 4 类计数与 CSV `value_counts` 一致；train/val URL 无交集 |
| 规则法 | 能跑通且**不再塌缩到 majority**（ip_host 等信号有真实命中）；仍预期 Macro-F1 偏低（L0 定位不变） |
| TF-IDF | `ngram=2,5` 打印正确；meta 的 `n_eval/seed` 与其他路一致 |
| BERT | `--max_steps 20` 冒烟通过；正式跑 val Macro-F1 显著高于 TF-IDF（文献上该数据集 BERT 可到 0.97+，远高于 TNEWS 的 0.55——URL 词法信号比新闻标题强得多，**预期结论会反转**：五路差距被拉开，注意 README/ARCHITECTURE 的叙事段要重写） |
| LLM 三路 | zero-shot 大概率仍混淆 phishing/malware（有类别定义后塌缩应缓解）；LoRA 起码覆盖 4/4 类 |
| eval_all | 各路 `n=500, seed=42` 一致，无「未运行」缺失 |
| 服务 | CLI 输出新契约：`queue ∈ {pass, review, block}`，`model` 字段显式标注后端 |

---

## 12. 风险与决策记录

1. **训练/推理 max_length 六处一致性**是最大隐性坑：
   train_bert.py:167、predict_bert.py:36、classify_news.py:80、classify_cli.py:158、
   classify_cli.py:223、train_lora.py:97——漏一处则训练/推理输入分布不一致，指标莫名掉。
2. **类别不均衡（实测 66:14:15:5，benign 65.7% / phishing 14.4% / defacement 14.8% / malware 5.0%）**：
   BERT 类别加权默认开（继承）；评测 500 条随机抽样里 malware 仅约 25 条，per-class recall 噪声大——
   若要更稳的横向对比，可在转换脚本里把 val 改成每类等量分层（备选项，会改变「随机 500」口径，
   二选一后写进 README）。
3. **phishing 子集为人工合成**（PhishTank/Malware 合成语料），与真实部署分布有偏，
   报告结论须注明「基准数字」而非「线上预期」。
4. **URL 截断 200 字符**：极少数超长 URL 尾部特征丢失，属于可控精度换工程稳定性的取舍。
5. **文档同步**：README / ARCHITECTURE / USAGE_GUIDE / RESUME_GUIDE 四份 md 的叙事与
   结果表全部基于 TNEWS 数字，重跑后需整段重写（尤其 11 节所述「结论反转」风险）。
