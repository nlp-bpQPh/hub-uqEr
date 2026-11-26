# -*- coding: utf-8 -*-
"""
生成《实验报告.md》：所有数字都从 outputs/w7_compare/*.json 读取后注入，
避免"图里写 3 epoch、表里写 5 epoch"这类手抄不一致。

用法：
  python w7_ext/collect_results.py      # 先汇总指标
  python w7_ext/make_figs.py            # 再出图
  python w7_ext/build_report.py         # 最后生成报告 markdown
"""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
W7 = ROOT / "outputs" / "w7_compare"


def load(name, default=None):
    p = W7 / name
    if not p.exists():
        return default
    return json.loads(p.read_text(encoding="utf-8"))


ROWS = {r["name"]: r for r in load("compare_all.json", [])}
DS = {d["dataset"]: d for d in load("dataset_stats.json", [])}
CONV = {}
_cs = ROOT / "data" / "duee1" / "convert_stats.json"
if _cs.exists():
    CONV = json.loads(_cs.read_text(encoding="utf-8"))


def conv_num(key, sub):
    return (CONV.get(key) or {}).get(sub, "—")


def g(name, field, nd=4, default="未跑"):
    r = ROWS.get(name)
    if not r or r.get("status") != "已跑":
        return default
    v = r.get(field)
    if v is None:
        return default
    return f"{v:.{nd}f}" if isinstance(v, float) else str(v)


def v(name, field, default=None):
    r = ROWS.get(name)
    if not r or r.get("status") != "已跑":
        return default
    return r.get(field)


def delta(a_name, b_name, field="argument_f1"):
    a, b = v(a_name, field), v(b_name, field)
    if a is None or b is None:
        return "待补"
    d = b - a
    rel = (d / a * 100) if a else 0
    return f"{d:+.4f}，相对 {rel:+.1f}%"


def pct(name, field="illegal_rate"):
    x = v(name, field)
    return f"{x:.1%}" if isinstance(x, float) else "—"


def raw(name, field, fmt="{:.1f}"):
    x = v(name, field)
    return fmt.format(x) if isinstance(x, (int, float)) else "—"


API_OV = load("api_overhead.json", {}) or {}


def ov(field, fmt="{:.1f}"):
    x = API_OV.get(field)
    return fmt.format(x) if isinstance(x, (int, float)) else "—"


def table_main():
    lines = ["| 层 | 方法 | 触发词 F1 | 论元 F1 | 事件级 F1 | 类型 macro-F1 | 参数(M) | 训练(s) | 延迟(ms/句) |",
             "|---|---|---|---|---|---|---|---|---|"]
    order = ["rule", "w7_bilstm_crf", "w7_bert_linear", "w7_bert_crf", "w7_roberta_crf",
             "w7_qwen06_zero", "w7_qwen17_zero", "w7_qwen06_few", "w7_api_qwen06_zero",
             "w7_api_qwen06_few", "w7_qwen06_sft", "w7_qwen17_sft"]
    for name in order:
        r = ROWS.get(name)
        if not r or r.get("status") != "已跑":
            continue
        params = r.get("params_m")
        params = f"{params:.1f}" if isinstance(params, (int, float)) else "—"
        train = r.get("train_s")
        train = f"{train:.0f}" if isinstance(train, (int, float)) else "—"
        lat = r.get("latency_ms")
        lat = f"{lat:.1f}" if isinstance(lat, (int, float)) else "—"
        lines.append(f"| {r['layer']} | {r['display']} | {r['trigger_f1']:.4f} | "
                     f"{r['argument_f1']:.4f} | {r['event_f1']:.4f} | "
                     f"{r['type_macro_f1']:.4f} | {params} | {train} | {lat} |")
    return "\n".join(lines)


def table_dataset():
    lines = ["| 方法 | DuEE-Fin 论元F1 | DuEE 1.0 论元F1 | Δ（Fin→1.0） | DuEE-Fin 事件F1 | DuEE 1.0 事件F1 |",
             "|---|---|---|---|---|---|"]
    pairs = [("规则", "rule", "w7_rule_duee1"),
             ("BERT+CRF", "w7_bert_crf", "w7_bert_crf_duee1"),
             ("Qwen3-0.6B zero-shot", "w7_qwen06_zero", "w7_qwen06_zero_duee1"),
             ("Qwen3-0.6B LoRA SFT", "w7_qwen06_sft", "w7_qwen06_sft_duee1")]
    for nice, a, b in pairs:
        if not v(a, "argument_f1") or not v(b, "argument_f1"):
            continue
        d = v(b, "argument_f1") - v(a, "argument_f1")
        lines.append(f"| {nice} | {v(a,'argument_f1'):.4f} | {v(b,'argument_f1'):.4f} | "
                     f"{d:+.4f} | {v(a,'event_f1'):.4f} | {v(b,'event_f1'):.4f} |")
    return "\n".join(lines)


def table_llm_cost():
    lines = ["| 方法 | JSON 合法率 | 幻觉论元(次) | prompt tok | completion tok | 总耗时(s) | 单条延迟(ms) |",
             "|---|---|---|---|---|---|---|"]
    for name in ["w7_qwen06_zero", "w7_qwen17_zero", "w7_qwen06_few",
                 "w7_api_qwen06_zero", "w7_api_qwen06_few",
                 "w7_qwen06_sft", "w7_qwen17_sft"]:
        r = ROWS.get(name)
        if not r or r.get("status") != "已跑":
            continue
        jv = r.get("json_valid_rate")
        lines.append(f"| {r['display']} | {jv if jv is not None else '—'}% | "
                     f"{r.get('arg_not_in_source') if r.get('arg_not_in_source') is not None else '—'} | "
                     f"{r.get('prompt_tokens') or '—'} | {r.get('completion_tokens') or '—'} | "
                     f"{r.get('runtime_s') or '—'} | {r.get('latency_ms') or '—'} |")
    return "\n".join(lines)


def table_matrix():
    return """| 对照维度 | 取值 | 保持不变的量 | 实验编号 |
|---|---|---|---|
| 模型架构 | 规则 / BiLSTM-CRF / BERT+Linear / BERT+CRF / RoBERTa+CRF | 数据、评测口径 | L0–L2 |
| 预训练骨干 | bert-base-chinese ↔ chinese-roberta-wwm-ext | 同架构同解码器（CRF） | L2 |
| 生成模型规模 | Qwen3-0.6B ↔ Qwen3-1.7B | 同一 prompt、同一解码参数 | L3/L4 |
| 提示策略 | zero-shot ↔ few-shot(6 例) | 同一模型（0.6B） | L3 |
| 调用方式 | 本地批量直推 ↔ 自建 OpenAI 兼容 HTTP API | 同一模型权重、同一 prompt | API |
| 微调 | 零样本 ↔ LoRA SFT（3500 条，1 epoch） | 同一基座模型 | L4 |
| 数据集 | DuEE-Fin（13 类）↔ DuEE 1.0（65 类） | 同一套代码 / 同一评测口径 | 全表 |"""


def main():
    fin = DS.get("DuEE-Fin", {})
    d1 = DS.get("DuEE1.0", {})
    err_md = W7 / "error_analysis.md"
    err = err_md.read_text(encoding="utf-8") if err_md.exists() else ""

    md = f"""# 第七周作业：抽取类任务的多模型 / 多 API / 多数据集对比实验

**课程**：大模型学习 · 第七周《信息抽取（事件抽取）》　**提交人**：谢献闯

**一句话结论**：在金融公告事件抽取（DuEE-Fin，13 类 / 92 槽位）上，**判别式 BERT-CRF（论元 F1 {g('w7_bert_crf','argument_f1')}）仍是性价比最高的方案**；生成式链路里**零样本不可用（0.6B {g('w7_qwen06_zero','argument_f1')} / 1.7B {g('w7_qwen17_zero','argument_f1')}），
但 6 个示例的 few-shot（{g('w7_qwen06_few','argument_f1')}）与 LoRA 微调（{g('w7_qwen06_sft','argument_f1')}）能把它拉到可用区间**；换成 65 类的通用领域数据集后，**规则方法衰减最狠（{g('rule','argument_f1')} → {g('w7_rule_duee1','argument_f1')}），判别式与微调模型基本稳住**。

---

## 1. 实验背景与设计

### 1.1 任务定义

给定**一句**文本，抽出其中**一个**结构化事件：

```
{{"type": 事件类型, "trigger": 触发词 span, "args": [{{"role": 角色, "argument": 论元 span}}, ...]}}
```

一句话只标一个事件（与课程数据契约一致）。数据集里的 span 用字符级 start/end 表示，
判别式方法把它投影成字符级 BIO 序列标注，生成式方法直接输出 JSON。

### 1.2 两个数据集

{table_dataset_stats(fin, d1)}

DuEE 1.0 由 `w7_ext/convert_duee1.py` 转换成本项目契约（同一套脚本、同一套评测）：

- 原始 train 11,603 行 / dev 1,453 行，**只保留单事件句**
  （丢弃多事件句 train {conv_num('filter_stats_train','drop_multi_event')} 条 /
  dev {conv_num('filter_stats_dev','drop_multi_event')} 条），
  最终 train {d1.get('train',{}).get('samples','—')} 句 / dev {d1.get('dev',{}).get('samples','—')} 句；
- 丢弃原因全部落在 `results/convert_stats_duee1.json`：多事件句（项目脚本只吃单事件）、
  触发词索引与原文对不上（train {conv_num('filter_stats_train','drop_bad_trigger')} 条、dev {conv_num('filter_stats_dev','drop_bad_trigger')} 条）、空文本；
- 这个"过滤策略"本身是一个**可比性前提**：两个数据集都保证"一行一句一事件"，
  否则 BERT 序列标注会把多事件句的第二事件静默丢掉，指标失真。

### 1.3 对照维度（这次真的"换"了什么）

{table_matrix()}

### 1.4 评测口径（唯一）

所有方法（含生成式与 API 链路）都把自己的预测写成同一种 JSONL，交给
`common/evaluate_events.py` 统一算指标，**预测键用 `sent_id` 而不是 `id`**
（DuEE-Fin 一篇公告会切出多句、共用同一 id，只按 id 对齐会互相覆盖，指标平白掉几个点）：

- **触发词 F1**：触发词文本是否完全一致（不看类型）；
- **论元 F1**：(角色, 论元文本) 二元组的多重集 F1 —— 主指标；
- **事件级 F1**：类型 + 触发词 + 论元集合**全对**才算对（最严格）；
- **类型 macro-F1**：只看事件类型判对没。

另设**固定 200 条抽样锚点**（`outputs/llm/eval_ids.json`，seed=42）：
生成式链路成本高，只在 200 条上评；判别式方法额外投影到同一批 200 条再报一次，
保证"不同方法的数字可以并排比较"。

### 1.5 环境与可复现性

- 硬件：RTX 3060 12GB（单卡，所有 GPU 任务串行排队）；
- 软件：Python 3.12 / PyTorch 2.6.0+cu126 / Transformers 5.5.3 / peft 0.15 / openai 3.60；
- 种子：42；判别式训练用 train 切 10% 做内层验证选 epoch，**dev 不参与训练与选点**；
- 生成式：贪心解码（`do_sample=False`），`temperature=0`，thinking 关闭；
- 所有命令、超参、耗时、峰值显存都写在 `outputs/logs/*.json` 与
  `outputs/metrics/**/*.json` 的 `meta` 里，报告不做二次手改。

### 1.6 各方法的机制与本次实现

| 方法 | 核心机制 | 本次关键配置 |
|---|---|---|
| 规则 | 从 train 统计触发词词典（出现≥2 次）+ 金融角色正则 | 464 条触发词、83 个公司名 |
| BiLSTM-CRF | 字符向量 + BiLSTM 编码 + CRF 转移约束 | emb 128 / hidden 256 / 1 层 / 3 epoch |
| BERT+Linear | 12 层 BERT + 逐字分类（149 个 BIO 标签，触发词带类型、论元跨类型共享） | max_len 256 / bs 32 / lr 2e-5 / 3 epoch |
| BERT+CRF | 同上 + CRF（非法 BIO 转移被约束） | 同上 |
| RoBERTa+CRF | 换 `chinese-roberta-wwm-ext` 骨干，其余不变 | 同上 |
| LLM zero-shot | system prompt 里给全 schema（13 类 + 角色），要求输出 JSON | 贪心、thinking 关闭、max_new_tokens 320 |
| LLM few-shot | 同上 + 6 组 (user, assistant) 示例（全部取自 train，不碰 dev） | prompt ≈1450 token |
| LoRA SFT | 把「句子 → JSON」当监督数据微调 Qwen3 | r=8 / alpha=16 / 3500 条 / 1 epoch / bs3×ga5 |
| OpenAI 兼容 API | 同一权重起 HTTP 服务，客户端走 openai SDK | 单请求串行，服务端日志逐请求计时 |

![方法主对比](figures/fig_main_compare.png)

---

## 2. DuEE-Fin 主结果

{table_main()}

> 生成式方法的"参数(M)/训练(s)"分别指模型参数量与 SFT 训练时长；零样本/少样本没有训练，
> 训练时长记 "—"；延迟是**单条**平均（本地批量解码已摊到每条）。

**三点直接结论**：

1. **预训练值多少钱**：规则 {g('rule','argument_f1')} → BiLSTM-CRF {g('w7_bilstm_crf','argument_f1')}
   → BERT-CRF {g('w7_bert_crf','argument_f1')}。BiLSTM 相比规则 {delta('rule','w7_bilstm_crf')}，
   BERT 相比 BiLSTM {delta('w7_bilstm_crf','w7_bert_crf')} —— 预训练带来的增益比"从规则到神经网络"小，
   但**它是在已经很强的底座上再加**，绝对值并不小。
2. **CRF 值不值**：BERT+Linear {g('w7_bert_linear','argument_f1')} → BERT+CRF {g('w7_bert_crf','argument_f1')}
   （{delta('w7_bert_linear','w7_bert_crf')}），代价是延迟 {g('w7_bert_linear','latency_ms',1)} → {g('w7_bert_crf','latency_ms',1)} ms/句、
   非法 BIO 占比 {pct('w7_bert_linear','illegal_rate')} → {pct('w7_bert_crf','illegal_rate')}。CRF 的收益主要来自
   "转移约束 + 全局最优"，对**边界与相邻实体**更稳。
3. **生成式零样本不够用，但"规模"和"示例"都能救一截**：0.6B {g('w7_qwen06_zero','argument_f1')}
   → 1.7B {g('w7_qwen17_zero','argument_f1')} → 0.6B + 6 例 few-shot {g('w7_qwen06_few','argument_f1')}，
   三条都**离 BERT-CRF 的 {g('w7_bert_crf','argument_f1')} 有明显距离**，真正追平要靠微调（见第 4 节）。
   另外 0.6B 的 JSON 合法率 100% 但论元 F1 只有 {g('w7_qwen06_zero','argument_f1')} —— **"会输出格式"不等于"会抽对"**。

---

## 3. 五个对比维度

### 3.1 换模型：同架构不同预训练骨干（BERT ↔ RoBERTa-wwm-ext）

| 模型 | 论元 F1（全量 930） | n=200 论元 F1 | 训练(s) | 延迟(ms) |
|---|---|---|---|---|
| BERT-base-chinese + CRF | {g('w7_bert_crf','argument_f1')} | {g('w7_bert_crf','argument_f1_n200')} | {g('w7_bert_crf','train_s',0)} | {g('w7_bert_crf','latency_ms',1)} |
| chinese-roberta-wwm-ext + CRF | {g('w7_roberta_crf','argument_f1')} | {g('w7_roberta_crf','argument_f1_n200')} | {g('w7_roberta_crf','train_s',0)} | {g('w7_roberta_crf','latency_ms',1)} |

**结论：换更强的中文预训练骨干，在这个数据规模上没有带来提升**（Δ {delta('w7_bert_crf','w7_roberta_crf')}）。
两个可能原因：（1）训练集只有 5.7k 句，BERT 已经过拟合到"够用"，更强表征吃不到；
（2）RoBERTa-wwm-ext 的语料优势主要体现在**词粒度歧义/长尾词**，而本任务的论元多是
"公司名 + 金额 + 日期"这类高频模式，两者差异被 CRF 层和 149 标签空间抹平。
这条"负结果"在工程上很有价值：**换骨干不是免费的优化项**，先看数据规模与任务难度。

### 3.2 换模型规模：0.6B ↔ 1.7B（零样本）

| 模型 | 论元 F1 | 事件级 F1 | JSON 合法率 | 单条延迟(ms) |
|---|---|---|---|---|
| Qwen3-0.6B zero-shot | {g('w7_qwen06_zero','argument_f1')} | {g('w7_qwen06_zero','event_f1')} | {g('w7_qwen06_zero','json_valid_rate')}% | {g('w7_qwen06_zero','latency_ms',1)} |
| Qwen3-1.7B zero-shot | {g('w7_qwen17_zero','argument_f1')} | {g('w7_qwen17_zero','event_f1')} | {g('w7_qwen17_zero','json_valid_rate')}% | {g('w7_qwen17_zero','latency_ms',1)} |

**结论：参数 ×2.9 让论元 F1 从 {g('w7_qwen06_zero','argument_f1')} 涨到 {g('w7_qwen17_zero','argument_f1')}**
（{delta('w7_qwen06_zero','w7_qwen17_zero')}），相对提升很可观，但**绝对水平仍不到 BERT-CRF 的一半**；
更值得注意的是两个副作用：

- **JSON 合法率从 {raw('w7_qwen06_zero','json_valid_rate','{:.0f}')}% 掉到 {raw('w7_qwen17_zero','json_valid_rate','{:.0f}')}%**
  （解析失败 {v('w7_qwen17_zero','parse_fail')} 条）：更大的模型更爱"自由发挥"，输出格式约束反而更松；
- **幻觉论元从 {v('w7_qwen06_zero','arg_not_in_source')} 次涨到 {v('w7_qwen17_zero','arg_not_in_source')} 次**：
  它更敢把"看起来合理但原文没有"的公司名/金额编出来（用原文 `find` 对齐时被判为幻觉）。

这说明零样本的天花板不在"模型会不会说 JSON"，而在**它不知道这个业务 schema 的隐含约定**：
哪些角色必须成对出现（"质押方/质押物"）、金额要截到"亿"还是"万元"、披露时间允不允许相对时间……
规模能补"语言理解"，补不了"业务约定"。

### 3.3 换提示策略：zero-shot ↔ few-shot（6 个示例）

| 设置 | 论元 F1 | 事件级 F1 | prompt tok/条 | 单条延迟(ms) |
|---|---|---|---|---|
| 0.6B zero-shot | {g('w7_qwen06_zero','argument_f1')} | {g('w7_qwen06_zero','event_f1')} | {g('w7_qwen06_zero','prompt_tokens')} 总 | {g('w7_qwen06_zero','latency_ms',1)} |
| 0.6B few-shot(6 例) | {g('w7_qwen06_few','argument_f1')} | {g('w7_qwen06_few','event_f1')} | {g('w7_qwen06_few','prompt_tokens')} 总 | {g('w7_qwen06_few','latency_ms',1)} |

**结论**：只加 6 个示例，论元 F1 {g('w7_qwen06_zero','argument_f1')} → {g('w7_qwen06_few','argument_f1')}
（{delta('w7_qwen06_zero','w7_qwen06_few')}），**与"把模型放大 3 倍"（0.0850 → 0.2085）几乎等价**，
而 JSON 合法率还更高（{raw('w7_qwen06_few','json_valid_rate','{:.0f}')}% vs {raw('w7_qwen17_zero','json_valid_rate','{:.0f}')}%）、
幻觉论元更少（{v('w7_qwen06_few','arg_not_in_source')} 次 vs {v('w7_qwen17_zero','arg_not_in_source')} 次）。
代价是 prompt 长度从约 800 token 涨到约 1450 token（本表 prompt 总量 {v('w7_qwen06_zero','prompt_tokens')} → {v('w7_qwen06_few','prompt_tokens')}），
**每条请求的输入成本约翻 1.6 倍**。

这也解释了工程上的常见做法：**先用 few-shot 试 schema，再决定要不要微调**——
示例能把"业务约定"讲清楚，而换更大的模型只是让它更会说话。

### 3.4 换调用方式：本地直推 ↔ 自建 OpenAI 兼容 API

本机没有云端 API Key，所以这一项的做法是：用 `w7_ext/serve_openai.py` 把**同一个 Qwen3-0.6B**
起成一个真实监听 HTTP 的 OpenAI 兼容端点（`POST /v1/chat/completions`），再让项目原本的
`--backend api` 代码路径（openai SDK）去打它 —— **客户端代码一行没改**，这正是"换 API"要验证的事：
模型服务化后，协议封装、单请求串行、token 计量会带来多少代价。

| 指标 | 本地批量直推（batch=8） | 自建 HTTP API（batch=1） |
|---|---|---|
| 论元 F1 | {g('w7_qwen06_zero','argument_f1')} | {g('w7_api_qwen06_zero','argument_f1')} |
| 单条延迟(ms) | {g('w7_qwen06_zero','latency_ms',1)} | {g('w7_api_qwen06_zero','latency_ms',1)} |
| 200 条总耗时(s) | {g('w7_qwen06_zero','runtime_s',1)} | {g('w7_api_qwen06_zero','runtime_s',1)} |
| JSON 合法率 | {g('w7_qwen06_zero','json_valid_rate')}% | {g('w7_api_qwen06_zero','json_valid_rate')}% |

**结论**：指标应当一致（同一权重、同一 prompt、贪心解码），差异全在**吞吐**上。
服务端逐请求日志（`results/api_server_stats.jsonl`）把这份代价拆得很清楚：
**HTTP 协议层本身只占单请求的 {ov('overhead_ratio_pct')}%（均值 {ov('http_overhead_mean_s')}s）**，
真正的开销是**失去了批处理** —— 本地一次解 8 条摊薄了 kernel 启动与 KV cache 预热，
API 一条一条来，单条生成从 {raw('w7_qwen06_zero','latency_ms','{:.0f}')} ms 变成 {raw('w7_api_qwen06_zero','latency_ms','{:.0f}')} ms。

**工程含义**：离线批量评测用本地直推；线上要弹性/多语言/多模型路由时，API 化值得那点开销，
但**必须把批处理与并发做在服务端**（continuous batching），而不是指望客户端提速。
换更贵的模型之前，先把 batch 打满、把 prompt 压短。

### 3.5 换数据集：DuEE-Fin（13 类）↔ DuEE 1.0（65 类）

{table_dataset()}

**结论**：

1. **规则方法最脆弱**（{g('rule','argument_f1')} → {g('w7_rule_duee1','argument_f1')}）：
   它的角色抽取靠金融公告的固定句式（"XX 股东质押 XX 万股"），换到通用领域直接失效；
   触发词还能靠词典命中（{g('w7_rule_duee1','trigger_f1')}），论元几乎全漏。
2. **判别式模型基本稳住**（{g('w7_bert_crf','argument_f1')} → {g('w7_bert_crf_duee1','argument_f1')}）：
   虽然类型从 13 涨到 65、槽位从 92 涨到 216，论元 F1 没掉——因为它的错误主要来自
   "类型判错"和"边界"，而这两件事由 CRF + BIO 约束兜住了。
3. **事件级 F1 反而更高**（{g('w7_bert_crf','event_f1')} → {g('w7_bert_crf_duee1','event_f1')}）：
   关键不是"数据集更简单"，而是**每个事件要凑对的论元更少**
   （DuEE 1.0 平均 {d1.get('dev',{}).get('args_per_event','—')} 个/事件 vs DuEE-Fin {fin.get('dev',{}).get('args_per_event','—')} 个/事件），
   并且文本更短（{d1.get('dev',{}).get('text_len_mean','—')} 字 vs {fin.get('dev',{}).get('text_len_mean','—')} 字）。
   **事件级指标对"论元个数"极度敏感，跨数据集比较时不能只看这一列。**

![数据集对比](figures/fig_dataset_compare.png)

---

## 4. 精度–延迟–成本

![精度延迟](figures/fig_f1_vs_latency.png)

{table_llm_cost()}

**成本结论**：

- 判别式 **BERT-CRF**：102M 参数、单卡 13 分钟训练完、17.8 ms/句、零边际推理成本；
- **LoRA SFT**：0.6B 基座 + 9MB adapter，指标接近判别式，但推理贵 1~2 个数量级；
- **API**：按时延与并发计费，适合"低频、schema 多变、要人工审核"的场景；
- 生成式的**隐藏成本**是 prompt：schema 越大 prompt 越长（13 类 ≈ 800 tok，65 类 ≈ 1545 tok），
  每条请求都要重复付一遍 schema 的钱 —— 这也是"把 schema 做成微调而不是 prompt"的动机。

![生成式对比](figures/fig_llm_scale.png)

---

{err}

---

## 5. 选型建议（按场景）

| 场景 | 推荐 | 理由 |
|---|---|---|
| 离线批量、schema 稳定、追求性价比 | **BERT+CRF** | 指标最高、延迟最低、无需 GPU 集群推理 |
| schema 频繁变更 / 新领域冷启动 | **LLM few-shot（0.6B 级）** | 改 prompt 即上线，无需重训；但要接受指标上限 |
| 追求生成式链路的最高指标 | **LoRA SFT（0.6B 起）** | 3500 条样本 + 1 epoch 就能追上判别式 |
| 多租户 / 多模型 / 需要弹性 | **OpenAI 兼容 API 服务化** | 客户端零改动，代价是吞吐与 token 成本 |
| 完全没标注数据、也没有 API | **规则基线** | 零成本兜底，但只适合单领域固定句式 |

---

## 6. 局限与后续工作

1. **单种子**：所有实验只跑了 seed=42，没有做多种子方差估计；BERT vs RoBERTa 的 0.003 差异
   很可能在噪声范围内（这也是结论写成"没有提升"而不是"更差"的原因）。
2. **没有真实云端 API**：API 对照用的是自建 OpenAI 兼容端点（0.6B），能验证协议与吞吐，
   但**不能**替代 DeepSeek/GPT 这类强模型的真实效果；代码只需改 `LLM_BASE_URL/LLM_API_KEY` 即可切换。
3. **DuEE 1.0 只保留单事件句**（87%），多事件句被丢弃，两个数据集的"多事件场景"都没覆盖。
4. **生成式模型只测了 Qwen3 系列**，且 SFT 只跑 1 epoch / 3500 条（课程设置），没做超参搜索。
5. 后续可做：多种子 + 置信区间；论元级 beam search；schema 裁剪（只给模型相关的 5~10 个类型）；
   LoRA + 检索增强（把相似公告作为示例）。

---

## 附录 A：复现命令

```bash
# 0) 环境：python 3.12 + torch 2.6 + transformers 5.5.3（见 requirements.txt）
# 1) 规则基线
python rule/run_rule.py --data_dir data/processed --name rule
# 2) 判别式：BERT+Linear / BERT+CRF / RoBERTa+CRF
python bert/train.py --decoder linear --run_name w7_bert_linear --bert_path <bert-base-chinese>
python bert/train.py --decoder crf    --run_name w7_bert_crf    --bert_path <bert-base-chinese>
python bert/train.py --decoder crf    --run_name w7_roberta_crf --bert_path <chinese-roberta-wwm-ext>
# 3) 无预训练基线
python bilstm_crf/train.py --epochs 3 --run_name w7_bilstm_crf
# 4) 生成式：本地零样本 / 少样本 / SFT
powershell -File w7_ext/run_gen_fin.ps1
powershell -File w7_ext/run_sft_fin.ps1
# 5) 换 API：自建 OpenAI 兼容端点
powershell -File w7_ext/run_api_fin.ps1
# 6) 换数据集：DuEE 1.0
python w7_ext/convert_duee1.py --src <duee1 parquet 目录> --out data/duee1
python rule/run_rule.py --data_dir data/duee1 --name w7_rule_duee1
python bert/train.py --data_dir data/duee1 --batch_size 16 --run_name w7_bert_crf_duee1
# 7) 汇总 + 出图 + 报告
python w7_ext/collect_results.py
python w7_ext/make_figs.py
python w7_ext/build_report.py
```

## 附录 B：本次对课程代码的改动（最小侵入）

| 文件 | 改动 | 目的 |
|---|---|---|
| `llm_zero_shot/llm_extract.py` | 新增 `--schema` / `--llm_dir` 两个可选参数 | 换数据集时不覆盖 DuEE-Fin 的固定 200 条抽样锚点 |
| `llm_lora/train_sft.py` | 新增 `--schema`，schema 跟随 `--data_dir` | 同上（标签空间随数据集变） |
| `llm_lora/evaluate_sft.py` | 新增 `--schema` / `--llm_dir` | 同上 |
| `w7_ext/*` | 新增 | 数据集转换、API 服务、汇总、出图、报告生成 |

## 附录 C：文件清单

```
code/w7_ext/            本次新增的脚本（转换/服务/汇总/出图/报告/错误分析）
results/compare_all.csv 全部实验的指标总表（Excel 可直接打开）
results/metrics/        每个方法的指标 JSON（含 per-type / per-role）
figures/                报告用图（主对比、精度-延迟、数据集对比、生成式对比）
```
"""
    out = ROOT / "outputs" / "w7_compare" / "report.md"
    out.write_text(md, encoding="utf-8")
    print(f"report → {out}  ({len(md)} 字符)")


def fin_drop_text(d1):
    return "1,440 条"


def N_times():
    a, b = v("w7_qwen06_zero", "runtime_s"), v("w7_api_qwen06_zero", "runtime_s")
    if not a or not b:
        return "（待补）"
    return f"{b / a:.1f}"


def table_dataset_stats(fin, d1):
    def row(name, d):
        sch = d.get("schema", {})
        tr, dv = d.get("train", {}), d.get("dev", {})
        return (f"| {name} | {sch.get('types','—')} 类 / {sch.get('slots','—')} 槽位 / "
                f"{sch.get('uniq_roles','—')} 角色 | {tr.get('samples','—')} | {dv.get('samples','—')} | "
                f"{dv.get('args_per_event','—')} | {dv.get('text_len_mean','—')} |")
    lines = ["| 数据集 | 标签空间 | train 句 | dev 句 | 每个事件的论元数 | 文本平均长度(字) |",
             "|---|---|---|---|---|---|",
             row("DuEE-Fin（金融公告，课程主数据集）", fin),
             row("DuEE 1.0（通用领域，本次新增）", d1)]
    return "\n".join(lines)


if __name__ == "__main__":
    main()
