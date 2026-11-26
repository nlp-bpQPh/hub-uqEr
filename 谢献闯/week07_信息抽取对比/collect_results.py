# -*- coding: utf-8 -*-
"""
汇总所有实验的指标，生成一张「同一口径」的对比表（CSV + Markdown + JSON）。

设计原则（和项目 README 一致）：
  - 只读 outputs/metrics 下真实存在的 JSON；**缺文件就跳过并在报告里写"未跑"**，
    绝不为了表格好看填估算值。
  - 判别式方法报 dev 全量（DuEE-Fin 930 句 / DuEE1 1258 句）+ n=200 子集；
    生成式方法只跑固定 200 条抽样（seed=42 的 eval_ids.json），因此"全量"列写"仅 200"。
  - 表里的延迟/成本字段来自各自 run 的日志（训练时长/推理延迟/token 数），
    不在这里重新测——保证"报告里的数字 = 当次运行的日志"。

用法：
  python w7_ext/collect_results.py
产物：
  outputs/w7_compare/compare_all.csv / .md / .json
"""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MET_DISC = ROOT / "outputs" / "metrics" / "disc"
MET_LLM = ROOT / "outputs" / "metrics" / "llm"
LLM_REPORT = ROOT / "outputs" / "llm"
OUT_DIR = ROOT / "outputs" / "w7_compare"

# 实验登记表：name 必须与指标文件名一致
EXPERIMENTS = [
    # ── DuEE-Fin（金融公告，13 类 / 92 槽位）──
    dict(dataset="DuEE-Fin", layer="L0", name="rule", display="规则（词典+正则）", kind="disc",
         goal="不用模型的零训练基线"),
    dict(dataset="DuEE-Fin", layer="L1", name="w7_bilstm_crf", display="BiLSTM-CRF（无预训练）",
         kind="disc", goal="不借预训练能到多少"),
    dict(dataset="DuEE-Fin", layer="L2", name="w7_bert_linear", display="BERT+Linear（联合 149 标签）",
         kind="disc", goal="预训练 + 线性解码"),
    dict(dataset="DuEE-Fin", layer="L2", name="w7_bert_crf", display="BERT+CRF（联合 149 标签）",
         kind="disc", goal="CRF 约束解码"),
    dict(dataset="DuEE-Fin", layer="L2", name="w7_roberta_crf", display="RoBERTa-wwm-ext+CRF",
         kind="disc", goal="换预训练骨干（同架构不同语料）"),
    dict(dataset="DuEE-Fin", layer="L2", name="w7_pipeline", display="两段式流水线（检测→类型→论元）",
         kind="disc", goal="拆细任务的收益与代价", optional=True),
    dict(dataset="DuEE-Fin", layer="L3", name="w7_qwen06_zero", display="Qwen3-0.6B zero-shot",
         kind="llm", n=200, goal="生成式零样本（小模型）"),
    dict(dataset="DuEE-Fin", layer="L3", name="w7_qwen17_zero", display="Qwen3-1.7B zero-shot",
         kind="llm", n=200, goal="换模型规模（×3 参数）"),
    dict(dataset="DuEE-Fin", layer="L3", name="w7_qwen06_few", display="Qwen3-0.6B few-shot(6 例)",
         kind="llm", n=200, goal="换提示策略（示例注入）"),
    dict(dataset="DuEE-Fin", layer="L4", name="w7_qwen06_sft", display="Qwen3-0.6B + LoRA SFT",
         kind="llm", n=200, goal="生成式微调（小模型）"),
    dict(dataset="DuEE-Fin", layer="L4", name="w7_qwen17_sft", display="Qwen3-1.7B + LoRA SFT",
         kind="llm", n=200, goal="生成式微调（大一点的模型）"),
    dict(dataset="DuEE-Fin", layer="API", name="w7_api_qwen06_zero",
         display="Qwen3-0.6B via 自建 OpenAI API", kind="llm", n=200,
         goal="换调用方式（HTTP 协议封装 vs 本地直推）"),
    dict(dataset="DuEE-Fin", layer="API", name="w7_api_qwen06_few",
         display="Qwen3-0.6B few-shot via API", kind="llm", n=200,
         goal="API + few-shot 的成本/收益"),
    # ── DuEE 1.0（通用领域，65 类 / 216 槽位）──
    dict(dataset="DuEE1.0", layer="L0", name="w7_rule_duee1", display="规则（词典+正则）", kind="disc",
         goal="同方法换数据集（schema ×5）"),
    dict(dataset="DuEE1.0", layer="L2", name="w7_bert_crf_duee1", display="BERT+CRF", kind="disc",
         goal="判别式模型在 65 类上的表现"),
    dict(dataset="DuEE1.0", layer="L3", name="w7_qwen06_zero_duee1", display="Qwen3-0.6B zero-shot",
         kind="llm", n=200, goal="零样本在新领域/大 schema 下"),
    dict(dataset="DuEE1.0", layer="L4", name="w7_qwen06_sft_duee1", display="Qwen3-0.6B + LoRA SFT",
         kind="llm", n=200, goal="微调后迁移到新数据集", optional=True),
]


def _f(v, nd=4):
    return round(float(v), nd) if isinstance(v, (int, float)) else v


# 生成式模型的参数量（按模型型号登记；不是实测值，表里注明）
LLM_PARAMS_M = {
    "w7_qwen06_zero": 0.6, "w7_qwen06_few": 0.6, "w7_qwen06_sft": 0.6,
    "w7_api_qwen06_zero": 0.6, "w7_api_qwen06_few": 0.6,
    "w7_qwen06_zero_duee1": 0.6, "w7_qwen06_sft_duee1": 0.6,
    "w7_qwen17_zero": 1.7, "w7_qwen17_sft": 1.7,
}


def load_metrics(exp):
    base = MET_DISC if exp["kind"] == "disc" else MET_LLM
    p = base / f"{exp['name']}.json"
    if not p.exists():
        return None, None
    m = json.loads(p.read_text(encoding="utf-8"))
    rep = None
    rp = LLM_REPORT / f"report_{exp['name']}.json"
    if rp.exists():
        rep = json.loads(rp.read_text(encoding="utf-8"))
    return m, rep


def row_from(exp):
    m, rep = load_metrics(exp)
    if m is None:
        return {"status": "未跑", "name": exp["name"],
                **{k: exp[k] for k in ("dataset", "layer", "display", "goal")}}
    meta = m.get("meta") or {}
    r = {
        "status": "已跑",
        "dataset": exp["dataset"],
        "layer": exp["layer"],
        "name": exp["name"],
        "display": exp["display"],
        "goal": exp["goal"],
        "trigger_f1": _f(m["trigger"]["f1"]),
        "argument_f1": _f(m["argument"]["f1"]),
        "event_f1": _f(m["event_exact"]["f1"]),
        "type_macro_f1": _f(m.get("type_macro_f1", 0.0)),
        "n_samples_eval": m.get("num_samples"),
    }
    if exp["kind"] == "disc":
        n2 = m.get("n200_summary") or {}
        # 只有 DuEE-Fin 有固定 200 条抽样锚点；DuEE1.0 的 n200 投影对着 Fin 的 id，
        # 无重叠会算出 0（假数字），这里直接置空，绝不写进表里。
        if exp["dataset"] != "DuEE-Fin":
            n2 = {}
        r.update({
            "trigger_f1_n200": _f(n2.get("trigger_f1")) if n2 else None,
            "argument_f1_n200": _f(n2.get("argument_f1")) if n2 else None,
            "event_f1_n200": _f(n2.get("event_f1")) if n2 else None,
            "params_m": meta.get("params_m"),
            "train_s": meta.get("train_s"),
            "latency_ms": meta.get("ms_per_sentence"),
            "illegal_rate": (meta.get("illegal_dev") or {}).get("illegal_seq_rate"),
            "peak_gpu_mb": meta.get("peak_gpu_mb"),
        })
    else:
        r.update({
            "trigger_f1_n200": r["trigger_f1"],
            "argument_f1_n200": r["argument_f1"],
            "event_f1_n200": r["event_f1"],
            "params_m": LLM_PARAMS_M.get(exp["name"]),
            "train_s": meta.get("train_s"),
            "latency_ms": round(1000 * rep["latency_mean_s"], 1) if rep and rep.get("latency_mean_s") else None,
            "illegal_rate": None,
            "peak_gpu_mb": meta.get("peak_gpu_mb"),
            "json_valid_rate": rep.get("json_valid_rate") if rep else None,
            "parse_fail": rep.get("parse_fail") if rep else None,
            "arg_not_in_source": rep.get("arg_not_in_source") if rep else None,
            "total_tokens": rep.get("total_tokens") if rep else None,
            "prompt_tokens": rep.get("prompt_tokens") if rep else None,
            "completion_tokens": rep.get("completion_tokens") if rep else None,
            "runtime_s": rep.get("total_s") if rep else None,
            "mode": rep.get("mode") if rep else None,
            "backend": rep.get("backend") if rep else None,
        })
    return r


def fmt(v, nd=4):
    if v is None:
        return "—"
    if isinstance(v, float):
        return f"{v:.{nd}f}"
    return str(v)


def to_md(rows, dataset):
    sub = [r for r in rows if r["dataset"] == dataset and r["status"] == "已跑"]
    head = ("| 层 | 方法 | 触发词F1 | 论元F1 | 事件F1 | 类型macroF1 | n=200 论元F1 | "
            "n=200 事件F1 | 参数(M) | 训练(s) | 延迟(ms/句) | 非法序列占比 |")
    sep = "|" + "|".join(["---"] * 12) + "|"
    lines = [f"### {dataset}", "", head, sep]
    for r in sub:
        lines.append("| " + " | ".join([
            r["layer"], r["display"], fmt(r["trigger_f1"]), fmt(r["argument_f1"]),
            fmt(r["event_f1"]), fmt(r["type_macro_f1"]),
            fmt(r.get("argument_f1_n200")), fmt(r.get("event_f1_n200")),
            fmt(r.get("params_m"), 1), fmt(r.get("train_s"), 0), fmt(r.get("latency_ms"), 1),
            (f"{r['illegal_rate']:.2%}" if isinstance(r.get("illegal_rate"), float) else "—"),
        ]) + " |")
    missing = [r["display"] for r in rows if r["dataset"] == dataset and r["status"] == "未跑"]
    if missing:
        lines += ["", "> 未跑：" + "、".join(missing)]
    return "\n".join(lines)


def to_md_llm(rows, dataset):
    sub = [r for r in rows if r["dataset"] == dataset and r["status"] == "已跑"
           and r["layer"] in ("L3", "L4", "API")]
    if not sub:
        return ""
    head = ("| 方法 | 后端/模式 | 论元F1 | 事件F1 | JSON合法率 | 幻觉论元数 | "
            "延迟(ms/条) | prompt tok | completion tok | 总耗时(s) |")
    sep = "|" + "|".join(["---"] * 10) + "|"
    lines = [f"### {dataset} · 生成式链路工程指标", "", head, sep]
    for r in sub:
        lines.append("| " + " | ".join([
            r["display"], f'{r.get("backend") or "—"}/{r.get("mode") or "—"}',
            fmt(r["argument_f1"]), fmt(r["event_f1"]),
            (f'{r["json_valid_rate"]}%' if r.get("json_valid_rate") is not None else "—"),
            fmt(r.get("arg_not_in_source")), fmt(r.get("latency_ms"), 1),
            fmt(r.get("prompt_tokens"), 0), fmt(r.get("completion_tokens"), 0),
            fmt(r.get("runtime_s"), 1),
        ]) + " |")
    return "\n".join(lines)


def api_overhead():
    """服务端逐请求日志 → 「HTTP 协议开销」占单请求总耗时的比例。

    意义：把「服务化的代价」拆成两块 —— 协议/网络（http_total - server_gen）与
    失去批处理（单条生成 vs 批量摊薄）。报告 3.4 节直接引用这两个数。
    """
    p = ROOT / "outputs" / "runlogs" / "api_server_stats.jsonl"
    if not p.exists():
        return None
    rows = [json.loads(l) for l in p.read_text(encoding="utf-8").splitlines() if l.strip()]
    if not rows:
        return None
    diffs = [r["http_total_s"] - r["server_gen_s"] for r in rows if "server_gen_s" in r]
    gens = [r["server_gen_s"] for r in rows if "server_gen_s" in r]
    diffs.sort()
    out = {
        "requests": len(rows),
        "http_overhead_mean_s": round(sum(diffs) / len(diffs), 4),
        "http_overhead_p50_s": round(diffs[len(diffs) // 2], 4),
        "http_overhead_max_s": round(max(diffs), 4),
        "server_gen_mean_s": round(sum(gens) / len(gens), 4),
        "overhead_ratio_pct": round(100 * (sum(diffs) / len(diffs)) / (sum(gens) / len(gens)), 2),
        "total_prompt_tokens": sum(r.get("prompt_tokens", 0) for r in rows),
        "total_completion_tokens": sum(r.get("completion_tokens", 0) for r in rows),
    }
    (OUT_DIR / "api_overhead.json").write_text(
        json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    return out


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rows = [row_from(e) for e in EXPERIMENTS]

    (OUT_DIR / "compare_all.json").write_text(
        json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")

    # CSV（中文表头，Excel 直接打开）
    cols = ["dataset", "layer", "display", "trigger_f1", "argument_f1", "event_f1",
            "type_macro_f1", "argument_f1_n200", "event_f1_n200", "params_m", "train_s",
            "latency_ms", "illegal_rate", "json_valid_rate", "arg_not_in_source",
            "prompt_tokens", "completion_tokens", "runtime_s"]
    lines = [",".join(cols)]
    for r in rows:
        if r["status"] != "已跑":
            continue
        lines.append(",".join(str(r.get(c, "")) if r.get(c) is not None else "" for c in cols))
    (OUT_DIR / "compare_all.csv").write_text("\n".join(lines) + "\n", encoding="utf-8-sig")

    md = ["# 事件抽取方法对比（统一评测口径 common/evaluate_events.py）", ""]
    md.append("> 表中每个数字都来自 `outputs/metrics/**` 下的真实运行结果；缺文件的实验标为「未跑」。")
    md.append("")
    md.append(to_md(rows, "DuEE-Fin"))
    md.append("")
    md.append(to_md_llm(rows, "DuEE-Fin"))
    md.append("")
    md.append(to_md(rows, "DuEE1.0"))
    md.append("")
    md.append(to_md_llm(rows, "DuEE1.0"))
    (OUT_DIR / "compare_all.md").write_text("\n".join(md) + "\n", encoding="utf-8")

    done = sum(1 for r in rows if r["status"] == "已跑")
    print(f"已汇总 {done}/{len(rows)} 个实验 → {OUT_DIR}")
    ov = api_overhead()
    if ov:
        print(f"API 服务端日志：{ov['requests']} 个请求，HTTP 层开销均值 "
              f"{ov['http_overhead_mean_s']}s（占单请求 {ov['overhead_ratio_pct']}%），"
              f"纯生成均值 {ov['server_gen_mean_s']}s")
    for r in rows:
        if r["status"] != "已跑":
            print(f"  [未跑] {r['display']}")


if __name__ == "__main__":
    main()
