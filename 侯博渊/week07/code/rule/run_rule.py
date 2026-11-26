# -*- coding: utf-8 -*-
"""
L0 规则基线（DuEE 1.0 适配版）：触发词词典 + 时间/数字正则 + jieba 词性实体

与原版（DuEE-Fin）的区别（换数据集必须改的三处）：
  1. 触发词表：仍然从 train 统计（≥2 次）+ 最长优先匹配，但**不再手写金融同义词表**
     —— DuEE 1.0 是 65 类开放域事件，手写 65 类同义词既不可维护、也会把测试集答案抄进来。
  2. 角色语义：DuEE 1.0 的角色名是「时间/地点/裁员方/死者/胜者/融资轮次…」，
     用「角色名关键词 → 抽取器」的通用映射，而不是金融公告专用的「股票简称词典」。
  3. 实体候选：金融公告里公司名有「有限公司/集团」后缀可抓，开放式新闻里人名/地名/机构
     没有统一后缀，所以这里用 jieba 词性标注（nr 人名 / ns 地名 / nt 机构）做候选池，
     再按角色语义依次领取 —— 这正是「规则能判类型、填不满槽位」在开放域上的放大版。

结论（写进报告）：规则在 DuEE 1.0 上 trigger 仍然能到 0.6~0.7（触发词词典覆盖率决定），
但 argument 会明显低于 DuEE-Fin —— 因为论元从「金额/日期」变成了大量自由文本实体，
没有 NER 的规则系统只能靠词性猜，且一个实体只能领一次。

用法：
  python rule/run_rule.py                     # data/processed（主实验：15 类子集）
  python rule/run_rule.py --data_dir data/processed_full65 --name rule_full65
"""

import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import argparse
import json
import re
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]   # 项目根目录
sys.path.insert(0, str(ROOT / "common"))

from event_data import (DATA_DIR, load_extract_split, load_schema, make_pred_row, row_event)
from eval_utils import METRIC_DIR, evaluate_predictions
from log_utils import EvalLogger

MOCK_DIR = ROOT / "data" / "_mock"

# 人工补充触发词：DuEE 1.0 不手写（65 类开放域，写不全且容易泄漏测试集），保留空表占位
MANUAL_TRIGGERS: dict[str, list[str]] = {}

# ── 正则库：时间 / 数字（人数、金额、幅度、年龄、刑期、震级、深度…）──
NUM = r"[0-9０-９][0-9０-９,，]*\.?[0-9０-９]*"
DATE_RE = re.compile(
    r"(?:\d{4}\s*年\s*\d{1,2}\s*月\s*\d{1,2}\s*日"
    r"|\d{4}\s*年\s*\d{1,2}\s*月"
    r"|\d{4}\s*年(?:\s*(?:上半年|下半年|第[一二三四]季度|一季度|二季度|三季度|四季度|全年))?"
    r"|\d{1,2}\s*月\s*\d{1,2}\s*日|\d{1,2}\s*月|\d{1,2}\s*日"
    r"|(?:今天|昨天|前天|近日|日前|当天|当日|本周|上周|本月|上月|今年|去年|明年|凌晨|上午|下午|晚上|晚间))")
PERCENT_RE = re.compile(rf"{NUM}\s*(?:%|％|个百分点)")
MONEY_RE = re.compile(rf"{NUM}\s*(?:亿|万|千)?\s*(?:元|美元|港元|欧元|英镑|日元|万元|亿元)")
COUNT_RE = re.compile(rf"{NUM}\s*(?:人|名|位|岁|年|天|小时|分钟|米|公里|级|次|场|部|辆|架|艘|吨|万元|亿元|%|％)")
NUMBER_RE = re.compile(NUM)

# 角色名 → 抽取器
DATE_KEYS = ("时间", "日期", "时候", "年代")
PLACE_KEYS = ("地点", "地点", "震中", "地区", "场所")
PERCENT_KEYS = ("幅度", "比例", "占比", "率")
MONEY_KEYS = ("金额", "价格", "费用", "罚款", "薪资", "票房", "融资")
COUNT_KEYS = ("人数", "年龄", "刑期", "时长", "震级", "深度", "数量", "次数")
ORG_KEYS = ("机构", "组织", "公司", "企业", "法院", "学校", "球队", "俱乐部", "方")
PERSON_KEYS = ("人", "者", "双方", "主体", "对象", "原告", "被告")


def role_to_kind(role: str) -> str:
    """把角色名归类到「用哪种抽取器」——规则系统里唯一的泛化方式。顺序有讲究。"""
    if any(k in role for k in DATE_KEYS):
        return "date"
    if any(k in role for k in PLACE_KEYS):
        return "place"
    if any(k in role for k in PERCENT_KEYS):
        return "percent"
    if any(k in role for k in MONEY_KEYS):
        return "money"
    if any(k in role for k in COUNT_KEYS):
        return "count"
    if any(k in role for k in ORG_KEYS):
        return "org"
    if any(k in role for k in PERSON_KEYS):
        return "person"
    return "entity"


# ─────────────────────────── 1. 触发词词典 ───────────────────────────

def build_trigger_dict(train_rows: list[dict], min_count: int = 2) -> dict:
    """统计 {类型: [(触发词, 频次)]}，只保留出现 ≥ min_count 次的词（去噪）。"""
    cnt = defaultdict(Counter)
    for r in train_rows:
        ev = row_event(r)
        if not ev:
            continue
        trig = (ev.get("trigger") or {}).get("text", "")
        if trig:
            cnt[ev["type"]][trig] += 1
    return {t: [(w, c) for w, c in c.most_common() if c >= min_count]
            for t, c in cnt.items()}


def merge_trigger_dict(auto: dict, manual: dict) -> list[tuple[str, str]]:
    """合并成「(触发词, 类型)」扁平表，按词长降序（最长优先，避免「质押」抢「解除质押」）。"""
    pairs = []
    for t, words in auto.items():
        for w, _ in words:
            pairs.append((w, t))
    for t, words in manual.items():
        for w in words:
            pairs.append((w, t))
    return sorted(set(pairs), key=lambda x: (-len(x[0]), x[1]))


def find_trigger(text: str, trig_pairs: list[tuple[str, str]]):
    """最长优先扫触发词，返回 (类型, 触发词, start, end)；没命中返回 None。"""
    best = None
    for word, etype in trig_pairs:
        pos = text.find(word)
        if pos < 0:
            continue
        cand = (len(word), -pos, etype)
        if best is None or cand[:2] > best[0][:2]:
            best = (cand, etype, word, pos, pos + len(word))
    if best is None:
        return None
    _, etype, word, s, e = best
    return etype, word, s, e


# ─────────────────────────── 2. 实体候选池（jieba 词性）───────────────────────────

def build_entity_pool(text: str) -> list[dict]:
    """
    用 jieba 词性标注抽候选实体：nr 人名 / ns 地名 / nt 机构名 / nz 专名。

    为什么用词性而不是词典：DuEE 1.0 的论元大量是人名、地名、赛事名、影视名，
    没有「有限公司」这种统一后缀可抓；jieba 的 nr/ns/nt 是最便宜的可解释替代。
    代价也很清楚：分词错 → 实体边界错，且新闻里的人名经常被切成普通名词。
    """
    import jieba.posseg as pseg

    pool = []
    for w, flag in pseg.cut(text):
        w = w.strip()
        if len(w) < 2:
            continue
        if flag in ("nr", "ns", "nt", "nz", "nrt", "nrfg", "s", "j"):
            kind = "person" if flag.startswith("nr") else ("place" if flag in ("ns", "s") else "org")
            pool.append({"text": w, "start": text.find(w), "end": text.find(w) + len(w), "kind": kind})
    # 去掉位置找不到的（jieba 会把空白/标点单独切），按出现位置排序
    pool = [c for c in pool if c["start"] >= 0]
    return sorted(pool, key=lambda x: x["start"])


ENTITIES = None


def extract_args(text: str, etype: str, role: str, pool: list[dict],
                 used: list[tuple[int, int]]) -> list[dict]:
    """按角色语义抽一个角色（每个角色最多 1 个；实体从候选池里按出现顺序领）。"""
    kind = role_to_kind(role)
    out = []

    def free(s, e):
        return all(not (s < ue and e > us) for us, ue in used)

    if kind in ("date", "percent", "money", "count"):
        pat = {"date": DATE_RE, "percent": PERCENT_RE, "money": MONEY_RE, "count": COUNT_RE}[kind]
        for m in pat.finditer(text):
            s, e = m.start(), m.end()
            if free(s, e):
                out.append({"role": role, "text": text[s:e].strip(), "start": s, "end": e})
                break   # 每个角色只领 1 个：规则分不清哪个数字属于哪个角色

    elif kind in ("person", "place", "org", "entity"):
        want = {"person": ("person",), "place": ("place",), "org": ("org",),
                "entity": ("person", "org", "place")}[kind]
        for i, c in enumerate(pool):
            if c["kind"] in want and free(c["start"], c["end"]):
                out.append({"role": role, "text": c["text"], "start": c["start"], "end": c["end"]})
                pool.pop(i)
                break
    return out


def extract_one(text: str, schema: dict, trig_pairs: list[tuple[str, str]]) -> dict | None:
    hit = find_trigger(text, trig_pairs)
    if hit is None:
        return None
    etype, word, ts, te = hit
    roles = schema.get(etype, [])
    used = [(ts, te)]
    pool = build_entity_pool(text)
    args = []
    for role in roles:
        got = extract_args(text, etype, role, pool, used)
        for a in got:
            used.append((a["start"], a["end"]))
        args.extend(got)
    return {"type": etype, "trigger": {"text": word, "start": ts, "end": te}, "args": args}


# ─────────────────────────── 3. 失败分析 ───────────────────────────

def analyze_failures(gold_rows: list[dict], pred_rows: list[dict], top_n: int = 8) -> dict:
    """把错误拆成互斥类别：no_trigger / wrong_type / wrong_trigger / arg_missing / arg_spurious / exact。"""
    pred_map = {(r.get("id"), r.get("sent_id")): r for r in pred_rows}
    cats = Counter()
    examples = defaultdict(list)
    n_trig_hit = 0

    for g in gold_rows:
        ge = row_event(g) or {}
        p = pred_map.get((g.get("id"), g.get("sent_id")), {"events": []})
        pe = (p.get("events") or [{}])[0] if p.get("events") else {}
        g_trig = (ge.get("trigger") or {}).get("text", "")
        p_trig = (pe.get("trigger") or {}).get("text", "")
        g_args = {(a["role"], a["text"]) for a in ge.get("args") or []}
        p_args = {(a["role"], a["text"]) for a in pe.get("args") or []}

        if not p_trig:
            cat = "no_trigger"
        else:
            n_trig_hit += 1
            if pe.get("type") != ge.get("type"):
                cat = "wrong_type"
            elif p_trig != g_trig:
                cat = "wrong_trigger"
            elif p_args == g_args:
                cat = "exact"
            elif g_args - p_args and not (p_args - g_args):
                cat = "arg_missing"
            else:
                cat = "arg_spurious" if p_args - g_args else "arg_missing"
        cats[cat] += 1
        if len(examples[cat]) < top_n:
            examples[cat].append({"text": g.get("text", ""),
                                  "gold": {"type": ge.get("type"), "trigger": g_trig,
                                           "args": sorted(g_args)},
                                  "pred": {"type": pe.get("type"), "trigger": p_trig,
                                           "args": sorted(p_args)}})

    total = max(len(gold_rows), 1)
    return {
        "counts": dict(cats),
        "ratios": {k: round(v / total, 4) for k, v in cats.items()},
        "trigger_hit_rate": round(n_trig_hit / total, 4),
        "examples": dict(examples),
    }


# ─────────────────────────── 主流程 ───────────────────────────

def run(data_dir: Path, name: str, min_trigger: int) -> dict:
    eval_logger = EvalLogger(method=name, root=ROOT, notes="L0 规则基线（触发词词典 + 正则 + jieba 词性）")
    schema = load_schema(data_dir / "event_schema.json")
    train_rows = load_extract_split("train", data_dir)
    dev_rows = load_extract_split("dev", data_dir)

    auto = build_trigger_dict(train_rows, min_count=min_trigger)
    trig_pairs = merge_trigger_dict(auto, MANUAL_TRIGGERS)
    print(f"触发词表：{len(trig_pairs)} 条（自动 {sum(len(v) for v in auto.values())}，人工 0）")

    t0 = time.perf_counter()
    preds = []
    for r in dev_rows:
        ev = extract_one(r["text"], schema, trig_pairs)
        preds.append(make_pred_row(r, [ev] if ev else []))
    elapsed = time.perf_counter() - t0
    ms_per_sent = elapsed * 1000 / max(len(dev_rows), 1)
    print(f"推理完成：{len(dev_rows)} 句，耗时 {elapsed:.2f}s（{ms_per_sent:.3f} ms/句，纯 CPU）")

    meta = {
        "display_name": "L0 规则（词典+正则+词性）",
        "train_s": 0,
        "params_m": 0,
        "ms_per_sentence": round(ms_per_sent, 3),
        "num_triggers": len(trig_pairs),
        "min_trigger_count": min_trigger,
        "note": "零训练基线；触发词从 train 统计（最长匹配），时间/数字用正则，实体用 jieba 词性",
    }
    res = evaluate_predictions(name, preds, meta=meta,
                               gold_path=data_dir / "dev_extract.jsonl", with_detect=False)

    failures = analyze_failures(dev_rows, preds)
    res["failure_analysis"] = failures
    METRIC_DIR.mkdir(parents=True, exist_ok=True)
    with open(METRIC_DIR / f"{name}.json", "w", encoding="utf-8") as f:
        json.dump(res, f, ensure_ascii=False, indent=2)

    print("\n【规则基线失败分析】")
    for k, v in sorted(failures["counts"].items(), key=lambda x: -x[1]):
        print(f"  {k:14s} {v:5d} 条 ({failures['ratios'][k]:.1%})")
    print(f"  触发词命中率（整句有事件的比例）：{failures['trigger_hit_rate']:.1%}")
    eval_logger.finalize(
        metrics={k: res[k]["f1"] for k in ("trigger", "argument", "event_exact")},
        type_macro_f1=res.get("type_macro_f1"),
        num_triggers=len(trig_pairs),
        trigger_hit_rate=failures["trigger_hit_rate"],
        failure_counts=failures["counts"],
    )
    return res


def main():
    ap = argparse.ArgumentParser(description="L0 规则基线：触发词词典 + 正则 + 词性实体")
    ap.add_argument("--mock", action="store_true", help="用 data/_mock 的 20 条手造数据自检")
    ap.add_argument("--data_dir", default="", help="数据目录；留空=全量 data/processed")
    ap.add_argument("--name", default="", help="预测/指标文件名，默认 rule")
    ap.add_argument("--min_trigger_count", type=int, default=2)
    args = ap.parse_args()

    data_dir = (Path(args.data_dir) if args.data_dir
                else (MOCK_DIR if args.mock else DATA_DIR))
    name = args.name or ("rule_mock" if args.mock else "rule")
    print(f"数据目录：{data_dir}")
    run(data_dir, name, args.min_trigger_count)


if __name__ == "__main__":
    main()
