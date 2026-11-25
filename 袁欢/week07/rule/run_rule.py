"""
L0 规则基线：触发词词典的「零训练」事件检测

为什么课里要从规则讲起（而不是直接上 BERT）：
  规则基线是**最诚实的对照组**。它告诉我们「不用学模型、只靠词典能拿到多少」，
  后面的 BERT / CRF / 两段式涨了多少点，才有意义。
  更重要的是：规则的**失败模式非常清楚**，正好用来引出后面每个模型要解决的问题：
    - 触发词没收录 → 整句漏抽（召回天花板卡在词典覆盖率上）
    - 一个词有多种事件类型（「漏洞发现」vs「漏洞补丁」）→ 类型判错
    - 触发词是自由变体（句中写「电子邮件钓鱼攻击」、词典只有「网络钓鱼」）→ 扫不中

口径说明（2026-11-24 换装 CCED 起）：论元链路已砍，规则只做
「事件类型 + 触发词 span」；原金融版的金额/日期正则与公司名词典已删除。

做法（全部从训练集自动统计，不手写几百个词）：
  1. 触发词：统计训练集里各类型出现过的 trigger 文本，保留出现 ≥2 次的（去噪），
     再加一小张人工同义词表补召回。匹配时**最长优先**（否则「供应链攻击」会被「供应链」抢走）。
  2. 论元：CCED 无论元标注，本方法不再抽论元（事件检测口径）。

使用方式：
  python rule/run_rule.py                          # 跑 dev 集，出指标 + 失败分析
  python rule/run_rule.py --mock                   # 用 data/_mock 的 20 条手造数据自检
  python rule/run_rule.py --min_trigger_count 3    # 提高触发词入典门槛
"""

import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]   # 项目根目录（方法目录的上一级）
sys.path.insert(0, str(ROOT / "common"))     # 公共模块：event_data / crf / eval_utils / models / log_utils

from event_data import (DATA_DIR, ROOT, load_extract_split,
                        make_pred_row, row_event)
from eval_utils import METRIC_DIR, evaluate_predictions
from log_utils import EvalLogger

MOCK_DIR = ROOT / "data" / "_mock"

# ── 人工补充的同义词 / 变体（教学点：统计自动入典之后，人工补的边际收益有多大）──
# 只补「训练集里没出现、但网络安全新闻里显然会出现」的说法，避免把测试集答案抄进来。
# 注意与类型归并口径一致（「安全漏洞」按归并表属漏洞影响）。
MANUAL_TRIGGERS = {
    "数据泄露": ["数据泄露", "信息泄露", "泄露了"],
    "网络钓鱼": ["网络钓鱼", "钓鱼攻击", "钓鱼邮件", "钓鱼网站"],
    "勒索赎金": ["勒索", "赎金", "勒索软件", "敲诈"],
    "DDoS攻击": ["DDoS攻击", "DDos攻击", "分布式拒绝服务", "拒绝服务攻击"],
    "恶意软件": ["恶意软件", "木马", "僵尸网络", "后门程序"],
    "供应链攻击": ["供应链攻击"],
    "漏洞影响": ["存在漏洞", "安全漏洞"],
    "漏洞发现": ["发现漏洞", "披露漏洞", "爆出漏洞"],
    "漏洞补丁": ["漏洞补丁", "修复漏洞", "发布补丁", "安全补丁"],
}


# ─────────────────────────── 1. 从训练集统计触发词 ───────────────────────────

def build_trigger_dict(train_rows: list[dict], min_count: int = 2) -> dict:
    """
    统计 {类型: [(触发词, 频次), ...]}。

    只保留出现 ≥ min_count 次的触发词：新闻里有很多「一次性」表述
    （某家媒体独有的写法），把它们全收进词典会让规则在训练集上虚高、在验证集上不泛化。
    """
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
    """
    合并成「(触发词, 类型) 的扁平列表」，按**词长降序**排好。

    为什么要按长度排序：触发词存在包含关系 ——「供应链攻击」里含「供应链」，
    「漏洞补丁」里含「补丁」。如果把短的排在前面，一句「漏洞补丁已发布」会被判错类型，
    macro F1 会掉得很明显。最长优先（maximal match）是规则系统的标准做法。
    """
    pairs = []
    for t, words in auto.items():
        for w, _ in words:
            pairs.append((w, t))
    for t, words in manual.items():
        for w in words:
            pairs.append((w, t))
    # 去重（同一 (词,类型) 只留一份），再按词长降序、类型名升序保证确定性
    pairs = sorted(set(pairs), key=lambda x: (-len(x[0]), x[1]))
    return pairs


# ─────────────────────────── 2. 单句抽取 ───────────────────────────

def find_trigger(text: str, trig_pairs: list[tuple[str, str]]) -> tuple[str, str, int, int] | None:
    """最长优先扫触发词，返回 (类型, 触发词, start, end)；没命中返回 None。"""
    best = None
    for word, etype in trig_pairs:
        pos = text.find(word)
        if pos < 0:
            continue
        # 词长优先；同长度时取最靠前出现的位置（贴近人类阅读顺序）
        cand = (len(word), -pos)
        if best is None or cand > best[0]:
            best = (cand, etype, word, pos, pos + len(word))
    if best is None:
        return None
    _, etype, word, s, e = best
    return etype, word, s, e


def extract_one(text: str, trig_pairs: list[tuple[str, str]]) -> dict | None:
    """对一句话抽一个事件（类型+触发词）；抽不到返回 None（评测时写成 events: []）。"""
    hit = find_trigger(text, trig_pairs)
    if hit is None:
        return None
    etype, word, s, e = hit
    return {"type": etype, "trigger": {"text": word, "start": s, "end": e}, "args": []}


# ─────────────────────────── 3. 失败分析（教学点）───────────────────────────

def analyze_failures(gold_rows: list[dict], pred_rows: list[dict], top_n: int = 8) -> dict:
    """
    把错误拆成互斥的几类，告诉学生「规则到底输在哪」：

      no_trigger      没扫到触发词 → 整句漏抽（词典覆盖率 = 召回天花板）
      wrong_type      触发词命中了，但类型判错（「漏洞发现」vs「漏洞补丁」）
      wrong_trigger   类型对了，触发词边界/文本不对
      exact           完全正确（类型 + 触发词）
    """
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

        if not p_trig:
            cat = "no_trigger"
        else:
            n_trig_hit += 1
            if pe.get("type") != ge.get("type"):
                cat = "wrong_type"
            elif p_trig != g_trig:
                cat = "wrong_trigger"
            else:
                cat = "exact"
        cats[cat] += 1
        if len(examples[cat]) < top_n:
            examples[cat].append({"text": g.get("text", ""),
                                  "gold": {"type": ge.get("type"), "trigger": g_trig},
                                  "pred": {"type": pe.get("type"), "trigger": p_trig}})

    total = max(len(gold_rows), 1)
    return {
        "counts": dict(cats),
        "ratios": {k: round(v / total, 4) for k, v in cats.items()},
        "trigger_hit_rate": round(n_trig_hit / total, 4),
        "examples": dict(examples),
    }


# ─────────────────────────── 主流程 ───────────────────────────

def run(data_dir: Path, name: str, min_trigger: int) -> dict:
    # 规则基线不训练，用 EvalLogger 记录"这次评测用了什么词典、结果如何"
    eval_logger = EvalLogger(method=name, root=ROOT, notes="L0 规则基线（触发词词典）")
    train_rows = load_extract_split("train", data_dir)
    dev_rows = load_extract_split("dev", data_dir)

    auto = build_trigger_dict(train_rows, min_count=min_trigger)
    trig_pairs = merge_trigger_dict(auto, MANUAL_TRIGGERS)
    print(f"触发词词典：{len(trig_pairs)} 条（自动 {sum(len(v) for v in auto.values())} + "
          f"人工 {sum(len(v) for v in MANUAL_TRIGGERS.values())}，含去重）")

    import time
    t0 = time.perf_counter()
    preds = []
    for r in dev_rows:
        ev = extract_one(r["text"], trig_pairs)
        preds.append(make_pred_row(r, [ev] if ev else []))
    elapsed = time.perf_counter() - t0
    ms_per_sent = elapsed * 1000 / max(len(dev_rows), 1)
    print(f"推理完成：{len(dev_rows)} 句，耗时 {elapsed:.2f}s（{ms_per_sent:.3f} ms/句，纯 CPU）")

    meta = {
        "display_name": "L0 规则（触发词词典）",
        "train_s": 0,
        "params_m": 0,
        "ms_per_sentence": round(ms_per_sent, 3),
        "num_triggers": len(trig_pairs),
        "min_trigger_count": min_trigger,
        "note": "零训练基线；触发词从 train 统计 + 人工同义词（事件检测口径，无论元）",
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
        metrics={k: res[k]["f1"] for k in ("trigger", "event_exact")},
        type_macro_f1=res.get("type_macro_f1"),
        num_triggers=len(trig_pairs),
        trigger_hit_rate=failures["trigger_hit_rate"],
        failure_counts=failures["counts"],
    )
    return res


def main():
    ap = argparse.ArgumentParser(description="L0 规则基线：触发词词典（事件检测口径）")
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
