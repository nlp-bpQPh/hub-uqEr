"""评估指标：实体级 P/R/F1（BIO 解码）+ token 级准确率 + 分类型统计。

序列标注真正关心的是"实体"而不是"标签 token"：
  一个实体只有边界和类型都正确才算 TP，因此实体级 F1 是主指标。
"""
from __future__ import annotations

from collections import defaultdict


def decode_entities(tags: list[str]) -> set[tuple[str, int, int]]:
    """BIO → {(type, start, end)}；非法 I- 开头按 B- 处理，保证鲁棒。"""
    ents: set[tuple[str, int, int]] = set()
    cur: tuple[str, int, int] | None = None
    for i, tag in enumerate(tags):
        if tag == "O":
            if cur:
                ents.add(cur)
                cur = None
        elif tag.startswith("B-"):
            if cur:
                ents.add(cur)
            cur = (tag[2:], i, i + 1)
        elif tag.startswith("I-"):
            typ = tag[2:]
            if cur and cur[0] == typ:
                cur = (typ, cur[1], i + 1)
            else:
                if cur:
                    ents.add(cur)
                cur = (typ, i, i + 1)
        else:
            raise ValueError(f"未知标签: {tag}")
    if cur:
        ents.add(cur)
    return ents


def _prf(tp: int, fp: int, fn: int) -> dict:
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    f = 2 * p * r / (p + r) if p + r else 0.0
    return {"precision": p, "recall": r, "f1": f, "tp": tp, "fp": fp, "fn": fn}


def evaluate(gold_tags: list[list[str]], pred_tags: list[list[str]]) -> dict:
    """gold/pred 都是"每句话的标签列表"。返回 micro / per-type / token_acc。"""
    assert len(gold_tags) == len(pred_tags)

    tp = fp = fn = 0
    per_type_tp: dict[str, int] = defaultdict(int)
    per_type_fp: dict[str, int] = defaultdict(int)
    per_type_fn: dict[str, int] = defaultdict(int)

    correct = total = 0
    for gold, pred in zip(gold_tags, pred_tags):
        for g, p in zip(gold, pred):
            total += 1
            correct += int(g == p)

        g_ents = decode_entities(gold)
        p_ents = decode_entities(pred)
        tp_set = g_ents & p_ents
        fp_set = p_ents - g_ents
        fn_set = g_ents - p_ents
        tp += len(tp_set)
        fp += len(fp_set)
        fn += len(fn_set)
        for typ, _, _ in tp_set:
            per_type_tp[typ] += 1
        for typ, _, _ in fp_set:
            per_type_fp[typ] += 1
        for typ, _, _ in fn_set:
            per_type_fn[typ] += 1

    micro = _prf(tp, fp, fn)
    micro["token_acc"] = correct / total if total else 0.0
    per_type = {
        typ: _prf(per_type_tp[typ], per_type_fp[typ], per_type_fn[typ])
        for typ in sorted(set(per_type_tp) | set(per_type_fp) | set(per_type_fn))
    }
    return {"micro": micro, "per_type": per_type}


def format_report(metrics: dict, title: str = "") -> str:
    m = metrics["micro"]
    lines = []
    if title:
        lines.append(f"===== {title} =====")
    lines.append(
        f"entity P={m['precision']:.4f}  R={m['recall']:.4f}  F1={m['f1']:.4f}  "
        f"(tp={m['tp']} fp={m['fp']} fn={m['fn']})  token_acc={m['token_acc']:.4f}"
    )
    lines.append(f"{'类型':<8}{'P':>8}{'R':>8}{'F1':>8}{'tp':>8}{'fp':>8}{'fn':>8}")
    for typ, v in metrics["per_type"].items():
        lines.append(
            f"{typ:<8}{v['precision']:>8.4f}{v['recall']:>8.4f}{v['f1']:>8.4f}"
            f"{v['tp']:>8}{v['fp']:>8}{v['fn']:>8}"
        )
    return "\n".join(lines)
