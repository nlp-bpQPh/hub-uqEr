# -*- coding: utf-8 -*-
"""按已定契约，把 CCED / CSED.json（中文网络安全事件检测语料）处理成 week7 教学用的数据文件。

与 build_dataset.py（DuEE-Fin 版）的关系：本脚本是换装 CCED 后的替代构建入口，
契约（data/processed/ 字段）保持一致，下游方法与评测不用改读取代码。

已定口径（2026-11-24 用户确认，详见 MIGRATION_PLAN.md）：
  1. **事件检测降级**：CSED 只有「类型+触发词」，论元链路已砍——event.args 恒空、
     event_schema.json 角色列表全空；
  2. 类型归并 18→9（见 TYPE_MAP；空串/漏洞/安全漏洞 3 条并入或丢弃见 DROPPED）；
  3. 95 行多触发词取首个位置；
  4. 无负样本 → 不产 _detect.jsonl（两段式已砍检测段）；
  5. 无篇章 → dev_docs 每句包成单句「文档」保 service 可跑；
  6. 无官方切分 → stratify 8:2 自切，seed=42。

span 口径（与 DuEE-Fin 版一致）：字符级、左闭右开、text[start:end] == 声明文本。
CSED 自带 trigger_positions（实测字符级），直接采信并逐条校验，不自洽行丢弃计数。

复现命令：
  python data/scripts/build_cced.py
  python data/scripts/verify_processed.py   # （改造后）全量 span 校验
"""
import json
import random
from collections import Counter
from pathlib import Path

DATA = Path(__file__).resolve().parent.parent
SRC = DATA / "CCED-main" / "CSED.json"
PROC = DATA / "processed"
STATS = DATA / "stats"

SEED = 42
VAL_FRAC = 0.2

# 18 个原始变体 → 9 类（键=归并后类型，值=归并前原始字符串集合）
TYPE_MAP = {
    "数据泄露": {"数据泄露"},
    "网络钓鱼": {"网络钓鱼", "钓鱼攻击", "网络钓鱼攻击", "钓鱼邮件"},
    "勒索赎金": {"赎金", "勒索"},
    "DDoS攻击": {"DDos攻击", "DDoS攻击"},
    "恶意软件": {"恶意软件"},
    "供应链攻击": {"供应链攻击", "供应链"},
    "漏洞影响": {"漏洞影响", "安全漏洞"},
    "漏洞发现": {"漏洞发现", "漏洞"},
    "漏洞补丁": {"漏洞补丁"},
}
ALIAS = {raw: canon for canon, raws in TYPE_MAP.items() for raw in raws}


def dump_jsonl(path, rows):
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def iter_src(path):
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                yield None          # 脏行：外层计数


def main():
    n_total = n_bad = n_dropped_type = n_bad_span = n_multi = n_dup_sent = 0
    rows, seen_sent = [], set()

    for d in iter_src(SRC):
        n_total += 1
        if d is None:
            n_bad += 1
            continue
        raw_type = (d.get("eventype") or "").strip()
        canon = ALIAS.get(raw_type)
        if not canon:
            n_dropped_type += 1
            continue

        sent = (d.get("sentence") or "").strip()
        trig_text = (d.get("trigger") or "").strip()
        positions = d.get("trigger_positions") or []
        if not sent or not trig_text:
            n_bad_span += 1
            continue
        if len(positions) > 1:
            n_multi += 1           # 多触发词：取首个（已定口径）
        if positions:
            s, e = int(positions[0][0]), int(positions[0][1])
            if sent[s:e] != trig_text:
                # 自洽校验：CSED 自带 offset 偶有失准（如 trigger 与 sentence 尾标点差异），
                # 回退用 find 定位首次出现；再找不到才丢弃
                pos = sent.find(trig_text)
                if pos < 0:
                    n_bad_span += 1
                    continue
                s, e = pos, pos + len(trig_text)
        else:
            # CSED 实测有 47 行只有 trigger 文本、没有 trigger_positions：
            # 与 DuEE-Fin 版同思路用 find 回填（回填口径本就是项目惯例），实在找不到才丢
            pos = sent.find(trig_text)
            if pos < 0:
                n_bad_span += 1
                continue
            s, e = pos, pos + len(trig_text)

        if sent in seen_sent:
            n_dup_sent += 1
            continue               # 同句重复标注只保留第一条
        seen_sent.add(sent)

        # sent_id 用全局序号而非原 id：CSED 实测有 2 组重复 id（不同句子共用同一 id），
        # row_key = id|sent_id 的唯一性是评测正确对齐的前提（DuEE-Fin 坑 1 的同款教训）
        sent_id = f"s{len(rows)}"
        rows.append({"id": d["id"], "sent_id": sent_id, "text": sent, "canon": canon,
                     "trigger": {"text": sent[s:e], "start": s, "end": e}})

    # stratify 8:2 切分（每类内部独立切，比例按全量算）
    rng = random.Random(SEED)
    by_type = {}
    for r in rows:
        by_type.setdefault(r["canon"], []).append(r)
    train_rows, dev_rows = [], []
    for canon in TYPE_MAP:
        group = by_type.get(canon, [])
        rng.shuffle(group)
        n_val = round(len(group) * VAL_FRAC)
        dev_rows.extend(group[:n_val])
        train_rows.extend(group[n_val:])
    rng.shuffle(train_rows)
    rng.shuffle(dev_rows)

    def to_extract(rs):
        return [{"id": r["id"], "sent_id": r["sent_id"], "text": r["text"],
                 "event": {"type": r["canon"], "trigger": r["trigger"], "args": []}}
                for r in rs]

    train_extract = to_extract(train_rows)
    dev_extract = to_extract(dev_rows)
    # 单句「文档」：保 service 层切句→逐句→事件流的演示可跑（无真实篇章）
    dev_docs = [{"id": r["id"], "title": "", "text": r["text"],
                 "events": [{"type": r["canon"], "trigger": r["trigger"], "args": []}]}
                for r in dev_rows]
    schema = {canon: [] for canon in TYPE_MAP}

    PROC.mkdir(parents=True, exist_ok=True)
    dump_jsonl(PROC / "train_extract.jsonl", train_extract)
    dump_jsonl(PROC / "dev_extract.jsonl", dev_extract)
    dump_jsonl(PROC / "dev_docs.jsonl", dev_docs)
    (PROC / "event_schema.json").write_text(
        json.dumps(schema, ensure_ascii=False, indent=2), encoding="utf-8")

    stats = {
        "src_total": n_total, "src_bad_json": n_bad, "dropped_unknown_type": n_dropped_type,
        "dropped_bad_span": n_bad_span, "multi_trigger_take_first": n_multi,
        "dup_sentence_dropped": n_dup_sent,
        "kept": len(rows),
        "train": len(train_extract), "dev": len(dev_extract), "dev_docs": len(dev_docs),
        "train_types": dict(Counter(r["canon"] for r in train_rows)),
        "dev_types": dict(Counter(r["canon"] for r in dev_rows)),
        "seed": SEED, "val_frac": VAL_FRAC,
        "types": list(TYPE_MAP),
    }
    STATS.mkdir(parents=True, exist_ok=True)
    (STATS / "build_cced.json").write_text(
        json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"src={n_total} kept={len(rows)}（bad_json={n_bad} unknown_type={n_dropped_type} "
          f"bad_span={n_bad_span} multi_take_first={n_multi} dup_sent={n_dup_sent}）")
    print(f"train={len(train_extract)} dev={len(dev_extract)} dev_docs={len(dev_docs)}")
    print("train 各类:", stats["train_types"])
    print("dev   各类:", stats["dev_types"])


if __name__ == "__main__":
    main()
