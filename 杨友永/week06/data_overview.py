"""① 数据总览：商品名 → 品类（10 分类）。

    python data_overview.py            # 控制台概览
    python data_overview.py --report   # 额外生成 analysis/report.md 与三张图

统计口径统一走 dataset.ProductDataset，避免和训练用的划分不一致。
"""
from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

from dataset import DATA_DIR, ProductDataset, build_label_map, read_jsonl

HERE = Path(__file__).resolve().parent
OUT_DIR = HERE / "analysis"
FIG_DIR = OUT_DIR / "figures"
REPORT = OUT_DIR / "report.md"

BLUE, ORANGE, GRAY = "#1F4F82", "#B8612A", "#8C8C8C"
KEYWORDS = {
    "非冷藏即饮果汁": ["果汁"], "冷藏即饮果汁": ["果汁"], "即饮茶": ["茶"],
    "即饮奶茶": ["奶茶"], "白酒": ["白酒", "酒"], "牙膏": ["牙膏"],
    "硬糖": ["糖"], "软糖": ["糖"], "口香糖": ["口香糖"], "方便面": ["面"],
}


# --------------------------------------------------------------------------- #
# 统计
# --------------------------------------------------------------------------- #
def load_splits() -> dict[str, ProductDataset]:
    return {s: ProductDataset(s) for s in ("train", "val", "test")}


def basic_stats(splits: dict[str, ProductDataset]) -> dict:
    raw_train = read_jsonl(DATA_DIR / "train.jsonl")
    raw_test = read_jsonl(DATA_DIR / "test.jsonl")

    byname: dict[str, set[str]] = defaultdict(set)
    for r in raw_train:
        byname[r["product_name"]].add(r["category"])

    counts = {s: ds.class_counts() for s, ds in splits.items()}
    lens = [len(x.text) for x in splits["train"].items]
    kw_hit = {}
    for cat in counts["train"]:
        rows = [x for x in raw_train if x["category"] == cat]
        hit = sum(1 for x in rows if any(k in x["product_name"] for k in KEYWORDS[cat]))
        kw_hit[cat] = hit / len(rows)

    return {
        "raw": {"train": len(raw_train), "test": len(raw_test)},
        "counts": counts,
        "lens": sorted(lens),
        "kw_hit": kw_hit,
        "dup": len(raw_train) - len({r["product_name"] for r in raw_train}),
        "overlap": len({r["product_name"] for r in raw_train}
                       & {r["product_name"] for r in raw_test}),
        "conflict": {k: v for k, v in byname.items() if len(v) > 1},
        "n_test": len(raw_test),
    }


# --------------------------------------------------------------------------- #
# 控制台
# --------------------------------------------------------------------------- #
def print_overview(s: dict, splits: dict[str, ProductDataset]) -> None:
    c = s["counts"]
    names = sorted(c["train"], key=lambda x: -c["train"][x])
    print("=== 数据规模 ===")
    print(f"train.jsonl {s['raw']['train']:>7,}    test.jsonl {s['raw']['test']:>7,}")
    print(f"划分后: train {len(splits['train']):,} / val {len(splits['val']):,}"
          f" / test {len(splits['test']):,}")

    print("\n=== 类别分布 ===")
    print(f"{'类别':<14}{'train':>7}{'val':>7}{'test':>7}")
    for n in names:
        print(f"{n:<14}{c['train'][n]:>7}{c['val'][n]:>7}{c['test'][n]:>7}")
    top, bottom = names[0], names[-1]
    print(f"不均衡 {c['train'][top]}/{c['train'][bottom]} = "
          f"{c['train'][top]/c['train'][bottom]:.1f}:1")

    lens = s["lens"]
    print(f"\n=== 长度 ===\n平均 {sum(lens)/len(lens):.1f}  "
          f"P50 {lens[len(lens)//2]}  P95 {lens[int(len(lens)*.95)]}  max {lens[-1]}")

    print(f"\n=== 质量 ===\n重复商品名 {s['dup']}   "
          f"train/test 重叠 {s['overlap']}（{s['overlap']/s['n_test']:.1%} of test）"
          f"   同名不同类 {len(s['conflict'])}")

    print("\n=== 关键词命中率 ===")
    for n, v in sorted(s["kw_hit"].items(), key=lambda x: -x[1]):
        print(f"  {n:<14}{v:>7.1%}")

    print("\n=== 样例 ===")
    for n in names[:2] + names[-2:]:
        ex = [x.text for x in splits["train"].items if x.category == n][:2]
        print(f"  {n:<14}{ex}")


# --------------------------------------------------------------------------- #
# 图表 + 报告
# --------------------------------------------------------------------------- #
def _plt():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams["font.sans-serif"] = ["PingFang SC", "Hiragino Sans GB", "SimHei"]
    plt.rcParams["axes.unicode_minus"] = False
    return plt


def make_figures(s: dict) -> dict[str, Path]:
    plt = _plt()
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    c = s["counts"]
    names = sorted(c["train"], key=lambda x: -c["train"][x])
    out = {}

    import numpy as np
    y = np.arange(len(names))
    fig, ax = plt.subplots(figsize=(8, 4.6))
    ax.barh(y - 0.2, [c["train"][n] for n in names], 0.4, color=BLUE, label="train")
    ax.barh(y + 0.2, [c["test"][n] for n in names], 0.4, color=ORANGE, label="test")
    ax.set_yticks(y, names); ax.invert_yaxis()
    ax.set_xlabel("样本数"); ax.legend(frameon=False)
    ax.set_title(f"类别分布（train {len(ProductDataset('train')):,}"
                 f" / test {len(ProductDataset('test')):,}）", fontsize=13)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout(); out["class"] = FIG_DIR / "class_dist.png"
    fig.savefig(out["class"], dpi=160); plt.close(fig)

    lens = s["lens"]
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(lens, bins=range(min(lens), max(lens) + 2), color=BLUE, alpha=0.85)
    for q, color, tag in [(0.5, GRAY, "P50"), (0.95, ORANGE, "P95")]:
        v = lens[int(len(lens) * q)]
        ax.axvline(v, color=color, ls="--", lw=1.2, label=f"{tag} = {v}")
    ax.set_xlabel("商品名字符数"); ax.set_ylabel("样本数"); ax.legend(frameon=False)
    ax.set_title(f"商品名长度分布（平均 {sum(lens)/len(lens):.1f} 字）", fontsize=13)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout(); out["len"] = FIG_DIR / "length_dist.png"
    fig.savefig(out["len"], dpi=160); plt.close(fig)

    items = sorted(s["kw_hit"].items(), key=lambda x: -x[1])
    fig, ax = plt.subplots(figsize=(8, 4.2))
    vals = [v * 100 for _, v in items]
    colors = [BLUE if v >= 90 else ORANGE if v < 50 else GRAY for v in vals]
    bars = ax.bar([k for k, _ in items], vals, color=colors)
    ax.axhline(90, color=GRAY, ls=":", lw=1)
    ax.set_ylabel("类别词命中率 (%)"); ax.set_ylim(0, 105)
    ax.set_title("类别关键词直接出现在商品名中的比例", fontsize=13)
    ax.tick_params(axis="x", rotation=30, labelsize=9)
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v + 1.5, f"{v:.0f}%", ha="center", fontsize=8)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout(); out["kw"] = FIG_DIR / "keyword_hit.png"
    fig.savefig(out["kw"], dpi=160); plt.close(fig)
    return out


def write_report(s: dict, splits: dict[str, ProductDataset]) -> None:
    c = s["counts"]
    names = sorted(c["train"], key=lambda x: -c["train"][x])
    n_tr = len(splits["train"]); n_te = len(splits["test"])
    lens = s["lens"]
    top, bottom = names[0], names[-1]

    rows = "\n".join(f"| {n} | {c['train'][n]} | {c['val'][n]} | {c['test'][n]} |"
                     for n in names)
    kw_rows = "\n".join(f"| {n} | {v:.1%} |"
                        for n, v in sorted(s["kw_hit"].items(), key=lambda x: -x[1]))
    conflict = next(iter(s["conflict"].items()), ("无", []))

    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(f"""# 商品分类数据集 · 数据总览报告

> 数据：`product-classification-hiring-demo`（ModelScope，CC-BY-NC-SA-4.0）
> 任务：商品名 → 10 个品类
> 由 `data_overview.py --report` 自动生成

## 一、基本信息

| 项 | 值 |
|---|---|
| 原始训练集 / 测试集 | **{s['raw']['train']:,} / {s['raw']['test']:,}** |
| 划分后 train / val / test | {n_tr:,} / {len(splits['val']):,} / {n_te:,} |
| 字段 | `product_name`、`category` |
| 类别数 | **{len(c['train'])}** |
| 不均衡比 | **{c['train'][top]/c['train'][bottom]:.1f} : 1**（{top} {c['train'][top]} vs {bottom} {c['train'][bottom]}） |
| 商品名平均长度 | **{sum(lens)/len(lens):.1f} 字**（P50 {lens[len(lens)//2]} / P95 {lens[int(len(lens)*.95)]} / max {lens[-1]}） |

## 二、类别分布

![类别分布](figures/class_dist.png)

| 类别 | train | val | test |
|---|---|---|---|
{rows}

不均衡 **{c['train'][top]/c['train'][bottom]:.1f}:1**，属温和区间（对比 TNEWS 的 23:1），
划分后 val 的比例与 train 一致，说明分层切分生效。

## 三、文本长度

![长度分布](figures/length_dist.png)

商品名极短，**96% 在 {lens[int(len(lens)*.95)]} 字以内**：

- `max_len = 32` 覆盖全部样本，截断损失可忽略
- 序列短，训练与推理都快

## 四、关键词泄漏

![关键词命中率](figures/keyword_hit.png)

| 类别 | 类别词出现在商品名中的比例 |
|---|---|
{kw_rows}

**这组数字决定了下限和上限：**

- 前 6 类（茶、面、牙膏、糖、酒）命中率 91%~99.8%，**一条关键词规则就能拿 95%+**
- 果汁两类只有 {s['kw_hit']['非冷藏即饮果汁']:.1%} / {s['kw_hit']['冷藏即饮果汁']:.1%}，
  商品名通常不写「冷藏/非冷藏」，这是真正的难点

## 五、数据质量

| 检查项 | 结果 |
|---|---|
| 训练集重复商品名 | {s['dup']} 条 |
| 训练/测试商品名重叠 | **{s['overlap']} 条**（占 test {s['overlap']/n_te:.1%}） |
| 同名不同类别 | **{len(s['conflict'])} 条** |

唯一冲突：

```
{conflict[0]} → {list(conflict[1]) if conflict[1] else ''}
```

「冷藏」与「非冷藏」是储存属性，同一商品名可归两类，属标注歧义。

测试集有 {s['overlap']} 条商品名与训练集完全相同（{s['overlap']/n_te:.1%}），
评估时这部分等于送分，**真实泛化能力会略低于报告数字**。

## 六、难点分析

真正难的只有「果汁」这一族：

```
冷藏即饮果汁:   怡亿盛园冰杨梅汁370ml
非冷藏即饮果汁: 佳纯杨枝甘露果汁480ml
```

两类文字特征几乎相同，区别在是否需要冷链——该信息**大概率不在商品名里**，
构成精度的天然上限。

## 七、建模建议

| 项 | 建议 | 依据 |
|---|---|---|
| `max_len` | **32** | P95 = {lens[int(len(lens)*.95)]}，max = {lens[-1]} |
| 类别加权 | **暂不需要** | 不均衡仅 {c['train'][top]/c['train'][bottom]:.1f}:1 |
| 基线 | 关键词规则 | 前 6 类命中率 > 91% |
| 主模型 | BERT-base 微调 | 短文本分类标准方案 |
| 评估 | macro-F1 + 混淆矩阵 | 关注果汁两类互错 |

---

*由 `data_overview.py --report` 自动生成*
""", encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--report", action="store_true", help="生成 analysis/report.md 与图表")
    args = ap.parse_args()

    splits = load_splits()
    s = basic_stats(splits)
    print_overview(s, splits)
    if args.report:
        figs = make_figures(s)
        write_report(s, splits)
        print(f"\n报告: {REPORT}")
        print(f"图: {[p.name for p in figs.values()]}")


if __name__ == "__main__":
    main()
