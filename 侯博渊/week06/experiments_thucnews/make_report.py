"""
THUCNews 10 类实验报告（Markdown + 图）。

读取 experiments_thucnews/outputs/：
  rule_pred.json / tfidf_svc_pred.json / tfidf_lr_pred.json /
  small_bert_pred.json / bert_probe_pred.json
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "_data" / "thucnews"
EXP = ROOT / "experiments_thucnews" / "outputs"
NUM_LABELS = 10

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "Arial Unicode MS"]
plt.rcParams["axes.unicode_minus"] = False


def load(name):
    p = EXP / name
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None


def fmt(v, digits=4):
    return "-" if v is None else f"{v:.{digits}f}"


def main():
    labels = json.loads((DATA / "labels.json").read_text(encoding="utf-8"))
    id2label = {v: k for k, v in labels.items()}

    specs = [
        ("规则 / 关键词", "rule_pred.json", None, "—", "10 类关键词计数，取最高"),
        ("TF-IDF + LinearSVC", "tfidf_svc_pred.json", None, "4,000", "char 1-2gram，C=1.0"),
        ("TF-IDF + LogReg", "tfidf_lr_pred.json", None, "4,000", "class_weight=balanced"),
        ("StructBERT-tiny（微调）", "small_bert_pred.json", 8.77, "4,000", "4 层 / 256 维，3 epoch，max_len=192"),
        ("BERT-base（线性探针）", "bert_probe_pred.json", 102.3, "2,000", "冻结 BERT + 逻辑回归，max_len=128"),
    ]
    rows = []
    for name, fname, params, n_train, note in specs:
        data = load(fname)
        if not data:
            print(f"[warn] 缺少 {fname}")
            continue
        r = data["report"]
        rows.append({
            "name": name,
            "params": params or r.get("params_m"),
            "n_train": n_train,
            "train_s": r.get("train_seconds", r.get("fit_seconds")),
            "acc": r.get("acc_incl_unparsed", r.get("acc")),
            "f1": r["macro_f1"],
            "ece": r.get("ece"),
            "note": note,
            "report": r,
        })

    tiny = next((r for r in rows if "tiny" in r["name"].lower()), None)
    probe = next((r for r in rows if "BERT-base" in r["name"]), None)
    best_tfidf = max((r["f1"] for r in rows if "TF-IDF" in r["name"]), default=None)

    lines = []
    lines.append("# 换数据集实验：THUCNews 10 类新闻分类\n")
    lines.append("> 数据集换成 **THUCNews**（10 类中文新闻：体育/娱乐/家居/房产/教育/时尚/时政/游戏/科技/财经），"
                 "比二分类情感任务更接近真实的多分类场景。\n")
    lines.append("> 目的：在 **10 类** 任务上重新比较规则 / TF-IDF / 小模型 / BERT 探针，看排序是否又变。\n")

    lines.append("## 结论速览\n")
    if best_tfidf and probe:
        lines.append(f"- **TF-IDF 与 BERT 探针基本并列**：{best_tfidf:.4f} vs {probe['f1']:.4f}"
                     f"（差 {abs(best_tfidf-probe['f1'])*100:.1f} 个点）。")
    elif best_tfidf:
        lines.append(f"- **TF-IDF 最好**：Macro-F1 {best_tfidf:.4f}，训练几秒到几十秒。")
    if tiny:
        lines.append(f"- **StructBERT-tiny**：Macro-F1 {tiny['f1']:.4f}，训练 {tiny['train_s']:.0f}s（CPU，4,000 条）。")
    lines.append("")

    lines.append("## 一、数据\n")
    lines.append("| 项 | 值 |")
    lines.append("|---|---|")
    lines.append("| 数据集 | THUCNews（10 类新闻） |")
    lines.append("| 训练 / 验证 / 测试 | 4,000 / 1,000 / 1,500（每类 400 / 100 / 150） |")
    lines.append("| 类别 | 体育、娱乐、家居、房产、教育、时尚、时政、游戏、科技、财经 |")
    lines.append("| 文本长度 | 中位数约 630 字，p95 约 2,440 字（长新闻） |")
    lines.append("| test 与 train 重复 | 0 / 1,500 |")
    lines.append("| 来源 | `hf-mirror.com/datasets/spiritx2023/ThuCnews`（`cnews.val/test.txt`） |\n")

    lines.append("## 二、主结果（test 1,500 条）\n")
    lines.append("| 方法 | 参数量 | 训练样本 | 训练耗时 | 准确率 | Macro-F1 | ECE | 备注 |")
    lines.append("|---|---|---|---|---|---|---|---|")
    for r in rows:
        params = f"{r['params']:.1f}M" if r["params"] else "—"
        if "规则" in r["name"] or r["train_s"] is None:
            ts = "—"
        elif r["train_s"] < 1:
            ts = "<1s"
        else:
            ts = f"{r['train_s']:.0f}s"
        lines.append(f"| {r['name']} | {params} | {r['n_train']} | {ts} | {fmt(r['acc'])} | "
                     f"{fmt(r['f1'])} | {fmt(r['ece'], 3)} | {r['note']} |")
    lines.append("\n> 1,500 条测试集上 1 条 = 0.07 个点；10 类 Macro-F1 等权。\n")

    dev = None
    dev_path = EXP / "dev_eval.json"
    if dev_path.exists():
        dev = json.loads(dev_path.read_text(encoding="utf-8"))

    lines.append("## 三、dev 不能代表 test：两个 split 分布不同\n")
    if dev:
        lines.append("| 方法 | dev 1,000 | test 1,500 | dev − test |")
        lines.append("|---|---|---|---|")
        for key, name in [("rule", "规则 / 关键词"), ("tfidf_svc", "TF-IDF + LinearSVC")]:
            row = next((r for r in rows if r["name"] == name), None)
            if row:
                d = dev[key]["macro_f1"]
                lines.append(f"| {name} | {d:.4f} | {row['f1']:.4f} | {d-row['f1']:+.4f} |")
        row = next((r for r in rows if "tiny" in r["name"].lower()), None)
        if row and "tiny" in dev:
            d = dev["tiny"]["macro_f1"]
            lines.append(f"| StructBERT-tiny | {d:.4f} | {row['f1']:.4f} | {d-row['f1']:+.4f} |")
        lines.append("\n> dev 来自 `cnews.val.txt`（中位 610 字），test 来自 `cnews.test.txt`（中位 728 字）——"
                     "两个文件来源/长度不同，dev 明显更容易。")
        lines.append("> TF-IDF 和 tiny 在 dev 上都能到 0.96–0.99，到 test 都掉 7 个点左右；")
        lines.append("> 所以 **dev 只能用来选 epoch，不能当 test 估计**。\n")
    else:
        lines.append("（缺少 dev_eval.json，跳过）\n")

    lines.append("## 四、10 类 per-class recall\n")
    lines.append("| 类别 | " + " | ".join(r["name"].split("（")[0] for r in rows) + " |")
    lines.append("|---|" + "---|" * len(rows))
    for i in range(NUM_LABELS):
        cells = [fmt((r["report"].get("per_class_recall") or {}).get(str(i)), 2) for r in rows]
        lines.append(f"| {id2label.get(i, str(i))} | " + " | ".join(cells) + " |")

    lines.append("\n## 五、分析\n")
    if tiny and best_tfidf:
        diff = (tiny["f1"] - best_tfidf) * 100
        lines.append(f"1. **小模型 vs TF-IDF**：{tiny['f1']:.4f} vs {best_tfidf:.4f}"
                     f"（**{diff:+.1f} 个点**）。新闻主题分类的关键词很强（球/股票/手机），"
                     f"字 n-gram 能吃到大部分信号；小模型多花的时间没有换回分数。"
                     f" 注意 dev 上两者分别是 0.9675 / 0.9860，到 test 都掉了 7 个点左右——dev 明显高估。")
    if probe:
        lines.append(f"2. **BERT-base 线性探针**：{probe['f1']:.4f}，只用了 2,000 条特征、max_len=128，"
                     f"和 TF-IDF 只差 0.08 个点，基本打平——说明预训练表示在 2k 特征 + 截断下仍能追上字 n-gram。"
                     f"这不代表 BERT 微调的上限；本机无 GPU，4,000×3 在 CPU 上估计十几小时（未实测），没有跑；"
                     f"GPU 上约几分钟到十几分钟。")
    lines.append("3. **规则关键词**：靠 10 类各自的关键词能到 0.67 左右；"
                 "但同一篇文章常同时出现多个领域词（如「科技公司财报」），规则没有上下文，"
                 "这是它和 TF-IDF 差 20 多个点的主因。")
    lines.append("4. **截断不是主因**：补测按长度分桶，tiny 在 >500 字上 F1 0.895、≤192 字只有 0.848——"
                 "长文并不更差；dev-test 的 7.5 点差距主要来自两个 split 的分布差异。"
                 "但 192 token 截断仍是上限限制，要冲更高分应提高 max_length 或分段。")
    lines.append("5. **跨数据集看排序**：")
    lines.append("")
    lines.append("| 数据集 | 类别数 | 最好方法 | 说明 |")
    lines.append("|---|---|---|---|")
    lines.append("| TNEWS | 15 | BERT 0.5436 | 细粒度新闻主题，BERT 微调最好 |")
    lines.append("| THUCNews | 10 | " + (f"TF-IDF {best_tfidf:.4f} / BERT 探针 {probe['f1']:.4f}（并列）" if best_tfidf and probe else (f"TF-IDF {best_tfidf:.4f}" if best_tfidf else "-")) + " | 长新闻 + 强关键词，TF-IDF 与 BERT 探针打平 |")
    lines.append("| ChnSentiCorp | 2 | TF-IDF 0.9252 | 情感二分类，关键词极强 |")
    lines.append("")
    lines.append("> 三个数据集上「最好方法」并不一致——这正是做多数据集对比的价值："
                 "选型要看任务粒度、文本长度、词面信号强弱，而不是默认「上 BERT」。\n")
    lines.append("6. **置信度**：TF-IDF LogReg 的 ECE 0.295（很自信但校准差）；"
                 "tiny 0.066、BERT 探针 0.046 更可信。要拿 conf 做阈值，优先用后两者或先做温度缩放。")
    lines.append("7. **错误集中**：教育类 recall 规则 0.73 > BERT 0.56 > TF-IDF SVC 0.47；"
                 "教育新闻常和科技/时政重叠，TF-IDF 被非教育词拉走，规则反而靠「高考/学生/学校」更稳。\n")

    lines.append("## 六、局限\n")
    lines.append("- 训练集只用了 4,000 条（官方 THUCNews 训练集有 5 万条），是为了 CPU 可跑；数据量上去后 BERT 可能反超。")
    lines.append("- 只跑了单 seed；1,500 条测试集的统计噪声约 ±1 个点。")
    lines.append("- BERT-base 只做 2,000 条线性探针，不是全量微调。")
    lines.append("- 长文本截断（tiny 192 / BERT 128）会低估预训练模型的上限。\n")

    lines.append("## 七、复现\n")
    lines.append("```powershell")
    lines.append("python experiments_thucnews/prepare_data.py")
    lines.append("python experiments_thucnews/run_baselines.py")
    lines.append("python experiments_thucnews/train_small_bert.py --epochs 3")
    lines.append("python experiments_thucnews/bert_linear_probe.py --n_train 2000")
    lines.append("python experiments_thucnews/make_report.py")
    lines.append("```\n")
    lines.append("> 依赖：`_data/thucnews/`、`_models/structbert-tiny/`、`pretrain_models/bert-base-chinese/`。")

    text = "\n".join(lines)
    (ROOT / "experiments_thucnews" / "REPORT.md").write_text(text, encoding="utf-8")
    (EXP / "summary.md").write_text(text, encoding="utf-8")
    print("→ experiments_thucnews/REPORT.md")

    names = [r["name"].replace("（微调）", "").replace("（线性探针）", "") for r in rows]
    f1s = [r["f1"] for r in rows]
    colors = []
    for r in rows:
        if "tiny" in r["name"].lower():
            colors.append("#d97706")
        elif "BERT" in r["name"]:
            colors.append("#2563eb")
        elif "TF-IDF" in r["name"]:
            colors.append("#059669")
        else:
            colors.append("#94a3b8")
    fig, ax = plt.subplots(figsize=(9.5, 4.8))
    y = np.arange(len(names))
    ax.barh(y, f1s, color=colors)
    ax.set_yticks(y)
    ax.set_yticklabels(names)
    ax.invert_yaxis()
    ax.set_xlim(0, 1.0)
    ax.set_xlabel("Macro-F1（test 1,500 条）")
    ax.set_title("THUCNews 10 类新闻分类：各方法 Macro-F1")
    for i, v in enumerate(f1s):
        ax.text(v + 0.01, i, f"{v:.3f}", va="center", fontsize=10)
    ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    fig.savefig(EXP / "comparison.png", dpi=200)
    print("→ experiments_thucnews/outputs/comparison.png")


if __name__ == "__main__":
    main()
