"""独立运行 CNews 分类对比实验，默认只读取本目录的数据与源码。"""

import argparse
import csv
import json
from collections import Counter
from pathlib import Path

from bert.train_bert import train_predict as train_bert
from common.data import load_or_prepare_data
from common.metrics import classification_report
from rule.classify_rule import majority_predict, predict as rule_predict
from tfidf.classify_tfidf import predict as tfidf_predict

ROOT = Path(__file__).resolve().parent
RESULT_DIR = ROOT / "results"
SEED = 42


def write_report(result):
    methods = result["methods"]
    lines = [
        "# CNews 新闻分类方法对比实验报告", "",
        "## 1. 实验说明", "",
        "本实验参照课程 `code` 目录结构组织：`common` 存放数据与指标，`rule` 存放规则基线，`tfidf` 存放 TF-IDF + SVM，`bert` 存放 BERT 微调。运行不导入外部 `code` 目录。", "",
        f"- 数据来源：{result['evaluation_note']}",
        f"- 随机种子：{result['seed']}；切分数量：训练 {result['split_sizes']['train']} / 验证 {result['split_sizes']['valid']} / 测试 {result['split_sizes']['test']}。",
        "- 指标：Accuracy、Macro-F1、Weighted-F1、分类别 Precision/Recall/F1 和混淆矩阵。", "",
        "## 2. 数据清洗", "",
        "按首个 Tab 分离标签和正文，规范空白，剔除空行、坏行、空标签/空正文，并按（标签，正文）去除完全重复；不删除标点、不做繁简转换。", "",
        f"- 清洗后样本数：{result['cleaning_audit'].get('kept_records', sum(result['class_counts_cleaned'].values()))}；完全重复剔除：{result['cleaning_audit'].get('duplicate_records', 0)}；坏行：{result['cleaning_audit'].get('malformed_lines', 0)}；空行：{result['cleaning_audit'].get('blank_lines', 0)}；空标签/正文：{result['cleaning_audit'].get('empty_label_or_text', 0)}。",
        "- 类别分布：" + "；".join(f"{label} {count}" for label, count in result["class_counts_cleaned"].items()), "",
        "## 3. 对比结果", "",
        "| 方法 | Accuracy | Macro-F1 | Weighted-F1 | 测试数 | 用时（秒） |", "|---|---:|---:|---:|---:|---:|",
    ]
    for name, report in methods.items():
        lines.append(f"| {name} | {report['accuracy']:.4f} | {report['macro_f1']:.4f} | {report['weighted_f1']:.4f} | {report['n']} | {report['seconds']:.2f} |")
    lines += ["", "## 4. 方法与复现说明", "",
              "- 多数类基线：预测训练集中最多的类别。",
              "- 规则关键词：文本出现类别名时输出该类，否则回退至训练集多数类。",
              "- TF-IDF + LinearSVC：字符级 n-gram，比较 char(1,2) 与 char(1,3)，C=1.0。",
              "- BERT：CLS 池化 + dropout + 动态类别线性头；需要本地预训练权重及兼容的 torch/transformers 环境。", "",
              "在本目录执行 `python run_cnews.py` 复现实验；指定原始数据并重做清洗切分：`python run_cnews.py --source <原始数据路径>`。启用 BERT：`python run_cnews.py --bert --model-dir <本地BERT目录>`。", "",
              "## 5. 每类别指标", ""]
    for name, report in methods.items():
        lines += [f"### {name}", "", "| 类别 | Precision | Recall | F1 | Support |", "|---|---:|---:|---:|---:|"]
        for label, metric in report["per_class"].items():
            lines.append(f"| {label} | {metric['precision']:.4f} | {metric['recall']:.4f} | {metric['f1']:.4f} | {metric['support']} |")
        lines.append("")
    lines += ["## 6. 局限", "",
              "1. 原目录只提供 `cnews.test.txt`，本实验对该文件内部切分，不可视作官方测试集成绩。",
              "2. 去重只处理完全重复文本，未处理近似转载样本。",
              "3. BERT 权重文件没有复制到本目录，启用时需显式提供本机模型目录；BERT 依赖版本冲突时，不影响默认基线和 TF-IDF 实验。", "",
              "## 7. 输出文件", "",
              "- `run_cnews.py`：实验调度与报告生成。",
              "- `common/`、`rule/`、`tfidf/`、`bert/`：数据、指标与各分类方法源码。",
              "- `data/`：清洗语料和固定切分；`results/`：指标 JSON、逐样本预测和本报告。", ""]
    (RESULT_DIR / "对比实验报告.md").write_text("\n".join(lines), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description="运行 week6 内的 CNews 分类对比实验")
    parser.add_argument("--source", type=Path, default=None, help="可选：原始 label<TAB>text 数据并重新清洗切分")
    parser.add_argument("--bert", action="store_true", help="可选：微调本地 BERT")
    parser.add_argument("--model-dir", type=Path, default=None, help="本地 BERT 权重目录；启用 --bert 时必填")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-length", type=int, default=128)
    args = parser.parse_args()

    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    train, valid, test, cleaned, audit, evaluation_note = load_or_prepare_data(args.source)
    labels = sorted({row["label"] for row in cleaned})
    y_true = [row["label"] for row in test]
    methods, predictions_by_method = {}, {}

    baselines = [
        ("多数类基线", majority_predict(train, len(test))),
        ("规则关键词（类别名命中）", rule_predict(train, [row["text"] for row in test], labels)),
    ]
    for name, predictions in baselines:
        methods[name] = {**classification_report(y_true, predictions, labels), "seconds": 0.0}
        predictions_by_method[name] = predictions

    for ngram in ((1, 2), (1, 3)):
        predictions, elapsed = tfidf_predict(train, test, ngram_range=ngram, c=1.0)
        name = f"字符 TF-IDF + LinearSVC (char {ngram[0]}-{ngram[1]}, C=1.0)"
        methods[name] = {**classification_report(y_true, predictions, labels), "seconds": elapsed}
        predictions_by_method[name] = predictions

    if args.bert:
        if args.model_dir is None or not args.model_dir.is_dir():
            raise FileNotFoundError("启用 BERT 时请用 --model-dir 指定有效的本地预训练模型目录。")
        predictions, elapsed, checkpoint = train_bert(
            train, test, labels, args.model_dir, args.epochs,
            args.batch_size, args.max_length, RESULT_DIR,
        )
        name = f"BERT 微调（{args.model_dir.name}）"
        methods[name] = {**classification_report(y_true, predictions, labels),
                         "seconds": elapsed, "checkpoint": checkpoint,
                         "epochs": args.epochs, "max_length": args.max_length}
        predictions_by_method[name] = predictions

    with (RESULT_DIR / "predictions.tsv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f, delimiter="\t", lineterminator="\n")
        writer.writerow(["true_label", "text", *predictions_by_method.keys()])
        for i, row in enumerate(test):
            writer.writerow([row["label"], row["text"], *[predictions_by_method[name][i] for name in predictions_by_method]])

    result = {
        "source": "week6/data", "evaluation_note": evaluation_note,
        "seed": SEED,
        "split_sizes": {"train": len(train), "valid": len(valid), "test": len(test)},
        "labels": labels, "cleaning_audit": audit,
        "class_counts_cleaned": dict(Counter(row["label"] for row in cleaned)),
        "methods": methods,
    }
    (RESULT_DIR / "metrics.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    write_report(result)
    print(json.dumps({
        "split_sizes": result["split_sizes"], "cleaning_audit": audit,
        "methods": {name: {"accuracy": value["accuracy"], "macro_f1": value["macro_f1"], "seconds": value["seconds"]}
                    for name, value in methods.items()},
    }, ensure_ascii=False, indent=2))
    print(f"报告：{RESULT_DIR / '对比实验报告.md'}")


if __name__ == "__main__":
    main()
