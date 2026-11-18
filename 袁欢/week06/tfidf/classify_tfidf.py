"""
方法二：TF-IDF + 线性分类器（L1 基线）

教学重点：
  1. 中文短文本用字 1-2gram 抓「股/盘/楼」这类信号；英文 URL 的信号在
     **token 片段**（`.tk`、`//login`、`-verify-`），char 2-5gram 恰好覆盖，
     仍然不用分词/分符号，秒级训完。
  1b. 但它不是「强基线」的结论保持：URL 词法信号比新闻标题强得多，
     本数据集上线性模型上限远高于 TNEWS（重跑后见 outputs/summary.md）。
  2. 同一个脚本可通过 --ngram / --C 复现调参对比（URL 上建议从 char(2,5) 起步）。
  3. 它最怕「数据泄漏」：TNEWS 的 val 里混进 11% 训练样本能把 F1 抬 3 个点——
     本数据集转换时已源头去重，--leak 的泄漏演示不再生效（历史对照保留）。

用法：
  python tfidf/classify_tfidf.py                  # 干净集，默认 char(2,5) + C=1.0
  python tfidf/classify_tfidf.py --leak           # 在全量 val（含重复）上评测，演示泄漏虚高
  python tfidf/classify_tfidf.py --n 500 --C 0.5 --ngram 1,3
"""

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from common.data import clean_val, load_clean_val, load_splits
from common.metrics import classification_report, to_markdown_table


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=0, help="抽样条数，0=全部")
    ap.add_argument("--ngram", default="2,5", help="字符 n-gram 范围，如 2,5；URL 信号在 token 片段（.tk / -verify-），char 2-5gram 恰好覆盖")
    ap.add_argument("--C", type=float, default=1.0, help="LinearSVC 正则强度")
    ap.add_argument("--leak", action="store_true", help="在全量 val（不去重）上评测，对照泄漏影响")
    ap.add_argument("--out", default="outputs/tfidf_pred.json")
    args = ap.parse_args()

    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.svm import LinearSVC

    train, val = load_splits()
    if args.leak:
        # 故意不剔除重复：用来量化「数据泄漏」到底能虚高多少
        samples = val[: args.n] if args.n else val
        meta = {"mode": "leaky_full_val", "n_eval": len(samples)}
    else:
        samples, meta = load_clean_val(args.n or None)

    lo, hi = (int(x) for x in args.ngram.split(","))
    vec = TfidfVectorizer(analyzer="char", ngram_range=(lo, hi), max_features=300000)

    t0 = time.time()
    Xtr = vec.fit_transform([d["sentence"] for d in train])          # 稀疏矩阵，内存友好
    Xte = vec.transform([d["sentence"] for d in samples])
    clf = LinearSVC(C=args.C).fit(Xtr, [d["label"] for d in train])  # 线性 SVM：高维稀疏文本上的经典线性基线
    y_pred = clf.predict(Xte).tolist()
    fit_seconds = round(time.time() - t0, 1)

    y_true = [d["label"] for d in samples]
    report = classification_report(y_true, y_pred)
    report.update({"method": "tfidf_svm", "fit_seconds": fit_seconds,
                   "ngram": args.ngram, "C": args.C})

    out = Path(__file__).resolve().parent.parent / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"meta": meta, "report": report}, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"评测 {meta['n_eval']} 条 | 训练+推理 {fit_seconds}s | ngram={args.ngram} C={args.C}")
    print(to_markdown_table({f"TF-IDF+SVM (ngram={args.ngram}, C={args.C})": report}))
    print(f"→ 结果写入 {out}")


if __name__ == "__main__":
    main()
