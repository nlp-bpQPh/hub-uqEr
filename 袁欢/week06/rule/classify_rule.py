"""
方法一：规则 / 关键词（L0 基线）

教学重点：
  1. 这是**最便宜的**方法：不训练、不加载模型，10 行代码。
     先跑它，是为了给后面所有方法一个「下限」——连它都没有的东西不配谈提升。
  2. 「URL 含类别名」在英文 URL 上完全不成立，所以规则换成**URL 词法特征**：
     裸 IP 做 host、host 段含 @、可疑 TLD、钓鱼话术关键词。这些是恶意 URL 的
     经典硬信号，但只够分「恶意/良性」，分不了 phishing/malware/defacement——
     这不是代码写得差，是 L0 的天生上限（分流主力在模型）。
  3. 规则并非一无是处：生产里常用它做「快筛」，一条裸 IP 正则就能挡掉大量模型调用
     （见 service/classify_news.py 的 FAST_RULES）。

用法：
  python rule/classify_rule.py            # 全量干净验证集
  python rule/classify_rule.py --n 500    # 抽样 500 条
"""

import argparse
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from common.data import load_clean_val, load_splits
from common.labels import LABEL2ID, LABELS
from common.metrics import classification_report, pred_distribution, to_markdown_table

SUSPICIOUS_TLDS = {".tk", ".ml", ".ga", ".cf", ".gq", ".top", ".xyz",
                   ".buzz", ".click", ".country", ".review"}
SUSPICIOUS_WORDS = ("login", "verify", "account", "secure", "banking", "confirm",
                    "webscr", "signin", "update", "password", "recover")
IP_HOST = re.compile(r"^https?://[^/@]+")


def url_rule_hits(url: str) -> list:
    """返回命中的规则名列表（可多条），供快筛与错误分析"""
    low = url.lower()
    m = IP_HOST.match(low)
    host = m.group(0)[8:].split("/")[0] if m else ""        # 提取 host 段
    hits = []
    if host and re.fullmatch(r"[0-9.]+", host.split(":")[0]):
        hits.append("ip_host")                               # 裸 IP 做 host
    if "@" in host:
        hits.append("at_in_host")                           # http://baidu.com@evil.com
    if any(host.endswith(t) for t in SUSPICIOUS_TLDS):
        hits.append("suspicious_tld")                       # 廉价/滥用高发 TLD
    if any(w in low for w in SUSPICIOUS_WORDS):
        hits.append("keyword")                              # 路径中的钓鱼话术
    if len(url) > 75:
        hits.append("very_long")                            # 超长编码/子域堆叠
    return hits


def classify(text: str, fallback: int) -> int:
    """
    L0 规则：命中的词法特征全部指向 phishing 档（安全处置动作相同：拦）；
    一条不中 → majority 兜底（本数据集 majority = benign）。

    为什么全部指向 phishing 档而不是细分：
      规则能分「恶意/良性」，分不了「phishing/malware/defacement」——
      硬造细分规则就变成人工调参，反而掩盖 L0 的真实上限。
    """
    if url_rule_hits(text):
        return LABEL2ID["phishing"]
    return fallback


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=0, help="抽样条数，0=用全部干净样本")
    ap.add_argument("--out", default="outputs/rule_pred.json")
    args = ap.parse_args()

    samples, meta = load_clean_val(args.n or None)
    train, _ = load_splits()

    # 兜底类：训练集里样本最多的类别（majority baseline，所有方法的下限）
    from collections import Counter
    fallback = Counter(d["label"] for d in train).most_common(1)[0][0]

    y_true = [d["label"] for d in samples]
    y_pred = [classify(d["sentence"], fallback) for d in samples]

    report = classification_report(y_true, y_pred)
    report["method"] = "rule_url_lexical"
    report["pred_distribution"] = pred_distribution(y_pred)

    out = Path(__file__).resolve().parent.parent / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"meta": meta, "report": report}, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"样本 {meta['n_eval']} 条（干净集，重复已剔除 {meta['n_dup']} 条）")
    hit_n = sum(1 for d in samples if url_rule_hits(d["sentence"]))
    print(f"规则命中率 {hit_n / len(samples):.2%}（ip_host/at_in_host/TLD/关键词/超长任一命中）")
    print(to_markdown_table({"规则/URL词法": report}))
    print(f"→ 结果写入 {out}")


if __name__ == "__main__":
    main()
