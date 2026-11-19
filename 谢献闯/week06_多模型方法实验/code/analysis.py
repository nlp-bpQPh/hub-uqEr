"""报告用的细粒度分析：混淆对、类别增益、方法一致性、典型错误样例。"""

import json
from collections import Counter
from pathlib import Path

from cfg import DATASET_SPECS, OUT_DIR


def load(fname):
    p = Path(OUT_DIR) / fname
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None


def top_confusions(d, labels, k=6):
    cm = d["report"]["confusion_matrix"]
    pairs = []
    for i, row in enumerate(cm):
        for j, v in enumerate(row):
            if i != j and v > 0:
                pairs.append((v, labels[i], labels[j]))
    pairs.sort(reverse=True)
    return [{"n": v, "true": t, "pred": p} for v, t, p in pairs[:k]]


def examples(d, labels, k=5):
    """挑置信度最高的错误（这里用出现次数最多的混淆对举例）。"""
    preds = d.get("preds", [])
    return preds[:k]


def main():
    out = {}
    for ds, spec in DATASET_SPECS.items():
        models = {}
        for key in ("bert-base", "roberta-wwm-ext", "macbert-base"):
            d = load(f"enc_{key}_{ds}.json")
            if d:
                models[key] = d
        if not models:
            continue
        best_key = max(models, key=lambda k: models[k]["report"]["macro_f1"])
        best = models[best_key]
        labels = spec["labels"]

        info = {
            "best_model": best["model_display"],
            "best_key": best_key,
            "best_acc": best["report"]["acc"],
            "best_f1": best["report"]["macro_f1"],
            "top_confusions": top_confusions(best, labels),
            "per_class": {labels[i]: best["report"]["per_class_recall"][str(i)]
                          for i in range(len(labels))},
        }
        # 换模型带来的每类召回变化
        if "bert-base" in models and best_key != "bert-base":
            delta = {}
            for i, name in enumerate(labels):
                a = models["bert-base"]["report"]["per_class_recall"][str(i)]
                b = best["report"]["per_class_recall"][str(i)]
                if a is not None and b is not None:
                    delta[name] = round(b - a, 4)
            info["recall_delta_vs_bert"] = delta
        out[ds] = info

    (OUT_DIR / "analysis.json").write_text(
        json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(out, ensure_ascii=False, indent=2)[:2000])


if __name__ == "__main__":
    main()
