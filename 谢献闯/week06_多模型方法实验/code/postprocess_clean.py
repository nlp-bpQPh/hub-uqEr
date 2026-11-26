"""给已训练好的结果补「去重测试集」指标（不用重训）。

encoder 结果里存了逐条预测，直接按「test 文本是否出现在 train」过滤后重算指标。
"""

import json
from pathlib import Path

from cfg import DATASET_SPECS, OUT_DIR
from data_lib import load_data
from metrics_lib import classification_report

CACHE = {}


def get_data(name):
    if name not in CACHE:
        CACHE[name] = load_data(name)
    return CACHE[name]


def main():
    for p in sorted(Path(OUT_DIR).glob("*.json")):
        if p.name in ("dataset_stats.json", "smoke.json", "results.csv"):
            continue
        try:
            d = json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            continue
        if "preds" not in d or "dataset" not in d:
            continue
        if "report_clean" in d:
            continue
        data = get_data(d["dataset"])
        train_texts = {x["text"] for x in data["train"]}
        keep = [i for i, x in enumerate(d["preds"]) if x["text"] not in train_texts]
        rep = classification_report(
            [d["preds"][i]["true"] for i in keep],
            [d["preds"][i]["pred"] for i in keep],
            labels=list(range(DATASET_SPECS[d["dataset"]]["num_labels"])))
        rep["n_removed"] = len(d["preds"]) - len(keep)
        rep["removed_rate"] = round((len(d["preds"]) - len(keep)) / len(d["preds"]), 4)
        d["report_clean"] = rep
        p.write_text(json.dumps(d, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"{p.name}: clean n={rep['n']} acc={rep['acc']} f1={rep['macro_f1']}")


if __name__ == "__main__":
    main()
