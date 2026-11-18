"""独立复核交付的模型、数据、指标及输入处理；不重新训练。"""
import csv
import json
from pathlib import Path

import numpy as np
import torch

from classifier import evaluate, load_model, normalize


def main():
    torch.set_num_threads(2)
    root = Path(__file__).parent / "results"
    data = np.load(root / "dataset.npz")
    metrics = json.loads((root / "metrics.json").read_text())
    print("WEEK 02 | Independent artifact verification")
    combined = []
    for split in ["train", "val", "test"]:
        x, y = data[f"{split}_x"], data[f"{split}_y"]
        assert x.shape == (len(y), 5)
        assert np.array_equal(x.argmax(1), y)
        assert np.isfinite(x).all()
        combined.append(x)
    rows = np.concatenate(combined)
    assert np.unique(rows, axis=0).shape == rows.shape
    print("[PASS] Labels match the rule; no duplicated rows across the three splits")
    model = load_model(root / "best_model.pt")
    actual, pred = evaluate(model, normalize(data["test_x"]), torch.from_numpy(data["test_y"]))
    assert actual == metrics["test"]
    assert actual["accuracy"] > .98
    print(f"[PASS] Reloaded checkpoint exactly reproduces test metrics: {actual['accuracy']:.2%}")
    with (root / "test_errors.csv").open(newline="") as stream:
        errors = list(csv.DictReader(stream))
    assert len(errors) == actual["size"] - actual["correct"]
    print(f"[PASS] Error CSV includes all {len(errors)} incorrect test predictions")
    sample = data["test_x"][:128]
    np.testing.assert_allclose(normalize(sample), normalize(sample * 2 + 17), atol=1e-6)
    for index in range(5):
        vector = np.full((1, 5), -10.)
        vector[0, index] = 10
        with torch.inference_mode():
            assert int(model(normalize(vector)).argmax(1)) == index
    print("[PASS] Positive scaling / translation invariance; clear examples for all five classes")
    for invalid in [[], [[1, 2, 3, 4]], [[0, 0, float('nan'), 0, 0]], [[float('inf')] * 5]]:
        try:
            normalize(invalid)
        except ValueError:
            continue
        raise AssertionError(f"Accepted invalid input: {invalid}")
    assert torch.isfinite(normalize([[0] * 5, [1e308, -1e308, 0, 1, -1]])).all()
    print("[PASS] Invalid shape / NaN / infinity rejected; zero and large finite vectors remain finite")
    print("ALL ARTIFACT CHECKS PASSED")


if __name__ == "__main__":
    main()
