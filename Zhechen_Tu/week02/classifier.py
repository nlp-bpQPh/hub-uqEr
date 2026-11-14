"""五维向量五分类：训练、评估、保存模型，并预测用户输入。"""
import argparse
import copy
import csv
import json
import platform
import random
from pathlib import Path

import numpy as np
import torch
from torch import nn

SEED = 20261022


def normalize(vectors):
    """逐行平移与正比例缩放，保留维度顺序；先缩放以避免大数溢出。"""
    values = np.asarray(vectors, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] != 5 or len(values) == 0:
        raise ValueError("输入必须是非空的 N×5 数组")
    if not np.isfinite(values).all():
        raise ValueError("输入必须全部为有限数字，不能包含 NaN 或无穷大")
    magnitude = np.max(np.abs(values), axis=1, keepdims=True)
    scaled = values / np.where(magnitude > 0, magnitude, 1)
    width = np.ptp(scaled, axis=1, keepdims=True)
    result = (scaled - scaled.mean(axis=1, keepdims=True)) / np.where(width > 0, width, 1)
    return torch.from_numpy(result.astype(np.float32))


def make_model():
    # 返回未经 softmax 的 logits，交叉熵内部完成相应计算。
    return nn.Sequential(nn.Linear(5, 64), nn.ReLU(), nn.Linear(64, 32),
                         nn.ReLU(), nn.Linear(32, 5))


def make_dataset(size, seed, distribution="uniform"):
    rng = np.random.default_rng(seed)
    if distribution == "uniform":
        vectors = rng.uniform(-100, 100, (size, 5))
    elif distribution == "normal":
        vectors = rng.normal(0, 20, (size, 5))
    elif distribution == "negative":
        vectors = -rng.exponential(30, (size, 5))
    elif distribution == "near_tie":
        vectors = rng.uniform(-1, 1, (size, 5))
        order = np.argsort(vectors, axis=1)
        rows = np.arange(size)
        vectors[rows, order[:, -2]] = vectors[rows, order[:, -1]] - rng.uniform(1e-6, 1e-3, size)
    else:
        raise ValueError(distribution)
    # argmax 只生成监督标签；内部类别 0..4，对外显示为第 1..5 类。
    labels = np.argmax(vectors, axis=1).astype(np.int64)
    return vectors, normalize(vectors), torch.from_numpy(labels)


@torch.inference_mode()
def evaluate(model, x, y):
    model.eval()
    logits = model(x)
    pred = logits.argmax(dim=1)
    matrix = np.zeros((5, 5), dtype=np.int64)
    np.add.at(matrix, (y.numpy(), pred.numpy()), 1)
    diagonal = matrix.diagonal().astype(float)
    precision = diagonal / np.maximum(matrix.sum(axis=0), 1)
    recall = diagonal / np.maximum(matrix.sum(axis=1), 1)
    f1 = 2 * precision * recall / np.maximum(precision + recall, 1e-12)
    return {
        "size": len(y), "correct": int((pred == y).sum()),
        "accuracy": float((pred == y).float().mean()),
        "loss": float(nn.functional.cross_entropy(logits, y)),
        "macro_f1": float(f1.mean()), "class_precision": precision.tolist(),
        "class_recall": recall.tolist(), "class_f1": f1.tolist(),
        "confusion_matrix": matrix.tolist(),
    }, pred.numpy()


def save_plots(history, metrics, target):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    epochs = [r["epoch"] for r in history]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for key, label in [("train_loss", "Train"), ("val_loss", "Validation")]:
        axes[0].plot(epochs, [r[key] for r in history], label=label)
    for key, label in [("train_accuracy", "Train"), ("val_accuracy", "Validation")]:
        axes[1].plot(epochs, np.array([r[key] for r in history]) * 100, label=label)
    axes[0].set(title="Cross-entropy loss", xlabel="Epoch", ylabel="Loss")
    axes[1].set(title="Classification accuracy", xlabel="Epoch", ylabel="Accuracy (%)")
    for ax in axes:
        ax.grid(alpha=.2)
        ax.legend()
    fig.tight_layout()
    fig.savefig(target / "training_curves.png", dpi=160)
    plt.close(fig)
    matrix = np.array(metrics["test"]["confusion_matrix"])
    fig, ax = plt.subplots(figsize=(6, 5))
    picture = ax.imshow(matrix, cmap="Blues")
    for i in range(5):
        for j in range(5):
            ax.text(j, i, str(matrix[i, j]), ha="center", va="center",
                    color="white" if matrix[i, j] > matrix.max() / 2 else "black")
    ax.set(xticks=range(5), yticks=range(5), xticklabels=range(1, 6),
           yticklabels=range(1, 6), xlabel="Predicted class", ylabel="True class",
           title=f"Independent test set: {metrics['test']['accuracy']:.2%} accuracy")
    fig.colorbar(picture, ax=ax)
    fig.tight_layout()
    fig.savefig(target / "confusion_matrix.png", dpi=160)
    plt.close(fig)


def train(args):
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    target = args.output_dir
    target.mkdir(parents=True, exist_ok=True)
    raw_train, train_x, train_y = make_dataset(60000, SEED)
    raw_val, val_x, val_y = make_dataset(10000, SEED + 1)
    # 最终测试数据在选好模型之后才生成和评估。
    model = make_model()
    initial, _ = evaluate(model, val_x, val_y)
    print("WEEK 02 | Zhechen Tu | Five-dimensional vector classification", flush=True)
    print(f"Python {platform.python_version()} | PyTorch {torch.__version__} | CPU | seed={SEED}", flush=True)
    print("Class = first maximum position (displayed as 1..5)", flush=True)
    print("MLP: 5 -> 64 -> 32 -> 5 | ReLU | CrossEntropyLoss | Adam", flush=True)
    print("Train / validation / test: 60000 / 10000 / 10000 (independent seeds)", flush=True)
    print(f"Initial validation accuracy: {initial['accuracy']:.2%}", flush=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=.002)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=.00005)
    criterion = nn.CrossEntropyLoss()
    best_loss, best_epoch, best_weights = float("inf"), 0, None
    history = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        order = torch.randperm(len(train_y))
        loss_sum, correct = 0., 0
        for indices in order.split(256):
            optimizer.zero_grad()
            logits = model(train_x[indices])
            loss = criterion(logits, train_y[indices])
            loss.backward()
            optimizer.step()
            loss_sum += loss.item() * len(indices)
            correct += int((logits.argmax(1) == train_y[indices]).sum())
        val, _ = evaluate(model, val_x, val_y)
        row = {"epoch": epoch, "train_loss": loss_sum / len(train_y),
               "train_accuracy": correct / len(train_y), "val_loss": val["loss"],
               "val_accuracy": val["accuracy"]}
        history.append(row)
        if val["loss"] < best_loss:
            best_loss, best_epoch = val["loss"], epoch
            best_weights = copy.deepcopy(model.state_dict())
        if epoch == 1 or epoch % 10 == 0:
            print(f"Epoch {epoch:02d} | train loss={row['train_loss']:.5f} acc={row['train_accuracy']:.2%}"
                  f" | val loss={row['val_loss']:.5f} acc={row['val_accuracy']:.2%}", flush=True)
        scheduler.step()
    model.load_state_dict(best_weights)
    checkpoint = {"state_dict": best_weights, "seed": SEED, "best_epoch": best_epoch,
                  "architecture": "5-64-32-5", "normalization": "row_center_range_v1"}
    torch.save(checkpoint, target / "best_model.pt")
    # 从保存文件重新加载后再最终评估，验证交付模型可独立使用。
    model = load_model(target / "best_model.pt")
    raw_test, test_x, test_y = make_dataset(10000, SEED + 2)
    metrics = {"seed": SEED, "best_epoch": best_epoch, "epochs": args.epochs,
               "initial_validation": initial,
               "training_counts": torch.bincount(train_y, minlength=5).tolist(),
               "validation": evaluate(model, val_x, val_y)[0]}
    metrics["test"], predicted = evaluate(model, test_x, test_y)
    print(f"Best epoch selected by validation loss: {best_epoch}", flush=True)
    for name, seed in [("normal", SEED + 3), ("negative", SEED + 4), ("near_tie", SEED + 5)]:
        _, x, y = make_dataset(5000, seed, name)
        metrics[name] = evaluate(model, x, y)[0]
    for name in ["test", "normal", "negative", "near_tie"]:
        result = metrics[name]
        print(f"{name:10s}: {result['correct']}/{result['size']} | accuracy={result['accuracy']:.2%}"
              f" | macro F1={result['macro_f1']:.4f}", flush=True)
    examples = np.array([[9, 1, 2, 3, 4], [1, 9, 2, 3, 4], [1, 2, 9, 3, 4],
                         [1, 2, 3, 9, 4], [1, 2, 3, 4, 9], [-9, -7, -1, -5, -3],
                         [1e100, 2e100, 5e100, -3e100, 0], [3, 3, 1, 0, -1], [0, 0, 0, 0, 0]], dtype=float)
    _, example_pred = evaluate(model, normalize(examples), torch.from_numpy(examples.argmax(1)))
    print("Examples: vector -> model class / expected class", flush=True)
    for v, pred in zip(examples, example_pred):
        print(f"{v.tolist()} -> {pred + 1} / {v.argmax() + 1}", flush=True)
    metrics["examples"] = [{"vector": v.tolist(), "predicted": int(p + 1),
                             "expected": int(v.argmax() + 1)} for v, p in zip(examples, example_pred)]
    np.savez_compressed(target / "dataset.npz", train_x=raw_train, train_y=train_y.numpy(),
                        val_x=raw_val, val_y=val_y.numpy(), test_x=raw_test, test_y=test_y.numpy())
    (target / "metrics.json").write_text(json.dumps(metrics, indent=2, ensure_ascii=False) + "\n")
    with (target / "training_history.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(history[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(history)
    with (target / "test_errors.csv").open("w", newline="") as stream:
        writer = csv.writer(stream, lineterminator="\n")
        writer.writerow(["test_index", "x1", "x2", "x3", "x4", "x5", "true_class", "model_class"])
        for index in np.flatnonzero(predicted != test_y.numpy()):
            writer.writerow([int(index), *raw_test[index], int(test_y[index]) + 1, int(predicted[index]) + 1])
    save_plots(history, metrics, target)
    print("Saved: best_model.pt, dataset.npz, metrics.json, CSV logs and two plots", flush=True)
    print("NOTE: Near ties and exact ties can be misclassified; this is a learned approximation.", flush=True)
    assert metrics["test"]["accuracy"] > .98, "Final test accuracy below the preset 98% acceptance threshold"
    print("TRAINING AND EVALUATION COMPLETED", flush=True)


def load_model(checkpoint_path):
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    model = make_model()
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    return model


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predict", type=float, nargs=5, metavar="X", help="输入五个有限数字，使用保存模型预测")
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).parent / "results")
    args = parser.parse_args()
    if args.epochs < 1:
        parser.error("epochs 必须为正整数")
    if args.predict is not None:
        try:
            features = normalize([args.predict])
        except ValueError as error:
            parser.error(str(error))
        model = load_model(args.output_dir / "best_model.pt")
        with torch.inference_mode():
            probabilities = model(features).softmax(dim=1)[0]
        print(f"Vector: {args.predict}")
        print(f"Model predicted class (1..5): {int(probabilities.argmax()) + 1}")
        print("Probabilities:", [round(float(p), 6) for p in probabilities])
        print(f"Rule reference class (first maximum): {int(np.argmax(args.predict)) + 1}")
        print("The model prediction is approximate; exact ties follow the first-maximum label convention.")
    else:
        train(args)


if __name__ == "__main__":
    main()
