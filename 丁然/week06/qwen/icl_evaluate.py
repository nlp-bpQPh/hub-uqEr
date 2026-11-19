"""Qwen in-context learning 的 zero-shot 和 few-shot 评估。"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from pathlib import Path

WEEK_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(WEEK_DIR))

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import seaborn as sns  # noqa: E402
import torch  # noqa: E402
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score  # noqa: E402
from transformers import AutoModelForCausalLM  # noqa: E402

from labels import LABELS  # noqa: E402
from qwen_config import (  # noqa: E402
    DEFAULT_MODEL_PATH,
    DEFAULT_TRAIN_PATH,
    DEFAULT_VAL_PATH,
    EVAL_LIMIT,
    SHOTS_PER_CLASS,
    get_device,
)
from qwen_data import balanced_subset, few_shot_subset, load_rows  # noqa: E402
from qwen_model import load_tokenizer  # noqa: E402


LABEL_TEXT = "、".join(LABELS)
LOG_EVERY = 10


def build_messages(
    sentence: str,
    demonstrations: list[dict],
) -> list[dict[str, str]]:
    """构造符合 Qwen 对话格式的 zero-shot/few-shot 消息。"""

    messages = [
        {
            "role": "system",
            "content": (
                "你是中文新闻分类器。可选类别只有："
                f"{LABEL_TEXT}。必须输出一个类别名称。"
            ),
        }
    ]
    for row in demonstrations:
        messages.append(
            {
                "role": "user",
                "content": f"新闻：{row['sentence']}",
            }
        )
        messages.append(
            {
                "role": "assistant",
                "content": LABELS[row["label"]],
            }
        )
    messages.append(
        {
            "role": "user",
            "content": f"新闻：{sentence}",
        }
    )
    return messages


def save_results(
    output_dir: Path,
    mode: str,
    rows: list[dict],
    predictions: list[int],
    confidences: list[float],
    demo_count: int,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    labels = [row["label"] for row in rows]
    invalid_count = sum(prediction < 0 for prediction in predictions)
    metrics = {
        "accuracy": accuracy_score(labels, predictions),
        "macro_f1": f1_score(
            labels,
            predictions,
            labels=list(range(15)),
            average="macro",
            zero_division=0,
        ),
        "weighted_f1": f1_score(
            labels,
            predictions,
            labels=list(range(15)),
            average="weighted",
            zero_division=0,
        ),
        "invalid_predictions": invalid_count,
        "invalid_rate": invalid_count / max(len(predictions), 1),
        "samples": len(predictions),
        "demonstrations": demo_count,
    }
    (output_dir / "metrics.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (output_dir / "meta.json").write_text(
        json.dumps(
            {
                "experiment": mode,
                "shots_per_class": demo_count // 15 if demo_count else 0,
                "scoring": "constrained_next_token",
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    with (output_dir / "predictions.csv").open(
        "w",
        encoding="utf-8",
        newline="",
    ) as file:
        writer = csv.writer(file)
        writer.writerow(
            [
                "idx",
                "sentence",
                "true_label",
                "prediction",
                "predicted_label",
                "confidence",
            ]
        )
        for row, prediction, confidence in zip(rows, predictions, confidences):
            writer.writerow(
                [
                    row.get("idx"),
                    row["sentence"],
                    row["label"],
                    prediction,
                    LABELS[prediction],
                    confidence,
                ]
            )

    confusion_predictions = [
        prediction if prediction >= 0 else 15 for prediction in predictions
    ]
    matrix = confusion_matrix(
        labels,
        confusion_predictions,
        labels=list(range(16)),
    )
    np.savetxt(output_dir / "confusion_matrix.csv", matrix, delimiter=",", fmt="%d")

    plt.rcParams["font.sans-serif"] = ["PingFang SC", "Arial Unicode MS"]
    plt.rcParams["axes.unicode_minus"] = False
    plt.figure(figsize=(11, 9))
    sns.heatmap(
        matrix,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=[*LABELS, "无法解析"],
        yticklabels=LABELS,
    )
    plt.xlabel("预测类别")
    plt.ylabel("真实类别")
    plt.title(f"Qwen ICL 混淆矩阵：{mode}")
    plt.tight_layout()
    plt.savefig(output_dir / "confusion_matrix.png", dpi=180)
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Qwen zero-shot/few-shot ICL")
    parser.add_argument("--mode", choices=["zero_shot", "few_shot"], required=True)
    parser.add_argument("--model-path", default=DEFAULT_MODEL_PATH)
    parser.add_argument("--train-path", default=str(DEFAULT_TRAIN_PATH))
    parser.add_argument("--val-path", default=str(DEFAULT_VAL_PATH))
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--shots-per-class", type=int, default=SHOTS_PER_CLASS)
    parser.add_argument("--eval-limit", type=int, default=EVAL_LIMIT)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()

    os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/week06-qwen-mpl")
    device = get_device(args.device)
    tokenizer = load_tokenizer(args.model_path)
    tokenizer.padding_side = "left"
    model_dtype = torch.float16 if device.type == "mps" else torch.float32
    model = AutoModelForCausalLM.from_pretrained(
        str(Path(args.model_path).expanduser().resolve()),
        local_files_only=True,
        dtype=model_dtype,
    ).to(device)
    model.eval()
    label_token_ids = [
        tokenizer.encode(label, add_special_tokens=False)[0]
        for label in LABELS
    ]
    if len(set(label_token_ids)) != len(label_token_ids):
        raise ValueError("类别名称首 token 不唯一，无法使用约束打分")

    val_rows = load_rows(args.val_path)
    if args.smoke:
        val_rows = balanced_subset(val_rows, 30, args.seed + 1)
    elif args.eval_limit > 0:
        val_rows = balanced_subset(val_rows, args.eval_limit, args.seed + 1)

    demonstrations = []
    if args.mode == "few_shot":
        train_rows = load_rows(args.train_path)
        shots = 1 if args.smoke else args.shots_per_class
        demonstrations = few_shot_subset(train_rows, shots, args.seed)

    if not tokenizer.chat_template:
        raise ValueError("当前 Qwen tokenizer 没有 chat_template，无法生成标准 ICL 输入")
    prompts = [
        tokenizer.apply_chat_template(
            build_messages(row["sentence"], demonstrations),
            tokenize=False,
            add_generation_prompt=True,
        )
        for row in val_rows
    ]
    predictions: list[int] = []
    confidences: list[float] = []
    print(
        f"ICL模式={args.mode} 验证={len(val_rows)} "
        f"示例数={len(demonstrations)} 设备={device}",
        flush=True,
    )

    # 只比较 15 个类别名称首 token，不让模型自由生成解释文本。
    for index, prompt in enumerate(prompts, start=1):
        encoded = tokenizer(
            prompt,
            truncation=True,
            max_length=2048,
            return_tensors="pt",
        )
        encoded = {name: value.to(device) for name, value in encoded.items()}
        with torch.inference_mode():
            next_logits = model(**encoded).logits[0, -1]
        candidate_logits = next_logits[
            torch.tensor(label_token_ids, device=device)
        ]
        probabilities = candidate_logits.softmax(dim=-1)
        confidence, prediction = probabilities.max(dim=-1)
        predictions.append(int(prediction.item()))
        confidences.append(float(confidence.item()))

        if index == 1 or index % LOG_EVERY == 0 or index == len(prompts):
            print(f"[ICL] {index}/{len(prompts)}", flush=True)

    output_dir = Path(args.output_dir)
    save_results(
        output_dir,
        args.mode,
        val_rows,
        predictions,
        confidences,
        len(demonstrations),
    )
    metrics = json.loads((output_dir / "metrics.json").read_text(encoding="utf-8"))
    print(
        f"结果 accuracy={metrics['accuracy']:.4f} "
        f"macro_f1={metrics['macro_f1']:.4f} "
        f"无法解析={metrics['invalid_predictions']}",
        flush=True,
    )


if __name__ == "__main__":
    main()
