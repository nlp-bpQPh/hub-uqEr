import json
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np


def plot_steps_jsonl(jsonl_path: Path):
    """
    读取 steps.jsonl，画图，png 和 jsonl 放在同一个文件夹
    """
    p = Path(jsonl_path)
    raw = p.read_text(encoding="utf-8").strip()
    lines = raw.splitlines()
    if not lines:
        print("[auto plot] 警告：steps.jsonl 为空，没有日志，跳过绘图")
        return

    records = [json.loads(line) for line in lines]

    # 按日志顺序生成连续全局采样点（修复原先 epoch*10000 断层bug）
    global_steps = list(range(len(records)))
    loss = [r["loss"] for r in records]
    lr = [r["lr"] if r["lr"] is not None else np.nan for r in records]

    # 中文支持
    plt.rcParams["font.sans-serif"] = ["SimHei"]
    plt.rcParams["axes.unicode_minus"] = False

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
    ax1.plot(global_steps, loss, color="#e74c3c", label="train loss")
    ax1.set_ylabel("Loss")
    ax1.set_title("Train Loss & LR")
    ax1.grid(alpha=0.3)
    ax1.legend()
    ax1.ticklabel_format(style="plain", axis="x")

    ax2.plot(global_steps, lr, color="#2980b9", label="lr")
    ax2.set_xlabel("sample point")
    ax2.set_ylabel("Learning Rate")
    ax2.grid(alpha=0.3)
    ax2.legend()
    ax2.ticklabel_format(style="plain", axis="x")

    plt.tight_layout()
    save_png = p.with_suffix(".png")
    plt.savefig(save_png, dpi=300)
    plt.close()
    print(f"[auto plot] 曲线已保存：{save_png}")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("jsonl_path", help="训练步日志文件路径，train_steps.jsonl")
    args = ap.parse_args()
    plot_steps_jsonl(Path(args.jsonl_path))
