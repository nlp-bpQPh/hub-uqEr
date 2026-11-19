"""第 6 周作业：更换预训练模型，对比 TNEWS 15 类标题分类方法。

在本文件所在的 Homework 目录运行：
    python compare_classifiers.py
    python compare_classifiers.py --methods rule tfidf
    python compare_classifiers.py --smoke --encoder-model /path/to/encoder --llm-model /path/to/llm

用课件的原模型做对照时，保持实验参数一致，只把 --encoder-model 和
--llm-model 分别换成 google-bert/bert-base-chinese 和 Qwen/Qwen3-0.6B。
完整实验依赖：torch>=2.6、transformers>=4.55、peft、scikit-learn。
"""

import argparse
import json
import random
import sys
from contextlib import ExitStack
from importlib import import_module
from math import gcd
from pathlib import Path
from tempfile import mkdtemp
from unittest.mock import patch


HOMEWORK = Path(__file__).resolve().parent
CODE = HOMEWORK.parent / "code"
sys.path[:0] = [str(CODE), str(CODE / "bert")]

from common.data import clean_val, load_clean_val, load_splits, sample_subset
from common.labels import LABELS
from common.metrics import classification_report, pred_distribution, to_markdown_table


SEED = 42
METHODS = {
    "rule": ("规则 / 关键词", "rule.classify_rule"),
    "tfidf": ("TF-IDF + SVM", "tfidf.classify_tfidf"),
    "bert": ("编码器微调", "predict_bert"),
    "zero_shot": ("本地 LLM Zero-shot", "llm_zero_shot.classify_zero_shot"),
    "lora": ("LoRA SFT", "llm_lora.eval_lora"),
    "full_ft": ("全量 SFT", "llm_lora.eval_lora"),
}


def model_source(value):
    path = Path(value).expanduser()
    return str(path.resolve()) if path.is_dir() else value


def run_example(module_name, arguments, **overrides):
    """直接运行示例 main()，在本次调用内替换模型路径和实验参数。"""
    module = import_module(module_name)
    with ExitStack() as stack:
        stack.enter_context(patch.object(sys, "argv", [module.__file__, *map(str, arguments)]))
        stack.enter_context(patch.multiple(module, **overrides))
        if "qwen3_path" in overrides:
            from packaging.version import Version
            from transformers import AutoModelForCausalLM, __version__

            if Version(__version__) < Version("4.56"):
                # 课件使用 dtype。Transformers 4.55 的同一个参数名是 torch_dtype。
                load_model = AutoModelForCausalLM.from_pretrained

                def load_legacy(*args, dtype=None, **kwargs):
                    return load_model(*args, torch_dtype=dtype, **kwargs)

                stack.enter_context(patch.object(AutoModelForCausalLM, "from_pretrained", load_legacy))
        module.main()


def report_all(y_true, y_pred):
    """固定 15 个类别计算 Macro-F1，生成式的解析失败按答错计入。"""
    from sklearn.metrics import f1_score

    # 用额外的类别编号保留失败样本，沿用示例对 Accuracy 和每类召回率的计算。
    scored_preds = [p if p >= 0 else len(LABELS) for p in y_pred]
    report = classification_report(y_true, scored_preds)
    report["macro_f1"] = round(float(f1_score(
        y_true, y_pred, labels=range(len(LABELS)), average="macro", zero_division=0,
    )), 4)
    report["unparsed"] = sum(p < 0 for p in y_pred)
    report["unparsed_rate"] = round(report["unparsed"] / len(y_pred), 4)
    report["pred_distribution"] = pred_distribution(y_pred)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--encoder-model", default="hfl/chinese-roberta-wwm-ext", help="编码器的 Hugging Face ID 或本地权重目录")
    parser.add_argument("--llm-model", default="Qwen/Qwen2.5-0.5B-Instruct", help="兼容 Qwen 聊天模板及 LoRA 层名的模型 ID 或本地目录")
    parser.add_argument("--methods", nargs="+", choices=METHODS, default=list(METHODS), help="默认运行全部六种方法")
    parser.add_argument("--n", type=int, help="固定 seed 抽取的干净评测样本数，默认 500")
    parser.add_argument("--pool", choices=("cls", "mean", "max"), default="cls", help="编码器池化方法")
    parser.add_argument("--num-train", type=int, default=5000, help="LoRA 和全量 SFT 共用的训练样本数")
    parser.add_argument("--smoke", action="store_true", help="微调各跑一步，评测默认 8 条，仅检查流程")
    parser.add_argument("--out-dir", type=Path, default=HOMEWORK / "outputs", help="每次实验在此目录下新建 run-* 子目录")
    args = parser.parse_args()
    n = args.n if args.n is not None else (8 if args.smoke else 500)
    if n < 1 or args.num_train < 1:
        parser.error("--n 和 --num-train 必须为正整数")

    import numpy as np

    encoder = model_source(args.encoder_model)
    llm = model_source(args.llm_model)
    models = {"bert": encoder, "zero_shot": llm, "lora": llm, "full_ft": llm}
    samples, meta = load_clean_val(n, seed=SEED)
    methods = list(dict.fromkeys(args.methods))
    training_methods = {"bert", "lora", "full_ft"}.intersection(methods)
    if training_methods:
        train, val = load_splits()
        clean, _ = clean_val(train, val)
        test_titles = {d["sentence"] for d in samples}
        dev = sample_subset([d for d in clean if d["sentence"] not in test_titles],
                            64 if args.smoke else 500, seed=SEED)
        if not dev:
            parser.error("请减小 --n，给微调留出独立的 checkpoint 验证样本")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    out_dir = Path(mkdtemp(prefix="run-", dir=args.out_dir.resolve()))
    print(f"实验目录 {out_dir}", flush=True)
    rows = [
        "| 方法 | 预训练模型 | Accuracy | Macro-F1 | 预测类别数 | 解析失败 |",
        "|---|---|---:|---:|---:|---:|",
    ]
    notes = [f"TNEWS 去重评测集 {len(samples)} 条，抽样 seed={SEED}。Accuracy 和 Macro-F1 均计入解析失败。"]
    if args.smoke:
        notes.append("冒烟模式，微调各更新一步，分数仅用于检查流程。")
    if training_methods:
        notes.append(f"微调验证集 {len(dev)} 条，与最终评测标题无重叠。")
    if {"lora", "full_ft"}.intersection(methods) and not args.smoke:
        notes.append(f"LoRA 与全量 SFT 使用相同的 {min(args.num_train, len(train))} 条训练样本。")

    for method in methods:
        label, module_name = METHODS[method]
        pretrained = models.get(method)
        print(f"\n→ {label} {pretrained or ''}", flush=True)
        random.seed(SEED)
        np.random.seed(SEED)

        # 各路评测读取同一份样本，继续使用示例的推理和预测文件格式。
        pred_file = out_dir / f"{method}_pred.json"
        eval_args = ["--n", n, "--out", pred_file]
        eval_config = {"load_clean_val": lambda *a, **kw: (samples, meta.copy()),
                       "classification_report": report_all}
        train_args = []

        if method == "bert":
            import torch

            torch.manual_seed(SEED)
            train_args = ["--pool", args.pool]
            if args.smoke:
                train_args += ["--epochs", 1, "--batch_size", 2, "--max_steps", 1]
            run_example("train_bert", train_args, bert_path=lambda: encoder,
                        OUT_DIR=out_dir, load_splits=lambda: (train, dev))
            eval_args += ["--ckpt", out_dir / "bert_ckpt.pt"]
            eval_config["bert_path"] = lambda: encoder
            label += f" ({args.pool})"
        elif method in {"lora", "full_ft"}:
            import torch

            torch.manual_seed(SEED)
            # 给聊天模板和答案留足长度。两种 SFT 共用数据、长度及有效 batch。
            train_args = ["--num_train", args.num_train, "--max_length", 256]
            if args.smoke:
                train_args += ["--epochs", 1, "--batch_size", 1, "--grad_accum", 1, "--max_steps", 1]
            else:
                # 示例在累积满时更新。选择能整除批次数的累积步数，避免剩余梯度跨 epoch。
                batches = (min(args.num_train, len(train)) + 1) // 2
                train_args += ["--batch_size", 2, "--grad_accum", gcd(batches, 4)]
            checkpoint = out_dir / "lora_adapter"
            if method == "full_ft":
                train_args += ["--full_ft", "--gradient_checkpointing"]
                checkpoint = out_dir / "full_ft_ckpt"
            run_example("llm_lora.train_lora", train_args, qwen3_path=lambda: llm,
                        OUT_DIR=out_dir, load_splits=lambda: (train, dev))
            eval_args += ["--adapter", checkpoint]
            eval_config["qwen3_path"] = lambda: llm
        elif method == "zero_shot":
            eval_config["qwen3_path"] = lambda: llm

        eval_config["to_markdown_table"] = lambda reports: to_markdown_table({label: next(iter(reports.values()))})
        run_example(module_name, eval_args, **eval_config)
        result = json.loads(pred_file.read_text(encoding="utf-8"))
        result["meta"].update(pretrained_model=pretrained, smoke=args.smoke, train_args=train_args)
        # 示例中的方法名带有写死的模型名称，结果中记录本次实际使用的模型。
        report = result["report"]
        report["method"] = method
        pred_file.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
        rows.append(f"| {label} | {pretrained or '—'} | {report['acc']:.4f} | {report['macro_f1']:.4f} | "
                    f"{report['pred_distribution']['used_classes']} / {len(LABELS)} | {report['unparsed']} |")
        (out_dir / "summary.md").write_text("\n".join(rows + [""] + notes) + "\n", encoding="utf-8")

    print("\n" + "\n".join(rows))
    print(f"\n对比结果写入 {out_dir / 'summary.md'}")


if __name__ == "__main__":
    main()
