# -*- coding: utf-8 -*-
"""RoBERTa-wwm 离线训练入口。

默认将 ``hfl/chinese-roberta-wwm-ext`` 解析为项目根目录下的
``pretrain_models/chinese-roberta-wwm-ext``，并强制 Hugging Face 离线模式。
用户原有命令可以继续使用模型 ID，无需改写命令。
"""
from pathlib import Path
import os
import sys

ROOT = Path(__file__).resolve().parents[1]   # code/
sys.path.insert(0, str(ROOT))
DEFAULT_MODEL_ID = "hfl/chinese-roberta-wwm-ext"

def _resolve_local_model(value: str | None) -> str:
    from common.model_paths import ROBERTA_MODEL_DIR, resolve_model_path

    raw = (value or "").strip()
    # 旧命令传入 Hub ID 时，优先解析本地预训练目录。
    if not raw or raw == DEFAULT_MODEL_ID or raw == ROBERTA_MODEL_DIR:
        return str(resolve_model_path(ROBERTA_MODEL_DIR))

    p = Path(raw).expanduser()
    if p.exists():
        return str(p.resolve())

    # 兼容从项目根目录或 code/ 目录执行时使用的相对路径。
    candidates = [Path.cwd() / p, ROOT / p, ROOT.parent / p]
    for item in candidates:
        if item.exists():
            return str(item.resolve())

    # 若传入的路径尾部是标准模型目录名，仍尝试通过项目目录解析。
    if p.name == ROBERTA_MODEL_DIR:
        return str(resolve_model_path(ROBERTA_MODEL_DIR))

    raise FileNotFoundError(
        f"RoBERTa 预训练模型目录不存在：{raw}\n"
        f"请将模型放到项目根目录/pretrain_models/{ROBERTA_MODEL_DIR}，"
        "或通过 --bert_path 指定本地目录。当前入口默认离线，不会联网下载。"
    )

def _extract_model_arg(argv: list[str]) -> tuple[str | None, list[str]]:
    """读取 --bert_path/--model_path 的值，兼容 --flag value 与 --flag=value。"""
    result = list(argv)
    for i, item in enumerate(result):
        if item in ("--bert_path", "--model_path") and i + 1 < len(result):
            return result[i + 1], result
        for flag in ("--bert_path=", "--model_path="):
            if item.startswith(flag):
                return item[len(flag):], result
    return None, result

def _replace_model_arg(argv: list[str], model_path: str) -> list[str]:
    result = list(argv)
    for i, item in enumerate(result):
        if item in ("--bert_path", "--model_path") and i + 1 < len(result):
            result[i + 1] = model_path
            return result
        if item.startswith("--bert_path="):
            result[i] = "--bert_path=" + model_path
            return result
        if item.startswith("--model_path="):
            result[i] = "--model_path=" + model_path
            return result
    return ["--bert_path", model_path] + result

def main():
    # 明确离线：不访问 Hugging Face Hub，也不尝试后台下载 tokenizer/权重。
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"

    argv = sys.argv[1:]
    requested, _ = _extract_model_arg(argv)
    local_model = _resolve_local_model(requested)
    argv = _replace_model_arg(argv, local_model)

    if "--encoder_name" not in argv and not any(x.startswith("--encoder_name=") for x in argv):
        argv = ["--encoder_name", "RoBERTa-WWM-Ext"] + argv
    if "--run_name" not in argv and not any(x.startswith("--run_name=") for x in argv):
        argv = ["--run_name", "roberta_crf_256"] + argv

    print(f"RoBERTa 本地权重：{local_model}")
    print("Hugging Face 离线模式：已开启（不会联网下载）")
    sys.argv = [sys.argv[0]] + argv
    from bert.train import main as shared_main
    shared_main()

if __name__ == "__main__":
    main()
