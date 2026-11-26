"""
预训练模型路径解析（同时兼容「母版」与「交付副本」两套目录树）

为什么需要这个模块：
  同一个项目会存在于两套目录树里，且层级深度不同：
    母版：    项目材料准备/event_extraction/<方法目录>/xxx.py
    交付副本：八斗课程/week7 信息抽取/event_extraction/<方法目录>/xxx.py
  脚本所在目录到 pretrain_models/ 的层级深度不同（一个差 1 级、一个差 2 级），
  所以任何写死的 "../pretrain_models" 或 "../../pretrain_models"
  都必然在其中一套里失效。
  正确做法：从脚本所在目录开始**向上逐级查找** pretrain_models/<模型名>。

  目录里必须有真实权重文件（*.safetensors / pytorch_model.bin）才算命中，
  避免拿到半下载的空目录——这是这类「本地模型 + 离线课堂」项目最常见的坑。

使用方式：
  from model_paths import resolve_model_path, BERT_MODEL_DIR
  BERT_PATH = resolve_model_path(BERT_MODEL_DIR)   # → .../pretrain_models/bert-base-chinese
"""

import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")   # Windows: torch 与 numpy 各带 libiomp5md，不设会 OMP Error #15

from pathlib import Path

# ── 模型目录名（与 pretrain_models/ 下的实际目录一致）──────────────────────────
BERT_MODEL_DIR = "bert-base-chinese"   # 本项目判别式链路全部使用 BERT

PRETRAIN_DIR_NAME = "pretrain_models"

QWEN_MODEL_DIR = "Qwen3-0.6B"

def _is_model_dir(path: Path) -> bool:
    """目录里必须有真实权重文件才算模型（避免拿到半下载的空目录）"""
    if not path.is_dir():  #判断这个路径是不是一个文件夹
        return False
    if any(path.glob("*.safetensors")): #path.glob("*.safetensors")：在这个文件夹下面，查找所有后缀是 .safetensors 的文件;any(...)：只要找到任意一个匹配的文件，结果就是 True
        return True
    return (path / "pytorch_model.bin").exists() #存在 → 返回 True；不存在 → 返回 False


def candidate_paths(dir_name: str, extra_roots: list[Path] | None = None) -> list[Path]:
    """列出所有候选路径（供 resolve_model_path 使用，也便于报错时展示）"""
    here = Path(__file__).resolve()  #当前脚本文件的完整绝对路径
    candidates: list[Path] = []  #: list[Path]-(冒号：类型注解的分隔符，list[Path]：代表：这是一个列表，列表里面每一个元素都必须是 Path 对象)

    # 1. 调用方显式指定的根目录（可给模型目录本身，也可给它的父目录）
    for root in (extra_roots or []):
        root = Path(root) #Path 是一个类（class）：把传入的路径字符串，包装成 Path 对象，赋值给变量 root
        candidates.append(root if root.name == dir_name else root / dir_name)

    # 2. 从本文件所在目录向上逐级查找 pretrain_models/<dir_name>
    #    母版：命中 项目材料准备/pretrain_models/
    #    副本：命中 八斗课程/pretrain_models/
    for parent in here.parents:
        candidates.append(parent / PRETRAIN_DIR_NAME / dir_name)
    #arents 是 Path 对象自带的属性，是一个迭代器，依次取出当前文件所有上层父目录，由近到远：
    # 第 1 个 parent：D:\AI\week7\common
    # 第 2 个 parent：D:\AI\week7
    # 第 3 个 parent：D:\AI
    # 第 4 个 parent：D:\
    # 一直到根目录

    return candidates


def resolve_model_path(dir_name: str = BERT_MODEL_DIR,
                       extra_roots: list[Path] | None = None) -> Path:
    """
    定位本地预训练模型目录，返回绝对路径。

    优先级：
      1. extra_roots 里显式给出的目录
      2. 逐级向上找到的第一个含真实权重的 pretrain_models/<dir_name>

    找不到时抛 FileNotFoundError，并打印全部已尝试路径（不静默回退到联网下载，
    保证课堂离线可用）。
    """
    candidates = candidate_paths(dir_name, extra_roots)
    for path in candidates:
        if _is_model_dir(path):
            return path

    tried = "\n  ".join(str(c) for c in candidates)
    raise FileNotFoundError(
        f"未找到本地模型目录：{dir_name}\n"
        f"已尝试的路径：\n  {tried}\n\n"
        f"请确认 pretrain_models/{dir_name}/ 下存在模型权重；"
        f"本课程约定直接复用已有权重，不重新下载。"
    )


if __name__ == "__main__":
    # 自检：直接打印解析结果，便于课堂排查「模型找不到」问题
    print(f"{BERT_MODEL_DIR:<18} -> {resolve_model_path(BERT_MODEL_DIR)}")
