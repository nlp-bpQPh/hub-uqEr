# -*- coding: utf-8 -*-
"""
第五周作业配置：基于 Qwen3-0.6B-Base 的 SFT（LoRA）训练。

思路与课件第五部分一致：
  预训练模型只会「续写」，SFT 用（指令, 回答）数据 + ChatML 模板 + Loss Mask，
  把它变成「被问就答、答完就停」的助手；模型结构一个新层都不加。
"""
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
PROCESSED_DIR = DATA_DIR / "processed"
MODEL_DIR = ROOT / "models" / "Qwen3-1.7B-Base"
ADAPTER_DIR = ROOT / "models" / "qwen3-1.7b-sft-lora"
OUTPUT_DIR = ROOT / "outputs"

# 先用 0.6B 试过一版：能学会「作答」，但不会生成 <|im_end|> 收尾，回答后面容易重复。
# 换成 1.7B 基座后明显好转，所以最终结果用 1.7B。两者都是 base 模型，脚本通用：
#   python src/train_sft.py --model models/Qwen3-0.6B-Base --save_dir models/qwen3-0.6b-sft-lora

# ---------- ChatML 模板（Qwen 系列用的对话格式）----------
# <|im_start|>user\n问题<|im_end|>\n<|im_start|>assistant\n回答<|im_end|>
IM_START = "<|im_start|>"
IM_END = "<|im_end|>"


def build_prompt(instruction: str) -> str:
    """用户提问部分：这段在 loss 里会被 mask 掉。"""
    return f"{IM_START}user\n{instruction.strip()}{IM_END}\n{IM_START}assistant\n"


def build_target(output: str) -> str:
    """助手回答部分：loss 只落在这里，末尾的 <|im_end|> 教模型「说到哪算完」。"""
    return f"{output.strip()}{IM_END}"


# ---------- 训练超参 ----------
MAX_LEN = 512          # 数据都很短（平均 60 字符），512 足够
BATCH_SIZE = 8
GRAD_ACCUM = 1         # 3000 条数据偏少，等效 batch 小一点 → 每轮步数更多
LEARNING_RATE = 1e-4   # LoRA 常用 1e-4 ~ 2e-4；1.7B 上用 1e-4 更稳
EPOCHS = 3
WARMUP_RATIO = 0.03
WEIGHT_DECAY = 0.0
MAX_GRAD_NORM = 1.0
SEED = 42

# ---------- LoRA ----------
LORA_R = 16
LORA_ALPHA = 32
LORA_DROPOUT = 0.05
LORA_TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]

# ---------- 生成 ----------
MAX_NEW_TOKENS = 64


def ensure_dirs():
    for d in (PROCESSED_DIR, OUTPUT_DIR, ADAPTER_DIR.parent):
        d.mkdir(parents=True, exist_ok=True)
