"""
基于 Decoder-only Transformer 字符级语言模型的拼音输入法。

先运行 language_model.py 训练模型，再运行：
    python pinyin_ime.py
    python pinyin_ime.py --model_path best_model_transformer.pt --topk 8 --beam 10

输入拼音后，程序使用拼音映射表生成候选汉字，并通过 Transformer
计算上下文条件概率，使用束搜索输出候选文本。
"""

import argparse
import collections
import glob
import json
import math
import os

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------
# 拼音映射表
# ---------------------------------------------------------------------

def _load_pinyin_map(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"找不到拼音映射表文件：{path}")
    with open(path, "r", encoding="utf-8") as f:
        mapping = json.load(f)
    if not isinstance(mapping, dict):
        raise ValueError("拼音映射表格式错误，顶层应为 JSON 对象。")
    return mapping


PINYIN_MAP = {}
_SYLLABLES = []


def segment(pinyin_str):
    """
    将拼音切分为音节。
    支持空格分隔（ni hao）和连续输入（nihao）。
    使用最长匹配优先策略。
    """
    syllables = []
    for token in pinyin_str.strip().lower().split():
        i = 0
        while i < len(token):
            matched = next(
                (s for s in _SYLLABLES if token.startswith(s, i)),
                None,
            )
            if matched:
                syllables.append(matched)
                i += len(matched)
            else:
                i += 1
    return syllables


# ---------------------------------------------------------------------
# 与 language_model.py 完全一致的 Transformer 定义
# ---------------------------------------------------------------------

class PositionalEmbedding(nn.Module):
    def __init__(self, max_len, d_model):
        super().__init__()
        self.position_embedding = nn.Embedding(max_len, d_model)

    def forward(self, x):
        seq_len = x.size(1)
        positions = torch.arange(seq_len, device=x.device).unsqueeze(0)
        return x + self.position_embedding(positions)


class TransformerLM(nn.Module):
    def __init__(
        self,
        vocab_size,
        d_model=256,
        nhead=8,
        num_layers=4,
        dim_feedforward=1024,
        dropout=0.1,
        max_len=256,
    ):
        super().__init__()
        if d_model % nhead != 0:
            raise ValueError("d_model 必须能被 nhead 整除。")

        self.vocab_size = vocab_size
        self.d_model = d_model
        self.max_len = max_len

        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.position_embedding = PositionalEmbedding(max_len, d_model)
        self.embedding_dropout = nn.Dropout(dropout)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers,
            norm=nn.LayerNorm(d_model),
        )
        self.final_dropout = nn.Dropout(dropout)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        self.lm_head.weight = self.token_embedding.weight

    def forward(self, input_ids):
        if input_ids.dim() != 2:
            raise ValueError("input_ids 必须是二维张量 (batch, seq_len)。")

        seq_len = input_ids.size(1)
        if seq_len > self.max_len:
            raise ValueError(
                f"输入长度 {seq_len} 超过模型最大长度 {self.max_len}。"
            )

        x = self.token_embedding(input_ids) * math.sqrt(self.d_model)
        x = self.position_embedding(x)
        x = self.embedding_dropout(x)

        causal_mask = torch.triu(
            torch.ones(seq_len, seq_len, device=input_ids.device, dtype=torch.bool),
            diagonal=1,
        )
        hidden = self.transformer(x, mask=causal_mask)
        hidden = self.final_dropout(hidden)
        return self.lm_head(hidden)


LM = TransformerLM


# ---------------------------------------------------------------------
# 字频先验：仅在没有任何历史上下文时使用
# ---------------------------------------------------------------------

UNI_LOGPROB = None


def build_unigram_logprob(pattern, char2idx, device):
    """统计训练语料字频，并进行拉普拉斯平滑。"""
    counter = collections.Counter()
    paths = glob.glob(pattern)
    if not paths:
        raise FileNotFoundError(
            f"没有找到用于统计字频的语料：{pattern}。"
        )

    for path in paths:
        if os.path.isfile(path):
            with open(path, "r", encoding="utf-8", errors="ignore") as f:
                for line in f:
                    counter.update(c for c in line if c in char2idx)

    probs = torch.ones(len(char2idx), dtype=torch.float)
    for char, count in counter.items():
        probs[char2idx[char]] += count
    probs /= probs.sum()
    return torch.log(probs).to(device)


# ---------------------------------------------------------------------
# Transformer 下一字符概率
# ---------------------------------------------------------------------

def next_char_log_probs(context, model, char2idx, device):
    """
    返回给定上下文后的下一字符 log probability。
    上下文超过 max_len 时保留最近的 max_len 个字符。
    """
    ids = [char2idx[c] for c in context if c in char2idx]
    if not ids:
        return UNI_LOGPROB

    ids = ids[-model.max_len:]
    x = torch.tensor([ids], dtype=torch.long, device=device)
    with torch.no_grad():
        logits = model(x)
        log_probs = F.log_softmax(logits[0, -1, :], dim=-1)
    return log_probs


# ---------------------------------------------------------------------
# 束搜索
# ---------------------------------------------------------------------

def beam_search(
    syllables,
    prefix,
    model,
    char2idx,
    idx2char,
    beam_size,
    device,
):
    """
    逐音节展开候选汉字。
    每一步使用 Transformer 的 P(next_char | context) 计算增量得分。
    返回 [(累计 log probability, 转换结果), ...]。
    """
    beams = [(0.0, "")]

    for syllable in syllables:
        candidates = list(dict.fromkeys(
            char for char in PINYIN_MAP.get(syllable, [])
            if char in char2idx
        ))
        if not candidates:
            continue

        new_beams = []
        for score, partial in beams:
            context = prefix + partial
            log_probs = next_char_log_probs(
                context, model, char2idx, device
            )

            for char in candidates:
                char_score = log_probs[char2idx[char]].item()
                new_beams.append((score + char_score, partial + char))

        new_beams.sort(key=lambda item: item[0], reverse=True)
        beams = new_beams[:beam_size]

        if not beams:
            break

    return beams


# ---------------------------------------------------------------------
# 交互式输入法
# ---------------------------------------------------------------------

def run(model, char2idx, idx2char, topk, beam_size, device):
    print("=" * 58)
    print("  拼音输入法（字符级 Transformer 语言模型）")
    print("  输入拼音回车，选择候选编号追加到已输入文字")
    print("  r = 重置上下文，q = 退出")
    print("=" * 58)

    confirmed = ""

    while True:
        print(f"\n已输入：「{confirmed}」" if confirmed else "\n已输入：（空）")
        raw = input("拼音> ").strip()

        if not raw:
            continue
        if raw.lower() == "q":
            print("退出。")
            break
        if raw.lower() == "r":
            confirmed = ""
            continue

        syllables = segment(raw)
        if not syllables:
            print("无法识别任何音节，请检查拼音拼写或映射表。")
            continue

        print(f"音节：{' '.join(syllables)}")
        results = beam_search(
            syllables, confirmed, model, char2idx, idx2char,
            beam_size, device,
        )

        if not results:
            print("没有找到词表内的候选汉字。")
            continue

        print("候选：")
        for i, (score, candidate_text) in enumerate(results[:topk]):
            print(f"  [{i}] {candidate_text}  ({score:.3f})")

        choice = input("选择编号（回车跳过）：").strip()
        if choice.isdigit():
            index = int(choice)
            if 0 <= index < min(topk, len(results)):
                confirmed += results[index][1]
            else:
                print("编号超出范围。")


# ---------------------------------------------------------------------
# 加载模型
# ---------------------------------------------------------------------

def load_transformer_checkpoint(model_path, device):
    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"找不到模型文件：{model_path}。请先运行 language_model.py 训练模型。"
        )

    # checkpoint 是本项目训练脚本生成的可信本地文件
    try:
        checkpoint = torch.load(
            model_path, map_location=device, weights_only=False
        )
    except TypeError:
        checkpoint = torch.load(model_path, map_location=device)

    char2idx = checkpoint["char2idx"]
    idx2char = checkpoint.get(
        "idx2char", {i: c for c, i in char2idx.items()}
    )
    cfg = checkpoint["args"]

    model = TransformerLM(
        vocab_size=len(char2idx),
        d_model=cfg["d_model"],
        nhead=cfg["nhead"],
        num_layers=cfg["num_layers"],
        dim_feedforward=cfg["dim_feedforward"],
        dropout=0.0,
        max_len=cfg["seq_len"],
    ).to(device)

    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    return model, char2idx, idx2char, cfg


def main():
    parser = argparse.ArgumentParser(
        description="基于 Transformer 语言模型的拼音输入法"
    )
    parser.add_argument("--model_path", default="best_model_transformer.pt")
    parser.add_argument("--pinyin_map", default="pinyin_map.json")
    parser.add_argument("--topk", type=int, default=5, help="展示候选数")
    parser.add_argument("--beam", type=int, default=10, help="束搜索宽度")
    parser.add_argument(
        "--unigram", default="*.txt",
        help="统计字频先验所使用的语料文件匹配模式",
    )
    args = parser.parse_args()

    if args.topk <= 0 or args.beam <= 0:
        parser.error("topk 和 beam 必须为正整数。")

    global PINYIN_MAP, _SYLLABLES, UNI_LOGPROB
    PINYIN_MAP = _load_pinyin_map(args.pinyin_map)
    _SYLLABLES = sorted(PINYIN_MAP.keys(), key=len, reverse=True)
    print(f"拼音映射表：{args.pinyin_map}（{len(PINYIN_MAP)} 个音节）")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"运行设备：{device}")

    model, char2idx, idx2char, cfg = load_transformer_checkpoint(
        args.model_path, device
    )

    UNI_LOGPROB = build_unigram_logprob(
        args.unigram, char2idx, device
    )

    print(
        f"模型：{args.model_path}（Transformer，词表 {len(char2idx)} 字，"
        f"最大上下文 {model.max_len}）"
    )
    run(model, char2idx, idx2char, args.topk, args.beam, device)


if __name__ == "__main__":
    main()
