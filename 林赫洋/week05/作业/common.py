"""公共工具：路径、基座模型、chat 模板、loss mask、生成循环。

本文件把第 5 周 PPT 里的三处知识点落成代码：

1. chat 模板（PPT 第 27 页）
   <|user|>问题<|assistant|>回答<|end|>
   三个角色标记让模型知道「谁在说话、说到哪算完」。

2. loss mask（PPT 第 28 页）
   提问部分的 label 全部设成 -100，只有回答部分参与 loss。
   这样模型学的是「怎么答」，不是「怎么背问题」。

3. 生成循环（PPT 第 3-4 页）
   模型一次只出一个 token，把它拼回输入再前向一次，直到生成 <|end|>。

用法：
    import common
    tok, model = common.load_base_model()

文件与脚本放在同一目录，整个文件夹可以整体拷走。
"""

import os
import sys
import json
import math
import time

import torch


# ─────────────────────────── 路径 ───────────────────────────

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "model", "wenzhong-gpt2-110m")
DATA_DIR = os.path.join(BASE_DIR, "data")
OUT_DIR = os.path.join(BASE_DIR, "out")

SFT_DATA = os.path.join(DATA_DIR, "sft_data.jsonl")
GENERAL_CORPUS = os.path.join(DATA_DIR, "general_zh.txt")
SFT_MODEL_DIR = os.path.join(OUT_DIR, "sft_model")


# ─────────────────────────── chat 模板 ───────────────────────────

USER_TAG = "<|user|>"
ASST_TAG = "<|assistant|>"
END_TAG = "<|end|>"
SPECIAL_TOKENS = [USER_TAG, ASST_TAG, END_TAG]

IGNORE_INDEX = -100   # PyTorch 的约定：label = -100 的位置不参与 loss


# ─────────────────────────── 基座模型 ───────────────────────────

MODEL_ID = "IDEA-CCNL/Wenzhong-GPT2-110M"
HF_MIRROR = "https://hf-mirror.com"
MODEL_FILES = [
    "config.json",
    "model.safetensors",
    "tokenizer.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "vocab.json",
    "merges.txt",
]


def log(msg=""):
    print(msg, flush=True)


def pick_device(name="auto"):
    """自动选 GPU，没有 GPU 就用 CPU（CPU 也能跑，只是慢）。"""
    if name and name != "auto":
        return torch.device(name)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def ensure_model(mirror=HF_MIRROR):
    """本地没有基座权重才下载，目录已完整就直接返回。

    用「镜像 + requests 直连」下载，不依赖 hf CLI；下载失败会把可选方案打印出来。
    """
    missing = [f for f in MODEL_FILES if not os.path.isfile(os.path.join(MODEL_DIR, f))]
    if not missing:
        return MODEL_DIR

    os.makedirs(MODEL_DIR, exist_ok=True)
    log(f"[模型] 本地缺少 {len(missing)} 个文件，开始从镜像下载（约 275 MB）……")
    log(f"[模型] 源：{mirror}/{MODEL_ID}")

    import requests  # transformers 的依赖里已经带了 requests

    base = f"{mirror}/{MODEL_ID}/resolve/main/"
    for name in missing:
        path = os.path.join(MODEL_DIR, name)
        with requests.get(base + name, stream=True, timeout=60, allow_redirects=True) as r:
            r.raise_for_status()
            total = int(r.headers.get("Content-Length", 0))
            done = 0
            tmp = path + ".part"
            with open(tmp, "wb") as f:
                for chunk in r.iter_content(1024 * 512):
                    f.write(chunk)
                    done += len(chunk)
                    if total > 20 * 1024 * 1024 and done % (20 * 1024 * 1024) < 512 * 1024:
                        log(f"       {name}: {done / 1e6:.0f}/{total / 1e6:.0f} MB")
            os.replace(tmp, path)
        log(f"       完成 {name}")

    log("[模型] 下载完成。")
    return MODEL_DIR


def load_tokenizer():
    """加载分词器，并补上三个 chat 模板专用标记。"""
    from transformers import AutoTokenizer

    ensure_model()
    tok = AutoTokenizer.from_pretrained(MODEL_DIR)
    tok.add_special_tokens({"additional_special_tokens": SPECIAL_TOKENS})
    # GPT-2 本身没有 pad 标记，用 eos 顶替；真正参与 loss 的位置我们自己用 -100 控制
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    return tok


def load_base_model(device=None):
    """加载「只预训练过」的基座模型（PPT 第 25 页的那个续写器）。"""
    from transformers import AutoModelForCausalLM

    device = device or pick_device()
    tok = load_tokenizer()
    model = AutoModelForCausalLM.from_pretrained(MODEL_DIR)
    # 加了 3 个特殊 token，词表变大，必须同步扩展输出层，否则 id 越界
    model.resize_token_embeddings(len(tok))
    model.to(device)
    model.eval()
    return tok, model


def load_sft_model(device=None):
    """加载训练好的 SFT 模型（还没训练过时给出明确提示）。"""
    from transformers import AutoModelForCausalLM, AutoTokenizer

    if not os.path.isdir(SFT_MODEL_DIR):
        raise FileNotFoundError(
            "还没有训练好的 SFT 模型。先运行：python sft_train.py"
        )
    device = device or pick_device()
    tok = AutoTokenizer.from_pretrained(SFT_MODEL_DIR)
    model = AutoModelForCausalLM.from_pretrained(SFT_MODEL_DIR)
    model.to(device)
    model.eval()
    return tok, model


# ─────────────────────────── 一条样本 ───────────────────────────

def encode_turn(tok, question, answer):
    """一条 (提问, 回答) → (input_ids, labels)，提问段 label 全部是 -100。"""
    uid = tok.convert_tokens_to_ids(USER_TAG)
    aid = tok.convert_tokens_to_ids(ASST_TAG)
    eid = tok.convert_tokens_to_ids(END_TAG)

    prompt_ids = [uid] + tok(question, add_special_tokens=False)["input_ids"] + [aid]
    answer_ids = tok(answer, add_special_tokens=False)["input_ids"] + [eid]

    input_ids = prompt_ids + answer_ids
    labels = [IGNORE_INDEX] * len(prompt_ids) + answer_ids
    return input_ids, labels


def encode_sample(tok, question, answer, max_len):
    input_ids, labels = encode_turn(tok, question, answer)
    return input_ids[:max_len], labels[:max_len]


def encode_plain(tok, text, max_len):
    """通用语料：整段都是监督信号（label 与输入相同），用来对抗灾难性遗忘。"""
    ids = tok(text, add_special_tokens=False)["input_ids"][:max_len]
    return ids, list(ids)


class SFTDataset(torch.utils.data.Dataset):
    def __init__(self, pairs, tok, max_len):
        self.samples = [encode_sample(tok, q, a, max_len) for q, a in pairs]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, i):
        return self.samples[i]


class GeneralDataset(torch.utils.data.Dataset):
    """把通用语料切成定长块，随机取块——只做「混一点」用。

    块长默认比 SFT 样本短（256）：一个 batch 里有几块通用语料时，
    整批会被补齐到最长的那个，块太长会把每一步都拖慢好几倍。
    """

    def __init__(self, text, tok, max_len, chunk_len=256):
        ids = []
        # 分片 tokenize：整份几十万字一次性喂给 tokenizer 会触发超长警告，也吃内存
        for i in range(0, len(text), 100_000):
            ids += tok(text[i:i + 100_000], add_special_tokens=False)["input_ids"]
        chunk_len = max(64, min(chunk_len, max_len))
        self.chunks = [ids[i:i + chunk_len] for i in range(0, len(ids) - chunk_len, chunk_len)]
        self.max_len = max_len

    def __len__(self):
        return max(1, len(self.chunks))

    def __getitem__(self, i):
        ids = self.chunks[i % len(self.chunks)]
        return ids, list(ids)


def collate(batch, pad_id):
    """把变长样本补到同一长度。

    返回 (input_ids, labels, attention_mask)：
    - 补出来的位置 label 是 -100，不参与 loss
    - attention_mask 为 0，让注意力不去看补齐出来的位置
    """
    max_len = max(len(x) for x, _ in batch)
    xs, ys, ms = [], [], []
    for ids, labels in batch:
        pad = max_len - len(ids)
        xs.append(ids + [pad_id] * pad)
        ys.append(labels + [IGNORE_INDEX] * pad)
        ms.append([1] * len(ids) + [0] * pad)
    return (torch.tensor(xs, dtype=torch.long),
            torch.tensor(ys, dtype=torch.long),
            torch.tensor(ms, dtype=torch.long))


# ─────────────────────────── 生成循环 ───────────────────────────

def sample_next(logits, temperature=0.7, top_p=0.9):
    """从一个 vocab 维的打分里选下一个 token（PPT 第 5 页的选词策略）。"""
    if temperature is None or temperature <= 0:
        return int(torch.argmax(logits))
    probs = torch.softmax(logits / temperature, dim=-1)
    if top_p and top_p < 1.0:
        sorted_probs, sorted_idx = torch.sort(probs, descending=True)
        cumulative = torch.cumsum(sorted_probs, dim=-1)
        cut = cumulative > top_p
        cut[1:] = cut[:-1].clone()      # 至少保留概率最高的那一个
        cut[0] = False
        sorted_probs[cut] = 0.0
        probs = torch.zeros_like(probs).scatter_(0, sorted_idx, sorted_probs)
        probs = probs / probs.sum()
    return int(torch.multinomial(probs, num_samples=1))


@torch.no_grad()
def generate(model, tok, question, device, history=None,
             max_new_tokens=64, temperature=0.7, top_p=0.9):
    """生成一个回答：每次只出一个 token，拼回输入再算（PPT 第 4 页）。"""
    ids = []
    for q, a in history or []:
        turn_ids, _ = encode_turn(tok, q, a)
        ids += turn_ids

    uid = tok.convert_tokens_to_ids(USER_TAG)
    aid = tok.convert_tokens_to_ids(ASST_TAG)
    eid = tok.convert_tokens_to_ids(END_TAG)

    ids += [uid] + tok(question, add_special_tokens=False)["input_ids"] + [aid]
    ids = ids[-(model.config.n_positions - max_new_tokens):]   # 超长就截掉最早的对话

    x = torch.tensor([ids], dtype=torch.long, device=device)
    out_ids = []
    for _ in range(max_new_tokens):
        logits = model(x).logits[0, -1].float()
        nxt = sample_next(logits, temperature, top_p)
        if nxt == eid:          # 学会输出 <|end|> 就停
            break
        out_ids.append(nxt)
        x = torch.cat([x, torch.tensor([[nxt]], dtype=torch.long, device=device)], dim=1)
    text = tok.decode(out_ids, skip_special_tokens=True)
    # 字节级 BPE 有可能在最后一个汉字中间停下，解码会留下半个字（U+FFFD），去掉
    return text.replace("\ufffd", "").strip()


# ─────────────────────────── 小工具 ───────────────────────────

def read_jsonl(path):
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def split_pairs(items, val_ratio=0.1, seed=42, group_key=None):
    """固定随机种子切训练/验证集，保证每次运行切法一致。

    给了 group_key 就「按组」切：同一个问题的几种问法必须整组落在同一边，
    否则验证集里全是训练集见过的答案，val_loss 会虚低。
    """
    import random
    rng = random.Random(seed)
    items = list(items)
    if group_key is None:
        rng.shuffle(items)
        n_val = max(1, int(len(items) * val_ratio))
        return items[n_val:], items[:n_val]

    groups = {}
    for it in items:
        groups.setdefault(group_key(it), []).append(it)
    keys = list(groups)
    rng.shuffle(keys)
    n_val = max(1, int(len(keys) * val_ratio))
    val_keys = set(keys[:n_val])
    train = [it for it in items if group_key(it) not in val_keys]
    val = [it for it in items if group_key(it) in val_keys]
    return train, val


def lr_lambda_factory(total_steps, warmup_ratio=0.05):
    """线性 warmup + 余弦退火（PPT 第 31 页「小学习率、少 epoch」的配套）。"""
    warmup = max(1, int(total_steps * warmup_ratio))

    def fn(step):
        if step < warmup:
            return step / warmup
        progress = (step - warmup) / max(1, total_steps - warmup)
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    return fn
