"""
字符级 Transformer 语言模型 → SFT（指令微调）。单文件版：模型定义内联，
不 import 上周的 language_model_transformer.py，复制到别处也能单独运行。

上周的 language_model_transformer.py 训练的是"无监督语言模型"：
喂一段连续文本，让它预测下一个字。它学到的是"中文接下来大概率写什么"，
所以给它一句"用户：给定两个数字，请选择较大数"，它只会顺着这个开头
往下胡诌，而不是"回答"这个问题。

SFT（Supervised Fine-Tuning，监督微调 / 指令微调）就是在预训练模型上
再做一轮监督学习，把"续写"扭成"回答"：

    输入（prompt，不参与 loss）: <bos>用户：给定两个数字…\n助手：
    目标（answer，参与 loss）  : 较大的数字是 564。<eos>

本文件自带全部代码：模型定义（LM / CausalSelfAttention / TransformerBlock /
PositionalEncoding）从上周脚本原样搬进来，见下面的「模型定义」一节，不再
import 上周的文件。于是它的依赖只剩两样数据：

    - 训练要读上周产出的 best_model_transformer.pt 当起点 —— SFT 本来就是
      "从预训练权重接着训"，没有它就不是 SFT（想对照可以加 --scratch）；
    - 训练产出的 best_model_sft.pt 自带词表 / 模板 / 结构超参，
      --chat / --prompt / --eval 只认这一个文件。

和上周脚本相比，模型结构一个字没改，只多了三件事：
    1. 扩词表   —— 上周的 4000 字词表是从别的语料统计出来的，SFT 数据里
                   有 100 多个字它没见过（最典型的是空格，还有 = + [ ] 这类
                   符号）。直接把 OOV 字丢掉会让"数字1: 238"变成"数字1238"，
                   内容被悄悄改坏；全映射成 <unk> 又会把符号信息抹平。
                   所以把缺的字追加进词表，并把 embedding / 输出层一起扩长。
    2. 拼模板   —— 把 (instruction, output) 拼成一条字符序列，并给样本加上
                   <bos> / <eos>，模型才知道"回答从这里开始、到哪里结束"。
    3. 掩 prompt —— 只对回答段算 loss，问题段 label 置 -100（ignore_index）。

关键点 —— 为什么必须做 prompt 掩码
    如果整条序列都算 loss，模型就会被要求把"用户的问题"也当作可生成的
    文本来拟合。这部分信号在推理时毫无用处（问题是用户给定的），却会跟
    回答抢模型容量；更糟的是问题里常出现的格式符号会被学成生成模式。
    把 prompt 段的 label 置成 -100，交叉熵直接跳过这些位置，梯度只来自
    回答 —— 模型全部容量都用在"怎么答"上。

关键点 —— 扩词表后旧的 embedding 不能动
    预训练词表里每个字对应 embedding 的第 0..3999 行，id 一旦重排，
    checkpoint 里那 4000 行权重就全错位了。所以字表只允许"往后追加"，
    老字的 id 必须原样保留；新追加的行用老 embedding 的均值加小噪声初始化，
    尺度和老字同量级，模型不用先花几轮把新字的向量尺度"校"回来。

用法:
    python language_model_sft.py                          # 默认超参训练（CPU 约 20 分钟）
    python language_model_sft.py --epochs 5 --batch_size 16
    python language_model_sft.py --self_test              # 只跑标签/掩码自检，不训练
    python language_model_sft.py --chat --ckpt best_model_sft.pt
    python language_model_sft.py --prompt "计算 3 的平方根" --ckpt best_model_sft.pt
    python language_model_sft.py --eval --ckpt best_model_sft.pt   # 只算验证集 loss
    python language_model_sft.py --scratch                # 对照实验：不加载预训练权重

生成参数就是课上 generation_lab 里那几个旋钮：--temperature 0.8 --top_k 5，
还可以加 --top_p 0.9 按累积概率切长尾；--temperature 0 退化成贪心。

从任意目录启动都可以：--ckpt / --train / --val 的相对路径先按当前工作目录
查找，找不到时自动回退到脚本所在目录。
"""

import argparse
import json
import math
import os
import random
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset


# ─────────────────────────── 模板与常量 ───────────────────────────

# -100 是 nn.CrossEntropyLoss 的 ignore_index：这些位置不进 loss、不产生梯度
IGNORE_INDEX = -100

# 会话模板。用"用户：/助手："这种自然语言标记，而不是凭空造几个特殊符号：
# 预训练语言模型本来就见过"用户""助手""："这些字，换成新符号等于让模型从零
# 开始学一套标记，小模型要花更多步才能学会。多轮对话以后就是重复拼接。
PROMPT_TEMPLATE = "<bos>用户：{instruction}\n助手："


def prompt_of(instruction):
    """问题 → 模型实际看到的输入前缀（生成时也用它，训练和推理必须一致）。"""
    return PROMPT_TEMPLATE.format(instruction=instruction)


# ─────────────────────────── 路径 ───────────────────────────

def resolve_path(path):
    """
    相对路径先按当前工作目录解析，找不到再回退到脚本所在目录。

    和上周 corpus_paths() 的思路一样：脚本、数据、checkpoint 分在不同子目录，
    在哪个目录下启动都能跑起来，不用每次都写绝对路径。
    """
    if not path or os.path.isabs(path):
        return path
    if os.path.exists(path):
        return path
    here = os.path.dirname(os.path.abspath(__file__))
    alt = os.path.join(here, path)
    return alt if os.path.exists(alt) else path


# ─────────────────────────── 词表 / 分词 ───────────────────────────

class Tokenizer:
    """
    字符级分词器 + 4 个特殊符号。

    词表布局（id 从 0 递增）：
        [0 .. V-1]        上周预训练词表，顺序和 id 原样保留
        [V .. V+3]        <pad> <bos> <eos> <unk>
        [V+4 .. ]         SFT 数据里新增的字（按出现频率排序）

    <pad>  变长样本 batch 时补齐，logits 位置和 label 位置都要靠它对齐
    <bos>  序列起点（模板开头）
    <eos>  回答结束标记：训练时教模型"说完就停"，生成时遇到它就收工
    <unk>  训练时没见过的字（比如 emoji），训练集里不出现也不算浪费 ——
           推理时用户随时可能打出词表外的字，总得有个位置接住它
    """

    SPECIALS = ("<pad>", "<bos>", "<eos>", "<unk>")

    def __init__(self, base_chars=()):
        self.char2idx = {}
        for c in base_chars:
            if c not in self.char2idx:
                self.char2idx[c] = len(self.char2idx)
        for s in self.SPECIALS:
            self.char2idx[s] = len(self.char2idx)
        self.idx2char = {i: c for c, i in self.char2idx.items()}
        self._bind_specials()

    @classmethod
    def from_char2idx(cls, char2idx):
        """从 checkpoint 里存的字表恢复分词器（id 已经是最终形态，直接沿用）。"""
        tok = cls.__new__(cls)
        tok.char2idx = dict(char2idx)
        tok.idx2char = {i: c for c, i in tok.char2idx.items()}
        tok._bind_specials()
        return tok

    def _bind_specials(self):
        self.pad_id = self.char2idx["<pad>"]
        self.bos_id = self.char2idx["<bos>"]
        self.eos_id = self.char2idx["<eos>"]
        self.unk_id = self.char2idx["<unk>"]

    def __len__(self):
        return len(self.char2idx)

    def add(self, chars):
        """把新字追加到词表末尾，返回实际新增的数量（已存在的跳过）。"""
        added = 0
        for c in chars:
            if c not in self.char2idx:
                i = len(self.char2idx)
                self.char2idx[c] = i
                self.idx2char[i] = c
                added += 1
        return added

    def encode(self, text):
        """字 → id。词表外的字统一落到 <unk>，长度不变。"""
        return [self.char2idx.get(c, self.unk_id) for c in text]

    def decode(self, ids, skip_special=True):
        """id → 字。默认跳过特殊符号，直接得到人读的文本。"""
        out = []
        for i in ids:
            c = self.idx2char.get(int(i), "")
            if skip_special and c in self.SPECIALS:
                continue
            out.append(c)
        return "".join(out)


def chars_missing(tok, texts):
    """
    按出现频率从高到低，返回 texts 里 tok 还不认识的字。

    频率高的先进词表：万一以后要限制新增字数上限，先保高频字更划算。
    """
    from collections import Counter
    counter = Counter()
    for t in texts:
        counter.update(c for c in t if c not in tok.char2idx)
    return [c for c, _ in counter.most_common()]


def count_unk(tok, rows):
    """统计一批样本里有多少个字符会落到 <unk>（衡量词表覆盖率）。"""
    n = 0
    for row in rows:
        n += sum(1 for i in tok.encode(row["instruction"] + row["output"]) if i == tok.unk_id)
    return n


# ─────────────────────────── 数据 ───────────────────────────

def read_jsonl(path):
    """读 SFT 数据：每行一个 {"instruction": ..., "output": ...}。"""
    rows = []
    with open(path, encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            if "instruction" not in row or "output" not in row:
                raise ValueError(f"{path} 第 {line_no} 行缺少 instruction / output 字段: {row}")
            rows.append(row)
    return rows


class SFTDataset(Dataset):
    """
    一条 (instruction, output) → 一条 (input_ids, labels)。

        input  : <bos>用户：{instruction}\n助手：{output}<eos>
        label  : -100 ... -100                | {output}<eos>
                 ^^^^^^^^^^^^^^^^^^^^^^^^^^     ^^^^^^^^^^^^
                 只当条件用，不算 loss            真正要学的部分

    label 和 input 是等长的：位置 t 的 label 是"位置 t 这个字该是什么"。
    真正算 loss 时要错一位：用位置 t 的输出 logits 去预测 label[t+1]，
    这和上周 CharDataset 里 x=前 n 字、y=后错一位是同一件事。

    labels 里 prompt 段全是 -100，CrossEntropyLoss 会把它们跳过 ——
    这就是 SFT 的 prompt 掩码，一行数据构造就能看出效果。
    """

    def __init__(self, rows, tok, max_len=None):
        self.samples = []
        self.n_too_long = 0
        for row in rows:
            prompt_ids = tok.encode(prompt_of(row["instruction"]))
            answer_ids = tok.encode(row["output"]) + [tok.eos_id]
            ids = prompt_ids + answer_ids
            if max_len and len(ids) > max_len:
                # 超长样本整条丢掉：只截断回答会把 <eos> 截掉，模型就学不会
                # "说完就停"；只截断问题又会把条件改坏。数据量够时直接跳过最干净。
                self.n_too_long += 1
                continue
            labels = [IGNORE_INDEX] * len(prompt_ids) + answer_ids
            self.samples.append((ids, labels))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


def make_collate_fn(pad_id):
    """把不同长度的样本补齐到 batch 内最大长度，pad 位置的 label 保持 -100。"""

    def collate(batch):
        max_len = max(len(ids) for ids, _ in batch)
        input_ids = torch.full((len(batch), max_len), pad_id, dtype=torch.long)
        labels = torch.full((len(batch), max_len), IGNORE_INDEX, dtype=torch.long)
        for i, (ids, lab) in enumerate(batch):
            input_ids[i, :len(ids)] = torch.tensor(ids, dtype=torch.long)
            labels[i, :len(lab)] = torch.tensor(lab, dtype=torch.long)
        return input_ids, labels

    return collate


# ─────────────────────────── 模型定义（从上周脚本原样搬来） ───────────────────────────
# 这一节和上周 language_model_transformer.py 里的 LM 及子模块逐行一致，
# 搬过来的唯一目的是让本文件能独立运行。要让上周的预训练 checkpoint
# （embed.weight / blocks.* / norm_f.* / fc.*）能直接 load 进来，这里的
# 模块名、层数组织、参数形状一个都不能改；以后改模型结构，两处要同步改。

NEG_INF = float("-inf")


def build_causal_mask(seq_len, device=None):
    """
    加性因果 mask，形状 (T, T)：
        mask[i, j] = 0      当 j <= i  —— 第 i 个位置可以看到自己及其左侧
        mask[i, j] = -inf   当 j >  i  —— 右侧（未来）位置被彻底屏蔽

    用 -inf 而不是 0/1 开关：它可以直接加在 softmax 的输入上，
    exp(-inf) = 0，未来位置的注意力权重天然就是 0。
    对角线是 0（diagonal=1 表示从对角线上方一格开始屏蔽），因为预测当前
    位置时当前位置的输入是允许看的 —— 这正是"错一位"的由来。
    """
    mask = torch.full((seq_len, seq_len), NEG_INF, device=device)
    return torch.triu(mask, diagonal=1)


class PositionalEncoding(nn.Module):
    """
    正弦位置编码（Attention Is All You Need 原文公式），无参数：
        PE[pos, 2i]   = sin(pos / 10000^(2i/d))
        PE[pos, 2i+1] = cos(pos / 10000^(2i/d))

    自注意力对顺序是无感知的，位置编码负责把"这是第几个字"注入进去。
    按需现算，所以 SFT 的样本比预训练的 seq_len=64 更长（最长 128）也不会越界。
    """

    def __init__(self, d_model):
        super().__init__()
        assert d_model % 2 == 0, "d_model 取偶数，正弦/余弦才能成对排布"
        self.d_model = d_model

    def forward(self, x):
        # x: (B, T, d_model)
        T, d = x.size(-2), self.d_model
        device, dtype = x.device, x.dtype
        pos = torch.arange(T, device=device, dtype=torch.float32).unsqueeze(1)   # (T, 1)
        idx = torch.arange(0, d, 2, device=device, dtype=torch.float32)          # (d/2,)
        div = torch.exp(-math.log(10000.0) * idx / d)                            # (d/2,)
        pe = torch.zeros(T, d, device=device, dtype=torch.float32)
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        return x + pe.to(dtype)


class CausalSelfAttention(nn.Module):
    """多头自注意力，attn_mask 为加性 mask（因果 mask / padding mask 都在这里生效）。"""

    def __init__(self, d_model, nhead, dropout):
        super().__init__()
        assert d_model % nhead == 0, f"d_model({d_model}) 必须能被 nhead({nhead}) 整除"
        self.nhead = nhead
        self.head_dim = d_model // nhead
        self.scale = self.head_dim ** -0.5        # 1/√d_head，防止点积过大让 softmax 饱和
        self.qkv = nn.Linear(d_model, 3 * d_model)  # 一个矩阵一次算出 Q/K/V
        self.proj = nn.Linear(d_model, d_model)
        self.attn_drop = nn.Dropout(dropout)
        self.proj_drop = nn.Dropout(dropout)

    def forward(self, x, attn_mask=None):
        B, T, C = x.shape
        q, k, v = self.qkv(x).split(C, dim=-1)                     # 各 (B, T, C)
        # 拆多头：(B, T, H, Dh) → (B, H, T, Dh)
        q = q.reshape(B, T, self.nhead, self.head_dim).transpose(1, 2)
        k = k.reshape(B, T, self.nhead, self.head_dim).transpose(1, 2)
        v = v.reshape(B, T, self.nhead, self.head_dim).transpose(1, 2)

        scores = (q @ k.transpose(-2, -1)) * self.scale             # (B, H, T, T)
        if attn_mask is not None:
            scores = scores + attn_mask                             # ← mask 在这里生效
        attn = self.attn_drop(torch.softmax(scores, dim=-1))        # 未来位置权重 = 0
        out = attn @ v                                              # (B, H, T, Dh)
        out = out.transpose(1, 2).reshape(B, T, C)
        return self.proj_drop(self.proj(out))


class TransformerBlock(nn.Module):
    """
    一层 Transformer（Pre-LN 写法）：
        x = x + Attention(LN(x))
        x = x + FFN(LN(x))
    """

    def __init__(self, d_model, nhead, ffn_dim, dropout):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.attn = CausalSelfAttention(d_model, nhead, dropout)
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, ffn_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_dim, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x, attn_mask=None):
        x = x + self.attn(self.norm1(x), attn_mask)
        x = x + self.ffn(self.norm2(x))
        return x


class LM(nn.Module):
    """字符级 Transformer 语言模型：Embedding → (PE) → N×Block → LayerNorm → Linear。"""

    def __init__(self, vocab_size, d_model, nhead, num_layers, ffn_dim, dropout):
        super().__init__()
        self.d_model = d_model
        self.embed = nn.Embedding(vocab_size, d_model)
        self.pos = PositionalEncoding(d_model)
        self.blocks = nn.ModuleList(
            TransformerBlock(d_model, nhead, ffn_dim, dropout)
            for _ in range(num_layers)
        )
        self.norm_f = nn.LayerNorm(d_model)     # 所有 block 之后再归一化一次
        self.drop = nn.Dropout(dropout)
        self.fc = nn.Linear(d_model, vocab_size)

    def forward(self, x, key_padding_mask=None, causal=True):
        """
        x: (B, T) 字符 id
        key_padding_mask: (B, T) bool，True 表示该位置是 padding、不参与注意力。
            SFT 的 batch 里样本长短不一，右侧补齐，靠它显式标明 pad。
            （注意：右侧 padding 在因果模式下本来就在"未来"、已被因果 mask 挡住，
            这个参数主要是把语义写明确，以及给以后左侧 padding 留好接口。）
        causal: 仅给 --self_test 做对照实验用，正常训练/推理恒为 True。
        返回: (B, T, vocab_size)，每个位置的"下一个字"打分
        """
        B, T = x.shape

        attn_mask = build_causal_mask(T, x.device) if causal else None   # (T, T)
        if key_padding_mask is not None:
            pad_mask = torch.zeros(B, 1, 1, T, device=x.device,
                                   dtype=self.embed.weight.dtype)
            pad_mask = pad_mask.masked_fill(key_padding_mask[:, None, None, :], NEG_INF)
            attn_mask = pad_mask if attn_mask is None else attn_mask + pad_mask

        # 乘 √d_model：让 embedding 的尺度与位置编码、残差同量级
        h = self.drop(self.pos(self.embed(x) * math.sqrt(self.d_model)))
        for block in self.blocks:
            h = block(h, attn_mask)
        return self.fc(self.norm_f(h))          # (B, T, V)


# ─────────────────────────── 模型：扩词表 ───────────────────────────

def expand_model_vocab(model, old_vocab_size, new_vocab_size):
    """
    把输入 embedding 和输出层从 old_vocab_size 扩到 new_vocab_size。

    为什么不是"重新建一个模型再拷贝"：
        LM.__init__ 里创建的是旧形状（vocab_size 决定 embed/fc 的行数），
        直接 load_state_dict 会因为形状不匹配报错；这里先按旧形状加载权重，
        再把这两层换成新形状，旧行原样搬过去，其余层（attention/FFN/LN）
        完全不动 —— 预训练学到的东西一点不丢。

    新行为什么用"均值 + 小噪声"初始化：
        nn.Embedding 默认是 N(0, 1)，和预训练 embedding 的实际尺度对不上，
        新字一进来就成了离群点；用旧行均值起步，尺度自然一致。
        但也不能所有新行都取同一个均值 —— 那样模型一开始区分不了
        "空格"和"="，梯度给它们的更新完全一样，要多花好几轮才能分开；
        加 0.01 的小噪声就能让它们从第一步起就有区别。
    """
    if new_vocab_size <= old_vocab_size:
        return model

    d_model = model.embed.embedding_dim
    old_embed = model.embed.weight.detach().clone()
    old_fc_w = model.fc.weight.detach().clone()
    old_fc_b = model.fc.bias.detach().clone()
    n_new = new_vocab_size - old_vocab_size

    new_embed = nn.Embedding(new_vocab_size, d_model)
    new_fc = nn.Linear(d_model, new_vocab_size)
    with torch.no_grad():
        new_embed.weight[:old_vocab_size] = old_embed
        new_embed.weight[old_vocab_size:] = (
            old_embed.mean(dim=0, keepdim=True) + 0.01 * torch.randn(n_new, d_model)
        )
        new_fc.weight[:old_vocab_size] = old_fc_w
        new_fc.bias[:old_vocab_size] = old_fc_b
        # 输出层新行取旧行的均值、bias 置 0：新字初始 logit 是中性值，
        # 不会一上来就把某些老字的概率压下去
        new_fc.weight[old_vocab_size:] = old_fc_w.mean(dim=0, keepdim=True)
        new_fc.bias[old_vocab_size:] = 0.0

    model.embed = new_embed
    model.fc = new_fc
    return model


def load_sft_checkpoint(path, device):
    """
    读 SFT checkpoint，重建模型和分词器，供对话/评估复用：
        from language_model_sft import load_sft_checkpoint, generate
    """
    ckpt = torch.load(path, map_location=device)
    cfg = ckpt.get("args", {})
    if cfg.get("arch") != "sft-transformer":
        raise ValueError(
            f"{path} 不是 SFT checkpoint（arch={cfg.get('arch')!r}）。\n"
            "  这个文件是上周的无监督语言模型：它有词表但没有 SFT 的扩展词表、\n"
            "  也不知道 <bos>/<eos> 和 “用户：/助手：” 这套模板。\n"
            "  先运行 python language_model_sft.py 训练出 best_model_sft.pt，\n"
            "  再用 --ckpt best_model_sft.pt。"
        )
    tok = Tokenizer.from_char2idx(ckpt["char2idx"])
    model = LM(
        vocab_size=len(tok),
        d_model=cfg["d_model"],
        nhead=cfg["nhead"],
        num_layers=cfg["num_layers"],
        ffn_dim=cfg["ffn_dim"],
        dropout=0.0,        # 推理阶段关闭 dropout
    ).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return model, tok, ckpt


# ─────────────────────────── 生成 ───────────────────────────

def generate(model, tok, instruction, max_new_tokens=64, top_k=5, top_p=1.0,
             temperature=0.8, device="cpu"):
    """
    从 prompt 开始逐字自回归，遇到 <eos> 或达到长度上限就停。

    每步都把"整段 prompt + 已生成的字"重新送进模型，只取最后一个位置的
    logits —— 没做 KV cache。2 层 128 维的模型重算几十个 token 代价很小，
    代码少一半；真按大模型那套做 cache，复杂度会盖过 SFT 本身。

    解码顺序和课上 generation_lab 一致：温度 → top-k → top-p → 归一化采样。
    temperature=0 退化成贪心（确定性，但小模型贪心最容易复读，看演示时建议
    用默认的温度采样）。

    特殊符号里 <pad>/<bos>/<unk> 直接从候选里禁掉（-inf）：这三个符号
    只在特定位置有意义，让模型自由生成它们基本只会制造噪音；
    <eos> 必须留着，它是"停止"的唯一出口。
    """
    model.eval()
    ids = tok.encode(prompt_of(instruction))
    x = torch.tensor([ids], dtype=torch.long, device=device)
    out_ids = []
    banned = [tok.pad_id, tok.bos_id, tok.unk_id]

    with torch.no_grad():
        for _ in range(max_new_tokens):
            logits = model(x)[0, -1].float().clone()
            logits[banned] = float("-inf")

            if temperature and temperature > 0:
                logits = logits / temperature
                if top_k and top_k > 0:
                    # 只保留分数最高的 top_k 个候选，其余置 -inf 后再归一化
                    k = min(top_k, logits.numel())
                    kth = torch.topk(logits, k).values[-1]
                    logits = logits.masked_fill(logits < kth, float("-inf"))
                if top_p and 0 < top_p < 1.0:
                    # 核采样：按概率从高到低累加，凑够 p 就把长尾切掉。
                    # 和 top-k 的区别：top-k 固定切 k 个，top-p 按"概率质量"切，
                    # 分布很尖的时候可能只留一两个候选。
                    sorted_logits, sorted_idx = logits.sort(descending=True)
                    probs_sorted = torch.softmax(sorted_logits, dim=-1)
                    cum = probs_sorted.cumsum(dim=-1)
                    # 累积概率"在它之前"已经 >= p 的候选丢掉，跨过阈值那个保留
                    remove = (cum - probs_sorted) >= top_p
                    sorted_logits = sorted_logits.masked_fill(remove, float("-inf"))
                    logits = torch.full_like(logits, float("-inf")).scatter(0, sorted_idx, sorted_logits)
                probs = torch.softmax(logits, dim=-1)
                nxt = int(torch.multinomial(probs, 1))
            else:
                nxt = int(logits.argmax())      # temperature=0 → 贪心

            if nxt == tok.eos_id:
                break
            out_ids.append(nxt)
            x = torch.cat([x, torch.tensor([[nxt]], device=device)], dim=1)

    return tok.decode(out_ids)


def show_preview(model, tok, rows, gen_kwargs, device, tag=""):
    """训练前后各打印几条生成结果，直观看到"胡诌"变成"回答"。"""
    model.eval()
    clip = lambda s, n=60: s if len(s) <= n else s[:n] + "…"
    for row in rows:
        ans = generate(model, tok, row["instruction"], device=device, **gen_kwargs)
        print(f"  {tag}问：{clip(row['instruction'])}")
        print(f"  {' ' * len(tag)}答：{clip(ans) if ans else '（空）'}")


# ─────────────────────────── 训练 / 评估 ───────────────────────────

def run_epoch(model, loader, criterion, tok, device, optimizer=None,
              grad_clip=0.0, log_every=0, log_file="", epoch=0, t0=None):
    """
    跑一轮。optimizer=None 表示只评估。

    返回 (平均 loss, token 准确率)，两者都只在"回答 token"上统计：
    prompt 和 padding 的 label 是 -100，被 ignore_index 跳过。
    """
    train = optimizer is not None
    model.train(train)
    total_loss = 0.0
    total_tokens = 0
    correct = 0
    n_steps = len(loader)

    for step, (x, y) in enumerate(loader, 1):
        x, y = x.to(device), y.to(device)
        # 只标记右侧 padding。因果 mask 已经挡住了"未来"的 pad，
        # 这里再传一次是为了让 padding 语义显式化：以后要是改成左侧 padding
        # 或者双向注意力，mask 不用重写。代价可以忽略。
        pad_mask = x == tok.pad_id

        logits = model(x, key_padding_mask=pad_mask)
        logits = logits[:, :-1]         # 位置 t 的输出，用来预测位置 t+1
        target = y[:, 1:]               # 错一位；-100 的位置不贡献 loss
        loss = criterion(logits.reshape(-1, logits.size(-1)), target.reshape(-1))

        if train:
            optimizer.zero_grad()
            loss.backward()
            # 微调同样要裁梯度：新加的 100 多个字 embedding 初始是噪声，
            # 前几步梯度可能偏大
            if grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()

            if log_every and step % log_every == 0:
                elapsed = (time.time() - t0) / 60 if t0 else 0.0
                msg = (f"  epoch {epoch}  step {step:>5}/{n_steps}  "
                       f"loss {loss.item():.4f}  elapsed {elapsed:5.1f} min")
                print(msg, flush=True)
                if log_file:
                    with open(log_file, "a", encoding="utf-8") as f:
                        f.write(msg + "\n")

        with torch.no_grad():
            valid = target != IGNORE_INDEX
            n = int(valid.sum())
            correct += int((logits.argmax(dim=-1)[valid] == target[valid]).sum())
        total_loss += loss.item() * n
        total_tokens += n

    return total_loss / max(1, total_tokens), correct / max(1, total_tokens)


def save_checkpoint(path, model, optimizer, epoch, best_val_loss, tok, args):
    """
    存 checkpoint。除了权重，词表和模板也必须一起存：
    SFT 后的模型带着一个"扩过的词表"，下游只有拿到同一份 char2idx
    才能把字对上 id；模板也一样，训练和推理用的模板不一致，模型表现会崩。
    """
    torch.save({
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "epoch": epoch,
        "best_val_loss": best_val_loss,
        "char2idx": tok.char2idx,
        "idx2char": tok.idx2char,
        "special_tokens": {s: tok.char2idx[s] for s in Tokenizer.SPECIALS},
        "template": PROMPT_TEMPLATE,
        # arch 用来和上周的 checkpoint 区分：下游一看就知道要不要扩词表
        "args": dict(vars(args), arch="sft-transformer", model="transformer"),
    }, path)


# ─────────────────────────── SFT 数据管线自检 ───────────────────────────

def self_test():
    """
    不训练，只验证 SFT 管线的四件事 —— 任何一件坏了，训练都会"看起来在收敛"
    但其实是错的，所以单独跑一遍：

      1. 标签对齐：prompt 段必须全是 -100，回答段必须正好是 output + <eos>；
      2. 因果 mask：改掉回答的最后一个字，前面位置的 logits 不能变；
      3. padding mask：pad 区内容随便换，有效位置的输出不能变（用双向注意力验证，
         因为因果模式下右侧 padding 本来就在"未来"，看不出 padding mask 的作用）；
      4. prompt 掩码：整条算 loss 再只取回答位置求平均，必须等于直接忽略
         回答以外位置的 loss。
    """
    torch.manual_seed(0)
    q = "给定两个数字，请选择较大的那个。 数字1: 238 数字2: 564"
    a = "较大的数字是 564。"
    alphabet = sorted(set(q + a + prompt_of("")))
    tok = Tokenizer(base_chars=alphabet)
    ds = SFTDataset([{"instruction": q, "output": a}], tok)
    ids, labels = ds[0]
    prompt_len = len(tok.encode(prompt_of(q)))
    answer_ids = tok.encode(a) + [tok.eos_id]

    # 1) 标签对齐
    align_ok = (
        len(ids) == len(labels)
        and labels[:prompt_len] == [IGNORE_INDEX] * prompt_len
        and labels[prompt_len:] == answer_ids
        and ids[prompt_len:] == answer_ids
    )

    model = LM(vocab_size=len(tok), d_model=32, nhead=4, num_layers=2,
               ffn_dim=64, dropout=0.0)
    model.eval()
    x = torch.tensor([ids], dtype=torch.long)

    # 2) 因果 mask：把最后的 <eos> 换掉，前面位置的 logits 不能受影响
    with torch.no_grad():
        base = model(x)
        changed = x.clone()
        changed[0, -1] = tok.char2idx["数"]
        new = model(changed)
    causal_ok = torch.allclose(base[0, :-1], new[0, :-1], atol=1e-6)

    # 3) padding mask：pad 区填不同的字，有效位置输出必须一致
    pad_len = 6
    pad_ids = torch.full((1, pad_len), tok.pad_id, dtype=torch.long)
    xp = torch.cat([x, pad_ids], dim=1)
    pad_mask = xp == tok.pad_id
    xp2 = xp.clone()
    xp2[0, len(ids):] = torch.randint(0, len(tok), (pad_len,))
    with torch.no_grad():
        p1 = model(xp, key_padding_mask=pad_mask, causal=False)
        p2 = model(xp2, key_padding_mask=pad_mask, causal=False)
    pad_ok = torch.allclose(p1[0, :len(ids)], p2[0, :len(ids)], atol=1e-6)

    # 4) prompt 掩码：loss 必须只等于"回答 token"的平均
    with torch.no_grad():
        logits = model(xp, key_padding_mask=pad_mask)[:, :-1]
    target = torch.full((1, xp.size(1) - 1), IGNORE_INDEX, dtype=torch.long)
    target[0, :len(labels) - 1] = torch.tensor(labels[1:], dtype=torch.long)
    flat_target = target.reshape(-1)
    loss_masked = F.cross_entropy(logits.reshape(-1, len(tok)), flat_target,
                                  ignore_index=IGNORE_INDEX)
    # 把 -100 换成真实 token（pad 位置用 pad id），得到"整段都算 loss"的逐 token 损失，
    # 再只对原回答位置求平均，应该和上面的值完全相等
    unmasked = torch.where(flat_target == IGNORE_INDEX, xp[:, 1:].reshape(-1), flat_target)
    per_token = F.cross_entropy(logits.reshape(-1, len(tok)), unmasked, reduction="none")
    only_answer = per_token[flat_target != IGNORE_INDEX].mean()
    mask_ok = torch.isclose(only_answer, loss_masked, atol=1e-6)

    print("SFT 数据管线自检")
    print(f"  [{'PASS' if align_ok else 'FAIL'}] 标签对齐：prompt 段全是 -100，回答段 = output + <eos>")
    print(f"  [{'PASS' if causal_ok else 'FAIL'}] 因果 mask：改掉回答最后一个字，前面位置的 logits 不变")
    print(f"  [{'PASS' if pad_ok else 'FAIL'}] padding mask：pad 区内容随意替换，有效位置输出不变")
    print(f"  [{'PASS' if mask_ok else 'FAIL'}] prompt 掩码：loss 只统计回答 token "
          f"({float(loss_masked):.4f} vs {float(only_answer):.4f})")

    if not (align_ok and causal_ok and pad_ok and mask_ok):
        raise SystemExit("SFT 数据管线自检失败")
    print("全部通过。")


# ─────────────────────────── 训练主流程 ───────────────────────────

def train(args):
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_rows = read_jsonl(resolve_path(args.train))
    val_rows = read_jsonl(resolve_path(args.val))
    print(f"device: {device}")
    print(f"SFT 数据: {resolve_path(args.train)} ({len(train_rows)} 条) / "
          f"{resolve_path(args.val)} ({len(val_rows)} 条)")

    data_texts = [r["instruction"] + r["output"] for r in train_rows]
    data_texts.append(prompt_of(""))        # 模板本身用到的字也要进词表

    if args.scratch:
        # 对照实验：不加载预训练，词表从 SFT 数据现建 —— 用来回答
        # "没有上周的预训练权重，光靠这 3000 条 SFT 数据能训成什么样"
        tok = Tokenizer()
        n_added = tok.add(chars_missing(tok, data_texts))
        model = LM(
            vocab_size=len(tok),
            d_model=args.d_model,
            nhead=args.nhead,
            num_layers=args.num_layers,
            ffn_dim=args.ffn_dim,
            dropout=args.dropout,
        )
        print(f"对照实验 --scratch：不加载预训练权重，词表从 SFT 数据现建（{len(tok)} 字）")
    else:
        ckpt_path = resolve_path(args.ckpt)
        pretrain = torch.load(ckpt_path, map_location="cpu")
        cfg = pretrain["args"]
        if cfg.get("arch") != "transformer":
            raise ValueError(f"{ckpt_path} 不是上周的 Transformer 语言模型 checkpoint"
                             f"（arch={cfg.get('arch')!r}）。")
        old_vocab = len(pretrain["char2idx"])
        old_model = LM(
            vocab_size=old_vocab,
            d_model=cfg["d_model"],
            nhead=cfg["nhead"],
            num_layers=cfg["num_layers"],
            ffn_dim=cfg["ffn_dim"],
            dropout=args.dropout,
        )
        old_model.load_state_dict(pretrain["model_state"])

        # 老字 id 原样保留（顺序即 id），新字追加在后面
        base_chars = [c for c, _ in sorted(pretrain["char2idx"].items(), key=lambda kv: kv[1])]
        tok = Tokenizer(base_chars=base_chars)
        n_added = tok.add(chars_missing(tok, data_texts))
        model = expand_model_vocab(old_model, old_vocab, len(tok))

        print(f"预训练 checkpoint: {ckpt_path}  (d_model={cfg['d_model']}, "
              f"{cfg['num_layers']} 层, {cfg['nhead']} 头, 词表 {old_vocab})")
        print(f"词表: {old_vocab} → {len(tok)}"
              f"（+{len(Tokenizer.SPECIALS)} 特殊符号, +{n_added} 个 SFT 新字）")

    model = model.to(device)
    print(f"模型参数量: {sum(p.numel() for p in model.parameters()):,}")

    train_ds = SFTDataset(train_rows, tok, max_len=args.max_len)
    val_ds = SFTDataset(val_rows, tok, max_len=args.max_len)
    if not train_ds:
        raise SystemExit("没有可用训练样本（全部超过 --max_len？）")
    print(f"参与训练: {len(train_ds)} 条（超长跳过 {train_ds.n_too_long} 条）  "
          f"验证: {len(val_ds)} 条（超长跳过 {val_ds.n_too_long} 条）")
    print(f"验证集未登录字符 → <unk>: {count_unk(tok, val_rows)} 处")

    collate = make_collate_fn(tok.pad_id)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              collate_fn=collate)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                            collate_fn=collate)

    criterion = nn.CrossEntropyLoss(ignore_index=IGNORE_INDEX)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr,
                                  weight_decay=args.weight_decay)

    gen_kwargs = dict(max_new_tokens=args.max_new_tokens, top_k=args.top_k,
                      top_p=args.top_p, temperature=args.temperature)
    preview_rows = val_rows[:args.preview] if args.preview > 0 else []

    # 训练前先量一个基线：同一份验证集、同一个"只算回答"的口径。
    # 这样后面表里的每一个数都是相对这个基线的变化，而不是一个孤立的 loss。
    init_name = "随机初始化" if args.scratch else "预训练权重"
    if val_ds:
        with torch.no_grad():
            base_loss, base_acc = run_epoch(model, val_loader, criterion, tok, device)
        print(f"\n训练前（{init_name}）验证集: loss {base_loss:.4f}  token 准确率 {base_acc:.4f}")

    if preview_rows:
        print(f"\nSFT 前（{init_name}，没见过一条指令数据）:")
        show_preview(model, tok, preview_rows, gen_kwargs, device)

    print(f"\n{'Epoch':>6}  {'Train Loss':>10}  {'Train Acc':>9}  {'Val Loss':>9}  {'Val Acc':>8}")
    print("-" * 52)

    best_val_loss = float("inf")
    save_path = resolve_path(args.save)
    t0 = time.time()
    for epoch in range(1, args.epochs + 1):
        tr_loss, tr_acc = run_epoch(model, train_loader, criterion, tok, device,
                                    optimizer=optimizer, grad_clip=args.grad_clip,
                                    log_every=args.log_every, log_file=args.log_file,
                                    epoch=epoch, t0=t0)
        with torch.no_grad():
            va_loss, va_acc = run_epoch(model, val_loader, criterion, tok, device)

        marker = ""
        if va_loss < best_val_loss:
            best_val_loss = va_loss
            marker = "  *"
            save_checkpoint(save_path, model, optimizer, epoch, best_val_loss, tok, args)

        print(f"{epoch:>6}  {tr_loss:>10.4f}  {tr_acc:>9.4f}  "
              f"{va_loss:>9.4f}  {va_acc:>8.4f}{marker}")

        if preview_rows:
            show_preview(model, tok, preview_rows, gen_kwargs, device, tag=f"ep{epoch} ")

    print(f"\n训练完成。最佳验证 loss: {best_val_loss:.4f}  已保存至 {save_path}  "
          f"（用时 {(time.time() - t0) / 60:.1f} 分钟）")

    # 用最优 checkpoint 再生成一遍（训练末轮未必是最优那轮）
    if preview_rows and os.path.exists(save_path):
        best = torch.load(save_path, map_location=device)
        model.load_state_dict(best["model_state"])
        print(f"\n最佳 checkpoint（第 {best['epoch']} 轮）的生成结果:")
        show_preview(model, tok, preview_rows, gen_kwargs, device)
        print(f"\n继续用: python language_model_sft.py --chat --ckpt {args.save}")


# ─────────────────────────── 推理模式 ───────────────────────────

def answer_once(model, tok, device, question, gen_kwargs):
    ans = generate(model, tok, question, device=device, **gen_kwargs)
    print(f"用户：{question}")
    print(f"助手：{ans if ans else '（空回答）'}")


def chat(model, tok, device, gen_kwargs):
    """交互式对话。每一轮都是独立的一问一答（模板只有单轮）。"""
    print("\n" + "=" * 52)
    print("  SFT 对话（字符级 Transformer）  q / exit 退出")
    print("=" * 52)
    while True:
        try:
            q = input("\n用户> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n退出。")
            break
        if not q:
            continue
        if q.lower() in ("q", "quit", "exit"):
            print("退出。")
            break
        answer_once(model, tok, device, q, gen_kwargs)


def evaluate(args, model, tok, device):
    """只算验证集 loss / token 准确率，用来对比不同 checkpoint。"""
    val_rows = read_jsonl(resolve_path(args.val))
    ds = SFTDataset(val_rows, tok, max_len=args.max_len)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False,
                        collate_fn=make_collate_fn(tok.pad_id))
    criterion = nn.CrossEntropyLoss(ignore_index=IGNORE_INDEX)
    with torch.no_grad():
        loss, acc = run_epoch(model, loader, criterion, tok, device)
    print(f"验证集: loss {loss:.4f}  token 准确率 {acc:.4f}  "
          f"({len(ds)} 条, 超长跳过 {ds.n_too_long} 条)")


# ─────────────────────────── 主函数 ───────────────────────────

def main():
    parser = argparse.ArgumentParser(description="字符级 Transformer 语言模型的 SFT 指令微调")
    # ── 路径 ──
    parser.add_argument("--ckpt", default=os.path.join("循环神经网络语言模型", "best_model_transformer.pt"),
                        help="训练时是上周的预训练 checkpoint；--chat/--prompt/--eval 时换成 SFT checkpoint")
    parser.add_argument("--train", default=os.path.join("sft_train", "sft_train.jsonl"))
    parser.add_argument("--val", default=os.path.join("sft_train", "sft_val.jsonl"))
    parser.add_argument("--save", default="best_model_sft.pt", help="最优 checkpoint 保存路径")
    # ── 训练超参 ──
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3,
                        help="这个 1.4M 的玩具模型严重欠拟合，1e-3 收敛最快；"
                             "真实大模型 SFT 一般用 1e-5~1e-4")
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--grad_clip", type=float, default=1.0, help="梯度裁剪阈值，0=关闭")
    parser.add_argument("--max_len", type=int, default=128,
                        help="prompt+回答 的最大字符数，超过的样本跳过")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--log_every", type=int, default=0, help="每 N 个 batch 打印一次进度，0=关闭")
    parser.add_argument("--log_file", default="", help="进度同时追加写入该文件")
    parser.add_argument("--preview", type=int, default=3,
                        help="训练前 / 每轮后打印几条生成样例，0=关闭")
    # ── 生成超参 ──
    parser.add_argument("--max_new_tokens", type=int, default=64)
    parser.add_argument("--top_k", type=int, default=5, help="采样候选数，0=不限制")
    parser.add_argument("--top_p", type=float, default=1.0,
                        help="核采样阈值（按累积概率切长尾），1.0=关闭")
    parser.add_argument("--temperature", type=float, default=0.8, help="0=贪心解码")
    # ── 仅 --scratch 生效的结构超参 ──
    parser.add_argument("--d_model", type=int, default=128, help="仅 --scratch 生效")
    parser.add_argument("--nhead", type=int, default=4, help="仅 --scratch 生效")
    parser.add_argument("--num_layers", type=int, default=2, help="仅 --scratch 生效")
    parser.add_argument("--ffn_dim", type=int, default=512, help="仅 --scratch 生效")
    # ── 模式 ──
    parser.add_argument("--self_test", action="store_true", help="只跑数据管线自检，不训练")
    parser.add_argument("--eval", action="store_true", help="只评估 --ckpt 的验证集指标")
    parser.add_argument("--chat", action="store_true", help="交互式对话")
    parser.add_argument("--prompt", default="", help="单条问题，生成回答")
    parser.add_argument("--scratch", action="store_true", help="对照实验：不加载预训练权重")
    args = parser.parse_args()

    if args.self_test:
        self_test()
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    gen_kwargs = dict(max_new_tokens=args.max_new_tokens, top_k=args.top_k,
                      top_p=args.top_p, temperature=args.temperature)

    if args.chat or args.prompt or args.eval:
        model, tok, ckpt = load_sft_checkpoint(resolve_path(args.ckpt), device)
        print(f"模型: {args.ckpt}  ({ckpt['args']['num_layers']} 层, "
              f"d_model={ckpt['args']['d_model']}, 词表 {len(tok)} 字, "
              f"训练 {ckpt['epoch']} 轮)")
        if args.eval:
            evaluate(args, model, tok, device)
            return
        if args.prompt:
            answer_once(model, tok, device, args.prompt, gen_kwargs)
        if args.chat:
            chat(model, tok, device, gen_kwargs)
        return

    train(args)


if __name__ == "__main__":
    main()
