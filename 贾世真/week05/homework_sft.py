"""
第五周作业：基于已有语言模型进行 SFT（监督微调）。

复用第四周的字符级 Decoder-only Transformer 和预训练权重，只改变：
1. 数据：连续语料滑窗 -> 一条完整指令问答；
2. 模板：<user>问题<assistant>回答<end>；
3. 标签：问题与 PAD 置 -100，只让回答和 EOS 产生损失；
4. 生成：遇到 <end> 停止，而不是遇到换行停止。

训练：
    python3 homework_sft.py --base_model base_model.pt --data sft_data.jsonl --epochs 30
查看一条样本的 input/target 对齐：
    python3 homework_sft.py --inspect
使用微调后的模型回答：
    python3 homework_sft.py --save sft_model.pt --prompt "什么是过拟合？"

说明：小数据作业用于验证 SFT 的数据、Loss Mask、EOS 和训练流程，不用于获得通用问答能力。
"""

import argparse
import json
import math
import random
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

PAD, USER, ASSISTANT, END, UNK = "<pad>", "<user>", "<assistant>", "<end>", "<unk>"
SPECIAL_TOKENS = [PAD, USER, ASSISTANT, END, UNK]
IGNORE_INDEX = -100


def load_checkpoint(path, device="cpu"):
    """兼容 PyTorch 2.6 的 weights_only 默认值与旧版本。"""
    try:
        return torch.load(path, map_location=device, weights_only=True)
    except TypeError:
        return torch.load(path, map_location=device)


class ChatTokenizer:
    """原字表 + 多字符特殊标记；特殊标记必须优先于单字切分。"""

    def __init__(self, token2idx):
        self.token2idx = dict(token2idx)
        self.idx2token = {i: t for t, i in self.token2idx.items()}
        for token in SPECIAL_TOKENS:
            if token not in self.token2idx:
                raise ValueError(f"词表缺少特殊标记: {token}")

    @classmethod
    def from_base_vocab(cls, char2idx):
        token2idx = dict(char2idx)
        for token in SPECIAL_TOKENS:
            if token not in token2idx:
                token2idx[token] = len(token2idx)
        return cls(token2idx)

    def encode(self, text):
        ids, i = [], 0
        while i < len(text):
            for token in SPECIAL_TOKENS:
                if text.startswith(token, i):
                    ids.append(self.token2idx[token])
                    i += len(token)
                    break
            else:
                ids.append(self.token2idx.get(text[i], self.token2idx[UNK]))
                i += 1
        return ids

    def decode(self, ids, skip_template=True):
        chars = []
        for i in ids:
            token = self.idx2token[int(i)]
            if token in (PAD, END):
                continue
            if skip_template and token in (USER, ASSISTANT):
                continue
            chars.append("□" if token == UNK else token)
        return "".join(chars)


# ─────────────────────────── 第四周模型：仅增加 key padding mask ───────────────────────────

class CausalSelfAttention(nn.Module):
    def __init__(self, d_model, n_head, dropout=0.1):
        super().__init__()
        assert d_model % n_head == 0
        self.n_head = n_head
        self.d_head = d_model // n_head
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.proj = nn.Linear(d_model, d_model)
        self.drop = nn.Dropout(dropout)

    def forward(self, x, attention_mask=None):
        B, T, C = x.shape
        q, k, v = self.qkv(x).chunk(3, dim=-1)
        q = q.view(B, T, self.n_head, self.d_head).transpose(1, 2)
        k = k.view(B, T, self.n_head, self.d_head).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.d_head).transpose(1, 2)

        scores = (q @ k.transpose(-2, -1)) / math.sqrt(self.d_head)
        causal = torch.triu(torch.ones(T, T, dtype=torch.bool, device=x.device), diagonal=1)
        scores = scores.masked_fill(causal, float("-inf"))
        if attention_mask is not None:
            # 只屏蔽 PAD key，不屏蔽 PAD query，避免整行 -inf 导致 softmax NaN。
            key_pad = attention_mask[:, None, None, :] == 0
            scores = scores.masked_fill(key_pad, float("-inf"))
        weights = self.drop(F.softmax(scores, dim=-1))
        y = weights @ v
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.proj(self.drop(y))


class Block(nn.Module):
    def __init__(self, d_model, n_head, dropout=0.1):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = CausalSelfAttention(d_model, n_head, dropout)
        self.ln2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, 4 * d_model), nn.GELU(),
            nn.Linear(4 * d_model, d_model), nn.Dropout(dropout),
        )

    def forward(self, x, attention_mask=None):
        x = x + self.attn(self.ln1(x), attention_mask)
        x = x + self.ffn(self.ln2(x))
        return x


class TransformerLM(nn.Module):
    def __init__(self, vocab_size, d_model=192, n_head=4, n_layer=4,
                 seq_len=64, dropout=0.1):
        super().__init__()
        self.tok_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(seq_len, d_model)
        self.drop = nn.Dropout(dropout)
        self.blocks = nn.ModuleList([Block(d_model, n_head, dropout) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size, bias=False)
        self.head.weight = self.tok_emb.weight  # weight tying
        self.seq_len = seq_len
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.LayerNorm):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)

    def forward(self, x, attention_mask=None):
        T = x.shape[1]
        if T > self.seq_len:
            raise ValueError(f"输入长度 {T} 超过位置表长度 {self.seq_len}")
        pos = torch.arange(T, device=x.device)
        h = self.drop(self.tok_emb(x) + self.pos_emb(pos))
        for block in self.blocks:
            h = block(h, attention_mask)
        return self.head(self.ln_f(h))


def model_from_base(checkpoint, tokenizer, device):
    """先建立扩词表后的模型，再逐块复制旧权重；新特殊标记保留 std=0.02 初始化。"""
    cfg = checkpoint["args"]
    model = TransformerLM(
        vocab_size=len(tokenizer.token2idx), d_model=cfg["d_model"],
        n_head=cfg["n_head"], n_layer=cfg["n_layer"],
        seq_len=cfg["seq_len"], dropout=0.0,
    ).to(device)
    state = model.state_dict()
    old_state = checkpoint["model_state"]
    old_vocab = old_state["tok_emb.weight"].shape[0]
    for name, value in old_state.items():
        if name in ("tok_emb.weight", "head.weight"):
            state[name][:old_vocab].copy_(value)
        else:
            state[name].copy_(value)
    model.load_state_dict(state)
    return model


# ─────────────────────────── SFT 数据与 Loss Mask ───────────────────────────

def format_prompt(instruction):
    return f"{USER}{instruction}{ASSISTANT}"


def read_jsonl(path):
    records = []
    with open(path, encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not row.get("instruction") or not row.get("response"):
                raise ValueError(f"{path}:{line_no} 需要非空 instruction 和 response")
            records.append({"instruction": row["instruction"], "response": row["response"]})
    if not records:
        raise ValueError("SFT 数据为空")
    return records


def build_example(record, tokenizer, seq_len):
    """
    full = prompt + response + EOS；模型输入和目标错一位。
    关键：只 mask 前 len(prompt)-1 个 target，最后 prompt token 仍负责预测第一个回答 token。
    """
    prompt_ids = tokenizer.encode(format_prompt(record["instruction"]))
    answer_ids = tokenizer.encode(record["response"]) + [tokenizer.token2idx[END]]
    if len(prompt_ids) >= seq_len:
        raise ValueError(f"问题过长（{len(prompt_ids)} tokens），无法保留回答和 EOS")

    # 截断只砍回答尾部，并强制保留 EOS；绝不截掉回答开头或整条丢弃。
    available = seq_len + 1 - len(prompt_ids)
    if len(answer_ids) > available:
        answer_ids = answer_ids[:available - 1] + [tokenizer.token2idx[END]]
    full_ids = prompt_ids + answer_ids
    x = full_ids[:-1]
    y = full_ids[1:]
    y[:len(prompt_ids) - 1] = [IGNORE_INDEX] * (len(prompt_ids) - 1)
    assert len(x) == len(y) <= seq_len
    assert any(label != IGNORE_INDEX for label in y), "不能出现全 -100 样本"
    assert tokenizer.token2idx[END] in y, "目标必须包含 EOS"
    return torch.tensor(x), torch.tensor(y)


class SFTDataset(Dataset):
    def __init__(self, records, tokenizer, seq_len):
        self.examples = [build_example(row, tokenizer, seq_len) for row in records]

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        return self.examples[idx]


def collate_sft(examples, pad_id):
    """右 padding：输入补 PAD，标签补 -100，attention_mask 标记真实 key。"""
    max_len = max(len(x) for x, _ in examples)
    xs = torch.full((len(examples), max_len), pad_id, dtype=torch.long)
    ys = torch.full_like(xs, IGNORE_INDEX)
    mask = torch.zeros_like(xs)
    for i, (x, y) in enumerate(examples):
        xs[i, :len(x)] = x
        ys[i, :len(y)] = y
        mask[i, :len(x)] = 1
    return xs, ys, mask


def run_epoch(model, loader, device, optimizer=None, scheduler=None):
    training = optimizer is not None
    model.train(training)
    total_loss, total_tokens = 0.0, 0
    for x, y, attention_mask in loader:
        x, y, attention_mask = x.to(device), y.to(device), attention_mask.to(device)
        valid_tokens = (y != IGNORE_INDEX).sum()
        if valid_tokens.item() == 0:
            raise ValueError("batch 没有有效回答 token，请检查 Loss Mask")
        logits = model(x, attention_mask)
        # sum 用于正确累计不同 batch 的有效 token；反向仍使用每个有效 token 的平均损失。
        loss_sum = F.cross_entropy(
            logits.reshape(-1, logits.size(-1)), y.reshape(-1),
            ignore_index=IGNORE_INDEX, reduction="sum",
        )
        if training:
            optimizer.zero_grad()
            (loss_sum / valid_tokens).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            if scheduler is not None:
                scheduler.step()
        total_loss += loss_sum.item()
        total_tokens += valid_tokens.item()
    avg_loss = total_loss / total_tokens
    return avg_loss, math.exp(min(avg_loss, 20.0)), total_tokens


# ─────────────────────────── 生成与检查 ───────────────────────────

@torch.no_grad()
def generate_answer(model, tokenizer, instruction, device, max_new_tokens=80,
                    temperature=0.0, top_k=0):
    model.eval()
    prompt_ids = tokenizer.encode(format_prompt(instruction))
    ids = list(prompt_ids)
    for _ in range(max_new_tokens):
        context = ids[-model.seq_len:]
        x = torch.tensor([context], dtype=torch.long, device=device)
        mask = torch.ones_like(x)
        logits = model(x, mask)[0, -1]
        if temperature <= 0:
            next_id = logits.argmax().item()
        else:
            logits = logits / temperature
            if top_k > 0:
                values, _ = torch.topk(logits, min(top_k, logits.numel()))
                logits[logits < values[-1]] = float("-inf")
            next_id = torch.multinomial(F.softmax(logits, dim=-1), 1).item()
        if next_id == tokenizer.token2idx[END]:
            break
        ids.append(next_id)
    return tokenizer.decode(ids[len(prompt_ids):])


def inspect_example(record, tokenizer, seq_len):
    x, y = build_example(record, tokenizer, seq_len)
    print("模板:", format_prompt(record["instruction"]) + record["response"] + END)
    print("位置 | 输入 token -> 目标 token | 是否计算 Loss")
    for i, (x_id, y_id) in enumerate(zip(x.tolist(), y.tolist())):
        target = "PAD(-100)" if y_id == IGNORE_INDEX else tokenizer.idx2token[y_id]
        print(f"{i:>4} | {tokenizer.idx2token[x_id]} -> {target} | {'否' if y_id == IGNORE_INDEX else '是'}")
    print(f"有效回答 token 数: {(y != IGNORE_INDEX).sum().item()}；EOS 已在目标中。")


def save_model(path, model, tokenizer, args):
    torch.save({
        "model_state": model.state_dict(),
        "token2idx": tokenizer.token2idx,
        "idx2token": tokenizer.idx2token,
        "model_config": {
            "vocab_size": len(tokenizer.token2idx),
            "d_model": model.tok_emb.embedding_dim,
            "n_head": model.blocks[0].attn.n_head,
            "n_layer": len(model.blocks),
            "seq_len": model.seq_len,
        },
        "args": vars(args),
    }, path)


def load_sft_model(path, device):
    checkpoint = load_checkpoint(path, device)
    tokenizer = ChatTokenizer(checkpoint["token2idx"])
    model = TransformerLM(**checkpoint["model_config"], dropout=0.0).to(device)
    model.load_state_dict(checkpoint["model_state"])
    return model, tokenizer


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--base_model", default="base_model.pt", help="第四周预训练权重")
    parser.add_argument("--data", default="sft_data.jsonl")
    parser.add_argument("--save", default="sft_model.pt")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--val_ratio", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max_new_tokens", type=int, default=80)
    parser.add_argument("--prompt", default=None, help="设置后只加载 SFT 模型回答，不训练")
    parser.add_argument("--inspect", action="store_true", help="只展示第一条样本的 token 对齐，不训练")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if args.prompt is not None:
        model, tokenizer = load_sft_model(args.save, device)
        print(generate_answer(model, tokenizer, args.prompt, device, args.max_new_tokens))
        return

    base_path = Path(args.base_model)
    data_path = Path(args.data)
    if not base_path.exists() or not data_path.exists():
        raise FileNotFoundError("训练需要 base_model.pt 和 sft_data.jsonl；可用 --help 查看用法")
    base = load_checkpoint(base_path, device)
    tokenizer = ChatTokenizer.from_base_vocab(base["char2idx"])
    records = read_jsonl(data_path)
    if args.inspect:
        inspect_example(records[0], tokenizer, base["args"]["seq_len"])
        return

    random.shuffle(records)
    val_count = max(1, round(len(records) * args.val_ratio))
    val_count = min(val_count, len(records) - 1)
    val_records, train_records = records[:val_count], records[val_count:]
    seq_len = base["args"]["seq_len"]
    train_ds = SFTDataset(train_records, tokenizer, seq_len)
    val_ds = SFTDataset(val_records, tokenizer, seq_len)
    collate = lambda batch: collate_sft(batch, tokenizer.token2idx[PAD])
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, collate_fn=collate)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, drop_last=False, collate_fn=collate)

    model = model_from_base(base, tokenizer, device)
    print(f"device={device}；预训练词表={len(base['char2idx'])}，SFT 词表={len(tokenizer.token2idx)}")
    print(f"训练问答={len(train_ds)}，验证问答={len(val_ds)}，序列上限={seq_len}")
    print(f"可训练参数={sum(p.numel() for p in model.parameters()):,}")

    demo_questions = ["什么是过拟合？", "交叉熵有什么作用？", "什么是因果掩码？"]
    print("\n[SFT 前：同一 chat template]")
    for question in demo_questions:
        print(f"问：{question}\n答：{generate_answer(model, tokenizer, question, device, args.max_new_tokens)}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    total_steps = max(1, len(train_loader) * args.epochs)
    warmup_steps = min(args.warmup, max(1, total_steps // 5))

    def lr_lambda(step):
        if step < warmup_steps:
            return (step + 1) / warmup_steps
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1 + math.cos(math.pi * min(1.0, progress)))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    best_val = float("inf")
    save_path = Path(args.save)
    best_path = save_path.with_name("best_" + save_path.name)
    print(f"\n{'Epoch':>5} {'Train Loss':>11} {'Val Loss':>9} {'Val PPL':>9} {'回答tokens':>10}")
    for epoch in range(1, args.epochs + 1):
        train_loss, _, train_tokens = run_epoch(model, train_loader, device, optimizer, scheduler)
        with torch.no_grad():
            val_loss, val_ppl, _ = run_epoch(model, val_loader, device)
        marker = ""
        if val_loss < best_val:
            best_val = val_loss
            save_model(best_path, model, tokenizer, args)
            marker = " *"
        print(f"{epoch:>5} {train_loss:>11.4f} {val_loss:>9.4f} {val_ppl:>9.2f} {train_tokens:>10}{marker}")

    # 同时保存最后一轮与最佳验证 Loss 两轮检查点：二者回答行为可能差异很大。
    # 小数据作业默认用最终模型观察格式学习，不能用最低 Val Loss 代替生成质量评估。
    save_model(save_path, model, tokenizer, args)
    model, tokenizer = load_sft_model(save_path, device)
    print(f"\n训练完成，最佳验证回答 Loss={best_val:.4f}")
    print(f"最终模型：{save_path}；最佳验证模型：{best_path}")
    print("\n[SFT 后：同一 chat template；观察回答格式与 EOS 停止，不以小数据证明通用能力]")
    for question in demo_questions:
        print(f"问：{question}\n答：{generate_answer(model, tokenizer, question, device, args.max_new_tokens)}")


if __name__ == "__main__":
    main()
