import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
import argparse
import glob
import json
import math
import random
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset


HERE = os.path.dirname(os.path.abspath(__file__))
# ===================== 统一输出路径 =====================
SAVE_DIR = "D:/AI/week5"
PRETRAIN_CKPT = f"{SAVE_DIR}/lm_pretrain.pt"
SFT_CKPT = f"{SAVE_DIR}/lm_sft_best.pt"


# ─────────────────────── 数据 ───────────────────────
def load_corpus(pattern):
    texts = []
    for path in sorted(glob.glob(pattern)):
        with open(path, encoding="utf-8", errors="ignore") as f:
            texts.append(f.read())
    full_text = "".join(texts)
    if not full_text:
        raise FileNotFoundError(f"没读到语料: {pattern}")
    return full_text


def build_vocab(text, max_vocab):
    from collections import Counter
    chars = [c for c, _ in Counter(text).most_common(max_vocab)] if max_vocab else sorted(set(text))
    char2idx = {"<pad>": 0}
    for idx, c in enumerate(chars, start=1):
        char2idx[c] = idx
    idx2char = {i: c for c, i in char2idx.items()}
    return char2idx, idx2char


class CharDataset(Dataset):
    def __init__(self, text, char2idx, seq_len, stride):
        ids = [char2idx[c] for c in text if c in char2idx]
        self.data = torch.tensor(ids, dtype=torch.long)
        self.seq_len = seq_len
        self.stride = max(1, stride)

    def __len__(self):
        return max(0, (len(self.data) - self.seq_len - 1) // self.stride + 1)

    def __getitem__(self, i):
        s = i * self.stride
        x = self.data[s:s + self.seq_len]
        y = self.data[s + 1:s + self.seq_len + 1]
        return x, y


class SftInstructionDataset(Dataset):
    """
    SFT数据集：每条问答仅生成1条样本，不做滑动窗口；
    labels mask：「指令：...\\n回答：」部分=-100不计算loss，仅回答部分算loss
    修复：短句padding、单条最大512token截断，移除内部滑动窗口
    """
    def __init__(self, jsonl_path, char2idx, seq_len, stride):
        self.window_list = []   # 存放样本 (win_x, win_label)
        self.seq_len = seq_len
        self.stride = max(1, stride)
        self.char2idx = char2idx
        MAX_ONE_ITEM = 512

        with open(jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                item = json.loads(line)
                inst = item["instruction"]
                out = item["output"]
                prompt_part = f"指令：{inst}\n回答："

                prompt_ids = [char2idx[c] for c in prompt_part if c in char2idx]
                output_ids = [char2idx[c] for c in out if c in char2idx]
                full_ids = prompt_ids + output_ids
                full_labels = [-100]*len(prompt_ids) + output_ids

                # 单条问答最大512token截断，防止超长
                if len(full_ids) > MAX_ONE_ITEM:
                    full_ids = full_ids[:MAX_ONE_ITEM]
                    full_labels = full_labels[:MAX_ONE_ITEM]

                total_len = len(full_ids)
                need_len = self.seq_len + 1

                # 短句padding补齐到 need_len（65）
                if total_len < need_len:
                    pad_num = need_len - total_len
                    pad_id = self.char2idx["<pad>"]
                    full_ids += [pad_id] * pad_num
                    full_labels += [-100] * pad_num
                    total_len = len(full_ids)

                # --------不再滑动，只取从0开始的第一个窗口--------
                win_in = full_ids[0 : 0 + self.seq_len + 1]
                win_lab = full_labels[0 : 0 + self.seq_len + 1]
                x_win = torch.tensor(win_in[:-1], dtype=torch.long)
                y_win = torch.tensor(win_lab[1:], dtype=torch.long)
                self.window_list.append((x_win, y_win))

    def __len__(self):
        return len(self.window_list)

    def __getitem__(self, i):
        return self.window_list[i]



# ─────────────────────── 模型 ───────────────────────
class TransformerLM(nn.Module):
    def __init__(self, vocab_size, d_model=192, n_head=4, n_layer=4, max_len=512,
                 dropout=0.1, use_pos=True, causal=True):
        super().__init__()
        self.causal = causal
        self.use_pos = use_pos
        self.tok = nn.Embedding(vocab_size, d_model)
        self.pos = nn.Embedding(max_len, d_model) if use_pos else None
        self.drop = nn.Dropout(dropout)

        layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_head, dim_feedforward=4 * d_model,
            dropout=dropout, activation="gelu", batch_first=True, norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(
            layer, num_layers=n_layer, norm=nn.LayerNorm(d_model),
            enable_nested_tensor=False,
        )
        self.head = nn.Linear(d_model, vocab_size, bias=False)
        self.head.weight = self.tok.weight

        nn.init.normal_(self.tok.weight, std=0.02)
        if self.pos is not None:
            nn.init.normal_(self.pos.weight, std=0.02)

    def forward(self, idx):
        # idx: [B, T]
        B, T = idx.shape
        x = self.tok(idx)
        if self.use_pos:
            pos_ids = torch.arange(T, device=idx.device)
            x = x + self.pos(pos_ids)
        x = self.drop(x)

        mask = None
        if self.causal:
            mask = nn.Transformer.generate_square_subsequent_mask(T, device=idx.device)
        x = self.encoder(x, mask=mask)
        return self.head(x)


# ─────────────────────── 评估、训练、生成 ───────────────────────
@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    total, tokens = 0.0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)),
                                y.reshape(-1), reduction="sum", ignore_index=-100)
        total += loss.item()   # 修复警告 float(loss) → loss.item()
        valid_mask = (y != -100)
        tokens += int(valid_mask.sum())
    loss = total / tokens if tokens > 0 else 0.0
    return loss, math.exp(loss)


def train(args):
    os.makedirs(SAVE_DIR, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(args.seed)
    random.seed(args.seed)

    char2idx, idx2char = None, None
    tr, va = None, None
    pretrain_ck = None

    if args.sft_jsonl_train is not None:
        print("===== SFT问答微调模式【Loss Mask，仅回答算loss】 =====")
        sft_text_buf = []
        with open(args.sft_jsonl_train, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                d = json.loads(line)
                sft_text_buf.append(f"指令：{d['instruction']}\n回答：{d['output']}")
        with open(args.sft_jsonl_val, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                d = json.loads(line)
                sft_text_buf.append(f"指令：{d['instruction']}\n回答：{d['output']}")
        sft_full_text = "\n".join(sft_text_buf)

        if args.pretrain_ckpt is not None:
            print(f"加载预训练权重: {args.pretrain_ckpt}")
            pretrain_ck = torch.load(args.pretrain_ckpt, map_location=device)
            old_char2idx = pretrain_ck["char2idx"]

            sft_new_chars = set(sft_full_text) - set(old_char2idx.keys())
            add_char_list = sorted(list(sft_new_chars))
            print(f"预训练词表大小：{len(old_char2idx)}，SFT新增未登录字符数量：{len(add_char_list)}")

            char2idx = dict(old_char2idx)
            next_id = len(char2idx)
            for c in add_char_list:
                char2idx[c] = next_id
                next_id += 1
            idx2char = {i: c for c, i in char2idx.items()}
        else:
            print("【警告】未提供预训练权重，从零开始SFT训练！模型缺少通用知识")
            char2idx, idx2char = build_vocab(sft_full_text, args.max_vocab)

        V = len(char2idx)
        tr = SftInstructionDataset(args.sft_jsonl_train, char2idx, args.seq_len, args.stride)
        va = SftInstructionDataset(args.sft_jsonl_val, char2idx, args.seq_len, args.stride)
    else:
        print("===== 普通字符语言模型模式 =====")
        text = load_corpus(args.corpus)
        if args.max_chars and len(text) > args.max_chars:
            text = text[:args.max_chars]
        char2idx, idx2char = build_vocab(text, args.max_vocab)
        V = len(char2idx)
        lines = [line for line in text.splitlines() if line.strip()]
        random.shuffle(lines)
        split = int(len(lines) * (1 - args.val_ratio))
        tr = CharDataset("\n".join(lines[:split]), char2idx, args.seq_len, args.stride)
        va = CharDataset("\n".join(lines[split:]), char2idx, args.seq_len, args.stride)

    print(f"device: {device}  最终词表大小 {V}")

    train_loader = DataLoader(tr, batch_size=args.batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(va, batch_size=args.batch_size, shuffle=False, drop_last=True)

    # =========修复：关键字传参 max_len=512 =========
    model = TransformerLM(
        vocab_size=V,
        d_model=args.d_model,
        n_head=args.heads,
        n_layer=args.layers,
        max_len=512,
        dropout=args.dropout,
        use_pos=not args.no_pos,
        causal=not args.no_mask
    ).to(device)

    if pretrain_ck is not None:
        old_vocab_size = len(pretrain_ck["char2idx"])
        pretrain_state = pretrain_ck["model_state"]
        new_state = model.state_dict()

        new_state["tok.weight"][:old_vocab_size, :] = pretrain_state["tok.weight"][:old_vocab_size, :]

        # --------位置编码权重扩容--------
        if "pos.weight" in pretrain_state:
            old_pos_weight = pretrain_state["pos.weight"]
            new_pos_weight = new_state["pos.weight"]
            new_pos_weight[:old_pos_weight.shape[0], :] = old_pos_weight
            nn.init.normal_(new_pos_weight[old_pos_weight.shape[0]:, :], std=0.02)
            new_state["pos.weight"] = new_pos_weight

        for key in pretrain_state.keys():
            if key not in ("tok.weight", "head.weight", "pos.weight"):
                new_state[key] = pretrain_state[key]

        model.load_state_dict(new_state)
        print("已加载预训练权重；SFT新增字符embedding、位置编码新增部分使用随机初始化")

    n_par = sum(p.numel() for p in model.parameters())
    print(f"参数量 {n_par/1e6:.2f}M  位置编码={'有' if not args.no_pos else '无'}  因果掩码={'有' if not args.no_mask else '无'}")

    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    best = float("inf")

    print(f"\n{'Epoch':>6} {'TrainLoss':>10} {'ValPPL':>10} {'tok/s':>9}")
    print("-" * 50)

    for ep in range(1, args.epochs + 1):
        model.train()
        st, seen, tot = time.time(), 0, 0.0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            loss = F.cross_entropy(logits.reshape(-1, V), y.reshape(-1), ignore_index=-100)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

            valid_mask = (y != -100)
            valid_cnt = int(valid_mask.sum())
            tot += loss.item() * valid_cnt
            seen += valid_cnt

        dt = time.time() - st
        v_loss, v_ppl = evaluate(model, val_loader, device)

        mark = ""
        if v_ppl < best:
            best = v_ppl
            mark = " *"
            torch.save({
                "model_state": model.state_dict(),
                "char2idx": char2idx,
                "idx2char": idx2char,
                "args": vars(args)
            }, args.save)

        train_ppl = math.exp(tot / seen) if seen > 0 else float("inf")
        print(f"{ep:>6} {train_ppl:>10.4f} {v_ppl:>10.2f} {seen/dt:>9,.0f}{mark}")

    print(f"\n最佳验证PPL: {best:.2f}  已保存 {args.save}")


@torch.no_grad()
def generate(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ck = torch.load(args.ckpt, map_location=device)
    c = ck["args"]
    char2idx, idx2char = ck["char2idx"], ck["idx2char"]

    # =========修复推理重建模型 max_len=512=========
    model = TransformerLM(
        vocab_size=len(char2idx),
        d_model=c["d_model"],
        n_head=c["heads"],
        n_layer=c["layers"],
        max_len=512,
        dropout=0.0,
        use_pos=not c.get("no_pos", False),
        causal=not c.get("no_mask", False)
    ).to(device)

    model.load_state_dict(ck["model_state"])
    model.eval()

    ids = [char2idx[ch] for ch in args.generate if ch in char2idx]
    out = list(ids)
    for _ in range(args.n_chars):
        feed = torch.tensor([out[-args.seq_len:]], device=device)
        logits = model(feed)[0, -1]
        if args.temperature <= 0.01:
            nxt = int(logits.argmax())
        else:
            logits = logits / args.temperature
            top_p = 0.9
            sorted_probs, sorted_indices = torch.sort(torch.softmax(logits, dim=-1), descending=True)
            cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
            remove_mask = cumulative_probs > top_p
            remove_mask[1:] = remove_mask[:-1].clone()
            remove_mask[0] = False
            logits[sorted_indices[remove_mask]] = -float("inf")

            prob = F.softmax(logits, dim=-1)
            nxt = int(torch.multinomial(prob, 1))
#         else:
#             prob = F.softmax(logits / args.temperature, dim=-1)
#             nxt = int(torch.multinomial(prob, 1))
        out.append(nxt)

    print(f"输入:\n{args.generate}")
    print(f"续写:\n{''.join(idx2char[i] for i in out[len(ids):])}")


def main():
    ap = argparse.ArgumentParser()
    default_corpus = os.path.join(HERE, "corpus_sample.txt")
    ap.add_argument("--corpus", default=default_corpus)
    ap.add_argument("--max_chars", type=int, default=0)
    ap.add_argument("--max_vocab", type=int, default=4000)
    ap.add_argument("--seq_len", type=int, default=64)
    ap.add_argument("--stride", type=int, default=64)
    ap.add_argument("--batch_size", type=int, default=128)
    ap.add_argument("--epochs", type=int, default=6)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--d_model", type=int, default=192)
    ap.add_argument("--heads", type=int, default=4)
    ap.add_argument("--layers", type=int, default=4)
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--val_ratio", type=float, default=0.05)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--no-mask", dest="no_mask", action="store_true")
    ap.add_argument("--no-pos", dest="no_pos", action="store_true")
    ap.add_argument("--save", default=PRETRAIN_CKPT)

    ap.add_argument("--sft_jsonl_train", default=None, help="SFT训练集jsonl路径")
    ap.add_argument("--sft_jsonl_val", default=None, help="SFT验证集jsonl路径")
    ap.add_argument("--pretrain_ckpt", default=None, help="【两阶段SFT】预训练LM的ckpt路径")

    ap.add_argument("--generate", default=None)
    ap.add_argument("--ckpt", default=SFT_CKPT)
    ap.add_argument("--n_chars", type=int, default=80)
    ap.add_argument("--temperature", type=float, default=0.7)

    args = ap.parse_args()
    if args.generate:
        generate(args)
    else:
        train(args)


if __name__ == "__main__":
    main()

"""
预训练
python transformer_lm_sft.py --corpus corpus_general.txt --seq_len 64 --stride 64 --batch_size 64 --epochs 6 --lr 1e-3 --d_model 192 --heads 4 --layers 4 --save "lm_pretrain.pt"

SFT微调
python transformer_lm_sft.py --sft_jsonl_train sft_train.jsonl --sft_jsonl_val sft_val.jsonl --pretrain_ckpt "lm_pretrain.pt" --seq_len 64 --stride 64 --batch_size 64 --epochs 6 --lr 5e-4 --d_model 192 --heads 4 --layers 4 --save "lm_sft_best.pt"

推理
python transformer_lm_sft.py --generate "指令：你好\n回答：" --ckpt "lm_sft_best.pt" --n_chars 10 --temperature 0.7
"""