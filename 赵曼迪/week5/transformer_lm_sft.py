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
# 全局上下文长度
CTX_LEN = 128


# ─────────────────────── 数据 ───────────────────────
def load_corpus(pattern):
    texts = []
    for path in sorted(glob.glob(pattern)):
        with open(path, encoding="utf-8", errors="ignore") as f:
            texts.append(f.read())#提取当前运行终端的符合传参的文件，将他们阅读出来存在texts列表里
    full_text = "".join(texts)#把列表变成一个长字符串，带"\n"
    if not full_text:
        raise FileNotFoundError(f"没读到语料: {pattern}")
    return full_text


def build_vocab(text, max_vocab):
    from collections import Counter
    chars = [c for c, _ in Counter(text).most_common(max_vocab)] if max_vocab else sorted(set(text))
    char2idx = {"<pad>": 0, "<eos>": 1}#新增补0padding字符,结束字符：<eos>，防止无限续写
    for idx, c in enumerate(chars, start=2):
        char2idx[c] = idx
    idx2char = {i: c for c, i in char2idx.items()}
    return char2idx, idx2char


class CharDataset(Dataset):
    """
    结合234~244行的用法：
    把全部文本按训练集/验证机拼成一整条超长字符串，然后用以seq_len为一组长度，stride为步长滑动窗口不断切窗口，得到一组一组 (x,y)
    """
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
    短句增加padding：0补齐128位、单条最大128token截断，添加eos结束标识；
    labels mask：「指令：...\\n回答：」部分，padding部分=-100不计算loss，仅回答部分算loss
    """
    def __init__(self, jsonl_path, char2idx, seq_len, stride):
        self.window_list = []   # 存放样本 (win_x, win_label)
        self.seq_len = seq_len
        self.stride = max(1, stride)
        self.char2idx = char2idx
        MAX_ONE_ITEM = CTX_LEN

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
                eos_id = char2idx["<eos>"]
                output_ids = [char2idx[c] for c in out if c in char2idx] + [eos_id]
                full_ids = prompt_ids + output_ids
                full_labels = [-100]*len(prompt_ids) + output_ids

                # 单条问答最大128token截断，防止超长
                if len(full_ids) > MAX_ONE_ITEM:
                    full_ids = full_ids[:MAX_ONE_ITEM]
                    full_labels = full_labels[:MAX_ONE_ITEM]

                total_len = len(full_ids)
                need_len = self.seq_len + 1

                # 短句padding补齐到 need_len（129）
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
    def __init__(self, vocab_size, d_model=192, n_head=4, n_layer=4, max_len=CTX_LEN,
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
        )#norm_first=True和下面norm=nn.LayerNorm(d_model)联动使用，norm_first=True代表先layernorm残差再加输入x,这样最后还是没有归一化，所以最后一步加一个：
         #norm=nn.LayerNorm(d_model)在所有层都做完后在算一次LayerNorm
        self.encoder = nn.TransformerEncoder(
            layer, num_layers=n_layer, norm=nn.LayerNorm(d_model),
            enable_nested_tensor=False, #enable_nested_tensor=False为后续增加mask掩码做准备
        )
        self.head = nn.Linear(d_model, vocab_size, bias=False)
        self.head.weight = self.tok.weight #添加bias=False:去掉b,因为后续要做线性层的权重和tok-embedding层的权重相同，tok-embedding层只有w权重
                                           #权重相等先决条件：1.shape相同(V,D)V:词表长度，D：每个词的向量长度；2.含义相同：都是对字的理解

        nn.init.normal_(self.tok.weight, std=0.02)
        if self.pos is not None:
            nn.init.normal_(self.pos.weight, std=0.02)#正常embedding每个词的向量长度默认（mean(大概正太分布的中心位置)=0,std（正太分布的x标准差),
                                                      #logits与std的关系：logits的方差=D的长度 * 输入向量x里面单个元素的方差 * 权重矩阵W里面单个元素的方差
                                                      #所以权重矩阵W里面单个元素的方差越大，logits的方差越大，过softmax越尖锐，loss越大

    def forward(self, idx):
        # idx: [B, T]
        B, T = idx.shape
        x = self.tok(idx) #(B,T,D)
        if self.use_pos:
            pos_ids = torch.arange(T, device=idx.device) #(T,D)
            x = x + self.pos(pos_ids) #(B,T,D) + (T,D) = (B,T,D) + (1,T,D)
        x = self.drop(x)

        mask = None
        if self.causal:
            mask = nn.Transformer.generate_square_subsequent_mask(T, device=idx.device)
        x = self.encoder(x, mask=mask)
        return self.head(x)


# ─────────────────────── 评估、训练、生成 ───────────────────────
@torch.no_grad()
def evaluate(model, loader, device):
    """
    生成验证集的loss和ppl
    """
    model.eval()
    total, tokens = 0.0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)),
                                y.reshape(-1), reduction="sum", ignore_index=-100) #将每一组的loss相加，真实值标注为-100的不参与loss计算
        total += loss.item()   
        valid_mask = (y != -100)
        tokens += int(valid_mask.sum())
    loss = total / tokens if tokens > 0 else 0.0
    return loss, math.exp(loss)


def train(args):
    os.makedirs(SAVE_DIR, exist_ok=True) #自动创建SAVE_DIR的路径，如果已有这个路径不再新建
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(args.seed)
    random.seed(args.seed)

    char2idx, idx2char = None, None
    tr, va = None, None
    pretrain_ck = None

    #该if条件生成训练集和验证集样本数据，根据传参控制走else（预训练）还是if(sft训练)
    #生成样本数据的class中的参数时，为了防止sft中的训练语句出现了transformers训练时词表没有的字符，进行了预训练词表扩容，将新词加在原有词表后面
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

    #建立模型，当预训练跑完后，将预训练的模型参数覆盖到sft模型初始化的参数，让sft模型保留transformers训练的权重，不用重新训练
    #由于新建了词表扩容，所以扩容部分的tok-embedding部分及head.weight对应扩容部分（原模型设置head.weight =tok.weight）权重初始化是随机的
    model = TransformerLM(
        vocab_size=V,
        d_model=args.d_model,
        n_head=args.heads,
        n_layer=args.layers,
        max_len=CTX_LEN,
        dropout=args.dropout,
        use_pos=not args.no_pos,
        causal=not args.no_mask
    ).to(device)

    if pretrain_ck is not None:
        old_vocab_size = len(pretrain_ck["char2idx"])
        pretrain_state = pretrain_ck["model_state"]
        new_state = model.state_dict()

        new_state["tok.weight"][:old_vocab_size, :] = pretrain_state["tok.weight"][:old_vocab_size, :]

        # # --------位置编码权重扩容--------
        # if "pos.weight" in pretrain_state:
        #     old_pos_weight = pretrain_state["pos.weight"]
        #     new_pos_weight = new_state["pos.weight"]
        #     new_pos_weight[:old_pos_weight.shape[0], :] = old_pos_weight
        #     nn.init.normal_(new_pos_weight[old_pos_weight.shape[0]:, :], std=0.02)
        #     new_state["pos.weight"] = new_pos_weight

        # for key in pretrain_state.keys():
        #     if key not in ("tok.weight", "head.weight", "pos.weight"):
        #         new_state[key] = pretrain_state[key]

        # 位置编码max_len固定128，尺寸不变，直接完整继承，不需要扩容
        for key in pretrain_state.keys():
            if key not in ("tok.weight", "head.weight"):
                new_state[key] = pretrain_state[key]

        model.load_state_dict(new_state)
        model.head.weight = model.tok.weight
        print("已加载预训练权重；SFT新增字符embedding、位置编码新增部分使用随机初始化")

    n_par = sum(p.numel() for p in model.parameters())
    print(f"参数量 {n_par/1e6:.2f}M  位置编码={'有' if not args.no_pos else '无'}  因果掩码={'有' if not args.no_mask else '无'}")

    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    best = float("inf")

    print(f"\n{'Epoch':>6} {'TrainLoss':>10} {'ValPPL':>10} {'tok/s':>9}")
    print("-" * 50)

    #开始训练，生成每组的训练数据和验证数据
    for ep in range(1, args.epochs + 1):
        model.train()
        st, seen, tot = time.time(), 0, 0.0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            loss = F.cross_entropy(logits.reshape(-1, V), y.reshape(-1), ignore_index=-100) #y=-100部分不参与损失计算
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0) #限制所有参数梯度的 L2 范数最大不超过 1.0：不足1.0的正常计算，查过1.0的等比缩减到1.0内
            # torch.nn.utils.clip_grad_norm_(
            # parameters,   # 要裁剪的参数：model.parameters()
            # max_norm=1.0, # 允许的最大L2范数
            # norm_type=2.0 # 默认就是L2范数，可以不写
            # )
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


@torch.no_grad() #在这个函数内部，关闭自动求导（不构建计算图、不保存梯度信息），只做推理 / 评估
def generate(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ck = torch.load(args.ckpt, map_location=device)
    c = ck["args"]
    char2idx, idx2char = ck["char2idx"], ck["idx2char"]
    eos_id = char2idx["<eos>"]

    model = TransformerLM(
        vocab_size=len(char2idx),
        d_model=c["d_model"],
        n_head=c["heads"],
        n_layer=c["layers"],
        max_len=CTX_LEN,
        dropout=0.0,
        use_pos=not c.get("no_pos", False), #训练和推理的模型结构一致
        causal=not c.get("no_mask", False)  #训练和推理的模型结构一致
    ).to(device)

    model.load_state_dict(ck["model_state"])#将训练生成的权重加载进来
    model.eval()

    ids = [char2idx[ch] for ch in args.generate if ch in char2idx]
    out = list(ids)
    for _ in range(args.n_chars):
        feed = torch.tensor([out[-args.seq_len:]], device=device)
        logits = model(feed)[0, -1] #model(feed) 前向传播，得到 [1, T, vocab_size] 的 logits它的
                                    #[0, -1]：取 batch 里第 0 条、最后一个位置的输出，例：把你好作为输入，输出为下一个字，再把这三个字变为输入，再做下一次输出，一共做n_chars次

        if args.temperature <= 0.01:
            nxt = int(logits.argmax()) #温度接近 0 → 贪心解码：直接选概率最大的 token
        #设置temperature=0.7，top-p=0.9
        else:
            logits = logits / args.temperature
            top_p = 0.9
            sorted_probs, sorted_indices = torch.sort(torch.softmax(logits, dim=-1), descending=True)
            #softmax(logits)：原始分数 → 概率分布，所有概率之和 = 1
            #sort(..., descending=True)：按概率从高到低排序
            #sorted_probs：排好序的概率值 [p0, p1, p2, p3,...]，p0 最大
            #sorted_indices：对应每个概率原来的 token 编号
            cumulative_probs = torch.cumsum(sorted_probs, dim=-1)#cumsum = 累加求和
            remove_mask = cumulative_probs > top_p #判断累加概率是否超过 0.9，没有超过位置的为FALSE,超过的位置为TRUE
            remove_mask[1:] = remove_mask[:-1].clone() #把从第二位开始算的结果克隆从第一位到倒数第二位的值，即把标注为FALSE的位置往后加一位，因为要把第一次超过top-p的位置也要算进来
            remove_mask[0] = False #无论如何至少保留概率最高的那一个 token，防止极端情况
            logits[sorted_indices[remove_mask]] = -float("inf") #拿到那些等于TRUE的位置，置为负无穷

            prob = F.softmax(logits, dim=-1) #重新计算现在结果的softmax值
            nxt = int(torch.multinomial(prob, 1)) #根据现在的softmax值(即下一个取到的概率)取出下一个字
#         else:
#             prob = F.softmax(logits / args.temperature, dim=-1)
#             nxt = int(torch.multinomial(prob, 1))

        #防止结束后为了满足args.n_chars续写
        if nxt == eos_id:
            break
        out.append(nxt)

    print(f"输入:\n{args.generate}")
    print(f"\n{''.join(idx2char[i] for i in out[len(ids):])}")


def main():
    ap = argparse.ArgumentParser()
    default_corpus = os.path.join(HERE, "corpus_sample.txt")
    ap.add_argument("--corpus", default=default_corpus)
    ap.add_argument("--max_chars", type=int, default=0)
    ap.add_argument("--max_vocab", type=int, default=4000)
    ap.add_argument("--seq_len", type=int, default=CTX_LEN)
    ap.add_argument("--stride", type=int, default=CTX_LEN)
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
python transformer_lm_sft.py --corpus corpus_general.txt --seq_len 128 --stride 128 --batch_size 64 --epochs 6 --lr 1e-3 --d_model 192 --heads 4 --layers 4 --save "lm_pretrain.pt"

SFT微调
python transformer_lm_sft.py --sft_jsonl_train sft_train.jsonl --sft_jsonl_val sft_val.jsonl --pretrain_ckpt "lm_pretrain.pt" --seq_len 128 --stride 128 --batch_size 64 --epochs 6 --lr 5e-4 --d_model 192 --heads 4 --layers 4 --save "lm_sft_best.pt"

推理
python transformer_lm_sft.py --generate "指令：你好\n回答：" --ckpt "lm_sft_best.pt" --n_chars 10 --temperature 0.7
"""