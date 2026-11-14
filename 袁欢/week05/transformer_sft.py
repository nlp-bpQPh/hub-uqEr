import argparse
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import json

class TransformerLM(nn.Module):
    def __init__(self, vocab_size, d_model, max_len=512, use_pos=True, nhead=4, 
                 num_layer=4, causal=True, dropout=0.1):
        super().__init__()
        self.causal = causal
        self.tok = nn.Embedding(vocab_size, d_model)
        self.use_pos = use_pos
        self.pos = nn.Embedding(max_len, d_model) if use_pos else None
        self.dropout = nn.Dropout(dropout)
        layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, 
            dim_feedforward=d_model*4, dropout=dropout, activation="gelu", 
            batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(encoder_layer=layer, num_layers=num_layer, 
            norm=nn.LayerNorm(d_model), enable_nested_tensor=False)
        self.head = nn.Linear(d_model, vocab_size, bias=False)
        self.head.weight = self.tok.weight

        # 将embedding的参数初始化为0.02方差，默认用(0，1)初始化
        nn.init.normal_(self.tok.weight, std=0.02)
        if use_pos is not None:
            nn.init.normal_(self.pos.weight, std=0.02)

    def forward(self, idx, key_padding_mask=None):
        x = self.tok(idx)
        if self.use_pos:
            x = x + self.pos(torch.arange(idx.size(1), device=idx.device))
        x = self.dropout(x)
        mask = None
        if self.causal:
            T = idx.size(1)
            mask = torch.triu(torch.ones(T, T, device=idx.device, dtype=torch.bool), diagonal=1)
        x = self.encoder(x, mask=mask, src_key_padding_mask=key_padding_mask)
        return self.head(x)

def encode_with_special(sentence, char2idx):
    idx = []
    i = 0
    while i < len(sentence):
        if sentence.startswith("<|user|>", i):
            idx.append("<|user|>")
            i += 8
        elif sentence.startswith("<|assistant|>", i):
            idx.append("<|assistant|>")
            i += 13
        elif sentence.startswith("<|end|>", i):
            idx.append("<|end|>")
            i += 7
        else:
            idx.append(sentence[i])
            i += 1
    return [char2idx.get(i, 1) for i in idx]

class SFTDataset(Dataset):
    def __init__(self, data_path, char2idx, max_len):
        self.data = []
        with open(data_path, "r", encoding="utf-8") as f:
            for line in f:
                item = json.loads(line)
                prompt = f"<|user|>{item['instruction']}<|assistant|>"
                answer = f"{item['output']}<|end|>"
                p_idx = encode_with_special(prompt, char2idx)
                a_idx = encode_with_special(answer, char2idx)

                if len(p_idx) > max_len - 10:
                    p_idx = p_idx[:max_len-10]
                full_idx = p_idx + a_idx
                if len(full_idx) > max_len - 10:
                    full_idx = full_idx[:max_len-10]
                elif len(full_idx) < 2:
                    continue

                x = full_idx[:-1]
                y = full_idx[1:]

                mask_len = len(p_idx) - 1
                y = [-100] * mask_len + y[mask_len:]
                self.data.append((torch.tensor(x), torch.tensor(y)))

    def __len__(self):
        return len(self.data)
    def __getitem__(self, index):
        return self.data[index]
    
def collate_fn(batch):
    max_len = max(len(x) for x, y in batch)
    batch_x, batch_y = [], []
    for x, y in batch:
        padding_len = max_len - len(x)
        batch_x.append(torch.cat([x, torch.zeros(padding_len, dtype=torch.long)]))
        batch_y.append(torch.cat([y, torch.full((padding_len,), -100, dtype=torch.long)]))
    return torch.stack(batch_x), torch.stack(batch_y)

def evaluate_sft(model, val_loader, device):
    model.eval()
    with torch.no_grad():
        loss_total = 0.0
        token_total = 0
        for x, y in val_loader:
            x, y = x.to(device), y.to(device)
            mask = (x == 0)
            logits = model(x, mask)
            loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), y.reshape(-1), 
                    ignore_index=-100, reduction="sum")
            loss_total += loss.item()
            token_total += (y != -100).sum().item()
    return loss_total / token_total

def train_sft(args, device):
    # 加载预训练模型参数
    ckpt = torch.load(args.pretrained, map_location="cpu")
    old_args = ckpt["args"]
    char2idx = ckpt["char2idx"]
    idx2char = ckpt["idx2char"]
    old_state = ckpt["model_state"]
    old_vocab_size = len(char2idx)

    # 固定随机值
    torch.manual_seed(old_args["seed"])

    # 注入sft首尾分隔符
    special_token = ["<|user|>", "<|assistant|>", "<|end|>"]
    for token in special_token:
        if token not in char2idx:
            char_len = len(char2idx)
            char2idx[token] = char_len
            idx2char[char_len] = token
    vocab_size = len(char2idx)
    
    # 实例化新模型
    model = TransformerLM(vocab_size, old_args["d_model"], max_len=old_args["seq_len"], use_pos=not old_args["no_pos"], 
            nhead=old_args["heads"], num_layer=old_args["layers"], causal=not old_args["no_mask"]).to(device=device)
    
    # 拷贝预训练模型参数
    new_state = model.state_dict()
    #print(new_state.keys())
    for i in new_state:
        if "tok.weight" in i or "head.weight" in i:
            new_state[i][:old_vocab_size] = old_state[i]
        else:
            new_state[i] = old_state[i]
    model.load_state_dict(new_state)
    model.head.weight = model.tok.weight

    # 加载数据
    train_dataset = SFTDataset(args.train_data, char2idx=char2idx, max_len=old_args["seq_len"])
    val_dataset = SFTDataset(args.val_data, char2idx=char2idx, max_len=old_args["seq_len"])
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn, drop_last=True)

    # 实例化优化器
    adam = torch.optim.Adam(model.parameters(), lr=args.lr)

    # 训练并验证
    loss_best = float("inf")
    for epoch in range(args.epochs):
        model.train()
        loss_total = 0.0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            pad_mask = (x == 0)
            logits = model(x, pad_mask)
            loss = F.cross_entropy(logits.reshape(-1, vocab_size), y.reshape(-1), ignore_index=-100)

            adam.zero_grad()
            loss.backward()
            adam.step()
            loss_total += loss.item()

        loss_eval = evaluate_sft(model, val_loader, device)
        print(f"epoch[{epoch}]: train[{loss_total/len(train_loader):.4f}] eval[{loss_eval:.4f}]")

        if loss_eval < loss_best:
            loss_best = loss_eval
            torch.save({"model_state": model.state_dict(), "char2idx": char2idx, "idx2char": idx2char, 
                    "args": old_args}, args.save)

def generate_sft(args, device):
    # 加载模型参数
    ckpt = torch.load(args.ckpt, map_location="cpu")
    old_args = ckpt["args"]
    char2idx = ckpt["char2idx"]
    idx2char = ckpt["idx2char"]

    prompt = args.generate
    prompt = f"<|user|>{prompt}<|assistant|>"
    p_idx = encode_with_special(prompt, char2idx)
    out = list(p_idx)

    model = TransformerLM(len(char2idx), old_args["d_model"], old_args["seq_len"], not old_args["no_pos"], 
                old_args["heads"], old_args["layers"], not old_args["no_mask"], old_args["dropout"]).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    for _ in range(args.n_chars):
        input_tensor = torch.tensor([out[-old_args["seq_len"]:]], device=device)
        logits = model(input_tensor)[0, -1]
        if args.temperature < 0.01:
            next_token = int(logits.argmax())
        else:
            next_token = int(torch.multinomial(F.softmax(logits / args.temperature, dim=-1), 1))

        if next_token == char2idx["<|end|>"]:
            break

        out.append(next_token)
    print(f"User Input:{args.generate}")
    print(f"AI:{''.join(idx2char[i] for i in out[len(p_idx):])}")
    
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pretrained", default="tf_best.pt", help="预训练模型")
    ap.add_argument("--train_data", default="sft_train.jsonl", help="训练数据")
    ap.add_argument("--val_data", default="sft_val.jsonl", help="验证集数据")
    ap.add_argument("--save", default="sft_best.pt", help="模型保存路径")
    ap.add_argument("--batch_size", type=int, default=32, help="训练批次大小")
    ap.add_argument("--epochs", type=int, default=5, help="训练批次")
    ap.add_argument("--lr", type=float, default=1e-4, help="学习率")
    ap.add_argument("--generate", default=None, help="prompt，用于推理测试")
    ap.add_argument("--ckpt", default="sft_best.pt", help="推理模型")
    ap.add_argument("--n_chars", type=int, default=50, help="推理长度")
    ap.add_argument("--temperature", type=float, default=0.8, help="温度")
    args = ap.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if args.generate:
        generate_sft(args, device)
    else:
        train_sft(args, device)

if __name__ == "__main__":
    main()
