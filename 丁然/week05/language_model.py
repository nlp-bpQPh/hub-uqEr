import math
import torch
import torch.nn as nn
from torch.utils.data import DataLoader,Dataset
import os
import random

# device = torch.device("mps")
# if torch.cuda.is_available():
#     device = torch.device("cuda")
# elif torch.backends.mps.is_available():
#     device = torch.device("mps")
# else:
#     device = torch.device("cpu")
device = torch.device("cpu")

def load_corpus(path):
    texts = []
    with open(path,"r",encoding="utf-8") as f:
        texts.append(f.read())
    return "".join(texts)

def build_vocab(texts):
    #先去重再排序
    chars = sorted(set(texts))
    vocab = {}
    for i,char in enumerate(chars):
        vocab[char] = i
    return vocab

class TextDataset(Dataset):
    def __init__(self,vocab,text,seq_len,):
        self.seq_len = seq_len
        self.ids = [vocab.get(char) for char in text if char in vocab]

    def __len__(self):
        # 每个样本使用一段不重叠的文本，并额外保留一个字符作为 y 的末尾
        return max(0, (len(self.ids) - 1) // self.seq_len)
    def __getitem__(self,index):

        start = index * self.seq_len
        x = self.ids[start:start + self.seq_len]
        y = self.ids[start + 1:start + self.seq_len + 1]
        return (
            torch.tensor(x, dtype=torch.long),
            torch.tensor(y, dtype=torch.long),
        )

class LanguageModel(nn.Module):
    def __init__(self,vocab_size,input_size,hidden_size,dropout):
        super(LanguageModel,self).__init__()
        self.embedding = nn.Embedding(num_embeddings=vocab_size,embedding_dim=input_size)
        
        # 位置编码
        self.pos_embedding = nn.Embedding(128, input_size)
        one_block = nn.TransformerEncoderLayer(
            d_model=input_size, 
            nhead=8, 
            dim_feedforward=hidden_size, 
            dropout=dropout,
            batch_first=True
        )

        self.transformer = nn.TransformerEncoder(one_block, num_layers=2)


        self.dropout = nn.Dropout(dropout)
        self.linear = nn.Linear(input_size, vocab_size)
        self.loss = nn.CrossEntropyLoss()


    def generate_square_subsequent_mask(self, sz):
        """生成因果掩码（下三角矩阵），防止看到未来的词"""
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask
    
    def forward(self,x,y = None):

        seq_len = x.size(1)
        
        # 1. 算词向量和位置向量，然后加起来
        e = self.embedding(x)
        positions = torch.arange(seq_len, device=x.device).unsqueeze(0)
        e = e + self.pos_embedding(positions)
        e = self.dropout(e)

        # 2. 生成因果掩码（这是替换 LSTM 的灵魂，保证单向生成）
        # causal_mask = self.generate_square_subsequent_mask(seq_len).to(device)
         # 用自带的生成，并且不需要 .to(device) 因为它本来就在！
        causal_mask = torch.nn.Transformer.generate_square_subsequent_mask(seq_len).to(x.device)
        # 3. 经过 Transformer 提取特征
        # 注意这里传入了 mask，告诉模型只能往前看
        r = self.transformer(e, mask=causal_mask)

        # 4. 预测下一个词
        logits = self.linear(self.dropout(r))
        
        # 5. 计算损失
        if y is not None:
            return self.loss(logits.reshape(-1, logits.size(-1)), y.reshape(-1))
        else:
            return torch.softmax(logits, dim=-1)



def evaluate(model,loader):
    model.eval()

    total_loss = 0.0
    total_tokens = 0
    with torch.no_grad():
        for x,y in loader:
            x, y = x.to(device), y.to(device)
            loss = model(x,y)

            total_loss += loss.item() * y.numel()
            total_tokens += y.numel()

    avg_loss = total_loss / total_tokens
    ppl = math.exp(avg_loss)
    return avg_loss,ppl


def main():


    current_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(current_dir, "data")


    epoch_size = 8
    seq_len = 128
    batch_size = 128
    embedding_dim = 128
    hidden_dim = 256
    num_layers = 2
    lr = 1e-3
    train_radio = 0.5
    dropout = 0.3
    corpus = os.path.join(data_dir, "corpus_general.txt")


    save_path = os.path.join(data_dir, "best_model.pt")


    #读取语料
    texts = load_corpus(corpus)
    #构建词表
    vocab = build_vocab(texts=texts)
    vocab_size = len(vocab)

    lines = texts.splitlines()
    random.shuffle(lines)
    split = int(len(lines)*train_radio)

    train_text = "\n".join(lines[:split])
    val_text   = "\n".join(lines[split:])

    print(f"词表大小=   {vocab_size}")

    train_loader = DataLoader(dataset=TextDataset(text = train_text,seq_len = seq_len,vocab=vocab),batch_size=batch_size,shuffle=True,drop_last=True                         
    ,num_workers=4,       # 👈 强行开启 4 个 CPU 进程去切数据
    pin_memory=True  )
    val_loader = DataLoader(dataset=TextDataset(text = val_text,seq_len = seq_len,vocab=vocab),batch_size=batch_size,shuffle=True,drop_last=True)


    #创建模型
    model = LanguageModel(vocab_size=vocab_size,input_size=embedding_dim,hidden_size=hidden_dim,dropout=dropout).to(device=device)
    #创建优化器
    opt = torch.optim.Adam(model.parameters(),lr= lr)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"模型参数量: {total_params:,}")

    best_val_ppl = float("inf")

    print(f"\n{'Epoch':>6}  {'Train Loss':>10}  {'Train PPL':>10}  {'Val Loss':>10}  {'Val PPL':>10}")
    print("-" * 56)

    for epoch in range(epoch_size):
        model.train()
        print(
            f"\nEpoch {epoch} 开始训练，共 {len(train_loader)} 个批次",
            flush=True,
        )

        total_loss = 0.0
        total_tokens = 0

        for step, (x, y) in enumerate(train_loader, start=1):
            x, y = x.to(device), y.to(device)
            loss = model(x,y)
            opt.zero_grad()
            loss.backward()
            opt.step()

        
            total_loss += loss.item() * y.numel()
            total_tokens += y.numel()

            if step % 20 == 0 or step == len(train_loader):
                print(
                    f"  训练进度 {step}/{len(train_loader)}，"
                    f"当前批次 Loss: {loss.item():.4f}",
                    flush=True,
                )
        
        avg_loss = total_loss / total_tokens
        ppl = math.exp(avg_loss)

        print(f"Epoch {epoch} 训练完成，开始验证", flush=True)
        eval_loss,eval_ppl = evaluate(model,val_loader)
        if eval_ppl < best_val_ppl:
            best_val_ppl = eval_ppl
            torch.save({
                "model_state": model.state_dict(),
                "vocab": vocab,
            }, save_path)

        print(f"{epoch:>6}  {avg_loss:>10.4f}  {ppl:>10.2f}  {eval_loss:>10.4f}  {eval_ppl:>10.2f}")
    print(f"\n训练完成。最佳验证 PPL: {best_val_ppl:.2f}  已保存至 best_model.pt")

if __name__ == '__main__':
    main()
