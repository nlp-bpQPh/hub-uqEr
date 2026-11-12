import math
import torch
import torch.nn as nn
from torch.utils.data import DataLoader,Dataset
import os
import random
import json

device = torch.device("mps")
if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
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
    def __init__(self,vocab,data_path,seq_len,):
        self.seq_len = seq_len
        self.datas = []
        self.vocab = vocab
        with open(data_path,"r",encoding="utf-8") as f:

            for line_number, line in enumerate(f, start=1):

                item = json.loads(line)
                    
                instruction = item['instruction']
                output = item['output']

                self.datas.append((instruction, output))
        

    def __len__(self):
        
        return len(self.datas)
    def __getitem__(self,index):
        instruction,assist = self.datas[index]
        #找不到都先用0填充
        prompt = f"用户：{instruction}\n助手："
        assist = f"{assist}\n"
        prompt_ids = [self.vocab.get(char,0) for char in prompt ] 
        assist_ids = [self.vocab.get(char,0) for char in assist ]
        input_ids = prompt_ids + assist_ids
        labels = [-100]*len(prompt_ids)+assist_ids

        input_ids = input_ids[:self.seq_len]
        labels = labels[:self.seq_len]
                
        x = input_ids[:-1]
        y = labels[1:]

        # 右侧 padding；y 的 padding 不参与 loss
        x += [0] * (self.seq_len - len(x))
        y += [-100] * (self.seq_len - len(y))
        return (
            torch.tensor(x, dtype=torch.long),
            torch.tensor(y, dtype=torch.long),
        )

class SFTModel(nn.Module):
    def __init__(self,vocab_size,input_size,hidden_size,dropout,num_layers=12):
        super(SFTModel,self).__init__()
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

        self.transformer = nn.TransformerEncoder(one_block, num_layers=num_layers)


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
            valid_tokens = (y != -100).sum().item()
            total_loss += loss.item() * valid_tokens
            total_tokens += valid_tokens

    avg_loss = total_loss / total_tokens
    ppl = math.exp(avg_loss)
    return avg_loss,ppl


def main():


    current_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(current_dir, "data")


    epoch_size = 80
    seq_len = 128
    batch_size = 32
    embedding_dim = 128
    hidden_dim = 256
    num_layers = 12
    lr = 3e-4
    train_radio = 0.5
    dropout = 0.1



    #基座
    pretrained_path = os.path.join(data_dir, "best_model.pt")
    train_path = os.path.join(data_dir, "sft_train.jsonl")
    val_path = os.path.join(data_dir, "sft_val.jsonl")
    # 保存sft模型
    save_path = os.path.join(data_dir, "sft_best_model.pt")

    checkpoint = torch.load(
        pretrained_path,
        map_location="cpu",
     weights_only=False,
    )
    vocab = checkpoint["vocab"]


    vocab_size = len(vocab)


    print(f"词表大小=   {vocab_size}")

    train_loader = DataLoader(dataset=TextDataset(data_path= train_path,seq_len = seq_len,vocab=vocab),batch_size=batch_size,shuffle=True,drop_last=False )
    val_loader = DataLoader(dataset=TextDataset(data_path= val_path,seq_len = seq_len,vocab=vocab),batch_size=batch_size,shuffle=False,drop_last=False)


    #创建模型
    model = SFTModel(vocab_size=vocab_size,input_size=embedding_dim,hidden_size=hidden_dim,dropout=dropout,num_layers=num_layers).to(device=device)

    missing_keys, unexpected_keys = model.load_state_dict(
        checkpoint["model_state"],
        strict=False,
    )
    if unexpected_keys:
        raise RuntimeError(f"预训练权重包含无法加载的参数: {unexpected_keys}")
    if missing_keys:
        print(
            f"预训练 checkpoint 缺少 {num_layers - 2} 层参数，"
            "新增层将随机初始化"
        )

    
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

            valid_tokens = (y != -100).sum().item()
            total_loss += loss.item() * valid_tokens
            total_tokens += valid_tokens

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
    print(f"\n训练完成。最佳验证 PPL: {best_val_ppl:.2f}  已保存至 sft_best_model.pt")

if __name__ == '__main__':
    main()
