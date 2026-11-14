import torch
import torch.nn as nn
from collections import Counter
from torch.utils.data import Dataset, DataLoader
# 第一步 将输入L长度，D维张量，根据维度切成12个头，Q，K，V。 计算Q乘以K的转置，得到L长度，L维度的张量。
# 每行都要经过softmax(行除以根号下转置后的K的维度)。之后得到的结果乘以V
# 第二步。经过两层线性函数：gelu(W1*X+B1)W2*X+B2


"""定义一个模型"""

class Transformer(nn.Module):
    def __init__(self,vocab_size,hidden_size,max_len):
        super().__init__()
        self.token_embeddings = nn.Embedding(vocab_size,hidden_size)
        self.pos_embeddings = nn.Embedding(max_len,hidden_size)
        # 2. Transformer 层 (用一个 EncoderLayer 来充当单层 Decoder)
        # 这个模块内部自带了：多头自注意力 -> 残差 -> GELU前馈网络 -> 残差
        self.transformer = nn.TransformerEncoderLayer(
            d_model=hidden_size,
            nhead=4,
            dim_feedforward=hidden_size * 4,  # 对应你注释里的两层线性网络维度
            activation='gelu',  # 对应你说的 gelu
            batch_first=True
        )
        self.layer=nn.Linear(hidden_size,vocab_size)

    def forward(self,x):
        token = self.token_embeddings(x)
        positions = torch.arange(0, x.size(1), device=x.device).unsqueeze(0)
        positions=self.pos_embeddings(positions)
        embeddings = token + positions
        # 生成因果掩码 (Causal Mask)：防止模型看到未来的词
        # 这一步非常关键，语言模型只能看前面的词
        mask = nn.Transformer.generate_square_subsequent_mask(x.size(1)).to(x.device)
        embeddings = self.transformer(embeddings, mask)
        output = self.layer(embeddings)
        return output

"""生成词表"""

def build_vocab(text_path,max_vocab):
    with open(text_path,'r',encoding='utf-8') as f:
        counter = Counter()
        texts=[]
        for line in f:
            texts.append(line)

        for word in texts:
            for char in word:
                counter[char]+=1
        chars = list(['<pad>', '<cls>', '<sep>', '<unk>'])
        for c, _ in counter.most_common(max_vocab-4):
            chars.append(c)

        char2idx = {}
        idx2char = {}
        for i, c in enumerate(chars):
            char2idx[c] = i
            idx2char[i] = c
        return char2idx, idx2char,texts


"""定义无监督训练数据"""

class TextDataset(Dataset):
    def __init__(self,char2idx,texts,max_len,seq_len):
        self.seq_len = seq_len
        self.stride = max(1, max_len)

        ids = []
        for c in texts:
            if c in char2idx:
                ids.append(char2idx[c])
            else:
                ids.append(char2idx['<unk>'])
        self.data = torch.tensor(ids, dtype=torch.long)

    def __len__(self):
        return max(0, (len(self.data) - self.seq_len - 1) // self.stride + 1)

    def __getitem__(self, idx):
        start = idx * self.stride
        x = self.data[start: start + self.seq_len]
        y = self.data[start + 1: start + self.seq_len + 1]
        return x, y



to_idx,to_chars,seq=build_vocab("./data/corpus_general.txt",4000)
test_word="谁有范冰冰的苹果这部电影"
train_text = "".join(seq)

dataset=TextDataset(to_idx,train_text,512,512)
train_loader = DataLoader(dataset, batch_size=50, shuffle=True, drop_last=True)

# 3. 模型实例化
VOCAB = len(to_idx)          # 4000
HIDDEN = 256
MAX_LEN = 512
model = Transformer(VOCAB, HIDDEN, MAX_LEN)
optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
criterion = nn.CrossEntropyLoss()
EPOCHS = 20
model.train()
for epoch in range(EPOCHS):
    total_loss = 0.0
    for step, (x, y) in enumerate(train_loader):

        logits = model(x)                        # (B, L, V)
        loss = criterion(logits.reshape(-1, VOCAB), y.reshape(-1))
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        total_loss += loss.item()
    avg = total_loss / len(train_loader)
    print(f"Epoch {epoch+1}/{EPOCHS}  loss={avg:.4f}")

def generate(model, char2idx, idx2char, prompt, max_new=30, temperature=1.0):
    model.eval()
    ids = [char2idx.get(c, char2idx['<unk>']) for c in prompt]
    ids = torch.tensor(ids, dtype=torch.long).unsqueeze(0)
    for _ in range(max_new):
        inp = ids[:, -MAX_LEN:]                  # 防止超过位置编码长度
        logits = model(inp)[:, -1, :] / temperature
        probs = torch.softmax(logits, dim=-1)
        nxt = torch.multinomial(probs, num_samples=1)
        ids = torch.cat([ids, nxt], dim=1)
    out = ids[0].tolist()
    return "".join(idx2char[i] for i in out)

test_word = "谁有范冰冰的苹果这部电影"
print("生成结果:", generate(model, to_idx, to_chars, test_word, max_new=30))












