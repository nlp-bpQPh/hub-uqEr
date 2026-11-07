"""
字符级语言模型训练脚本，支持 RNN / LSTM 切换，含 PPL 计算。
用法:
    python language_model.py --model lstm --epochs 20
    python language_model.py --model rnn  --epochs 20
"""
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
import math
import argparse
import glob
import random
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

script_dir = os.path.dirname(os.path.abspath(__file__))
pattern = os.path.join(script_dir, "*.txt")

# ─────────────────────────── 数据 ───────────────────────────

def load_corpus(pattern):
    texts = []
    for path in glob.glob(pattern):
        with open(path, encoding="utf-8", errors="ignore") as f:
            texts.append(f.read())
    return "".join(texts)
"""
glob.glob（）找出当前文件夹下所有符合的文件，默认 py 执行的终端文件夹（不一定是当前 py 所在的文件夹）
"*.txt":该目录下所有的TXT文件
由于我的脚本位置：`D:/AI/week4/测试.py`，在终端 `C:/Users/Administrator` 执行，
进程的工作目录 `os.getcwd()` → `C:/Users/Administrator`，`glob.glob("*.txt")` 去 C 盘这个目录找 txt，**不是去 D:/AI/week4
所以改进：
import glob
import os

# __file__ 代表当前这个py文件
script_dir = os.path.dirname(os.path.abspath(__file__))
# 拼接搜索pattern
pattern = os.path.join(script_dir, "*.txt")

files = glob.glob(pattern)
for f in files:
    print(f)
"""


def build_vocab(text, max_vocab=None):
    """
    构建字表：按出现频率取前 max_vocab 个字（None = 全部字符）。
    语料里的生僻字超出字表后会被 CharDataset 过滤掉 —— 字表越小模型越小，
    但覆盖不足会丢信息，这是词表大小的取舍。
    """
    from collections import Counter
    counter = Counter(text)
    chars = [c for c, _ in counter.most_common(max_vocab)] if max_vocab else sorted(set(text))
    char2idx = {c: i for i, c in enumerate(chars)}
    idx2char = {i: c for c, i in char2idx.items()}
    return char2idx, idx2char
"""
counter.most_common(N)**是一个方法**：counter.most_common () 如果不加参数，就是全部字符按频率排序
- 作用：**按频次从高到低排序**，返回 `(字符, 次数)` 的列表
- 传 `max_vocab`：只取前 `max_vocab` 个高频字符
sorted(set(text))`：排序规则是 **Unicode 编码从小到大**
- **当 max_vocab 有值（比如 3000）**：
顺序：**字符出现频率从高到低**，取前 max_vocab 个
- **当 max_vocab = None（假）**：
顺序：**Unicode 编码升序**，全部字符都要，不看频率
"""

class CharDataset(Dataset):
    """
    字符级语言模型的训练样本：输入 x = 连续 seq_len 个字符，目标 y = 往后错一位。

    stride 控制切块间隔：
        stride=1  —— 每个字符位置都当一个样本（样本大量重叠，一期语料能切出上百万个样本）
        stride=seq_len —— 不重叠切块（标准做法，样本数 = 语料长度 / seq_len）
    语料一大，stride=1 会让一轮训练变得极慢，这不是模型慢，而是样本数被放大了 seq_len 倍。
    """

    def __init__(self, text, char2idx, seq_len, stride=1):
        self.seq_len = seq_len
        self.stride = max(1, stride)
        ids = [char2idx[c] for c in text if c in char2idx]
        self.data = torch.tensor(ids, dtype=torch.long)

    def __len__(self):
        return max(0, (len(self.data) - self.seq_len - 1) // self.stride + 1)

    def __getitem__(self, idx):
        start = idx * self.stride
        x = self.data[start: start + self.seq_len]
        y = self.data[start + 1: start + self.seq_len + 1]
        return x, y
"""
`torch.utils.data.Dataset` 是抽象基类，本身没有固定传参；传参完全由你自己的数据集逻辑决定，
__init__：加载原始大数据，存本体（只跑一次），即下面提到的数据集
只强制要求实现两个魔法方法：
__len__：设定数据集一共有多少样本
__getitem__(idx)：idx是由上面得来的数据集样本索引，根据上面的规定分成的组一组一组带入，按需现场切片，造出这一份 x,y(可以没有y，无监督）（每取一个样本就跑一次）

一般文本类继承Dataset用CharDataset，其他，比如图像类用ImagePairDataset


通用文本类 Dataset 常见入参
1. **原始数据**：`text` / `lines` / `data_list`（字符串列表、id 列表）
2. **词表 / 分词器**：`tokenizer` / `vocab` / `char2idx`（负责文本→id）
3. **序列超参**：`seq_len`、`max_len`（截断）
4. **任务相关参数**：mask 概率、是否做 padding、是否截断
5. **预处理开关**：`shuffle`（一般不放这里，shuffle 交给 DataLoader！）、`truncate`、`filter`

后续Dataset生成实例后用DataLoader作为外面的加载器：
Dataset 只管：按索引拿单条样本 (x,y)
DataLoader 负责：打包成 batch、打乱、多线程加载、预取、padding 等
例：
#实例化 Dataset
ds = CharDataset(raw_text, block_size=block_size)

#包装 DataLoader
dl = DataLoader(
    dataset=ds,
    batch_size=4,        # 一次拿4条样本组成一个batch
    shuffle=True,         # 每个epoch开始打乱样本顺序（训练集一般开，验证集False）
    num_workers=0,        # 加载子进程，Windows建议写0，linux可以>=1
    drop_last=False,      # 最后不够一个batch的样本要不要丢掉
)

# 送入模型
for x, y in dl:
    print("x:", x.shape)
    print("y:", y.shape)

    logits = model(x)
    loss = loss_fn(logits, y)
    
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
"""

# # ─────────────────────────── 模型 ───────────────────────────

# class LM(nn.Module):
#     def __init__(self, vocab_size, embed_dim, hidden_dim, num_layers, model_type, dropout):
#         super().__init__()
#         self.embed = nn.Embedding(vocab_size, embed_dim)
#         rnn_cls = nn.LSTM if model_type == "lstm" else nn.RNN
#         self.rnn = rnn_cls(
#             embed_dim, hidden_dim,
#             num_layers=num_layers,
#             batch_first=True,
#             dropout=dropout if num_layers > 1 else 0.0,
#         )
#         self.drop = nn.Dropout(dropout)
#         self.fc = nn.Linear(hidden_dim, vocab_size)

#     def forward(self, x):
#         e = self.drop(self.embed(x))
#         out, _ = self.rnn(e)
#         logits = self.fc(self.drop(out))   # (B, T, V)
#         return logits
"""
RNN/LSTM的h是每个h单独拿出来过后面的函数，还是全部h拿出来过后面的函数，还是最后一个h拿出来过后面的函数，取决于后面的函数：
分支 1（逐位置）：h[t] → Linear → Softmax → 预测token
分支 2（取最后一个）：h[-1] → Linear → Sigmoid → 情感分类
分支 3（时序聚合）：h全部 → MeanPool → Linear
分支 4（加 norm）：h[t] → LayerNorm → Linear
在该文本生成字监督任务中，走的是分支1，每一个隐状态h都单独过后面的Linear 投影头 + softmax
"""

# # ─────────────────────────── Transformer 模型（替换原来的 LM LSTM/RNN） ───────────────────────────
# class TransformerLM(nn.Module):
#     def __init__(self,vocab_size,embed_dim,hidden_dim,num_layers,nhead,dropout,max_seq_len):
#         super().__init__()
#         self.embed_dim = embed_dim

#         # token embedding
#         self.token_emb = nn.Embedding(vocab_size, embed_dim)

#         # 位置编码（可学习位置编码，也可以换成正弦）
#         self.pos_emb = nn.Embedding(max_seq_len, embed_dim)

#         decoder_layer = nn.TransformerDecoderLayer(
#             d_model=embed_dim,
#             nhead=nhead,
#             dim_feedforward=hidden_dim,
#             dropout=dropout,
#             batch_first=True,
#             activation="gelu"
#         )
#         self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)
#         self.drop = nn.Dropout(dropout)
#         self.fc = nn.Linear(embed_dim, vocab_size)

#     def forward(self, x):
#         B, T = x.shape
#         # token embedding
#         tok_emb = self.token_emb(x)

#         # 位置索引
#         pos = torch.arange(0, T, device=x.device).unsqueeze(0).expand(B, T)
#         #torch.arange(0, T, device=x.device)：生成一维向量：[0, 1, 2, ..., T-1]，长度为 T，shape：(T,),和 x 在同一个 GPU/CPU 设备上
#         #.unsqueeze(0):在第 0 维增加一个维度，变成 (1, T),如果写.unsqueeze(1)在第 0 维增加一个维度，变成 (T，1)
#         #.expand(B, T)：在第 0 维复制扩展 `B` 份，得到 `(B, T)`

#         pos_emb = self.pos_emb(pos) #和tok_emb的维度保持一致
#         e = self.drop(tok_emb + pos_emb)

#         # causal mask：上三角mask，禁止看到未来token
#         causal_mask = nn.Transformer.generate_square_subsequent_mask(T, device=x.device)
#         # TransformerDecoder 需要 memory，这里 Decoder-only，memory传全0
#         memory = torch.zeros_like(e)
#         out = self.decoder(tgt=e, memory=memory, tgt_mask=causal_mask)
#         out = self.drop(out)
#         logits = self.fc(out)   # [B, T, vocab_size]
#         return logits
"""
关于transformers:
1.建立入参x:一般是x的token和x的位置token相加，token均为数字在nn.Embedding对应的向量
  在代码中表示为：
  准备部分：self.embed_dim = embed_dim
           self.pos_emb = nn.Embedding(max_seq_len, embed_dim)
  生成部分：B, T = x.shape 
           tok_emb = self.token_emb(x)
           pos = torch.arange(0, T, device=x.device).unsqueeze(0).expand(B, T)
           pos_emb = self.pos_emb(pos)
           e = self.drop(tok_emb + pos_emb)
建立位置：文本是有位置的概念的，文字顺序打乱和原本并不是一个意思，
LNN/LSTM本身就是顺时串行顺序，但transformers是矩阵相乘，换了位置相乘后结果一样，为了要不一样，添加了位置token
现在传入一行文字，设最大的位置为512，那么这个传入的单个文本样本不能超过512个字符，否则会报错
现在设传入的一批样本为（B,L,T)，
B:单次传入的样本数（这些样本一起算损失值）
L:每个样本的字符长度
T：每个字符的向量长度

2.
1）通过线性层生成三个输出：Q,K,V（和X的输入维度保持一致）
2）将Q和K按设定的头参数nhead分成多份，例：设定头为12，就是将T这个长度分成12份，其中L一定要被nhead整除
3）每一份Q*每一份K的转置：得到每个字和L中的每个字的关联程度矩阵
   由于当前任务是文本生成，所以给下一个生成的字手动挡上，这里采用了causal_mask = nn.Transformer.generate_square_subsequent_mask(T, device=x.device)，
   即：将生成的矩阵中对角线斜上方的内容全部设为负无穷(-inf),这样后面softmax时，就会把那些设为负无穷的值基本上屏蔽
4）逐行进行softmax(scores/根号T)：逐行softmax是为了匹配上面说的：得到每个字和L中的每个字的关联程度矩阵，
                                 除以T开根号：根据softmax函数（e的输入次方/每个输入的e的输入次方），为了输入本来相差差不多的两个值不要因为e的x次方差距过大
5)将这个得到的值*V：将每个字和L中的每个字的关联程度实体化，
  例如[0.7,0.2,0,1]*[v1,v2,v3]的转置，得到0.7v1+0.2v2+0.1v3,这个值替换到原来的第一个字，得到了这个字和样本中那些字的对应关系
  结合上面的将生成的矩阵中对角线斜上方的内容全部设为负无穷，第一个字实际情况包含的内容就是他自己，这里的输出就根据这个字后面生成的，符合输入x:[1,2,3],输出y:[2,3,4]
6）将分成12份得到的值拼起来，得到和输入相同维度的输出
7）将这个输出的值+原输入x相加后做Layernorm归一化，或者将这个输出的值做Layernorm归一化再和原输入x相加
8）进入到前馈网络（FFN）FFN(x)=GELU(W1*x+b1)*W2+b2
   第一层线性层W1*x+b1进入后，一般会将T改为4T，不过也可以时2T,3T，一般会使T边长：增加可训练参数，使模型学习效果更好
   过激活函数GELU，也可以时其他激活函数，再过线性层，这时会把原来扩大的T（例：4T）还原回(B,L,T)
9）将这个输出的值+原输入x相加后做Layernorm归一化，或者将这个输出的值做Layernorm归一化再和原输入x相加
10）将结果经过后续的线性层等其他函数，得到输出的预测值y，
    和真实值y做nn.CrossEntropyLoss(预测值y,真实值y)损失计算，
    模型输出：logits.shape = [B, T, V]，V = 词表大小（每个位置一个长度为 V 的分数向量，这就是传给 CrossEntropyLoss 的预测侧输入）
    标签（真实 y）：labels.shape = [B, T]，是每个位置的 token id（整数，0~V-1）
    但是！PyTorch 原生 nn.CrossEntropyLoss 默认要求：输入 [N, C]，标签 [N],即logits: [B, T, V] → flatten → [B*T, V]，labels: [B, T]    → flatten → [B*T]所以：
    后续需要logits_flat = logits.reshape(B * T, V) ，labels_flat = labels.reshape(B * T) 
    CrossEntropyLoss内部会做log(softmax (预测值y)),即先把y每一行做总和=1的计算，例：该位置 logits：[2.1, 0.3, -1.0]，词表大小 3
    softmax 后概率：[0.84, 0.14, 0.02]，真实标签是 0（第 0 类），损失 = - log(0.84)
11）注：关于memory = torch.zeros_like(e)
           out = self.decoder(tgt=e, memory=memory, tgt_mask=causal_mask)
     是适配TransformerDecoder的写法，是encoder编码器→decoder解码器，它的输入是（句子1，句子1的翻译句子2）
     tgt自注意力部分就是这个文本生成任务要做的部分，
     交叉注意力memory就是解码器的部分，query来自tgt，key/value来自memory，它的序列长度L可以和 tgt 不一样（因为通过上面Q,K,V的算法，最后也是（B,L,T)，
     由于这个任务不需要解码器也没有翻译的样本，所以memory = torch.zeros_like(e)全部致0
     关于pythorch里打包好的transformers有五种：
     nn.Transformer                # 完整的encoder-decoder整套（seq2seq）
     nn.TransformerEncoder         # 堆叠多层 encoder layer 的容器
     nn.TransformerEncoderLayer    # 单层 encoder block（只有自注意力，无cross-attn）
     nn.TransformerDecoder         # 堆叠多层 decoder layer 的容器
     nn.TransformerDecoderLayer    # 单层 decoder block（自注意力 + cross注意力 + FFN）  

     其中TransformerEncoder是只有编码器，下面有根据这个完成，也有手写法
12）目前输入是序列长度为64的句子，每个位置都有值，没有padding0，
    如果最后的输入是，例：x = [12,35,78, 0, 0, 0],self.blocks需要添加src_key_padding_mask=pad_mask：self.blocks(src, mask=causal_mask, src_key_padding_mask=pad_mask)
    这样在上述3）生成L*L的矩阵时（经过generate_square_subsequent_mask使对角线以上都变成-inf的基础上），将右下角3*3的矩阵也变成负无穷，将这个矩阵带入到后面*V的计算
    并且在后续计算损失的时候：loss_fn = CrossEntropyLoss(ignore_index=0)，
    这样当标签label（即真实值，由于这个是文本生成任务，对应的样本y就是[35,78, 0, 0, 0，0]）== 0 的时候，这个字符为0的位置不参与 loss 计算，梯度不回传。
"""
# ─────────────────────────── 改成TransformerEncoder  ───────────────────────────
class TransformerLM(nn.Module):
    def __init__(self,vocab_size,embed_dim,hidden_dim,num_layers,nhead,dropout,max_seq_len):
        super().__init__()
        self.embed_dim = embed_dim
        self.token_emb = nn.Embedding(vocab_size, embed_dim)
        self.pos_emb = nn.Embedding(max_seq_len, embed_dim)

        # 用EncoderLayer！只有self-attn，没有cross-attn，完美适配GPT
        enc_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=nhead,
            dim_feedforward=hidden_dim,
            dropout=dropout,
            batch_first=True,
            activation="gelu"
        )
        self.blocks = nn.TransformerEncoder(enc_layer, num_layers=num_layers)
        self.drop = nn.Dropout(dropout)
        self.fc = nn.Linear(embed_dim, vocab_size)

    def forward(self, x):
        B, T = x.shape
        tok_emb = self.token_emb(x)
        pos = torch.arange(0, T, device=x.device).unsqueeze(0).expand(B, T)
        pos_emb = self.pos_emb(pos)
        e = self.drop(tok_emb + pos_emb)

        # causal mask，遮住未来token
        #torch.ones (T, T, dtype=torch.bool, device=x.device) 创建一个 T 行 × T 列 的布尔张量，全部填充为 True
        #torch.triu(inp, diagonal=1)：从主对角线向右偏移1的那条对角线开始，往上的上三角区域保留原值(True)，剩下所有位置置为 False
        #下三角变成 False，本质是 triu 在非上三角位置填充 0，再因为 dtype 是 bool，0 等价 False
        causal_mask = torch.triu(torch.ones(T, T, dtype=torch.bool, device=x.device), diagonal=1)
        # 不需要memory！！
        out = self.blocks(e, mask=causal_mask)
        out = self.drop(out)
        logits = self.fc(out)
        return logits
    
# # ─────────────────────────── 手写多头注意力 + GPT Block（Pre-LN，GPT2标准） ───────────────────────────
# class MaskedMultiHeadSelfAttention(nn.Module):
#     def __init__(self, d_model, n_head):
#         super().__init__()
#         assert d_model % n_head == 0
#         self.d_model = d_model
#         self.n_head = n_head
#         self.d_k = d_model // n_head

#         self.w_qkv = nn.Linear(d_model, 3 * d_model)
#         self.w_o = nn.Linear(d_model, d_model)

#     def forward(self, x, mask=None):
#         B, L, _ = x.shape
#         qkv = self.w_qkv(x)
#         q, k, v = torch.chunk(qkv, 3, dim=-1)

#         q = q.view(B, L, self.n_head, self.d_k).transpose(1, 2)
#         k = k.view(B, L, self.n_head, self.d_k).transpose(1, 2)
#         v = v.view(B, L, self.n_head, self.d_k).transpose(1, 2)

#         score = q @ k.transpose(-1, -2) / (self.d_k ** 0.5)
#         if mask is not None:
#             score = score.masked_fill(mask, -1e9)
#         attn_weight = F.softmax(score, dim=-1)
#         out = attn_weight @ v

#         out = out.transpose(1, 2).contiguous().view(B, L, self.d_model)
#         out = self.w_o(out)
#         return out


# class GPTBlock(nn.Module):
#     def __init__(self, d_model, n_head, ff_dim, dropout):
#         super().__init__()
#         self.attn = MaskedMultiHeadSelfAttention(d_model, n_head)
#         self.ln1 = nn.LayerNorm(d_model)
#         self.ln2 = nn.LayerNorm(d_model)
#         self.dropout = nn.Dropout(dropout)
#         self.ffn = nn.Sequential(
#             nn.Linear(d_model, ff_dim),
#             nn.GELU(),
#             nn.Linear(ff_dim, d_model),
#             nn.Dropout(dropout)
#         )

#     def forward(self, x, mask=None):
#         x = x + self.dropout(self.attn(self.ln1(x), mask))
#         x = x + self.dropout(self.ffn(self.ln2(x)))
#         return x


# # ─────────────────────────── 替换后的 TransformerLM 模型（手写版） ───────────────────────────
# class TransformerLM(nn.Module):
#     def __init__(self, vocab_size, embed_dim, hidden_dim, num_layers, nhead, dropout, max_seq_len):
#         super().__init__()
#         self.embed_dim = embed_dim
#         self.token_emb = nn.Embedding(vocab_size, embed_dim)
#         self.pos_emb = nn.Embedding(max_seq_len, embed_dim)
#         self.drop = nn.Dropout(dropout)

#         self.blocks = nn.ModuleList([
#             GPTBlock(d_model=embed_dim, n_head=nhead, ff_dim=hidden_dim, dropout=dropout)
#             for _ in range(num_layers)
#         ])
#         self.ln_final = nn.LayerNorm(embed_dim)
#         self.fc = nn.Linear(embed_dim, vocab_size)

#     def forward(self, x):
#         B, T = x.shape
#         tok_emb = self.token_emb(x)
#         pos = torch.arange(0, T, device=x.device).unsqueeze(0).expand(B, T)
#         pos_emb = self.pos_emb(pos)
#         e = self.drop(tok_emb + pos_emb)

#         # causal mask，挡住未来token
#         causal_mask = torch.triu(torch.ones(T, T, dtype=torch.bool, device=x.device), diagonal=1)

#         for block in self.blocks:
#             e = block(e, mask=causal_mask)

#         e = self.ln_final(e)
#         logits = self.fc(e)
#         return logits

# ─────────────────────────── 训练评估 ───────────────────────────
def run_epoch(model, loader, criterion, optimizer, device, train=True):
    model.train(train)
    total_loss = 0.0
    total_tokens = 0

    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        loss = criterion(logits.reshape(-1, logits.size(-1)), y.reshape(-1))

        if train:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        total_loss += loss.item() * y.numel()#y.numel()：这个batch里有多少个token
        total_tokens += y.numel()

    avg_loss = total_loss / total_tokens
    ppl = math.exp(avg_loss)#ppl的公式为：e的单个-log(预测概率)再取平均次方，这里avg_loss正好符合这个值，ppl越小预测值越准，比如结果时12，代表在这12个数里取一个
    return avg_loss, ppl 


# ─────────────────────────── 主函数，修改模型实例化，新增 transformer 参数 ───────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs",     type=int,   default=20)
    parser.add_argument("--seq_len",    type=int,   default=64)
    parser.add_argument("--stride",     type=int,   default=64, help="切块间隔：1=密集重叠，seq_len=不重叠")
    parser.add_argument("--batch_size", type=int,   default=128)
    parser.add_argument("--embed_dim",  type=int,   default=128)
    parser.add_argument("--hidden_dim", type=int,   default=256)
    parser.add_argument("--num_layers", type=int,  default=2)
    parser.add_argument("--nhead",      type=int,   default=4, help="多头注意力头数，embed_dim必须能被nhead整除")
    parser.add_argument("--dropout",    type=float, default=0.3)
    parser.add_argument("--lr",         type=float, default=1e-3)
    parser.add_argument("--val_ratio",  type=float, default=0.05)
    parser.add_argument("--max_vocab",  type=int,   default=4000, help="字表最大字数，0=不限")
    parser.add_argument("--seed",       type=int,   default=42)
    parser.add_argument("--corpus",     default=pattern)
    parser.add_argument("--save",       default="best_transformer.pt")
    args = parser.parse_args()

    #下面的随机取要按seed=42的方式取，即每次取/打乱的方式一样
    torch.manual_seed(args.seed)
    random.seed(args.seed)

    #有GPU用GPU，没有用CPU，因为正常电脑会默认走CPU，后面调用device表示这个调用的在GPU或CPU中运行
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}  model: Transformer Decoder-only")

    # 数据准备
    text = load_corpus(args.corpus)
    if not text:
        raise FileNotFoundError("未找到任何 .txt 文件，请确认路径正确。")
    print(f"语料字符数: {len(text):,}")

    char2idx, idx2char = build_vocab(text, args.max_vocab or None)
    vocab_size = len(char2idx)
    print(f"词表大小: {vocab_size}")

    lines = text.splitlines()
    random.shuffle(lines)
    split = int(len(lines) * (1 - args.val_ratio))
    train_text = "\n".join(lines[:split])
    val_text   = "\n".join(lines[split:])

    train_ds = CharDataset(train_text, char2idx, args.seq_len, args.stride)
    val_ds   = CharDataset(val_text,   char2idx, args.seq_len, args.stride)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, drop_last=True,num_workers=0)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size, shuffle=False, drop_last=True,num_workers=0)

    # 实例化 Transformer LM
    model = TransformerLM(
        vocab_size=vocab_size,
        embed_dim=args.embed_dim,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        nhead=args.nhead,
        dropout=args.dropout,
        max_seq_len=args.seq_len
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"模型参数量: {total_params:,}")

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    best_val_ppl = float("inf")

    print(f"\n{'Epoch':>6}  {'Train Loss':>10}  {'Train PPL':>10}  {'Val Loss':>10}  {'Val PPL':>10}")
    print("-" * 56)

    for epoch in range(1, args.epochs + 1):
        tr_loss, tr_ppl = run_epoch(model, train_loader, criterion, optimizer, device, train=True)
        with torch.no_grad():
            va_loss, va_ppl = run_epoch(model, val_loader, criterion, optimizer, device, train=False)#这里train=False等同于model.eval()

        marker = "  *" if va_ppl < best_val_ppl else ""
        if va_ppl < best_val_ppl:
            best_val_ppl = va_ppl
            torch.save({
                "model_state": model.state_dict(), #模型权重参数，后面加载模型要用
                "char2idx": char2idx, #字符→数字的映射表
                "idx2char": idx2char, #数字→字符的映射表，生成文本解码的时候必须用
                "args": vars(args), #本次训练所有超参数（batch、seq_len、层数、学习率等），方便复现实验
            }, args.save)

        print(f"{epoch:>6}  {tr_loss:>10.4f}  {tr_ppl:>10.2f}  {va_loss:>10.4f}  {va_ppl:>10.2f}{marker}")

    print(f"\n训练完成。最佳验证 PPL: {best_val_ppl:.2f}  已保存至 {args.save}")


if __name__ == "__main__":
    main()
