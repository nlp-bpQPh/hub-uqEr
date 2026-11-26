"""
使用了下载的模型chinese-roberta-wwm-ext并进行微调
其中微调的内容：
1：使用模型输出值后的池化：可选三种：cls,avg,max
2.对loss的计算：采用类别加权；Focal Loss(根据难度加权)；Label Smoothing(当有符合多个类别时加入，防止过度自信)
3.对学习率调整：采用AdamW；对预训练和后续随机初始化参数采用不同的学习率；学习率随着step采用先增大再减小的方式：0~1~0
4.再embedding层加入了FAG对抗训练
5.减小精度，释放内存
5.以上调整的内容可分别带入算损失，精确率，F1
"""
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
import argparse
import json
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from common.data import load_splits,load_clean_val
from common.labels import LABELS
from common.model_paths import bert_path

CODE_ROOT = Path(__file__).resolve().parent #当前py文件所在的完整路径的上一级，即当前py文件所在的文件夹
OUT_DIR = CODE_ROOT / "outputs"



class TnewsDataset(Dataset):
    """把 {"sentence","label"} 变成 BERT 的输入张量"""

    def __init__(self, data, tok, max_length=64):
        self.data, self.tok, self.max_length = data, tok, max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        d = self.data[i]
        enc = self.tok(d["sentence"], truncation=True, max_length=self.max_length,
                       padding="max_length", return_tensors="pt")
        #truncation=True：句子太长就截断，不超过 max_length
        #padding="max_length"：统一补 padding 到固定长度 max_length（不是动态 padding）
        #return_tensors="pt"：直接输出 pytorch tensor:(1,L)，而不是 list,不加后续需要自己torch.tensor，加了后续需要降维
        return {"input_ids": enc["input_ids"][0], #取 dim=0 这个维度上的第 0 个元素:(L)
                "attention_mask": enc["attention_mask"][0],
                "labels": torch.tensor(d["label"])}


class BertClassifier(nn.Module):
    """
    BERT + 池化 +  Dropout + 线性头。

    pool 三选一：
      cls  —— 取 [CLS] 向量（预训练就为它设计，但不一定最适合短文本）
      mean —— 对所有有效 token 取均值（短文本常更稳）
      max  —— 逐维取最大（抓最显著特征，噪声也大）
    """

    def __init__(self, model_dir, num_labels=15, pool="cls", dropout=0.1):
        super().__init__()
        from transformers import AutoModel
        self.bert = AutoModel.from_pretrained(str(model_dir))
        self.pool = pool
        hidden = self.bert.config.hidden_size          # bert-base = 768
        #config内容是model_dir的，可能包含：
#         {
#   "hidden_size": 768,                // 隐藏向量维度，你代码用到的！
#   "num_hidden_layers": 12,           // encoder层数
#   "num_attention_heads": 12,         // 多头注意力头数
#   "intermediate_size": 3072,         // FFN中间层维度（4×768）
#   "hidden_act": "gelu",              // 激活函数
#   "vocab_size": 21128,               // 词表大小，和vocab.txt对应
#   "max_position_embeddings": 512,    // 模型支持最大长度512
#   "type_vocab_size": 2,              // token_type_ids 只有0/1两类
#   "hidden_dropout_prob": 0.1,        // encoder内部dropout
#   "attention_probs_dropout_prob": 0.1,// attention分数dropout
#   "initializer_range": 0.02,         // 权重初始化标准差
#   "layer_norm_eps": 1e-12,
#   "model_type": "bert",
#   "pad_token_id": 0
# }
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden, num_labels)

    def forward(self, input_ids, attention_mask):
        out = self.bert(input_ids=input_ids, attention_mask=attention_mask, return_dict=True)
        #model_dir 里面的 config.json 写了 model_type: bert → 加载出来的是 BertModel 这个架构类
        #forward 能接收哪些参数，是 BertModel 这个类的代码定义的，不是 model_dir 里单独写的一份参数清单；
        #只要它是 BERT 架构，就天然支持 input_ids、attention_mask、token_type_ids 这些入参。
        #return_dict=True开启后，out 是字典式对象（ModelOutput），可以用 .last_hidden_state 这种名字取字段，
        #不写 / return_dict=False，返回的是 tuple，顺序是 (last_hidden_state, pooler_output, ...)，靠下标取，版本升级很容易踩坑
        #out输出重点字段：
        # out.last_hidden_state：shape:[B, L, hidden_size]，每个 token 对应的隐层向量
        # out.pooler_output：BERT 原生自带的 pooler，是把 CLS 再过一层线性 + tanh，很多人不用它，自己写 cls/mean/max 池化
        # out.hidden_states（需要开启 output_hidden_states=True 才会有）：每一层 encoder 的输出
        # out.attentions（需要开启 output_attentions=True）：注意力分数
        #输入attention_mask，当 attention_mask=0 时：
        #模型会把这个位置的注意力分数减去一个很大的负数（类似 -1e9），经过 softmax 之后，这个位置权重≈0，不会分到注意力
        hs = out.last_hidden_state                     # [B, L, 768]
        if self.pool == "cls":
            vec = hs[:, 0] 
            #等价于 hs[:, 0, :]
            # : → 取当维的全部 
            # 0 → 取当维的第 0 个 token，也就是 [CLS]
            # 结果：丢掉长度 L 那一维，输出 vec.shape = [B, H]
        else:
            m = attention_mask.unsqueeze(-1).float()   # unsqueeze(-1)：在原有的最后一维加一维：[B, L, 1]
            #attention_mask 读出来默认是 整数张量（dtype = torch.int64 /long）
            #hs（last_hidden_state）是 浮点张量（dtype = torch.float32）
            #整数 × 浮点，不是不能乘，但有坑；而且后面要做除法、求平均，必须是浮点，所以主动转 .float() 是稳妥写法
            if self.pool == "mean":
                vec = (hs * m).sum(1) / m.sum(1).clamp(min=1e-9) 
                #hs * m：输出shape:[B, L, 768],只是attention_mask L上为0的位置hs的L上也为0
                #sum(1)沿着第1维相加：(hs * m).sum(1)shape:[B, 768]
                #clamp(min=1e-9) 最小值为1e-9，防止除数为0
            else:                                      # max：PAD 位置压到极小值再取 max
                vec = hs.masked_fill(m == 0, -1e9).max(1).values
                #masked_fill(条件, 填充值)：条件为 True 的位置，把 hs 里面的值替换成 -1e9,即将m==0的位置换成-1e9
                #.max(1)就是沿着L（序列长度）维度取最大值
                #.max(dim=1) 返回的是一个 torch.return_types.max 命名元组，里面包含两部分：
                # values：最大值张量（就是你要的 vec）
                # indices：最大值所在的下标（哪个 token 取到了这个最大值）
        return self.classifier(self.dropout(vec))


class FGM:
    """
    FGM（Fast Gradient Method）对抗训练 —— 三步：
      ① 正常前向 + 反传，拿到参数的梯度 g；
      ② 在 **word_embeddings** 上沿梯度方向加一点扰动：r = eps * g / ||g||；
      ③ 用被扰动的权重再前向 + 反传一次，两次梯度一起更新，最后把权重还原。

    为什么只扰动 embedding：文本任务里「输入」就是 embedding，扰它等于在字的表示上加噪；
    扰动全部参数既慢又容易破坏预训练知识。为什么必须归一化：不除 ||g||，eps 的单位随梯度大小漂，
    同一个 eps 在不同 step 上的扰动强度完全不同。AMP 下用的是缩放后的梯度，方向不变，可以直接用。
    """

    def __init__(self, model, emb_name="word_embeddings", eps=1.0):
        self.model, self.emb_name, self.eps = model, emb_name, eps
        self.backup = {}

    def attack(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad and self.emb_name in name and param.grad is not None:
                #param本身是 Tensor 的子类，不止存权重数值，附带很多自动微分相关标记（除下方列举还有）
                #param.data：参数本身的数值张量（权重、embedding 矩阵）
                #param.requires_grad（布尔标记）：这个参数要不要算梯度、要不要被优化器更新
                #param.grad（梯度张量）：loss 对这个参数的导数，初始是 None，只有跑完 loss.backward() 之后才会被填充成张量，shape 和 param 完全一样
                self.backup[name] = param.data.clone()
                norm = torch.norm(param.grad) #算梯度的范数（向量平方之和再开根号）
                if norm != 0 and not torch.isnan(norm): #not torch.isnan(norm)：防止梯度里面已经出现 NaN，再计算就污染权重
                    param.data.add_(self.eps * param.grad / norm) #add_：原地加法，等价：param.data = param.data + (eps * grad / norm)FGM加扰动对抗公式

    def restore(self):
        for name, param in self.model.named_parameters():
            if name in self.backup:
                param.data = self.backup[name] #再将权重还原
        self.backup = {}


def focal_loss(logits, targets, weight=None, gamma=2.0): #logits：[B, C]，模型原始输出，targets：[B]，真实标签下标, weight：类别加权参数 gamma：按难度控制loss的参数
    """
    Focal Loss：把「已经学会的样本」的权重压下去，让梯度集中在难样本上。
        L = -alpha_t * (1 - p_t)^gamma * log(p_t)
    alpha 直接用类别权重（balanced）—— 和「类别加权」共用同一张表，不额外加超参。
    """
    logp = F.log_softmax(logits, dim=-1)#先 softmax，再取自然对数
    logp_t = logp.gather(1, targets.unsqueeze(1)).squeeze(1) 
    #targets.unsqueeze(1):(B)-(B,1),gather(dim=1, index=xxx) 的规则:ndex 的形状，必须和输入张量 shape 一样，或者可以广播对齐
    #logp.gather(1, targets.unsqueeze(1)):输出结果 shape：[B, 1],含义：每个样本，取出它真实标签对应的那一条 logp
    #.squeeze(1) 把多余的维度删掉，变回 [B],最后 logp_t.shape = [B]，正好是每个样本的 log(pt)，方便后面计算 loss
    p_t = logp_t.exp()
    alpha = weight[targets] if weight is not None else 1.0 #weight[targets]：weight一维张量，返回一个一维张量[B] 
    return (-alpha * ((1.0 - p_t) ** gamma) * logp_t).mean() #将一组样本，同时采用类别权重和Focal Loss，再求平均,手动算loss


@torch.no_grad() #评估阶段核心装饰器，关闭计算图、不记录梯度，节省显存、加速推理
def evaluate(model, loader, device):
    from sklearn.metrics import accuracy_score, f1_score
    model.eval()
    ys, ps, logits_all = [], [], []
    for batch in loader:
        logits = model(batch["input_ids"].to(device), batch["attention_mask"].to(device)) #[B, num_classes] ,每一行是各个类别的得分
        ps.extend(logits.argmax(-1).cpu().tolist()) #在最后一维找最大值下标 → 得到预测类别 id,转成 python 列表,并把这一批的多个结果追加进总列表（append 会把整个 batch 当成一个元素）
        ys.extend(batch["labels"].tolist())
        logits_all.append(logits.cpu())
    return {
        "acc": round(float(accuracy_score(ys, ps)), 4), #准确率：正确预测的样本数量/全部样本数量
        "macro_f1": round(float(f1_score(ys, ps, average="macro")), 4), #macro：宏 F1，每个类别单独算 F1，直接取平均，类别不均衡场景更适合, F1=(2*精确率*召回率)/(精确率+召回率)
                                                                        #精确率:所有预测成i的，实际有多少是i;召回率:所有真实是i的，有多少预测成了i
        "logits": torch.cat(logits_all), #(B*循环的轮数,num_classes)
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pool", default="cls", choices=["cls", "mean", "max"]) #池化方式
    ap.add_argument("--epochs", type=int, default=3) #训练轮数
    ap.add_argument("--batch_size", type=int, default=64) #一次传入的样本数
    ap.add_argument("--lr", type=float, default=2e-5, help="BERT 层学习率，越小越保守") #学习率
    ap.add_argument("--head_lr_mult", type=float, default=5.0, help="分类头 lr = BERT lr × 该倍数") #分类头（即最后的线性层，非预训练部分）学习率放大倍数
    ap.add_argument("--max_length", type=int, default=64) #L的最大长度
    ap.add_argument("--no_class_weight", action="store_true") #命令行不加这个参数：开启类别权重，加上：不用类别权重
    # ↓ 四个训练技巧（实测在本任务上都不涨，保留是为了让学生自己复现/在别的任务上试）
    ap.add_argument("--label_smoothing", type=float, default=0.0, help="软标签 ε，常用 0.05~0.1") #加上：防止过度自信
    ap.add_argument("--focal_gamma", type=float, default=0.0, help="Focal Loss 的 γ，>0 才启用（与类别加权共用权重）") #传大于 0 的值（比如 2.0），才真正启用 Focal，压制简单样本
    ap.add_argument("--fgm", action="store_true", help="打开 FGM 对抗训练（训练时间约 ×1.5）") #FGM 总开关，不加：跳过对抗，加：实例化 FGM 类
    ap.add_argument("--fgm_eps", type=float, default=1.0, help="FGM 扰动强度 ε") #FGM 的扰动强度 ε，只有--fgm生效时才会用到，默认 1.0
    # ↓ 冒烟/调试用：不改变默认行为
    ap.add_argument("--max_steps", type=int, default=0, help="每个 epoch 最多跑多少步，0=全跑（冒烟用）")
    ap.add_argument("--max_eval", type=int, default=0, help="验证集最多评多少条，0=全评（冒烟用）")
    ap.add_argument("--no_save", action="store_true", help="不落盘 checkpoint（冒烟用，省 390 MB）")
    args = ap.parse_args()

    from transformers import AutoTokenizer

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(str(bert_path()))
    #bert_path()返回 BERT 模型文件夹路径（本地）或者模型名
    #一个标准 BERT 文件夹，里面两类文件：
#     bert/
# ├── pytorch_model.bin       # 模型权重（神经网络参数，用来做编码、预测）
# ├── config.json             # 模型结构配置（层数、头数、隐藏维度）
# ├── vocab.txt               # 【分词器核心】词表！
# ├── tokenizer_config.json   # 【分词器配置】是否大小写、特殊token、截断规则
# └── special_tokens_map.json # 特殊token：[CLS] [SEP] [PAD] [UNK]
    #AutoTokenizer:分词器只读取：vocab.txt + tokenizer_config.json + special_tokens_map.json
    #模型（AutoModel这里后续用到的模型）读取：pytorch_model.bin + config.json
    # text = "今天天气很好"
    # out = tok(text)
    # print(out)  
    #输出：
    # {
    # 'input_ids': [101, 791, 1921, 1921, 3698, 2548, 1962, 102],
    # 'token_type_ids': [0, 0, 0, 0, 0, 0, 0, 0],标记这个 token 属于第 1 句，还是第 2 句
    # 'attention_mask': [1, 1, 1, 1, 1, 1, 1, 1]标记这个是真实要用（展示1）的还是padding补0,若是补0展示0
    # }
    #101 = [CLS]，102 = [SEP]，是 BERT 内置特殊标记
    #
    #
    #train, val = load_splits() #加载训练集和验证集
    train, _ = load_splits()
    val, meta_val = load_clean_val(n=None)

    if args.max_eval:
        val = val[: args.max_eval]

    train_dl = DataLoader(TnewsDataset(train, tok, args.max_length), batch_size=args.batch_size, shuffle=True)
    #shuffle=True：每个 epoch 开始前打乱样本顺序（训练集一定要开；验证集 val_dl 一般设 shuffle=False）
    val_dl = DataLoader(TnewsDataset(val, tok, args.max_length), batch_size=args.batch_size)

    model = BertClassifier(bert_path(), num_labels=len(LABELS), pool=args.pool).to(device)

    # 分层学习率：头是随机初始化的，需要学得更快；BERT 要小步，否则冲掉预训练知识
    head_params = list(model.classifier.parameters()) + list(model.dropout.parameters()) #列表相加，组成一个大列表
    bert_params = [p for n, p in model.named_parameters() if not n.startswith("classifier") and not n.startswith("dropout")]
    opt = torch.optim.AdamW([
        {"params": bert_params, "lr": args.lr},
        {"params": head_params, "lr": args.lr * args.head_lr_mult},
    ], weight_decay=0.01)
    #model.classifier：就是最后的线性层 nn.Linear(768, num_labels)
    #model.dropout：nn.Dropout，Dropout 层本身不带可训练参数，list(model.dropout.parameters()) 是空列表
    #model.named_parameters()：遍历整个模型所有参数，同时返回 参数名n 和 参数p，parameters()：只会返回参数p
    #n.startswith("classifier"):参数名n以classifier开头
    #bert_params过滤规则：名字不以 classifier 开头，并且不以 dropout 开头 → 剩下全部是 bert 里面的参数
    #w = w(1 - 学习率 * weight_decay) - 学习率 * 梯度
    #AdamW 里 bert_params/head_params 对应的参数已经默认了是哪一部分的参数，所以不用传参数名告诉是哪个部分的参数
    #weight_decay的作用：每次更新参数时，额外给权重一个往 0 拉的拉力，抑制权重变得过大，用来缓解过拟合

    # 类别加权：23× 不均衡，不加权时尾部类 recall 会趋近 0
    if args.no_class_weight:
        weight = None
    else:
        from sklearn.utils.class_weight import compute_class_weight
        import numpy as np
        cw = compute_class_weight("balanced", classes=np.arange(len(LABELS)), y=[d["label"] for d in train])
        # compute_class_weight用法：
        # 第一个参数：class_weight="balanced"
        # 可选值只有两种：
        # "balanced"：总样本数/(类别的样本数量*类别数量)
        # None：全部权重 = 1，返回全 1 数组
        #  第二个参数 classes：不是必须传，但强烈建议传
        # 不传 classes：sklearn 自动从 y 里扫描出现过的标签，只返回 y 中存在的类别权重。
        # 坑：如果某个类别在训练集一条样本都没有，返回的 cw 数组长度会变短，后面传给 CrossEntropyLoss 直接维度不匹配报错。
        # 写 classes=np.arange(len(LABELS))：主动告诉它全部合法类别编号是 0,1,2…C-1。哪怕某一类在 y 里没有样本，输出数组长度依然等于总类别数；
        # 但是！如果某类数量 =0，公式分母 = 0，直接抛异常，这个要自己提前处理。
        #第三个参数 y：标签数组，来源规则固定：只能用训练集标签
        #cw返回numpy 一维数组 
        weight = torch.tensor(cw, dtype=torch.float).to(device) #torch.tensor返回张量，不加维度

    # 损失函数：三个旋钮选一个（Focal 与 Label Smoothing 不叠加，Focal 优先）
    if args.focal_gamma > 0:
        if args.label_smoothing > 0:
            print("[warn] 同时给了 --focal_gamma 和 --label_smoothing，按 Focal 跑（两者都是在改目标分布）", flush=True)
        crit = lambda logits, y: focal_loss(logits, y, weight, args.focal_gamma)   # noqa: E731
        #lambda 参数列表: 返回表达式，lambda是 Python 自带的一个关键字，用来快速定义匿名小函数，即这个if分支后续使用crit时，需要传入logits, 此时自动返回focal_loss这个函数的返回值
        # 造一个匿名函数，把这个函数对象赋值给 crit。
        # 此时不会运行 focal_loss(...)！
        # 这里的 logits, y 只是写在函数的参数列表上，只是一张 “申请表”：以后调用我的时候，给我两个东西。
        crit_desc = f"Focal(gamma={args.focal_gamma})" #生成一个字符串
    else:
        crit = nn.CrossEntropyLoss(weight=weight, label_smoothing=args.label_smoothing)
        #nn.CrossEntropyLoss是类
        # ① 实例化（这里只能放配置：weight、label_smoothing、reduction……）
        # loss_fn = nn.CrossEntropyLoss(weight=weight, label_smoothing=0.1)
        # ② 调用实例，这里才传数据 logits、y
        # loss = loss_fn(logits, y)
        #F.cross_entropy（函数）一步直接传所有参数，当场算出 loss，没有 “提前保存计算器” 这一步
        #loss = F.cross_entropy(logits, y, weight=weight, label_smoothing=0.1)
        #原生 CrossEntropy 不支持 Focal 的调制因子,所以需要用时调用上面那个手写的计算loss函数
        crit_desc = "CE" + (f"+LS({args.label_smoothing})" if args.label_smoothing > 0 else "")

    steps_per_epoch = min(len(train_dl), args.max_steps) if args.max_steps else len(train_dl) #train_dl 经过了 DataLoader，所以 len (train_dl)是一轮一共有多少组
    total_steps = steps_per_epoch * args.epochs
    warmup = int(total_steps * 0.1)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda step: step / max(1, warmup) if step < warmup else max(0.0, (total_steps - step) / max(1, total_steps - warmup)))
    #LambdaLR(optimizer, lr_lambda)：opt：绑定优化器；lr_lambda：一个函数，输入当前 step，输出缩放倍数
    #前0.1，系数变化0~1，后0.9，系数变化1~0
    scaler = torch.amp.GradScaler("cuda", enabled=(device == "cuda"))   # AMP：显存约降 40%，训练更快
    #torch.amp.GradScaler(device_type, enabled=True)
    #device_type：只能写 "cuda"，enabled：布尔开关，控制这个 scaler 要不要干活
    #一部分计算用 FP16（半精度），一部分用 FP32（单精度）
    fgm = FGM(model, eps=args.fgm_eps) if args.fgm else None

    print(f"[cfg] pool={args.pool} loss={crit_desc} class_weight={'off' if weight is None else 'on'} "
          f"fgm={'on(eps=%s)' % args.fgm_eps if fgm else 'off'} epochs={args.epochs} "
          f"steps/epoch={steps_per_epoch} eval_n={len(val)} save={'off' if args.no_save else 'on'}", flush=True)

    best, log = 0.0, []
    for epoch in range(1, args.epochs + 1):
        model.train()
        t0, running, n_step = time.time(), 0.0, 0
        for i, batch in enumerate(train_dl):
            ids = batch["input_ids"].to(device)  #([B, L])
            mask = batch["attention_mask"].to(device) #([B, L])
            y = batch["labels"].to(device) #([B])

            opt.zero_grad(set_to_none=True) #清空梯度 set_to_none=True：直接把 param.grad = None，释放这部分显存，显存更省
            with torch.autocast("cuda", dtype=torch.float16, enabled=(device == "cuda")):
                loss = crit(model(ids, mask), y) #torch.autocast也是降精度降现存，只作用于前向算子，只作用于with里面的位置，和scaler = torch.amp.GradScaler()联合使用一起将显存
            scaler.scale(loss).backward() #更新梯度，scaler.scale(loss) 就是专门对 loss 生效的，它的作用就是把 loss 乘上放大系数，然后再 backward
                                          #乘上放大系数因为 torch.autocast降了精度

            if fgm is not None:
                # 对抗训练：先按梯度扰动 embedding，再前向一次、反传一次（两次梯度累加后一起更新）
                fgm.attack()
                with torch.autocast("cuda", dtype=torch.float16, enabled=(device == "cuda")):
                    loss_adv = crit(model(ids, mask), y)
                scaler.scale(loss_adv).backward() 
                fgm.restore()

            scaler.step(opt) #把放大的梯度除以缩放系数，还原梯度，然后执行 AdamW 更新参数
            scaler.update() #调整下一轮要用的缩放系数 k，不影响这一轮的参数更新,如果刚才成功：适当增大 k，争取下一轮继续用更大的缩放，不容易下溢,如果刚才溢出了：缩小 k，避免下一轮继续爆溢出
            sched.step() #使用上面定义的sched更新参数，每调用一次，step加1
            running += loss.item() #一组一组累加loss
            n_step += 1 #更新样本组数
            if i % 200 == 0:
                print(f"  epoch {epoch} step {i}/{steps_per_epoch} loss={loss.item():.4f}", flush=True)
            if args.max_steps and n_step >= args.max_steps:
                break

        res = evaluate(model, val_dl, device)
        log.append({"epoch": epoch, "train_loss": round(running / max(1, n_step), 4),
                    "val_acc": res["acc"], "val_macro_f1": res["macro_f1"],
                    "elapsed_s": round(time.time() - t0, 1)})
        print(f"epoch {epoch}: val_acc={res['acc']} val_f1={res['macro_f1']}", flush=True)

        if res["macro_f1"] > best and not args.no_save:
            best = res["macro_f1"]
            OUT_DIR.mkdir(parents=True, exist_ok=True) #没有outputs文件夹，自动创建
            torch.save({"state_dict": model.state_dict(), "pool": args.pool,
                        "val_acc": res["acc"], "val_macro_f1": res["macro_f1"], "args": vars(args)},
                       OUT_DIR / "bert_ckpt.pt")
            #`vars(args)` 把 argparse 对象转成字典，保存全部训练参数（学习率、batch、epoch、模型版本等），方便复现
            #OUT_DIR / "bert_ckpt.pt"存储的文件路径

    if not args.no_save:
        (OUT_DIR / "bert_train_log.json").write_text(json.dumps(log, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"最佳 val Macro-F1 = {best} → {OUT_DIR / 'bert_ckpt.pt'}")
    else:
        print("--no_save：本次不落盘（冒烟模式），ckpt 与日志都没写")
    #- `.write_text()`：pathlib 自带方法，直接写入文本，不用 `open()`
    # `json.dumps(log, ensure_ascii=False, indent=2)`
    # `log`：字典，里面存训练日志（每个 epoch 的 acc、f1、loss 等）
    #  `ensure_ascii=False`：**支持中文，不会变成 \uXXXX 编码**
    # `indent=2`：json 格式化，缩进 2 空格，方便人看
    # `encoding="utf-8"`：指定 utf8 编码，避免中文乱码


if __name__ == "__main__":
    main()


"""
1.直接跑：
采用类别加权，cls 池化，学习率采用 AdamW；对预训练和后续随机初始化参数采用不同的学习率；学习率随着 step 采用先增大再减小的方式：0~1~0
[cfg] pool=cls loss=CE class_weight=on fgm=off epochs=3 steps/epoch=834 eval_n=10000 save=on
  epoch 1 step 0/834 loss=2.8188
  epoch 1 step 200/834 loss=1.5746
  epoch 1 step 400/834 loss=1.1630
  epoch 1 step 600/834 loss=0.9992
  epoch 1 step 800/834 loss=1.3937
epoch 1: val_acc=0.5542 val_f1=0.5461
  epoch 2 step 0/834 loss=1.5773
  epoch 2 step 200/834 loss=1.2503
  epoch 2 step 400/834 loss=0.9792
  epoch 2 step 600/834 loss=1.1430
  epoch 2 step 800/834 loss=1.0539
epoch 2: val_acc=0.5588 val_f1=0.5474
  epoch 3 step 0/834 loss=0.9636
  epoch 3 step 200/834 loss=0.9650
  epoch 3 step 400/834 loss=0.9430
  epoch 3 step 600/834 loss=0.9256
  epoch 3 step 800/834 loss=0.8223
epoch 3: val_acc=0.5651 val_f1=0.5573
最佳 val Macro-F1 = 0.5573 → D:/AI/week6/outputs/bert_ckpt.pt
2.修改池化：
python bert.py --pool mean
python bert.py --pool max

3.加FGM
python bert.py --fgm

4.加focal_gamma
python bert.py --focal_gamma 2.0

5.修改max_length
python bert.py --max_length 32

###########
冒烟快速调试（任意组合后面追加）
--epochs 1 --max_eval 2000 --no_save
"""