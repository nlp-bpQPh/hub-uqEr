"""
方法五：Qwen3-0.6B 指令微调（LoRA 与全量微调两种模式）

教学重点：
  1. SFT 的关键不是模型，而是三件事：**chat 模板、Loss Masking、数据格式**。
  2. Loss Masking：prompt 部分 labels 全设 -100（不算 loss），只在答案 token 上算；
     不 mask 的话模型学的是「复述模板」，不是「答标签」。
  3. 两种微调模式的分工：
       LoRA（默认）     只训低秩旁路，2.29M / 596M = 0.38%，显存约 3 GB，adapter 9 MB
       全量微调 --full_ft  596M 参数全更新，显存 6~8 GB（8G 卡要开梯度检查点），权重 1.2 GB
     实测在这个 15 类任务上全量与 LoRA 的差距很小（都约 0.48–0.49），
     但成本差一个量级 —— 这就是「什么时候不值得全量微调」的答案。
  4. Qwen3 必须 enable_thinking=False，且训练与推理的模板必须完全一致。

用法：
  python llm_lora/train_lora.py                                   # LoRA，5000 条 × 2 epoch
  python llm_lora/train_lora.py --num_train 1000 --epochs 1        # 课堂快速版
  python llm_lora/train_lora.py --full_ft --num_train 3000 --epochs 2   # 全量微调
  python llm_lora/train_lora.py --max_steps 20                     # 冒烟测流程

####################
启用大模型微调
数据传输样本格式：如下方代码所示
若想减小可调参数量，采用LORA模式：对大模型的所有线性层都可另外加一个降维矩阵A，恢复维度的矩阵B，具体在哪个线性层加可自己控制
线性层新加LORA的计算方式：h=wx+(缩放系数/低秩矩阵 A、B 的内部维度)*B*A*x
可显著降低显存，并可以将LORA部分单独保存传输
"""
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
import argparse
import json
import sys
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from common.data import load_splits,load_clean_val
from common.labels import LABELS
from common.model_paths import qwen3_path

CODE_ROOT = Path(__file__).resolve().parent
OUT_DIR = CODE_ROOT / "outputs"

SYSTEM = ("你是一个新闻标题分类助手。请将给定的新闻标题分类到以下类别之一，"
          "只输出类别名称，不要输出任何其他内容。\n可选类别：" + "、".join(LABELS))


class SFTDataset(Dataset):
    """
    把分类样本转成 chat 格式的 SFT 样本，即符合大模型输入的样本。

    labels 的构造是本文件最核心的一行：
      prompt 部分 = -100（CrossEntropyLoss 会忽略 -100）
      答案部分   = 真实 token id（只有这里产生梯度）
    """

    def __init__(self, data, tok, max_length=128):
        self.data, self.tok, self.max_length = data, tok, max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        d = self.data[i]
        answer = LABELS[d["label"]] #取第 i 条样本；`LABELS` 是映射表，把数字 label（0/1/2…）变成类别文本（比如 “体育”、“科技”）
        msgs = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": d["sentence"]}]
        #拼成对话消息列表：system 提示词 + 用户输入新闻标题，符合 Qwen 的 chat 模板格式
        # tokenize=False 先拿文本：transformers 5.x 里 tokenize=True 返回 BatchEncoding，不好拼
        prompt = self.tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False)
        #`apply_chat_template`,：**按 Qwen 规定的 chat 模板拼接字符串**（加 <|im_start|>、<|im_end|> 这些特殊标记）,前提：模型`tokenizer.chat_template is not None`，才能直接调用 `apply_chat_template`
        #`tokenize=False`：只返回文本，不转 token id
        #`add_generation_prompt=True`：末尾自动加上 assistant 的起始标记，告诉模型「接下来该模型输出了」
        #`enable_thinking=False`：关闭 deepseek/qwen 的思考标记
        p_ids = self.tok(prompt, add_special_tokens=False)["input_ids"]  #prompt（system+user）这一整段的 token id ，input_ids：把字符串转成**整数 token 编号列表**
                                                                         #add_special_tokens控制不要自动追加 tokenizer 自带的开头 / 结尾特殊标记（`<s>`、`</s>`这类
                                                                         #`prompt` 已经由 `apply_chat_template` 拼好了完整对话文本，里面已经包含模型需要的 `<|im_start|> <|im_end|>` 这些标记了；
                                                                         #如果这里再开 `add_special_tokens=True`，会**重复多加 bos/eos，标记错乱
        a_ids = self.tok(answer, add_special_tokens=False)["input_ids"] + [self.tok.eos_token_id] #答案文本转 token，**末尾追加 EOS 结束符**，教会模型什么时候一句话结束
        input_ids = (p_ids + a_ids)[: self.max_length]  #把【prompt token】+【答案 token】拼在一起，超长就截断。
        labels = ([-100] * len(p_ids) + a_ids)[: self.max_length]  #prompt 对应的位置全部填 `-100` → loss 忽略，不更新参数
        return {"input_ids": torch.tensor(input_ids), "labels": torch.tensor(labels)}
        #算loss时：
        #shift_logits = logits[:, :-1, :]:去掉最后一个token的logits
        #shift_labels = labels[:, 1:]:label从1开始取，丢掉label[0]
        #loss = F.cross_entropy(shift_logits.reshape(-1, vocab_size), shift_labels.reshape(-1), ignore_index=-100)

def collate(batch, pad_id):
    """手写 collate：input_ids 用 pad 补齐，labels 用 -100 补齐（PAD 不产生 loss）"""
    L = max(len(b["input_ids"]) for b in batch) #L = 当前 batch 最大序列长度
    ids = torch.full((len(batch), L), pad_id, dtype=torch.long) #将len(batch)行L列的矩阵都用pad_id填充
    lab = torch.full((len(batch), L), -100, dtype=torch.long)
    att = torch.zeros((len(batch), L), dtype=torch.long) #将len(batch)行L列的矩阵都用0填充
    for i, b in enumerate(batch):
        n = len(b["input_ids"])
        ids[i, :n] = b["input_ids"] #将样本长度不够的位置保留pad_id
        lab[i, :n] = b["labels"]  #将样本长度不够的位置保留-100
        att[i, :n] = 1 ##将样本长度不够的位置保留0，其余为1
    return {"input_ids": ids, "labels": lab, "attention_mask": att}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--num_train", type=int, default=5000)
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--batch_size", type=int, default=4, help="全量微调建议 1~2")
    ap.add_argument("--grad_accum", type=int, default=4)
    ap.add_argument("--lr", type=float, default=None, help="不填则 LoRA=2e-4、全量=2e-5")
    ap.add_argument("--lora_r", type=int, default=8)
    ap.add_argument("--lora_alpha", type=int, default=16)
    ap.add_argument("--full_ft", action="store_true", help="全量微调；不加则 LoRA")
    ap.add_argument("--gradient_checkpointing", action="store_true", help="省显存（全量微调在 8G 卡上必开）")
    ap.add_argument("--max_length", type=int, default=128)
    ap.add_argument("--max_steps", type=int, default=0, help=">0 时只跑这么多步（冒烟测试）")
    args = ap.parse_args()

    from transformers import AutoModelForCausalLM, AutoTokenizer

    device = "cuda" if torch.cuda.is_available() else "cpu"
    # 全量微调：用 bf16 训练（fp16 + Adam 容易溢出）；LoRA：fp16 足够
    dtype = torch.bfloat16 if (args.full_ft and device == "cuda") else (
        torch.float16 if device == "cuda" else torch.float32)  #bfloat16/float16/float32:决定了梯度张量能表示的数值范围（上限、下限）
    tok = AutoTokenizer.from_pretrained(str(qwen3_path()))
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    #Qwen3.special_tokens_map.json
    # {
    # "eos_token": "<|im_end|>",
    # "bos_token": "<|im_start|>",
    # "unk_token": "<|endoftext|>"
    # }
    #DeepSeek.special_tokens_map.json
    # {
    # "bos_token": "<｜begin▁of▁sentence｜>",
    # "eos_token": "<｜end▁of▁sentence｜>"
    # }

    base = AutoModelForCausalLM.from_pretrained(str(qwen3_path()), dtype=dtype)
    #AutoModelForCausalLM:加载带自回归 LM 头（放在 Transformer 主干的最后一层的线性层）、自带因果掩码（下三角 mask）的 Transformer,输出：[batch, seq_len, vocab_size]）
    #AutoModel：只加载 Transformer 主干，**不带任务头，也不会自动加因果掩码**

    if args.full_ft:
        model = base.to(device)                     # 所有参数默认 requires_grad=True
        if args.gradient_checkpointing:
            # 梯度检查点：用时间换显存，激活值不再常驻（全量微调在 8G 卡上基本必开）
            model.gradient_checkpointing_enable()
            #正常前向传播：
            #每一层的中间激活值（activation）全部存在显存里，反向传播的时候直接拿过来算梯度。
            #优点：快；缺点：显存占用巨大，大模型全量微调很容易 OOM 爆显存。
            #开梯度检查点之后：
            #前向时**不保存大部分中间激活**，只保存少量关键节点；
            #等到反向传播的时候，**重新跑一遍前向，现场算出需要的激活值**，再算梯度。
            #收益：显存大幅下降（8G 卡全量微调 Qwen 基本必须开）
            #代价：训练速度变慢，多了一遍前向计算
            model.enable_input_require_grads()
            #梯度检查点开启后，梯度检查点的内部逻辑默认把输入当成不需要梯度的张量（`requires_grad=False`）
            #哪怕它可以重算出中间层激活，**梯度传到输入这一步就断了**
            #model.enable_input_require_grads()：补丁 —— 把输入的 `requires_grad` 打通，让重算出来的梯度能正常回传
            
        lr = args.lr or 2e-5                        # 全量更新必须用更小的 lr，否则冲掉预训练知识
        print(f"[full-ft] 参数量 {sum(p.numel() for p in model.parameters())/1e6:.1f}M（全部可训练），lr={lr}")
    else:
        from peft import LoraConfig, TaskType, get_peft_model #LoraConfig:LoRA 的超参配置类; TaskType:声明任务大类; get_peft_model:给基座模型挂上 LoRA 旁路的函数
        cfg = LoraConfig(task_type=TaskType.CAUSAL_LM, r=args.lora_r, lora_alpha=args.lora_alpha,
                         lora_dropout=0.05, target_modules=["q_proj", "k_proj", "v_proj", "o_proj"])
        #task_type=TaskType.CAUSAL_LM:因果语言模型，就是自回归 decoder 模型（Qwen、DeepSeek、Llama）,如果是 BERT 分类，这里用 SEQ_CLS
        #r=args.lora_r:LoRA 秩，低秩矩阵 A、B 的内部维度；r 越大，旁路容量越大，可学的信息越多，参数量轻微上涨
        #lora_alpha=args.lora_alpha:缩放系数：scale = alpha / r，旁路输出 × 这个 scale。它是控制旁路贡献强度的旋钮
        #lora_dropout=0.05:LoRA 分支内部的 dropout，随机把一部分旁路输出置 0，用来防止过拟合；小数据集可以开，大数据集也可以设 0
        #target_modules=["q_proj", "k_proj", "v_proj", "o_proj"]:指定哪些线性层上面挂 LoRA
        model = get_peft_model(base, cfg).to(device) 
        lr = args.lr or 2e-4
        # 一定要打印：target_modules 写错名字不会报错，只会「训了等于没训」
        model.print_trainable_parameters()
        #PEFT 自带打印函数，输出类似：
        #trainable params: 2.29M || all params: 596M || trainable%: 0.38%
        #trainable params：LoRA 新增的 A、B 低秩矩阵参数（就是这次要训练的参数）
        #all params：整个包装后 PeftModel 里面全部参数总和 = 基座原有全部参数 + LoRA 新增参数
        #trainable% = 可训练参数 ÷ 全部参数

    #train, val = load_splits()
    train, _ = load_splits()
    val, meta_val = load_clean_val(n=None)
    train = train[: args.num_train]
    #val = val[: min(500, len(val))] 
    val = val[: len(val)] 
    train_dl = DataLoader(SFTDataset(train, tok, args.max_length), batch_size=args.batch_size,
                          shuffle=True, collate_fn=lambda b: collate(b, tok.pad_token_id))
    #DataLoader 会从 Dataset 取 `batch_size` 条样本，拼成一个列表，DataLoader 调用lambda的时候，自动把这个样本列表传给 `b`
    #shuffle=True:每个 epoch 开始前打乱样本顺序，防止模型记住样本顺序，提升泛化。**训练集一般开 True；验证集 shuffle=False**
    val_dl = DataLoader(SFTDataset(val, tok, args.max_length), batch_size=args.batch_size,
                        collate_fn=lambda b: collate(b, tok.pad_token_id))

    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=lr) #上面建立模型：get_peft_model(base, cfg) 内部，自动设置好 requires_grad（只有新加的 LoRA A/B 矩阵：`requires_grad=True`）
    scaler = torch.amp.GradScaler("cuda", enabled=(device == "cuda" and dtype == torch.float16))
    log, step = [], 0
    t_start = time.time()

    for epoch in range(1, args.epochs + 1):
        model.train()
        running, n_batch = 0.0, 0
        for i, batch in enumerate(train_dl):
            batch = {k: v.to(device) for k, v in batch.items()}
            with torch.autocast("cuda", dtype=torch.float16, enabled=(device == "cuda" and dtype == torch.float16)):
                loss = model(**batch).loss / args.grad_accum     # 计算loss是HuggingFace AutoModelForCausalLM 这类模型自带默认行为：model(**batch).loss
            scaler.scale(loss).backward()  #多次 forward+backward，一次 step，显存更小
            if (i + 1) % args.grad_accum == 0: #每grad_accum的倍数位置在更新权重
                scaler.step(opt)
                scaler.update()
                opt.zero_grad(set_to_none=True) #把 .grad 直接赋值成 None，而不是填充 0 张量，好处：省显存，大模型 / QLoRA 推荐用
                                                #由于上面是梯度累加的形式，这里后清0。正常先清0：防止由于中间过程崩了算不到最后的梯度清0导致下一次循环没有清0
                step += 1
            running += loss.item() * args.grad_accum
            n_batch += 1
            if i % 100 == 0:
                print(f"  epoch {epoch} {i}/{len(train_dl)} loss={loss.item()*args.grad_accum:.4f}", flush=True)
            if args.max_steps and step >= args.max_steps:
                break

        model.eval()
        vloss, vn = 0.0, 0
        with torch.no_grad():
            for batch in val_dl:
                batch = {k: v.to(device) for k, v in batch.items()}
                vloss += model(**batch).loss.item()
                vn += 1
        log.append({"epoch": epoch, "train_loss": round(running / max(1, n_batch), 4),
                    "val_loss": round(vloss / max(1, vn), 4), "steps": step,
                    "elapsed_s": round(time.time() - t_start, 1),
                    "peak_mem_gb": round(torch.cuda.max_memory_allocated() / 1024**3, 2) if device == "cuda" else None})
        #torch.cuda.max_memory_allocated()：PyTorch CUDA 峰值分配显存（allocated，模型 / 张量占用，不含驱动、CUDA 上下文等），返回单位：字节 Byte
        print(f"epoch {epoch}: train_loss={log[-1]['train_loss']} val_loss={log[-1]['val_loss']}", flush=True)
        if args.max_steps:
            break

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    if args.full_ft:
        # 全量微调存整个权重（约 1.2 GB）；交付件里不带，学生自己跑出来
        model.save_pretrained(str(OUT_DIR / "full_ft_ckpt"),safe_serialization=True)
        #model.save_pretrained存法更符合Lora模式
        #自动识别是不是 PeftModel（LoRA），只导出可训练的 LoRA 权重，自动生成 adapter_config.json
        # 自动保存 config.json、generation_config，分片、safetensors、权重 key 对齐全部帮你处理
        # 输出是标准 hf 目录结构，直接 from_pretrained / PeftModel.from_pretrained 加载，vLLM、trl、hub 全都兼容
        log_path = OUT_DIR / "full_ft_train_log.json"
        print(f"全量权重 → {OUT_DIR / 'full_ft_ckpt'}")
    else:
        model.save_pretrained(str(OUT_DIR / "lora_adapter"),safe_serialization=True)   # 只存 adapter，9 MB 左右
        log_path = OUT_DIR / "lora_train_log.json"
        print(f"adapter → {OUT_DIR / 'lora_adapter'}")

    log_path.write_text(json.dumps(log, ensure_ascii=False, indent=2), encoding="utf-8") #将上面储存的内容做一个以 json 格式写入的动作


if __name__ == "__main__":
    main()


#python lora.py --num_train 5000 --epochs 2 --batch_size 4 --grad_accum 4 --lora_r 8 --lora_alpha 16 --max_length 128
"""
trainable params: 2,293,760 || all params: 598,343,680 || trainable%: 0.3834
  epoch 1 0/1250 loss=1.3971
  epoch 1 100/1250 loss=1.5050
  epoch 1 200/1250 loss=1.3241
  epoch 1 300/1250 loss=0.4489
  epoch 1 400/1250 loss=0.6367
  epoch 1 500/1250 loss=0.5617
  epoch 1 600/1250 loss=0.7598
  epoch 1 700/1250 loss=0.8780
  epoch 1 800/1250 loss=1.0750
  epoch 1 900/1250 loss=0.6272
  epoch 1 1000/1250 loss=0.6218
  epoch 1 1100/1250 loss=0.5959
  epoch 1 1200/1250 loss=1.0482
epoch 1: train_loss=0.7977 val_loss=0.7314
  epoch 2 0/1250 loss=0.8328
  epoch 2 100/1250 loss=0.4656
  epoch 2 200/1250 loss=0.6337
"""