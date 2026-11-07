# 任务描述：训练基于transformer的语言模型
# 其中 diy_transformer_block 为演示一个block 的计算过程
# model 为实际测试效果
# compare 为对比手搓与实际bert的transformer-block 的输出结果

import os
import torch
from typing import List, Dict, Tuple
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertModel, BertTokenizer


class BERT_BASE_CONFIG:
    hidden_size = 768
    num_hidden_layers = 12
    num_attention_heads = 12
    hidden_size_per_attenion_head = 64
    intermediate_size = 3072


def diy_transformer_block():
    """手动实现bert 的transformer-block的过程
    实际要串行跑12层
    如果第0层的权重从Bert 内拷贝，测出来的结果会几乎一致

    输入 x
        ↓
    Multi-Head Self-Attention
        ↓
    Add & Norm（残差 + 层归一化）
        ↓
    Feed Forward Network（前馈网络）
        ↓
    Add & Norm（残差 + 层归一化）
        ↓
        输出
    """

    # embedding 后的输入数据示例
    # 64 是batch size
    # 768 是embedding size
    # 256 是sequence length
    # input = torch.rand(64, 256, 768)

    w_q = nn.Linear(BERT_BASE_CONFIG.hidden_size, BERT_BASE_CONFIG.hidden_size)
    w_k = nn.Linear(BERT_BASE_CONFIG.hidden_size, BERT_BASE_CONFIG.hidden_size)
    w_v = nn.Linear(BERT_BASE_CONFIG.hidden_size, BERT_BASE_CONFIG.hidden_size)

    # 增加mac mps 计算 / windows cuda 计算
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using {device} device")

    # 输入数据示例
    # shape: [64, 256, 768]
    input = torch.rand(64, 256, BERT_BASE_CONFIG.hidden_size).to(device)
    print(input.shape)

    # Multi-Head
    w_q = w_q.to(device)
    w_k = w_k.to(device)
    w_v = w_v.to(device)

    q = w_q(input)
    k = w_k(input)
    v = w_v(input)

    # q 拆成 h=12 个头，每个头 d_k=64
    q = q.reshape(
        64,
        256,
        BERT_BASE_CONFIG.num_attention_heads,
        BERT_BASE_CONFIG.hidden_size_per_attenion_head,
    )
    # k 拆成 h=12 个头，每个头 d_k=64
    k = k.reshape(
        64,
        256,
        BERT_BASE_CONFIG.num_attention_heads,
        BERT_BASE_CONFIG.hidden_size_per_attenion_head,
    )
    # v 拆成 h=12 个头，每个头 d_k=64
    v = v.reshape(
        64,
        256,
        BERT_BASE_CONFIG.num_attention_heads,
        BERT_BASE_CONFIG.hidden_size_per_attenion_head,
    )

    print(q.shape, k.shape, v.shape)
    # 12个头并行计算，每个头独立做Self-Attention
    # 64 是 d_k
    # 256 是 sequence length
    # 12 是 head 数量
    # Self-Attention

    head_outputs = []
    for i in range(12):
        q_i = q[:, :, i, :]
        k_i = k[:, :, i, :]
        v_i = v[:, :, i, :]

        # q_i, k_i, v_i 都是 [64, 256, 64]
        # 计算attention score
        # k_i.transpose(1, 2) = Ki^T
        # k_i.shape = [B, T, d_k]  k_i.transpose(1, 2).shape = [B, d_k, T]
        attention_score = torch.matmul(q_i, k_i.transpose(1, 2)) / torch.sqrt(
            torch.tensor(64, dtype=torch.float32)
        )

        # 计算attention weight
        attention_weight = F.softmax(attention_score, dim=-1)

        # 计算attention output
        attention_output = torch.matmul(attention_weight, v_i)
        print(attention_output.shape)

        head_outputs.append(attention_output)  # ← 保存每个头的结果

    # 将12个头拼接起来，得到最终的attention output
    attention_output = torch.cat(head_outputs, dim=-1)

    print(attention_output.shape)
    # 线性变换
    attention_output = nn.Linear(
        BERT_BASE_CONFIG.hidden_size, BERT_BASE_CONFIG.hidden_size
    ).to(device)(attention_output)

    attention_output = nn.Dropout(0.3).to(device)(attention_output)
    print(attention_output.shape)

    # 第一遍残差连接和layer normalization
    attention_output = nn.LayerNorm(BERT_BASE_CONFIG.hidden_size).to(device)(
        attention_output + input
    )
    print(attention_output.shape)

    # FEN
    fc1 = nn.Linear(768, 3072).to(device)
    fc2 = nn.Linear(3072, 768).to(device)
    fen_output = fc2(F.gelu(fc1(attention_output)))
    fen_output = nn.Dropout(0.3).to(device)(fen_output)
    print(fen_output.shape)

    # 第二遍残差连接和layer normalization
    fen_output = nn.LayerNorm(BERT_BASE_CONFIG.hidden_size).to(device)(
        fen_output + attention_output
    )
    print(fen_output.shape)


# ===================== 1. 手动实现 =====================
class ManualTransformerBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.num_heads = BERT_BASE_CONFIG.num_attention_heads
        self.head_dim = (
            BERT_BASE_CONFIG.hidden_size // BERT_BASE_CONFIG.num_attention_heads
        )  # 64

        # 增加mac mps 计算 / windows cuda 计算
        if torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using {device} device")

        # Q/K/V/W_O
        self.w_q = nn.Linear(
            BERT_BASE_CONFIG.hidden_size, BERT_BASE_CONFIG.hidden_size
        ).to(device)
        self.w_k = nn.Linear(
            BERT_BASE_CONFIG.hidden_size, BERT_BASE_CONFIG.hidden_size
        ).to(device)
        self.w_v = nn.Linear(
            BERT_BASE_CONFIG.hidden_size, BERT_BASE_CONFIG.hidden_size
        ).to(device)
        self.w_o = nn.Linear(
            BERT_BASE_CONFIG.hidden_size, BERT_BASE_CONFIG.hidden_size
        ).to(device)

        # 两个 LayerNorm
        self.ln1 = nn.LayerNorm(BERT_BASE_CONFIG.hidden_size).to(device)
        self.ln2 = nn.LayerNorm(BERT_BASE_CONFIG.hidden_size).to(device)

        # FFN
        self.fc1 = nn.Linear(
            BERT_BASE_CONFIG.hidden_size, BERT_BASE_CONFIG.intermediate_size
        ).to(device)
        self.fc2 = nn.Linear(
            BERT_BASE_CONFIG.intermediate_size, BERT_BASE_CONFIG.hidden_size
        ).to(device)

    def forward(self, x, mask=None):
        B, T, D = x.shape

        # ① Q/K/V
        q = self.w_q(x)  # [B, T, 768]
        k = self.w_k(x)
        v = self.w_v(x)

        # ② 拆成多头 [B, T, 12, 64] → [B, 12, T, 64]
        q = q.reshape(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.reshape(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.reshape(B, T, self.num_heads, self.head_dim).transpose(1, 2)

        # ③ 并行算所有头的注意力 [B, 12, T, T]
        scores = torch.matmul(q, k.transpose(-1, -2)) / (self.head_dim**0.5)

        # Padding Mask
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)

        attn = F.softmax(scores, dim=-1)
        out = torch.matmul(attn, v)  # [B, 12, T, 64]

        # ④ 拼接多头 → [B, T, 768]
        out = out.transpose(1, 2).reshape(B, T, D)
        out = self.w_o(out)

        # ⑤ 残差 + LN
        out = self.ln1(out + x)

        # ⑥ FFN
        ffn = self.fc2(F.gelu(self.fc1(out)))

        # ⑦ 残差 + LN
        out = self.ln2(ffn + out)
        return out


# ===================== 2. 对比函数 =====================
def compare():
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 加载 BERT 官方模型和 tokenizer
    print("加载 BERT 模型...")
    model_dir = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "bert-base-chinese"
    )
    tokenizer = BertTokenizer.from_pretrained(model_dir)
    bert = BertModel.from_pretrained(model_dir).to(device)
    bert.eval()

    # 取 BERT 的第 0 层 Transformer Block（官方实现）
    official_block = bert.encoder.layer[0].to(device)
    official_block.eval()

    # 手动实现
    manual_block = ManualTransformerBlock().to(device)

    # 拷贝权重
    copy_weights(official_block, manual_block)

    manual_block.eval()

    # ===================== 3. 准备真实输入 =====================
    text = "我喜欢玩游戏，因为很有趣"
    print(f"\n输入文本: {text}")

    # 用 BERT tokenizer 编码
    encoding = tokenizer(
        text,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=32,
    )
    input_ids = encoding["input_ids"].to(device)  # [1, T]
    attention_mask = encoding["attention_mask"].to(device)  # [1, T]

    print(f"input_ids: {input_ids.shape}")
    print(f"attention_mask: {attention_mask.shape}")
    print(f"token 文本: {tokenizer.convert_ids_to_tokens(input_ids[0])}")

    # BERT 的 Embedding 输出（作为两个 block 的共同输入）
    with torch.no_grad():
        embedding_output = bert.embeddings(input_ids)  # [1, T, 768]
    print(f"\nembedding 输出: {embedding_output.shape}")

    # ===================== 4. 跑两个 block =====================
    # attention_mask 转成 bool 类型（官方 block 要求）
    attn_mask = attention_mask[:, None, None, :].bool()  # [1,1,1,T]

    with torch.no_grad():
        # 官方实现
        official_out = official_block(
            embedding_output,
            attention_mask=attn_mask,
        )[0]

        # 手动实现
        manual_out = manual_block(embedding_output, mask=attn_mask)

    # ===================== 5. 对比结果 =====================
    print(f"\n===== 对比结果 =====")
    print(f"官方输出 shape: {official_out.shape}")
    print(f"手动输出 shape: {manual_out.shape}")

    # 计算差异
    diff = (official_out - manual_out).abs()
    print(f"\n最大绝对误差: {diff.max().item():.6f}")
    print(f"平均绝对误差: {diff.mean().item():.6f}")
    print(f"误差 < 0.001 的元素占比: {(diff < 0.001).float().mean().item() * 100:.1f}%")

    # 注意：因为权重没对齐，数值不会完全一样
    # 但 shape 和量级应该一致
    print(f"\n官方输出前5个值: {official_out[0, 0, :5].tolist()}")
    print(f"手动输出前5个值: {manual_out[0, 0, :5].tolist()}")


# ===================== 6. 拷贝权重 =====================
def copy_weights(official_block, manual_block):
    """把 BERT 第0层的权重完整拷贝到手动实现"""

    # ===== Attention: Q/K/V =====
    manual_block.w_q.weight.data = official_block.attention.self.query.weight.data
    manual_block.w_q.bias.data = official_block.attention.self.query.bias.data

    manual_block.w_k.weight.data = official_block.attention.self.key.weight.data
    manual_block.w_k.bias.data = official_block.attention.self.key.bias.data

    manual_block.w_v.weight.data = official_block.attention.self.value.weight.data
    manual_block.w_v.bias.data = official_block.attention.self.value.bias.data

    # ===== Attention: W_O =====
    manual_block.w_o.weight.data = official_block.attention.output.dense.weight.data
    manual_block.w_o.bias.data = official_block.attention.output.dense.bias.data

    # ===== Attention: LayerNorm 1 =====
    manual_block.ln1.weight.data = official_block.attention.output.LayerNorm.weight.data
    manual_block.ln1.bias.data = official_block.attention.output.LayerNorm.bias.data

    # ===== FFN: fc1 (Intermediate) =====
    manual_block.fc1.weight.data = official_block.intermediate.dense.weight.data
    manual_block.fc1.bias.data = official_block.intermediate.dense.bias.data

    # ===== FFN: fc2 (Output) =====
    manual_block.fc2.weight.data = official_block.output.dense.weight.data
    manual_block.fc2.bias.data = official_block.output.dense.bias.data

    # ===== FFN: LayerNorm 2 =====
    manual_block.ln2.weight.data = official_block.output.LayerNorm.weight.data
    manual_block.ln2.bias.data = official_block.output.LayerNorm.bias.data

    print("所有权重拷贝完成！")


def main():
    # 手动实现bert 的transformer-block的过程
    # diy_transformer_block()

    # 对比手搓与实际bert的transformer-block
    compare()


if __name__ == "__main__":
    main()
