# -*- coding: utf-8 -*-


import json
import os
import random
import time
from pathlib import Path

import torch

# Transformers 在导入模型组件时会检查 torchvision 的部分算子。
# 当前环境的 torchvision 缺少 nms 算子注册，但 Qwen3 是纯文本模型，
# 实际训练不会调用图像相关功能；这里注册一个占位定义，避免导入时报错。
_torchvision_stub = torch.library.Library("torchvision", "DEF")
_torchvision_stub.define("nms(Tensor boxes, Tensor scores, float iou_threshold) -> Tensor")

# AutoTokenizer 根据模型目录读取对应的分词器；
# AutoModelForCausalLM 读取适合“根据前文预测下一个 token”的因果语言模型。
from transformers import AutoModelForCausalLM, AutoTokenizer


# Path(__file__) 表示当前脚本文件的位置。
# 后面的模型、数据和输出目录都相对当前脚本定位，换工作目录运行也不易找错文件。
HERE = Path(__file__).resolve().parent
MODEL_DIR = HERE / "Qwen3-0.6B"
TRAIN_PATH = HERE / "sft_train" / "firefly_train_small.jsonl"
VAL_PATH = HERE / "sft_val" / "firefly_val_small.jsonl"
OUTPUT_DIR = HERE / "outputs" / "qwen3-firefly-sft"


SEED = 42                 # 固定随机数种子，让随机行为尽量可重复
MAX_LENGTH = 128          # 每条对话最多允许 128 个 token；超长样本跳过
LEARNING_RATE = 2e-5      # 学习率：每次更新参数时迈多大的步子
TRAIN_STEPS = 1           # 只更新一次，验证训练流程能否执行
MAX_TRAIN_SAMPLES = 4     # 最多准备 4 条训练样本
MAX_VAL_SAMPLES = 2       # 最多用 2 条验证样本计算 loss


def load_rows(path):
    """读取 JSONL 数据：每一行都是一条独立的 JSON 训练样本。

    每条样本需要有两个字段：
      instruction：用户问题/任务说明
      output：希望模型学习生成的参考回答
    """
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def encode_example(tokenizer, row):
    """把一条问答样本转换成模型能处理的 token 编号和训练标签。

    关键点：SFT 通常只训练模型生成 assistant 回答，不要求模型预测用户问题。
    因此，用户 prompt 对应的 labels 会设为 -100；PyTorch 的交叉熵会忽略 -100。
    """
    user_message = {"role": "user", "content": row["instruction"]}
    messages = [user_message, {"role": "assistant", "content": row["output"]}]

    # 单独编码用户部分，并要求模板补上 assistant 的开头。
    # 这段 token 是完整对话序列的前缀，也就是模型开始回答前看到的内容。
    # transformers 5.x 的 apply_chat_template 返回 BatchEncoding，
    # 需要显式取出 input_ids；旧版本直接返回 id 列表。
    prompt_ids = list(tokenizer.apply_chat_template(
        [user_message], tokenize=True, add_generation_prompt=True,
        enable_thinking=False,
    )["input_ids"])

    # 编码完整的 user + assistant 对话。
    # tokenize=True 表示直接得到整数 token 列表，不是可读字符串。
    full_ids = list(tokenizer.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=False,
        enable_thinking=False,
    )["input_ids"])

    # 为了演示简单，这里不裁剪过长回答，直接丢弃超长样本，
    # 避免把回答截断后仍误以为训练目标完整。
    if len(full_ids) > MAX_LENGTH:
        return None

    # 确认 prompt 确实是完整对话的开头。只有这样才能准确确定
    # 从哪个 token 开始是 assistant 回答，并应用 loss mask。
    if full_ids[:len(prompt_ids)] != prompt_ids:
        raise ValueError("对话模板的 prompt 不是完整样本的前缀，无法正确屏蔽提示词 loss")

    # labels 与 full_ids 等长。
    # 用户问题和 assistant 开头部分用 -100 屏蔽；回答 token 保留原 token id。
    labels = [-100] * len(prompt_ids) + full_ids[len(prompt_ids):]

    # 至少要有两个 token，且确实存在一个未屏蔽的回答 token，才能训练。
    if len(full_ids) < 2 or not any(token_id != -100 for token_id in labels):
        return None
    return full_ids, labels


def make_batch(tokenizer, encoded_rows):
    """把若干条不同长度的样本拼成一个 batch（批次）。

    神经网络要求同一批输入形状相同，所以短样本右侧补 pad token。
    attention_mask 中 1 表示真实 token、0 表示 padding；padding 的 label
    设为 -100，保证它既不被注意力当作内容，也不参与 loss 计算。
    """
    max_len = max(len(ids) for ids, _ in encoded_rows)
    pad_id = tokenizer.pad_token_id
    input_ids = torch.full((len(encoded_rows), max_len), pad_id, dtype=torch.long)
    labels = torch.full((len(encoded_rows), max_len), -100, dtype=torch.long)
    attention_mask = torch.zeros((len(encoded_rows), max_len), dtype=torch.long)

    for index, (ids, target) in enumerate(encoded_rows):
        length = len(ids)
        input_ids[index, :length] = torch.tensor(ids, dtype=torch.long)
        labels[index, :length] = torch.tensor(target, dtype=torch.long)
        attention_mask[index, :length] = 1

    # 返回字典，字段名称正好对应 Hugging Face 模型 forward() 的参数。
    return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}


def prepare_examples(tokenizer, rows, limit):
    """编码数据集中的样本，最多保留 limit 条有效样本。"""
    encoded = []
    dropped = 0
    for row in rows:
        item = encode_example(tokenizer, row)
        if item is None:
            # None 表示样本超长，或者没有可用的回答 token。
            dropped += 1
        else:
            encoded.append(item)

        # 准备到指定条数后即可停止；我们只想做小规模流程测试。
        if len(encoded) >= limit:
            break

    if not encoded:
        raise ValueError("没有符合 MAX_LENGTH 的有效样本")
    return encoded, dropped


def evaluate(model, tokenizer, rows):
    """在少量验证样本上计算平均 loss，不更新模型参数。"""
    model.eval()  # 切换到评估模式；Dropout 等训练专用行为会关闭
    encoded, dropped = prepare_examples(tokenizer, rows, MAX_VAL_SAMPLES)
    losses = []

    # no_grad 表示不记录梯度，节约验证阶段的内存。
    with torch.no_grad():
        for item in encoded:
            batch = make_batch(tokenizer, [item])
            # labels 传入模型后会自动计算交叉熵 loss。
            losses.append(model(**batch, use_cache=False).loss.item())

    return sum(losses) / len(losses), len(encoded), dropped


def main():
    """训练主流程：加载模型 → 全参数更新 → 验证 → 保存 → 重载生成。"""
    random.seed(SEED)
    torch.manual_seed(SEED)
    # CPU 上线程并非越多越快；限制线程数，避免小任务线程调度开销过大。
    torch.set_num_threads(min(4, os.cpu_count() or 1))

    # 先检查需要的文件，错误时尽早给出清楚提示。
    if not (MODEL_DIR / "model.safetensors").is_file():
        raise FileNotFoundError(f"模型权重不存在：{MODEL_DIR / 'model.safetensors'}")
    train_rows = load_rows(TRAIN_PATH)
    val_rows = load_rows(VAL_PATH)
    if not train_rows or not val_rows:
        raise ValueError("训练集和验证集都必须至少有一条样本")

    # tokenizer 把文本转换为 token id，也负责把生成出来的 token id 还原成文字。
    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR, local_files_only=True)
    tokenizer.padding_side = "right"
    if tokenizer.pad_token_id is None:
        # 有些生成模型没有专门的 padding token；这里复用结束符作为补齐符。
        tokenizer.pad_token = tokenizer.eos_token

    # 加载本地 Qwen3 权重。CPU 使用 bfloat16 以降低内存占用：
    # float32 全参数训练需要约 9.6 GB（参数+梯度+优化器状态），本机内存不足；
    # bf16 把模型和梯度减半，仍保留全参数更新，只是数值精度略低。
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_DIR,
        torch_dtype=torch.bfloat16,
        local_files_only=True,
        low_cpu_mem_usage=True,
    )

    # 训练时关闭 KV cache，因为它主要用于生成加速，训练时会额外占内存。
    # Gradient checkpointing 通过反向传播时重新计算部分中间结果，减少激活内存，
    # 代价是计算会更慢。它不会冻结参数，也不会把全参数训练变成 LoRA。
    model.config.use_cache = False
    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()

    # 把 JSONL 样本编码成 token id 和 labels。
    train_encoded, train_dropped = prepare_examples(
        tokenizer, train_rows, MAX_TRAIN_SAMPLES)
    val_encoded, val_dropped = prepare_examples(
        tokenizer, val_rows, MAX_VAL_SAMPLES)

    # 不设置任何参数的 requires_grad=False：模型所有参数都参加训练。
    # total 是总参数量，trainable 是实际会计算梯度并更新的参数量。
    parameters = list(model.parameters())
    trainable = sum(parameter.numel() for parameter in parameters if parameter.requires_grad)
    total = sum(parameter.numel() for parameter in parameters)
    print(f"训练样本：{len(train_encoded)} 条；验证样本：{len(val_encoded)} 条")
    print(f"因长度限制跳过：训练 {train_dropped} 条 / 验证 {val_dropped} 条")
    print(f"可训练参数：{trainable:,} / {total:,}（全部参数）")
    print("设备：CPU；精度：bfloat16；gradient checkpointing：开启")

    # AdamW 根据 loss 的梯度更新参数。全参数 SFT 会为所有参数维护优化器状态，
    # 因此即使训练样本只有几条，模型本身的内存需求仍然很大。
    optimizer = torch.optim.AdamW(parameters, lr=LEARNING_RATE, weight_decay=0.01)
    model.train()  # 切换到训练模式
    started = time.time()

    for step in range(TRAIN_STEPS):
        # 每一步取一条样本，组成 batch。单样本减少激活内存，但不会减少模型参数/优化器状态。
        row = train_encoded[step % len(train_encoded)]
        batch = make_batch(tokenizer, [row])

        # 标准 PyTorch 训练五步：清梯度 → 前向算 loss → 反向求梯度 → 裁剪梯度 → 更新参数。
        optimizer.zero_grad(set_to_none=True)
        loss = model(**batch, use_cache=False).loss
        loss.backward()
        # 限制梯度范数，减少梯度异常过大时对训练稳定性的影响。
        torch.nn.utils.clip_grad_norm_(parameters, 1.0)
        optimizer.step()
        print(f"训练 step {step + 1}/{TRAIN_STEPS}，loss={loss.item():.4f}")

    # 更新一次后，用验证样本看看模型能否正常计算 loss。
    val_loss, val_count, _ = evaluate(model, tokenizer, val_rows)
    print(f"验证 loss（{val_count} 条）：{val_loss:.4f}")

    # 参考 week05 的做法保存完整模型，而不是仅保存少量 adapter 权重。
    # save_pretrained 会将更新后的模型参数和配置写到输出目录。
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    model.config.use_cache = True
    model.save_pretrained(OUTPUT_DIR, safe_serialization=True)
    tokenizer.save_pretrained(OUTPUT_DIR)

    # 保存训练摘要，方便回头确认模型来源、训练参数和 loss。
    result = {
        "base_model": str(MODEL_DIR),
        "method": "full_parameter_sft",
        "framework": "PyTorch",
        "train_samples": len(train_encoded),
        "validation_samples": val_count,
        "max_length": MAX_LENGTH,
        "train_steps": TRAIN_STEPS,
        "learning_rate": LEARNING_RATE,
        "trainable_parameters": trainable,
        "total_parameters": total,
        "train_loss": float(loss.item()),
        "validation_loss": float(val_loss),
        "train_seconds": round(time.time() - started, 1),
        "device": "cpu",
        "precision": "bfloat16",
    }
    with (OUTPUT_DIR / "train_result.json").open("w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    # 从保存目录重新加载一次模型，验证 checkpoint 确实可以被读取。
    check_model = AutoModelForCausalLM.from_pretrained(
        OUTPUT_DIR, torch_dtype=torch.bfloat16, local_files_only=True,
        low_cpu_mem_usage=True,
    )
    check_model.eval()

    # 只给模型验证集中的用户问题，让模型自行续写回答。
    # 同样要取出 input_ids：新版返回 BatchEncoding，直接传字典无法作为输入张量。
    prompt = tokenizer.apply_chat_template(
        [{"role": "user", "content": val_rows[0]["instruction"]}],
        tokenize=True, add_generation_prompt=True, enable_thinking=False,
        return_tensors="pt",
    )["input_ids"]
    with torch.no_grad():
        generated = check_model.generate(
            prompt, max_new_tokens=24, do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    # 输入 prompt 本身也在 generated 里，所以只解码它后面的新 token。
    answer = tokenizer.decode(generated[0, prompt.size(1):], skip_special_tokens=True)
    print("重新加载后的生成验证：", answer.strip())
    print(f"完整 SFT checkpoint 已保存：{OUTPUT_DIR}")


if __name__ == "__main__":
    # 只有直接运行本文件时才启动训练；被其他脚本 import 时不会自动训练。
    main()
