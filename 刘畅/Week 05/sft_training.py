import argparse
import json
from functools import partial
from pathlib import Path


def load_data(path, tokenizer, max_length):
    """instruction/output → chat template → 回答部分有监督的 token 序列。"""
    samples = []
    skipped = 0
    with open(path, encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            for key in ("instruction", "output"):
                if not isinstance(row.get(key), str) or not row[key].strip():
                    raise ValueError(f"{path}:{line_no} 的 {key} 必须是非空字符串")
            messages = [{"role": "user", "content": row["instruction"]}]
            # prompt 包含 assistant 的起始标记，但不包含回答。
            prompt_ids = tokenizer.apply_chat_template(
                messages, tokenize=True, add_generation_prompt=True
            )
            input_ids = tokenizer.apply_chat_template(
                messages + [{"role": "assistant", "content": row["output"]}],
                tokenize=True, add_generation_prompt=False,
            )
            # 检查真实 token 边界，避免替换模型后把问题或回答错标。
            if input_ids[:len(prompt_ids)] != prompt_ids:
                raise ValueError("模型的 chat template 前缀不一致，请使用兼容的非思考对话模型")
            if len(input_ids) <= len(prompt_ids):
                raise ValueError(f"{path}:{line_no} 没有可训练的回答 token")
            # 保留完整回答及模板自带的结束标记；超长样本跳过，不把截断答案当完整答案。
            if len(input_ids) > max_length:
                skipped += 1
                continue
            labels = [-100] * len(prompt_ids) + input_ids[len(prompt_ids):]
            samples.append({"input_ids": input_ids, "labels": labels})
    print(f"{Path(path).name}: 使用 {len(samples)} 条，跳过超长样本 {skipped} 条")
    if not samples:
        raise ValueError(f"{path} 无有效样本，请检查数据或增大 --max-length")
    return samples


def collate_batch(samples, pad_token_id):
    import torch

    length = max(len(s["input_ids"]) for s in samples)
    batch = {"input_ids": [], "attention_mask": [], "labels": []}
    for sample in samples:
        size = len(sample["input_ids"])
        padding = length - size
        batch["input_ids"].append(sample["input_ids"] + [pad_token_id] * padding)
        batch["attention_mask"].append([1] * size + [0] * padding)
        batch["labels"].append(sample["labels"] + [-100] * padding)
    return {key: torch.tensor(value, dtype=torch.long) for key, value in batch.items()}


def evaluate(model, loader, device):
    import torch

    model.eval()
    total_loss, total_tokens = 0.0, 0
    with torch.no_grad():
        for batch in loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            loss = model(**batch).loss
            if not torch.isfinite(loss):
                raise RuntimeError("验证 loss 非有限值，请检查数据和模型")
            # 模型内部 shift 后，labels 第 0 位不会参与 loss。
            count = (batch["labels"][:, 1:] != -100).sum().item()
            total_loss += loss.item() * count
            total_tokens += count
    return total_loss / total_tokens


def main():
    folder = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description="按 PPT 的回答区间 loss mask 进行 SFT")
    parser.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--train-file", type=Path, default=folder / "sft_train.jsonl")
    parser.add_argument("--val-file", type=Path, default=folder / "sft_val.jsonl")
    parser.add_argument("--output-dir", type=Path, default=folder / "sft_model")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--max-steps", type=int, default=0, help="0 表示完整训练")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda", "mps"], default="auto")
    args = parser.parse_args()
    if (args.epochs < 1 or args.batch_size < 1 or args.max_length < 2
            or args.learning_rate <= 0 or args.max_steps < 0):
        parser.error("epochs/batch-size/learning-rate 必须为正，max-length >= 2，max-steps >= 0")
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        parser.error("输出目录非空，请通过 --output-dir 指定新目录，避免覆盖已有模型")

    import torch
    from torch.utils.data import DataLoader
    from transformers import AutoModelForCausalLM, AutoTokenizer

    torch.manual_seed(42)
    device = args.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else (
            "mps" if torch.backends.mps.is_available() else "cpu"
        )
    print(f"训练设备: {device}，模型: {args.model}（全参数 SFT）")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if not tokenizer.chat_template:
        raise ValueError("模型没有 chat template，请选用带模板的模型")
    if tokenizer.pad_token_id is None:
        if tokenizer.eos_token_id is None:
            raise ValueError("tokenizer 缺少 pad_token 和 eos_token")
        tokenizer.pad_token = tokenizer.eos_token
    train_data = load_data(args.train_file, tokenizer, args.max_length)
    val_data = load_data(args.val_file, tokenizer, args.max_length)
    collate_fn = partial(collate_batch, pad_token_id=tokenizer.pad_token_id)
    train_loader = DataLoader(train_data, batch_size=args.batch_size,
                              shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_data, batch_size=args.batch_size, collate_fn=collate_fn)

    # 模型自带 causal mask；attention_mask 只负责屏蔽 padding。
    model = AutoModelForCausalLM.from_pretrained(args.model).to(device)
    model.config.use_cache = False
    model.gradient_checkpointing_enable()  # 用额外计算换取较低的激活内存占用
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
    print(f"训练前 val_loss={evaluate(model, val_loader, device):.4f}")
    step = 0
    for epoch in range(args.epochs):
        model.train()
        total_loss, total_tokens = 0.0, 0
        for batch in train_loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            optimizer.zero_grad(set_to_none=True)
            # AutoModelForCausalLM 内部完成错位一位和 ignore_index=-100。
            # labels 与 input_ids 保持同位置，不能再手动 shift 一次。
            loss = model(**batch).loss
            if not torch.isfinite(loss):
                raise RuntimeError("训练 loss 非有限值，请检查数据和学习率")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            count = (batch["labels"][:, 1:] != -100).sum().item()
            total_loss += loss.item() * count
            total_tokens += count
            step += 1
            if step == 1 or step % 20 == 0:
                print(f"epoch={epoch + 1} step={step} loss={loss.item():.4f}")
            if args.max_steps and step >= args.max_steps:
                break
        val_loss = evaluate(model, val_loader, device)
        print(f"epoch={epoch + 1} train_loss={total_loss / total_tokens:.4f} "
              f"val_loss={val_loss:.4f}")
        if args.max_steps and step >= args.max_steps:
            break

    model.config.use_cache = True
    model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print(f"模型与 tokenizer 已保存到: {args.output_dir}")


if __name__ == "__main__":
    main()
