import argparse
import random
from pathlib import Path

import torch

from sft_model import SFTModel, device


DATA_DIR = Path(__file__).resolve().parent / "data"
MODEL_PATH = DATA_DIR / "sft_best_model.pt"


def load_model():
    checkpoint = torch.load(
        MODEL_PATH,
        map_location="cpu",
        weights_only=False,
    )
    vocab = checkpoint["vocab"]
    state_dict = checkpoint["model_state"]

    # 从 checkpoint 推断模型维度，避免和训练脚本配置不一致
    vocab_size, input_size = state_dict["embedding.weight"].shape
    hidden_size = state_dict["transformer.layers.0.linear1.weight"].shape[0]

    model = SFTModel(
        vocab_size=vocab_size,
        input_size=input_size,
        hidden_size=hidden_size,
        dropout=0.0,
    )
    model.load_state_dict(state_dict, strict=True)
    model.to(device)
    model.eval()
    return model, vocab


def encode(text, vocab):
    # 与 SFT 训练保持一致，未登录字符暂时映射为 ID 0
    return [vocab.get(char, 0) for char in text]


def generate(
    model,
    vocab,
    prefix,
    max_new_tokens=64,
    temperature=0.8,
    top_k=10,
    greedy=False,
):
    reverse_vocab = {index: char for char, index in vocab.items()}
    input_ids = encode(prefix, vocab)

    max_context = model.pos_embedding.num_embeddings
    if len(input_ids) >= max_context:
        raise ValueError(
            f"输入长度为 {len(input_ids)}，已经达到模型上限 {max_context}"
        )

    generated = []
    with torch.no_grad():
        for _ in range(min(max_new_tokens, max_context - len(input_ids))):
            x = torch.tensor([input_ids], dtype=torch.long, device=device)
            probabilities = model(x)[0, -1]

            if greedy:
                next_id = torch.argmax(probabilities).item()
            else:
                k = min(max(1, top_k), probabilities.numel())
                top_probs, top_ids = torch.topk(probabilities, k)
                top_probs = (top_probs / max(temperature, 1e-5)).softmax(dim=-1)
                selected = torch.multinomial(top_probs, num_samples=1).item()
                next_id = top_ids[selected].item()

            next_char = reverse_vocab[next_id]

            generated.append(next_char)
            input_ids.append(next_id)

    return "".join(generated).strip()


def main():
    parser = argparse.ArgumentParser(description="字符级语言模型 SFT 问答 Demo")
    parser.add_argument(
        "text",
        nargs="?",
        type=str,
        default=None,
        help="用户问题；不传时进入交互输入",
    )
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--greedy", action="store_true", help="使用确定性贪心解码")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)

    model, vocab = load_model()
    print(f"模型: {MODEL_PATH}")
    print(f"设备: {device}")
    print(f"词表大小: {len(vocab)}\n")

    question = args.text
    if question is None:
        question = input("请输入问题：")

    question = question.strip()
    if not question:
        raise ValueError("问题不能为空")

    # 必须与 TextDataset 中的 SFT 训练模板保持一致
    model_input = f"用户：{question}\n助手："

    answer = generate(
        model=model,
        vocab=vocab,
        prefix=model_input,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_k=args.top_k,
        greedy=args.greedy,
    )

    print(f"问题：{question}")
    print(f"回答：{answer}")


if __name__ == "__main__":
    main()
