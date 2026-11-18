from pathlib import Path

import torch

from language_model import LanguageModel, device


def load_language_model():
    """加载同目录下已训练的模型"""
    checkpoint_path = Path(__file__).with_name("best_model.pt")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    vocab = checkpoint["vocab"]

    model = LanguageModel(
        vocab_size=len(vocab),
        input_size=128,
        hidden_size=256,
        dropout=0.0,
    )
    model.load_state_dict(checkpoint["model_state"])
    model.to(device)
    model.eval()
    return model, vocab


@torch.no_grad()
def generate_text(model, prompt, vocab, max_new_chars=80):
    """每次选择概率最高的字符，并立即打印"""
    id_to_char = {index: char for char, index in vocab.items()}
    input_ids = [vocab[char] for char in prompt if char in vocab]

    if not input_ids:
        raise ValueError("提示文本中没有词表内的字符")

    model_device = next(model.parameters()).device
    max_context = model.pos_embedding.num_embeddings
    result = prompt
    print(result, end="", flush=True)

    for _ in range(max_new_chars):
        context_ids = input_ids[-max_context:]
        x = torch.tensor([context_ids], dtype=torch.long, device=model_device)

        # 取最后一个字符位置的概率分布，选择概率最高的下一个字符
        probabilities = model(x)[0, -1, :]
        next_id = torch.argmax(probabilities).item()
        next_char = id_to_char[next_id]

        input_ids.append(next_id)
        result += next_char
        print(next_char, end="", flush=True)

    print()
    return result


if __name__ == "__main__":
    prompt = input("请输入续写开头：")
    model, vocab = load_language_model()
    generate_text(model, prompt, vocab)
