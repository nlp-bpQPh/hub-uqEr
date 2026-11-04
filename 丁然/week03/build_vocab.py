import csv
import json
import unicodedata
from pathlib import Path


PAD_TOKEN = "<PAD>"
UNK_TOKEN = "<UNK>"





def build_vocab(sentences):
    vocab = {PAD_TOKEN: 0, UNK_TOKEN: 1}
    print(f"正在构建词表...sentences = {len(sentences)}")
    for sentence in sentences:
        for char in sentence:
            if char not in vocab:
                vocab[char] = len(vocab)

    return vocab


def load_sentences(csv_paths):
    # 找出所有句子
    sentences = []

    for csv_path in csv_paths:
        with Path(csv_path).open("r", encoding="utf-8-sig", newline="") as file:
            reader = csv.DictReader(file)
            if "text" not in (reader.fieldnames or []):
                raise ValueError(f"{csv_path} 中缺少 text 列")

            for row in reader:
                text = (row["text"] or "").strip()
                if text:
                    sentences.append(text)

    return sentences


def build_v12_vocab(data_dir):
    data_dir = Path(data_dir)
    csv_paths = [
        data_dir / "train.csv",
    ]

    sentences = load_sentences(csv_paths)
    return build_vocab(sentences)


def main():
    week_dir = Path(__file__).resolve().parent
    vocab = build_v12_vocab(week_dir / "data" / "v12")
    output_path = week_dir  / "data"/ "vocab.json"

    with output_path.open("w", encoding="utf-8") as file:
        json.dump(vocab, file, ensure_ascii=False, indent=2)

    print(f"词表大小: {len(vocab)}")


if __name__ == "__main__":
    main()
