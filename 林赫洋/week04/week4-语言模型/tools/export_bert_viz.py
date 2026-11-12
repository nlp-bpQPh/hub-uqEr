"""
export_bert_viz.py — 用本地 bert-base-chinese 真实跑一遍，导出网页可视化要用的数据

导出内容（每句）:
    tokens / ids                 分词结果（WordPiece，中文基本按单字）
    attention[layer][head]       L×L 注意力权重，12 层 × 12 头（uint16 量化，误差 <1.5e-5）
    last_hidden / word_emb       末层上下文向量 / 静态字向量（fp16），给余弦对比用
    mlm[position]                遮住该位置后模型的 top-10 预测（真实概率 + 原字排名）
    scores[layer=1][head=1]      未归一化的 QK^T/√d，用来讲「分数 → softmax」两步

用法:
    python export_bert_viz.py                 # 导出预置句子
    python export_bert_viz.py --sentence "你的句子"   # 临时加一句（追加到预置列表）
输出:
    web/bert_data.js
"""

import argparse
import base64
import json
import os

import numpy as np
import torch
from transformers import BertForMaskedLM, BertTokenizer

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = r"D:\badou"
BERT_DIR = os.path.join(ROOT, r"八斗课程\pretrain_models\bert-base-chinese")
OUT = os.path.normpath(os.path.join(HERE, "..", "web", "bert_data.js"))

SENTENCES = [
    "小明昨天在图书馆遇到了他的老师，所以今天特别高兴。",
    "我昨天吃了一个很甜的苹果。",
    "这台苹果手机的性能非常强。",
    "中国的首都是北京，北京有很多名胜古迹。",
    "那只在河边散步的猫突然跳进了水里。",
    "银行宣布下调存款利率，储户的收益会受到影响。",
    "虽然下雨了，但是我们还是决定去爬山。",
    "他把书放在桌子上，然后就离开了。",
    "这家餐厅的服务态度很差，我再也不会来了。",
    "马云创办了阿里巴巴，这家公司改变了中国人的购物方式。",
]

MLM_TOPK = 10
QUANT_LAYER, QUANT_HEAD = 0, 0      # 导出原始 scores 的层/头（0 基）


def b64_fp16(arr: np.ndarray) -> str:
    return base64.b64encode(np.ascontiguousarray(arr).astype(np.float16).tobytes()).decode("ascii")


def b64_u16(arr: np.ndarray) -> str:
    return base64.b64encode(np.ascontiguousarray(arr).astype("<u2").tobytes()).decode("ascii")


def quant_attn(a: np.ndarray) -> np.ndarray:
    """注意力权重（0~1）→ uint16 线性量化：误差 < 1.5e-5，行和仍为 1.000。

    早先用 uint8 + sqrt 映射，体积小一半，但平方反解会系统性低估权重
    （每行加总只剩 0.98 左右），课堂上解释「每行和为 1」就说不通了。
    """
    return np.clip(np.round(np.clip(a, 0, 1) * 65535.0), 0, 65535).astype(np.uint16)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bert_dir", default=BERT_DIR)
    ap.add_argument("--out", default=OUT)
    ap.add_argument("--sentence", action="append", default=None, help="额外句子，可重复")
    args = ap.parse_args()

    sentences = list(SENTENCES) + (args.sentence or [])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")
    print(f"加载: {args.bert_dir}")

    tokenizer = BertTokenizer.from_pretrained(args.bert_dir)
    model = BertForMaskedLM.from_pretrained(
        args.bert_dir, output_attentions=True, output_hidden_states=True,
        attn_implementation="eager",
    ).to(device).eval()
    word_emb = model.bert.embeddings.word_embeddings.weight.detach()

    mask_id = tokenizer.mask_token_id
    data = {"meta": {"model": "bert-base-chinese", "layers": model.config.num_hidden_layers,
                     "heads": model.config.num_attention_heads,
                     "hidden": model.config.hidden_size}, "sentences": []}

    for si, text in enumerate(sentences):
        enc = tokenizer(text, return_tensors="pt")
        ids = enc["input_ids"][0]
        toks = tokenizer.convert_ids_to_tokens(ids)
        L = len(toks)
        with torch.no_grad():
            out = model(**{k: v.to(device) for k, v in enc.items()}, return_dict=True)
            attn = out.attentions                       # tuple(L层) of (1, heads, L, L)
            hidden = out.hidden_states[-1][0].float().cpu().numpy()   # (L, 768)
            static = word_emb[ids].float().cpu().numpy()              # (L, 768)

            # 未归一化 scores：用第 1 层第 1 头的 Q/K 手算，便于讲 softmax 前后
            layer = model.bert.encoder.layer[QUANT_LAYER]
            H = model.config.hidden_size
            heads = model.config.num_attention_heads
            dk = H // heads
            x = model.bert.embeddings(ids.unsqueeze(0).to(device))[0]
            q = layer.attention.self.query(x)[:, :dk]
            k = layer.attention.self.key(x)[:, :dk]
            scores = (q @ k.T / np.sqrt(dk)).float().cpu().numpy()

            # MLM：逐位置遮住，看模型猜什么
            mlm = []
            for pos in range(1, L - 1):
                masked = ids.clone()
                original = int(masked[pos])
                masked[pos] = mask_id
                logits = model(input_ids=masked.unsqueeze(0).to(device), return_dict=True).logits[0, pos]
                probs = torch.softmax(logits.float(), dim=-1)
                top = torch.topk(probs, MLM_TOPK)
                rank_of_original = int((probs > probs[original]).sum().item())
                mlm.append({
                    "pos": pos,
                    "original": tokenizer.convert_ids_to_tokens(original),
                    "original_prob": round(float(probs[original]), 5),
                    "original_rank": rank_of_original,
                    "top": [{"token": tokenizer.convert_ids_to_tokens(int(i)),
                             "p": round(float(p), 5)}
                            for p, i in zip(top.values, top.indices)],
                })

        data["sentences"].append({
            "text": text,
            "tokens": toks,
            "ids": ids.tolist(),
            "attn": [b64_u16(quant_attn(a[0].float().cpu().numpy())) for a in attn],
            "hidden": b64_fp16(hidden),
            "static": b64_fp16(static),
            "mlm": mlm,
            "scores": {"layer": QUANT_LAYER, "head": QUANT_HEAD,
                       "matrix": [round(float(v), 4) for row in scores for v in row]},
        })
        print(f"[{si + 1}/{len(sentences)}] {text}  tokens={L}")

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    js = "window.BERT_DATA = " + json.dumps(data, ensure_ascii=False, separators=(",", ":")) + ";\n"
    with open(args.out, "w", encoding="utf-8") as f:
        f.write(js)
    size = os.path.getsize(args.out) / 1024 / 1024
    print(f"已写出: {args.out} ({size:.2f} MB)")

    # 顺带打印一个「多义词」检查：同一 token 在不同句子里上下文向量是否不同
    def vec(si, tok):
        s = data["sentences"][si]
        i = s["tokens"].index(tok)
        h = np.frombuffer(base64.b64decode(s["hidden"]), dtype=np.float16).reshape(-1, 768)
        st = np.frombuffer(base64.b64decode(s["static"]), dtype=np.float16).reshape(-1, 768)
        return h[i].astype(np.float32), st[i].astype(np.float32)

    cos = lambda a, b: float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b)))
    try:
        h1, s1 = vec(1, "苹")
        h2, s2 = vec(2, "苹")
        print(f"多数词对照「苹」：上下文向量 cos={cos(h1, h2):.3f}   静态字向量 cos={cos(s1, s2):.3f}")
    except Exception as e:
        print("多义词检查跳过:", e)


if __name__ == "__main__":
    main()
