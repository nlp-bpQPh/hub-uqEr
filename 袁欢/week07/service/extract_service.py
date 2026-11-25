"""
L5 系统组合：把「切句 → 事件检测模型 → 校验 → 篇章合并」串成一个可调用的服务

为什么模型指标很好、系统还是不能直接上线（这是本节课的落脚点）：
  1. **模型会抽错但"看起来对"**：触发词必须能在原文里找到（span 校验），
     类型必须属于 9 类 schema（schema 校验），否则就是幻觉式结构化输出。
  2. **要能人工复核**：置信度低的样本打 needs_review，交给人审 ——
     真实业务里"机器抽 80% + 人审 20%"比"全自动但 15% 错"更可行。
  3. **篇章 ≠ 句子堆叠**：一篇新闻里同一事件可能在多句被抽到，
     要按句子顺序整理成事件流（带 sent_idx 溯源），而不是把碎片直接丢给下游。

口径说明（2026-11-24 换装 CCED 起）：论元链路已砍——原「正则兜底（金额/日期/
比例/股数）、核心角色判据、枚举校验」全部删除；服务层只剩「切句 → 检测 →
schema/span 校验 → 事件流」。原金融版「模型抽结构 + 正则补数字」的混合架构
教学点保留在 git 历史与 DuEE-Fin 归档目录中。

本模块不训练任何模型，只做「编排 + 校验」，模型全部来自 bert/ 与 bert_pipeline/ 的训练产物。

使用方式：
  from extract_service import ExtractService
  svc = ExtractService(backend="joint")
  events = svc.extract_document({"id": "...", "title": "...", "text": "..."})

命令行演示见 service/demo_extract.py
"""

import json
import os
import re
import sys
from pathlib import Path

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import torch

ROOT = Path(__file__).resolve().parents[1]   # 项目根目录（方法目录的上一级）
sys.path.insert(0, str(ROOT / "common"))     # 公共模块：event_data / crf / eval_utils / models / log_utils

from event_data import (ROOT, build_bio_constraint_mask, build_joint_schema,
                        build_start_constraint_mask, decode_bio_spans, load_enums,
                        load_schema, spans_to_event)
from model_paths import BERT_MODEL_DIR, resolve_model_path

CKPT_DIR = ROOT / "outputs" / "checkpoints"


def split_sentences(text: str, min_chars: int = 30, max_chars: int = 160) -> list[tuple[int, str]]:
    """
    中文新闻切句：按 。！？； 和换行切成最小单元，再把**过短的相邻单元合并**成 30~160 字的片段。
    返回 [(起始字符偏移, 片段)]。

    为什么要合并（本项目实测踩到的坑，课堂上必讲）：
      新闻原文里大量句子只有十几个字，按标点硬切会得到一堆 15~30 字的碎片；
      而模型训练时样本平均更长。输入长度分布和训练时对不上，同一篇新闻
      从「金标多个事件」变成只抽出 0~1 个。**模型不是不会抽，是被喂了它
      没见过的输入形态** —— 这是部署时最常见的掉点原因之一。
      工程做法：仍按句子边界切，但把碎片合并到接近训练分布的长度。
    保留偏移是为了后面能把「片段内坐标」换成「全文坐标」。
    """
    # 1) 先按标点切成最小单元。**只记区间 (start, end)**，片段文本一律从原文切片——
    #    早先版本用 `start += len(seg)` 累加偏移，遇到「连续标点」「空行」会漏加（正则
    #    `[^。！？；\n]+` 要求至少一个非标点字符），偏移逐段向左漂移（实测 dev 985 篇里
    #    81% 的文档出现漂移，最大 1249 字），篇章级 span 的全文坐标因此对错位置。
    units: list[tuple[int, int]] = []
    for m in re.finditer(r"[^。！？；\n]+[。！？；\n]*", text):
        if text[m.start():m.end()].strip():
            units.append((m.start(), m.end()))
    if not units:
        return [(0, text)]

    # 2) 把过短的相邻单元合并到 min_chars 以上（但不超过 max_chars）。
    #    用「起点 + 终点」滑动，片段文本 = text[cur_s:cur_e]，天然连续：
    #    既保证 text[off:off+len(seg)] == seg（offset 可直接拿去换全文坐标），
    #    也不会把单元之间的换行/连续标点吞掉。
    merged: list[tuple[int, str]] = []
    cur_s, cur_e = units[0]
    for seg_s, seg_e in units[1:]:
        if (cur_e - cur_s) < min_chars and (seg_e - cur_s) <= max_chars:
            cur_e = seg_e
        else:
            merged.append((cur_s, text[cur_s:cur_e]))
            cur_s, cur_e = seg_s, seg_e
    merged.append((cur_s, text[cur_s:cur_e]))
    return merged

class ExtractService:
    """
    事件检测服务：加载模型 → 逐句抽 → 校验 → 篇章事件流 → 标注复核。

    两种后端：
      backend="joint"    一个 BERT 模型端到端（bert/train.py 的产物），延迟低
      backend="pipeline" 类型分类 + 类型条件触发词专家（bert_pipeline/train.py 的产物），F1 通常更高
    """

    def __init__(self, backend: str = "joint", joint_ckpt: str = "", device: str = "",
                 review_threshold: float = 0.55,
                 bert_path: str = "", hard_constraint: bool = True):
        self.backend = backend
        self.review_threshold = review_threshold
        self.schema = load_schema()
        self.enums = load_enums()
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.bert_path = bert_path or str(resolve_model_path(BERT_MODEL_DIR))
        self.joint_ckpt = joint_ckpt or self._pick_default_joint_ckpt()
        # 解码期硬约束：默认开。理由是它约束的是「BIO 结构」而不是「内容」——
        # 不允许 I- 开头、不允许 B-x→I-y，这些本来就是标注规范，不可能伤害正确结果，
        # 却能把模型偶发的非法序列直接修掉（本项目实测非法转移 −87%，且论元 F1 提升）。
        self.hard_constraint = hard_constraint
        self._loaded = False

    @staticmethod
    def _pick_default_joint_ckpt() -> str:
        """默认优先用 CRF 版（非法序列更少），没有就退回 linear 版。"""
        for name in ("bert_crf_256.pt", "bert_linear_256.pt", "bert4L_crf_256.pt",
                     "bert_crf_256_mock.pt"):
            if (CKPT_DIR / name).exists():
                return str(CKPT_DIR / name)
        return ""

    # ── 模型加载（懒加载，避免 import 时就吃掉显存）──
    def load(self) -> None:
        if self._loaded:
            return
        from transformers import BertTokenizerFast
        self.tokenizer = BertTokenizerFast.from_pretrained(self.bert_path)
        if self.backend == "joint":
            self._load_joint()
        else:
            self._load_pipeline()
        self._loaded = True

    def _load_joint(self) -> None:
        from crf import CRFClassifier
        from models import BertEncoder

        if not self.joint_ckpt or not Path(self.joint_ckpt).exists():
            raise FileNotFoundError(
                f"找不到联合模型 checkpoint：{self.joint_ckpt}\n"
                f"请先运行 python bert/train.py --decoder crf")
        ckpt = torch.load(self.joint_ckpt, map_location=self.device, weights_only=False)
        self.label_scheme = ckpt["label_scheme"]
        scheme = {"labels": self.label_scheme["labels"],
                  "label2id": {l: i for i, l in enumerate(self.label_scheme["labels"])},
                  "id2label": {i: l for i, l in enumerate(self.label_scheme["labels"])}}
        self.id2label = scheme["id2label"]
        # 按 ckpt 记录的层数重建 encoder（mock/消融 ckpt 只存前 4 层，按 12 层建会缺权重）
        n_layers = int((ckpt.get("args") or {}).get("bert_layers", 0) or 0)
        encoder = BertEncoder(self.bert_path, n_layers)
        self.joint_model = CRFClassifier(
            encoder=encoder, hidden_size=encoder.hidden_size,
            num_tags=len(scheme["labels"]),
            use_crf=ckpt["args"].get("decoder", "crf") == "crf").to(self.device)
        self.joint_model.load_state_dict(ckpt["state_dict"])
        if self.hard_constraint and ckpt["args"].get("decoder", "crf") == "crf":
            # ⚠ constraint 是 persistent buffer，load_state_dict 会把它覆盖成 checkpoint 里的全 True，
            # 所以必须 load 之后再装一次（这个坑在 bert/evaluate_hard_constraint.py 里踩过）
            self.joint_model.crf.constraint = build_bio_constraint_mask(scheme).to(self.device)
            self.joint_model.crf.start_constraint = build_start_constraint_mask(scheme).to(self.device)
        self.joint_model.eval()
        self.joint_scheme = scheme

    def _load_pipeline(self) -> None:
        from models import BertCls
        self.types_sorted = sorted(self.schema)
        self.type2id = {t: i for i, t in enumerate(self.types_sorted)}
        self.type_model = BertCls(self.bert_path, len(self.types_sorted)).to(self.device)
        self.type_model.load_state_dict(torch.load(
            CKPT_DIR / "pipe_type.pt", map_location=self.device,
            weights_only=False)["state_dict"])
        self.type_model.eval()
        self._arg_models = {}   # 按需加载：9 个专家不一定要全load

    def _arg_model_for(self, etype: str):
        """按需加载某个类型的专家模型（缓存起来，第二次不再加载）。"""
        if etype in self._arg_models:
            return self._arg_models[etype]
        from models import build_arg_model
        path = CKPT_DIR / f"pipe_arg_{etype}.pt"
        if not path.exists():
            self._arg_models[etype] = None
            return None
        ckpt = torch.load(path, map_location=self.device, weights_only=False)
        scheme = {"labels": ckpt["labels"],
                  "label2id": {l: i for i, l in enumerate(ckpt["labels"])},
                  "id2label": {i: l for i, l in enumerate(ckpt["labels"])}}
        model = build_arg_model(self.bert_path, scheme,
                                 _ArgArgs(int((ckpt.get("args") or {}).get("arg_layers", 4)))).to(self.device)
        model.load_state_dict(ckpt["state_dict"])
        model.eval()
        self._arg_models[etype] = (model, scheme)
        return self._arg_models[etype]

    # ── 单句抽取 ──
    @torch.no_grad()
    def extract_sentence(self, text: str) -> dict:
        """抽一句话 → {"events": [...], "confidence": float, "flag": 诊断信息}"""
        if not text.strip():
            return {"events": [], "confidence": 0.0, "flags": ["empty_text"]}
        enc = self.tokenizer(list(text), is_split_into_words=True, max_length=384,
                             truncation=True, return_tensors="pt")
        word_ids = enc.word_ids()
        ids = enc["input_ids"].to(self.device)
        mask = enc["attention_mask"].to(self.device)
        tti = enc["token_type_ids"].to(self.device) if "token_type_ids" in enc else None

        if self.backend == "joint":
            paths, scores = self.joint_model.predict(ids, mask, tti, return_scores=True)
            tags = self._subword_to_char(paths[0], word_ids, len(text), self.id2label)
            ev = spans_to_event(decode_bio_spans(tags), text, enums=self.enums)
            events = [ev] if ev else []
            conf = float(scores[0])
        else:
            events, conf = self._pipeline_sentence(ids, mask, tti, word_ids, text)

        events, flags = self._validate(text, events)
        return {"events": events, "confidence": round(conf, 4), "flags": flags}

    def _pipeline_sentence(self, ids, mask, tti, word_ids, text):
        """类型分类 → 类型条件触发词专家，返回 (events, confidence)。"""
        from event_data import BioDataset  # 仅为复用 collate 逻辑，这里直接单条前向

        tlogits, _ = self.type_model(ids, mask, tti)
        type_prob = float(torch.softmax(tlogits, -1).max())
        etype = self.types_sorted[int(tlogits.argmax(-1))]
        got = self._arg_model_for(etype)
        if got is None:
            return [], type_prob
        model, scheme = got
        paths, scores = model.predict(ids, mask, tti, return_scores=True)
        tags = self._subword_to_char(paths[0], word_ids, len(text), scheme["id2label"])
        ev = spans_to_event(decode_bio_spans(tags), text, typed_type=etype, enums=self.enums)
        return ([ev] if ev else []), type_prob * float(scores[0])

    @staticmethod
    def _subword_to_char(path, word_ids, n_chars: int, id2label: dict) -> list[str]:
        """
        子词级预测 → 字符级标签。

        **只取每个汉字的首个子词**：训练时只有首子词背标签，非首子词训练时是 -100，
        它们的预测是噪声；如果让后面的子词覆盖前面的，字符标签会被污染，
        还会制造假的非法 BIO 转移（本项目实测过，见 train_bio.predict_char_tags 的注释）。
        """
        tags = ["O"] * n_chars
        prev_wid = None
        for j, wid in enumerate(word_ids):
            if wid is None or wid >= n_chars or j >= len(path):
                prev_wid = wid
                continue
            if wid != prev_wid:
                tags[wid] = id2label.get(path[j], "O")
            prev_wid = wid
        return tags

    # ── 校验 ──
    def _validate(self, text: str, events: list[dict]) -> tuple[list[dict], list[str]]:
        """
        两件事：
          1. 事件类型必须属于 schema 的 9 类 —— 防"幻觉式结构化输出"
          2. 触发词文本必须在原文里出现（span 校验）
        （论元链路已砍：无角色/枚举校验、无正则兜底）
        """
        flags = []
        clean = []
        valid_types = set(self.schema)
        for ev in events:
            etype = ev.get("type", "")
            if etype not in valid_types:
                flags.append(f"unknown_type:{etype}")
                continue
            trig = ev.get("trigger") or {}
            if trig.get("text") and trig["text"] not in text:
                flags.append(f"trigger_not_in_text:{etype}")
                continue
            clean.append({**ev, "args": []})
        return clean, flags

    def _needs_review(self, conf: float, events: list[dict], flags: list[str]) -> tuple[bool, list[str]]:
        """决定是否需要人工复核，并给出原因（下游工单要能看懂）。"""
        reasons = []
        if conf < self.review_threshold:
            reasons.append(f"low_confidence({conf:.2f})")
        if not events:
            reasons.append("no_event")
        for f in flags:
            if f.startswith(("unknown_type", "trigger_not_in_text")):
                reasons.append(f)
        return bool(reasons), reasons

    # ── 篇章级 ──
    def extract_document(self, doc: dict) -> dict:
        """
        篇章级抽取：切句 → 逐句抽 → 合并成事件流。

        合并规则（简单可解释）：同一事件类型的事件合并为一条，
        论元按角色累加（同角色同文本去重），并记录来自哪几句 ——
        这样一篇公告最终输出「1~3 个事件流」而不是「7 个碎片事件」。
        """
        text = doc.get("text") or ""
        title = doc.get("title") or ""
        sents = split_sentences(text)
        events: list[dict] = []   # 事件流：按句子顺序排列，**不做跨句同类型合并**
        sent_records = []

        for idx, (offset, seg) in enumerate(sents):
            res = self.extract_sentence(seg)
            review, reasons = self._needs_review(res["confidence"], res["events"], res["flags"])
            # 句子内坐标 → 全文坐标
            for ev in res["events"]:
                ev["trigger"] = dict(ev.get("trigger") or {})
                for key in ("start", "end"):
                    if ev["trigger"].get(key) is not None:
                        ev["trigger"][key] += offset
                for a in ev.get("args") or []:
                    for key in ("start", "end"):
                        if a.get(key) is not None:
                            a[key] += offset
            sent_records.append({"sent_idx": idx, "offset": offset, "text": seg.strip(),
                                 "events": res["events"], "confidence": res["confidence"],
                                 "needs_review": review, "review_reasons": reasons})
            # 每句最多 1 个事件，按出现顺序拼成事件流，并记下来自哪一句（sent_idx）。
            #
            # 为什么**不**做「同类型跨句合并」：一篇公告里「6 家子公司分别回购」「6 家平台分别被约谈」
            # 是 6 个**独立事件**，按类型合并会把 6 个事件压成 1 个（本项目实测：一篇金标 6 个
            # "股份回购" 的公告，合并后只剩 1 个，召回直接掉到 1/6）。
            # 真正需要合并的是「同一事件的信息分散在多句」（如金额在第 2 句、时间在第 3 句），
            # 那要靠事件共指/去重，属于更重的工程，本周不做 —— 这里保持"一句一事件、按序成流"，
            # 语义上更忠实，也更容易在 PPT 上讲清楚。
            for ev in res["events"]:
                events.append({
                    "type": ev["type"],
                    "trigger": ev.get("trigger"),
                    "args": ev.get("args") or [],
                    "sent_idx": idx,
                    "confidence": round(float(res["confidence"]), 4),
                })
        doc_review = any(r["needs_review"] for r in sent_records)
        return {
            "id": doc.get("id", ""),
            "title": title,
            "n_sentences": len(sents),
            "events": events,
            "needs_review": doc_review,
            "sentences": sent_records,
        }

class _ArgArgs:
    """给 build_arg_model 用的最小参数壳（它只关心 arg_layers / dropout）。"""

    dropout = 0.1

    def __init__(self, arg_layers: int = 4):
        self.arg_layers = arg_layers

if __name__ == "__main__":
    # 自检：不加载模型，只验证切句 + 校验逻辑（无 GPU 也能跑）
    svc = ExtractService.__new__(ExtractService)
    svc.schema = load_schema()
    svc.enums = load_enums()
    text = "黑客利用勒索软件加密了某医院的服务器，并索要赎金。事后安全团队披露了相关漏洞细节。"
    sents = split_sentences(text)
    print("切句：", [(o, s[:30]) for o, s in sents])
    ev = {"type": "勒索赎金", "trigger": {"text": "勒索软件", "start": 2, "end": 6},
          "args": []}
    bad = {"type": "不存在的类型", "trigger": {"text": "披露", "start": 0, "end": 2}, "args": []}
    hallucinated = {"type": "数据泄露", "trigger": {"text": "原文没有的词", "start": 0, "end": 6},
                    "args": []}
    clean, flags = svc._validate(text, [ev, bad, hallucinated])
    print("校验后：", json.dumps(clean, ensure_ascii=False))
    print("被拦下的问题：", flags)
