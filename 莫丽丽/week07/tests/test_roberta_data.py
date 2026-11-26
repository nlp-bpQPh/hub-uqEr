# -*- coding: utf-8 -*-
"""不依赖 transformers/显卡的 RoBERTa 数据侧回归测试。"""
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "common"))

from event_data import (BioDataset, build_joint_schema, load_jsonl, load_schema,
                        sample_negative_rows)


class DummyEncoding(dict):
    def __init__(self, input_ids, attention_mask, word_ids):
        super().__init__(input_ids=input_ids, attention_mask=attention_mask)
        self._word_ids = word_ids

    def word_ids(self):
        return self._word_ids


class DummyFastTokenizer:
    is_fast = True

    def __call__(self, chars, **kwargs):
        # 为每个字符生成一个 token，首尾分别模拟 CLS / SEP。
        return DummyEncoding([101] + list(range(1000, 1000 + len(chars))) + [102],
                             [1] * (len(chars) + 2),
                             [None] + list(range(len(chars))) + [None])


class RobertaDataTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.data_dir = ROOT / "data" / "processed"
        cls.schema = build_joint_schema(load_schema(cls.data_dir / "event_schema.json"))

    def test_negative_sampling_is_deterministic_and_all_o(self):
        positive = load_jsonl(self.data_dir / "train_extract.jsonl")[:25]
        detect = load_jsonl(self.data_dir / "train_detect.jsonl")
        a, na = sample_negative_rows(positive, detect, 1.0, seed=7)
        b, nb = sample_negative_rows(positive, detect, 1.0, seed=7)
        self.assertEqual(na, 25)
        self.assertEqual(na, nb)
        self.assertEqual([x["text"] for x in a], [x["text"] for x in b])
        negatives = a[len(positive):]
        self.assertTrue(all("event" not in x and "events" not in x for x in negatives))
        self.assertEqual(len({x["text"] for x in negatives}), len(negatives))

    def test_negative_examples_receive_only_o_tags(self):
        row = {"id": "n1", "sent_id": 0, "text": "这是一条没有事件的测试句。"}
        ds = BioDataset([row], DummyFastTokenizer(), self.schema, max_length=64)
        item = ds[0]
        labels = item["labels"].tolist()
        non_ignored = [x for x in labels if x != -100]
        self.assertTrue(non_ignored)
        self.assertTrue(all(x == self.schema["label2id"]["O"] for x in non_ignored))

    def test_clean_dev_exists_and_is_smaller_than_original(self):
        clean = load_jsonl(self.data_dir / "dev_extract_clean.jsonl")
        original = load_jsonl(self.data_dir / "dev_extract.jsonl")
        self.assertEqual(len(clean), 920)
        self.assertEqual(len(original), 930)


if __name__ == "__main__":
    unittest.main()

class PromptVersionTests(unittest.TestCase):
    def test_prompt_v2_adds_closed_schema_and_evidence_check(self):
        prompt_dir = ROOT / "llm_zero_shot"
        sys.path.insert(0, str(prompt_dir))
        import json
        from event_prompt import build_system_prompt
        raw = json.loads((ROOT / "data" / "processed" / "event_schema.json").read_text(encoding="utf-8"))
        schema = {"event_types": {k: v for k, v in raw.items() if not k.startswith("__")}}
        v1 = build_system_prompt(schema, "v1")
        v2 = build_system_prompt(schema, "v2")
        self.assertNotIn("prompt v2", v1)
        self.assertIn("prompt v2", v2)
        self.assertIn("论元 role 必须逐字来自", v2)
        with self.assertRaises(ValueError):
            build_system_prompt(schema, "v3")
