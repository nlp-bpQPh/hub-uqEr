import tempfile
import unittest
from pathlib import Path

from common.model_paths import _is_model_dir, resolve_model_path


class ModelPathTests(unittest.TestCase):
    def test_rejects_incomplete_model_directory(self):
        with tempfile.TemporaryDirectory() as tmp:
            model = Path(tmp) / "chinese-roberta-wwm-ext"
            model.mkdir()
            (model / "model.safetensors").write_bytes(b"stub")
            self.assertFalse(_is_model_dir(model))

    def test_accepts_complete_local_model_directory(self):
        with tempfile.TemporaryDirectory() as tmp:
            model = Path(tmp) / "chinese-roberta-wwm-ext"
            model.mkdir()
            (model / "config.json").write_text("{}", encoding="utf-8")
            (model / "model.safetensors").write_bytes(b"stub")
            (model / "vocab.txt").write_text("[PAD]\n[UNK]\n", encoding="utf-8")
            self.assertTrue(_is_model_dir(model))

    def test_resolves_explicit_local_model_root(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "pretrain_models"
            model = root / "chinese-roberta-wwm-ext"
            model.mkdir(parents=True)
            (model / "config.json").write_text("{}", encoding="utf-8")
            (model / "model.safetensors").write_bytes(b"stub")
            (model / "tokenizer.json").write_text("{}", encoding="utf-8")
            resolved = resolve_model_path("chinese-roberta-wwm-ext", extra_roots=[root])
            self.assertEqual(resolved, model)


if __name__ == "__main__":
    unittest.main()
