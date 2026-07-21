import json
import unittest
from pathlib import Path
from types import SimpleNamespace

from cprc_vlm_runtime import qwen3_vl_model_class


class Qwen3VLModelLoadingTest(unittest.TestCase):
    def test_local_30b_checkpoint_declares_moe_architecture(self):
        config_path = Path("models/Qwen3-VL-30B-A3B-Instruct/config.json")
        if not config_path.exists():
            self.skipTest("local 30B checkpoint is not installed")

        config = json.loads(config_path.read_text(encoding="utf-8"))
        self.assertEqual(config["model_type"], "qwen3_vl_moe")
        self.assertIn("Qwen3VLMoeForConditionalGeneration", config["architectures"])

    def test_transformers_exposes_dense_and_moe_qwen3_vl_classes(self):
        import transformers

        self.assertTrue(hasattr(transformers, "Qwen3VLForConditionalGeneration"))
        self.assertTrue(hasattr(transformers, "Qwen3VLMoeForConditionalGeneration"))

    def test_model_type_selects_the_official_dense_or_moe_class(self):
        dense = object()
        moe = object()
        module = SimpleNamespace(
            Qwen3VLForConditionalGeneration=dense,
            Qwen3VLMoeForConditionalGeneration=moe,
        )
        self.assertIs(qwen3_vl_model_class("qwen3_vl", module), dense)
        self.assertIs(qwen3_vl_model_class("qwen3_vl_moe", module), moe)

    def test_unknown_model_type_is_rejected(self):
        module = SimpleNamespace(
            Qwen3VLForConditionalGeneration=object(),
            Qwen3VLMoeForConditionalGeneration=object(),
        )
        with self.assertRaisesRegex(ValueError, "unsupported Qwen3-VL"):
            qwen3_vl_model_class("qwen3_vl_unknown", module)


if __name__ == "__main__":
    unittest.main()
