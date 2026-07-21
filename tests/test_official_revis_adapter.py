import unittest
from unittest.mock import patch

import torch

from official_revis_adapter import (
    build_retrospective_inputs,
    official_risk,
    resolve_revis_paper_alpha,
    select_official_layer,
)


class _FakeTensor:
    def __init__(self):
        self.device = None

    def to(self, device):
        self.device = device
        return self


class _FakeProcessor:
    def __init__(self):
        self.messages = None
        self.text = None
        self.images = None

    def apply_chat_template(self, messages, **kwargs):
        self.messages = messages
        self.template_kwargs = kwargs
        return "<assistant-prefix>"

    def __call__(self, *, text, images, **kwargs):
        self.text = text
        self.images = images
        self.processor_kwargs = kwargs
        return {"input_ids": _FakeTensor(), "pixel_values": _FakeTensor()}


class OfficialRevisAdapterTest(unittest.TestCase):
    def test_paper_alpha_is_family_specific(self):
        self.assertEqual(resolve_revis_paper_alpha("qwen3_vl"), 1.6)
        self.assertEqual(resolve_revis_paper_alpha("llava_next"), 1.1)

    def test_llava_retrospective_input_matches_released_append_style(self):
        processor = _FakeProcessor()
        image = object()
        with patch("official_revis_adapter._input_device", return_value="cuda:0"):
            inputs = build_retrospective_inputs(
                object(), processor, "llava_next", "What is shown?", "A red car.", image
            )

        self.assertEqual(processor.text, ["<assistant-prefix> A red car."])
        self.assertEqual(processor.images, [image])
        self.assertTrue(processor.template_kwargs["add_generation_prompt"])
        self.assertEqual(processor.messages[0]["content"][0], {"type": "image"})
        self.assertNotIn("assistant", [message["role"] for message in processor.messages])
        self.assertTrue(all(value.device == "cuda:0" for value in inputs.values()))

    def test_llava_retrospective_no_image_omits_placeholder(self):
        processor = _FakeProcessor()
        with patch("official_revis_adapter._input_device", return_value="cuda:1"):
            build_retrospective_inputs(
                object(), processor, "llava_next", "What is shown?", "Unknown.", None
            )

        self.assertIsNone(processor.images)
        self.assertEqual(processor.messages[0]["content"], [{"type": "text", "text": "What is shown?"}])

    def test_risk_is_negative_cosine_similarity(self):
        hidden = torch.tensor([[1.0, 0.0]])
        visual = torch.tensor([1.0, 0.0])
        self.assertAlmostEqual(float(official_risk(hidden, visual).item()), -1.0)

    def test_backward_search_selects_deepest_positive_layer(self):
        rows = {
            11: [
                {"category": "TP", "risk": 0.1},
                {"category": "TN", "risk": 0.3},
                {"category": "FP", "risk": 0.5},
            ],
            12: [
                {"category": "TP", "risk": 0.5},
                {"category": "TN", "risk": 0.3},
                {"category": "FP", "risk": 0.2},
            ],
        }
        result = select_official_layer(rows, percentile=85, min_layer=10)
        self.assertEqual(result["selected"]["layer"], 11)
        self.assertEqual(result["selected"]["vector_index"], 12)

    def test_search_respects_minimum_layer(self):
        rows = {
            9: [
                {"category": "TP", "risk": 0.1},
                {"category": "FP", "risk": 0.9},
            ]
        }
        self.assertIsNone(select_official_layer(rows, min_layer=10)["selected"])


if __name__ == "__main__":
    unittest.main()
