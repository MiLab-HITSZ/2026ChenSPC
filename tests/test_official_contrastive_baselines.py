import math
import unittest
from types import SimpleNamespace
from unittest import mock

import torch

from analyze_candidate_baseline_sweep import (
    _method_block,
    _replay_mfcd,
    _replay_nolan,
    _replay_vcd,
    _symmetric_kl,
)
from evaluate_cdh_bench import (
    _candidate_logprob_qwen3_vl,
    _candidate_logprobs_shared_prefix,
    _candidate_plausibility_mask,
    _mfcd_candidate_score,
    _pai_candidate_score,
    _vcd_candidate_score,
)


class _Tokenizer:
    tokenizer = None

    def __init__(self):
        self.tokenizer = self

    def __call__(self, text, **_kwargs):
        values = {"": [], " yes": [1, 2], " no": [1, 3]}[text]
        return {"input_ids": torch.tensor([values], dtype=torch.long)}


class _Model:
    def __init__(self):
        self.calls = 0

    def __call__(self, input_ids, **_kwargs):
        self.calls += 1
        length = input_ids.shape[-1]
        logits = torch.arange(8, dtype=torch.float32).repeat(1, length, 1)
        return SimpleNamespace(logits=logits)


class OfficialContrastiveBaselineTest(unittest.TestCase):
    def test_shared_prefix_scores_match_individual_forwards_exactly(self):
        processor = _Tokenizer()
        prompt_inputs = {
            "input_ids": torch.tensor([[5, 6, 7]], dtype=torch.long),
            "attention_mask": torch.ones((1, 3), dtype=torch.long),
        }
        with mock.patch(
            "evaluate_cdh_bench._candidate_vlm_inputs",
            side_effect=lambda *_args, **_kwargs: {
                key: value.clone() for key, value in prompt_inputs.items()
            },
        ):
            shared_model = _Model()
            shared = _candidate_logprobs_shared_prefix(
                shared_model,
                processor,
                "prompt",
                None,
                [" yes", " no"],
            )
            individual_model = _Model()
            individual = [
                _candidate_logprob_qwen3_vl(
                    individual_model, processor, "prompt", None, candidate
                )
                for candidate in (" yes", " no")
            ]
        self.assertEqual(shared_model.calls, 1)
        self.assertEqual(individual_model.calls, 2)
        self.assertEqual(shared, individual)

    def test_vcd_matches_official_score_form(self):
        self.assertAlmostEqual(
            _vcd_candidate_score(-2.0, -5.0, 1.0),
            (1.0 + 1.0) * -2.0 - 1.0 * -5.0,
        )

    def test_mfcd_subtracts_both_frequency_views(self):
        self.assertAlmostEqual(
            _mfcd_candidate_score(-2.0, -4.0, -6.0, 1.0, 0.5),
            (1.0 + 1.0 + 0.5) * -2.0 - 1.0 * -4.0 - 0.5 * -6.0,
        )

    def test_pai_matches_official_cfg_score_form(self):
        self.assertAlmostEqual(
            _pai_candidate_score(-2.0, -5.0, 1.1),
            1.1 * -2.0 - 0.1 * -5.0,
        )

    def test_nolan_uses_released_dynamic_suppression_rule(self):
        block = {
            "baseline_key": "a",
            "candidates": [
                {"key": "a", "logp_image": -0.2, "logp_no_image": -0.1},
                {"key": "b", "logp_image": -0.3, "logp_no_image": -3.0},
            ],
        }
        self.assertEqual(_replay_nolan(block, {"beta": 0.8}), "b")

    def test_symmetric_kl_is_zero_only_for_matching_distributions(self):
        self.assertAlmostEqual(_symmetric_kl([0.8, 0.2], [0.8, 0.2]), 0.0)
        self.assertGreater(_symmetric_kl([0.8, 0.2], [0.2, 0.8]), 0.0)

    def test_plausibility_mask_is_relative_to_image_maximum(self):
        rows = [
            {"key": "a", "logp_image": math.log(0.8)},
            {"key": "b", "logp_image": math.log(0.2)},
            {"key": "c", "logp_image": math.log(0.05)},
        ]
        self.assertEqual(
            _candidate_plausibility_mask(rows, "logp_image", 0.1),
            {"a": True, "b": True, "c": False},
        )

    def test_zero_beta_keeps_all_candidates(self):
        rows = [{"key": "a", "logp_image": -1.0}, {"key": "b", "logp_image": -100.0}]
        self.assertEqual(
            _candidate_plausibility_mask(rows, "logp_image", 0.0),
            {"a": True, "b": True},
        )

    def test_vcd_can_reuse_identical_mfcd_low_frequency_scores(self):
        record = {
            "mfcd": {
                "baseline_key": "a",
                "low_mode": "blur",
                "candidates": [
                    {"key": "a", "logp_image": -1.0, "logp_low_freq": -2.0},
                    {"key": "b", "logp_image": -2.0, "logp_low_freq": -4.0},
                ],
            }
        }
        block = _method_block(record, "vcd")
        self.assertEqual(block["degrade_mode"], "blur")
        self.assertEqual(block["candidates"][1]["logp_degraded"], -4.0)
        self.assertEqual(
            block["score_source"], "shared_mfcd_image_low_frequency_candidates"
        )

    def test_view_collection_replays_vcd_and_mfcd_without_rescoring_image(self):
        block = {
            "baseline_key": "a",
            "candidates": [
                {
                    "key": "a",
                    "logp_image": -1.0,
                    "views": {
                        "blur": {"logprob": -1.0},
                        "edges": {"logprob": -3.0},
                    },
                },
                {
                    "key": "b",
                    "logp_image": -1.2,
                    "views": {
                        "blur": {"logprob": -3.0},
                        "edges": {"logprob": -1.0},
                    },
                },
            ],
        }
        self.assertEqual(
            _replay_vcd(
                block,
                {
                    "alpha": 1.0,
                    "beta": 0.0,
                    "contrast_margin": 0.0,
                    "degrade_mode": "blur",
                    "gate": "always",
                },
            ),
            "b",
        )
        self.assertEqual(
            _replay_mfcd(
                block,
                {
                    "alpha_low": 1.0,
                    "alpha_high": 0.5,
                    "beta": 0.0,
                    "contrast_margin": 0.0,
                    "low_mode": "blur",
                    "high_mode": "edges",
                    "gate": "always",
                },
            ),
            "b",
        )


if __name__ == "__main__":
    unittest.main()
