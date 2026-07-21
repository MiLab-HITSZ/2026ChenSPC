import unittest

from analyze_qa_polarity_stress import score_view_diagnostics
from build_qa_polarity_stress import (
    label_counts,
    transform,
    validate_pairwise_complements,
)


def sample_item():
    return {
        "pair_id": "Pair 1",
        "category": "Counting Anomalies",
        "subcategory": "Body Parts",
        "counterfactual_prompt": "six fingers",
        "commonsense_prompt": "five fingers",
        "direct_qa": {
            "question": "Does the hand have five fingers?",
            "counterfactual_gt": "no",
            "commonsense_gt": "yes",
        },
        "multiple_choice": {
            "question": "How many fingers are visible?",
            "options": ["A. five", "B. six", "C. four", "D. seven"],
            "counterfactual_gt": "B",
            "commonsense_gt": "A",
        },
        "captioning": {
            "question": "Describe the hand.",
            "counterfactual_gt": "six fingers",
            "commonsense_gt": "five fingers",
        },
    }


class QAPolarityStressTest(unittest.TestCase):
    def test_variants_reverse_labels_and_answer_order(self):
        cs_first = transform(sample_item(), "cs_first")
        cf_first = transform(sample_item(), "cf_first")
        validate_pairwise_complements([cs_first], [cf_first])
        self.assertIn('"five" rather than "six"', cs_first["direct_qa"]["question"])
        self.assertIn('"six" rather than "five"', cf_first["direct_qa"]["question"])
        self.assertEqual(cs_first["direct_qa"]["counterfactual_gt"], "no")
        self.assertEqual(cf_first["direct_qa"]["counterfactual_gt"], "yes")
        self.assertEqual(
            label_counts([cs_first, cf_first]),
            {
                "counterfactual": {"no": 1, "yes": 1},
                "commonsense": {"no": 1, "yes": 1},
            },
        )

    def test_invalid_mc_pair_uses_caption_anchor_without_manual_rule(self):
        item = sample_item()
        item["multiple_choice"]["counterfactual_gt"] = "A"
        transformed = transform(item, "cf_first")
        metadata = transformed["qa_polarity_stress"]
        self.assertEqual(
            metadata["anchor_source"], "caption_gt_fallback_for_invalid_mc_pair"
        )
        self.assertIn("six fingers", transformed["direct_qa"]["question"])

    def test_score_view_diagnostics_detects_noncomplementary_prior(self):
        rows = {}
        for variant, gt_by_side in {
            "cs_first": {"counterfactual": "no", "commonsense": "yes"},
            "cf_first": {"counterfactual": "yes", "commonsense": "no"},
        }.items():
            for side, gt in gt_by_side.items():
                rows[(f"test_{variant}", "Pair 1", side)] = {
                    "gt": gt,
                    "cp_vbc": {
                        "baseline_key": gt,
                        "image_top": gt,
                        "prior_top": "yes",
                        "candidates": [],
                    },
                }

        diagnostics = score_view_diagnostics(rows)
        cf = diagnostics["paired_complement"]["counterfactual"]
        self.assertEqual(cf["native_complement_consistency"], 1.0)
        self.assertEqual(cf["image_score_complement_consistency"], 1.0)
        self.assertEqual(cf["no_image_prior_complement_consistency"], 0.0)
        self.assertEqual(
            diagnostics["by_variant"]["cf_first"]["counterfactual"][
                "no_image_prior_acc"
            ],
            1.0,
        )


if __name__ == "__main__":
    unittest.main()
