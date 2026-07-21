import unittest

from analyze_cprc_attribution import (
    fixed_lambda_prediction,
    policy_summary,
    prior_alignment,
    row_key,
)


def make_row(side, gt, baseline, image_scores, prior_scores):
    return {
        "status": "ok",
        "task": "qa",
        "pair_id": "Pair 1",
        "side": side,
        "gt": gt,
        "cp_vbc": {
            "baseline_key": baseline,
            "candidates": [
                {
                    "key": key,
                    "logp_image": image_scores[key],
                    "logp_prior": prior_scores[key],
                }
                for key in ("yes", "no")
            ],
        },
    }


class CPRCAttributionTest(unittest.TestCase):
    def setUp(self):
        self.cs = make_row(
            "commonsense",
            "yes",
            "yes",
            {"yes": 3.0, "no": 0.0},
            {"yes": 3.0, "no": 0.0},
        )
        self.cf = make_row(
            "counterfactual",
            "no",
            "yes",
            {"yes": 2.0, "no": 1.0},
            {"yes": 4.0, "no": 0.0},
        )

    def test_prior_subtraction_changes_the_candidate_ranking(self):
        self.assertEqual(fixed_lambda_prediction(self.cf, 0.0), "yes")
        self.assertEqual(fixed_lambda_prediction(self.cf, 1.0), "no")

    def test_prior_alignment_uses_paired_cs_target(self):
        values = prior_alignment([self.cf, self.cs])["qa"]["native_counterfactual_errors"]
        self.assertEqual(values["n"], 1)
        self.assertEqual(values["baseline_equals_prior_top"], 1.0)
        self.assertEqual(values["prior_top_equals_cs_gt"], 1.0)

    def test_summary_separates_cf_repairs_from_cs_harms(self):
        predictions = {row_key(self.cf): "no", row_key(self.cs): "no"}
        values = policy_summary([self.cf, self.cs], predictions, bootstrap=20, seed=3)
        self.assertEqual(values["by_side"]["counterfactual"]["repairs"], 1)
        self.assertEqual(values["by_side"]["commonsense"]["harms"], 1)


if __name__ == "__main__":
    unittest.main()
