import unittest

from analyze_cp_vbc_bayes_path_cv import (
    bayes_path_features,
    fold_ids,
    metric,
    predict_bayes_path,
    selection_key,
)


def make_row(
    *,
    baseline="A",
    image_a=0.0,
    image_b=-1.0,
    prior_a=0.0,
    prior_b=-3.0,
    subcategory="Synthetic",
    pair_id="Pair 1",
    side="counterfactual",
):
    return {
        "status": "ok",
        "task": "mc",
        "side": side,
        "subcategory": subcategory,
        "pair_id": pair_id,
        "gt": "B",
        "cp_vbc": {
            "baseline_key": baseline,
            "candidates": [
                {"key": "A", "logp_image": image_a, "logp_prior": prior_a},
                {"key": "B", "logp_image": image_b, "logp_prior": prior_b},
            ],
        },
    }


class BayesPathTest(unittest.TestCase):
    def test_net_utility_trades_cf_gain_against_cs_loss(self):
        balanced = {
            "counterfactual": {"correct": 8, "delta": 0.2},
            "commonsense": {"correct": 8, "delta": -0.1, "overrides": 1},
        }
        aggressive = {
            "counterfactual": {"correct": 9, "delta": 0.3},
            "commonsense": {"correct": 6, "delta": -0.3, "overrides": 3},
        }
        self.assertGreater(
            selection_key(balanced, 2, 0, "net_utility", 1.0),
            selection_key(aggressive, 4, 1, "net_utility", 1.0),
        )
        self.assertGreater(
            selection_key(aggressive, 4, 1, "net_utility", 0.25),
            selection_key(balanced, 2, 0, "net_utility", 0.25),
        )

    def test_metric_does_not_count_key_case_as_override(self):
        row = make_row(baseline="A")
        row["gt"] = "A"
        result = metric([row], ["a"], [0])

        self.assertEqual(result["counterfactual"]["overrides"], 0)
        self.assertEqual(result["counterfactual"]["correct"], 1)

    def test_prior_absorption_path_can_override(self):
        row = make_row()
        features = bayes_path_features(row, 0.5, 1.0, 9)
        self.assertEqual(features["proposal"], "B")
        self.assertTrue(features["prior_absorption"])
        self.assertGreaterEqual(features["path_stability"], 0.5)
        prediction = predict_bayes_path(
            features,
            {
                "visual_margin_min": 0.0,
                "absorption_margin_max": 2.0,
                "path_stability_min": 0.5,
                "path_margin_min": 0.0,
                "prior_relief_min": 0.0,
            },
        )
        self.assertEqual(prediction, "B")

    def test_prior_relief_gate_can_abstain(self):
        row = make_row()
        features = bayes_path_features(row, 0.5, 1.0, 9)
        prediction = predict_bayes_path(
            features,
            {
                "visual_margin_min": 0.0,
                "absorption_margin_max": 2.0,
                "path_stability_min": 0.5,
                "path_margin_min": 0.0,
                "prior_relief_min": 4.0,
            },
        )
        self.assertEqual(prediction, "A")

    def test_absorption_path_margin_max_can_abstain(self):
        row = make_row()
        features = bayes_path_features(row, 0.5, 1.0, 9)
        prediction = predict_bayes_path(
            features,
            {
                "visual_margin_min": 0.0,
                "absorption_margin_max": 2.0,
                "absorption_path_margin_max": 0.0,
                "path_stability_min": 0.5,
                "path_margin_min": 0.0,
                "prior_relief_min": 0.0,
            },
        )
        self.assertEqual(prediction, "A")

    def test_visual_conflict_follows_image_supported_candidate(self):
        row = make_row(image_a=-1.0, image_b=0.0, prior_a=0.0, prior_b=-1.0)
        features = bayes_path_features(row, 0.2, 0.8, 7)
        self.assertEqual(features["proposal"], "B")
        self.assertTrue(features["visual_conflict"])
        prediction = predict_bayes_path(
            features,
            {
                "visual_margin_min": 0.5,
                "absorption_margin_max": 0.0,
                "path_stability_min": 0.5,
                "path_margin_min": 0.0,
                "prior_relief_min": 0.0,
            },
        )
        self.assertEqual(prediction, "B")

    def test_fold_assignment_keeps_pair_sides_together(self):
        rows = []
        for subcategory in ("A", "B"):
            for pair_id in ("Pair 1", "Pair 2"):
                rows.append(make_row(subcategory=subcategory, pair_id=pair_id, side="counterfactual"))
                rows.append(make_row(subcategory=subcategory, pair_id=pair_id, side="commonsense"))
        folds = fold_ids(rows)
        seen = {}
        for row, fold in zip(rows, folds):
            key = (row["subcategory"], row["pair_id"])
            seen.setdefault(key, fold)
            self.assertEqual(seen[key], fold)
        self.assertEqual(set(folds), {0, 1})

    def test_explicit_fold_count_handles_single_large_subcategory(self):
        rows = []
        for pair_id in range(1, 11):
            rows.append(
                make_row(
                    subcategory="POPEv2",
                    pair_id=f"Pair {pair_id}",
                    side="counterfactual",
                )
            )
            rows.append(
                make_row(
                    subcategory="POPEv2",
                    pair_id=f"Pair {pair_id}",
                    side="commonsense",
                )
            )
        folds = fold_ids(rows, requested_folds=5)

        self.assertEqual(set(folds), set(range(5)))
        for index in range(0, len(rows), 2):
            self.assertEqual(folds[index], folds[index + 1])

    def test_cv_group_keeps_related_questions_in_one_fold(self):
        rows = [
            {
                **make_row(pair_id=f"Pair {pair_id}", side=side),
                "cv_group": group,
            }
            for group, pair_id in (("figure-a", 1), ("figure-a", 2), ("figure-b", 3))
            for side in ("counterfactual", "commonsense")
        ]
        folds = fold_ids(rows)

        self.assertEqual(len({folds[0], folds[1], folds[2], folds[3]}), 1)
        self.assertNotEqual(folds[0], folds[4])


if __name__ == "__main__":
    unittest.main()
