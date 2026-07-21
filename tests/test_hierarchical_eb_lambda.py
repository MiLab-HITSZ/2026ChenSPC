import copy
import math
import unittest
from unittest.mock import patch

import numpy as np

from analyze_hierarchical_eb_lambda_cv import (
    apply_gate,
    candidate_arrays,
    context_features,
    exact_top_support,
    evaluation_splits,
    exact_mcnemar_pvalue,
    joint_folds,
    policy_details_gauss_hermite,
    run_cv,
    selection_key,
)


def make_row(pair_id: str = "Pair 1", cv_group: str = "group-1"):
    return {
        "status": "ok",
        "_dataset": "synthetic",
        "_task": "qa",
        "pair_id": pair_id,
        "cv_group": cv_group,
        "subcategory": "Synthetic",
        "side": "counterfactual",
        "gt": "no",
        "cp_vbc": {
            "baseline_key": "yes",
            "candidates": [
                {"key": "yes", "logp_image": -0.2, "logp_prior": -0.1},
                {"key": "no", "logp_image": -0.8, "logp_prior": -2.0},
            ],
        },
    }


class HierarchicalEmpiricalBayesLambdaTest(unittest.TestCase):
    def test_nolan_overlap_adds_one_bounded_distribution_feature(self):
        row = make_row()
        native = context_features(row, feature_mode="full")
        augmented = context_features(row, feature_mode="nolan_overlap")
        self.assertEqual(len(augmented), len(native) + 1)
        np.testing.assert_allclose(augmented[:-1], native)
        self.assertGreaterEqual(augmented[-1], 0.0)
        self.assertLessEqual(augmented[-1], 1.0)

    def test_exact_mcnemar_pvalue(self):
        self.assertEqual(exact_mcnemar_pvalue(0, 0), 1.0)
        self.assertAlmostEqual(exact_mcnemar_pvalue(3, 0), 0.25)
        self.assertAlmostEqual(exact_mcnemar_pvalue(5, 1), 0.21875)

    def test_frozen_transfer_never_trains_on_target_labels(self):
        rows = []
        for label in ("dev_mc", "dev_qa", "test_mc", "test_qa"):
            row = make_row(pair_id=label)
            row["_dataset"] = label
            rows.append(row)
        splits, evaluation_labels = evaluation_splits(
            rows,
            "frozen_transfer",
            5,
            train_labels=("dev_mc", "dev_qa"),
        )
        _, train_ids, test_ids = splits[0]

        self.assertEqual({rows[index]["_dataset"] for index in train_ids}, {"dev_mc", "dev_qa"})
        self.assertEqual({rows[index]["_dataset"] for index in test_ids}, {"test_mc", "test_qa"})
        self.assertEqual(evaluation_labels, ["test_mc", "test_qa"])

    def test_frozen_transfer_does_not_refit_on_all_rows(self):
        rows = []
        for label in ("dev_mc", "dev_qa", "test_mc", "test_qa"):
            row = make_row(pair_id=label)
            row["_dataset"] = label
            rows.append(row)
        fit_ids = []

        def fake_fit(rows, raw_contexts, train_ids, *args, **kwargs):
            fit_ids.append(list(train_ids))
            predictions = [row["cp_vbc"]["baseline_key"] for row in rows]
            details = [{} for _ in rows]
            return predictions, details, {"synthetic": True}

        with patch(
            "analyze_hierarchical_eb_lambda_cv.fit_and_select",
            side_effect=fake_fit,
        ):
            result = run_cv(
                rows,
                mode="frozen_transfer",
                folds=5,
                max_cs_drop=0.2,
                require_cf_nonnegative=True,
                quadrature_points=8,
                bootstrap=2,
                seed=13,
                selection_objective="net_utility",
                cs_cost=0.5,
                train_labels=("dev_mc", "dev_qa"),
            )

        self.assertEqual(fit_ids, [[0, 1]])
        self.assertEqual(result["evaluation_labels"], ["test_mc", "test_qa"])
        self.assertIn("no target labels", result["full_data_selected"]["note"])

    def test_net_utility_can_accept_a_paid_cs_loss(self):
        balanced = {
            "synthetic": {
                "counterfactual": {"correct": 8, "delta": 0.2},
                "commonsense": {"correct": 8, "delta": -0.1},
            }
        }
        aggressive = {
            "synthetic": {
                "counterfactual": {"correct": 9, "delta": 0.3},
                "commonsense": {"correct": 6, "delta": -0.3},
            }
        }
        rows = [make_row()]
        self.assertGreater(
            selection_key(balanced, ["yes"], rows, [0], "net_utility", 1.0),
            selection_key(aggressive, ["yes"], rows, [0], "net_utility", 1.0),
        )
        self.assertGreater(
            selection_key(aggressive, ["yes"], rows, [0], "net_utility", 0.25),
            selection_key(balanced, ["yes"], rows, [0], "net_utility", 0.25),
        )

    def test_exact_top_support_matches_binary_threshold(self):
        image = np.asarray([0.0, -1.0])
        prior = np.asarray([0.0, -2.0])
        support = exact_top_support(
            proposal_index=1,
            image=image,
            prior=prior,
            lambda_cap=1.0,
            latent_mean=0.0,
            latent_std=1.0,
        )
        threshold = np.log(np.expm1(0.5))
        expected = 0.5 * (1.0 - math.erf(threshold / math.sqrt(2.0)))

        self.assertAlmostEqual(support, expected, places=8)

    def test_gauss_hermite_is_exact_for_degenerate_posterior(self):
        row = make_row()
        context = context_features(row)[None, :]
        mean = np.zeros(context.shape[1])
        covariance = np.zeros((context.shape[1], context.shape[1]))

        first = policy_details_gauss_hermite(
            [row], context, mean, covariance, quadrature_points=8
        )[0]
        second = policy_details_gauss_hermite(
            [row], context, mean, covariance, quadrature_points=64
        )[0]

        self.assertEqual(first["proposal"], second["proposal"])
        self.assertAlmostEqual(first["lambda_mean"], second["lambda_mean"], places=8)
        self.assertAlmostEqual(
            first["posterior_margin"], second["posterior_margin"], places=8
        )

    def test_features_are_invariant_to_candidate_logprob_offsets(self):
        row = make_row()
        shifted = copy.deepcopy(row)
        for candidate in shifted["cp_vbc"]["candidates"]:
            candidate["logp_image"] += 13.0
            candidate["logp_prior"] -= 7.0

        self.assertTrue(np.allclose(candidate_arrays(row)[1], candidate_arrays(shifted)[1]))
        self.assertTrue(np.allclose(candidate_arrays(row)[2], candidate_arrays(shifted)[2]))
        self.assertTrue(np.allclose(context_features(row), context_features(shifted)))

    def test_gate_requires_numeric_prior_risk(self):
        row = make_row()
        params = {
            "visual_support_min": 0.8,
            "absorption_support_min": 0.8,
            "posterior_margin_min": 0.1,
        }
        accepted = {
            "baseline": "yes",
            "proposal": "no",
            "visual_conflict": False,
            "prior_absorption": True,
            "posterior_support": 0.9,
            "posterior_margin": 0.2,
        }
        rejected = {**accepted, "prior_absorption": False}

        self.assertEqual(apply_gate([row], [accepted], params), ["no"])
        self.assertEqual(apply_gate([row], [rejected], params), ["yes"])

    def test_joint_folds_keep_cv_groups_together(self):
        rows = []
        for group_index in range(5):
            for question_index in range(2):
                row = make_row(
                    pair_id=f"Pair {group_index}-{question_index}",
                    cv_group=f"group-{group_index}",
                )
                rows.append(row)
        folds = joint_folds(rows, 5)

        for group_index in range(5):
            positions = [
                index
                for index, row in enumerate(rows)
                if row["cv_group"] == f"group-{group_index}"
            ]
            self.assertEqual(len({folds[index] for index in positions}), 1)
        self.assertEqual(set(folds), set(range(5)))


if __name__ == "__main__":
    unittest.main()
