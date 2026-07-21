import unittest
from unittest.mock import patch

from analyze_hierarchical_eb_group_holdout import run_group_holdout


def make_row(split, task, group, pair_id):
    return {
        "_split": split,
        "_task": task,
        "_dataset": f"{split}_{task}",
        "status": "ok",
        "task": task,
        "pair_id": pair_id,
        "category": "Synthetic",
        "subcategory": group,
        "side": "counterfactual",
        "gt": "no",
        "cp_vbc": {
            "baseline_key": "yes",
            "candidates": [
                {"key": "yes", "logp_image": -0.1, "logp_prior": -0.1},
                {"key": "no", "logp_image": -0.5, "logp_prior": -1.0},
            ],
        },
    }


class GroupHoldoutTest(unittest.TestCase):
    def test_heldout_group_is_absent_from_each_fit(self):
        rows = []
        for split in ("development", "test"):
            for task in ("mc", "qa"):
                for group in ("A", "B"):
                    rows.append(make_row(split, task, group, f"{split}-{task}-{group}"))
        fits = []

        def fake_fit(all_rows, contexts, train_ids, *args, **kwargs):
            fits.append({all_rows[index]["subcategory"] for index in train_ids})
            predictions = [row["cp_vbc"]["baseline_key"] for row in all_rows]
            return predictions, [{} for _ in all_rows], {"synthetic": True}

        with patch(
            "analyze_hierarchical_eb_group_holdout.fit_and_select", side_effect=fake_fit
        ):
            result = run_group_holdout(
                rows,
                "subcategory",
                max_cs_drop=0.1,
                quadrature_points=8,
                bootstrap=2,
                seed=1,
                selection_objective="net_utility",
                cs_cost=0.5,
            )

        self.assertEqual(fits, [{"B"}, {"A"}])
        self.assertEqual(result["heldout_rows"], 4)


if __name__ == "__main__":
    unittest.main()
