import json
import tempfile
import unittest
from pathlib import Path

from analyze_cp_vbc_frozen_transfer import (
    exact_mcnemar_pvalue,
    predictions_for_params,
    run_frozen_transfer,
)


def row(pair_id: str, side: str = "counterfactual"):
    return {
        "status": "ok",
        "task": "qa",
        "pair_id": pair_id,
        "side": side,
        "subcategory": "Synthetic",
        "gt": "no",
        "cp_vbc": {
            "baseline_key": "yes",
            "candidates": [
                {"key": "yes", "logp_image": -0.1, "logp_prior": -0.1},
                {"key": "no", "logp_image": -0.8, "logp_prior": -2.0},
            ],
        },
    }


class CPVBCFrozenTransferTest(unittest.TestCase):
    def test_fixed_prediction_uses_only_supplied_params(self):
        predictions = predictions_for_params(
            [row("Pair 1")],
            {
                "family": "fixed",
                "lambda": 1.0,
                "contrast_margin_min": 0.0,
                "absorption_margin_max": 2.0,
                "visual_margin_min": 0.0,
                "prior_relief_min": 0.0,
            },
        )
        self.assertEqual(predictions, ["no"])

    def test_rejects_development_test_pair_overlap_before_selection(self):
        with tempfile.TemporaryDirectory() as temporary:
            development = Path(temporary) / "development.jsonl"
            test = Path(temporary) / "test.jsonl"
            payload = json.dumps(row("Pair 1")) + "\n"
            development.write_text(payload)
            test.write_text(payload)

            with self.assertRaisesRegex(ValueError, "pair overlap"):
                run_frozen_transfer(
                    development,
                    test,
                    "qa",
                    0.2,
                    "net_utility",
                    0.5,
                    3,
                    2,
                    13,
                )

    def test_imported_exact_test_is_available(self):
        self.assertAlmostEqual(exact_mcnemar_pvalue(3, 0), 0.25)


if __name__ == "__main__":
    unittest.main()
