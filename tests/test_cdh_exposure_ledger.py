import json
import tempfile
import unittest
from pathlib import Path

from build_cdh_exposure_ledger import build_ledger


class CDHExposureLedgerTest(unittest.TestCase):
    def test_separates_baseline_and_method_exposure(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            benchmark = root / "benchmark.jsonl"
            benchmark.write_text(
                "\n".join(
                    json.dumps(
                        {
                            "pair_id": f"Pair {index}",
                            "category": "Synthetic",
                            "subcategory": "Type A",
                            "pair_name": f"case {index}",
                        }
                    )
                    for index in range(1, 4)
                )
                + "\n"
            )
            baseline = root / "result" / "Model" / "results.jsonl"
            baseline.parent.mkdir(parents=True)
            baseline.write_text(
                json.dumps(
                    {
                        "pair_id": "Pair 1",
                        "subcategory": "Type A",
                        "image_path": "images/Type_A/Pair_1/counterfactual.png",
                        "task": "mc",
                    }
                )
                + "\n"
            )
            method = root / "result" / "cp_vbc_trial" / "results.jsonl"
            method.parent.mkdir(parents=True)
            method.write_text(
                json.dumps(
                    {
                        "pair_id": "Pair 2",
                        "subcategory": "Type A",
                        "image_path": "images/Type_A/Pair_2/counterfactual.png",
                        "task": "qa",
                    }
                )
                + "\n"
            )

            output = build_ledger(benchmark, root / "result")

            self.assertEqual(output["summary"]["baseline_exposed_pairs"], 1)
            self.assertEqual(output["summary"]["method_exposed_pairs"], 1)
            self.assertEqual(output["summary"]["cpr_exposed_pairs"], 1)
            self.assertEqual(output["summary"]["mitigation_unseen_pairs"], 2)
            self.assertEqual(
                output["split_manifests"]["mitigation_unseen"]["pair_ids"],
                ["Pair 1", "Pair 3"],
            )


if __name__ == "__main__":
    unittest.main()
