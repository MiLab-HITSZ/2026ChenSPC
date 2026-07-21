import json
import tempfile
import unittest
from pathlib import Path

from analyze_official_revis_transfer import exact_mcnemar, load_rows, summarize
from spc_vlm_runtime import extract_first_letter, extract_yes_no, load_model_specs
from evaluate_paired_qa_candidate_scores import load_baselines


class OfficialRevisTransferTest(unittest.TestCase):
    def test_summary_counts_repairs_and_harms(self):
        rows = [
            {
                "baseline_correct": False,
                "steered_correct": True,
                "changed": True,
                "baseline_key": "yes",
                "steered_key": "no",
                "revis": {
                    "gate_observation": {
                        "positions": 10,
                        "triggered_positions": 2,
                        "risk_sum": -1.0,
                    }
                },
            },
            {
                "baseline_correct": True,
                "steered_correct": False,
                "changed": True,
                "baseline_key": "no",
                "steered_key": "yes",
            },
            {"baseline_correct": True, "steered_correct": True, "changed": False},
        ]
        result = summarize(rows)
        self.assertEqual(result["repairs"], 1)
        self.assertEqual(result["harms"], 1)
        self.assertAlmostEqual(result["delta_accuracy"], 0.0)
        self.assertEqual(result["answer_transitions"], {"no->yes": 1, "yes->no": 1})
        self.assertAlmostEqual(result["gate"]["trigger_rate"], 0.2)

    def test_exact_mcnemar_handles_no_changes(self):
        self.assertEqual(exact_mcnemar(0, 0), 1.0)

    def test_load_rows_accepts_disjoint_shards(self):
        with tempfile.TemporaryDirectory() as directory:
            paths = []
            for index, side in enumerate(("counterfactual", "commonsense")):
                path = Path(directory) / f"shard{index}.jsonl"
                path.write_text(
                    json.dumps(
                        {
                            "status": "ok",
                            "pair_id": "Pair 1",
                            "task": "qa",
                            "side": side,
                            "subcategory": "test",
                            "gt": "yes",
                            "pred": "yes",
                            "correct": True,
                            "revis": {
                                "baseline_prediction": "no",
                                "steered_prediction": "yes",
                            },
                        }
                    )
                    + "\n",
                    encoding="utf-8",
                )
                paths.append(path)
            rows = load_rows(paths)
        self.assertEqual(len(rows), 2)
        self.assertEqual({row["side"] for row in rows}, {"counterfactual", "commonsense"})

    def test_precision_matched_baseline_loader_uses_revis_baseline(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "results.jsonl"
            path.write_text(
                json.dumps(
                    {
                        "status": "ok",
                        "pair_id": "Pair 1",
                        "task": "qa",
                        "side": "counterfactual",
                        "revis": {"baseline_prediction": "yes"},
                        "raw": {"baseline_raw": {"dtype": "bfloat16"}},
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            baselines = load_baselines([path])
        self.assertEqual(
            baselines[("Pair 1", "qa", "counterfactual")],
            ("yes", {"dtype": "bfloat16"}),
        )

    def test_lean_runtime_preserves_answer_parsing_and_precision(self):
        self.assertEqual(extract_yes_no("Final answer: No."), "no")
        self.assertEqual(extract_first_letter("Final answer: C"), "C")
        spec = load_model_specs("configs/qwen_32b_instruct_bf16_fair.json")[0]
        self.assertEqual(spec.quantization, "none")
        self.assertEqual(spec.dtype, "bfloat16")


if __name__ == "__main__":
    unittest.main()
