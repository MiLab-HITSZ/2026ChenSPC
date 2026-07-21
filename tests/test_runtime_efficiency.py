import unittest

from analyze_runtime_efficiency import percentile, summarize_rows


class RuntimeEfficiencyTest(unittest.TestCase):
    def test_percentile_interpolates(self):
        self.assertEqual(2.5, percentile([1, 2, 3, 4], 0.5))

    def test_cprc_call_structure(self):
        rows = [
            {
                "latency_ms": 100,
                "answer_latency_ms": 40,
                "cp_vbc": {"candidates": [{}, {}]},
            },
            {
                "latency_ms": 120,
                "answer_latency_ms": 50,
                "cp_vbc": {"candidates": [{}, {}, {}, {}]},
            },
        ]
        result = summarize_rows(rows)
        self.assertEqual(65.0, result["method_overhead_ms"]["median"])
        calls = result["call_structure"]
        self.assertEqual(6.0, calls["teacher_forced_candidate_forwards_per_row"]["median"])
        self.assertEqual(0, calls["post_score_h_eb_model_forwards"])


if __name__ == "__main__":
    unittest.main()
