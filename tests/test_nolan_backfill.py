import unittest

from analyze_nolan_backfill import backfill_predictions


class NolanBackfillTest(unittest.TestCase):
    def test_primary_override_has_priority(self):
        rows = [
            {"cp_vbc": {"baseline_key": "a"}},
            {"cp_vbc": {"baseline_key": "a"}},
        ]
        self.assertEqual(
            backfill_predictions(rows, ["b", "a"], ["c", "c"]),
            ["b", "c"],
        )


if __name__ == "__main__":
    unittest.main()
