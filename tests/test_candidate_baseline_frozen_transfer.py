import unittest

from analyze_candidate_baseline_frozen_transfer import _exact_mcnemar, _quantile


class CandidateBaselineFrozenTransferTest(unittest.TestCase):
    def test_exact_mcnemar_is_two_sided(self):
        self.assertAlmostEqual(_exact_mcnemar(6, 4), 0.75390625)
        self.assertAlmostEqual(_exact_mcnemar(4, 6), 0.75390625)
        self.assertEqual(_exact_mcnemar(0, 0), 1.0)

    def test_quantile_interpolates(self):
        self.assertAlmostEqual(_quantile([0.0, 1.0], 0.25), 0.25)


if __name__ == "__main__":
    unittest.main()
