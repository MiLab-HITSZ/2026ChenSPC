import math
import unittest

import numpy as np

from analyze_hierarchical_eb_lambda_cv import inverse_softplus
from analyze_nolan_anchored_spc import nolan_lambda, nolan_predictions


class NolanAnchoredSpcTest(unittest.TestCase):
    def setUp(self):
        self.row = {
            "cp_vbc": {
                "candidates": [
                    {"key": "a", "logp_image": -0.2, "logp_prior": -0.1},
                    {"key": "b", "logp_image": -0.3, "logp_prior": -3.0},
                ]
            }
        }

    def test_anchor_is_exact_at_zero_latent_residual(self):
        anchor = nolan_lambda(self.row, beta=0.8)
        recovered = math.log1p(math.exp(inverse_softplus(anchor)))
        self.assertAlmostEqual(recovered, anchor)

    def test_candidate_ranking_matches_nolan_score_rule(self):
        anchor = nolan_lambda(self.row, beta=0.8)
        self.assertEqual(nolan_predictions([self.row], np.asarray([anchor])), ["b"])

    def test_anchor_strength_increases_monotonically_with_beta(self):
        values = [nolan_lambda(self.row, beta) for beta in (0.2, 0.8, 1.6)]
        self.assertLess(values[0], values[1])
        self.assertLess(values[1], values[2])
        self.assertTrue(all(0.0 < value < 1.0 for value in values))


if __name__ == "__main__":
    unittest.main()
