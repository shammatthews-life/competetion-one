import unittest

import numpy as np

from experiments.exp01_meta_weights import FIXED_WEIGHTS, MEMBERS, ranked_probit


class MetaWeightsTests(unittest.TestCase):
    def test_fixed_weights_are_positive_and_normalizable(self):
        self.assertEqual(len(FIXED_WEIGHTS), len(MEMBERS))
        self.assertTrue(np.all(FIXED_WEIGHTS > 0.0))
        normalized = FIXED_WEIGHTS / FIXED_WEIGHTS.sum()
        self.assertAlmostEqual(float(normalized.sum()), 1.0)

    def test_ranked_probit_is_finite_and_order_preserving(self):
        values = np.asarray([0.3, 0.1, 0.2, 0.4])
        scores = ranked_probit(values)
        self.assertTrue(np.isfinite(scores).all())
        self.assertTrue(np.all(np.diff(scores[np.argsort(values)]) > 0.0))

    def test_ranked_probit_rejects_nonfinite_inputs(self):
        with self.assertRaises(ValueError):
            ranked_probit(np.asarray([0.1, np.nan, 0.9]))


if __name__ == "__main__":
    unittest.main()
