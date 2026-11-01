from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "source"))

from prediction_blender import combine_phase2_predictions, mean_fold_predictions


class BlendingTests(unittest.TestCase):
    def test_exact_half_probability_blend(self):
        lgb = np.asarray([0.2, 0.4], dtype=np.float32)
        cat = np.asarray([0.6, 0.8], dtype=np.float32)
        np.testing.assert_allclose(
            combine_phase2_predictions(lgb, cat), [0.4, 0.6], rtol=0.0, atol=2e-8
        )

    def test_average_fold_test_predictions_before_blend(self):
        result = mean_fold_predictions(
            [np.asarray([0.1, 0.5]), np.asarray([0.3, 0.9])]
        )
        np.testing.assert_array_equal(result, np.asarray([0.2, 0.7], dtype=np.float32))

    def test_rejects_shape_nonfinite_and_out_of_range_inputs(self):
        with self.assertRaisesRegex(ValueError, "shapes differ"):
            combine_phase2_predictions(np.asarray([0.2]), np.asarray([0.2, 0.3]))
        with self.assertRaisesRegex(ValueError, "finite"):
            combine_phase2_predictions(np.asarray([np.nan]), np.asarray([0.2]))
        with self.assertRaisesRegex(ValueError, "lie in"):
            combine_phase2_predictions(np.asarray([1.2]), np.asarray([0.2]))


if __name__ == "__main__":
    unittest.main()
