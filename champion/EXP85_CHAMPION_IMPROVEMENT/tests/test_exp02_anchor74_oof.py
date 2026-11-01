import unittest

import numpy as np
import pandas as pd

from experiments.exp02_anchor74_oof.run_anchor74_oof import (
    calibrate_fold,
    compute_regime_delta,
    entity_group_ids,
    make_grouped_splits,
    rank_probit,
    tail_sharpen,
)


class Anchor74OofTests(unittest.TestCase):
    def test_rank_probit_is_finite_and_order_preserving(self):
        values = np.asarray([0.4, 0.1, 0.3, 0.2])
        scores = rank_probit(values)
        self.assertTrue(np.isfinite(scores).all())
        self.assertTrue(np.all(np.diff(scores[np.argsort(values)]) > 0))

    def test_tail_sharpen_preserves_central_values(self):
        values = np.asarray([-1.0, 0.0, 1.0])
        np.testing.assert_array_equal(tail_sharpen(values), values)

    def test_regime_delta_requires_five_models_and_aligns_rows(self):
        models = np.arange(20, dtype=np.float64).reshape(4, 5)
        result = compute_regime_delta(models, np.asarray([0, 1, 2, 3]))
        self.assertEqual(result.shape, (4,))
        with self.assertRaises(ValueError):
            compute_regime_delta(models[:, :4], np.asarray([0, 1, 2, 3]))

    def test_fold_calibration_does_not_read_heldout_targets(self):
        fit = pd.DataFrame(
            {
                "DAYS_BIRTH": [-15000, -15001],
                "DAYS_REGISTRATION": [-1000, -1001],
                "DAYS_ID_PUBLISH": [-3000, -3001],
                "CODE_GENDER": ["F", "F"],
                "NAME_EDUCATION_TYPE": ["Higher", "Higher"],
                "DAYS_EMPLOYED": [-500, -501],
                "DAYS_LAST_PHONE_CHANGE": [-200, -201],
            }
        )
        valid = pd.DataFrame(
            {
                "DAYS_BIRTH": [-15002],
                "DAYS_REGISTRATION": [-1002],
                "DAYS_ID_PUBLISH": [-3002],
                "CODE_GENDER": ["F"],
                "NAME_EDUCATION_TYPE": ["Higher"],
                "DAYS_EMPLOYED": [-502],
                "DAYS_LAST_PHONE_CHANGE": [-202],
            }
        )
        z_models = np.asarray([[0.1, 0.2, 0.3, 0.4, 0.5]])
        pred_a, counts_a = calibrate_fold(
            np.asarray([0.2]), z_models, np.asarray([0]), fit, valid,
            np.asarray([0, 0])
        )
        pred_b, counts_b = calibrate_fold(
            np.asarray([0.2]), z_models, np.asarray([0]), fit, valid,
            np.asarray([1, 1])
        )
        self.assertEqual(counts_a["train_to_validation_pairs"], 2)
        self.assertEqual(counts_a, counts_b)
        self.assertNotEqual(float(pred_a[0]), float(pred_b[0]))

    def test_grouped_folds_keep_entity_groups_disjoint(self):
        rows = []
        labels = []
        for group in range(20):
            target = group % 2
            for repeat in range(2):
                birth = -12000 - group * 20
                rows.append(
                    {
                        "DAYS_BIRTH": birth,
                        "DAYS_REGISTRATION": birth + 100 + group,
                        "DAYS_ID_PUBLISH": birth + 400 + group * 3,
                        "CODE_GENDER": "F" if group % 2 else "M",
                        "NAME_EDUCATION_TYPE": "Higher",
                        "DAYS_EMPLOYED": -500 - repeat,
                        "DAYS_LAST_PHONE_CHANGE": -200 - repeat,
                    }
                )
                labels.append(target)
        frame = pd.DataFrame(rows)
        y = np.asarray(labels, dtype=np.int32)
        groups = entity_group_ids(frame)
        self.assertEqual(len(np.unique(groups)), 20)
        splits = make_grouped_splits(frame, y, groups)
        self.assertEqual(len(splits), 5)
        for fit_idx, valid_idx in splits:
            self.assertFalse(set(groups[fit_idx]) & set(groups[valid_idx]))


if __name__ == "__main__":
    unittest.main()
