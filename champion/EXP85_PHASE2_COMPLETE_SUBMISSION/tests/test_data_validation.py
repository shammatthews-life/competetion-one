from __future__ import annotations

import sys
import unittest
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "source"))

from data_validation import validate_frames


def frames():
    train = pd.DataFrame(
        {"SK_ID_CURR": [1, 2, 3, 4], "feature": [1.0, 2.0, 3.0, 4.0], "TARGET": [0, 1, 0, 1]}
    )
    test = pd.DataFrame({"SK_ID_CURR": [5, 6], "feature": [5.0, 6.0]})
    sample = pd.DataFrame({"SK_ID_CURR": [5, 6], "TARGET": [0, 0]})
    return train, test, sample


class DataValidationTests(unittest.TestCase):
    def test_valid_schema_target_and_id_order(self):
        labels, metadata = validate_frames(*frames())
        self.assertEqual(labels.tolist(), [0, 1, 0, 1])
        self.assertTrue(metadata["test_ids_equal_sample_order"])

    def test_rejects_missing_target_or_test_target(self):
        train, test, sample = frames()
        with self.assertRaisesRegex(ValueError, "TARGET"):
            validate_frames(train.drop(columns="TARGET"), test, sample)
        with self.assertRaisesRegex(ValueError, "exclude"):
            validate_frames(train, test.assign(TARGET=0), sample)

    def test_rejects_incompatible_predictor_order_and_sample_schema(self):
        train, test, sample = frames()
        with self.assertRaisesRegex(ValueError, "incompatible"):
            validate_frames(train, test[["feature", "SK_ID_CURR"]], sample)
        with self.assertRaisesRegex(ValueError, "sample schema"):
            validate_frames(train, test, sample[["TARGET", "SK_ID_CURR"]])

    def test_rejects_duplicate_ids_and_nonbinary_labels(self):
        train, test, sample = frames()
        with self.assertRaisesRegex(ValueError, "duplicated"):
            validate_frames(train, test.assign(SK_ID_CURR=[5, 5]), sample)
        train.loc[0, "TARGET"] = 2
        with self.assertRaisesRegex(ValueError, "binary"):
            validate_frames(train, test, sample)


if __name__ == "__main__":
    unittest.main()
