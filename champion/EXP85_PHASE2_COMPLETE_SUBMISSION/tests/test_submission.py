from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "source"))

from data_validation import sha256_file
from submission_builder import build_submission, validate_submission


class SubmissionTests(unittest.TestCase):
    def setUp(self):
        self.test = pd.DataFrame({"SK_ID_CURR": [10, 20, 30]})
        self.sample = pd.DataFrame({"SK_ID_CURR": [10, 20, 30], "TARGET": [0, 0, 0]})

    def test_build_writes_and_reloads_competition_schema(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "submission.csv"
            checks = build_submission(
                self.test, self.sample, np.asarray([0.2, 0.5, 0.8]), path
            )
            self.assertEqual(checks["status"], "PASS")
            self.assertEqual(list(pd.read_csv(path).columns), ["SK_ID_CURR", "TARGET"])
            self.assertEqual(len(sha256_file(path)), 64)

    def test_duplicate_ids_bad_order_and_invalid_probabilities_fail(self):
        invalid = pd.DataFrame(
            {"SK_ID_CURR": [10, 10, 30], "TARGET": [0.2, 0.5, 0.8]}
        )
        with self.assertRaisesRegex(ValueError, "duplicated"):
            validate_submission(invalid, self.test, self.sample)
        invalid = pd.DataFrame(
            {"SK_ID_CURR": [20, 10, 30], "TARGET": [0.2, 0.5, 0.8]}
        )
        with self.assertRaisesRegex(ValueError, "order"):
            validate_submission(invalid, self.test, self.sample)
        invalid = pd.DataFrame(
            {"SK_ID_CURR": [10, 20, 30], "TARGET": [0.2, np.inf, 0.8]}
        )
        with self.assertRaisesRegex(ValueError, "non-finite"):
            validate_submission(invalid, self.test, self.sample)


if __name__ == "__main__":
    unittest.main()
