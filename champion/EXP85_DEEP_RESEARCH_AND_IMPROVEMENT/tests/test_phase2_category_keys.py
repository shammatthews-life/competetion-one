from __future__ import annotations

import sys
import unittest
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "source"))

from phase2_category_keys import CATEGORY_SPECS, MISSING_KEY, build_category_keys


class CategoryKeyTests(unittest.TestCase):
    def setUp(self) -> None:
        self.train = pd.DataFrame(
            {
                "ORGANIZATION_TYPE": ["Business", None, "School"],
                "OCCUPATION_TYPE": ["Laborer", "Managers", None],
                "NAME_INCOME_TYPE": ["Working", "Pensioner", "Working"],
                "NAME_EDUCATION_TYPE": ["Higher", "Secondary", "Higher"],
            }
        )
        self.test = self.train.iloc[:2].copy().reset_index(drop=True)

    def test_keys_are_ordered_and_missing_categories_are_explicit(self) -> None:
        keys = build_category_keys(self.train, self.test)
        self.assertEqual(list(keys), [name for name, _, _ in CATEGORY_SPECS])
        self.assertEqual(len(keys["ORG_TYPE"]), len(self.train) + len(self.test))
        self.assertEqual(keys["ORG_TYPE"][1], MISSING_KEY)
        self.assertEqual(keys["ORG_TYPE"][3], "Business")
        self.assertTrue(all(values.dtype.kind in "US" for values in keys.values()))

    def test_target_and_ids_are_rejected(self) -> None:
        train = self.train.assign(TARGET=[0, 1, 0])
        with self.assertRaisesRegex(ValueError, "exclude TARGET and ID"):
            build_category_keys(train, self.test)
        test = self.test.assign(SK_ID_CURR=[10, 20])
        with self.assertRaisesRegex(ValueError, "exclude TARGET and ID"):
            build_category_keys(self.train, test)

if __name__ == "__main__":
    unittest.main()
