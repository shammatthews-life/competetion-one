from __future__ import annotations

import numpy as np
import pandas as pd


CATEGORY_SPECS = (
    ("ORG_TYPE", "ORGANIZATION_TYPE", 100.0),
    ("OCC_TYPE", "OCCUPATION_TYPE", 50.0),
    ("INCOME_TYPE", "NAME_INCOME_TYPE", 30.0),
    ("EDUCATION_TYPE", "NAME_EDUCATION_TYPE", 30.0),
)
MISSING_KEY = "__MISSING_KEY__"


def build_category_keys(
    train_raw: pd.DataFrame, test_raw: pd.DataFrame
) -> dict[str, np.ndarray]:
    for label, frame in (("train", train_raw), ("test", test_raw)):
        if "TARGET" in frame.columns or "SK_ID_CURR" in frame.columns:
            raise ValueError(f"{label} category input must exclude TARGET and ID")
        missing = {
            column for _, column, _ in CATEGORY_SPECS
        } - set(frame.columns)
        if missing:
            raise ValueError(
                f"{label} category input is missing: {sorted(missing)}"
            )

    combined = pd.concat(
        [
            train_raw.loc[:, [column for _, column, _ in CATEGORY_SPECS]],
            test_raw.loc[:, [column for _, column, _ in CATEGORY_SPECS]],
        ],
        ignore_index=True,
    )
    result = {}
    for name, column, _ in CATEGORY_SPECS:
        result[name] = (
            combined[column]
            .astype("string")
            .fillna(MISSING_KEY)
            .astype(str)
            .to_numpy(dtype=str)
        )
    expected_length = len(train_raw) + len(test_raw)
    if any(len(values) != expected_length for values in result.values()):
        raise ValueError("Category key arrays do not align with train/test rows")
    return result
