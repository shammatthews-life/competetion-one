from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


ID_COLUMN = "SK_ID_CURR"
TARGET_COLUMN = "TARGET"


def validate_submission(
    candidate: pd.DataFrame,
    test: pd.DataFrame,
    sample: pd.DataFrame,
) -> dict[str, Any]:
    if list(candidate.columns) != list(sample.columns):
        raise ValueError("Submission columns/order differ from sample_submission.csv.")
    if list(candidate.columns) != [ID_COLUMN, TARGET_COLUMN]:
        raise ValueError("Submission schema must be SK_ID_CURR,TARGET.")
    if len(candidate) != len(test) or len(candidate) != len(sample):
        raise ValueError("Submission row count differs from test/sample input.")
    if candidate[ID_COLUMN].isna().any() or candidate[ID_COLUMN].duplicated().any():
        raise ValueError("Submission IDs are missing or duplicated.")
    if not np.array_equal(candidate[ID_COLUMN].to_numpy(), test[ID_COLUMN].to_numpy()):
        raise ValueError("Submission IDs/order differ from test.csv.")
    if not np.array_equal(candidate[ID_COLUMN].to_numpy(), sample[ID_COLUMN].to_numpy()):
        raise ValueError("Submission IDs/order differ from sample_submission.csv.")
    values = pd.to_numeric(candidate[TARGET_COLUMN], errors="coerce").to_numpy(
        dtype=np.float64
    )
    if not np.isfinite(values).all() or np.any((values < 0.0) | (values > 1.0)):
        raise ValueError("Submission contains non-finite or out-of-range probabilities.")
    return {
        "status": "PASS",
        "rows": int(len(candidate)),
        "schema": list(candidate.columns),
        "ids_unique": True,
        "ids_match_test_order": True,
        "ids_match_sample_order": True,
        "probabilities_finite": True,
        "probability_min": float(values.min()),
        "probability_max": float(values.max()),
    }


def build_submission(
    test: pd.DataFrame,
    sample: pd.DataFrame,
    predictions: np.ndarray,
    path: Path,
) -> dict[str, Any]:
    predictions = np.asarray(predictions, dtype=np.float64)
    if predictions.ndim != 1 or len(predictions) != len(test):
        raise ValueError("Final predictions must be a vector aligned to test rows.")
    candidate = pd.DataFrame(
        {ID_COLUMN: test[ID_COLUMN].to_numpy(), TARGET_COLUMN: predictions}
    )
    validate_submission(candidate, test, sample)
    path.parent.mkdir(parents=True, exist_ok=True)
    candidate.to_csv(path, index=False, float_format="%.10f")
    read_back = pd.read_csv(path)
    return validate_submission(read_back, test, sample)
