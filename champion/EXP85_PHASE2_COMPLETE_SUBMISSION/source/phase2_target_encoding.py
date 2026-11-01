from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold


MISSING_KEY = "__MISSING_KEY__"
KEY_SPECS = (
    ("FEE_Q3", "ANNUITY_CREDIT_RATIO_DECODED_ROUND_3", 100.0),
    ("FEE_Q4", "ANNUITY_CREDIT_RATIO_DECODED_ROUND_4", 30.0),
    ("FEE_Q5", "ANNUITY_CREDIT_RATIO_DECODED_ROUND_5", 30.0),
    ("PRODUCT_K", "PRODUCT_KEY_K", 50.0),
)
INNER_SPLITS = 5
INNER_SEED = 7


def key_arrays(
    fee_train: pd.DataFrame, fee_test: pd.DataFrame
) -> dict[str, np.ndarray]:
    combined = pd.concat([fee_train, fee_test], ignore_index=True)
    result = {}
    for name, column, _ in KEY_SPECS:
        values = combined[column].round(0 if name == "PRODUCT_K" else int(column[-1]))
        result[name] = values.astype("string").fillna(MISSING_KEY).to_numpy(dtype=str)
    return result


def encode_keys(
    keys: np.ndarray,
    labels: np.ndarray,
    fit_indices: np.ndarray,
    apply_indices: np.ndarray,
    key_name: str,
    smoothing: float,
) -> tuple[np.ndarray, np.ndarray]:
    keys = np.asarray(keys)
    labels = np.asarray(labels)
    fit_indices = np.asarray(fit_indices, dtype=np.int64)
    apply_indices = np.asarray(apply_indices, dtype=np.int64)
    if keys.ndim != 1 or labels.ndim != 1:
        raise ValueError("Target-encoding keys and labels must be one-dimensional.")
    if len(keys) < len(labels) or not np.isfinite(labels).all():
        raise ValueError("Invalid target-encoding key or label arrays.")
    if fit_indices.size == 0 or np.any(fit_indices < 0) or np.any(fit_indices >= len(labels)):
        raise ValueError("Target-encoding fit indices are empty or out of label bounds.")
    if np.any(apply_indices < 0) or np.any(apply_indices >= len(keys)):
        raise ValueError("Target-encoding apply indices are out of key bounds.")
    if np.intersect1d(fit_indices, apply_indices).size:
        raise ValueError("Target-encoding fit and apply indices overlap.")
    if not np.isfinite(smoothing) or smoothing <= 0:
        raise ValueError("Target-encoding smoothing must be positive and finite.")

    prior = float(np.mean(labels[fit_indices]))
    fit = pd.DataFrame(
        {"key": keys[fit_indices], "target": labels[fit_indices].astype(np.float64)}
    )
    grouped = fit.groupby("key", sort=False)["target"].agg(["sum", "count"])
    smoothed = (grouped["sum"] + smoothing * prior) / (grouped["count"] + smoothing)
    apply_keys = pd.Series(keys[apply_indices])
    encoded = apply_keys.map(smoothed).fillna(prior).to_numpy(dtype=np.float32)
    counts = apply_keys.map(grouped["count"]).fillna(0).to_numpy(dtype=np.float32)
    return encoded, counts


def nested_training_encodings(
    keys_by_name: dict[str, np.ndarray],
    labels: np.ndarray,
    outer_train_indices: np.ndarray,
) -> pd.DataFrame:
    labels = np.asarray(labels)
    outer_train_indices = np.asarray(outer_train_indices, dtype=np.int64)
    if np.any(outer_train_indices < 0) or np.any(outer_train_indices >= len(labels)):
        raise ValueError("Outer training indices are out of label bounds.")
    output = np.full((len(outer_train_indices), 2 * len(KEY_SPECS)), np.nan, dtype=np.float32)
    inner = StratifiedKFold(n_splits=INNER_SPLITS, shuffle=True, random_state=INNER_SEED)
    local_y = labels[outer_train_indices]
    for inner_fit_local, inner_apply_local in inner.split(
        np.zeros(len(outer_train_indices)), local_y
    ):
        inner_fit = outer_train_indices[inner_fit_local]
        inner_apply = outer_train_indices[inner_apply_local]
        if np.intersect1d(inner_fit, inner_apply).size:
            raise RuntimeError("Nested target-encoding split overlaps.")
        for key_index, (name, _, smoothing) in enumerate(KEY_SPECS):
            encoded, count = encode_keys(
                keys_by_name[name], labels, inner_fit, inner_apply, name, smoothing
            )
            output[inner_apply_local, 2 * key_index] = encoded
            output[inner_apply_local, 2 * key_index + 1] = np.log1p(count)
    if not np.isfinite(output).all():
        raise ValueError("Nested target encodings are incomplete or non-finite.")
    columns = [
        column
        for name, _, _ in KEY_SPECS
        for column in (f"TE_{name}", f"LOG_COUNT_{name}")
    ]
    return pd.DataFrame(output, columns=columns)


def apply_training_encodings(
    keys_by_name: dict[str, np.ndarray],
    labels: np.ndarray,
    fit_indices: np.ndarray,
    apply_indices: np.ndarray,
) -> pd.DataFrame:
    values = {}
    for name, _, smoothing in KEY_SPECS:
        encoded, count = encode_keys(
            keys_by_name[name], labels, fit_indices, apply_indices, name, smoothing
        )
        values[f"TE_{name}"] = encoded
        values[f"LOG_COUNT_{name}"] = np.log1p(count)
    return pd.DataFrame(values).reset_index(drop=True)
