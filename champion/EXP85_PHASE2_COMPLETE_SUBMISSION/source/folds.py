from __future__ import annotations

import numpy as np
from sklearn.model_selection import StratifiedKFold


def build_folds(
    labels: np.ndarray, n_splits: int = 5, seed: int = 42
) -> list[tuple[np.ndarray, np.ndarray]]:
    labels = np.asarray(labels)
    if labels.ndim != 1 or len(labels) == 0:
        raise ValueError("Fold labels must be a non-empty vector.")
    if n_splits < 2:
        raise ValueError("At least two folds are required.")
    splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    return [
        (fit.astype(np.int64), valid.astype(np.int64))
        for fit, valid in splitter.split(np.zeros(len(labels)), labels)
    ]


def fold_id_vector(
    labels: np.ndarray, folds: list[tuple[np.ndarray, np.ndarray]]
) -> np.ndarray:
    result = np.full(len(labels), -1, dtype=np.int16)
    for fold_number, (_, valid) in enumerate(folds, 1):
        if np.any(result[valid] != -1):
            raise ValueError("Validation fold assignments overlap.")
        result[valid] = fold_number
    if np.any(result < 1):
        raise ValueError("Validation folds do not cover every training row.")
    return result
