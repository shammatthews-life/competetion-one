from __future__ import annotations

import numpy as np


def _validate_prediction(values: np.ndarray, name: str) -> np.ndarray:
    values = np.asarray(values)
    if values.ndim != 1:
        raise ValueError(f"{name} must be a one-dimensional prediction vector.")
    if not np.issubdtype(values.dtype, np.number) or not np.isfinite(values).all():
        raise ValueError(f"{name} must contain finite numeric predictions.")
    if np.any((values < 0.0) | (values > 1.0)):
        raise ValueError(f"{name} probabilities must lie in [0, 1].")
    return values


def combine_phase2_predictions(
    lightgbm_predictions: np.ndarray, catboost_predictions: np.ndarray
) -> np.ndarray:
    lightgbm_predictions = _validate_prediction(lightgbm_predictions, "LightGBM")
    catboost_predictions = _validate_prediction(catboost_predictions, "CatBoost")
    if lightgbm_predictions.shape != catboost_predictions.shape:
        raise ValueError("LightGBM and CatBoost prediction shapes differ.")
    return (
        lightgbm_predictions.astype(np.float64)
        + catboost_predictions.astype(np.float64)
    ) / 2.0


def mean_fold_predictions(predictions: list[np.ndarray]) -> np.ndarray:
    if not predictions:
        raise ValueError("No fold predictions were provided.")
    arrays = [_validate_prediction(value, f"fold_{index}") for index, value in enumerate(predictions, 1)]
    if any(array.shape != arrays[0].shape for array in arrays):
        raise ValueError("Fold prediction shapes differ.")
    return np.mean(np.stack(arrays), axis=0, dtype=np.float64).astype(np.float32)
