from __future__ import annotations

from pathlib import Path
from typing import Any

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score


def train_lightgbm_fold(
    X_train: pd.DataFrame,
    labels: np.ndarray,
    X_test: pd.DataFrame,
    fit_indices: np.ndarray,
    valid_indices: np.ndarray,
    fold: int,
    settings: dict[str, Any],
    seed: int,
    model_path: Path,
) -> dict[str, Any]:
    params = {
        key: value
        for key, value in settings.items()
        if key not in ("device", "num_boost_round", "early_stopping_rounds")
    }
    params["seed"] = seed
    params["num_threads"] = min(int(settings["num_threads"]), 8)
    train_set = lgb.Dataset(
        X_train.iloc[fit_indices], label=labels[fit_indices], free_raw_data=True
    )
    valid_set = lgb.Dataset(
        X_train.iloc[valid_indices],
        label=labels[valid_indices],
        reference=train_set,
        free_raw_data=True,
    )
    model = lgb.train(
        params,
        train_set,
        num_boost_round=int(settings["num_boost_round"]),
        valid_sets=[valid_set],
        callbacks=[
            lgb.early_stopping(
                int(settings["early_stopping_rounds"]), verbose=False
            )
        ],
    )
    valid_prediction = model.predict(
        X_train.iloc[valid_indices], num_iteration=model.best_iteration
    )
    test_prediction = model.predict(X_test, num_iteration=model.best_iteration)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    model.save_model(str(model_path), num_iteration=model.best_iteration)
    importance = pd.DataFrame(
        {
            "feature": X_train.columns,
            "gain": model.feature_importance(importance_type="gain"),
            "split": model.feature_importance(importance_type="split"),
            "fold": fold,
        }
    )
    return {
        "valid_prediction": valid_prediction.astype(np.float32),
        "test_prediction": test_prediction.astype(np.float32),
        "best_iteration": int(model.best_iteration),
        "auc": float(roc_auc_score(labels[valid_indices], valid_prediction)),
        "importance": importance,
        "device": "CPU",
        "model_path": str(model_path),
    }
