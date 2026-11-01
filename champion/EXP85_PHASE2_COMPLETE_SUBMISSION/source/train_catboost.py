from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.metrics import roc_auc_score

from gpu_telemetry import GpuMonitor


def train_catboost_fold(
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
    if settings.get("task_type") != "GPU" or settings.get("devices") != "0":
        raise ValueError("Selected Phase 2 CatBoost configuration requires GPU 0.")
    model = CatBoostClassifier(
        iterations=int(settings["iterations"]),
        depth=int(settings["depth"]),
        learning_rate=float(settings["learning_rate"]),
        loss_function=settings["loss_function"],
        eval_metric=settings["eval_metric"],
        l2_leaf_reg=float(settings["l2_leaf_reg"]),
        random_strength=float(settings["random_strength"]),
        bootstrap_type=settings["bootstrap_type"],
        subsample=float(settings["subsample"]),
        border_count=int(settings["border_count"]),
        task_type=settings["task_type"],
        devices=settings["devices"],
        random_seed=seed,
        allow_writing_files=False,
        verbose=False,
        thread_count=int(settings["num_threads"]),
    )
    monitor = GpuMonitor()
    with monitor:
        model.fit(
            X_train.iloc[fit_indices],
            labels[fit_indices],
            cat_features=list(settings["categorical_columns"]),
            eval_set=(X_train.iloc[valid_indices], labels[valid_indices]),
            early_stopping_rounds=int(settings["early_stopping_rounds"]),
            verbose=False,
        )
    valid_prediction = model.predict_proba(X_train.iloc[valid_indices])[:, 1]
    test_prediction = model.predict_proba(X_test)[:, 1]
    model_path.parent.mkdir(parents=True, exist_ok=True)
    model.save_model(str(model_path))
    importance = pd.DataFrame(
        {
            "feature": X_train.columns,
            "importance": model.get_feature_importance(),
            "fold": fold,
        }
    )
    telemetry = monitor.report()
    return {
        "valid_prediction": valid_prediction.astype(np.float32),
        "test_prediction": test_prediction.astype(np.float32),
        "best_iteration": int(model.best_iteration_),
        "auc": float(roc_auc_score(labels[valid_indices], valid_prediction)),
        "importance": importance,
        "device": "GPU:0",
        "model_path": str(model_path),
        "gpu_telemetry": telemetry,
    }
