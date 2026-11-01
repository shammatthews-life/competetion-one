from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


ID_COLUMN = "SK_ID_CURR"
TARGET_COLUMN = "TARGET"
INPUT_NAMES = ("train.csv", "test.csv", "sample_submission.csv")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_frames(
    train: pd.DataFrame,
    test: pd.DataFrame,
    sample: pd.DataFrame,
    expected_target_hash: str | None = None,
) -> tuple[np.ndarray, dict[str, Any]]:
    if ID_COLUMN not in train or TARGET_COLUMN not in train:
        raise ValueError(f"Training input must include {ID_COLUMN} and {TARGET_COLUMN}.")
    if ID_COLUMN not in test or TARGET_COLUMN in test:
        raise ValueError(f"Test input must include {ID_COLUMN} and exclude {TARGET_COLUMN}.")
    if list(sample.columns) != [ID_COLUMN, TARGET_COLUMN]:
        raise ValueError(f"Unexpected sample schema: {sample.columns.tolist()}")
    expected_test_columns = [c for c in train.columns if c != TARGET_COLUMN]
    if expected_test_columns != list(test.columns):
        raise ValueError("Train predictors and test columns/order are incompatible.")
    if train.empty or test.empty or len(sample) != len(test):
        raise ValueError("Train, test, and sample inputs must be non-empty and aligned.")

    for label, frame in (("train", train), ("test", test), ("sample", sample)):
        if frame[ID_COLUMN].isna().any() or frame[ID_COLUMN].duplicated().any():
            raise ValueError(f"{label} IDs are missing or duplicated.")
    if not np.array_equal(test[ID_COLUMN].to_numpy(), sample[ID_COLUMN].to_numpy()):
        raise ValueError("Test and sample IDs/order differ.")

    target = pd.to_numeric(train[TARGET_COLUMN], errors="coerce")
    if target.isna().any() or not set(target.unique()).issubset({0, 1}):
        raise ValueError("Training TARGET must contain only non-missing binary 0/1 labels.")
    labels = target.to_numpy(dtype=np.int8)
    if np.unique(labels).size != 2:
        raise ValueError("Training TARGET must contain both classes.")
    if expected_target_hash is not None:
        actual = hashlib.sha256(labels.tobytes()).hexdigest()
        if actual != expected_target_hash:
            raise ValueError(
                f"Target-order fingerprint mismatch: expected {expected_target_hash}, got {actual}."
            )
    metadata = {
        "train_rows": int(len(train)),
        "test_rows": int(len(test)),
        "sample_rows": int(len(sample)),
        "train_columns": list(train.columns),
        "test_columns": list(test.columns),
        "sample_columns": list(sample.columns),
        "feature_columns": expected_test_columns,
        "class_counts": {
            str(int(value)): int(count)
            for value, count in zip(*np.unique(labels, return_counts=True))
        },
        "ids_unique": True,
        "test_ids_equal_sample_order": True,
        "train_test_predictor_schema_equal": True,
    }
    return labels, metadata


def load_inputs(
    data_dir: Path, expected_hashes: dict[str, str] | None = None
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, np.ndarray, dict[str, Any]]:
    paths = {name: data_dir / name for name in INPUT_NAMES}
    for name, path in paths.items():
        if not path.is_file():
            raise FileNotFoundError(f"Required input is missing: {path}")
        if expected_hashes and name in expected_hashes:
            actual = sha256_file(path)
            if actual != expected_hashes[name]:
                raise ValueError(
                    f"{name} SHA-256 mismatch: expected {expected_hashes[name]}, got {actual}."
                )
    train, test, sample = (
        pd.read_csv(paths["train.csv"]),
        pd.read_csv(paths["test.csv"]),
        pd.read_csv(paths["sample_submission.csv"]),
    )
    labels, metadata = validate_frames(train, test, sample)
    metadata["input_hashes"] = {
        name: {"sha256": sha256_file(path), "size_bytes": path.stat().st_size}
        for name, path in paths.items()
    }
    return train, test, sample, labels, metadata
