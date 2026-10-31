from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold, StratifiedKFold


def fold_record(
    name: str,
    fold_ids: np.ndarray,
    y: np.ndarray | None = None,
) -> dict[str, Any]:
    encoded = np.asarray(fold_ids, dtype="<i2")
    digest = hashlib.sha256(encoded.tobytes()).hexdigest()
    counts = np.bincount(encoded.astype(np.int64))
    result: dict[str, Any] = {
        "name": name,
        "rows": int(encoded.size),
        "fold_count": int(counts.size),
        "fold_ids_sha256_little_endian_int16": digest,
        "fold_row_counts": counts.astype(int).tolist(),
    }
    if y is not None:
        positives = [
            int(np.count_nonzero(np.asarray(y)[encoded == fold]))
            for fold in range(counts.size)
        ]
        result["positive_label_counts_by_fold"] = positives
    return result


def assign_stratified_folds(y: np.ndarray, n_splits: int, seed: int) -> np.ndarray:
    assignment = np.full(y.size, -1, dtype=np.int16)
    splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    for fold, (_, validation_indices) in enumerate(
        splitter.split(np.zeros(y.size, dtype=np.uint8), y)
    ):
        assignment[validation_indices] = fold
    if np.any(assignment < 0):
        raise AssertionError("Some train rows were not assigned to a validation fold.")
    return assignment


def assign_kfold(n_rows: int, n_splits: int, seed: int) -> np.ndarray:
    assignment = np.full(n_rows, -1, dtype=np.int16)
    splitter = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
    for fold, (_, validation_indices) in enumerate(splitter.split(np.zeros(n_rows))):
        assignment[validation_indices] = fold
    if np.any(assignment < 0):
        raise AssertionError("Some train rows were not assigned to a KFold partition.")
    return assignment


def audit(raw_dir: Path, output_path: Path) -> dict[str, Any]:
    started = time.monotonic()
    train = pd.read_csv(raw_dir / "train.csv", usecols=["SK_ID_CURR", "TARGET"])
    test = pd.read_csv(raw_dir / "test.csv", usecols=["SK_ID_CURR"])
    ids = train["SK_ID_CURR"].to_numpy()
    test_ids = test["SK_ID_CURR"].to_numpy()
    y = train["TARGET"].to_numpy(dtype=np.int32)
    if len(np.unique(ids)) != len(ids) or len(np.unique(test_ids)) != len(test_ids):
        raise ValueError("Raw train/test IDs are not unique.")
    if np.intersect1d(ids, test_ids).size:
        raise ValueError("Raw train and test ID sets overlap.")

    stratified_5 = assign_stratified_folds(y, 5, 42)
    fold_schemes = [
        fold_record("XGB10 StratifiedKFold", assign_stratified_folds(y, 10, 42), y),
        fold_record("V55 StratifiedKFold", stratified_5, y),
        fold_record("V17 StratifiedKFold", stratified_5, y),
        fold_record("Anchor74 StratifiedKFold", stratified_5, y),
        fold_record(
            "Anchor74 transductive Ridge-residual KFold",
            assign_kfold(len(y) + len(test_ids), 5, 42),
        ),
    ]
    record = {
        "schema_version": 1,
        "classification": "CURRENT_SOURCE_FOLD_ASSIGNMENTS_RECONSTRUCTED",
        "command": [sys.executable, *sys.argv],
        "exit_code": 0,
        "historical_fold_assignments_verified": False,
        "row_alignment": "raw train.csv row order",
        "raw_train_rows": len(train),
        "raw_test_rows": len(test),
        "transductive_feature_rows": len(train) + len(test),
        "raw_train_id_sha256_little_endian_int64": hashlib.sha256(
            np.asarray(ids, dtype="<i8").tobytes()
        ).hexdigest(),
        "raw_test_id_sha256_little_endian_int64": hashlib.sha256(
            np.asarray(test_ids, dtype="<i8").tobytes()
        ).hexdigest(),
        "concatenated_train_test_id_sha256_little_endian_int64": hashlib.sha256(
            np.concatenate([ids, test_ids]).astype("<i8", copy=False).tobytes()
        ).hexdigest(),
        "raw_target_sha256_little_endian_int32": hashlib.sha256(
            np.asarray(y, dtype="<i4").tobytes()
        ).hexdigest(),
        "fold_schemes": fold_schemes,
        "elapsed_seconds": time.monotonic() - started,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return record


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Record deterministic current-source CV fold assignments from raw labels."
    )
    parser.add_argument("--raw-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = audit(args.raw_dir, args.output)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
