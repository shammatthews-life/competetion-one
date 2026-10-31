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
from scipy.stats import pearsonr, rankdata, spearmanr


ARRAY_PAIRS = (
    ("xgb10", "oof_10f_xgb.npy", "oof_10f_xgb.npy"),
    ("xgb10", "test_10f_xgb.npy", "test_10f_xgb.npy"),
    ("v17", "oof_old", "oof_old"),
    ("v17", "test_old", "test_old"),
    ("v17", "oof_e1", "oof_e1"),
    ("v17", "test_e1", "test_e1"),
    ("v17", "oof_e1et", "oof_e1et"),
    ("v17", "test_e1et", "test_e1et"),
    ("v17", "oof_e1Aet", "oof_e1Aet"),
    ("v17", "test_e1Aet", "test_e1Aet"),
    ("v17", "oof_cb", "oof_cb"),
    ("v17", "test_cb", "test_cb"),
    ("v17", "final_oof", "final_oof"),
    ("v17", "final_test", "final_test"),
    ("v55", "oof_qf_shallow", "oof_qf_shallow"),
    ("v55", "test_qf_shallow", "test_qf_shallow"),
    ("v55", "oof_qf_reg", "oof_qf_reg"),
    ("v55", "test_qf_reg", "test_qf_reg"),
    ("v55", "oof_v1_allfam_oh", "oof_v1_allfam_oh"),
    ("v55", "test_v1_allfam_oh", "test_v1_allfam_oh"),
    ("v55", "oof_v2_allfam_oh", "oof_v2_allfam_oh"),
    ("v55", "test_v2_allfam_oh", "test_v2_allfam_oh"),
    ("v55", "oof_v2_mono", "oof_v2_mono"),
    ("v55", "test_v2_mono", "test_v2_mono"),
    ("v55", "oof_deep_et", "oof_deep_et"),
    ("v55", "test_deep_et", "test_deep_et"),
    ("v55", "oof_cb_product", "oof_cb_product"),
    ("v55", "test_cb_product", "test_cb_product"),
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_array(directory: Path, group: str, member: str) -> np.ndarray:
    if group == "xgb10":
        return np.load(directory / member, allow_pickle=False)
    archive_path = directory / (
        "v_17_full_suite_bundle.npz" if group == "v17" else "v55_members.npz"
    )
    with np.load(archive_path, allow_pickle=False) as archive:
        return archive[member]


def compare_arrays(fresh: np.ndarray, historical: np.ndarray) -> dict[str, Any]:
    fresh = np.asarray(fresh, dtype=np.float64).ravel()
    historical = np.asarray(historical, dtype=np.float64).ravel()
    if fresh.shape != historical.shape:
        return {
            "comparable": False,
            "fresh_shape": list(fresh.shape),
            "historical_shape": list(historical.shape),
            "reason": "shape mismatch",
        }
    if not np.isfinite(fresh).all() or not np.isfinite(historical).all():
        return {"comparable": False, "reason": "non-finite values"}
    difference = fresh - historical
    fresh_ranks = rankdata(fresh, method="average")
    historical_ranks = rankdata(historical, method="average")
    rank_displacement = np.abs(fresh_ranks - historical_ranks)
    fresh_percentiles = (fresh_ranks - 0.5) / fresh.size
    historical_percentiles = (historical_ranks - 0.5) / historical.size
    percentile_difference = fresh_percentiles - historical_percentiles
    return {
        "comparable": True,
        "rows": int(fresh.size),
        "pearson": float(pearsonr(fresh, historical).statistic),
        "spearman": float(spearmanr(fresh, historical).statistic),
        "mae": float(np.mean(np.abs(difference))),
        "rmse": float(np.sqrt(np.mean(np.square(difference)))),
        "maximum_absolute_difference": float(np.max(np.abs(difference))),
        "different_cells": int(np.count_nonzero(difference)),
        "mean_rank_displacement": float(np.mean(rank_displacement)),
        "maximum_rank_displacement": float(np.max(rank_displacement)),
        "exact_array_match": bool(np.array_equal(fresh, historical)),
        "rank_percentile_comparison": {
            "mae": float(np.mean(np.abs(percentile_difference))),
            "rmse": float(np.sqrt(np.mean(np.square(percentile_difference)))),
            "maximum_absolute_difference": float(
                np.max(np.abs(percentile_difference))
            ),
            "different_cells": int(np.count_nonzero(percentile_difference)),
            "exact_rank_match": bool(np.array_equal(fresh_ranks, historical_ranks)),
        },
    }


def run_comparison(
    generated_dir: Path,
    historical_dir: Path,
    raw_dir: Path,
    output_path: Path,
    historical_anchor_path: Path | None = None,
) -> dict[str, Any]:
    started = time.monotonic()
    raw_train = pd.read_csv(raw_dir / "train.csv", usecols=["SK_ID_CURR", "TARGET"])
    raw_test = pd.read_csv(raw_dir / "test.csv", usecols=["SK_ID_CURR"])
    generated_y = np.load(generated_dir / "y_train.npy", allow_pickle=False)
    historical_y_path = historical_dir / "y_train.npy"
    historical_y = np.load(historical_y_path, allow_pickle=False)
    raw_y = raw_train["TARGET"].to_numpy(dtype=generated_y.dtype)
    if not np.array_equal(generated_y, raw_y):
        raise ValueError("Fresh generated labels do not match raw train TARGET.")
    if not np.array_equal(historical_y, raw_y):
        raise ValueError("Historical labels do not match raw train TARGET.")

    comparisons: dict[str, Any] = {}
    for group, fresh_member, historical_member in ARRAY_PAIRS:
        fresh = load_array(generated_dir, group, fresh_member)
        historical = load_array(historical_dir, group, historical_member)
        is_oof = fresh_member.startswith("oof_") or fresh_member.endswith("_oof")
        expected_rows = len(raw_train) if is_oof else len(raw_test)
        if fresh.size != expected_rows or historical.size != expected_rows:
            raise ValueError(
                f"{group}:{fresh_member} has row counts "
                f"{fresh.size}/{historical.size}, expected {expected_rows}."
            )
        comparisons[f"{group}:{fresh_member}"] = compare_arrays(fresh, historical)

    historical_anchor_comparison = None
    if historical_anchor_path is not None:
        historical_anchor = pd.read_csv(historical_anchor_path)
        fresh_anchor = pd.read_csv(
            generated_dir / "v_20261027_precision_transductive_apex.csv"
        )
        expected_ids = raw_test["SK_ID_CURR"].to_numpy()
        for label, frame in (
            ("fresh", fresh_anchor),
            ("historical", historical_anchor),
        ):
            if list(frame.columns) != ["SK_ID_CURR", "TARGET"]:
                raise ValueError(f"{label} Anchor74 has an unexpected schema.")
            if (
                len(frame) != len(expected_ids)
                or frame["SK_ID_CURR"].duplicated().any()
                or set(frame["SK_ID_CURR"]) != set(expected_ids)
            ):
                raise ValueError(f"{label} Anchor74 IDs differ from raw test IDs.")
        fresh_positions = pd.Index(fresh_anchor["SK_ID_CURR"]).get_indexer(expected_ids)
        historical_positions = pd.Index(historical_anchor["SK_ID_CURR"]).get_indexer(
            expected_ids
        )
        if np.any(fresh_positions < 0) or np.any(historical_positions < 0):
            raise ValueError("Anchor74 ID alignment failed.")
        fresh_values = fresh_anchor["TARGET"].to_numpy(dtype=np.float64)[fresh_positions]
        historical_values = historical_anchor["TARGET"].to_numpy(dtype=np.float64)[
            historical_positions
        ]
        if (
            not np.isfinite(fresh_values).all()
            or not np.isfinite(historical_values).all()
        ):
            raise ValueError("Anchor74 contains non-finite target values.")
        if (
            np.any((fresh_values < 0) | (fresh_values > 1))
            or np.any((historical_values < 0) | (historical_values > 1))
        ):
            raise ValueError("Anchor74 targets are outside [0, 1].")
        historical_anchor_comparison = {
            "path": str(historical_anchor_path),
            "sha256": sha256_file(historical_anchor_path),
            "id_aligned_metrics": compare_arrays(fresh_values, historical_values),
        }

    record = {
        "schema_version": 1,
        "classification": "READ_ONLY_POST_RUN_MODEL_OUTPUT_COMPARISON",
        "command": [sys.executable, *sys.argv],
        "exit_code": 0,
        "historical_arrays_used_for_candidate_generation": False,
        "alignment_limit": (
            "Prediction arrays have no ID sidecars. OOF comparisons are positional "
            "against raw train row order after exact label identity checks; test "
            "comparisons are positional and cannot prove per-row ID alignment."
        ),
        "raw_inputs": {
            "train_sha256": sha256_file(raw_dir / "train.csv"),
            "test_sha256": sha256_file(raw_dir / "test.csv"),
            "train_rows": len(raw_train),
            "test_rows": len(raw_test),
            "generated_and_historical_labels_exact_raw_match": True,
        },
        "generated_directory": str(generated_dir),
        "historical_directory": str(historical_dir),
        "historical_artifact_sha256": {
            "oof_10f_xgb.npy": sha256_file(historical_dir / "oof_10f_xgb.npy"),
            "test_10f_xgb.npy": sha256_file(historical_dir / "test_10f_xgb.npy"),
            "v55_members.npz": sha256_file(historical_dir / "v55_members.npz"),
            "v_17_full_suite_bundle.npz": sha256_file(
                historical_dir / "v_17_full_suite_bundle.npz"
            ),
            "y_train.npy": sha256_file(historical_y_path),
        },
        "comparisons": comparisons,
        "anchor74_vs_historical": historical_anchor_comparison,
        "elapsed_seconds": time.monotonic() - started,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return record


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Read-only compare fresh predictions with historical caches after synthesis."
    )
    parser.add_argument("--generated-dir", type=Path, required=True)
    parser.add_argument("--historical-dir", type=Path, required=True)
    parser.add_argument("--raw-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--historical-anchor", type=Path)
    args = parser.parse_args()
    result = run_comparison(
        args.generated_dir,
        args.historical_dir,
        args.raw_dir,
        args.output,
        args.historical_anchor,
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
