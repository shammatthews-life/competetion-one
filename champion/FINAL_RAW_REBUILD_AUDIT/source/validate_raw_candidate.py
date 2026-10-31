from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import norm, pearsonr, rankdata, spearmanr


N_TEST = 61_500
BLEND_WEIGHTS = (
    ("v17", "test_e1Aet", 0.2932),
    ("xgb", "test_10f_xgb.npy", 0.2596),
    ("v17", "final_test", 0.2051),
    ("v55", "test_cb_product", 0.0874),
    ("v17", "test_e1et", 0.0489),
    ("v55", "test_deep_et", 0.0442),
    ("v55", "test_v2_allfam_oh", 0.0383),
    ("v55", "test_v2_mono", 0.0233),
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def align_submission(path: Path, expected_ids: np.ndarray) -> np.ndarray:
    expected_ids = np.asarray(expected_ids)
    if (
        expected_ids.ndim != 1
        or len(expected_ids) != N_TEST
        or len(np.unique(expected_ids)) != N_TEST
    ):
        raise ValueError("Expected IDs must be a unique 61,500-element vector.")
    frame = pd.read_csv(path)
    if list(frame.columns) != ["SK_ID_CURR", "TARGET"]:
        raise ValueError(f"{path} columns are {list(frame.columns)}, not the required schema.")
    ids = frame["SK_ID_CURR"].to_numpy()
    if len(ids) != N_TEST or len(np.unique(ids)) != N_TEST:
        raise ValueError(f"{path} must have 61,500 unique IDs.")
    positions = pd.Index(ids).get_indexer(expected_ids)
    if np.any(positions < 0) or len(np.unique(positions)) != N_TEST:
        raise ValueError(f"{path} ID set differs from raw test IDs.")
    values = frame["TARGET"].to_numpy(dtype=np.float64)[positions]
    if not np.isfinite(values).all() or np.any((values < 0) | (values > 1)):
        raise ValueError(f"{path} has non-finite targets or values outside [0, 1].")
    return values


def rank_percentile(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64).ravel()
    if not values.size or not np.isfinite(values).all():
        raise ValueError("Rank input must be non-empty and finite.")
    return (rankdata(values, method="average") - 0.5) / values.size


def independently_recompute(
    components_dir: Path, raw_ids: np.ndarray
) -> np.ndarray:
    anchor = pd.read_csv(
        components_dir / "v_20261027_precision_transductive_apex.csv"
    )
    if not np.array_equal(anchor["SK_ID_CURR"].to_numpy(), raw_ids):
        raise ValueError("Fresh Anchor74 output is not in raw test ID order.")
    anchor_values = anchor["TARGET"].to_numpy(dtype=np.float64)

    with np.load(components_dir / "v_17_full_suite_bundle.npz", allow_pickle=False) as z17:
        v17 = {name: z17[name] for name in z17.files}
    with np.load(components_dir / "v55_members.npz", allow_pickle=False) as z55:
        v55 = {name: z55[name] for name in z55.files}
    xgb = np.load(components_dir / "test_10f_xgb.npy", allow_pickle=False)
    member_arrays = {
        ("v17", "test_e1Aet"): v17["test_e1Aet"],
        ("v17", "final_test"): v17["final_test"],
        ("v17", "test_e1et"): v17["test_e1et"],
        ("v55", "test_cb_product"): v55["test_cb_product"],
        ("v55", "test_deep_et"): v55["test_deep_et"],
        ("v55", "test_v2_allfam_oh"): v55["test_v2_allfam_oh"],
        ("v55", "test_v2_mono"): v55["test_v2_mono"],
        ("xgb", "test_10f_xgb.npy"): xgb,
    }
    normalized = np.asarray([weight for _, _, weight in BLEND_WEIGHTS], dtype=np.float64)
    normalized /= normalized.sum()
    suite = np.zeros(N_TEST, dtype=np.float64)
    for (archive, key, _), weight in zip(BLEND_WEIGHTS, normalized):
        values = np.asarray(member_arrays[(archive, key)], dtype=np.float64).ravel()
        if values.shape != (N_TEST,) or not np.isfinite(values).all():
            raise ValueError(f"Fresh member {archive}:{key} has invalid shape or values.")
        uniforms = rank_percentile(values)
        suite += weight * norm.ppf(np.clip(uniforms, 1e-6, 1.0 - 1e-6))

    anchor_uniform = rank_percentile(anchor_values)
    z_final = (
        0.85 * norm.ppf(np.clip(anchor_uniform, 1e-6, 1.0 - 1e-6))
        + 0.15 * suite
    )
    return (rankdata(z_final, method="ordinal") - 0.5) / N_TEST


def comparison_metrics(left: np.ndarray, right: np.ndarray) -> dict[str, Any]:
    difference = left - right
    left_ranks = rankdata(left, method="average")
    right_ranks = rankdata(right, method="average")
    rank_displacement = np.abs(left_ranks - right_ranks)
    return {
        "pearson": float(pearsonr(left, right).statistic),
        "spearman": float(spearmanr(left, right).statistic),
        "mae": float(np.mean(np.abs(difference))),
        "rmse": float(np.sqrt(np.mean(np.square(difference)))),
        "maximum_absolute_difference": float(np.max(np.abs(difference))),
        "mean_rank_displacement": float(np.mean(rank_displacement)),
        "maximum_rank_displacement": float(np.max(rank_displacement)),
    }


def write_validation(
    candidate_path: Path,
    components_dir: Path,
    raw_dir: Path,
    frozen_champion_path: Path,
    previous_candidate_path: Path | None,
    output_path: Path,
) -> dict[str, Any]:
    raw_test = pd.read_csv(raw_dir / "test.csv", usecols=["SK_ID_CURR"])
    expected_ids = raw_test["SK_ID_CURR"].to_numpy()
    if len(expected_ids) != N_TEST or len(np.unique(expected_ids)) != N_TEST:
        raise ValueError("Raw test IDs are not 61,500 unique rows.")

    candidate = align_submission(candidate_path, expected_ids)
    if len(np.unique(candidate)) != N_TEST:
        raise ValueError("Candidate TARGET values must be unique ordinal percentiles.")
    formula_target = independently_recompute(components_dir, expected_ids)
    formula_error = candidate - formula_target
    if np.max(np.abs(formula_error)) > 5.1e-11:
        raise AssertionError(
            "Candidate values do not match an independent EXP85 formula recomputation "
            "to the CSV's 10-decimal precision."
        )

    frozen = align_submission(frozen_champion_path, expected_ids)
    comparison: dict[str, Any] = {
        "frozen_champion": comparison_metrics(candidate, frozen)
    }
    if previous_candidate_path is not None:
        if not previous_candidate_path.is_file():
            raise FileNotFoundError(previous_candidate_path)
        previous = align_submission(previous_candidate_path, expected_ids)
        comparison["previous_raw_candidate"] = comparison_metrics(candidate, previous)

    record = {
        "schema_version": 1,
        "classification": "RAW_COMPONENT_CANDIDATE; HISTORICAL_PARITY_NOT_ESTABLISHED",
        "candidate": {
            "path": str(candidate_path),
            "sha256": sha256(candidate_path),
            "size_bytes": candidate_path.stat().st_size,
            "rows": N_TEST,
            "columns": ["SK_ID_CURR", "TARGET"],
            "unique_ids": True,
            "id_set_equals_raw_test": True,
            "unique_targets": True,
            "row_order_equals_raw_test": bool(
                np.array_equal(
                    pd.read_csv(candidate_path)["SK_ID_CURR"].to_numpy(),
                    expected_ids,
                )
            ),
            "finite_targets": bool(np.isfinite(candidate).all()),
            "target_min": float(candidate.min()),
            "target_max": float(candidate.max()),
        },
        "raw_test": {
            "path": str(raw_dir / "test.csv"),
            "sha256": sha256(raw_dir / "test.csv"),
        },
        "frozen_champion": {
            "path": str(frozen_champion_path),
            "sha256": sha256(frozen_champion_path),
        },
        "formula_recomputation": {
            "maximum_absolute_difference_after_csv_rounding": float(
                np.max(np.abs(formula_error))
            ),
            "within_10_decimal_csv_precision": True,
            "weight_sum_before_normalization": float(
                sum(weight for _, _, weight in BLEND_WEIGHTS)
            ),
            "anchor_weight": 0.85,
            "suite_weight": 0.15,
        },
        "comparisons": comparison,
        "official_leaderboard_score": None,
        "score_note": (
            "No competition submission was made; public/private leaderboard "
            "score cannot be inferred from local rank correlation."
        ),
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return record


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate a fresh EXP85 raw-component candidate.")
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--components-dir", type=Path, required=True)
    parser.add_argument("--raw-dir", type=Path, required=True)
    parser.add_argument("--frozen-champion", type=Path, required=True)
    parser.add_argument("--previous-candidate", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = write_validation(
        args.candidate,
        args.components_dir,
        args.raw_dir,
        args.frozen_champion,
        args.previous_candidate,
        args.output,
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
