from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, rankdata, spearmanr


def align_submission(
    reference: pd.DataFrame,
    candidate: pd.DataFrame,
    canonical_ids: pd.Series,
    id_column: str = "SK_ID_CURR",
    prediction_column: str = "TARGET",
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    required = {id_column, prediction_column}
    for label, frame in (("reference", reference), ("candidate", candidate)):
        if not required.issubset(frame.columns):
            raise ValueError(f"{label} is missing required columns {sorted(required)}")
        if frame[id_column].isna().any() or frame[id_column].duplicated().any():
            raise ValueError(f"{label} contains missing or duplicate IDs")
    if canonical_ids.isna().any() or canonical_ids.duplicated().any():
        raise ValueError("Canonical IDs contain missing or duplicate IDs")

    canonical = pd.Index(canonical_ids)
    reference_ids = pd.Index(reference[id_column])
    candidate_ids = pd.Index(candidate[id_column])
    if len(reference) != len(canonical) or not reference_ids.difference(canonical).empty or not canonical.difference(reference_ids).empty:
        raise ValueError("Reference ID set does not equal canonical raw test IDs")
    if len(candidate) != len(canonical) or not candidate_ids.difference(canonical).empty or not canonical.difference(candidate_ids).empty:
        raise ValueError("Candidate ID set does not equal canonical raw test IDs")

    reference_aligned = reference.set_index(id_column).reindex(canonical)
    candidate_aligned = candidate.set_index(id_column).reindex(canonical)
    ref_values = pd.to_numeric(reference_aligned[prediction_column], errors="coerce").to_numpy(
        dtype=np.float64
    )
    cand_values = pd.to_numeric(candidate_aligned[prediction_column], errors="coerce").to_numpy(
        dtype=np.float64
    )
    for label, values in (("reference", ref_values), ("candidate", cand_values)):
        if not np.isfinite(values).all():
            raise ValueError(f"{label} predictions contain non-finite values")
        if np.any((values < 0.0) | (values > 1.0)):
            raise ValueError(f"{label} predictions fall outside [0, 1]")
    return ref_values, cand_values, {
        "rows": int(len(canonical)),
        "reference_row_order_matches_test": reference_ids.equals(canonical),
        "candidate_row_order_matches_test": candidate_ids.equals(canonical),
        "duplicate_ids": 0,
        "missing_or_extra_ids": 0,
    }


def prediction_metrics(reference: np.ndarray, candidate: np.ndarray) -> dict[str, Any]:
    reference = np.asarray(reference, dtype=np.float64)
    candidate = np.asarray(candidate, dtype=np.float64)
    if reference.shape != candidate.shape or reference.ndim != 1 or len(reference) < 2:
        raise ValueError("Predictions must be one-dimensional, equal-shaped, and nontrivial")
    if not np.isfinite(reference).all() or not np.isfinite(candidate).all():
        raise ValueError("Predictions must be finite")
    if np.ptp(reference) == 0.0 or np.ptp(candidate) == 0.0:
        raise ValueError("Correlation is undefined for zero-variance predictions")
    difference = candidate - reference
    abs_difference = np.abs(difference)
    rank_difference = np.abs(
        rankdata(candidate, method="average") - rankdata(reference, method="average")
    )
    return {
        "pearson": float(pearsonr(reference, candidate).statistic),
        "spearman": float(spearmanr(reference, candidate).statistic),
        "mae": float(abs_difference.mean()),
        "rmse": float(np.sqrt(np.mean(np.square(difference)))),
        "maximum_absolute_error": float(abs_difference.max()),
        "mean_signed_error": float(difference.mean()),
        "mean_rank_displacement": float(rank_difference.mean()),
        "median_rank_displacement": float(np.median(rank_difference)),
        "p95_rank_displacement": float(np.quantile(rank_difference, 0.95)),
        "maximum_rank_displacement": float(rank_difference.max()),
        "exact_rank_agreement_count": int(np.count_nonzero(rank_difference == 0.0)),
    }


def _rank_quartile(values: pd.Series, labels: list[str]) -> pd.Series:
    ranks = pd.Series(
        rankdata(values.to_numpy(dtype=np.float64), method="ordinal"),
        index=values.index,
    )
    return pd.qcut(ranks, q=len(labels), labels=labels).astype(str)


def make_prediction_slices(
    reference: np.ndarray, test_features: pd.DataFrame
) -> dict[str, pd.Series]:
    if len(reference) != len(test_features):
        raise ValueError("Reference predictions and test features have different row counts")
    source = test_features.reset_index(drop=True)
    if "TARGET" in source.columns:
        raise ValueError("Test covariates unexpectedly contain TARGET")
    slices: dict[str, pd.Series] = {}
    reference_rank = pd.Series(
        rankdata(np.asarray(reference, dtype=np.float64), method="ordinal")
    )
    slices["reference_prediction_decile"] = pd.qcut(
        reference_rank,
        q=10,
        labels=[f"D{i}" for i in range(1, 11)],
    ).astype(str)

    ext_columns = ["EXT_SOURCE_1", "EXT_SOURCE_2", "EXT_SOURCE_3"]
    missing_ext = source[ext_columns].isna().sum(axis=1)
    slices["ext_source_missingness"] = pd.Series(
        np.select(
            [missing_ext.eq(0), missing_ext.eq(1)],
            ["0_missing", "1_missing"],
            default="2plus_missing",
        )
    )
    all_missing = source.drop(columns=["SK_ID_CURR"], errors="ignore").isna().sum(axis=1)
    slices["total_feature_missingness"] = pd.Series(
        np.select(
            [all_missing.eq(0), all_missing.le(5), all_missing.le(10)],
            ["0", "1_to_5", "6_to_10"],
            default="over_10",
        )
    )
    income = pd.to_numeric(source["AMT_INCOME_TOTAL"], errors="coerce")
    slices["income_quartile"] = _rank_quartile(
        income.fillna(income.median()), ["Q1", "Q2", "Q3", "Q4"]
    )
    age = -pd.to_numeric(source["DAYS_BIRTH"], errors="coerce") / 365.25
    slices["age_band"] = pd.cut(
        age,
        bins=[-np.inf, 25, 35, 45, 55, np.inf],
        labels=["under_25", "25_to_34", "35_to_44", "45_to_54", "55plus"],
        right=False,
    ).astype("string").fillna("missing").astype(str)
    for column, output_name in (
        ("NAME_CONTRACT_TYPE", "contract_type"),
        ("CODE_GENDER", "gender"),
    ):
        slices[output_name] = source[column].astype("string").fillna("MISSING").astype(str)
    return slices


def grouped_error_metrics(
    candidate_name: str,
    reference: np.ndarray,
    candidate: np.ndarray,
    slices: dict[str, pd.Series],
) -> list[dict[str, Any]]:
    reference = np.asarray(reference, dtype=np.float64)
    candidate = np.asarray(candidate, dtype=np.float64)
    if len(reference) != len(candidate):
        raise ValueError("Reference and candidate lengths differ")
    rank_difference = np.abs(
        rankdata(candidate, method="average") - rankdata(reference, method="average")
    )
    difference = candidate - reference
    rows = []
    for slice_name, groups in slices.items():
        groups = pd.Series(groups).reset_index(drop=True)
        if len(groups) != len(reference) or groups.isna().any():
            raise ValueError(f"Invalid rows in slice {slice_name}")
        for group_value in sorted(groups.unique()):
            mask = groups.to_numpy() == group_value
            group_diff = difference[mask]
            rows.append(
                {
                    "candidate": candidate_name,
                    "slice": slice_name,
                    "group": str(group_value),
                    "rows": int(mask.sum()),
                    "mean_signed_error": float(group_diff.mean()),
                    "mae": float(np.abs(group_diff).mean()),
                    "rmse": float(np.sqrt(np.mean(np.square(group_diff)))),
                    "maximum_absolute_error": float(np.abs(group_diff).max()),
                    "mean_global_rank_displacement": float(rank_difference[mask].mean()),
                    "p95_global_rank_displacement": float(
                        np.quantile(rank_difference[mask], 0.95)
                    ),
                }
            )
    return rows
