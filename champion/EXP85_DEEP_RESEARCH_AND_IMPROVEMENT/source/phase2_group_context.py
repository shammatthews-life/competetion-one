from __future__ import annotations

import numpy as np
import pandas as pd


GROUP_COLUMNS = (
    "NAME_INCOME_TYPE",
    "NAME_EDUCATION_TYPE",
    "OCCUPATION_TYPE",
    "ORGANIZATION_TYPE",
)
NUMERIC_COLUMNS = (
    "AMT_INCOME_TOTAL",
    "AMT_CREDIT",
    "AMT_ANNUITY",
    "AMT_GOODS_PRICE",
    "EXT_SOURCE_1",
    "EXT_SOURCE_2",
    "EXT_SOURCE_3",
    "AGE_YEARS",
    "EMPLOYED_YEARS",
)
REQUIRED_COLUMNS = set(GROUP_COLUMNS) | {
    "AMT_INCOME_TOTAL",
    "AMT_CREDIT",
    "AMT_ANNUITY",
    "AMT_GOODS_PRICE",
    "EXT_SOURCE_1",
    "EXT_SOURCE_2",
    "EXT_SOURCE_3",
    "DAYS_BIRTH",
    "DAYS_EMPLOYED",
}


def _numeric_values(frame: pd.DataFrame, column: str) -> np.ndarray:
    if column == "AGE_YEARS":
        values = -pd.to_numeric(frame["DAYS_BIRTH"], errors="raise").to_numpy(
            dtype=np.float64
        ) / 365.25
    elif column == "EMPLOYED_YEARS":
        employed = pd.to_numeric(
            frame["DAYS_EMPLOYED"], errors="raise"
        ).to_numpy(dtype=np.float64)
        employed = np.where(employed == 365243.0, np.nan, employed)
        values = -employed / 365.25
    else:
        values = pd.to_numeric(frame[column], errors="raise").to_numpy(
            dtype=np.float64
        )
    if np.isinf(values).any():
        raise ValueError(f"{column} contains infinite values")
    return values


def build_fold_group_context(
    train_raw: pd.DataFrame,
    test_raw: pd.DataFrame,
    fit_indices: np.ndarray,
    valid_indices: np.ndarray,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Create fold-local peer statistics with leave-one-out training values."""
    for label, frame in (("train", train_raw), ("test", test_raw)):
        missing = REQUIRED_COLUMNS - set(frame.columns)
        if missing:
            raise ValueError(f"{label} is missing required columns: {sorted(missing)}")
        if "TARGET" in frame.columns or "SK_ID_CURR" in frame.columns:
            raise ValueError(f"{label} passed target or ID into group feature builder")

    fit_indices = np.asarray(fit_indices, dtype=np.int64)
    valid_indices = np.asarray(valid_indices, dtype=np.int64)
    expected = np.arange(len(train_raw), dtype=np.int64)
    if (
        len(np.unique(fit_indices)) != len(fit_indices)
        or len(np.unique(valid_indices)) != len(valid_indices)
        or np.intersect1d(fit_indices, valid_indices).size
        or not np.array_equal(np.sort(np.concatenate([fit_indices, valid_indices])), expected)
    ):
        raise ValueError("Fit and validation indices must partition training rows")

    fit = train_raw.iloc[fit_indices].reset_index(drop=True)
    valid = train_raw.iloc[valid_indices].reset_index(drop=True)
    fit_features = pd.DataFrame(index=np.arange(len(fit_indices)))
    valid_features = pd.DataFrame(index=np.arange(len(valid_indices)))
    test_features = pd.DataFrame(index=np.arange(len(test_raw)))

    for group_column in GROUP_COLUMNS:
        fit_keys = fit[group_column].astype("string").fillna("__MISSING__").astype(str)
        valid_keys = valid[group_column].astype("string").fillna("__MISSING__").astype(str)
        test_keys = test_raw[group_column].astype("string").fillna("__MISSING__").astype(str)
        group_sizes = fit_keys.value_counts(dropna=False)
        fit_group_size = fit_keys.map(group_sizes).to_numpy(dtype=np.float64) - 1.0
        valid_group_size = valid_keys.map(group_sizes).fillna(0).to_numpy(dtype=np.float64)
        test_group_size = test_keys.map(group_sizes).fillna(0).to_numpy(dtype=np.float64)
        count_name = f"LOG_GROUP_COUNT__{group_column}"
        fit_features[count_name] = np.log1p(np.maximum(fit_group_size, 0.0))
        valid_features[count_name] = np.log1p(valid_group_size)
        test_features[count_name] = np.log1p(test_group_size)

        for numeric_column in NUMERIC_COLUMNS:
            fit_values = _numeric_values(fit, numeric_column)
            valid_values = _numeric_values(valid, numeric_column)
            test_values = _numeric_values(test_raw, numeric_column)
            grouped = pd.DataFrame(
                {"key": fit_keys, "value": fit_values}
            ).groupby("key", sort=False)["value"]
            group_sum = grouped.sum(min_count=1)
            group_count = grouped.count()
            global_sum = float(np.nansum(fit_values))
            global_count = int(np.isfinite(fit_values).sum())

            fit_sum = fit_keys.map(group_sum).to_numpy(dtype=np.float64)
            fit_count = fit_keys.map(group_count).fillna(0).to_numpy(dtype=np.float64)
            fit_is_observed = np.isfinite(fit_values).astype(np.float64)
            loo_sum = fit_sum - np.where(np.isfinite(fit_values), fit_values, 0.0)
            loo_count = fit_count - fit_is_observed
            fit_mean = np.divide(
                loo_sum,
                loo_count,
                out=np.full(len(fit_values), np.nan, dtype=np.float64),
                where=loo_count > 0,
            )
            global_loo_count = global_count - fit_is_observed
            global_loo_sum = global_sum - np.where(
                np.isfinite(fit_values), fit_values, 0.0
            )
            global_loo_mean = np.divide(
                global_loo_sum,
                global_loo_count,
                out=np.full(len(fit_values), np.nan, dtype=np.float64),
                where=global_loo_count > 0,
            )
            fit_mean = np.where(
                np.isfinite(fit_mean), fit_mean, global_loo_mean
            )

            if global_count:
                global_mean = global_sum / global_count
            else:
                global_mean = np.nan
            mapped_valid = valid_keys.map(group_sum).to_numpy(dtype=np.float64)
            valid_counts = valid_keys.map(group_count).fillna(0).to_numpy(dtype=np.float64)
            valid_mean = np.divide(
                mapped_valid,
                valid_counts,
                out=np.full(len(valid_values), np.nan, dtype=np.float64),
                where=valid_counts > 0,
            )
            valid_mean = np.where(np.isfinite(valid_mean), valid_mean, global_mean)

            mapped_test = test_keys.map(group_sum).to_numpy(dtype=np.float64)
            test_counts = test_keys.map(group_count).fillna(0).to_numpy(dtype=np.float64)
            test_mean = np.divide(
                mapped_test,
                test_counts,
                out=np.full(len(test_values), np.nan, dtype=np.float64),
                where=test_counts > 0,
            )
            test_mean = np.where(np.isfinite(test_mean), test_mean, global_mean)

            mean_name = f"GROUP_MEAN__{group_column}__{numeric_column}"
            delta_name = f"GROUP_DELTA__{group_column}__{numeric_column}"
            fit_features[mean_name] = fit_mean.astype(np.float32)
            fit_features[delta_name] = (fit_values - fit_mean).astype(np.float32)
            valid_features[mean_name] = valid_mean.astype(np.float32)
            valid_features[delta_name] = (valid_values - valid_mean).astype(np.float32)
            test_features[mean_name] = test_mean.astype(np.float32)
            test_features[delta_name] = (test_values - test_mean).astype(np.float32)

    if list(fit_features.columns) != list(valid_features.columns) or list(
        fit_features.columns
    ) != list(test_features.columns):
        raise ValueError("Group-context feature schemas differ")
    if np.isinf(
        fit_features.select_dtypes(include=[np.number]).to_numpy(dtype=np.float64)
    ).any():
        raise ValueError("Fit group-context features contain infinite values")
    if np.isinf(
        valid_features.select_dtypes(include=[np.number]).to_numpy(dtype=np.float64)
    ).any():
        raise ValueError("Validation group-context features contain infinite values")
    if np.isinf(
        test_features.select_dtypes(include=[np.number]).to_numpy(dtype=np.float64)
    ).any():
        raise ValueError("Test group-context features contain infinite values")

    train_features = pd.DataFrame(
        index=np.arange(len(train_raw)),
        columns=fit_features.columns,
        dtype=np.float32,
    )
    train_features.iloc[fit_indices] = fit_features.to_numpy(dtype=np.float32)
    train_features.iloc[valid_indices] = valid_features.to_numpy(dtype=np.float32)
    return train_features.reset_index(drop=True), test_features.reset_index(drop=True)
