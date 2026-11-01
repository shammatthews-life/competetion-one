from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.special import ndtr


CATEGORICAL_COLUMNS = (
    "NAME_CONTRACT_TYPE",
    "CODE_GENDER",
    "NAME_INCOME_TYPE",
    "NAME_EDUCATION_TYPE",
    "NAME_FAMILY_STATUS",
    "NAME_HOUSING_TYPE",
    "OCCUPATION_TYPE",
    "ORGANIZATION_TYPE",
)
ID_COLUMN = "SK_ID_CURR"
TARGET_COLUMN = "TARGET"


def safe_ratio(numerator: np.ndarray, denominator: np.ndarray) -> np.ndarray:
    numerator = np.asarray(numerator, dtype=np.float64)
    denominator = np.asarray(denominator, dtype=np.float64)
    result = np.full(numerator.shape, np.nan, dtype=np.float64)
    valid = np.isfinite(numerator) & np.isfinite(denominator) & (denominator != 0)
    result[valid] = numerator[valid] / denominator[valid]
    return result


def build_rowwise_features(raw: pd.DataFrame) -> pd.DataFrame:
    missing_columns = set(CATEGORICAL_COLUMNS) - set(raw.columns)
    if missing_columns:
        raise ValueError(f"Missing categorical columns: {sorted(missing_columns)}")
    if ID_COLUMN not in raw or TARGET_COLUMN in raw:
        if TARGET_COLUMN in raw:
            raise ValueError("Feature builder must not receive the target column.")
        raise ValueError(f"Missing required ID column: {ID_COLUMN}")

    numeric = raw.drop(columns=[ID_COLUMN]).select_dtypes(include=[np.number]).copy()
    if np.isinf(numeric.to_numpy(dtype=np.float64, copy=False)).any():
        raise ValueError("Raw numeric features contain infinite values.")
    categorical = raw.loc[:, CATEGORICAL_COLUMNS].reset_index(drop=True).copy()
    categorical = categorical.fillna("MISSING").astype(str)
    features = numeric.reset_index(drop=True)

    for column in numeric.columns:
        features[f"NA_{column}"] = numeric[column].isna().to_numpy(dtype=np.float32)
    features["RAW_MISSING_COUNT"] = (
        raw.drop(columns=[ID_COLUMN]).isna().sum(axis=1).to_numpy(dtype=np.float32)
    )

    get = lambda name: raw[name].to_numpy(dtype=np.float64)
    income = get("AMT_INCOME_TOTAL")
    credit = get("AMT_CREDIT")
    annuity = get("AMT_ANNUITY")
    goods = get("AMT_GOODS_PRICE")
    family = raw["CNT_FAM_MEMBERS"].fillna(1.0).to_numpy(dtype=np.float64)
    birth = get("DAYS_BIRTH")
    emp_raw = get("DAYS_EMPLOYED")
    emp_clean = np.where(emp_raw == 365243.0, np.nan, emp_raw)
    registration = get("DAYS_REGISTRATION")
    id_published = get("DAYS_ID_PUBLISH")
    phone = get("DAYS_LAST_PHONE_CHANGE")
    goods_filled = np.where(np.isnan(goods), credit, goods)

    derived: dict[str, np.ndarray] = {
        "DAYS_EMPLOYED_CLEAN": emp_clean,
        "DAYS_EMPLOYED_SENTINEL": (emp_raw == 365243.0).astype(np.float32),
        "AGE_YEARS": -birth / 365.25,
        "EMPLOYED_YEARS": -emp_clean / 365.25,
        "EMPLOYED_TO_AGE": safe_ratio(emp_clean, birth + 1.0),
        "REGISTRATION_TO_AGE": safe_ratio(registration, birth + 1.0),
        "ID_PUBLISH_TO_AGE": safe_ratio(id_published, birth + 1.0),
        "PHONE_CHANGE_YEARS": -phone / 365.25,
        "CREDIT_TO_INCOME": safe_ratio(credit, income),
        "ANNUITY_TO_INCOME": safe_ratio(annuity, income),
        "CREDIT_TO_GOODS": safe_ratio(credit, goods_filled),
        "ANNUITY_TO_CREDIT": safe_ratio(annuity, credit),
        "CREDIT_TERM": safe_ratio(credit, annuity),
        "INCOME_PER_FAMILY": safe_ratio(income, family),
        "CREDIT_PER_FAMILY": safe_ratio(credit, family),
        "ANNUITY_PER_FAMILY": safe_ratio(annuity, family),
        "DISPOSABLE_INCOME": income - 12.0 * annuity,
        "DISPOSABLE_INCOME_TO_INCOME": safe_ratio(income - 12.0 * annuity, income),
        "DEBT_SERVICE_COVERAGE": safe_ratio(income / 12.0, annuity),
        "CREDIT_TERM_ROUND_1": safe_ratio(credit, annuity + 1.0).round(1),
        "CREDIT_TO_GOODS_ROUND_2": safe_ratio(credit, goods_filled + 1.0).round(2),
    }

    ext = np.column_stack([get("EXT_SOURCE_1"), get("EXT_SOURCE_2"), get("EXT_SOURCE_3")])
    ext_count = np.isfinite(ext).sum(axis=1)
    ext_sum = np.where(np.isfinite(ext), ext, 0.0).sum(axis=1)
    ext_mean = np.divide(
        ext_sum,
        ext_count,
        out=np.full(len(ext), np.nan, dtype=np.float64),
        where=ext_count > 0,
    )
    ext_min = np.where(np.isfinite(ext), ext, np.inf).min(axis=1)
    ext_max = np.where(np.isfinite(ext), ext, -np.inf).max(axis=1)
    ext_min[ext_count == 0] = np.nan
    ext_max[ext_count == 0] = np.nan
    ext_variance = np.divide(
        np.where(np.isfinite(ext), (ext - ext_mean[:, None]) ** 2, 0.0).sum(axis=1),
        ext_count,
        out=np.full(len(ext), np.nan, dtype=np.float64),
        where=ext_count > 0,
    )
    ext_std = np.sqrt(ext_variance)
    ext_product = np.where(np.isfinite(ext), ext, 1.0).prod(axis=1)
    ext_product[ext_count == 0] = np.nan
    derived.update(
        {
            "EXT_MEAN": ext_mean,
            "EXT_MIN": ext_min,
            "EXT_MAX": ext_max,
            "EXT_STD": np.where(ext_count == 0, 0.0, ext_std),
            "EXT_RANGE": ext_max - ext_min,
            "EXT_COUNT": ext_count.astype(np.float32),
            "EXT_PRODUCT": ext_product,
        }
    )
    for left, right in ((0, 1), (0, 2), (1, 2)):
        derived[f"EXT_DIFF_{left}{right}"] = ext[:, left] - ext[:, right]
        derived[f"EXT_PRODUCT_{left}{right}"] = ext[:, left] * ext[:, right]
        derived[f"EXT_RATIO_{left}{right}"] = safe_ratio(ext[:, left], ext[:, right])

    term_years = np.clip(safe_ratio(credit, annuity + 1.0) / 12.0, 0.1, 10.0)
    disposable_ratio = np.maximum(0.01, safe_ratio(income / 12.0, annuity + 1.0))
    for mu, sigma, suffix in ((0.02, 0.20, "20"), (0.03, 0.25, "25")):
        distance = (
            np.log(disposable_ratio) + (mu - 0.5 * sigma**2) * term_years
        ) / (sigma * np.sqrt(term_years))
        derived[f"MERTON_DISTANCE_{suffix}"] = np.clip(distance, -10.0, 10.0)
        derived[f"MERTON_PD_{suffix}"] = ndtr(-distance)

    bureau_columns = [
        column for column in raw.columns if column.startswith("AMT_REQ_CREDIT_BUREAU_")
    ]
    bureau = raw[bureau_columns]
    derived["BUREAU_INQUIRY_SUM"] = bureau.sum(axis=1, min_count=1).to_numpy()
    derived["INCOME_LOG1P"] = np.log1p(np.maximum(income, 0.0))
    derived["CREDIT_LOG1P"] = np.log1p(np.maximum(credit, 0.0))
    derived["ANNUITY_LOG1P"] = np.log1p(np.maximum(annuity, 0.0))

    derived_frame = pd.DataFrame(derived)
    if np.isinf(derived_frame.to_numpy(dtype=np.float64, copy=False)).any():
        raise ValueError("Feature engineering produced infinite values.")
    features = pd.concat([features, derived_frame], axis=1)
    features = features.loc[:, ~features.columns.duplicated()]
    for column in features.columns:
        if not pd.api.types.is_numeric_dtype(features[column]):
            raise TypeError(f"Non-numeric derived feature: {column}")
    return pd.concat([features, categorical], axis=1).reset_index(drop=True)


def make_lightgbm_frames(
    train_features: pd.DataFrame, test_features: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    numeric_columns = [
        column for column in train_features.columns if column not in CATEGORICAL_COLUMNS
    ]
    train_parts = [train_features[numeric_columns].astype(np.float32)]
    test_parts = [test_features[numeric_columns].astype(np.float32)]
    for column in CATEGORICAL_COLUMNS:
        levels = sorted(train_features[column].astype(str).unique().tolist())
        train_cat = pd.Categorical(train_features[column].astype(str), categories=levels)
        test_cat = pd.Categorical(test_features[column].astype(str), categories=levels)
        category_columns = [f"CAT_{column}_{index}" for index in range(len(levels))]
        train_dummy = pd.get_dummies(train_cat, dtype=np.float32)
        test_dummy = pd.get_dummies(test_cat, dtype=np.float32)
        train_dummy.columns = category_columns
        test_dummy.columns = category_columns
        train_parts.append(train_dummy)
        test_parts.append(test_dummy)
    X_train = pd.concat(train_parts, axis=1)
    X_test = pd.concat(test_parts, axis=1).reindex(columns=X_train.columns, fill_value=0.0)
    if not X_train.columns.is_unique or list(X_train.columns) != list(X_test.columns):
        raise ValueError("LightGBM train/test feature schemas differ.")
    if any(not column.replace("_", "").isalnum() for column in X_train.columns):
        raise ValueError("LightGBM feature names must be alphanumeric with underscores.")
    if np.isinf(X_train.to_numpy(dtype=np.float64, copy=False)).any() or np.isinf(
        X_test.to_numpy(dtype=np.float64, copy=False)
    ).any():
        raise ValueError("LightGBM features contain infinite values.")
    return X_train, X_test
