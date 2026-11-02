#!/usr/bin/env python3
"""Rebuild reconstructable Hasan features with fold-local rank transforms."""

from __future__ import annotations

import re
from itertools import combinations

import numpy as np
import pandas as pd
from scipy.stats import norm, rankdata


ID_COLUMN = "SK_ID_CURR"
TARGET_COLUMN = "TARGET"
CORE_COLUMNS = (
    "EXT_MEAN",
    "DTI",
    "EAR",
    "LTV",
    "DSCR",
    "SURPLUS",
    "CREDIT_TERM",
    "DAYS_BIRTH",
    "DAYS_EMPLOYED_CLEAN",
    "DAYS_ID_PUBLISH",
    "DAYS_REGISTRATION",
    "DAYS_LAST_PHONE_CHANGE",
    "AMT_CREDIT",
    "AMT_INCOME_TOTAL",
    "AMT_ANNUITY",
    "AMT_GOODS_PRICE",
    "MERTON_PD_25",
    "CLAYTON_COPULA_2_5",
    "BUFFER_MONTHS",
    "DEBT_BURDEN",
)
EVALUATED_GROUPS = ("A_FINANCIAL", "B_RISK", "C_TEMPORAL", "D_DEMOGRAPHIC",
                    "E_INQUIRY", "F_EXTERNAL", "H_INTERACTIONS")
EXCLUDED_GROUPS = {
    "G_ENTITY_TIMELINE": (
        "Uses the same birth/registration/ID-publish/gender/education "
        "signature tested in EXP07; excluded to avoid repeating the rejected "
        "hypothesis. Historical counts and next/previous application gaps "
        "also depend on a pooled train/test cohort."
    ),
    "UIDY_SUPERVISED_MEMORY": (
        "Uses labels joined on the same signature with a global seed-42 "
        "five-fold assignment, which is not nested inside EXP08 outer folds. "
        "Excluded because that construction leaks outer-validation labels."
    ),
}


class FeatureAuditError(ValueError):
    """Raised when a source feature cannot be reconstructed safely."""


def _raw_values(frame: pd.DataFrame, column: str) -> np.ndarray:
    if column not in frame.columns:
        raise FeatureAuditError(f"Historical feature requires raw column {column}")
    return frame[column].to_numpy(dtype=np.float64, na_value=np.nan)


def _rank_fit(values: np.ndarray) -> np.ndarray:
    finite = np.isfinite(values)
    result = np.full(len(values), 0.5, dtype=np.float64)
    if finite.any():
        result[finite] = rankdata(values[finite], method="average") / (
            int(finite.sum()) + 1.0
        )
    return result


def _rank_apply(reference: np.ndarray, values: np.ndarray) -> np.ndarray:
    reference = np.asarray(reference, dtype=np.float64)
    reference = np.sort(reference[np.isfinite(reference)])
    result = np.full(len(values), 0.5, dtype=np.float64)
    finite = np.isfinite(values)
    if len(reference) and finite.any():
        query = values[finite]
        left = np.searchsorted(reference, query, side="left")
        right = np.searchsorted(reference, query, side="right")
        result[finite] = (left + right + 1.0) / (2.0 * (len(reference) + 1.0))
    return result


def _validate_raw(frame: pd.DataFrame, label: str) -> None:
    if TARGET_COLUMN in frame.columns:
        raise FeatureAuditError(f"{label} contains TARGET; feature builder refuses labels")
    if ID_COLUMN in frame.columns:
        raise FeatureAuditError(f"{label} contains applicant ID; remove it before building")
    if not frame.columns.is_unique:
        raise FeatureAuditError(f"{label} has duplicate raw column names")
    numeric = frame.select_dtypes(include=[np.number]).to_numpy(dtype=np.float64)
    if np.isinf(numeric).any():
        raise FeatureAuditError(f"{label} raw numeric columns contain infinity")


def _feature_values(
    frame: pd.DataFrame,
    fit_ext: tuple[np.ndarray, np.ndarray, np.ndarray],
    fit_rows: bool,
    include_interactions: bool,
) -> dict[str, np.ndarray]:
    get = lambda name: _raw_values(frame, name)
    income = get("AMT_INCOME_TOTAL")
    credit = get("AMT_CREDIT")
    annuity = get("AMT_ANNUITY")
    goods_raw = get("AMT_GOODS_PRICE")
    goods = np.where(np.isnan(goods_raw), credit, goods_raw)
    birth = get("DAYS_BIRTH")
    emp_raw = get("DAYS_EMPLOYED")
    retired = (emp_raw == 365243.0).astype(np.float32)
    emp = emp_raw.copy()
    emp[emp == 365243.0] = np.nan
    registration = get("DAYS_REGISTRATION")
    id_published = get("DAYS_ID_PUBLISH")
    phone = get("DAYS_LAST_PHONE_CHANGE")
    family = frame["CNT_FAM_MEMBERS"].fillna(1.0).to_numpy(dtype=np.float64)
    children = get("CNT_CHILDREN")
    ext = tuple(get(f"EXT_SOURCE_{index}") for index in (1, 2, 3))

    with np.errstate(divide="ignore", invalid="ignore"):
        housing_stability = registration / (id_published + 1.0)

    term = credit / (annuity + 1.0)
    ear = 24.0 * (annuity * term - credit) / (credit * (term + 1.0) + 1.0)
    surplus = income - 12.0 * annuity
    true_cost = annuity * term - credit
    ltv = credit / (goods + 1.0)
    debt_burden = annuity / (income / 12.0 + 1.0)
    dti = credit / (income + 1.0)
    dscr = ((income / 12.0) * 0.65) / (annuity + 1.0)
    buffer_months = surplus / (12.0 * annuity + 1.0)

    result: dict[str, np.ndarray] = {
        "IS_RETIRED": retired,
        "DAYS_EMPLOYED_CLEAN": emp.astype(np.float32),
        "CREDIT_TERM": term.astype(np.float32),
        "EAR": np.clip(ear, -1.0, 5.0).astype(np.float32),
        "SURPLUS": surplus.astype(np.float32),
        "SURPLUS_TO_INC": (surplus / (income + 1.0)).astype(np.float32),
        "TRUE_COST": true_cost.astype(np.float32),
        "TRUE_COST_RATIO": (true_cost / (credit + 1.0)).astype(np.float32),
        "LTV": np.clip(ltv, 0.0, 5.0).astype(np.float32),
        "GOODS_CREDIT_DIFF": (goods - credit).astype(np.float32),
        "DEBT_BURDEN": np.clip(debt_burden, 0.0, 5.0).astype(np.float32),
        "DISPOSABLE_INC": ((income / 12.0) - annuity).astype(np.float32),
        "DTI": np.clip(dti, 0.0, 20.0).astype(np.float32),
        "DSCR": np.clip(dscr, 0.0, 20.0).astype(np.float32),
        "FOIR": np.clip(annuity / (income / 12.0 + 1.0), 0.0, 5.0).astype(np.float32),
        "TDSR": np.clip(
            (annuity + credit * 0.05 / 12.0) / (income / 12.0 + 1.0),
            0.0,
            5.0,
        ).astype(np.float32),
        "SOLVENCY": np.clip(
            (income / 12.0) / (annuity + 1.0), 0.0, 20.0
        ).astype(np.float32),
        "LIQUIDITY": (surplus / (credit + 1.0)).astype(np.float32),
        "BUFFER_MONTHS": np.clip(buffer_months, -10.0, 50.0).astype(np.float32),
        "RESERVE_MONTHS": np.clip(
            (income - 12.0 * annuity) / (annuity + 1.0), -10.0, 50.0
        ).astype(np.float32),
        "DTI_40": (dti > 0.40).astype(np.float32),
        "DTI_46": (dti > 0.46).astype(np.float32),
        "DTI_50": (dti > 0.50).astype(np.float32),
        "LTV_75": (ltv > 0.75).astype(np.float32),
        "LTV_100": (ltv > 1.00).astype(np.float32),
        "DSCR_BELOW1": (dscr < 1.0).astype(np.float32),
        "DSCR_BELOW1_25": (dscr < 1.25).astype(np.float32),
        "PAYMENT_SHOCK_10": (
            annuity * 1.10 / (income / 12.0 + 1.0) > 0.50
        ).astype(np.float32),
        "PAYMENT_SHOCK_20": (
            annuity * 1.20 / (income / 12.0 + 1.0) > 0.50
        ).astype(np.float32),
        "SHOCK_BREACH20": (
            annuity * 1.20 / (income / 12.0 + 1.0) > 0.50
        ).astype(np.float32),
    }

    z_score = (
        1.2 * (surplus / (credit + 1.0))
        + 1.4 * (surplus / (income + 1.0))
        + 3.3 * (income / (credit + 1.0))
    )
    result.update(
        {
            "Z_SCORE": np.clip(z_score, -20.0, 50.0).astype(np.float32),
            "HFVI": (
                (dti > 0.50).astype(float)
                + (ltv > 0.90).astype(float)
                + (dscr < 1.0).astype(float)
                + (buffer_months < 1.0).astype(float)
            ).astype(np.float32),
            "FFI": (
                ((surplus / (income + 1.0)) < 0.2).astype(float)
                + (debt_burden > 0.4).astype(float)
            ).astype(np.float32),
            "ZOMBIE": ((surplus < 0.0) & (ltv > 1.0)).astype(np.float32),
        }
    )

    years = np.clip(term / 12.0, 0.1, 10.0)
    disposable_ratio = np.maximum(0.01, (income / 12.0) / (annuity + 1.0))
    for mu, sigma, suffix in (
        (0.02, 0.20, "20"),
        (0.03, 0.25, "25"),
        (0.04, 0.35, "35"),
    ):
        distance = (
            np.log(disposable_ratio) + (mu - 0.5 * sigma**2) * years
        ) / (sigma * np.sqrt(years))
        result[f"MERTON_DD_{suffix}"] = np.clip(
            distance, -10.0, 10.0
        ).astype(np.float32)
        result[f"MERTON_PD_{suffix}"] = norm.cdf(-distance).astype(np.float32)

    units = (
        tuple(_rank_fit(value) for value in ext)
        if fit_rows
        else tuple(_rank_apply(reference, value) for reference, value in zip(fit_ext, ext))
    )
    for theta in (1.5, 2.5, 4.0):
        powered = [
            np.power(np.maximum(1e-4, unit), -theta) for unit in units
        ]
        inside = np.maximum(0.0, powered[0] + powered[1] + powered[2] - 2.0)
        copula = np.power(inside, -1.0 / theta)
        result[f"CLAYTON_COPULA_{str(theta).replace('.', '_')}"] = np.nan_to_num(
            copula, nan=0.5
        ).astype(np.float32)

    req_hour = frame["AMT_REQ_CREDIT_BUREAU_HOUR"].fillna(0.0).to_numpy(
        dtype=np.float64
    )
    req_month = frame["AMT_REQ_CREDIT_BUREAU_MON"].fillna(0.0).to_numpy(
        dtype=np.float64
    )
    req_quarter = frame["AMT_REQ_CREDIT_BUREAU_QRT"].fillna(0.0).to_numpy(
        dtype=np.float64
    )
    req_year = frame["AMT_REQ_CREDIT_BUREAU_YEAR"].fillna(0.0).to_numpy(
        dtype=np.float64
    )
    total_requests = req_hour + req_month + req_quarter + req_year
    result.update(
        {
            "TERM_KEY_1D": term.round(1).astype(np.float32),
            "MARKUP_KEY_2D": (credit / (goods + 1.0)).round(2).astype(np.float32),
            "ID_VS_REG": (id_published - registration).astype(np.float32),
            "EMP_VS_REG": (emp - registration).astype(np.float32),
            "PHONE_VS_ID": (phone - id_published).astype(np.float32),
            "PHONE_VS_REG": (phone - registration).astype(np.float32),
            "EMP_STABILITY": (emp / (birth + 1.0)).astype(np.float32),
            "REG_STABILITY": (registration / (birth + 1.0)).astype(np.float32),
            "PHONE_STABILITY": (phone / (birth + 1.0)).astype(np.float32),
            "HOUSING_STABILITY": housing_stability.astype(np.float32),
            "PHONE_CHANGED_RECENTLY": (phone > -365.0).astype(np.float32),
            "ID_CHANGED_RECENTLY": (id_published > -365.0).astype(np.float32),
            "REG_CHANGED_RECENTLY": (registration > -365.0).astype(np.float32),
            "SPEND_PRESSURE": (
                annuity / (income / family + 1.0)
            ).astype(np.float32),
            "HOUSING_BURDEN": (annuity / family).astype(np.float32),
            "FULLY_FINANCED": (credit >= goods).astype(np.float32),
            "OVERSPENDING": (credit > income * 3.0).astype(np.float32),
            "INCOME_PER_MEMBER": (income / family).astype(np.float32),
            "CREDIT_PER_MEMBER": (credit / family).astype(np.float32),
            "DEPENDENCY": (children / family).astype(np.float32),
            "IS_MARRIED": (
                frame["NAME_FAMILY_STATUS"] == "Married"
            ).astype(np.float32),
            "TOTAL_INQUIRIES": (
                total_requests
            ).astype(np.float32),
            "INQUIRY_BURST": (
                (req_hour + req_month) / (req_year / 12.0 + 1.0)
            ).astype(np.float32),
            "INQUIRY_QRT_RATIO": (
                req_quarter / (req_year / 4.0 + 1.0)
            ).astype(np.float32),
            "INQUIRY_PER_CREDIT": (
                total_requests / (credit / 100000.0 + 1.0)
            ).astype(np.float32),
            "INQUIRY_ZERO_FLAG": (total_requests == 0.0).astype(np.float32),
            "PHONE_WORK_EMP_MATCH": (
                _raw_values(frame, "FLAG_EMP_PHONE")
                == _raw_values(frame, "FLAG_WORK_PHONE")
            ).astype(np.float32),
            "PHONE_CONTACT_SCORE": (
                _raw_values(frame, "FLAG_EMP_PHONE")
                + _raw_values(frame, "FLAG_WORK_PHONE")
                + _raw_values(frame, "FLAG_MOBIL")
            ).astype(np.float32),
            "REGION_RATING_CREDIT": (
                credit
                / (
                    frame["REGION_RATING_CLIENT"].fillna(2.0).to_numpy()
                    + 1.0
                )
            ).astype(np.float32),
            "REGION_RATING_ANNUITY": (
                annuity
                / (
                    frame["REGION_RATING_CLIENT"].fillna(2.0).to_numpy()
                    + 1.0
                )
            ).astype(np.float32),
            "REGION_POP_INCOME": (
                _raw_values(frame, "REGION_POPULATION_RELATIVE") * income
            ).astype(np.float32),
        }
    )

    ext_stack = np.column_stack(ext)
    ext_present = ~np.isnan(ext_stack)
    ext_count = ext_present.sum(axis=1)
    ext_sum = np.where(ext_present, ext_stack, 0.0).sum(axis=1)
    ext_mean = np.full(len(frame), np.nan, dtype=np.float64)
    np.divide(ext_sum, ext_count, out=ext_mean, where=ext_count > 0)
    ext_squared_deviation = np.where(
        ext_present,
        (ext_stack - ext_mean[:, None]) ** 2,
        0.0,
    ).sum(axis=1)
    ext_variance = np.zeros(len(frame), dtype=np.float64)
    np.divide(
        ext_squared_deviation,
        ext_count,
        out=ext_variance,
        where=ext_count > 0,
    )
    ext_std = np.sqrt(ext_variance)
    ext_min = np.min(np.where(ext_present, ext_stack, np.inf), axis=1)
    ext_max = np.max(np.where(ext_present, ext_stack, -np.inf), axis=1)
    ext_min[ext_count == 0] = np.nan
    ext_max[ext_count == 0] = np.nan
    ext_product = np.prod(np.where(ext_present, ext_stack, 1.0), axis=1)
    ext_weighted = (
        np.nan_to_num(0.2 * ext[0], nan=0.0)
        + np.nan_to_num(0.4 * ext[1], nan=0.0)
        + np.nan_to_num(0.4 * ext[2], nan=0.0)
    )
    result.update(
        {
            "EXT_MEAN": ext_mean.astype(np.float32),
            "EXT_STD": np.nan_to_num(ext_std, nan=0.0).astype(np.float32),
            "EXT_MIN": ext_min.astype(np.float32),
            "EXT_MAX": ext_max.astype(np.float32),
            "EXT_PROD": ext_product.astype(np.float32),
            "EXT_WEIGHTED": ext_weighted.astype(np.float32),
            "EXT_SQ": (ext_mean**2).astype(np.float32),
            "EXT_CUBE": (ext_mean**3).astype(np.float32),
            "EXT_DIFF_12": (ext[0] - ext[1]).astype(np.float32),
            "EXT_DIFF_23": (ext[1] - ext[2]).astype(np.float32),
            "EXT_DIFF_13": (ext[0] - ext[2]).astype(np.float32),
            "EXT_x_DTI": (ext_mean * dti).astype(np.float32),
            "EXT_x_SURPLUS": (
                ext_mean * result["SURPLUS_TO_INC"]
            ).astype(np.float32),
            "EXT_x_EAR": (ext_mean * ear).astype(np.float32),
            "EXT_x_LTV": (ext_mean * ltv).astype(np.float32),
            "EXT_x_DEBT_BURDEN": (ext_mean * debt_burden).astype(np.float32),
        }
    )

    if include_interactions:
        if not all(
            name in result or name in frame.columns for name in CORE_COLUMNS
        ):
            raise FeatureAuditError("Historical interaction core is incomplete")
        for left, right in combinations(CORE_COLUMNS, 2):
            left_values = (
                result[left] if left in result else _raw_values(frame, left)
            )
            right_values = (
                result[right] if right in result else _raw_values(frame, right)
            )
            result[f"P_PROD_{left}_x_{right}"] = (
                left_values * right_values
            ).astype(np.float32)
            result[f"P_RATIO_{left}_div_{right}"] = (
                left_values / (np.abs(right_values) + 1.0)
            ).astype(np.float32)
            result[f"P_DIFF_{left}_sub_{right}"] = (
                left_values - right_values
            ).astype(np.float32)
    return result


def build_feature_frames(
    fit_raw: pd.DataFrame,
    apply_raw: pd.DataFrame,
    groups: tuple[str, ...] = EVALUATED_GROUPS,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build source-derived features using fit-only EXT rank distributions."""
    unknown = set(groups) - set(EVALUATED_GROUPS)
    if unknown:
        raise FeatureAuditError(f"Unknown or excluded feature groups: {sorted(unknown)}")
    _validate_raw(fit_raw, "fit raw frame")
    _validate_raw(apply_raw, "apply raw frame")
    required = set(fit_raw.columns)
    missing = required - set(apply_raw.columns)
    if missing:
        raise FeatureAuditError(f"Apply frame is missing columns: {sorted(missing)}")
    if list(fit_raw.columns) != list(apply_raw.columns):
        raise FeatureAuditError("Fit/apply raw schemas differ in order")
    fit_ext = tuple(
        _raw_values(fit_raw, f"EXT_SOURCE_{index}") for index in (1, 2, 3)
    )
    with_interactions = "H_INTERACTIONS" in groups
    all_fit_values = _feature_values(
        fit_raw, fit_ext, fit_rows=True, include_interactions=with_interactions
    )
    all_apply_values = _feature_values(
        apply_raw, fit_ext, fit_rows=False, include_interactions=with_interactions
    )
    grouped_names = feature_groups(tuple(all_fit_values))
    selected_columns = [
        name
        for name in all_fit_values
        if any(name in grouped_names[group] for group in groups)
    ]
    fit_values = {name: all_fit_values[name] for name in selected_columns}
    apply_values = {name: all_apply_values[name] for name in selected_columns}
    if list(fit_values) != list(apply_values):
        raise FeatureAuditError("Fit/apply historical feature schemas differ")
    fit_frame = pd.DataFrame(fit_values, index=fit_raw.index).reset_index(drop=True)
    apply_frame = pd.DataFrame(apply_values, index=apply_raw.index).reset_index(drop=True)
    return fit_frame, apply_frame


def feature_groups(feature_columns: list[str] | tuple[str, ...]) -> dict[str, list[str]]:
    groups: dict[str, list[str]] = {group: [] for group in EVALUATED_GROUPS}
    for name in feature_columns:
        if name.startswith("P_"):
            group = "H_INTERACTIONS"
        elif name.startswith(("MERTON_", "CLAYTON_")):
            group = "B_RISK"
        elif name in ("TERM_KEY_1D", "MARKUP_KEY_2D"):
            group = "B_RISK"
        elif name.startswith(("ID_VS_", "EMP_VS_", "PHONE_VS_", "EMP_STABILITY",
                              "REG_STABILITY", "PHONE_STABILITY", "HOUSING_STABILITY",
                              "PHONE_CHANGED_", "ID_CHANGED_", "REG_CHANGED_")):
            group = "C_TEMPORAL"
        elif name.startswith(("SPEND_", "HOUSING_BURDEN", "FULLY_FINANCED",
                              "OVERSPENDING", "INCOME_PER_MEMBER",
                              "CREDIT_PER_MEMBER", "DEPENDENCY", "IS_MARRIED")):
            group = "D_DEMOGRAPHIC"
        elif name.startswith(("TOTAL_INQUIRIES", "INQUIRY_", "PHONE_WORK_",
                              "PHONE_CONTACT_", "REGION_RATING_", "REGION_POP_")):
            group = "E_INQUIRY"
        elif name.startswith(("EXT_",)):
            group = "F_EXTERNAL"
        else:
            group = "A_FINANCIAL"
        groups[group].append(name)
    return groups


DIRECT_FORMULAS = {
    "IS_RETIRED": "1[DAYS_EMPLOYED == 365243]",
    "DAYS_EMPLOYED_CLEAN": "DAYS_EMPLOYED; replace 365243 with NaN",
    "CREDIT_TERM": "AMT_CREDIT / (AMT_ANNUITY + 1)",
    "EAR": "clip(24*(AMT_ANNUITY*CREDIT_TERM-AMT_CREDIT)/(AMT_CREDIT*(CREDIT_TERM+1)+1), -1, 5)",
    "SURPLUS": "AMT_INCOME_TOTAL - 12*AMT_ANNUITY",
    "SURPLUS_TO_INC": "SURPLUS / (AMT_INCOME_TOTAL + 1)",
    "TRUE_COST": "AMT_ANNUITY*CREDIT_TERM - AMT_CREDIT",
    "TRUE_COST_RATIO": "TRUE_COST / (AMT_CREDIT + 1)",
    "LTV": "clip(AMT_CREDIT / (goods_filled + 1), 0, 5)",
    "GOODS_CREDIT_DIFF": "goods_filled - AMT_CREDIT",
    "DEBT_BURDEN": "clip(AMT_ANNUITY / (AMT_INCOME_TOTAL/12 + 1), 0, 5)",
    "DISPOSABLE_INC": "AMT_INCOME_TOTAL/12 - AMT_ANNUITY",
    "DTI": "clip(AMT_CREDIT / (AMT_INCOME_TOTAL + 1), 0, 20)",
    "DSCR": "clip((AMT_INCOME_TOTAL/12*0.65)/(AMT_ANNUITY+1), 0, 20)",
    "FOIR": "clip(AMT_ANNUITY/(AMT_INCOME_TOTAL/12+1), 0, 5)",
    "TDSR": "clip((AMT_ANNUITY+AMT_CREDIT*0.05/12)/(AMT_INCOME_TOTAL/12+1), 0, 5)",
    "SOLVENCY": "clip((AMT_INCOME_TOTAL/12)/(AMT_ANNUITY+1), 0, 20)",
    "LIQUIDITY": "SURPLUS / (AMT_CREDIT + 1)",
    "BUFFER_MONTHS": "clip(SURPLUS/(12*AMT_ANNUITY+1), -10, 50)",
    "RESERVE_MONTHS": "clip((AMT_INCOME_TOTAL-12*AMT_ANNUITY)/(AMT_ANNUITY+1), -10, 50)",
    "DTI_40": "1[unclipped DTI > 0.40]",
    "DTI_46": "1[unclipped DTI > 0.46]",
    "DTI_50": "1[unclipped DTI > 0.50]",
    "LTV_75": "1[unclipped LTV > 0.75]",
    "LTV_100": "1[unclipped LTV > 1.00]",
    "DSCR_BELOW1": "1[unclipped DSCR < 1]",
    "DSCR_BELOW1_25": "1[unclipped DSCR < 1.25]",
    "PAYMENT_SHOCK_10": "1[AMT_ANNUITY*1.10/(AMT_INCOME_TOTAL/12+1) > 0.50]",
    "PAYMENT_SHOCK_20": "1[AMT_ANNUITY*1.20/(AMT_INCOME_TOTAL/12+1) > 0.50]",
    "SHOCK_BREACH20": "alias of PAYMENT_SHOCK_20",
    "Z_SCORE": "clip(1.2*SURPLUS/(AMT_CREDIT+1)+1.4*SURPLUS/(AMT_INCOME_TOTAL+1)+3.3*AMT_INCOME_TOTAL/(AMT_CREDIT+1), -20, 50)",
    "HFVI": "1[DTI>.50]+1[LTV>.90]+1[DSCR<1]+1[BUFFER_MONTHS<1]",
    "FFI": "1[SURPLUS_TO_INC<.2]+1[unclipped DEBT_BURDEN>.4]",
    "ZOMBIE": "1[SURPLUS<0 and unclipped LTV>1]",
    "TERM_KEY_1D": "round(AMT_CREDIT/(AMT_ANNUITY+1), 1)",
    "MARKUP_KEY_2D": "round(AMT_CREDIT/(goods_filled+1), 2)",
    "ID_VS_REG": "DAYS_ID_PUBLISH - DAYS_REGISTRATION",
    "EMP_VS_REG": "DAYS_EMPLOYED_CLEAN - DAYS_REGISTRATION",
    "PHONE_VS_ID": "DAYS_LAST_PHONE_CHANGE - DAYS_ID_PUBLISH",
    "PHONE_VS_REG": "DAYS_LAST_PHONE_CHANGE - DAYS_REGISTRATION",
    "EMP_STABILITY": "DAYS_EMPLOYED_CLEAN/(DAYS_BIRTH+1)",
    "REG_STABILITY": "DAYS_REGISTRATION/(DAYS_BIRTH+1)",
    "PHONE_STABILITY": "DAYS_LAST_PHONE_CHANGE/(DAYS_BIRTH+1)",
    "HOUSING_STABILITY": "DAYS_REGISTRATION/(DAYS_ID_PUBLISH+1)",
    "PHONE_CHANGED_RECENTLY": "1[DAYS_LAST_PHONE_CHANGE > -365]",
    "ID_CHANGED_RECENTLY": "1[DAYS_ID_PUBLISH > -365]",
    "REG_CHANGED_RECENTLY": "1[DAYS_REGISTRATION > -365]",
    "SPEND_PRESSURE": "AMT_ANNUITY/(AMT_INCOME_TOTAL/CNT_FAM_MEMBERS_FILLED+1)",
    "HOUSING_BURDEN": "AMT_ANNUITY/CNT_FAM_MEMBERS_FILLED",
    "FULLY_FINANCED": "1[AMT_CREDIT >= goods_filled]",
    "OVERSPENDING": "1[AMT_CREDIT > 3*AMT_INCOME_TOTAL]",
    "INCOME_PER_MEMBER": "AMT_INCOME_TOTAL/CNT_FAM_MEMBERS_FILLED",
    "CREDIT_PER_MEMBER": "AMT_CREDIT/CNT_FAM_MEMBERS_FILLED",
    "DEPENDENCY": "CNT_CHILDREN/CNT_FAM_MEMBERS_FILLED",
    "IS_MARRIED": "1[NAME_FAMILY_STATUS == 'Married']",
    "TOTAL_INQUIRIES": "sum(fillna(0), AMT_REQ_CREDIT_BUREAU_HOUR/MON/QRT/YEAR)",
    "INQUIRY_BURST": "(HOUR+MON)/(YEAR/12+1), each missing count set to zero",
    "INQUIRY_QRT_RATIO": "QRT/(YEAR/4+1), each missing count set to zero",
    "INQUIRY_PER_CREDIT": "TOTAL_INQUIRIES/(AMT_CREDIT/100000+1)",
    "INQUIRY_ZERO_FLAG": "1[TOTAL_INQUIRIES == 0]",
    "PHONE_WORK_EMP_MATCH": "1[FLAG_EMP_PHONE == FLAG_WORK_PHONE]",
    "PHONE_CONTACT_SCORE": "FLAG_EMP_PHONE + FLAG_WORK_PHONE + FLAG_MOBIL",
    "REGION_RATING_CREDIT": "AMT_CREDIT/(fillna(REGION_RATING_CLIENT,2)+1)",
    "REGION_RATING_ANNUITY": "AMT_ANNUITY/(fillna(REGION_RATING_CLIENT,2)+1)",
    "REGION_POP_INCOME": "REGION_POPULATION_RELATIVE*AMT_INCOME_TOTAL",
    "EXT_MEAN": "nanmean(EXT_SOURCE_1, EXT_SOURCE_2, EXT_SOURCE_3)",
    "EXT_STD": "nanstd(EXT_SOURCE_1, EXT_SOURCE_2, EXT_SOURCE_3), ddof=0; all-missing becomes 0",
    "EXT_MIN": "nanmin(EXT_SOURCE_1, EXT_SOURCE_2, EXT_SOURCE_3)",
    "EXT_MAX": "nanmax(EXT_SOURCE_1, EXT_SOURCE_2, EXT_SOURCE_3)",
    "EXT_PROD": "nanprod(EXT_SOURCE_1, EXT_SOURCE_2, EXT_SOURCE_3); all-missing is 1",
    "EXT_WEIGHTED": "0.2*nan_to_num(EXT_SOURCE_1)+0.4*nan_to_num(EXT_SOURCE_2)+0.4*nan_to_num(EXT_SOURCE_3)",
    "EXT_SQ": "EXT_MEAN ** 2",
    "EXT_CUBE": "EXT_MEAN ** 3",
    "EXT_DIFF_12": "EXT_SOURCE_1 - EXT_SOURCE_2",
    "EXT_DIFF_23": "EXT_SOURCE_2 - EXT_SOURCE_3",
    "EXT_DIFF_13": "EXT_SOURCE_1 - EXT_SOURCE_3",
    "EXT_x_DTI": "EXT_MEAN * unclipped DTI",
    "EXT_x_SURPLUS": "EXT_MEAN * SURPLUS_TO_INC",
    "EXT_x_EAR": "EXT_MEAN * unclipped EAR",
    "EXT_x_LTV": "EXT_MEAN * unclipped LTV",
    "EXT_x_DEBT_BURDEN": "EXT_MEAN * unclipped DEBT_BURDEN",
}


def formula_for(name: str) -> str:
    if name.startswith("P_PROD_"):
        return "core_left * core_right (float32)"
    if name.startswith("P_RATIO_"):
        return "core_left / (abs(core_right) + 1) (float32)"
    if name.startswith("P_DIFF_"):
        return "core_left - core_right (float32)"
    if name.startswith("CLAYTON_COPULA_"):
        theta = float(name.removeprefix("CLAYTON_COPULA_").replace("_", "."))
        return (
            f"max(0, sum(max(1e-4, U_i)^(-{theta})) - 2)^(-1/{theta}); "
            "trivariate EXT_SOURCE_1/2/3 using fit-only average ranks"
        )
    if name.startswith("MERTON_DD_") or name.startswith("MERTON_PD_"):
        return (
            "T=clip((AMT_CREDIT/(AMT_ANNUITY+1))/12,0.1,10); "
            "q=max(0.01,(AMT_INCOME_TOTAL/12)/(AMT_ANNUITY+1)); "
            "DD=(ln(q)+(mu-0.5*sigma^2)*T)/(sigma*sqrt(T)); "
            + ("clip(DD,-10,10)" if name.startswith("MERTON_DD_") else "Phi(-DD)")
        )
    if name in DIRECT_FORMULAS:
        return DIRECT_FORMULAS[name]
    return f"Historical executable formula preserved; see source expression for {name}"


def source_columns_for(name: str, raw_columns: set[str]) -> tuple[str, ...]:
    if name.startswith("P_PROD_"):
        pair = name.removeprefix("P_PROD_").split("_x_", maxsplit=1)
        return tuple(pair)
    if name.startswith("P_RATIO_"):
        pair = name.removeprefix("P_RATIO_").split("_div_", maxsplit=1)
        return tuple(pair)
    if name.startswith("P_DIFF_"):
        pair = name.removeprefix("P_DIFF_").split("_sub_", maxsplit=1)
        return tuple(pair)
    expression = formula_for(name)
    tokens = set(
        re.findall(r"\b[A-Z][A-Z0-9_]*\b", expression)
    )
    return tuple(sorted(tokens & raw_columns))
