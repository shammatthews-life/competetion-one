from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from preprocessing import TARGET_COLUMN


FEE_AMOUNT = 129.5
PRODUCT_RATE = 0.0132


def build_decoded_loan_features(
    train_raw: pd.DataFrame,
    test_raw: pd.DataFrame,
    pooled_frequency_reference: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    if not pooled_frequency_reference:
        raise ValueError(
            "Selected seed-42 Phase 2 recipe uses pooled unlabeled train/test frequencies."
        )
    combined = pd.concat(
        [train_raw.drop(columns=[TARGET_COLUMN], errors="ignore"), test_raw],
        ignore_index=True,
    )
    credit = combined["AMT_CREDIT"].to_numpy(dtype=np.float64)
    annuity = combined["AMT_ANNUITY"].to_numpy(dtype=np.float64)
    goods = combined["AMT_GOODS_PRICE"].to_numpy(dtype=np.float64)
    contract = combined["NAME_CONTRACT_TYPE"].astype("string").to_numpy()

    raw_ratio = np.full(len(combined), np.nan, dtype=np.float64)
    fee_ratio = np.full(len(combined), np.nan, dtype=np.float64)
    valid_credit = np.isfinite(credit) & (credit != 0.0)
    raw_ratio[valid_credit] = annuity[valid_credit] / credit[valid_credit]
    fee_ratio[valid_credit] = (annuity[valid_credit] - FEE_AMOUNT) / credit[valid_credit]

    raw_key = pd.Series(raw_ratio).round(5)
    fee_key = pd.Series(fee_ratio).round(5)
    raw_counts = raw_key.map(raw_key.value_counts()).fillna(0).to_numpy()
    fee_counts = fee_key.map(fee_key.value_counts()).fillna(0).to_numpy()
    fee_decoded = (
        (contract == "Cash loans")
        & (fee_counts > raw_counts)
        & np.isfinite(raw_ratio)
    )
    decoded_ratio = np.where(fee_decoded, fee_ratio, raw_ratio)

    goods_valid = np.isfinite(goods) & (goods != 0.0)
    credit_goods_ratio = np.full(len(combined), np.nan, dtype=np.float64)
    credit_goods_ratio[goods_valid] = credit[goods_valid] / goods[goods_valid]
    implied_product = (credit_goods_ratio - 1.0) / PRODUCT_RATE
    nearest_product = np.rint(implied_product)
    product_valid = (
        (contract == "Cash loans")
        & np.isfinite(implied_product)
        & (credit > goods)
        & (np.abs(implied_product - nearest_product) < 0.1)
    )

    features = pd.DataFrame(
        {
            "FEE_DECODED": fee_decoded.astype(np.float32),
            "FEE_RAW_RATIO_COUNT": np.log1p(raw_counts).astype(np.float32),
            "FEE_ADJUSTED_RATIO_COUNT": np.log1p(fee_counts).astype(np.float32),
            "ANNUITY_CREDIT_RATIO_FEE_ADJUSTED": fee_ratio,
            "ANNUITY_CREDIT_RATIO_DECODED": decoded_ratio,
            "ANNUITY_CREDIT_RATIO_DECODED_ROUND_3": np.round(decoded_ratio, 3),
            "ANNUITY_CREDIT_RATIO_DECODED_ROUND_4": np.round(decoded_ratio, 4),
            "ANNUITY_CREDIT_RATIO_DECODED_ROUND_5": np.round(decoded_ratio, 5),
            "CREDIT_GOODS_RATIO": credit_goods_ratio,
            "PRODUCT_KEY_VALID": product_valid.astype(np.float32),
            "PRODUCT_KEY_K": np.where(product_valid, nearest_product, np.nan),
            "PRODUCT_KEY_DISTANCE": np.where(
                product_valid, implied_product - nearest_product, np.nan
            ),
        }
    )
    if np.isinf(features.to_numpy(dtype=np.float64, copy=False)).any():
        raise ValueError("Decoded loan features contain infinite values.")

    n_train = len(train_raw)
    train_features = features.iloc[:n_train].reset_index(drop=True)
    test_features = features.iloc[n_train:].reset_index(drop=True)
    diagnostics = {
        "fee_amount": FEE_AMOUNT,
        "product_rate": PRODUCT_RATE,
        "fee_decoded_rows": int(fee_decoded.sum()),
        "fee_decoded_cash_loan_fraction": float(fee_decoded[contract == "Cash loans"].mean()),
        "product_key_valid_rows": int(product_valid.sum()),
        "product_key_valid_fraction_of_cash_loans": float(
            product_valid[contract == "Cash loans"].mean()
        ),
        "feature_names": features.columns.tolist(),
        "frequency_counts_use_train_and_test_covariates": True,
        "frequency_counts_use_target": False,
    }
    return train_features, test_features, diagnostics
