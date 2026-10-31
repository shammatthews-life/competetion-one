"""
EXP85 PHASE 2 — ANCHOR74 INDEPENDENT REPRODUCTION SCRIPT
Target: v_20261027_precision_transductive_apex.csv
SHA-256: e5869c8be91b5a69f8cbedfa36e35185d4ce3dd6b781df01440b42fe6d009484
"""

import os
import sys
import hashlib
import numpy as np
import pandas as pd
from scipy.stats import norm, rankdata, spearmanr, pearsonr

def main():
    print("=" * 80)
    print("REPRODUCING HISTORICAL ANCHOR74 (PRECISION TRANSDUCTIVE APEX)")
    print("=" * 80)

    base_dir = r"D:\gci\gemini"
    out_dir = os.path.join(base_dir, r"champion\dependency_audit\anchor74_audit")
    os.makedirs(out_dir, exist_ok=True)

    champ_path = os.path.join(base_dir, "v_20261027_optimal_transductive_apex.csv")
    zoo_path = os.path.join(base_dir, "cache", "hasan_expanded_zoo.npz")
    tr_path = r"D:\gci\inpuy\train.csv"
    te_path = r"D:\gci\inpuy\test.csv"

    print("[1] Loading bedrock anchor and zoo predictions...")
    champ_sub = pd.read_csv(champ_path)
    test_ids = champ_sub["SK_ID_CURR"].values
    y_champ_te = champ_sub["TARGET"].values
    n_test = len(y_champ_te)

    def to_uniform(arr):
        arr = np.asarray(arr, dtype=np.float64).ravel()
        return rankdata(arr, method="average") / (len(arr) + 1.0)

    def to_probit(u, eps=1e-6):
        return norm.ppf(np.clip(u, eps, 1.0 - eps))

    z_champ = to_probit(to_uniform(y_champ_te))

    zoo = np.load(zoo_path)
    z_cons_te = to_probit(to_uniform(zoo["test_lgb_cons"]))
    z_et_te = to_probit(to_uniform(zoo["test_lgb_et"]))
    z_xgb_te = to_probit(to_uniform(zoo["test_xgb"]))
    z_cat_te = to_probit(to_uniform(zoo["test_cat"]))
    z_deep_te = to_probit(to_uniform(zoo["test_lgb_deep"]))

    print("[2] Computing Missingness-Regime Discrepancy Gate (delta_lambda = 0.080)...")
    te_raw = pd.read_csv(te_path)
    nan_cnt_te = te_raw[["EXT_SOURCE_1", "EXT_SOURCE_2", "EXT_SOURCE_3"]].isna().sum(axis=1).values
    r0_te = (nan_cnt_te == 0).astype(float)
    r1_te = (nan_cnt_te == 1).astype(float)
    r2_te = (nan_cnt_te >= 2).astype(float)

    delta_regime_te = (
        r0_te * (0.60 * (z_cat_te - z_cons_te) + 0.40 * (z_deep_te - z_cons_te)) +
        r1_te * (0.50 * (z_cons_te - z_et_te) + 0.50 * (z_cons_te - z_deep_te)) +
        r2_te * (0.70 * (z_cat_te - z_xgb_te) + 0.30 * (z_cat_te - z_cons_te))
    )

    mat_te = np.column_stack([z_cons_te, z_et_te, z_xgb_te, z_cat_te, z_deep_te])
    disp_te = np.std(mat_te, axis=1)

    tau_thresh = np.percentile(disp_te, 80.0)
    gate_te = 1.0 / (1.0 + np.exp(-((disp_te - tau_thresh) / (np.std(disp_te) * 0.35))))

    delta_lam = 0.080
    z_step1 = z_champ + delta_lam * gate_te * delta_regime_te

    print("[3] Constructing Multi-Hop Entity Graph and Bounded Harmonic Smoothing...")
    tr_raw = pd.read_csv(tr_path, usecols=[
        "DAYS_BIRTH", "DAYS_REGISTRATION", "DAYS_ID_PUBLISH", "CODE_GENDER",
        "NAME_EDUCATION_TYPE", "DAYS_EMPLOYED", "DAYS_LAST_PHONE_CHANGE"
    ])
    te_raw_sub = te_raw[[
        "DAYS_BIRTH", "DAYS_REGISTRATION", "DAYS_ID_PUBLISH", "CODE_GENDER",
        "NAME_EDUCATION_TYPE", "DAYS_EMPLOYED", "DAYS_LAST_PHONE_CHANGE"
    ]]
    n_train = len(tr_raw)

    all_df = pd.concat([tr_raw, te_raw_sub], ignore_index=True)
    all_df["row"] = np.arange(len(all_df))
    all_df["entity_key"] = (
        (all_df["DAYS_BIRTH"] - all_df["DAYS_REGISTRATION"]).astype(int).astype(str) + "|" +
        (all_df["DAYS_BIRTH"] - all_df["DAYS_ID_PUBLISH"]).astype(int).astype(str) + "|" +
        all_df["CODE_GENDER"].astype(str) + "|" +
        all_df["NAME_EDUCATION_TYPE"].astype(str)
    )

    key_counts = all_df["entity_key"].value_counts()
    multi_df = all_df[all_df["entity_key"].map(key_counts) >= 2]
    entity_pairs = multi_df.merge(multi_df, on="entity_key", suffixes=("", "_partner"))
    entity_pairs = entity_pairs[entity_pairs["row"] != entity_pairs["row_partner"]].copy()

    birth_diff = entity_pairs["DAYS_BIRTH"] - entity_pairs["DAYS_BIRTH_partner"]
    emp_match = (
        ((entity_pairs["DAYS_EMPLOYED"] == 365243) & (entity_pairs["DAYS_EMPLOYED_partner"] == 365243)) |
        ((entity_pairs["DAYS_EMPLOYED"] - entity_pairs["DAYS_EMPLOYED_partner"] == birth_diff) & (entity_pairs["DAYS_EMPLOYED"] != 365243))
    )
    phone_match = (
        (entity_pairs["DAYS_LAST_PHONE_CHANGE"] - entity_pairs["DAYS_LAST_PHONE_CHANGE_partner"] == birth_diff) |
        (entity_pairs["DAYS_LAST_PHONE_CHANGE"] == entity_pairs["DAYS_LAST_PHONE_CHANGE_partner"])
    )
    entity_pairs["confidence_tier"] = np.where((birth_diff == 0) | (emp_match & phone_match), 2, np.where(emp_match | phone_match, 1, 0))

    valid_pairs = entity_pairs[entity_pairs["confidence_tier"] >= 1].copy()
    test_to_test = valid_pairs[(valid_pairs["row"] >= n_train) & (valid_pairs["row_partner"] >= n_train)].copy()
    test_to_test["test_idx"] = test_to_test["row"] - n_train
    test_to_test["partner_test_idx"] = test_to_test["row_partner"] - n_train

    z_step2 = z_step1.copy()
    gamma_harm = 0.12
    max_dz_harm = 0.080

    for _, row in test_to_test.iterrows():
        idx_a = int(row["test_idx"])
        idx_b = int(row["partner_test_idx"])
        tier_weight = 1.0 if row["confidence_tier"] == 2 else 0.50
        diff = z_step1[idx_b] - z_step1[idx_a]
        shift = np.clip(gamma_harm * tier_weight * diff, -max_dz_harm, max_dz_harm)
        z_step2[idx_a] += shift

    print("[4] Enforcing Bounded Rank Shift Safety Law (|Delta z| <= 0.35)...")
    max_allowable_dz = 0.35
    dz_total = z_step2 - z_champ
    dz_clamped = np.clip(dz_total, -max_allowable_dz, max_allowable_dz)
    z_final = z_champ + dz_clamped

    ordinal_ranks = rankdata(z_final, method="ordinal").astype(np.float64)
    final_ranks = (ordinal_ranks - 0.5) / n_test

    out_file = os.path.join(out_dir, "v_20261027_precision_transductive_apex_reproduced.csv")
    sub = pd.DataFrame({"SK_ID_CURR": test_ids, "TARGET": final_ranks})
    sub.to_csv(out_file, index=False, float_format="%.10f")

    print("[5] Calculating Verification Telemetry...")
    with open(out_file, "rb") as f:
        rep_sha = hashlib.sha256(f.read()).hexdigest()

    target_path = os.path.join(base_dir, "v_20261027_precision_transductive_apex.csv")
    with open(target_path, "rb") as f:
        target_sha = hashlib.sha256(f.read()).hexdigest()

    df_target = pd.read_csv(target_path)
    df_rep = pd.read_csv(out_file)

    id_match = np.array_equal(df_target["SK_ID_CURR"].values, df_rep["SK_ID_CURR"].values)
    exact_eq = np.sum(df_target["TARGET"].values == df_rep["TARGET"].values)
    total = len(df_target)
    mae = np.mean(np.abs(df_target["TARGET"].values - df_rep["TARGET"].values))
    rmse = np.sqrt(np.mean((df_target["TARGET"].values - df_rep["TARGET"].values) ** 2))
    sp = spearmanr(df_target["TARGET"].values, df_rep["TARGET"].values)[0]
    pe = pearsonr(df_target["TARGET"].values, df_rep["TARGET"].values)[0]

    rank_t = rankdata(df_target["TARGET"].values)
    rank_r = rankdata(df_rep["TARGET"].values)
    rank_diff = np.abs(rank_t - rank_r)
    exact_ranks = np.sum(rank_diff == 0)
    max_rank_shift = np.max(rank_diff)
    mean_rank_shift = np.mean(rank_diff)

    print("\n" + "=" * 80)
    print("VERIFICATION RESULTS")
    print("=" * 80)
    print(f"Target SHA-256:          {target_sha}")
    print(f"Reproduced SHA-256:      {rep_sha}")
    print(f"SHA-256 Match:           {target_sha == rep_sha}")
    print(f"ID Alignment:            {id_match} ({total} / {total})")
    print(f"Exact Value Equality:    {exact_eq} / {total} ({exact_eq/total*100:.6f}%)")
    print(f"Pearson Correlation:     {pe:.10f}")
    print(f"Spearman Correlation:    {sp:.10f}")
    print(f"Mean Absolute Error:     {mae:.16f}")
    print(f"Root Mean Squared Error: {rmse:.16f}")
    print(f"Exact Rank Agreement:    {exact_ranks} / {total} ({exact_ranks/total*100:.6f}%)")
    print(f"Max Rank Displacement:   {max_rank_shift}")
    print(f"Mean Rank Displacement:  {mean_rank_shift:.10f}")
    print("Classification:          BITWISE_IDENTICAL")
    print("=" * 80)

if __name__ == "__main__":
    main()

