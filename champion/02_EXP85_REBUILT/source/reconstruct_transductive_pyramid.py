"""
EXP85 Phase 3 - Workstream 4 & 5: Complete Upstream Pyramid & EXP85 Synthesis
Consumes the independently regenerated five-model zoo from independent_rebuild
and executes the complete calibration and synthesis chain:
1. Asymmetric Tail-Gated Apex (from Level-2 stacker + newly trained zoo)
2. Optimal Transductive Apex (from newly trained tail apex + newly trained zoo)
3. Precision Transductive Anchor74 (from newly trained optimal apex + newly trained zoo)
4. EXP85 85/15 Probit Synthesis (from newly trained Anchor74 + 8-engine suite)

Outputs all intermediate and final submissions to independent_rebuild/.
Then evaluates against historical champion benchmarks.
"""

import os
import sys
import time
import hashlib
import numpy as np
import pandas as pd
from scipy.stats import norm, rankdata, spearmanr, pearsonr
from sklearn.metrics import roc_auc_score
import warnings
warnings.filterwarnings('ignore')

def to_uniform(arr):
    arr = np.asarray(arr, dtype=np.float64).ravel()
    return rankdata(arr, method="average") / (len(arr) + 1.0)

def to_probit(u, eps=1e-6):
    return norm.ppf(np.clip(u, eps, 1.0 - eps))

def main():
    t0 = time.time()
    print("=" * 80)
    print("WORKSTREAM 4 & 5: RECONSTRUCTING TRANSDUCTIVE PYRAMID & EXP85 SYNTHESIS")
    print("=" * 80)

    rebuild_dir = r"D:\gci\gemini\champion\phase3_reconstruction\independent_rebuild"
    raw_train_path = r"D:\gci\inpuy\train.csv"
    raw_test_path = r"D:\gci\inpuy\test.csv"
    base_gemini = r"D:\gci\gemini"

    zoo_path = os.path.join(rebuild_dir, "hasan_expanded_zoo.npz")
    if not os.path.exists(zoo_path):
        raise FileNotFoundError(f"Missing rebuilt zoo: {zoo_path}")

    print("[1] Loading independently rebuilt foundation model zoo...")
    zoo = np.load(zoo_path)
    oof_lgb_cons = zoo['oof_lgb_cons']
    test_lgb_cons = zoo['test_lgb_cons']
    oof_lgb_et = zoo['oof_lgb_et']
    test_lgb_et = zoo['test_lgb_et']
    oof_xgb = zoo['oof_xgb']
    test_xgb = zoo['test_xgb']
    oof_cat = zoo['oof_cat']
    test_cat = zoo['test_cat']
    oof_lgb_deep = zoo['oof_lgb_deep']
    test_lgb_deep = zoo['test_lgb_deep']
    y_tr = zoo['y_true']

    n_train = len(y_tr)
    n_test = len(test_lgb_cons)

    z_models_tr = [to_probit(to_uniform(m)) for m in [oof_lgb_cons, oof_lgb_et, oof_xgb, oof_cat, oof_lgb_deep]]
    z_models_te = [to_probit(to_uniform(m)) for m in [test_lgb_cons, test_lgb_et, test_xgb, test_cat, test_lgb_deep]]

    # =========================================================================
    # Step 1: Reconstruct Asymmetric Tail-Gated Apex
    # =========================================================================
    print("\n[Step 1] Building Asymmetric Tail-Gated Apex from Rebuilt Zoo...")
    champ_l2_path = os.path.join(base_gemini, "v_solo_080300_level2_regime_stacker.csv")
    champ_sub = pd.read_csv(champ_l2_path)
    test_ids = champ_sub["SK_ID_CURR"].values
    y_champ_te = champ_sub["TARGET"].values

    z_champ_te = to_probit(to_uniform(y_champ_te))
    opt_zoo_weights = np.array([0.22, 0.18, 0.24, 0.20, 0.16])
    z_champ_tr = np.column_stack(z_models_tr) @ opt_zoo_weights

    tr_raw_ext = pd.read_csv(raw_train_path, usecols=["EXT_SOURCE_1", "EXT_SOURCE_2", "EXT_SOURCE_3"])
    te_raw_ext = pd.read_csv(raw_test_path, usecols=["EXT_SOURCE_1", "EXT_SOURCE_2", "EXT_SOURCE_3"])

    m_cnt_tr = tr_raw_ext.isna().sum(axis=1).values
    m_cnt_te = te_raw_ext.isna().sum(axis=1).values

    regime0_tr = (m_cnt_tr == 0).astype(float)
    regime1_tr = (m_cnt_tr == 1).astype(float)
    regime2_tr = (m_cnt_tr >= 2).astype(float)

    regime0_te = (m_cnt_te == 0).astype(float)
    regime1_te = (m_cnt_te == 1).astype(float)
    regime2_te = (m_cnt_te >= 2).astype(float)

    mat_tr = np.column_stack(z_models_tr)
    mat_te = np.column_stack(z_models_te)

    disp_tr = np.std(mat_tr, axis=1)
    disp_te = np.std(mat_te, axis=1)

    tau_thresh = np.percentile(disp_tr, 85.0)
    gate_tr = 1.0 / (1.0 + np.exp(-((disp_tr - tau_thresh) / (np.std(disp_tr) * 0.40))))
    gate_te = 1.0 / (1.0 + np.exp(-((disp_te - tau_thresh) / (np.std(disp_te) * 0.40))))

    def compute_regime_delta(z_mods, r0, r1, r2):
        z_c, z_e, z_x, z_cat, z_d = z_mods
        delta = np.zeros(len(r0), dtype=np.float64)
        delta += r0 * (0.60 * (z_cat - z_c) + 0.40 * (z_d - z_c))
        delta += r1 * (0.50 * (z_c - z_e) + 0.50 * (z_c - z_d))
        delta += r2 * (0.70 * (z_cat - z_x) + 0.30 * (z_cat - z_c))
        return delta

    delta_regime_tr = compute_regime_delta(z_models_tr, regime0_tr, regime1_tr, regime2_tr)
    delta_regime_te = compute_regime_delta(z_models_te, regime0_te, regime1_te, regime2_te)

    best_auc = roc_auc_score(y_tr, z_champ_tr)
    best_p = 1.00
    best_lambda = 0.00

    for p_exp in [1.00, 1.02, 1.04, 1.06, 1.08]:
        for lam in [0.00, 0.02, 0.04, 0.06, 0.08]:
            z_adj = z_champ_tr.copy()
            if lam > 0:
                z_adj += lam * gate_tr * delta_regime_tr
            if p_exp > 1.0:
                pos_mask = z_adj > 0.0
                z_adj[pos_mask] = np.power(z_adj[pos_mask], p_exp)
            score = roc_auc_score(y_tr, z_adj)
            if score > best_auc:
                best_auc = score
                best_p = p_exp
                best_lambda = lam

    z_test_tail = z_champ_te.copy()
    if best_lambda > 0:
        z_test_tail += best_lambda * gate_te * delta_regime_te
    else:
        z_test_tail += 0.035 * gate_te * delta_regime_te

    p_final = best_p if best_p > 1.0 else 1.035
    pos_mask_te = z_test_tail > 0.0
    z_test_tail[pos_mask_te] = np.power(z_test_tail[pos_mask_te], p_final)
    tail_ranks = to_uniform(z_test_tail)

    tail_out = os.path.join(rebuild_dir, "v_20261027_asymmetric_tail_gated_apex.csv")
    pd.DataFrame({"SK_ID_CURR": test_ids, "TARGET": tail_ranks}).to_csv(tail_out, index=False, float_format="%.10f")
    print(f"  * Generated Rebuilt Tail Apex: {tail_out}")

    # =========================================================================
    # Step 2: Reconstruct Optimal Transductive Apex
    # =========================================================================
    print("\n[Step 2] Building Optimal Transductive Apex from Rebuilt Tail Apex...")
    tr_raw_graph = pd.read_csv(raw_train_path, usecols=[
        "SK_ID_CURR", "DAYS_BIRTH", "DAYS_REGISTRATION", "DAYS_ID_PUBLISH",
        "CODE_GENDER", "NAME_EDUCATION_TYPE", "DAYS_EMPLOYED", "DAYS_LAST_PHONE_CHANGE", "TARGET"
    ])
    te_raw_graph = pd.read_csv(raw_test_path, usecols=[
        "SK_ID_CURR", "DAYS_BIRTH", "DAYS_REGISTRATION", "DAYS_ID_PUBLISH",
        "CODE_GENDER", "NAME_EDUCATION_TYPE", "DAYS_EMPLOYED", "DAYS_LAST_PHONE_CHANGE"
    ])

    all_graph = pd.concat([tr_raw_graph.drop(columns=["TARGET"]), te_raw_graph], ignore_index=True)
    all_graph["row"] = np.arange(len(all_graph), dtype=np.int32)
    all_graph["entity_key"] = (
        (all_graph["DAYS_BIRTH"] - all_graph["DAYS_REGISTRATION"]).astype(int).astype(str) + "|" +
        (all_graph["DAYS_BIRTH"] - all_graph["DAYS_ID_PUBLISH"]).astype(int).astype(str) + "|" +
        all_graph["CODE_GENDER"].astype(str) + "|" +
        all_graph["NAME_EDUCATION_TYPE"].astype(str)
    )

    key_counts = all_graph["entity_key"].value_counts()
    multi_df = all_graph[all_graph["entity_key"].map(key_counts) >= 2]
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
    valid_pairs["weight"] = np.where(valid_pairs["confidence_tier"] == 2, 1.0, 0.60)

    test_to_train = valid_pairs[(valid_pairs["row"] >= n_train) & (valid_pairs["row_partner"] < n_train)].copy()
    test_to_train["test_idx"] = test_to_train["row"] - n_train
    test_to_train["train_target"] = y_tr[test_to_train["row_partner"].values]

    test_to_test = valid_pairs[(valid_pairs["row"] >= n_train) & (valid_pairs["row_partner"] >= n_train)].copy()
    test_to_test["test_idx"] = test_to_test["row"] - n_train
    test_to_test["partner_test_idx"] = test_to_test["row_partner"] - n_train

    z_test_opt = to_probit(to_uniform(tail_ranks))

    # Repeat borrower calibration
    gt_summary = test_to_train.groupby("test_idx")["train_target"].mean()
    gamma_gt = 0.12
    s_def = 2.50
    s_rep = -1.30

    for t_idx, target_mean in gt_summary.items():
        if target_mean >= 0.5:
            z_test_opt[t_idx] = (1.0 - gamma_gt) * z_test_opt[t_idx] + gamma_gt * s_def
        elif target_mean == 0.0:
            z_test_opt[t_idx] = (1.0 - gamma_gt) * z_test_opt[t_idx] + gamma_gt * s_rep

    # Graph diffusion across test pairs
    gamma_tt = 0.15
    z_diffused = z_test_opt.copy()
    partner_groups = test_to_test.groupby("test_idx")

    for t_idx, group in partner_groups:
        weights = group["weight"].values
        partners = group["partner_test_idx"].values
        weighted_partner_z = np.sum(weights * z_test_opt[partners]) / np.sum(weights)
        z_diffused[t_idx] = (1.0 - gamma_tt) * z_test_opt[t_idx] + gamma_tt * weighted_partner_z

    # Gated correction using rebuilt models
    tau_thresh_80 = np.percentile(disp_te, 80.0)
    gate_te_80 = 1.0 / (1.0 + np.exp(-((disp_te - tau_thresh_80) / (np.std(disp_te) * 0.35))))
    z_gated_opt = z_diffused + 0.045 * gate_te_80 * delta_regime_te

    def apply_piecewise_tail(z_arr, p_high=1.025, p_low=1.010, tau=1.28):
        z_out = z_arr.copy()
        mask_high = z_out > tau
        z_out[mask_high] = tau + np.power(z_out[mask_high] - tau, p_high)
        mask_low = z_out < -tau
        z_out[mask_low] = -tau - np.power(np.abs(z_out[mask_low]) - tau, p_low)
        return z_out

    z_final_opt = apply_piecewise_tail(z_gated_opt, p_high=1.025, p_low=1.010, tau=1.28)
    final_ranks_opt = to_uniform(z_final_opt)

    opt_out = os.path.join(rebuild_dir, "v_20261027_optimal_transductive_apex.csv")
    pd.DataFrame({"SK_ID_CURR": test_ids, "TARGET": final_ranks_opt}).to_csv(opt_out, index=False, float_format="%.10f")
    print(f"  * Generated Rebuilt Optimal Apex: {opt_out}")

    # =========================================================================
    # Step 3: Reconstruct Precision Transductive Anchor74
    # =========================================================================
    print("\n[Step 3] Building Precision Transductive Anchor74 from Rebuilt Optimal Apex...")
    z_champ_prec = to_probit(to_uniform(final_ranks_opt))
    z_step1_prec = z_champ_prec + 0.080 * gate_te_80 * delta_regime_te

    z_step2_prec = z_step1_prec.copy()
    gamma_harm = 0.12
    max_dz_harm = 0.080

    for _, row in test_to_test.iterrows():
        idx_a = int(row["test_idx"])
        idx_b = int(row["partner_test_idx"])
        tier_weight = 1.0 if row["confidence_tier"] == 2 else 0.50
        diff = z_step1_prec[idx_b] - z_step1_prec[idx_a]
        shift = np.clip(gamma_harm * tier_weight * diff, -max_dz_harm, max_dz_harm)
        z_step2_prec[idx_a] += shift

    max_allowable_dz = 0.35
    dz_total = z_step2_prec - z_champ_prec
    dz_clamped = np.clip(dz_total, -max_allowable_dz, max_allowable_dz)
    z_final_prec = z_champ_prec + dz_clamped

    ordinal_ranks_prec = rankdata(z_final_prec, method="ordinal").astype(np.float64)
    final_ranks_prec = (ordinal_ranks_prec - 0.5) / n_test

    anchor74_out = os.path.join(rebuild_dir, "v_20261027_precision_transductive_apex.csv")
    pd.DataFrame({"SK_ID_CURR": test_ids, "TARGET": final_ranks_prec}).to_csv(anchor74_out, index=False, float_format="%.10f")
    print(f"  * Generated Rebuilt Precision Anchor74: {anchor74_out}")

    # =========================================================================
    # Step 4: EXP85 85/15 Probit Synthesis
    # =========================================================================
    print("\n[Step 4] Executing EXP85 85/15 Probit Synthesis...")
    # Load 8-engine suite from cache
    v17_path = os.path.join(base_gemini, "dependent", "exp85_notebook_run", "historical_cache", "v_17_full_suite_bundle.npz")
    v55_path = os.path.join(base_gemini, "dependent", "exp85_notebook_run", "historical_cache", "v55_members.npz")
    xgb10_oof_path = os.path.join(base_gemini, "dependent", "exp85_notebook_run", "historical_cache", "oof_10f_xgb.npy")
    xgb10_test_path = os.path.join(base_gemini, "dependent", "exp85_notebook_run", "historical_cache", "test_10f_xgb.npy")

    v17_bundle = np.load(v17_path)
    v55_bundle = np.load(v55_path)
    oof_xgb10 = np.load(xgb10_oof_path)
    test_xgb10 = np.load(xgb10_test_path)

    v17_oof_keys = [k for k in v17_bundle.files if k.startswith("oof_")][:6]
    v17_test_keys = [k for k in v17_bundle.files if k.startswith("test_")][:6]

    suite_oof_list = [v17_bundle[k] for k in v17_oof_keys]
    suite_test_list = [v17_bundle[k] for k in v17_test_keys]

    v55_oof_keys = [k for k in v55_bundle.files if k.startswith("oof_") or "oof" in k.lower()]
    v55_test_keys = [k for k in v55_bundle.files if k.startswith("test_") or "test" in k.lower()]

    suite_oof_list.append(np.mean([v55_bundle[k] for k in v55_oof_keys], axis=0))
    suite_test_list.append(np.mean([v55_bundle[k] for k in v55_test_keys], axis=0))

    suite_oof_list.append(oof_xgb10)
    suite_test_list.append(test_xgb10)

    from scipy.optimize import minimize
    def neg_auc(w):
        w = np.maximum(0, w)
        w = w / (np.sum(w) + 1e-12)
        pred = np.zeros(n_train, dtype=np.float64)
        for i in range(len(w)):
            pred += w[i] * suite_oof_list[i]
        return -roc_auc_score(y_tr, pred)

    init_w = np.ones(len(suite_oof_list)) / len(suite_oof_list)
    res = minimize(neg_auc, init_w, method="Nelder-Mead", options={"maxiter": 300, "disp": False})
    opt_w = np.maximum(0, res.x)
    opt_w /= np.sum(opt_w)

    suite_test_pred = np.zeros(n_test, dtype=np.float64)
    for i in range(len(opt_w)):
        suite_test_pred += opt_w[i] * suite_test_list[i]

    # Probit blend 85/15
    z_anchor = to_probit(final_ranks_prec)
    z_suite = to_probit(to_uniform(suite_test_pred))
    z_macro = 0.850 * z_anchor + 0.150 * z_suite
    p_macro = norm.cdf(z_macro)

    final_exp85_out = os.path.join(rebuild_dir, "submission_exp85_rebuilt.csv")
    pd.DataFrame({"SK_ID_CURR": test_ids, "TARGET": np.round(p_macro, 8)}).to_csv(final_exp85_out, index=False, float_format="%.8f")
    print(f"  * Generated Rebuilt EXP85 Submission: {final_exp85_out}")

    # =========================================================================
    # Step 5: Verification against Historical EXP85 Champion
    # =========================================================================
    hist_champ_path = os.path.join(base_gemini, "v_20261027_optimal_macro_apex_v85.csv")
    df_hist = pd.read_csv(hist_champ_path)
    df_reb = pd.read_csv(final_exp85_out)

    sp = spearmanr(df_reb["TARGET"].values, df_hist["TARGET"].values)[0]
    pe = pearsonr(df_reb["TARGET"].values, df_hist["TARGET"].values)[0]
    mae = np.mean(np.abs(df_reb["TARGET"].values - df_hist["TARGET"].values))
    r_diff = np.abs(rankdata(df_reb["TARGET"].values) - rankdata(df_hist["TARGET"].values))
    exact_ranks = np.sum(r_diff == 0)
    max_rank_shift = np.max(r_diff)
    mean_rank_shift = np.mean(r_diff)

    with open(final_exp85_out, "rb") as f: reb_sha = hashlib.sha256(f.read()).hexdigest()
    with open(hist_champ_path, "rb") as f: hist_sha = hashlib.sha256(f.read()).hexdigest()

    print("\n" + "=" * 80)
    print("FINAL EXP85 REBUILD VERIFICATION TELEMETRY")
    print("=" * 80)
    print(f"Historical Champion SHA-256: {hist_sha}")
    print(f"Rebuilt Champion SHA-256:    {reb_sha}")
    print(f"SHA-256 Match:               {hist_sha == reb_sha}")
    print(f"Spearman Rank Correlation:   {sp:.10f}")
    print(f"Pearson Linear Correlation:  {pe:.10f}")
    print(f"Mean Absolute Error:         {mae:.16f}")
    print(f"Exact Rank Agreement:        {exact_ranks} / {len(df_hist)} ({exact_ranks/len(df_hist)*100:.4f}%)")
    print(f"Max Rank Displacement:       {max_rank_shift}")
    print(f"Mean Rank Displacement:      {mean_rank_shift:.4f}")
    print(f"Total Execution Time:        {time.time()-t0:.1f}s")
    print("=" * 80)

if __name__ == '__main__':
    main()

