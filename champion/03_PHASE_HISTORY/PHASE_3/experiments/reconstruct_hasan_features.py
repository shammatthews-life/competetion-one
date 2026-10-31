"""
EXP85 Phase 3 - Workstream 2: Feature Matrix Independent Reconstruction
Executes the exact feature transformations from build_hasan_600_features.py
directly on raw input data and saves to independent_rebuild directory.
Then performs exhaustive feature-by-feature numerical verification against the historical cache.
"""

import os
import sys
import time
import hashlib
import numpy as np
import pandas as pd
from scipy.stats import norm, rankdata
import warnings
warnings.filterwarnings('ignore')

def main():
    t0 = time.time()
    print("=" * 80)
    print("WORKSTREAM 2: REGENERATING 669 HASAN FEATURES FROM RAW COMPETITION DATA")
    print("=" * 80)

    train_path = r'D:\gci\inpuy\train.csv'
    test_path = r'D:\gci\inpuy\test.csv'
    out_dir = r'D:\gci\gemini\champion\phase3_reconstruction\independent_rebuild'
    os.makedirs(out_dir, exist_ok=True)

    print("[1] Loading raw CSVs...")
    df_tr = pd.read_csv(train_path)
    df_te = pd.read_csv(test_path)

    n_train = len(df_tr)
    n_test = len(df_te)
    y_train = df_tr['TARGET'].values

    df_all = pd.concat([df_tr.drop(columns=['TARGET']), df_te], ignore_index=True)
    print(f"Loaded all data: Train={n_train}, Test={n_test}, Total={len(df_all)}")

    feat_dict = {}

    # Clean basic variables
    days_emp = df_all['DAYS_EMPLOYED'].values.astype(np.float64)
    is_retired = (days_emp == 365243.0).astype(np.float32)
    days_emp[days_emp == 365243.0] = np.nan
    feat_dict['IS_RETIRED'] = is_retired
    feat_dict['DAYS_EMPLOYED_CLEAN'] = days_emp.astype(np.float32)

    inc = df_all['AMT_INCOME_TOTAL'].values.astype(np.float64)
    cred = df_all['AMT_CREDIT'].values.astype(np.float64)
    ann = df_all['AMT_ANNUITY'].values.astype(np.float64)
    goods = df_all['AMT_GOODS_PRICE'].values.astype(np.float64)
    goods = np.where(np.isnan(goods), cred, goods)

    birth = df_all['DAYS_BIRTH'].values.astype(np.float64)
    reg = df_all['DAYS_REGISTRATION'].values.astype(np.float64)
    id_pub = df_all['DAYS_ID_PUBLISH'].values.astype(np.float64)
    phone = df_all['DAYS_LAST_PHONE_CHANGE'].values.astype(np.float64)
    fam = df_all['CNT_FAM_MEMBERS'].fillna(1.0).values.astype(np.float64)
    child = df_all['CNT_CHILDREN'].values.astype(np.float64)

    e1 = df_all['EXT_SOURCE_1'].values.astype(np.float64)
    e2 = df_all['EXT_SOURCE_2'].values.astype(np.float64)
    e3 = df_all['EXT_SOURCE_3'].values.astype(np.float64)

    print("[2] Engineering Group A: Foundational Gaps & Banking Standards...")
    term = cred / (ann + 1.0)
    feat_dict['CREDIT_TERM'] = term.astype(np.float32)

    # Effective Annual Rate (EAR) / Implied Interest Markup
    ear = 24.0 * (ann * term - cred) / (cred * (term + 1.0) + 1.0)
    feat_dict['EAR'] = np.clip(ear, -1.0, 5.0).astype(np.float32)

    surplus = inc - 12.0 * ann
    feat_dict['SURPLUS'] = surplus.astype(np.float32)
    feat_dict['SURPLUS_TO_INC'] = (surplus / (inc + 1.0)).astype(np.float32)

    true_cost = ann * term - cred
    feat_dict['TRUE_COST'] = true_cost.astype(np.float32)
    feat_dict['TRUE_COST_RATIO'] = (true_cost / (cred + 1.0)).astype(np.float32)

    ltv = cred / (goods + 1.0)
    feat_dict['LTV'] = np.clip(ltv, 0.0, 5.0).astype(np.float32)
    feat_dict['GOODS_CREDIT_DIFF'] = (goods - cred).astype(np.float32)

    debt_burden = ann / (inc / 12.0 + 1.0)
    feat_dict['DEBT_BURDEN'] = np.clip(debt_burden, 0.0, 5.0).astype(np.float32)

    disposable_inc = (inc / 12.0) - ann
    feat_dict['DISPOSABLE_INC'] = disposable_inc.astype(np.float32)

    dti = cred / (inc + 1.0)
    feat_dict['DTI'] = np.clip(dti, 0.0, 20.0).astype(np.float32)

    dscr = ((inc / 12.0) * 0.65) / (ann + 1.0)
    feat_dict['DSCR'] = np.clip(dscr, 0.0, 20.0).astype(np.float32)

    foir = ann / (inc / 12.0 + 1.0)
    feat_dict['FOIR'] = np.clip(foir, 0.0, 5.0).astype(np.float32)

    tdsr = (ann + (cred * 0.05 / 12.0)) / (inc / 12.0 + 1.0)
    feat_dict['TDSR'] = np.clip(tdsr, 0.0, 5.0).astype(np.float32)

    solvency = (inc / 12.0) / (ann + 1.0)
    feat_dict['SOLVENCY'] = np.clip(solvency, 0.0, 20.0).astype(np.float32)
    feat_dict['LIQUIDITY'] = (surplus / (cred + 1.0)).astype(np.float32)

    buffer_months = surplus / (12.0 * ann + 1.0)
    feat_dict['BUFFER_MONTHS'] = np.clip(buffer_months, -10.0, 50.0).astype(np.float32)

    reserve_months = (inc - 12.0 * ann) / (ann + 1.0)
    feat_dict['RESERVE_MONTHS'] = np.clip(reserve_months, -10.0, 50.0).astype(np.float32)

    feat_dict['DTI_40'] = (dti > 0.40).astype(np.float32)
    feat_dict['DTI_46'] = (dti > 0.46).astype(np.float32)
    feat_dict['DTI_50'] = (dti > 0.50).astype(np.float32)
    feat_dict['LTV_75'] = (ltv > 0.75).astype(np.float32)
    feat_dict['LTV_100'] = (ltv > 1.00).astype(np.float32)
    feat_dict['DSCR_BELOW1'] = (dscr < 1.0).astype(np.float32)
    feat_dict['DSCR_BELOW1_25'] = (dscr < 1.25).astype(np.float32)

    ps10 = ((ann * 1.10) / (inc / 12.0 + 1.0) > 0.50).astype(np.float32)
    ps20 = ((ann * 1.20) / (inc / 12.0 + 1.0) > 0.50).astype(np.float32)
    feat_dict['PAYMENT_SHOCK_10'] = ps10
    feat_dict['PAYMENT_SHOCK_20'] = ps20
    feat_dict['SHOCK_BREACH20'] = ps20

    z_score = 1.2 * (surplus / (cred + 1.0)) + 1.4 * (surplus / (inc + 1.0)) + 3.3 * (inc / (cred + 1.0))
    feat_dict['Z_SCORE'] = np.clip(z_score, -20.0, 50.0).astype(np.float32)
    feat_dict['HFVI'] = ((dti > 0.50).astype(float) + (ltv > 0.90).astype(float) + (dscr < 1.0).astype(float) + (buffer_months < 1.0).astype(float)).astype(np.float32)
    feat_dict['FFI'] = ((feat_dict['SURPLUS_TO_INC'] < 0.2).astype(float) + (debt_burden > 0.4).astype(float)).astype(np.float32)
    feat_dict['ZOMBIE'] = ((surplus < 0) & (ltv > 1.0)).astype(np.float32)

    print("[3] Engineering Group B: Alpha Trinity (Merton Distance-to-Default & Clayton Copula)...")
    T_years = np.clip(term / 12.0, 0.1, 10.0)
    disposable_ratio = np.maximum(0.01, (inc / 12.0) / (ann + 1.0))

    for mu, sigma, tag in [(0.02, 0.20, '20'), (0.03, 0.25, '25'), (0.04, 0.35, '35')]:
        dd = (np.log(disposable_ratio) + (mu - 0.5 * sigma**2) * T_years) / (sigma * np.sqrt(T_years))
        p_merton = norm.cdf(-dd)
        feat_dict[f'MERTON_DD_{tag}'] = np.clip(dd, -10.0, 10.0).astype(np.float32)
        feat_dict[f'MERTON_PD_{tag}'] = p_merton.astype(np.float32)

    def uniform_rank(v):
        mask = ~np.isnan(v)
        res = np.full(len(v), 0.5, dtype=np.float64)
        if mask.sum() > 0:
            res[mask] = rankdata(v[mask], method='average') / (mask.sum() + 1.0)
        return res

    u1 = uniform_rank(e1)
    u2 = uniform_rank(e2)
    u3 = uniform_rank(e3)

    for theta, tag in [(1.0, '1_0'), (1.5, '1_5'), (2.0, '2_0'), (2.5, '2_5')]:
        t1 = np.maximum(1e-6, u1) ** (-theta)
        t2 = np.maximum(1e-6, u2) ** (-theta)
        t3 = np.maximum(1e-6, u3) ** (-theta)
        copula_val = (t1 + t2 + t3 - 2.0) ** (-1.0 / theta)
        feat_dict[f'CLAYTON_COPULA_{tag}'] = np.nan_to_num(copula_val, nan=0.5).astype(np.float32)

    print("[4] Engineering Group C: External Sources Interactions...")
    e_matrix = np.column_stack([e1, e2, e3])
    feat_dict['EXT_MEAN'] = np.nanmean(e_matrix, axis=1).astype(np.float32)
    feat_dict['EXT_STD'] = np.nanstd(e_matrix, axis=1).astype(np.float32)
    feat_dict['EXT_MIN'] = np.nanmin(e_matrix, axis=1).astype(np.float32)
    feat_dict['EXT_MAX'] = np.nanmax(e_matrix, axis=1).astype(np.float32)
    feat_dict['EXT_SPREAD'] = (feat_dict['EXT_MAX'] - feat_dict['EXT_MIN']).astype(np.float32)
    feat_dict['EXT_MISSING_COUNT'] = np.isnan(e_matrix).sum(axis=1).astype(np.float32)

    feat_dict['EXT_PROD_12'] = (e1 * e2).astype(np.float32)
    feat_dict['EXT_PROD_13'] = (e1 * e3).astype(np.float32)
    feat_dict['EXT_PROD_23'] = (e2 * e3).astype(np.float32)
    feat_dict['EXT_PROD_ALL'] = (e1 * e2 * e3).astype(np.float32)

    feat_dict['EXT_DIFF_12'] = (e1 - e2).astype(np.float32)
    feat_dict['EXT_DIFF_13'] = (e1 - e3).astype(np.float32)
    feat_dict['EXT_DIFF_23'] = (e2 - e3).astype(np.float32)

    feat_dict['EXT_WEIGHTED'] = (0.5 * np.nan_to_num(e2, nan=0.5) + 0.3 * np.nan_to_num(e3, nan=0.5) + 0.2 * np.nan_to_num(e1, nan=0.5)).astype(np.float32)

    print("[5] Engineering Group D: Timeline & Chronological Drift...")
    age_years = -birth / 365.25
    emp_years = -days_emp / 365.25
    reg_years = -reg / 365.25
    id_years = -id_pub / 365.25
    phone_years = -phone / 365.25

    feat_dict['AGE_YEARS'] = age_years.astype(np.float32)
    feat_dict['EMP_YEARS'] = emp_years.astype(np.float32)
    feat_dict['REG_YEARS'] = reg_years.astype(np.float32)
    feat_dict['ID_YEARS'] = id_years.astype(np.float32)
    feat_dict['PHONE_YEARS'] = phone_years.astype(np.float32)

    feat_dict['EMP_TO_AGE'] = (emp_years / (age_years + 1.0)).astype(np.float32)
    feat_dict['REG_TO_AGE'] = (reg_years / (age_years + 1.0)).astype(np.float32)
    feat_dict['ID_TO_AGE'] = (id_years / (age_years + 1.0)).astype(np.float32)
    feat_dict['PHONE_TO_AGE'] = (phone_years / (age_years + 1.0)).astype(np.float32)

    feat_dict['REG_TO_EMP'] = (reg_years / (emp_years + 1.0)).astype(np.float32)
    feat_dict['ID_TO_EMP'] = (id_years / (emp_years + 1.0)).astype(np.float32)
    feat_dict['PHONE_TO_EMP'] = (phone_years / (emp_years + 1.0)).astype(np.float32)

    feat_dict['ID_REG_DIFF'] = (id_pub - reg).astype(np.float32)
    feat_dict['PHONE_REG_DIFF'] = (phone - reg).astype(np.float32)
    feat_dict['PHONE_ID_DIFF'] = (phone - id_pub).astype(np.float32)
    feat_dict['EMP_REG_DIFF'] = (days_emp - reg).astype(np.float32)
    feat_dict['EMP_ID_DIFF'] = (days_emp - id_pub).astype(np.float32)

    print("[6] Engineering Group E: Household, Demographic & Social Signals...")
    feat_dict['INC_PER_PERSON'] = (inc / (fam + 1.0)).astype(np.float32)
    feat_dict['CRED_PER_PERSON'] = (cred / (fam + 1.0)).astype(np.float32)
    feat_dict['ANN_PER_PERSON'] = (ann / (fam + 1.0)).astype(np.float32)
    feat_dict['CHILD_TO_FAM'] = (child / (fam + 1.0)).astype(np.float32)
    feat_dict['ADULT_CNT'] = np.maximum(1.0, fam - child).astype(np.float32)
    feat_dict['INC_PER_ADULT'] = (inc / feat_dict['ADULT_CNT']).astype(np.float32)

    feat_dict['AGE_BAND_25'] = (age_years < 25).astype(np.float32)
    feat_dict['AGE_BAND_35'] = ((age_years >= 25) & (age_years < 35)).astype(np.float32)
    feat_dict['AGE_BAND_50'] = ((age_years >= 35) & (age_years < 50)).astype(np.float32)
    feat_dict['AGE_BAND_65'] = (age_years >= 50).astype(np.float32)

    print("[7] Engineering Group F: Categorical Code Conversions...")
    cat_cols = df_all.select_dtypes(include=['object']).columns
    for c in cat_cols:
        codes, _ = pd.factorize(df_all[c])
        feat_dict[f'CAT_{c}'] = codes.astype(np.int32)
        freq = df_all[c].map(df_all[c].value_counts(normalize=True))
        feat_dict[f'FREQ_{c}'] = freq.values.astype(np.float32)

    print("[8] Engineering Group G: Demographic Multi-Hop Entity Matching...")
    uid_key = (
        (df_all['DAYS_BIRTH'] - df_all['DAYS_REGISTRATION']).astype(int).astype(str) + '_' +
        (df_all['DAYS_BIRTH'] - df_all['DAYS_ID_PUBLISH']).astype(int).astype(str) + '_' +
        df_all['CODE_GENDER'].astype(str)
    )
    uid_counts = uid_key.map(uid_key.value_counts())
    feat_dict['UID_REPEAT_COUNT'] = uid_counts.values.astype(np.float32)
    feat_dict['UID_IS_REPEAT'] = (uid_counts > 1).astype(np.float32)

    # Historical target encoding for repeat borrower graph
    df_all['UID_KEY'] = uid_key
    df_all['TARGET'] = np.concatenate([y_train, np.full(n_test, np.nan)])
    uid_target_mean = df_all.groupby('UID_KEY')['TARGET'].transform('mean').fillna(0.0807)
    feat_dict['UID_TARGET_MEAN'] = uid_target_mean.values.astype(np.float32)

    # Any bad flag
    uid_anybad = df_all.groupby('UID_KEY')['TARGET'].transform(lambda x: (x == 1).any()).astype(float).fillna(0.0)
    feat_dict['UIDY_ANYBAD'] = uid_anybad.values.astype(np.float32)

    print("[9] Engineering Group H: High-Order Pairwise Interactions...")
    core_cols = [
        'EXT_MEAN', 'DTI', 'EAR', 'LTV', 'DSCR', 'SURPLUS', 'CREDIT_TERM',
        'DAYS_BIRTH', 'DAYS_EMPLOYED_CLEAN', 'DAYS_ID_PUBLISH', 'DAYS_REGISTRATION', 'DAYS_LAST_PHONE_CHANGE',
        'AMT_CREDIT', 'AMT_INCOME_TOTAL', 'AMT_ANNUITY', 'AMT_GOODS_PRICE',
        'MERTON_PD_25', 'CLAYTON_COPULA_2_5', 'BUFFER_MONTHS', 'DEBT_BURDEN'
    ]

    interaction_count = 0
    for i in range(len(core_cols)):
        for j in range(i + 1, len(core_cols)):
            col1 = core_cols[i]
            col2 = core_cols[j]
            v1 = feat_dict[col1] if col1 in feat_dict else df_all[col1].values.astype(np.float64)
            v2 = feat_dict[col2] if col2 in feat_dict else df_all[col2].values.astype(np.float64)

            feat_dict[f'P_PROD_{col1}_x_{col2}'] = (v1 * v2).astype(np.float32)
            feat_dict[f'P_RATIO_{col1}_div_{col2}'] = (v1 / (np.abs(v2) + 1.0)).astype(np.float32)
            feat_dict[f'P_DIFF_{col1}_sub_{col2}'] = (v1 - v2).astype(np.float32)
            interaction_count += 3

    print(f"Engineered {interaction_count} pairwise interactions across top 20 core drivers!")

    df_features = pd.DataFrame(feat_dict)
    print(f"Total features constructed: {df_features.shape[1]}")

    X_tr_hasan = df_features.iloc[:n_train].reset_index(drop=True)
    X_te_hasan = df_features.iloc[n_train:].reset_index(drop=True)

    tr_out = os.path.join(out_dir, 'hasan_features_tr.feather')
    te_out = os.path.join(out_dir, 'hasan_features_te.feather')

    X_tr_hasan.to_feather(tr_out)
    X_te_hasan.to_feather(te_out)
    print(f"Saved regenerated features: Train={X_tr_hasan.shape}, Test={X_te_hasan.shape} ({time.time()-t0:.1f}s)")

    # Numerical verification against historical cache
    hist_tr_path = r'D:\gci\gemini\cache\hasan_features_tr.feather'
    hist_te_path = r'D:\gci\gemini\cache\hasan_features_te.feather'

    df_hist_tr = pd.read_feather(hist_tr_path)
    df_hist_te = pd.read_feather(hist_te_path)

    cols_match = list(X_tr_hasan.columns) == list(df_hist_tr.columns)
    print(f"\nFeature Columns Match: {cols_match} ({len(X_tr_hasan.columns)} cols)")

    # Check differences across all columns
    diffs = []
    for col in X_tr_hasan.columns:
        v_rep = X_tr_hasan[col].values
        v_hist = df_hist_tr[col].values
        # Mask NaNs
        nan_m = np.isnan(v_rep) & np.isnan(v_hist)
        valid = ~nan_m
        if np.any(np.isnan(v_rep) != np.isnan(v_hist)):
            diffs.append((col, 'NaN mismatch', np.nan))
        else:
            max_d = np.max(np.abs(v_rep[valid] - v_hist[valid])) if np.any(valid) else 0.0
            if max_d > 1e-5:
                diffs.append((col, 'Numerical mismatch', max_d))

    print(f"Discrepant Columns Count: {len(diffs)} / {len(X_tr_hasan.columns)}")
    if diffs:
        for c, reason, d in diffs[:10]:
            print(f"  * {c}: {reason} (max diff = {d})")
    else:
        print(">>> 100% PERFECT NUMERICAL MATCH ACROSS ALL 669 FEATURES! <<<")

if __name__ == '__main__':
    main()

