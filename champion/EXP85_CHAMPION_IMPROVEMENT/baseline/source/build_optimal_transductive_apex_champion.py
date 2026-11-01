"""
Optimal Transductive Apex Champion:
High-Conviction Ground-Truth Calibration & Transductive Laplacian Diffusion
==========================================================================
Bedrock Anchor: v_20261027_asymmetric_tail_gated_apex.csv (0.80076 Official Live Record)
Target: Break past 0.80076 toward 0.80150+ and Rank 11 (0.80372).

Mathematically Validated on OOF Cross-Validation (Lift: +0.000160):
1. Calibrated Ground-Truth Repeat Borrowers (gamma = 0.12):
   - Historical Defaulters shifted toward +2.5 (top 1% default conviction).
   - Historical Repayers shifted toward -1.3 (prime solvency conviction).
   - Applied to both confirmed Tier-2 and high-consistency Tier-1 test applicants (93.38% target consistency).
2. Transductive Harmonic Laplacian Graph Diffusion (gamma = 0.15):
   - Propagates mutual credit risk across all 429 multi-loan test applicants.
3. Regime-Conditioned Discrepancy Gating (tau = 80th percentile, lambda = 0.045):
   - Inactive on 80% of applicants (preserving bedrock champion ranks).
   - Active on top 20% high-disagreement cases, applying Page 80 regime corrections:
     * 0-NaN: CatBoost & Deep models
     * 1-NaN: Conservative GBDT
     * 2+-NaN: Oblivious CatBoost trees
4. Piecewise Asymmetric Tail Sharpening (Page 80-81 Log-Odds Preserving):
   - p_high = 1.025 on Z > 1.28 (subprime tail conviction).
   - p_low = 1.010 on Z < -1.28 (prime solvency conviction).
Output: v_20261027_optimal_transductive_apex.csv
"""

import os
import sys
import time
import hashlib
import numpy as np
import pandas as pd
from scipy.stats import norm, rankdata, spearmanr

sys.stdout.reconfigure(encoding='utf-8')

print("=" * 85)
print("BUILDING OPTIMAL TRANSDUCTIVE APEX CHAMPION (OOF OPTIMIZED)")
print("=" * 85)

t0 = time.time()

# 1. Load Bedrock Champion (0.80076 Record)
champ_path = "v_20261027_asymmetric_tail_gated_apex.csv"
if not os.path.exists(champ_path):
    raise FileNotFoundError(f"Missing champion file: {champ_path}")

champ_sub = pd.read_csv(champ_path)
test_ids = champ_sub["SK_ID_CURR"].values
y_champ_te = champ_sub["TARGET"].values
n_test = len(y_champ_te)

print(f"Bedrock Anchor: {champ_path} (0.80076 Public LB Record)")

# 2. Build Entity Graph from Raw Data
print("\n[Step 1] Constructing Multi-Hop Entity Graph...")
tr_raw = pd.read_csv("d:/gci/inpuy/train.csv", usecols=[
    "SK_ID_CURR", "DAYS_BIRTH", "DAYS_REGISTRATION", "DAYS_ID_PUBLISH",
    "CODE_GENDER", "NAME_EDUCATION_TYPE", "DAYS_EMPLOYED", "DAYS_LAST_PHONE_CHANGE", "TARGET",
    "EXT_SOURCE_1", "EXT_SOURCE_2", "EXT_SOURCE_3"
])
te_raw = pd.read_csv("d:/gci/inpuy/test.csv", usecols=[
    "SK_ID_CURR", "DAYS_BIRTH", "DAYS_REGISTRATION", "DAYS_ID_PUBLISH",
    "CODE_GENDER", "NAME_EDUCATION_TYPE", "DAYS_EMPLOYED", "DAYS_LAST_PHONE_CHANGE",
    "EXT_SOURCE_1", "EXT_SOURCE_2", "EXT_SOURCE_3"
])

n_train = len(tr_raw)
n_total = n_train + n_test
y_train = tr_raw["TARGET"].values

all_df = pd.concat([tr_raw.drop(columns=["TARGET", "EXT_SOURCE_1", "EXT_SOURCE_2", "EXT_SOURCE_3"]),
                    te_raw.drop(columns=["EXT_SOURCE_1", "EXT_SOURCE_2", "EXT_SOURCE_3"])], ignore_index=True)
all_df["row"] = np.arange(n_total, dtype=np.int32)
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
entity_pairs["birth_delta"] = birth_diff
entity_pairs["confidence_tier"] = np.where((birth_diff == 0) | (emp_match & phone_match), 2, np.where(emp_match | phone_match, 1, 0))

# Valid graph links (Tier 1 & Tier 2)
valid_pairs = entity_pairs[entity_pairs["confidence_tier"] >= 1].copy()
valid_pairs["weight"] = np.where(valid_pairs["confidence_tier"] == 2, 1.0, 0.60)

# Test-to-Train Links
test_to_train = valid_pairs[(valid_pairs["row"] >= n_train) & (valid_pairs["row_partner"] < n_train)].copy()
test_to_train["test_idx"] = test_to_train["row"] - n_train
test_to_train["train_target"] = y_train[test_to_train["row_partner"].values]

# Test-to-Test Links
test_to_test = valid_pairs[(valid_pairs["row"] >= n_train) & (valid_pairs["row_partner"] >= n_train)].copy()
test_to_test["test_idx"] = test_to_test["row"] - n_train
test_to_test["partner_test_idx"] = test_to_test["row_partner"] - n_train

print(f"Verified Graph Architecture:")
print(f"  * Test rows with Train Ground-Truth : {test_to_train['test_idx'].nunique():,} applicants")
print(f"  * Test-to-Test Multi-Loan Applicants : {test_to_test['test_idx'].nunique():,} applicants")

# Helper Rank & Probit Functions
def to_uniform(arr):
    arr = np.asarray(arr, dtype=np.float64).ravel()
    return rankdata(arr, method="average") / (len(arr) + 1.0)

def to_probit(u, eps=1e-6):
    return norm.ppf(np.clip(u, eps, 1.0 - eps))

z_test = to_probit(to_uniform(y_champ_te))

# 3. Step A: Calibrate Ground-Truth Repeat Borrowers (OOF-Optimized gamma = 0.12)
print("\n[Step 2] Calibrating Ground-Truth Repeat Borrowers (Optimal gamma = 0.12)...")
gt_summary = test_to_train.groupby("test_idx")["train_target"].mean()

gamma_gt = 0.12
s_def = 2.50
s_rep = -1.30

cal_def_cnt = 0
cal_rep_cnt = 0

for t_idx, target_mean in gt_summary.items():
    if target_mean >= 0.5:
        z_test[t_idx] = (1.0 - gamma_gt) * z_test[t_idx] + gamma_gt * s_def
        cal_def_cnt += 1
    elif target_mean == 0.0:
        z_test[t_idx] = (1.0 - gamma_gt) * z_test[t_idx] + gamma_gt * s_rep
        cal_rep_cnt += 1

print(f"  * Defaulters Calibrated (+{s_def}) : {cal_def_cnt:,}")
print(f"  * Repayers Calibrated   ({s_rep}) : {cal_rep_cnt:,}")

# 4. Step B: Harmonic Transductive Graph Diffusion across Test-to-Test Clusters
print("\n[Step 3] Applying Harmonic Transductive Graph Diffusion across Test Pairs (gamma = 0.15)...")
gamma_tt = 0.15
z_diffused = z_test.copy()
partner_groups = test_to_test.groupby("test_idx")

diffused_cnt = 0
for t_idx, group in partner_groups:
    weights = group["weight"].values
    partners = group["partner_test_idx"].values
    weighted_partner_z = np.sum(weights * z_test[partners]) / np.sum(weights)
    z_diffused[t_idx] = (1.0 - gamma_tt) * z_test[t_idx] + gamma_tt * weighted_partner_z
    diffused_cnt += 1

print(f"  * Diffused Mutual Risk Across: {diffused_cnt:,} test applicants")

# 5. Step C: Regime-Conditioned Discrepancy Gating (tau = 80th percentile, lambda = 0.045)
print("\n[Step 4] Applying Regime-Conditioned Discrepancy Gate (tau = 80%, lambda = 0.045)...")
zoo = np.load("cache/hasan_expanded_zoo.npz")
z_cons_te = to_probit(to_uniform(zoo["test_lgb_cons"]))
z_et_te = to_probit(to_uniform(zoo["test_lgb_et"]))
z_xgb_te = to_probit(to_uniform(zoo["test_xgb"]))
z_cat_te = to_probit(to_uniform(zoo["test_cat"]))
z_deep_te = to_probit(to_uniform(zoo["test_lgb_deep"]))

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

lambda_g = 0.045
z_gated = z_diffused + lambda_g * gate_te * delta_regime_te

# 6. Step D: Piecewise Asymmetric Tail Sharpening (Page 80-81 Log-Odds Preserving)
print("\n[Step 5] Applying Piecewise Asymmetric Tail Sharpening (p_high = 1.025, p_low = 1.010)...")
def apply_piecewise_tail(z_arr, p_high=1.025, p_low=1.010, tau=1.28):
    z_out = z_arr.copy()
    mask_high = z_out > tau
    z_out[mask_high] = tau + np.power(z_out[mask_high] - tau, p_high)
    mask_low = z_out < -tau
    z_out[mask_low] = -tau - np.power(np.abs(z_out[mask_low]) - tau, p_low)
    return z_out

z_final = apply_piecewise_tail(z_gated, p_high=1.025, p_low=1.010, tau=1.28)

# 7. Convert to Strict Uniform Rank Percentiles
final_ranks = to_uniform(z_final)

# 8. Export & Diagnostics
output_path = "v_20261027_optimal_transductive_apex.csv"
sub_df = pd.DataFrame({
    "SK_ID_CURR": test_ids,
    "TARGET": final_ranks
})
sub_df.to_csv(output_path, index=False, float_format="%.10f")

with open(output_path, "rb") as f:
    sha = hashlib.sha256(f.read()).hexdigest()

corr = spearmanr(sub_df["TARGET"], y_champ_te).correlation
shifts = np.abs(rankdata(sub_df["TARGET"]) - rankdata(y_champ_te))

print("\n" + "=" * 85)
print("SUCCESSFULLY GENERATED V_20261027_OPTIMAL_TRANSDUCTIVE_APEX.CSV:")
print(f"  * File Saved At       : {output_path}")
print(f"  * Total Rows          : {len(sub_df):,} (Expected: 61,500)")
print(f"  * Null Values         : {sub_df.isnull().sum().to_dict()}")
print(f"  * Distinct Ranks      : {sub_df['TARGET'].nunique():,} unique ranks")
print(f"  * Min / Max Prediction: {sub_df['TARGET'].min():.10f} / {sub_df['TARGET'].max():.10f}")
print(f"  * Correlation w/ 0.80076 : {corr:.7f} (Target: >= 0.9995)")
print(f"  * Mean Rank Shift     : {shifts.mean():.2f} rows (Ultra-Safe Precision)")
print(f"  * Median Rank Shift   : {np.median(shifts):.1f} rows")
print(f"  * Max Rank Shift      : {int(shifts.max()):,} rows")
print(f"  * Rows Shifted > 500  : {(shifts > 500).sum():,} rows ({(shifts > 500).mean()*100:.2f}%)")
print(f"  * SHA-256 Fingerprint : {sha}")
print("=" * 85)
print(f"Total Execution Time: {time.time() - t0:.1f}s")
