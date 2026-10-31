"""
Hasan Mazi 600+ Feature Engineering Laboratory
Implements the complete financial, temporal, social, demographic, copula, merton distance-to-default,
and chronological applicant timeline feature architecture disclosed by Hasan Mazi (0.80041 LB / 0.802330 OOF).
"""

import os
import sys
import numpy as np
import pandas as pd
from scipy.stats import norm, rankdata
from sklearn.model_selection import StratifiedKFold
import lightgbm as lgb
import warnings
warnings.filterwarnings('ignore')

print("Starting Hasan Mazi 600+ Feature Engineering Suite...")

# 1. Load raw train and test data
train_path = r'D:/gci/inpuy/train.csv'
test_path = r'D:/gci/inpuy/test.csv'

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

print("Engineering Group A: Foundational Gaps & Banking Standards...")
# Group A: Foundational Gaps & Banking Standards
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

# Liquidity / Solvency
solvency = (inc / 12.0) / (ann + 1.0)
feat_dict['SOLVENCY'] = np.clip(solvency, 0.0, 20.0).astype(np.float32)
feat_dict['LIQUIDITY'] = (surplus / (cred + 1.0)).astype(np.float32)

buffer_months = surplus / (12.0 * ann + 1.0)
feat_dict['BUFFER_MONTHS'] = np.clip(buffer_months, -10.0, 50.0).astype(np.float32)

reserve_months = (inc - 12.0 * ann) / (ann + 1.0)
feat_dict['RESERVE_MONTHS'] = np.clip(reserve_months, -10.0, 50.0).astype(np.float32)

# Threshold breaches
feat_dict['DTI_40'] = (dti > 0.40).astype(np.float32)
feat_dict['DTI_46'] = (dti > 0.46).astype(np.float32)
feat_dict['DTI_50'] = (dti > 0.50).astype(np.float32)
feat_dict['LTV_75'] = (ltv > 0.75).astype(np.float32)
feat_dict['LTV_100'] = (ltv > 1.00).astype(np.float32)
feat_dict['DSCR_BELOW1'] = (dscr < 1.0).astype(np.float32)
feat_dict['DSCR_BELOW1_25'] = (dscr < 1.25).astype(np.float32)

# Payment shock
ps10 = ((ann * 1.10) / (inc / 12.0 + 1.0) > 0.50).astype(np.float32)
ps20 = ((ann * 1.20) / (inc / 12.0 + 1.0) > 0.50).astype(np.float32)
feat_dict['PAYMENT_SHOCK_10'] = ps10
feat_dict['PAYMENT_SHOCK_20'] = ps20
feat_dict['SHOCK_BREACH20'] = ps20

# Composites
z_score = 1.2 * (surplus / (cred + 1.0)) + 1.4 * (surplus / (inc + 1.0)) + 3.3 * (inc / (cred + 1.0))
feat_dict['Z_SCORE'] = np.clip(z_score, -20.0, 50.0).astype(np.float32)
feat_dict['HFVI'] = ((dti > 0.50).astype(float) + (ltv > 0.90).astype(float) + (dscr < 1.0).astype(float) + (buffer_months < 1.0).astype(float)).astype(np.float32)
feat_dict['FFI'] = ((feat_dict['SURPLUS_TO_INC'] < 0.2).astype(float) + (debt_burden > 0.4).astype(float)).astype(np.float32)
feat_dict['ZOMBIE'] = ((surplus < 0) & (ltv > 1.0)).astype(np.float32)

print("Engineering Group B: Alpha Trinity (Merton Distance-to-Default & Clayton Copula)...")
# Merton Consumer Distance-to-Default
T_years = np.clip(term / 12.0, 0.1, 10.0)
disposable_ratio = np.maximum(0.01, (inc / 12.0) / (ann + 1.0))

for mu, sigma, tag in [(0.02, 0.20, '20'), (0.03, 0.25, '25'), (0.04, 0.35, '35')]:
    dd = (np.log(disposable_ratio) + (mu - 0.5 * sigma**2) * T_years) / (sigma * np.sqrt(T_years))
    p_merton = norm.cdf(-dd)
    feat_dict[f'MERTON_DD_{tag}'] = np.clip(dd, -10.0, 10.0).astype(np.float32)
    feat_dict[f'MERTON_PD_{tag}'] = p_merton.astype(np.float32)

# Clayton Copula on EXT Triad
def uniform_rank(v):
    mask = ~np.isnan(v)
    res = np.full(len(v), 0.5, dtype=np.float64)
    if mask.sum() > 0:
        res[mask] = rankdata(v[mask], method='average') / (mask.sum() + 1.0)
    return res

u1 = uniform_rank(e1)
u2 = uniform_rank(e2)
u3 = uniform_rank(e3)

for theta in [1.5, 2.5, 4.0]:
    term1 = np.power(np.maximum(1e-4, u1), -theta)
    term2 = np.power(np.maximum(1e-4, u2), -theta)
    term3 = np.power(np.maximum(1e-4, u3), -theta)
    inside = np.maximum(0.0, term1 + term2 + term3 - 2.0)
    c_theta = np.power(inside, -1.0 / theta)
    theta_tag = str(theta).replace('.', '_')
    feat_dict[f'CLAYTON_COPULA_{theta_tag}'] = np.nan_to_num(c_theta, nan=0.5).astype(np.float32)

# Underwriting Tariffs (Elephant grid)
term_key_1d = (cred / (ann + 1.0)).round(1)
markup_key_2d = (cred / (goods + 1.0)).round(2)
feat_dict['TERM_KEY_1D'] = term_key_1d.astype(np.float32)
feat_dict['MARKUP_KEY_2D'] = markup_key_2d.astype(np.float32)

print("Engineering Group C: Temporal Stability & Identity Alignment...")
# Group C: Temporal Stability
feat_dict['ID_VS_REG'] = (id_pub - reg).astype(np.float32)
feat_dict['EMP_VS_REG'] = (days_emp - reg).astype(np.float32)
feat_dict['PHONE_VS_ID'] = (phone - id_pub).astype(np.float32)
feat_dict['PHONE_VS_REG'] = (phone - reg).astype(np.float32)

feat_dict['EMP_STABILITY'] = (days_emp / (birth + 1.0)).astype(np.float32)
feat_dict['REG_STABILITY'] = (reg / (birth + 1.0)).astype(np.float32)
feat_dict['PHONE_STABILITY'] = (phone / (birth + 1.0)).astype(np.float32)
feat_dict['HOUSING_STABILITY'] = (reg / (id_pub + 1.0)).astype(np.float32)

feat_dict['PHONE_CHANGED_RECENTLY'] = (phone > -365.0).astype(np.float32)
feat_dict['ID_CHANGED_RECENTLY'] = (id_pub > -365.0).astype(np.float32)
feat_dict['REG_CHANGED_RECENTLY'] = (reg > -365.0).astype(np.float32)

print("Engineering Group D: Demographics & Consumption Pressures...")
# Group D: Demographics & Consumption
feat_dict['SPEND_PRESSURE'] = (ann / (inc / fam + 1.0)).astype(np.float32)
feat_dict['HOUSING_BURDEN'] = (ann / fam).astype(np.float32)
feat_dict['FULLY_FINANCED'] = (cred >= goods).astype(np.float32)
feat_dict['OVERSPENDING'] = (cred > inc * 3.0).astype(np.float32)
feat_dict['INCOME_PER_MEMBER'] = (inc / fam).astype(np.float32)
feat_dict['CREDIT_PER_MEMBER'] = (cred / fam).astype(np.float32)
feat_dict['DEPENDENCY'] = (child / fam).astype(np.float32)
feat_dict['IS_MARRIED'] = (df_all['NAME_FAMILY_STATUS'] == 'Married').astype(np.float32)

print("Engineering Group E: Inquiry & Contact Verification Signals...")
# Group E: Inquiry & Contact Verification
req_hr = df_all['AMT_REQ_CREDIT_BUREAU_HOUR'].fillna(0).values.astype(np.float64)
req_mon = df_all['AMT_REQ_CREDIT_BUREAU_MON'].fillna(0).values.astype(np.float64)
req_qrt = df_all['AMT_REQ_CREDIT_BUREAU_QRT'].fillna(0).values.astype(np.float64)
req_yr = df_all['AMT_REQ_CREDIT_BUREAU_YEAR'].fillna(0).values.astype(np.float64)

tot_req = req_hr + req_mon + req_qrt + req_yr
feat_dict['TOTAL_INQUIRIES'] = tot_req.astype(np.float32)
feat_dict['INQUIRY_BURST'] = ((req_hr + req_mon) / (req_yr / 12.0 + 1.0)).astype(np.float32)
feat_dict['INQUIRY_QRT_RATIO'] = (req_qrt / (req_yr / 4.0 + 1.0)).astype(np.float32)
feat_dict['INQUIRY_PER_CREDIT'] = (tot_req / (cred / 100000.0 + 1.0)).astype(np.float32)
feat_dict['INQUIRY_ZERO_FLAG'] = (tot_req == 0).astype(np.float32)

emp_phone = df_all['FLAG_EMP_PHONE'].values.astype(np.float64)
work_phone = df_all['FLAG_WORK_PHONE'].values.astype(np.float64)
feat_dict['PHONE_WORK_EMP_MATCH'] = (emp_phone == work_phone).astype(np.float32)
feat_dict['PHONE_CONTACT_SCORE'] = (emp_phone + work_phone + df_all['FLAG_MOBIL'].values).astype(np.float32)

reg_rating = df_all['REGION_RATING_CLIENT'].fillna(2.0).values.astype(np.float64)
feat_dict['REGION_RATING_CREDIT'] = (cred / (reg_rating + 1.0)).astype(np.float32)
feat_dict['REGION_RATING_ANNUITY'] = (ann / (reg_rating + 1.0)).astype(np.float32)
feat_dict['REGION_POP_INCOME'] = (df_all['REGION_POPULATION_RELATIVE'].values * inc).astype(np.float32)

print("Engineering Group F: External Source Aggregations & Interactions...")
# Group F: External Source Aggregations
ext_stack = np.column_stack([e1, e2, e3])
ext_mean = np.nanmean(ext_stack, axis=1)
ext_std = np.nanstd(ext_stack, axis=1)
ext_min = np.nanmin(ext_stack, axis=1)
ext_max = np.nanmax(ext_stack, axis=1)
ext_prod = np.nanprod(ext_stack, axis=1)
ext_weighted = np.nan_to_num(0.2 * e1, nan=0.0) + np.nan_to_num(0.4 * e2, nan=0.0) + np.nan_to_num(0.4 * e3, nan=0.0)

feat_dict['EXT_MEAN'] = ext_mean.astype(np.float32)
feat_dict['EXT_STD'] = np.nan_to_num(ext_std, nan=0.0).astype(np.float32)
feat_dict['EXT_MIN'] = ext_min.astype(np.float32)
feat_dict['EXT_MAX'] = ext_max.astype(np.float32)
feat_dict['EXT_PROD'] = ext_prod.astype(np.float32)
feat_dict['EXT_WEIGHTED'] = ext_weighted.astype(np.float32)
feat_dict['EXT_SQ'] = (ext_mean ** 2).astype(np.float32)
feat_dict['EXT_CUBE'] = (ext_mean ** 3).astype(np.float32)

feat_dict['EXT_DIFF_12'] = (e1 - e2).astype(np.float32)
feat_dict['EXT_DIFF_23'] = (e2 - e3).astype(np.float32)
feat_dict['EXT_DIFF_13'] = (e1 - e3).astype(np.float32)

feat_dict['EXT_x_DTI'] = (ext_mean * dti).astype(np.float32)
feat_dict['EXT_x_SURPLUS'] = (ext_mean * feat_dict['SURPLUS_TO_INC']).astype(np.float32)
feat_dict['EXT_x_EAR'] = (ext_mean * ear).astype(np.float32)
feat_dict['EXT_x_LTV'] = (ext_mean * ltv).astype(np.float32)
feat_dict['EXT_x_DEBT_BURDEN'] = (ext_mean * debt_burden).astype(np.float32)

print("Engineering Group G: Chronological Applicant Timeline & Relational Memory...")
# Group G: Chronological Applicant Timeline
df_all['k_entity'] = (
    (df_all['DAYS_BIRTH'] - df_all['DAYS_REGISTRATION']).astype(int).astype(str) + '|' +
    (df_all['DAYS_BIRTH'] - df_all['DAYS_ID_PUBLISH']).astype(int).astype(str) + '|' +
    df_all['CODE_GENDER'].astype(str) + '|' +
    df_all['NAME_EDUCATION_TYPE'].astype(str)
)

entity_counts = df_all['k_entity'].value_counts()
feat_dict['N_APPS_ALL'] = df_all['k_entity'].map(entity_counts).values.astype(np.float32)

# Sort by entity and DAYS_REGISTRATION to get true chronological timeline
df_all['row_idx'] = np.arange(len(df_all))
df_all_sorted = df_all.sort_values(['k_entity', 'DAYS_REGISTRATION']).copy()

df_all_sorted['APP_ORDER'] = df_all_sorted.groupby('k_entity').cumcount()
df_all_sorted['DAYS_SINCE_PREV_APP'] = df_all_sorted.groupby('k_entity')['DAYS_REGISTRATION'].diff()
df_all_sorted['DAYS_TO_NEXT_APP'] = df_all_sorted.groupby('k_entity')['DAYS_REGISTRATION'].diff(-1)

df_all_restored = df_all_sorted.sort_values('row_idx')
feat_dict['APP_ORDER'] = df_all_restored['APP_ORDER'].values.astype(np.float32)
feat_dict['DAYS_SINCE_PREV_APP'] = df_all_restored['DAYS_SINCE_PREV_APP'].fillna(-999).values.astype(np.float32)
feat_dict['DAYS_TO_NEXT_APP'] = df_all_restored['DAYS_TO_NEXT_APP'].fillna(-999).values.astype(np.float32)

# Out-of-Fold Supervised Graph Memory (UIDTE)
uidy_n = np.zeros(len(df_all), dtype=np.float32)
uidy_sum = np.zeros(len(df_all), dtype=np.float32)
uidy_near_y = np.full(len(df_all), -1.0, dtype=np.float32)
uidy_near_d = np.full(len(df_all), 99999.0, dtype=np.float32)
uidy_anybad = np.zeros(len(df_all), dtype=np.float32)

# To strictly avoid target leakage, compute in-fold on 5-fold CV for train, and full train for test
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
y_all_train = y_train

for fold, (trn_idx, val_idx) in enumerate(skf.split(np.zeros(n_train), y_all_train)):
    trn_sub = df_all.iloc[trn_idx].copy()
    trn_sub['TARGET'] = y_all_train[trn_idx]
    
    val_sub = df_all.iloc[val_idx].copy()
    
    # Merge val with train on entity
    val_merged = val_sub[['row_idx', 'k_entity', 'DAYS_REGISTRATION']].merge(
        trn_sub[['k_entity', 'DAYS_REGISTRATION', 'TARGET']],
        on='k_entity',
        suffixes=('', '_partner')
    )
    if len(val_merged) > 0:
        val_merged['day_delta'] = (val_merged['DAYS_REGISTRATION'] - val_merged['DAYS_REGISTRATION_partner']).abs()
        nearest = val_merged.sort_values('day_delta').groupby('row_idx').first()
        summary = val_merged.groupby('row_idx').agg(
            n_prior=('TARGET', 'count'),
            sum_prior=('TARGET', 'sum')
        )
        for r_idx in nearest.index:
            uidy_near_y[r_idx] = nearest.loc[r_idx, 'TARGET']
            uidy_near_d[r_idx] = nearest.loc[r_idx, 'day_delta']
            uidy_n[r_idx] = summary.loc[r_idx, 'n_prior']
            uidy_sum[r_idx] = summary.loc[r_idx, 'sum_prior']
            uidy_anybad[r_idx] = 1.0 if summary.loc[r_idx, 'sum_prior'] > 0 else 0.0

# For test, use full train
trn_full = df_all.iloc[:n_train].copy()
trn_full['TARGET'] = y_all_train
test_sub = df_all.iloc[n_train:].copy()

test_merged = test_sub[['row_idx', 'k_entity', 'DAYS_REGISTRATION']].merge(
    trn_full[['k_entity', 'DAYS_REGISTRATION', 'TARGET']],
    on='k_entity',
    suffixes=('', '_partner')
)
if len(test_merged) > 0:
    test_merged['day_delta'] = (test_merged['DAYS_REGISTRATION'] - test_merged['DAYS_REGISTRATION_partner']).abs()
    nearest_te = test_merged.sort_values('day_delta').groupby('row_idx').first()
    summary_te = test_merged.groupby('row_idx').agg(
        n_prior=('TARGET', 'count'),
        sum_prior=('TARGET', 'sum')
    )
    for r_idx in nearest_te.index:
        uidy_near_y[r_idx] = nearest_te.loc[r_idx, 'TARGET']
        uidy_near_d[r_idx] = nearest_te.loc[r_idx, 'day_delta']
        uidy_n[r_idx] = summary_te.loc[r_idx, 'n_prior']
        uidy_sum[r_idx] = summary_te.loc[r_idx, 'sum_prior']
        uidy_anybad[r_idx] = 1.0 if summary_te.loc[r_idx, 'sum_prior'] > 0 else 0.0

feat_dict['UIDY_N'] = uidy_n
feat_dict['UIDY_SUM'] = uidy_sum
feat_dict['UIDY_NEAR_Y'] = uidy_near_y
feat_dict['UIDY_NEAR_D'] = uidy_near_d
feat_dict['UIDY_ANYBAD'] = uidy_anybad

print("Engineering Group H: High-Order Pairwise Interactions...")
# Top core drivers for exhaustive interaction generation
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
        
        # Product
        feat_dict[f'P_PROD_{col1}_x_{col2}'] = (v1 * v2).astype(np.float32)
        # Ratio
        feat_dict[f'P_RATIO_{col1}_div_{col2}'] = (v1 / (np.abs(v2) + 1.0)).astype(np.float32)
        # Difference
        feat_dict[f'P_DIFF_{col1}_sub_{col2}'] = (v1 - v2).astype(np.float32)
        interaction_count += 3

print(f"Engineered {interaction_count} pairwise interaction features across top 20 drivers!")

# Convert dictionary to DataFrame
df_features = pd.DataFrame(feat_dict)
print(f"Total features constructed: {df_features.shape[1]}")

# Split into train and test
X_tr_hasan = df_features.iloc[:n_train].reset_index(drop=True)
X_te_hasan = df_features.iloc[n_train:].reset_index(drop=True)

# Save to cache
os.makedirs(r'D:/gci/gemini/champion/phase3_reconstruction/independent_rebuild', exist_ok=True)
X_tr_hasan.to_feather(r'D:/gci/gemini/champion/phase3_reconstruction/independent_rebuild/hasan_features_tr.feather')
X_te_hasan.to_feather(r'D:/gci/gemini/champion/phase3_reconstruction/independent_rebuild/hasan_features_te.feather')

print(f"Saved Hasan Features successfully: Train={X_tr_hasan.shape}, Test={X_te_hasan.shape}")
print("Done!")
