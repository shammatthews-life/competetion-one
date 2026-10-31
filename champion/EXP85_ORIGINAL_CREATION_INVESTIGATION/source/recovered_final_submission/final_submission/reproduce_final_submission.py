import os
import sys
import json
import hashlib
import numpy as np
import pandas as pd
from scipy.stats import norm, rankdata
from scipy.optimize import minimize
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression
import lightgbm as lgb

output_dir = r"C:\Users\User\Documents\Personal Stuffs\GCI World\output"
downloads_dir = r"c:\Users\User\Downloads"

def resolve_path(filename: str) -> str:
    path_out = os.path.join(output_dir, filename)
    path_dl = os.path.join(downloads_dir, filename)
    path_dl_sub = os.path.join(downloads_dir, "output", filename)
    if os.path.exists(path_out):
        return path_out
    for candidate in [path_dl, path_dl_sub, filename]:
        if os.path.exists(candidate):
            return candidate
    return path_out

def to_average_uniform_rank_f64(values: np.ndarray) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float64).ravel()
    valid_mask = ~np.isnan(arr) & ~np.isinf(arr)
    ranked_output = np.full_like(arr, 0.5, dtype=np.float64)
    valid_count = int(valid_mask.sum())
    if valid_count > 0:
        ranked_output[valid_mask] = rankdata(arr[valid_mask], method="average").astype(np.float64) / (valid_count + 1.0)
    return ranked_output

def to_strict_uniform_rank_f64(primary_values: np.ndarray, tie_breaker_values: np.ndarray) -> np.ndarray:
    primary_arr = np.asarray(primary_values, dtype=np.float64).ravel()
    tie_breaker_arr = np.asarray(tie_breaker_values, dtype=np.float64).ravel()
    sample_count = primary_arr.shape[0]
    sort_order = np.lexsort((tie_breaker_arr, primary_arr))
    strict_ranks = np.empty(sample_count, dtype=np.float64)
    strict_ranks[sort_order] = np.arange(1, sample_count + 1, dtype=np.float64) / (sample_count + 1.0)
    return strict_ranks

def to_probit_f64(uniform_ranks: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    clipped_ranks = np.clip(np.asarray(uniform_ranks, dtype=np.float64), eps, 1.0 - eps)
    return norm.ppf(clipped_ranks).astype(np.float64)

def to_probit_blom_f64(probabilities: np.ndarray) -> np.ndarray:
    prob_arr = np.asarray(probabilities, dtype=np.float64).ravel()
    cleaned_probs = np.where(np.isnan(prob_arr), np.nanmedian(prob_arr), prob_arr)
    sample_count = len(cleaned_probs)
    return norm.ppf((rankdata(cleaned_probs, method="average") - 0.375) / (sample_count + 0.25))

def to_logit_f64(probabilities: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    clipped_probs = np.clip(np.asarray(probabilities, dtype=np.float64), eps, 1.0 - eps)
    log_odds = np.log(clipped_probs / (1.0 - clipped_probs))
    return (log_odds - np.mean(log_odds)) / (np.std(log_odds) + 1e-12)

def quantile_matched_logit(oof_ranks: np.ndarray, test_probabilities: np.ndarray) -> tuple:
    logit_test = to_logit_f64(test_probabilities)
    uniform_oof = to_average_uniform_rank_f64(oof_ranks)
    sorted_logit_test = np.sort(logit_test)
    quantile_grid = (np.arange(len(sorted_logit_test), dtype=np.float64) + 1.0) / (len(sorted_logit_test) + 1.0)
    logit_train = np.interp(uniform_oof, quantile_grid, sorted_logit_test)
    logit_train = (logit_train - np.mean(logit_train)) / (np.std(logit_train) + 1e-12)
    return logit_train, logit_test

def dual_test_probit_equal(rank_values: np.ndarray, prob_values: np.ndarray) -> np.ndarray:
    uniform_rank = to_average_uniform_rank_f64(rank_values)
    uniform_prob = to_average_uniform_rank_f64(prob_values)
    blended_uniform = to_average_uniform_rank_f64(0.50 * uniform_rank + 0.50 * uniform_prob)
    return to_probit_f64(blended_uniform)

def dual_probit_pair(oof_values: np.ndarray, test_rank_values: np.ndarray, test_prob_values: np.ndarray) -> tuple:
    probit_train = to_probit_f64(to_average_uniform_rank_f64(oof_values))
    probit_test_rank = to_probit_f64(to_average_uniform_rank_f64(test_rank_values))
    probit_test_prob = to_probit_f64(to_average_uniform_rank_f64(test_prob_values))
    probit_test = to_probit_f64(to_average_uniform_rank_f64(0.55 * probit_test_rank + 0.45 * probit_test_prob))
    return probit_train, probit_test

def continuous_probit_power_blend(
    uniform_matrix: np.ndarray,
    probit_matrix: np.ndarray,
    portfolio_weights: np.ndarray,
    beta_probit: float = 0.85,
    power_exponent: float = 1.65
) -> np.ndarray:
    weights = np.asarray(portfolio_weights, dtype=np.float64).ravel()
    weights = np.maximum(0.0, weights)
    weights /= np.sum(weights)
    clipped_uniform = np.clip(np.asarray(uniform_matrix, dtype=np.float64), 1e-6, 1.0 - 1e-6)
    linear_probit = probit_matrix @ weights
    standardized_linear = (linear_probit - np.mean(linear_probit)) / (np.std(linear_probit) + 1e-12)
    power_combination = np.power(clipped_uniform, power_exponent) @ weights
    power_probit = to_probit_f64(to_average_uniform_rank_f64(power_combination))
    standardized_power = (power_probit - np.mean(power_probit)) / (np.std(power_probit) + 1e-12)
    return (beta_probit * standardized_linear + (1.0 - beta_probit) * standardized_power).astype(np.float64)

def compute_hierarchical_macro_synthesis(y_true: np.ndarray) -> tuple:
    with open(resolve_path("meta.json"), "r", encoding="utf-8") as f:
        meta_config = json.load(f)
    relational_oof_matrix = np.load(resolve_path("group_a_oof_10stream.npy"))
    relational_test_matrix = np.load(resolve_path("group_a_test_10stream.npy"))
    stream_order = meta_config["order10"]
    stream_weights_map = meta_config["weights"]
    prior_weights = np.array([stream_weights_map[name] for name in stream_order], dtype=np.float64)
    prior_weights /= np.sum(prior_weights)

    uniform_rel_oof = np.column_stack([to_average_uniform_rank_f64(relational_oof_matrix[:, j]) for j in range(10)])
    uniform_rel_test = np.column_stack([to_average_uniform_rank_f64(relational_test_matrix[:, j]) for j in range(10)])
    probit_rel_oof = to_probit_f64(uniform_rel_oof)
    probit_rel_test = to_probit_f64(uniform_rel_test)

    xgb_indices = [3, 4, 5, 8, 9]
    lgb_cb_indices = [0, 1, 2, 6, 7]
    weights_xgb = prior_weights[xgb_indices] / np.sum(prior_weights[xgb_indices])
    weights_lgb_cb = prior_weights[lgb_cb_indices] / np.sum(prior_weights[lgb_cb_indices])

    z_rel_xgb_oof = to_probit_f64(to_average_uniform_rank_f64(continuous_probit_power_blend(uniform_rel_oof[:, xgb_indices], probit_rel_oof[:, xgb_indices], weights_xgb, beta_probit=0.70, power_exponent=1.70)))
    z_rel_xgb_test = to_probit_f64(to_average_uniform_rank_f64(continuous_probit_power_blend(uniform_rel_test[:, xgb_indices], probit_rel_test[:, xgb_indices], weights_xgb, beta_probit=0.70, power_exponent=1.70)))
    u_rel_xgb_oof = to_average_uniform_rank_f64(z_rel_xgb_oof)
    u_rel_xgb_test = to_average_uniform_rank_f64(z_rel_xgb_test)

    z_rel_lgb_oof = to_probit_f64(to_average_uniform_rank_f64(continuous_probit_power_blend(uniform_rel_oof[:, lgb_cb_indices], probit_rel_oof[:, lgb_cb_indices], weights_lgb_cb, beta_probit=0.70, power_exponent=1.70)))
    z_rel_lgb_test = to_probit_f64(to_average_uniform_rank_f64(continuous_probit_power_blend(uniform_rel_test[:, lgb_cb_indices], probit_rel_test[:, lgb_cb_indices], weights_lgb_cb, beta_probit=0.70, power_exponent=1.70)))
    u_rel_lgb_oof = to_average_uniform_rank_f64(z_rel_lgb_oof)
    u_rel_lgb_test = to_average_uniform_rank_f64(z_rel_lgb_test)

    data_tabm463 = np.load(resolve_path("oof_and_test_tabm32_463_dae.npz"), allow_pickle=True)
    z_tabm463_oof = to_probit_f64(to_average_uniform_rank_f64(data_tabm463["oof_tabm463_rank"]))
    u_tabm463_oof = to_average_uniform_rank_f64(z_tabm463_oof)
    u_tabm463_test = to_strict_uniform_rank_f64(data_tabm463["test_tabm463_rank"], tie_breaker_values=data_tabm463["test_tabm463_prob"])
    z_tabm463_test = to_probit_f64(u_tabm463_test)

    data_lgb_dae = np.load(resolve_path("oof_and_test_lgbm_dae491.npz"), allow_pickle=True)
    z_lgb_dae_oof = to_probit_f64(to_average_uniform_rank_f64(data_lgb_dae["oof_lgb_dae_rank"]))
    u_lgb_dae_oof = to_average_uniform_rank_f64(z_lgb_dae_oof)
    u_lgb_dae_test = to_strict_uniform_rank_f64(data_lgb_dae["test_lgb_dae_rank"], tie_breaker_values=data_lgb_dae["test_lgb_dae_prob"])
    z_lgb_dae_test = to_probit_f64(u_lgb_dae_test)

    data_resnet = np.load(resolve_path("oof_and_test_tabular_neural_net.npz"), allow_pickle=True)
    z_resnet_oof = to_probit_f64(to_average_uniform_rank_f64(data_resnet["oof_nn_rank"]))
    u_resnet_test = to_strict_uniform_rank_f64(data_resnet["test_nn"], tie_breaker_values=data_resnet["test_nn_prob"])
    z_resnet_test = to_probit_f64(u_resnet_test)

    tabm_tier2_path = resolve_path("oof_and_test_tabm_32heads.npz")
    if os.path.exists(tabm_tier2_path):
        data_tabm226 = np.load(tabm_tier2_path, allow_pickle=True)
        z_tabm226_oof = to_probit_f64(to_average_uniform_rank_f64(data_tabm226["oof_tabm_rank"]))
        u_tabm226_test = to_strict_uniform_rank_f64(data_tabm226["test_tabm_rank"], tie_breaker_values=data_tabm226["test_nn_prob"])
        z_tabm226_test = to_probit_f64(u_tabm226_test)
        z_tier2_nn_oof_raw = 0.72 * z_tabm226_oof + 0.28 * z_resnet_oof
        z_tier2_nn_test_raw = 0.72 * z_tabm226_test + 0.28 * z_resnet_test
    else:
        z_tier2_nn_oof_raw = z_resnet_oof
        z_tier2_nn_test_raw = z_resnet_test
    u_tier2_nn_oof = to_average_uniform_rank_f64(z_tier2_nn_oof_raw)
    u_tier2_nn_test = to_average_uniform_rank_f64(z_tier2_nn_test_raw)
    z_tier2_nn_oof = to_probit_f64(u_tier2_nn_oof)
    z_tier2_nn_test = to_probit_f64(u_tier2_nn_test)

    data_buhlmann = np.load(resolve_path("oof_and_test_buhlmann_tier2_v21_models.npz"), allow_pickle=True)
    path_cloud8 = resolve_path("oof_and_test_cloud_burst_zoo_v21_8seed.npz")
    path_acer10 = resolve_path("oof_and_test_laptop_acer_zoo_10seed.npz")
    path_cloud5 = resolve_path("oof_and_test_cloud_burst_5seed.npz")
    if os.path.exists(path_cloud8) and os.path.exists(path_acer10) and os.path.exists(path_cloud5):
        data_cloud8 = np.load(path_cloud8, allow_pickle=True)
        data_acer10 = np.load(path_acer10, allow_pickle=True)
        data_cloud5 = np.load(path_cloud5, allow_pickle=True)
        buhlmann_weights = np.array([0.5043, 0.0793, 0.0334, 0.1269, 0.0661, 0.0416, 0.1382, 0.0103], dtype=np.float64)
        gbdt_tier2_oof = np.column_stack([
            to_probit_blom_f64(data_buhlmann["oof_lgb_cons"]),
            to_probit_blom_f64(data_buhlmann["oof_lgb_et"]),
            to_probit_blom_f64(data_buhlmann["oof_xgb"]),
            to_probit_blom_f64(data_buhlmann["oof_cat"]),
            to_probit_blom_f64(data_cloud8["oof_cat"]),
            to_probit_blom_f64(data_acer10["oof_cat"]),
            to_probit_blom_f64(data_cloud5["oof_xgb_cuda"]),
            to_probit_blom_f64(data_cloud5["oof_cb_lossguide"])
        ]) @ buhlmann_weights
        gbdt_tier2_test = np.column_stack([
            to_probit_blom_f64(data_buhlmann["test_lgb_cons"]),
            to_probit_blom_f64(data_buhlmann["test_lgb_et"]),
            to_probit_blom_f64(data_buhlmann["test_xgb"]),
            to_probit_blom_f64(data_buhlmann["test_cat"]),
            to_probit_blom_f64(data_cloud8["test_cat"]),
            to_probit_blom_f64(data_acer10["test_cat"]),
            to_probit_blom_f64(data_cloud5["test_xgb_cuda"]),
            to_probit_blom_f64(data_cloud5["test_cb_lossguide"])
        ]) @ buhlmann_weights
    else:
        buhlmann_weights = np.array([0.5254, 0.1165, 0.1589, 0.1993], dtype=np.float64)
        gbdt_tier2_oof = np.column_stack([
            to_probit_blom_f64(data_buhlmann["oof_lgb_cons"]),
            to_probit_blom_f64(data_buhlmann["oof_lgb_et"]),
            to_probit_blom_f64(data_buhlmann["oof_xgb"]),
            to_probit_blom_f64(data_buhlmann["oof_cat"])
        ]) @ buhlmann_weights
        gbdt_tier2_test = np.column_stack([
            to_probit_blom_f64(data_buhlmann["test_lgb_cons"]),
            to_probit_blom_f64(data_buhlmann["test_lgb_et"]),
            to_probit_blom_f64(data_buhlmann["test_xgb"]),
            to_probit_blom_f64(data_buhlmann["test_cat"])
        ]) @ buhlmann_weights

    z_gbdt_tier2_oof = to_probit_f64(to_average_uniform_rank_f64(gbdt_tier2_oof))
    u_gbdt_tier2_oof = to_average_uniform_rank_f64(z_gbdt_tier2_oof)
    u_gbdt_tier2_test = to_strict_uniform_rank_f64(gbdt_tier2_test, tie_breaker_values=gbdt_tier2_test)
    z_gbdt_tier2_test = to_probit_f64(u_gbdt_tier2_test)

    data_grand595 = np.load(resolve_path("oof_and_test_grand_synthesis_595.npz"), allow_pickle=True)
    z_grand595_oof = to_probit_f64(to_average_uniform_rank_f64(data_grand595["oof_grand_rank"]))
    u_grand595_oof = to_average_uniform_rank_f64(z_grand595_oof)
    u_grand595_test = to_strict_uniform_rank_f64(data_grand595["test_grand_rank"], tie_breaker_values=data_grand595["test_grand_prob"])
    z_grand595_test = to_probit_f64(u_grand595_test)

    data_neural646 = np.load(resolve_path("oof_and_test_neural_grand_646.npz"), allow_pickle=True)
    z_neural646_oof = to_probit_f64(to_average_uniform_rank_f64(data_neural646["oof_neural646_rank"]))
    u_neural646_oof = to_average_uniform_rank_f64(z_neural646_oof)
    u_neural646_test = to_strict_uniform_rank_f64(data_neural646["test_neural646_rank"], tie_breaker_values=data_neural646["test_neural646_prob"])
    z_neural646_test = to_probit_f64(u_neural646_test)

    data_grand646 = np.load(resolve_path("oof_and_test_grand_synthesis_646.npz"), allow_pickle=True)
    z_grand646_oof = to_probit_f64(to_average_uniform_rank_f64(data_grand646["oof_grand_rank"]))
    u_grand646_oof = to_average_uniform_rank_f64(z_grand646_oof)
    u_grand646_test = to_strict_uniform_rank_f64(data_grand646["test_grand_rank"], tie_breaker_values=data_grand646["test_grand_prob"])
    z_grand646_test = to_probit_f64(u_grand646_test)

    z_macro_oof = np.column_stack([
        z_rel_xgb_oof, z_rel_lgb_oof, z_tabm463_oof, z_lgb_dae_oof,
        z_tier2_nn_oof, z_gbdt_tier2_oof, z_grand595_oof, z_neural646_oof, z_grand646_oof
    ])
    z_macro_test = np.column_stack([
        z_rel_xgb_test, z_rel_lgb_test, z_tabm463_test, z_lgb_dae_test,
        z_tier2_nn_test, z_gbdt_tier2_test, z_grand595_test, z_neural646_test, z_grand646_test
    ])
    u_macro_oof = np.column_stack([
        u_rel_xgb_oof, u_rel_lgb_oof, u_tabm463_oof, u_lgb_dae_oof,
        u_tier2_nn_oof, u_gbdt_tier2_oof, u_grand595_oof, u_neural646_oof, u_grand646_oof
    ])
    u_macro_test = np.column_stack([
        u_rel_xgb_test, u_rel_lgb_test, u_tabm463_test, u_lgb_dae_test,
        u_tier2_nn_test, u_gbdt_tier2_test, u_grand595_test, u_neural646_test, u_grand646_test
    ])

    pillar_count = z_macro_oof.shape[1]
    fold_bonuses = np.array([0.0014, 0.0012, 0.0003, 0.0003, 0.0002, 0.0002, 0.0004, 0.0004, 0.0005], dtype=np.float64)
    aucs_oof = np.array([roc_auc_score(y_true, z_macro_oof[:, j]) for j in range(pillar_count)], dtype=np.float64)
    aucs_adjusted = aucs_oof + fold_bonuses
    discriminative_signal = np.sqrt(2.0) * norm.ppf(np.clip(aucs_adjusted, 0.501, 0.999))
    covariance_test = np.cov(z_macro_test, rowvar=False) + np.eye(pillar_count) * 0.015
    weights_markowitz = np.maximum(0.015, np.linalg.solve(covariance_test, discriminative_signal))
    weights_markowitz /= np.sum(weights_markowitz)

    structural_prior = np.array([0.020, 0.010, 0.060, 0.015, 0.015, 0.030, 0.180, 0.270, 0.400], dtype=np.float64)
    structural_prior /= np.sum(structural_prior)
    weights_combined = 0.50 * structural_prior + 0.50 * weights_markowitz
    weights_combined /= np.sum(weights_combined)

    z_hier_oof = continuous_probit_power_blend(u_macro_oof, z_macro_oof, weights_combined, beta_probit=0.85, power_exponent=1.65)
    z_hier_test = continuous_probit_power_blend(u_macro_test, z_macro_test, weights_combined, beta_probit=0.85, power_exponent=1.65)
    return z_hier_oof, z_hier_test

def compute_unbundled_tri_space_synthesis(y_true: np.ndarray) -> tuple:
    with open(resolve_path("meta.json"), "r", encoding="utf-8") as f:
        meta_config = json.load(f)
    relational_oof_matrix = np.load(resolve_path("group_a_oof_10stream.npy"))
    relational_test_matrix = np.load(resolve_path("group_a_test_10stream.npy"))
    stream_order = meta_config["order10"]
    stream_weights_map = meta_config["weights"]
    prior_weights = np.array([stream_weights_map[name] for name in stream_order], dtype=np.float64)
    prior_weights /= np.sum(prior_weights)

    uniform_rel_oof = np.column_stack([to_average_uniform_rank_f64(relational_oof_matrix[:, j]) for j in range(10)])
    uniform_rel_test = np.column_stack([to_average_uniform_rank_f64(relational_test_matrix[:, j]) for j in range(10)])
    probit_rel_oof = to_probit_f64(uniform_rel_oof)
    probit_rel_test = to_probit_f64(uniform_rel_test)

    xgb_indices = [3, 4, 5, 8, 9]
    lgb_cb_indices = [0, 1, 2, 6, 7]
    weights_xgb = prior_weights[xgb_indices] / np.sum(prior_weights[xgb_indices])
    weights_lgb_cb = prior_weights[lgb_cb_indices] / np.sum(prior_weights[lgb_cb_indices])

    z_rel_xgb_oof = to_probit_f64(to_average_uniform_rank_f64(continuous_probit_power_blend(uniform_rel_oof[:, xgb_indices], probit_rel_oof[:, xgb_indices], weights_xgb, beta_probit=0.70, power_exponent=1.70)))
    z_rel_xgb_test = to_probit_f64(to_average_uniform_rank_f64(continuous_probit_power_blend(uniform_rel_test[:, xgb_indices], probit_rel_test[:, xgb_indices], weights_xgb, beta_probit=0.70, power_exponent=1.70)))

    z_rel_lgb_oof = to_probit_f64(to_average_uniform_rank_f64(continuous_probit_power_blend(uniform_rel_oof[:, lgb_cb_indices], probit_rel_oof[:, lgb_cb_indices], weights_lgb_cb, beta_probit=0.70, power_exponent=1.70)))
    z_rel_lgb_test = to_probit_f64(to_average_uniform_rank_f64(continuous_probit_power_blend(uniform_rel_test[:, lgb_cb_indices], probit_rel_test[:, lgb_cb_indices], weights_lgb_cb, beta_probit=0.70, power_exponent=1.70)))

    data_lgb_dae = np.load(resolve_path("oof_and_test_lgbm_dae491.npz"), allow_pickle=True)
    z_lgb_dae_oof = to_probit_f64(to_average_uniform_rank_f64(data_lgb_dae["oof_lgb_dae_rank"]))
    z_lgb_dae_test = dual_test_probit_equal(data_lgb_dae["test_lgb_dae_rank"], data_lgb_dae["test_lgb_dae_prob"])

    data_neural646 = np.load(resolve_path("oof_and_test_neural_grand_646.npz"), allow_pickle=True)
    data_grand646 = np.load(resolve_path("oof_and_test_grand_synthesis_646.npz"), allow_pickle=True)
    data_grand595 = np.load(resolve_path("oof_and_test_grand_synthesis_595.npz"), allow_pickle=True)

    stream_specifications = [
        ("TabM32_646_Probit", to_probit_f64(to_average_uniform_rank_f64(data_neural646["oof_tabm646_rank"])), dual_test_probit_equal(data_neural646["test_tabm646_rank"], data_neural646["test_tabm646_prob"]), 0.0004, 0.050, 0.320),
        ("TabM32_646_Logit",  to_logit_f64(data_neural646["oof_tabm646_prob"]),                               to_logit_f64(data_neural646["test_tabm646_prob"]),                                                 0.0004, 0.040, 0.220),
        ("ResNet_646_Probit", to_probit_f64(to_average_uniform_rank_f64(data_neural646["oof_resnet646_rank"])), dual_test_probit_equal(data_neural646["test_resnet646_rank"], data_neural646["test_resnet646_prob"]), 0.0004, 0.025, 0.120),
        ("ResNet_646_Logit",  to_logit_f64(data_neural646["oof_resnet646_prob"]),                             to_logit_f64(data_neural646["test_resnet646_prob"]),                                               0.0004, 0.005, 0.040),
        ("LGB_Cons_646",      to_probit_f64(to_average_uniform_rank_f64(data_grand646["oof_lgb_cons"])),      dual_test_probit_equal(data_grand646["test_lgb_cons"], data_grand646["test_lgb_cons_prob"]),       0.0005, 0.060, 0.220),
        ("LGB_Deep_646",      to_probit_f64(to_average_uniform_rank_f64(data_grand646["oof_lgb_deep"])),      dual_test_probit_equal(data_grand646["test_lgb_deep"], data_grand646["test_lgb_deep_prob"]),       0.0005, 0.020, 0.100),
        ("XGB_D6_646",        to_probit_f64(to_average_uniform_rank_f64(data_grand646["oof_xgb_d6"])),        dual_test_probit_equal(data_grand646["test_xgb_d6"], data_grand646["test_xgb_d6_prob"]),           0.0005, 0.035, 0.140),
        ("XGB_D4_646",        to_probit_f64(to_average_uniform_rank_f64(data_grand646["oof_xgb_d4"])),        dual_test_probit_equal(data_grand646["test_xgb_d4"], data_grand646["test_xgb_d4_prob"]),           0.0005, 0.035, 0.140),
        ("CB_Sym_646",        to_probit_f64(to_average_uniform_rank_f64(data_grand646["oof_cb_sym"])),        dual_test_probit_equal(data_grand646["test_cb_sym"], data_grand646["test_cb_sym_prob"]),           0.0005, 0.035, 0.140),
        ("LGB_Cons_595",      to_probit_f64(to_average_uniform_rank_f64(data_grand595["oof_lgb_cons"])),      dual_test_probit_equal(data_grand595["test_lgb_cons"], data_grand595["test_lgb_cons_prob"]),       0.0004, 0.025, 0.120),
        ("LGB_Deep_595",      to_probit_f64(to_average_uniform_rank_f64(data_grand595["oof_lgb_deep"])),      dual_test_probit_equal(data_grand595["test_lgb_deep"], data_grand595["test_lgb_deep_prob"]),       0.0004, 0.010, 0.080),
        ("XGB_D6_595",        to_probit_f64(to_average_uniform_rank_f64(data_grand595["oof_xgb_d6"])),        dual_test_probit_equal(data_grand595["test_xgb_d6"], data_grand595["test_xgb_d6_prob"]),           0.0004, 0.020, 0.100),
        ("CB_Sym_595",        to_probit_f64(to_average_uniform_rank_f64(data_grand595["oof_cb_sym"])),        dual_test_probit_equal(data_grand595["test_cb_sym"], data_grand595["test_cb_sym_prob"]),           0.0004, 0.025, 0.120),
        ("LGBM_DAE_491",      z_lgb_dae_oof,                                                                  z_lgb_dae_test,                                                                                    0.0003, 0.005, 0.025),
        ("Rel_XGB5",          z_rel_xgb_oof,                                                                  z_rel_xgb_test,                                                                                    0.0014, 0.005, 0.025),
        ("Rel_LGB_CB5",       z_rel_lgb_oof,                                                                  z_rel_lgb_test,                                                                                    0.0012, 0.005, 0.025),
    ]

    stream_count = len(stream_specifications)
    z_streams_oof = np.column_stack([spec[1] for spec in stream_specifications])
    z_streams_test = np.column_stack([spec[2] for spec in stream_specifications])
    u_streams_oof = np.column_stack([to_average_uniform_rank_f64(z_streams_oof[:, j]) for j in range(stream_count)])
    u_streams_test = np.column_stack([to_average_uniform_rank_f64(z_streams_test[:, j]) for j in range(stream_count)])
    stream_bonuses = np.array([spec[3] for spec in stream_specifications], dtype=np.float64)
    weight_bounds = [(spec[4], spec[5]) for spec in stream_specifications]

    aucs_oof = np.array([roc_auc_score(y_true, z_streams_oof[:, j]) for j in range(stream_count)], dtype=np.float64)
    aucs_adjusted = aucs_oof + stream_bonuses
    discriminative_signal = np.sqrt(2.0) * norm.ppf(np.clip(aucs_adjusted, 0.501, 0.999))
    covariance_test = np.cov(z_streams_test, rowvar=False) + np.eye(stream_count) * 0.025
    weights_markowitz_raw = np.linalg.solve(covariance_test, discriminative_signal)
    weights_markowitz = np.clip(weights_markowitz_raw, [b[0] for b in weight_bounds], [b[1] for b in weight_bounds])
    weights_markowitz /= np.sum(weights_markowitz)

    skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
    fold_weights_list = []

    for _, (train_idx, val_idx) in enumerate(skf.split(z_streams_oof, y_true)):
        z_fold_train = z_streams_oof[train_idx]
        y_fold_train = y_true[train_idx]
        cov_fold_train = np.cov(z_fold_train, rowvar=False) + np.eye(stream_count) * 0.020
        aucs_fold_train = np.array([roc_auc_score(y_fold_train, z_fold_train[:, j]) for j in range(stream_count)], dtype=np.float64)
        signal_fold_train = np.sqrt(2.0) * norm.ppf(np.clip(aucs_fold_train, 0.501, 0.999))

        def portfolio_objective(weights):
            normalized_w = weights / np.sum(weights)
            portfolio_return = np.dot(normalized_w, signal_fold_train)
            portfolio_variance = np.dot(normalized_w, np.dot(cov_fold_train, normalized_w))
            sharpe_ratio = portfolio_return / np.sqrt(2.0 * portfolio_variance + 1e-12)
            regularization_penalty = 0.25 * np.sum((normalized_w - weights_markowitz) ** 2)
            return -sharpe_ratio + regularization_penalty

        opt_result = minimize(
            portfolio_objective,
            weights_markowitz,
            method="SLSQP",
            bounds=weight_bounds,
            constraints={"type": "eq", "fun": lambda w: np.sum(w) - 1.0},
            options={"maxiter": 200, "ftol": 1e-9}
        )
        fold_w = np.clip(opt_result.x, [b[0] for b in weight_bounds], [b[1] for b in weight_bounds])
        fold_w /= np.sum(fold_w)
        fold_weights_list.append(fold_w)

    weights_nested_mean = np.mean(fold_weights_list, axis=0)
    weights_nested_mean /= np.sum(weights_nested_mean)

    weights_final = 0.50 * weights_markowitz + 0.50 * weights_nested_mean
    weights_final = np.clip(weights_final, [b[0] for b in weight_bounds], [b[1] for b in weight_bounds])
    weights_final /= np.sum(weights_final)

    z_unbundled_oof = continuous_probit_power_blend(u_streams_oof, z_streams_oof, weights_final, beta_probit=0.88, power_exponent=1.60)
    z_unbundled_test = continuous_probit_power_blend(u_streams_test, z_streams_test, weights_final, beta_probit=0.88, power_exponent=1.60)
    return z_unbundled_oof, z_unbundled_test

def main():
    print("=" * 90)
    print("HOME CREDIT DEFAULT RISK: MULTI-MANIFOLD META-STACKING & TRANSDUCTIVE GRAPH SYNTHESIS")
    print("=" * 90)

    y_true = np.load(resolve_path("y.npy")).astype(np.int64).ravel()
    test_ids = np.load(resolve_path("test_ids.npy")).astype(np.int64).ravel()
    n_train = len(y_true)
    n_test = len(test_ids)

    tier2_archive = np.load(resolve_path("matrix_226_features_buhlmann_tier2.npz"), allow_pickle=True)
    X_features_train = np.asarray(tier2_archive["X_tr"], dtype=np.float32)
    X_features_test = np.asarray(tier2_archive["X_te"], dtype=np.float32)
    X_features_all = np.vstack([X_features_train, X_features_test])
    n_total = X_features_all.shape[0]
    feature_names = [str(col) for col in tier2_archive["feature_names"]]
    feature_index_map = {col: idx for idx, col in enumerate(feature_names)}

    days_birth = np.nan_to_num(X_features_all[:, feature_index_map["DAYS_BIRTH"]].astype(np.float64), nan=0.0).round(0)
    days_registration = np.nan_to_num(X_features_all[:, feature_index_map["DAYS_REGISTRATION"]].astype(np.float64), nan=0.0).round(0)
    days_id_publish = np.nan_to_num(X_features_all[:, feature_index_map["DAYS_ID_PUBLISH"]].astype(np.float64), nan=0.0).round(0)
    days_employed = np.nan_to_num(X_features_all[:, feature_index_map["DAYS_EMPLOYED"]].astype(np.float64), nan=365243.0).round(0)
    days_phone_change = np.nan_to_num(X_features_all[:, feature_index_map["DAYS_LAST_PHONE_CHANGE"]].astype(np.float64), nan=0.0).round(0)
    code_gender = np.nan_to_num(X_features_all[:, feature_index_map["CODE_GENDER"]].astype(np.float64), nan=-1.0).round(0).astype(int)
    education_type = np.nan_to_num(X_features_all[:, feature_index_map["NAME_EDUCATION_TYPE"]].astype(np.float64), nan=-1.0).round(0).astype(int)

    entity_df = pd.DataFrame({
        "row": np.arange(n_total, dtype=np.int32),
        "DAYS_BIRTH": days_birth,
        "DAYS_EMPLOYED": days_employed,
        "DAYS_LAST_PHONE_CHANGE": days_phone_change,
        "CODE_GENDER": code_gender,
        "NAME_EDUCATION_TYPE": education_type
    })
    entity_df["entity_key"] = (
        pd.Series(days_birth - days_registration).astype(int).astype(str) + "|" +
        pd.Series(days_birth - days_id_publish).astype(int).astype(str) + "|" +
        pd.Series(code_gender).astype(str) + "|" +
        pd.Series(education_type).astype(str)
    )
    key_counts = entity_df["entity_key"].value_counts()
    multi_entity_df = entity_df[entity_df["entity_key"].map(key_counts) >= 2]
    entity_pairs = multi_entity_df.merge(multi_entity_df, on="entity_key", suffixes=("", "_partner"))
    entity_pairs = entity_pairs[entity_pairs["row"] != entity_pairs["row_partner"]].copy()

    birth_diff = entity_pairs["DAYS_BIRTH"] - entity_pairs["DAYS_BIRTH_partner"]
    employment_match = (
        ((entity_pairs["DAYS_EMPLOYED"] == 365243) & (entity_pairs["DAYS_EMPLOYED_partner"] == 365243)) |
        ((entity_pairs["DAYS_EMPLOYED"] - entity_pairs["DAYS_EMPLOYED_partner"] == birth_diff) & (entity_pairs["DAYS_EMPLOYED"] != 365243) & (entity_pairs["DAYS_EMPLOYED_partner"] != 365243))
    )
    phone_match = (
        (entity_pairs["DAYS_LAST_PHONE_CHANGE"] - entity_pairs["DAYS_LAST_PHONE_CHANGE_partner"] == birth_diff) |
        (entity_pairs["DAYS_LAST_PHONE_CHANGE"] == entity_pairs["DAYS_LAST_PHONE_CHANGE_partner"])
    )
    entity_pairs["birth_delta"] = birth_diff
    entity_pairs["abs_birth_delta"] = birth_diff.abs()
    entity_pairs["confidence_tier"] = np.where((birth_diff == 0) | (employment_match & phone_match), 2, np.where(employment_match | phone_match, 1, 0))

    confirmed_pairs = entity_pairs[entity_pairs["confidence_tier"] == 2][["row", "row_partner", "birth_delta", "abs_birth_delta"]].reset_index(drop=True)
    soft_tier1_pairs = entity_pairs[entity_pairs["confidence_tier"] == 1][["row", "row_partner", "birth_delta", "abs_birth_delta"]].reset_index(drop=True)

    z_hier_oof, z_hier_test = compute_hierarchical_macro_synthesis(y_true)
    z_unbundled_oof, z_unbundled_test = compute_unbundled_tri_space_synthesis(y_true)

    probit_hier_train = to_probit_f64(to_average_uniform_rank_f64(z_hier_oof))
    probit_hier_test = to_probit_f64(to_average_uniform_rank_f64(z_hier_test))
    probit_unbundled_train = to_probit_f64(to_average_uniform_rank_f64(z_unbundled_oof))
    probit_unbundled_test = to_probit_f64(to_average_uniform_rank_f64(z_unbundled_test))

    composite_linear_train = to_probit_f64(to_average_uniform_rank_f64(0.70 * probit_unbundled_train + 0.30 * probit_hier_train))
    composite_linear_test = to_probit_f64(to_average_uniform_rank_f64(0.70 * probit_unbundled_test + 0.30 * probit_hier_test))

    data_neural646 = np.load(resolve_path("oof_and_test_neural_grand_646.npz"), allow_pickle=True)
    data_grand646 = np.load(resolve_path("oof_and_test_grand_synthesis_646.npz"), allow_pickle=True)
    data_grand595 = np.load(resolve_path("oof_and_test_grand_synthesis_595.npz"), allow_pickle=True)
    data_lgb_dae = np.load(resolve_path("oof_and_test_lgbm_dae491.npz"), allow_pickle=True)
    data_tabm463 = np.load(resolve_path("oof_and_test_tabm32_463_dae.npz"), allow_pickle=True)
    data_hetero_zoo = np.load(resolve_path("oof_and_test_v26_heterogeneous_zoo_9stream.npz"), allow_pickle=True)
    data_buhlmann = np.load(resolve_path("oof_and_test_buhlmann_tier2_v21_models.npz"), allow_pickle=True)

    z_neural646_train = to_probit_f64(to_average_uniform_rank_f64(data_neural646["oof_neural646_rank"]))
    z_neural646_test = to_probit_f64(to_average_uniform_rank_f64(data_neural646["test_neural646_rank"]))
    z_grand646_train = to_probit_f64(to_average_uniform_rank_f64(data_grand646["oof_grand_rank"]))
    z_grand646_test = to_probit_f64(to_average_uniform_rank_f64(data_grand646["test_grand_rank"]))
    z_grand595_train = to_probit_f64(to_average_uniform_rank_f64(data_grand595["oof_grand_rank"]))
    z_grand595_test = to_probit_f64(to_average_uniform_rank_f64(data_grand595["test_grand_rank"]))

    z_tabm_train, z_tabm_test = dual_probit_pair(data_neural646["oof_tabm646_rank"], data_neural646["test_tabm646_rank"], data_neural646["test_tabm646_prob"])
    logit_tabm_train, logit_tabm_test = to_logit_f64(data_neural646["oof_tabm646_prob"]), to_logit_f64(data_neural646["test_tabm646_prob"])
    z_resnet_train, z_resnet_test = dual_probit_pair(data_neural646["oof_resnet646_rank"], data_neural646["test_resnet646_rank"], data_neural646["test_resnet646_prob"])
    logit_resnet_train, logit_resnet_test = to_logit_f64(data_neural646["oof_resnet646_prob"]), to_logit_f64(data_neural646["test_resnet646_prob"])

    z_lgb_cons6_train, z_lgb_cons6_test = dual_probit_pair(data_grand646["oof_lgb_cons"], data_grand646["test_lgb_cons"], data_grand646["test_lgb_cons_prob"])
    logit_lgb_cons6_train, logit_lgb_cons6_test = quantile_matched_logit(data_grand646["oof_lgb_cons"], data_grand646["test_lgb_cons_prob"])
    z_lgb_deep6_train, z_lgb_deep6_test = dual_probit_pair(data_grand646["oof_lgb_deep"], data_grand646["test_lgb_deep"], data_grand646["test_lgb_deep_prob"])
    logit_lgb_deep6_train, logit_lgb_deep6_test = quantile_matched_logit(data_grand646["oof_lgb_deep"], data_grand646["test_lgb_deep_prob"])
    z_xgb_d6_6_train, z_xgb_d6_6_test = dual_probit_pair(data_grand646["oof_xgb_d6"], data_grand646["test_xgb_d6"], data_grand646["test_xgb_d6_prob"])
    logit_xgb_d6_6_train, logit_xgb_d6_6_test = quantile_matched_logit(data_grand646["oof_xgb_d6"], data_grand646["test_xgb_d6_prob"])
    z_xgb_d4_6_train, z_xgb_d4_6_test = dual_probit_pair(data_grand646["oof_xgb_d4"], data_grand646["test_xgb_d4"], data_grand646["test_xgb_d4_prob"])
    logit_xgb_d4_6_train, logit_xgb_d4_6_test = quantile_matched_logit(data_grand646["oof_xgb_d4"], data_grand646["test_xgb_d4_prob"])
    z_cat_sym6_train, z_cat_sym6_test = dual_probit_pair(data_grand646["oof_cb_sym"], data_grand646["test_cb_sym"], data_grand646["test_cb_sym_prob"])
    logit_cat_sym6_train, logit_cat_sym6_test = quantile_matched_logit(data_grand646["oof_cb_sym"], data_grand646["test_cb_sym_prob"])

    z_lgb_cons5_train, z_lgb_cons5_test = dual_probit_pair(data_grand595["oof_lgb_cons"], data_grand595["test_lgb_cons"], data_grand595["test_lgb_cons_prob"])
    logit_lgb_cons5_train, logit_lgb_cons5_test = quantile_matched_logit(data_grand595["oof_lgb_cons"], data_grand595["test_lgb_cons_prob"])
    z_lgb_deep5_train, z_lgb_deep5_test = dual_probit_pair(data_grand595["oof_lgb_deep"], data_grand595["test_lgb_deep"], data_grand595["test_lgb_deep_prob"])
    logit_lgb_deep5_train, logit_lgb_deep5_test = quantile_matched_logit(data_grand595["oof_lgb_deep"], data_grand595["test_lgb_deep_prob"])
    z_xgb_d6_5_train, z_xgb_d6_5_test = dual_probit_pair(data_grand595["oof_xgb_d6"], data_grand595["test_xgb_d6"], data_grand595["test_xgb_d6_prob"])
    logit_xgb_d6_5_train, logit_xgb_d6_5_test = quantile_matched_logit(data_grand595["oof_xgb_d6"], data_grand595["test_xgb_d6_prob"])
    z_cat_sym5_train, z_cat_sym5_test = dual_probit_pair(data_grand595["oof_cb_sym"], data_grand595["test_cb_sym"], data_grand595["test_cb_sym_prob"])
    logit_cat_sym5_train, logit_cat_sym5_test = quantile_matched_logit(data_grand595["oof_cb_sym"], data_grand595["test_cb_sym_prob"])

    z_dae_train, z_dae_test = dual_probit_pair(data_lgb_dae["oof_lgb_dae_rank"], data_lgb_dae["test_lgb_dae_rank"], data_lgb_dae["test_lgb_dae_prob"])
    logit_dae_train, logit_dae_test = to_logit_f64(data_lgb_dae["oof_lgb_dae_prob"]), to_logit_f64(data_lgb_dae["test_lgb_dae_prob"])
    z_tabm463_train, z_tabm463_test = dual_probit_pair(data_tabm463["oof_tabm463_rank"], data_tabm463["test_tabm463_rank"], data_tabm463["test_tabm463_prob"])
    logit_tabm463_train, logit_tabm463_test = to_logit_f64(data_tabm463["oof_tabm463_prob"]), to_logit_f64(data_tabm463["test_tabm463_prob"])

    z_hetero_train = to_probit_f64(to_average_uniform_rank_f64(np.mean([to_probit_f64(to_average_uniform_rank_f64(data_hetero_zoo["oof_top11_9model"][:, j])) for j in [6, 7, 8]], axis=0)))
    z_hetero_test = to_probit_f64(to_average_uniform_rank_f64(np.mean([to_probit_f64(to_average_uniform_rank_f64(data_hetero_zoo["test_top11_9model"][:, j])) for j in [6, 7, 8]], axis=0)))
    z_buhlmann_train = to_probit_f64(to_average_uniform_rank_f64(data_buhlmann["oof_lgb_cons"]))
    z_buhlmann_test = to_probit_f64(to_average_uniform_rank_f64(data_buhlmann["test_lgb_cons"]))

    z_target_free_all = np.concatenate([
        to_probit_f64(to_average_uniform_rank_f64(0.35 * z_dae_train + 0.35 * z_tabm463_train + 0.20 * z_hetero_train + 0.10 * z_buhlmann_train)),
        to_probit_f64(to_average_uniform_rank_f64(0.35 * z_dae_test + 0.35 * z_tabm463_test + 0.20 * z_hetero_test + 0.10 * z_buhlmann_test))
    ])

    confirmed_df = confirmed_pairs.copy()
    confirmed_df["partner_target_free_risk"] = z_target_free_all[confirmed_df["row_partner"].to_numpy()]
    nearest_confirmed = confirmed_df.sort_values("abs_birth_delta").groupby("row").first()
    mean_confirmed = confirmed_df.groupby("row")["partner_target_free_risk"].mean()
    count_confirmed = confirmed_df.groupby("row").size()

    has_confirmed_partner = np.zeros(n_total, dtype=np.float64)
    nearest_partner_risk = z_target_free_all.copy()
    mean_partner_risk = z_target_free_all.copy()
    log_partner_count = np.zeros(n_total, dtype=np.float64)
    confirmed_indices = nearest_confirmed.index.to_numpy()
    has_confirmed_partner[confirmed_indices] = 1.0
    nearest_partner_risk[confirmed_indices] = nearest_confirmed["partner_target_free_risk"].to_numpy()
    mean_partner_risk[confirmed_indices] = mean_confirmed.to_numpy()
    log_partner_count[confirmed_indices] = np.log1p(count_confirmed.to_numpy().astype(np.float64))

    soft_tier1_df = soft_tier1_pairs.copy()
    soft_tier1_df["partner_target_free_risk"] = z_target_free_all[soft_tier1_df["row_partner"].to_numpy()]
    nearest_soft_tier1 = soft_tier1_df.sort_values("abs_birth_delta").groupby("row").first()
    has_soft_partner = np.zeros(n_total, dtype=np.float64)
    nearest_soft_partner_risk = z_target_free_all.copy()
    soft_indices = nearest_soft_tier1.index.to_numpy()
    has_soft_partner[soft_indices] = 1.0
    nearest_soft_partner_risk[soft_indices] = nearest_soft_tier1["partner_target_free_risk"].to_numpy()

    diff_confirmed_nearest = has_confirmed_partner * (nearest_partner_risk - z_target_free_all)
    diff_confirmed_mean = has_confirmed_partner * (mean_partner_risk - z_target_free_all)
    diff_soft_nearest = has_soft_partner * (nearest_soft_partner_risk - z_target_free_all)

    def build_meta_feature_matrix(
        composite_score, probit_unbundled, probit_hier, z_neural, z_gbdt646, z_gbdt595,
        z_tabm, logit_tabm, z_resnet, logit_resnet,
        z_lgb_c6, logit_lgb_c6, z_lgb_d6, logit_lgb_d6, z_xgb6_6, logit_xgb6_6, z_xgb4_6, logit_xgb4_6, z_cb6, logit_cb6,
        z_lgb_c5, logit_lgb_c5, z_lgb_d5, logit_lgb_d5, z_xgb6_5, logit_xgb6_5, z_cb5, logit_cb5,
        z_dae, logit_dae, z_tabm463, logit_tabm463, z_hetero, z_buhlmann,
        X_raw_slice, has_conf, diff_conf_near, diff_conf_mean, log_conf_cnt, has_soft, diff_soft_near
    ):
        ext1_raw = X_raw_slice[:, feature_index_map["EXT_SOURCE_1"]].astype(np.float64)
        ext2_raw = X_raw_slice[:, feature_index_map["EXT_SOURCE_2"]].astype(np.float64)
        ext3_raw = X_raw_slice[:, feature_index_map["EXT_SOURCE_3"]].astype(np.float64)
        ext1_missing = np.isnan(ext1_raw).astype(np.float64)
        ext2_missing = np.isnan(ext2_raw).astype(np.float64)
        ext3_missing = np.isnan(ext3_raw).astype(np.float64)
        missing_ext_count = ext1_missing + ext2_missing + ext3_missing

        disagreement_neural_gbdt = z_neural - z_gbdt646
        abs_disagreement = np.abs(disagreement_neural_gbdt)
        family_matrix = np.column_stack([z_tabm, z_resnet, z_lgb_c6, z_lgb_d6, z_xgb6_6, z_xgb4_6, z_cb6])
        family_std = np.std(family_matrix, axis=1)
        family_max = np.max(family_matrix, axis=1)
        family_min = np.min(family_matrix, axis=1)
        family_skew = family_max + family_min - 2.0 * np.median(family_matrix, axis=1)

        logit_gbdt646 = 0.35 * logit_lgb_c6 + 0.15 * logit_lgb_d6 + 0.20 * logit_xgb6_6 + 0.15 * logit_xgb4_6 + 0.15 * logit_cb6
        logit_gbdt595 = 0.35 * logit_lgb_c5 + 0.20 * logit_lgb_d5 + 0.25 * logit_xgb6_5 + 0.20 * logit_cb5
        logit_neural = 0.72 * logit_tabm + 0.28 * logit_resnet

        return np.column_stack([
            composite_score, probit_unbundled, probit_hier,
            z_tabm, logit_tabm, z_resnet, logit_resnet,
            z_lgb_c6, logit_lgb_c6, z_lgb_d6, logit_lgb_d6, z_xgb6_6, logit_xgb6_6, z_xgb4_6, logit_xgb4_6, z_cb6, logit_cb6,
            z_lgb_c5, logit_lgb_c5, z_lgb_d5, logit_lgb_d5, z_xgb6_5, logit_xgb6_5, z_cb5, logit_cb5,
            z_dae, logit_dae, z_tabm463, logit_tabm463, z_hetero, z_buhlmann,
            logit_neural, logit_gbdt646, logit_gbdt595,
            disagreement_neural_gbdt, abs_disagreement,
            disagreement_neural_gbdt * ext1_missing, disagreement_neural_gbdt * ext2_missing,
            disagreement_neural_gbdt * ext3_missing, disagreement_neural_gbdt * missing_ext_count,
            (logit_neural - logit_gbdt646) * missing_ext_count,
            family_std, family_max - composite_score, family_min - composite_score, family_skew,
            composite_score ** 2, composite_score ** 3, z_neural * z_gbdt646,
            has_conf, diff_conf_near, diff_conf_mean, log_conf_cnt, has_soft, diff_soft_near
        ])

    X_meta_train = build_meta_feature_matrix(
        composite_linear_train, probit_unbundled_train, probit_hier_train, z_neural646_train, z_grand646_train, z_grand595_train,
        z_tabm_train, logit_tabm_train, z_resnet_train, logit_resnet_train,
        z_lgb_cons6_train, logit_lgb_cons6_train, z_lgb_deep6_train, logit_lgb_deep6_train,
        z_xgb_d6_6_train, logit_xgb_d6_6_train, z_xgb_d4_6_train, logit_xgb_d4_6_train, z_cat_sym6_train, logit_cat_sym6_train,
        z_lgb_cons5_train, logit_lgb_cons5_train, z_lgb_deep5_train, logit_lgb_deep5_train,
        z_xgb_d6_5_train, logit_xgb_d6_5_train, z_cat_sym5_train, logit_cat_sym5_train,
        z_dae_train, logit_dae_train, z_tabm463_train, logit_tabm463_train, z_hetero_train, z_buhlmann_train,
        X_features_train, has_confirmed_partner[:n_train], diff_confirmed_nearest[:n_train],
        diff_confirmed_mean[:n_train], log_partner_count[:n_train], has_soft_partner[:n_train], diff_soft_nearest[:n_train]
    )
    X_meta_test = build_meta_feature_matrix(
        composite_linear_test, probit_unbundled_test, probit_hier_test, z_neural646_test, z_grand646_test, z_grand595_test,
        z_tabm_test, logit_tabm_test, z_resnet_test, logit_resnet_test,
        z_lgb_cons6_test, logit_lgb_cons6_test, z_lgb_deep6_test, logit_lgb_deep6_test,
        z_xgb_d6_6_test, logit_xgb_d6_6_test, z_xgb_d4_6_test, logit_xgb_d4_6_test, z_cat_sym6_test, logit_cat_sym6_test,
        z_lgb_cons5_test, logit_lgb_cons5_test, z_lgb_deep5_test, logit_lgb_deep5_test,
        z_xgb_d6_5_test, logit_xgb_d6_5_test, z_cat_sym5_test, logit_cat_sym5_test,
        z_dae_test, logit_dae_test, z_tabm463_test, logit_tabm463_test, z_hetero_test, z_buhlmann_test,
        X_features_test, has_confirmed_partner[n_train:], diff_confirmed_nearest[n_train:],
        diff_confirmed_mean[n_train:], log_partner_count[n_train:], has_soft_partner[n_train:], diff_soft_nearest[n_train:]
    )

    meta_mean = np.mean(X_meta_train, axis=0)
    meta_std = np.std(X_meta_train, axis=0) + 1e-8
    X_meta_train_scaled = (X_meta_train - meta_mean) / meta_std
    X_meta_test_scaled = (X_meta_test - meta_mean) / meta_std

    print(f"Meta-Feature Matrix Dimensions : Train {X_meta_train_scaled.shape} | Test {X_meta_test_scaled.shape}")
    print(f"Linear Synthesis OOF ROC-AUC   : {roc_auc_score(y_true, composite_linear_train):.6f}")

    oof_logistic_predictions = []
    test_logistic_predictions = []
    for seed in [42, 2026, 777]:
        skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=seed)
        for c_reg in [0.002, 0.005, 0.015]:
            oof_pred = np.zeros(n_train, dtype=np.float64)
            test_pred = np.zeros(n_test, dtype=np.float64)
            for train_idx, val_idx in skf.split(X_meta_train_scaled, y_true):
                classifier = LogisticRegression(C=c_reg, max_iter=600, solver="lbfgs")
                classifier.fit(X_meta_train_scaled[train_idx], y_true[train_idx])
                oof_pred[val_idx] = classifier.decision_function(X_meta_train_scaled[val_idx])
                test_pred += classifier.decision_function(X_meta_test_scaled) / 10.0
            oof_logistic_predictions.append(to_probit_f64(to_average_uniform_rank_f64(oof_pred)))
            test_logistic_predictions.append(to_probit_f64(to_average_uniform_rank_f64(test_pred)))

    z_meta_logistic_train = to_probit_f64(to_average_uniform_rank_f64(np.mean(oof_logistic_predictions, axis=0)))
    z_meta_logistic_test = to_probit_f64(to_average_uniform_rank_f64(np.mean(test_logistic_predictions, axis=0)))
    print(f"L2 Logistic Meta-Stacker OOF ROC-AUC          : {roc_auc_score(y_true, z_meta_logistic_train):.6f}")

    oof_lgb_meta = np.zeros(n_train, dtype=np.float64)
    test_lgb_meta = np.zeros(n_test, dtype=np.float64)
    init_margin_train = np.log(0.0807 / (1.0 - 0.0807)) + 0.85 * composite_linear_train
    init_margin_test = np.log(0.0807 / (1.0 - 0.0807)) + 0.85 * composite_linear_test

    skf_gbdt = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
    for train_idx, val_idx in skf_gbdt.split(X_meta_train, y_true):
        dtrain = lgb.Dataset(X_meta_train[train_idx], label=y_true[train_idx], init_score=init_margin_train[train_idx])
        dval = lgb.Dataset(X_meta_train[val_idx], label=y_true[val_idx], init_score=init_margin_train[val_idx], reference=dtrain)
        lgb_params = {
            "objective": "binary",
            "learning_rate": 0.015,
            "num_leaves": 7,
            "max_depth": 3,
            "min_child_samples": 600,
            "subsample": 0.75,
            "subsample_freq": 1,
            "colsample_bytree": 0.65,
            "reg_alpha": 35.0,
            "reg_lambda": 120.0,
            "verbose": -1,
            "n_jobs": 4,
            "random_state": 42 + int(train_idx[0])
        }
        booster = lgb.train(
            lgb_params,
            dtrain,
            num_boost_round=140,
            valid_sets=[dval],
            callbacks=[lgb.early_stopping(25, verbose=False)]
        )
        oof_lgb_meta[val_idx] = init_margin_train[val_idx] + booster.predict(X_meta_train[val_idx], raw_score=True)
        test_lgb_meta += (init_margin_test + booster.predict(X_meta_test, raw_score=True)) / 10.0

    z_meta_lgb_train = to_probit_f64(to_average_uniform_rank_f64(oof_lgb_meta))
    z_meta_lgb_test = to_probit_f64(to_average_uniform_rank_f64(test_lgb_meta))
    print(f"Residual Shallow LightGBM Meta OOF ROC-AUC    : {roc_auc_score(y_true, z_meta_lgb_train):.6f}")

    best_auc = 0.0
    best_weights = (0.45, 0.40, 0.15, 0.04)
    for w_linear in [0.35, 0.40, 0.45, 0.50]:
        for w_logistic in [0.30, 0.35, 0.40, 0.45]:
            w_gbdt = round(1.0 - w_linear - w_logistic, 4)
            if w_gbdt < 0.05 or w_gbdt > 0.30:
                continue
            for w_graph in [0.0, 0.02, 0.04, 0.06]:
                candidate_train = (
                    w_linear * composite_linear_train +
                    w_logistic * z_meta_logistic_train +
                    w_gbdt * z_meta_lgb_train +
                    w_graph * diff_confirmed_nearest[:n_train]
                )
                candidate_auc = roc_auc_score(y_true, candidate_train)
                if candidate_auc > best_auc:
                    best_auc = candidate_auc
                    best_weights = (w_linear, w_logistic, w_gbdt, w_graph)

    w_linear, w_logistic, w_gbdt, w_graph = best_weights
    print(f"Optimal Meta-Synthesis Weights (linear={w_linear}, logistic={w_logistic}, gbdt={w_gbdt}, graph={w_graph})")

    z_stacked_train = to_probit_f64(to_average_uniform_rank_f64(
        w_linear * composite_linear_train + w_logistic * z_meta_logistic_train + w_gbdt * z_meta_lgb_train + w_graph * diff_confirmed_nearest[:n_train]
    ))
    z_stacked_test = to_probit_f64(to_average_uniform_rank_f64(
        w_linear * composite_linear_test + w_logistic * z_meta_logistic_test + w_gbdt * z_meta_lgb_test + w_graph * diff_confirmed_nearest[n_train:]
    ))

    fold_assignment = np.full(n_total, -1, dtype=np.int32)
    for fold_idx, (_, val_idx) in enumerate(skf_gbdt.split(np.zeros(n_train), y_true)):
        fold_assignment[val_idx] = fold_idx
    confirmed_train_pairs = confirmed_pairs[(confirmed_pairs["row"] < n_train) & (confirmed_pairs["row_partner"] < n_train)].copy()
    confirmed_train_pairs["same_fold"] = fold_assignment[confirmed_train_pairs["row"].to_numpy()] == fold_assignment[confirmed_train_pairs["row_partner"].to_numpy()]
    same_fold_pairs = confirmed_train_pairs[confirmed_train_pairs["same_fold"]].sort_values("abs_birth_delta").groupby("row").first()
    val_pair_rows = same_fold_pairs.index.to_numpy()
    val_pair_partners = same_fold_pairs["row_partner"].to_numpy()

    coupling_gamma = 0.15
    z_final_train = z_stacked_train.copy()
    z_final_train[val_pair_rows] = (1.0 - coupling_gamma) * z_stacked_train[val_pair_rows] + coupling_gamma * z_stacked_train[val_pair_partners]

    confirmed_test_pairs = confirmed_pairs[(confirmed_pairs["row"] >= n_train) & (confirmed_pairs["row_partner"] >= n_train)].copy()
    nearest_test_pairs = confirmed_test_pairs.sort_values("abs_birth_delta").groupby("row").first()
    test_pair_rows = nearest_test_pairs.index.to_numpy() - n_train
    test_pair_partners = nearest_test_pairs["row_partner"].to_numpy() - n_train

    z_final_test = z_stacked_test.copy()
    z_final_test[test_pair_rows] = (1.0 - coupling_gamma) * z_stacked_test[test_pair_rows] + coupling_gamma * z_stacked_test[test_pair_partners]

    final_oof_auc = roc_auc_score(y_true, z_final_train)
    print("-" * 90)
    print(f"Transductive Partner Coupling Applied to Test Pairs : {len(test_pair_rows)} borrowers")
    print(f"Final Cross-Validation OOF ROC-AUC                  : {final_oof_auc:.6f}")
    print("-" * 90)

    final_test_predictions = to_average_uniform_rank_f64(z_final_test)

    submission_df = pd.DataFrame({
        "SK_ID_CURR": test_ids,
        "TARGET": final_test_predictions
    })
    script_dir = os.path.dirname(os.path.abspath(__file__))
    submission_path = os.path.join(script_dir, "submission_final.csv")
    submission_df.to_csv(submission_path, index=False, float_format="%.10f")

    with open(submission_path, "rb") as f:
        submission_sha256 = hashlib.sha256(f.read()).hexdigest()

    print(f"Saved Final Submission : {submission_path}")
    print(f"Submission SHA-256     : {submission_sha256}")
    print("=" * 90)

if __name__ == "__main__":
    main()
