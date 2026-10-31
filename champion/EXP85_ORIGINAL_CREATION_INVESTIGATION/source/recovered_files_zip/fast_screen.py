"""fast_screen.py - minutes-not-hours A/B screening of feature ideas for the GCI Home Credit task (train.csv/test.csv only).

WHY THIS EXISTS
  Every idea is tested the same way: same 5 folds as the real pipeline, same fold-safe target encodings and partner features,
  one fixed fast model (GPU XGBoost), baseline and candidate trained on IDENTICAL frames, paired per-fold deltas.
  The baseline INCLUDES the label-based blocks (product keys, partner features) - a screen without them rewards ideas that merely
  duplicate those blocks (that is how the v21 gate was fooled).

USAGE (Windows, venv with pandas numpy scikit-learn scipy xgboost):
  set GCI_DATA_DIR=D:\\gci\\inpuy            (folder with train.csv, test.csv, sample_submission.csv)
  python fast_screen.py --validate           (first run: checks the proxy reproduces KNOWN effects + null control, ~3-5 min)
  python fast_screen.py                      (all candidates, 5 folds)
  python fast_screen.py --only e_poly,research_lean --folds 0,1,2,3,4 --seeds 2

DECISION RULE (printed per candidate): PROMISING if mean paired gain >= +0.0002, >= 4/5 folds non-negative and mean > 2 x SE.
  A proxy can disagree with the real LightGBM pipeline: confirm any PROMISING block with GCI_v25_block_tester.ipynb before wiring it in.
ADD YOUR OWN IDEA: write a function returning a DataFrame with len(al) rows (label-free) and register it in FREE; or a label-based
  function fn(fit_idx, apply_idx) -> DataFrame and register it in LABEL (see te_combos / partner_emp_phone below).
"""
import os, sys, time, json, argparse, itertools, warnings; warnings.filterwarnings("ignore")
ap = argparse.ArgumentParser()
ap.add_argument("--data", default=os.environ.get("GCI_DATA_DIR", ".")); ap.add_argument("--folds", default="0,1,2,3,4"); ap.add_argument("--only", default="")
ap.add_argument("--seeds", type=int, default=1); ap.add_argument("--validate", action="store_true"); ap.add_argument("--model", default="auto", choices=["auto", "xgb", "hgb"])
ap.add_argument("--out", default="fast_screen_results.csv"); args = ap.parse_args(); os.environ["GCI_DATA_DIR"] = args.data
exec(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "gci_core.py"), encoding="utf-8").read())      # defines al, y, ntr, nt, n, Xa, Xat, KS, M0, enc, train_enc, partner_enc, nested_generic ...
from sklearn.linear_model import LogisticRegression
# ------------------------------------------------------------------ fast model (GPU XGBoost; sklearn HGB fallback for machines without it)
def make_model(kind):
    if kind in ("auto", "xgb"):
        try:
            import xgboost as xgb; major = int(xgb.__version__.split(".")[0]); dev = dict(device="cuda", tree_method="hist") if major >= 2 else dict(tree_method="gpu_hist")
            def fp(Xa_, ya, Xb_, seed):
                m = xgb.XGBClassifier(n_estimators=500, max_depth=5, learning_rate=0.06, subsample=0.8, colsample_bytree=0.5, min_child_weight=20, reg_lambda=20, eval_metric="auc", random_state=seed, verbosity=0, **dev)
                m.fit(Xa_, ya); return m.predict_proba(Xb_)[:, 1]
            print("proxy model: XGBoost", xgb.__version__, dev); return fp
        except Exception as e:
            if kind == "xgb": raise
            print("xgboost unavailable (%r) -> sklearn HistGradientBoosting fallback (slower)" % (e,))
    from sklearn.ensemble import HistGradientBoostingClassifier as HC
    its = 40 if os.environ.get("FAST_TEST") == "1" else 400
    return lambda Xa_, ya, Xb_, seed: HC(max_iter=its, learning_rate=0.05, max_leaf_nodes=15, min_samples_leaf=150, l2_regularization=20, max_features=0.5, early_stopping=False, random_state=seed).fit(Xa_, ya).predict_proba(Xb_)[:, 1]
fit_predict = make_model(args.model)
# ------------------------------------------------------------------ candidate blocks
E3 = al[["EXT_SOURCE_1", "EXT_SOURCE_2", "EXT_SOURCE_3"]]; em = E3.mean(axis=1); cr_, an_, inc_, gp_ = al.AMT_CREDIT, al.AMT_ANNUITY, al.AMT_INCOME_TOTAL, al.AMT_GOODS_PRICE; age_ = -al.DAYS_BIRTH / 365.25
sc = lambda c: al[c].astype(str); agegrp = (age_ // 5).astype(int).astype(str)
def c_e_poly(): return pd.DataFrame({"P_EXTMEAN_PAY": em * an_ / cr_, "P_EXTMEAN_CTI": em * cr_ / inc_, "P_EXTMEAN_PAY_AGE": em * an_ / cr_ * age_, "P_CREDIT_GOODS_INC": (cr_ - gp_) / inc_, "P_EXTMEAN_ANN_INC": em * an_ / inc_})
def c_ext_pct():
    pct = lambda keys: em.groupby(keys).rank(pct=True)
    return pd.DataFrame({"EXTF_PCT_OCC": pct(sc("OCCUPATION_TYPE")), "EXTF_PCT_GEN_AGE": pct(sc("CODE_GENDER") + "|" + agegrp), "EXTF_PCT_ORG": pct(sc("ORGANIZATION_TYPE")), "EXTF_PCT_INCTYPE": pct(sc("NAME_INCOME_TYPE"))})
def c_date_diffs():
    dc = ["DAYS_BIRTH", "DAYS_EMPLOYED", "DAYS_REGISTRATION", "DAYS_ID_PUBLISH", "DAYS_LAST_PHONE_CHANGE"]; D = al[dc].copy(); D["DAYS_EMPLOYED"] = D["DAYS_EMPLOYED"].where(D["DAYS_EMPLOYED"] != 365243)
    return pd.DataFrame({f"RDIFF_{a_[5:]}_{b_[5:]}": (D[a_] - D[b_]).values for a_, b_ in itertools.combinations(dc, 2)})
def c_implied_peaks():
    cr = cr_.astype(float).values; an = an_.astype(float).values; q = an / cr; term = cr / an; d = {}
    for n_ in (12, 24, 36, 48, 60):
        lo, hi = np.full(len(q), 1e-9), np.full(len(q), 0.5)
        for _ in range(50): mid = (lo + hi) / 2; f_ = mid / (1 - (1 + mid) ** (-n_)) - q; hi = np.where(f_ > 0, mid, hi); lo = np.where(f_ > 0, lo, mid)
        d[f"R_IMPLIED_RATE_{n_}M"] = np.where(q > 1.0 / n_, 12 * (lo + hi) / 2, np.nan)
    vc = pd.Series(np.round(term, 1)).value_counts(); peaks = np.sort(vc.index[vc >= 400].values); j_ = np.clip(np.searchsorted(peaks, term), 1, len(peaks) - 1)
    near = np.where(np.abs(term - peaks[j_ - 1]) <= np.abs(term - peaks[j_]), peaks[j_ - 1], peaks[j_])
    d.update(R_TERM_PEAK_NEAREST=near, R_TERM_PEAK_DISTANCE=term - near, R_TERM_PEAK_REL_DISTANCE=(term - near) / near, R_TERM_RANK=pd.Series(term).rank(pct=True).values, R_TERM_RANK_BY_CONTRACT=pd.Series(term).groupby(al.NAME_CONTRACT_TYPE.values).rank(pct=True).values)
    return pd.DataFrame(d)
def c_tier_z():
    tier = KS["Q4"]; gsz = tier.map(tier.value_counts()); gm = em.groupby(tier).transform("mean"); gsd = em.groupby(tier).transform("std")
    return pd.DataFrame({"TIER_EXT_Z": ((em - gm) / gsd).where(gsz >= 30), "TIER_EXT_DIFF": (em - gm).where(gsz >= 30)})
def c_amt_pct():          # NEW (untested by the author): peer-relative money amounts
    pct = lambda s, keys: s.groupby(keys).rank(pct=True)
    return pd.DataFrame({"AP_INC_OCC": pct(inc_, sc("OCCUPATION_TYPE")), "AP_INC_EDU_AGE": pct(inc_, sc("NAME_EDUCATION_TYPE") + "|" + agegrp), "AP_CR_INCTYPE": pct(cr_, sc("NAME_INCOME_TYPE")), "AP_ANNINC_GEN_AGE": pct(an_ / inc_, sc("CODE_GENDER") + "|" + agegrp), "AP_CR_ORG": pct(cr_, sc("ORGANIZATION_TYPE"))})
def c_missing_pattern():  # NEW (untested by the author): joint NaN pattern of all incomplete columns + how common that pattern is
    cols = [c for c in al.columns if al[c].isna().any()][:40]; M = al[cols].isna().values.astype(np.int64); code = (M * (1 << np.arange(len(cols), dtype=np.int64))).sum(1)
    return pd.DataFrame({"NANPAT_CODE": code.astype(float), "NANPAT_FREQ": pd.Series(code).map(pd.Series(code).value_counts()).values, "NANPAT_N": M.sum(1)})
def c_ctrl_noise(): rs = np.random.RandomState(123); return pd.DataFrame(rs.randn(len(al), 3), columns=["CTRL_N1", "CTRL_N2", "CTRL_N3"])        # null control: expected gain ~ 0
def c_ctrl_dup(): return pd.DataFrame({"CTRL_DUP_EXTMEAN": em.values, "CTRL_DUP_AGE": age_.values})                                        # redundancy control: expected ~ 0
FREE = {"e_poly": c_e_poly, "ext_pct": c_ext_pct, "date_diffs": c_date_diffs, "implied_peaks": c_implied_peaks, "tier_z": c_tier_z, "amt_pct": c_amt_pct, "missing_pattern": c_missing_pattern, "ctrl_noise": c_ctrl_noise, "ctrl_dup": c_ctrl_dup}
FREE["research_lean"] = lambda: pd.concat([c_date_diffs(), c_implied_peaks()], axis=1)
# ---- label-based candidates: fn(fit_idx, apply_idx) -> DataFrame (nested out-of-fold automatically) ----
def _mk_key(parts):
    B = al.DAYS_BIRTH; emp = al.DAYS_EMPLOYED.where(al.DAYS_EMPLOYED != 365243); off = {"id": al.DAYS_ID_PUBLISH - B, "reg": al.DAYS_REGISTRATION - B, "emp": emp - B, "ph": al.DAYS_LAST_PHONE_CHANGE - B}
    k = sc("CODE_GENDER"); ok = pd.Series(True, index=al.index)
    for p_ in parts: k = k + "|" + off[p_].astype(str); ok &= off[p_].notna()          # NaN keys are EXCLUDED (the '22% coverage' claim came from NaN keys matching each other)
    if "reg" in parts: ok &= (al.DAYS_REGISTRATION != 0)
    codes = pd.factorize(k.where(ok))[0]; sz = pd.Series(codes).map(pd.Series(codes[codes >= 0]).value_counts()).fillna(0).values; return np.where((codes >= 0) & (sz > 1), codes, -1)
_NEWK = {"PYE_EMPID": _mk_key(["emp", "id"]), "PYE_PHID": _mk_key(["ph", "id"]), "PYE_REGEMP": _mk_key(["reg", "emp"])}
def l_partner_emp_phone(fit, idx):
    idx = np.asarray(idx); lab = np.full(len(al), np.nan); lab[fit] = y[fit]; out = {}
    for nm, codes in _NEWK.items():
        m = codes >= 0; s = np.bincount(codes[m], weights=np.nan_to_num(lab[m]), minlength=codes.max() + 1); c = np.bincount(codes[m], weights=(~np.isnan(lab[m])).astype(float), minlength=codes.max() + 1)
        ci = codes[idx]; own = np.nan_to_num(lab[idx]); ownc = (~np.isnan(lab[idx])).astype(float); ps = np.where(ci >= 0, s[np.maximum(ci, 0)] - own, np.nan); pc = np.where(ci >= 0, c[np.maximum(ci, 0)] - ownc, np.nan)
        out[nm] = np.where((ci >= 0) & (pc > 0), (ps > 0).astype(float), np.nan)
    return pd.DataFrame(out)
_CK = {"ORG_OCC": sc("ORGANIZATION_TYPE") + "|" + sc("OCCUPATION_TYPE"), "OCC_INC_EDU": sc("OCCUPATION_TYPE") + "|" + sc("NAME_INCOME_TYPE") + "|" + sc("NAME_EDUCATION_TYPE"), "REG_ORG": sc("REGION_POPULATION_RELATIVE") + "|" + sc("ORGANIZATION_TYPE"),
       "REG_OCC_GEN": sc("REGION_POPULATION_RELATIVE") + "|" + sc("OCCUPATION_TYPE") + "|" + sc("CODE_GENDER"), "FAM_INC_HOUSE": sc("NAME_FAMILY_STATUS") + "|" + sc("NAME_INCOME_TYPE") + "|" + sc("NAME_HOUSING_TYPE") + "|" + sc("CNT_CHILDREN")}
_CC = {k: pd.factorize(v)[0] for k, v in _CK.items()}; _CN = {k: int(v.max()) + 1 for k, v in _CC.items()}
def l_te_combos(fit, qi):
    qi = np.asarray(qi); pr = float(y[fit].mean()); yf = y[fit].astype(float); out = {}
    for k, c in _CC.items(): ct = np.bincount(c[fit], minlength=_CN[k]).astype(float); st = np.bincount(c[fit], weights=yf, minlength=_CN[k]); out["TEC_" + k] = (st[c[qi]] + 50 * pr) / (ct[c[qi]] + 50); out["CNC_" + k] = ct[c[qi]]
    return pd.DataFrame(out)
def _woe_fit_pred(Fa, ya, Fb, nb=24):
    p0 = ya.mean(); base = np.log(p0 / (1 - p0)); maps = {}
    for c in Fa.columns:
        v = Fa[c].values; nn_ = ~np.isnan(v)
        if nn_.sum() < 1000 or np.nanstd(v) == 0: continue
        qs = np.unique(np.nanquantile(v[nn_], np.linspace(0, 1, nb + 1)[1:-1])); idx = np.where(nn_, np.searchsorted(qs, np.where(nn_, v, 0), side="right"), len(qs) + 1); nbin = len(qs) + 2
        cnt = np.bincount(idx, minlength=nbin).astype(float); pos = np.bincount(idx, weights=ya, minlength=nbin); r = (pos + 20 * p0) / (cnt + 20); maps[c] = (qs, np.log(r / (1 - r)) - base)
    def tf(F):
        out = np.zeros((len(F), len(maps)), np.float32)
        for j, (c, (qs, woe)) in enumerate(maps.items()): v = F[c].values; nn_ = ~np.isnan(v); idx = np.where(nn_, np.searchsorted(qs, np.where(nn_, v, 0), side="right"), len(qs) + 1); out[:, j] = woe[idx]
        return out
    return LogisticRegression(C=0.05, max_iter=200).fit(tf(Fa), ya).decision_function(tf(Fb))
XF = Xa.reset_index(drop=True).astype(np.float32)
def l_woe_view(fit, qi): return pd.DataFrame({"V_WOE": _woe_fit_pred(XF.iloc[fit], y[fit], XF.iloc[np.asarray(qi)])})
LABEL = {"partner_emp_phone": l_partner_emp_phone, "te_combos": l_te_combos, "woe_view": l_woe_view}
PARTNER_COLS = ["PY_ANY", "PY_N", "PY_NEAR180", "PY_EARLIER", "PY_LATER", "PYL_ANY"]
DROP = {"abl_no_partner_labels": lambda cols: [c for c in PARTNER_COLS if c in cols], "abl_no_product_keys": lambda cols: [c for c in cols if c.startswith(("TE_", "CNT_"))]}     # validation: known effects ~ -0.001 / ~ -0.008
# ------------------------------------------------------------------ screening loop
Xb = Xa.reset_index(drop=True); FL = list(StratifiedKFold(FOLDS, shuffle=True, random_state=42).split(np.zeros(n), y)); folds = [int(x) for x in args.folds.split(",") if int(x) < FOLDS]
if args.validate: names = ["abl_no_partner_labels", "abl_no_product_keys", "ctrl_noise", "ctrl_dup"]
elif args.only: names = [x.strip() for x in args.only.split(",")]
else: names = list(FREE) + list(LABEL)
unknown = [x for x in names if x not in FREE and x not in LABEL and x not in DROP]; assert not unknown, f"unknown candidates {unknown}; available {list(FREE) + list(LABEL) + list(DROP)}"
FREE_CACHE = {k: FREE[k]().replace([np.inf, -np.inf], np.nan).astype(np.float32).reset_index(drop=True) for k in names if k in FREE}
res = {}; t0 = time.time(); print(f"candidates: {names} | folds {folds} | seeds {args.seeds}", flush=True)
def score(A_, ya, B_, yb):
    return float(np.mean([roc_auc_score(yb, fit_predict(A_, ya, B_, 42 + s)) for s in range(args.seeds)]))
for k in folds:
    a, b = FL[k]; ea, eb = train_enc(a), enc(a, b); A0 = pd.concat([Xb.iloc[a].reset_index(drop=True), ea], axis=1).astype(np.float32); B0 = pd.concat([Xb.iloc[b].reset_index(drop=True), eb], axis=1).astype(np.float32)
    res[("baseline", k)] = score(A0, y[a], B0, y[b]); print(f"fold {k+1}: baseline {res[('baseline', k)]:.5f}  ({time.time()-t0:.0f}s)", flush=True)
    for nm in names:
        if nm in DROP: dc = DROP[nm](list(A0.columns)); A_, B_ = A0.drop(columns=dc), B0.drop(columns=dc)
        elif nm in FREE: A_ = pd.concat([A0, FREE_CACHE[nm].iloc[:n].iloc[a].reset_index(drop=True)], axis=1); B_ = pd.concat([B0, FREE_CACHE[nm].iloc[:n].iloc[b].reset_index(drop=True)], axis=1)
        else: A_ = pd.concat([A0, nested_generic(a, LABEL[nm]).astype(np.float32)], axis=1); B_ = pd.concat([B0, LABEL[nm](a, b).astype(np.float32).reset_index(drop=True)], axis=1)
        res[(nm, k)] = score(A_, y[a], B_, y[b]); print(f"   {nm:22s} {res[(nm, k)]:.5f}  delta {res[(nm, k)] - res[('baseline', k)]:+.5f}  ({time.time()-t0:.0f}s)", flush=True)
rows = []
for nm in names:
    d = np.array([res[(nm, k)] - res[("baseline", k)] for k in folds]); se = d.std(ddof=1) / np.sqrt(len(d)) if len(d) > 1 else float("nan"); ok = (d.mean() >= 0.0002) and ((d >= 0).sum() >= min(4, len(d))) and (len(d) < 3 or d.mean() > 2 * se)
    rows.append(dict(candidate=nm, mean_gain=round(float(d.mean()), 5), se=round(float(se), 5), per_fold=[round(float(x), 5) for x in d], nonneg=f"{(d >= 0).sum()}/{len(d)}", verdict="PROMISING" if (ok and not nm.startswith(('ctrl', 'abl'))) else ("control/ablation" if nm.startswith(("ctrl", "abl")) else "reject")))
T = pd.DataFrame(rows).sort_values("mean_gain", ascending=False); pd.set_option("display.width", 220, "display.max_colwidth", 70)
print("\nBASELINE per fold:", [round(res[("baseline", k)], 5) for k in folds]); print(T.to_string(index=False)); T.to_csv(args.out, index=False); print("saved", args.out)
if args.validate: print("\nVALIDATION READING: abl_no_partner_labels should be about -0.001 and abl_no_product_keys about -0.008 (real-pipeline ablation values); ctrl_noise / ctrl_dup should sit within about +-0.0003. If so, the proxy is trustworthy for screening.")
