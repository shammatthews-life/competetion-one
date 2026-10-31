"""gci_core.py - standalone feature/encoding setup used by fast_screen.py (extracted from the v17 pipeline).
Builds from train.csv/test.csv only: the label-free feature families (Xa/Xat), the decoded loan-product keys, the
fold-safe target encodings (enc/train_enc) and the repeat-applicant (partner) features. No external data.
Environment: GCI_DATA_DIR = folder containing train.csv, test.csv, sample_submission.csv. Optional: GCI_CACHE_BASE.
"""
# cell 1 - setup (same input discovery as v15: finds train/test/sample in /content, Drive or cwd)
import os, sys, glob, time, warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
try:
    import lightgbm as lgb
except Exception:
    lgb = None
from scipy.stats import rankdata
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
# ---- v16 switches ----
SETS = ["old", "e1", "e1et", "e1Aet"]      # LightGBM sets to run. Drop some (e.g. ["e1et","e1Aet"]) for a faster ~4h run; the final formula adapts.
REUSE_V15_CACHE = True                     # label-free helpers (residuals, view features) are re-used from the v15 cache folder if present
USE_PARTNER = True                         # repeat-applicant (partner) features
USE_PERSON = True                          # person-pooled features (EXT scores averaged over a person's applications, partner deltas); 5-fold test: mean +0.0005 AUC
USE_COMMUNITY_KEYS = True                  # rounded CREDIT/ANNUITY keys (1/2/3 dec), combos with goods/credit/annuity, 3 smoothing levels
SMOKE = os.environ.get("SMOKE") == "1"                   # SMOKE=1 -> tiny quick test of the code path only
def find(name, roots=(os.environ.get("GCI_DATA_DIR", "."), os.getcwd(), "/content")):
    for r in roots:
        for pat in (name, "*/"+name, "*/*/"+name, "*/*/*/"+name, "*/*/*/*/"+name):
            hit = glob.glob(os.path.join(r, pat))
            if hit: return hit[0]
    return None
def find_data():   # validated finder: 34 columns, TARGET present, 171,202 rows (skips any other train.csv with the same name)
    cands = []
    for r in (os.environ.get("GCI_DATA_DIR", "."), os.getcwd(), "/content"):
        for pat in ("train.csv", "*/train.csv", "*/*/train.csv", "*/*/*/train.csv", "*/*/*/*/train.csv"): cands += glob.glob(os.path.join(r, pat))
    for f in sorted(set(cands)):
        d = os.path.dirname(f)
        if not all(os.path.exists(os.path.join(d, x)) for x in ("test.csv", "sample_submission.csv")): continue
        try: head = open(f, "rb").readline().decode("utf-8", "ignore").strip().split(","); nrows = sum(1 for _ in open(f, "rb")) - 1
        except Exception: continue
        if len(head) == 34 and "TARGET" in head and "AMT_ANNUITY" in head and nrows == 171202: print("using competition files in", d); return [os.path.join(d, x) for x in ("train.csv", "test.csv", "sample_submission.csv")]
        print("skipping a different train.csv:", f, "| columns", len(head), "| rows", nrows)
    raise SystemExit("Real competition files not found: upload train.csv, test.csv, sample_submission.csv from the competition input folder to /content")
TRAIN_PATH, TEST_PATH, SAMPLE_PATH = find_data()
assert TRAIN_PATH and TEST_PATH and SAMPLE_PATH, "Could not find train.csv / test.csv / sample_submission.csv - upload them to /content or put them in Drive"
HGB_PATH = find("GCI_hgb_product_test.npy"); HGB_OOF = find("GCI_hgb_product_oof.npy")
base_dir = os.environ.get("GCI_CACHE_BASE", os.getcwd())
CACHE_DIR = os.path.join(base_dir, "gci_cache_v17") if not SMOKE else "/tmp/gci_cache_smoke"; OUT_DIR = os.path.join(base_dir, "gci_output_v17") if not SMOKE else "/tmp/gci_out_smoke"
os.makedirs(CACHE_DIR, exist_ok=True); os.makedirs(OUT_DIR, exist_ok=True)
THREADS = os.cpu_count() or 1
OLD_CACHE = os.path.join(base_dir, "gci_cache_final")
print("train:", TRAIN_PATH, "\ntest:", TEST_PATH, "\nsample:", SAMPLE_PATH, "\nhgb oof:", HGB_OOF, "\nhgb npy:", HGB_PATH, "\ncache:", CACHE_DIR, "| threads:", THREADS)

# cell 2 - OPTIMISED: features (v15 families) + decoded product keys (v15) + community keys + partner features (v16)
CATS = ["NAME_CONTRACT_TYPE","CODE_GENDER","NAME_INCOME_TYPE","NAME_EDUCATION_TYPE","NAME_FAMILY_STATUS","NAME_HOUSING_TYPE","OCCUPATION_TYPE","ORGANIZATION_TYPE"]
tr = pd.read_csv(TRAIN_PATH); te = pd.read_csv(TEST_PATH); sample = pd.read_csv(SAMPLE_PATH)
y = tr["TARGET"].values; ntr = len(tr)
al = pd.concat([tr.drop(columns=["TARGET"]), te], ignore_index=True)
def new_cols(d): return pd.DataFrame(d, index=al.index)
def base(al):
    X = al.drop(columns=["SK_ID_CURR"]).copy()
    for c in CATS: X[c] = X[c].astype("category").cat.codes.replace(-1, np.nan).astype(float)
    return X
def f_clean(al, X):
    d = {"DE_SENT": (al.DAYS_EMPLOYED == 365243).astype(int), "DAYS_EMPLOYED_CLEAN": al.DAYS_EMPLOYED.where(al.DAYS_EMPLOYED != 365243),
         "ORG_XNA": (al.ORGANIZATION_TYPE == "XNA").astype(int), "CAR_AGE_OUT": (al.OWN_CAR_AGE >= 60).astype(int), "MISS_CNT": al.isna().sum(axis=1)}
    for c in ["EXT_SOURCE_1","EXT_SOURCE_3","OWN_CAR_AGE","AMT_REQ_CREDIT_BUREAU_YEAR"]: d[c + "_NA"] = al[c].isna().astype(int)
    return pd.concat([X, new_cols(d)], axis=1)
def f_ext(al, X):
    e = al[["EXT_SOURCE_1","EXT_SOURCE_2","EXT_SOURCE_3"]]; d = {}
    d["EXT_MEAN"] = e.mean(axis=1); d["EXT_MIN"] = e.min(axis=1); d["EXT_MAX"] = e.max(axis=1); d["EXT_STD"] = e.std(axis=1)
    d["EXT_RANGE"] = d["EXT_MAX"] - d["EXT_MIN"]; d["EXT_NOBS"] = e.notna().sum(axis=1); d["EXT_PROD"] = e.prod(axis=1)
    d["EXT_W"] = (al.EXT_SOURCE_1*2 + al.EXT_SOURCE_2*3 + al.EXT_SOURCE_3*4) / 9
    d["E12"] = al.EXT_SOURCE_1*al.EXT_SOURCE_2; d["E13"] = al.EXT_SOURCE_1*al.EXT_SOURCE_3; d["E23"] = al.EXT_SOURCE_2*al.EXT_SOURCE_3
    d["D12"] = al.EXT_SOURCE_1-al.EXT_SOURCE_2; d["D13"] = al.EXT_SOURCE_1-al.EXT_SOURCE_3; d["D23"] = al.EXT_SOURCE_2-al.EXT_SOURCE_3
    return pd.concat([X, new_cols(d)], axis=1)
def f_fin(al, X):
    inc, cr, an, gp = al.AMT_INCOME_TOTAL, al.AMT_CREDIT, al.AMT_ANNUITY, al.AMT_GOODS_PRICE
    d = {"CREDIT_INC": cr/inc, "ANN_INC": an/inc, "ANN_CREDIT": an/cr, "TERM": cr/an, "GOODS_CREDIT": gp/cr, "CREDIT_MINUS_GOODS": cr-gp, "GOODS_INC": gp/inc,
         "INC_PER_FAM": inc/al.CNT_FAM_MEMBERS, "CREDIT_PER_FAM": cr/al.CNT_FAM_MEMBERS, "ANN_PER_FAM": an/al.CNT_FAM_MEMBERS}
    for c in ["AMT_INCOME_TOTAL","AMT_CREDIT","AMT_ANNUITY","AMT_GOODS_PRICE"]: d["LOG_" + c] = np.log1p(al[c])
    return pd.concat([X, new_cols(d)], axis=1)
def f_age(al, X):
    age = -al.DAYS_BIRTH; de = al.DAYS_EMPLOYED.where(al.DAYS_EMPLOYED != 365243)
    d = {"AGE": age/365.25, "EMP_AGE": de/al.DAYS_BIRTH, "EMP_YEARS": -de/365.25, "REG_AGE": al.DAYS_REGISTRATION/al.DAYS_BIRTH, "ID_AGE": al.DAYS_ID_PUBLISH/al.DAYS_BIRTH,
         "PHONE_AGE": al.DAYS_LAST_PHONE_CHANGE/al.DAYS_BIRTH, "CAR_AGE_RATIO": al.OWN_CAR_AGE/(age/365.25), "CREDIT_PER_AGE": al.AMT_CREDIT/(age/365.25), "INC_PER_AGE": al.AMT_INCOME_TOTAL/(age/365.25)}
    return pd.concat([X, new_cols(d)], axis=1)
def f_bureau(al, X):
    b = [c for c in al.columns if c.startswith("AMT_REQ")]
    d = {"BUR_SUM": al[b].sum(axis=1, min_count=1), "BUR_RECENT": al[["AMT_REQ_CREDIT_BUREAU_HOUR","AMT_REQ_CREDIT_BUREAU_MON"]].sum(axis=1, min_count=1),
         "BUR_MON_YEAR": al.AMT_REQ_CREDIT_BUREAU_MON/(al.AMT_REQ_CREDIT_BUREAU_YEAR+1)}
    return pd.concat([X, new_cols(d)], axis=1)
def f_count(al, X):
    d = {"CNT_" + c: al[c].map(al[c].value_counts()) for c in CATS + ["REGION_POPULATION_RELATIVE","DAYS_BIRTH","AMT_CREDIT","AMT_GOODS_PRICE","AMT_ANNUITY","AMT_INCOME_TOTAL"]}
    return pd.concat([X, new_cols(d)], axis=1)
def f_group(al, X):
    tmp = al.copy(); tmp["CREDIT_INC"] = al.AMT_CREDIT/al.AMT_INCOME_TOTAL; d = {}
    for k in ["ORGANIZATION_TYPE","OCCUPATION_TYPE","NAME_INCOME_TYPE","NAME_EDUCATION_TYPE","REGION_POPULATION_RELATIVE"]:
        for v in ["AMT_INCOME_TOTAL","AMT_CREDIT","EXT_SOURCE_1","EXT_SOURCE_2","EXT_SOURCE_3","DAYS_BIRTH","CREDIT_INC"]:
            d[f"G_{k[:6]}_{v[-6:]}_DIFF"] = tmp[v] - tmp.groupby(k)[v].transform("mean")
    return pd.concat([X, new_cols(d)], axis=1)
def f_kdec(al, X):    # credit = goods * (1 + 0.0132*k) for ~90% of cash loans with credit > goods -> integer k (insurance/tariff tier); NaN where another tariff family applies
    C, G = al.AMT_CREDIT, al.AMT_GOODS_PRICE; cash = al.NAME_CONTRACT_TYPE == "Cash loans"; kf = (C / G - 1) / 0.0132; k = kf.round(); ok = cash & (C > G) & ((kf - k).abs() < 0.1)
    K = pd.Series(np.where(ok, k, np.where(cash & (C == G), 0.0, np.nan)), index=al.index)
    return pd.concat([X, new_cols({"K_DEC": K, "CG_RATIO": C / G})], axis=1)
# ---- PARTNER / REPEAT-APPLICANT FEATURES (new in v16) ----
# same person re-applying keeps constant gaps: DAYS_REGISTRATION-DAYS_BIRTH and DAYS_ID_PUBLISH-DAYS_BIRTH. Two fingerprints: strict (+ education, region, mobile flag) and loose (gender + gaps).
_i1 = al.DAYS_REGISTRATION - al.DAYS_BIRTH; _i2 = al.DAYS_ID_PUBLISH - al.DAYS_BIRTH
def _groups(parts):
    k = al[parts].astype(str).agg("|".join, axis=1) + "|" + _i1.astype(str) + "|" + _i2.astype(str); k = k.where(al.DAYS_REGISTRATION != 0)
    sz = k.map(k.value_counts()); k = k.where(sz > 1); g = pd.Series(pd.factorize(k)[0]).replace(-1, np.nan).values
    mem = pd.DataFrame({"g": g}).groupby("g").groups; return g, sz.where(k.notna()).values, mem
_gb = al.DAYS_BIRTH.values
def _plist(g, mem):         # for every row: (indices of the other applications, signed gap: >0 means the partner applied EARLIER)
    out = {}
    for _, ix in mem.items():
        ix = np.asarray(ix)
        for i in ix: o = ix[ix != i]; out[i] = (o, _gb[o] - _gb[i])
    return out
G_S, SZ_S, MEM_S = _groups(["CODE_GENDER", "NAME_EDUCATION_TYPE", "REGION_POPULATION_RELATIVE", "FLAG_MOBIL"]); PL_S = _plist(G_S, MEM_S)
G_L, SZ_L, MEM_L = _groups(["CODE_GENDER"]); PL_L = _plist(G_L, MEM_L)
def f_partner(al, X):
    gap = np.full(len(al), np.nan)
    for i, (o, gp_) in PL_S.items(): gap[i] = gp_[np.argmin(np.abs(gp_))]
    d = {"HAS_PARTNER": (~np.isnan(G_S)).astype(int), "PARTNER_N": SZ_S, "PARTNER_GAP": gap, "PARTNER_ABSGAP": np.abs(gap), "HAS_PARTNER_LOOSE": (~np.isnan(G_L)).astype(int), "PARTNER_N_LOOSE": SZ_L}
    return pd.concat([X, new_cols(d)], axis=1)
def f_person(al, X):      # label-free PERSON-level pooling: repeat applications are noisy re-measurements of the same person (EXT corr with partner 0.72-0.83) -> pool them
    N = len(al); E = al[["EXT_SOURCE_1", "EXT_SOURCE_2", "EXT_SOURCE_3"]].values.astype(float); pe = np.full((N, 3), np.nan); dif = np.full((N, 3), np.nan)
    cr = np.full(N, np.nan); an = np.full(N, np.nan); inc = np.full(N, np.nan); same = np.full(N, np.nan); qd = np.full(N, np.nan)
    Qv = (al.AMT_ANNUITY / al.AMT_CREDIT).values; ct = (al.NAME_CONTRACT_TYPE == "Cash loans").values; CR, AN_, INC = al.AMT_CREDIT.values, al.AMT_ANNUITY.values, al.AMT_INCOME_TOTAL.values
    for i, (o, g) in PL_S.items():
        j = o[np.argmin(np.abs(g))]; allr = np.r_[i, o]
        for k in range(3):
            v = E[allr, k]; pe[i, k] = np.nanmean(v) if np.isfinite(v).any() else np.nan; dif[i, k] = E[i, k] - E[j, k]
        cr[i] = CR[i] / CR[j]; an[i] = AN_[i] / AN_[j]; inc[i] = INC[i] / INC[j]; same[i] = float(ct[i] == ct[j]); qd[i] = Qv[i] - Qv[j]
    d = {f"PERS_EXT{k+1}": pe[:, k] for k in range(3)}; d.update({f"PDIF_EXT{k+1}": dif[:, k] for k in range(3)}); d.update(PCR_RATIO=cr, PAN_RATIO=an, PINC_RATIO=inc, PSAME_CT=same, PQ_DIFF=qd)
    P = pd.DataFrame(d); P["PERS_EXT_MEAN"] = P[["PERS_EXT1", "PERS_EXT2", "PERS_EXT3"]].mean(axis=1)
    return pd.concat([X, new_cols({c: P[c].values for c in P.columns})], axis=1)
N_PARTNER_ENC = 6
def partner_enc(fit, idx):     # label-based: uses ONLY labels of rows in `fit` (so for training rows it is leave-self-out and, inside train_enc, nested)
    idx = np.asarray(idx); cols = ["PY_ANY", "PY_N", "PY_NEAR180", "PY_EARLIER", "PY_LATER", "PYL_ANY"]
    out = np.full((len(idx), 6), np.nan); out[:, 1] = 0
    if SMOKE or not USE_PARTNER: return pd.DataFrame(out, columns=cols)
    lab = np.full(len(al), np.nan); lab[fit] = y[fit]
    for j, i in enumerate(idx):
        if i in PL_S:
            o, gp_ = PL_S[i]; v = lab[o]; k = ~np.isnan(v)
            if k.any():
                v, gp_ = v[k], gp_[k]; out[j, 0] = v.max(); out[j, 1] = k.sum()
                if (np.abs(gp_) <= 180).any(): out[j, 2] = v[np.abs(gp_) <= 180].max()
                if (gp_ > 0).any(): out[j, 3] = v[gp_ > 0].max()
                if (gp_ < 0).any(): out[j, 4] = v[gp_ < 0].max()
        if i in PL_L:
            o, gp_ = PL_L[i]; v = lab[o]; v = v[~np.isnan(v)]
            if len(v): out[j, 5] = v.max()
    return pd.DataFrame(out, columns=cols)
FAMS = {"person": f_person, "partner": f_partner, "kdec": f_kdec, "clean": f_clean, "ext": f_ext, "fin": f_fin, "age": f_age, "bureau": f_bureau, "count": f_count, "group": f_group}
def build(fams):
    X = base(al)
    for f in fams: X = FAMS[f](al, X)
    X = X.replace([np.inf, -np.inf], np.nan)
    return X.iloc[:ntr].reset_index(drop=True), X.iloc[ntr:].reset_index(drop=True)
CORE = ["clean", "fin", "age", "bureau"] + (["partner"] if USE_PARTNER else []) + (["person"] if (USE_PARTNER and USE_PERSON) else []); ALLF = ["clean", "ext", "fin", "age", "bureau", "count", "group"] + (["partner"] if USE_PARTNER else []) + (["person"] if (USE_PARTNER and USE_PERSON) else [])
Xc, Xct = build(CORE); Xa, Xat = build(ALLF)                              # members WITHOUT k
Xck, Xckt = build(CORE + ["kdec"]); Xak, Xakt = build(ALLF + ["kdec"])   # members WITH decoded k (v2 members)
# keep the frames WITHOUT residual features; residuals are attached per experiment
BASEF = dict(Xc=Xc, Xct=Xct, Xa=Xa, Xat=Xat, Xck=Xck, Xckt=Xckt, Xak=Xak, Xakt=Xakt)
if SMOKE: y = y[:8000]
n, nt = len(y), len(Xct); tidx = np.arange(ntr, ntr + nt) if not SMOKE else np.arange(8000, 8000 + nt)
# ---- q, and the decoded product: annuity = Q0*credit + F  (F ~ 129.5 monthly fee on ~75% of cash loans); label-free ----
q = al.AMT_ANNUITY / al.AMT_CREDIT; _cash = (al.NAME_CONTRACT_TYPE == "Cash loans").values; _qb = (al.AMT_ANNUITY - 129.5) / al.AMT_CREDIT
_q5, _qb5 = q.round(5), _qb.round(5)
FEE = ((_qb5.map(_qb5.value_counts()).values > _q5.map(_q5.value_counts()).values) & _cash).astype(int); Q0 = pd.Series(np.where(FEE, _qb, q))
KS = {"Q3": q.round(3).astype(str), "Q4": q.round(4).astype(str), "Q5": q.round(5).astype(str),
      "P3": Q0.round(3).astype(str), "P4": Q0.round(4).astype(str), "P5": Q0.round(5).astype(str), "PF4": Q0.round(4).astype(str) + "|" + pd.Series(FEE).astype(str)}
M0 = {"Q3": 100., "Q4": 30., "Q5": 30., "P3": 100., "P4": 30., "P5": 30., "PF4": 30.}
# ---- COMMUNITY KEYS (research: rounded CREDIT/ANNUITY = interest-rate tiers; 1/2/3 decimals; combos; counts; smoothing 10/50/200) ----
if USE_COMMUNITY_KEYS:
    CT = al.AMT_CREDIT / al.AMT_ANNUITY            # CREDIT_TERM ~ number of payments
    CT1, CT2, CT3 = CT.round(1).astype(str), CT.round(2).astype(str), CT.round(3).astype(str)
    KS.update({"CT1": CT1, "CT2": CT2, "CT3": CT3,
               "CT1_s10": CT1, "CT1_s200": CT1,                                  # same key, different smoothing (community tried 10 and 200)
               "CT1_GP": CT1 + "|" + al.AMT_GOODS_PRICE.astype(str), "CT1_CR": CT1 + "|" + al.AMT_CREDIT.astype(str), "CT1_AN": CT1 + "|" + al.AMT_ANNUITY.astype(str),
               "GP": al.AMT_GOODS_PRICE.astype(str), "CR": al.AMT_CREDIT.astype(str), "AN": al.AMT_ANNUITY.astype(str)})
    M0.update({"CT1": 50., "CT2": 50., "CT3": 50., "CT1_s10": 10., "CT1_s200": 200., "CT1_GP": 50., "CT1_CR": 50., "CT1_AN": 50., "GP": 50., "CR": 50., "AN": 50.})
if SMOKE: KS = {k: v.iloc[list(range(8000)) + list(range(ntr, ntr + nt))].reset_index(drop=True) for k, v in KS.items()}
print("decoded products:", Q0.round(5).nunique(), "(q5 distinct:", q.round(5).nunique(), ") | fee share of cash loans:", round(float(FEE[_cash].mean()), 3), "| features core/all:", Xc.shape[1], Xa.shape[1])
import hashlib
_h = lambda a: hashlib.md5(np.ascontiguousarray(a).tobytes()).hexdigest()
KC = {k: pd.factorize(v)[0] for k, v in KS.items()}; KN = {k: int(v.max()) + 1 for k, v in KC.items()}     # integer-coded keys: bincount encoder is ~8x faster than pandas groupby on strings and bit-identical
_MEMO = {}                                                                                              # every one of the ~20 members needs the SAME fold encodings: compute them once (key includes a hash of the labels, so shuffled-label tests never reuse real ones)
def enc(fit, qi, keys=None, partner=True):
    mk = ("enc", _h(y), _h(fit), _h(qi), partner) if keys is None else None
    if mk is not None and mk in _MEMO: return _MEMO[mk]
    pr = float(y[fit].mean()); yf = y[fit].astype(float); out = {}
    for k in (keys or KS):
        c = KC[k]; cnt_t = np.bincount(c[fit], minlength=KN[k]).astype(float); sm_t = np.bincount(c[fit], weights=yf, minlength=KN[k]); cq = c[qi]
        out[f"TE_{k}"] = (sm_t[cq] + M0[k] * pr) / (cnt_t[cq] + M0[k]); out[f"CNT_{k}"] = cnt_t[cq]
    res = pd.concat([pd.DataFrame(out), partner_enc(fit, qi)], axis=1) if (keys is None and partner) else pd.DataFrame(out)
    if mk is not None: _MEMO[mk] = res
    return res
def train_enc(a):     # nested out-of-fold encodings for training rows (a row's own label never enters its own feature)
    mk = ("tr", _h(y), _h(a))
    if mk in _MEMO: return _MEMO[mk]
    cols = enc(a[:10], a[:5]).columns; inner = np.zeros((len(a), len(cols)))
    for ia, ib in StratifiedKFold(5, shuffle=True, random_state=7).split(a, y[a]): inner[ib] = enc(a[ia], a[ib]).values
    _MEMO[mk] = pd.DataFrame(inner, columns=cols); return _MEMO[mk]
def onehot(Xa_, Xt_):
    z = pd.concat([Xa_, Xt_], ignore_index=True); oh = pd.get_dummies(z[CATS].astype("category"), dummy_na=True, dtype=np.float32)
    oh.columns = [f"OH_{i}" for i in range(oh.shape[1])]; Z = pd.concat([z.drop(columns=CATS), oh], axis=1)
    return Z.iloc[:len(Xa_)].reset_index(drop=True), Z.iloc[len(Xa_):].reset_index(drop=True)
Xc, Xa = BASEF["Xc"], BASEF["Xa"]   # for the print above
BASE = dict(objective="binary", force_col_wise=True, bagging_freq=1, verbose=-1, num_threads=THREADS, seed=42)
rk = lambda v: (rankdata(v) - 1) / (len(v) - 1)
FOLDS = 2 if SMOKE else 5
W = {"qf_shallow": 0.24, "qf_reg": 0.12, "v1_allfam_oh": 0.32, "v2_allfam_oh": 0.16, "v2_mono": 0.16}   # the fixed weights of the submitted blend

# ---- learned-expectation (residual) features; seeds = how many different fold splits are averaged (E2 uses 3) ----
from sklearn.ensemble import HistGradientBoostingRegressor as HR, HistGradientBoostingClassifier as HC
from sklearn.model_selection import KFold, train_test_split
def compute_resid(seeds=(0,)):
    raw = al.drop(columns=["SK_ID_CURR"]).copy()
    for c in CATS: raw[c] = raw[c].astype("category").cat.codes.replace(-1, np.nan).astype(float)
    raw["DAYS_EMPLOYED"] = raw.DAYS_EMPLOYED.where(raw.DAYS_EMPLOYED != 365243)
    demo = [c for c in raw.columns if c.startswith("EXT_SOURCE") or c.startswith("AMT_")]
    T = {"EXT1": (raw.EXT_SOURCE_1, ["EXT_SOURCE_1"]), "EXT2": (raw.EXT_SOURCE_2, ["EXT_SOURCE_2"]), "EXT3": (raw.EXT_SOURCE_3, ["EXT_SOURCE_3"]),
         "LOGINC": (np.log(raw.AMT_INCOME_TOTAL), ["AMT_INCOME_TOTAL"]), "LOGCR": (np.log(raw.AMT_CREDIT), ["AMT_CREDIT", "AMT_ANNUITY", "AMT_GOODS_PRICE"]),
         "EMP": (raw.DAYS_EMPLOYED, ["DAYS_EMPLOYED"]), "REG": (raw.DAYS_REGISTRATION, ["DAYS_REGISTRATION"]), "AGE": (raw.DAYS_BIRTH, ["DAYS_BIRTH"]),
         "IDP": (raw.DAYS_ID_PUBLISH, ["DAYS_ID_PUBLISH"]), "PHONE": (raw.DAYS_LAST_PHONE_CHANGE, ["DAYS_LAST_PHONE_CHANGE"]), "REGPOP": (raw.REGION_POPULATION_RELATIVE, ["REGION_POPULATION_RELATIVE"]),
         "FAM": (raw.CNT_FAM_MEMBERS, ["CNT_FAM_MEMBERS"]), "CAR": (raw.OWN_CAR_AGE, ["OWN_CAR_AGE"]),
         "EXT1D": (raw.EXT_SOURCE_1, demo), "EXT2D": (raw.EXT_SOURCE_2, demo), "EXT3D": (raw.EXT_SOURCE_3, demo)}
    if SMOKE: T = dict(list(T.items())[:2])
    out = {}; t0 = time.time()
    for nm, (yt, drop_cols) in T.items():
        Xp = raw.drop(columns=drop_cols); pred = np.zeros(len(raw)); obs = yt.notna().values
        for sd in seeds:
            for tri, tei in KFold(2 if SMOKE else 5, shuffle=True, random_state=sd).split(Xp):
                m = HR(max_iter=5 if SMOKE else 120, learning_rate=0.1, max_leaf_nodes=31, min_samples_leaf=100, random_state=0).fit(Xp.iloc[tri][obs[tri]], yt.iloc[tri][obs[tri]]); pred[tei] += m.predict(Xp.iloc[tei]) / len(seeds)
        out[f"RP_{nm}"] = pred; out[f"RR_{nm}"] = np.where(obs, yt.values - pred, np.nan); print(f"  residual target {nm} ({time.time()-t0:.0f}s)", flush=True)
    return pd.DataFrame(out)
def get_resid(seeds, name):
    path = os.path.join(CACHE_DIR, name)
    old = next((p_ for p_ in [os.path.join(base_dir, "gci_cache_v6", "resid16.pkl"), os.path.join(OLD_CACHE, "resid16.pkl")] if os.path.exists(p_) and REUSE_V15_CACHE), "/nonexistent")
    if not SMOKE and os.path.exists(path): return pd.read_pickle(path)
    if not SMOKE and seeds == (0,) and os.path.exists(old): return pd.read_pickle(old)       # reuse the v6 cache (identical recipe)
    R = compute_resid(seeds)
    if not SMOKE: R.to_pickle(path)
    return R
def with_resid(R):
    R_tr, R_te = R.iloc[:ntr].reset_index(drop=True), R.iloc[ntr:].reset_index(drop=True)
    assert len(R) == ntr + len(BASEF["Xct"])
    return {k: pd.concat([BASEF[k], R_te if k.endswith("t") else R_tr], axis=1) for k in BASEF}
def build_members(f):
    Xc_, Xct_, Xa_, Xat_, Xck_, Xckt_, Xak_, Xakt_ = [f[k] for k in ("Xc", "Xct", "Xa", "Xat", "Xck", "Xckt", "Xak", "Xakt")]
    if SMOKE: Xc_, Xa_, Xck_, Xak_ = [d.iloc[:8000] for d in (Xc_, Xa_, Xck_, Xak_)]
    Xco, Xcot = onehot(Xc_, Xct_); Xao, Xaot = onehot(Xa_, Xat_); Xaok, Xaokt = onehot(Xak_, Xakt_)
    MONO_K = [-1 if c in ("EXT_SOURCE_1", "EXT_SOURCE_2", "EXT_SOURCE_3") else 0 for c in Xck_.columns] + [0] * (2 * len(KS) + N_PARTNER_ENC)
    return {
     "qf_shallow":   (Xco,  Xcot,  dict(learning_rate=0.05, num_leaves=7, max_depth=3, min_child_samples=150, feature_fraction=0.5, bagging_fraction=0.8, lambda_l2=20), 1300, 1430, [42, 7, 2024, 11, 99]),
     "qf_reg":       (Xco,  Xcot,  dict(learning_rate=0.03, num_leaves=11, min_child_samples=300, feature_fraction=0.4, bagging_fraction=0.8, lambda_l2=40), 1939, 2133, [42, 7, 2024, 11, 99]),
     "v1_allfam_oh": (Xao,  Xaot,  dict(learning_rate=0.04, num_leaves=15, min_child_samples=150, feature_fraction=0.3, bagging_fraction=0.8, lambda_l2=20), 881, 969, [42, 7, 2024]),
     "v2_allfam_oh": (Xaok, Xaokt, dict(learning_rate=0.04, num_leaves=15, min_child_samples=150, feature_fraction=0.3, bagging_fraction=0.8, lambda_l2=20), 881, 969, [42, 7, 2024]),
     "v2_mono":      (Xck_.reset_index(drop=True), Xckt_.reset_index(drop=True), dict(learning_rate=0.02, num_leaves=15, min_child_samples=150, feature_fraction=0.5, bagging_fraction=0.8, lambda_l2=20, monotone_constraints=MONO_K), 1571, 1729, [42, 7, 2024])}
def run_member(name, spec, tag, view=None):
    X, Xt, p, R, Rf, seeds = spec[name]; f = os.path.join(CACHE_DIR, f"{tag}_{name}.npz")
    if view is not None and "monotone_constraints" in p: p = {**p, "monotone_constraints": list(p["monotone_constraints"]) + [0] * view["tr"][0].shape[1]}   # extra view columns are unconstrained
    if os.path.exists(f) and not SMOKE:
        z = np.load(f); print(f"{tag}_{name}: loaded from cache | held-out AUC {roc_auc_score(y, z['oof']):.5f}"); return z["test"], z["oof"]
    tf = np.zeros(nt); oof = np.zeros(n); t0 = time.time(); ex = lambda key, k=None: ([] if view is None else [view[key] if k is None else view[key][k]])
    for k, (a, b) in enumerate(StratifiedKFold(FOLDS, shuffle=True, random_state=42).split(np.zeros(n), y)):
        fa = pd.concat([X.iloc[a].reset_index(drop=True), train_enc(a)] + ex("tr", k), axis=1); fb = pd.concat([X.iloc[b].reset_index(drop=True), enc(a, b)] + ex("va", k), axis=1); ft = pd.concat([Xt.reset_index(drop=True), enc(a, tidx)] + ex("te", k), axis=1)
        assert "monotone_constraints" not in p or len(p["monotone_constraints"]) == fa.shape[1], f"monotone constraints ({len(p['monotone_constraints'])}) != features ({fa.shape[1]})"
        m = lgb.train({**BASE, **p}, lgb.Dataset(fa, y[a]), 60 if SMOKE else R); oof[b] = m.predict(fb); tf += m.predict(ft) / FOLDS
        print(f"  {tag}_{name} fold {k+1}/{FOLDS}: AUC {roc_auc_score(y[b], oof[b]):.5f} ({time.time()-t0:.0f}s)", flush=True)
    allr = np.arange(n); fa = pd.concat([X.reset_index(drop=True), train_enc(allr)] + ex("all_tr"), axis=1); ft = pd.concat([Xt.reset_index(drop=True), enc(allr, tidx)] + ex("all_te"), axis=1); fr = []
    for sd in ([42] if SMOKE else seeds):
        fr.append(rk(lgb.train({**BASE, **p, "seed": sd}, lgb.Dataset(fa, y), 60 if SMOKE else Rf).predict(ft))); print(f"  {tag}_{name} full-data seed {sd} done ({time.time()-t0:.0f}s)", flush=True)
    test = rk(0.5 * rk(tf) + 0.5 * np.mean(fr, 0)); print(f"{tag}_{name}: held-out AUC {roc_auc_score(y, oof):.5f}", flush=True)
    if not SMOKE: np.savez(f, test=test, oof=oof)
    return test, oof
RESV = {}
def nested_generic(a, fn):      # nested out-of-fold wrapper for any label-based encoder fn(fit_idx, apply_idx) -> DataFrame (used by CatBoost cell and the ablation)
    parts = []
    for ia, ib in StratifiedKFold(5, shuffle=True, random_state=7).split(a, y[a]):
        f = fn(a[ia], a[ib]); f.index = ib; parts.append(f)
    return pd.concat(parts).sort_index().reset_index(drop=True)
print("keys:", list(KS), "| partner rows (strict):", int((~np.isnan(G_S)).sum()), "| loose:", int((~np.isnan(G_L)).sum()))
