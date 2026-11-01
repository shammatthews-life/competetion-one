from __future__ import annotations

import hashlib
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm, rankdata, spearmanr
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


WORKSPACE = Path(__file__).resolve().parents[1]
ARTIFACTS = WORKSPACE / "baseline" / "artifacts"
RUNS = WORKSPACE / "experiments" / "runs"
RAW_TRAIN = Path(r"D:\gci\inpuy\train.csv")
RAW_TEST = Path(r"D:\gci\inpuy\test.csv")
SEED = 20261101
C_VALUE = 0.1
MIN_MEAN_GAIN = 0.0002
MIN_POSITIVE_FOLDS = 4
MIN_GAIN_TO_STANDARD_ERROR = 2.0
MEMBERS = (
    "v17_e1Aet",
    "xgb10",
    "v17_final",
    "v55_cbprod",
    "v17_e1et",
    "v55_deep_et",
    "v55_allfam",
    "v55_v2_mono",
)
FIXED_WEIGHTS = np.asarray(
    [0.2932, 0.2596, 0.2051, 0.0874, 0.0489, 0.0442, 0.0383, 0.0233],
    dtype=np.float64,
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def ranked_probit(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64).ravel()
    if not np.isfinite(values).all():
        raise ValueError("Prediction stream contains non-finite values")
    uniform = (rankdata(values, method="average") - 0.5) / len(values)
    return norm.ppf(np.clip(uniform, 1e-6, 1.0 - 1e-6))


def load_streams() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    v17_path = ARTIFACTS / "v_17_full_suite_bundle.npz"
    v55_path = ARTIFACTS / "v55_members.npz"
    xgb_oof_path = ARTIFACTS / "oof_10f_xgb.npy"
    xgb_test_path = ARTIFACTS / "test_10f_xgb.npy"
    labels_path = ARTIFACTS / "y_train.npy"
    with np.load(v17_path, allow_pickle=False) as v17:
        oof_values = {
            "v17_e1Aet": v17["oof_e1Aet"],
            "v17_final": v17["final_oof"],
            "v17_e1et": v17["oof_e1et"],
        }
        test_values = {
            "v17_e1Aet": v17["test_e1Aet"],
            "v17_final": v17["final_test"],
            "v17_e1et": v17["test_e1et"],
        }
    with np.load(v55_path, allow_pickle=False) as v55:
        oof_values.update(
            {
                "v55_cbprod": v55["oof_cb_product"],
                "v55_deep_et": v55["oof_deep_et"],
                "v55_allfam": v55["oof_v2_allfam_oh"],
                "v55_v2_mono": v55["oof_v2_mono"],
            }
        )
        test_values.update(
            {
                "v55_cbprod": v55["test_cb_product"],
                "v55_deep_et": v55["test_deep_et"],
                "v55_allfam": v55["test_v2_allfam_oh"],
                "v55_v2_mono": v55["test_v2_mono"],
            }
        )
    oof_values["xgb10"] = np.load(xgb_oof_path, allow_pickle=False)
    test_values["xgb10"] = np.load(xgb_test_path, allow_pickle=False)
    y = np.load(labels_path, allow_pickle=False).astype(np.int32)
    if not RAW_TRAIN.is_file():
        raise FileNotFoundError(f"Raw training labels are required: {RAW_TRAIN}")
    raw_y = pd.read_csv(RAW_TRAIN, usecols=["TARGET"])["TARGET"].to_numpy()
    if not np.array_equal(y, raw_y):
        raise ValueError("Frozen label array differs from raw train.csv TARGET")
    x_oof = np.column_stack([ranked_probit(oof_values[name]) for name in MEMBERS])
    x_test = np.column_stack([ranked_probit(test_values[name]) for name in MEMBERS])
    if x_oof.shape != (len(y), len(MEMBERS)):
        raise ValueError(f"Unexpected OOF stack shape: {x_oof.shape}")
    if x_test.shape != (61_500, len(MEMBERS)):
        raise ValueError(f"Unexpected test stack shape: {x_test.shape}")
    if not np.isfinite(x_oof).all() or not np.isfinite(x_test).all():
        raise ValueError("Rank-transformed feature matrix contains non-finite values")
    return x_oof, x_test, y


def main() -> None:
    if not (WORKSPACE / "baseline" / "BASELINE_MANIFEST.json").is_file():
        raise FileNotFoundError("Run source\\prepare_baseline.py first.")
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_dir = RUNS / f"{run_id}_exp01_oof_meta_weights"
    run_dir.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    x_oof, x_test, y = load_streams()
    if set(np.unique(y)) != {0, 1}:
        raise ValueError("Labels must be binary 0/1")

    fixed_weights = FIXED_WEIGHTS / FIXED_WEIGHTS.sum()
    fixed_score = x_oof @ fixed_weights
    splitter = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    candidate_oof = np.full(len(y), np.nan, dtype=np.float64)
    fold_results = []
    fold_membership = np.full(len(y), -1, dtype=np.int8)

    for fold_number, (fit_idx, valid_idx) in enumerate(
        splitter.split(x_oof, y), start=1
    ):
        if np.intersect1d(fit_idx, valid_idx).size:
            raise RuntimeError("Meta fit/validation indices overlap")
        model = make_pipeline(
            StandardScaler(),
            LogisticRegression(
                C=C_VALUE,
                penalty="l2",
                solver="lbfgs",
                max_iter=500,
                random_state=SEED + fold_number,
            ),
        )
        model.fit(x_oof[fit_idx], y[fit_idx])
        candidate_oof[valid_idx] = model.decision_function(x_oof[valid_idx])
        fold_membership[valid_idx] = fold_number - 1
        fold_results.append(
            {
                "fold": fold_number,
                "fit_rows": int(len(fit_idx)),
                "valid_rows": int(len(valid_idx)),
                "fixed_suite_auc": float(
                    roc_auc_score(y[valid_idx], fixed_score[valid_idx])
                ),
                "meta_stack_auc": float(
                    roc_auc_score(y[valid_idx], candidate_oof[valid_idx])
                ),
                "paired_auc_delta": float(
                    roc_auc_score(y[valid_idx], candidate_oof[valid_idx])
                    - roc_auc_score(y[valid_idx], fixed_score[valid_idx])
                ),
            }
        )

    if not np.isfinite(candidate_oof).all() or (fold_membership < 0).any():
        raise RuntimeError("Meta OOF predictions do not cover every training row")
    deltas = np.asarray(
        [result["paired_auc_delta"] for result in fold_results],
        dtype=np.float64,
    )
    mean_delta = float(deltas.mean())
    standard_error = float(deltas.std(ddof=1) / np.sqrt(len(deltas)))
    positives = int((deltas > 0).sum())
    gate_passed = (
        mean_delta >= MIN_MEAN_GAIN
        and positives >= MIN_POSITIVE_FOLDS
        and mean_delta > MIN_GAIN_TO_STANDARD_ERROR * standard_error
    )
    report: dict[str, object] = {
        "experiment_id": run_id,
        "classification": "HISTORICAL_CACHE_ASSISTED_META_VALIDATION",
        "status": "PASS_SCREEN" if gate_passed else "REJECTED_SCREEN",
        "hypothesis": (
            "An L2-regularized logistic stack over the existing eight OOF "
            "streams may generalize better than the fixed historical weights."
        ),
        "configuration": {
            "outer_folds": 5,
            "shuffle": True,
            "seed": SEED,
            "standardization": "fit on outer training rows only",
            "logistic_C": C_VALUE,
            "fixed_weights_raw": FIXED_WEIGHTS.tolist(),
            "members": list(MEMBERS),
        },
        "execution_identity": {
            "source_path": str(Path(__file__).resolve()),
            "source_sha256": sha256_file(Path(__file__).resolve()),
            "configuration_sha256": hashlib.sha256(
                json.dumps(
                    {
                        "seed": SEED,
                        "C": C_VALUE,
                        "members": MEMBERS,
                        "fixed_weights": FIXED_WEIGHTS.tolist(),
                        "outer_folds": 5,
                    },
                    sort_keys=True,
                ).encode("utf-8")
            ).hexdigest(),
            "raw_train_sha256": sha256_file(RAW_TRAIN),
            "raw_test_sha256": sha256_file(RAW_TEST),
        },
        "validation": {
            "fixed_suite_pooled_auc": float(roc_auc_score(y, fixed_score)),
            "crossfit_meta_pooled_auc": float(roc_auc_score(y, candidate_oof)),
            "mean_paired_fold_delta": mean_delta,
            "paired_fold_delta_standard_error": standard_error,
            "positive_folds": positives,
            "folds": fold_results,
            "gate": {
                "minimum_mean_gain": MIN_MEAN_GAIN,
                "minimum_positive_folds": MIN_POSITIVE_FOLDS,
                "minimum_gain_to_standard_error": MIN_GAIN_TO_STANDARD_ERROR,
                "passed": gate_passed,
            },
            "OOF_fold_sidecars_available": False,
        },
        "provenance_limitations": [
            "Historical V17/V55/XGB10 OOF artifacts are inputs; original fold sidecars and producer run manifest are unavailable.",
            "The historical fixed weights were selected using the complete OOF label set; this makes the fixed-weight fold comparator optimistic and the paired-fold comparison non-nested.",
            "This is not an independent raw-data reconstruction.",
            "The historical 85% Anchor74 has no verified aligned OOF vector; this experiment evaluates the 15% suite, not the full champion.",
            "No historical test prediction was used as a label or tuning target.",
        ],
        "inputs": {
            name: {
                "path": str(ARTIFACTS / name),
                "sha256": sha256_file(ARTIFACTS / name),
            }
            for name in (
                "v_17_full_suite_bundle.npz",
                "v55_members.npz",
                "oof_10f_xgb.npy",
                "test_10f_xgb.npy",
                "y_train.npy",
            )
        },
        "runtime": {
            "elapsed_seconds": time.perf_counter() - started,
            "device": "CPU; rank transforms and logistic meta-fit do not use GPU",
        },
    }
    np.save(run_dir / "fixed_suite_oof.npy", fixed_score, allow_pickle=False)
    np.save(run_dir / "crossfit_meta_oof.npy", candidate_oof, allow_pickle=False)
    np.save(run_dir / "fold_membership.npy", fold_membership, allow_pickle=False)
    report_path = run_dir / "experiment_report.json"
    report_path.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    report["outputs"] = {
        path.name: sha256_file(path)
        for path in run_dir.iterdir()
        if path.is_file() and path != report_path
    }

    if gate_passed:
        final_model = make_pipeline(
            StandardScaler(),
            LogisticRegression(
                C=C_VALUE,
                penalty="l2",
                solver="lbfgs",
                max_iter=500,
                random_state=SEED,
            ),
        )
        final_model.fit(x_oof, y)
        test_score = final_model.decision_function(x_test)
        suite_test_probit = ranked_probit(test_score)
        anchor = pd.read_csv(
            ARTIFACTS / "v_20261027_precision_transductive_apex.csv"
        )
        if len(anchor) != len(test_score):
            raise ValueError("Anchor74 and candidate test row counts differ")
        raw_test_ids = pd.read_csv(
            RAW_TEST, usecols=["SK_ID_CURR"]
        )["SK_ID_CURR"].to_numpy()
        anchor_ids = anchor["SK_ID_CURR"].to_numpy()
        if not np.array_equal(anchor_ids, raw_test_ids):
            raise ValueError("Anchor74 IDs do not match raw test.csv order")
        anchor_test_probit = norm.ppf(
            np.clip(
                (rankdata(anchor["TARGET"].to_numpy(), method="average") - 0.5)
                / len(anchor),
                1e-6,
                1.0 - 1e-6,
            )
        )
        blended = 0.85 * anchor_test_probit + 0.15 * suite_test_probit
        ranks = rankdata(blended, method="ordinal")
        candidate = pd.DataFrame(
            {
                "SK_ID_CURR": anchor_ids,
                "TARGET": (ranks - 0.5) / len(ranks),
            }
        )
        if (
            not candidate["SK_ID_CURR"].is_unique
            or not np.array_equal(candidate["SK_ID_CURR"].to_numpy(), raw_test_ids)
            or not np.isfinite(candidate["TARGET"].to_numpy()).all()
            or ((candidate["TARGET"] < 0.0) | (candidate["TARGET"] > 1.0)).any()
        ):
            raise ValueError("Experimental candidate failed submission validation")
        candidate_path = run_dir / "candidate_submission.csv"
        candidate.to_csv(candidate_path, index=False, float_format="%.10f")
        report["candidate"] = {
            "path": str(candidate_path),
            "sha256": sha256_file(candidate_path),
            "rows": len(candidate),
            "classification": "HISTORICAL_CACHE_ASSISTED_EXPERIMENTAL_CANDIDATE",
            "official_score": "NOT_EVALUATED",
        }
    else:
        report["candidate"] = None
        report["decision"] = (
            "No test candidate generated because the predeclared OOF gate failed."
        )

    report["outputs"] = {
        path.name: sha256_file(path)
        for path in run_dir.iterdir()
        if path.is_file() and path != report_path
    }
    report["runtime"]["elapsed_seconds"] = time.perf_counter() - started
    report_path.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (run_dir / "experiment_report.md").write_text(
        "# EXP01: OOF meta-weight screen\n\n"
        f"- Status: **{report['status']}**\n"
        f"- Fixed suite pooled OOF AUC: "
        f"`{report['validation']['fixed_suite_pooled_auc']:.10f}`\n"
        f"- Cross-fit meta pooled OOF AUC: "
        f"`{report['validation']['crossfit_meta_pooled_auc']:.10f}`\n"
        f"- Mean paired-fold delta: `{mean_delta:+.10f}` "
        f"(SE `{standard_error:.10f}`; {positives}/5 positive folds)\n"
        f"- Candidate generated: `{report['candidate'] is not None}`\n"
        "- Classification: historical-cache-assisted meta-validation; not an "
        "independent raw rebuild or official score.\n",
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
