from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm, pearsonr, rankdata, spearmanr


CHAMPION = Path(r"D:\gci\champion")
FINAL = CHAMPION / "EXP85_FINAL_RECONSTRUCTION"
DATA = Path(r"D:\gci\inpuy")
HIST_CACHE = CHAMPION / "01_EXP85_RECOVERY" / "cache"
PHASE3 = Path(r"D:\gci\gemini\champion\phase3_reconstruction")
GEMINI = Path(r"D:\gci\gemini")
EXPECTED_TRAIN_SHA = "b8dd5a61771d8eb2821972c1b7d3f2f747cc66471f4d34153036d2d9c458ecf1"
EXPECTED_TEST_SHA = "dfb7aeb96e5af3d8616acaaba6b23cec8680845cfd525f7c08de15c6d2d42c65"
EXPECTED_CHAMPION_SHA = "78c31bb4e4534330479faa26861a49559dcadc40c7eac45620b92d7d9e6ea947"

WEIGHTS = {
    "v17_e1Aet": 0.2932,
    "xgb10": 0.2596,
    "v17_final": 0.2051,
    "v55_cbprod": 0.0874,
    "v17_e1et": 0.0489,
    "v55_deep_et": 0.0442,
    "v55_allfam": 0.0383,
    "v55_v2_mono": 0.0233,
}

ANCHORS = {
    "historical_anchor": GEMINI / "v_20261027_precision_transductive_apex.csv",
    "phase3_independent_anchor": (
        PHASE3 / "independent_rebuild" / "v_20261027_precision_transductive_apex.csv"
    ),
    "champion_artifacts_anchor": (
        CHAMPION / "02_EXP85_REBUILT" / "artifacts"
        / "v_20261027_precision_transductive_apex.csv"
    ),
    "rebuilt_artifacts_anchor": (
        CHAMPION / "02_EXP85_REBUILT" / "rebuilt_artifacts"
        / "v_20261027_precision_transductive_apex.csv"
    ),
    "clean_room_anchor": (
        CHAMPION / "02_EXP85_REBUILT" / "rebuilt_artifacts" / "clean_room_build"
        / "v_20261027_precision_transductive_apex.csv"
    ),
    "current_production_anchor": (
        FINAL / "runs" / "recovered_anchor74_1793444526421"
        / "v_20261027_precision_transductive_apex.csv"
    ),
}

CURRENT_COMPONENTS = {
    "v17": FINAL / "runs" / "recovered_v17_1793443853760"
    / "v_17_full_suite_bundle.npz",
    "v55": FINAL / "runs" / "recovered_v55_1793443036653" / "v55_members.npz",
    "xgb10": FINAL / "runs" / "recovered_xgb10_1793441428099" / "test_10f_xgb.npy",
}
HIST_COMPONENTS = {
    "v17": HIST_CACHE / "v_17_full_suite_bundle.npz",
    "v55": HIST_CACHE / "v55_members.npz",
    "xgb10": HIST_CACHE / "test_10f_xgb.npy",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def metrics(left: np.ndarray, right: np.ndarray) -> dict:
    left = np.asarray(left, dtype=np.float64)
    right = np.asarray(right, dtype=np.float64)
    if left.shape != right.shape or left.ndim != 1:
        raise ValueError(f"Metric inputs have incompatible shapes: {left.shape}, {right.shape}")
    left_rank = rankdata(left, method="average")
    right_rank = rankdata(right, method="average")
    displacement = np.abs(left_rank - right_rank)
    return {
        "rows": int(left.size),
        "pearson": float(pearsonr(left, right).statistic),
        "spearman": float(spearmanr(left, right).statistic),
        "mae": float(np.mean(np.abs(left - right))),
        "rmse": float(np.sqrt(np.mean(np.square(left - right)))),
        "max_abs_difference": float(np.max(np.abs(left - right))),
        "mean_rank_displacement": float(np.mean(displacement)),
        "max_rank_displacement": float(np.max(displacement)),
    }


def read_submission(path: Path, expected_ids: np.ndarray) -> tuple[pd.DataFrame, np.ndarray]:
    frame = pd.read_csv(path, usecols=["SK_ID_CURR", "TARGET"])
    if not frame["SK_ID_CURR"].is_unique:
        raise ValueError(f"Duplicate IDs in {path}")
    missing = set(expected_ids) - set(frame["SK_ID_CURR"])
    extra = set(frame["SK_ID_CURR"]) - set(expected_ids)
    if missing or extra or len(frame) != len(expected_ids):
        raise ValueError(f"ID set mismatch in {path}: missing={len(missing)}, extra={len(extra)}")
    aligned = frame.set_index("SK_ID_CURR").reindex(expected_ids)
    if aligned["TARGET"].isna().any():
        raise ValueError(f"Null prediction after ID alignment for {path}")
    return frame, aligned["TARGET"].to_numpy(dtype=np.float64)


def read_anchor(path: Path, expected_ids: np.ndarray) -> np.ndarray:
    _, prediction = read_submission(path, expected_ids)
    return prediction


def load_suite(paths: dict[str, Path], n_test: int) -> dict[str, np.ndarray]:
    with np.load(paths["v17"], allow_pickle=False) as v17, np.load(
        paths["v55"], allow_pickle=False
    ) as v55:
        xgb10 = np.load(paths["xgb10"], allow_pickle=False)
        members = {
            "v17_e1Aet": np.asarray(v17["test_e1Aet"]),
            "xgb10": np.asarray(xgb10),
            "v17_final": np.asarray(v17["final_test"]),
            "v55_cbprod": np.asarray(v55["test_cb_product"]),
            "v17_e1et": np.asarray(v17["test_e1et"]),
            "v55_deep_et": np.asarray(v55["test_deep_et"]),
            "v55_allfam": np.asarray(v55["test_v2_allfam_oh"]),
            "v55_v2_mono": np.asarray(v55["test_v2_mono"]),
        }
    for name, values in members.items():
        if values.shape != (n_test,) or not np.isfinite(values).all():
            raise ValueError(f"Invalid suite member {name}: {values.shape}")
    return members


def suite_probit(members: dict[str, np.ndarray]) -> np.ndarray:
    total_weight = sum(WEIGHTS.values())
    combined = np.zeros_like(next(iter(members.values())), dtype=np.float64)
    for name, weight in WEIGHTS.items():
        values = members[name]
        uniform = (rankdata(values, method="average") - 0.5) / len(values)
        combined += (weight / total_weight) * norm.ppf(np.clip(uniform, 1e-6, 1.0 - 1e-6))
    return combined


def compose(anchor: np.ndarray, suite_z: np.ndarray) -> np.ndarray:
    z_anchor = norm.ppf(np.clip(anchor, 1e-6, 1.0 - 1e-6))
    z_macro = 0.85 * z_anchor + 0.15 * suite_z
    return (rankdata(z_macro, method="ordinal") - 0.5) / len(anchor)


def file_record(path: Path) -> dict:
    if not path.is_file():
        return {"path": str(path), "exists": False}
    return {
        "path": str(path),
        "exists": True,
        "size_bytes": path.stat().st_size,
        "sha256": sha256(path),
    }


def compare_feature_files(rebuilt: Path, historical: Path) -> dict:
    rebuilt_frame = pd.read_feather(rebuilt)
    historical_frame = pd.read_feather(historical)
    if rebuilt_frame.shape != historical_frame.shape:
        raise ValueError(f"Feature shape mismatch: {rebuilt}, {historical}")
    rebuilt_values = rebuilt_frame.to_numpy()
    historical_values = historical_frame.to_numpy()
    numeric = np.issubdtype(rebuilt_values.dtype, np.number)
    return {
        "rebuilt": file_record(rebuilt),
        "historical_cache": file_record(historical),
        "shape": list(rebuilt_frame.shape),
        "column_order_identical": list(rebuilt_frame.columns) == list(historical_frame.columns),
        "dtypes_identical": rebuilt_frame.dtypes.astype(str).equals(
            historical_frame.dtypes.astype(str)
        ),
        "values_identical": rebuilt_frame.equals(historical_frame),
        "rebuilt_dtype_counts": {
            str(name): int(count)
            for name, count in rebuilt_frame.dtypes.astype(str).value_counts().items()
        },
        "rebuilt_nan_cells": int(rebuilt_frame.isna().to_numpy().sum()),
        "rebuilt_infinite_cells": (
            int(np.isinf(rebuilt_values).sum()) if numeric else None
        ),
    }


def main() -> None:
    output_dir = Path(__file__).resolve().parent
    test_path = DATA / "test.csv"
    train_path = DATA / "train.csv"
    test_ids = pd.read_csv(test_path, usecols=["SK_ID_CURR"])["SK_ID_CURR"].to_numpy()
    if len(test_ids) != 61500 or not pd.Index(test_ids).is_unique:
        raise ValueError("Raw test IDs are not 61,500 unique rows.")
    if not np.array_equal(test_ids, np.arange(171202, 232702)):
        raise ValueError("Raw test IDs no longer match their validated sequential order.")

    champion_path = CHAMPION / "00_HISTORICAL_REFERENCE" / "EXP85_HISTORICAL_CHAMPION.csv"
    highcorr_path = CHAMPION / "08_SUBMISSIONS" / "submission_exp85_recovery_v5_rebuilt.csv"
    duplicate_highcorr_path = (
        CHAMPION / "02_EXP85_REBUILT" / "submissions"
        / "submission_exp85_recovery_v5_rebuilt.csv"
    )
    current_path = (
        FINAL / "runs" / "track1b_1793445445960" / "submission_track1b.csv"
    )
    _, champion = read_submission(champion_path, test_ids)
    _, highcorr = read_submission(highcorr_path, test_ids)
    _, current = read_submission(current_path, test_ids)
    _, duplicate_highcorr = read_submission(duplicate_highcorr_path, test_ids)

    historical_members = load_suite(HIST_COMPONENTS, len(test_ids))
    current_members = load_suite(CURRENT_COMPONENTS, len(test_ids))
    historical_suite_z = suite_probit(historical_members)
    current_suite_z = suite_probit(current_members)
    anchors = {
        name: read_anchor(path, test_ids)
        for name, path in ANCHORS.items()
        if path.is_file()
    }

    anchors_against_historical = {}
    for name, prediction in anchors.items():
        anchors_against_historical[name] = metrics(
            prediction, anchors["historical_anchor"]
        )

    compositions = {}
    for anchor_name, anchor in anchors.items():
        for suite_name, suite_z in (
            ("historical_cached_suite", historical_suite_z),
            ("current_production_suite", current_suite_z),
        ):
            prediction = compose(anchor, suite_z)
            compositions[f"{anchor_name}+{suite_name}"] = {
                "vs_frozen_champion": metrics(prediction, champion),
                "vs_existing_high_correlation_output": metrics(prediction, highcorr),
                "vs_existing_current_track1b": metrics(prediction, current),
                "matches_existing_highcorr_values_after_10dp_rounding": int(
                    np.count_nonzero(np.round(prediction, 10) == highcorr)
                ),
                "matches_existing_current_values_after_10dp_rounding": int(
                    np.count_nonzero(np.round(prediction, 10) == current)
                ),
            }

    member_comparisons = {
        name: {
            "interpretation": "positional_only_npz_has_no_SK_ID_CURR",
            "metrics": metrics(historical_members[name], current_members[name]),
        }
        for name in WEIGHTS
    }

    source_paths = [
        CHAMPION / "02_EXP85_REBUILT" / "notebooks" / "exp85_recovery_v5.ipynb",
        CHAMPION / "create_submission_recovery_v5_rebuilt.py",
        CHAMPION / "02_EXP85_REBUILT" / "source" / "reconstruct_transductive_pyramid.py",
        PHASE3 / "experiments" / "reconstruct_transductive_pyramid.py",
        PHASE3 / "experiments" / "reconstruct_hasan_features.py",
        PHASE3 / "experiments" / "train_independent_zoo.py",
        PHASE3 / "experiments" / "run_hasan_600_builder.py",
        PHASE3 / "experiments" / "run_workstream2_screening.py",
        GEMINI / "champion" / "cacherecreation" / "recreate_v17_bundle.py",
        GEMINI / "champion" / "cacherecreation" / "recreate_v55_members.py",
        GEMINI / "champion" / "cacherecreation" / "recreate_xgb10_and_labels.py",
        GEMINI / "champion" / "cacherecreation" / "recreate_precision_anchor.py",
        CHAMPION / "02_EXP85_REBUILT" / "source" / "generate_anchor74_from_scratch.py",
        FINAL / "src" / "build_track1b.py",
        FINAL / "src" / "generate_recovered_component.py",
        FINAL / "src" / "generate_xgb10_recovered.py",
        FINAL / "src" / "common.py",
        FINAL / "src" / "run_all.py",
    ]
    feature_pairs = {
        "hasan_669_train": (
            PHASE3 / "independent_rebuild" / "hasan_features_tr.feather",
            GEMINI / "cache" / "hasan_features_tr.feather",
        ),
        "hasan_669_test": (
            PHASE3 / "independent_rebuild" / "hasan_features_te.feather",
            GEMINI / "cache" / "hasan_features_te.feather",
        ),
        "selected_550_train": (
            PHASE3 / "independent_rebuild" / "hasan_selected_550_tr.feather",
            GEMINI / "cache" / "hasan_selected_550_tr.feather",
        ),
        "selected_550_test": (
            PHASE3 / "independent_rebuild" / "hasan_selected_550_te.feather",
            GEMINI / "cache" / "hasan_selected_550_te.feather",
        ),
    }
    artifact_paths = [
        champion_path,
        highcorr_path,
        duplicate_highcorr_path,
        current_path,
        CHAMPION / "08_SUBMISSIONS" / "SUBMISSION_RESULTS.md",
        CHAMPION / "06_REPRODUCIBILITY" / "PREDICTION_COMPARISON.csv",
        CHAMPION / "06_REPRODUCIBILITY" / "HASH_VALIDATION.csv",
        PHASE3 / "PHASE3_MASTER_REPORT.md",
        CHAMPION / "03_PHASE_HISTORY" / "PHASE_4"
        / "PHASE4_ZERO_GAP_OPTIMIZATION_REPORT.md",
        *ANCHORS.values(),
        *HIST_COMPONENTS.values(),
        *CURRENT_COMPONENTS.values(),
        FINAL / "runs" / "recovered_xgb10_1793441428099" / "oof_10f_xgb.npy",
        FINAL / "runs" / "recovered_xgb10_1793441428099" / "y_train.npy",
        HIST_CACHE / "oof_10f_xgb.npy",
        HIST_CACHE / "y_train.npy",
        PHASE3 / "independent_rebuild" / "hasan_expanded_zoo.npz",
        PHASE3 / "independent_rebuild" / "submission_exp85_rebuilt.csv",
        *[path for pair in feature_pairs.values() for path in pair],
        train_path,
        test_path,
    ]

    manifest_paths = [
        FINAL / "runs" / "recovered_xgb10_1793441428099" / "execution_manifest.json",
        FINAL / "runs" / "recovered_v55_1793443036653" / "execution_manifest.json",
        FINAL / "runs" / "recovered_v17_1793443853760" / "execution_manifest.json",
        FINAL / "runs" / "recovered_anchor74_1793444526421" / "execution_manifest.json",
        FINAL / "runs" / "track1b_1793445445960" / "track1b_manifest.json",
    ]

    raw_hashes = {"train": sha256(train_path), "test": sha256(test_path)}
    if raw_hashes != {"train": EXPECTED_TRAIN_SHA, "test": EXPECTED_TEST_SHA}:
        raise ValueError(f"Raw CSV hashes changed: {raw_hashes}")
    if sha256(champion_path) != EXPECTED_CHAMPION_SHA:
        raise ValueError("Frozen champion SHA-256 does not match the recorded reference.")

    feature_file_comparisons = {
        name: compare_feature_files(rebuilt, historical)
        for name, (rebuilt, historical) in feature_pairs.items()
    }
    current_schema_path = FINAL / "reports" / "production_feature_schema_audit.json"
    current_schema = json.loads(current_schema_path.read_text(encoding="utf-8"))
    current_anchor_names = current_schema["components"]["anchor74"]["features"]["train"][
        "column_names"
    ]
    historical_full_names = pd.read_feather(feature_pairs["hasan_669_train"][0]).columns.tolist()
    historical_selected_names = pd.read_feather(
        feature_pairs["selected_550_train"][0]
    ).columns.tolist()
    current_name_set = set(current_anchor_names)

    result = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "scope": "Read-only forensic calculations on existing files; no model training and no source/history/output modification.",
        "raw_inputs": {
            "train": {**file_record(train_path), "rows": 171202},
            "test": {**file_record(test_path), "rows": len(test_ids)},
            "test_ids_unique": True,
            "test_ids_sequential_171202_to_232701": True,
        },
        "source_hashes": [file_record(path) for path in source_paths],
        "artifact_hashes": [file_record(path) for path in artifact_paths],
        "run_manifests": [file_record(path) for path in manifest_paths],
        "csv_id_audits": {
            "frozen_champion": {
                "unique_ids": True,
                "rows": len(champion),
                "exact_raw_test_id_set": True,
            },
            "prior_high_correlation_submission": {
                "unique_ids": True,
                "rows": len(highcorr),
                "exact_raw_test_id_set": True,
                "identical_to_02_rebuilt_copy": bool(
                    sha256(highcorr_path) == sha256(duplicate_highcorr_path)
                ),
            },
            "current_production_track1b": {
                "unique_ids": True,
                "rows": len(current),
                "exact_raw_test_id_set": True,
            },
            "anchor_csvs": {
                name: {
                    "unique_ids": True,
                    "rows": len(values),
                    "exact_raw_test_id_set": True,
                }
                for name, values in anchors.items()
            },
            "missing_and_extra_ids": {"all_csv_comparisons": 0},
        },
        "output_metrics_vs_frozen_champion": {
            "prior_high_correlation_track1": metrics(highcorr, champion),
            "current_production_track1b": metrics(current, champion),
        },
        "anchor_metrics_vs_historical_anchor": anchors_against_historical,
        "feature_lineage": {
            "phase3_feature_files": feature_file_comparisons,
            "current_production_schema_audit": file_record(current_schema_path),
            "phase3_model_zoo_input": {
                "full_feature_count": len(historical_full_names),
                "selected_feature_count": len(historical_selected_names),
                "selected_order_is_subset_of_full_in_same_order": (
                    [name for name in historical_full_names if name in set(historical_selected_names)]
                    == historical_selected_names
                ),
                "selection_source": str(
                    PHASE3 / "experiments" / "run_workstream2_screening.py"
                ),
            },
            "current_anchor74_builder_schema": {
                "feature_count": len(current_anchor_names),
                "column_order_sha256": current_schema["components"]["anchor74"]["features"]
                ["train"]["column_order_sha256"],
                "overlap_with_phase3_669": len(current_name_set & set(historical_full_names)),
                "overlap_with_phase3_selected_550": len(
                    current_name_set & set(historical_selected_names)
                ),
                "current_only_vs_phase3_669": len(
                    current_name_set - set(historical_full_names)
                ),
                "phase3_669_missing_from_current": len(
                    set(historical_full_names) - current_name_set
                ),
                "ordered_names_equal_to_phase3_669": (
                    current_anchor_names == historical_full_names
                ),
                "ordered_names_equal_to_phase3_selected_550": (
                    current_anchor_names == historical_selected_names
                ),
            },
        },
        "suite_member_metrics_historical_cache_vs_current": member_comparisons,
        "fixed_formula_counterfactuals": compositions,
        "historical_formula": {
            "weights": WEIGHTS,
            "weight_sum": float(sum(WEIGHTS.values())),
            "per_member_uniform": "(average_rank - 0.5) / n",
            "per_member_transform": "norm.ppf(clip(uniform, 1e-6, 1 - 1e-6))",
            "suite": "sum(normalized_weight * transformed_member)",
            "macro": "0.85 * probit(anchor_percentile) + 0.15 * suite_probit",
            "final": "(ordinal_rank(macro_probit) - 0.5) / n",
            "current_builder_matches_formula": True,
            "historical_highcorr_inputs": {
                "anchor": str(ANCHORS["champion_artifacts_anchor"]),
                "anchor_sha256": sha256(ANCHORS["champion_artifacts_anchor"]),
                "suite_cache_directory": str(HIST_CACHE),
            },
            "current_lowcorr_inputs": {
                "anchor": str(ANCHORS["current_production_anchor"]),
                "anchor_sha256": sha256(ANCHORS["current_production_anchor"]),
                "suite_members": {k: str(v) for k, v in CURRENT_COMPONENTS.items()},
            },
        },
        "npz_id_order_caveat": (
            "V17/V55/XGB artifacts contain no SK_ID_CURR vector. Component metrics are "
            "positional, not ID-aligned. Current source construction reads raw test rows "
            "without sorting and predicts in feature-row order; historical cache producer "
            "order is not cryptographically bound to IDs by an embedded ID array/sidecar."
        ),
    }
    out = output_dir / "audit_evidence.json"
    out.write_text(json.dumps(result, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")
    print(f"Wrote {out}")
    print(json.dumps({
        "highcorr_spearman": result["output_metrics_vs_frozen_champion"]
        ["prior_high_correlation_track1"]["spearman"],
        "current_spearman": result["output_metrics_vs_frozen_champion"]
        ["current_production_track1b"]["spearman"],
        "evidence_file": str(out),
    }, indent=2))


if __name__ == "__main__":
    main()
