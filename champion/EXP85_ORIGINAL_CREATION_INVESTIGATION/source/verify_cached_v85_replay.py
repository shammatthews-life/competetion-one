from __future__ import annotations

import argparse
import hashlib
import io
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import norm, rankdata, spearmanr


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


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def rank_uniform(values: np.ndarray) -> np.ndarray:
    return (rankdata(values, method="average") - 0.5) / len(values)


def probit(values: np.ndarray) -> np.ndarray:
    return norm.ppf(np.clip(values, 1e-6, 1.0 - 1e-6))


def require_vector(
    archive: np.lib.npyio.NpzFile, key: str, expected_length: int, name: str
) -> np.ndarray:
    if key not in archive.files:
        raise KeyError(f"{name} archive is missing {key!r}")
    values = np.asarray(archive[key])
    if values.ndim != 1 or len(values) != expected_length:
        raise ValueError(
            f"{name}[{key!r}] shape {values.shape}; expected ({expected_length},)"
        )
    if not np.isfinite(values).all():
        raise ValueError(f"{name}[{key!r}] contains non-finite predictions")
    return values


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Recompute the historical v85 blend from saved prediction artifacts; "
            "does not train or persist a candidate."
        )
    )
    parser.add_argument("--repository-root", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--anchor-csv", type=Path, required=True)
    parser.add_argument("--raw-test-csv", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()

    started_at = datetime.now(timezone.utc)
    started_clock = time.perf_counter()
    root = args.repository_root.resolve()
    cache_dir = args.cache_dir.resolve()
    manifest_path = root / "gemini" / "EXP85_DEPENDENCY_MANIFEST.json"
    artifacts = {
        "anchor74": args.anchor_csv.resolve(),
        "v17": cache_dir / "v_17_full_suite_bundle.npz",
        "v55": cache_dir / "v55_members.npz",
        "xgb10_test": cache_dir / "test_10f_xgb.npy",
        "raw_test": args.raw_test_csv.resolve(),
        "historical_v85": root / "gemini" / "v_20261027_optimal_macro_apex_v85.csv",
        "frozen_champion": root / "gemini" / "frozen_champion_0801340_reference.csv",
        "dependency_manifest": manifest_path,
    }
    missing = [str(path) for path in artifacts.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError("Required replay inputs are missing: " + ", ".join(missing))

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest_keys = {
        "v17": "cache/v_17_full_suite_bundle.npz",
        "v55": "cache/v55_members.npz",
        "xgb10_test": "cache/test_10f_xgb.npy",
        "anchor74": "v_20261027_precision_transductive_apex.csv",
    }
    manifest_verification = {}
    for artifact_name, manifest_key in manifest_keys.items():
        expected = manifest[manifest_key]["sha256"]
        actual = sha256_file(artifacts[artifact_name])
        manifest_verification[artifact_name] = {
            "manifest_key": manifest_key,
            "expected_sha256": expected,
            "actual_sha256": actual,
            "matches": actual == expected,
        }
        if actual != expected:
            raise ValueError(
                f"{artifact_name} hash does not match the recorded EXP85 "
                f"dependency manifest: {actual} != {expected}"
            )

    anchor = pd.read_csv(artifacts["anchor74"], usecols=["SK_ID_CURR", "TARGET"])
    raw_ids = pd.read_csv(artifacts["raw_test"], usecols=["SK_ID_CURR"])[
        "SK_ID_CURR"
    ].to_numpy()
    ids = anchor["SK_ID_CURR"].to_numpy()
    anchor_values = anchor["TARGET"].to_numpy(dtype=np.float64)
    if len(ids) != len(raw_ids) or not np.array_equal(ids, raw_ids):
        raise ValueError("Historical Anchor74 IDs/order do not exactly match raw test.csv")
    if len(np.unique(ids)) != len(ids):
        raise ValueError("Historical Anchor74 contains duplicate IDs")
    if not np.isfinite(anchor_values).all():
        raise ValueError("Historical Anchor74 contains non-finite predictions")
    n_test = len(ids)

    with np.load(artifacts["v17"], allow_pickle=False) as v17:
        with np.load(artifacts["v55"], allow_pickle=False) as v55:
            xgb_test = np.load(artifacts["xgb10_test"], allow_pickle=False)
            if xgb_test.ndim != 1 or len(xgb_test) != n_test:
                raise ValueError(
                    f"XGB10 test shape {xgb_test.shape}; expected ({n_test},)"
                )
            if not np.isfinite(xgb_test).all():
                raise ValueError("XGB10 test predictions contain non-finite values")

            model_vectors = {
                "v17_e1Aet": require_vector(v17, "test_e1Aet", n_test, "V17"),
                "xgb10": xgb_test,
                "v17_final": require_vector(v17, "final_test", n_test, "V17"),
                "v55_cbprod": require_vector(
                    v55, "test_cb_product", n_test, "V55"
                ),
                "v17_e1et": require_vector(v17, "test_e1et", n_test, "V17"),
                "v55_deep_et": require_vector(
                    v55, "test_deep_et", n_test, "V55"
                ),
                "v55_allfam": require_vector(
                    v55, "test_v2_allfam_oh", n_test, "V55"
                ),
                "v55_v2_mono": require_vector(
                    v55, "test_v2_mono", n_test, "V55"
                ),
            }

    weight_total = sum(WEIGHTS.values())
    suite_z = np.zeros(n_test, dtype=np.float64)
    for name, weight in WEIGHTS.items():
        suite_z += (weight / weight_total) * probit(rank_uniform(model_vectors[name]))
    macro_z = 0.850 * probit(anchor_values) + 0.150 * suite_z
    final_target = (
        rankdata(macro_z, method="ordinal").astype(np.float64) - 0.5
    ) / float(n_test)
    recomputed = pd.DataFrame({"SK_ID_CURR": ids, "TARGET": final_target})
    buffer = io.StringIO(newline="")
    recomputed.to_csv(buffer, index=False, float_format="%.10f")
    candidate_bytes = buffer.getvalue().encode("utf-8")
    candidate_sha = hashlib.sha256(candidate_bytes).hexdigest()

    reference_data: dict[str, Any] = {}
    for name in ("historical_v85", "frozen_champion"):
        reference = pd.read_csv(artifacts[name], usecols=["SK_ID_CURR", "TARGET"])
        ref_ids = reference["SK_ID_CURR"].to_numpy()
        if len(ref_ids) != n_test or not np.array_equal(ref_ids, ids):
            raise ValueError(f"{name} IDs/order do not match raw test.csv")
        ref_target = reference["TARGET"].to_numpy(dtype=np.float64)
        if not np.isfinite(ref_target).all():
            raise ValueError(f"{name} contains non-finite targets")
        difference = np.abs(final_target - ref_target)
        reference_bytes = artifacts[name].read_bytes()
        reference_data[name] = {
            "path": str(artifacts[name]),
            "sha256": sha256_file(artifacts[name]),
            "candidate_bytes_equal": candidate_bytes == reference_bytes,
            "target_exact_matches": int(np.count_nonzero(final_target == ref_target)),
            "maximum_absolute_target_difference": float(difference.max()),
            "pearson": float(np.corrcoef(final_target, ref_target)[0, 1]),
            "spearman": float(spearmanr(final_target, ref_target).statistic),
        }

    elapsed = time.perf_counter() - started_clock
    result = {
        "classification": "CACHED_ARTIFACT_REPLAY_NOT_RAW_REBUILD",
        "training_executed": False,
        "candidate_file_persisted": False,
        "official_competition_score_evaluated": False,
        "method": (
            "Recomputed the fixed v85 test blend from existing artifacts whose "
            "hashes match the tracked EXP85 dependency manifest; no weights "
            "were tuned."
        ),
        "historical_manifest_verification": manifest_verification,
        "cache_namespace": str(cache_dir),
        "anchor_path": str(artifacts["anchor74"]),
        "formula": (
            "Normalized fixed model weights in probit rank space; 0.85 * "
            "Anchor74 probit + 0.15 * suite probit; ordinal rank projection."
        ),
        "test_rows": n_test,
        "ids_match_raw_test_exact_order": True,
        "all_prediction_vectors_finite_and_correct_length": True,
        "prediction_arrays_have_id_sidecars": False,
        "prediction_array_order_assumed_from_historical_producer": True,
        "recomputed_candidate": {
            "sha256": candidate_sha,
            "csv_bytes": len(candidate_bytes),
            "unique_targets": int(np.unique(final_target).size),
            "target_min": float(final_target.min()),
            "target_max": float(final_target.max()),
        },
        "references": reference_data,
        "input_sha256": {
            name: sha256_file(path)
            for name, path in artifacts.items()
        },
        "weights": WEIGHTS,
        "command": [
            sys.executable,
            str(Path(__file__).resolve()),
            *sys.argv[1:],
        ],
        "started_at_utc": started_at.isoformat(),
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "elapsed_seconds": elapsed,
        "exit_code": 0,
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
