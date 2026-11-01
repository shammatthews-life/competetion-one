from __future__ import annotations

import argparse
import hashlib
import json
import platform
import re
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import scipy
import sklearn

from gap_diagnostics import (
    align_submission,
    grouped_error_metrics,
    make_prediction_slices,
    prediction_metrics,
)


ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = ROOT.parents[1]
EXPECTED_SHA256 = {
    "historical_champion": "78c31bb4e4534330479faa26861a49559dcadc40c7eac45620b92d7d9e6ea947",
    "hybrid_v5": "91aa95c9501efb358e98b47e2def6678faa6c2f4ba34431e4f9d0b4d4b792482",
    "phase2": "2470cecc0819d696b03b8d0fcc80d7b91b680986e9c7fe877a57a1684a7411fa",
    "test": "dfb7aeb96e5af3d8616acaaba6b23cec8680845cfd525f7c08de15c6d2d42c65",
    "sample": "6e5ad587b7c0b3d9fa39c90ce03e161ff5ef292193e117a6a02b11075dffe7db",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_commit() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "-C", str(REPO_ROOT), "rev-parse", "HEAD"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def input_record(name: str, path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"Required input is missing: {path}")
    digest = sha256_file(path)
    if digest != EXPECTED_SHA256[name]:
        raise ValueError(
            f"{name} SHA-256 mismatch: expected {EXPECTED_SHA256[name]}, got {digest}"
        )
    return {
        "path": str(path.resolve()),
        "size_bytes": path.stat().st_size,
        "sha256": digest,
    }


def read_submission(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path, dtype={"SK_ID_CURR": "int64"})
    if list(frame.columns) != ["SK_ID_CURR", "TARGET"]:
        raise ValueError(f"Unexpected submission schema in {path}: {list(frame.columns)}")
    return frame


def run(
    reference_path: Path,
    phase2_path: Path,
    hybrid_path: Path,
    test_path: Path,
    sample_path: Path,
    run_dir: Path,
) -> dict[str, Any]:
    if run_dir.exists():
        raise FileExistsError(f"Run directory already exists; refusing overwrite: {run_dir}")
    started_at = datetime.now(timezone.utc).isoformat()
    identities = {
        "historical_champion": input_record("historical_champion", reference_path),
        "phase2": input_record("phase2", phase2_path),
        "hybrid_v5": input_record("hybrid_v5", hybrid_path),
        "test": input_record("test", test_path),
        "sample": input_record("sample", sample_path),
        "source_files": {
            path.name: {
                "path": str(path.resolve()),
                "sha256": sha256_file(path),
            }
            for path in (Path(__file__).resolve(), Path(__file__).with_name("gap_diagnostics.py"))
        },
    }

    test_frame = pd.read_csv(test_path)
    sample_frame = pd.read_csv(sample_path, dtype={"SK_ID_CURR": "int64"})
    if list(sample_frame.columns) != ["SK_ID_CURR", "TARGET"]:
        raise ValueError(f"Unexpected sample submission schema: {list(sample_frame.columns)}")
    sample_ids = sample_frame["SK_ID_CURR"]
    if "TARGET" in test_frame.columns or len(test_frame) != 61500:
        raise ValueError("Raw test rows have an unexpected target column or row count")
    if "SK_ID_CURR" not in test_frame.columns:
        raise ValueError("Raw test is missing SK_ID_CURR")
    test_frame["SK_ID_CURR"] = pd.to_numeric(
        test_frame["SK_ID_CURR"], errors="raise"
    ).astype("int64")
    if not test_frame["SK_ID_CURR"].is_unique or not sample_ids.is_unique:
        raise ValueError("Raw test or sample IDs are duplicated")
    if not test_frame["SK_ID_CURR"].reset_index(drop=True).equals(
        sample_ids.reset_index(drop=True)
    ):
        raise ValueError("Raw test and sample submission ID order differs")

    reference = read_submission(reference_path)
    phase2 = read_submission(phase2_path)
    hybrid = read_submission(hybrid_path)
    ref_values, phase2_values, phase2_alignment = align_submission(
        reference, phase2, test_frame["SK_ID_CURR"]
    )
    _, hybrid_values, hybrid_alignment = align_submission(
        reference, hybrid, test_frame["SK_ID_CURR"]
    )
    slices = make_prediction_slices(ref_values, test_frame)
    groups = []
    groups.extend(grouped_error_metrics("phase2_independent", ref_values, phase2_values, slices))
    groups.extend(grouped_error_metrics("v5_hybrid_replay", ref_values, hybrid_values, slices))

    comparisons = {
        "phase2_vs_historical_champion": prediction_metrics(ref_values, phase2_values),
        "v5_hybrid_vs_historical_champion": prediction_metrics(ref_values, hybrid_values),
        "phase2_vs_v5_hybrid": prediction_metrics(phase2_values, hybrid_values),
    }
    record = {
        "run_id": run_dir.name,
        "status": "VALID_DIAGNOSTIC_ONLY",
        "started_at_utc": started_at,
        "completed_at_utc": None,
        "git_commit": git_commit(),
        "command": sys.argv,
        "runtime": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scipy": scipy.__version__,
            "scikit_learn": sklearn.__version__,
        },
        "inputs": identities,
        "alignment": {
            "test_rows": int(len(test_frame)),
            "test_columns": int(len(test_frame.columns)),
            "target_in_test": False,
            "sample_order_matches_test": True,
            "phase2": phase2_alignment,
            "hybrid_v5": hybrid_alignment,
        },
        "comparisons": comparisons,
        "slices": {
            "reference_prediction_decile": "10 equal-sized groups from historical predictions",
            "ext_source_missingness": "0, 1, or 2+ missing values among EXT_SOURCE_1..3",
            "total_feature_missingness": "0, 1-5, 6-10, or >10 missing test covariates",
            "income_quartile": "rank quartiles of AMT_INCOME_TOTAL",
            "age_band": "under 25, 25-34, 35-44, 45-54, 55+",
            "contract_type": "raw categorical test covariate",
            "gender": "raw categorical test covariate",
        },
        "interpretation_guard": {
            "test_labels_available": False,
            "historical_predictions_used_as_labels": False,
            "training_or_tuning_performed": False,
            "model_selection_performed": False,
            "official_score_inferred": False,
            "candidate_promotion_allowed": False,
            "meaning": "Descriptive prediction-disagreement analysis only.",
        },
    }
    record["completed_at_utc"] = datetime.now(timezone.utc).isoformat()

    run_dir.mkdir(parents=True)
    (run_dir / "gap_report.json").write_text(
        json.dumps(record, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    pd.DataFrame(groups).to_csv(
        run_dir / "grouped_prediction_differences.csv",
        index=False,
        lineterminator="\n",
    )
    summary = [
        "# Prediction-Gap Diagnostic",
        "",
        f"- Run: `{record['run_id']}`",
        "- Status: `VALID_DIAGNOSTIC_ONLY`",
        "- Raw test labels: none; no training or tuning was performed.",
        "- All comparisons were joined to the exact raw test IDs.",
        "",
        "| Pair | Pearson | Spearman | MAE | RMSE | Max abs error | Mean rank displacement | p95 rank displacement | Max rank displacement |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for name, metrics in comparisons.items():
        summary.append(
            f"| {name} | {metrics['pearson']:.10f} | {metrics['spearman']:.10f} | "
            f"{metrics['mae']:.8g} | {metrics['rmse']:.8g} | "
            f"{metrics['maximum_absolute_error']:.8g} | "
            f"{metrics['mean_rank_displacement']:.4f} | "
            f"{metrics['p95_rank_displacement']:.4f} | "
            f"{metrics['maximum_rank_displacement']:.0f} |"
        )
    summary.extend(
        [
            "",
            "Grouped statistics are in `grouped_prediction_differences.csv`. They describe disagreement with a reference prediction, not subgroup default risk or model error against labels.",
            "",
            "No candidate was selected or promoted from this report. Historical test predictions are not ground truth.",
            "",
        ]
    )
    (run_dir / "gap_report.md").write_text(
        "\n".join(summary), encoding="utf-8", newline="\n"
    )
    outputs = {}
    for path in sorted(run_dir.iterdir()):
        if path.is_file():
            outputs[path.name] = {
                "size_bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
    (run_dir / "run_manifest.json").write_text(
        json.dumps(
            {
                "run_id": record["run_id"],
                "inputs": identities,
                "outputs": outputs,
            },
            indent=2,
            ensure_ascii=False,
            allow_nan=False,
        )
        + "\n",
        encoding="utf-8",
        newline="\n",
    )
    return record


def main() -> int:
    parser = argparse.ArgumentParser(description="Read-only, ID-aligned prediction-gap diagnostics.")
    parser.add_argument(
        "--reference",
        type=Path,
        default=Path(
            r"D:\gci\champion\00_HISTORICAL_REFERENCE\EXP85_HISTORICAL_CHAMPION.csv"
        ),
    )
    parser.add_argument(
        "--phase2",
        type=Path,
        default=ROOT.parent
        / "EXP85_PHASE2_FROZEN_RELEASE"
        / "CANDIDATE"
        / "submission_key_target_encoding.csv",
    )
    parser.add_argument(
        "--hybrid",
        type=Path,
        default=ROOT.parent
        / "EXP85_SUBMISSION_CORRELATION_REPLICATION"
        / "candidate_artifacts"
        / "track1_v5_rebuilt.csv",
    )
    parser.add_argument("--test", type=Path, default=Path(r"D:\gci\inpuy\test.csv"))
    parser.add_argument("--sample", type=Path, default=Path(r"D:\gci\inpuy\sample_submission.csv"))
    parser.add_argument("--run-id", required=True, help="Unique, human-readable run directory name.")
    args = parser.parse_args()
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{0,79}", args.run_id):
        parser.error("--run-id may contain only letters, numbers, dot, underscore, and hyphen")
    run_dir = ROOT / "runs" / args.run_id
    record = run(args.reference, args.phase2, args.hybrid, args.test, args.sample, run_dir)
    print(json.dumps(record["comparisons"], indent=2))
    print(f"Saved diagnostic-only run to {run_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
