from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, rankdata, spearmanr


SCRIPT_PATH = Path(__file__).resolve()
REPO_ROOT = SCRIPT_PATH.parents[3]


class ComparisonError(ValueError):
    def __init__(self, message: str, alignment: dict[str, Any] | None = None):
        super().__init__(message)
        self.alignment = alignment or {}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def file_identity(path: Path, source_path: str | None = None) -> dict[str, Any]:
    if not path.is_file():
        return {
            "path": str(path.resolve()),
            "source_path": source_path,
            "exists": False,
            "size_bytes": None,
            "sha256": None,
        }
    return {
        "path": str(path.resolve()),
        "source_path": source_path,
        "exists": True,
        "size_bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def current_git_commit() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "-C", str(REPO_ROOT), "rev-parse", "HEAD"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def load_predictions(path: Path, id_col: str, pred_col: str) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(f"Input file not found: {path}")
    frame = pd.read_csv(path, dtype={id_col: "string"})
    if frame.empty:
        raise ComparisonError(f"Input file is empty: {path}")
    missing_columns = [column for column in (id_col, pred_col) if column not in frame]
    if missing_columns:
        raise ComparisonError(
            f"Missing required column(s) {missing_columns} in {path}"
        )
    result = frame[[id_col, pred_col]].copy()
    result.columns = ["id", "prediction"]
    if result["id"].isna().any() or result["id"].str.len().eq(0).any():
        raise ComparisonError(f"Missing or blank IDs found in {path}")
    result["id"] = result["id"].astype(str)
    result["prediction"] = pd.to_numeric(result["prediction"], errors="coerce")
    values = result["prediction"].to_numpy(dtype=np.float64)
    if not np.isfinite(values).all():
        raise ComparisonError(f"Non-numeric or non-finite predictions found in {path}")
    if np.any((values < 0.0) | (values > 1.0)):
        raise ComparisonError(f"Predictions outside [0, 1] found in {path}")
    return result


def _rank_summary(
    reference_values: np.ndarray,
    candidate_values: np.ndarray,
    method: str,
) -> dict[str, float | int]:
    reference_rank = rankdata(reference_values, method=method)
    candidate_rank = rankdata(candidate_values, method=method)
    displacement = np.abs(reference_rank - candidate_rank)
    return {
        "mean": float(np.mean(displacement)),
        "median": float(np.median(displacement)),
        "p95": float(np.quantile(displacement, 0.95)),
        "maximum": float(np.max(displacement)),
        "exact_rank_agreement_count": int(np.count_nonzero(displacement == 0)),
    }


def compare_predictions(
    reference: pd.DataFrame,
    candidate: pd.DataFrame,
    method: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    duplicate_reference = int(reference["id"].duplicated(keep=False).sum())
    duplicate_candidate = int(candidate["id"].duplicated(keep=False).sum())
    reference_ids = set(reference["id"])
    candidate_ids = set(candidate["id"])
    missing_ids = sorted(reference_ids - candidate_ids)
    extra_ids = sorted(candidate_ids - reference_ids)
    row_order_matches = reference["id"].tolist() == candidate["id"].tolist()
    aligned_count = len(reference_ids & candidate_ids)
    alignment = {
        "reference_rows": int(len(reference)),
        "candidate_rows": int(len(candidate)),
        "reference_duplicate_id_rows": duplicate_reference,
        "candidate_duplicate_id_rows": duplicate_candidate,
        "missing_from_candidate_ids": len(missing_ids),
        "extra_in_candidate_ids": len(extra_ids),
        "aligned_row_count": aligned_count,
        "row_order_matches": row_order_matches,
        "valid": False,
    }
    if duplicate_reference or duplicate_candidate:
        raise ComparisonError("Duplicate IDs found; refusing comparison", alignment)
    if missing_ids or extra_ids:
        raise ComparisonError(
            f"ID sets differ: missing_from_candidate={len(missing_ids)}, "
            f"extra_in_candidate={len(extra_ids)}",
            alignment,
        )
    if aligned_count == 0:
        raise ComparisonError("No matching IDs found", alignment)
    if method == "notebook_stage9" and not row_order_matches:
        raise ComparisonError(
            "Legacy notebook method requires identical verified ID row order",
            alignment,
        )

    if method == "id_aligned":
        ref = reference.set_index("id").sort_index()
        cand = candidate.set_index("id").reindex(ref.index)
        ref_values = ref["prediction"].to_numpy(dtype=np.float64)
        cand_values = cand["prediction"].to_numpy(dtype=np.float64)
        rank_method = "average"
        alignment_method = "one-to-one ID join, sorted by ID"
    elif method == "notebook_stage9":
        ref_values = reference["prediction"].to_numpy(dtype=np.float64)
        cand_values = candidate["prediction"].to_numpy(dtype=np.float64)
        rank_method = "ordinal"
        alignment_method = (
            "original notebook row-position comparison after exact ID-order validation"
        )
    else:
        raise ValueError(f"Unsupported comparison method: {method}")

    if len(ref_values) < 2:
        raise ComparisonError("At least two aligned rows are required", alignment)
    if np.ptp(ref_values) == 0 or np.ptp(cand_values) == 0:
        raise ComparisonError("Zero-variance predictions make correlation undefined", alignment)
    difference = cand_values - ref_values
    absolute_difference = np.abs(difference)
    rank_metrics = _rank_summary(ref_values, cand_values, rank_method)
    alignment["valid"] = True
    metrics = {
        "pearson": float(pearsonr(ref_values, cand_values).statistic),
        "spearman": float(spearmanr(ref_values, cand_values).statistic),
        "mae": float(np.mean(absolute_difference)),
        "rmse": float(np.sqrt(np.mean(np.square(difference)))),
        "maximum_absolute_prediction_error": float(np.max(absolute_difference)),
        "mean_absolute_prediction_error": float(np.mean(absolute_difference)),
        "rank_displacement": rank_metrics,
        "rank_method": rank_method,
        "alignment_method": alignment_method,
        "correlation_is_not_prediction_equality": True,
        "official_score_inferred": False,
    }
    return alignment, metrics


def markdown_report(record: dict[str, Any]) -> str:
    alignment = record.get("alignment", {})
    metrics = record.get("metrics")
    prediction_validation = record.get("prediction_validation", {})
    lines = [
        "# EXP85 Submission Correlation Replication",
        "",
        f"- Run ID: `{record['run_id']}`",
        f"- Status: **{record['status']}**",
        f"- Started (UTC): `{record['started_at_utc']}`",
        f"- Git commit: `{record.get('git_commit') or 'unavailable'}`",
        f"- Method: `{record['configuration']['method']}`",
        f"- Command exit code: `{record['exit_code']}`",
        "",
        "## Inputs",
        "",
        f"- Reference: `{record['reference']['path']}`",
        f"- Reference source path: `{record['reference'].get('source_path')}`",
        f"- Reference SHA-256: `{record['reference']['sha256']}`",
        f"- Candidate: `{record['candidate']['path']}`",
        f"- Candidate source path: `{record['candidate'].get('source_path')}`",
        f"- Candidate SHA-256: `{record['candidate']['sha256']}`",
        "",
        "## Alignment",
        "",
        f"- Matched rows: `{alignment.get('aligned_row_count')}`",
        f"- Duplicate ID rows (reference/candidate): "
        f"`{alignment.get('reference_duplicate_id_rows')}` / "
        f"`{alignment.get('candidate_duplicate_id_rows')}`",
        f"- Missing candidate IDs / extra candidate IDs: "
        f"`{alignment.get('missing_from_candidate_ids')}` / "
        f"`{alignment.get('extra_in_candidate_ids')}`",
        f"- Original row order matched: `{alignment.get('row_order_matches')}`",
        "",
    ]
    if prediction_validation:
        lines.extend(
            [
                "## Prediction validation",
                "",
                f"- Expected rows: `{record['configuration']['expected_rows']}`",
                f"- Reference min / max: "
                f"`{prediction_validation['reference']['minimum']:.17g}` / "
                f"`{prediction_validation['reference']['maximum']:.17g}`",
                f"- Candidate min / max: "
                f"`{prediction_validation['candidate']['minimum']:.17g}` / "
                f"`{prediction_validation['candidate']['maximum']:.17g}`",
                "- All predictions finite and within `[0, 1]`: `True`",
                "",
            ]
        )
    if metrics is not None:
        ranks = metrics["rank_displacement"]
        lines.extend(
            [
                "## Metrics",
                "",
                f"- Pearson: `{metrics['pearson']:.17g}`",
                f"- Spearman: `{metrics['spearman']:.17g}`",
                f"- MAE: `{metrics['mae']:.17g}`",
                f"- RMSE: `{metrics['rmse']:.17g}`",
                f"- Maximum absolute prediction error: "
                f"`{metrics['maximum_absolute_prediction_error']:.17g}`",
                f"- Rank displacement mean / median / p95 / max: "
                f"`{ranks['mean']:.17g}` / `{ranks['median']:.17g}` / "
                f"`{ranks['p95']:.17g}` / `{ranks['maximum']:.17g}`",
                f"- Exact rank agreement rows: `{ranks['exact_rank_agreement_count']}`",
                f"- Rank method: `{metrics['rank_method']}`",
                f"- Alignment method: {metrics['alignment_method']}",
                "",
            ]
        )
    if record.get("error"):
        lines.extend(["## Error", "", record["error"], ""])
    lines.extend(
        [
            "## Limitations",
            "",
            record.get("lineage_note", "Model lineage was not independently certified."),
            "Correlation does not prove prediction equality or model lineage. "
            "No official score is inferred from this local comparison.",
            "",
            f"- Script SHA-256: `{record['script_sha256']}`",
            f"- Configuration SHA-256: `{record['configuration_sha256']}`",
            "",
        ]
    )
    return "\n".join(lines)


def run(
    reference_path: Path,
    candidate_path: Path,
    run_dir: Path,
    id_col: str,
    pred_col: str,
    method: str,
    reference_source_path: str | None,
    candidate_source_path: str | None,
    lineage_note: str,
    expected_rows: int = 61500,
) -> dict[str, Any]:
    if run_dir.exists():
        raise FileExistsError(f"Run directory already exists; refusing overwrite: {run_dir}")
    if expected_rows < 2:
        raise ValueError(f"Expected row count must be at least 2: {expected_rows}")
    started = datetime.now(timezone.utc)
    configuration = {
        "id_column": id_col,
        "prediction_column": pred_col,
        "method": method,
        "rank_tie_method": "ordinal" if method == "notebook_stage9" else "average",
        "probability_bounds": [0.0, 1.0],
        "expected_rows": expected_rows,
    }
    config_bytes = json.dumps(configuration, sort_keys=True, separators=(",", ":")).encode()
    record: dict[str, Any] = {
        "run_id": run_dir.name,
        "status": "INVALID",
        "started_at_utc": started.isoformat(),
        "git_commit": current_git_commit(),
        "command": sys.argv,
        "exit_code": 1,
        "script_path": str(SCRIPT_PATH),
        "script_sha256": sha256_file(SCRIPT_PATH),
        "configuration": configuration,
        "configuration_sha256": hashlib.sha256(config_bytes).hexdigest(),
        "reference": file_identity(reference_path, reference_source_path),
        "candidate": file_identity(candidate_path, candidate_source_path),
        "lineage_note": lineage_note,
        "alignment": {},
        "prediction_validation": {},
        "metrics": None,
        "official_competition_score": "NOT_EVALUATED",
    }
    try:
        if not record["reference"]["exists"]:
            raise FileNotFoundError(f"Reference artifact not found: {reference_path}")
        if not record["candidate"]["exists"]:
            raise FileNotFoundError(f"Candidate artifact not found: {candidate_path}")
        reference = load_predictions(reference_path, id_col, pred_col)
        candidate = load_predictions(candidate_path, id_col, pred_col)
        if len(reference) != expected_rows or len(candidate) != expected_rows:
            alignment = {
                "reference_rows": int(len(reference)),
                "candidate_rows": int(len(candidate)),
                "valid": False,
            }
            raise ComparisonError(
                f"Expected exactly {expected_rows} rows in both inputs; "
                f"found reference={len(reference)}, candidate={len(candidate)}",
                alignment,
            )
        record["prediction_validation"] = {
            "reference": {
                "minimum": float(reference["prediction"].min()),
                "maximum": float(reference["prediction"].max()),
                "finite": True,
            },
            "candidate": {
                "minimum": float(candidate["prediction"].min()),
                "maximum": float(candidate["prediction"].max()),
                "finite": True,
            },
        }
        alignment, metrics = compare_predictions(reference, candidate, method)
        record["alignment"] = alignment
        record["metrics"] = metrics
        record["status"] = "VALID"
        record["exit_code"] = 0
    except ComparisonError as error:
        record["error"] = str(error)
        record["alignment"] = error.alignment
    except (OSError, pd.errors.ParserError, ValueError) as error:
        record["error"] = str(error)

    record["completed_at_utc"] = datetime.now(timezone.utc).isoformat()
    run_dir.mkdir(parents=True)
    (run_dir / "correlation_report.json").write_text(
        json.dumps(record, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    (run_dir / "correlation_report.md").write_text(
        markdown_report(record), encoding="utf-8"
    )
    return record


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Fail-closed, ID-aware EXP85 prediction comparison."
    )
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--id-col", default="SK_ID_CURR")
    parser.add_argument("--pred-col", default="TARGET")
    parser.add_argument("--expected-rows", type=int, default=61500)
    parser.add_argument(
        "--method",
        choices=("id_aligned", "notebook_stage9"),
        default="id_aligned",
    )
    parser.add_argument("--reference-source-path")
    parser.add_argument("--candidate-source-path")
    parser.add_argument(
        "--lineage-note",
        default="Candidate lineage has not been independently certified.",
    )
    args = parser.parse_args()
    record = run(
        args.reference,
        args.candidate,
        args.run_dir,
        args.id_col,
        args.pred_col,
        args.method,
        args.reference_source_path,
        args.candidate_source_path,
        args.lineage_note,
        args.expected_rows,
    )
    print(json.dumps(record, indent=2, ensure_ascii=False))
    return int(record["exit_code"])


if __name__ == "__main__":
    raise SystemExit(main())
