#!/usr/bin/env python3
"""Audit a candidate prediction file against a reference prediction file by explicit ID alignment.

This script is intentionally fail-closed. It refuses to compute correlations when the ID sets do
not match exactly or when prediction values are invalid.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr


def sha256_file(path: Path) -> str:
    import hashlib

    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def _normalize_id_values(series: pd.Series) -> pd.Series:
    # Normalize mixed numeric/string IDs to a stable string representation so duplicates and
    # mismatches are checked consistently without silently dropping rows.
    return series.map(lambda v: str(v)).astype(str)


def load_predictions(path: Path, id_col: str, pred_col: str) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Input file not found: {path}")

    df = pd.read_csv(path)
    if df.empty:
        raise ValueError(f"Input file is empty: {path}")
    if id_col not in df.columns:
        raise ValueError(f"Missing ID column '{id_col}' in {path}")
    if pred_col not in df.columns:
        raise ValueError(f"Missing prediction column '{pred_col}' in {path}")

    out = df[[id_col, pred_col]].copy()
    out.columns = ["id", "pred"]
    out["id"] = _normalize_id_values(out["id"])
    try:
        out["pred"] = pd.to_numeric(out["pred"], errors="raise")
    except Exception as exc:
        raise ValueError(f"Non-numeric predictions found in column '{pred_col}' in {path}") from exc
    if not np.isfinite(out["pred"].to_numpy(dtype=float)).all():
        raise ValueError(f"Non-finite values found in prediction column '{pred_col}' in {path}")
    if out["pred"].nunique() <= 1:
        raise ValueError("Prediction array has zero variance; Pearson/Spearman are undefined")
    return out


def validate_alignment(ref: pd.DataFrame, cand: pd.DataFrame) -> dict:
    ref = ref.copy()
    cand = cand.copy()

    findings = {
        "duplicate_ids_ref": int(ref["id"].duplicated().sum()),
        "duplicate_ids_cand": int(cand["id"].duplicated().sum()),
        "missing_ids": 0,
        "extra_ids": 0,
        "row_order_matches": bool(ref["id"].tolist() == cand["id"].tolist()),
        "aligned_row_count": 0,
        "valid": False,
    }

    if findings["duplicate_ids_ref"] > 0:
        raise ValueError("Duplicate IDs detected in reference data")
    if findings["duplicate_ids_cand"] > 0:
        raise ValueError("Duplicate IDs detected in candidate data")

    merged = ref.merge(cand, on="id", how="outer", suffixes=("_ref", "_cand"), indicator=True)
    findings["missing_ids"] = int(merged["_merge"].eq("right_only").sum())
    findings["extra_ids"] = int(merged["_merge"].eq("left_only").sum())
    findings["aligned_row_count"] = int((merged["_merge"] == "both").sum())

    if findings["missing_ids"] or findings["extra_ids"]:
        raise ValueError(
            f"ID alignment invalid: missing_ids={findings['missing_ids']}, extra_ids={findings['extra_ids']}"
        )

    if findings["aligned_row_count"] == 0:
        raise ValueError("No aligned IDs found between reference and candidate data")

    findings["valid"] = True
    return findings


def compute_metrics(ref: pd.DataFrame, cand: pd.DataFrame) -> dict:
    aligned = ref.merge(cand, on="id", how="inner", validate="one_to_one", suffixes=("_ref", "_cand"))
    aligned = aligned.sort_values("id").reset_index(drop=True)

    if aligned.empty:
        raise ValueError("No common IDs found between reference and candidate data")

    y = aligned["pred_ref"].to_numpy(dtype=float)
    y_hat = aligned["pred_cand"].to_numpy(dtype=float)

    if np.any(~np.isfinite(y)) or np.any(~np.isfinite(y_hat)):
        raise ValueError("Non-finite values detected during metric calculation")
    if np.allclose(y, y[0]) or np.allclose(y_hat, y_hat[0]):
        raise ValueError("Correlation is undefined for constant arrays")

    pearson = float(np.corrcoef(y, y_hat)[0, 1])
    spearman = float(spearmanr(y, y_hat).statistic)
    mae = float(np.mean(np.abs(y - y_hat)))
    rmse = float(np.sqrt(np.mean((y - y_hat) ** 2)))
    max_abs_diff = float(np.max(np.abs(y - y_hat)))
    mean_abs_diff = float(np.mean(np.abs(y - y_hat)))

    ref_rank = aligned["pred_ref"].rank(method="average", ascending=True).to_numpy()
    cand_rank = aligned["pred_cand"].rank(method="average", ascending=True).to_numpy()
    rank_disp = np.abs(ref_rank - cand_rank)

    metrics = {
        "pearson": pearson,
        "spearman": spearman,
        "mae": mae,
        "rmse": rmse,
        "max_abs_prediction_difference": max_abs_diff,
        "mean_abs_prediction_difference": mean_abs_diff,
        "max_rank_displacement": int(np.max(rank_disp)),
        "mean_rank_displacement": float(np.mean(rank_disp)),
        "median_rank_displacement": float(np.median(rank_disp)),
        "quantile_90_abs_pred_diff": float(np.quantile(np.abs(y - y_hat), 0.90)),
        "quantile_95_abs_pred_diff": float(np.quantile(np.abs(y - y_hat), 0.95)),
        "quantile_99_abs_pred_diff": float(np.quantile(np.abs(y - y_hat), 0.99)),
        "tie_handling": "Pandas rank(method='average') is used for ties in both reference and candidate ranks.",
        "note": "This is a correlation/error audit, not an official leaderboard score.",
    }
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare reference and candidate predictions by ID alignment")
    parser.add_argument("--reference", required=True, type=Path, help="Reference CSV path")
    parser.add_argument("--candidate", required=True, type=Path, help="Candidate CSV path")
    parser.add_argument("--id-col", required=True, help="ID column name")
    parser.add_argument("--pred-col", required=True, help="Prediction column name")
    parser.add_argument("--report", type=Path, default=None, help="Human-readable report path")
    parser.add_argument("--json-out", type=Path, default=None, help="JSON output path")
    args = parser.parse_args()

    try:
        ref_df = load_predictions(args.reference, args.id_col, args.pred_col)
        cand_df = load_predictions(args.candidate, args.id_col, args.pred_col)
        align = validate_alignment(ref_df, cand_df)
        metrics = compute_metrics(ref_df, cand_df)

        report = {
            "status": "VALID",
            "reference_path": str(args.reference),
            "candidate_path": str(args.candidate),
            "reference_sha256": sha256_file(args.reference),
            "candidate_sha256": sha256_file(args.candidate),
            "alignment": align,
            "metrics": metrics,
        }

        if args.json_out is not None:
            args.json_out.parent.mkdir(parents=True, exist_ok=True)
            args.json_out.write_text(json.dumps(report, indent=2), encoding="utf-8")

        if args.report is not None:
            lines = [
                "EXP85 Prediction Audit Report",
                "===========================",
                f"Reference: {args.reference}",
                f"Candidate: {args.candidate}",
                f"Reference SHA256: {report['reference_sha256']}",
                f"Candidate SHA256: {report['candidate_sha256']}",
                f"ID alignment valid: {align['valid']}",
                f"Aligned row count: {align['aligned_row_count']}",
                f"Original row order matched: {align['row_order_matches']}",
                f"Duplicate IDs in reference: {align['duplicate_ids_ref']}",
                f"Duplicate IDs in candidate: {align['duplicate_ids_cand']}",
                f"Missing IDs: {align['missing_ids']}",
                f"Extra IDs: {align['extra_ids']}",
                "",
                "Metrics",
                "-------",
                f"Pearson: {metrics['pearson']}",
                f"Spearman: {metrics['spearman']}",
                f"MAE: {metrics['mae']}",
                f"RMSE: {metrics['rmse']}",
                f"Max abs prediction difference: {metrics['max_abs_prediction_difference']}",
                f"Mean abs prediction difference: {metrics['mean_abs_prediction_difference']}",
                f"Max rank displacement: {metrics['max_rank_displacement']}",
                f"Mean rank displacement: {metrics['mean_rank_displacement']}",
                f"Median rank displacement: {metrics['median_rank_displacement']}",
                "",
                "This is not an official leaderboard score; it is a correlation and error audit.",
                metrics["tie_handling"],
            ]
            args.report.parent.mkdir(parents=True, exist_ok=True)
            args.report.write_text("\n".join(lines) + "\n", encoding="utf-8")

        print(json.dumps(report, indent=2))
        return
    except Exception as exc:  # fail closed and do not silently report a valid comparison
        payload = {
            "status": "INVALID",
            "error": str(exc),
            "reference_path": str(args.reference),
            "candidate_path": str(args.candidate),
        }
        if args.json_out is not None:
            args.json_out.parent.mkdir(parents=True, exist_ok=True)
            args.json_out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(json.dumps(payload, indent=2), file=sys.stderr)
        raise SystemExit(1)


if __name__ == "__main__":
    main()
