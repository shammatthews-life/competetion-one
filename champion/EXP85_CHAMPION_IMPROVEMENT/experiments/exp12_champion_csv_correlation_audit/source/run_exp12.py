from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import re
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
from scipy.stats import kendalltau, rankdata, spearmanr


EXPECTED_CHAMPION_SHA256 = (
    "78c31bb4e4534330479faa26861a49559dcadc40c7eac45620b92d7d9e6ea947"
)
DEFAULT_ROOTS = (
    Path(r"D:\gci\champion"),
    Path(r"D:\gci\_gci_push_tmp\champion"),
    Path(r"D:\gci\gemini"),
)
DEFAULT_CHAMPION = Path(
    r"D:\gci\champion\00_HISTORICAL_REFERENCE\EXP85_HISTORICAL_CHAMPION.csv"
)
DEFAULT_TRAIN = Path(r"D:\gci\inpuy\train.csv")
DEFAULT_TEST = Path(r"D:\gci\inpuy\test.csv")
EXPECTED_TRAIN_SHA256 = "b8dd5a61771d8eb2821972c1b7d3f2f747cc66471f4d34153036d2d9c458ecf1"
EXPECTED_TEST_SHA256 = "dfb7aeb96e5af3d8616acaaba6b23cec8680845cfd525f7c08de15c6d2d42c65"
DEFAULT_OUTPUT = Path(
    r"D:\gci\_gci_push_tmp\champion\EXP85_CHAMPION_IMPROVEMENT\experiments"
    r"\exp12_champion_csv_correlation_audit"
)
DEFAULT_RUN_DIR = DEFAULT_OUTPUT / "runs" / "exp12_full_csv_scan"
CHAMPION_ID_COLUMN = "SK_ID_CURR"
CHAMPION_PREDICTION_COLUMN = "TARGET"
NON_PREDICTION_COLUMNS = {
    "target_true",
    "y_true",
    "true_target",
    "label",
    "labels",
    "ground_truth",
    "actual",
    "actual_target",
}
PREDICTION_NAME = re.compile(
    r"(^|[_\W])(pred|prediction|prob|probability|score|logit|risk|oof)([_\W]|$)",
    re.IGNORECASE,
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _csv_encoding(path: Path) -> str:
    try:
        with path.open("r", encoding="utf-8-sig", newline="") as stream:
            next(csv.reader(stream), None)
        return "utf-8-sig"
    except UnicodeDecodeError:
        return "latin-1"


def _csv_shape(path: Path) -> tuple[list[str], int, str]:
    encoding = _csv_encoding(path)
    with path.open("r", encoding=encoding, newline="") as stream:
        reader = csv.reader(stream)
        columns = next(reader, [])
        row_count = sum(1 for _ in reader)
    return columns, row_count, encoding


def discover_csvs(roots: Iterable[Path], excluded_dir: Path) -> list[Path]:
    excluded = os.path.normcase(str(excluded_dir.resolve()))
    found: dict[str, Path] = {}
    for root in roots:
        if not root.exists():
            continue
        for path in root.rglob("*.csv"):
            if not path.is_file():
                continue
            resolved = path.resolve()
            normalized = os.path.normcase(str(resolved))
            common = os.path.normcase(os.path.commonpath((excluded, normalized)))
            if common == excluded:
                continue
            found.setdefault(normalized, resolved)
    return sorted(found.values(), key=lambda value: str(value).casefold())


def scan_csv(path: Path, roots: Iterable[Path]) -> dict[str, Any]:
    stat = path.stat()
    try:
        columns, rows, encoding = _csv_shape(path)
        error = ""
    except (OSError, UnicodeError, csv.Error) as exc:
        columns, rows, encoding = [], None, ""
        error = f"{type(exc).__name__}: {exc}"
    containing_roots = [
        str(root.resolve())
        for root in roots
        if root.exists()
        and os.path.commonpath(
            (os.path.normcase(str(root.resolve())), os.path.normcase(str(path.resolve())))
        )
        == os.path.normcase(str(root.resolve()))
    ]
    return {
        "path": str(path.resolve()),
        "roots_json": json.dumps(containing_roots, ensure_ascii=False),
        "file_name": path.name,
        "size_bytes": stat.st_size,
        "mtime_utc": datetime.fromtimestamp(stat.st_mtime, timezone.utc).isoformat(),
        "row_count": rows,
        "columns": columns,
        "columns_json": json.dumps(columns, ensure_ascii=False),
        "encoding": encoding,
        "sha256": sha256_file(path),
        "read_error": error,
    }


def normalize_id(value: Any) -> str:
    if value is None:
        return ""
    text = str(value).strip()
    if not text:
        return ""
    try:
        number = float(text)
    except ValueError:
        return text
    if math.isfinite(number) and number.is_integer():
        return str(int(number))
    return text


def choose_id_column(columns: list[str]) -> str | None:
    by_lower = {column.strip().lower(): column for column in columns}
    for preferred in (
        "sk_id_curr",
        "id",
        "customer_id",
        "borrower_id",
        "loan_id",
        "applicant_id",
        "id_test",
    ):
        if preferred in by_lower:
            return by_lower[preferred]
    candidates = [
        column
        for column in columns
        if column.strip().lower().endswith("_id")
        or column.strip().lower().startswith("id_")
    ]
    return candidates[0] if candidates else None


def choose_prediction_columns(columns: list[str], path: Path) -> list[str]:
    if len(set(columns)) != len(columns):
        return []
    lower = {column: column.strip().lower() for column in columns}
    selected: list[str] = []
    for column, normalized in lower.items():
        if normalized in NON_PREDICTION_COLUMNS or normalized.endswith("_true"):
            continue
        if PREDICTION_NAME.search(normalized):
            selected.append(column)
            continue
        if normalized in {"target", "probability", "prediction"}:
            stem = path.stem.lower()
            if (
                len(columns) == 2
                or any(word in stem for word in ("submission", "candidate", "prediction"))
            ) and stem not in {"train", "training", "test", "sample_submission"}:
                selected.append(column)
    return selected


def _load_ids(path: Path, id_column: str) -> list[str]:
    values = pd.read_csv(
        path, usecols=[id_column], dtype=str, keep_default_na=False
    )[id_column]
    return [normalize_id(value) for value in values.to_numpy()]


def _read_prediction_frame(
    path: Path, id_column: str, prediction_columns: list[str]
) -> pd.DataFrame:
    selected = list(dict.fromkeys([id_column, *prediction_columns]))
    return pd.read_csv(path, usecols=selected, dtype=str, keep_default_na=False)


def align_by_id(
    reference_ids: Iterable[Any],
    reference_values: Iterable[float],
    candidate_ids: Iterable[Any],
    candidate_values: Iterable[float],
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    ref_ids = [normalize_id(value) for value in reference_ids]
    cand_ids = [normalize_id(value) for value in candidate_ids]
    ref_values = np.asarray(list(reference_values), dtype=np.float64)
    cand_values = np.asarray(list(candidate_values), dtype=np.float64)
    if len(ref_ids) != len(ref_values) or len(cand_ids) != len(cand_values):
        raise ValueError("ID and prediction lengths differ")
    if any(not value for value in ref_ids + cand_ids):
        raise ValueError("IDs contain missing values")
    if len(set(ref_ids)) != len(ref_ids) or len(set(cand_ids)) != len(cand_ids):
        raise ValueError("IDs contain duplicates")
    ref_lookup = dict(zip(ref_ids, ref_values))
    cand_lookup = dict(zip(cand_ids, cand_values))
    common = [value for value in ref_ids if value in cand_lookup]
    if not common:
        raise ValueError("No matching IDs")
    return (
        np.asarray([ref_lookup[value] for value in common], dtype=np.float64),
        np.asarray([cand_lookup[value] for value in common], dtype=np.float64),
        common,
    )


def stream_sha256(ids: Iterable[str], values: Iterable[float]) -> str:
    pairs = sorted(zip(ids, values), key=lambda pair: pair[0])
    digest = hashlib.sha256()
    for identifier, value in pairs:
        digest.update(identifier.encode("utf-8"))
        digest.update(b"\0")
        digest.update(np.asarray([value], dtype="<f8").tobytes())
    return digest.hexdigest()


def _metric_corr(left: np.ndarray, right: np.ndarray, method: str) -> float | None:
    if len(left) < 2 or np.ptp(left) == 0 or np.ptp(right) == 0:
        return None
    if method == "pearson":
        value = np.corrcoef(left, right)[0, 1]
    elif method == "spearman":
        value = spearmanr(left, right).statistic
    else:
        value = kendalltau(left, right, variant="b").statistic
    return float(value) if np.isfinite(value) else None


def _risk_overlap(
    reference: np.ndarray, candidate: np.ndarray, ids: list[str], fraction: float
) -> dict[str, float | int]:
    count = max(1, int(math.ceil(len(ids) * fraction)))
    id_array = np.asarray(ids, dtype=str)
    reference_order = np.lexsort((id_array, -reference))[:count]
    candidate_order = np.lexsort((id_array, -candidate))[:count]
    reference_set = set(reference_order.tolist())
    candidate_set = set(candidate_order.tolist())
    overlap = len(reference_set & candidate_set)
    return {
        "group_size": count,
        "overlap_count": overlap,
        "overlap_fraction": overlap / count,
        "jaccard": overlap / len(reference_set | candidate_set),
    }


def calculate_metrics(
    reference: Iterable[float],
    candidate: Iterable[float],
    ids: Iterable[str],
) -> dict[str, Any]:
    ref = np.asarray(list(reference), dtype=np.float64)
    cand = np.asarray(list(candidate), dtype=np.float64)
    ids_list = [normalize_id(value) for value in ids]
    if len(ref) != len(cand) or len(ref) != len(ids_list):
        raise ValueError("IDs and aligned prediction arrays must have equal lengths")
    if len(ref) == 0 or not np.isfinite(ref).all() or not np.isfinite(cand).all():
        raise ValueError("Prediction arrays must be nonempty and finite")
    difference = cand - ref
    metrics: dict[str, Any] = {
        "pearson": _metric_corr(ref, cand, "pearson"),
        "spearman": _metric_corr(ref, cand, "spearman"),
        "kendall_tau": _metric_corr(ref, cand, "kendall"),
        "mean_absolute_difference": float(np.mean(np.abs(difference))),
        "rmse": float(np.sqrt(np.mean(np.square(difference)))),
        "maximum_absolute_difference": float(np.max(np.abs(difference))),
        "mean_signed_difference": float(np.mean(difference)),
        "below_0_001_percent": float(np.mean(np.abs(difference) < 0.001) * 100.0),
        "below_0_005_percent": float(np.mean(np.abs(difference) < 0.005) * 100.0),
        "below_0_01_percent": float(np.mean(np.abs(difference) < 0.01) * 100.0),
    }
    for fraction, label in ((0.01, "top_1"), (0.05, "top_5"), (0.10, "top_10")):
        overlap = _risk_overlap(ref, cand, ids_list, fraction)
        metrics[f"{label}_overlap_count"] = overlap["overlap_count"]
        metrics[f"{label}_overlap_fraction"] = overlap["overlap_fraction"]
        metrics[f"{label}_jaccard"] = overlap["jaccard"]
    bottom = _risk_overlap(-ref, -cand, ids_list, 0.10)
    metrics["bottom_10_overlap_count"] = bottom["overlap_count"]
    metrics["bottom_10_overlap_fraction"] = bottom["overlap_fraction"]
    metrics["bottom_10_jaccard"] = bottom["jaccard"]
    for label, values in (("champion", ref), ("candidate", cand)):
        quantiles = np.quantile(values, [0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99])
        metrics.update(
            {
                f"{label}_mean": float(np.mean(values)),
                f"{label}_std": float(np.std(values)),
                f"{label}_min": float(np.min(values)),
                f"{label}_q01": float(quantiles[0]),
                f"{label}_q05": float(quantiles[1]),
                f"{label}_q25": float(quantiles[2]),
                f"{label}_median": float(quantiles[3]),
                f"{label}_q75": float(quantiles[4]),
                f"{label}_q95": float(quantiles[5]),
                f"{label}_q99": float(quantiles[6]),
                f"{label}_max": float(np.max(values)),
            }
        )
    return metrics


def _classify_path(path: Path, file_hash: str, champion_hash: str) -> str:
    if file_hash == champion_hash:
        return "EXACT_CHAMPION_FILE_COPY"
    parts = {part.lower() for part in path.parts}
    stem = path.stem.lower()
    if any(word in stem for word in ("replay", "historical", "cache")) or any(
        word in parts for word in ("cache", "replay", "recovery", "historical")
    ):
        return "HISTORICAL_OR_CACHE_PATH_EVIDENCE_UNVERIFIED"
    if "experiments" in parts or "runs" in parts:
        return "EXPERIMENT_OUTPUT_PROVENANCE_REQUIRES_SOURCE_CONFIRMATION"
    return "GENERATED_OUTPUT_PROVENANCE_UNVERIFIED"


def _training_provenance(path: Path, file_hash: str) -> dict[str, Any]:
    for run_dir in (path.parent, *path.parents):
        manifest_path = next(
            (
                run_dir / name
                for name in ("sha256_manifest.json", "SHA256_MANIFEST.json")
                if (run_dir / name).is_file()
            ),
            None,
        )
        summary_path = run_dir / "experiment_summary.json"
        if not manifest_path or not summary_path.is_file():
            continue
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        output_entries = manifest.get("outputs", {})
        matching_output = next(
            (
                entry
                for name, entry in output_entries.items()
                if name.casefold() == path.name.casefold()
                or Path(str(entry.get("path", ""))).resolve() == path.resolve()
            ),
            None,
        )
        if not matching_output or matching_output.get("sha256") != file_hash:
            continue
        raw_input_hashes = {
            Path(str(entry.get("path", name))).name.lower(): entry.get("sha256")
            for name, entry in manifest.get("inputs", {}).items()
        }
        raw_data_verified = (
            EXPECTED_TRAIN_SHA256 in raw_input_hashes.values()
            and EXPECTED_TEST_SHA256 in raw_input_hashes.values()
        )
        model_count = sum(
            1
            for candidate in run_dir.glob("fold_*")
            if candidate.is_file()
            and candidate.suffix.lower() in {".cbm", ".txt", ".json", ".ubj"}
        )
        if not raw_data_verified or model_count < 5:
            continue
        experiment_root = run_dir.parent.parent
        results_path = experiment_root / "RESULTS.md"
        results_text = (
            results_path.read_text(encoding="utf-8", errors="replace").lower()
            if results_path.is_file()
            else ""
        )
        gate = summary.get("advancement_gate", {})
        overall_status = (
            "EXPERIMENT_REJECTED"
            if re.search(r"decision\s*:\s*\**\s*rejected", results_text)
            or re.search(r"status\s*:\s*.*rejected", results_text)
            or gate.get("passed") is False
            else "RUN_GATE_STATUS_RECORDED_IN_SUMMARY"
        )
        return {
            "provenance_class": "INDEPENDENT_RAW_DATA_TRAINING_ARTIFACTS_HASH_VERIFIED",
            "evidence": (
                "Candidate hash matches its run manifest; run manifest pins the known raw train and "
                "test inputs; five-fold model artifacts and experiment summary are present."
            ),
            "run_dir": str(run_dir.resolve()),
            "experiment_id": manifest.get("experiment_id", summary.get("experiment_id", "")),
            "model_artifact_count": model_count,
            "overall_experiment_status": overall_status,
            "run_advancement_gate": summary.get("advancement_gate", {}),
        }
    return {
        "provenance_class": "NOT_ESTABLISHED_FROM_CSV_AND_ADJACENT_RUN_ARTIFACTS",
        "evidence": "No matching run manifest, pinned raw train/test hashes, summary, and five-fold saved-model set were all verified.",
        "run_dir": "",
        "experiment_id": "",
        "model_artifact_count": 0,
        "overall_experiment_status": "UNVERIFIED",
        "run_advancement_gate": {},
    }


def _is_dataset_or_sample(path: Path) -> str:
    stem = path.stem.lower()
    if stem in {"train", "training", "application_train"}:
        return "Training dataset; labels/features are not generated predictions."
    if stem in {"test", "application_test"}:
        return "Test feature dataset; it contains inputs, not generated predictions."
    if stem in {"sample_submission", "submission_sample"}:
        return "Sample submission placeholder; excluded even if its schema matches."
    return ""


def _to_float_prediction(values: pd.Series) -> tuple[np.ndarray, int]:
    numeric = pd.to_numeric(values, errors="coerce").to_numpy(dtype=np.float64)
    invalid_count = int((~np.isfinite(numeric)).sum())
    return numeric, invalid_count


def _prediction_role(
    path: Path,
    column: str,
    ids: list[str],
    train_ids: set[str],
    test_ids: set[str],
    champion_ids: set[str],
    valid_numeric: bool,
    id_error: bool,
) -> tuple[str, str]:
    special = _is_dataset_or_sample(path)
    if special:
        return "INVALID_OR_UNKNOWN", special
    if not valid_numeric:
        return "INVALID_OR_UNKNOWN", "Prediction column is nonnumeric, missing, or non-finite."
    if id_error or not ids:
        return "INVALID_OR_UNKNOWN", "IDs are missing or duplicated; safe ID alignment is impossible."
    unique_ids = set(ids)
    if unique_ids.issubset(test_ids) and unique_ids & champion_ids:
        return "TEST_PREDICTION_COMPARABLE", ""
    oof_evidence = (
        "oof" in path.stem.lower()
        or column.strip().lower().startswith("oof_")
        or column.strip().lower().endswith("_oof")
    )
    if unique_ids.issubset(train_ids) and unique_ids.isdisjoint(test_ids) and oof_evidence:
        return "OOF_TRAIN_PREDICTION", (
            "Training-row OOF predictions; deliberately not compared with the test champion."
        )
    if unique_ids & champion_ids:
        return "DIFFERENT_DATASET", (
            "IDs mix champion-test rows with rows outside the verified test ID universe."
        )
    return "DIFFERENT_DATASET", (
        "Prediction IDs do not identify observations in the verified champion test set."
    )


def _write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(
            stream, fieldnames=fieldnames, extrasaction="ignore", lineterminator="\n"
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    key: json.dumps(value, ensure_ascii=False, sort_keys=True)
                    if isinstance(value, (dict, list, tuple))
                    else value
                    for key, value in row.items()
                }
            )


def _csv_inventory_row(row: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in row.items() if key != "columns"}


def _pairwise_correlations(
    streams: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    if len(streams) < 2:
        return []
    all_ids = streams[0]["ids"]
    if not all(np.array_equal(item["ids"], all_ids) for item in streams[1:]):
        return _pairwise_correlations_by_id(streams)
    values = np.vstack([item["values"] for item in streams])
    centered = values - values.mean(axis=1, keepdims=True)
    norms = np.linalg.norm(centered, axis=1)
    safe_norms = np.where(norms == 0, np.nan, norms)
    pearson = (centered / safe_norms[:, None]) @ (centered / safe_norms[:, None]).T
    ranks = rankdata(values, axis=1, method="average")
    centered_ranks = ranks - ranks.mean(axis=1, keepdims=True)
    rank_norms = np.linalg.norm(centered_ranks, axis=1)
    safe_rank_norms = np.where(rank_norms == 0, np.nan, rank_norms)
    spearman = (centered_ranks / safe_rank_norms[:, None]) @ (
        centered_ranks / safe_rank_norms[:, None]
    ).T
    output = []
    for left in range(len(streams)):
        for right in range(left + 1, len(streams)):
            output.append(
                {
                    "candidate_a": streams[left]["representative"],
                    "prediction_a": streams[left]["prediction_column"],
                    "prediction_stream_sha256_a": streams[left]["stream_sha256"],
                    "source_file_count_a": streams[left]["source_file_count"],
                    "candidate_b": streams[right]["representative"],
                    "prediction_b": streams[right]["prediction_column"],
                    "prediction_stream_sha256_b": streams[right]["stream_sha256"],
                    "source_file_count_b": streams[right]["source_file_count"],
                    "common_id_count": len(all_ids),
                    "pearson": _finite_or_none(pearson[left, right]),
                    "spearman": _finite_or_none(spearman[left, right]),
                    "alignment": "exact common ID universe; ID-sorted",
                }
            )
    return output


def _pairwise_correlations_by_id(
    streams: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    output = []
    for left in range(len(streams)):
        for right in range(left + 1, len(streams)):
            left_map = dict(zip(streams[left]["ids"], streams[left]["values"]))
            right_map = dict(zip(streams[right]["ids"], streams[right]["values"]))
            common = sorted(left_map.keys() & right_map.keys())
            a = np.asarray([left_map[key] for key in common], dtype=np.float64)
            b = np.asarray([right_map[key] for key in common], dtype=np.float64)
            output.append(
                {
                    "candidate_a": streams[left]["representative"],
                    "prediction_a": streams[left]["prediction_column"],
                    "prediction_stream_sha256_a": streams[left]["stream_sha256"],
                    "source_file_count_a": streams[left]["source_file_count"],
                    "candidate_b": streams[right]["representative"],
                    "prediction_b": streams[right]["prediction_column"],
                    "prediction_stream_sha256_b": streams[right]["stream_sha256"],
                    "source_file_count_b": streams[right]["source_file_count"],
                    "common_id_count": len(common),
                    "pearson": _metric_corr(a, b, "pearson"),
                    "spearman": _metric_corr(a, b, "spearman"),
                    "alignment": "ID intersection; sorted by ID",
                }
            )
    return output


def _finite_or_none(value: Any) -> float | None:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    return numeric if math.isfinite(numeric) else None


def _duplicate_rows(
    inventory: list[dict[str, Any]], streams: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    file_groups: dict[str, list[str]] = defaultdict(list)
    for row in inventory:
        if row["sha256"]:
            file_groups[row["sha256"]].append(row["path"])
    for file_hash, paths in sorted(file_groups.items()):
        if len(paths) > 1:
            output.append(
                {
                    "duplicate_type": "BYTE_IDENTICAL_CSV",
                    "duplicate_hash": file_hash,
                    "file_count": len(paths),
                    "representative_path": paths[0],
                    "paths_json": json.dumps(paths, ensure_ascii=False),
                }
            )
    for item in streams:
        if item["source_file_count"] > 1:
            output.append(
                {
                    "duplicate_type": "IDENTICAL_ID_ALIGNED_PREDICTION_STREAM",
                    "duplicate_hash": item["stream_sha256"],
                    "file_count": item["source_file_count"],
                    "representative_path": item["representative"],
                    "paths_json": json.dumps(item["source_paths"], ensure_ascii=False),
                }
            )
    return output


def _render_results(
    summary: dict[str, Any],
    inventory: list[dict[str, Any]],
    ranking: list[dict[str, Any]],
    duplicate_rows: list[dict[str, Any]],
    matrix: list[dict[str, Any]],
) -> str:
    lines = [
        "# EXP12 — Champion CSV correlation audit",
        "",
        "## Executive summary",
        "",
        f"- CSV paths inventoried: **{summary['discovered_csv_count']}** across all requested roots.",
        f"- Champion hash validated: **{summary['champion']['sha256']}**.",
        f"- Test-comparable prediction columns: **{summary['test_comparable_prediction_count']}** "
        f"from **{summary['test_comparable_file_count']}** files; all correlations use ID alignment.",
        f"- OOF prediction columns excluded from champion correlation: **{summary['oof_prediction_count']}**.",
        f"- Exact champion-prediction streams (including byte-identical copies): **{summary['champion_identical_stream_count']}**.",
        f"- Distinct pairwise-comparable prediction streams: **{summary['distinct_test_prediction_stream_count']}**.",
        f"- Historical public champion score: **{summary['competition_context']['historical_champion_public_score']:.6f}**; target: **{summary['competition_context']['target_public_score']:.5f}**. This audit creates no new leaderboard score.",
        "",
        "Correlations and distances describe prediction similarity only. They do not establish "
        "higher AUC, better calibration, or public leaderboard improvement.",
        "",
        "## Answers to the requested questions",
        "",
    ]
    answers = summary["decision_answers"]
    for question, answer in answers.items():
        lines.extend([f"**{question}** {answer}", ""])
    lines.extend(
        [
            "## Champion validation",
            "",
            "| Check | Result |",
            "|---|---:|",
        ]
    )
    for key, value in summary["champion"].items():
        if key in {"path", "sha256"}:
            lines.append(f"| {key} | `{value}` |")
        else:
            lines.append(f"| {key} | {value} |")
    lines.extend(
        [
            "",
            "## Ranked comparable predictions",
            "",
            "| Rank | Prediction source | Prediction column | Provenance assessment | IDs | Pearson | Spearman | Kendall tau-b | MAE | RMSE | Max abs diff |",
            "|---:|---|---|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for index, row in enumerate(ranking, start=1):
        lines.append(
            "| {rank} | `{path}` | `{column}` | {provenance} | {n} | {pearson} | "
            "{spearman} | {kendall} | {mae} | {rmse} | {maximum} |".format(
                rank=index,
                path=row["candidate_path"].replace("|", "%7C"),
                column=row["prediction_column"].replace("|", "%7C"),
                provenance=row["provenance_class"],
                n=row["common_id_count"],
                pearson=_fmt(row.get("pearson")),
                spearman=_fmt(row.get("spearman")),
                kendall=_fmt(row.get("kendall_tau")),
                mae=_fmt(row.get("mean_absolute_difference")),
                rmse=_fmt(row.get("rmse")),
                maximum=_fmt(row.get("maximum_absolute_difference")),
            )
        )
    lines.extend(
        [
            "",
            "## Similarity, duplication, and diversity findings",
            "",
            f"- Exact champion stream: **{len(summary['champion_identical_stream_file_paths'])} other CSV paths**; these are copies/replays, not independent discoveries.",
            f"- Near-champion noncopy streams (Spearman >= 0.999 and MAE <= 0.005): **{summary['noncopy_similarity_diagnostics']['near_champion_stream_count']}** distinct streams.",
            f"- High rank correlation with sizeable probability differences (Spearman >= 0.90 and MAE >= 0.05): **{summary['noncopy_similarity_diagnostics']['high_rank_large_probability_difference_count']}** distinct streams; first 25 are listed.",
            "- The greatest-diversity candidates are listed in `CORRELATION_AUDIT.json`; diversity is not evidence of improvement.",
            "",
        ]
    )
    for label, rows, limit in (
        (
            "Near-champion noncopy streams",
            summary["noncopy_similarity_diagnostics"]["near_champion_streams"],
            None,
        ),
        (
            "High-rank streams with sizeable probability differences",
            summary["noncopy_similarity_diagnostics"][
                "high_rank_large_probability_difference"
            ],
            25,
        ),
        (
            "Most diverse unique streams",
            summary["noncopy_similarity_diagnostics"]["most_diverse_unique_streams"],
            None,
        ),
    ):
        lines.extend(
            [
                f"### {label}",
                "",
                "| Candidate path | Prediction column | Provenance | Spearman | Pearson | MAE |",
                "|---|---|---|---:|---:|---:|",
            ]
        )
        shown = rows if limit is None else rows[:limit]
        if not shown:
            lines.append("| None found | — | — | — | — | — |")
        for row in shown:
            lines.append(
                "| `{}` | `{}` | {} | {} | {} | {} |".format(
                    row["candidate_path"].replace("|", "%7C"),
                    row["prediction_column"].replace("|", "%7C"),
                    row["provenance_class"],
                    _fmt(row.get("spearman")),
                    _fmt(row.get("pearson")),
                    _fmt(row.get("mean_absolute_difference")),
                )
            )
        lines.append("")
    lines.extend(
        [
            "## Separate OOF performance evidence",
            "",
            "These are matched OOF experiment outcomes, not metrics derived from test-set correlation:",
            "",
            "| Experiment | Pooled OOF delta vs matched Phase 2 | Decision |",
            "|---|---:|---|",
        ]
    )
    for row in summary["related_oof_performance_evidence"]:
        lines.append(
            f"| {row['experiment']} | {_fmt(row['pooled_auc_delta_vs_matched_phase2'])} | "
            f"{row['decision']} |"
        )
    lines.extend(
        [
            "",
            "The exact report paths for this context are recorded in `CORRELATION_AUDIT.json`. "
            "None of these rejections is converted into a positive claim by prediction similarity.",
            "",
            "## Duplicate and replay evidence",
            "",
            f"Duplicate groups recorded: **{len(duplicate_rows)}** (exact file duplicates and "
            "identical ID-aligned prediction streams are separately labeled).",
            "",
            "The exact champion file/prediction-stream hashes are evidence of copying or replay, "
            "not independent training. A path containing `cache`, `replay`, `recovery`, or "
            "`historical` is marked as a provenance clue only; the path name alone is not proof.",
            "",
            "## Complete CSV inventory (including excluded files)",
            "",
            f"Every one of the **{len(inventory)}** discovered CSV paths is listed below. "
            "Feature datasets, sample submissions, metric reports, OOF files and invalid outputs "
            "are retained in this inventory with their explicit classification and skip reason.",
            "",
            "| # | Path | Rows | Columns | Classification | Prediction columns | Reason / status | SHA-256 |",
            "|---:|---|---:|---|---|---|---|---|",
        ]
    )
    for index, row in enumerate(inventory, start=1):
        classification = row.get("classification", "INVALID_OR_UNKNOWN")
        reason = row.get("skip_reason", "")
        if row.get("comparison_eligible"):
            reason = "ID-aligned test prediction; see correlation ranking."
        if classification == "OOF_TRAIN_PREDICTION":
            reason = "OOF train prediction; never compared with test champion."
        lines.append(
            "| {index} | `{path}` | {rows} | `{columns}` | {classification} | `{preds}` | "
            "{reason} | `{sha}` |".format(
                index=index,
                path=row["path"].replace("|", "%7C"),
                rows=row.get("row_count", ""),
                columns=row.get("columns_json", "[]").replace("|", "%7C"),
                classification=classification,
                preds=row.get("prediction_columns_json", "[]").replace("|", "%7C"),
                reason=(reason or "—").replace("|", "%7C").replace("\n", " "),
                sha=row.get("sha256", ""),
            )
        )
    lines.extend(
        [
            "",
            "## Interpretation limitations",
            "",
            "- Test data has no labels in this audit. No test AUC, error rate, or improvement can be inferred.",
            "- OOF rows are explicitly separated from test rows; they are not correlated against the champion.",
            "- A high correlation means similar ordering/predictions, not a better model.",
            "- Exact source hashes and identical aligned prediction streams identify copies; other provenance labels remain cautious unless supported by run artifacts.",
            "- Full metrics, alignments, file metadata, duplicates, pairwise candidate correlations and hashes are in the adjacent CSV/JSON artifacts.",
            "",
        ]
    )
    return "\n".join(lines)


def _fmt(value: Any) -> str:
    if value is None or value == "":
        return "—"
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return str(value)
    if not math.isfinite(numeric):
        return "—"
    return f"{numeric:.12g}"


def run_audit(
    roots: Iterable[Path],
    champion_path: Path,
    train_path: Path | None,
    test_path: Path | None,
    output_dir: Path,
    run_dir: Path,
) -> dict[str, Any]:
    roots = tuple(Path(root) for root in roots)
    output_dir = output_dir.resolve()
    run_dir = run_dir.resolve()
    missing_roots = [str(root) for root in roots if not root.is_dir()]
    if missing_roots:
        raise FileNotFoundError(
            "CSV discovery roots are missing; refusing an incomplete audit: "
            + ", ".join(missing_roots)
        )
    if not champion_path.is_file():
        raise FileNotFoundError(f"Historical champion not found: {champion_path}")
    if not train_path or not train_path.is_file():
        raise FileNotFoundError(f"Raw training CSV not found: {train_path}")
    if not test_path or not test_path.is_file():
        raise FileNotFoundError(f"Raw test CSV not found: {test_path}")
    champion_hash = sha256_file(champion_path)
    if champion_hash != EXPECTED_CHAMPION_SHA256:
        raise ValueError(
            f"Historical champion hash mismatch: expected {EXPECTED_CHAMPION_SHA256}, "
            f"got {champion_hash}"
        )
    if sha256_file(train_path) != EXPECTED_TRAIN_SHA256:
        raise ValueError(f"Raw training CSV hash mismatch: {train_path}")
    if sha256_file(test_path) != EXPECTED_TEST_SHA256:
        raise ValueError(f"Raw test CSV hash mismatch: {test_path}")
    champion_frame = pd.read_csv(champion_path, dtype=str, keep_default_na=False)
    if CHAMPION_ID_COLUMN not in champion_frame or CHAMPION_PREDICTION_COLUMN not in champion_frame:
        raise ValueError("Champion lacks the expected SK_ID_CURR,TARGET columns")
    champion_ids = [normalize_id(x) for x in champion_frame[CHAMPION_ID_COLUMN]]
    champion_values, champion_invalid = _to_float_prediction(
        champion_frame[CHAMPION_PREDICTION_COLUMN]
    )
    if champion_invalid:
        raise ValueError(f"Champion has {champion_invalid} missing/non-finite predictions")
    if len(set(champion_ids)) != len(champion_ids) or any(not value for value in champion_ids):
        raise ValueError("Champion has missing or duplicate IDs")
    if not np.isfinite(champion_values).all() or (
        np.any(champion_values < 0) or np.any(champion_values > 1)
    ):
        raise ValueError("Champion predictions are not finite probabilities in [0,1]")
    champion_id_set = set(champion_ids)
    champion_stream_hash = stream_sha256(champion_ids, champion_values)
    test_id_values = _load_ids(test_path, CHAMPION_ID_COLUMN)
    train_id_values = _load_ids(train_path, CHAMPION_ID_COLUMN)
    if len(test_id_values) != len(set(test_id_values)) or any(not value for value in test_id_values):
        raise ValueError("Raw test data contains missing or duplicate IDs")
    if len(train_id_values) != len(set(train_id_values)) or any(not value for value in train_id_values):
        raise ValueError("Raw training data contains missing or duplicate IDs")
    test_ids = set(test_id_values)
    train_ids = set(train_id_values)
    if train_ids & test_ids:
        raise ValueError("Raw training and test ID universes overlap")
    if champion_id_set != test_ids:
        raise ValueError("Champion IDs do not exactly match the raw test ID universe")
    champion_test_overlap = len(champion_id_set & test_ids)
    inventory_paths = discover_csvs(roots, output_dir)
    scanned = [scan_csv(path, roots) for path in inventory_paths]
    champion_correlation_rows: list[dict[str, Any]] = []
    alignment_rows: list[dict[str, Any]] = []
    inventory_rows: list[dict[str, Any]] = []
    stream_groups: dict[str, dict[str, Any]] = {}
    prediction_count = 0
    oof_count = 0
    for record in scanned:
        path = Path(record["path"])
        columns = record["columns"]
        id_column = choose_id_column(columns)
        prediction_columns = choose_prediction_columns(columns, path)
        record["id_column"] = id_column or ""
        record["prediction_columns_json"] = json.dumps(prediction_columns)
        file_class = "INVALID_OR_UNKNOWN"
        skip_reason = record["read_error"]
        if not prediction_columns:
            skip_reason = skip_reason or _is_dataset_or_sample(path) or (
                "No named numeric prediction columns; excluded as data, features, metric report, or unknown CSV."
            )
            alignment_rows.append(
                {
                    "candidate_path": record["path"],
                    "prediction_column": "",
                    "classification": file_class,
                    "row_count": record["row_count"],
                    "id_column": id_column or "",
                    "skip_reason": skip_reason,
                }
            )
        elif not id_column:
            skip_reason = "Prediction-like column(s) found, but no usable ID column exists."
            alignment_rows.append(
                {
                    "candidate_path": record["path"],
                    "prediction_column": "",
                    "classification": file_class,
                    "row_count": record["row_count"],
                    "id_column": "",
                    "skip_reason": skip_reason,
                }
            )
        elif record["read_error"]:
            skip_reason = record["read_error"]
            alignment_rows.append(
                {
                    "candidate_path": record["path"],
                    "prediction_column": "",
                    "classification": file_class,
                    "row_count": record["row_count"],
                    "id_column": id_column,
                    "skip_reason": skip_reason,
                }
            )
        else:
            frame = _read_prediction_frame(path, id_column, prediction_columns)
            candidate_ids = [normalize_id(x) for x in frame[id_column]]
            missing_id_count = sum(not value for value in candidate_ids)
            duplicate_id_count = len(candidate_ids) - len(set(x for x in candidate_ids if x))
            id_error = missing_id_count > 0 or duplicate_id_count > 0
            overlap_champion = len(set(candidate_ids) & champion_id_set)
            overlap_test = len(set(candidate_ids) & test_ids) if test_ids else 0
            overlap_train = len(set(candidate_ids) & train_ids) if train_ids else 0
            complete_details: list[dict[str, Any]] = []
            for prediction_column in prediction_columns:
                prediction_count += 1
                values, invalid_count = _to_float_prediction(frame[prediction_column])
                numeric_valid = invalid_count == 0 and len(values) == len(candidate_ids)
                classification, reason = _prediction_role(
                    path,
                    prediction_column,
                    candidate_ids,
                    train_ids,
                    test_ids,
                    champion_id_set,
                    numeric_valid,
                    id_error,
                )
                if classification == "OOF_TRAIN_PREDICTION":
                    oof_count += 1
                minimum = float(np.min(values)) if len(values) and np.isfinite(values).any() else None
                maximum = float(np.max(values)) if len(values) and np.isfinite(values).any() else None
                probability_range = bool(
                    numeric_valid
                    and np.all(values >= 0.0)
                    and np.all(values <= 1.0)
                )
                stream_hash = ""
                metrics: dict[str, Any] | None = None
                common_ids: list[str] = []
                if classification == "TEST_PREDICTION_COMPARABLE":
                    aligned_ref, aligned_candidate, common_ids = align_by_id(
                        champion_ids, champion_values, candidate_ids, values
                    )
                    metrics = calculate_metrics(aligned_ref, aligned_candidate, common_ids)
                    stream_hash = stream_sha256(candidate_ids, values)
                    provenance = _training_provenance(path, record["sha256"])
                    if stream_hash == champion_stream_hash:
                        same_reference_file = path.resolve() == champion_path.resolve()
                        provenance["provenance_class"] = (
                            "PINNED_HISTORICAL_CHAMPION_REFERENCE"
                            if same_reference_file
                            else "EXACT_CHAMPION_PREDICTION_STREAM_COPY"
                        )
                        provenance["evidence"] = (
                            "ID-sorted prediction values are exactly equal to the pinned champion stream."
                        )
                    details = {
                        "candidate_path": record["path"],
                        "prediction_column": prediction_column,
                        "classification": classification,
                        "id_column": id_column,
                        "path_provenance_clue": _classify_path(
                            path, record["sha256"], champion_hash
                        ),
                        "provenance_class": provenance["provenance_class"],
                        "provenance_evidence": provenance["evidence"],
                        "training_run_dir": provenance["run_dir"],
                        "experiment_id": provenance["experiment_id"],
                        "saved_fold_model_count": provenance["model_artifact_count"],
                        "overall_experiment_status": provenance["overall_experiment_status"],
                        "run_advancement_gate": provenance["run_advancement_gate"],
                        "common_id_count": len(common_ids),
                        "candidate_row_count": len(candidate_ids),
                        "candidate_id_coverage_of_champion": len(common_ids) / len(champion_ids),
                        "candidate_id_coverage_of_test": (
                            len(set(candidate_ids) & test_ids) / len(test_ids) if test_ids else None
                        ),
                        "prediction_stream_sha256": stream_hash,
                        "prediction_in_probability_range_0_1": probability_range,
                        **metrics,
                    }
                    champion_correlation_rows.append(details)
                    sorted_ids = np.asarray(sorted(candidate_ids), dtype=str)
                    value_lookup = dict(zip(candidate_ids, values))
                    sorted_values = np.asarray([value_lookup[x] for x in sorted_ids], dtype=np.float64)
                    group = stream_groups.setdefault(
                        stream_hash,
                        {
                            "stream_sha256": stream_hash,
                            "ids": sorted_ids,
                            "values": sorted_values,
                            "representative": record["path"],
                            "prediction_column": prediction_column,
                            "source_paths": [],
                        },
                    )
                    group["source_paths"].append(record["path"])
                complete_details.append(
                    {
                        "prediction_column": prediction_column,
                        "classification": classification,
                        "reason": reason,
                        "numeric_invalid_count": invalid_count,
                        "prediction_min": minimum,
                        "prediction_max": maximum,
                        "probability_range_0_1": probability_range,
                        "prediction_stream_sha256": stream_hash,
                    }
                )
                alignment_rows.append(
                    {
                        "candidate_path": record["path"],
                        "prediction_column": prediction_column,
                        "classification": classification,
                        "row_count": len(candidate_ids),
                        "id_column": id_column,
                        "unique_id_count": len(set(x for x in candidate_ids if x)),
                        "missing_id_count": missing_id_count,
                        "duplicate_id_count": duplicate_id_count,
                        "champion_id_overlap_count": overlap_champion,
                        "champion_id_overlap_fraction": (
                            overlap_champion / len(champion_id_set) if champion_id_set else None
                        ),
                        "test_id_overlap_count": overlap_test,
                        "train_id_overlap_count": overlap_train,
                        "candidate_ids_not_in_champion_count": (
                            len(set(candidate_ids) - champion_id_set) if candidate_ids else 0
                        ),
                        "common_champion_rows_used": len(common_ids),
                        "numeric_invalid_count": invalid_count,
                        "prediction_min": minimum,
                        "prediction_max": maximum,
                        "prediction_in_probability_range_0_1": probability_range,
                        "prediction_stream_sha256": stream_hash,
                        "skip_reason": reason,
                    }
                )
            classes = {detail["classification"] for detail in complete_details}
            file_class = (
                "TEST_PREDICTION_COMPARABLE"
                if "TEST_PREDICTION_COMPARABLE" in classes
                else "OOF_TRAIN_PREDICTION"
                if classes == {"OOF_TRAIN_PREDICTION"}
                else "DIFFERENT_DATASET"
                if "DIFFERENT_DATASET" in classes
                else "INVALID_OR_UNKNOWN"
            )
            skip_reason = (
                ""
                if file_class == "TEST_PREDICTION_COMPARABLE"
                else "; ".join(
                    sorted({detail["reason"] for detail in complete_details if detail["reason"]})
                )
            )
            record["prediction_details_json"] = json.dumps(
                complete_details, ensure_ascii=False, sort_keys=True
            )
        record["classification"] = file_class
        record["comparison_eligible"] = file_class == "TEST_PREDICTION_COMPARABLE"
        record["skip_reason"] = skip_reason
        inventory_rows.append(_csv_inventory_row(record))

    streams = list(stream_groups.values())
    for stream in streams:
        stream["source_file_count"] = len(set(stream["source_paths"]))
        stream["source_paths"] = sorted(set(stream["source_paths"]))
    pairwise_rows = _pairwise_correlations(streams)
    duplicate_rows = _duplicate_rows(scanned, streams)
    ranking = sorted(
        champion_correlation_rows,
        key=lambda row: (
            row.get("spearman") is None,
            -(row.get("spearman") or 0.0),
            row.get("pearson") is None,
            -(row.get("pearson") or 0.0),
            row.get("mean_absolute_difference", math.inf),
            row["candidate_path"].casefold(),
            row["prediction_column"].casefold(),
        ),
    )
    champion_identical = [
        row
        for row in champion_correlation_rows
        if row["prediction_stream_sha256"] == champion_stream_hash
    ]
    independent_rows = [
        row
        for row in ranking
        if row["provenance_class"]
        == "INDEPENDENT_RAW_DATA_TRAINING_ARTIFACTS_HASH_VERIFIED"
        and row["prediction_stream_sha256"] != champion_stream_hash
    ]
    most_diverse = min(
        champion_correlation_rows,
        key=lambda row: (
            row.get("spearman") if row.get("spearman") is not None else math.inf,
            row.get("pearson") if row.get("pearson") is not None else math.inf,
            -row.get("mean_absolute_difference", 0.0),
        ),
        default=None,
    )
    independent_best = independent_rows[0] if independent_rows else None
    unique_prediction_rows: list[dict[str, Any]] = []
    seen_streams: set[str] = set()
    for row in ranking:
        if row["prediction_stream_sha256"] in seen_streams:
            continue
        seen_streams.add(row["prediction_stream_sha256"])
        unique_prediction_rows.append(row)
    non_champion_unique = [
        row
        for row in unique_prediction_rows
        if row["prediction_stream_sha256"] != champion_stream_hash
    ]
    near_champion_streams = [
        row
        for row in non_champion_unique
        if (row.get("spearman") or -1.0) >= 0.999
        and row.get("mean_absolute_difference", math.inf) <= 0.005
    ]
    high_rank_large_difference = [
        row
        for row in non_champion_unique
        if (row.get("spearman") or -1.0) >= 0.90
        and row.get("mean_absolute_difference", 0.0) >= 0.05
    ]
    raw_input_facts = {}
    for label, path in (("train", train_path), ("test", test_path)):
        if path and path.is_file():
            raw_input_facts[label] = {
                "path": str(path.resolve()),
                "sha256": sha256_file(path),
                "size_bytes": path.stat().st_size,
                "row_id_count": len(train_ids if label == "train" else test_ids),
            }
    exact_candidate_paths = [
        row["candidate_path"]
        for row in champion_identical
        if Path(row["candidate_path"]).resolve() != champion_path.resolve()
    ]
    candidate_metric_count = len(champion_correlation_rows)
    summary = {
        "audit_id": "exp12_full_csv_scan",
        "classification_policy": {
            "test_comparable": "prediction IDs are unique/nonmissing and all belong to verified raw test IDs with overlap to champion IDs; align by ID",
            "oof": "prediction-like column and IDs are within verified raw training IDs, disjoint from test IDs, with explicit OOF naming",
            "different_dataset": "valid prediction and IDs identify another/mixed dataset",
            "invalid_or_unknown": "nonprediction/data/report/sample/unknown CSV, invalid values, or unsafe IDs",
        },
        "discovery_roots": [str(path) for path in roots],
        "excluded_self_output_directory": str(output_dir),
        "discovered_csv_count": len(scanned),
        "competition_context": {
            "historical_champion_public_score": 0.801340,
            "target_public_score": 0.80314,
            "new_official_score_created": False,
        },
        "inventory_row_count": len(inventory_rows),
        "prediction_column_count": prediction_count,
        "test_comparable_prediction_count": len(champion_correlation_rows),
        "test_comparable_file_count": len(
            {row["candidate_path"] for row in champion_correlation_rows}
        ),
        "oof_prediction_count": oof_count,
        "distinct_test_prediction_stream_count": len(streams),
        "pairwise_unique_stream_pair_count": len(pairwise_rows),
        "champion_identical_stream_count": len(champion_identical),
        "champion_identical_stream_file_paths": exact_candidate_paths,
        "duplicate_group_count": len(duplicate_rows),
        "class_counts": {
            str(label): int(count)
            for label, count in pd.Series(
                [row["classification"] for row in inventory_rows]
            )
            .value_counts()
            .sort_index()
            .items()
        },
        "champion": {
            "path": str(champion_path.resolve()),
            "sha256": champion_hash,
            "row_count": len(champion_ids),
            "id_column": CHAMPION_ID_COLUMN,
            "prediction_column": CHAMPION_PREDICTION_COLUMN,
            "unique_id_count": len(set(champion_ids)),
            "missing_id_count": sum(not value for value in champion_ids),
            "duplicate_id_count": len(champion_ids) - len(set(champion_ids)),
            "missing_prediction_count": champion_invalid,
            "prediction_min": float(np.min(champion_values)),
            "prediction_max": float(np.max(champion_values)),
            "prediction_mean": float(np.mean(champion_values)),
            "prediction_std": float(np.std(champion_values)),
            "probabilities_in_0_1": True,
            "known_raw_test_id_count": len(test_ids) if test_ids else None,
            "overlap_with_raw_test_ids": champion_test_overlap,
            "champion_ids_missing_from_raw_test": (
                len(champion_id_set - test_ids) if test_ids else None
            ),
        },
        "raw_inputs": {
            "train": raw_input_facts.get("train"),
            "test": raw_input_facts.get("test"),
            "test_ids_equal_champion_ids": bool(test_ids == champion_id_set) if test_ids else None,
        },
        "independently_trained_candidate_evidence": {
            "scope": "A test CSV cannot prove training independence. Evidence is derived from saved folds/models/run summaries and related experiment documentation.",
            "highest_similarity_candidate": independent_best,
            "candidate_paths": [row["candidate_path"] for row in independent_rows],
        },
        "related_oof_performance_evidence": [
            {
                "experiment": "EXP07 independent seed 42",
                "pooled_auc_delta_vs_matched_phase2": 0.0006971,
                "decision": "REJECTED; missed the +0.001 gate",
                "report_path": str(
                    (output_dir.parent / "exp07_gbdt_feature_ensemble" / "RESULTS.md").resolve()
                ),
            },
            {
                "experiment": "EXP08 historical feature bridge",
                "pooled_auc_delta_vs_matched_phase2": 0.00019798495073664046,
                "decision": "REJECTED",
                "report_path": str(
                    (output_dir.parent / "exp08_historical_feature_bridge" / "RESULTS.md").resolve()
                ),
            },
            {
                "experiment": "EXP10 independently replicated pair",
                "pooled_auc_delta_vs_matched_phase2": 0.0007935762321820139,
                "decision": "REJECTED; missed the +0.001 gate",
                "report_path": str(
                    (
                        output_dir.parent
                        / "exp10_pair_replication"
                        / "runs"
                        / "exp10_pair_seed20261102_final_verified"
                        / "RESULTS.md"
                    ).resolve()
                ),
            },
            {
                "experiment": "EXP11 fresh XGB10 refit",
                "pooled_auc_delta_vs_matched_phase2": -0.010879549792374488,
                "decision": "REJECTED",
                "report_path": str(
                    (
                        output_dir.parent
                        / "exp11_historical_component_recovery"
                        / "runs"
                        / "exp11_xgb10_seed20261102"
                        / "RESULTS.md"
                    ).resolve()
                ),
            },
        ],
        "decision_answers": {
            "1. Closest generated CSV": (
                f"{len(exact_candidate_paths)} non-reference CSV path(s) are exact champion-stream "
                "copies and therefore tied as closest. The closest non-identical prediction stream is "
                f"{next((row['candidate_path'] + ' (Spearman ' + _fmt(row.get('spearman')) + ', MAE ' + _fmt(row.get('mean_absolute_difference')) + ')' for row in ranking if row['prediction_stream_sha256'] != champion_stream_hash), 'none')}."
                if exact_candidate_paths
                else f"{ranking[0]['candidate_path']} ({ranking[0]['prediction_column']}; "
                f"Spearman {_fmt(ranking[0].get('spearman'))})."
                if ranking
                else "No test-comparable generated predictions were found."
            ),
            "2. Highest-similarity independently trained candidate": (
                f"{independent_best['candidate_path']} ({independent_best['prediction_column']}; "
                f"Spearman {_fmt(independent_best.get('spearman'))}, Pearson "
                f"{_fmt(independent_best.get('pearson'))}; "
                f"{independent_best['overall_experiment_status']}). Similarity does not imply a better model."
                if independent_best
                else "No independently trained test candidate could be established from the available CSV and run artifacts."
            ),
            "3. Greatest prediction diversity": (
                f"{most_diverse['candidate_path']} ({most_diverse['prediction_column']}; "
                f"Spearman {_fmt(most_diverse.get('spearman'))}, MAE "
                f"{_fmt(most_diverse.get('mean_absolute_difference'))}). Diversity is not improvement."
                if most_diverse
                else "No test-comparable candidate was available."
            ),
            "4. Copies or historical-cache replays": (
                f"Yes. {len(exact_candidate_paths)} CSV path(s) reproduce the champion prediction stream exactly; "
                f"{sum(row['duplicate_type'] == 'BYTE_IDENTICAL_CSV' for row in duplicate_rows)} "
                "byte-identical file duplicate group(s) and additional aligned-stream duplicates are listed. "
                "Other replay/cache labels are path clues, not proven lineage."
            ),
            "5. Independently trained candidate worth investigating further": (
                "No new candidate is recommended on correlation alone. The strongest verified fresh-run candidates "
                "are the rejected EXP07/EXP08 outputs; consult their OOF gates rather than rerunning or promoting by similarity."
                if independent_rows
                else "None can be recommended from correlation alone; no independently trained output was established."
            ),
            "6. OOF files not comparable to the test champion": (
                f"{oof_count} OOF prediction column(s) were identified and excluded; full paths and reasons are in CSV_INVENTORY.csv and ALIGNMENT_VALIDATION.csv."
            ),
            "7. Evidence of improved predictive performance": (
                "No test-label or new leaderboard evidence exists in this audit. The related predeclared OOF replications remain rejected: EXP07 independent seed +0.0006971, EXP08 +0.00019798, EXP10 +0.00079358, and EXP11 -0.01087955 pooled AUC delta. EXP07's first screen (+0.0010171) required independent replication and was not a promotion result. Correlation does not establish improvement."
            ),
        },
        "top_ranked_predictions": ranking[:25],
        "lowest_spearman_prediction": most_diverse,
        "noncopy_similarity_diagnostics": {
            "near_champion_stream_definition": "Spearman >= 0.999 and MAE <= 0.005, excluding exact champion-stream copies and duplicate prediction streams.",
            "near_champion_stream_count": len(near_champion_streams),
            "near_champion_streams": near_champion_streams,
            "high_rank_large_probability_difference_definition": "Spearman >= 0.90 and MAE >= 0.05, excluding exact champion-stream copies and duplicate prediction streams.",
            "high_rank_large_probability_difference_count": len(
                high_rank_large_difference
            ),
            "high_rank_large_probability_difference": high_rank_large_difference,
            "most_diverse_unique_streams": sorted(
                non_champion_unique,
                key=lambda row: (
                    row.get("spearman") if row.get("spearman") is not None else math.inf,
                    row.get("pearson") if row.get("pearson") is not None else math.inf,
                ),
            )[:10],
        },
        "limitations": [
            "No test labels are used or available for the test-set comparison.",
            "Correlation does not measure AUC or public leaderboard performance.",
            "OOF prediction rows were not compared with the test champion.",
            "Path-based provenance labels are conservative and do not by themselves prove a historical replay.",
            "Pairwise Pearson and Spearman are emitted for unique aligned streams; pairwise Kendall is omitted as impractical at full pair coverage.",
        ],
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    run_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(
        run_dir / "CSV_INVENTORY.csv",
        inventory_rows,
        [
            "path",
            "roots_json",
            "file_name",
            "size_bytes",
            "mtime_utc",
            "row_count",
            "columns_json",
            "sha256",
            "encoding",
            "read_error",
            "id_column",
            "prediction_columns_json",
            "prediction_details_json",
            "classification",
            "comparison_eligible",
            "skip_reason",
        ],
    )
    matrix_fields = list(
        dict.fromkeys(
            key for row in champion_correlation_rows for key in row.keys()
        )
    )
    _write_csv(run_dir / "CHAMPION_CORRELATION_MATRIX.csv", champion_correlation_rows, matrix_fields)
    _write_csv(run_dir / "CHAMPION_CORRELATION_RANKING.csv", ranking, matrix_fields)
    _write_csv(
        run_dir / "PAIRWISE_CANDIDATE_CORRELATIONS.csv",
        pairwise_rows,
        [
            "candidate_a",
            "prediction_a",
            "prediction_stream_sha256_a",
            "source_file_count_a",
            "candidate_b",
            "prediction_b",
            "prediction_stream_sha256_b",
            "source_file_count_b",
            "common_id_count",
            "pearson",
            "spearman",
            "alignment",
        ],
    )
    alignment_fields = list(
        dict.fromkeys(key for row in alignment_rows for key in row.keys())
    )
    _write_csv(run_dir / "ALIGNMENT_VALIDATION.csv", alignment_rows, alignment_fields)
    _write_csv(
        run_dir / "DUPLICATE_FILES.csv",
        duplicate_rows,
        [
            "duplicate_type",
            "duplicate_hash",
            "file_count",
            "representative_path",
            "paths_json",
        ],
    )
    (run_dir / "CORRELATION_AUDIT.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False, sort_keys=True, allow_nan=False)
        + "\n",
        encoding="utf-8",
        newline="\n",
    )
    (run_dir / "RESULTS.md").write_text(
        _render_results(summary, inventory_rows, ranking, duplicate_rows, champion_correlation_rows),
        encoding="utf-8",
        newline="\n",
    )
    artifacts = [
        "CSV_INVENTORY.csv",
        "CHAMPION_CORRELATION_MATRIX.csv",
        "CHAMPION_CORRELATION_RANKING.csv",
        "PAIRWISE_CANDIDATE_CORRELATIONS.csv",
        "ALIGNMENT_VALIDATION.csv",
        "DUPLICATE_FILES.csv",
        "CORRELATION_AUDIT.json",
        "RESULTS.md",
    ]
    manifest = {
        "manifest_version": 1,
        "audit_id": summary["audit_id"],
        "champion_sha256": champion_hash,
        "discovered_csv_count": len(scanned),
        "inputs": {
            row["path"]: {
                "sha256": row["sha256"],
                "size_bytes": row["size_bytes"],
            }
            for row in scanned
        },
        "references": {
            "champion": {
                "path": str(champion_path.resolve()),
                "sha256": champion_hash,
                "size_bytes": champion_path.stat().st_size,
            },
            **raw_input_facts,
        },
        "sources": {
            "run_exp12.py": {
                "path": str(Path(__file__).resolve()),
                "sha256": sha256_file(Path(__file__).resolve()),
                "size_bytes": Path(__file__).stat().st_size,
            },
            "verify_exp12.py": {
                "path": str(Path(__file__).resolve().with_name("verify_exp12.py")),
                "sha256": sha256_file(Path(__file__).resolve().with_name("verify_exp12.py")),
                "size_bytes": Path(__file__).resolve().with_name("verify_exp12.py").stat().st_size,
            },
        },
        "outputs": {
            name: {
                "sha256": sha256_file(run_dir / name),
                "size_bytes": (run_dir / name).stat().st_size,
            }
            for name in artifacts
        },
        "self_hash_excluded": True,
    }
    (run_dir / "SHA256_MANIFEST.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False, sort_keys=True, allow_nan=False)
        + "\n",
        encoding="utf-8",
        newline="\n",
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", action="append", type=Path, dest="roots")
    parser.add_argument("--champion", type=Path, default=DEFAULT_CHAMPION)
    parser.add_argument("--train", type=Path, default=DEFAULT_TRAIN)
    parser.add_argument("--test", type=Path, default=DEFAULT_TEST)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN_DIR)
    args = parser.parse_args()
    result = run_audit(
        args.roots or DEFAULT_ROOTS,
        args.champion,
        args.train,
        args.test,
        args.output_dir,
        args.run_dir,
    )
    print(
        json.dumps(
            {
                "discovered_csv_count": result["discovered_csv_count"],
                "test_comparable_prediction_count": result["test_comparable_prediction_count"],
                "oof_prediction_count": result["oof_prediction_count"],
                "distinct_test_prediction_stream_count": result[
                    "distinct_test_prediction_stream_count"
                ],
                "run_dir": str(args.run_dir.resolve()),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
