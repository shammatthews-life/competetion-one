from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import kendalltau, rankdata, spearmanr


EXPECTED_CHAMPION_SHA256 = (
    "78c31bb4e4534330479faa26861a49559dcadc40c7eac45620b92d7d9e6ea947"
)
EXPECTED_TRAIN_SHA256 = "b8dd5a61771d8eb2821972c1b7d3f2f747cc66471f4d34153036d2d9c458ecf1"
EXPECTED_TEST_SHA256 = "dfb7aeb96e5af3d8616acaaba6b23cec8680845cfd525f7c08de15c6d2d42c65"
DEFAULT_RUN_DIR = (
    Path(r"D:\gci\_gci_push_tmp\champion\EXP85_CHAMPION_IMPROVEMENT\experiments")
    / "exp12_champion_csv_correlation_audit"
    / "runs"
    / "exp12_full_csv_scan"
)
DEFAULT_CHAMPION = Path(
    r"D:\gci\champion\00_HISTORICAL_REFERENCE\EXP85_HISTORICAL_CHAMPION.csv"
)
NUMERIC_METRICS = (
    "pearson",
    "spearman",
    "kendall_tau",
    "mean_absolute_difference",
    "rmse",
    "maximum_absolute_difference",
    "mean_signed_difference",
    "below_0_001_percent",
    "below_0_005_percent",
    "below_0_01_percent",
    "top_1_overlap_fraction",
    "top_5_overlap_fraction",
    "top_10_overlap_fraction",
    "bottom_10_overlap_fraction",
)


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_id(value: Any) -> str:
    text = str(value).strip()
    if not text:
        return ""
    try:
        number = float(text)
    except ValueError:
        return text
    return str(int(number)) if math.isfinite(number) and number.is_integer() else text


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def _pearson(left: np.ndarray, right: np.ndarray) -> float | None:
    if len(left) < 2 or np.ptp(left) == 0 or np.ptp(right) == 0:
        return None
    value = float(np.corrcoef(left, right)[0, 1])
    return value if math.isfinite(value) else None


def _spearman(left: np.ndarray, right: np.ndarray) -> float | None:
    if len(left) < 2 or np.ptp(left) == 0 or np.ptp(right) == 0:
        return None
    value = float(spearmanr(left, right).statistic)
    return value if math.isfinite(value) else None


def _kendall(left: np.ndarray, right: np.ndarray) -> float | None:
    if len(left) < 2 or np.ptp(left) == 0 or np.ptp(right) == 0:
        return None
    value = float(kendalltau(left, right, variant="b").statistic)
    return value if math.isfinite(value) else None


def _risk_sets(
    reference: np.ndarray, candidate: np.ndarray, ids: list[str], fraction: float
) -> tuple[int, float]:
    size = max(1, int(math.ceil(len(ids) * fraction)))
    id_array = np.asarray(ids, dtype=str)
    ref_set = set(np.lexsort((id_array, -reference))[:size].tolist())
    candidate_set = set(np.lexsort((id_array, -candidate))[:size].tolist())
    return len(ref_set & candidate_set), len(ref_set & candidate_set) / size


def _recompute_metrics(
    reference: np.ndarray, candidate: np.ndarray, ids: list[str]
) -> dict[str, float | int | None]:
    error = candidate - reference
    result: dict[str, float | int | None] = {
        "pearson": _pearson(reference, candidate),
        "spearman": _spearman(reference, candidate),
        "kendall_tau": _kendall(reference, candidate),
        "mean_absolute_difference": float(np.abs(error).mean()),
        "rmse": float(np.sqrt(np.square(error).mean())),
        "maximum_absolute_difference": float(np.abs(error).max()),
        "mean_signed_difference": float(error.mean()),
        "below_0_001_percent": float((np.abs(error) < 0.001).mean() * 100.0),
        "below_0_005_percent": float((np.abs(error) < 0.005).mean() * 100.0),
        "below_0_01_percent": float((np.abs(error) < 0.01).mean() * 100.0),
    }
    for fraction, label in ((0.01, "top_1"), (0.05, "top_5"), (0.10, "top_10")):
        count, ratio = _risk_sets(reference, candidate, ids, fraction)
        result[f"{label}_overlap_count"] = count
        result[f"{label}_overlap_fraction"] = ratio
    count, ratio = _risk_sets(-reference, -candidate, ids, 0.10)
    result["bottom_10_overlap_count"] = count
    result["bottom_10_overlap_fraction"] = ratio
    return result


def _check_numeric(actual: str, expected: float | int | None, label: str) -> None:
    if expected is None:
        require(actual == "", f"{label} should be empty for an undefined correlation")
        return
    require(actual != "", f"{label} is missing")
    require(
        math.isclose(float(actual), float(expected), rel_tol=1e-9, abs_tol=1e-10),
        f"{label} mismatch: stored={actual}, recomputed={expected}",
    )


def _aligned_predictions(
    path: Path, id_column: str, prediction_column: str, champion_ids: list[str]
) -> tuple[list[str], np.ndarray, np.ndarray]:
    frame = pd.read_csv(
        path,
        usecols=[id_column, prediction_column],
        dtype=str,
        keep_default_na=False,
    )
    ids = [canonical_id(value) for value in frame[id_column].to_numpy()]
    values = pd.to_numeric(frame[prediction_column], errors="coerce").to_numpy(
        dtype=np.float64
    )
    require(all(ids), f"Missing ID in candidate: {path}")
    require(len(ids) == len(set(ids)), f"Duplicate ID in candidate: {path}")
    require(np.isfinite(values).all(), f"Invalid prediction values in candidate: {path}")
    lookup = dict(zip(ids, values))
    common = [identifier for identifier in champion_ids if identifier in lookup]
    require(common, f"No champion ID overlap for purported comparable file: {path}")
    champion_frame = pd.read_csv(
        DEFAULT_CHAMPION,
        usecols=["SK_ID_CURR", "TARGET"],
        dtype=str,
        keep_default_na=False,
    )
    reference_lookup = {
        canonical_id(identifier): float(value)
        for identifier, value in zip(
            champion_frame["SK_ID_CURR"], champion_frame["TARGET"]
        )
    }
    return (
        common,
        np.asarray([reference_lookup[identifier] for identifier in common]),
        np.asarray([lookup[identifier] for identifier in common]),
    )


def verify(run_dir: Path) -> dict[str, Any]:
    run_dir = run_dir.resolve()
    required_files = (
        "CSV_INVENTORY.csv",
        "CHAMPION_CORRELATION_MATRIX.csv",
        "CHAMPION_CORRELATION_RANKING.csv",
        "PAIRWISE_CANDIDATE_CORRELATIONS.csv",
        "ALIGNMENT_VALIDATION.csv",
        "DUPLICATE_FILES.csv",
        "CORRELATION_AUDIT.json",
        "RESULTS.md",
    )
    manifest_path = run_dir / "SHA256_MANIFEST.json"
    require(manifest_path.is_file(), "SHA256 manifest is missing")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    require(manifest["champion_sha256"] == EXPECTED_CHAMPION_SHA256, "Pinned champion hash mismatch")
    for name in required_files:
        require(name in manifest["outputs"], f"Required output missing from manifest: {name}")
    for name, record in manifest["outputs"].items():
        path = run_dir / name
        require(path.is_file(), f"Required output missing: {name}")
        require(
            file_hash(path) == record["sha256"]
            and path.stat().st_size == record["size_bytes"],
            f"Output hash/size mismatch: {name}",
        )
    for name, record in manifest["sources"].items():
        path = Path(record["path"])
        require(path.is_file(), f"Source file missing: {name}")
        require(
            file_hash(path) == record["sha256"]
            and path.stat().st_size == record["size_bytes"],
            f"Source hash/size mismatch: {name}",
        )
    inventory = _read_csv(run_dir / "CSV_INVENTORY.csv")
    require(
        len(inventory) == manifest["discovered_csv_count"],
        "Inventory count does not match manifest",
    )
    input_records = manifest["inputs"]
    require(
        len(input_records) == len(inventory),
        "Input hash record count does not match CSV inventory",
    )
    for row in inventory:
        record = input_records.get(row["path"])
        path = Path(row["path"])
        require(record is not None and path.is_file(), f"Input disappeared: {row['path']}")
        require(
            row["sha256"] == record["sha256"] == file_hash(path)
            and int(row["size_bytes"]) == int(record["size_bytes"]) == path.stat().st_size,
            f"Input hash/size mismatch: {row['path']}",
        )

    audit = json.loads((run_dir / "CORRELATION_AUDIT.json").read_text(encoding="utf-8"))
    for label, expected_hash in (
        ("champion", EXPECTED_CHAMPION_SHA256),
        ("train", EXPECTED_TRAIN_SHA256),
        ("test", EXPECTED_TEST_SHA256),
    ):
        record = manifest["references"].get(label)
        if label == "champion" or record:
            require(record is not None, f"Reference hash record is missing: {label}")
            reference_path = Path(record["path"])
            require(reference_path.is_file(), f"Reference file missing: {label}")
            require(
                record["sha256"] == expected_hash
                and file_hash(reference_path) == expected_hash
                and reference_path.stat().st_size == record["size_bytes"],
                f"Reference hash/size mismatch: {label}",
            )
    require(
        audit["raw_inputs"]["train"]["sha256"] == EXPECTED_TRAIN_SHA256
        and audit["raw_inputs"]["test"]["sha256"] == EXPECTED_TEST_SHA256,
        "Pinned raw train/test hashes do not match the audit summary",
    )
    champion_path = Path(audit["champion"]["path"])
    require(file_hash(champion_path) == EXPECTED_CHAMPION_SHA256, "Champion source hash changed")
    champion_frame = pd.read_csv(
        champion_path,
        usecols=["SK_ID_CURR", "TARGET"],
        dtype=str,
        keep_default_na=False,
    )
    champion_ids = [canonical_id(value) for value in champion_frame["SK_ID_CURR"]]
    champion_values = pd.to_numeric(
        champion_frame["TARGET"], errors="coerce"
    ).to_numpy(dtype=np.float64)
    require(len(champion_ids) == 61500, "Champion row count changed")
    require(len(champion_ids) == len(set(champion_ids)) and all(champion_ids), "Champion IDs invalid")
    require(np.isfinite(champion_values).all(), "Champion predictions are non-finite")
    require(
        int(audit["champion"]["unique_id_count"]) == len(set(champion_ids))
        and int(audit["champion"]["missing_id_count"]) == 0
        and int(audit["champion"]["duplicate_id_count"]) == 0
        and int(audit["champion"]["missing_prediction_count"]) == 0,
        "Champion validation summary is inconsistent",
    )

    matrix = _read_csv(run_dir / "CHAMPION_CORRELATION_MATRIX.csv")
    alignment = _read_csv(run_dir / "ALIGNMENT_VALIDATION.csv")
    oof_paths = {
        row["candidate_path"]
        for row in alignment
        if row.get("classification") == "OOF_TRAIN_PREDICTION"
    }
    require(
        not any(row["candidate_path"] in oof_paths for row in matrix),
        "An OOF file was incorrectly compared with the test champion",
    )
    recomputed_streams: dict[str, dict[str, Any]] = {}
    metric_count = 0
    for row in matrix:
        path = Path(row["candidate_path"])
        ids, reference, candidate = _aligned_predictions(
            path,
            row["id_column"] if "id_column" in row else "SK_ID_CURR",
            row["prediction_column"],
            champion_ids,
        )
        actual = _recompute_metrics(reference, candidate, ids)
        for key in NUMERIC_METRICS:
            _check_numeric(row.get(key, ""), actual[key], f"{path.name}:{key}")
        for key in (
            "top_1_overlap_count",
            "top_5_overlap_count",
            "top_10_overlap_count",
            "bottom_10_overlap_count",
        ):
            require(
                int(float(row[key])) == int(actual[key]),
                f"{path.name}:{key} mismatch",
            )
        stream_hash = row["prediction_stream_sha256"]
        sorted_order = np.argsort(np.asarray(ids, dtype=str), kind="stable")
        id_array = np.asarray(ids, dtype=str)[sorted_order]
        candidate_sorted = candidate[sorted_order]
        stream_digest = hashlib.sha256()
        for identifier, value in zip(id_array, candidate_sorted):
            stream_digest.update(identifier.encode("utf-8"))
            stream_digest.update(b"\0")
            stream_digest.update(np.asarray([value], dtype="<f8").tobytes())
        require(stream_digest.hexdigest() == stream_hash, f"Prediction stream hash mismatch: {path}")
        recomputed_streams.setdefault(
            stream_hash,
            {
                "ids": id_array,
                "values": candidate_sorted,
                "representative": str(path),
                "prediction_column": row["prediction_column"],
            },
        )
        metric_count += 1

    ranking = _read_csv(run_dir / "CHAMPION_CORRELATION_RANKING.csv")
    require(len(ranking) == len(matrix), "Ranking and matrix row counts differ")
    for left, right in zip(ranking, sorted(
        matrix,
        key=lambda item: (
            item["spearman"] == "",
            -float(item["spearman"] or 0.0),
            item["pearson"] == "",
            -float(item["pearson"] or 0.0),
            float(item["mean_absolute_difference"]),
            item["candidate_path"].casefold(),
            item["prediction_column"].casefold(),
        ),
    )):
        require(
            left["candidate_path"] == right["candidate_path"]
            and left["prediction_column"] == right["prediction_column"],
            "Ranking order is not reproducible from its declared sort fields",
        )

    pairwise = _read_csv(run_dir / "PAIRWISE_CANDIDATE_CORRELATIONS.csv")
    unique_streams = list(recomputed_streams.items())
    expected_pair_count = len(unique_streams) * (len(unique_streams) - 1) // 2
    require(len(pairwise) == expected_pair_count, "Pairwise row count is incomplete")
    pair_index = {
        stream_hash: index for index, (stream_hash, _) in enumerate(unique_streams)
    }
    if unique_streams and all(
        np.array_equal(item["ids"], unique_streams[0][1]["ids"])
        for _, item in unique_streams
    ):
        matrix_values = np.vstack([item["values"] for _, item in unique_streams])
        centered = matrix_values - matrix_values.mean(axis=1, keepdims=True)
        norm = np.linalg.norm(centered, axis=1)
        centered = centered / np.where(norm == 0, np.nan, norm)[:, None]
        pair_pearson = centered @ centered.T
        ranks = rankdata(matrix_values, axis=1, method="average")
        centered_ranks = ranks - ranks.mean(axis=1, keepdims=True)
        rank_norm = np.linalg.norm(centered_ranks, axis=1)
        centered_ranks = centered_ranks / np.where(rank_norm == 0, np.nan, rank_norm)[:, None]
        pair_spearman = centered_ranks @ centered_ranks.T
        for row in pairwise:
            left = pair_index[row["prediction_stream_sha256_a"]]
            right = pair_index[row["prediction_stream_sha256_b"]]
            _check_numeric(row["pearson"], _finite_or_none(pair_pearson[left, right]), "pairwise Pearson")
            _check_numeric(row["spearman"], _finite_or_none(pair_spearman[left, right]), "pairwise Spearman")
    else:
        for row in pairwise:
            left = recomputed_streams[row["prediction_stream_sha256_a"]]
            right = recomputed_streams[row["prediction_stream_sha256_b"]]
            left_map = dict(zip(left["ids"], left["values"]))
            right_map = dict(zip(right["ids"], right["values"]))
            common = sorted(left_map.keys() & right_map.keys())
            a = np.asarray([left_map[key] for key in common])
            b = np.asarray([right_map[key] for key in common])
            _check_numeric(row["pearson"], _pearson(a, b), "pairwise Pearson")
            _check_numeric(row["spearman"], _spearman(a, b), "pairwise Spearman")

    duplicate_rows = _read_csv(run_dir / "DUPLICATE_FILES.csv")
    expected_duplicates: dict[tuple[str, str], int] = {}
    file_hash_groups: dict[str, int] = {}
    for row in inventory:
        file_hash_groups[row["sha256"]] = file_hash_groups.get(row["sha256"], 0) + 1
    expected_duplicates.update(
        {
            ("BYTE_IDENTICAL_CSV", digest): count
            for digest, count in file_hash_groups.items()
            if count > 1
        }
    )
    stream_paths: dict[str, set[str]] = {}
    for row in matrix:
        stream_paths.setdefault(row["prediction_stream_sha256"], set()).add(
            row["candidate_path"]
        )
    expected_duplicates.update(
        {
            ("IDENTICAL_ID_ALIGNED_PREDICTION_STREAM", digest): len(paths)
            for digest, paths in stream_paths.items()
            if len(paths) > 1
        }
    )
    actual_duplicates = {
        (row["duplicate_type"], row["duplicate_hash"]): int(row["file_count"])
        for row in duplicate_rows
    }
    require(
        actual_duplicates == expected_duplicates,
        "Duplicate file/stream groups do not match independently recomputed hashes",
    )

    verified = {
        "audit_id": audit["audit_id"],
        "passed": True,
        "champion_sha256_verified": True,
        "verified_input_csv_count": len(inventory),
        "verified_output_count": len(manifest["outputs"]) + (
            0 if "VERIFICATION.json" in manifest["outputs"] else 1
        ),
        "verified_prediction_metric_rows": metric_count,
        "verified_unique_prediction_streams": len(unique_streams),
        "verified_pairwise_correlation_rows": len(pairwise),
        "verified_oof_files_excluded": len(oof_paths),
        "verified_oof_prediction_columns_excluded": sum(
            row.get("classification") == "OOF_TRAIN_PREDICTION" for row in alignment
        ),
        "id_aligned_recalculation": True,
        "pairwise_recalculation": True,
    }
    verification_path = run_dir / "VERIFICATION.json"
    verification_path.write_text(
        json.dumps(verified, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    manifest["sources"]["verify_exp12.py"] = {
        "path": str(Path(__file__).resolve()),
        "sha256": file_hash(Path(__file__).resolve()),
        "size_bytes": Path(__file__).stat().st_size,
    }
    manifest["outputs"]["VERIFICATION.json"] = {
        "sha256": file_hash(verification_path),
        "size_bytes": verification_path.stat().st_size,
    }
    manifest_path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False, sort_keys=True, allow_nan=False)
        + "\n",
        encoding="utf-8",
        newline="\n",
    )
    return verified


def _finite_or_none(value: Any) -> float | None:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    return numeric if math.isfinite(numeric) else None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN_DIR)
    args = parser.parse_args()
    print(json.dumps(verify(args.run_dir), indent=2))


if __name__ == "__main__":
    main()
