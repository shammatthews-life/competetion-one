from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import norm, pearsonr, rankdata, spearmanr


WORKSPACE = Path(__file__).resolve().parents[1]
GCI_ROOT = Path(r"D:\gci")
DATA_DIR = GCI_ROOT / "inpuy"
HISTORICAL_CACHE = GCI_ROOT / "champion" / "01_EXP85_RECOVERY" / "cache"
ANCHOR_PATH = (
    GCI_ROOT
    / "champion"
    / "02_EXP85_REBUILT"
    / "artifacts"
    / "v_20261027_precision_transductive_apex.csv"
)
REFERENCE_PATH = WORKSPACE / "reference_artifacts" / "historical_champion.csv"
REFERENCE_SOURCE_PATH = (
    GCI_ROOT
    / "champion"
    / "00_HISTORICAL_REFERENCE"
    / "EXP85_HISTORICAL_CHAMPION.csv"
)
PRIOR_CANDIDATE_PATH = WORKSPACE / "candidate_artifacts" / "track1_v5_rebuilt.csv"
PRIOR_CANDIDATE_SOURCE_PATH = (
    GCI_ROOT
    / "champion"
    / "08_SUBMISSIONS"
    / "submission_exp85_recovery_v5_rebuilt.csv"
)
ALTERNATE_ANCHOR_PATH = (
    GCI_ROOT
    / "champion"
    / "02_EXP85_REBUILT"
    / "rebuilt_artifacts"
    / "v_20261027_precision_transductive_apex.csv"
)
NOTEBOOK_PATH = WORKSPACE / "source" / "exp85_recovery_v5_original.ipynb"
PHASE3_SOURCE_PATH = (
    GCI_ROOT
    / "champion"
    / "02_EXP85_REBUILT"
    / "source"
    / "reconstruct_transductive_pyramid.py"
)
EQUIVALENCE_REPORT_PATH = (
    GCI_ROOT
    / "champion"
    / "01_EXP85_RECOVERY"
    / "reports"
    / "EXP85_ARTIFACT_EQUIVALENCE.csv"
)

TRAIN_ROWS = 171202
TEST_ROWS = 61500
EXPECTED_HASHES = {
    "raw_test": "dfb7aeb96e5af3d8616acaaba6b23cec8680845cfd525f7c08de15c6d2d42c65",
    "sample_submission": "6e5ad587b7c0b3d9fa39c90ce03e161ff5ef292193e117a6a02b11075dffe7db",
    "anchor74": "315b0bd80e4698240a7a902648a157d84da1480dea77901cdb60892502a7b55b",
    "v17": "c1dfacea535ddac5d46a6d69ffcbbddf05ccbe876fc0fd115742feb91a701223",
    "v55": "995b0e80162b706e49e2d4b5551a34244bd33f6e8ae4893de1fd1fad38cbbc0f",
    "xgb10_test": "5a8d8a23b2f36ac6f27703b1f8c4b0b29aa15fefe4292edc4a628702fff76195",
    "historical_reference": "78c31bb4e4534330479faa26861a49559dcadc40c7eac45620b92d7d9e6ea947",
    "prior_candidate": "91aa95c9501efb358e98b47e2def6678faa6c2f4ba34431e4f9d0b4d4b792482",
    "notebook": "d7d16d00599061b041e2b85619eb4dbc3f3607d437766c9c1e72d928a913f7b7",
    "phase3_source": "a99bd2ba02ed673a6660eb67c39fb16e80d9d01d4560e02ef3ba3ae0fefbd0f4",
    "artifact_equivalence_report": "01d5c090aab9b18dc1bb8470e12fb17909b406362724b415036954407181a8ad",
}
SUITE_WEIGHTS = {
    "v17_e1Aet": 0.2932,
    "xgb10": 0.2596,
    "v17_final": 0.2051,
    "v55_cbprod": 0.0874,
    "v17_e1et": 0.0489,
    "v55_deep_et": 0.0442,
    "v55_allfam": 0.0383,
    "v55_v2_mono": 0.0233,
}
SUITE_KEYS = tuple(SUITE_WEIGHTS)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def require_hash(path: Path, expected_sha256: str, label: str) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"{label} is missing: {path}")
    actual = sha256_file(path)
    if actual != expected_sha256:
        raise ValueError(
            f"{label} SHA-256 mismatch: expected {expected_sha256}, got {actual} "
            f"at {path}"
        )
    return {
        "path": str(path.resolve()),
        "size_bytes": path.stat().st_size,
        "sha256": actual,
    }


def rank_uniform(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values)
    if (
        values.ndim != 1
        or values.size == 0
        or not np.issubdtype(values.dtype, np.number)
        or np.issubdtype(values.dtype, np.complexfloating)
        or not np.isfinite(values).all()
    ):
        raise ValueError("Rank input must be a non-empty finite one-dimensional vector.")
    return (rankdata(values, method="average") - 0.5) / values.size


def probit(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values)
    if (
        values.ndim != 1
        or not np.issubdtype(values.dtype, np.number)
        or np.issubdtype(values.dtype, np.complexfloating)
        or not np.isfinite(values).all()
    ):
        raise ValueError("Probit input must be a finite one-dimensional vector.")
    return norm.ppf(np.clip(values, 1e-6, 1.0 - 1e-6))


def validate_probability_vector(
    values: np.ndarray, name: str, expected_rows: int
) -> np.ndarray:
    values = np.asarray(values)
    if values.shape != (expected_rows,):
        raise ValueError(
            f"{name} must have shape ({expected_rows},), got {values.shape}."
        )
    if (
        not np.issubdtype(values.dtype, np.number)
        or np.issubdtype(values.dtype, np.complexfloating)
        or not np.isfinite(values).all()
    ):
        raise ValueError(f"{name} contains non-finite values.")
    if np.any((values < 0.0) | (values > 1.0)):
        raise ValueError(f"{name} contains values outside [0, 1].")
    return values


def suite_probit(members: dict[str, np.ndarray]) -> np.ndarray:
    if set(members) != set(SUITE_KEYS):
        missing = sorted(set(SUITE_KEYS) - set(members))
        extra = sorted(set(members) - set(SUITE_KEYS))
        raise ValueError(f"Suite member mapping mismatch: missing={missing}, extra={extra}")
    arrays = {name: np.asarray(values) for name, values in members.items()}
    if any(values.ndim != 1 for values in arrays.values()):
        raise ValueError("Suite members must be one-dimensional vectors.")
    sizes = {values.shape[0] for values in arrays.values()}
    if len(sizes) != 1:
        raise ValueError("Suite members have inconsistent row counts.")
    n_rows = sizes.pop()
    if n_rows == 0:
        raise ValueError("Suite members are empty.")
    weight_total = sum(SUITE_WEIGHTS.values())
    if not np.isclose(weight_total, 1.0, rtol=0.0, atol=1e-12):
        raise ValueError(f"Historical suite weights must sum to 1, got {weight_total}.")

    combined = np.zeros(n_rows, dtype=np.float64)
    for name, weight in SUITE_WEIGHTS.items():
        values = validate_probability_vector(arrays[name], name, n_rows)
        combined += (weight / weight_total) * probit(rank_uniform(values))
    return combined


def compose_candidate(
    anchor_values: np.ndarray, suite_z: np.ndarray
) -> np.ndarray:
    anchor_values = np.asarray(anchor_values, dtype=np.float64)
    suite_z = np.asarray(suite_z, dtype=np.float64)
    if (
        anchor_values.ndim != 1
        or suite_z.ndim != 1
        or anchor_values.shape != suite_z.shape
        or not np.isfinite(anchor_values).all()
        or not np.isfinite(suite_z).all()
    ):
        raise ValueError("Anchor and suite inputs must be aligned finite vectors.")
    if np.any((anchor_values < 0.0) | (anchor_values > 1.0)):
        raise ValueError("Anchor74 values must lie in [0, 1].")
    z_macro = 0.85 * probit(anchor_values) + 0.15 * suite_z
    final = (rankdata(z_macro, method="ordinal") - 0.5) / anchor_values.size
    if not np.isfinite(final).all() or np.unique(final).size != final.size:
        raise ValueError("Final output must be finite, distinct percentile ranks.")
    return final


def read_submission(path: Path, label: str) -> pd.DataFrame:
    frame = pd.read_csv(path, dtype={"SK_ID_CURR": "int64"})
    if list(frame.columns) != ["SK_ID_CURR", "TARGET"]:
        raise ValueError(f"{label} must have schema SK_ID_CURR,TARGET: {path}")
    if frame.empty or frame["SK_ID_CURR"].isna().any():
        raise ValueError(f"{label} contains no rows or missing IDs.")
    if frame["SK_ID_CURR"].duplicated().any():
        raise ValueError(f"{label} contains duplicate IDs.")
    predictions = pd.to_numeric(frame["TARGET"], errors="coerce").to_numpy(
        dtype=np.float64
    )
    if not np.isfinite(predictions).all():
        raise ValueError(f"{label} contains non-finite predictions.")
    if np.any((predictions < 0.0) | (predictions > 1.0)):
        raise ValueError(f"{label} has predictions outside [0, 1].")
    frame["TARGET"] = predictions
    return frame


def id_aligned_metrics(
    reference: pd.DataFrame, candidate: pd.DataFrame
) -> dict[str, Any]:
    reference_ids = set(reference["SK_ID_CURR"])
    candidate_ids = set(candidate["SK_ID_CURR"])
    if reference_ids != candidate_ids or len(reference) != len(candidate):
        raise ValueError(
            "Reference and candidate IDs differ: "
            f"missing={len(reference_ids - candidate_ids)}, "
            f"extra={len(candidate_ids - reference_ids)}."
        )
    aligned_reference = reference.set_index("SK_ID_CURR").reindex(
        candidate["SK_ID_CURR"]
    )
    if aligned_reference["TARGET"].isna().any():
        raise ValueError("Reference ID alignment produced missing predictions.")
    ref = aligned_reference["TARGET"].to_numpy(dtype=np.float64)
    cand = candidate["TARGET"].to_numpy(dtype=np.float64)
    difference = cand - ref
    ref_rank = rankdata(ref, method="average")
    cand_rank = rankdata(cand, method="average")
    displacement = np.abs(ref_rank - cand_rank)
    return {
        "pearson": float(pearsonr(ref, cand).statistic),
        "spearman": float(spearmanr(ref, cand).statistic),
        "mae": float(np.mean(np.abs(difference))),
        "rmse": float(np.sqrt(np.mean(np.square(difference)))),
        "maximum_absolute_difference": float(np.max(np.abs(difference))),
        "mean_rank_displacement": float(np.mean(displacement)),
        "maximum_rank_displacement": int(np.max(displacement)),
        "exact_prediction_matches": int(np.count_nonzero(ref == cand)),
        "exact_rank_matches": int(np.count_nonzero(displacement == 0)),
        "aligned_rows": int(len(candidate)),
        "alignment": "one-to-one SK_ID_CURR join in raw test ID order",
    }


def load_suite_members(
    v17_path: Path, v55_path: Path, xgb10_path: Path
) -> tuple[dict[str, np.ndarray], dict[str, list[str]]]:
    with np.load(v17_path, allow_pickle=False) as archive:
        v17_keys = list(archive.files)
        v17 = {
            "v17_e1Aet": archive["test_e1Aet"].copy(),
            "v17_final": archive["final_test"].copy(),
            "v17_e1et": archive["test_e1et"].copy(),
        }
    with np.load(v55_path, allow_pickle=False) as archive:
        v55_keys = list(archive.files)
        v55 = {
            "v55_cbprod": archive["test_cb_product"].copy(),
            "v55_deep_et": archive["test_deep_et"].copy(),
            "v55_allfam": archive["test_v2_allfam_oh"].copy(),
            "v55_v2_mono": archive["test_v2_mono"].copy(),
        }
    xgb10 = np.load(xgb10_path, allow_pickle=False)
    members = {**v17, "xgb10": xgb10, **v55}
    for name, values in members.items():
        validate_probability_vector(values, name, TEST_ROWS)
    return members, {"v17_archive_keys": v17_keys, "v55_archive_keys": v55_keys}


def build_report_markdown(report: dict[str, Any]) -> str:
    metrics = report["comparison_to_historical_champion"]
    lines = [
        "# Historical EXP85 V5 Cache Reconstruction",
        "",
        f"- Run ID: `{report['run_id']}`",
        f"- Status: **{report['status']}**",
        "- Classification: **hybrid historical-cache reconstruction**.",
        "- Official competition score: **NOT EVALUATED**.",
        "",
        "## Reconstructed submission",
        "",
        f"- SHA-256: `{report['candidate']['sha256']}`",
        f"- Previous V5 candidate SHA-256: `{report['candidate']['prior_candidate_sha256']}`",
        f"- Byte-identical to previous V5 candidate: `{report['candidate']['byte_identical_to_prior_v5']}`",
        f"- Rows/schema: `{report['candidate']['rows']}` / `{','.join(report['candidate']['schema'])}`",
        f"- IDs match raw test and sample order: `{report['candidate']['ids_match_raw_test_and_sample_order']}`",
        "",
        "## ID-aligned comparison to the historical champion",
        "",
        f"- Pearson: `{metrics['pearson']:.16f}`",
        f"- Spearman: `{metrics['spearman']:.16f}`",
        f"- MAE / RMSE: `{metrics['mae']:.16f}` / `{metrics['rmse']:.16f}`",
        f"- Maximum absolute difference: `{metrics['maximum_absolute_difference']:.16f}`",
        f"- Mean / maximum rank displacement: `{metrics['mean_rank_displacement']:.6f}` / `{metrics['maximum_rank_displacement']}`",
        f"- Exactly matching predictions: `{metrics['exact_prediction_matches']}`",
        "",
        "## Artifact dependency graph",
        "",
        "| Artifact | Exact path | SHA-256 | Producer / lineage status | Consumer |",
        "|---|---|---|---|---|",
    ]
    for item in report["dependency_graph"]:
        lines.append(
            f"| {item['artifact']} | `{item['path']}` | `{item['sha256']}` | "
            f"{item['producer_lineage']} | {item['consumer']} |"
        )
    lines.extend(
        [
            "",
            "## Audit findings",
            "",
            "- The original notebook's exact 85/15 probit formula and eight explicit suite weights were used.",
            "- The correct V5 Anchor74 generation is pinned by its expected SHA-256. Alternate same-named Anchor74 outputs were not substituted.",
            "- V17/V55/XGB10 cache producers are not fully provenance-bound. Recovered raw-data generators produce different arrays; their outputs were not substituted.",
            "- Historical suite arrays contain no embedded SK_ID_CURR IDs or verified row-ID sidecars. Their positional order is retained as in the original notebook; the final submission's ID order is independently checked.",
            "- This is not a full raw-data historical pipeline reconstruction and does not certify the official score.",
            "",
            "## Incomplete dependencies",
            "",
        ]
    )
    for item in report["unresolved_dependencies"]:
        lines.append(f"- **{item['item']}**: {item['status']}. {item['impact']}")
    lines.extend(["", ""])
    return "\n".join(lines)


def run(run_id: str | None = None) -> dict[str, Any]:
    if run_id is None:
        run_id = datetime.now(timezone.utc).strftime(
            "%Y%m%dT%H%M%SZ_historical_cache_v5"
        )
    if not re.fullmatch(r"[A-Za-z0-9._-]+", run_id):
        raise ValueError("Run ID may contain only letters, digits, dot, underscore, and hyphen.")
    run_dir = WORKSPACE / "runs" / run_id
    run_dir.mkdir(parents=False, exist_ok=False)
    started_at = datetime.now(timezone.utc).isoformat()
    started = time.perf_counter()
    try:
        inputs = {
            "raw_test": require_hash(
                DATA_DIR / "test.csv", EXPECTED_HASHES["raw_test"], "Raw test CSV"
            ),
            "sample_submission": require_hash(
                DATA_DIR / "sample_submission.csv",
                EXPECTED_HASHES["sample_submission"],
                "Sample submission",
            ),
            "anchor74_v5_rebuilt": require_hash(
                ANCHOR_PATH, EXPECTED_HASHES["anchor74"], "V5 Anchor74"
            ),
            "v17_historical_bundle": require_hash(
                HISTORICAL_CACHE / "v_17_full_suite_bundle.npz",
                EXPECTED_HASHES["v17"],
                "Historical V17 bundle",
            ),
            "v55_historical_members": require_hash(
                HISTORICAL_CACHE / "v55_members.npz",
                EXPECTED_HASHES["v55"],
                "Historical V55 bundle",
            ),
            "xgb10_historical_test": require_hash(
                HISTORICAL_CACHE / "test_10f_xgb.npy",
                EXPECTED_HASHES["xgb10_test"],
                "Historical XGB10 test vector",
            ),
            "historical_champion": require_hash(
                REFERENCE_PATH,
                EXPECTED_HASHES["historical_reference"],
                "Historical champion reference copy",
            ),
            "historical_champion_source": require_hash(
                REFERENCE_SOURCE_PATH,
                EXPECTED_HASHES["historical_reference"],
                "Historical champion source",
            ),
            "prior_v5_candidate": require_hash(
                PRIOR_CANDIDATE_PATH,
                EXPECTED_HASHES["prior_candidate"],
                "Prior V5 candidate copy",
            ),
            "prior_v5_candidate_source": require_hash(
                PRIOR_CANDIDATE_SOURCE_PATH,
                EXPECTED_HASHES["prior_candidate"],
                "Prior V5 candidate source",
            ),
            "original_notebook": require_hash(
                NOTEBOOK_PATH, EXPECTED_HASHES["notebook"], "Original V5 notebook"
            ),
            "phase3_source": require_hash(
                PHASE3_SOURCE_PATH,
                EXPECTED_HASHES["phase3_source"],
                "Phase 3 source snapshot",
            ),
            "artifact_equivalence_evidence": require_hash(
                EQUIVALENCE_REPORT_PATH,
                EXPECTED_HASHES["artifact_equivalence_report"],
                "Cache equivalence report",
            ),
        }
        alternate_anchor = {
            "path": str(ALTERNATE_ANCHOR_PATH.resolve()),
            "exists": ALTERNATE_ANCHOR_PATH.is_file(),
            "sha256": (
                sha256_file(ALTERNATE_ANCHOR_PATH)
                if ALTERNATE_ANCHOR_PATH.is_file()
                else None
            ),
            "used": False,
            "reason": (
                "Different Anchor74 generation; hash is not the V5 notebook input."
            ),
        }

        raw_test_ids = pd.read_csv(
            DATA_DIR / "test.csv", usecols=["SK_ID_CURR"]
        )["SK_ID_CURR"]
        sample_ids = pd.read_csv(
            DATA_DIR / "sample_submission.csv", usecols=["SK_ID_CURR"]
        )["SK_ID_CURR"]
        if (
            len(raw_test_ids) != TEST_ROWS
            or raw_test_ids.isna().any()
            or raw_test_ids.duplicated().any()
            or not raw_test_ids.equals(sample_ids)
        ):
            raise ValueError("Raw test/sample IDs are not 61,500 unique ordered matches.")

        anchor_frame = read_submission(ANCHOR_PATH, "V5 Anchor74")
        if not anchor_frame["SK_ID_CURR"].equals(raw_test_ids):
            raise ValueError("Pinned V5 Anchor74 IDs do not match raw test order.")
        anchor = validate_probability_vector(
            anchor_frame["TARGET"].to_numpy(), "Anchor74", TEST_ROWS
        )
        members, archive_keys = load_suite_members(
            HISTORICAL_CACHE / "v_17_full_suite_bundle.npz",
            HISTORICAL_CACHE / "v55_members.npz",
            HISTORICAL_CACHE / "test_10f_xgb.npy",
        )
        suite_z = suite_probit(members)
        candidate_values = compose_candidate(anchor, suite_z)
        candidate_frame = pd.DataFrame(
            {"SK_ID_CURR": raw_test_ids, "TARGET": candidate_values}
        )
        candidate_path = run_dir / "submission_historical_cache_v5.csv"
        candidate_frame.to_csv(
            candidate_path,
            index=False,
            float_format="%.10f",
            lineterminator="\r\n",
        )

        candidate = read_submission(candidate_path, "Reconstructed candidate")
        reference = read_submission(REFERENCE_PATH, "Historical champion")
        prior = read_submission(PRIOR_CANDIDATE_PATH, "Prior V5 candidate")
        if not candidate["SK_ID_CURR"].equals(raw_test_ids):
            raise ValueError("Reconstructed candidate IDs do not match raw test order.")
        if not prior["SK_ID_CURR"].equals(raw_test_ids):
            raise ValueError("Prior V5 candidate IDs do not match raw test order.")
        if not reference["SK_ID_CURR"].equals(raw_test_ids):
            raise ValueError("Historical reference IDs do not match raw test order.")

        metrics = id_aligned_metrics(reference, candidate)
        candidate_sha = sha256_file(candidate_path)
        prior_values_equal = np.array_equal(
            candidate["TARGET"].to_numpy(), prior["TARGET"].to_numpy()
        )
        prior_bytes_equal = candidate_sha == EXPECTED_HASHES["prior_candidate"]
        if prior_values_equal != prior_bytes_equal:
            raise RuntimeError(
                "Candidate value/byte identity checks disagree with the prior V5 artifact."
            )

        producer_scripts = {
            "v17_recovered_generator": (
                GCI_ROOT / "gemini" / "champion" / "cacherecreation"
                / "recreate_v17_bundle.py"
            ),
            "v55_recovered_generator": (
                GCI_ROOT / "gemini" / "champion" / "cacherecreation"
                / "recreate_v55_members.py"
            ),
            "xgb10_recovered_generator": (
                GCI_ROOT / "gemini" / "champion" / "cacherecreation"
                / "recreate_xgb10_and_labels.py"
            ),
        }
        supporting_sources = {}
        for name, path in producer_scripts.items():
            supporting_sources[name] = (
                {
                    "path": str(path.resolve()),
                    "exists": True,
                    "sha256": sha256_file(path),
                    "exact_historical_cache_producer_verified": False,
                }
                if path.is_file()
                else {"path": str(path), "exists": False}
            )

        generated_at = datetime.now(timezone.utc).isoformat()
        report = {
            "run_id": run_id,
            "status": (
                "REPRODUCED_EXACT_PRIOR_V5"
                if prior_bytes_equal
                else "RECONSTRUCTED_DIFFERENT_FROM_PRIOR_V5"
            ),
            "started_at_utc": started_at,
            "completed_at_utc": generated_at,
            "elapsed_seconds": time.perf_counter() - started,
            "classification": "HYBRID_HISTORICAL_CACHE_RECONSTRUCTION",
            "historical_prediction_cache_inputs": True,
            "full_raw_data_historical_pipeline_reconstruction": False,
            "official_competition_score": "NOT_EVALUATED",
            "formula": {
                "suite_member_weights": SUITE_WEIGHTS,
                "suite_weight_sum": sum(SUITE_WEIGHTS.values()),
                "member_transform": "(average rank - 0.5) / n; probit clip eps=1e-6",
                "suite_aggregation": "weighted sum of eight transformed test members",
                "final_transform": (
                    "ordinal rank(0.85 * probit(Anchor74) + "
                    "0.15 * SuiteZ); output=(rank - 0.5) / 61500"
                ),
                "anchor_weight": 0.85,
                "suite_weight": 0.15,
            },
            "input_artifacts": inputs,
            "rejected_alternate_anchor": alternate_anchor,
            "supporting_recovered_generator_sources": supporting_sources,
            "suite_archive_keys": archive_keys,
            "cache_row_identity": {
                "cache_arrays_embed_ids": False,
                "verified_row_id_sidecars_found": False,
                "assembly_order": (
                    "Historical cache array order is consumed unchanged, as in the "
                    "original notebook. Final candidate IDs are checked against raw "
                    "test and sample-submission order."
                ),
            },
            "candidate": {
                "path": str(candidate_path.resolve()),
                "sha256": candidate_sha,
                "rows": int(len(candidate)),
                "schema": list(candidate.columns),
                "unique_ids": bool(candidate["SK_ID_CURR"].is_unique),
                "ids_match_raw_test_and_sample_order": True,
                "finite_predictions": bool(
                    np.isfinite(candidate["TARGET"].to_numpy()).all()
                ),
                "probability_min": float(candidate["TARGET"].min()),
                "probability_max": float(candidate["TARGET"].max()),
                "byte_identical_to_prior_v5": prior_bytes_equal,
                "value_identical_to_prior_v5": prior_values_equal,
                "prior_candidate_sha256": inputs["prior_v5_candidate"]["sha256"],
            },
            "historical_reference": {
                "path": str(REFERENCE_PATH.resolve()),
                "sha256": inputs["historical_champion"]["sha256"],
            },
            "comparison_to_historical_champion": metrics,
            "dependency_graph": [
                {
                    "artifact": "Rebuilt Anchor74 used by V5",
                    "path": str(ANCHOR_PATH),
                    "sha256": inputs["anchor74_v5_rebuilt"]["sha256"],
                    "producer_lineage": (
                        "Exact notebook input identified; producer execution is not "
                        "bound to this file by a run manifest."
                    ),
                    "consumer": "85% Anchor74 probit branch",
                },
                {
                    "artifact": "V17 historical test bundle",
                    "path": str(HISTORICAL_CACHE / "v_17_full_suite_bundle.npz"),
                    "sha256": inputs["v17_historical_bundle"]["sha256"],
                    "producer_lineage": (
                        "Exact historical producer run unavailable; recovered "
                        "generator output is not byte/array-identical."
                    ),
                    "consumer": "Three explicitly named suite members",
                },
                {
                    "artifact": "V55 historical test members",
                    "path": str(HISTORICAL_CACHE / "v55_members.npz"),
                    "sha256": inputs["v55_historical_members"]["sha256"],
                    "producer_lineage": (
                        "Exact historical producer run unavailable; recovered "
                        "generator output is not byte/array-identical."
                    ),
                    "consumer": "Four explicitly named suite members",
                },
                {
                    "artifact": "XGB10 historical test vector",
                    "path": str(HISTORICAL_CACHE / "test_10f_xgb.npy"),
                    "sha256": inputs["xgb10_historical_test"]["sha256"],
                    "producer_lineage": (
                        "Exact historical producer run unavailable; recovered "
                        "generator output is not byte/array-identical."
                    ),
                    "consumer": "One explicitly named suite member",
                },
                {
                    "artifact": "Raw test IDs and sample submission",
                    "path": str(DATA_DIR),
                    "sha256": (
                        f"test={inputs['raw_test']['sha256']}; "
                        f"sample={inputs['sample_submission']['sha256']}"
                    ),
                    "producer_lineage": "Supplied competition inputs; hash-verified.",
                    "consumer": "ID-order validation only; not model features.",
                },
            ],
            "unresolved_dependencies": [
                {
                    "item": "Original V17/V55/XGB10 training execution metadata and cache ID sidecars",
                    "status": "UNAVAILABLE",
                    "impact": (
                        "The historical vectors are exactly identified and reproduce "
                        "the prior candidate, but their original per-row ID mapping "
                        "and training environment cannot be independently certified."
                    ),
                },
                {
                    "item": "Exact Anchor74 producer run manifest for SHA-256 "
                    + inputs["anchor74_v5_rebuilt"]["sha256"],
                    "status": "UNAVAILABLE",
                    "impact": (
                        "The exact V5 input is present and hash-pinned; its upstream "
                        "training/cache ancestry is not fully proven."
                    ),
                },
                {
                    "item": "Official platform record linking score 0.801340 to this candidate hash",
                    "status": "UNAVAILABLE",
                    "impact": (
                        "The local reconstruction proves numeric correlation only; "
                        "the official score is not claimed."
                    ),
                },
            ],
        }
        report_path = run_dir / "reconstruction_report.json"
        report_path.write_text(
            json.dumps(report, indent=2, allow_nan=False) + "\n",
            encoding="utf-8",
            newline="\n",
        )
        markdown_path = run_dir / "reconstruction_report.md"
        markdown_path.write_text(
            build_report_markdown(report), encoding="utf-8", newline="\n"
        )
        output_paths = (candidate_path, report_path, markdown_path)
        run_manifest = {
            "run_id": run_id,
            "status": report["status"],
            "command": [
                sys.executable,
                "-B",
                str(Path(__file__).resolve()),
                "--run-id",
                run_id,
            ],
            "exit_code": 0,
            "python_version": sys.version,
            "source": {
                "path": str(Path(__file__).resolve()),
                "sha256": sha256_file(Path(__file__).resolve()),
            },
            "inputs": inputs,
            "outputs": {
                path.name: {
                    "path": str(path.resolve()),
                    "size_bytes": path.stat().st_size,
                    "sha256": sha256_file(path),
                }
                for path in output_paths
            },
        }
        (run_dir / "run_manifest.json").write_text(
            json.dumps(run_manifest, indent=2, allow_nan=False) + "\n",
            encoding="utf-8",
            newline="\n",
        )
        print(json.dumps(report, indent=2, allow_nan=False), flush=True)
        return report
    except Exception as error:
        failure_path = run_dir / "failure_report.json"
        failure_path.write_text(
            json.dumps(
                {
                    "run_id": run_id,
                    "status": "FAILED",
                    "failed_at_utc": datetime.now(timezone.utc).isoformat(),
                    "error_type": type(error).__name__,
                    "error": str(error),
                    "traceback": traceback.format_exc(),
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
            newline="\n",
        )
        raise


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Reconstruct the verified historical-cache V5 submission into a new "
            "non-overwriting run directory."
        )
    )
    parser.add_argument("--run-id", help="Optional unique run ID for controlled execution.")
    args = parser.parse_args()
    report = run(args.run_id)
    return 0 if report["status"] == "REPRODUCED_EXACT_PRIOR_V5" else 2


if __name__ == "__main__":
    raise SystemExit(main())
