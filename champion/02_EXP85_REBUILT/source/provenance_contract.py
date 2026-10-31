from __future__ import annotations

from typing import Any


EXPECTED_GENERATORS = {
    "generate_xgb10_from_scratch.py",
    "generate_v55_from_scratch.py",
    "generate_v17_from_scratch.py",
    "generate_anchor74_from_scratch.py",
}
EXPECTED_ARTIFACTS = {
    "y_train.npy",
    "oof_10f_xgb.npy",
    "test_10f_xgb.npy",
    "v55_members.npz",
    "v_17_full_suite_bundle.npz",
    "v_20261027_precision_transductive_apex.csv",
}


def determine_provenance_status(
    *, verify_only: bool, fast: bool, execution_records: list[dict[str, Any]]
) -> str:
    if verify_only:
        return "ARTIFACTS_VERIFIED"
    if fast:
        return "SMOKE_TEST_ONLY"
    if len(execution_records) != len(EXPECTED_GENERATORS):
        return "INCOMPLETE"
    if {record.get("script") for record in execution_records} != EXPECTED_GENERATORS:
        return "INCOMPLETE"
    if any(record.get("exit_code") != 0 for record in execution_records):
        return "INCOMPLETE"
    return "RAW_REBUILD_CERTIFIED"


def require_raw_rebuild_manifest(manifest: dict[str, Any]) -> None:
    errors: list[str] = []
    expected_status = "RAW_REBUILD_CERTIFIED"
    if manifest.get("provenance_status") != expected_status:
        errors.append("top-level provenance_status is not RAW_REBUILD_CERTIFIED")
    summary = manifest.get("verification_summary", {})
    if summary.get("provenance_status") != expected_status:
        errors.append("verification_summary provenance_status is not RAW_REBUILD_CERTIFIED")
    if manifest.get("execution_mode") != "full_rebuild":
        errors.append("execution_mode is not full_rebuild")

    executions = manifest.get("execution_telemetry")
    if not isinstance(executions, list) or len(executions) != len(EXPECTED_GENERATORS):
        errors.append("four generator execution records are required")
        executions = []
    else:
        executed_names = {record.get("script") for record in executions}
        if executed_names != EXPECTED_GENERATORS:
            errors.append("execution records do not cover the four required generators")
        if any(record.get("exit_code") != 0 for record in executions):
            errors.append("every generator must have exit_code 0")
        if any(
            "--fast" in record.get("command", [])
            if isinstance(record.get("command"), list)
            else "--fast" in str(record.get("command", ""))
            for record in executions
        ):
            errors.append("fast/smoke executions cannot certify a raw rebuild")

    source_hashes = manifest.get("generator_source_hashes", {})
    if not isinstance(source_hashes, dict) or not EXPECTED_GENERATORS.issubset(
        source_hashes
    ):
        errors.append("source hashes for all four required generators are required")
    else:
        for name in EXPECTED_GENERATORS:
            value = source_hashes.get(name)
            if not isinstance(value, str) or len(value) != 64:
                errors.append(f"invalid SHA-256 source hash for {name}")

    raw_inputs = manifest.get("raw_inputs", {})
    for name in ("train_csv", "test_csv"):
        value = raw_inputs.get(name, {}).get("sha256")
        if not isinstance(value, str) or len(value) != 64:
            errors.append(f"raw input hash missing for {name}")

    artifacts = manifest.get("artifacts", {})
    if not isinstance(artifacts, dict) or not EXPECTED_ARTIFACTS.issubset(artifacts):
        errors.append("the six required generated artifacts are not all recorded")

    if errors:
        raise ValueError("Manifest is not eligible for full-scratch synthesis: " + "; ".join(errors))
