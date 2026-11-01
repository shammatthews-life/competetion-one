from __future__ import annotations

import csv
import hashlib
import json
import platform
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np


AUDIT_DIR = Path(__file__).resolve().parents[1]
REPO_ROOT = Path(__file__).resolve().parents[3]
GRAPH_PATH = AUDIT_DIR / "manifests" / "dependency_graph.json"
FEATURE_PARITY_REPORT = Path(
    r"D:\gci\champion\EXP85_MASTER_RECOVERY\runs\feature_parity_20261031_200506\parity_report.json"
)
FEATURE_PARITY_REPORT_SHA256 = "636ba1f46ecbf7ef8c8b7aceb5b9a07ce009c91e4406c770589f0dc9d3d3154c"
SELECTOR_MANIFEST = Path(
    r"D:\gci\champion\EXP85_MASTER_RECOVERY\runs\selector_replay_20261031_202500\selector_manifest.json"
)
SELECTOR_MANIFEST_SHA256 = "712ee5095064df60013e3fad80bbe7d002f4a1d781dc0a4d7ce906ad36214c4c"
FEATURE_COMPARISONS = {
    "historical_669_train": (171202, 669),
    "historical_669_test": (61500, 669),
    "historical_550_train": (171202, 550),
    "historical_550_test": (61500, 550),
}
OUTPUT_DIRS = (
    AUDIT_DIR / "manifests",
    AUDIT_DIR / "logs",
)

SOURCE_FILES = [
    ("historical_final_blend", r"D:\gci\gemini\build_exp85_optimal_macro_apex.py", "f4eeb19a5f3f161f186c318e5b942035e592cc001c710e18a679f589e39ed555"),
    ("historical_anchor_builder", r"D:\gci\gemini\build_precision_transductive_apex_champion.py", "7b80c196bad395a2aaba478d08dc2087af4280d0f2b41ff75e36010a984b1fdc"),
    ("historical_feature_builder", r"D:\gci\gemini\build_hasan_600_features.py", "cafd70abd8da3cb6d544c26b668ed61f329cd89ef98ff61cec3983ffc1c08341"),
    ("historical_feature_selector", r"D:\gci\gemini\screen_and_select_features.py", "0a881c5bb2090e77e6cb3020e0134ba172650fe3ecfb1f7701ae57a5e8caafb7"),
    ("raw_selector_replay_runner", r"D:\gci\champion\EXP85_MASTER_RECOVERY\source\replay_historical_selector.py", "1b57bd666f836f16b8a296537bbe4b339c047ea411ae5d23d66ec96705284103"),
    ("raw_zoo_training_runner", r"D:\gci\champion\EXP85_MASTER_RECOVERY\source\train_historical_anchor_zoo.py", "a1dd1aa87631ba79cd19a2138ca7400b10aec1509a87c088f55fcb5d8b19a1bc"),
    ("historical_zoo_trainer_a_source", r"D:\gci\gemini\train_hasan_master_zoo.py", "bb8a2e8d8205fcbdc534854899ca46bf7ffeddbff16ade82c618c578f8a51009"),
    ("historical_zoo_trainer_b_source", r"D:\gci\gemini\train_catboost_and_deep_lgb.py", "1c74ca4620ace7e927e8b9ec5c6bb0ecec9f6e4c577c9d01e44da1eff42f0290"),
    ("historical_vgold_builder", r"D:\gci\gemini\train_v_gold_grand_master_apex.py", "bcce858e3cba4f0d11fe727dec4e4f07c2fe6605322c588463d5c7b9b16c8edf"),
    ("historical_v080150_builder", r"D:\gci\gemini\solo_grand_master_apex_080.py", "38917ee975beeb5abb1ff522490732d42e351ef161fc179e0b5bf6ad44871756"),
    ("historical_v080200_builder", r"D:\gci\gemini\build_v80200_apex.py", "312e760a1f09c71620968cc80938cab6c612bf1576dfe2b474ba10f152053c28"),
    ("historical_level2_builder", r"D:\gci\gemini\build_level2_regime_meta_stacker.py", "f1224d8f351daa8b8058fca99ed1f2eb5aa35b98408e2c306beddce5fd0a8490"),
    ("historical_asymmetric_builder", r"D:\gci\gemini\build_asymmetric_convex_gated_champion.py", "23f14c6fd6f3f15e34efad05e805d6caa21a79702440110e8d31b5891a49fd64"),
    ("historical_optimal_transductive_builder", r"D:\gci\gemini\build_optimal_transductive_apex_champion.py", "e5d3febfc71e5194c05ce6db85358db5f9a93601619bea29ef3633dbd47e3011"),
    ("historical_v17_trainer", r"D:\gci\gemini\train_full_v16_suite.py", "eb20c21039c3a09f63367a538d9b46d2f34b8c04879b0b83fa4c29ef880bcec0"),
    ("v16_feature_builder", r"D:\gci\gemini\build_v16_features.py", None),
    ("historical_v55_trainer", r"D:\gci\gemini\train_clean_honors_champion.py", "fcc248cea1c7cc18e9277f676995e5578407ca7fc4493b34411ac9351697756a"),
    ("v17_raw_recreation", r"D:\gci\gemini\champion\cacherecreation\recreate_v17_bundle.py", "f9c15e292161a4dd8c2d71419d1a95134a5ebba4b8d366e0a86885b79a7c65a6"),
    ("v55_raw_recreation", r"D:\gci\gemini\champion\cacherecreation\recreate_v55_members.py", "3c2979bc8941c61d9f2ebbd0de725d15f09bca32245d9dcd46812bbb9e0cf290"),
    ("xgb10_raw_recreation", r"D:\gci\gemini\champion\cacherecreation\recreate_xgb10_and_labels.py", "7fbeaa89fb7b4473aefa9dbdb2c83b6ee337389b1ffc783b442a1640f9c1c900"),
    ("anchor74_raw_candidate", r"D:\gci\gemini\champion\cacherecreation\recreate_precision_anchor.py", "ad38e92988428f04a59f66ab9a3746cbaf01b633c6295b61f5876c6e03fc7dff"),
    ("historical_zoo_trainer_a", r"D:\gci\champion\EXP85_FINAL_RECONSTRUCTION\gemini_recovered\train_hasan_master_zoo.py", "bb8a2e8d8205fcbdc534854899ca46bf7ffeddbff16ade82c618c578f8a51009"),
    ("historical_zoo_trainer_b", r"D:\gci\champion\EXP85_FINAL_RECONSTRUCTION\gemini_recovered\train_catboost_and_deep_lgb.py", "1c74ca4620ace7e927e8b9ec5c6bb0ecec9f6e4c577c9d01e44da1eff42f0290"),
    ("anchor74_cache_replay", r"D:\gci\champion\EXP85_FINAL_RECONSTRUCTION\gemini_recovered\champion\dependency_audit\anchor74_audit\reproduce_anchor74.py", "08d7f5da963d2fb1b324dfd6884601e50b736ecc23313d2a25aa89a5a94ab79b"),
    ("v080041_export_reproducer", r"D:\gci\gemini\final_submission\reproduce_final_submission.py", "8a90d233723639297b1a433ee474b37ea1d00879fdb518ac39ee75a0fe575662"),
    ("origin_notebook_repo", r"D:\gci\_gci_push_tmp\champion\exp85_solo.ipynb", "d7d16d00599061b041e2b85619eb4dbc3f3607d437766c9c1e72d928a913f7b7"),
    ("origin_notebook_backup", r"D:\gci\_gci_push_tmp\champion\exp85_solo.pre_independent_wiring_backup.ipynb", "1ccfab28ef8534fc176c757c79250547c03a041a922851d683c82d0df1241d23"),
    ("origin_notebook_backup_pre_fix", r"D:\gci\_gci_push_tmp\champion\exp85_solo_backup_pre_fix.ipynb", "34b66ca02ca74f1c8ee80a24d80a519292f3d0d6f00e197e1900e16f94dfeb4b"),
    ("gemini_notebook_repo_copy", r"D:\gci\_gci_push_tmp\gemini\champion\exp85_solo.ipynb", "2640eaa9b828c9b9485497f9ce35060bd4d89db179820a5a42bb153039862312"),
    ("gemini_notebook", r"D:\gci\gemini\champion\exp85_solo.ipynb", "52996cf1a2c9b018195672ea00bc316cd6cf01bab8018f9956b55d8cae22c9a3"),
    ("gemini_recovery_notebook", r"D:\gci\gemini\champion\cacherecreation\exp85_recovery_v5.ipynb", None),
    ("gemini_rebuild_notebook", r"D:\gci\gemini\exp85_rebuilt\EXP85_LEGITIMATE_RECONSTRUCTION.ipynb", "05086c2a109209ce7cf1d2c37ec05982ce43988f43a825cd0aaa81a7db772686"),
    ("raw_xgb_generator", r"D:\gci\_gci_push_tmp\champion\02_EXP85_REBUILT\source\generate_xgb10_from_scratch.py", None),
    ("raw_v55_generator", r"D:\gci\_gci_push_tmp\champion\02_EXP85_REBUILT\source\generate_v55_from_scratch.py", None),
    ("raw_v17_generator", r"D:\gci\_gci_push_tmp\champion\02_EXP85_REBUILT\source\generate_v17_from_scratch.py", None),
    ("raw_anchor_generator", r"D:\gci\_gci_push_tmp\champion\02_EXP85_REBUILT\source\generate_anchor74_from_scratch.py", None),
    ("raw_v5_anchor_reconstructor", r"D:\gci\_gci_push_tmp\champion\02_EXP85_REBUILT\source\reconstruct_transductive_pyramid.py", "a99bd2ba02ed673a6660eb67c39fb16e80d9d01d4560e02ef3ba3ae0fefbd0f4"),
]

CSV_CHECKS = [
    ("raw_test", Path(r"D:\gci\inpuy\test.csv"), "SK_ID_CURR", None),
    ("sample_submission", Path(r"D:\gci\inpuy\sample_submission.csv"), "SK_ID_CURR", "TARGET"),
    ("historical_champion", Path(r"D:\gci\champion\00_HISTORICAL_REFERENCE\EXP85_HISTORICAL_CHAMPION.csv"), "SK_ID_CURR", "TARGET"),
    ("historical_v85", Path(r"D:\gci\gemini\v_20261027_optimal_macro_apex_v85.csv"), "SK_ID_CURR", "TARGET"),
    ("historical_v20", Path(r"D:\gci\gemini\v_20_apex_grand_champion_submission.csv"), "SK_ID_CURR", "TARGET"),
    ("historical_v_gold", Path(r"D:\gci\gemini\v_gold_grand_master_apex.csv"), "SK_ID_CURR", "TARGET"),
    ("historical_v080041", Path(r"D:\gci\gemini\v_solo_080041_pure_champion.csv"), "SK_ID_CURR", "TARGET"),
    ("historical_v080150", Path(r"D:\gci\gemini\v_solo_080150_micro_apex_fusion.csv"), "SK_ID_CURR", "TARGET"),
    ("historical_v080200", Path(r"D:\gci\gemini\v_solo_080200_grand_master_apex.csv"), "SK_ID_CURR", "TARGET"),
    ("historical_level2_stacker", Path(r"D:\gci\gemini\v_solo_080300_level2_regime_stacker.csv"), "SK_ID_CURR", "TARGET"),
    ("historical_asymmetric_apex", Path(r"D:\gci\gemini\v_20261027_asymmetric_tail_gated_apex.csv"), "SK_ID_CURR", "TARGET"),
    ("historical_optimal_transductive_apex", Path(r"D:\gci\gemini\v_20261027_optimal_transductive_apex.csv"), "SK_ID_CURR", "TARGET"),
    ("historical_precision_apex", Path(r"D:\gci\gemini\v_20261027_precision_transductive_apex.csv"), "SK_ID_CURR", "TARGET"),
    ("v5_candidate", Path(r"D:\gci\champion\08_SUBMISSIONS\submission_exp85_recovery_v5_rebuilt.csv"), "SK_ID_CURR", "TARGET"),
    ("fresh_raw_candidate", Path(r"D:\gci\_gci_push_tmp\champion\FINAL_RAW_REBUILD_AUDIT\runs\raw_full_gpu_20261031_235900\candidate_submission.csv"), "SK_ID_CURR", "TARGET"),
    ("raw_bedrock_candidate", Path(r"D:\gci\champion\EXP85_MASTER_RECOVERY\runs\bedrock_chain_rebuild_20261101\optimal\v_20261027_optimal_transductive_apex.csv"), "SK_ID_CURR", "TARGET"),
]


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_graph(path: Path = GRAPH_PATH) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as stream:
        value = json.load(stream)
    if not isinstance(value, dict) or not isinstance(value.get("nodes"), list):
        raise ValueError(f"Invalid dependency graph: {path}")
    return value


def read_csv_columns(path: Path, columns: set[str]) -> tuple[list[str], int, dict[str, list[str]]]:
    selected: dict[str, list[str]] = {column: [] for column in columns}
    row_count = 0
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames is None:
            raise ValueError(f"CSV has no header: {path}")
        headers = list(reader.fieldnames)
        for row in reader:
            row_count += 1
            for column in columns.intersection(headers):
                selected[column].append(row[column] or "")
    return headers, row_count, selected


def compare_id_order(expected: list[str], actual: list[str]) -> dict[str, Any]:
    return {
        "same_order": expected == actual,
        "same_count": len(expected) == len(actual),
        "expected_rows": len(expected),
        "actual_rows": len(actual),
        "expected_unique": len(expected) == len(set(expected)),
        "actual_unique": len(actual) == len(set(actual)),
    }


def _validate_feature_comparison(
    name: str, comparison: dict[str, Any], rows: int, columns: int
) -> list[str]:
    errors: list[str] = []
    if comparison.get("rows") != rows or comparison.get("columns") != columns:
        errors.append(f"{name} has unexpected dimensions")
    for key in ("schema_exact", "column_order_equal"):
        if comparison.get(key) is not True:
            errors.append(f"{name} does not have exact schema/column order")
    for key in (
        "dtype_mismatch_count",
        "missing_mask_mismatch_count",
        "differing_feature_count",
        "differing_cell_count",
    ):
        if comparison.get(key) != 0:
            errors.append(f"{name} has nonzero {key}")
    if comparison.get("max_abs_difference") != 0.0:
        errors.append(f"{name} has nonzero maximum absolute difference")
    return errors


def _feature_parity_evidence() -> tuple[dict[str, Any], list[str]]:
    errors: list[str] = []
    summary: dict[str, Any] = {
        "status": "INSPECTED",
        "report_path": str(FEATURE_PARITY_REPORT),
        "selector_manifest_path": str(SELECTOR_MANIFEST),
    }
    if not FEATURE_PARITY_REPORT.is_file():
        return {**summary, "status": "MISSING"}, ["Feature parity report is unavailable"]
    report_hash = sha256_file(FEATURE_PARITY_REPORT)
    summary["report_sha256"] = report_hash
    if report_hash != FEATURE_PARITY_REPORT_SHA256:
        errors.append("Feature parity report hash differs from the pinned evidence")
    with FEATURE_PARITY_REPORT.open("r", encoding="utf-8") as stream:
        report = json.load(stream)
    if report.get("exit_code") != 0:
        errors.append("Raw feature parity replay did not exit successfully")
    expected_raw_hashes = {
        "train": "b8dd5a61771d8eb2821972c1b7d3f2f747cc66471f4d34153036d2d9c458ecf1",
        "test": "dfb7aeb96e5af3d8616acaaba6b23cec8680845cfd525f7c08de15c6d2d42c65",
    }
    for name, expected_hash in expected_raw_hashes.items():
        if report.get("raw_inputs", {}).get(name, {}).get("sha256") != expected_hash:
            errors.append(f"Feature replay used an unexpected raw {name} file")
    source = report.get("historical_source_execution", {})
    if (
        source.get("matches_audited_source_hash") is not True
        or source.get("sha256")
        != "cafd70abd8da3cb6d544c26b668ed61f329cd89ef98ff61cec3983ffc1c08341"
    ):
        errors.append("Feature replay did not verify its historical source hash")
    comparisons = report.get("comparisons", {})
    for name, (rows, columns) in FEATURE_COMPARISONS.items():
        comparison = comparisons.get(name)
        if not isinstance(comparison, dict):
            errors.append(f"Missing feature parity comparison: {name}")
            continue
        errors.extend(_validate_feature_comparison(name, comparison, rows, columns))
    summary["comparisons_exact"] = not errors

    if not SELECTOR_MANIFEST.is_file():
        errors.append("Raw selector replay manifest is unavailable")
        summary["selector_manifest_status"] = "MISSING"
        return summary, errors
    selector_hash = sha256_file(SELECTOR_MANIFEST)
    summary["selector_manifest_sha256"] = selector_hash
    if selector_hash != SELECTOR_MANIFEST_SHA256:
        errors.append("Raw selector manifest hash differs from the pinned evidence")
    with SELECTOR_MANIFEST.open("r", encoding="utf-8") as stream:
        selector = json.load(stream)
    selector_ok = (
        selector.get("exit_code") == 0
        and selector.get("training_performed") is True
        and selector.get("selected_order_matches_historical") is True
        and selector.get("rebuilt_selected_feature_count") == 550
        and selector.get("selected_set_intersection") == 550
        and selector.get("selected_set_jaccard") == 1.0
        and selector.get("sources", {}).get("raw_target", {}).get("sha256")
        == expected_raw_hashes["train"]
    )
    summary["selector_order_exact"] = selector_ok
    summary["selector_manifest_status"] = "INSPECTED"
    if not selector_ok:
        errors.append("Raw selector replay does not verify exact historical selection order")
    return summary, errors


def _array_summary(name: str, array: np.ndarray) -> dict[str, Any]:
    item: dict[str, Any] = {
        "member": name,
        "shape": list(array.shape),
        "dtype": str(array.dtype),
        "element_count": int(array.size),
        "id_or_fold_name": any(token in name.lower() for token in ("id", "fold", "split")),
        "finite_count": None,
        "minimum": None,
        "maximum": None,
        "inspection_error": None,
    }
    if np.issubdtype(array.dtype, np.number):
        finite = np.isfinite(array)
        item["finite_count"] = int(finite.sum())
        if item["finite_count"]:
            item["minimum"] = float(np.min(array[finite]))
            item["maximum"] = float(np.max(array[finite]))
    return item


def inspect_arrays(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    try:
        loaded = np.load(path, allow_pickle=False)
        if isinstance(loaded, np.lib.npyio.NpzFile):
            with loaded as archive:
                for member in archive.files:
                    try:
                        rows.append(_array_summary(member, archive[member]))
                    except (ValueError, OSError) as exc:
                        rows.append({
                            "member": member,
                            "shape": None,
                            "dtype": None,
                            "element_count": None,
                            "finite_count": None,
                            "minimum": None,
                            "maximum": None,
                            "inspection_error": str(exc),
                        })
        else:
            rows.append(_array_summary(path.stem, loaded))
    except (ValueError, OSError, EOFError) as exc:
        rows.append({
            "member": path.name,
            "shape": None,
            "dtype": None,
            "element_count": None,
            "finite_count": None,
            "minimum": None,
            "maximum": None,
            "inspection_error": str(exc),
        })
    return rows


def write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _run_command(command: list[str], cwd: Path, commands: list[dict[str, Any]]) -> subprocess.CompletedProcess[str]:
    start = time.perf_counter()
    try:
        result = subprocess.run(
            command,
            cwd=str(cwd),
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            check=False,
        )
        commands.append({
            "command": command,
            "cwd": str(cwd),
            "exit_code": result.returncode,
            "elapsed_seconds": round(time.perf_counter() - start, 6),
            "stdout": result.stdout[-12000:],
            "stderr": result.stderr[-12000:],
        })
        return result
    except OSError as exc:
        commands.append({
            "command": command,
            "cwd": str(cwd),
            "exit_code": None,
            "elapsed_seconds": round(time.perf_counter() - start, 6),
            "stdout": "",
            "stderr": str(exc),
        })
        return subprocess.CompletedProcess(command, 127, "", str(exc))


def _artifact_inventory(graph: dict[str, Any], errors: list[str]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    artifact_rows: list[dict[str, Any]] = []
    array_rows: list[dict[str, Any]] = []
    for node in graph["nodes"]:
        raw_path = node.get("path")
        if not raw_path:
            continue
        path = Path(raw_path)
        exists = path.is_file()
        actual_hash = sha256_file(path) if exists else None
        expected_hash = node.get("sha256")
        hash_status = "NOT_PINNED"
        if expected_hash:
            hash_status = "MATCH" if actual_hash == expected_hash else ("MISSING" if not exists else "MISMATCH")
            if hash_status != "MATCH":
                errors.append(f"{node['id']}: expected {expected_hash}, found {actual_hash}")
        artifact_rows.append({
            "id": node["id"],
            "kind": node.get("kind"),
            "path": str(path),
            "exists": exists,
            "size_bytes": path.stat().st_size if exists else None,
            "expected_sha256": expected_hash,
            "actual_sha256": actual_hash,
            "hash_status": hash_status,
            "producer": node.get("producer"),
            "consumer": node.get("consumer"),
            "classification": node.get("classification"),
            "lineage": node.get("lineage"),
        })
        if exists and path.suffix.lower() in {".npy", ".npz"}:
            for item in inspect_arrays(path):
                array_rows.append({
                    "artifact_id": node["id"],
                    "path": str(path),
                    **item,
                })
    return artifact_rows, array_rows


def _source_inventory(errors: list[str]) -> list[dict[str, Any]]:
    rows = []
    for role, raw_path, expected_hash in SOURCE_FILES:
        path = Path(raw_path)
        exists = path.is_file()
        actual_hash = sha256_file(path) if exists else None
        hash_status = "NOT_PINNED"
        if expected_hash:
            hash_status = "MATCH" if actual_hash == expected_hash else ("MISSING" if not exists else "MISMATCH")
            if hash_status != "MATCH":
                errors.append(f"source {role}: expected {expected_hash}, found {actual_hash}")
        rows.append({
            "role": role,
            "path": str(path),
            "exists": exists,
            "size_bytes": path.stat().st_size if exists else None,
            "expected_sha256": expected_hash,
            "actual_sha256": actual_hash,
            "hash_status": hash_status,
        })
    return rows


def _validate_csvs() -> dict[str, Any]:
    summaries: dict[str, Any] = {}
    test_ids: list[str] | None = None
    train_labels: list[int] | None = None
    for name, path, id_column, target_column in CSV_CHECKS:
        if not path.is_file():
            summaries[name] = {"status": "MISSING", "path": str(path)}
            continue
        needed = {column for column in (id_column, target_column) if column}
        headers, row_count, columns = read_csv_columns(path, needed)
        details: dict[str, Any] = {
            "status": "INSPECTED",
            "path": str(path),
            "rows": row_count,
            "columns": headers,
        }
        if id_column in headers:
            ids = columns[id_column]
            details["unique_ids"] = len(ids) == len(set(ids))
            if name == "raw_test":
                test_ids = ids
            elif test_ids is not None and name in {
                "sample_submission", "historical_champion", "historical_v85",
                "historical_v20", "historical_v_gold", "historical_v080041",
                "historical_v080150", "historical_v080200",
                "historical_level2_stacker", "historical_asymmetric_apex",
                "historical_optimal_transductive_apex", "historical_precision_apex",
                "v5_candidate", "fresh_raw_candidate", "raw_bedrock_candidate",
            }:
                details["test_id_order"] = compare_id_order(test_ids, ids)
        if target_column and target_column in headers:
            values = np.asarray([float(value) for value in columns[target_column]], dtype=np.float64)
            details["target_finite"] = bool(np.isfinite(values).all())
            details["target_minimum"] = float(values.min()) if values.size else None
            details["target_maximum"] = float(values.max()) if values.size else None
            details["target_in_unit_interval"] = bool(
                values.size and np.isfinite(values).all() and (values >= 0).all() and (values <= 1).all()
            )
        summaries[name] = details

    raw_train = Path(r"D:\gci\inpuy\train.csv")
    if raw_train.is_file():
        headers, row_count, columns = read_csv_columns(raw_train, {"TARGET"})
        result: dict[str, Any] = {
            "status": "INSPECTED",
            "path": str(raw_train),
            "rows": row_count,
            "columns": headers,
        }
        if "TARGET" in headers:
            try:
                train_labels = [int(float(value)) for value in columns["TARGET"]]
                result["target_values"] = sorted(set(train_labels))
            except (TypeError, ValueError) as exc:
                result["target_error"] = str(exc)
        summaries["raw_train"] = result

    y_path = Path(r"D:\gci\champion\cache\y_train.npy")
    if y_path.is_file() and train_labels is not None:
        labels = np.load(y_path, allow_pickle=False)
        summaries["raw_label_array"] = {
            "path": str(y_path),
            "same_length_as_raw_train": len(labels) == len(train_labels),
            "exact_raw_label_order_match": bool(
                labels.shape == (len(train_labels),)
                and np.array_equal(labels, np.asarray(train_labels, dtype=labels.dtype))
            ),
        }
    return summaries


def _data_errors(data: dict[str, Any], array_rows: list[dict[str, Any]]) -> list[str]:
    errors: list[str] = []
    if data.get("raw_test", {}).get("rows") != 61500:
        errors.append("Raw test row count is not exactly 61,500")
    if data.get("raw_train", {}).get("rows") != 171202:
        errors.append("Raw train row count is not exactly 171,202")
    if not data.get("raw_test", {}).get("unique_ids", False):
        errors.append("Raw test IDs are missing or duplicated")
    required_submissions = (
        "sample_submission",
        "historical_champion",
        "historical_v85",
        "historical_v20",
        "historical_v_gold",
        "historical_v080041",
        "historical_v080150",
        "historical_v080200",
        "historical_level2_stacker",
        "historical_asymmetric_apex",
        "historical_optimal_transductive_apex",
        "historical_precision_apex",
        "v5_candidate",
        "fresh_raw_candidate",
        "raw_bedrock_candidate",
    )
    for name in required_submissions:
        item = data.get(name, {})
        if item.get("status") != "INSPECTED":
            errors.append(f"Required CSV validation input is unavailable: {name}")
            continue
        if not item.get("unique_ids", False):
            errors.append(f"{name} contains duplicate IDs")
        order = item.get("test_id_order", {})
        if not order.get("same_order", False):
            errors.append(f"{name} IDs do not exactly match raw test order")
        if "target_finite" in item and not item["target_finite"]:
            errors.append(f"{name} contains non-finite targets")
        if "target_in_unit_interval" in item and not item["target_in_unit_interval"]:
            errors.append(f"{name} targets are not all in [0, 1]")
    labels = data.get("raw_label_array", {})
    if labels and not labels.get("exact_raw_label_order_match", False):
        errors.append("Saved y_train.npy does not exactly match raw train TARGET order")

    train_rows = data.get("raw_train", {}).get("rows")
    test_rows = data.get("raw_test", {}).get("rows")
    if train_rows is not None and test_rows is not None:
        for item in array_rows:
            if item.get("inspection_error"):
                errors.append(f"{item.get('artifact_id')}:{item.get('member')} could not be inspected safely")
            if item.get("finite_count") is not None and item["finite_count"] != item.get("element_count"):
                errors.append(f"{item.get('artifact_id')}:{item.get('member')} contains non-finite numeric values")
            shape = item.get("shape")
            if not shape or len(shape) != 1:
                continue
            member = str(item.get("member", "")).lower()
            expected_rows = None
            if member.startswith("oof") or member in {"final_oof", "y_true", "y_train"}:
                expected_rows = train_rows
            elif member.startswith("test") or member == "final_test":
                expected_rows = test_rows
            if expected_rows is not None and shape[0] != expected_rows:
                errors.append(
                    f"{artifact_id}:{member} has {shape[0]} rows; expected {expected_rows}"
                )
    return errors


def _git_evidence(commands: list[dict[str, Any]]) -> dict[str, Any]:
    evidence: dict[str, Any] = {}
    for key, args in [
        ("head", ["rev-parse", "HEAD"]),
        ("shallow", ["rev-parse", "--is-shallow-repository"]),
        ("status", ["status", "--short", "--branch"]),
    ]:
        result = _run_command(["git", "-C", str(REPO_ROOT), *args], REPO_ROOT, commands)
        evidence[key] = {
            "exit_code": result.returncode,
            "stdout": result.stdout.strip(),
            "stderr": result.stderr.strip(),
        }
    for term in [
        "v_17_full_suite_bundle.npz",
        "v55_members.npz",
        "oof_10f_xgb.npy",
        "v_20_apex_grand_champion_submission.csv",
    ]:
        result = _run_command(
            ["git", "-C", str(REPO_ROOT), "log", "--all", "--oneline", "-S", term, "--", "champion"],
            REPO_ROOT,
            commands,
        )
        evidence[f"history_search:{term}"] = {
            "exit_code": result.returncode,
            "stdout": result.stdout.strip(),
            "stderr": result.stderr.strip(),
        }
    return evidence


def run_audit() -> tuple[int, dict[str, Any]]:
    for directory in OUTPUT_DIRS:
        directory.mkdir(parents=True, exist_ok=True)

    started_at = utc_now()
    started = time.perf_counter()
    graph = load_graph()
    errors: list[str] = []
    commands: list[dict[str, Any]] = []

    artifact_rows, array_rows = _artifact_inventory(graph, errors)
    source_rows = _source_inventory(errors)
    data_validation = _validate_csvs()
    feature_parity, feature_parity_errors = _feature_parity_evidence()
    data_validation["raw_feature_parity_evidence"] = feature_parity
    errors.extend(_data_errors(data_validation, array_rows))
    errors.extend(feature_parity_errors)

    write_csv(
        AUDIT_DIR / "manifests" / "artifact_inventory.csv",
        ["id", "kind", "path", "exists", "size_bytes", "expected_sha256", "actual_sha256",
         "hash_status", "producer", "consumer", "classification", "lineage"],
        artifact_rows,
    )
    write_csv(
        AUDIT_DIR / "manifests" / "source_inventory.csv",
        ["role", "path", "exists", "size_bytes", "expected_sha256", "actual_sha256", "hash_status"],
        source_rows,
    )
    write_csv(
        AUDIT_DIR / "manifests" / "array_schema.csv",
        ["artifact_id", "path", "member", "shape", "dtype", "element_count",
         "id_or_fold_name", "finite_count", "minimum", "maximum", "inspection_error"],
        [
            {**row, "shape": json.dumps(row["shape"], separators=(",", ":"))}
            for row in array_rows
        ],
    )
    with (AUDIT_DIR / "manifests" / "data_validation.json").open("w", encoding="utf-8") as stream:
        json.dump(data_validation, stream, indent=2, sort_keys=True)
        stream.write("\n")

    git_evidence = _git_evidence(commands)
    test_command = [
        sys.executable, "-m", "unittest", "discover",
        "-s", "tests", "-p", "test_phase0_audit.py", "-v",
    ]
    test_result = _run_command(test_command, AUDIT_DIR, commands)
    if test_result.returncode != 0:
        errors.append("Focused Phase 0 unit tests failed.")

    exit_code = 1 if errors else 0
    record = {
        "audit_id": "exp85_phase0_forensic_audit",
        "verdict": graph["verdict"],
        "started_at_utc": started_at,
        "completed_at_utc": utc_now(),
        "elapsed_seconds": round(time.perf_counter() - started, 6),
        "invocation": [sys.executable, str(Path(__file__).resolve()), *sys.argv[1:]],
        "script_exit_code": exit_code,
        "training_or_model_inference_performed": False,
        "historical_artifacts_modified": False,
        "environment": {
            "python": sys.version,
            "platform": platform.platform(),
            "numpy": np.__version__,
            "working_directory": str(Path.cwd()),
            "audit_directory": str(AUDIT_DIR),
            "repository_root": str(REPO_ROOT),
        },
        "inventory_counts": {
            "dependency_nodes": len(graph["nodes"]),
            "dependency_edges": len(graph["edges"]),
            "artifact_files": len(artifact_rows),
            "source_files": len(source_rows),
            "array_members": len(array_rows),
        },
        "git_evidence": git_evidence,
        "commands": commands,
        "verification_errors": errors,
        "data_validation_file": str(AUDIT_DIR / "manifests" / "data_validation.json"),
    }
    with (AUDIT_DIR / "logs" / "phase0_execution.json").open("w", encoding="utf-8") as stream:
        json.dump(record, stream, indent=2, sort_keys=True)
        stream.write("\n")
    return exit_code, record


if __name__ == "__main__":
    code, result = run_audit()
    print(json.dumps({
        "verdict": result["verdict"],
        "script_exit_code": code,
        "verification_errors": result["verification_errors"],
        "inventory_counts": result["inventory_counts"],
    }, indent=2))
    raise SystemExit(code)
