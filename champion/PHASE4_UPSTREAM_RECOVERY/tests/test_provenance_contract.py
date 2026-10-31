import sys
import unittest
from pathlib import Path

SOURCE_DIR = (
    Path(__file__).resolve().parents[2]
    / "02_EXP85_REBUILT"
    / "source"
)
sys.path.insert(0, str(SOURCE_DIR))

from provenance_contract import (  # noqa: E402
    EXPECTED_ARTIFACTS,
    EXPECTED_GENERATORS,
    determine_provenance_status,
    require_raw_rebuild_manifest,
)


def valid_manifest():
    scripts = sorted(EXPECTED_GENERATORS)
    return {
        "provenance_status": "RAW_REBUILD_CERTIFIED",
        "execution_mode": "full_rebuild",
        "verification_summary": {
            "provenance_status": "RAW_REBUILD_CERTIFIED",
        },
        "execution_telemetry": [
            {
                "script": name,
                "exit_code": 0,
                "command": ["python", name],
            }
            for name in scripts
        ],
        "generator_source_hashes": {name: "a" * 64 for name in scripts},
        "raw_inputs": {
            "train_csv": {"sha256": "b" * 64},
            "test_csv": {"sha256": "c" * 64},
        },
        "artifacts": {name: {"sha256": "d" * 64} for name in EXPECTED_ARTIFACTS},
    }


class ProvenanceContractTests(unittest.TestCase):
    def test_fast_execution_is_smoke_only(self):
        records = [
            {"script": name, "exit_code": 0}
            for name in EXPECTED_GENERATORS
        ]
        self.assertEqual(
            determine_provenance_status(
                verify_only=False, fast=True, execution_records=records
            ),
            "SMOKE_TEST_ONLY",
        )

    def test_verify_only_is_not_a_raw_rebuild(self):
        self.assertEqual(
            determine_provenance_status(
                verify_only=True, fast=False, execution_records=[]
            ),
            "ARTIFACTS_VERIFIED",
        )

    def test_missing_or_failed_generator_is_incomplete(self):
        records = [
            {"script": name, "exit_code": 0}
            for name in list(EXPECTED_GENERATORS)[:-1]
        ]
        self.assertEqual(
            determine_provenance_status(
                verify_only=False, fast=False, execution_records=records
            ),
            "INCOMPLETE",
        )

    def test_full_manifest_requires_all_execution_and_hash_evidence(self):
        require_raw_rebuild_manifest(valid_manifest())

    def test_artifact_verification_manifest_cannot_authorize_synthesis(self):
        manifest = valid_manifest()
        manifest["provenance_status"] = "ARTIFACTS_VERIFIED"
        manifest["verification_summary"]["provenance_status"] = "ARTIFACTS_VERIFIED"
        with self.assertRaises(ValueError):
            require_raw_rebuild_manifest(manifest)

    def test_fast_execution_cannot_authorize_synthesis(self):
        manifest = valid_manifest()
        manifest["execution_telemetry"][0]["command"].append("--fast")
        with self.assertRaises(ValueError):
            require_raw_rebuild_manifest(manifest)


if __name__ == "__main__":
    unittest.main()
