from __future__ import annotations

import hashlib
import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from verify_pipeline import verify_run


class ManifestTests(unittest.TestCase):
    def test_detects_modified_run_output_hash(self):
        with tempfile.TemporaryDirectory() as directory:
            run = Path(directory) / "run"
            (run / "reports").mkdir(parents=True)
            (run / "submissions").mkdir()
            (run / "predictions").mkdir()
            (run / "submissions" / "submission_phase2_complete.csv").write_text(
                "SK_ID_CURR,TARGET\n1,0.5\n", encoding="utf-8"
            )
            report = {
                "reproduction_validation": {"status": "PASS"},
                "submission": {"promoted_path": None},
            }
            (run / "reports" / "implementation_report.json").write_text(
                json.dumps(report), encoding="utf-8"
            )
            (run / "run_manifest.json").write_text(
                json.dumps({"status": "PASS", "run_id": "test"}), encoding="utf-8"
            )
            files = {}
            for path in run.rglob("*"):
                if path.is_file():
                    files[path.relative_to(run).as_posix()] = {
                        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                        "size_bytes": path.stat().st_size,
                    }
            (run / "sha256_manifest.json").write_text(
                json.dumps({"files": files}), encoding="utf-8"
            )
            input_dir = Path(directory) / "data"
            input_dir.mkdir()
            pd = __import__("pandas")
            pd.DataFrame({"SK_ID_CURR": [1], "feature": [0]}).to_csv(input_dir / "test.csv", index=False)
            pd.DataFrame({"SK_ID_CURR": [1], "TARGET": [0]}).to_csv(input_dir / "sample_submission.csv", index=False)
            (run / "submissions" / "submission_phase2_complete.csv").write_text(
                "SK_ID_CURR,TARGET\n1,0.6\n", encoding="utf-8"
            )
            with self.assertRaisesRegex(ValueError, "hash mismatch"):
                verify_run(run, input_dir)

    def test_requires_every_oof_and_test_prediction_file(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            run = root / "run"
            for name in ("reports", "submissions", "predictions"):
                (run / name).mkdir(parents=True, exist_ok=True)
            data = root / "data"
            data.mkdir()
            pd.DataFrame({"SK_ID_CURR": [1, 2], "feature": [0, 1], "TARGET": [0, 1]}).to_csv(
                data / "train.csv", index=False
            )
            pd.DataFrame({"SK_ID_CURR": [3], "feature": [2]}).to_csv(
                data / "test.csv", index=False
            )
            pd.DataFrame({"SK_ID_CURR": [3], "TARGET": [0.5]}).to_csv(
                data / "sample_submission.csv", index=False
            )
            pd.DataFrame({"SK_ID_CURR": [3], "TARGET": [0.5]}).to_csv(
                run / "submissions" / "submission_phase2_complete.csv", index=False
            )
            (run / "reports" / "implementation_report.json").write_text(
                json.dumps(
                    {
                        "reproduction_validation": {"status": "PASS"},
                        "submission": {"promoted_path": None},
                    }
                ),
                encoding="utf-8",
            )
            (run / "run_manifest.json").write_text(
                json.dumps({"status": "PASS", "run_id": "test"}), encoding="utf-8"
            )
            for model in ("lightgbm", "catboost", "phase2_blend"):
                np.save(run / "predictions" / f"test_{model}.npy", np.array([0.5]))
                if model != "lightgbm":
                    np.save(
                        run / "predictions" / f"oof_{model}.npy",
                        np.array([0.2, 0.8]),
                    )
            files = {}
            for path in run.rglob("*"):
                if path.is_file():
                    files[path.relative_to(run).as_posix()] = {
                        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                        "size_bytes": path.stat().st_size,
                    }
            (run / "sha256_manifest.json").write_text(
                json.dumps({"files": files}), encoding="utf-8"
            )
            with self.assertRaisesRegex(FileNotFoundError, "Required oof prediction"):
                verify_run(run, data)


if __name__ == "__main__":
    unittest.main()
