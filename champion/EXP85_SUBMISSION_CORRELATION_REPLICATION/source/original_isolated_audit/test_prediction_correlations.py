#!/usr/bin/env python3
"""Synthetic smoke tests for the prediction correlation auditor.

This script intentionally uses only synthetic data. It does not touch the original project files.
"""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

SCRIPT = Path(__file__).with_name("audit_prediction_correlations.py")


def run_case(ref_df: pd.DataFrame | None, cand_df: pd.DataFrame | None, *, expect_valid: bool, expected_phrase: str | None = None, id_col: str = "SK_ID_CURR", pred_col: str = "pred") -> None:
    with tempfile.TemporaryDirectory() as td:
        td_path = Path(td)
        ref_path = td_path / "ref.csv"
        cand_path = td_path / "cand.csv"
        out_path = td_path / "result.json"

        if ref_df is not None:
            ref_df.to_csv(ref_path, index=False)
        if cand_df is not None:
            cand_df.to_csv(cand_path, index=False)

        cmd = [
            sys.executable,
            str(SCRIPT),
            "--reference",
            str(ref_path),
            "--candidate",
            str(cand_path),
            "--id-col",
            id_col,
            "--pred-col",
            pred_col,
            "--json-out",
            str(out_path),
        ]
        proc = subprocess.run(cmd, capture_output=True, text=True)

        if expect_valid:
            assert proc.returncode == 0, f"expected valid case but failed: {proc.stderr or proc.stdout}"
            payload = json.loads(out_path.read_text(encoding="utf-8"))
            assert payload["status"] == "VALID", payload
            assert payload["alignment"]["valid"] is True, payload
            assert payload["alignment"]["aligned_row_count"] > 0, payload
            assert payload["metrics"]["pearson"] is not None, payload
            assert payload["metrics"]["spearman"] is not None, payload
        else:
            assert proc.returncode != 0, f"expected invalid case but script succeeded: {proc.stdout}"
            payload = json.loads(out_path.read_text(encoding="utf-8")) if out_path.exists() else {}
            if expected_phrase is not None:
                message = payload.get("error", proc.stderr).lower()
                assert expected_phrase.lower() in message, (expected_phrase, message)


def main() -> None:
    run_case(
        pd.DataFrame({"SK_ID_CURR": [1, 2, 3], "pred": [0.1, 0.2, 0.3]}),
        pd.DataFrame({"SK_ID_CURR": [1, 2, 3], "pred": [0.1, 0.2, 0.3]}),
        expect_valid=True,
    )
    run_case(
        pd.DataFrame({"SK_ID_CURR": [1, 2, 3], "pred": [0.1, 0.2, 0.3]}),
        pd.DataFrame({"SK_ID_CURR": [3, 1, 2], "pred": [0.3, 0.1, 0.2]}),
        expect_valid=True,
    )
    run_case(
        pd.DataFrame({"SK_ID_CURR": [1, 1, 2], "pred": [0.1, 0.2, 0.3]}),
        pd.DataFrame({"SK_ID_CURR": [1, 2], "pred": [0.1, 0.3]}),
        expect_valid=False,
        expected_phrase="duplicate",
    )
    run_case(
        pd.DataFrame({"SK_ID_CURR": [1, 2], "pred": [0.1, 0.3]}),
        pd.DataFrame({"SK_ID_CURR": [1, 1, 2], "pred": [0.1, 0.2, 0.3]}),
        expect_valid=False,
        expected_phrase="duplicate",
    )
    run_case(
        pd.DataFrame({"SK_ID_CURR": [1, 2, 3], "pred": [0.1, 0.2, 0.3]}),
        pd.DataFrame({"SK_ID_CURR": [1, 3], "pred": [0.1, 0.3]}),
        expect_valid=False,
        expected_phrase="missing",
    )
    run_case(
        pd.DataFrame({"SK_ID_CURR": [1, 2, 3], "pred": [0.1, 0.2, 0.3]}),
        pd.DataFrame({"SK_ID_CURR": [1, 2, 4], "pred": [0.1, 0.2, 0.4]}),
        expect_valid=False,
        expected_phrase="extra",
    )
    run_case(
        pd.DataFrame({"SK_ID_CURR": [1, 2], "x": [0.1, 0.2]}),
        pd.DataFrame({"SK_ID_CURR": [1, 2], "pred": [0.1, 0.2]}),
        expect_valid=False,
        expected_phrase="missing prediction column",
    )
    run_case(
        pd.DataFrame({"SK_ID_CURR": [1, 2], "pred": [0.1, np.nan]}),
        pd.DataFrame({"SK_ID_CURR": [1, 2], "pred": [0.1, 0.2]}),
        expect_valid=False,
        expected_phrase="non-finite",
    )
    run_case(
        pd.DataFrame({"SK_ID_CURR": [1, 2], "pred": [0.1, np.inf]}),
        pd.DataFrame({"SK_ID_CURR": [1, 2], "pred": [0.1, 0.2]}),
        expect_valid=False,
        expected_phrase="non-finite",
    )
    run_case(
        pd.DataFrame({"SK_ID_CURR": [1, 2], "pred": ["a", "b"]}),
        pd.DataFrame({"SK_ID_CURR": [1, 2], "pred": [0.1, 0.2]}),
        expect_valid=False,
        expected_phrase="non-numeric",
    )
    run_case(
        pd.DataFrame({"SK_ID_CURR": [1, 2, 3, 4], "pred": [1.0, 2.0, 3.0, 4.0]}),
        pd.DataFrame({"SK_ID_CURR": [1, 2, 3, 4], "pred": [1.0, 3.0, 2.0, 5.0]}),
        expect_valid=True,
    )
    run_case(
        pd.DataFrame({"SK_ID_CURR": [1, 2, 3, 4], "pred": [1.0, 2.0, 3.0, 40.0]}),
        pd.DataFrame({"SK_ID_CURR": [1, 2, 3, 4], "pred": [1.0, 2.0, 40.0, 3.0]}),
        expect_valid=True,
    )
    run_case(
        pd.DataFrame({"SK_ID_CURR": [1, 2, 3], "pred": [0.5, 0.5, 0.5]}),
        pd.DataFrame({"SK_ID_CURR": [1, 2, 3], "pred": [0.5, 0.5, 0.5]}),
        expect_valid=False,
        expected_phrase="zero variance",
    )
    run_case(
        None,
        None,
        expect_valid=False,
        expected_phrase="not found",
    )
    run_case(
        pd.DataFrame({"SK_ID_CURR": [], "pred": []}),
        pd.DataFrame({"SK_ID_CURR": [], "pred": []}),
        expect_valid=False,
        expected_phrase="empty",
    )

    print("All synthetic prediction correlation tests passed.")


if __name__ == "__main__":
    main()
