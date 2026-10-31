from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm, rankdata


WEIGHTS = {
    "v17_e1Aet": 0.2932,
    "xgb10": 0.2596,
    "v17_final": 0.2051,
    "v55_cbprod": 0.0874,
    "v17_e1et": 0.0489,
    "v55_deep_et": 0.0442,
    "v55_allfam": 0.0383,
    "v55_v2_mono": 0.0233,
}
N_TEST = 61_500


def rank_uniform(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64).ravel()
    if not np.isfinite(values).all():
        raise ValueError("Prediction input contains NaN or infinity.")
    if values.size == 0:
        raise ValueError("Prediction input is empty.")
    return (rankdata(values, method="average") - 0.5) / values.size


def probit(values: np.ndarray) -> np.ndarray:
    return norm.ppf(np.clip(values, 1e-6, 1.0 - 1e-6))


def synthesize(
    raw_test_ids: np.ndarray,
    anchor_path: Path,
    v17_path: Path,
    v55_path: Path,
    xgb_test_path: Path,
) -> pd.DataFrame:
    ids = np.asarray(raw_test_ids)
    if ids.ndim != 1 or len(ids) != N_TEST or len(np.unique(ids)) != N_TEST:
        raise ValueError("Raw test IDs must be a unique 61,500-element vector.")

    anchor = pd.read_csv(anchor_path)
    if list(anchor.columns) != ["SK_ID_CURR", "TARGET"]:
        raise ValueError("Anchor74 must have exactly SK_ID_CURR,TARGET columns.")
    if not np.array_equal(anchor["SK_ID_CURR"].to_numpy(), ids):
        raise ValueError("Anchor74 IDs do not exactly match raw test order.")
    anchor_values = anchor["TARGET"].to_numpy(dtype=np.float64)
    if not np.isfinite(anchor_values).all() or np.any((anchor_values < 0) | (anchor_values > 1)):
        raise ValueError("Anchor74 targets must be finite and in [0, 1].")

    with np.load(v17_path, allow_pickle=False) as archive:
        v17 = {key: archive[key] for key in archive.files}
    with np.load(v55_path, allow_pickle=False) as archive:
        v55 = {key: archive[key] for key in archive.files}
    xgb_test = np.load(xgb_test_path, allow_pickle=False)

    member_predictions = {
        "v17_e1Aet": v17["test_e1Aet"],
        "xgb10": xgb_test,
        "v17_final": v17["final_test"],
        "v55_cbprod": v55["test_cb_product"],
        "v17_e1et": v17["test_e1et"],
        "v55_deep_et": v55["test_deep_et"],
        "v55_allfam": v55["test_v2_allfam_oh"],
        "v55_v2_mono": v55["test_v2_mono"],
    }

    weights_total = sum(WEIGHTS.values())
    normalized_weights = {key: value / weights_total for key, value in WEIGHTS.items()}
    suite_z = np.zeros(N_TEST, dtype=np.float64)
    for name, weight in normalized_weights.items():
        values = np.asarray(member_predictions[name], dtype=np.float64).ravel()
        if values.shape != (N_TEST,):
            raise ValueError(f"{name} has shape {values.shape}, expected ({N_TEST},).")
        suite_z += weight * probit(rank_uniform(values))

    macro_z = 0.85 * probit(rank_uniform(anchor_values)) + 0.15 * suite_z
    ordinal_ranks = rankdata(macro_z, method="ordinal")
    target = (ordinal_ranks - 0.5) / N_TEST
    if len(np.unique(target)) != N_TEST:
        raise AssertionError("Final rank projection is not strictly unique.")
    return pd.DataFrame({"SK_ID_CURR": ids, "TARGET": target})


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build a new raw-component EXP85 candidate without touching existing submissions."
    )
    parser.add_argument("--raw-dir", type=Path, required=True)
    parser.add_argument("--components-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    if args.output.exists():
        raise FileExistsError(f"Refusing to overwrite candidate: {args.output}")
    raw_ids = pd.read_csv(args.raw_dir / "test.csv", usecols=["SK_ID_CURR"])[
        "SK_ID_CURR"
    ].to_numpy()
    frame = synthesize(
        raw_ids,
        args.components_dir / "v_20261027_precision_transductive_apex.csv",
        args.components_dir / "v_17_full_suite_bundle.npz",
        args.components_dir / "v55_members.npz",
        args.components_dir / "test_10f_xgb.npy",
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(args.output, index=False, float_format="%.10f")
    print(f"Candidate written: {args.output}")
    print(f"Rows: {len(frame)}; unique IDs: {frame['SK_ID_CURR'].nunique()}")


if __name__ == "__main__":
    main()
