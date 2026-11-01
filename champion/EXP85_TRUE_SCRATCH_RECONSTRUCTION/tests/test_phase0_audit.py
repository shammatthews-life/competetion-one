import csv
import tempfile
import unittest
from pathlib import Path

import numpy as np

from source.run_phase0_audit import (
    AUDIT_DIR,
    _validate_feature_comparison,
    compare_id_order,
    inspect_arrays,
    load_graph,
    sha256_file,
)


class Phase0AuditTests(unittest.TestCase):
    def test_hashes_file_contents(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "sample.txt"
            path.write_bytes(b"EXP85 audit")
            self.assertEqual(
                sha256_file(path),
                "a38fdd233b6f1ac3f74fc5a659853ad1b0767f1d22885109c106246d614e04c2",
            )

    def test_compares_exact_id_order_and_duplicates(self):
        self.assertTrue(compare_id_order(["1", "2"], ["1", "2"])["same_order"])
        self.assertFalse(compare_id_order(["1", "2"], ["2", "1"])["same_order"])
        self.assertFalse(compare_id_order(["1", "1"], ["1", "1"])["expected_unique"])

    def test_feature_parity_requires_exact_values_and_schema(self):
        exact = {
            "rows": 171202,
            "columns": 669,
            "schema_exact": True,
            "column_order_equal": True,
            "dtype_mismatch_count": 0,
            "missing_mask_mismatch_count": 0,
            "differing_feature_count": 0,
            "differing_cell_count": 0,
            "max_abs_difference": 0.0,
        }
        self.assertEqual(_validate_feature_comparison("train", exact, 171202, 669), [])
        altered = {**exact, "differing_cell_count": 1}
        self.assertIn(
            "train has nonzero differing_cell_count",
            _validate_feature_comparison("train", altered, 171202, 669),
        )

    def test_inspects_npy_and_npz_without_pickle(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            npy = root / "predictions.npy"
            np.save(npy, np.asarray([0.1, 0.9], dtype=np.float32), allow_pickle=False)
            npz = root / "bundle.npz"
            np.savez(npz, oof=np.asarray([0.2, 0.8]), test_ids=np.asarray([10, 11]))
            npy_rows = inspect_arrays(npy)
            npz_rows = inspect_arrays(npz)
            self.assertEqual(npy_rows[0]["shape"], [2])
            self.assertEqual(npy_rows[0]["dtype"], "float32")
            self.assertEqual(npy_rows[0]["finite_count"], 2)
            self.assertEqual({row["member"] for row in npz_rows}, {"oof", "test_ids"})

    def test_graph_edges_and_producer_inventory_are_well_formed(self):
        graph = load_graph()
        node_ids = {node["id"] for node in graph["nodes"]}
        self.assertEqual(len(node_ids), len(graph["nodes"]))
        for edge in graph["edges"]:
            self.assertIn(edge["from"], node_ids)
            self.assertIn(edge["to"], node_ids)

        config_path = AUDIT_DIR / "manifests" / "producer_config_inventory.csv"
        with config_path.open("r", encoding="utf-8", newline="") as stream:
            reader = csv.DictReader(stream)
            self.assertIn("component", reader.fieldnames or [])
            rows = list(reader)
        self.assertGreaterEqual(len(rows), 8)
        self.assertTrue(all(row["component"] and row["configuration_status"] for row in rows))


if __name__ == "__main__":
    unittest.main()
