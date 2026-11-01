import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np


EXPERIMENT_DIR = Path(__file__).resolve().parents[1]
MODULE_PATH = EXPERIMENT_DIR / "source" / "rebuild_expanded_zoo.py"
SPEC = importlib.util.spec_from_file_location("rebuild_expanded_zoo", MODULE_PATH)
rebuild = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(rebuild)


class RebuildExpandedZooTests(unittest.TestCase):
    def test_pinned_original_sources_match(self):
        config = json.loads((EXPERIMENT_DIR / "config" / "expanded_zoo.json").read_text(encoding="utf-8"))
        hashes = rebuild.verify_sources(config)
        self.assertEqual(hashes, config["source_sha256"])

    def test_empty_runtime_cache_guard_rejects_prediction_reuse(self):
        with tempfile.TemporaryDirectory() as temporary:
            cache = Path(temporary)
            rebuild.assert_prediction_cache_clean(cache)
            (cache / "oof_xgb.npy").touch()
            with self.assertRaisesRegex(RuntimeError, "pre-existing prediction caches"):
                rebuild.assert_prediction_cache_clean(cache)

    def test_expanded_zoo_schema_and_raw_label_alignment(self):
        config = {
            "artifact": {
                "keys_in_order": list(rebuild.ARRAY_KEYS),
                "train_rows": 4,
                "test_rows": 3,
                "prediction_dtype": "float64",
                "label_dtype": "int64",
                "prediction_range": [0.0, 1.0],
            }
        }
        labels = np.asarray([0, 1, 0, 1], dtype=np.int64)
        arrays = {
            key: np.asarray([0.1, 0.2, 0.3, 0.4] if key.startswith("oof_") else [0.2, 0.5, 0.8], dtype=np.float64)
            for key in rebuild.ARRAY_KEYS
            if key != "y_true"
        }
        arrays["y_true"] = labels
        with tempfile.TemporaryDirectory() as temporary:
            archive = Path(temporary) / "zoo.npz"
            np.savez(archive, **arrays)
            details = rebuild.validate_zoo(archive, config, labels)
            self.assertEqual(details["arrays"]["oof_xgb"]["shape"], [4])
            self.assertEqual(details["arrays"]["y_true"]["dtype"], "int64")
            arrays["y_true"] = labels[::-1]
            np.savez(archive, **arrays)
            with self.assertRaisesRegex(ValueError, "does not exactly match raw train TARGET order"):
                rebuild.validate_zoo(archive, config, labels)

    def test_feature_interface_shape_fails_closed(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "features.feather"
            import pandas as pd

            pd.DataFrame({"first": [1.0, 2.0], "second": [3.0, 4.0]}).to_feather(path)
            details = rebuild.validate_feature_frame(path, 2, 2)
            self.assertEqual(details["shape"], [2, 2])
            with self.assertRaisesRegex(ValueError, "expected"):
                rebuild.validate_feature_frame(path, 2, 3)

    def test_feature_infinities_require_exact_historical_reference(self):
        import pandas as pd

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            generated = root / "generated.feather"
            reference = root / "reference.feather"
            frame = pd.DataFrame({"value": [1.0, np.inf], "other": [np.nan, 2.0]})
            frame.to_feather(generated)
            frame.to_feather(reference)
            details = rebuild.validate_feature_frame(generated, 2, 2, reference)
            self.assertEqual(details["infinity_count"], 1)
            self.assertTrue(details["infinity_reference_parity"]["exact_values"])
            with self.assertRaisesRegex(ValueError, "no historical reference"):
                rebuild.validate_feature_frame(generated, 2, 2)

            frame.loc[0, "value"] = 3.0
            frame.to_feather(reference)
            with self.assertRaisesRegex(ValueError, "infinities that do not exactly match"):
                rebuild.validate_feature_frame(generated, 2, 2, reference)

    def test_base_zoo_schema_includes_aligned_raw_labels(self):
        names = (
            "oof_lgb_cons",
            "test_lgb_cons",
            "oof_lgb_et",
            "test_lgb_et",
            "oof_xgb",
            "test_xgb",
            "y_true",
        )
        arrays = {
            name: np.asarray([0, 1, 0, 1], dtype=np.int64)
            if name == "y_true"
            else np.asarray([0.1, 0.2, 0.8] if name.startswith("test_") else [0.1, 0.2, 0.3, 0.8], dtype=np.float64)
            for name in names
        }
        with tempfile.TemporaryDirectory() as temporary:
            archive = Path(temporary) / "base.npz"
            np.savez(archive, **arrays)
            details = rebuild.validate_base_zoo(archive, train_rows=4, test_rows=3)
            self.assertEqual(details["arrays"]["y_true"]["dtype"], "int64")
            arrays["y_true"] = np.asarray([0, 1, 0], dtype=np.int64)
            np.savez(archive, **arrays)
            with self.assertRaisesRegex(ValueError, "got shape"):
                rebuild.validate_base_zoo(archive, train_rows=4, test_rows=3)


if __name__ == "__main__":
    unittest.main()
