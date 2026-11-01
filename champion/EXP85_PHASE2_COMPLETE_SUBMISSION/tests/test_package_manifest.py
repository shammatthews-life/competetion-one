from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from package_manifest import create_manifest, verify_manifest


class PackageManifestTests(unittest.TestCase):
    def test_create_and_verify_package_manifest(self):
        with tempfile.TemporaryDirectory() as directory:
            package = Path(directory)
            (package / "source").mkdir()
            (package / "source" / "module.py").write_text("value = 1\n", encoding="utf-8")
            create_manifest(package)
            self.assertEqual(verify_manifest(package)["status"], "PASS")
            (package / "source" / "module.py").write_text("value = 2\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "mismatched"):
                verify_manifest(package)


if __name__ == "__main__":
    unittest.main()
