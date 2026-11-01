# Source Tools

- `correlation_gap_diagnostics.py` and `gap_diagnostics.py`: ID-aligned, test-covariate-only prediction gap report; no model training or tuning.
- `build_manifests.py` / `verify_manifests.py`: build and verify the SHA-256 manifest for all non-cache project files.

The diagnostic defaults to the documented raw-data and frozen-release paths and accepts explicit input-path overrides. It must fail closed on hash, schema, ID, or output-directory conflicts. Run it from any directory with `python source\correlation_gap_diagnostics.py --run-id <unique-name>`. Build and verify the package manifest with `python source\build_manifests.py` and `python source\verify_manifests.py`.
