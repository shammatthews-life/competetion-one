# EXP03: Rebuild the historical five-family zoo dependency

The audit did **not** find a missing direct EXP85 input. The seven artifacts
read by the final EXP85 script are present. It did find that the historical
Anchor74 branch consumes `cache/hasan_expanded_zoo.npz`, which is present but
has no run manifest proving which of two candidate producers created it.

This experiment regenerates that exact NPZ interface from the original raw
train/test data by running the preserved historical source snapshots for:

1. the 669-column Hasan feature builder;
2. five-fold LightGBM feature-gain selection of 550 columns;
3. the original five-fold LightGBM/ExtraTrees/XGBoost model zoo; and
4. the original CatBoost/deep-LightGBM expansion.

The snapshot hashes are pinned in `config/expanded_zoo.json`. All scripts run
in a fresh, isolated runtime cache. Historical prediction arrays are loaded
only after generation, for parity comparison; they are not training inputs.
No historical cache is copied into the training runtime.

Run:

```powershell
python experiments\exp03_missing_dependency_rebuild\source\rebuild_expanded_zoo.py
```

If all training stages finish but a post-run validation gate needs correction,
preserve the original run and resume validation/integration without retraining:

```powershell
python experiments\exp03_missing_dependency_rebuild\source\rebuild_expanded_zoo.py `
  --continue-run experiments\exp03_missing_dependency_rebuild\runs\<run-id>
```

Check fixed-seed CatBoost repeatability and exercise downstream integration
from that second raw rebuild in separate isolated outputs:

```powershell
python experiments\exp03_missing_dependency_rebuild\source\rebuild_expanded_zoo.py `
  --repeat-cuda-run experiments\exp03_missing_dependency_rebuild\runs\<run-id>
python experiments\exp03_missing_dependency_rebuild\source\rebuild_expanded_zoo.py `
  --integration-repeat experiments\exp03_missing_dependency_rebuild\runs\<run-id>
```

The repeat output is compared with the historical arrays only after training.
It is not selected or blended to optimize reference parity; the report
preserves each run and classifies downstream use of historical anchors and
suite bundles as hybrid.

By default the runner also re-executes the original downstream chain in an
isolated working directory, rewiring output paths only. The chain starts from
the available historical submission and auxiliary anchor references, so it
is a **hybrid reconstruction**, not a complete raw-data rebuild. Use
`--zoo-only` to stop after the artifact validation.

The generated archive contains `y_true`, a copy of the competition training
labels required by the original consumer. It is kept in the ignored runtime
directory and must not be committed. The JSON report, logs, feature parity
checks, model parity metrics, downstream output comparisons, and GPU
telemetry are kept separately.

The original feature code constructs supervised UIDY features using an inner
five-fold split, then the original zoo trainer uses a separate five-fold
split. The reconstruction preserves this historical behavior. It does not
certify the resulting OOF arrays as nested/leakage-safe.

The historical feature builder intentionally emits a small number of
infinities in `HOUSING_STABILITY`. They are accepted only when the generated
feature matrix is bit-exact to the corresponding historical feature reference,
including the positions of infinities and NaNs; they are not silently cleaned
or altered.
