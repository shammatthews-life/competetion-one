# EXP05: independent PyTorch tabular neural network

EXP05 tests one model family: a residual MLP with fold-local categorical
embeddings. It trains from the raw application CSVs using the existing Phase 2
rowwise and fold-local fee/product/target-encoding features. The feature
builder, preprocessing statistics, category maps, inner stopping split, and
neural weights are fit independently inside each fold.

CUDA is mandatory. The runner requires CUDA-enabled PyTorch, verifies RTX 4050
memory before work, caps this process's allocator at 76% of device memory,
records `nvidia-smi` telemetry, checks PyTorch's reported CUDA device, and
fails rather than falling back to CPU. The repository's shared Python
environment is not modified; this experiment uses its dedicated `.venv`.

Create the environment and install PyTorch 2.11 with CUDA 12.8:

```powershell
python -m venv --system-site-packages champion\EXP85_CHAMPION_IMPROVEMENT\experiments\exp05_tabular_neural\.venv
champion\EXP85_CHAMPION_IMPROVEMENT\experiments\exp05_tabular_neural\.venv\Scripts\python.exe -m pip install --index-url https://download.pytorch.org/whl/cu128 torch==2.11.0+cu128
```

Run from the repository root:

```powershell
champion\EXP85_CHAMPION_IMPROVEMENT\experiments\exp05_tabular_neural\.venv\Scripts\python.exe champion\EXP85_CHAMPION_IMPROVEMENT\experiments\exp05_tabular_neural\run_experiment.py
```

The five outer folds and OOF comparators are matched exactly to the saved
Phase 2 and EXP04 fold indices. For each outer fold, a stratified inner
validation split is drawn only from outer-fit rows. It selects the epoch
count; the final neural net is then refit on the complete outer-fit partition.
Outer validation labels do not influence stopping, preprocessing, feature
generation, or model fitting.

Each run saves OOF and test arrays, fold indices and fold predictions,
per-fold inner/final checkpoints, GPU telemetry, the feature list and weight
diagnostics, a candidate CSV, and input/source/output SHA-256 manifests.
The runner uses fixed per-fold seeds, PyTorch deterministic algorithms, and a
CUDA deterministic-GEMM workspace. This supports repeatable execution, but a
full second training run is not implied by a single run report.
Historical predictions are validation comparators only; the historical
champion is read after training only for test diagnostics. Neither is used as
a model feature or training target. No public leaderboard score can be
claimed from local OOF metrics.
