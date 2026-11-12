import json
from pathlib import Path

py_path = Path(r"D:\gci\solo_champion_v28_certified.py")
nb_path = Path(r"D:\gci\solo_champion_v28_certified.ipynb")

src = py_path.read_text(encoding="utf-8")

markers = [
    ("# ==============================================================================\n# 1. PURE ROW-LEVEL FEATURE ENGINEERING", "Step 2"),
    ("# ==============================================================================\n# 2. OUTER-FOLD LOCAL COVARIATE RESIDUALS", "Step 3"),
    ("# ==============================================================================\n# 3. FOLD-LOCAL 6-VIEW INNER-OOF STACKING", "Step 4"),
    ("# ==============================================================================\n# 4. FOLD-LOCAL 27 BAYESIAN TARGET ENCODINGS", "Step 5"),
    ("# ==============================================================================\n# 5. ORTHOGONAL GPU NEURAL ARCHITECTURES", "Step 6"),
    ("# ==============================================================================\n# 6. MAIN TWO-LEVEL OUTER-NESTED CLEAN-ROOM EXECUTION PIPELINE", "Step 7"),
]

chunks = []
pos = 0
for marker, _ in markers:
    idx = src.find(marker, pos)
    assert idx != -1, f"Marker not found: {marker}"
    chunks.append((pos, idx))
    pos = idx
chunks.append((pos, len(src)))

cells = [
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "# GCI World 2026: Solo Champion V28 Clean-Room GPU Pipeline (`SEED_EVAL=2027` Two-Level Nested CV)\n",
            "\n",
            "**Clean-Room Two-Level Outer-Nested Validation:**\n",
            "- **Zero External Data / Zero Historical CSVs / Zero Precomputed Caches** (`train.csv`, `test.csv`, `sample_submission.csv` only, with live SHA-256 provenance printing)\n",
            "- **Zero Target Propagation (`partner_enc` omitted)** & **Zero Manual Test-Subset Weights**\n",
            "- **Strictly Fold-Local Preprocessing & Predeclared Meta-Learner (Inner & Outer):** Category vocabularies (including inner-fold TE vocabularies), medians, 13 covariate residual regressors, aligned 6-view inner-OOF stackers, 29 fold-fit group/frequency features, 27 Bayesian TE keys (`prior_inner = y_fit[ia].mean()`), scalers, empirical probit CDFs, and a predeclared L2-regularized (`C=0.05`) `LogisticRegression` meta-learner with post-hoc nonnegative unit-simplex normalized coefficients are ALL fitted strictly inside each active training fold (`df_ia` for inner folds, `df_a` for outer folds), with zero held-out `b` or `test` leakage.\n",
            "- **Honest Evaluation Scope (`SEED_EVAL=2027`):** Evaluated across all 171,202 competition training rows via 5-fold outer-nested CV following prior development on the same competition training data, with GPU-accelerated `TabResMLP_NaNMask_GPU`, `TabPeriodicSwiGLU_GPU`, `CB_GPU_D7_Product` (identical 850 iterations across inner/outer), and `XGB_CUDA_B512`.\n",
            "- **Pinned Reproducibility & Best-Effort Determinism:** Exact library versions pinned in `requirements.txt` (`torch==2.14.1+cu130`), best-effort deterministic CUDA flags (`CUBLAS_WORKSPACE_CONFIG=:4096:8`, `cudnn.deterministic=True`, `torch.use_deterministic_algorithms(True, warn_only=True)`), and unique-ID assertions on `test`/`sample_sub`/`sub`.\n"
        ]
    }
]

section_titles = [
    "## Step 1: Environment, Deterministic Seeds & Dependency Verification",
    "## Step 2: Pure Row-Level Feature Engineering (Fold-Local Vocabularies)",
    "## Step 3: Outer-Fold Local Full-Covariate Expectation Residuals",
    "## Step 4: Fold-Local Aligned 6-View Inner-OOF Stacking & Group View",
    "## Step 5: Fold-Local 27 Bayesian Target Encodings",
    "## Step 6: Orthogonal GPU Neural Architectures (`TabResMLP` + `TabPeriodicSwiGLU`) & 10-Model Trainer",
    "## Step 7: True Two-Level Outer-Nested Execution (`SEED_EVAL=2027`) & Certified Submission Export",
]

for idx_c, (a, b) in enumerate(chunks):
    cells.append({
        "cell_type": "markdown",
        "metadata": {},
        "source": [section_titles[idx_c] + "\n"]
    })
    code_block = src[a:b]
    if idx_c == 0:
        code_block = code_block.rstrip() + (
            "\nprint(f'Env ready: numpy={np.__version__}, pandas={pd.__version__}, '"
            "f'lgb={lgb.__version__}, xgb={xgb.__version__}, torch={torch.__version__}')\n"
        )
    elif idx_c < len(chunks) - 1:
        fn_name = section_titles[idx_c].split(":")[1].strip()
        code_block = code_block.rstrip() + f"\nprint('[+] Defined stage: {fn_name}')\n"
    else:
        code_block = code_block.replace(
            'if __name__ == "__main__":\n    main()',
            'if __name__ == "__main__":\n    sys.argv = ["solo_champion_v28_certified.py"]\n    main()'
        )
    cells.append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [line + "\n" for line in code_block.splitlines()]
    })

nb = {
    "cells": cells,
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.12"}
    },
    "nbformat": 4,
    "nbformat_minor": 5
}

nb_path.write_text(json.dumps(nb, indent=1, ensure_ascii=False), encoding="utf-8")
print(f"[+] Built standalone clean-room notebook: {nb_path} ({len(cells)} cells)")
