import json
from pathlib import Path

py_path = Path(r"D:\gci\solo_champion_v28_certified.py")
nb_path = Path(r"D:\gci\solo_champion_v28_certified.ipynb")

src = py_path.read_text(encoding="utf-8")

# Split cleanly by major section markers into self-contained Jupyter cells
markers = [
    ("# ==============================================================================\n# 1. PURE ROW-LEVEL FEATURE ENGINEERING", "Cell 2: Pure Row-Level Feature Engineering"),
    ("# ==============================================================================\n# 2. STRICTLY TRAIN-ONLY EXPECTATION RESIDUALS", "Cell 3: Strictly Train-Only Expectation Residuals"),
    ("# ==============================================================================\n# 3. STRICTLY ALIGNED INNER-OOF 6-VIEW STACKING", "Cell 4: Strictly Aligned Inner-OOF 6-View Stacking"),
    ("# ==============================================================================\n# 4. STRICTLY IN-FOLD BAYESIAN TARGET ENCODINGS", "Cell 5: Strictly In-Fold Bayesian Target Encodings"),
    ("# ==============================================================================\n# 5. ORTHOGONAL NEURAL MODEL", "Cell 6: Orthogonal Neural Architecture & Nested Calibration Helpers"),
    ("# ==============================================================================\n# 7. MAIN EXECUTION PIPELINE", "Cell 7: Full Pipeline Execution, Nested Unbiased Validation & Certified CSV Export"),
]

chunks = []
pos = 0
for marker, title in markers:
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
            "# GCI World 2026: Solo Champion V28 Audit-Certified Clean-Room Pipeline\n",
            "\n",
            "**100% Rule-Compliant & Unbiased Validation Certified:**\n",
            "- **Zero External Data / Zero Historical CSVs / Zero Precomputed Caches** (`train.csv`, `test.csv`, `sample_submission.csv` only)\n",
            "- **Zero Target Propagation (`partner_enc` removed)** & **Zero Conditional Manual Test Weights**\n",
            "- **Zero Test-Set Transductive Pooling** (all residuals, medians, group stats, scalers, target encodings, and probit CDFs fitted strictly on training folds)\n",
            "- **Strictly Row-Aligned Inner-OOF Multi-View Stacking** (fixes all in-sample & concatenation alignment defects)\n",
            "- **Outer 5-Fold Nested Empirical-CDF Probit Weight Optimization** (unbiased evaluation strictly on untouched validation folds)\n"
        ]
    }
]

section_titles = [
    "## Step 1: Environment, Deterministic Seeds & Dependency Pin Verification",
    "## Step 2: Pure Row-Level Feature Engineering (Independent Train/Test)",
    "## Step 3: Strictly Train-Only Full-Covariate Expectation Residuals",
    "## Step 4: Strictly Row-Aligned Inner-OOF 6-View Stacking + Fold-Safe Group View",
    "## Step 5: Strictly In-Fold Bayesian Target Encodings (27 Tariff & Demographic Keys)",
    "## Step 6: Orthogonal Neural Architecture & Nested Empirical-CDF Probit Helpers",
    "## Step 7: Live Multi-Architecture Training, Nested Unbiased Validation & Submission Certification",
]

for idx_c, (a, b) in enumerate(chunks):
    cells.append({
        "cell_type": "markdown",
        "metadata": {},
        "source": [section_titles[idx_c] + "\n"]
    })
    code_block = src[a:b]
    if idx_c == len(chunks) - 1:
        # Set default sys.argv cleanly inside Jupyter so argparse never picks up ipykernel -f flags
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

