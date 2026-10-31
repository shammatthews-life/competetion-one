# EXP85 End-to-End Reproduction Instructions

## Step 1: Environment Setup
Ensure Python 3.12 is installed with CUDA-enabled GPU (NVIDIA RTX 4050 or compatible):
```bash
pip install -r DEPENDENCIES.txt
```

## Step 2: Feature Generation & Screening
Run the feature generator and screening scripts on raw competition CSVs:
```bash
python 02_EXP85_REBUILT/source/run_hasan_600_builder.py
python 02_EXP85_REBUILT/source/run_workstream2_screening.py
```

## Step 3: Foundation Model Zoo Retraining
Train the 5 foundation models across 5 outer folds:
```bash
python 02_EXP85_REBUILT/source/train_independent_zoo.py
```

## Step 4: Transductive Calibration Pyramid
Execute non-linear calibration to reconstruct Anchor 74:
```bash
python 02_EXP85_REBUILT/source/reconstruct_transductive_pyramid.py
```

## Step 5: Master Notebook Execution
Run the master notebook to generate both final submissions:
```bash
jupyter nbconvert --to notebook --execute 01_EXP85_RECOVERY/notebooks/exp85_recovery_v5.ipynb
```
