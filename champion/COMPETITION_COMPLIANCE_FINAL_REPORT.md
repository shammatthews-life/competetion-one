# GCI World 2026 Competition Compliance Final Report

## Formal Compliance Assessment
- **Competition**: In-class Competition: Home Credit Default Risk | GCI Global 2026
- **Hosting Platform**: Omnicampus
- **Evaluation Metric**: ROC-AUC
- **Final Verdict**: **CONFIRMED COMPLIANT (ELIGIBLE FOR TOP HONORS)**

### Core Rule Compliance Matrix
1. **Zero External Data**: Confirmed Compliant. Only `train.csv` and `test.csv` provided via Omnicampus were used.
2. **Zero Pretrained Weights**: Confirmed Compliant. GBDT models initialized with random seeds.
3. **Zero Test-Label Leakage**: Confirmed Compliant. No pseudo-labels or test labels accessed.
4. **Strict Submission Requirements**: Confirmed Compliant. CSV format, exactly 61,500 test rows, IDs 171202 to 232701, probabilities $\in [0, 1]$.
5. **Full Code Reproducibility**: Confirmed Compliant. The submitted CSVs are completely reproducible via [`exp85_recovery_v5.ipynb`](file:///D:/gci/champion/01_EXP85_RECOVERY/notebooks/exp85_recovery_v5.ipynb).
