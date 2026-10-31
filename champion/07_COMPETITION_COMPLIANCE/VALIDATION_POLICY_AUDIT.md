# Validation Policy Audit

## Audit Item: Leakage-Free Out-Of-Fold Validation
- **Requirement**: Cross-validation out-of-fold predictions must have zero data leakage from validation folds into training folds.
- **Inspection**:
  * Cross-validation scheme: Stratified 5-Fold (`StratifiedKFold(n_splits=5, shuffle=True, random_state=42)`).
  * Target Encodings: Computed strictly out-of-fold within train folds.
  * Foundation Models: Out-of-fold predictions populated strictly on validation indices `val_idx`.
- **Verdict**: **CONFIRMED COMPLIANT** (Zero target leakage; strict out-of-fold separation).
