# Model and Pretrained Weights Audit

## Audit Item: Pretrained Models & Third-Party Weights
- **Requirement**: Models must be trained from scratch on competition data without importing prohibited pretrained neural weights.
- **Inspection**:
  * Foundation zoo comprises LightGBM, XGBoost, and CatBoost GBDT architectures initialized from random seeds.
  * No pretrained weights (e.g. HuggingFace, torchvision) were imported.
- **Verdict**: **CONFIRMED COMPLIANT**.
