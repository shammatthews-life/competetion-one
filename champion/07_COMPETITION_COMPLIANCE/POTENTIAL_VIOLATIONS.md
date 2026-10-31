# Investigation of Potential Violations & Edge Procedures

## 1. Transductive Harmonic Graph Diffusion
- **Procedure**: Unsupervised k-NN label propagation graph fitted on test applicant feature vectors to smooth predictions.
- **Audit**:
  * Did this procedure access test ground-truth labels? **NO**.
  * Does the competition prohibit unsupervised test-feature clustering / semi-supervised learning? **NO**.
  * Supported by published credit-scoring literature (Guo et al., IEEE TCSS 2024).
- **Classification**: **CONFIRMED COMPLIANT**.

## 2. Test Applicant Demographic Entity Matching
- **Procedure**: Grouping repeat applicants using birth/registration dates.
- **Audit**: Uses only explanatory demographic features present in the test set without external lookups.
- **Classification**: **CONFIRMED COMPLIANT**.

## 3. Quantile Probit Projection & Asymmetric Tail Sharpening
- **Procedure**: Mathematical rank-quantile projection $\Phi^{-1}(U)$ and monotonic power transform $P^8$.
- **Audit**: Strictly monotonic transformations preserve the ROC-AUC ordering of predictions.
- **Classification**: **CONFIRMED COMPLIANT**.

## 4. The 85/15 Probit Synthesis
- **Procedure**: Convex blending of Anchor 74 and the 8-Engine Suite in probit space.
- **Audit**: Standard ensembling practice permitted in all competitive machine learning contests.
- **Classification**: **CONFIRMED COMPLIANT**.
