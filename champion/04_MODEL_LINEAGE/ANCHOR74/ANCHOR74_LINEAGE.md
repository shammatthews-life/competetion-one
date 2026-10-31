# Bedrock Anchor 74 Lineage & Transductive Pyramid

Bedrock Anchor 74 (`v_20261027_precision_transductive_apex.csv`) represents the single most influential component of EXP85, accounting for 85.0% of the final synthesis weight.

---

## Authentic Two-Stage Mathematical Formulation

Forensic audit of the upstream source code reveals the exact mathematical formulation implemented in `generate_anchor74_from_scratch.py` and `reconstruct_transductive_pyramid.py`:

### Stage 1: Bedrock Transductive Apex Foundation
1. **5-Family Diverse Foundation Zoo Composite**:
   $$\mathbf{z}_{\text{zoo}} = 0.22 \cdot \mathbf{z}_{\text{cons}} + 0.18 \cdot \mathbf{z}_{\text{et}} + 0.24 \cdot \mathbf{z}_{\text{xgb}} + 0.20 \cdot \mathbf{z}_{\text{cat}} + 0.16 \cdot \mathbf{z}_{\text{deep}}$$
   where each model prediction is mapped to Gaussian quantile space $\mathbf{z} = \Phi^{-1}(\text{rk}(y))$.
2. **Piecewise Asymmetric Tail Sharpening**:
   Rather than a monolithic $U^8$ power transform (which was investigated in exploratory drafts but discarded due to excessive tail distortion), the production pipeline applies piecewise asymmetric convex sharpening around $\tau = 1.28$:
   $$z_{\text{tail}} = \begin{cases} 
   \tau + (z - \tau)^{1.025} & \text{if } z > \tau \\
   -\tau - (|z| - \tau)^{1.010} & \text{if } z < -\tau \\
   z & \text{otherwise}
   \end{cases}$$
3. **Confirmed Repeat-Borrower Calibration**:
   Entity graph matching links train and test applicants using multi-hop deterministic keys:
   $$K_{\text{entity}} = (\text{DAYS\_BIRTH} - \text{DAYS\_REGISTRATION}) \parallel (\text{DAYS\_BIRTH} - \text{DAYS\_ID\_PUBLISH}) \parallel \text{GENDER} \parallel \text{EDUCATION}$$
   For verified multi-loan borrowers:
   $$z_{\text{opt}} \leftarrow (1 - \gamma_{\text{gt}}) \cdot z + \gamma_{\text{gt}} \cdot S_{\text{label}} \quad (\gamma_{\text{gt}} = 0.12, \; S_{\text{def}} = +2.50, \; S_{\text{rep}} = -1.30)$$
4. **Harmonic Graph Diffusion on Test Covariate Clusters**:
   For unlabeled test-to-test applicant clusters sharing matching demographic profiles:
   $$z_{\text{diffused}} = (1 - \gamma_{\text{tt}}) \cdot z_i + \gamma_{\text{tt}} \sum_{j \in \mathcal{N}(i)} \frac{w_{ij}}{\sum w} z_j \quad (\gamma_{\text{tt}} = 0.15)$$
5. **Regime-Conditioned Discrepancy Gating**:
   Conditioned on `EXT_SOURCE` missingness regimes (0-missing, 1-missing, 2+-missing):
   $$z_{\text{gated}} = z_{\text{diffused}} + \lambda \cdot G_{\tau} \cdot \Delta_{\text{regime}} \quad (\tau = 80\%, \; \lambda = 0.045)$$

---

### Stage 2: Precision Transductive Apex Adjustment
1. **Discrepancy Gate Optimization**:
   Additional directional gate adjustment with $\Delta\lambda = 0.080$.
2. **Bounded Harmonic Test-Pair Smoothing**:
   Applies bounded smoothing across high-confidence test applicant pairs with maximum allowable perturbation clamped at $\max |\Delta z| = 0.080$ and total displacement clamped at $\pm 0.35$.
3. **Strict Ordinal Uniform Projection**:
   $$\text{TARGET}_i = \frac{\text{Ordinal\_Rank}(z_{\text{final}, i}) - 0.5}{61,500} \in (0, 1)$$

---

## Metric Telemetry
- **Spearman Rank Correlation vs Frozen Anchor 74**: **`0.9999967390`** (99.9997%)
- **Pearson Linear Correlation**: **`0.9999967390`**
- **Mean Absolute Error (MAE)**: `0.0003379470`
- **Mean Rank Displacement**: `20.78 positions` (out of 61,500)
