# EXP85 Macro Synthesis Mathematical Proof

## Theoretical Formulation
Let $U_{74} \in [0, 1]$ denote the percentile rank of Anchor 74, and let $U_{\text{suite}} \in [0, 1]$ denote the composite percentile rank of the 8-Engine Suite.

The latent default propensity is mapped to Gaussian quantiles via the probit link function:
$$Z_{74} = \Phi^{-1}(\text{clip}(U_{74}, 10^{-6}, 1 - 10^{-6}))$$
$$Z_{\text{suite}} = \Phi^{-1}(\text{clip}(U_{\text{suite}}, 10^{-6}, 1 - 10^{-6}))$$

The optimal convex synthesis in probit space is defined by:
$$Z_{\text{macro}} = w_{74} \cdot Z_{74} + (1 - w_{74}) \cdot Z_{\text{suite}}$$
where empirical Nelder-Mead simplex optimization on local cross-validation yields $w_{74} = 0.850$.

Final submission probabilities are mapped via distinct ordinal ranking:
$$\text{TARGET} = \frac{\text{rank}_{\text{ordinal}}(Z_{\text{macro}}) - 0.5}{N}$$
This operation is strictly monotonic, preserving ROC-AUC ranking invariants while ensuring zero ties and uniform output distribution.
