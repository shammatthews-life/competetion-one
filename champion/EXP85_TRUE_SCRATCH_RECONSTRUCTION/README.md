# EXP85 Phase 0 Forensic Audit

This workspace contains a read-only inventory of the historical EXP85 pipeline
and its raw-data reconstruction gaps. It does not contain model-training code
or generated prediction artifacts.

## Result

**Phase 0 evidence checks pass; full raw reconstruction remains blocked.**
The prior feature hash discrepancy was resolved as Feather serialization
variation after exact cell-level parity checks. Raw feature/selector/zoo
training and bedrock-chain runs already exist in separate isolated workspaces.
The historical bedrock ancestry still depends on unrecovered V20/V080041
model inputs; see [RECONSTRUCTION_STATUS.md](RECONSTRUCTION_STATUS.md).

The audit keeps these results separate:

- The historical-cache V5 reconstruction reproduces its earlier candidate,
  with reported ID-aligned Pearson/Spearman correlations of approximately
  `0.99999758` against the frozen champion. It is not a raw-only rebuild.
- A prior fresh raw-component candidate is valid as a separate candidate, but
  it did not reproduce the historical champion.
- The independent Phase 2 model is unrelated to the historical EXP85 lineage
  and remains untouched.

## Reproduce the inventory checks

From this directory, run:

```powershell
python source\run_phase0_audit.py
```

The verifier hashes named evidence files, inspects prediction-array schemas
without loading pickle files, checks raw/test/intermediate submission ID
ordering, validates the pinned feature-parity and selector manifests, records
selected Git-history searches, and runs the focused unit tests. It does not
train, infer, tune, submit, or overwrite any historical artifact.

See [reports/PHASE0_FORENSIC_AUDIT.md](reports/PHASE0_FORENSIC_AUDIT.md),
[manifests/dependency_graph.json](manifests/dependency_graph.json), and the
generated inventories under `manifests/` and `logs/`, including
[producer_config_inventory.csv](manifests/producer_config_inventory.csv).

The run depends on the historical roots at `D:\gci\champion`,
`D:\gci\gemini`, and raw inputs at `D:\gci\inpuy`. Missing files and incomplete
provenance are recorded as audit findings, not silently replaced.
