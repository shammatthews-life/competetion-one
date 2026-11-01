# EXP85 Champion Improvement Track

This workspace preserves the historical EXP85 implementation as a
cache-assisted baseline and evaluates new ideas in isolated experiments.
Nothing here modifies the historical champion, the original EXP85 source, the
Phase 2 release, or experiments in other workspaces.

The target requested by the user is an **official public ROC-AUC of 0.80314
or higher**. No local OOF score, replay, prediction correlation, or embedded
historical prediction can establish that target.

## Current verified state

- The isolated historical synthesis replay passed byte-for-byte against the
  frozen 61,500-row reference (SHA-256
  `78c31bb4e4534330479faa26861a49559dcadc40c7eac45620b92d7d9e6ea947`).
- EXP01's predeclared OOF meta-stacker screen was rejected; no candidate
  submission was generated.
- The reported historical public score is not independently bound to this
  CSV hash by an official upload receipt, and no new public score was measured.
- The source and cache inputs are classified as historical-cache-assisted;
  no upstream model was independently retrained in this workspace.

## Start here

1. Run `python source\prepare_baseline.py` to snapshot the original synthesis
   source, its direct historical inputs, and the frozen reference.
2. Run `python source\run_baseline.py` to execute an isolated copy of the
   historical 85:15 synthesis. It never writes to the original source or
   output paths.
3. Run `python experiments\exp01_meta_weights.py` to reproduce the isolated
   five-fold meta-stacker screen. Its recorded run failed the predeclared gate
   and generated no test candidate.
4. Run `python -m unittest discover -s tests -v`.

The 85% Anchor74 input is a historical cached prediction. Its upstream
lineage is preserved as evidence, not relabeled as raw training. The direct
V20/V080041 dependencies are upstream of that cached Anchor74, but are not
direct inputs to the final EXP85 synthesis script.

See [ARCHITECTURE.md](ARCHITECTURE.md),
[EXPERIMENT_PLAN.md](EXPERIMENT_PLAN.md), and the generated
[baseline manifest](baseline/BASELINE_MANIFEST.json).
