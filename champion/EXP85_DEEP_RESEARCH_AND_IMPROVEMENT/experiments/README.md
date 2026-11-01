# Experiment Protocols

Each actual execution gets a new `runs/<UTC timestamp>_<purpose>_<unique suffix>/` folder. Refuse to reuse or overwrite it.

For model runs, a run is incomplete unless it records:

- exact raw input and code/config SHA-256 values;
- exact fold IDs and seeds;
- per-fold metrics and saved OOF/test arrays with IDs;
- device/library versions plus observed GPU evidence or explicit CPU classification;
- execution command, runtime, logs, output hashes, and integrity result;
- baseline/previous-best comparisons on the same OOF rows;
- leakage tests and a clear no-official-score caveat.

Historical prediction artifacts may appear only in replay/diagnostic modes. They are prohibited as training features, labels, validation targets, or tuning criteria.

The Stage 0 read-only comparison is implemented by `..\source\correlation_gap_diagnostics.py`; it needs no model training and writes only to a new run directory. The final full-covariate run is `..\runs\20261102T003000Z_stage0_final\`. The first run is preserved but superseded for grouped missingness slices; see `STATUS.md`.
