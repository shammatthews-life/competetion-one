# EXP85 Notebook Lineage Comparison

## Scope

The listed notebooks and runners were parsed as JSON/Python source. No
notebook cells were executed. Each cell's full source, execution count,
metadata, and text outputs are retained in `NOTEBOOK_CELL_DIFF.json`;
non-text MIME payloads are represented by size and SHA-256.

## Notebook inventory

| Notebook | Available | Cells | Code | Markdown | Executed code | File SHA-256 |
|---|---:|---:|---:|---:|---:|---|
| `champion/exp85_solo.ipynb` | Yes | 19 | 9 | 10 | 9 | `d7d16d00599061b041e2b85619eb4dbc3f3607d437766c9c1e72d928a913f7b7` |
| `champion/exp85_solo.pre_independent_wiring_backup.ipynb` | Yes | 13 | 12 | 1 | 0 | `1ccfab28ef8534fc176c757c79250547c03a041a922851d683c82d0df1241d23` |
| `champion/exp85_solo_backup_pre_fix.ipynb` | Yes | 11 | 10 | 1 | 0 | `34b66ca02ca74f1c8ee80a24d80a519292f3d0d6f00e197e1900e16f94dfeb4b` |
| `gemini/champion/exp85_solo.ipynb` | Yes | 10 | 9 | 1 | 0 | `2640eaa9b828c9b9485497f9ce35060bd4d89db179820a5a42bb153039862312` |
| `gemini/EXP85_LEGITIMATE_RECONSTRUCTION.ipynb` | Yes | 9 | 8 | 1 | 0 | `05086c2a109209ce7cf1d2c37ec05982ce43988f43a825cd0aaa81a7db772686` |
| `gemini/GCI_CHAMPION_0801340_CLEAN_ROOM.ipynb` | Yes | 19 | 18 | 1 | 18 | `52f21f66bf3ed35bd47859136b33b3250c8ca7582b96151405ca17c38d047e34` |
| `gemini/GCI_CHAMPION_0801340_EXACT_REPRODUCIBLE.ipynb` | Yes | 18 | 17 | 1 | 0 | `d02b5e186774822636981b0a245a4810c0fb9b02cd6dec7201d08b9a11174368` |
| `gemini/GCI_CHAMPION_0801340_SELF_CONTAINED.ipynb` | Yes | 19 | 18 | 1 | 18 | `52f21f66bf3ed35bd47859136b33b3250c8ca7582b96151405ca17c38d047e34` |
| `gemini/GCI_WORLD_2026_CHAMPION_080134_REPRODUCIBLE.ipynb` | Yes | 7 | 6 | 1 | 0 | `51c97fea9d165c4bb14bd7fbed9bfbbf8b07cc01b0384c9b74d79aad5850dcd7` |

## Code-source comparisons

- `champion/exp85_solo.ipynb` vs `champion/exp85_solo.pre_independent_wiring_backup.ipynb`: line similarity 0.020; exact code-cell sequence similarity 0.000; 0 exact code cell(s) shared; 12 changed line block(s).
- `champion/exp85_solo.ipynb` vs `champion/exp85_solo_backup_pre_fix.ipynb`: line similarity 0.034; exact code-cell sequence similarity 0.000; 0 exact code cell(s) shared; 13 changed line block(s).
- `champion/exp85_solo.ipynb` vs `gemini/champion/exp85_solo.ipynb`: line similarity 0.065; exact code-cell sequence similarity 0.000; 0 exact code cell(s) shared; 12 changed line block(s).
- `champion/exp85_solo.ipynb` vs `gemini/EXP85_LEGITIMATE_RECONSTRUCTION.ipynb`: line similarity 0.031; exact code-cell sequence similarity 0.000; 0 exact code cell(s) shared; 14 changed line block(s).
- `champion/exp85_solo.ipynb` vs `gemini/GCI_CHAMPION_0801340_CLEAN_ROOM.ipynb`: line similarity 0.042; exact code-cell sequence similarity 0.000; 0 exact code cell(s) shared; 15 changed line block(s).
- `champion/exp85_solo.ipynb` vs `gemini/GCI_CHAMPION_0801340_EXACT_REPRODUCIBLE.ipynb`: line similarity 0.024; exact code-cell sequence similarity 0.000; 0 exact code cell(s) shared; 9 changed line block(s).
- `champion/exp85_solo.ipynb` vs `gemini/GCI_CHAMPION_0801340_SELF_CONTAINED.ipynb`: line similarity 0.042; exact code-cell sequence similarity 0.000; 0 exact code cell(s) shared; 15 changed line block(s).
- `champion/exp85_solo.ipynb` vs `gemini/GCI_WORLD_2026_CHAMPION_080134_REPRODUCIBLE.ipynb`: line similarity 0.029; exact code-cell sequence similarity 0.000; 0 exact code cell(s) shared; 9 changed line block(s).
- `champion/exp85_solo.pre_independent_wiring_backup.ipynb` vs `champion/exp85_solo_backup_pre_fix.ipynb`: line similarity 0.052; exact code-cell sequence similarity 0.000; 0 exact code cell(s) shared; 31 changed line block(s).
- `champion/exp85_solo.pre_independent_wiring_backup.ipynb` vs `gemini/champion/exp85_solo.ipynb`: line similarity 0.318; exact code-cell sequence similarity 0.857; 9 exact code cell(s) shared; 1 changed line block(s).
- `champion/exp85_solo.pre_independent_wiring_backup.ipynb` vs `gemini/EXP85_LEGITIMATE_RECONSTRUCTION.ipynb`: line similarity 0.013; exact code-cell sequence similarity 0.000; 0 exact code cell(s) shared; 7 changed line block(s).
- `champion/exp85_solo.pre_independent_wiring_backup.ipynb` vs `gemini/GCI_CHAMPION_0801340_CLEAN_ROOM.ipynb`: line similarity 0.014; exact code-cell sequence similarity 0.000; 0 exact code cell(s) shared; 10 changed line block(s).
- `champion/exp85_solo.pre_independent_wiring_backup.ipynb` vs `gemini/GCI_CHAMPION_0801340_EXACT_REPRODUCIBLE.ipynb`: line similarity 0.013; exact code-cell sequence similarity 0.000; 0 exact code cell(s) shared; 11 changed line block(s).
- `champion/exp85_solo.pre_independent_wiring_backup.ipynb` vs `gemini/GCI_CHAMPION_0801340_SELF_CONTAINED.ipynb`: line similarity 0.014; exact code-cell sequence similarity 0.000; 0 exact code cell(s) shared; 10 changed line block(s).
- `champion/exp85_solo.pre_independent_wiring_backup.ipynb` vs `gemini/GCI_WORLD_2026_CHAMPION_080134_REPRODUCIBLE.ipynb`: line similarity 0.012; exact code-cell sequence similarity 0.000; 0 exact code cell(s) shared; 11 changed line block(s).
- `champion/exp85_solo_backup_pre_fix.ipynb` vs `gemini/champion/exp85_solo.ipynb`: line similarity 0.129; exact code-cell sequence similarity 0.000; 0 exact code cell(s) shared; 27 changed line block(s).
- `champion/exp85_solo_backup_pre_fix.ipynb` vs `gemini/EXP85_LEGITIMATE_RECONSTRUCTION.ipynb`: line similarity 0.120; exact code-cell sequence similarity 0.000; 0 exact code cell(s) shared; 22 changed line block(s).
- `champion/exp85_solo_backup_pre_fix.ipynb` vs `gemini/GCI_CHAMPION_0801340_CLEAN_ROOM.ipynb`: line similarity 0.041; exact code-cell sequence similarity 0.000; 0 exact code cell(s) shared; 15 changed line block(s).
- `champion/exp85_solo_backup_pre_fix.ipynb` vs `gemini/GCI_CHAMPION_0801340_EXACT_REPRODUCIBLE.ipynb`: line similarity 0.095; exact code-cell sequence similarity 0.000; 0 exact code cell(s) shared; 22 changed line block(s).
- `champion/exp85_solo_backup_pre_fix.ipynb` vs `gemini/GCI_CHAMPION_0801340_SELF_CONTAINED.ipynb`: line similarity 0.041; exact code-cell sequence similarity 0.000; 0 exact code cell(s) shared; 15 changed line block(s).
- `champion/exp85_solo_backup_pre_fix.ipynb` vs `gemini/GCI_WORLD_2026_CHAMPION_080134_REPRODUCIBLE.ipynb`: line similarity 0.042; exact code-cell sequence similarity 0.000; 0 exact code cell(s) shared; 18 changed line block(s).
- `gemini/champion/exp85_solo.ipynb` vs `gemini/EXP85_LEGITIMATE_RECONSTRUCTION.ipynb`: line similarity 0.027; exact code-cell sequence similarity 0.000; 0 exact code cell(s) shared; 6 changed line block(s).
- `gemini/champion/exp85_solo.ipynb` vs `gemini/GCI_CHAMPION_0801340_CLEAN_ROOM.ipynb`: line similarity 0.030; exact code-cell sequence similarity 0.000; 0 exact code cell(s) shared; 10 changed line block(s).
- `gemini/champion/exp85_solo.ipynb` vs `gemini/GCI_CHAMPION_0801340_EXACT_REPRODUCIBLE.ipynb`: line similarity 0.020; exact code-cell sequence similarity 0.000; 0 exact code cell(s) shared; 8 changed line block(s).
- `gemini/champion/exp85_solo.ipynb` vs `gemini/GCI_CHAMPION_0801340_SELF_CONTAINED.ipynb`: line similarity 0.030; exact code-cell sequence similarity 0.000; 0 exact code cell(s) shared; 10 changed line block(s).
- `gemini/champion/exp85_solo.ipynb` vs `gemini/GCI_WORLD_2026_CHAMPION_080134_REPRODUCIBLE.ipynb`: line similarity 0.029; exact code-cell sequence similarity 0.000; 0 exact code cell(s) shared; 9 changed line block(s).
- `gemini/EXP85_LEGITIMATE_RECONSTRUCTION.ipynb` vs `gemini/GCI_CHAMPION_0801340_CLEAN_ROOM.ipynb`: line similarity 0.037; exact code-cell sequence similarity 0.000; 0 exact code cell(s) shared; 15 changed line block(s).
- `gemini/EXP85_LEGITIMATE_RECONSTRUCTION.ipynb` vs `gemini/GCI_CHAMPION_0801340_EXACT_REPRODUCIBLE.ipynb`: line similarity 0.527; exact code-cell sequence similarity 0.000; 0 exact code cell(s) shared; 40 changed line block(s).
- `gemini/EXP85_LEGITIMATE_RECONSTRUCTION.ipynb` vs `gemini/GCI_CHAMPION_0801340_SELF_CONTAINED.ipynb`: line similarity 0.037; exact code-cell sequence similarity 0.000; 0 exact code cell(s) shared; 15 changed line block(s).
- `gemini/EXP85_LEGITIMATE_RECONSTRUCTION.ipynb` vs `gemini/GCI_WORLD_2026_CHAMPION_080134_REPRODUCIBLE.ipynb`: line similarity 0.246; exact code-cell sequence similarity 0.000; 0 exact code cell(s) shared; 52 changed line block(s).
- `gemini/GCI_CHAMPION_0801340_CLEAN_ROOM.ipynb` vs `gemini/GCI_CHAMPION_0801340_EXACT_REPRODUCIBLE.ipynb`: line similarity 0.042; exact code-cell sequence similarity 0.000; 0 exact code cell(s) shared; 13 changed line block(s).
- `gemini/GCI_CHAMPION_0801340_CLEAN_ROOM.ipynb` vs `gemini/GCI_CHAMPION_0801340_SELF_CONTAINED.ipynb`: line similarity 1.000; exact code-cell sequence similarity 1.000; 18 exact code cell(s) shared; 0 changed line block(s).
- `gemini/GCI_CHAMPION_0801340_CLEAN_ROOM.ipynb` vs `gemini/GCI_WORLD_2026_CHAMPION_080134_REPRODUCIBLE.ipynb`: line similarity 0.031; exact code-cell sequence similarity 0.000; 0 exact code cell(s) shared; 13 changed line block(s).
- `gemini/GCI_CHAMPION_0801340_EXACT_REPRODUCIBLE.ipynb` vs `gemini/GCI_CHAMPION_0801340_SELF_CONTAINED.ipynb`: line similarity 0.042; exact code-cell sequence similarity 0.000; 0 exact code cell(s) shared; 13 changed line block(s).
- `gemini/GCI_CHAMPION_0801340_EXACT_REPRODUCIBLE.ipynb` vs `gemini/GCI_WORLD_2026_CHAMPION_080134_REPRODUCIBLE.ipynb`: line similarity 0.387; exact code-cell sequence similarity 0.000; 0 exact code cell(s) shared; 77 changed line block(s).
- `gemini/GCI_CHAMPION_0801340_SELF_CONTAINED.ipynb` vs `gemini/GCI_WORLD_2026_CHAMPION_080134_REPRODUCIBLE.ipynb`: line similarity 0.031; exact code-cell sequence similarity 0.000; 0 exact code cell(s) shared; 13 changed line block(s).

## Runner identity checks

- `champion/exp85_solo_runner.py` vs `champion/exp85_solo.ipynb`: not identical; similarity is diagnostic only (line sequence similarity 0.062).
- `champion/exp85_solo_runner.py` vs `champion/exp85_solo.pre_independent_wiring_backup.ipynb`: not identical; similarity is diagnostic only (line sequence similarity 0.322).
- `champion/exp85_solo_runner.py` vs `champion/exp85_solo_backup_pre_fix.ipynb`: not identical; similarity is diagnostic only (line sequence similarity 0.128).
- `champion/exp85_solo_runner.py` vs `gemini/champion/exp85_solo.ipynb`: not identical; similarity is diagnostic only (line sequence similarity 0.964).
- `champion/exp85_solo_runner.py` vs `gemini/EXP85_LEGITIMATE_RECONSTRUCTION.ipynb`: not identical; similarity is diagnostic only (line sequence similarity 0.026).
- `champion/exp85_solo_runner.py` vs `gemini/GCI_CHAMPION_0801340_CLEAN_ROOM.ipynb`: not identical; similarity is diagnostic only (line sequence similarity 0.028).
- `champion/exp85_solo_runner.py` vs `gemini/GCI_CHAMPION_0801340_EXACT_REPRODUCIBLE.ipynb`: not identical; similarity is diagnostic only (line sequence similarity 0.017).
- `champion/exp85_solo_runner.py` vs `gemini/GCI_CHAMPION_0801340_SELF_CONTAINED.ipynb`: not identical; similarity is diagnostic only (line sequence similarity 0.028).
- `champion/exp85_solo_runner.py` vs `gemini/GCI_WORLD_2026_CHAMPION_080134_REPRODUCIBLE.ipynb`: not identical; similarity is diagnostic only (line sequence similarity 0.027).
- `gemini/champion/exp85_solo_runner.py` vs `champion/exp85_solo.ipynb`: not identical; similarity is diagnostic only (line sequence similarity 0.064).
- `gemini/champion/exp85_solo_runner.py` vs `champion/exp85_solo.pre_independent_wiring_backup.ipynb`: not identical; similarity is diagnostic only (line sequence similarity 0.302).
- `gemini/champion/exp85_solo_runner.py` vs `champion/exp85_solo_backup_pre_fix.ipynb`: not identical; similarity is diagnostic only (line sequence similarity 0.130).
- `gemini/champion/exp85_solo_runner.py` vs `gemini/champion/exp85_solo.ipynb`: not identical; similarity is diagnostic only (line sequence similarity 0.912).
- `gemini/champion/exp85_solo_runner.py` vs `gemini/EXP85_LEGITIMATE_RECONSTRUCTION.ipynb`: not identical; similarity is diagnostic only (line sequence similarity 0.026).
- `gemini/champion/exp85_solo_runner.py` vs `gemini/GCI_CHAMPION_0801340_CLEAN_ROOM.ipynb`: not identical; similarity is diagnostic only (line sequence similarity 0.028).
- `gemini/champion/exp85_solo_runner.py` vs `gemini/GCI_CHAMPION_0801340_EXACT_REPRODUCIBLE.ipynb`: not identical; similarity is diagnostic only (line sequence similarity 0.018).
- `gemini/champion/exp85_solo_runner.py` vs `gemini/GCI_CHAMPION_0801340_SELF_CONTAINED.ipynb`: not identical; similarity is diagnostic only (line sequence similarity 0.028).
- `gemini/champion/exp85_solo_runner.py` vs `gemini/GCI_WORLD_2026_CHAMPION_080134_REPRODUCIBLE.ipynb`: not identical; similarity is diagnostic only (line sequence similarity 0.028).

These string comparisons are provenance checks, not proof that a
notebook was executed or that its recorded outputs were produced by
the displayed source. Execution counts and saved outputs are
reported separately and must be corroborated by run evidence.
