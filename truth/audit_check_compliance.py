import json
import re
from pathlib import Path

CHECKS = [
    ("External/Historical CSV or Cache Read", r"(GCI_submission|v_22|pre info|\.npz|\.pkl|\.npy)"),
    ("Target Propagation (partner_enc/link_key)", r"(partner_enc|link_key|PY_ANY|PY_NEAR180|PL_S)"),
    ("Hard-Coded Test Subset Weights", r"(w17\[|ext_nobs_te|is_pensioner_te|is_revolving_te)"),
    ("Train+Test Concatenation Pooling", r"pd\.concat\(\[df_train|pd\.concat\(\[tr,\s*te\]"),
    ("Scrambled View Concatenation Bug", r"np\.concatenate\(pv\)"),
    ("Validation Fold Early-Stopping Peeking", r"early_stopping_rounds"),
]

targets = [
    Path(r"D:\gci\GCI_v22_pure_champion_master.ipynb"),
    Path(r"D:\gci\solo_champion_v23_master.py"),
    Path(r"D:\gci\solo_champion_v27_grandmaster.py"),
    Path(r"D:\gci\solo_champion_v28_certified.py"),
    Path(r"D:\gci\solo_champion_v28_certified.ipynb"),
]

def extract_code(p: Path):
    if p.suffix == ".ipynb":
        nb = json.loads(p.read_text(encoding="utf-8"))
        lines = []
        for idx, c in enumerate(nb.get("cells", [])):
            if c.get("cell_type") == "code":
                for lno, line in enumerate("".join(c.get("source", [])).splitlines(), 1):
                    lines.append((f"C{idx}:L{lno}", line))
        return lines
    lines = []
    in_doc = False
    for i, line in enumerate(p.read_text(encoding="utf-8", errors="ignore").splitlines(), 1):
        s = line.strip()
        if s.startswith('"""') or s.startswith("'''"):
            if s.count('"""') == 1 or s.count("'''") == 1:
                in_doc = not in_doc
            continue
        if in_doc or s.startswith("#"):
            continue
        lines.append((f"L{i}", line))
    return lines

for p in targets:
    print(f"\n=== AUDIT SCAN (EXECUTABLE CODE ONLY): {p.name} ===")
    if not p.exists():
        print("  (not found)")
        continue
    code_lines = extract_code(p)
    clean = True
    for label, pat in CHECKS:
        hits = [(loc, ln.strip()[:100]) for loc, ln in code_lines if re.search(pat, ln)]
        if hits:
            clean = False
            print(f"  [FLAG] {label}: {len(hits)} hit(s) -> first at {hits[0][0]}: {hits[0][1]}")
    if clean:
        print("  [PASS] ZERO rule/validation flags detected in executable code!")
