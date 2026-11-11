import ast
import json
import re
from pathlib import Path

CHECKS = [
    ("External/Historical CSV or Cache Read", r"(GCI_submission|pre info|\.npz|\.pkl|\.npy)"),
    ("Target Propagation (partner_enc/link_key)", r"(partner_enc|link_key|PY_ANY|PY_NEAR180|PL_S)"),
    ("Hard-Coded Test Subset Weights", r"(w17\[|ext_nobs_te|is_pensioner_te|is_revolving_te)"),
    ("Train+Test Concatenation Pooling", r"pd\.concat\(\[df_train|pd\.concat\(\[tr,\s*te\]"),
    ("Scrambled View Concatenation Bug", r"np\.concatenate\(pv\)"),
    ("Validation Fold Early-Stopping Peeking", r"early_stopping_rounds"),
    ("Incomplete Cache Key Collision Risk", r"\(len\(fit\),\s*len\(qi\)|\(\'tr\',\s*len\(a\)"),
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
        if in_doc or s.startswith("#") or s.startswith("print("):
            continue
        lines.append((f"L{i}", line))
    return lines

def check_requirements(py_path: Path, req_path: Path):
    if not py_path.exists() or not req_path.exists():
        return []
    req_pkgs = set()
    for ln in req_path.read_text(encoding="utf-8").splitlines():
        ln = ln.strip()
        if ln and not ln.startswith("#"):
            pkg = ln.split("==")[0].lower().replace("-", "_")
            req_pkgs.add(pkg)
    import_map = {"sklearn": "scikit_learn", "catboost": "catboost", "lightgbm": "lightgbm",
                  "xgboost": "xgboost", "torch": "torch", "scipy": "scipy", "numpy": "numpy", "pandas": "pandas"}
    missing = []
    tree = ast.parse(py_path.read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        mods = []
        if isinstance(node, ast.Import):
            mods = [n.name.split(".")[0] for n in node.names]
        elif isinstance(node, ast.ImportFrom) and node.module:
            mods = [node.module.split(".")[0]]
        for m in mods:
            if m in import_map and import_map[m] not in req_pkgs:
                missing.append(m)
    return sorted(set(missing))

def check_notebook_executed(nb_path: Path):
    nb = json.loads(nb_path.read_text(encoding="utf-8"))
    unexec = []
    for idx, c in enumerate(nb.get("cells", [])):
        if c.get("cell_type") == "code":
            if c.get("execution_count") is None:
                unexec.append(idx)
    return unexec

def main():
    targets = [
        Path(r"D:\gci\GCI_v22_pure_champion_master.ipynb"),
        Path(r"D:\gci\solo_champion_v23_master.py"),
        Path(r"D:\gci\solo_champion_v27_grandmaster.py"),
        Path(r"D:\gci\solo_champion_v28_certified.py"),
        Path(r"D:\gci\solo_champion_v28_certified.ipynb"),
    ]
    req_path = Path(r"D:\gci\requirements.txt")

    for p in targets:
        print(f"\n=== AUDIT SCAN: {p.name} ===")
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
        if p.name == "solo_champion_v28_certified.py":
            missing_req = check_requirements(p, req_path)
            if missing_req:
                clean = False
                print(f"  [FLAG] Missing in requirements.txt: {missing_req}")
            # Verify preprocessing + inner CV calibration are strictly inside outer fold loop
            src = p.read_text(encoding="utf-8")
            if "fit_category_maps(df_train)" in src or "compute_train_only_residuals(" in src:
                if "for fi, (a, b) in enumerate(outer_folds):" in src:
                    # Check that fit_category_maps is NOT called outside the outer loop
                    before_loop = src.split("for fi, (a, b) in enumerate(outer_folds):")[0]
                    if re.search(r"(fit_category_maps|compute_fold_residuals|compute_fold_views)\(", before_loop.split("def main():")[-1]):
                        clean = False
                        print("  [FLAG] Preprocessing called before outer_folds loop!")
            if "s_oof[a]" in src:
                clean = False
                print("  [FLAG] Global OOF sliced across outer folds instead of true inner_oof!")
        if p.name == "solo_champion_v28_certified.ipynb":
            unexec = check_notebook_executed(p)
            if unexec:
                clean = False
                print(f"  [FLAG] Unexecuted notebook cells: {unexec}")
        if clean:
            print("  [PASS] ZERO rule, preprocessing, nesting, requirement, or notebook execution flags!")

if __name__ == "__main__":
    main()
