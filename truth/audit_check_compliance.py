import ast
import json
import re
from pathlib import Path

CHECKS = [
    ("External/Historical CSV or Cache Read", r"(GCI_submission|pre info|\.npz|\.pkl|\.npy)"),
    ("Target Propagation (partner_enc/link_key)", r"(partner_enc|link_key|PY_ANY|PY_NEAR180)"),
    ("Legacy Covariate Person-Link Pooling", r"\bPL_S\b"),
    ("Hard-Coded Test Subset Weights", r"(w17\[|ext_nobs_te|is_pensioner_te|is_revolving_te)"),
    ("Train+Test Concatenation Pooling", r"pd\.concat\(\[df_train|pd\.concat\(\[tr,\s*te\]"),
    ("Scrambled View Concatenation Bug", r"np\.concatenate\(pv\)"),
    ("Validation Fold Early-Stopping Peeking", r"early_stopping_rounds"),
    ("Incomplete Cache Key Collision Risk", r"\(len\(fit\),\s*len\(qi\)|\(\'tr\',\s*len\(a\)"),
]

PREPROC_FUNCS = {
    "fit_category_maps", "build_row_features", "compute_fold_residuals",
    "compute_local_te", "compute_local_views", "StandardScaler"
}

def extract_code(p: Path):
    if p.suffix == ".ipynb":
        nb = json.loads(p.read_text(encoding="utf-8"))
        lines = []
        for idx, c in enumerate(nb.get("cells", [])):
            if c.get("cell_type") == "code":
                for lno, line in enumerate("".join(c.get("source", [])).splitlines(), 1):
                    s = line.strip()
                    if s.startswith("#") or s.startswith("print("):
                        continue
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

def extract_ipynb_source(p: Path) -> str:
    nb = json.loads(p.read_text(encoding="utf-8"))
    parts = []
    for c in nb.get("cells", []):
        if c.get("cell_type") == "code":
            parts.append("".join(c.get("source", [])))
    return "\n".join(parts)

def check_requirements(src: str, req_path: Path):
    if not req_path.exists():
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
    tree = ast.parse(src)
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

def _calls_in_nodes(nodes):
    calls = []
    for node in nodes:
        for sub in ast.walk(node):
            if isinstance(sub, ast.Call):
                if isinstance(sub.func, ast.Name):
                    calls.append((sub.lineno, sub.func.id))
                elif isinstance(sub.func, ast.Attribute):
                    calls.append((sub.lineno, sub.func.attr))
    return calls

def _has_yfit_mean(node):
    """Returns True if AST subtree contains bare `y_fit.mean()` (not `y_fit[ia].mean()`)."""
    for sub in ast.walk(node):
        if isinstance(sub, ast.Call) and isinstance(sub.func, ast.Attribute) and sub.func.attr == "mean":
            val = sub.func.value
            if isinstance(val, ast.Name) and val.id == "y_fit":
                return True
    return False

def _has_yfit_ia_mean(node):
    """Returns True if AST subtree contains `y_fit[ia].mean()`."""
    for sub in ast.walk(node):
        if isinstance(sub, ast.Call) and isinstance(sub.func, ast.Attribute) and sub.func.attr == "mean":
            val = sub.func.value
            if isinstance(val, ast.Subscript) and isinstance(val.value, ast.Name) and val.value.id == "y_fit":
                return True
    return False

def check_v28_semantics(src: str):
    flags = []
    tree = ast.parse(src)
    fn_map = {n.name: n for n in tree.body if isinstance(n, ast.FunctionDef)}

    # 1. Semantic Check: compute_local_te prior isolation
    if "compute_local_te" not in fn_map:
        flags.append("Missing compute_local_te function")
    else:
        te_fn = fn_map["compute_local_te"]
        # Find top-level loop over `in_splits` or inner loop inside `for kname`
        inner_loops = []
        pre_loop_stmts = []
        for stmt in te_fn.body:
            if isinstance(stmt, ast.For):
                for sub in ast.walk(stmt):
                    if isinstance(sub, ast.For) and isinstance(sub.iter, ast.Name) and sub.iter.id == "in_splits":
                        inner_loops.append(sub)
                # Check direct children of `for kname` before `for ia, ib in in_splits:`
                if any(isinstance(sub, ast.For) for sub in stmt.body):
                    for s2 in stmt.body:
                        if isinstance(s2, ast.For) and isinstance(s2.iter, ast.Name) and s2.iter.id == "in_splits":
                            break
                        pre_loop_stmts.append(s2)
            else:
                pre_loop_stmts.append(stmt)
        for s in pre_loop_stmts:
            if _has_yfit_mean(s):
                flags.append(f"Full `y_fit.mean()` computed before inner-fold loop in compute_local_te (L{s.lineno})")
        if not inner_loops:
            flags.append("Missing `in_splits` loop inside compute_local_te")
        for lp in inner_loops:
            if _has_yfit_mean(lp):
                flags.append(f"Full `y_fit.mean()` used inside `in_splits` loop in compute_local_te (L{lp.lineno})")
            if not _has_yfit_ia_mean(lp):
                flags.append(f"Missing fold-local `y_fit[ia].mean()` inside `in_splits` loop (L{lp.lineno})")

    # 2. Semantic Check: outer_folds & inner_folds preprocessing placement in main()
    if "main" not in fn_map:
        flags.append("Missing main() function")
    else:
        main_fn = fn_map["main"]
        outer_loop = None
        before_outer = []
        for stmt in main_fn.body:
            if isinstance(stmt, ast.For) and isinstance(stmt.iter, ast.Call) and isinstance(stmt.iter.func, ast.Name) and stmt.iter.func.id == "enumerate":
                if stmt.iter.args and isinstance(stmt.iter.args[0], ast.Name) and stmt.iter.args[0].id == "outer_folds":
                    outer_loop = stmt
                    break
            before_outer.append(stmt)
        if outer_loop is None:
            flags.append("Missing `for ... in enumerate(outer_folds):` loop in main()")
        else:
            for lno, fn in _calls_in_nodes(before_outer):
                if fn in PREPROC_FUNCS:
                    flags.append(f"Preprocessing `{fn}()` called before outer_folds at L{lno}")

            inner_loop = None
            before_inner = []
            for stmt in outer_loop.body:
                if isinstance(stmt, ast.For) and isinstance(stmt.iter, ast.Call) and isinstance(stmt.iter.func, ast.Name) and stmt.iter.func.id == "enumerate":
                    if stmt.iter.args and isinstance(stmt.iter.args[0], ast.Name) and stmt.iter.args[0].id == "inner_folds":
                        inner_loop = stmt
                        break
                before_inner.append(stmt)
            if inner_loop is None:
                flags.append("Missing `for ... in enumerate(inner_folds):` loop inside outer_folds")
            else:
                for lno, fn in _calls_in_nodes(before_inner):
                    if fn in PREPROC_FUNCS:
                        flags.append(f"Outer preprocessing `{fn}()` called before inner_folds (shared across ia/ib) at L{lno}")
                inner_calls = {fn for _, fn in _calls_in_nodes([inner_loop])}
                required_inner = {"fit_category_maps", "build_row_features", "compute_fold_residuals", "compute_local_te", "compute_local_views"}
                missing_inner = sorted(required_inner - inner_calls)
                if missing_inner:
                    flags.append(f"Missing inside inner_folds loop: {missing_inner}")

    if "s_oof[a]" in src:
        flags.append("Global `s_oof[a]` slicing detected instead of true `inner_oof`")
    return flags

def check_notebook_executed(nb_path: Path):
    nb = json.loads(nb_path.read_text(encoding="utf-8"))
    unexec = []
    no_out = []
    for idx, c in enumerate(nb.get("cells", [])):
        if c.get("cell_type") == "code":
            if not isinstance(c.get("execution_count"), int):
                unexec.append(idx)
            if len(c.get("outputs", [])) == 0:
                no_out.append(idx)
    return unexec, no_out

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
        if p.name.startswith("solo_champion_v28"):
            src = extract_ipynb_source(p) if p.suffix == ".ipynb" else p.read_text(encoding="utf-8")
            missing_req = check_requirements(src, req_path)
            if missing_req:
                clean = False
                print(f"  [FLAG] Missing in requirements.txt: {missing_req}")
            for fl in check_v28_semantics(src):
                clean = False
                print(f"  [FLAG] Semantic Nesting/Prior Check: {fl}")
        if p.name == "solo_champion_v28_certified.ipynb":
            unexec, no_out = check_notebook_executed(p)
            if unexec or no_out:
                clean = False
                print(f"  [FLAG] Unexecuted/empty notebook cells: unexec={unexec}, empty_outputs={no_out}")
        if clean:
            print("  [PASS] ZERO rule, preprocessing, inner/outer nesting, prior, requirement, or execution flags!")

if __name__ == "__main__":
    main()
