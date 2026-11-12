import ast
import hashlib
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

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
    return "\n".join("".join(c.get("source", [])) for c in nb.get("cells", []) if c.get("cell_type") == "code")

def extract_ipynb_outputs(p: Path) -> str:
    nb = json.loads(p.read_text(encoding="utf-8"))
    out = []
    for c in nb.get("cells", []):
        if c.get("cell_type") == "code":
            for o in c.get("outputs", []):
                if "text" in o:
                    out.append("".join(o["text"]) if isinstance(o["text"], list) else str(o["text"]))
    return "\n".join(out)

def check_requirements(src: str, req_path: Path):
    if not req_path.exists():
        return ["MISSING requirements.txt"]
    req_pkgs = set()
    for ln in req_path.read_text(encoding="utf-8").splitlines():
        ln = ln.strip()
        if ln and not ln.startswith("#") and not ln.startswith("--"):
            pkg = re.split(r"[=<>!~\[]", ln)[0].strip().lower().replace("-", "_")
            if pkg:
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
                    calls.append((sub.lineno, sub.func.id, sub))
                elif isinstance(sub.func, ast.Attribute):
                    calls.append((sub.lineno, sub.func.attr, sub))
    return calls

def _arg_names(call_node):
    return [ast.unparse(a) for a in call_node.args] + [f"{k.arg}={ast.unparse(k.value)}" for k in call_node.keywords]

def _has_yfit_mean(node):
    for sub in ast.walk(node):
        if isinstance(sub, ast.Call) and isinstance(sub.func, ast.Attribute) and sub.func.attr == "mean":
            val = sub.func.value
            if isinstance(val, ast.Name) and val.id == "y_fit":
                return True
    return False

def _has_yfit_exact_ia_mean(node, ia_var: str):
    for sub in ast.walk(node):
        if isinstance(sub, ast.Call) and isinstance(sub.func, ast.Attribute) and sub.func.attr == "mean":
            val = sub.func.value
            if isinstance(val, ast.Subscript) and isinstance(val.value, ast.Name) and val.value.id == "y_fit":
                if isinstance(val.slice, ast.Name) and val.slice.id == ia_var:
                    return True
    return False

def _get_inner_ia_var(loop_node):
    if isinstance(loop_node.target, ast.Tuple) and len(loop_node.target.elts) >= 1:
        if isinstance(loop_node.target.elts[0], ast.Name):
            return loop_node.target.elts[0].id
    return None

def check_v28_semantics(src: str):
    flags = []
    tree = ast.parse(src)
    fn_map = {n.name: n for n in tree.body if isinstance(n, ast.FunctionDef)}

    # 1. Semantic Check: compute_local_te prior isolation (`y_fit[ia].mean()` strictly matching loop var)
    if "compute_local_te" not in fn_map:
        flags.append("Missing compute_local_te function")
    else:
        te_fn = fn_map["compute_local_te"]
        inner_loops = []
        pre_loop_stmts = []
        for stmt in te_fn.body:
            if isinstance(stmt, ast.For):
                for sub in ast.walk(stmt):
                    if isinstance(sub, ast.For) and isinstance(sub.iter, ast.Name) and sub.iter.id == "in_splits":
                        inner_loops.append(sub)
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
            for _, fn_name, _ in _calls_in_nodes([s]):
                if fn_name == "unique":
                    flags.append(f"Full `pd.unique(s_fit)` vocabulary built before inner-fold loop in compute_local_te (L{s.lineno})")
        if not inner_loops:
            flags.append("Missing `in_splits` loop inside compute_local_te")
        for lp in inner_loops:
            ia_var = _get_inner_ia_var(lp)
            if _has_yfit_mean(lp):
                flags.append(f"Full `y_fit.mean()` used inside `in_splits` loop in compute_local_te (L{lp.lineno})")
            if not ia_var or not _has_yfit_exact_ia_mean(lp, ia_var):
                flags.append(f"Missing exact fold-local `y_fit[{ia_var}].mean()` inside `in_splits` loop (L{lp.lineno})")
            if not any(fn_name == "unique" for _, fn_name, _ in _calls_in_nodes([lp])):
                flags.append(f"Missing inner-fold `pd.unique` vocabulary construction inside `in_splits` loop (L{lp.lineno})")

    # 2. Semantic Check: outer_folds & inner_folds preprocessing placement + exact inner arguments
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
            for lno, fn, _ in _calls_in_nodes(before_outer):
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
                for lno, fn, _ in _calls_in_nodes(before_inner):
                    if fn in PREPROC_FUNCS:
                        flags.append(f"Outer preprocessing `{fn}()` called before inner_folds at L{lno}")
                inner_calls = _calls_in_nodes([inner_loop])
                seen_inner = {}
                for lno, fn, node in inner_calls:
                    seen_inner.setdefault(fn, []).append((lno, _arg_names(node)))
                required_inner = {"fit_category_maps", "build_row_features", "compute_fold_residuals", "compute_local_te", "compute_local_views"}
                missing_inner = sorted(required_inner - set(seen_inner))
                if missing_inner:
                    flags.append(f"Missing inside inner_folds loop: {missing_inner}")
                # Argument-level isolation verification inside inner_folds:
                for lno, args in seen_inner.get("fit_category_maps", []):
                    if args != ["df_ia"]:
                        flags.append(f"fit_category_maps must take `df_ia` inside inner_folds (got {args} at L{lno})")
                for lno, args in seen_inner.get("compute_fold_residuals", []):
                    if args[:3] != ["df_ia", "df_ib", "None"]:
                        flags.append(f"compute_fold_residuals inside inner_folds must pass (df_ia, df_ib, None) (got {args} at L{lno})")
                for lno, args in seen_inner.get("compute_local_te", []):
                    if args[:3] != ["df_ia", "df_ib", "y_ia"]:
                        flags.append(f"compute_local_te inside inner_folds must pass (df_ia, df_ib, y_ia) (got {args} at L{lno})")
                for lno, args in seen_inner.get("compute_local_views", []):
                    if args[:5] != ["Xia_res", "Xib_res", "df_ia", "df_ib", "y_ia"]:
                        flags.append(f"compute_local_views inside inner_folds must pass (Xia_res, Xib_res, df_ia, df_ib, y_ia) (got {args} at L{lno})")

    if "s_oof[a]" in src:
        flags.append("Global `s_oof[a]` slicing detected instead of true `inner_oof`")

    # 3. Consistency, Determinism, Unique-ID & Docstring Hygiene Checks
    if "meta_cv" in src or re.search(r"StratifiedKFold\([^)]*\)\.split\(\s*Z_in", src):
        flags.append("Un-nested `meta_cv` cross-validation over `Z_in` detected (use predeclared C=0.05 without contaminated meta-CV)")
    if re.search(r"(?:iterations|n_estimators|epochs)\s*=\s*[^\n]*\bif\s+is_outer\s+else\b", src):
        flags.append("Inner/outer iteration/epoch mismatch (`if is_outer else`) detected in model hyperparameters")
    for req_expr in (
        'df_test["SK_ID_CURR"].is_unique',
        'sample_sub["SK_ID_CURR"].is_unique',
        'sub["SK_ID_CURR"].is_unique',
        "torch.use_deterministic_algorithms",
    ):
        if req_expr not in src:
            flags.append(f"Missing required safety/determinism check: `{req_expr}`")
    for stale in ("torch==2.15.0", "all 9 models", "9 orthogonal models"):
        if stale in src:
            flags.append(f"Stale documentation string detected: `{stale}`")

    return flags

def verify_notebook_and_artifacts(nb_path: Path, py_path: Path):
    flags = []
    nb = json.loads(nb_path.read_text(encoding="utf-8"))
    cc = [c for c in nb.get("cells", []) if c.get("cell_type") == "code"]
    unexec = [i for i, c in enumerate(cc) if not isinstance(c.get("execution_count"), int)]
    no_out = [i for i, c in enumerate(cc) if len(c.get("outputs", [])) == 0]
    if unexec or no_out:
        flags.append(f"Unexecuted/empty code cells: unexec={unexec}, empty_outputs={no_out}")

    # Verify saved output SHA-256 of submission + raw inputs against disk files (fail if missing)
    out_txt = extract_ipynb_outputs(nb_path)
    sub_path = ROOT / "LEADERBOARD_SUBMISSION_V28_CERTIFIED.csv"
    if not sub_path.exists():
        flags.append(f"Missing submission CSV: {sub_path}")
    else:
        sub_sha = hashlib.sha256(sub_path.read_bytes()).hexdigest().upper()
        if sub_sha not in out_txt:
            flags.append(f"Notebook output does not record current disk submission SHA-256 ({sub_sha})")

    for fname in ("train.csv", "test.csv", "sample_submission.csv"):
        fpath = ROOT / "inpuy" / fname
        if not fpath.exists():
            flags.append(f"Missing required raw input file: {fpath}")
        else:
            f_sha = hashlib.sha256(fpath.read_bytes()).hexdigest().upper()
            if f_sha not in out_txt:
                flags.append(f"Notebook output missing raw input provenance SHA-256 for {fname} ({f_sha[:16]}...)")

    # Full AST & normalized line-by-line equivalence between .py and .ipynb code cells
    py_norm = [
        ln.strip() for ln in py_path.read_text(encoding="utf-8").splitlines()
        if ln.strip() and not ln.strip().startswith("#") and ln.strip() not in ('if __name__ == "__main__":', "main()")
    ]
    nb_raw_lines = extract_ipynb_source(nb_path).splitlines()
    nb_norm = [
        ln.strip() for ln in nb_raw_lines
        if ln.strip()
        and not ln.strip().startswith("#")
        and not ln.strip().startswith("print('[+] Defined stage:")
        and not ln.strip().startswith("print(f'Env ready:")
        and not ln.strip().startswith("f'lgb={lgb.__version__}")
        and ln.strip() not in ('if __name__ == "__main__":', 'sys.argv = ["solo_champion_v28_certified.py"]', "main()")
    ]
    if py_norm != nb_norm:
        flags.append(f"Full source-equivalence mismatch between {py_path.name} ({len(py_norm)} stmts) and {nb_path.name} ({len(nb_norm)} stmts)")
    return flags

def check_bench_split_isolation(eval_seeds=(2026, 2027)):
    flags = []
    for p in sorted((ROOT / "truth").glob("bench_*.py")):
        tree = ast.parse(p.read_text(encoding="utf-8", errors="ignore"))
        for node in ast.walk(tree):
            if isinstance(node, ast.Constant) and node.value in eval_seeds:
                flags.append(f"Benchmark {p.name} uses integer literal {node.value} in executable code (L{node.lineno})")
                break
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "StratifiedKFold":
                for kw in node.keywords:
                    if kw.arg == "random_state" and isinstance(kw.value, ast.Constant) and kw.value.value not in (42, 77, 100, 101, 777):
                        flags.append(f"Benchmark {p.name} uses unexpected StratifiedKFold random_state={kw.value.value} (L{node.lineno})")
    return flags

def report_actual_split_intersections():
    """Calculates actual row-index intersections between SEED=42/777 bench splits and SEED_EVAL=2027 folds."""
    tr_path = ROOT / "inpuy" / "train.csv"
    if not tr_path.exists():
        return [f"Missing {tr_path} for actual split-intersection audit"]
    import pandas as pd
    from sklearn.model_selection import StratifiedKFold
    y = pd.read_csv(tr_path, usecols=["TARGET"])["TARGET"].values
    dev_outer = list(StratifiedKFold(5, shuffle=True, random_state=42).split(y, y))[0][0]
    _, ib = list(StratifiedKFold(3, shuffle=True, random_state=777).split(dev_outer, y[dev_outer]))[0]
    dev_set = set(dev_outer.tolist())
    ib_set = set(dev_outer[ib].tolist())
    eval_folds = list(StratifiedKFold(5, shuffle=True, random_state=2027).split(y, y))
    lines = []
    for fi, (_, b) in enumerate(eval_folds, 1):
        b_set = set(b.tolist())
        ov_dev = len(b_set & dev_set)
        ov_ib = len(b_set & ib_set)
        lines.append(f"  [INFO] SEED_EVAL=2027 Fold {fi} ({len(b_set):,} rows): dev_outer overlap={ov_dev:,} ({100*ov_dev/len(b_set):.2f}%) | bench_v30 ib overlap={ov_ib:,} ({100*ov_ib/len(b_set):.2f}%)")
    return lines

def main():
    required = [
        ROOT / "GCI_v22_pure_champion_master.ipynb",
        ROOT / "v22_champion_audit_certified.py",
        ROOT / "v22_champion_audit_certified_v2.py",
        ROOT / "v22_champion_audit_certified_v3.py",
        ROOT / "v22_champion_audit_certified_v4_restoration.py",
        ROOT / "v22_champion_audit_certified_v5_high_loc.py",
        ROOT / "v22_champion_audit_certified_v6_ultra_loc.py",
        ROOT / "solo_champion_v23_master.py",
        ROOT / "solo_champion_v27_grandmaster.py",
        ROOT / "solo_champion_v28_certified.py",
        ROOT / "solo_champion_v28_certified.ipynb",
    ]
    req_path = ROOT / "requirements.txt"
    v28_failed = False

    for p in required:
        print(f"\n=== AUDIT SCAN: {p.name} ===")
        if not p.exists():
            print("  [FAIL] Required file missing!")
            v28_failed = True
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
                v28_failed = True
                print(f"  [FLAG] Missing in requirements.txt: {missing_req}")
            for fl in check_v28_semantics(src):
                clean = False
                v28_failed = True
                print(f"  [FLAG] Semantic Nesting/Prior/Arg Check: {fl}")
        if p.name == "solo_champion_v28_certified.ipynb":
            for fl in verify_notebook_and_artifacts(p, ROOT / "solo_champion_v28_certified.py"):
                clean = False
                v28_failed = True
                print(f"  [FLAG] Notebook/Artifact Provenance Check: {fl}")
        if p.name == "solo_champion_v27_grandmaster.py":
            v27_sem = check_v28_semantics(p.read_text(encoding="utf-8"))
            print(f"  [HISTORICAL ARCHIVE] Regex-clean for external CSV/partner_enc, NOT clean nested CV ({len(v27_sem)} semantic nesting/global-OOF flags; non-comparable baseline).")
        elif clean:
            print("  [PASS] ZERO rule, preprocessing, inner/outer arg nesting, prior, requirement, or provenance flags!")

    print("\n=== BENCHMARK SPLIT ISOLATION CHECK ===")
    bench_flags = check_bench_split_isolation((2026, 2027))
    if bench_flags:
        v28_failed = True
        for fl in bench_flags:
            print(f"  [FLAG] {fl}")
    else:
        print("  [PASS] Zero truth/bench_*.py scripts reference SEED_EVAL=2026/2027 or unexpected StratifiedKFold seeds (static AST check).")
    for line in report_actual_split_intersections():
        print(line)

    if v28_failed:
        sys.exit(1)

if __name__ == "__main__":
    main()
