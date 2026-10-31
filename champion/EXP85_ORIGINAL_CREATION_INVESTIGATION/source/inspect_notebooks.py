from __future__ import annotations

import argparse
import difflib
import hashlib
import json
import re
from pathlib import Path
from typing import Any


NOTEBOOKS = (
    "champion/exp85_solo.ipynb",
    "champion/exp85_solo.pre_independent_wiring_backup.ipynb",
    "champion/exp85_solo_backup_pre_fix.ipynb",
    "gemini/champion/exp85_solo.ipynb",
    "gemini/EXP85_LEGITIMATE_RECONSTRUCTION.ipynb",
    "gemini/GCI_CHAMPION_0801340_CLEAN_ROOM.ipynb",
    "gemini/GCI_CHAMPION_0801340_EXACT_REPRODUCIBLE.ipynb",
    "gemini/GCI_CHAMPION_0801340_SELF_CONTAINED.ipynb",
    "gemini/GCI_WORLD_2026_CHAMPION_080134_REPRODUCIBLE.ipynb",
)

RUNNERS = (
    "champion/exp85_solo_runner.py",
    "gemini/champion/exp85_solo_runner.py",
)

RELATED_SCRIPTS = (
    "gemini/build_exp85_optimal_macro_apex.py",
    "gemini/build_grand_master_macro_stacker_apex.py",
    "gemini/build_v80250_power_apex.py",
    "gemini/solo_grand_master_apex_080.py",
    "champion/02_EXP85_REBUILT/source/generate_xgb10_from_scratch.py",
    "champion/02_EXP85_REBUILT/source/generate_v55_from_scratch.py",
    "champion/02_EXP85_REBUILT/source/generate_v17_from_scratch.py",
    "champion/02_EXP85_REBUILT/source/generate_anchor74_from_scratch.py",
    "champion/02_EXP85_REBUILT/source/run_all_recreations.py",
    "gemini/EXP85_DEPENDENCY_MANIFEST.json",
    "gemini/EXP85_GENERATED_ARTIFACTS.json",
    "gemini/EXP85_SHA256_MANIFEST.json",
)

CATEGORY_PATTERNS = {
    "feature_generation": re.compile(
        r"feature|transform|encode|ratio|aggregate|matrix|select|column",
        re.IGNORECASE,
    ),
    "models": re.compile(
        r"lightgbm|lgbm|xgboost|xgb|catboost|extra.?trees|ridge|classifier|"
        r"regressor|fit\(",
        re.IGNORECASE,
    ),
    "folds_and_seeds": re.compile(
        r"StratifiedKFold|KFold|GroupKFold|random_state|seed|fold",
        re.IGNORECASE,
    ),
    "cache_and_file_discovery": re.compile(
        r"glob\(|rglob\(|os\.walk|listdir|cache|\.npz|\.npy|pickle|"
        r"read_parquet|read_pickle|load\(",
        re.IGNORECASE,
    ),
    "raw_inputs": re.compile(
        r"train\.csv|test\.csv|read_csv|TARGET|SK_ID_CURR",
        re.IGNORECASE,
    ),
    "predictions_and_outputs": re.compile(
        r"oof|prediction|predict|to_csv|submission|\.csv",
        re.IGNORECASE,
    ),
    "blend_formula": re.compile(
        r"probit|rankdata|spearman|weight|blend|0\.85|0\.15|0\.90|0\.10",
        re.IGNORECASE,
    ),
}


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def normalize_source(source: str) -> str:
    return "\n".join(line.rstrip() for line in source.replace("\r\n", "\n").splitlines()).strip()


def output_record(output: dict[str, Any]) -> dict[str, Any]:
    result: dict[str, Any] = {
        "output_type": output.get("output_type"),
        "name": output.get("name"),
        "text": None,
        "data": {},
        "execution_count": output.get("execution_count"),
    }
    text = output.get("text")
    if isinstance(text, list):
        result["text"] = "".join(str(part) for part in text)
    elif isinstance(text, str):
        result["text"] = text

    data = output.get("data", {})
    if isinstance(data, dict):
        for mime, payload in data.items():
            if isinstance(payload, list):
                rendered = "".join(str(part) for part in payload)
            elif isinstance(payload, str):
                rendered = payload
            else:
                rendered = json.dumps(payload, sort_keys=True, default=str)
            result["data"][mime] = {
                "sha256": sha256_bytes(rendered.encode("utf-8")),
                "characters": len(rendered),
                "text": rendered if mime.startswith("text/") else None,
            }

    traceback = output.get("traceback")
    if isinstance(traceback, list):
        result["traceback"] = "\n".join(str(line) for line in traceback)
    elif traceback is not None:
        result["traceback"] = str(traceback)
    return result


def notebook_record(root: Path, relative_path: str) -> dict[str, Any]:
    path = root / relative_path
    if not path.is_file():
        return {"path": relative_path, "available": False}
    raw = path.read_bytes()
    notebook = json.loads(raw.decode("utf-8"))
    cells: list[dict[str, Any]] = []
    for index, cell in enumerate(notebook.get("cells", [])):
        source_value = cell.get("source", [])
        source = (
            "".join(str(part) for part in source_value)
            if isinstance(source_value, list)
            else str(source_value)
        )
        normalized = normalize_source(source)
        outputs = [
            output_record(output)
            for output in cell.get("outputs", [])
            if isinstance(output, dict)
        ]
        cells.append(
            {
                "index": index,
                "cell_type": cell.get("cell_type"),
                "execution_count": cell.get("execution_count"),
                "source": source,
                "source_sha256": sha256_bytes(normalized.encode("utf-8")),
                "source_characters": len(source),
                "outputs": outputs,
                "metadata": cell.get("metadata", {}),
            }
        )

    all_source = "\n\n".join(
        cell["source"] for cell in cells if cell["cell_type"] == "code"
    )
    categorized_lines = {
        category: [
            {"cell_index": cell["index"], "line": line.strip()}
            for cell in cells
            if cell["cell_type"] == "code"
            for line in cell["source"].splitlines()
            if pattern.search(line)
        ]
        for category, pattern in CATEGORY_PATTERNS.items()
    }
    return {
        "path": relative_path,
        "available": True,
        "file_sha256": sha256_bytes(raw),
        "file_bytes": len(raw),
        "format": notebook.get("nbformat"),
        "format_minor": notebook.get("nbformat_minor"),
        "metadata": notebook.get("metadata", {}),
        "cell_count": len(cells),
        "code_cell_count": sum(cell["cell_type"] == "code" for cell in cells),
        "code_line_count": sum(
            len(cell["source"].splitlines())
            for cell in cells
            if cell["cell_type"] == "code"
        ),
        "markdown_cell_count": sum(cell["cell_type"] == "markdown" for cell in cells),
        "executed_code_cell_count": sum(
            cell["cell_type"] == "code" and cell["execution_count"] is not None
            for cell in cells
        ),
        "code_source_sha256": sha256_bytes(
            normalize_source(all_source).encode("utf-8")
        ),
        "category_lines": categorized_lines,
        "cells": cells,
    }


def code_record(root: Path, relative_path: str) -> dict[str, Any]:
    path = root / relative_path
    if not path.is_file():
        return {"path": relative_path, "available": False}
    raw = path.read_bytes()
    source = raw.decode("utf-8", errors="replace")
    return {
        "path": relative_path,
        "available": True,
        "sha256": sha256_bytes(raw),
        "bytes": len(raw),
        "source_characters": len(source),
        "source": source,
    }


def compare_notebooks(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    available = [record for record in records if record.get("available")]
    comparisons: list[dict[str, Any]] = []
    for left_index, left in enumerate(available):
        for right in available[left_index + 1 :]:
            left_cells = left["cells"]
            right_cells = right["cells"]
            left_code_cells = [
                cell for cell in left_cells if cell["cell_type"] == "code"
            ]
            right_code_cells = [
                cell for cell in right_cells if cell["cell_type"] == "code"
            ]
            left_keys = [cell["source_sha256"] for cell in left_code_cells]
            right_keys = [cell["source_sha256"] for cell in right_code_cells]
            cell_matcher = difflib.SequenceMatcher(
                a=left_keys, b=right_keys, autojunk=False
            )
            cell_operations = []
            for tag, i1, i2, j1, j2 in cell_matcher.get_opcodes():
                if tag != "equal":
                    cell_operations.append(
                        {
                            "operation": tag,
                            "left_cell_indices": [
                                cell["index"] for cell in left_code_cells[i1:i2]
                            ],
                            "right_cell_indices": [
                                cell["index"] for cell in right_code_cells[j1:j2]
                            ],
                            "left_sources": [
                                cell["source"] for cell in left_code_cells[i1:i2]
                            ],
                            "right_sources": [
                                cell["source"] for cell in right_code_cells[j1:j2]
                            ],
                        }
                    )
            left_lines = [
                line
                for cell in left_cells
                if cell["cell_type"] == "code"
                for line in cell["source"].splitlines()
            ]
            right_lines = [
                line
                for cell in right_cells
                if cell["cell_type"] == "code"
                for line in cell["source"].splitlines()
            ]
            line_matcher = difflib.SequenceMatcher(
                a=left_lines, b=right_lines, autojunk=True
            )
            line_operations = []
            for tag, i1, i2, j1, j2 in line_matcher.get_opcodes():
                if tag == "equal":
                    continue
                line_operations.append(
                    {
                        "operation": tag,
                        "left_line_range": [i1 + 1, i2],
                        "right_line_range": [j1 + 1, j2],
                        "left_line_count": i2 - i1,
                        "right_line_count": j2 - j1,
                        "left_excerpt": left_lines[i1 : min(i1 + 6, i2)],
                        "right_excerpt": right_lines[j1 : min(j1 + 6, j2)],
                    }
                )
            shared_cell_count = sum(
                i2 - i1
                for tag, i1, i2, _, _ in cell_matcher.get_opcodes()
                if tag == "equal"
            )
            comparisons.append(
                {
                    "left": left["path"],
                    "right": right["path"],
                    "left_cells": len(left_cells),
                    "right_cells": len(right_cells),
                    "exact_code_cell_sequence_similarity": cell_matcher.ratio(),
                    "shared_exact_code_cell_count": shared_cell_count,
                    "left_code_line_count": len(left_lines),
                    "right_code_line_count": len(right_lines),
                    "code_line_sequence_similarity": line_matcher.ratio(),
                    "changed_cell_blocks": cell_operations,
                    "changed_code_line_blocks": line_operations,
                }
            )
    return comparisons


def runner_comparisons(
    notebooks: list[dict[str, Any]], runners: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    comparisons = []
    for runner in runners:
        if not runner.get("available"):
            continue
        runner_source = normalize_source(runner["source"])
        for notebook in notebooks:
            if not notebook.get("available"):
                continue
            code_cells = [
                cell for cell in notebook["cells"] if cell["cell_type"] == "code"
            ]
            joined = normalize_source(
                "\n\n".join(cell["source"] for cell in code_cells)
            )
            exact_contains = runner_source in joined or joined in runner_source
            matcher = difflib.SequenceMatcher(
                a=runner_source.splitlines(), b=joined.splitlines(), autojunk=True
            )
            comparisons.append(
                {
                    "runner": runner["path"],
                    "notebook": notebook["path"],
                    "runner_sha256": runner["sha256"],
                    "notebook_code_sha256": notebook["code_source_sha256"],
                    "runner_exactly_contained_in_notebook_cells": runner_source
                    in joined,
                    "notebook_code_exactly_contained_in_runner": joined
                    in runner_source,
                    "either_exact_containment": exact_contains,
                    "line_sequence_similarity": matcher.ratio(),
                    "interpretation": (
                        "exact text containment"
                        if exact_contains
                        else "not identical; similarity is diagnostic only"
                    ),
                }
            )
    return comparisons


def build(root: Path, output_dir: Path) -> dict[str, Any]:
    notebooks = [notebook_record(root, path) for path in NOTEBOOKS]
    runners = [code_record(root, path) for path in RUNNERS]
    scripts = [code_record(root, path) for path in RELATED_SCRIPTS]
    comparisons = compare_notebooks(notebooks)
    runner_matches = runner_comparisons(notebooks, runners)
    report = {
        "schema_version": 1,
        "repository_root": str(root),
        "notebooks": notebooks,
        "runners": runners,
        "related_scripts": scripts,
        "notebook_cell_comparisons": comparisons,
        "runner_notebook_comparisons": runner_matches,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "NOTEBOOK_CELL_DIFF.json").write_text(
        json.dumps(report, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    code_hashes = {
        record["path"]: {
            "available": record.get("available", False),
            "file_sha256": record.get("file_sha256", record.get("sha256")),
            "code_source_sha256": record.get("code_source_sha256"),
            "file_bytes": record.get("file_bytes", record.get("bytes")),
            "cell_count": record.get("cell_count"),
        }
        for record in [*notebooks, *runners, *scripts]
    }
    (output_dir / "NOTEBOOK_CODE_HASHES.json").write_text(
        json.dumps(code_hashes, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    lines = [
        "# EXP85 Notebook Lineage Comparison",
        "",
        "## Scope",
        "",
        "The listed notebooks and runners were parsed as JSON/Python source. No",
        "notebook cells were executed. Each cell's full source, execution count,",
        "metadata, and text outputs are retained in `NOTEBOOK_CELL_DIFF.json`;",
        "non-text MIME payloads are represented by size and SHA-256.",
        "",
        "## Notebook inventory",
        "",
        "| Notebook | Available | Cells | Code | Markdown | Executed code | File SHA-256 |",
        "|---|---:|---:|---:|---:|---:|---|",
    ]
    for record in notebooks:
        if not record.get("available"):
            lines.append(f"| `{record['path']}` | No | - | - | - | - | - |")
        else:
            lines.append(
                "| `{}` | Yes | {} | {} | {} | {} | `{}` |".format(
                    record["path"],
                    record["cell_count"],
                    record["code_cell_count"],
                    record["markdown_cell_count"],
                    record["executed_code_cell_count"],
                    record["file_sha256"],
                )
            )
    lines.extend(["", "## Code-source comparisons", ""])
    for comparison in comparisons:
        lines.append(
            "- `{}` vs `{}`: line similarity {:.3f}; exact code-cell sequence "
            "similarity {:.3f}; {} exact code cell(s) shared; {} changed line block(s).".format(
                comparison["left"],
                comparison["right"],
                comparison["code_line_sequence_similarity"],
                comparison["exact_code_cell_sequence_similarity"],
                comparison["shared_exact_code_cell_count"],
                len(comparison["changed_code_line_blocks"]),
            )
        )
    lines.extend(["", "## Runner identity checks", ""])
    for comparison in runner_matches:
        lines.append(
            "- `{}` vs `{}`: {} (line sequence similarity {:.3f}).".format(
                comparison["runner"],
                comparison["notebook"],
                comparison["interpretation"],
                comparison["line_sequence_similarity"],
            )
        )
    lines.extend(
        [
            "",
            "These string comparisons are provenance checks, not proof that a",
            "notebook was executed or that its recorded outputs were produced by",
            "the displayed source. Execution counts and saved outputs are",
            "reported separately and must be corroborated by run evidence.",
            "",
        ]
    )
    (output_dir / "NOTEBOOK_LINEAGE_COMPARISON.md").write_text(
        "\n".join(lines), encoding="utf-8"
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Inspect EXP85 notebook cells, outputs, hashes, and runner overlap."
    )
    parser.add_argument("--repository-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    report = build(args.repository_root.resolve(), args.output_dir.resolve())
    print(
        "Notebook records: {}; comparisons: {}; runner comparisons: {}".format(
            len(report["notebooks"]),
            len(report["notebook_cell_comparisons"]),
            len(report["runner_notebook_comparisons"]),
        )
    )
    for notebook in report["notebooks"]:
        print(
            "{}: available={} cells={} code={} executed={}".format(
                notebook["path"],
                notebook.get("available", False),
                notebook.get("cell_count", 0),
                notebook.get("code_cell_count", 0),
                notebook.get("executed_code_cell_count", 0),
            )
        )


if __name__ == "__main__":
    main()
