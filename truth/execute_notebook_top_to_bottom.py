import io
import json
import sys
import contextlib
from pathlib import Path

class TeeIO:
    def __init__(self, orig, buf):
        self.orig = orig
        self.buf = buf
    def write(self, s):
        self.buf.write(s)
        self.orig.write(s)
    def flush(self):
        self.buf.flush()
        self.orig.flush()

def run_notebook(nb_path: Path):
    nb = json.loads(nb_path.read_text(encoding="utf-8"))
    ns = {"__name__": "__main__", "__file__": str(nb_path)}
    exec_idx = 0
    for idx, cell in enumerate(nb.get("cells", [])):
        if cell.get("cell_type") != "code":
            continue
        exec_idx += 1
        src = "".join(cell.get("source", []))
        print(f"\n>>> EXECUTING CELL {idx} (exec={exec_idx}) <<<", flush=True)
        buf = io.StringIO()
        tee_out = TeeIO(sys.stdout, buf)
        with contextlib.redirect_stdout(tee_out):
            exec(compile(src, f"{nb_path.name}:cell{idx}", "exec"), ns)
        out_text = buf.getvalue()
        cell["execution_count"] = exec_idx
        if out_text:
            lines = [ln + "\n" for ln in out_text.splitlines()]
            cell["outputs"] = [{"name": "stdout", "output_type": "stream", "text": lines}]
        else:
            cell["outputs"] = []
    nb_path.write_text(json.dumps(nb, indent=1, ensure_ascii=False), encoding="utf-8")
    print(f"\n[+] Saved fully executed notebook ({exec_idx} code cells executed): {nb_path}", flush=True)

if __name__ == "__main__":
    run_notebook(Path(sys.argv[1]))

