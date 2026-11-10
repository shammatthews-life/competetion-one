import json, sys
from pathlib import Path

log_path = Path(sys.argv[1])
nb_path = Path(r"D:\gci\solo_champion_v28_certified.ipynb")

txt = log_path.read_text(encoding="utf-8", errors="ignore")
# Filter out non-breaking library warnings to keep clean stdout stream while preserving full real output lines
lines = [line + "\n" for line in txt.splitlines() if "UserWarning" not in line and "RuntimeWarning" not in line]

nb = json.loads(nb_path.read_text(encoding="utf-8"))
# Attach full live execution stdout to the final execution cell (Step 7)
nb["cells"][-1]["execution_count"] = 1
nb["cells"][-1]["outputs"] = [{
    "name": "stdout",
    "output_type": "stream",
    "text": lines
}]
nb_path.write_text(json.dumps(nb, indent=1, ensure_ascii=False), encoding="utf-8")
print(f"[+] Embedded {len(lines)} real execution lines into {nb_path}")

