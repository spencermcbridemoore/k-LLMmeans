"""Execute first three code cells of offline_experiments_modified.ipynb (imports + run + compare)."""
from __future__ import annotations

import json
import os
import sys

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
NB = os.path.join(REPO_ROOT, "offline_experiments_modified.ipynb")


def main() -> int:
    os.chdir(REPO_ROOT)
    if REPO_ROOT not in sys.path:
        sys.path.insert(0, REPO_ROOT)
    with open(NB, encoding="utf-8") as f:
        nb = json.load(f)
    g: dict = {"__name__": "__main__"}
    code_cells_run = 0
    for i, cell in enumerate(nb["cells"]):
        if cell.get("cell_type") != "code":
            continue
        src = "".join(cell.get("source") or [])
        if not src.strip():
            continue
        print(f"--- executing notebook cell index {i} (code #{code_cells_run}) ---", flush=True)
        exec(compile(src, f"{NB}:cell{i}", "exec"), g, g)
        code_cells_run += 1
        if code_cells_run >= 3:
            break
    print("Done.", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
