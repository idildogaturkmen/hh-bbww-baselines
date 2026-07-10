#!/usr/bin/env python3

import os
from pathlib import Path
import uproot
import pandas as pd

STORE = Path(os.environ["HH4B_STORE"])
REPO = Path(os.environ["HH4B_REPO"])
OUTDIR = REPO / "outputs/tables/hh4b_spanet_root_branch_audit_2026_07_10"
OUTDIR.mkdir(parents=True, exist_ok=True)

root_files = sorted((STORE / "root").glob("*HH4b*delphes.root"))[:3]
if not root_files:
    root_files = sorted((STORE / "root").glob("*delphes.root"))[:3]

rows = []

for path in root_files:
    with uproot.open(path) as f:
        tree = f["Delphes"]
        branches = list(tree.keys())

    for b in branches:
        bl = b.lower()
        if (
            bl.startswith("jet")
            or bl.startswith("particle")
            or "btag" in bl
            or "flavor" in bl
            or "pid" in bl
            or "status" in bl
            or "mother" in bl
            or "daughter" in bl
        ):
            rows.append({"root_file": path.name, "branch": b})

df = pd.DataFrame(rows)
df.to_csv(OUTDIR / "spanet_relevant_root_branches.csv", index=False)
(OUTDIR / "spanet_relevant_root_branches.md").write_text(df.to_markdown(index=False) + "\n")

print(df.to_string(index=False))
print("\nWrote:", OUTDIR)
