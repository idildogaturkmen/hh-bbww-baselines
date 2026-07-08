#!/usr/bin/env python3
from pathlib import Path
import os
import pandas as pd

release = Path("data_release/hh4b_delphes_analysis_v0_2026_07_08")

rows = []
for path in sorted(release.rglob("*")):
    if path.is_file():
        rows.append({
            "path": str(path),
            "size_MB": path.stat().st_size / 1e6,
            "suffix": path.suffix,
        })

df = pd.DataFrame(rows)
out_csv = release / "dataset_inventory.csv"
df.to_csv(out_csv, index=False)

out_md = release / "dataset_inventory.md"
out_md.write_text(df.to_markdown(index=False) + "\n")

print(df.to_string(index=False))
print("\nWrote:", out_csv)
print("Wrote:", out_md)
print("Total size MB:", df["size_MB"].sum() if len(df) else 0)
