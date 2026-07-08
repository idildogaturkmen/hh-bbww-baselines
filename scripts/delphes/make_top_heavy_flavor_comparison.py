#!/usr/bin/env python3
from pathlib import Path
import os
import pandas as pd

store = Path(os.environ["HH4B_STORE"])
outdir = Path("outputs/tables/hh4b_background_campaigns_2026_07_08")
outdir.mkdir(parents=True, exist_ok=True)

samples = [
    ("ttbar_100k", "inclusive ttbar"),
    ("ttbb_50k", "ttbb enriched diagnostic"),
]

rows = []

for tag, label in samples:
    ev_path = store / "parquet" / f"{tag}_merged_event_summary.parquet"
    cand_path = store / "parquet" / f"{tag}_merged_hh4b_candidates.parquet"

    ev = pd.read_parquet(ev_path)
    cand = pd.read_parquet(cand_path)

    xsec = float(ev["event_cross_section_pb"].median())
    n_gen = len(ev)
    n_cand = len(cand)
    eff = n_cand / n_gen
    eff_xsec = xsec * eff

    row = {
        "sample": tag,
        "label": label,
        "n_generated": n_gen,
        "xsec_pb_median": xsec,
        "candidate_rows": n_cand,
        "candidate_eff": eff,
        "effective_candidate_xsec_pb": eff_xsec,
    }

    for col in ["mbb1", "mbb2", "avg_mbb", "delta_mbb", "mhh", "n_selected_bjets"]:
        if col in cand.columns and len(cand):
            row[f"median_{col}"] = float(cand[col].median())

    rows.append(row)

df = pd.DataFrame(rows)

csv_path = outdir / "top_heavy_flavor_comparison.csv"
md_path = outdir / "top_heavy_flavor_comparison.md"

df.to_csv(csv_path, index=False)
md_path.write_text(df.to_markdown(index=False) + "\n")

print(df.to_string(index=False))
print("\nWrote:", csv_path)
print("Wrote:", md_path)
