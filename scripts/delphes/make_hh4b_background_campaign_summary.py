#!/usr/bin/env python3
from pathlib import Path
import os
import pandas as pd
import re

store = Path(os.environ["HH4B_STORE"])
outdir = Path("outputs/tables/hh4b_background_campaigns_2026_07_08")
outdir.mkdir(parents=True, exist_ok=True)

samples = [
    "qcd_bbbb_presel_100k",
    "zbbbb_presel_100k",
    "ttbar_100k",
]

rows = []

for sample in samples:
    summary_path = store / "metadata" / f"{sample}_summary.txt"
    event_path = store / "parquet" / f"{sample}_merged_event_summary.parquet"
    cand_path = store / "parquet" / f"{sample}_merged_hh4b_candidates.parquet"

    row = {"sample": sample}

    if event_path.exists():
        ev = pd.read_parquet(event_path)
        row["n_generated"] = len(ev)
        row["xsec_pb_median"] = float(ev["event_cross_section_pb"].median())
    else:
        row["n_generated"] = None
        row["xsec_pb_median"] = None

    if cand_path.exists():
        cand = pd.read_parquet(cand_path)
        row["candidate_rows"] = len(cand)
        row["candidate_eff"] = len(cand) / row["n_generated"] if row["n_generated"] else None
        row["effective_candidate_xsec_pb"] = (
            row["xsec_pb_median"] * row["candidate_eff"]
            if row["xsec_pb_median"] is not None and row["candidate_eff"] is not None
            else None
        )
        for col in ["mbb1", "mbb2", "avg_mbb", "delta_mbb", "mhh", "n_selected_bjets"]:
            if col in cand.columns and len(cand):
                row[f"median_{col}"] = float(cand[col].median())
    else:
        row["candidate_rows"] = None
        row["candidate_eff"] = None
        row["effective_candidate_xsec_pb"] = None

    rows.append(row)

df = pd.DataFrame(rows)
csv_path = outdir / "background_campaign_summary_100k.csv"
md_path = outdir / "background_campaign_summary_100k.md"

df.to_csv(csv_path, index=False)
md_path.write_text(df.to_markdown(index=False) + "\n")

print(df.to_string(index=False))
print("\nWrote:", csv_path)
print("Wrote:", md_path)
