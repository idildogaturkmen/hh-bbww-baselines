'''
Audit the signal normalization for HH4b analysis, using ggF+VBF signal and nominal backgrounds.
This script is intended to be run in the `scripts/delphes` directory, with the HH4b parquet store available at the path specified by the HH4B_STORE environment variable.
The script will produce a table in the outputs/tables/hh4b_signal_audit_YYYY_MM_DD directory        
'''
from pathlib import Path
import os
import numpy as np
import pandas as pd

store = Path(os.environ["HH4B_STORE"])
outdir = Path("outputs/tables/hh4b_signal_audit_2026_07_08")
outdir.mkdir(parents=True, exist_ok=True)

samples = [
    (
        "ggF HH4b",
        "HH4b_ggf_hh4b_10k_transfer_cluster_84730223_merged_10k_event_summary.parquet",
        "HH4b_ggf_hh4b_10k_transfer_cluster_84730223_merged_10k_hh4b_candidates.parquet",
    ),
    (
        "VBF HH4b",
        "HH4b_vbf_hh4b_10k_transfer_cluster_3473183_merged_10k_event_summary.parquet",
        "HH4b_vbf_hh4b_10k_transfer_cluster_3473183_merged_10k_hh4b_candidates.parquet",
    ),
]

rows = []

for label, ev_name, cand_name in samples:
    ev_path = store / "parquet" / ev_name
    cand_path = store / "parquet" / cand_name

    ev = pd.read_parquet(ev_path)
    cand = pd.read_parquet(cand_path)

    xsec_pb = float(ev["event_cross_section_pb"].median())
    n_generated = len(ev)
    n_candidates = len(cand)
    cand_eff = n_candidates / n_generated
    eff_xsec_pb = xsec_pb * cand_eff

    rows.append({
        "sample": label,
        "event_file": ev_name,
        "candidate_file": cand_name,
        "n_generated": n_generated,
        "candidate_rows": n_candidates,
        "generator_xsec_pb_median": xsec_pb,
        "generator_xsec_fb_median": xsec_pb * 1000.0,
        "candidate_efficiency": cand_eff,
        "effective_candidate_xsec_pb": eff_xsec_pb,
        "effective_candidate_xsec_fb": eff_xsec_pb * 1000.0,
        "expected_candidate_events_450fb": eff_xsec_pb * 1000.0 * 450.0,
        "median_mbb1": float(cand["mbb1"].median()),
        "median_mbb2": float(cand["mbb2"].median()),
        "median_avg_mbb": float(cand["avg_mbb"].median()),
        "median_mhh": float(cand["mhh"].median()),
    })

df = pd.DataFrame(rows)

csv_path = outdir / "signal_normalization_audit.csv"
md_path = outdir / "signal_normalization_audit.md"

df.to_csv(csv_path, index=False)
md_path.write_text(df.to_markdown(index=False) + "\n")

print(df.to_string(index=False))
print("\nWrote:", csv_path)
print("Wrote:", md_path)
