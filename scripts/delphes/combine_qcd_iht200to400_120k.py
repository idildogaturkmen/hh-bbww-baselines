#!/usr/bin/env python3

import os
from pathlib import Path
import pandas as pd
import numpy as np

STORE = Path(os.environ["HH4B_STORE"])
REPO = Path(os.environ["HH4B_REPO"])

OUTDIR = REPO / "outputs/tables/qcd_iht200to400_combined120k_2026_07_09"
OUTDIR.mkdir(parents=True, exist_ok=True)

ORIG_TAG = "qcd_bbbb_iht200to400_20000"
EXTRA_TAG = "qcd_bbbb_iht200to400_extra100k"
COMBINED_TAG = "qcd_bbbb_iht200to400_combined120k"

paths = {
    "orig_event": STORE / "parquet" / f"{ORIG_TAG}_event_summary.parquet",
    "orig_cand": STORE / "parquet" / f"{ORIG_TAG}_hh4b_candidates.parquet",
    "extra_event": STORE / "parquet" / f"{EXTRA_TAG}_merged_event_summary.parquet",
    "extra_cand": STORE / "parquet" / f"{EXTRA_TAG}_merged_hh4b_candidates.parquet",
}

for name, path in paths.items():
    if not path.exists():
        raise FileNotFoundError(f"Missing {name}: {path}")

orig_ev = pd.read_parquet(paths["orig_event"]).copy()
orig_ca = pd.read_parquet(paths["orig_cand"]).copy()
extra_ev = pd.read_parquet(paths["extra_event"]).copy()
extra_ca = pd.read_parquet(paths["extra_cand"]).copy()

orig_ev["source_campaign"] = ORIG_TAG
orig_ca["source_campaign"] = ORIG_TAG
extra_ev["source_campaign"] = EXTRA_TAG
extra_ca["source_campaign"] = EXTRA_TAG

events = pd.concat([orig_ev, extra_ev], ignore_index=True)
cands = pd.concat([orig_ca, extra_ca], ignore_index=True)

n_orig = len(orig_ev)
n_extra = len(extra_ev)
n_total = len(events)

xsec_orig = float(orig_ev["event_cross_section_pb"].median())
xsec_extra = float(extra_ev["event_cross_section_pb"].median())

# For same phase-space slice, combine by using event-count weighted average xsec estimate.
xsec_combined = (xsec_orig * n_orig + xsec_extra * n_extra) / n_total

event_out = STORE / "parquet" / f"{COMBINED_TAG}_merged_event_summary.parquet"
cand_out = STORE / "parquet" / f"{COMBINED_TAG}_merged_hh4b_candidates.parquet"

events.to_parquet(event_out, index=False)
cands.to_parquet(cand_out, index=False)

summary = pd.DataFrame([{
    "combined_tag": COMBINED_TAG,
    "original_tag": ORIG_TAG,
    "extra_tag": EXTRA_TAG,
    "n_original": n_orig,
    "n_extra": n_extra,
    "n_total": n_total,
    "candidate_rows_original": len(orig_ca),
    "candidate_rows_extra": len(extra_ca),
    "candidate_rows_total": len(cands),
    "xsec_original_pb": xsec_orig,
    "xsec_extra_pb": xsec_extra,
    "xsec_combined_pb": xsec_combined,
    "candidate_eff_total": len(cands) / n_total,
    "effective_candidate_xsec_pb": xsec_combined * len(cands) / n_total,
    "median_mbb1": cands["mbb1"].median(),
    "median_mbb2": cands["mbb2"].median(),
    "median_avg_mbb": cands["avg_mbb"].median(),
    "median_delta_mbb": cands["delta_mbb"].median(),
    "median_mhh": cands["mhh"].median(),
    "event_output": str(event_out),
    "candidate_output": str(cand_out),
}])

summary.to_csv(OUTDIR / "qcd_iht200to400_combined120k_summary.csv", index=False)
(OUTDIR / "qcd_iht200to400_combined120k_summary.md").write_text(summary.to_markdown(index=False) + "\n")

metadata_out = STORE / "metadata" / f"{COMBINED_TAG}_summary.csv"
summary.to_csv(metadata_out, index=False)

print("\n=== Combined QCD iht200to400 summary ===")
print(summary.to_string(index=False))
print(f"\nWrote:")
print(event_out)
print(cand_out)
print(metadata_out)
print(OUTDIR)
