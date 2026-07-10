#!/usr/bin/env python3

import os
from pathlib import Path
import pandas as pd

STORE = Path(os.environ["HH4B_STORE"])
REPO = Path(os.environ["HH4B_REPO"])
V2 = STORE / "parquet_v2_nominal"
OUTDIR = REPO / "outputs/tables/hh4b_spanet_readiness_v2_2026_07_10"
OUTDIR.mkdir(parents=True, exist_ok=True)

paths = sorted(V2.glob("*_v2_hh4b_candidates.parquet"))

needed_fourjet = []
for i in range(1, 5):
    needed_fourjet += [f"j{i}_pt", f"j{i}_eta", f"j{i}_phi", f"j{i}_mass"]

optional_fourjet = []
for i in range(1, 5):
    optional_fourjet += [f"j{i}_btag", f"j{i}_bjet_rank", f"j{i}_flavor"]

event_features = [
    "n_selected_jets", "n_selected_bjets", "n_extra_selected_jets",
    "n_extra_selected_bjets", "ht_selected_jets", "ht_selected_bjets",
    "ht_candidate_jets",
]

assignment_diagnostics = [
    "pairing", "pairing_combo_bjet_ranks", "pairing_score_125_125",
    "higgs_ordering",
]

rows = []
column_rows = []

for path in paths:
    df = pd.read_parquet(path)
    sample = path.name.replace("_v2_hh4b_candidates.parquet", "")

    cols = set(df.columns)
    missing_fourjet = [c for c in needed_fourjet if c not in cols]
    present_optional = [c for c in optional_fourjet if c in cols]
    present_event = [c for c in event_features if c in cols]
    present_assignment = [c for c in assignment_diagnostics if c in cols]

    rows.append({
        "sample": sample,
        "rows": len(df),
        "has_fourjet_kinematics_for_toy_spanet": len(missing_fourjet) == 0,
        "missing_fourjet_columns": ", ".join(missing_fourjet),
        "n_optional_jet_columns_present": len(present_optional),
        "optional_jet_columns_present": ", ".join(present_optional),
        "n_event_features_present": len(present_event),
        "event_features_present": ", ".join(present_event),
        "assignment_diagnostic_columns_present": ", ".join(present_assignment),
        "has_truth_flavor_columns": any(c.endswith("_flavor") for c in cols),
    })

    for c in sorted(df.columns):
        cl = c.lower()
        if any(key in cl for key in ["j1_", "j2_", "j3_", "j4_", "pair", "truth", "flavor", "rank", "higgs"]):
            column_rows.append({"sample": sample, "column": c})

summary = pd.DataFrame(rows)
columns = pd.DataFrame(column_rows)

summary.to_csv(OUTDIR / "spanet_readiness_summary.csv", index=False)
columns.to_csv(OUTDIR / "spanet_relevant_columns.csv", index=False)

(OUTDIR / "spanet_readiness_summary.md").write_text(summary.to_markdown(index=False) + "\n")
(OUTDIR / "spanet_relevant_columns.md").write_text(columns.to_markdown(index=False) + "\n")

notes = """# SPA-Net readiness notes

This audit checks whether the current v2 candidate parquets are enough for a toy SPA-Net-style model.

Interpretation:
- If j1/j2/j3/j4 pt/eta/phi/mass are present, a toy 4-candidate-jet assignment/classification model is possible.
- A full SPA-Net model should use leading-N selected jets, not only the four already selected candidate jets.
- Full SPA-Net also needs truth assignment labels mapping reconstructed jets to H1/H2/background.
- If only candidate jets are stored, the next step is to build a ROOT-to-SPA-Net dataset writer using Jet branches and generator-level H→bb truth daughters.
"""
(OUTDIR / "README.md").write_text(notes)

print("\n=== SPA-Net readiness summary ===")
print(summary.to_string(index=False))

print("\nWrote:", OUTDIR)
