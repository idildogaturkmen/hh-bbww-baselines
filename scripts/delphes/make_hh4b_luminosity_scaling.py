#!/usr/bin/env python3

from pathlib import Path
import os
import pandas as pd
import numpy as np

REPO = Path(os.environ["HH4B_REPO"])

OUTDIR = REPO / "outputs/tables/hh4b_luminosity_scaling_2026_07_10"
OUTDIR.mkdir(parents=True, exist_ok=True)

SRC = REPO / "outputs/tables/sm_normalized_hh4b_two_bdt_event_features_safe_resampling_v2_ablation_wp_2026_07_10/resampling_poisson_bootstrap_summary.csv"

df = pd.read_csv(SRC)

# Current nominal best point after focused bootstrap
row = df[df["working_point"].eq("previous_v2_loose_mid")].iloc[0]

L0 = 450.0
z0 = float(row["S_over_sqrtB_mean"])
sob0 = float(row["S_over_B_mean"])
b0 = float(row["B_events_p50"])
s0 = sob0 * b0

lumis = [138.0, 350.0, 450.0, 4000.0]

rows = []
for L in lumis:
    scale = L / L0
    S = s0 * scale
    B = b0 * scale

    rows.append({
        "luminosity_fb": L,
        "signal_events": S,
        "background_events": B,
        "S_over_B": S / B if B > 0 else np.nan,
        "S_over_sqrtB_stat_only": S / np.sqrt(B) if B > 0 else np.nan,
        "S_over_sqrtB_scaled_from_450": z0 * np.sqrt(scale),
        "S_over_sqrt_B_plus_10pctB_syst": S / np.sqrt(B + (0.10 * B)**2) if B > 0 else np.nan,
    })

out = pd.DataFrame(rows)
out.to_csv(OUTDIR / "hh4b_luminosity_scaling.csv", index=False)
(OUTDIR / "hh4b_luminosity_scaling.md").write_text(out.to_markdown(index=False) + "\n")

notes = f"""# HH4b luminosity scaling

Reference working point: previous_v2_loose_mid

Reference luminosity: {L0} fb^-1

Reference BDT cuts:
- BDT_QCD > {row["qcd_threshold"]}
- BDT_top > {row["top_threshold"]}

Reference metrics:
- S/sqrt(B) = {z0}
- S/B = {sob0}
- median B events = {b0}
- inferred S events = {s0}

Scaling assumptions:
- S and B scale linearly with luminosity.
- S/B is constant under simple luminosity scaling.
- S/sqrt(B) scales as sqrt(L).
- The 10% background systematic column is only a crude orientation check.
"""
(OUTDIR / "README.md").write_text(notes)

print(out.to_string(index=False))
print("\nWrote:", OUTDIR)
