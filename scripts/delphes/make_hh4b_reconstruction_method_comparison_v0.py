#!/usr/bin/env python3

import os
from pathlib import Path
import numpy as np
import pandas as pd

REPO = Path(os.environ["HH4B_REPO"])
STORE = Path(os.environ["HH4B_STORE"])
PARQ = STORE / "parquet"

OUTDIR = REPO / "outputs/tables/hh4b_reconstruction_method_comparison_2026_07_09"
OUTDIR.mkdir(parents=True, exist_ok=True)

SAFE_AUC = REPO / "outputs/tables/sm_normalized_hh4b_two_bdt_event_features_safe_2026_07_09/two_bdt_event_features_auc_summary_safe.csv"
RESAMPLE = REPO / "outputs/tables/sm_normalized_hh4b_two_bdt_event_features_safe_resampling_2026_07_09/resampling_nominal_seed_summary.csv"


def find_one(patterns):
    matches = []
    for pat in patterns:
        matches.extend(sorted(PARQ.glob(pat)))
    matches = [p for p in matches if "test" not in p.name.lower()]
    if not matches:
        raise FileNotFoundError(f"No files found for patterns: {patterns}")
    return matches[-1]


def central68_half_width(x):
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if len(x) == 0:
        return np.nan
    q16, q84 = np.quantile(x, [0.16, 0.84])
    return 0.5 * (q84 - q16)


def mass_metrics(df):
    return {
        "median_mbb1": float(df["mbb1"].median()),
        "median_mbb2": float(df["mbb2"].median()),
        "median_avg_mbb": float(df["avg_mbb"].median()),
        "avg_mbb_central68_half_width": central68_half_width(df["avg_mbb"]),
        "median_abs_avg_mbb_minus_125": float(np.median(np.abs(df["avg_mbb"] - 125.0))),
        "median_mhh": float(df["mhh"].median()),
    }


signal_samples = {
    "ggF_HH4b": {
        "n_generated": 10000,
        "path": find_one(["HH4b_ggf_hh4b_10k*merged*hh4b_candidates.parquet"]),
    },
    "VBF_HH4b": {
        "n_generated": 10000,
        "path": find_one(["HH4b_vbf_hh4b_10k*merged*hh4b_candidates.parquet"]),
    },
}

sig_frames = []
per_signal_rows = []

for name, cfg in signal_samples.items():
    df = pd.read_parquet(cfg["path"])
    sig_frames.append(df.assign(signal_sample=name))

    m = mass_metrics(df)
    per_signal_rows.append({
        "method": "strict_4b_closest_mass",
        "signal_sample": name,
        "n_generated": cfg["n_generated"],
        "candidate_rows": len(df),
        "candidate_efficiency": len(df) / cfg["n_generated"],
        **m,
        "correct_pairing_accuracy": np.nan,
        "correct_pairing_note": "requires ROOT-level truth matching to Higgs-daughter b quarks",
    })

sig_all = pd.concat(sig_frames, ignore_index=True)
m_all = mass_metrics(sig_all)

auc = pd.read_csv(SAFE_AUC)
res = pd.read_csv(RESAMPLE)
nominal = res[res["working_point"].eq("nominal_balanced")].iloc[0]

qcd_auc = float(auc.loc[auc["classifier"].str.contains("QCD"), "physics_weighted_auc"].iloc[0])
top_auc = float(auc.loc[auc["classifier"].str.contains("top"), "physics_weighted_auc"].iloc[0])

summary_rows = [
    {
        "method": "strict_4b_closest_mass",
        "status": "current nominal v1 baseline",
        "jet_selection": "require >=4 selected b-tagged jets; use leading four selected b-tagged jets",
        "pairing": "choose pairing minimizing |mbb1-125| + |mbb2-125|",
        "candidate_eff_ggF": per_signal_rows[0]["candidate_efficiency"],
        "candidate_eff_VBF": per_signal_rows[1]["candidate_efficiency"],
        "candidate_eff_signal_average": len(sig_all) / 20000,
        "correct_pairing_accuracy": np.nan,
        "correct_pairing_note": "pending ROOT-level truth matching",
        **m_all,
        "BDT_QCD_weighted_AUC": qcd_auc,
        "BDT_top_weighted_AUC": top_auc,
        "nominal_BDT_working_point": "BDT_QCD > 0.825 and BDT_top > 0.525",
        "nominal_S_over_sqrtB_mean": float(nominal["S_over_sqrtB_mean"]),
        "nominal_S_over_sqrtB_seed_std": float(nominal["S_over_sqrtB_std_seed"]),
        "nominal_S_over_B_mean": float(nominal["S_over_B_mean"]),
    },
    {
        "method": "CMS_inspired_top_four_btag_priority_mass_balance",
        "status": "next to implement",
        "jet_selection": "require >=4 selected jets; choose four highest b-tag-priority jets, then pT; no hard 4-btag requirement",
        "pairing": "mass-balance pairing minimizing |mbb1-mbb2|",
        "candidate_eff_ggF": np.nan,
        "candidate_eff_VBF": np.nan,
        "candidate_eff_signal_average": np.nan,
        "correct_pairing_accuracy": np.nan,
        "correct_pairing_note": "will be computed with truth matching",
        "median_mbb1": np.nan,
        "median_mbb2": np.nan,
        "median_avg_mbb": np.nan,
        "avg_mbb_central68_half_width": np.nan,
        "median_abs_avg_mbb_minus_125": np.nan,
        "median_mhh": np.nan,
        "BDT_QCD_weighted_AUC": np.nan,
        "BDT_top_weighted_AUC": np.nan,
        "nominal_BDT_working_point": "pending",
        "nominal_S_over_sqrtB_mean": np.nan,
        "nominal_S_over_sqrtB_seed_std": np.nan,
        "nominal_S_over_B_mean": np.nan,
    },
    {
        "method": "v2_all_bjet_pairing_event_features",
        "status": "next to implement / compare",
        "jet_selection": "use all selected b-tagged jets up to leading 8 for candidate pairing",
        "pairing": "closest-Higgs-mass pairing among candidate b jets",
        "candidate_eff_ggF": np.nan,
        "candidate_eff_VBF": np.nan,
        "candidate_eff_signal_average": np.nan,
        "correct_pairing_accuracy": np.nan,
        "correct_pairing_note": "will be computed with truth matching",
        "median_mbb1": np.nan,
        "median_mbb2": np.nan,
        "median_avg_mbb": np.nan,
        "avg_mbb_central68_half_width": np.nan,
        "median_abs_avg_mbb_minus_125": np.nan,
        "median_mhh": np.nan,
        "BDT_QCD_weighted_AUC": np.nan,
        "BDT_top_weighted_AUC": np.nan,
        "nominal_BDT_working_point": "pending",
        "nominal_S_over_sqrtB_mean": np.nan,
        "nominal_S_over_sqrtB_seed_std": np.nan,
        "nominal_S_over_B_mean": np.nan,
    },
    {
        "method": "future_SPANet_assignment",
        "status": "future",
        "jet_selection": "set/graph assignment model",
        "pairing": "learned permutationless assignment",
        "candidate_eff_ggF": np.nan,
        "candidate_eff_VBF": np.nan,
        "candidate_eff_signal_average": np.nan,
        "correct_pairing_accuracy": np.nan,
        "correct_pairing_note": "future ML assignment benchmark",
        "median_mbb1": np.nan,
        "median_mbb2": np.nan,
        "median_avg_mbb": np.nan,
        "avg_mbb_central68_half_width": np.nan,
        "median_abs_avg_mbb_minus_125": np.nan,
        "median_mhh": np.nan,
        "BDT_QCD_weighted_AUC": np.nan,
        "BDT_top_weighted_AUC": np.nan,
        "nominal_BDT_working_point": "pending",
        "nominal_S_over_sqrtB_mean": np.nan,
        "nominal_S_over_sqrtB_seed_std": np.nan,
        "nominal_S_over_B_mean": np.nan,
    },
]

summary = pd.DataFrame(summary_rows)
per_signal = pd.DataFrame(per_signal_rows)

summary.to_csv(OUTDIR / "reconstruction_method_comparison_v0.csv", index=False)
per_signal.to_csv(OUTDIR / "strict_4b_signal_mass_efficiency_breakdown.csv", index=False)

(OUTDIR / "reconstruction_method_comparison_v0.md").write_text(summary.to_markdown(index=False) + "\n")
(OUTDIR / "strict_4b_signal_mass_efficiency_breakdown.md").write_text(per_signal.to_markdown(index=False) + "\n")

readme = """# HH4b reconstruction method comparison

This table tracks reconstruction methods for the HH->4b Delphes ML benchmark.

Current filled row:
- strict_4b_closest_mass: current v1 baseline used for the safe event-feature two-BDT.

Pending rows:
- CMS-inspired top-four-btag-priority mass-balance reconstruction.
- v2 all-bjet pairing with event features.
- future SPA-Net assignment model.

Correct pairing accuracy requires ROOT-level truth matching to Higgs-daughter b quarks and is intentionally left blank until that validation is implemented.
"""
(OUTDIR / "README.md").write_text(readme)

print("\n=== Reconstruction method comparison v0 ===")
print(summary.to_string(index=False))
print(f"\nWrote outputs to: {OUTDIR}")
