#!/usr/bin/env python3
from pathlib import Path
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

store = Path(os.environ["HH4B_STORE"])
outdir = store / "plots" / "qcd_iht_importance_sampling_2026_07_08"
outdir.mkdir(parents=True, exist_ok=True)

inclusive_event = store / "parquet" / "qcd_bbbb_presel_100k_merged_event_summary.parquet"
inclusive_cand = store / "parquet" / "qcd_bbbb_presel_100k_merged_hh4b_candidates.parquet"

slices = [
    "qcd_bbbb_iht100to200_2000",
    "qcd_bbbb_iht200to400_2000",
    "qcd_bbbb_iht400to600_2000",
    "qcd_bbbb_iht600plus_2000",
]

def load_candidate_sample(tag):
    ev_path = store / "parquet" / f"{tag}_event_summary.parquet"
    cand_path = store / "parquet" / f"{tag}_hh4b_candidates.parquet"

    ev = pd.read_parquet(ev_path)
    cand = pd.read_parquet(cand_path).copy()

    xsec = float(ev["event_cross_section_pb"].median())
    n_gen = len(ev)
    w = xsec / n_gen

    cand["sample"] = tag
    cand["event_weight_pb"] = w
    cand["r_hh"] = np.sqrt((cand["mbb1"] - 125.0)**2 + (cand["mbb2"] - 120.0)**2)

    return ev, cand, xsec, n_gen, w

# Inclusive QCD 100k
inc_ev = pd.read_parquet(inclusive_event)
inc = pd.read_parquet(inclusive_cand).copy()
inc_xsec = float(inc_ev["event_cross_section_pb"].median())
inc_n = len(inc_ev)
inc_w = inc_xsec / inc_n
inc["sample"] = "inclusive_qcd_100k"
inc["event_weight_pb"] = inc_w
inc["r_hh"] = np.sqrt((inc["mbb1"] - 125.0)**2 + (inc["mbb2"] - 120.0)**2)

# HT-sliced QCD
slice_frames = []
summary_rows = []
for tag in slices:
    ev, cand, xsec, n_gen, w = load_candidate_sample(tag)
    slice_frames.append(cand)
    summary_rows.append({
        "tag": tag,
        "n_generated": n_gen,
        "xsec_pb": xsec,
        "candidate_rows": len(cand),
        "event_weight_pb": w,
        "candidate_xsec_pb": w * len(cand),
        "median_mhh": cand["mhh"].median() if len(cand) else np.nan,
        "median_r_hh": cand["r_hh"].median() if len(cand) else np.nan,
    })

sliced = pd.concat(slice_frames, ignore_index=True)
summary = pd.DataFrame(summary_rows)

summary.loc[len(summary)] = {
    "tag": "SUM_iht_slices",
    "n_generated": summary["n_generated"].sum(),
    "xsec_pb": summary["xsec_pb"].sum(),
    "candidate_rows": summary["candidate_rows"].sum(),
    "event_weight_pb": np.nan,
    "candidate_xsec_pb": summary["candidate_xsec_pb"].sum(),
    "median_mhh": np.nan,
    "median_r_hh": np.nan,
}

summary.loc[len(summary)] = {
    "tag": "inclusive_qcd_100k",
    "n_generated": inc_n,
    "xsec_pb": inc_xsec,
    "candidate_rows": len(inc),
    "event_weight_pb": inc_w,
    "candidate_xsec_pb": inc_w * len(inc),
    "median_mhh": inc["mhh"].median(),
    "median_r_hh": inc["r_hh"].median(),
}

out_csv = store / "metadata" / "qcd_iht_importance_sampling_summary_2026_07_08.csv"
summary.to_csv(out_csv, index=False)
print(summary.to_string(index=False))
print("Wrote:", out_csv)

def save(name):
    out = outdir / name
    plt.tight_layout()
    plt.savefig(out, dpi=180)
    plt.close()
    print("Wrote:", out)

def plot_weighted(var, bins, xlabel, filename):
    plt.figure()
    plt.hist(
        inc[var],
        bins=bins,
        weights=inc["event_weight_pb"],
        histtype="step",
        label="Inclusive QCD 100k",
    )
    plt.hist(
        sliced[var],
        bins=bins,
        weights=sliced["event_weight_pb"],
        histtype="step",
        label="HT-sliced QCD, 2k/slice",
    )
    plt.xlabel(xlabel)
    plt.ylabel("Candidate-level cross section per bin [pb]")
    plt.title(f"Weighted QCD comparison: {xlabel}")
    plt.legend()
    save(filename)

def plot_normalized(var, bins, xlabel, filename):
    plt.figure()
    plt.hist(
        inc[var],
        bins=bins,
        weights=inc["event_weight_pb"],
        histtype="step",
        density=True,
        label="Inclusive QCD 100k",
    )
    plt.hist(
        sliced[var],
        bins=bins,
        weights=sliced["event_weight_pb"],
        histtype="step",
        density=True,
        label="HT-sliced QCD, 2k/slice",
    )
    plt.xlabel(xlabel)
    plt.ylabel("Normalized weighted candidates")
    plt.title(f"Shape comparison: {xlabel}")
    plt.legend()
    save(filename)

plot_weighted("mhh", np.linspace(0, 1600, 65), "mHH [GeV]", "weighted_mhh_inclusive_vs_iht_slices.png")
plot_normalized("mhh", np.linspace(0, 1600, 65), "mHH [GeV]", "normalized_mhh_inclusive_vs_iht_slices.png")

plot_weighted("avg_mbb", np.linspace(0, 300, 61), "average m_bb [GeV]", "weighted_avg_mbb_inclusive_vs_iht_slices.png")
plot_normalized("avg_mbb", np.linspace(0, 300, 61), "average m_bb [GeV]", "normalized_avg_mbb_inclusive_vs_iht_slices.png")

plot_weighted("r_hh", np.linspace(0, 300, 61), "R_HH-like distance [GeV]", "weighted_rhh_inclusive_vs_iht_slices.png")
plot_normalized("r_hh", np.linspace(0, 300, 61), "R_HH-like distance [GeV]", "normalized_rhh_inclusive_vs_iht_slices.png")

# Contribution by HT slice to mHH
plt.figure()
for tag, sub in sliced.groupby("sample"):
    plt.hist(
        sub["mhh"],
        bins=np.linspace(0, 1600, 65),
        weights=sub["event_weight_pb"],
        histtype="step",
        label=tag.replace("qcd_bbbb_", "").replace("_2000", ""),
    )
plt.xlabel("mHH [GeV]")
plt.ylabel("Candidate-level cross section per bin [pb]")
plt.title("Weighted mHH contributions by HT slice")
plt.legend(fontsize=7)
save("weighted_mhh_by_iht_slice.png")
