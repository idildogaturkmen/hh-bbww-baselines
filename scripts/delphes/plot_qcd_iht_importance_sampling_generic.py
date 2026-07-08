#!/usr/bin/env python3
from pathlib import Path
import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

if len(sys.argv) < 2:
    raise SystemExit("Usage: plot_qcd_iht_importance_sampling_generic.py N_EVENTS_PER_SLICE")

n_events = int(sys.argv[1])

store = Path(os.environ["HH4B_STORE"])
outdir = store / "plots" / f"qcd_iht_importance_sampling_{n_events}_2026_07_08"
outdir.mkdir(parents=True, exist_ok=True)

meta_path = store / "metadata" / f"qcd_bbbb_iht_slice_scan_{n_events}.csv"
meta = pd.read_csv(meta_path)

inclusive_event = store / "parquet" / "qcd_bbbb_presel_100k_merged_event_summary.parquet"
inclusive_cand = store / "parquet" / "qcd_bbbb_presel_100k_merged_hh4b_candidates.parquet"

inc_ev = pd.read_parquet(inclusive_event)
inc = pd.read_parquet(inclusive_cand).copy()
inc_xsec = float(inc_ev["event_cross_section_pb"].median())
inc_w = inc_xsec / len(inc_ev)
inc["event_weight_pb"] = inc_w
inc["r_hh"] = np.sqrt((inc["mbb1"] - 125.0)**2 + (inc["mbb2"] - 120.0)**2)

frames = []
rows = []

for _, row in meta.iterrows():
    tag = row["tag"]
    ev_path = store / "parquet" / f"{tag}_event_summary.parquet"
    cand_path = store / "parquet" / f"{tag}_hh4b_candidates.parquet"

    ev = pd.read_parquet(ev_path)
    cand = pd.read_parquet(cand_path).copy()

    xsec = float(row["xsec_pb"])
    n_gen = int(row["n_generated"])
    w = xsec / n_gen

    cand["sample"] = tag
    cand["event_weight_pb"] = w
    cand["r_hh"] = np.sqrt((cand["mbb1"] - 125.0)**2 + (cand["mbb2"] - 120.0)**2)

    frames.append(cand)

    rows.append({
        "tag": tag,
        "slice_label": row["slice_label"],
        "n_generated": n_gen,
        "xsec_pb": xsec,
        "candidate_rows": len(cand),
        "candidate_eff": len(cand) / n_gen,
        "effective_candidate_xsec_pb": w * len(cand),
        "event_weight_pb": w,
        "median_mhh": cand["mhh"].median(),
        "median_r_hh": cand["r_hh"].median(),
    })

sliced = pd.concat(frames, ignore_index=True)
summary = pd.DataFrame(rows)

summary.loc[len(summary)] = {
    "tag": "SUM_iht_slices",
    "slice_label": "sum",
    "n_generated": summary["n_generated"].sum(),
    "xsec_pb": summary["xsec_pb"].sum(),
    "candidate_rows": summary["candidate_rows"].sum(),
    "candidate_eff": np.nan,
    "effective_candidate_xsec_pb": summary["effective_candidate_xsec_pb"].sum(),
    "event_weight_pb": np.nan,
    "median_mhh": np.nan,
    "median_r_hh": np.nan,
}

summary.loc[len(summary)] = {
    "tag": "inclusive_qcd_100k",
    "slice_label": "inclusive_reference",
    "n_generated": len(inc_ev),
    "xsec_pb": inc_xsec,
    "candidate_rows": len(inc),
    "candidate_eff": len(inc) / len(inc_ev),
    "effective_candidate_xsec_pb": inc_w * len(inc),
    "event_weight_pb": inc_w,
    "median_mhh": inc["mhh"].median(),
    "median_r_hh": inc["r_hh"].median(),
}

out_csv = store / "metadata" / f"qcd_iht_importance_sampling_summary_{n_events}_2026_07_08.csv"
summary.to_csv(out_csv, index=False)
print(summary.to_string(index=False))
print("Wrote:", out_csv)

def save(name):
    out = outdir / name
    plt.tight_layout()
    plt.savefig(out, dpi=180)
    plt.close()
    print("Wrote:", out)

def weighted_plot(var, bins, xlabel, filename, logy=False):
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
        label=f"HT-sliced QCD {n_events}/slice",
    )
    if logy:
        plt.yscale("log")
    plt.xlabel(xlabel)
    plt.ylabel("Candidate-level cross section per bin [pb]")
    plt.title(f"Weighted closure: {xlabel}")
    plt.legend()
    save(filename)

def normalized_plot(var, bins, xlabel, filename):
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
        label=f"HT-sliced QCD {n_events}/slice",
    )
    plt.xlabel(xlabel)
    plt.ylabel("Normalized weighted candidates")
    plt.title(f"Shape closure: {xlabel}")
    plt.legend()
    save(filename)

def closure_metrics(var, bins):
    h_inc, _ = np.histogram(inc[var], bins=bins, weights=inc["event_weight_pb"])
    h_sl, _ = np.histogram(sliced[var], bins=bins, weights=sliced["event_weight_pb"])

    p = h_inc / h_inc.sum() if h_inc.sum() else h_inc
    q = h_sl / h_sl.sum() if h_sl.sum() else h_sl
    l1 = 0.5 * np.sum(np.abs(p - q))

    return {
        "variable": var,
        "inclusive_yield_pb": h_inc.sum(),
        "sliced_yield_pb": h_sl.sum(),
        "relative_yield_difference": (h_sl.sum() - h_inc.sum()) / h_inc.sum() if h_inc.sum() else np.nan,
        "normalized_shape_l1": l1,
    }

metrics = []

weighted_plot("mhh", np.linspace(0, 1600, 65), "mHH [GeV]", "weighted_mhh_closure.png", logy=True)
normalized_plot("mhh", np.linspace(0, 1600, 65), "mHH [GeV]", "normalized_mhh_closure.png")
metrics.append(closure_metrics("mhh", np.linspace(0, 1600, 65)))

weighted_plot("avg_mbb", np.linspace(0, 300, 61), "average m_bb [GeV]", "weighted_avg_mbb_closure.png", logy=True)
normalized_plot("avg_mbb", np.linspace(0, 300, 61), "average m_bb [GeV]", "normalized_avg_mbb_closure.png")
metrics.append(closure_metrics("avg_mbb", np.linspace(0, 300, 61)))

weighted_plot("r_hh", np.linspace(0, 300, 61), "R_HH-like distance [GeV]", "weighted_rhh_closure.png", logy=True)
normalized_plot("r_hh", np.linspace(0, 300, 61), "R_HH-like distance [GeV]", "normalized_rhh_closure.png")
metrics.append(closure_metrics("r_hh", np.linspace(0, 300, 61)))

# Per-slice mHH contribution
plt.figure()
for tag, sub in sliced.groupby("sample"):
    plt.hist(
        sub["mhh"],
        bins=np.linspace(0, 1600, 65),
        weights=sub["event_weight_pb"],
        histtype="step",
        label=tag.replace("qcd_bbbb_", "").replace(f"_{n_events}", ""),
    )
plt.xlabel("mHH [GeV]")
plt.ylabel("Candidate-level cross section per bin [pb]")
plt.title(f"Weighted mHH contribution by HT slice, {n_events}/slice")
plt.yscale("log")
plt.legend(fontsize=7)
save("weighted_mhh_by_iht_slice.png")

metrics_df = pd.DataFrame(metrics)
metrics_path = store / "metadata" / f"qcd_iht_closure_metrics_{n_events}_2026_07_08.csv"
metrics_df.to_csv(metrics_path, index=False)
print(metrics_df.to_string(index=False))
print("Wrote:", metrics_path)
