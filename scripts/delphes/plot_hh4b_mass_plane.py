#!/usr/bin/env python3
from pathlib import Path
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

store = Path(os.environ["HH4B_STORE"])
outdir = store / "plots" / "hh4b_mass_plane_2026_07_07"
outdir.mkdir(parents=True, exist_ok=True)

samples = {
    "VBF HH4b 10k": store / "parquet" / "HH4b_vbf_hh4b_10k_transfer_cluster_3473183_merged_10k_hh4b_candidates.parquet",
    "ggF HEFT HH4b 10k": store / "parquet" / "HH4b_ggf_hh4b_10k_transfer_cluster_84730223_merged_10k_hh4b_candidates.parquet",
    "QCD bbbb 10k": store / "parquet" / "qcd_bbbb_presel_10k_hh4b_candidates.parquet",
    "QCD bbbb 100k": store / "parquet" / "qcd_bbbb_presel_100k_merged_hh4b_candidates.parquet",
    "Zbbbb 10k": store / "parquet" / "zbbbb_presel_10k_hh4b_candidates.parquet",
    "ttbar 10k": store / "parquet" / "ttbar_10k_hh4b_candidates.parquet",
}

frames = []
for name, path in samples.items():
    if not path.exists():
        print("Missing:", path)
        continue
    df = pd.read_parquet(path).copy()
    df["process"] = name
    df["r_hh"] = np.sqrt((df["mbb1"] - 125.0)**2 + (df["mbb2"] - 120.0)**2)
    df["in_sr_rhh30"] = df["r_hh"] < 30.0
    df["in_cr_rhh30_55"] = (df["r_hh"] >= 30.0) & (df["r_hh"] < 55.0)
    frames.append(df)

all_df = pd.concat(frames, ignore_index=True)

def save(name):
    out = outdir / name
    plt.tight_layout()
    plt.savefig(out, dpi=180)
    plt.close()
    print("Wrote:", out)

# R_HH distribution by process
plt.figure()
for proc, sub in all_df.groupby("process"):
    plt.hist(sub["r_hh"], bins=60, range=(0, 250), histtype="step", density=True, label=proc)
plt.xlabel("R_HH-like mass-plane distance [GeV]")
plt.ylabel("Normalized candidates")
plt.title("HH4b mass-plane distance by process")
plt.legend(fontsize=7)
save("rhh_by_process.png")

# Signal vs background R_HH
signal_names = {"VBF HH4b 10k", "ggF HEFT HH4b 10k"}
all_df["label"] = all_df["process"].isin(signal_names).astype(int)

plt.figure()
for label, sub in all_df.groupby("label"):
    name = "signal" if label == 1 else "background"
    plt.hist(sub["r_hh"], bins=60, range=(0, 250), histtype="step", density=True, label=name)
plt.xlabel("R_HH-like mass-plane distance [GeV]")
plt.ylabel("Normalized candidates")
plt.title("HH4b mass-plane distance: signal vs background")
plt.legend()
save("rhh_signal_vs_background.png")

# mbb1 vs mbb2 mass plane, one plot per process
theta = np.linspace(0, 2*np.pi, 300)
for proc, sub in all_df.groupby("process"):
    plt.figure()
    plt.scatter(sub["mbb1"], sub["mbb2"], s=5, alpha=0.45)
    plt.plot(125 + 30*np.cos(theta), 120 + 30*np.sin(theta), linestyle="--", label="R_HH = 30")
    plt.plot(125 + 55*np.cos(theta), 120 + 55*np.sin(theta), linestyle=":", label="R_HH = 55")
    plt.xlabel("m_bb1 [GeV]")
    plt.ylabel("m_bb2 [GeV]")
    plt.xlim(40, 260)
    plt.ylim(30, 230)
    plt.title(f"Mass plane: {proc}")
    plt.legend(fontsize=8)
    safe = proc.lower().replace(" ", "_").replace("→", "").replace("/", "_")
    save(f"mass_plane_{safe}.png")

# Yield table in mass-plane regions
summary = (
    all_df.groupby("process")
    .agg(
        candidates=("r_hh", "count"),
        median_r_hh=("r_hh", "median"),
        n_rhh_lt_30=("in_sr_rhh30", "sum"),
        n_rhh_30_55=("in_cr_rhh30_55", "sum"),
    )
    .reset_index()
)
summary["frac_rhh_lt_30"] = summary["n_rhh_lt_30"] / summary["candidates"]
summary["frac_rhh_30_55"] = summary["n_rhh_30_55"] / summary["candidates"]

out_csv = store / "metadata" / "hh4b_mass_plane_summary_2026_07_07.csv"
summary.to_csv(out_csv, index=False)
print(summary.to_string(index=False))
print("Wrote:", out_csv)

out_md = outdir / "hh4b_mass_plane_summary_2026_07_07.md"
out_md.write_text(summary.to_markdown(index=False) + "\n")
print("Wrote:", out_md)
