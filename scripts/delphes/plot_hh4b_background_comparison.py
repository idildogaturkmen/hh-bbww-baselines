#!/usr/bin/env python3
from pathlib import Path
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

store = Path(os.environ["HH4B_STORE"])
outdir = Path("outputs/plots/hh4b_background_campaigns_2026_07_08")
outdir.mkdir(parents=True, exist_ok=True)

samples = [
    ("qcd_bbbb_presel_100k", "QCD bbbb 100k"),
    ("zbbbb_presel_100k", "Zbbbb 100k"),
    ("ttbar_100k", "inclusive ttbar 100k"),
    ("ttbb_10k", "ttbb 10k diagnostic"),
]

frames = []
summary_rows = []

for tag, label in samples:
    ev_path = store / "parquet" / f"{tag}_merged_event_summary.parquet"
    cand_path = store / "parquet" / f"{tag}_merged_hh4b_candidates.parquet"

    if not ev_path.exists() or not cand_path.exists():
        print(f"Skipping missing sample: {tag}")
        continue

    ev = pd.read_parquet(ev_path)
    cand = pd.read_parquet(cand_path).copy()

    xsec = float(ev["event_cross_section_pb"].median())
    n_gen = len(ev)
    weight = xsec / n_gen

    cand["sample"] = tag
    cand["label"] = label
    cand["event_weight_pb"] = weight
    cand["r_hh"] = np.sqrt((cand["mbb1"] - 125.0)**2 + (cand["mbb2"] - 120.0)**2)

    frames.append(cand)

    summary_rows.append({
        "sample": tag,
        "label": label,
        "n_generated": n_gen,
        "xsec_pb": xsec,
        "candidate_rows": len(cand),
        "candidate_eff": len(cand) / n_gen,
        "effective_candidate_xsec_pb": weight * len(cand),
        "median_mhh": cand["mhh"].median(),
        "median_r_hh": cand["r_hh"].median(),
    })

all_cand = pd.concat(frames, ignore_index=True)
summary = pd.DataFrame(summary_rows)

table_dir = Path("outputs/tables/hh4b_background_campaigns_2026_07_08")
table_dir.mkdir(parents=True, exist_ok=True)
summary.to_csv(table_dir / "background_comparison_with_ttbb10k.csv", index=False)
(table_dir / "background_comparison_with_ttbb10k.md").write_text(summary.to_markdown(index=False) + "\n")

print(summary.to_string(index=False))

def save(name):
    plt.tight_layout()
    out = outdir / name
    plt.savefig(out, dpi=180)
    plt.close()
    print("Wrote:", out)

def weighted_hist(var, bins, xlabel, filename, logy=True):
    plt.figure()
    for tag, label in [(t, l) for t, l in samples if t in set(all_cand["sample"])]:
        sub = all_cand[all_cand["sample"] == tag]
        plt.hist(
            sub[var],
            bins=bins,
            weights=sub["event_weight_pb"],
            histtype="step",
            label=label,
        )
    if logy:
        plt.yscale("log")
    plt.xlabel(xlabel)
    plt.ylabel("Candidate-level cross section [pb/bin]")
    plt.title(f"Background comparison: {xlabel}")
    plt.legend(fontsize=8)
    save(filename)

weighted_hist("mhh", np.linspace(0, 1600, 65), "mHH [GeV]", "background_weighted_mhh.png")
weighted_hist("avg_mbb", np.linspace(0, 300, 61), "average m_bb [GeV]", "background_weighted_avg_mbb.png")
weighted_hist("r_hh", np.linspace(0, 300, 61), "R_HH-like distance [GeV]", "background_weighted_rhh.png")
weighted_hist("delta_mbb", np.linspace(0, 250, 51), "delta m_bb [GeV]", "background_weighted_delta_mbb.png")

plt.figure()
plt.bar(summary["label"], summary["effective_candidate_xsec_pb"])
plt.xticks(rotation=25, ha="right")
plt.ylabel("Effective selected-candidate cross section [pb]")
plt.title("Selected HH4b-candidate background rates")
save("background_effective_candidate_xsec_bar.png")
