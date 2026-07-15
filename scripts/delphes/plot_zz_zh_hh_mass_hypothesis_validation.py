#!/usr/bin/env python3

from pathlib import Path
import os
import pandas as pd
import matplotlib.pyplot as plt

REPO = Path(os.environ["HH4B_REPO"])
INDIR = REPO / "outputs/tables/zz_zh_hh_mass_hypothesis_validation_ak4ak8_2026_07_14"
OUTDIR = REPO / "outputs/plots/zz_zh_hh_mass_hypothesis_validation_ak4ak8_2026_07_14"
OUTDIR.mkdir(parents=True, exist_ok=True)

cand = pd.read_csv(INDIR / "zz_zh_hh_mass_hypothesis_candidates.csv")
best = pd.read_csv(INDIR / "zz_zh_hh_best_hypothesis_fractions.csv")

# Plot best-hypothesis fractions.
pivot = best.pivot_table(
    index="sample",
    columns="hypothesis",
    values="fraction",
    fill_value=0.0,
)

ax = pivot[["HH", "ZH", "ZZ"]].plot(kind="bar", figsize=(9, 5))
ax.set_ylabel("Fraction of selected events")
ax.set_xlabel("")
ax.set_title("Best mass hypothesis by sample")
ax.legend(title="Best hypothesis")
plt.xticks(rotation=30, ha="right")
plt.tight_layout()
plt.savefig(OUTDIR / "best_hypothesis_fractions.png", dpi=200)
plt.savefig(OUTDIR / "best_hypothesis_fractions.pdf")
plt.close()

# Mass-plane plots using each sample's truth hypothesis.
truth_hyp = {
    "HH_ggF_ak4ak8_10k": "HH",
    "HH_VBF_ak4ak8_10k": "HH",
    "ZZ4b_pilot10k": "ZZ",
    "ZH4b_pilot10k": "ZH",
}

for sample, hyp in truth_hyp.items():
    df = cand[(cand["sample"] == sample) & (cand["hypothesis"] == hyp)].copy()
    if len(df) == 0:
        continue

    plt.figure(figsize=(6, 5))
    plt.hist2d(df["mbb1"], df["mbb2"], bins=40, range=[[40, 180], [40, 180]])
    plt.xlabel(r"$m_{bb,1}$ [GeV]")
    plt.ylabel(r"$m_{bb,2}$ [GeV]")
    plt.title(f"{sample}: pairing under {hyp} hypothesis")
    plt.colorbar(label="events")
    plt.tight_layout()
    safe = sample.lower()
    plt.savefig(OUTDIR / f"{safe}_mass_plane_{hyp.lower()}.png", dpi=200)
    plt.savefig(OUTDIR / f"{safe}_mass_plane_{hyp.lower()}.pdf")
    plt.close()

print("Wrote:", OUTDIR)
