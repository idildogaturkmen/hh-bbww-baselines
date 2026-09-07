#!/usr/bin/env python3
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

PKG = "/uscms_data/d3/iturkmen/hh4b_delphes/paper_exports/track_b_harvey_tail_kinematics_20260906_v1"
FIG = f"{PKG}/figures"

master = pd.read_parquet(f"{PKG}/HARVEY_MASTER_EVENT_TABLE.parquet")

# ---- Figure 1: signal u-distribution, has-5th-jet vs not ----
fig, ax = plt.subplots(figsize=(7, 5))
sig = master[master["label"] == 1]
bins = np.linspace(3.5, 8, 40)
ax.hist(sig[sig["has_5th_jet"]]["u"].clip(upper=8), bins=bins, histtype="step", lw=2, label=f"signal, >=5 jets (n={int(sig['has_5th_jet'].sum())})", density=True)
ax.hist(sig[~sig["has_5th_jet"]]["u"].clip(upper=8), bins=bins, histtype="step", lw=2, label=f"signal, =4 jets (n={int((~sig['has_5th_jet']).sum())})", density=True)
ax.axvline(4.5, color="k", ls=":", lw=1)
ax.set_xlabel("u = -log10(1-score)"); ax.set_ylabel("density (raw counts, normalized)")
ax.set_title("Signal u-distribution: with vs without a 5th selected jet")
ax.legend(fontsize=8)
fig.tight_layout(); fig.savefig(f"{FIG}/signal_u_with_vs_without_5th_jet.png", dpi=150); plt.close(fig)

# ---- Figure 2: 2D heatmap, signal truth-match fraction (pT5, minDR5) ----
heat = pd.read_csv(f"{PKG}/work/fifth_jet_signal_heatmap_truth.csv")
pt5_labels = heat["pt5_bin"].unique()
dr_labels = heat["dr_bin"].unique()
grid = heat.pivot(index="pt5_bin", columns="dr_bin", values="mean").reindex(index=pt5_labels, columns=dr_labels)
counts = heat.pivot(index="pt5_bin", columns="dr_bin", values="count").reindex(index=pt5_labels, columns=dr_labels)
fig, ax = plt.subplots(figsize=(7.5, 5.5))
im = ax.imshow(grid.values, aspect="auto", origin="lower", cmap="viridis", vmin=0, vmax=1)
ax.set_xticks(range(len(dr_labels))); ax.set_xticklabels([str(x) for x in dr_labels], rotation=45, ha="right", fontsize=7)
ax.set_yticks(range(len(pt5_labels))); ax.set_yticklabels([str(x) for x in pt5_labels], fontsize=7)
for i in range(grid.shape[0]):
    for j in range(grid.shape[1]):
        c = counts.values[i, j]
        v = grid.values[i, j]
        if np.isfinite(v):
            ax.text(j, i, f"{v:.2f}\n(n={int(c)})", ha="center", va="center", fontsize=6, color="white" if v < 0.6 else "black")
ax.set_xlabel("min DeltaR(j5, leading-4)"); ax.set_ylabel("pT5 [GeV]")
ax.set_title("Signal: P(j5 truth-matched to an HH b-jet) vs (pT5, minDeltaR5)")
fig.colorbar(im, ax=ax, label="fraction truth-matched")
fig.tight_layout(); fig.savefig(f"{FIG}/fifth_jet_signal_truth_heatmap.png", dpi=150); plt.close(fig)

# ---- Figure 3: 1D ranking bar chart ----
ranking = pd.read_csv(f"{PKG}/HARVEY_1D_FEATURE_RANKING.csv").head(12)
fig, ax = plt.subplots(figsize=(7, 5))
ax.barh(ranking["feature"][::-1], ranking["mean_abs_auc_minus_half_u3p5"][::-1],
        xerr=ranking["std_across_comparisons"][::-1])
ax.set_xlabel("mean |AUC - 0.5| across all_background/QCD/ttbar comparisons (u>3.5)")
ax.set_title("Top 12 features by robust 1D signal/background separation")
fig.tight_layout(); fig.savefig(f"{FIG}/1d_feature_ranking.png", dpi=150); plt.close(fig)

# ---- Figure 4: pT_H1 distribution, signal vs ttbar, with the candidate cut ----
fig, ax = plt.subplots(figsize=(7, 5))
sub35 = master[master["u"] > 3.5]
sig35 = sub35[sub35["label"] == 1]
tt35 = sub35[sub35["background_class"] == "ttbar"]
bins = np.linspace(0, 1200, 40)
ax.hist(sig35["pT_H1"].clip(upper=1200), bins=bins, density=True, histtype="step", lw=2, label=f"signal (n={len(sig35)})")
ax.hist(tt35["pT_H1"].clip(upper=1200), bins=bins, density=True, histtype="step", lw=2, label=f"ttbar (n={len(tt35)})")
ax.axvline(406.1, color="k", ls="--", lw=1, label="candidate cut: pT_H1<406 GeV")
ax.set_xlabel("pT_H1 [GeV]"); ax.set_ylabel("density"); ax.set_title("pT_H1: signal vs surviving ttbar (u>3.5)")
ax.legend(fontsize=8)
fig.tight_layout(); fig.savefig(f"{FIG}/ttbar_pTH1_candidate_cut.png", dpi=150); plt.close(fig)

# ---- Figure 5: QCD assignment heatmap (small-N, explicitly labeled) ----
heatq = pd.read_csv(f"{PKG}/work/fifth_jet_qcd_heatmap_assignment.csv")
gridq = heatq.pivot(index="pt5_bin", columns="dr_bin", values="mean").reindex(index=pt5_labels, columns=dr_labels)
countsq = heatq.pivot(index="pt5_bin", columns="dr_bin", values="count").reindex(index=pt5_labels, columns=dr_labels)
fig, ax = plt.subplots(figsize=(7.5, 5.5))
im = ax.imshow(gridq.values, aspect="auto", origin="lower", cmap="viridis", vmin=0, vmax=1)
ax.set_xticks(range(len(dr_labels))); ax.set_xticklabels([str(x) for x in dr_labels], rotation=45, ha="right", fontsize=7)
ax.set_yticks(range(len(pt5_labels))); ax.set_yticklabels([str(x) for x in pt5_labels], fontsize=7)
for i in range(gridq.shape[0]):
    for j in range(gridq.shape[1]):
        c = countsq.values[i, j]
        v = gridq.values[i, j]
        if np.isfinite(v) and c > 0:
            ax.text(j, i, f"{v:.2f}\n(n={int(c)})", ha="center", va="center", fontsize=6, color="white" if v < 0.6 else "black")
ax.set_xlabel("min DeltaR(j5, leading-4)"); ax.set_ylabel("pT5 [GeV]")
ax.set_title("QCD (n=28 total, EXTREMELY LIMITED per bin): P(j5 in genuine SPA-Net assignment)")
fig.colorbar(im, ax=ax, label="fraction")
fig.tight_layout(); fig.savefig(f"{FIG}/fifth_jet_qcd_assignment_heatmap.png", dpi=150); plt.close(fig)

print("wrote 5 figures")
