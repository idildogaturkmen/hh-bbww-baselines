#!/usr/bin/env python3
"""Task F figures. Every figure carries the required DEVELOPMENT ONLY
footer and prints N for every plotted population in its legend."""
import json, os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

OUT = "/uscms_data/d3/iturkmen/hh4b_delphes/paper_exports/track_b_development_residual_backgrounds_20260821_v1"
RES = f"{OUT}/results"
FIG = f"{OUT}/figures"
PRIMARY_WPS = [0.50, 0.40]
FOOTER = ("DEVELOPMENT ONLY — matched 400k training-production cohort; "
          "not independent Stage-C; not physically normalized.")
LIMITED_N, EXTREME_LIMITED_N = 100, 20

plt.rcParams.update({"font.size": 9, "figure.dpi": 150, "savefig.dpi": 300})


def n_label(name, n):
    if n < EXTREME_LIMITED_N:
        return f"{name} (N={n}, EXTREMELY LIMITED)"
    if n < LIMITED_N:
        return f"{name} (N={n}, limited)"
    return f"{name} (N={n})"


def add_footer(fig):
    fig.text(0.5, 0.005, FOOTER, ha="center", va="bottom", fontsize=7, style="italic", color="dimgray")


def save_all(fig, name):
    fig.tight_layout(rect=[0, 0.02, 1, 1])
    for ext in ("pdf", "svg", "png"):
        fig.savefig(f"{FIG}/{name}.{ext}", bbox_inches="tight")
    plt.close(fig)


def hist_panel(ax, series_dict, bins, xlabel, xlim=None):
    for label, arr in series_dict.items():
        arr = arr[~np.isnan(arr)]
        n = len(arr)
        if n == 0:
            continue
        ax.hist(arr, bins=bins, range=xlim, histtype="step", linewidth=1.4, density=True,
                label=n_label(label, n))
    ax.set_xlabel(xlabel)
    ax.set_ylabel("density")
    ax.legend(fontsize=6.5, loc="best")
    if xlim:
        ax.set_xlim(xlim)


def main():
    topo = np.load(f"{RES}/topology_full.npz")
    is_qcd, is_ttbar, is_bg = topo["is_qcd"], topo["is_ttbar"], topo["is_bg"]
    valid4 = topo["valid4"]
    score = {"K": topo["score_K"], "KF": topo["score_KF"], "SPANET2M": topo["score_SPANET2M"]}
    with open(f"{RES}/thresholds_frozen.json") as f:
        thresholds = json.load(f)

    def thr(arm, epsS):
        return thresholds[arm][str(epsS)] if str(epsS) in thresholds[arm] else thresholds[arm][epsS]

    # ---------------- Task A figure: composition vs epsS (stacked) ----------------
    comp = pd.read_csv(f"{OUT}/RESIDUAL_BACKGROUND_COMPOSITION.csv")
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5), sharey=False)
    for ax, model in zip(axes, ["K", "KF", "SPANET2M"]):
        sub = comp[comp["model"] == model].sort_values("epsS", ascending=False)
        x = np.arange(len(sub))
        ax.bar(x, sub["n_qcd_survive"], label="QCD", color="#4c72b0")
        ax.bar(x, sub["n_ttbar_survive"], bottom=sub["n_qcd_survive"], label="ttbar", color="#dd8452")
        ax.set_xticks(x)
        ax.set_xticklabels([f"{e:g}" for e in sub["epsS"]])
        ax.set_xlabel("target $\\epsilon_S$")
        ax.set_ylabel("surviving background events (raw count)")
        ax.set_title(f"BDT-{model}" if model != "SPANET2M" else "SPA-Net-2M")
        for i, (n, tot) in enumerate(zip(sub["epsS"], sub["n_bg_survive_total"])):
            ax.text(i, tot, str(int(tot)), ha="center", va="bottom", fontsize=7)
        ax.legend(fontsize=8)
    fig.suptitle("Residual QCD/ttbar composition vs $\\epsilon_S$ working point "
                 "(raw development-cohort counts, NOT physically normalized)", fontsize=11)
    add_footer(fig)
    save_all(fig, "TaskA_composition_vs_epsS")

    # ---------------- Task C: per (process, epsS) kinematics + topology panels ----------------
    def population_masks(epsS):
        return {
            "pre-selection": {"qcd": is_qcd, "ttbar": is_ttbar, "combined_background": is_bg},
            "BDT-K": {p: (score["K"] >= thr("K", epsS)) & m for p, m in
                      [("qcd", is_qcd), ("ttbar", is_ttbar), ("combined_background", is_bg)]},
            "BDT-KF": {p: (score["KF"] >= thr("KF", epsS)) & m for p, m in
                       [("qcd", is_qcd), ("ttbar", is_ttbar), ("combined_background", is_bg)]},
            "SPA-Net-2M": {p: (score["SPANET2M"] >= thr("SPANET2M", epsS)) & m for p, m in
                           [("qcd", is_qcd), ("ttbar", is_ttbar), ("combined_background", is_bg)]},
        }

    for epsS in PRIMARY_WPS:
        pops = population_masks(epsS)
        for proc in ["qcd", "ttbar", "combined_background"]:
            series = {label: pops[label][proc] for label in pops}

            # kinematics panel: njet, HT, leading-jet pT, flavor summary
            fig, axes = plt.subplots(2, 2, figsize=(11, 8))
            hist_panel(axes[0, 0], {lb: topo["n_selected_jets"][m] for lb, m in series.items()},
                       bins=np.arange(0, 15) - 0.5, xlabel="n_selected_jets")
            hist_panel(axes[0, 1], {lb: topo["HT"][m] for lb, m in series.items()},
                       bins=40, xlabel="HT [GeV]")
            hist_panel(axes[1, 0], {lb: topo["leading_jet_pt"][m] for lb, m in series.items()},
                       bins=40, xlabel="leading-jet $p_T$ [GeV]")
            # flavor summary: mean probB of leading 4 jets (most discriminating of the three)
            hist_panel(axes[1, 1], {lb: topo["probB_leading4_mean"][m & valid4] for lb, m in series.items()},
                       bins=40, xlim=(0, 1), xlabel="mean probB (leading 4 jets)")
            fig.suptitle(f"{proc}, $\\epsilon_S$={epsS}: kinematics & flavor "
                         f"(pre-selection vs K/KF/SPA-2M survivors)", fontsize=11)
            add_footer(fig)
            save_all(fig, f"TaskC_kinematics_{proc}_epsS{epsS}")

            # topology panel: m_bb1 vs m_bb2 (KF only, 2D, for legibility), R_HH, mass asymmetry, deltaR
            fig, axes = plt.subplots(2, 2, figsize=(11, 8))
            ax = axes[0, 0]
            for lb, m in series.items():
                sel = m & valid4
                m1, m2 = topo["m_bb1"][sel], topo["m_bb2"][sel]
                ax.scatter(m1, m2, s=6, alpha=0.5, label=n_label(lb, len(m1)))
            ax.plot([125], [125], marker="+", color="k", markersize=12)
            ax.set_xlabel("$m_{bb1}$ [GeV]"); ax.set_ylabel("$m_{bb2}$ [GeV]")
            ax.set_xlim(0, 400); ax.set_ylim(0, 400)
            ax.legend(fontsize=6, loc="upper right")
            ax.set_title("$m_{bb1}$ vs $m_{bb2}$ (descriptive $R_{HH}$-minimizing pairing)")

            hist_panel(axes[0, 1], {lb: topo["R_HH"][m & valid4] for lb, m in series.items()},
                       bins=40, xlim=(0, 400), xlabel="$R_{HH}=\\sqrt{(m_{bb1}-125)^2+(m_{bb2}-125)^2}$ [GeV]")
            hist_panel(axes[1, 0], {lb: topo["mass_asymmetry"][m & valid4] for lb, m in series.items()},
                       bins=40, xlim=(0, 1), xlabel="mass asymmetry $|m_1-m_2|/(m_1+m_2)$")
            hist_panel(axes[1, 1], {lb: topo["deltaR_bb_max"][m & valid4] for lb, m in series.items()},
                       bins=40, xlabel="$\\Delta R_{bb}$ (max of pair1, pair2)")
            fig.suptitle(f"{proc}, $\\epsilon_S$={epsS}: four-jet HH topology "
                         f"(pre-selection vs K/KF/SPA-2M survivors)", fontsize=11)
            add_footer(fig)
            save_all(fig, f"TaskC_topology_{proc}_epsS{epsS}")

            # pT(pair1)/pT(pair2) panel
            fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
            hist_panel(axes[0], {lb: topo["pT_pair1"][m & valid4] for lb, m in series.items()},
                       bins=40, xlabel="$p_T$(pair1) [GeV]")
            hist_panel(axes[1], {lb: topo["pT_pair2"][m & valid4] for lb, m in series.items()},
                       bins=40, xlabel="$p_T$(pair2) [GeV]")
            fig.suptitle(f"{proc}, $\\epsilon_S$={epsS}: dijet-pair $p_T$", fontsize=11)
            add_footer(fig)
            save_all(fig, f"TaskC_pairpt_{proc}_epsS{epsS}")

    # ---------------- Task D: KF-only vs SPA-only disagreement comparison ----------------
    for epsS in PRIMARY_WPS:
        kf_pass = score["KF"] >= thr("KF", epsS)
        spa_pass = score["SPANET2M"] >= thr("SPANET2M", epsS)
        kf_only = kf_pass & (~spa_pass) & is_bg
        spa_only = (~kf_pass) & spa_pass & is_bg
        series = {"KF-only survivors": kf_only, "SPA-Net-2M-only survivors": spa_only}

        fig, axes = plt.subplots(2, 3, figsize=(15, 8))
        hist_panel(axes[0, 0], {lb: topo["n_selected_jets"][m] for lb, m in series.items()},
                   bins=np.arange(0, 15) - 0.5, xlabel="n_selected_jets")
        hist_panel(axes[0, 1], {lb: topo["HT"][m] for lb, m in series.items()}, bins=30, xlabel="HT [GeV]")
        hist_panel(axes[0, 2], {lb: topo["probB_leading4_mean"][m & valid4] for lb, m in series.items()},
                   bins=30, xlim=(0, 1), xlabel="mean probB (leading 4 jets)")
        hist_panel(axes[1, 0], {lb: topo["R_HH"][m & valid4] for lb, m in series.items()},
                   bins=30, xlim=(0, 400), xlabel="$R_{HH}$ [GeV]")
        hist_panel(axes[1, 1], {lb: topo["mass_asymmetry"][m & valid4] for lb, m in series.items()},
                   bins=30, xlim=(0, 1), xlabel="mass asymmetry")
        hist_panel(axes[1, 2], {lb: topo["deltaR_bb_max"][m & valid4] for lb, m in series.items()},
                   bins=30, xlabel="$\\Delta R_{bb}$ (max)")
        fig.suptitle(f"$\\epsilon_S$={epsS}: KF-only vs SPA-Net-2M-only survivors "
                     f"(combined QCD+ttbar background) — observed differences only, no causal claim",
                     fontsize=11)
        add_footer(fig)
        save_all(fig, f"TaskD_disagreement_epsS{epsS}")

    files = sorted(os.listdir(FIG))
    print(f"wrote {len(files)} figure files")


if __name__ == "__main__":
    main()
