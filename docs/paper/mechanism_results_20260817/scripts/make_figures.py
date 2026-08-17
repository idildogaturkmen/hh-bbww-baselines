#!/usr/bin/env python3
"""Render the 4 paper figures from MECHANISM_RESULTS_DATA.json only.
Does not open any frozen source file directly -- everything here traces
through the extraction step (extract_frozen_inputs.py), which
independent_validate.py separately re-derives from the raw sources to
cross-check.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from style import (
    ARM_COLOR, ARM_MARKER, ARM_LABEL, GAIN_COLOR, GAIN_LABEL,
    TRACKA_MODEL_COLOR, TRACKA_MODEL_HATCH, apply_tier_style, sim_watermark,
    MUTED_INK, AXIS_INK, GRID_INK, BLUE, ORANGE, SIM_WATERMARK, TRACKA_WATERMARK,
)

ROOT = Path(__file__).resolve().parent.parent
DATA = json.loads((ROOT / "MECHANISM_RESULTS_DATA.json").read_text())
FIGDIR = ROOT / "figures"
FIGDIR.mkdir(exist_ok=True)


def save(fig, name):
    fig.savefig(FIGDIR / f"{name}.pdf", bbox_inches="tight")
    fig.savefig(FIGDIR / f"{name}.png", bbox_inches="tight")
    plt.close(fig)
    print(f"wrote figures/{name}.pdf and .png")


def draw_marker(ax, x, y, yerr, color, marker, tier, label=None, zorder=3):
    style = apply_tier_style(tier)
    mfc = color if style["markerfacecolor"] == "__color__" else style["markerfacecolor"]
    (_, caps, bars) = ax.errorbar(
        x, y, yerr=yerr, fmt=marker, color=color, markerfacecolor=mfc,
        markeredgecolor=color, markersize=style["markersize"], mew=style["mew"],
        elinewidth=1.3, capsize=3, alpha=style["alpha"], label=label, zorder=zorder,
        linestyle="none",
    )
    return


# ---------------------------------------------------------------------------
# Figure 1: Track-B holdout_A mechanism ablation (E0/E1/E2, R_B vs eps_S)
# ---------------------------------------------------------------------------

def fig1():
    wp = DATA["track_b"]["working_points"]
    fig, ax = plt.subplots(figsize=(7.0, 5.2))

    for arm in ["E0", "E1", "E2"]:
        rows = sorted([w for w in wp if w["arm"] == arm], key=lambda w: w["eps_S"])
        xs = [r["eps_S"] for r in rows]
        ys = [r["R_B_mean"] for r in rows]
        yerrs = [r["R_B_std"] for r in rows]
        color = ARM_COLOR[arm]
        ax.plot(xs, ys, "-", color=color, lw=1.6, alpha=0.55, zorder=2)
        for r in rows:
            draw_marker(ax, r["eps_S"], r["R_B_mean"], r["R_B_std"], color,
                        ARM_MARKER[arm], r["support_tier_of_mean"], zorder=3)
        # one legend proxy per arm (well-supported style, full color)
        ax.plot([], [], marker=ARM_MARKER[arm], color=color, linestyle="-",
                 lw=1.6, markersize=7, label=ARM_LABEL[arm])

    ax.set_yscale("log")
    ax.set_xlabel(r"Signal efficiency $\epsilon_S$")
    ax.set_ylabel(r"$R_B = 1/\epsilon_B$  (holdout_A, tight\_exact4)")
    ax.set_xlim(0.05, 0.65)
    ax.grid(True, which="major", axis="y", color=GRID_INK, lw=0.7, zorder=0)

    # support-tier marker key (separate from the color/arm legend)
    tier_handles = [
        plt.Line2D([], [], marker="o", color=MUTED_INK, markerfacecolor=MUTED_INK,
                    linestyle="none", markersize=7, label=r"well-supported ($\geq$100 raw QCD)"),
        plt.Line2D([], [], marker="o", color=MUTED_INK, markerfacecolor=MUTED_INK,
                    linestyle="none", markersize=6, alpha=0.62, label="limited but reportable (10-99 raw QCD)"),
        plt.Line2D([], [], marker="o", color=MUTED_INK, markerfacecolor="none",
                    linestyle="none", markersize=5, mew=1.4, label="exploratory only (<10 raw QCD)"),
    ]
    leg1 = ax.legend(loc="upper right", fontsize=8.7, title="Arm (color/marker)", title_fontsize=8.7)
    ax.add_artist(leg1)
    ax.legend(handles=tier_handles, loc="lower left", fontsize=7.8,
              title="Statistical-support tier (marker style)", title_fontsize=7.8,
              bbox_to_anchor=(0.0, 0.0))
    sim_watermark(ax, SIM_WATERMARK, loc="upper left")
    fig.tight_layout()
    save(fig, "trackb_holdoutA_mechanism_rejection")


# ---------------------------------------------------------------------------
# Figure 2: Track-B mechanism gain (E1/E0, E2/E1)
# ---------------------------------------------------------------------------

def fig2():
    gain = sorted(DATA["track_b"]["mechanism_gain"], key=lambda r: r["eps_S"])
    fig, ax = plt.subplots(figsize=(7.0, 5.0))

    for key in ["E1_over_E0", "E2_over_E1"]:
        xs = [r["eps_S"] for r in gain]
        ys = [r[key]["mean_of_per_seed_ratios"] for r in gain]
        yerrs = [r[key]["std_of_per_seed_ratios"] for r in gain]
        tiers = [r[key]["support_tier"] for r in gain]
        color = GAIN_COLOR[key]
        ax.plot(xs, ys, "-", color=color, lw=1.6, alpha=0.55, zorder=2)
        for x, y, ye, t in zip(xs, ys, yerrs, tiers):
            draw_marker(ax, x, y, ye, color, "o", t, zorder=3)
        ax.plot([], [], marker="o", color=color, linestyle="-", lw=1.6,
                 markersize=7, label=GAIN_LABEL[key])

    ax.axhline(1.0, color=AXIS_INK, lw=1.2, ls="--", zorder=1)
    ax.text(0.61, 1.0, "ratio = 1", color=AXIS_INK, fontsize=8.5, va="bottom", ha="right")

    ax.set_xlabel(r"Signal efficiency $\epsilon_S$")
    ax.set_ylabel(r"$R_B$ ratio (holdout_A, seed-paired)")
    ax.set_xlim(0.05, 0.65)
    ax.grid(True, which="major", axis="y", color=GRID_INK, lw=0.7, zorder=0)

    tier_handles = [
        plt.Line2D([], [], marker="o", color=MUTED_INK, markerfacecolor=MUTED_INK,
                    linestyle="none", markersize=7, label=r"well-supported ($\geq$100 raw QCD, both arms)"),
        plt.Line2D([], [], marker="o", color=MUTED_INK, markerfacecolor=MUTED_INK,
                    linestyle="none", markersize=6, alpha=0.62, label="limited but reportable (10-99 raw QCD)"),
        plt.Line2D([], [], marker="o", color=MUTED_INK, markerfacecolor="none",
                    linestyle="none", markersize=5, mew=1.4, label="exploratory only (<10 raw QCD)"),
    ]
    leg1 = ax.legend(loc="upper left", fontsize=8.7)
    ax.add_artist(leg1)
    ax.legend(handles=tier_handles, loc="upper right", fontsize=7.6,
              title="Statistical-support tier (min. of the two arms)", title_fontsize=7.6)
    sim_watermark(ax, SIM_WATERMARK, loc="lower left")
    fig.tight_layout()
    save(fig, "trackb_mechanism_gain")


# ---------------------------------------------------------------------------
# Figure 3: Track-A model comparison at eps_S = 0.40
# ---------------------------------------------------------------------------

def fig3():
    models = DATA["track_a"]["fig3_models"]
    fig, ax = plt.subplots(figsize=(8.6, 5.4))
    xs = np.arange(len(models))
    for i, m in enumerate(models):
        color = TRACKA_MODEL_COLOR[m["short"]]
        hatch = TRACKA_MODEL_HATCH[m["short"]]
        yerr = m["R_B_err"]
        bar = ax.bar(i, m["R_B"], color=color, edgecolor="white", linewidth=0.8,
                      hatch=hatch, width=0.62, zorder=2,
                      yerr=[[yerr], [yerr]] if yerr else None,
                      ecolor=MUTED_INK, capsize=4, error_kw=dict(lw=1.4, zorder=4))
        label = f"{m['R_B']:.2f}" + (r"$\pm$" + f"{yerr:.2f}" if yerr else "")
        ax.text(i, m["R_B"] + (yerr or 0) + 0.28, label, ha="center", va="bottom",
                 fontsize=8.7, color="#0b0b0b")

    ax.set_xticks(xs)
    ax.set_xticklabels([m["label"] for m in models], fontsize=8.6)
    ax.set_ylabel(r"$R_B = 1/\epsilon_B$ at $\epsilon_S = 0.40$")
    ax.set_ylim(0, max(m["R_B"] + (m["R_B_err"] or 0) for m in models) * 1.22)
    ax.grid(True, which="major", axis="y", color=GRID_INK, lw=0.7, zorder=0)

    from matplotlib.patches import Patch
    handles = [
        Patch(facecolor="white", edgecolor=MUTED_INK, hatch="///", label="frozen reference point (no interval computed here)"),
        Patch(facecolor=MUTED_INK, edgecolor="white", label="measured this benchmark, 5-fold OOF x 3 seeds\n(error bar = seed-to-seed std)"),
    ]
    ax.legend(handles=handles, loc="upper left", fontsize=8.0)

    ax.text(4, models[4]["R_B"] * 0.42,
             "frozen historical\nlate-fusion jet-level\nTransformer reference\n(NOT canonical ParT)",
             ha="center", va="center", fontsize=7.3, color=MUTED_INK, linespacing=1.3)

    sim_watermark(ax, TRACKA_WATERMARK, loc="upper right")
    fig.tight_layout()
    save(fig, "tracka_epsS040_model_comparison")


# ---------------------------------------------------------------------------
# Figure 4: Track-A DNN vs CMS-inspired transformer, R_B vs eps_S,
#           paired-bootstrap panel, unsupported-tail shading
# ---------------------------------------------------------------------------

def fig4():
    dnn_curve = sorted(DATA["track_a"]["dnn_curve"], key=lambda r: r["eps_S"])
    tf_curve = sorted(DATA["track_a"]["cmstransformer_curve"], key=lambda r: r["eps_S"])
    paired = DATA["track_a"]["paired_bootstrap"]["by_epsilon_s"]
    dnn_tail = sorted(DATA["track_a"]["eps_b_tail"]["dnn"], key=lambda r: r["realized_epsilon_S"])
    tf_tail = sorted(DATA["track_a"]["eps_b_tail"]["cmstransformer"], key=lambda r: r["realized_epsilon_S"])
    dnn_tail_unsupported = [r for r in dnn_tail if not r["scientifically_supportable"]]
    tf_tail_unsupported = [r for r in tf_tail if not r["scientifically_supportable"]]

    fig, (ax, ax2) = plt.subplots(
        2, 1, figsize=(7.2, 7.6), sharex=True,
        gridspec_kw=dict(height_ratios=[2.4, 1.3], hspace=0.06),
    )

    for curve, color, label, tail in [
        (dnn_curve, TRACKA_MODEL_COLOR["dnn"], "Dense DNN", dnn_tail_unsupported),
        (tf_curve, TRACKA_MODEL_COLOR["cmstransformer"], "CMS-inspired five-jet pairwise event Transformer", tf_tail_unsupported),
    ]:
        xs = [r["eps_S"] for r in curve]
        ys = [r["R_B_mean"] for r in curve]
        yerrs = [r["R_B_std"] for r in curve]
        ax.errorbar(xs, ys, yerr=yerrs, fmt="o-", color=color, markersize=6.5,
                     lw=1.7, capsize=3, elinewidth=1.3, label=label, zorder=3)
        # unsupported single-seed diagnostic tail (eps_B = 1e-3, 1e-4), not
        # connected to the main (fixed-eps_S) curve -- different slicing,
        # genuine gap in the middle, not interpolated.
        tx = [r["realized_epsilon_S"] for r in tail]
        ty = [r["R_B"] for r in tail]
        ax.plot(tx, ty, "o:", color=color, markerfacecolor="none", mew=1.4,
                 markersize=6, lw=1.1, alpha=0.85, zorder=2)

    shade_hi = max(dnn_tail_unsupported[-1]["realized_epsilon_S"] if dnn_tail_unsupported else 0,
                    tf_tail_unsupported[-1]["realized_epsilon_S"] if tf_tail_unsupported else 0)
    shade_lo_x = min([r["realized_epsilon_S"] for r in dnn_tail_unsupported + tf_tail_unsupported], default=0.003)
    span_lo = max(shade_lo_x * 0.7, 0.003)
    span_hi = max(r["realized_epsilon_S"] for r in dnn_tail_unsupported + tf_tail_unsupported)
    ax.axvspan(span_lo, span_hi, color="#d03b3b", alpha=0.08, zorder=0)

    ax.set_yscale("log")
    ax.set_xlim(span_lo * 0.55, 0.65)
    ax.set_ylabel(r"$R_B = 1/\epsilon_B$")
    ax.legend(loc="center right", fontsize=8.3, bbox_to_anchor=(1.0, 0.72))
    ax.grid(True, which="major", axis="y", color=GRID_INK, lw=0.6, zorder=0)

    ymin, ymax = ax.get_ylim()
    ax.annotate(
        r"$\epsilon_B \leq 10^{-3}$: $qcd\_N_{eff} < 10$ (both models)" "\nnot a supported physical claim\n(single primary seed only, shown for context)",
        xy=(span_hi, ymin), xytext=(span_hi * 1.35, ymin * 2.6),
        fontsize=7.2, color="#a12a2a", ha="left", va="bottom",
        arrowprops=dict(arrowstyle="-", color="#a12a2a", lw=0.8),
    )

    # bottom panel: paired-bootstrap delta R_B (transformer - DNN), primary seed
    eps_s_order = [0.6, 0.585957, 0.5, 0.4, 0.25, 0.1]
    dx = eps_s_order
    d_pt = [paired[f"eps_S={e}"]["point_estimate_diff_transformer_minus_dnn"] for e in dx]
    d_lo68 = [paired[f"eps_S={e}"]["bootstrap_diff_transformer_minus_dnn"]["ci68_lo"] for e in dx]
    d_hi68 = [paired[f"eps_S={e}"]["bootstrap_diff_transformer_minus_dnn"]["ci68_hi"] for e in dx]
    d_lo95 = [paired[f"eps_S={e}"]["bootstrap_diff_transformer_minus_dnn"]["ci95_lo"] for e in dx]
    d_hi95 = [paired[f"eps_S={e}"]["bootstrap_diff_transformer_minus_dnn"]["ci95_hi"] for e in dx]

    order = np.argsort(dx)
    dxo = np.array(dx)[order]
    ax2.fill_between(dxo, np.array(d_lo95)[order], np.array(d_hi95)[order],
                      color=ORANGE, alpha=0.16, lw=0, label="95% CI (paired bootstrap)")
    ax2.fill_between(dxo, np.array(d_lo68)[order], np.array(d_hi68)[order],
                      color=ORANGE, alpha=0.32, lw=0, label="68% CI (paired bootstrap)")
    ax2.plot(dxo, np.array(d_pt)[order], "o-", color=BLUE, markersize=5.5, lw=1.5,
              label=r"point estimate, primary seed", zorder=3)
    ax2.axhline(0.0, color=AXIS_INK, lw=1.1, ls="--", zorder=1)
    ax2.set_xlabel(r"Signal efficiency $\epsilon_S$")
    ax2.set_ylabel(r"$\Delta R_B$" "\n(transformer $-$ DNN)", fontsize=9.3)
    ax2.legend(loc="lower left", fontsize=7.3)
    ax2.grid(True, which="major", axis="y", color=GRID_INK, lw=0.6, zorder=0)
    sim_watermark(ax2, TRACKA_WATERMARK, loc="upper right")

    fig.align_ylabels([ax, ax2])
    save(fig, "tracka_dnn_vs_transformer_rejection")


if __name__ == "__main__":
    fig1()
    fig2()
    fig3()
    fig4()
