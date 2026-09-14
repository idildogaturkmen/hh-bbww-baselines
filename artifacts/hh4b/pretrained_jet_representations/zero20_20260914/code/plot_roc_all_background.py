#!/usr/bin/env python3
"""Publication ROC figure (signal efficiency vs. background mistag rate,
all-background) for all four models on the exact common 400k-event
matched cohort. Reads metrics/roc_curves_data.json (already computed,
already verified event_id/process-matched) -- no recomputation, no
inference, no bootstrap here.

Two-panel horizontal layout (NOT an inset): panel (a) is the full ROC,
panel (b) is a plain zoomed second axes over the same exact curve arrays
(no inset rectangle, no connector lines) covering the high-efficiency/
low-mistag corner where Native 2M, Native 10M, and ZERO20 nearly overlap.
Those three are additionally distinguished by linestyle (not color alone)
since color can be hard to tell apart when curves sit this close together;
ParT active20 keeps its own solid, differently-colored line since it is
never close enough to the other three to need a linestyle distinction.

No plot title (the cohort description belongs in the surrounding
documentation, not the figure); a single shared legend below both panels
carries the AUC values to 5 decimal places.
"""
import argparse
import json

import matplotlib
matplotlib.use("svg")
import matplotlib.pyplot as plt  # noqa: E402

# user-specified legend order (differs from the internal dict order)
MODEL_ORDER = ["native2M", "native10M", "ZERO20_2M", "ParT20_2M"]
# Native2M/10M/ZERO20 sit within ~0.001-0.003 signal-efficiency of each other
# over most of the zoom window -- distinct dash patterns alone are not enough
# once one opaque line fully occludes the others, so all three also get
# partial transparency (alpha<1) so whichever is plotted underneath still
# shows through, and the SOLID (native2M) line is drawn UNDERNEATH the two
# patterned ones (lowest zorder of the three) so their dash/dot gaps reveal
# it rather than a solid line masking their patterns.
STYLE = {
    # Blue/green/red/orange -- four well-separated hues (NOT blue+purple, which
    # are perceptually too close to tell apart once the lines are blended/
    # overlapping -- an earlier draft of this figure used purple for ZERO20 and
    # it was visually indistinguishable from native2M's blue on inspection).
    "native2M":   dict(color="#4C72B0", linestyle="-",              linewidth=1.8, alpha=0.85, zorder=2),
    "native10M":  dict(color="#55A868", linestyle=(0, (5, 2)),       linewidth=1.8, alpha=0.85, zorder=3),
    "ZERO20_2M":  dict(color="#C44E52", linestyle=(0, (6, 2, 1, 2)), linewidth=2.0, alpha=0.9,  zorder=4),
    "ParT20_2M":  dict(color="#DD8452", linestyle="-",              linewidth=1.9, alpha=1.0,  zorder=5),
}


def style_axes(ax):
    ax.grid(True, alpha=0.3, linewidth=0.6)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def draw_curves(ax, d, with_diagonal):
    lines = {}
    for key in MODEL_ORDER:
        m = d["models"][key]
        fpr, tpr = m["roc"]["fpr"], m["roc"]["tpr"]
        (line,) = ax.plot(fpr, tpr, **STYLE[key])
        lines[key] = line
    if with_diagonal:
        ax.plot([0, 1], [0, 1], color="gray", linewidth=0.7, linestyle=":", alpha=0.6, zorder=1)
    return lines


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--roc-curves-json", required=True)
    ap.add_argument("--out-svg", required=True)
    args = ap.parse_args()

    with open(args.roc_curves_json) as f:
        d = json.load(f)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10.5, 5.0))

    lines = draw_curves(ax1, d, with_diagonal=True)
    ax1.set_xlim(0, 1)
    ax1.set_ylim(0, 1)
    ax1.set_xlabel(r"Background efficiency $\epsilon_B$")
    ax1.set_ylabel(r"Signal efficiency $\epsilon_S$")
    ax1.text(0.03, 0.97, "(a)", transform=ax1.transAxes, fontsize=12, fontweight="bold", va="top")
    style_axes(ax1)

    draw_curves(ax2, d, with_diagonal=False)
    ax2.set_xlim(0.0, 0.08)
    ax2.set_ylim(0.72, 1.0)
    ax2.set_xlabel(r"Background efficiency $\epsilon_B$")
    ax2.set_ylabel(r"Signal efficiency $\epsilon_S$")
    ax2.text(0.03, 0.97, "(b)", transform=ax2.transAxes, fontsize=12, fontweight="bold", va="top")
    style_axes(ax2)

    legend_labels = [f"{d['models'][key]['label']} = {d['models'][key]['auc_all_background']:.5f}"
                      for key in MODEL_ORDER]
    fig.legend([lines[key] for key in MODEL_ORDER], legend_labels,
               loc="lower center", ncol=2, frameon=False, fontsize=9.5,
               bbox_to_anchor=(0.5, -0.06))

    fig.tight_layout()
    fig.subplots_adjust(bottom=0.28, wspace=0.28)
    fig.savefig(args.out_svg, format="svg", bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {args.out_svg}")


if __name__ == "__main__":
    main()
