#!/usr/bin/env python3
"""Publication ROC figure (signal efficiency vs. background mistag rate,
all-background) for all four models on the exact common 400k-event
matched cohort. Reads metrics/roc_curves_data.json (already computed,
already verified event_id/process-matched -- no recomputation here).

Native 2M, Native 10M, and ZERO20 nearly overlap (that IS the ZERO20
result), so a zoomed inset panel is included covering the high-efficiency/
low-mistag corner where the four curves are otherwise indistinguishable --
this does not alter or refit the underlying curve data, it is the same
array plotted twice at two different axis scales.
"""
import argparse
import json

import matplotlib
matplotlib.use("svg")
import matplotlib.pyplot as plt  # noqa: E402

MODEL_ORDER = ["native2M", "native10M", "ParT20_2M", "ZERO20_2M"]
PALETTE = {"native2M": "#4C72B0", "native10M": "#55A868", "ParT20_2M": "#DD8452", "ZERO20_2M": "#8172B2"}
ZORDER = {"native2M": 4, "native10M": 3, "ParT20_2M": 5, "ZERO20_2M": 2}


def style(ax):
    ax.grid(True, alpha=0.3, linewidth=0.6)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--roc-curves-json", required=True)
    ap.add_argument("--out-svg", required=True)
    args = ap.parse_args()

    with open(args.roc_curves_json) as f:
        d = json.load(f)

    fig, ax = plt.subplots(figsize=(7.2, 6.2))
    for key in MODEL_ORDER:
        m = d["models"][key]
        fpr, tpr = m["roc"]["fpr"], m["roc"]["tpr"]
        ax.plot(fpr, tpr, color=PALETTE[key], linewidth=1.6, zorder=ZORDER[key],
                label=f"{m['label']} (AUC={m['auc_all_background']:.4f})")
    ax.plot([0, 1], [0, 1], color="gray", linewidth=0.8, linestyle=":", zorder=1, label="chance")

    ax.set_xlabel("background mistag rate  εB  (QCD + ttbar)")
    ax.set_ylabel("signal efficiency  εS")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_title("ROC: all-background, common 400k matched cohort")
    ax.legend(frameon=False, fontsize=8.5, loc="lower right")
    style(ax)

    # Zoomed inset: the high-efficiency / low-mistag corner where
    # native2M / native10M / ZERO20 are otherwise visually indistinguishable.
    axins = ax.inset_axes([0.10, 0.46, 0.42, 0.42])
    for key in MODEL_ORDER:
        m = d["models"][key]
        fpr, tpr = m["roc"]["fpr"], m["roc"]["tpr"]
        axins.plot(fpr, tpr, color=PALETTE[key], linewidth=1.4, zorder=ZORDER[key])
    axins.set_xlim(0.0, 0.08)
    axins.set_ylim(0.75, 1.0)
    axins.grid(True, alpha=0.3, linewidth=0.5)
    axins.tick_params(labelsize=7)
    axins.set_title("zoom: εB∈[0,0.08], εS∈[0.75,1]", fontsize=7.5)
    ax.indicate_inset_zoom(axins, edgecolor="black", linewidth=0.8)

    fig.tight_layout()
    fig.savefig(args.out_svg, format="svg")
    plt.close(fig)
    print(f"wrote {args.out_svg}")


if __name__ == "__main__":
    main()
