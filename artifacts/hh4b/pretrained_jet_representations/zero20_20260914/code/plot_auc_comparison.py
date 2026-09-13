#!/usr/bin/env python3
"""AUC comparison bar chart: Native SPA-Net 2M / 10M, SPA-Net + ParT
active20 2M, SPA-Net + ZERO20 2M -- all-background/QCD/ttbar. Reads
metrics/model_summary.json (already-assembled point estimates, no
recomputation here). Matplotlib SVG, muted categorical palette, no
external stylesheet."""
import argparse
import json

import matplotlib
matplotlib.use("svg")
import matplotlib.pyplot as plt  # noqa: E402

MODEL_ORDER = ["native2M", "native10M", "ParT20_2M", "ZERO20_2M"]
MODEL_LABEL = {"native2M": "Native 2M", "native10M": "Native 10M",
               "ParT20_2M": "+ParT active20 (2M)", "ZERO20_2M": "+ZERO20 (2M)"}
PALETTE = {"native2M": "#4C72B0", "native10M": "#55A868",
           "ParT20_2M": "#DD8452", "ZERO20_2M": "#8172B2"}


def apply_style(ax):
    ax.grid(True, which="major", axis="y", alpha=0.3, linewidth=0.6)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-summary-json", required=True)
    ap.add_argument("--out-svg", required=True)
    args = ap.parse_args()

    with open(args.model_summary_json) as f:
        models = json.load(f)

    metrics = ["all_background", "qcd", "ttbar"]
    x = range(len(metrics))
    present = [m for m in MODEL_ORDER if m in models]
    width = 0.8 / len(present)

    fig, ax = plt.subplots(figsize=(8, 4.8))
    for i, name in enumerate(present):
        vals = [models[name]["auc"][m] for m in metrics]
        offsets = [xi + (i - (len(present) - 1) / 2) * width for xi in x]
        bars = ax.bar(offsets, vals, width=width * 0.9, label=MODEL_LABEL[name], color=PALETTE[name])
        for b, v in zip(bars, vals):
            ax.annotate(f"{v:.4f}", (b.get_x() + b.get_width() / 2, v), textcoords="offset points",
                        xytext=(0, 2), ha="center", fontsize=6, rotation=90, va="bottom")

    ax.set_xticks(list(x))
    ax.set_xticklabels(["all-background", "QCD", "ttbar"])
    ax.set_ylabel("AUC")
    ax.set_ylim(0.90, 0.985)
    ax.set_title("SPA-Net classification AUC: native scaling vs. frozen-embedding ablations")
    ax.legend(frameon=False, fontsize=8, ncol=2)
    apply_style(ax)
    fig.tight_layout()
    fig.savefig(args.out_svg, format="svg")
    plt.close(fig)
    print(f"wrote {args.out_svg}")


if __name__ == "__main__":
    main()
