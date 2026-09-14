#!/usr/bin/env python3
"""Exact-event and per-Higgs HH reconstruction bar chart: Native 2M,
Native 10M, ParT active20 2M, ZERO20 2M. Reads metrics/model_summary.json.
Native 10M is a point estimate only (no paired-bootstrap CI was computed
for it -- flagged in the plot, not just the caption)."""
import argparse
import json

import matplotlib
matplotlib.use("svg")
import matplotlib.pyplot as plt  # noqa: E402

MODEL_ORDER = ["native2M", "native10M", "ParT20_2M", "ZERO20_2M"]
MODEL_LABEL = {"native2M": "Native 2M", "native10M": "Native 10M*",
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

    metrics = [("exact_event_hh_reconstruction_efficiency", "exact-event (both Higgs correct)"),
               ("higgs_assignment_pairing_accuracy", "per-Higgs pairing accuracy")]
    present = [m for m in MODEL_ORDER if m in models and models[m]["reco"]]
    x = range(len(metrics))
    width = 0.8 / len(present)

    fig, ax = plt.subplots(figsize=(8.3, 4.8))
    for i, name in enumerate(present):
        vals = [models[name]["reco"][key] for key, _ in metrics]
        offsets = [xi + (i - (len(present) - 1) / 2) * width for xi in x]
        bars = ax.bar(offsets, vals, width=width * 0.9, label=MODEL_LABEL[name], color=PALETTE[name])
        for b, v in zip(bars, vals):
            # rotated + offset labels: the native2M/native10M/ZERO20 bars sit within
            # ~0.005 of each other in the per-Higgs group, so horizontal labels collide
            ax.annotate(f"{v:.4f}", (b.get_x() + b.get_width() / 2, v), textcoords="offset points",
                        xytext=(0, 4), ha="center", va="bottom", fontsize=7.5, rotation=90)

    ax.set_xticks(list(x))
    ax.set_xticklabels([label for _, label in metrics])
    ax.set_ylabel("reconstruction accuracy (assignment-defined events)")
    ax.set_ylim(0.0, 1.04)
    ax.set_title("HH reconstruction: native 2M/10M vs. ParT active20 vs. ZERO20")
    ax.legend(frameon=False, fontsize=9, loc="upper left", bbox_to_anchor=(1.01, 1.0))
    apply_style(ax)
    fig.tight_layout()
    if "native10M" in present:
        fig.subplots_adjust(bottom=0.16, right=0.78)
        fig.text(0.5, 0.02, "*Native 10M: point estimate only, no paired-bootstrap CI computed",
                 ha="center", va="bottom", fontsize=7.5, style="italic", color="dimgray")
    else:
        fig.subplots_adjust(right=0.78)
    fig.savefig(args.out_svg, format="svg")
    plt.close(fig)
    print(f"wrote {args.out_svg}")


if __name__ == "__main__":
    main()
