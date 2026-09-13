#!/usr/bin/env python3
"""Exact-event and per-Higgs HH reconstruction bar chart: Native 2M,
ParT active20 2M, ZERO20 2M. Reads metrics/model_summary.json."""
import argparse
import json

import matplotlib
matplotlib.use("svg")
import matplotlib.pyplot as plt  # noqa: E402

MODEL_ORDER = ["native2M", "ParT20_2M", "ZERO20_2M"]
MODEL_LABEL = {"native2M": "Native 2M", "ParT20_2M": "+ParT active20 (2M)", "ZERO20_2M": "+ZERO20 (2M)"}
PALETTE = {"native2M": "#4C72B0", "ParT20_2M": "#DD8452", "ZERO20_2M": "#8172B2"}


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

    fig, ax = plt.subplots(figsize=(7, 4.8))
    for i, name in enumerate(present):
        vals = [models[name]["reco"][key] for key, _ in metrics]
        offsets = [xi + (i - (len(present) - 1) / 2) * width for xi in x]
        bars = ax.bar(offsets, vals, width=width * 0.9, label=MODEL_LABEL[name], color=PALETTE[name])
        for b, v in zip(bars, vals):
            ax.annotate(f"{v:.4f}", (b.get_x() + b.get_width() / 2, v), textcoords="offset points",
                        xytext=(0, 3), ha="center", fontsize=8)

    ax.set_xticks(list(x))
    ax.set_xticklabels([label for _, label in metrics])
    ax.set_ylabel("reconstruction accuracy (assignment-defined events)")
    ax.set_ylim(0.0, 1.0)
    ax.set_title("HH reconstruction: native 2M vs. ParT active20 vs. ZERO20")
    ax.legend(frameon=False, fontsize=9)
    apply_style(ax)
    fig.tight_layout()
    fig.savefig(args.out_svg, format="svg")
    plt.close(fig)
    print(f"wrote {args.out_svg}")


if __name__ == "__main__":
    main()
