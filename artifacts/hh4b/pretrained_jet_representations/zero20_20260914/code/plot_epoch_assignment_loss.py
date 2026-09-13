#!/usr/bin/env python3
"""Epoch-by-epoch assignment-loss comparison (native / ParT active20 /
ZERO20), reading metrics/training_history_three_way.json's "rows" list
(keys "{native,part2m,zero20}_loss/h1/assignment_loss" etc, one row per
training epoch of the real, completed runs). Plots h1+h2 assignment loss
summed per epoch, plus an epoch-0 zoom panel."""
import argparse
import json

import matplotlib
matplotlib.use("svg")
import matplotlib.pyplot as plt  # noqa: E402

MODEL_KEYS = {"Native 2M": "native", "+ParT active20 (2M)": "part2m", "+ZERO20 (2M)": "zero20"}
PALETTE = {"Native 2M": "#4C72B0", "+ParT active20 (2M)": "#DD8452", "+ZERO20 (2M)": "#8172B2"}


def apply_style(ax):
    ax.grid(True, which="major", axis="both", alpha=0.3, linewidth=0.6)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--training-history-json", required=True)
    ap.add_argument("--out-svg", required=True)
    args = ap.parse_args()

    with open(args.training_history_json) as f:
        d = json.load(f)
    rows = sorted(d["rows"], key=lambda r: r["epoch"])

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5))
    for model, prefix in MODEL_KEYS.items():
        epochs, combined = [], []
        for r in rows:
            h1 = r.get(f"{prefix}_loss/h1/assignment_loss")
            h2 = r.get(f"{prefix}_loss/h2/assignment_loss")
            if h1 is None or h2 is None:
                continue
            epochs.append(r["epoch"])
            combined.append(h1 + h2)
        if epochs:
            ax1.plot(epochs, combined, label=model, color=PALETTE[model], linewidth=1.6)

    ax1.set_xlabel("epoch")
    ax1.set_ylabel("assignment loss (h1 + h2)")
    ax1.set_title("Assignment loss vs. epoch")
    ax1.legend(frameon=False, fontsize=9)
    apply_style(ax1)

    ep0 = next((r for r in rows if r["epoch"] == 0), None)
    if ep0:
        names, vals, colors = [], [], []
        for model, prefix in MODEL_KEYS.items():
            h1 = ep0.get(f"{prefix}_loss/h1/assignment_loss")
            h2 = ep0.get(f"{prefix}_loss/h2/assignment_loss")
            if h1 is None or h2 is None:
                continue
            names.append(model); vals.append(h1 + h2); colors.append(PALETTE[model])
        ax2.bar(names, vals, color=colors)
        for i, v in enumerate(vals):
            ax2.annotate(f"{v:.3f}", (i, v), textcoords="offset points", xytext=(0, 3), ha="center", fontsize=8)
        ax2.set_ylabel("assignment loss (h1 + h2)")
        ax2.set_title("Epoch 0 only")
        ax2.tick_params(axis="x", rotation=20)
        apply_style(ax2)

    fig.tight_layout()
    fig.savefig(args.out_svg, format="svg")
    plt.close(fig)
    print(f"wrote {args.out_svg}")


if __name__ == "__main__":
    main()
