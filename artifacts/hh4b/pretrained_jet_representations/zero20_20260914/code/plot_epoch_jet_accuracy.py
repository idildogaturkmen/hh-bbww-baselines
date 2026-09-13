#!/usr/bin/env python3
"""Validation jet-accuracy vs. epoch (native / ParT active20 / ZERO20),
reading metrics/training_history_three_way.json's
"{native,part2m,zero20}_validation_average_jet_accuracy" fields."""
import argparse
import json

import matplotlib
matplotlib.use("svg")
import matplotlib.pyplot as plt  # noqa: E402

MODEL_KEYS = {"Native 2M": "native", "+ParT active20 (2M)": "part2m", "+ZERO20 (2M)": "zero20"}
PALETTE = {"Native 2M": "#4C72B0", "+ParT active20 (2M)": "#DD8452", "+ZERO20 (2M)": "#8172B2"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--training-history-json", required=True)
    ap.add_argument("--out-svg", required=True)
    args = ap.parse_args()

    with open(args.training_history_json) as f:
        d = json.load(f)
    rows = sorted(d["rows"], key=lambda r: r["epoch"])

    fig, ax = plt.subplots(figsize=(7, 4.8))
    for model, prefix in MODEL_KEYS.items():
        epochs, vals = [], []
        for r in rows:
            v = r.get(f"{prefix}_validation_average_jet_accuracy")
            if v is None:
                continue
            epochs.append(r["epoch"])
            vals.append(v)
        if epochs:
            ax.plot(epochs, vals, label=model, color=PALETTE[model], linewidth=1.6)
            ax.plot(epochs[-1], vals[-1], "o", color=PALETTE[model], markersize=4)

    ax.set_xlabel("epoch")
    ax.set_ylabel("validation_average_jet_accuracy")
    ax.set_title("Validation jet-assignment accuracy vs. epoch")
    ax.legend(frameon=False, fontsize=9)
    ax.grid(True, which="major", axis="both", alpha=0.3, linewidth=0.6)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    fig.savefig(args.out_svg, format="svg")
    plt.close(fig)
    print(f"wrote {args.out_svg}")


if __name__ == "__main__":
    main()
