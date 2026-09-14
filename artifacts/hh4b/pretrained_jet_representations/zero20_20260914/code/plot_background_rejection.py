#!/usr/bin/env python3
"""Publication background-rejection figure: signal efficiency vs.
1/epsilon_B (log y-axis), all-background, common 400k matched cohort.
Reads metrics/roc_curves_data.json.

Each curve is drawn ONLY over its own raw finite support (n_bg_pass>=1) --
it stops exactly at the lowest signal efficiency with at least one real
background survivor; nothing beyond that point is fitted or extrapolated.
The last, sparsest segment of each curve (n_bg_pass < sparse_bg_threshold,
i.e. Poisson-limited, matching this project's own established "low-
statistics" flag) is drawn dashed and marked with hollow markers, and each
curve's terminal (lowest-efficiency) point is annotated with its raw
n_bg_pass count so the reader sees directly how few background events that
last segment rests on.
"""
import argparse
import json

import matplotlib
matplotlib.use("svg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np

MODEL_ORDER = ["native2M", "native10M", "ParT20_2M", "ZERO20_2M"]
PALETTE = {"native2M": "#4C72B0", "native10M": "#55A868", "ParT20_2M": "#DD8452", "ZERO20_2M": "#8172B2"}
ZORDER = {"native2M": 4, "native10M": 3, "ParT20_2M": 5, "ZERO20_2M": 2}


def style(ax):
    ax.grid(True, which="both", alpha=0.25, linewidth=0.5)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--roc-curves-json", required=True)
    ap.add_argument("--out-svg", required=True)
    args = ap.parse_args()

    with open(args.roc_curves_json) as f:
        d = json.load(f)
    sparse_thr = d["sparse_bg_threshold"]

    fig, ax = plt.subplots(figsize=(7.5, 5.5))
    for key in MODEL_ORDER:
        m = d["models"][key]
        eff = np.array(m["rejection"]["signal_efficiency"])
        rej = np.array(m["rejection"]["rejection"])
        n_bg_pass = np.array(m["rejection"]["n_bg_pass"])
        sparse = np.array(m["rejection"]["is_sparse"], dtype=bool)

        # solid line over the well-populated region
        dense = ~sparse
        if dense.any():
            ax.plot(eff[dense], rej[dense], color=PALETTE[key], linewidth=1.6, zorder=ZORDER[key],
                     label=f"{m['label']}")
        # dashed, hollow-marker line over the Poisson-limited tail (n_bg_pass < sparse_bg_threshold) --
        # drawn as its own segment, connecting continuously from the last dense point, never extrapolated
        # past the last point with n_bg_pass>=1
        if sparse.any():
            boundary = np.where(dense)[0].max() if dense.any() else None
            idx = np.r_[[boundary] if boundary is not None else [], np.where(sparse)[0]]
            ax.plot(eff[idx], rej[idx], color=PALETTE[key], linewidth=1.1, linestyle="--", zorder=ZORDER[key])
            ax.plot(eff[sparse], rej[sparse], "o", color=PALETTE[key], markerfacecolor="white",
                     markersize=4, zorder=ZORDER[key] + 0.1)

    # terminal-point n_bg_pass values are listed in one compact block instead of
    # per-curve inline annotations -- the four curves' sparsest endpoints cluster
    # too close together on the plot for inline text to stay legible/non-overlapping
    endpoint_lines = ["lowest εS point on each curve (n_bg_pass survivors):"]
    for key in MODEL_ORDER:
        m = d["models"][key]
        n_bg_pass = np.array(m["rejection"]["n_bg_pass"])
        eff = np.array(m["rejection"]["signal_efficiency"])
        i_min = int(np.argmin(eff))
        endpoint_lines.append(f"  {m['label']}: εS={eff[i_min]:.4f}, n_bg={n_bg_pass[i_min]}")
    ax.text(0.985, 0.985, "\n".join(endpoint_lines), transform=ax.transAxes, ha="right", va="top",
            fontsize=6.8, color="dimgray",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="lightgray", alpha=0.9))

    ax.set_yscale("log")
    ax.set_xlabel("signal efficiency  εS")
    ax.set_ylabel("background rejection  1 / εB")
    ax.set_xlim(0.0, 1.0)
    ax.set_title("Background rejection vs. signal efficiency (all-background, common cohort)")
    ax.legend(frameon=False, fontsize=8.5, loc="lower left")

    style(ax)
    fig.tight_layout()
    # explanatory note on the sparse-tail convention, placed as a figure-level
    # caption below the axes so it cannot collide with the legend or curves
    fig.text(0.5, 0.06,
             f"Dashed segments + hollow markers: <{sparse_thr} raw background survivors (Poisson-limited, not a precise rejection estimate).\n"
             "Every curve stops at its own last real background survivor -- no extrapolated tail is drawn.",
             ha="center", va="bottom", fontsize=7.5, style="italic", color="dimgray")
    fig.subplots_adjust(bottom=0.22)
    fig.savefig(args.out_svg, format="svg")
    plt.close(fig)
    print(f"wrote {args.out_svg}")


if __name__ == "__main__":
    main()
