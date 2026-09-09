#!/usr/bin/env python3
"""Figure generator: reads a comparison_result.json (evaluate_multi_model.py
output) and writes SVG figures -- following this project's established
results/<name>/plots/*.svg convention. A minimal, self-contained plot style
is used (grid, muted palette, explicit axis labels) rather than importing
scripts/plotting/hh4b_cms_style.py directly, since that module lives in a
separate repository checkout from this package and this package is meant
to be portable/self-contained; the visual conventions (muted categorical
palette, light grid, explicit units) are followed by hand for consistency.

Writes only what is present in the input JSON -- a missing block is skipped
with a printed note, never plotted as a fabricated placeholder curve.
"""
import argparse
import json
import os

import matplotlib
matplotlib.use("svg")
import matplotlib.pyplot as plt  # noqa: E402

TARGET_EFFICIENCIES = [0.10, 0.075, 0.05, 0.04, 0.03]
PALETTE = {"control": "#4C72B0", "test": "#DD8452", "native10m": "#55A868"}


def apply_style(ax):
    ax.grid(True, which="major", axis="both", alpha=0.3, linewidth=0.6)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def fig_delta_auc_forest(result, out_path):
    rows = []
    for name, block in (("all_background", result.get("paired_bootstrap_delta_auc", {}).get("all_background")),
                         ("qcd", result.get("paired_bootstrap_delta_auc", {}).get("qcd")),
                         ("ttbar", result.get("paired_bootstrap_delta_auc", {}).get("ttbar"))):
        if block:
            rows.append((f"delta_AUC.{name}", block))
    for eff in TARGET_EFFICIENCIES:
        block = result.get("paired_bootstrap_delta_rejection", {}).get(f"epsS_{eff}")
        if block:
            rows.append((f"delta_rejection.epsS_{eff}", block))
    if not rows:
        print(f"SKIP {out_path}: no paired_bootstrap_delta_* blocks present")
        return

    fig, ax = plt.subplots(figsize=(7, 0.55 * len(rows) + 1.2))
    ys = list(range(len(rows)))[::-1]
    for y, (label, block) in zip(ys, rows):
        lo, hi, obs = block["ci_low"], block["ci_high"], block["observed_delta_test_minus_control"]
        color = "#C44E52" if block["excludes_zero"] and obs < 0 else ("#55A868" if block["excludes_zero"] else "#8172B2")
        ax.plot([lo, hi], [y, y], color=color, linewidth=2, solid_capstyle="round")
        ax.plot([obs], [y], "o", color=color, markersize=5)
    ax.axvline(0, color="black", linewidth=0.8, linestyle="--")
    ax.set_yticks(ys)
    ax.set_yticklabels([r[0] for r in rows], fontsize=9)
    ax.set_xlabel("paired bootstrap delta (TEST − CONTROL), 95% CI")
    ax.set_title("Paired bootstrap deltas (green=excludes zero, positive; red=excludes zero, negative;\npurple=CI includes zero)", fontsize=9)
    apply_style(ax)
    fig.tight_layout()
    fig.savefig(out_path, format="svg")
    plt.close(fig)


def fig_auc_comparison_bar(result, out_path):
    models = [m for m in ("control", "test", "native10m") if result.get("auc", {}).get(m)]
    if not models:
        print(f"SKIP {out_path}: no auc block present")
        return
    backgrounds = ["all_background", "qcd", "ttbar"]
    fig, ax = plt.subplots(figsize=(6, 4))
    width = 0.8 / len(models)
    x = range(len(backgrounds))
    for i, m in enumerate(models):
        vals = [result["auc"][m].get(b) for b in backgrounds]
        xs = [xi + i * width for xi in x]
        ax.bar(xs, vals, width=width, label=m, color=PALETTE.get(m, "#888888"))
    ax.set_xticks([xi + width * (len(models) - 1) / 2 for xi in x])
    ax.set_xticklabels(backgrounds)
    ax.set_ylabel("ROC AUC")
    ax.set_ylim(0.5, 1.0)
    ax.legend(fontsize=8)
    apply_style(ax)
    fig.tight_layout()
    fig.savefig(out_path, format="svg")
    plt.close(fig)


def fig_jet_multiplicity_auc(result, out_path):
    strata = result.get("jet_multiplicity_stratified_auc", {})
    names = [k for k in ("le4", "eq4", "eq5", "ge5", "ge6") if isinstance(strata.get(k), dict) and "auc_control" in strata[k]]
    if not names:
        print(f"SKIP {out_path}: no jet_multiplicity_stratified_auc data present")
        return
    fig, ax = plt.subplots(figsize=(6, 4))
    x = range(len(names))
    ax.plot(x, [strata[n]["auc_control"] for n in names], "o-", color=PALETTE["control"], label="control")
    ax.plot(x, [strata[n]["auc_test"] for n in names], "s-", color=PALETTE["test"], label="test")
    ax.set_xticks(list(x))
    ax.set_xticklabels(names)
    ax.set_xlabel("jet-multiplicity stratum")
    ax.set_ylabel("all-background AUC")
    ax.legend(fontsize=8)
    apply_style(ax)
    fig.tight_layout()
    fig.savefig(out_path, format="svg")
    plt.close(fig)


def fig_rejection_vs_epss(result, out_path):
    fig, ax = plt.subplots(figsize=(6, 4))
    plotted = False
    for model, color in (("control", PALETTE["control"]), ("test", PALETTE["test"])):
        xs, ys = [], []
        for eff in TARGET_EFFICIENCIES:
            wp = result.get("rejection_at_fixed_efficiency", {}).get(model, {}).get(f"epsS_{eff}", {}).get("all_background")
            if wp and wp.get("rejection") is not None:
                xs.append(eff)
                ys.append(wp["rejection"])
        if xs:
            ax.plot(xs, ys, "o-", color=color, label=model)
            plotted = True
    if not plotted:
        print(f"SKIP {out_path}: no rejection_at_fixed_efficiency data present")
        plt.close(fig)
        return
    ax.set_yscale("log")
    ax.set_xlabel("target signal efficiency epsS")
    ax.set_ylabel("all-background rejection R = 1/epsB")
    ax.invert_xaxis()
    ax.legend(fontsize=8)
    apply_style(ax)
    fig.tight_layout()
    fig.savefig(out_path, format="svg")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--comparison-result", required=True)
    ap.add_argument("--out-dir", required=True)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    with open(args.comparison_result) as f:
        result = json.load(f)

    fig_delta_auc_forest(result, os.path.join(args.out_dir, "delta_auc_forest_plot.svg"))
    fig_auc_comparison_bar(result, os.path.join(args.out_dir, "auc_comparison_bar.svg"))
    fig_jet_multiplicity_auc(result, os.path.join(args.out_dir, "jet_multiplicity_stratified_auc.svg"))
    fig_rejection_vs_epss(result, os.path.join(args.out_dir, "rejection_vs_epsS.svg"))

    print(json.dumps({"out_dir": args.out_dir,
                       "figures_written": sorted(os.listdir(args.out_dir))}, indent=2))


if __name__ == "__main__":
    main()
