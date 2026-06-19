#!/usr/bin/env python3

from pathlib import Path
import json
import pandas as pd
import matplotlib.pyplot as plt


def markdown_table(df, cols, float_digits=5):
    """Make a simple Markdown table without requiring tabulate."""
    small = df[cols].copy()

    def fmt(x):
        if pd.isna(x):
            return ""
        if isinstance(x, float):
            return f"{x:.{float_digits}f}"
        return str(x)

    lines = []
    header = "| " + " | ".join(cols) + " |"
    sep = "| " + " | ".join(["---"] * len(cols)) + " |"
    lines.append(header)
    lines.append(sep)

    for _, row in small.iterrows():
        lines.append("| " + " | ".join(fmt(row[c]) for c in cols) + " |")

    return "\n".join(lines)


def get_args(results):
    """Handle slightly different possible JSON structures."""
    if not isinstance(results, dict):
        return {}
    return results.get("args", {}) or {}


def main():
    latest_path = Path("outputs/DNN_Hyperparameter_Search/latest_search_root.txt")
    if not latest_path.exists():
        raise FileNotFoundError("Could not find outputs/DNN_Hyperparameter_Search/latest_search_root.txt")

    root = Path(latest_path.read_text().strip())
    comparison_dir = root / "comparison"
    comparison_dir.mkdir(parents=True, exist_ok=True)

    rows = []

    for run_dir in sorted(root.iterdir()):
        if not run_dir.is_dir():
            continue
        if run_dir.name == "comparison":
            continue

        perjet_path = run_dir / "outputs" / "perjet_metrics_test.csv"
        mass_path = run_dir / "outputs" / "event_mass_summary_test.csv"
        results_path = run_dir / "outputs" / "results.json"

        if not perjet_path.exists() or not mass_path.exists():
            print(f"Skipping incomplete run: {run_dir.name}")
            continue

        perjet = pd.read_csv(perjet_path)
        mass = pd.read_csv(mass_path)

        results = {}
        args = {}
        if results_path.exists():
            with open(results_path) as f:
                results = json.load(f)
            args = get_args(results)

        best_epoch = results.get("best_epoch", None)
        best_val_loss = results.get("best_val_loss", None)
        n_features = results.get("n_features", None)
        feature_set = results.get("feature_set", args.get("feature_set", "all"))
        dnn_calib_offset = results.get("dnn_calib_offset", None)

        # Some runs have both dnn and dnn_calibrated.
        dnn_methods = [m for m in ["dnn", "dnn_calibrated"] if m in set(perjet["method"])]

        for method in dnn_methods:
            pj = perjet[perjet["method"] == method].iloc[0]
            mb = mass[mass["method"] == method].iloc[0]

            rows.append({
                "run": run_dir.name,
                "method": method,
                "hidden": args.get("hidden", None),
                "dropout": args.get("dropout", None),
                "huber_beta": args.get("huber_beta", None),
                "feature_set": feature_set,
                "n_features": n_features,
                "best_epoch": best_epoch,
                "best_val_loss": best_val_loss,
                "dnn_calib_offset": dnn_calib_offset,
                "target_mae": pj["target_mae"],
                "target_rmse": pj["target_rmse"],
                "closure_median": pj["closure_median"],
                "closure_width68": pj["closure_width68"],
                "mbb_median": mb["median"],
                "mbb_width68": mb["width68"],
                "frac_90_140": mb["frac_90_140"],
                "frac_100_150": mb["frac_100_150"],
                "median_abs_offset_from_125": mb["median_abs_offset_from_125"],
            })

    if not rows:
        raise RuntimeError(f"No completed DNN runs found under {root}")

    summary = pd.DataFrame(rows)

    # Ranking: lower target MAE is better, lower m_bb width is better,
    # higher 90-140 fraction is better.
    summary["rank_target_mae"] = summary["target_mae"].rank(method="min", ascending=True)
    summary["rank_mbb_width68"] = summary["mbb_width68"].rank(method="min", ascending=True)
    summary["rank_frac_90_140"] = summary["frac_90_140"].rank(method="min", ascending=False)
    summary["rank_total"] = (
        summary["rank_target_mae"]
        + summary["rank_mbb_width68"]
        + summary["rank_frac_90_140"]
    )

    summary = summary.sort_values(["rank_total", "target_mae", "mbb_width68"])

    csv_out = comparison_dir / "dnn_hyperparameter_summary.csv"
    md_out = comparison_dir / "dnn_hyperparameter_summary.md"

    summary.to_csv(csv_out, index=False)

    display_cols = [
        "run",
        "method",
        "hidden",
        "dropout",
        "huber_beta",
        "feature_set",
        "n_features",
        "best_epoch",
        "best_val_loss",
        "target_mae",
        "target_rmse",
        "closure_median",
        "closure_width68",
        "mbb_median",
        "mbb_width68",
        "frac_90_140",
        "frac_100_150",
        "median_abs_offset_from_125",
        "rank_total",
    ]

    with open(md_out, "w") as f:
        f.write("# DNN Hyperparameter Search Summary\n\n")
        f.write(f"Search root: `{root}`\n\n")
        f.write(markdown_table(summary, display_cols, float_digits=5))
        f.write("\n\n")
        f.write("Ranking uses target MAE, m_bb width68, and fraction in 90-140 GeV.\n")

    print("\n=== DNN hyperparameter summary ===")
    print(summary[display_cols].to_string(index=False))
    print(f"\nSaved CSV: {csv_out}")
    print(f"Saved Markdown: {md_out}")

    plot_df = summary.copy()
    plot_df["label"] = plot_df["run"] + " / " + plot_df["method"]

    # Plot 1: target MAE
    df = plot_df.sort_values("target_mae", ascending=True)
    plt.figure(figsize=(11, 5.5))
    plt.barh(df["label"], df["target_mae"])
    plt.xlabel("Target MAE, lower is better")
    plt.title("DNN hyperparameter comparison: target MAE")
    plt.tight_layout()
    out = comparison_dir / "compare_target_mae.png"
    plt.savefig(out, dpi=180)
    plt.close()
    print(f"Saved plot: {out}")

    # Plot 2: m_bb width68
    df = plot_df.sort_values("mbb_width68", ascending=True)
    plt.figure(figsize=(11, 5.5))
    plt.barh(df["label"], df["mbb_width68"])
    plt.xlabel("m_bb width68 [GeV], lower is better")
    plt.title("DNN hyperparameter comparison: m_bb resolution")
    plt.tight_layout()
    out = comparison_dir / "compare_mbb_width68.png"
    plt.savefig(out, dpi=180)
    plt.close()
    print(f"Saved plot: {out}")

    # Plot 3: fraction in 90-140 window
    df = plot_df.sort_values("frac_90_140", ascending=True)
    plt.figure(figsize=(11, 5.5))
    plt.barh(df["label"], df["frac_90_140"])
    plt.xlabel("Fraction in 90-140 GeV, higher is better")
    plt.title("DNN hyperparameter comparison: Higgs-window efficiency")
    plt.tight_layout()
    out = comparison_dir / "compare_frac_90_140.png"
    plt.savefig(out, dpi=180)
    plt.close()
    print(f"Saved plot: {out}")

    # Plot 4: target MAE vs m_bb width68
    plt.figure(figsize=(8, 6))
    plt.scatter(plot_df["target_mae"], plot_df["mbb_width68"])
    for _, row in plot_df.iterrows():
        short_label = row["run"].replace("_", "\n")
        if row["method"] == "dnn_calibrated":
            short_label += "\ncalib"
        plt.annotate(short_label, (row["target_mae"], row["mbb_width68"]), fontsize=7)
    plt.xlabel("Target MAE, lower is better")
    plt.ylabel("m_bb width68 [GeV], lower is better")
    plt.title("DNN tradeoff: per-jet accuracy vs m_bb resolution")
    plt.tight_layout()
    out = comparison_dir / "scatter_target_mae_vs_mbb_width68.png"
    plt.savefig(out, dpi=180)
    plt.close()
    print(f"Saved plot: {out}")


if __name__ == "__main__":
    main()
