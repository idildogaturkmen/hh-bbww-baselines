"""
Signal-only pT-threshold tradeoff study for truth-matched H->bb.

This script uses the detector-response diagnostic output:

  outputs/hbb_detector_response/event_response_diagnostics.csv

It studies how cuts on the lower-pT truth-matched b jet affect:
  - signal retention,
  - mean/median reconstructed m_bb,
  - m_bb mass-window fractions,
  - matched-jet reco/gen pT response.

Important caveat:
  This is a signal-only diagnostic using truth-matched b jets.
  It is not yet a directly data-applicable selection, because true H->bb
  matched jets are known only in simulation.
"""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


INPUT_CSV = Path("outputs/hbb_detector_response/event_response_diagnostics.csv")
OUTDIR = Path("outputs/hbb_pt_threshold_tradeoff")
PLOT_DIR = Path("outputs/plots/hbb_pt_threshold_tradeoff")
SUMMARY_MD = Path("RESULTS_HBB_PT_THRESHOLD_TRADEOFF.md")

MH = 125.0
DEFAULT_THRESHOLDS = [20, 25, 30, 40, 50, 60, 70, 80, 90, 110]


REQUIRED_COLUMNS = [
    "matched_mbb",
    "lower_matched_jet_pt",
    "lower_response_reco_pt_over_gen_pt",
    "average_response_reco_pt_over_gen_pt",
    "eta_category",
]


def ensure_dirs():
    OUTDIR.mkdir(parents=True, exist_ok=True)
    PLOT_DIR.mkdir(parents=True, exist_ok=True)


def check_input(df):
    missing = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    if missing:
        raise ValueError(
            "Input CSV is missing required columns: "
            + ", ".join(missing)
            + "\nRerun scripts/diagnose_hbb_detector_response.py first."
        )


def finite(values):
    arr = np.asarray(values, dtype=float)
    return arr[np.isfinite(arr)]


def bootstrap_median_interval(values, rng, n_bootstrap):
    '''
    Returns the 16th, 50th, and 84th percentiles of the bootstrap distribution of the median of the input values.
    '''
    values = finite(values)
    n = len(values)
    if n == 0:
        return np.nan, np.nan, np.nan
    if n == 1 or n_bootstrap <= 0:
        med = float(np.median(values))
        return med, med, med

    meds = np.empty(n_bootstrap, dtype=float)
    for i in range(n_bootstrap):
        sample = values[rng.integers(0, n, size=n)]
        meds[i] = np.median(sample)

    return (
        float(np.percentile(meds, 16)),
        float(np.median(meds)),
        float(np.percentile(meds, 84)),
    )


def summarize_subset(part, n_total, cut_label, cut_value, threshold_for_plot, rng, n_bootstrap):
    '''
    Summarizes the given subset of events, returning a dictionary of summary statistics.
     - part: DataFrame subset to summarize
     - n_total: total number of events in the original dataset (for retention calculation)
     - cut_label: string label for the cut applied to this subset (e.g. "lower_pT_ge_30")
     - cut_value: numeric value of the cut applied (e.g. 30.0), or np.nan for no cut
     - threshold_for_plot: numeric value to use for plotting on x-axis (e.g. 30.0), or 0.0 for no cut
     - rng: numpy random generator for bootstrapping
     - n_bootstrap: number of bootstrap samples to use for median confidence interval
     Returns a dictionary with summary statistics for this subset.
     '''

    mbb = finite(part["matched_mbb"].to_numpy())
    lower_pt = finite(part["lower_matched_jet_pt"].to_numpy())
    lower_response = finite(part["lower_response_reco_pt_over_gen_pt"].to_numpy())
    avg_response = finite(part["average_response_reco_pt_over_gen_pt"].to_numpy())

    n = len(mbb)
    boot16, boot50, boot84 = bootstrap_median_interval(mbb, rng, n_bootstrap)

    if n == 0:
        return {
            "cut_label": cut_label,
            "min_lower_matched_jet_pt_cut": cut_value,
            "threshold_for_plot": threshold_for_plot,
            "n_events": 0,
            "signal_retention": 0.0,
            "mean_mbb": np.nan,
            "median_mbb": np.nan,
            "median_mbb_boot_p16": np.nan,
            "median_mbb_boot_p50": np.nan,
            "median_mbb_boot_p84": np.nan,
            "p16_mbb": np.nan,
            "p84_mbb": np.nan,
            "std_mbb": np.nan,
            "median_abs_mbb_minus_125": np.nan,
            "mean_abs_event_mbb_minus_125": np.nan,
            "frac_90_140": np.nan,
            "frac_100_150": np.nan,
            "frac_105_145": np.nan,
            "median_lower_matched_jet_pt": np.nan,
            "median_lower_response": np.nan,
            "median_average_response": np.nan,
            "frac_both_central": np.nan,
        }

    return {
        "cut_label": cut_label,
        "min_lower_matched_jet_pt_cut": cut_value,
        "threshold_for_plot": threshold_for_plot,
        "n_events": int(n),
        "signal_retention": float(n / n_total),
        "mean_mbb": float(np.mean(mbb)),
        "median_mbb": float(np.median(mbb)),
        "median_mbb_boot_p16": boot16,
        "median_mbb_boot_p50": boot50,
        "median_mbb_boot_p84": boot84,
        "p16_mbb": float(np.percentile(mbb, 16)),
        "p84_mbb": float(np.percentile(mbb, 84)),
        "std_mbb": float(np.std(mbb)),
        "median_abs_mbb_minus_125": float(abs(np.median(mbb) - MH)),
        "mean_abs_event_mbb_minus_125": float(np.mean(np.abs(mbb - MH))),
        "frac_90_140": float(np.mean((mbb >= 90.0) & (mbb <= 140.0))),
        "frac_100_150": float(np.mean((mbb >= 100.0) & (mbb <= 150.0))),
        "frac_105_145": float(np.mean((mbb >= 105.0) & (mbb <= 145.0))),
        "median_lower_matched_jet_pt": float(np.median(lower_pt)) if len(lower_pt) else np.nan,
        "median_lower_response": float(np.median(lower_response)) if len(lower_response) else np.nan,
        "median_average_response": float(np.median(avg_response)) if len(avg_response) else np.nan,
        "frac_both_central": float(np.mean(part["eta_category"].astype(str).str.contains("both_central"))),
    }


def build_summary(df, thresholds, n_bootstrap, seed):
    '''
Builds a summary DataFrame containing statistics for each lower-matched-jet-pT cut defined in thresholds.
 - df: input DataFrame with event-level diagnostics
 - thresholds: list of numeric pT thresholds to apply to the lower matched jet (e.g. [20, 30, 40])
 - n_bootstrap: number of bootstrap samples to use for estimating median confidence intervals
 - seed: random seed for reproducibility of bootstrapping
 Returns a DataFrame with one row per cut (including no cut) and columns for various summary statistics 
 (e.g. mean/median m_bb, signal retention, mass-window fractions, response metrics).
    '''

    rng = np.random.default_rng(seed)
    n_total = len(df)
    rows = []

    rows.append(
        summarize_subset(
            df,
            n_total=n_total,
            cut_label="no_cut",
            cut_value=np.nan,
            threshold_for_plot=0.0,
            rng=rng,
            n_bootstrap=n_bootstrap,
        )
    )

    for cut in thresholds:
        part = df[df["lower_matched_jet_pt"] >= cut]
        rows.append(
            summarize_subset(
                part,
                n_total=n_total,
                cut_label=f"lower_pT_ge_{cut:g}",
                cut_value=float(cut),
                threshold_for_plot=float(cut),
                rng=rng,
                n_bootstrap=n_bootstrap,
            )
        )

    return pd.DataFrame(rows)


def plot_mass_vs_cut(summary):
    plt.figure(figsize=(8.0, 5.2))
    plt.plot(summary["threshold_for_plot"], summary["median_mbb"], marker="o", label="median m_bb")
    plt.plot(summary["threshold_for_plot"], summary["mean_mbb"], marker="s", label="mean m_bb")
    plt.axhline(MH, linestyle="--", linewidth=1.1, label="125 GeV")
    plt.xlabel("Lower truth-matched b-jet pT cut [GeV]; 0 = no cut")
    plt.ylabel("Reconstructed truth-matched m_bb [GeV]")
    plt.title("m_bb scale vs lower matched b-jet pT cut")
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(PLOT_DIR / "median_mean_mbb_vs_pt_cut.png", dpi=200)
    plt.close()


def plot_retention_vs_cut(summary):
    plt.figure(figsize=(8.0, 5.2))
    plt.plot(summary["threshold_for_plot"], summary["signal_retention"], marker="o")
    plt.xlabel("Lower truth-matched b-jet pT cut [GeV]; 0 = no cut")
    plt.ylabel("Fraction of usable HH events retained")
    plt.ylim(0, 1.05)
    plt.title("Signal retention vs lower matched b-jet pT cut")
    plt.tight_layout()
    plt.savefig(PLOT_DIR / "signal_retention_vs_pt_cut.png", dpi=200)
    plt.close()


def plot_window_fractions(summary):
    plt.figure(figsize=(8.0, 5.2))
    plt.plot(summary["threshold_for_plot"], summary["frac_90_140"], marker="o", label="90 < m_bb < 140")
    plt.plot(summary["threshold_for_plot"], summary["frac_100_150"], marker="s", label="100 < m_bb < 150")
    plt.plot(summary["threshold_for_plot"], summary["frac_105_145"], marker="^", label="105 < m_bb < 145")
    plt.xlabel("Lower truth-matched b-jet pT cut [GeV]; 0 = no cut")
    plt.ylabel("Fraction of retained events")
    plt.ylim(0, 1.05)
    plt.title("Mass-window fraction vs lower matched b-jet pT cut")
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(PLOT_DIR / "mass_window_fraction_vs_pt_cut.png", dpi=200)
    plt.close()


def plot_response_vs_cut(summary):
    plt.figure(figsize=(8.0, 5.2))
    plt.plot(summary["threshold_for_plot"], summary["median_lower_response"], marker="o", label="lower-jet response")
    plt.plot(summary["threshold_for_plot"], summary["median_average_response"], marker="s", label="average two-jet response")
    plt.axhline(1.0, linestyle="--", linewidth=1.1, label="response = 1")
    plt.xlabel("Lower truth-matched b-jet pT cut [GeV]; 0 = no cut")
    plt.ylabel("Reco jet pT / gen b pT")
    plt.title("Matched b-jet response vs lower matched b-jet pT cut")
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(PLOT_DIR / "response_vs_pt_cut.png", dpi=200)
    plt.close()


def plot_mass_distributions(df, cuts_to_show):
    plt.figure(figsize=(8.0, 5.2))
    bins = np.linspace(0, 260, 80)

    for cut in cuts_to_show:
        if cut is None:
            part = df
            label = "no cut"
        else:
            part = df[df["lower_matched_jet_pt"] >= cut]
            label = f"lower pT >= {cut:g} GeV"

        values = finite(part["matched_mbb"].to_numpy())
        if len(values) == 0:
            continue

        plt.hist(values, bins=bins, histtype="step", density=True, linewidth=1.4, label=label)

    plt.axvline(MH, linestyle="--", linewidth=1.1, label="125 GeV")
    plt.xlabel("Reconstructed truth-matched m_bb [GeV]")
    plt.ylabel("Normalized event density")
    plt.title("m_bb distributions after lower matched b-jet pT cuts")
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(PLOT_DIR / "mbb_distributions_for_selected_pt_cuts.png", dpi=200)
    plt.close()


def fmt(x, digits=3):
    if pd.isna(x):
        return "—"
    return f"{x:.{digits}f}"


def write_markdown(summary, args):
    no_cut = summary.iloc[0]
    test_cuts = summary.iloc[1:].copy()

    closest = test_cuts.loc[test_cuts["median_abs_mbb_minus_125"].idxmin()]
    crossing = test_cuts[test_cuts["median_mbb"] >= MH]
    first_crossing = crossing.iloc[0] if len(crossing) else None

    lines = [
        "# H->bb pT Threshold Tradeoff Study",
        "",
        "This is a signal-only follow-up to the detector-response diagnostic. It studies how cuts on the lower-pT truth-matched H->bb b jet change the reconstructed m_bb scale and the retained signal fraction.",
        "",
        "**Important caveat:** this uses truth-matched b jets, so it is not yet a directly data-applicable selection. For a background/data-like version, the analogous observable would need to be something like the lower pT of the selected b-tagged jet pair.",
        "",
        "## Input",
        "",
        f"- Input CSV: `{args.input}`",
        f"- Total usable truth-matched HH events in input: {int(no_cut['n_events'])}",
        "",
        "## Main takeaway",
        "",
        f"- With no additional lower-b-jet pT cut, the median matched m_bb is {no_cut['median_mbb']:.2f} GeV.",
        f"- The tested cut with median m_bb closest to 125 GeV is `{closest['cut_label']}`, with median m_bb = {closest['median_mbb']:.2f} GeV and signal retention = {closest['signal_retention']:.3f}.",
    ]

    if first_crossing is not None:
        lines.append(
            f"- The first tested threshold where the median m_bb reaches or exceeds 125 GeV is `{first_crossing['cut_label']}`, retaining {first_crossing['signal_retention']:.3f} of usable HH events."
        )
    else:
        lines.append("- None of the tested thresholds made the median m_bb reach 125 GeV.")

    lines.extend(
        [
            "",
            "This supports the interpretation that harder lower-b-jet pT requirements can recover the m_bb scale, but the efficiency cost can be large.",
            "",
            "## Summary table",
            "",
            "| Cut | n events | Signal retention | Mean m_bb [GeV] | Median m_bb [GeV] | Bootstrap median 16-84% [GeV] | 90 < m_bb < 140 | 100 < m_bb < 150 | Median lower response |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )

    for _, row in summary.iterrows():
        lines.append(
            f"| {row['cut_label']} | "
            f"{int(row['n_events'])} | "
            f"{fmt(row['signal_retention'], 3)} | "
            f"{fmt(row['mean_mbb'], 2)} | "
            f"{fmt(row['median_mbb'], 2)} | "
            f"{fmt(row['median_mbb_boot_p16'], 2)}–{fmt(row['median_mbb_boot_p84'], 2)} | "
            f"{fmt(row['frac_90_140'], 3)} | "
            f"{fmt(row['frac_100_150'], 3)} | "
            f"{fmt(row['median_lower_response'], 3)} |"
        )

    lines.extend(
        [
            "",
            "## How to interpret",
            "",
            "- If the median m_bb increases with the lower-b-jet pT cut, then the low inclusive mass is strongly tied to soft matched b jets.",
            "- If the signal retention drops quickly, then a hard pT cut is not a simple fix, even if it improves the mass peak.",
            "- If the response also increases with the pT cut, this supports the detector/reconstruction-response explanation.",
            "- The mean can become much larger than the median at high pT cuts because the remaining sample has broad high-mass tails, so the median is usually the more stable mass-scale diagnostic.",
            "",
            "## Outputs",
            "",
            "- `outputs/hbb_pt_threshold_tradeoff/pt_threshold_summary.csv`",
            "- `outputs/hbb_pt_threshold_tradeoff/summary.json`",
            "- `outputs/plots/hbb_pt_threshold_tradeoff/median_mean_mbb_vs_pt_cut.png`",
            "- `outputs/plots/hbb_pt_threshold_tradeoff/signal_retention_vs_pt_cut.png`",
            "- `outputs/plots/hbb_pt_threshold_tradeoff/mass_window_fraction_vs_pt_cut.png`",
            "- `outputs/plots/hbb_pt_threshold_tradeoff/response_vs_pt_cut.png`",
            "- `outputs/plots/hbb_pt_threshold_tradeoff/mbb_distributions_for_selected_pt_cuts.png`",
        ]
    )

    SUMMARY_MD.write_text("\n".join(lines) + "\n")


def parse_args():
    parser = argparse.ArgumentParser(description="Signal-only lower-b-jet pT threshold tradeoff study.")
    parser.add_argument("--input", type=Path, default=INPUT_CSV)
    parser.add_argument("--thresholds", type=float, nargs="+", default=DEFAULT_THRESHOLDS)
    parser.add_argument("--bootstrap", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=20260618)
    return parser.parse_args()


def main():
    args = parse_args()
    ensure_dirs()

    if not args.input.exists():
        raise FileNotFoundError(
            f"Could not find {args.input}. "
            "Run scripts/diagnose_hbb_detector_response.py first."
        )

    df = pd.read_csv(args.input)
    check_input(df)

    # Remove pathological rows if any.
    df = df[np.isfinite(df["matched_mbb"]) & np.isfinite(df["lower_matched_jet_pt"])].copy()
    if df.empty:
        raise RuntimeError("Input exists, but no finite matched_mbb/lower_matched_jet_pt rows were found.")

    thresholds = sorted(set(float(x) for x in args.thresholds))
    summary = build_summary(df, thresholds, args.bootstrap, args.seed)

    summary_csv = OUTDIR / "pt_threshold_summary.csv"
    summary.to_csv(summary_csv, index=False)

    summary_json = {
        "input": str(args.input),
        "n_input_events": int(len(df)),
        "thresholds": thresholds,
        "bootstrap": int(args.bootstrap),
        "seed": int(args.seed),
        "outputs": {
            "summary_csv": str(summary_csv),
            "summary_markdown": str(SUMMARY_MD),
            "plot_dir": str(PLOT_DIR),
        },
    }
    with open(OUTDIR / "summary.json", "w") as f:
        json.dump(summary_json, f, indent=2)

    plot_mass_vs_cut(summary)
    plot_retention_vs_cut(summary)
    plot_window_fractions(summary)
    plot_response_vs_cut(summary)
    plot_mass_distributions(df, cuts_to_show=[None, 25, 40, 70, 90])

    write_markdown(summary, args)

    print("\nSaved summary table:")
    print(summary_csv)

    print("\nSaved markdown summary:")
    print(SUMMARY_MD)

    print("\nSaved plots:")
    print(PLOT_DIR)

    print("\nSummary:")
    display_cols = [
        "cut_label",
        "n_events",
        "signal_retention",
        "mean_mbb",
        "median_mbb",
        "median_mbb_boot_p16",
        "median_mbb_boot_p84",
        "frac_90_140",
        "frac_100_150",
        "median_lower_response",
    ]
    print(summary[display_cols].to_string(index=False))


if __name__ == "__main__":
    main()