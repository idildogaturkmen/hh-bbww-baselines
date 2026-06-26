'''
Make plots for signal and background processes, including stacked histograms and cutflow plots.
'''

from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.lines import Line2D


# Fixed plotting order and colors, so every process has the same color in every plot.
PLOT_ORDER = [
    "qcd",
    "minbias",
    "gamma",
    "upsilon",
    "wjets",
    "dy_zjets",
    "diboson",
    "triboson",
    "single_higgs",
    "HH_other",
    "ttV_ttH_tttt",
    "ttbar",
]

PRETTY = {
    "signal": r"HH$\to$bbWW",
    "HH_other": "Other HH",
    "ttbar": r"$t\bar{t}$",
    "ttV_ttH_tttt": r"$t\bar{t}V/t\bar{t}H/t\bar{t}t\bar{t}$",
    "dy_zjets": "DY/Z+jets",
    "wjets": "W+jets",
    "diboson": "VV",
    "triboson": "VVV",
    "single_higgs": "Single Higgs",
    "qcd": "QCD",
    "gamma": r"$\gamma$ samples",
    "minbias": "MinBias",
    "upsilon": r"$\Upsilon\to\ell\ell$",
    "other": "Other",
}

COLORS = {
    "signal": "red",
    "ttbar": "#1f77b4",
    "ttV_ttH_tttt": "#ff7f0e",
    "HH_other": "#2ca02c",
    "single_higgs": "#9467bd",
    "diboson": "#8c564b",
    "triboson": "#e377c2",
    "dy_zjets": "#7f7f7f",
    "wjets": "#bcbd22",
    "qcd": "#17becf",
    "gamma": "#d62728",
    "minbias": "#aec7e8",
    "upsilon": "#ffbb78",
    "other": "#c7c7c7",
}

CHANNELS = {
    "inclusive_ge2jets": ("Inclusive, ≥2 AK4 jets", "pass_ge2jets"),
    "onelep_ge3jets": ("1 lepton, ≥3 AK4 jets", "pass_1lep_ge3jets"),
    "twolep_ge2jets": ("2 leptons, ≥2 AK4 jets", "pass_2lep_ge2jets"),
}

VARIABLES = [
    (
        "mbb_top2_btag",
        r"$m_{bb}$ from two highest-b-tag AK4 jets [GeV]",
        np.linspace(0, 300, 61),
    ),
    (
        "ht",
        r"$H_T$ [GeV]",
        np.linspace(0, 1200, 61),
    ),
    (
        "met",
        r"MET [GeV]",
        np.linspace(0, 400, 51),
    ),
    (
        "n_jets",
        "Number of selected AK4 jets",
        np.arange(-0.5, 12.5, 1.0),
    ),
    (
        "n_leptons",
        "Number of selected leptons",
        np.arange(-0.5, 5.5, 1.0),
    ),
    # These are binary b-tag flags in this skim, not continuous b-tag scores.
    (
        "b1_btag",
        "Highest b-tag flag",
        np.array([-0.5, 0.5, 1.5]),
    ),
    (
        "b2_btag",
        "Second-highest b-tag flag",
        np.array([-0.5, 0.5, 1.5]),
    ),
]


def get_weights(df: pd.DataFrame) -> np.ndarray:
    if "weight" in df.columns:
        return df["weight"].to_numpy(dtype=float)
    return np.ones(len(df), dtype=float)


def safe_asimov_z(s: float, b: float) -> float:
    if s <= 0:
        return 0.0
    if b <= 0:
        return math.sqrt(2.0 * s)
    return math.sqrt(2.0 * ((s + b) * math.log(1.0 + s / b) - s))


def add_analysis_header(ax, channel_label: str):
    """
    Put a minimal non-CMS label above the plotting region.
    """
    ax.text(
        0.00,
        1.13,
        "COLLIDE-1M Simulation",
        transform=ax.transAxes,
        fontsize=12,
        style="italic",
        va="bottom",
        ha="left",
        clip_on=False,
    )
    ax.text(
        0.00,
        1.055,
        channel_label,
        transform=ax.transAxes,
        fontsize=11.5,
        va="bottom",
        ha="left",
        clip_on=False,
    )

def make_legend(ax, signal_scale: float):
    handles = []
    labels = []

    for group in PLOT_ORDER:
        handles.append(Patch(facecolor=COLORS[group], edgecolor="black", linewidth=0.3))
        labels.append(PRETTY[group])

    handles.append(Line2D([0], [0], color=COLORS["signal"], linewidth=2.5))
    labels.append(f"{PRETTY['signal']} × {signal_scale:g}")

    ax.legend(
        handles,
        labels,
        fontsize=8,
        ncol=1,
        loc="upper left",
        bbox_to_anchor=(1.01, 1.0),
        frameon=False,
        borderaxespad=0.0,
    )


def make_yield_tables(df: pd.DataFrame, outdir: Path):
    yield_rows = []
    sig_rows = []

    for channel_name, (channel_label, mask_col) in CHANNELS.items():
        base_mask = df[mask_col].astype(bool)

        selections = {
            channel_name: base_mask,
            f"{channel_name}_mbb_90_140": base_mask & df["mbb_top2_btag"].between(90, 140),
        }

        for selection_name, mask in selections.items():
            sub = df[mask].copy()

            for group in ["signal"] + PLOT_ORDER:
                g = sub[sub["process_group"] == group]
                yield_rows.append(
                    {
                        "selection": selection_name,
                        "channel_label": channel_label,
                        "process_group": group,
                        "label": PRETTY.get(group, group),
                        "n_events": int(len(g)),
                        "raw_yield": float(get_weights(g).sum()),
                    }
                )

            s = float(get_weights(sub[sub["process_group"] == "signal"]).sum())
            b = float(get_weights(sub[sub["process_group"] != "signal"]).sum())

            sig_rows.append(
                {
                    "selection": selection_name,
                    "channel_label": channel_label,
                    "S_raw": s,
                    "B_raw": b,
                    "S_over_B_raw": s / b if b > 0 else np.nan,
                    "S_over_sqrt_SplusB_raw": s / math.sqrt(s + b) if (s + b) > 0 else 0.0,
                    "asimov_Z_raw": safe_asimov_z(s, b),
                }
            )

    pd.DataFrame(yield_rows).to_csv(outdir / "raw_yields_by_channel.csv", index=False)
    pd.DataFrame(sig_rows).to_csv(outdir / "raw_significance_by_channel.csv", index=False)


def plot_stack(
    df: pd.DataFrame,
    variable: str,
    xlabel: str,
    bins: np.ndarray,
    channel_name: str,
    channel_label: str,
    mask_col: str,
    outdir: Path,
    signal_scale: float,
    logy: bool,
):
    sub = df[df[mask_col].astype(bool)].copy()
    sub = sub[np.isfinite(sub[variable])]

    signal = sub[sub["process_group"] == "signal"]
    backgrounds = sub[sub["process_group"] != "signal"]

    bkg_values = []
    bkg_weights = []
    bkg_colors = []

    for group in PLOT_ORDER:
        g = backgrounds[backgrounds["process_group"] == group]
        g = g[np.isfinite(g[variable])]
        if len(g) == 0:
            continue

        bkg_values.append(g[variable].to_numpy(dtype=float))
        bkg_weights.append(get_weights(g))
        bkg_colors.append(COLORS[group])

    fig, (ax, rax) = plt.subplots(
        2,
        1,
        figsize=(10.8, 7.3),
        sharex=True,
        gridspec_kw={"height_ratios": [3.2, 1.0], "hspace": 0.05},
    )

    # Top space for header, right space for legend.
    fig.subplots_adjust(top=0.80, right=0.78)

    if bkg_values:
        ax.hist(
            bkg_values,
            bins=bins,
            weights=bkg_weights,
            stacked=True,
            histtype="stepfilled",
            color=bkg_colors,
            edgecolor="black",
            linewidth=0.25,
            alpha=0.88,
        )

    if len(signal):
        ax.hist(
            signal[variable].to_numpy(dtype=float),
            bins=bins,
            weights=get_weights(signal) * signal_scale,
            histtype="step",
            color=COLORS["signal"],
            linewidth=2.6,
        )

    b_hist = np.zeros(len(bins) - 1)
    for group in PLOT_ORDER:
        g = backgrounds[backgrounds["process_group"] == group]
        g = g[np.isfinite(g[variable])]
        if len(g) == 0:
            continue
        h, _ = np.histogram(
            g[variable].to_numpy(dtype=float),
            bins=bins,
            weights=get_weights(g),
        )
        b_hist += h

    if len(signal):
        s_hist, _ = np.histogram(
            signal[variable].to_numpy(dtype=float),
            bins=bins,
            weights=get_weights(signal),
        )
    else:
        s_hist = np.zeros(len(bins) - 1)

    purity = np.divide(
        s_hist,
        s_hist + b_hist,
        out=np.zeros_like(s_hist, dtype=float),
        where=(s_hist + b_hist) > 0,
    )

    centers = 0.5 * (bins[:-1] + bins[1:])
    rax.step(centers, purity, where="mid", color="black", linewidth=1.8)

    rax.set_ylabel("Raw\nS/(S+B)", fontsize=10)
    ymax = 1.25 * np.nanmax(purity) if len(purity) and np.nanmax(purity) > 0 else 0.05
    rax.set_ylim(0, min(1.0, max(0.05, ymax)))
    rax.grid(True, alpha=0.30)

    ax.set_ylabel("Raw MC events / bin")
    if logy:
        ax.set_yscale("log")
        ax.set_ylim(bottom=0.5)

    ax.grid(True, alpha=0.25)
    add_analysis_header(ax, channel_label)
    make_legend(ax, signal_scale)
    rax.set_xlabel(xlabel)

    if variable in ["b1_btag", "b2_btag"]:
        rax.set_xticks([0, 1])
        rax.set_xticklabels(["0", "1"])

    outpath = outdir / f"{variable}_stack_{channel_name}.png"
    fig.savefig(outpath, dpi=220, bbox_inches="tight")
    plt.close(fig)


def make_cutflow_plot(df: pd.DataFrame, outdir: Path, signal_scale: float):
    cuts = {
        "All": np.ones(len(df), dtype=bool),
        "≥2 jets": df["pass_ge2jets"].astype(bool),
        "1 lep, ≥3 jets": df["pass_1lep_ge3jets"].astype(bool),
        "1 lep, ≥3 jets,\n90<m_bb<140": df["pass_1lep_ge3jets"].astype(bool)
        & df["mbb_top2_btag"].between(90, 140),
        "2 lep, ≥2 jets": df["pass_2lep_ge2jets"].astype(bool),
        "2 lep, ≥2 jets,\n90<m_bb<140": df["pass_2lep_ge2jets"].astype(bool)
        & df["mbb_top2_btag"].between(90, 140),
    }

    x = np.arange(len(cuts))
    bottom = np.zeros(len(cuts))

    fig, ax = plt.subplots(figsize=(10.8, 6.8))
    fig.subplots_adjust(top=0.80, right=0.78)

    for group in PLOT_ORDER:
        counts = []
        for mask in cuts.values():
            g = df[(df["process_group"] == group) & mask]
            counts.append(get_weights(g).sum())

        counts = np.asarray(counts)
        if counts.sum() == 0:
            continue

        ax.bar(
            x,
            counts,
            bottom=bottom,
            color=COLORS[group],
            edgecolor="black",
            linewidth=0.25,
        )
        bottom += counts

    # Overlay signal as red line, not as part of the stacked background.
    sig_counts = []
    for mask in cuts.values():
        g = df[(df["process_group"] == "signal") & mask]
        sig_counts.append(get_weights(g).sum() * signal_scale)

    ax.step(
        x,
        sig_counts,
        where="mid",
        color=COLORS["signal"],
        linewidth=2.6,
    )
    ax.scatter(x, sig_counts, color=COLORS["signal"], s=28)

    ax.set_xticks(x)
    ax.set_xticklabels(list(cuts.keys()), rotation=25, ha="right")
    ax.set_ylabel("Raw MC events")
    ax.set_yscale("log")
    ax.set_ylim(bottom=0.5)
    ax.grid(True, axis="y", alpha=0.30)

    add_analysis_header(ax, "Reco-level cutflow")
    make_legend(ax, signal_scale)

    fig.savefig(outdir / "cutflow_stack.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--skim",
        default="outputs/all_background_reco_skim/all_processes_reco_skim.parquet",
    )
    parser.add_argument(
        "--outdir",
        default="outputs/signal_vs_background_plots",
    )
    parser.add_argument("--signal-scale", type=float, default=20.0)
    parser.add_argument("--linear", action="store_true", help="Use linear y-axis instead of log.")
    args = parser.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    df = pd.read_parquet(args.skim)

    print(f"Read skim: {args.skim}")
    print(f"Rows: {len(df)}")
    print("\nProcess counts:")
    print(df.groupby("process_group").size().sort_values(ascending=False).to_string())

    make_yield_tables(df, outdir)
    make_cutflow_plot(df, outdir, signal_scale=args.signal_scale)

    for channel_name, (channel_label, mask_col) in CHANNELS.items():
        for variable, xlabel, bins in VARIABLES:
            plot_stack(
                df=df,
                variable=variable,
                xlabel=xlabel,
                bins=bins,
                channel_name=channel_name,
                channel_label=channel_label,
                mask_col=mask_col,
                outdir=outdir,
                signal_scale=args.signal_scale,
                logy=not args.linear,
            )

    print(f"\nWrote signal-vs-background plots and yield tables to: {outdir}")
    print("Key files:")
    print(f"  {outdir / 'cutflow_stack.png'}")
    print(f"  {outdir / 'mbb_top2_btag_stack_onelep_ge3jets.png'}")
    print(f"  {outdir / 'ht_stack_onelep_ge3jets.png'}")
    print(f"  {outdir / 'met_stack_onelep_ge3jets.png'}")
    print(f"  {outdir / 'raw_significance_by_channel.csv'}")


if __name__ == "__main__":
    main()