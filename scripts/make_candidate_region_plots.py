'''
Make plots for candidate signal regions, including stacked histograms and process composition plots.
strongest candidate baseline:

onelep_ge3jets
90 < m_bb < 140
3 <= n_jets <= 5
b2_btag = 1
MET >= 0 or MET >= 20
HT >= 0 or HT >= 100
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

REGIONS = {
    "inclusive_highstat_mbb70_160_ge3_ht100": (
        r"Inclusive high-stat proxy: $70<m_{bb}<160$, $N_{jets}\geq3$, $H_T\geq100$",
        {
            "channel": "inclusive",
            "mbb_low": 70,
            "mbb_high": 160,
            "n_jets_min": 3,
            "n_jets_max": None,
            "b2eq1": False,
            "met_min": 0,
            "ht_min": 100,
        },
    ),
    "onelep_3to5_b2eq1_pre_mbb": (
        r"Candidate pre-mass region: 1 lepton, $3\leq N_{jets}\leq5$, second b-tag flag = 1",
        {
            "channel": "onelep",
            "mbb_low": None,
            "mbb_high": None,
            "n_jets_min": 3,
            "n_jets_max": 5,
            "b2eq1": True,
            "met_min": 0,
            "ht_min": 0,
        },
    ),
    "onelep_baseline_mbb90_140_3to5_b2eq1_ht100": (
        r"Candidate baseline: 1 lepton, $90<m_{bb}<140$, $3\leq N_{jets}\leq5$, second b-tag flag = 1, $H_T\geq100$",
        {
            "channel": "onelep",
            "mbb_low": 90,
            "mbb_high": 140,
            "n_jets_min": 3,
            "n_jets_max": 5,
            "b2eq1": True,
            "met_min": 0,
            "ht_min": 100,
        },
    ),
    "onelep_tight_highpurity_mbb90_140_3to5_b2eq1_met40_ht600": (
        r"Tight high-purity check: 1 lepton, $90<m_{bb}<140$, $3\leq N_{jets}\leq5$, second b-tag flag = 1, MET$\geq40$, $H_T\geq600$",
        {
            "channel": "onelep",
            "mbb_low": 90,
            "mbb_high": 140,
            "n_jets_min": 3,
            "n_jets_max": 5,
            "b2eq1": True,
            "met_min": 40,
            "ht_min": 600,
        },
    ),
}

VARIABLES = [
    ("mbb_top2_btag", r"$m_{bb}$ from two highest-b-tag AK4 jets [GeV]", np.linspace(0, 300, 61)),
    ("ht", r"$H_T$ [GeV]", np.linspace(0, 1200, 61)),
    ("met", "MET [GeV]", np.linspace(0, 400, 51)),
    ("n_jets", "Number of selected AK4 jets", np.arange(-0.5, 12.5, 1.0)),
]


def get_weights(df: pd.DataFrame) -> np.ndarray:
    if "weight" in df.columns:
        return df["weight"].to_numpy(dtype=float)
    return np.ones(len(df), dtype=float)


def asimov_z(s: float, b: float) -> float:
    if s <= 0:
        return 0.0
    if b <= 0:
        return math.sqrt(2.0 * s)
    return math.sqrt(2.0 * ((s + b) * math.log(1.0 + s / b) - s))


def region_mask(df: pd.DataFrame, cfg: dict) -> pd.Series:
    mask = pd.Series(True, index=df.index)

    if cfg["channel"] == "inclusive":
        mask &= df["pass_ge2jets"].astype(bool)
    elif cfg["channel"] == "onelep":
        mask &= df["pass_1lep_ge3jets"].astype(bool)
    elif cfg["channel"] == "twolep":
        mask &= df["pass_2lep_ge2jets"].astype(bool)
    else:
        raise ValueError(cfg["channel"])

    if cfg["mbb_low"] is not None:
        mask &= df["mbb_top2_btag"] >= cfg["mbb_low"]
    if cfg["mbb_high"] is not None:
        mask &= df["mbb_top2_btag"] <= cfg["mbb_high"]

    if cfg["n_jets_min"] is not None:
        mask &= df["n_jets"] >= cfg["n_jets_min"]
    if cfg["n_jets_max"] is not None:
        mask &= df["n_jets"] <= cfg["n_jets_max"]

    if cfg["b2eq1"]:
        mask &= df["b2_btag"] >= 1

    mask &= df["met"].fillna(-999.0) >= cfg["met_min"]
    mask &= df["ht"].fillna(-999.0) >= cfg["ht_min"]

    return mask


def add_header(ax, label: str):
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
        label,
        transform=ax.transAxes,
        fontsize=10.5,
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


def plot_stack(df: pd.DataFrame, region_name: str, region_label: str, variable: str, xlabel: str, bins, outdir: Path, signal_scale: float):
    sub = df[np.isfinite(df[variable])].copy()
    signal = sub[sub["process_group"] == "signal"]
    backgrounds = sub[sub["process_group"] != "signal"]

    bkg_values = []
    bkg_weights = []
    bkg_colors = []

    for group in PLOT_ORDER:
        g = backgrounds[backgrounds["process_group"] == group]
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
        if len(g) == 0:
            continue
        h, _ = np.histogram(g[variable].to_numpy(dtype=float), bins=bins, weights=get_weights(g))
        b_hist += h

    if len(signal):
        s_hist, _ = np.histogram(signal[variable].to_numpy(dtype=float), bins=bins, weights=get_weights(signal))
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
    ax.set_yscale("log")
    ax.set_ylim(bottom=0.5)
    ax.grid(True, alpha=0.25)

    add_header(ax, region_label)
    make_legend(ax, signal_scale)
    rax.set_xlabel(xlabel)

    region_dir = outdir / region_name
    region_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(region_dir / f"{variable}.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_composition(df: pd.DataFrame, region_name: str, region_label: str, outdir: Path):
    rows = []
    for group in ["signal"] + PLOT_ORDER:
        g = df[df["process_group"] == group]
        rows.append({"process_group": group, "label": PRETTY.get(group, group), "yield": float(get_weights(g).sum())})

    comp = pd.DataFrame(rows)
    comp = comp[comp["yield"] > 0].copy()

    fig, ax = plt.subplots(figsize=(9.5, 5.6))
    fig.subplots_adjust(top=0.80, right=0.96)

    labels = comp["label"].tolist()
    values = comp["yield"].to_numpy()
    colors = [COLORS[g] for g in comp["process_group"]]

    ax.bar(np.arange(len(values)), values, color=colors, edgecolor="black", linewidth=0.3)
    ax.set_xticks(np.arange(len(values)))
    ax.set_xticklabels(labels, rotation=35, ha="right")
    ax.set_ylabel("Raw MC events")
    ax.set_yscale("log")
    ax.grid(True, axis="y", alpha=0.30)
    add_header(ax, region_label)

    region_dir = outdir / region_name
    region_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(region_dir / "process_composition.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--skim", default="outputs/all_background_reco_skim/all_processes_reco_skim.parquet")
    parser.add_argument("--outdir", default="outputs/candidate_region_plots")
    parser.add_argument("--signal-scale", type=float, default=20.0)
    args = parser.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    df = pd.read_parquet(args.skim)

    summary_rows = []
    process_rows = []

    for region_name, (region_label, cfg) in REGIONS.items():
        sub = df[region_mask(df, cfg)].copy()

        s = float(get_weights(sub[sub["process_group"] == "signal"]).sum())
        b = float(get_weights(sub[sub["process_group"] != "signal"]).sum())

        summary_rows.append(
            {
                "region": region_name,
                "label": region_label,
                "S_raw": s,
                "B_raw": b,
                "S_over_B_raw": s / b if b > 0 else np.nan,
                "S_over_sqrt_SplusB_raw": s / math.sqrt(s + b) if (s + b) > 0 else 0.0,
                "asimov_Z_raw": asimov_z(s, b),
            }
        )

        for group in ["signal"] + PLOT_ORDER:
            g = sub[sub["process_group"] == group]
            process_rows.append(
                {
                    "region": region_name,
                    "process_group": group,
                    "label": PRETTY.get(group, group),
                    "raw_yield": float(get_weights(g).sum()),
                    "n_events": int(len(g)),
                }
            )

        plot_composition(sub, region_name, region_label, outdir)

        for variable, xlabel, bins in VARIABLES:
            plot_stack(
                df=sub,
                region_name=region_name,
                region_label=region_label,
                variable=variable,
                xlabel=xlabel,
                bins=bins,
                outdir=outdir,
                signal_scale=args.signal_scale,
            )

    summary = pd.DataFrame(summary_rows).sort_values("S_over_B_raw", ascending=False)
    process_yields = pd.DataFrame(process_rows)

    summary.to_csv(outdir / "candidate_region_summary.csv", index=False)
    process_yields.to_csv(outdir / "candidate_region_process_yields.csv", index=False)

    print("Read:", args.skim)
    print("Rows:", len(df))
    print("\nWrote candidate-region plots to:", outdir)
    print("\nCandidate region summary:")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()