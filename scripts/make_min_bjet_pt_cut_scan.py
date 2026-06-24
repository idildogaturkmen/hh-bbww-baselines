'''
Min B-jet pT cut scan script.
Performs a scan of the minimum b-jet pT cut and evaluates the effect on the m_bb distribution.
Usage:
    python scripts/make_min_bjet_pt_cut_scan.py --file data/spanet_hbb/spanet_hbb_test.h5 --tag test --outdir outputs/min_bjet_pt_cut_scan
    
'''

from __future__ import annotations

import argparse
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


M_HIGGS = 125.0
CUTS = [0.0, 30.0, 40.0, 50.0, 60.0]


def central_width68(values: np.ndarray) -> float:
    """Central 68% half-width, approximately sigma-like for a Gaussian."""
    if len(values) == 0:
        return float("nan")
    q16, q84 = np.percentile(values, [16, 84])
    return 0.5 * (q84 - q16)


def pair_mass(pt1, eta1, phi1, mass1, pt2, eta2, phi2, mass2) -> np.ndarray:
    """Invariant mass of two jets from pt, eta, phi, mass."""
    px1 = pt1 * np.cos(phi1)
    py1 = pt1 * np.sin(phi1)
    pz1 = pt1 * np.sinh(eta1)
    e1 = np.sqrt(np.maximum((pt1 * np.cosh(eta1)) ** 2 + mass1**2, 0.0))

    px2 = pt2 * np.cos(phi2)
    py2 = pt2 * np.sin(phi2)
    pz2 = pt2 * np.sinh(eta2)
    e2 = np.sqrt(np.maximum((pt2 * np.cosh(eta2)) ** 2 + mass2**2, 0.0))

    e = e1 + e2
    px = px1 + px2
    py = py1 + py2
    pz = pz1 + pz2

    m2 = e**2 - px**2 - py**2 - pz**2
    return np.sqrt(np.maximum(m2, 0.0))


def load_hbb_arrays(path: Path) -> pd.DataFrame:
    """ Load the relevant arrays from the HBB h5 file and return a DataFrame with the truth-matched b-jet pair information."""
    with h5py.File(path, "r") as f:
        log_pt = f["INPUTS/Jets/log_pt"][:]
        eta = f["INPUTS/Jets/eta"][:]
        sin_phi = f["INPUTS/Jets/sin_phi"][:]
        cos_phi = f["INPUTS/Jets/cos_phi"][:]
        log_mass = f["INPUTS/Jets/log_mass"][:]

        log_corr_pt = f["INPUTS/Jets/log_corrected_pt"][:]
        log_corr_mass = f["INPUTS/Jets/log_corrected_mass"][:]

        b1 = f["TARGETS/h/b1"][:].astype(int)
        b2 = f["TARGETS/h/b2"][:].astype(int)

    n = len(b1)
    idx = np.arange(n)

    pt = np.exp(log_pt)
    mass = np.exp(log_mass)
    corr_pt = np.exp(log_corr_pt)
    corr_mass = np.exp(log_corr_mass)
    phi = np.arctan2(sin_phi, cos_phi)

    b1_pt = pt[idx, b1]
    b2_pt = pt[idx, b2]
    min_bjet_pt = np.minimum(b1_pt, b2_pt)

    b1_corr_pt = corr_pt[idx, b1]
    b2_corr_pt = corr_pt[idx, b2]
    min_bjet_corr_pt = np.minimum(b1_corr_pt, b2_corr_pt)

    mbb_uncorr = pair_mass(
        pt[idx, b1], eta[idx, b1], phi[idx, b1], mass[idx, b1],
        pt[idx, b2], eta[idx, b2], phi[idx, b2], mass[idx, b2],
    )

    mbb_corr = pair_mass(
        corr_pt[idx, b1], eta[idx, b1], phi[idx, b1], corr_mass[idx, b1],
        corr_pt[idx, b2], eta[idx, b2], phi[idx, b2], corr_mass[idx, b2],
    )

    return pd.DataFrame(
        {
            "event_local_index": idx,
            "b1": b1,
            "b2": b2,
            "b1_pt": b1_pt,
            "b2_pt": b2_pt,
            "min_bjet_pt": min_bjet_pt,
            "b1_corrected_pt": b1_corr_pt,
            "b2_corrected_pt": b2_corr_pt,
            "min_bjet_corrected_pt": min_bjet_corr_pt,
            "mbb_uncorrected": mbb_uncorr,
            "mbb_corrected": mbb_corr,
        }
    )


def summarize(df: pd.DataFrame, cuts: list[float]) -> pd.DataFrame:
    '''Summarize the effect of different minimum b-jet pT cuts on the m_bb distribution and other metrics.'''
    rows = []
    n_total = len(df)

    for cut in cuts:
        selected = df[df["min_bjet_pt"] >= cut].copy()
        n_pass = len(selected)
        efficiency = n_pass / n_total if n_total else float("nan")

        for version, col in [
            ("uncorrected", "mbb_uncorrected"),
            ("dnn_corrected", "mbb_corrected"),
        ]:
            values = selected[col].to_numpy()
            if n_pass == 0:
                rows.append(
                    {
                        "min_bjet_pt_cut": cut,
                        "version": version,
                        "n_total": n_total,
                        "n_pass": n_pass,
                        "efficiency": efficiency,
                    }
                )
                continue

            in_90_140 = (values >= 90.0) & (values <= 140.0)
            in_100_150 = (values >= 100.0) & (values <= 150.0)

            rows.append(
                {
                    "min_bjet_pt_cut": cut,
                    "version": version,
                    "n_total": n_total,
                    "n_pass": n_pass,
                    "efficiency": efficiency,
                    "mbb_median": float(np.median(values)),
                    "mbb_mean": float(np.mean(values)),
                    "mbb_width68": float(central_width68(values)),
                    "mbb_rmse_to_125": float(np.sqrt(np.mean((values - M_HIGGS) ** 2))),
                    "frac_90_140": float(np.mean(in_90_140)),
                    "tail_outside_90_140": float(1.0 - np.mean(in_90_140)),
                    "tail_low_90": float(np.mean(values < 90.0)),
                    "tail_high_140": float(np.mean(values > 140.0)),
                    "frac_100_150": float(np.mean(in_100_150)),
                    "tail_outside_100_150": float(1.0 - np.mean(in_100_150)),
                }
            )

    return pd.DataFrame(rows)


def make_plots(df: pd.DataFrame, summary: pd.DataFrame, plot_dir: Path, tag: str) -> None:
    plot_dir.mkdir(parents=True, exist_ok=True)

    # 1. Distribution of lower-pT Hbb jet.
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.hist(df["min_bjet_pt"], bins=60, histtype="step", linewidth=1.8)
    for cut in [30, 40, 50, 60]:
        ax.axvline(cut, linestyle="--", linewidth=1.2)
    ax.set_xlabel("Lower-pT truth-matched H→bb jet pT [GeV]")
    ax.set_ylabel("Events")
    ax.set_title("Lower-pT H→bb jet distribution")
    fig.tight_layout()
    fig.savefig(plot_dir / f"{tag}_min_bjet_pt_distribution.png", dpi=180)
    plt.close(fig)

    # 2. Efficiency vs cut.
    eff = summary[summary["version"] == "dnn_corrected"].sort_values("min_bjet_pt_cut")
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(eff["min_bjet_pt_cut"], eff["efficiency"], marker="o")
    ax.set_xlabel("Lower-pT H→bb jet cut [GeV]")
    ax.set_ylabel("Efficiency")
    ax.set_ylim(0, 1.05)
    ax.set_title("Efficiency after lower-pT H→bb jet cut")
    fig.tight_layout()
    fig.savefig(plot_dir / f"{tag}_efficiency_vs_min_bjet_pt_cut.png", dpi=180)
    plt.close(fig)

    # 3. Width68 vs cut.
    fig, ax = plt.subplots(figsize=(6, 5))
    for version in ["uncorrected", "dnn_corrected"]:
        sub = summary[summary["version"] == version].sort_values("min_bjet_pt_cut")
        ax.plot(sub["min_bjet_pt_cut"], sub["mbb_width68"], marker="o", label=version)
    ax.set_xlabel("Lower-pT H→bb jet cut [GeV]")
    ax.set_ylabel("m_bb central 68% half-width [GeV]")
    ax.set_title("m_bb resolution vs lower-pT H→bb jet cut")
    ax.legend()
    fig.tight_layout()
    fig.savefig(plot_dir / f"{tag}_mbb_width68_vs_min_bjet_pt_cut.png", dpi=180)
    plt.close(fig)

    # 4. Tail fraction outside 90-140 vs cut.
    fig, ax = plt.subplots(figsize=(6, 5))
    for version in ["uncorrected", "dnn_corrected"]:
        sub = summary[summary["version"] == version].sort_values("min_bjet_pt_cut")
        ax.plot(sub["min_bjet_pt_cut"], sub["tail_outside_90_140"], marker="o", label=version)
    ax.set_xlabel("Lower-pT H→bb jet cut [GeV]")
    ax.set_ylabel("Fraction outside 90–140 GeV")
    ax.set_ylim(0, 1.0)
    ax.set_title("m_bb tail fraction vs lower-pT H→bb jet cut")
    ax.legend()
    fig.tight_layout()
    fig.savefig(plot_dir / f"{tag}_tail_fraction_vs_min_bjet_pt_cut.png", dpi=180)
    plt.close(fig)

    # 5. Corrected m_bb distributions under different cuts.
    fig, ax = plt.subplots(figsize=(7, 5))
    bins = np.linspace(40, 220, 70)
    for cut in CUTS:
        values = df.loc[df["min_bjet_pt"] >= cut, "mbb_corrected"].to_numpy()
        label = "no cut" if cut == 0 else f"pT min ≥ {cut:.0f}"
        ax.hist(values, bins=bins, histtype="step", density=True, linewidth=1.4, label=label)
    ax.axvline(125.0, linestyle="--", linewidth=1.2)
    ax.set_xlabel("DNN-corrected truth H→bb m_bb [GeV]")
    ax.set_ylabel("Normalized events")
    ax.set_title("Corrected m_bb after lower-pT jet cuts")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(plot_dir / f"{tag}_corrected_mbb_by_min_bjet_pt_cut.png", dpi=180)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--file", default="data/spanet_hbb/spanet_hbb_test.h5")
    parser.add_argument("--tag", default="test")
    parser.add_argument("--outdir", default="outputs/min_bjet_pt_cut_scan")
    parser.add_argument("--plotdir", default="outputs/plots/min_bjet_pt_cut_scan")
    args = parser.parse_args()

    outdir = Path(args.outdir)
    plotdir = Path(args.plotdir)
    outdir.mkdir(parents=True, exist_ok=True)

    df = load_hbb_arrays(Path(args.file))
    summary = summarize(df, CUTS)

    event_csv = outdir / f"{args.tag}_hbb_truth_pair_masses.csv"
    summary_csv = outdir / f"{args.tag}_min_bjet_pt_cut_scan.csv"

    df.to_csv(event_csv, index=False)
    summary.to_csv(summary_csv, index=False)

    make_plots(df, summary, plotdir, args.tag)

    print(f"Wrote {event_csv}")
    print(f"Wrote {summary_csv}")
    print()
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
