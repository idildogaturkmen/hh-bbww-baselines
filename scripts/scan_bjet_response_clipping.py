'''
DNN b-jet response clipping scan script.
Performs a scan of the effect of clipping the DNN b-jet response correction on the m_bb distribution and other metrics.
Usage:
    python scripts/scan_bjet_response_clipping.py --file data/spanet_hbb/spanet_hbb_test.h5 --tag test --outdir outputs/bjet_response_clipping
'''

from __future__ import annotations

import argparse
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


M_HIGGS = 125.0
PT_CUTS = [0.0, 30.0, 40.0, 50.0, 60.0]
CAPS = [None, 1.25, 1.50, 1.75, 2.00, 2.50] # different clipping values for the DNN response correction, None means no clipping (original DNN correction)


def central_width68(values: np.ndarray) -> float:
    if len(values) == 0:
        return float("nan")
    q16, q84 = np.percentile(values, [16, 84])
    return 0.5 * (q84 - q16)


def pair_mass(pt1, eta1, phi1, mass1, pt2, eta2, phi2, mass2) -> np.ndarray:
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


def summarize_mass(values: np.ndarray) -> dict[str, float]:
    in_90_140 = (values >= 90.0) & (values <= 140.0)
    in_100_150 = (values >= 100.0) & (values <= 150.0)

    return {
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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--file", default="data/spanet_hbb/spanet_hbb_test.h5")
    parser.add_argument("--tag", default="test")
    parser.add_argument("--outdir", default="outputs/bjet_response_clipping")
    parser.add_argument("--plotdir", default="outputs/plots/bjet_response_clipping")
    args = parser.parse_args()

    h5_path = Path(args.file)
    outdir = Path(args.outdir)
    plotdir = Path(args.plotdir)
    outdir.mkdir(parents=True, exist_ok=True)
    plotdir.mkdir(parents=True, exist_ok=True)

    with h5py.File(h5_path, "r") as f:
        log_pt = f["INPUTS/Jets/log_pt"][:]
        log_mass = f["INPUTS/Jets/log_mass"][:]
        log_corr_pt = f["INPUTS/Jets/log_corrected_pt"][:]
        log_corr_mass = f["INPUTS/Jets/log_corrected_mass"][:]
        response_corr = f["INPUTS/Jets/response_corr"][:]
        eta = f["INPUTS/Jets/eta"][:]
        sin_phi = f["INPUTS/Jets/sin_phi"][:]
        cos_phi = f["INPUTS/Jets/cos_phi"][:]
        b1 = f["TARGETS/h/b1"][:].astype(int)
        b2 = f["TARGETS/h/b2"][:].astype(int)

    idx = np.arange(len(b1))
    pt = np.exp(log_pt)
    mass = np.exp(log_mass)
    phi = np.arctan2(sin_phi, cos_phi)

    # Original DNN corrected values from the HDF5.
    original_corr_pt = np.exp(log_corr_pt)
    original_corr_mass = np.exp(log_corr_mass)

    # Sanity check: response_corr should approximately equal corrected_pt / pt.
    derived_corr = original_corr_pt / np.maximum(pt, 1e-12)
    valid = np.isfinite(derived_corr)
    print("response_corr sanity check:")
    print("  max |stored - derived|:", np.nanmax(np.abs(response_corr[valid] - derived_corr[valid])))
    print("  response_corr percentiles:", np.percentile(response_corr[np.isfinite(response_corr)], [0, 1, 5, 50, 95, 99, 100]))
    print()

    min_bjet_pt = np.minimum(pt[idx, b1], pt[idx, b2])

    # Uncorrected mass.
    mbb_uncorr = pair_mass(
        pt[idx, b1], eta[idx, b1], phi[idx, b1], mass[idx, b1],
        pt[idx, b2], eta[idx, b2], phi[idx, b2], mass[idx, b2],
    )

    event_rows = {
        "event_local_index": idx,
        "b1": b1,
        "b2": b2,
        "min_bjet_pt": min_bjet_pt,
        "mbb_uncorrected": mbb_uncorr,
    }

    # Add original DNN and clipped variants.
    mass_variants: dict[str, np.ndarray] = {
        "uncorrected": mbb_uncorr,
    }

    # Original correction directly from saved corrected pT/mass.
    mbb_dnn = pair_mass(
        original_corr_pt[idx, b1], eta[idx, b1], phi[idx, b1], original_corr_mass[idx, b1],
        original_corr_pt[idx, b2], eta[idx, b2], phi[idx, b2], original_corr_mass[idx, b2],
    )
    mass_variants["dnn_original"] = mbb_dnn
    event_rows["mbb_dnn_original"] = mbb_dnn

    for cap in [c for c in CAPS if c is not None]:
        clipped_corr = np.minimum(response_corr, cap)

        clipped_pt = pt * clipped_corr
        clipped_mass = mass * clipped_corr

        label = f"dnn_cap_{cap:.2f}".replace(".", "p")

        mbb_clip = pair_mass(
            clipped_pt[idx, b1], eta[idx, b1], phi[idx, b1], clipped_mass[idx, b1],
            clipped_pt[idx, b2], eta[idx, b2], phi[idx, b2], clipped_mass[idx, b2],
        )

        mass_variants[label] = mbb_clip
        event_rows[f"mbb_{label}"] = mbb_clip

    event_df = pd.DataFrame(event_rows)
    event_csv = outdir / f"{args.tag}_event_mbb_with_clipped_corrections.csv"
    event_df.to_csv(event_csv, index=False)

    # Summary table over pT cuts and correction variants.
    rows = []
    n_total = len(event_df)

    for cut in PT_CUTS:
        pass_mask = event_df["min_bjet_pt"].to_numpy() >= cut
        n_pass = int(np.sum(pass_mask))
        efficiency = n_pass / n_total

        for version, values_all in mass_variants.items():
            values = values_all[pass_mask]
            row = {
                "min_bjet_pt_cut": cut,
                "version": version,
                "n_total": n_total,
                "n_pass": n_pass,
                "efficiency": efficiency,
            }
            row.update(summarize_mass(values))
            rows.append(row)

    summary = pd.DataFrame(rows)
    summary_csv = outdir / f"{args.tag}_bjet_response_clipping_scan.csv"
    summary.to_csv(summary_csv, index=False)

    # Category summary for no pT cut.
    cat_rows = []
    for version, col in [(v, f"mbb_{v}") for v in mass_variants if v != "uncorrected"]:
        corrected = event_df[col].to_numpy()
        uncorr = event_df["mbb_uncorrected"].to_numpy()

        categories = {
            "already_high_uncorrected": uncorr > 140,
            "new_high_after_correction": (uncorr <= 140) & (corrected > 140),
            "fixed_low_to_window": (uncorr < 90) & ((corrected >= 90) & (corrected <= 140)),
            "stays_in_window": ((uncorr >= 90) & (uncorr <= 140)) & ((corrected >= 90) & (corrected <= 140)),
            "stays_low": (uncorr < 90) & (corrected < 90),
        }

        for name, mask in categories.items():
            sub_unc = uncorr[mask]
            sub_cor = corrected[mask]
            cat_rows.append(
                {
                    "version": version,
                    "category": name,
                    "n_events": int(np.sum(mask)),
                    "fraction": float(np.mean(mask)),
                    "median_uncorrected_mbb": float(np.median(sub_unc)) if len(sub_unc) else float("nan"),
                    "median_corrected_mbb": float(np.median(sub_cor)) if len(sub_cor) else float("nan"),
                    "median_mbb_shift": float(np.median(sub_cor - sub_unc)) if len(sub_unc) else float("nan"),
                    "median_mbb_ratio": float(np.median(sub_cor / sub_unc)) if len(sub_unc) else float("nan"),
                }
            )

    category_summary = pd.DataFrame(cat_rows)
    category_csv = outdir / f"{args.tag}_bjet_response_clipping_category_summary.csv"
    category_summary.to_csv(category_csv, index=False)

    # Plot 1: no-cut m_bb distributions.
    bins = np.linspace(40, 240, 80)
    fig, ax = plt.subplots(figsize=(8, 5))
    for version in ["uncorrected", "dnn_original", "dnn_cap_1p50", "dnn_cap_2p00"]:
        ax.hist(mass_variants[version], bins=bins, histtype="step", density=True, linewidth=1.5, label=version)
    ax.axvline(125.0, linestyle="--", linewidth=1.2)
    ax.set_xlabel("Truth H→bb m_bb [GeV]")
    ax.set_ylabel("Normalized events")
    ax.set_title("m_bb distributions with clipped DNN corrections")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(plotdir / f"{args.tag}_mbb_clipping_overlay.png", dpi=180)
    plt.close(fig)

    # Plot 2: width68 vs cap for no pT cut.
    no_cut = summary[summary["min_bjet_pt_cut"] == 0].copy()
    order = ["uncorrected", "dnn_original", "dnn_cap_1p25", "dnn_cap_1p50", "dnn_cap_1p75", "dnn_cap_2p00", "dnn_cap_2p50"]
    no_cut["order"] = no_cut["version"].map({v: i for i, v in enumerate(order)})
    no_cut = no_cut.sort_values("order")

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(no_cut["version"], no_cut["mbb_width68"], marker="o")
    ax.set_xlabel("Correction version")
    ax.set_ylabel("m_bb central 68% half-width [GeV]")
    ax.set_title("m_bb resolution vs DNN correction cap")
    ax.tick_params(axis="x", rotation=35)
    fig.tight_layout()
    fig.savefig(plotdir / f"{args.tag}_width68_vs_response_cap.png", dpi=180)
    plt.close(fig)

    # Plot 3: high tail vs cap for no pT cut.
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(no_cut["version"], no_cut["tail_high_140"], marker="o")
    ax.set_xlabel("Correction version")
    ax.set_ylabel("Fraction with m_bb > 140 GeV")
    ax.set_title("High-mass tail vs DNN correction cap")
    ax.tick_params(axis="x", rotation=35)
    fig.tight_layout()
    fig.savefig(plotdir / f"{args.tag}_high_tail_vs_response_cap.png", dpi=180)
    plt.close(fig)

    # Plot 4: fraction in 90-140 vs cap for no pT cut.
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(no_cut["version"], no_cut["frac_90_140"], marker="o")
    ax.set_xlabel("Correction version")
    ax.set_ylabel("Fraction in 90–140 GeV")
    ax.set_title("H→bb mass-window fraction vs DNN correction cap")
    ax.tick_params(axis="x", rotation=35)
    fig.tight_layout()
    fig.savefig(plotdir / f"{args.tag}_frac_90_140_vs_response_cap.png", dpi=180)
    plt.close(fig)

    print(f"Wrote {event_csv}")
    print(f"Wrote {summary_csv}")
    print(f"Wrote {category_csv}")
    print()
    print("No-cut summary:")
    print(no_cut.drop(columns=["order"]).to_string(index=False))
    print()
    print("Category summary:")
    print(category_summary.to_string(index=False))
    print()
    print("Plots written to:", plotdir)


if __name__ == "__main__":
    main()
