'''
Script to scan the high-mass tail of the m_bb distribution and produce a CSV with detailed information about the highest corrected-mbb events.
Usage:
    python scripts/make_high_mbb_tail_event_scan.py --h5 data/spanet_h
'''

from __future__ import annotations

import argparse
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def delta_phi(phi1: np.ndarray, phi2: np.ndarray) -> np.ndarray:
    return np.arctan2(np.sin(phi1 - phi2), np.cos(phi1 - phi2))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--h5",
        default="data/spanet_hbb/spanet_hbb_test.h5",
        help="SPA-Net-format HDF5 file.",
    )
    parser.add_argument(
        "--scan-csv",
        default="outputs/min_bjet_pt_cut_scan/test_hbb_truth_pair_masses.csv",
        help="CSV produced by make_min_bjet_pt_cut_scan.py.",
    )
    parser.add_argument(
        "--outdir",
        default="outputs/min_bjet_pt_cut_scan",
        help="Output directory for event-scan CSVs.",
    )
    parser.add_argument(
        "--plotdir",
        default="outputs/plots/min_bjet_pt_cut_scan",
        help="Output directory for plots.",
    )
    parser.add_argument(
        "--tag",
        default="test",
        help="Prefix for output files.",
    )
    parser.add_argument(
        "--n-events",
        type=int,
        default=50,
        help="Number of highest corrected-mbb events to save.",
    )
    args = parser.parse_args()

    h5_path = Path(args.h5)
    scan_path = Path(args.scan_csv)
    outdir = Path(args.outdir)
    plotdir = Path(args.plotdir)

    outdir.mkdir(parents=True, exist_ok=True)
    plotdir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(scan_path)

    # Recompute these in case they are not already present.
    df["mbb_shift"] = df["mbb_corrected"] - df["mbb_uncorrected"]
    df["mbb_ratio"] = df["mbb_corrected"] / df["mbb_uncorrected"]

    with h5py.File(h5_path, "r") as f:
        log_pt = f["INPUTS/Jets/log_pt"][:]
        eta = f["INPUTS/Jets/eta"][:]
        sin_phi = f["INPUTS/Jets/sin_phi"][:]
        cos_phi = f["INPUTS/Jets/cos_phi"][:]
        btag = f["INPUTS/Jets/btag"][:]
        btag_phys = f["INPUTS/Jets/btag_phys"][:]
        response_corr = f["INPUTS/Jets/response_corr"][:]
        log_corr_pt = f["INPUTS/Jets/log_corrected_pt"][:]

    pt = np.exp(log_pt)
    corrected_pt = np.exp(log_corr_pt)
    phi = np.arctan2(sin_phi, cos_phi)

    rows = []
    for _, r in df.sort_values("mbb_corrected", ascending=False).head(args.n_events).iterrows():
        i = int(r["event_local_index"])
        b1 = int(r["b1"])
        b2 = int(r["b2"])

        d_eta = eta[i, b1] - eta[i, b2]
        d_phi = delta_phi(phi[i, b1], phi[i, b2])
        d_r = float(np.sqrt(d_eta**2 + d_phi**2))

        b1_corr = float(response_corr[i, b1])
        b2_corr = float(response_corr[i, b2])

        rows.append(
            {
                "event_local_index": i,
                "b1": b1,
                "b2": b2,
                "b1_pt": float(pt[i, b1]),
                "b2_pt": float(pt[i, b2]),
                "min_bjet_pt": float(min(pt[i, b1], pt[i, b2])),
                "b1_corrected_pt": float(corrected_pt[i, b1]),
                "b2_corrected_pt": float(corrected_pt[i, b2]),
                "min_bjet_corrected_pt": float(min(corrected_pt[i, b1], corrected_pt[i, b2])),
                "b1_eta": float(eta[i, b1]),
                "b2_eta": float(eta[i, b2]),
                "b1_phi": float(phi[i, b1]),
                "b2_phi": float(phi[i, b2]),
                "deltaR_b1_b2": d_r,
                "b1_btag": float(btag[i, b1]),
                "b2_btag": float(btag[i, b2]),
                "b1_btag_phys": float(btag_phys[i, b1]),
                "b2_btag_phys": float(btag_phys[i, b2]),
                "b1_response_corr": b1_corr,
                "b2_response_corr": b2_corr,
                "max_response_corr": max(b1_corr, b2_corr),
                "mean_response_corr": 0.5 * (b1_corr + b2_corr),
                "mbb_uncorrected": float(r["mbb_uncorrected"]),
                "mbb_corrected": float(r["mbb_corrected"]),
                "mbb_shift": float(r["mbb_shift"]),
                "mbb_ratio": float(r["mbb_ratio"]),
            }
        )

    detailed = pd.DataFrame(rows)

    detailed_csv = outdir / f"{args.tag}_high_mbb_tail_event_scan_detailed.csv"
    detailed.to_csv(detailed_csv, index=False)

    print(f"Wrote {detailed_csv}")
    print()
    print(detailed.head(20).to_string(index=False))

    # Plot 1: corrected mbb vs DeltaR for high-tail events.
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.scatter(detailed["deltaR_b1_b2"], detailed["mbb_corrected"], s=30, alpha=0.8)
    ax.set_xlabel("ΔR between truth-matched H→bb jets")
    ax.set_ylabel("Corrected m_bb [GeV]")
    ax.set_title("High-mass tail events: corrected m_bb vs ΔR(b1,b2)")
    fig.tight_layout()
    fig.savefig(plotdir / f"{args.tag}_high_tail_mbb_vs_deltaR.png", dpi=180)
    plt.close(fig)

    # Plot 2: corrected/uncorrected mbb ratio vs max response correction.
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.scatter(detailed["max_response_corr"], detailed["mbb_ratio"], s=30, alpha=0.8)
    ax.axhline(1.0, linestyle="--", linewidth=1.2)
    ax.axhline(1.25, linestyle="--", linewidth=1.0)
    ax.set_xlabel("Max individual b-jet response correction")
    ax.set_ylabel("m_bb corrected / m_bb uncorrected")
    ax.set_title("High-mass tail events: correction size")
    fig.tight_layout()
    fig.savefig(plotdir / f"{args.tag}_high_tail_mbb_ratio_vs_max_response_corr.png", dpi=180)
    plt.close(fig)

    # Plot 3: corrected pT vs original pT for b jets in the high-tail events.
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.scatter(detailed["b1_pt"], detailed["b1_corrected_pt"], s=25, alpha=0.7, label="b1")
    ax.scatter(detailed["b2_pt"], detailed["b2_corrected_pt"], s=25, alpha=0.7, label="b2")
    max_pt = max(
        detailed["b1_pt"].max(),
        detailed["b2_pt"].max(),
        detailed["b1_corrected_pt"].max(),
        detailed["b2_corrected_pt"].max(),
    )
    ax.plot([0, max_pt], [0, max_pt], linestyle="--", linewidth=1.2)
    ax.set_xlabel("Original b-jet pT [GeV]")
    ax.set_ylabel("Corrected b-jet pT [GeV]")
    ax.set_title("High-mass tail events: corrected pT vs original pT")
    ax.legend()
    fig.tight_layout()
    fig.savefig(plotdir / f"{args.tag}_high_tail_corrected_pt_vs_original_pt.png", dpi=180)
    plt.close(fig)

    print()
    print("Wrote plots:")
    print(plotdir / f"{args.tag}_high_tail_mbb_vs_deltaR.png")
    print(plotdir / f"{args.tag}_high_tail_mbb_ratio_vs_max_response_corr.png")
    print(plotdir / f"{args.tag}_high_tail_corrected_pt_vs_original_pt.png")


if __name__ == "__main__":
    main()
