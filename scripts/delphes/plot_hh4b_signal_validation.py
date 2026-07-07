#!/usr/bin/env python3
from pathlib import Path
import argparse
import pandas as pd
import matplotlib.pyplot as plt

def load(path):
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(p)
    return pd.read_parquet(p)

def plot_overlay(dfs, labels, column, outdir, xlabel=None, bins=50, density=True):
    plt.figure()
    for df, label in zip(dfs, labels):
        if column not in df.columns:
            continue
        x = df[column].dropna()
        if len(x) == 0:
            continue
        plt.hist(x, bins=bins, histtype="step", density=density, label=label)
    plt.xlabel(xlabel or column)
    plt.ylabel("Normalized events" if density else "Events")
    plt.legend()
    plt.tight_layout()
    out = outdir / f"signal_compare_{column}.png"
    plt.savefig(out, dpi=160)
    plt.close()
    print(out)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--vbf-candidates", required=True)
    ap.add_argument("--ggf-candidates", required=True)
    ap.add_argument("--vbf-events", required=True)
    ap.add_argument("--ggf-events", required=True)
    ap.add_argument("--outdir", required=True)
    args = ap.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    vbf_c = load(args.vbf_candidates)
    ggf_c = load(args.ggf_candidates)
    vbf_e = load(args.vbf_events)
    ggf_e = load(args.ggf_events)

    labels = ["VBF HH→4b SM", "ggF HH→4b HEFT"]

    for col in ["mbb1", "mbb2", "avg_mbb", "delta_mbb", "mhh", "n_selected_bjets", "j1_pt", "j2_pt", "j3_pt", "j4_pt"]:
        plot_overlay([vbf_c, ggf_c], labels, col, outdir, xlabel=col)

    for col in ["n_jet", "n_jet_pt30_eta25", "n_bjet_pt30_eta25", "ht_pt30_eta25", "met", "event_weight", "event_cross_section_pb"]:
        plot_overlay([vbf_e, ggf_e], labels, col, outdir, xlabel=col)

    summary = outdir / "signal_validation_plot_summary.txt"
    summary.write_text(
        "Signal validation plots comparing VBF HH→4b SM and ggF HH→4b HEFT.\n"
        "Use these as first presentation-quality sanity plots.\n"
        "Caveat: ggF bulk sample is HEFT/effective approximation, not full SM loop-induced.\n"
    )
    print(summary)

if __name__ == "__main__":
    main()
