#!/usr/bin/env python3

import os
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def write_markdown(df, path):
    try:
        path.write_text(df.to_markdown(index=False) + "\n")
    except Exception:
        path.write_text(df.to_string(index=False) + "\n")


def normalize_iht_table(path, campaign_label):
    df = pd.read_csv(path)
    df.columns = [c.strip() for c in df.columns]

    # The real CSV uses slice_label. Older drafts/scripts may use slice.
    if "ht_slice" not in df.columns:
        if "slice_label" in df.columns:
            df = df.rename(columns={"slice_label": "ht_slice"})
        elif "slice" in df.columns:
            df = df.rename(columns={"slice": "ht_slice"})
        else:
            def infer_slice(tag):
                m = re.search(r"iht\d+to\d+|iht\d+plus", str(tag))
                return m.group(0) if m else "unknown"
            df["ht_slice"] = df["tag"].map(infer_slice)

    label_map = {
        "iht100to200": "100–200",
        "iht200to400": "200–400",
        "iht400to600": "400–600",
        "iht600plus": "600+",
    }
    order_map = {
        "iht100to200": 0,
        "iht200to400": 1,
        "iht400to600": 2,
        "iht600plus": 3,
    }

    df["ht_slice_display"] = df["ht_slice"].map(label_map).fillna(df["ht_slice"])
    df["slice_order"] = df["ht_slice"].map(order_map).fillna(999).astype(int)
    df["campaign"] = campaign_label

    numeric_cols = [
        "n_generated",
        "xsec_pb",
        "n_ge4_jets",
        "n_ge4_btags",
        "candidate_rows",
        "candidate_eff",
        "effective_candidate_xsec_pb",
        "median_mbb1",
        "median_mbb2",
        "median_avg_mbb",
        "median_delta_mbb",
        "median_mhh",
        "median_ht_pt30_eta25",
    ]
    for c in numeric_cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    return df.sort_values("slice_order").reset_index(drop=True)


def save_bar(df, xcol, ycol, ylabel, title, outpath):
    plt.figure(figsize=(7.2, 4.6))
    plt.bar(df[xcol], df[ycol])
    plt.xlabel("Generator HT slice [GeV]")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(outpath, dpi=200)
    plt.close()


def save_line(df, xcol, ycol, ylabel, title, outpath):
    plt.figure(figsize=(7.2, 4.6))
    plt.plot(df[xcol], df[ycol], marker="o")
    plt.xlabel("Generator HT slice [GeV]")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(outpath, dpi=200)
    plt.close()


def find_candidate_parquet(store, tag):
    parquet_dir = store / "parquet"
    patterns = [
        f"{tag}_merged_hh4b_candidates.parquet",
        f"{tag}*merged*hh4b*candidates*.parquet",
        f"{tag}*hh4b*candidates*.parquet",
    ]
    matches = []
    for pattern in patterns:
        matches.extend(sorted(parquet_dir.glob(pattern)))
    # Deduplicate while preserving order
    out = []
    seen = set()
    for p in matches:
        if p not in seen:
            out.append(p)
            seen.add(p)
    return out[0] if out else None


def main():
    repo = Path(os.environ["HH4B_REPO"])
    store = Path(os.environ["HH4B_STORE"])

    outdir = repo / "outputs/plots/qcd_iht20k_summary_2026_07_08"
    tabledir = repo / "outputs/tables/qcd_iht20k_summary_2026_07_08"
    outdir.mkdir(parents=True, exist_ok=True)
    tabledir.mkdir(parents=True, exist_ok=True)

    p20 = store / "metadata/qcd_bbbb_iht_slice_scan_20000.csv"
    p10 = store / "metadata/qcd_bbbb_iht_slice_scan_10000.csv"

    if not p20.exists():
        raise FileNotFoundError(f"Missing 20k table: {p20}")

    df20 = normalize_iht_table(p20, "20k/slice")

    print("\n=== 20k/slice normalized table ===")
    print(df20.to_string(index=False))

    # Save 20k tables
    keep_cols = [
        "tag",
        "ht_slice",
        "n_generated",
        "xsec_pb",
        "n_ge4_jets",
        "n_ge4_btags",
        "candidate_rows",
        "candidate_eff",
        "effective_candidate_xsec_pb",
        "median_mbb1",
        "median_mbb2",
        "median_avg_mbb",
        "median_delta_mbb",
        "median_mhh",
        "median_ht_pt30_eta25",
    ]
    keep_cols = [c for c in keep_cols if c in df20.columns]
    df20[keep_cols].to_csv(tabledir / "qcd_iht20k_summary.csv", index=False)
    write_markdown(df20[keep_cols], tabledir / "qcd_iht20k_summary.md")

    totals20 = pd.DataFrame([{
        "campaign": "20k/slice",
        "total_generated": int(df20["n_generated"].sum()),
        "total_candidate_rows": int(df20["candidate_rows"].sum()),
        "sum_xsec_pb": df20["xsec_pb"].sum(),
        "sum_effective_candidate_xsec_pb": df20["effective_candidate_xsec_pb"].sum(),
    }])

    # Basic 20k plots
    save_bar(
        df20,
        "ht_slice_display",
        "candidate_rows",
        "HH4b candidate rows",
        "QCD bbbb HT-sliced 20k: candidate rows",
        outdir / "qcd_iht20k_candidate_rows_by_slice.png",
    )

    save_bar(
        df20,
        "ht_slice_display",
        "xsec_pb",
        "Generator cross section [pb]",
        "QCD bbbb HT-sliced 20k: cross section by slice",
        outdir / "qcd_iht20k_xsec_by_slice.png",
    )

    save_bar(
        df20,
        "ht_slice_display",
        "effective_candidate_xsec_pb",
        "Effective candidate cross section [pb]",
        "QCD bbbb HT-sliced 20k: selected rate by slice",
        outdir / "qcd_iht20k_effective_candidate_xsec_by_slice.png",
    )

    df20["candidate_eff_percent"] = 100.0 * df20["candidate_eff"]
    save_bar(
        df20,
        "ht_slice_display",
        "candidate_eff_percent",
        "Candidate efficiency [%]",
        "QCD bbbb HT-sliced 20k: candidate efficiency",
        outdir / "qcd_iht20k_candidate_efficiency_by_slice.png",
    )

    save_line(
        df20,
        "ht_slice_display",
        "median_mhh",
        "Median reconstructed mHH [GeV]",
        "QCD bbbb HT-sliced 20k: mHH hardens with HT",
        outdir / "qcd_iht20k_median_mhh_by_slice.png",
    )

    save_line(
        df20,
        "ht_slice_display",
        "median_ht_pt30_eta25",
        "Median reconstructed HT [GeV]",
        "QCD bbbb HT-sliced 20k: reconstructed HT by slice",
        outdir / "qcd_iht20k_median_reco_ht_by_slice.png",
    )

    # 10k vs 20k comparison if 10k table exists
    if p10.exists():
        df10 = normalize_iht_table(p10, "10k/slice")

        print("\n=== 10k/slice normalized table ===")
        print(df10.to_string(index=False))

        totals10 = pd.DataFrame([{
            "campaign": "10k/slice",
            "total_generated": int(df10["n_generated"].sum()),
            "total_candidate_rows": int(df10["candidate_rows"].sum()),
            "sum_xsec_pb": df10["xsec_pb"].sum(),
            "sum_effective_candidate_xsec_pb": df10["effective_candidate_xsec_pb"].sum(),
        }])

        totals = pd.concat([totals10, totals20], ignore_index=True)
        totals.to_csv(tabledir / "qcd_iht10k_vs_20k_totals.csv", index=False)
        write_markdown(totals, tabledir / "qcd_iht10k_vs_20k_totals.md")

        comp = df10[[
            "ht_slice",
            "ht_slice_display",
            "xsec_pb",
            "candidate_rows",
            "candidate_eff",
            "effective_candidate_xsec_pb",
            "median_mhh",
        ]].merge(
            df20[[
                "ht_slice",
                "xsec_pb",
                "candidate_rows",
                "candidate_eff",
                "effective_candidate_xsec_pb",
                "median_mhh",
            ]],
            on="ht_slice",
            suffixes=("_10k", "_20k"),
        )

        comp["xsec_ratio_20k_over_10k"] = comp["xsec_pb_20k"] / comp["xsec_pb_10k"]
        comp["candidate_rows_ratio_20k_over_10k"] = comp["candidate_rows_20k"] / comp["candidate_rows_10k"]
        comp["effective_candidate_xsec_ratio_20k_over_10k"] = (
            comp["effective_candidate_xsec_pb_20k"] / comp["effective_candidate_xsec_pb_10k"]
        )
        comp["candidate_eff_ratio_20k_over_10k"] = comp["candidate_eff_20k"] / comp["candidate_eff_10k"]
        comp["median_mhh_delta_20k_minus_10k"] = comp["median_mhh_20k"] - comp["median_mhh_10k"]

        comp.to_csv(tabledir / "qcd_iht10k_vs_20k_by_slice.csv", index=False)
        write_markdown(comp, tabledir / "qcd_iht10k_vs_20k_by_slice.md")

        # Comparison plots
        plt.figure(figsize=(7.2, 4.6))
        plt.plot(comp["ht_slice_display"], comp["effective_candidate_xsec_pb_10k"], marker="o", label="10k/slice")
        plt.plot(comp["ht_slice_display"], comp["effective_candidate_xsec_pb_20k"], marker="o", label="20k/slice")
        plt.xlabel("Generator HT slice [GeV]")
        plt.ylabel("Effective candidate cross section [pb]")
        plt.title("QCD HT slicing: selected rate stability")
        plt.legend()
        plt.tight_layout()
        plt.savefig(outdir / "qcd_iht10k_vs_20k_effective_candidate_xsec.png", dpi=200)
        plt.close()

        plt.figure(figsize=(7.2, 4.6))
        plt.plot(comp["ht_slice_display"], comp["xsec_ratio_20k_over_10k"], marker="o", label="Generator xsec ratio")
        plt.plot(comp["ht_slice_display"], comp["effective_candidate_xsec_ratio_20k_over_10k"], marker="o", label="Selected xsec ratio")
        plt.axhline(1.0, linestyle="--")
        plt.xlabel("Generator HT slice [GeV]")
        plt.ylabel("20k / 10k ratio")
        plt.title("QCD HT slicing: 20k vs 10k closure ratios")
        plt.legend()
        plt.tight_layout()
        plt.savefig(outdir / "qcd_iht10k_vs_20k_closure_ratios.png", dpi=200)
        plt.close()

        plt.figure(figsize=(7.2, 4.6))
        plt.bar(comp["ht_slice_display"], comp["candidate_rows_ratio_20k_over_10k"])
        plt.axhline(2.0, linestyle="--")
        plt.xlabel("Generator HT slice [GeV]")
        plt.ylabel("Candidate row ratio: 20k / 10k")
        plt.title("QCD HT slicing: candidate statistics approximately double")
        plt.tight_layout()
        plt.savefig(outdir / "qcd_iht10k_vs_20k_candidate_row_ratio.png", dpi=200)
        plt.close()

    else:
        totals20.to_csv(tabledir / "qcd_iht20k_totals.csv", index=False)
        write_markdown(totals20, tabledir / "qcd_iht20k_totals.md")

    # Optional weighted mHH shape by slice if candidate parquets exist
    frames = []
    missing = []
    for _, row in df20.iterrows():
        tag = row["tag"]
        p = find_candidate_parquet(store, tag)
        if p is None:
            missing.append(tag)
            continue

        cand = pd.read_parquet(p)
        if "mhh" not in cand.columns:
            missing.append(tag)
            continue

        cand = cand.copy()
        cand["weight_pb"] = float(row["xsec_pb"]) / float(row["n_generated"])
        cand["ht_slice_display"] = row["ht_slice_display"]
        frames.append(cand[["mhh", "weight_pb", "ht_slice_display"]])

    if frames:
        allcand = pd.concat(frames, ignore_index=True)

        plt.figure(figsize=(7.8, 5.0))
        bins = np.linspace(200, 1400, 49)
        for label in ["100–200", "200–400", "400–600", "600+"]:
            sub = allcand[allcand["ht_slice_display"] == label]
            if len(sub) == 0:
                continue
            plt.hist(
                sub["mhh"],
                bins=bins,
                weights=sub["weight_pb"],
                histtype="step",
                linewidth=1.6,
                label=label,
            )
        plt.xlabel("Reconstructed mHH [GeV]")
        plt.ylabel("Weighted candidate cross section [pb / bin]")
        plt.title("QCD bbbb HT-sliced 20k: weighted mHH by slice")
        plt.legend(title="HT slice [GeV]")
        plt.tight_layout()
        plt.savefig(outdir / "qcd_iht20k_weighted_mhh_by_slice.png", dpi=200)
        plt.close()

        plt.figure(figsize=(7.8, 5.0))
        plt.hist(
            allcand["mhh"],
            bins=np.linspace(200, 1400, 49),
            weights=allcand["weight_pb"],
            histtype="step",
            linewidth=1.8,
        )
        plt.xlabel("Reconstructed mHH [GeV]")
        plt.ylabel("Weighted candidate cross section [pb / bin]")
        plt.title("QCD bbbb HT-sliced 20k: combined weighted mHH")
        plt.tight_layout()
        plt.savefig(outdir / "qcd_iht20k_combined_weighted_mhh.png", dpi=200)
        plt.close()

    if missing:
        print("\nWarning: no candidate parquet found for these tags, so weighted mHH plots may be incomplete:")
        for tag in missing:
            print(" ", tag)

    print("\nWrote plots to:", outdir)
    print("Wrote tables to:", tabledir)

    print("\nKey totals:")
    print(totals20.to_string(index=False))


if __name__ == "__main__":
    main()
