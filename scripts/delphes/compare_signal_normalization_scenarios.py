#!/usr/bin/env python3

import os
from pathlib import Path

import pandas as pd


STORE = Path(os.environ["HH4B_STORE"])
REPO = Path(os.environ["HH4B_REPO"])

OUTDIR = REPO / "outputs/tables/signal_normalization_whatif_2026_07_08"
OUTDIR.mkdir(parents=True, exist_ok=True)

LUMI_FB = 450.0

# Approximate external SM references discussed with Harvey.
# These should remain labeled as "what-if" until Harvey confirms.
OFFICIAL_SIGNAL_XSEC_FB = {
    "ggF": 10.55,
    "VBF": 0.587,
}

MG5_SIGNAL_XSEC_FB = {
    "ggF": 0.99443,
    "VBF": 0.944887,
}

REGIONS = {
    "all_candidates": lambda df: pd.Series(True, index=df.index),
    "rhh_125_125_lt_30": lambda df: df["r_hh_125_125"] < 30 if "r_hh_125_125" in df.columns else ((df["mbb1"] - 125.0) ** 2 + (df["mbb2"] - 125.0) ** 2) ** 0.5 < 30,
    "rhh_125_125_lt_55": lambda df: df["r_hh_125_125"] < 55 if "r_hh_125_125" in df.columns else ((df["mbb1"] - 125.0) ** 2 + (df["mbb2"] - 125.0) ** 2) ** 0.5 < 55,
    "mhh_gt_400": lambda df: df["mhh"] > 400,
    "mhh_gt_600": lambda df: df["mhh"] > 600,
    "rhh_lt_30_and_mhh_gt_400": lambda df: (((df["mbb1"] - 125.0) ** 2 + (df["mbb2"] - 125.0) ** 2) ** 0.5 < 30) & (df["mhh"] > 400),
    "rhh_lt_55_and_mhh_gt_400": lambda df: (((df["mbb1"] - 125.0) ** 2 + (df["mbb2"] - 125.0) ** 2) ** 0.5 < 55) & (df["mhh"] > 400),
}


def find_one(patterns):
    matches = []
    for pat in patterns:
        matches.extend(sorted((STORE / "parquet").glob(pat)))
    if not matches:
        return None
    return matches[-1]


def load_signal():
    paths = {
        "ggF": find_one([
            "*ggf*merged*hh4b_candidates.parquet",
            "*ggF*merged*hh4b_candidates.parquet",
            "*GGF*merged*hh4b_candidates.parquet",
        ]),
        "VBF": find_one([
            "*vbf*merged*hh4b_candidates.parquet",
            "*VBF*merged*hh4b_candidates.parquet",
        ]),
    }

    rows = []
    frames = {}

    for proc, path in paths.items():
        if path is None:
            print(f"WARNING: no file found for {proc}")
            continue

        df = pd.read_parquet(path)
        frames[proc] = df

        rows.append({
            "process": proc,
            "path": str(path),
            "candidate_rows": len(df),
            "mg5_xsec_fb": MG5_SIGNAL_XSEC_FB[proc],
            "official_whatif_xsec_fb": OFFICIAL_SIGNAL_XSEC_FB[proc],
            "scale_official_over_mg5": OFFICIAL_SIGNAL_XSEC_FB[proc] / MG5_SIGNAL_XSEC_FB[proc],
        })

    file_summary = pd.DataFrame(rows)
    return frames, file_summary


def main():
    frames, file_summary = load_signal()

    if not frames:
        raise RuntimeError("No signal frames found.")

    file_summary.to_csv(OUTDIR / "signal_files_and_scale_factors.csv", index=False)
    (OUTDIR / "signal_files_and_scale_factors.md").write_text(file_summary.to_markdown(index=False) + "\n")

    rows = []

    for proc, df in frames.items():
        n_total = len(df)
        if n_total == 0:
            continue

        mg5_xsec_fb = MG5_SIGNAL_XSEC_FB[proc]
        official_xsec_fb = OFFICIAL_SIGNAL_XSEC_FB[proc]

        for region_name, selector in REGIONS.items():
            mask = selector(df)
            n_pass = int(mask.sum())
            eff = n_pass / n_total

            rows.append({
                "process": proc,
                "region": region_name,
                "candidate_rows_total": n_total,
                "candidate_rows_pass": n_pass,
                "candidate_level_eff": eff,
                "mg5_xsec_fb": mg5_xsec_fb,
                "official_whatif_xsec_fb": official_xsec_fb,
                "mg5_region_xsec_fb": mg5_xsec_fb * eff,
                "official_whatif_region_xsec_fb": official_xsec_fb * eff,
                "mg5_expected_events_450fb": mg5_xsec_fb * eff * LUMI_FB,
                "official_whatif_expected_events_450fb": official_xsec_fb * eff * LUMI_FB,
                "official_over_mg5_scale": official_xsec_fb / mg5_xsec_fb,
            })

    detail = pd.DataFrame(rows)

    combined = (
        detail.groupby("region", as_index=False)
        .agg(
            mg5_region_xsec_fb=("mg5_region_xsec_fb", "sum"),
            official_whatif_region_xsec_fb=("official_whatif_region_xsec_fb", "sum"),
            mg5_expected_events_450fb=("mg5_expected_events_450fb", "sum"),
            official_whatif_expected_events_450fb=("official_whatif_expected_events_450fb", "sum"),
        )
    )

    combined["official_over_mg5_expected_event_ratio"] = (
        combined["official_whatif_expected_events_450fb"]
        / combined["mg5_expected_events_450fb"]
    )

    detail.to_csv(OUTDIR / "signal_normalization_whatif_by_process.csv", index=False)
    combined.to_csv(OUTDIR / "signal_normalization_whatif_combined.csv", index=False)

    (OUTDIR / "signal_normalization_whatif_by_process.md").write_text(detail.to_markdown(index=False) + "\n")
    (OUTDIR / "signal_normalization_whatif_combined.md").write_text(combined.to_markdown(index=False) + "\n")

    note = f"""# Signal normalization what-if study

This is an analysis-only diagnostic while waiting for Harvey's guidance.

It compares current raw MG5 signal normalization to an external-normalization scenario using approximate SM HH cross sections times BR(H→bb)^2:

- ggF HH→4b external what-if: {OFFICIAL_SIGNAL_XSEC_FB['ggF']} fb
- VBF HH→4b external what-if: {OFFICIAL_SIGNAL_XSEC_FB['VBF']} fb

Current MG5 signal cross sections used for comparison:

- ggF HH→4b MG5: {MG5_SIGNAL_XSEC_FB['ggF']} fb
- VBF HH→4b MG5: {MG5_SIGNAL_XSEC_FB['VBF']} fb

This should not be treated as the final normalization until Harvey confirms the convention.
"""

    (OUTDIR / "README.md").write_text(note)

    print("\n=== Signal files and scale factors ===")
    print(file_summary.to_string(index=False))

    print("\n=== Combined signal expected events by region ===")
    print(combined.to_string(index=False))

    print(f"\nWrote outputs to: {OUTDIR}")


if __name__ == "__main__":
    main()
