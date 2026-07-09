#!/usr/bin/env python3

import os
from pathlib import Path

import pandas as pd


STORE = Path(os.environ["HH4B_STORE"])
REPO = Path(os.environ["HH4B_REPO"])

OUTDIR = REPO / "outputs/tables/signal_normalization_whatif_2026_07_08"
OUTDIR.mkdir(parents=True, exist_ok=True)

LUMI_FB = 450.0

# What-if external SM xsecs × BR(H→bb)^2.
# Keep labeled as what-if until Harvey confirms.
OFFICIAL_SIGNAL_XSEC_FB = {
    "ggF": 10.55,
    "VBF": 0.587,
}

# Latest MG5 sanity-check values.
MG5_SIGNAL_XSEC_FB = {
    "ggF": 0.99443,
    "VBF": 0.94488703,
}

N_GENERATED = {
    "ggF": 10000,
    "VBF": 10000,
}


def find_required_10k_signal(process):
    parquet_dir = STORE / "parquet"

    if process == "ggF":
        patterns = [
            "HH4b_ggf_hh4b_10k*merged*hh4b_candidates.parquet",
            "*ggf*10k*merged*hh4b_candidates.parquet",
        ]
    elif process == "VBF":
        patterns = [
            "HH4b_vbf_hh4b_10k*merged*hh4b_candidates.parquet",
            "*vbf*10k*merged*hh4b_candidates.parquet",
        ]
    else:
        raise ValueError(process)

    matches = []
    for pat in patterns:
        matches.extend(sorted(parquet_dir.glob(pat)))

    matches = [p for p in matches if "test" not in p.name.lower() and "_1k_" not in p.name.lower()]

    if not matches:
        raise FileNotFoundError(f"No 10k non-test {process} candidate parquet found.")

    return matches[-1]


def r_hh_125_125(df):
    return ((df["mbb1"] - 125.0) ** 2 + (df["mbb2"] - 125.0) ** 2) ** 0.5


REGIONS = {
    "all_candidates": lambda df: pd.Series(True, index=df.index),
    "rhh_125_125_lt_30": lambda df: r_hh_125_125(df) < 30,
    "rhh_125_125_lt_55": lambda df: r_hh_125_125(df) < 55,
    "mhh_gt_400": lambda df: df["mhh"] > 400,
    "mhh_gt_600": lambda df: df["mhh"] > 600,
    "rhh_lt_30_and_mhh_gt_400": lambda df: (r_hh_125_125(df) < 30) & (df["mhh"] > 400),
    "rhh_lt_55_and_mhh_gt_400": lambda df: (r_hh_125_125(df) < 55) & (df["mhh"] > 400),
}


def main():
    file_rows = []
    detail_rows = []

    for process in ["ggF", "VBF"]:
        path = find_required_10k_signal(process)
        df = pd.read_parquet(path)

        n_generated = N_GENERATED[process]
        n_candidates = len(df)

        mg5_xsec_fb = MG5_SIGNAL_XSEC_FB[process]
        official_xsec_fb = OFFICIAL_SIGNAL_XSEC_FB[process]

        file_rows.append(
            {
                "process": process,
                "path": str(path),
                "n_generated": n_generated,
                "candidate_rows": n_candidates,
                "candidate_eff": n_candidates / n_generated,
                "mg5_xsec_fb": mg5_xsec_fb,
                "official_whatif_xsec_fb": official_xsec_fb,
                "scale_official_over_mg5": official_xsec_fb / mg5_xsec_fb,
                "mg5_candidate_xsec_fb": mg5_xsec_fb * n_candidates / n_generated,
                "official_candidate_xsec_fb": official_xsec_fb * n_candidates / n_generated,
                "mg5_candidate_events_450fb": mg5_xsec_fb * n_candidates / n_generated * LUMI_FB,
                "official_candidate_events_450fb": official_xsec_fb * n_candidates / n_generated * LUMI_FB,
            }
        )

        for region_name, selector in REGIONS.items():
            mask = selector(df)
            n_pass = int(mask.sum())
            eff_vs_generated = n_pass / n_generated
            eff_vs_candidates = n_pass / n_candidates if n_candidates else 0.0

            detail_rows.append(
                {
                    "process": process,
                    "region": region_name,
                    "n_generated": n_generated,
                    "candidate_rows_total": n_candidates,
                    "rows_pass_region": n_pass,
                    "eff_vs_generated": eff_vs_generated,
                    "eff_vs_candidates": eff_vs_candidates,
                    "mg5_region_xsec_fb": mg5_xsec_fb * eff_vs_generated,
                    "official_whatif_region_xsec_fb": official_xsec_fb * eff_vs_generated,
                    "mg5_expected_events_450fb": mg5_xsec_fb * eff_vs_generated * LUMI_FB,
                    "official_whatif_expected_events_450fb": official_xsec_fb * eff_vs_generated * LUMI_FB,
                }
            )

    files = pd.DataFrame(file_rows)
    detail = pd.DataFrame(detail_rows)

    combined = (
        detail.groupby("region", as_index=False)
        .agg(
            rows_pass_region=("rows_pass_region", "sum"),
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

    files.to_csv(OUTDIR / "signal_files_and_scale_factors.csv", index=False)
    detail.to_csv(OUTDIR / "signal_normalization_whatif_by_process.csv", index=False)
    combined.to_csv(OUTDIR / "signal_normalization_whatif_combined.csv", index=False)

    (OUTDIR / "signal_files_and_scale_factors.md").write_text(files.to_markdown(index=False) + "\n")
    (OUTDIR / "signal_normalization_whatif_by_process.md").write_text(detail.to_markdown(index=False) + "\n")
    (OUTDIR / "signal_normalization_whatif_combined.md").write_text(combined.to_markdown(index=False) + "\n")

    note = """# Signal normalization what-if study

Corrected version.

This uses the number of generated events as the denominator, not the number of candidate rows. Therefore, the reported expected events are after the HH4b candidate selection.

This is still a what-if study until Harvey confirms the signal normalization convention.
"""
    (OUTDIR / "README.md").write_text(note)

    print("\n=== Signal files and scale factors ===")
    print(files.to_string(index=False))

    print("\n=== Combined corrected signal expected events by region ===")
    print(combined.to_string(index=False))

    print(f"\nWrote outputs to: {OUTDIR}")


if __name__ == "__main__":
    main()
