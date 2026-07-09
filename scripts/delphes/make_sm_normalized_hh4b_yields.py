'''
Normalized yields for SM HH4b analysis scenario.
Uses the generated MG5/Delphes samples for shapes and efficiencies, but normalizes the signal samples to approximate SM HH cross sections times BR(H→bb)^2:
- ggF HH→4b = 10.55 fb
- VBF HH→4b = 0.587 fb

Requires the following environment variables to be set:
- HH4B_STORE: path to the store directory containing the parquet files
- HH4B_REPO: path to the hh-bbww-baselines repository

'''

import os
from pathlib import Path
import numpy as np
import pandas as pd

STORE = Path(os.environ["HH4B_STORE"])
REPO = Path(os.environ["HH4B_REPO"])

OUTDIR = REPO / "outputs/tables/sm_normalized_hh4b_yields_2026_07_09"
OUTDIR.mkdir(parents=True, exist_ok=True)

LUMI_FB = 450.0
LUMI_PB = LUMI_FB * 1000.0

SIGNALS = {
    "ggF_HH4b_SMnorm": {
        "patterns": ["HH4b_ggf_hh4b_10k*merged*hh4b_candidates.parquet"],
        "n_generated": 10000,
        "xsec_pb": 10.55 / 1000.0,
        "group": "signal",
    },
    "VBF_HH4b_SMnorm": {
        "patterns": ["HH4b_vbf_hh4b_10k*merged*hh4b_candidates.parquet"],
        "n_generated": 10000,
        "xsec_pb": 0.587 / 1000.0,
        "group": "signal",
    },
}

BACKGROUNDS = {
    "ttbar_200k": {
        "patterns": ["ttbar_200k_merged_hh4b_candidates.parquet"],
        "n_generated": 200000,
        "xsec_pb": 512.2746276855469,
        "group": "background",
    },
    "Zbbbb_100k": {
        "patterns": ["zbbbb*100k*merged*hh4b_candidates.parquet", "Zbbbb*100k*merged*hh4b_candidates.parquet"],
        "n_generated": 100000,
        "xsec_pb": 6.912992,
        "group": "background",
    },
}

QCD_META = STORE / "metadata/qcd_bbbb_iht_slice_scan_20000.csv"


def find_one(patterns):
    matches = []
    for pat in patterns:
        matches += sorted((STORE / "parquet").glob(pat))
    matches = [p for p in matches if "test" not in p.name.lower()]
    if not matches:
        raise FileNotFoundError(f"No match for patterns: {patterns}")
    return matches[-1]


def rhh(df):
    return np.sqrt((df["mbb1"] - 125.0) ** 2 + (df["mbb2"] - 125.0) ** 2)


REGIONS = {
    "all_candidates": lambda df: np.ones(len(df), dtype=bool),
    "rhh_lt_30": lambda df: rhh(df) < 30,
    "rhh_lt_55": lambda df: rhh(df) < 55,
    "mhh_gt_400": lambda df: df["mhh"] > 400,
    "mhh_gt_600": lambda df: df["mhh"] > 600,
    "mhh_gt_800": lambda df: df["mhh"] > 800,
    "rhh_lt_30_and_mhh_gt_400": lambda df: (rhh(df) < 30) & (df["mhh"] > 400),
    "rhh_lt_55_and_mhh_gt_400": lambda df: (rhh(df) < 55) & (df["mhh"] > 400),
}


def add_sample_rows(rows, name, cfg):
    path = find_one(cfg["patterns"])
    df = pd.read_parquet(path)

    weight_pb = cfg["xsec_pb"] / cfg["n_generated"]

    for region, selector in REGIONS.items():
        mask = selector(df)
        n = int(mask.sum())
        xsec_pb = n * weight_pb

        rows.append({
            "sample": name,
            "group": cfg["group"],
            "region": region,
            "path": str(path),
            "n_generated": cfg["n_generated"],
            "candidate_rows_total": len(df),
            "rows_in_region": n,
            "region_xsec_pb": xsec_pb,
            "expected_events_450fb": xsec_pb * LUMI_PB,
            "rel_mc_stat_unc": 1 / np.sqrt(n) if n > 0 else np.nan,
        })


def add_qcd_rows(rows):
    meta = pd.read_csv(QCD_META)

    for _, m in meta.iterrows():
        tag = str(m["tag"])
        n_generated = int(m["n_generated"])
        xsec_pb = float(m["xsec_pb"])

        path = find_one([f"{tag}*merged*hh4b_candidates.parquet", f"{tag}*hh4b_candidates.parquet"])
        df = pd.read_parquet(path)

        weight_pb = xsec_pb / n_generated

        for region, selector in REGIONS.items():
            mask = selector(df)
            n = int(mask.sum())
            region_xsec_pb = n * weight_pb

            rows.append({
                "sample": tag,
                "group": "background",
                "region": region,
                "path": str(path),
                "n_generated": n_generated,
                "candidate_rows_total": len(df),
                "rows_in_region": n,
                "region_xsec_pb": region_xsec_pb,
                "expected_events_450fb": region_xsec_pb * LUMI_PB,
                "rel_mc_stat_unc": 1 / np.sqrt(n) if n > 0 else np.nan,
            })


def main():
    rows = []

    for name, cfg in SIGNALS.items():
        add_sample_rows(rows, name, cfg)

    for name, cfg in BACKGROUNDS.items():
        add_sample_rows(rows, name, cfg)

    add_qcd_rows(rows)

    detail = pd.DataFrame(rows)

    grouped = (
        detail.assign(process_group=lambda d: np.where(d["group"] == "signal", "signal", "background"))
        .groupby(["region", "process_group"], as_index=False)
        .agg(
            xsec_pb=("region_xsec_pb", "sum"),
            expected_events_450fb=("expected_events_450fb", "sum"),
            rows_in_region=("rows_in_region", "sum"),
        )
    )

    wide = grouped.pivot(index="region", columns="process_group", values=["xsec_pb", "expected_events_450fb", "rows_in_region"])
    wide.columns = [f"{a}_{b}" for a, b in wide.columns]
    wide = wide.reset_index()

    for col in [
        "xsec_pb_signal", "xsec_pb_background",
        "expected_events_450fb_signal", "expected_events_450fb_background",
    ]:
        if col not in wide.columns:
            wide[col] = 0.0

    wide["S_over_B"] = wide["expected_events_450fb_signal"] / wide["expected_events_450fb_background"]
    wide["S_over_sqrtB"] = wide["expected_events_450fb_signal"] / np.sqrt(wide["expected_events_450fb_background"])
    wide["S_over_10pctB"] = wide["expected_events_450fb_signal"] / (0.10 * wide["expected_events_450fb_background"])

    by_sample = (
        detail.groupby(["region", "sample", "group"], as_index=False)
        .agg(
            rows_in_region=("rows_in_region", "sum"),
            region_xsec_pb=("region_xsec_pb", "sum"),
            expected_events_450fb=("expected_events_450fb", "sum"),
        )
    )

    detail.to_csv(OUTDIR / "sm_normalized_yields_detail.csv", index=False)
    by_sample.to_csv(OUTDIR / "sm_normalized_yields_by_sample.csv", index=False)
    wide.to_csv(OUTDIR / "sm_normalized_region_summary.csv", index=False)

    (OUTDIR / "sm_normalized_yields_by_sample.md").write_text(by_sample.to_markdown(index=False) + "\n")
    (OUTDIR / "sm_normalized_region_summary.md").write_text(wide.to_markdown(index=False) + "\n")

    readme = """# SM-normalized HH4b yield summary

This is a provisional analysis scenario.

Signal samples are normalized externally to approximate SM HH cross sections times BR(H→bb)^2:
- ggF HH→4b = 10.55 fb
- VBF HH→4b = 0.587 fb

The generated MG5/Delphes samples are used for shapes and efficiencies. Backgrounds use their MG5 generator cross sections.

Nominal samples:
- ggF HH4b 10k
- VBF HH4b 10k
- QCD bbbb HT-sliced 20k/slice
- ttbar 200k
- Zbbbb 100k
"""
    (OUTDIR / "README.md").write_text(readme)

    print("\n=== SM-normalized region summary ===")
    print(wide.to_string(index=False))
    print(f"\nWrote outputs to: {OUTDIR}")


if __name__ == "__main__":
    main()
