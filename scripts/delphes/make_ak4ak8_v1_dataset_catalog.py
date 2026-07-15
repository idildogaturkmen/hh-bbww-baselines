#!/usr/bin/env python3

from pathlib import Path
import os
import re
import json
import pandas as pd

STORE = Path(os.environ["HH4B_STORE"])
REPO = Path(os.environ["HH4B_REPO"])
OUTDIR = REPO / "datasets/ak4ak8_v1"
OUTDIR.mkdir(parents=True, exist_ok=True)

SAMPLES = [
    ("signal", "ggF_HH_4b", "ggf_hh4b_ak4ak8_10k"),
    ("signal", "VBF_HH_4b", "vbf_hh4b_ak4ak8_10k"),

    ("background", "ttbar", "ttbar_extra50k_ak4ak8_v1_shard000"),
    ("background", "ttbar", "ttbar_extra50k_ak4ak8_v1_shard001"),
    ("background", "ttbar", "ttbar_extra50k_ak4ak8_v1_shard002"),
    ("background", "ttbar", "ttbar_extra50k_ak4ak8_v1_shard003"),
    ("background", "ttbar", "ttbar_extra50k_ak4ak8_v1_shard004"),

    ("background", "QCD_bbbb", "qcd_bbbb_ak4ak8_extra50k_v1_shard000"),
    ("background", "QCD_bbbb", "qcd_bbbb_ak4ak8_extra50k_v1_shard001"),
    ("background", "QCD_bbbb", "qcd_bbbb_ak4ak8_extra50k_v1_shard002"),
    ("background", "QCD_bbbb", "qcd_bbbb_ak4ak8_extra50k_v1_shard003"),
    ("background", "QCD_bbbb", "qcd_bbbb_ak4ak8_extra50k_v1_shard004"),

    ("background", "Zbbbb", "zbbbb_ak4ak8_extra50k_v1_shard000"),
    ("background", "Zbbbb", "zbbbb_ak4ak8_extra50k_v1_shard001"),
    ("background", "Zbbbb", "zbbbb_ak4ak8_extra50k_v1_shard002"),
    ("background", "Zbbbb", "zbbbb_ak4ak8_extra50k_v1_shard003"),
    ("background", "Zbbbb", "zbbbb_ak4ak8_extra50k_v1_shard004"),

    ("validation", "ZZ_4b", "zz4b_ak4ak8_pilot10k"),
    ("validation", "ZH_4b", "zh4b_ak4ak8_pilot10k"),
]

def parse_summary(path):
    out = {}
    if not path.exists():
        return out
    text = path.read_text()
    keys = [
        "events",
        "candidate rows",
        "cross-section median pb",
        "events with >=4 selected jets",
        "events with >=4 selected b-tagged jets",
        "median mbb1",
        "median mbb2",
        "median avg_mbb",
        "median delta_mbb",
        "median mhh",
        "median n_selected_bjets",
    ]
    for key in keys:
        m = re.search(rf"^{re.escape(key)}:\s*([^\n]+)", text, flags=re.M)
        if m:
            clean = key.replace(" ", "_").replace(">=", "ge").replace("-", "_")
            out[clean] = m.group(1).strip()
    return out

def size_gb(path):
    return path.stat().st_size / 1024**3 if path.exists() else None

rows = []

for role, process, tag in SAMPLES:
    root = STORE / "root" / f"{tag}_pythia8_delphes.root"
    hepmc = STORE / "hepmc" / f"{tag}_pythia8.hepmc"
    event = STORE / "parquet" / f"{tag}_event_summary.parquet"
    cand = STORE / "parquet" / f"{tag}_hh4b_candidates.parquet"
    diag = STORE / "metadata" / f"{tag}_diagnostic.csv"
    summary = STORE / "metadata" / f"{tag}_summary.txt"

    row = {
        "role": role,
        "process": process,
        "tag": tag,
        "root_path": str(root),
        "root_exists": root.exists(),
        "root_size_GB": size_gb(root),
        "hepmc_path": str(hepmc),
        "hepmc_exists": hepmc.exists(),
        "hepmc_size_GB": size_gb(hepmc),
        "event_summary_parquet": str(event),
        "event_summary_exists": event.exists(),
        "candidate_parquet": str(cand),
        "candidate_parquet_exists": cand.exists(),
        "diagnostic_csv": str(diag),
        "diagnostic_exists": diag.exists(),
        "summary_txt": str(summary),
        "summary_exists": summary.exists(),
    }
    row.update(parse_summary(summary))
    rows.append(row)

df = pd.DataFrame(rows)

for col in [
    "events", "candidate_rows", "cross_section_median_pb",
    "events_with_ge4_selected_jets", "events_with_ge4_selected_b_tagged_jets",
    "root_size_GB", "hepmc_size_GB"
]:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")

df.to_csv(OUTDIR / "ak4ak8_v1_local_manifest.csv", index=False)
(OUTDIR / "ak4ak8_v1_local_manifest.md").write_text(df.to_markdown(index=False) + "\n")
(OUTDIR / "ak4ak8_v1_local_manifest.json").write_text(json.dumps(rows, indent=2))

by_process = (
    df.groupby(["role", "process"], as_index=False)
    .agg(
        n_shards=("tag", "count"),
        total_events=("events", "sum"),
        total_candidate_rows=("candidate_rows", "sum"),
        median_xsec_pb=("cross_section_median_pb", "median"),
        total_root_GB=("root_size_GB", "sum"),
        total_hepmc_GB=("hepmc_size_GB", "sum"),
    )
)

by_process.to_csv(OUTDIR / "ak4ak8_v1_process_summary.csv", index=False)
(OUTDIR / "ak4ak8_v1_process_summary.md").write_text(by_process.to_markdown(index=False) + "\n")

readme = """# AK4/AK8 v1 local Delphes dataset catalog

This folder records the local AK4/AK8 Delphes samples used for the HH→4b study.

The large ROOT and HepMC files are not committed to git. This catalog stores sample names, local file paths, event counts, candidate counts, generator cross sections, and derived parquet locations.

For a public release, these local paths should later be replaced by EOS/XRootD/Zenodo paths and checksums.
"""
(OUTDIR / "README.md").write_text(readme)

print("=== Process summary ===")
print(by_process.to_string(index=False))
print("\nWrote:", OUTDIR)
