#!/usr/bin/env python3

import os
from pathlib import Path
import pandas as pd
import uproot

STORE = Path(os.environ["HH4B_STORE"])
ROOTDIR = STORE / "root"
PARQ = STORE / "parquet"
REPO = Path(os.environ["HH4B_REPO"])

OUTDIR = REPO / "outputs/tables/v2_root_availability_audit_2026_07_09"
OUTDIR.mkdir(parents=True, exist_ok=True)

samples = {
    "ggF_HH4b_SMnorm": {
        "expected": 10000,
        "root_patterns": ["HH4b_ggf_hh4b_10k_transfer_cluster_84730223_shard_*_pythia8_delphes.root"],
        "parquet_patterns": ["HH4b_ggf_hh4b_10k_transfer_cluster_84730223_merged_10k_hh4b_candidates.parquet"],
    },
    "VBF_HH4b_SMnorm": {
        "expected": 10000,
        "root_patterns": ["HH4b_vbf_hh4b_10k_transfer_cluster_3473183_shard_*_pythia8_delphes.root"],
        "parquet_patterns": ["HH4b_vbf_hh4b_10k_transfer_cluster_3473183_merged_10k_hh4b_candidates.parquet"],
    },
    "ttbar_200k": {
        "expected": 200000,
        "root_patterns": ["ttbar_200k_shard*_pythia8_delphes.root"],
        "parquet_patterns": ["ttbar_200k_merged_hh4b_candidates.parquet"],
    },
    "Zbbbb_100k": {
        "expected": 100000,
        "root_patterns": ["zbbbb_presel_100k_shard*_pythia8_delphes.root", "Zbbbb*100k*delphes.root"],
        "parquet_patterns": ["zbbbb_presel_100k_merged_hh4b_candidates.parquet"],
    },
    "qcd_bbbb_iht100to200_20000": {
        "expected": 20000,
        "root_patterns": ["qcd_bbbb_iht100to200_20000*delphes.root"],
        "parquet_patterns": ["qcd_bbbb_iht100to200_20000_hh4b_candidates.parquet"],
    },
    "qcd_bbbb_iht200to400_combined120k": {
        "expected": 120000,
        "root_patterns": ["qcd_bbbb_iht200to400_20000*delphes.root", "qcd_bbbb_iht200to400_extra100k*delphes.root"],
        "parquet_patterns": ["qcd_bbbb_iht200to400_combined120k_merged_hh4b_candidates.parquet"],
    },
    "qcd_bbbb_iht400to600_20000": {
        "expected": 20000,
        "root_patterns": ["qcd_bbbb_iht400to600_20000*delphes.root"],
        "parquet_patterns": ["qcd_bbbb_iht400to600_20000_hh4b_candidates.parquet"],
    },
    "qcd_bbbb_iht600plus_20000": {
        "expected": 20000,
        "root_patterns": ["qcd_bbbb_iht600plus_20000*delphes.root"],
        "parquet_patterns": ["qcd_bbbb_iht600plus_20000_hh4b_candidates.parquet"],
    },
}

def unique(paths):
    out = []
    seen = set()
    for p in paths:
        if p not in seen:
            seen.add(p)
            out.append(p)
    return sorted(out)

def count_root_events(files):
    total = 0
    bad = []
    for p in files:
        try:
            with uproot.open(p) as f:
                total += f["Delphes"].num_entries
        except Exception as e:
            bad.append(f"{p.name}: {e}")
    return total, bad

rows = []

for sample, cfg in samples.items():
    roots = []
    for pat in cfg["root_patterns"]:
        roots.extend(ROOTDIR.glob(pat))
    roots = unique(roots)

    parquets = []
    for pat in cfg["parquet_patterns"]:
        parquets.extend(PARQ.glob(pat))
    parquets = unique(parquets)

    root_events, bad = count_root_events(roots)

    parquet_rows = None
    if parquets:
        try:
            parquet_rows = sum(len(pd.read_parquet(p)) for p in parquets)
        except Exception:
            parquet_rows = -1

    expected = cfg["expected"]

    if root_events == expected:
        status = "ROOT_COMPLETE"
    elif root_events == 0:
        status = "ROOT_MISSING"
    else:
        status = f"ROOT_PARTIAL_{root_events}_of_{expected}"

    rows.append({
        "sample": sample,
        "expected_generated_events": expected,
        "root_files_found": len(roots),
        "root_events_found": root_events,
        "root_status": status,
        "parquet_files_found": len(parquets),
        "parquet_candidate_rows": parquet_rows,
        "root_patterns": ";".join(cfg["root_patterns"]),
        "parquet_patterns": ";".join(cfg["parquet_patterns"]),
        "bad_root_files": ";".join(bad),
    })

df = pd.DataFrame(rows)
df.to_csv(OUTDIR / "v2_root_availability_audit.csv", index=False)
(OUTDIR / "v2_root_availability_audit.md").write_text(df.to_markdown(index=False) + "\n")

print(df.to_string(index=False))
print(f"\nWrote outputs to: {OUTDIR}")
