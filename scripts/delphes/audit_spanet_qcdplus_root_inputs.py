#!/usr/bin/env python3

import os
from pathlib import Path
import pandas as pd
import uproot

REPO = Path(os.environ["HH4B_REPO"])
STORE = Path(os.environ["HH4B_STORE"])
ROOT_DIR = STORE / "root"

OUTDIR = REPO / "outputs/tables/spanet_qcdplus_root_audit_2026_07_13"
OUTDIR.mkdir(parents=True, exist_ok=True)

CONFIGS = [
    {
        "sample": "ggF_HH4b_SMnorm",
        "group": "signal",
        "expected_generated": 10000,
        "patterns": ["HH4b_ggf_hh4b_10k_transfer_cluster_84730223_shard_*_pythia8_delphes.root"],
    },
    {
        "sample": "VBF_HH4b_SMnorm",
        "group": "signal",
        "expected_generated": 10000,
        "patterns": ["HH4b_vbf_hh4b_10k_transfer_cluster_3473183_shard_*_pythia8_delphes.root"],
    },
    {
        "sample": "ttbar_200k",
        "group": "background",
        "expected_generated": 200000,
        "patterns": ["ttbar_200k_shard*_pythia8_delphes.root"],
    },
    {
        "sample": "Zbbbb_100k",
        "group": "background",
        "expected_generated": 100000,
        "patterns": ["zbbbb_presel_100k_shard*_pythia8_delphes.root"],
    },
    {
        "sample": "qcd_bbbb_iht100to200_20000",
        "group": "background",
        "expected_generated": 20000,
        "patterns": ["qcd_bbbb_iht100to200_20000*delphes.root"],
    },
    {
        "sample": "qcd_bbbb_iht200to400_nominal120k",
        "group": "background",
        "expected_generated": 120000,
        "patterns": [
            "qcd_bbbb_iht200to400_20000*delphes.root",
            "qcd_bbbb_iht200to400_extra100k_shard*_pythia8_delphes.root",
        ],
    },
    {
        "sample": "qcd_bbbb_iht200to400_extra100k_v3",
        "group": "background",
        "expected_generated": 100000,
        "patterns": [
            "qcd_bbbb_iht200to400_extra100k_v3_shard*_pythia8_delphes.root",
        ],
    },
    {
        "sample": "qcd_bbbb_iht400to600_nominal20k",
        "group": "background",
        "expected_generated": 20000,
        "patterns": ["qcd_bbbb_iht400to600_20000*delphes.root"],
    },
    {
        "sample": "qcd_bbbb_iht400to600_extra100k_rootkeep_v1",
        "group": "background",
        "expected_generated": 100000,
        "patterns": [
            "qcd_bbbb_iht400to600_extra100k_rootkeep_v1_shard*_pythia8_delphes.root",
        ],
    },
    {
        "sample": "qcd_bbbb_iht600plus_20000",
        "group": "background",
        "expected_generated": 20000,
        "patterns": ["qcd_bbbb_iht600plus_20000*delphes.root"],
    },
]

def collect(patterns):
    files = []
    for pat in patterns:
        files.extend(sorted(ROOT_DIR.glob(pat)))
    out = []
    seen = set()
    for f in files:
        if f not in seen:
            out.append(f)
            seen.add(f)
    return out

def count_events(path):
    try:
        with uproot.open(path) as f:
            if "Delphes" not in f:
                return None, "missing Delphes tree"
            return int(f["Delphes"].num_entries), ""
    except Exception as e:
        return None, repr(e)

rows = []
file_rows = []

for cfg in CONFIGS:
    files = collect(cfg["patterns"])
    total_events = 0
    n_bad = 0

    for f in files:
        n, err = count_events(f)
        if n is None:
            n_bad += 1
            n_for_sum = 0
        else:
            n_for_sum = n
            total_events += n

        file_rows.append({
            "sample": cfg["sample"],
            "file": str(f),
            "events": n_for_sum,
            "error": err,
        })

    rows.append({
        "sample": cfg["sample"],
        "group": cfg["group"],
        "n_files": len(files),
        "root_events": total_events,
        "expected_generated": cfg["expected_generated"],
        "coverage_fraction": total_events / cfg["expected_generated"] if cfg["expected_generated"] else 0.0,
        "status": "OK" if total_events == cfg["expected_generated"] and n_bad == 0 else "CHECK",
        "patterns": "; ".join(cfg["patterns"]),
    })

summary = pd.DataFrame(rows)
details = pd.DataFrame(file_rows)

summary.to_csv(OUTDIR / "spanet_qcdplus_root_summary.csv", index=False)
details.to_csv(OUTDIR / "spanet_qcdplus_root_files.csv", index=False)

(OUTDIR / "spanet_qcdplus_root_summary.md").write_text(summary.to_markdown(index=False) + "\n")
(OUTDIR / "spanet_qcdplus_root_files.md").write_text(details.to_markdown(index=False) + "\n")

print("Wrote", OUTDIR)
print(summary.to_string(index=False))
