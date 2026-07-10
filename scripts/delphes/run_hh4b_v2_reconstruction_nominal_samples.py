#!/usr/bin/env python3

import os
import re
import subprocess
from pathlib import Path

import pandas as pd
import uproot


REPO = Path(os.environ["HH4B_REPO"])
STORE = Path(os.environ["HH4B_STORE"])
ROOTDIR = STORE / "root"
PARQ = STORE / "parquet"

V2_DIR = STORE / "parquet_v2_nominal"
TMP_DIR = V2_DIR / "per_root"
V2_DIR.mkdir(parents=True, exist_ok=True)
TMP_DIR.mkdir(parents=True, exist_ok=True)

RECO_SCRIPT = REPO / "scripts/delphes/reconstruct_hh4b_candidates_v2.py"

QCD_META = Path(os.environ.get(
    "QCD_META",
    STORE / "metadata/qcd_bbbb_iht_slice_scan_20k_with_iht200to400_combined120k.csv",
))


def find_one_parquet(patterns):
    matches = []
    for pat in patterns:
        matches.extend(sorted(PARQ.glob(pat)))
    matches = [p for p in matches if "test" not in p.name.lower()]
    if not matches:
        raise FileNotFoundError(f"No parquet match for {patterns}")
    return matches[-1]


def campaign_from_parquet_name(path):
    m = re.search(r"(transfer_cluster_[0-9]+)", path.name)
    if not m:
        raise RuntimeError(f"Could not infer transfer_cluster campaign from {path.name}")
    return m.group(1)


def n_events_root(path):
    with uproot.open(path) as f:
        return f["Delphes"].num_entries


def unique_sorted(paths):
    return sorted(set(Path(p) for p in paths))


def roots_from_patterns(patterns):
    roots = []
    for pat in patterns:
        roots.extend(ROOTDIR.glob(pat))
    return unique_sorted(roots)


def roots_for_signal(sample):
    if sample == "ggF_HH4b_SMnorm":
        cand = find_one_parquet(["HH4b_ggf_hh4b_10k*merged*hh4b_candidates.parquet"])
        campaign = campaign_from_parquet_name(cand)
        roots = roots_from_patterns([f"HH4b_ggf_hh4b_10k_{campaign}_shard_*_pythia8_delphes.root"])
        return campaign, roots

    if sample == "VBF_HH4b_SMnorm":
        cand = find_one_parquet(["HH4b_vbf_hh4b_10k*merged*hh4b_candidates.parquet"])
        campaign = campaign_from_parquet_name(cand)
        roots = roots_from_patterns([f"HH4b_vbf_hh4b_10k_{campaign}_shard_*_pythia8_delphes.root"])
        return campaign, roots

    raise ValueError(sample)


manifest = {
    "ggF_HH4b_SMnorm": {
        "group": "signal",
        "n_generated": 10000,
        "xsec_pb": 10.55 / 1000.0,
        "roots": None,
    },
    "VBF_HH4b_SMnorm": {
        "group": "signal",
        "n_generated": 10000,
        "xsec_pb": 0.587 / 1000.0,
        "roots": None,
    },
    "ttbar_200k": {
        "group": "background",
        "n_generated": 200000,
        "xsec_pb": 512.2746276855469,
        "roots": roots_from_patterns(["ttbar_200k*delphes.root", "ttbar*200k*delphes.root"]),
    },
    "Zbbbb_100k": {
        "group": "background",
        "n_generated": 100000,
        "xsec_pb": 6.912992,
        "roots": roots_from_patterns(["zbbbb_presel_100k*delphes.root", "Zbbbb*100k*delphes.root"]),
    },
    "qcd_bbbb_iht100to200_20000": {
        "group": "background",
        "n_generated": 20000,
        "xsec_pb": 206.239914,
        "roots": roots_from_patterns(["qcd_bbbb_iht100to200_20000*delphes.root"]),
    },
    "qcd_bbbb_iht200to400_combined120k": {
        "group": "background",
        "n_generated": 120000,
        "xsec_pb": 126.352264,
        "roots": roots_from_patterns([
            "qcd_bbbb_iht200to400_20000*delphes.root",
            "qcd_bbbb_iht200to400_extra100k*delphes.root",
        ]),
    },
    "qcd_bbbb_iht400to600_20000": {
        "group": "background",
        "n_generated": 20000,
        "xsec_pb": 9.774867,
        "roots": roots_from_patterns(["qcd_bbbb_iht400to600_20000*delphes.root"]),
    },
    "qcd_bbbb_iht600plus_20000": {
        "group": "background",
        "n_generated": 20000,
        "xsec_pb": 1.787688,
        "roots": roots_from_patterns(["qcd_bbbb_iht600plus_20000*delphes.root"]),
    },
}

for sample in ["ggF_HH4b_SMnorm", "VBF_HH4b_SMnorm"]:
    campaign, roots = roots_for_signal(sample)
    manifest[sample]["campaign"] = campaign
    manifest[sample]["roots"] = roots

summary_rows = []

for sample, cfg in manifest.items():
    roots = cfg["roots"]
    if not roots:
        print(f"SKIPPING {sample}: no ROOT files found. Full v2 BDT comparison cannot use this sample yet.")

        summary_rows.append({
            "analysis_sample": sample,
            "group": cfg["group"],
            "campaign": cfg.get("campaign", ""),
            "n_generated": cfg["n_generated"],
            "xsec_pb": cfg["xsec_pb"],
            "n_root_files": 0,
            "n_root_events": 0,
            "candidate_rows_v2": -1,
            "candidate_efficiency_v2": float("nan"),
            "output": "",
            "status": "SKIPPED_NO_ROOT_FILES",
        })
        continue

    root_events = sum(n_events_root(p) for p in roots)
    expected = cfg["n_generated"]

    print(f"\n=== {sample} ===")
    print(f"ROOT files: {len(roots)}")
    print(f"ROOT events: {root_events}")
    print(f"Expected generated events: {expected}")

    if root_events != expected:
        print("\nSelected ROOT files:")
        for p in roots:
            print(" ", p)

        summary_rows.append({
            "analysis_sample": sample,
            "group": cfg["group"],
            "campaign": cfg.get("campaign", ""),
            "n_generated": cfg["n_generated"],
            "xsec_pb": cfg["xsec_pb"],
            "n_root_files": len(roots),
            "n_root_events": root_events,
            "candidate_rows_v2": -1,
            "candidate_efficiency_v2": float("nan"),
            "output": "",
            "status": f"SKIPPED_ROOT_EVENT_MISMATCH_expected_{expected}_found_{root_events}",
        })

        print(
            f"SKIPPING {sample}: ROOT event count {root_events} does not match expected {expected}. "
            "This avoids sample double counting or undercounting."
        )
        continue

    per_root_frames = []

    for i, root in enumerate(roots):
        out = TMP_DIR / f"{sample}__{i:03d}__{root.stem}_v2_hh4b_candidates.parquet"

        if not out.exists():
            cmd = [
                "python3",
                str(RECO_SCRIPT),
                "--input",
                str(root),
                "--out",
                str(out),
                "--sample",
                root.stem,
                "--max-bjets-for-pairing",
                "8",
                "--higgs-ordering",
                "pt",
            ]
            print("Running:", " ".join(cmd))
            subprocess.run(cmd, check=True)
        else:
            print("Exists:", out.name)

        if out.exists():
            df = pd.read_parquet(out)
            if len(df):
                df = df.copy()
                df["analysis_sample"] = sample
                df["source_root"] = root.name
                df["source_root_index"] = i
                per_root_frames.append(df)

    if per_root_frames:
        merged = pd.concat(per_root_frames, ignore_index=True)
    else:
        merged = pd.DataFrame()

    merged_out = V2_DIR / f"{sample}_v2_hh4b_candidates.parquet"
    merged.to_parquet(merged_out, index=False)

    summary_rows.append({
        "analysis_sample": sample,
        "group": cfg["group"],
        "campaign": cfg.get("campaign", ""),
        "n_generated": cfg["n_generated"],
        "xsec_pb": cfg["xsec_pb"],
        "n_root_files": len(roots),
        "n_root_events": root_events,
        "candidate_rows_v2": len(merged),
        "candidate_efficiency_v2": len(merged) / cfg["n_generated"],
        "output": str(merged_out),
        "status": "OK",
    })

summary = pd.DataFrame(summary_rows)
summary_out = V2_DIR / "v2_nominal_reconstruction_summary.csv"
summary.to_csv(summary_out, index=False)
(V2_DIR / "v2_nominal_reconstruction_summary.md").write_text(summary.to_markdown(index=False) + "\n")

print("\n=== v2 nominal reconstruction summary ===")
print(summary.to_string(index=False))
print(f"\nWrote v2 parquets to: {V2_DIR}")
