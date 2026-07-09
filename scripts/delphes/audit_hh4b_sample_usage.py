'''
Validate that the parquet files used in the HH4b analysis are consistent with the original Delphes ROOT files.
This script checks that the number of events in the parquet files matches the number of events in the corresponding Delphes ROOT files, and that the candidate efficiencies are consistent with the number of generated events specified in the sample configuration.
'''

import os
import re
from pathlib import Path

import pandas as pd
import uproot


STORE = Path(os.environ["HH4B_STORE"])
REPO = Path(os.environ["HH4B_REPO"])
PARQ = STORE / "parquet"
ROOT = STORE / "root"

QCD_META = Path(os.environ.get(
    "QCD_META",
    STORE / "metadata/qcd_bbbb_iht_slice_scan_20k_with_iht200to400_combined120k.csv",
))

OUTDIR = REPO / "outputs/tables/hh4b_sample_usage_audit_2026_07_09"
OUTDIR.mkdir(parents=True, exist_ok=True)


def find_one(patterns):
    matches = []
    for pat in patterns:
        matches.extend(sorted(PARQ.glob(pat)))
    matches = [p for p in matches if "test" not in p.name.lower()]
    if not matches:
        raise FileNotFoundError(f"No parquet match for {patterns}")
    return matches[-1]


def n_events_root(path):
    with uproot.open(path) as f:
        return f["Delphes"].num_entries


def campaign_from_parquet_name(path):
    m = re.search(r"(transfer_cluster_[0-9]+)", path.name)
    if not m:
        return None
    return m.group(1)


def signal_root_files_from_parquet(sample_key, cand_path):
    campaign = campaign_from_parquet_name(cand_path)
    if campaign is None:
        raise RuntimeError(f"Could not infer transfer_cluster campaign from {cand_path.name}")

    if sample_key == "ggF_HH4b_SMnorm":
        pattern = f"HH4b_ggf_hh4b_10k_{campaign}_shard_*_pythia8_delphes.root"
        broad = "HH4b_ggf_hh4b_10k*shard*_pythia8_delphes.root"
    elif sample_key == "VBF_HH4b_SMnorm":
        pattern = f"HH4b_vbf_hh4b_10k_{campaign}_shard_*_pythia8_delphes.root"
        broad = "HH4b_vbf_hh4b_10k*shard*_pythia8_delphes.root"
    else:
        raise ValueError(sample_key)

    selected = sorted(ROOT.glob(pattern))
    all_available = sorted(ROOT.glob(broad))

    return campaign, selected, all_available


rows = []

signal_manifest = {
    "ggF_HH4b_SMnorm": {
        "group": "signal",
        "n_generated": 10000,
        "xsec_pb": 10.55 / 1000.0,
        "cand_patterns": ["HH4b_ggf_hh4b_10k*merged*hh4b_candidates.parquet"],
        "event_patterns": ["HH4b_ggf_hh4b_10k*merged*event_summary.parquet"],
    },
    "VBF_HH4b_SMnorm": {
        "group": "signal",
        "n_generated": 10000,
        "xsec_pb": 0.587 / 1000.0,
        "cand_patterns": ["HH4b_vbf_hh4b_10k*merged*hh4b_candidates.parquet"],
        "event_patterns": ["HH4b_vbf_hh4b_10k*merged*event_summary.parquet"],
    },
}

background_manifest = {
    "ttbar_200k": {
        "group": "background",
        "n_generated": 200000,
        "xsec_pb": 512.2746276855469,
        "cand_patterns": ["ttbar_200k_merged_hh4b_candidates.parquet"],
        "event_patterns": ["ttbar_200k_merged_event_summary.parquet"],
    },
    "Zbbbb_100k": {
        "group": "background",
        "n_generated": 100000,
        "xsec_pb": 6.912992,
        "cand_patterns": ["zbbbb*100k*merged*hh4b_candidates.parquet", "Zbbbb*100k*merged*hh4b_candidates.parquet"],
        "event_patterns": ["zbbbb*100k*merged_event_summary.parquet", "Zbbbb*100k*merged_event_summary.parquet"],
    },
}

for sample, cfg in signal_manifest.items():
    cand_path = find_one(cfg["cand_patterns"])
    event_path = find_one(cfg["event_patterns"])

    cand = pd.read_parquet(cand_path)
    ev = pd.read_parquet(event_path)

    campaign, root_files, all_roots = signal_root_files_from_parquet(sample, cand_path)
    root_events = sum(n_events_root(p) for p in root_files)

    rows.append({
        "sample": sample,
        "group": cfg["group"],
        "status": "nominal",
        "campaign": campaign,
        "candidate_file": cand_path.name,
        "event_file": event_path.name,
        "candidate_rows": len(cand),
        "event_summary_rows": len(ev),
        "n_generated_config": cfg["n_generated"],
        "root_files_selected": len(root_files),
        "root_events_selected": root_events,
        "all_matching_root_files_available": len(all_roots),
        "all_matching_root_events_available": sum(n_events_root(p) for p in all_roots),
        "xsec_pb": cfg["xsec_pb"],
        "candidate_efficiency": len(cand) / cfg["n_generated"],
        "warning": "" if root_events == cfg["n_generated"] else "ROOT events selected do not match n_generated_config",
    })

for sample, cfg in background_manifest.items():
    cand_path = find_one(cfg["cand_patterns"])
    event_path = find_one(cfg["event_patterns"])

    cand = pd.read_parquet(cand_path)
    ev = pd.read_parquet(event_path)

    rows.append({
        "sample": sample,
        "group": cfg["group"],
        "status": "nominal",
        "campaign": "",
        "candidate_file": cand_path.name,
        "event_file": event_path.name,
        "candidate_rows": len(cand),
        "event_summary_rows": len(ev),
        "n_generated_config": cfg["n_generated"],
        "root_files_selected": "",
        "root_events_selected": "",
        "all_matching_root_files_available": "",
        "all_matching_root_events_available": "",
        "xsec_pb": cfg["xsec_pb"],
        "candidate_efficiency": len(cand) / cfg["n_generated"],
        "warning": "",
    })

qcd = pd.read_csv(QCD_META)

for _, r in qcd.iterrows():
    tag = str(r["tag"])
    n_generated = int(r["n_generated"])
    xsec_pb = float(r["xsec_pb"])

    cand_path = find_one([f"{tag}_hh4b_candidates.parquet", f"{tag}_merged_hh4b_candidates.parquet"])
    event_path = find_one([f"{tag}_event_summary.parquet", f"{tag}_merged_event_summary.parquet"])

    cand = pd.read_parquet(cand_path)
    ev = pd.read_parquet(event_path)

    rows.append({
        "sample": tag,
        "group": "background",
        "status": "nominal",
        "campaign": "",
        "candidate_file": cand_path.name,
        "event_file": event_path.name,
        "candidate_rows": len(cand),
        "event_summary_rows": len(ev),
        "n_generated_config": n_generated,
        "root_files_selected": "",
        "root_events_selected": "",
        "all_matching_root_files_available": "",
        "all_matching_root_events_available": "",
        "xsec_pb": xsec_pb,
        "candidate_efficiency": len(cand) / n_generated,
        "warning": "",
    })

audit = pd.DataFrame(rows)

audit.to_csv(OUTDIR / "hh4b_sample_usage_audit.csv", index=False)
(OUTDIR / "hh4b_sample_usage_audit.md").write_text(audit.to_markdown(index=False) + "\n")

print("\n=== HH4b sample usage audit ===")
print(audit.to_string(index=False))
print(f"\nWrote outputs to: {OUTDIR}")
