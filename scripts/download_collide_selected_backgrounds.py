'''
This script downloads a small number of parquet files for selected background samples from the collide-1m dataset.
The selected samples are chosen to represent a variety of background processes, while avoiding excessive disk usage.
The downloaded files are stored in a local directory, which can be specified by the user.
The script uses the Hugging Face Hub API to list and download the parquet files.
Usage:
    python scripts/download_collide_selected_backgrounds.py --local-dir <output_directory> [--revision <commit_hash_or_tag>]
'''
from __future__ import annotations

import argparse
from collections import defaultdict, OrderedDict
from pathlib import Path

from huggingface_hub import HfApi, hf_hub_download

REPO_ID = "fastmachinelearning/collide-1m"
REPO_TYPE = "dataset"

# Download all relevant background CLASSES, but only a small number of files per sample.
# This avoids blowing the disk quota while still giving a CMS-style signal-vs-background study.
SELECTED_SAMPLES = OrderedDict(
    [
        # Signal
        ("HH_bbWW", 2),

        # Dominant ttbar backgrounds
        ("tt0123j_5f_ckm_LO_MLM_semiLeptonic", 2),
        ("tt0123j_5f_ckm_LO_MLM_leptonic", 1),
        ("tt0123j_5f_ckm_LO_MLM_hadronic", 1),

        # Drell-Yan and W/Z+jets
        ("DYJetsToLL_13TeV-madgraphMLM-pythia8", 1),
        ("WJetsToLNu_13TeV-madgraphMLM-pythia8", 1),
        ("WJetsToQQ_13TeV-madgraphMLM-pythia8", 1),
        ("ZJetsToQQ_13TeV-madgraphMLM-pythia8", 1),
        ("ZJetsTobb_13TeV-madgraphMLM-pythia8", 1),
        ("ZJetsTocc_13TeV-madgraphMLM-pythia8", 1),
        ("ZJetsTovv_13TeV-madgraphMLM-pythia8", 1),

        # Diboson
        ("WW_hadronic", 1),
        ("WW_leptonic", 1),
        ("WW_semileptonic", 1),
        ("WZ_hadronic", 1),
        ("WZ_leptonic", 1),
        ("WZ_semileptonic", 1),
        ("ZZ_hadronic", 1),
        ("ZZ_leptonic", 1),
        ("ZZ_semileptonic", 1),

        # Triboson
        ("VVV_incl", 1),

        # Top + boson / rare top
        ("ttH_incl", 1),
        ("ttW_incl", 1),
        ("ttZ_incl", 1),
        ("tttt_incl", 1),

        # Single-Higgs backgrounds, limited first pass
        ("ggHWW", 1),
        ("ggHZZ", 1),
        ("ggHbb", 1),
        ("ggHtautau", 1),
        ("VBFHWW", 1),
        ("VBFHZZ", 1),
        ("VBFHbb", 1),
        ("VBFHtautau", 1),
        ("VH_incl", 1),

        # QCD / fake-like stress background, limited
        ("QCD_HT50tobb", 1),
        ("QCD_HT50toInf", 1),
    ]
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--local-dir", default="outputs/collide_selected_backgrounds")
    parser.add_argument("--revision", default=None)
    args = parser.parse_args()

    local_dir = Path(args.local_dir)
    local_dir.mkdir(parents=True, exist_ok=True)

    api = HfApi()
    files = api.list_repo_files(repo_id=REPO_ID, repo_type=REPO_TYPE, revision=args.revision)
    parquets = sorted([f for f in files if f.endswith(".parquet")])

    by_sample = defaultdict(list)
    for f in parquets:
        by_sample[f.split("/")[0]].append(f)

    selected_files = []
    missing_samples = []

    for sample, n_files in SELECTED_SAMPLES.items():
        available = sorted(by_sample.get(sample, []))
        if not available:
            missing_samples.append(sample)
            continue
        selected_files.extend(available[:n_files])

    print("Selected samples:")
    for sample, n_files in SELECTED_SAMPLES.items():
        available = sorted(by_sample.get(sample, []))
        print(f"  {sample:45s} requested={n_files:2d} available={len(available):3d}")

    if missing_samples:
        print("\nMissing selected samples:")
        for s in missing_samples:
            print(" ", s)

    print(f"\nTotal selected parquet files: {len(selected_files)}")
    print(f"Local output directory: {local_dir}")

    for i, filename in enumerate(selected_files, start=1):
        out_path = local_dir / filename
        if out_path.exists() and out_path.stat().st_size > 0:
            print(f"[{i}/{len(selected_files)}] exists, skipping: {filename}")
            continue

        print(f"[{i}/{len(selected_files)}] downloading: {filename}")
        hf_hub_download(
            repo_id=REPO_ID,
            repo_type=REPO_TYPE,
            filename=filename,
            revision=args.revision,
            local_dir=str(local_dir),
        )

    print("\nDone. Downloaded selected COLLIDE-1M subset.")


if __name__ == "__main__":
    main()