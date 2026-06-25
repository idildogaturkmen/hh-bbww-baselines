'''
This script downloads a small number of parquet files for selected background samples from the collide-1m dataset.
The selected samples are chosen to represent a variety of background processes, while avoiding excessive disk usage.
The downloaded files are stored in a local directory, which can be specified by the user.
The script uses the Hugging Face Hub API to list and download the parquet files.
Usage:
    python scripts/download_collide_extended_backgrounds.py --local-dir <output_directory> [--

The remaining backgrounds: 
ggHcc, ggHgammagamma, ggHgluglu
VBFHcc, VBFHgammagamma, VBFHgluglu
gamma, gamma_V
minbias
upsilon_to_leptons
HH_4b, HH_bbZZ, HH_bbgammagamma, HH_bbtautau
'''
from __future__ import annotations

import argparse
import shutil
from collections import defaultdict, OrderedDict
from pathlib import Path

from huggingface_hub import HfApi, hf_hub_download

REPO_ID = "fastmachinelearning/collide-1m"
REPO_TYPE = "dataset"

# Balanced plan:
# - keep HH_bbWW full
# - make other HH full
# - make diboson and rare top mostly/full
# - increase dominant backgrounds, QCD, W/Z+jets, single Higgs, gamma/minbias modestly
BALANCED_TARGETS = OrderedDict(
    [
        # Signal
        ("HH_bbWW", 2),

        # Other HH modes, all available files
        ("HH_4b", 2),
        ("HH_bbZZ", 2),
        ("HH_bbgammagamma", 2),
        ("HH_bbtautau", 2),

        # ttbar: more files, but not all yet
        ("tt0123j_5f_ckm_LO_MLM_hadronic", 3),
        ("tt0123j_5f_ckm_LO_MLM_leptonic", 3),
        ("tt0123j_5f_ckm_LO_MLM_semiLeptonic", 5),

        # top-associated / rare top: all or nearly all available
        ("ttH_incl", 10),
        ("ttW_incl", 5),
        ("ttZ_incl", 5),
        ("tttt_incl", 2),

        # DY / Z+jets
        ("DYJetsToLL_13TeV-madgraphMLM-pythia8", 3),
        ("ZJetsToQQ_13TeV-madgraphMLM-pythia8", 3),
        ("ZJetsTobb_13TeV-madgraphMLM-pythia8", 3),
        ("ZJetsTocc_13TeV-madgraphMLM-pythia8", 3),
        ("ZJetsTovv_13TeV-madgraphMLM-pythia8", 3),

        # W+jets
        ("WJetsToLNu_13TeV-madgraphMLM-pythia8", 3),
        ("WJetsToQQ_13TeV-madgraphMLM-pythia8", 3),

        # Diboson: all available files, since each has only 3
        ("WW_hadronic", 3),
        ("WW_leptonic", 3),
        ("WW_semileptonic", 3),
        ("WZ_hadronic", 3),
        ("WZ_leptonic", 3),
        ("WZ_semileptonic", 3),
        ("ZZ_hadronic", 3),
        ("ZZ_leptonic", 3),
        ("ZZ_semileptonic", 3),

        # Triboson: all available
        ("VVV_incl", 2),

        # Single Higgs: increase to 3 files each for now
        ("ggHWW", 3),
        ("ggHZZ", 3),
        ("ggHbb", 3),
        ("ggHcc", 3),
        ("ggHgammagamma", 3),
        ("ggHgluglu", 3),
        ("ggHtautau", 3),

        ("VBFHWW", 3),
        ("VBFHZZ", 3),
        ("VBFHbb", 3),
        ("VBFHcc", 3),
        ("VBFHgammagamma", 3),
        ("VBFHgluglu", 3),
        ("VBFHtautau", 3),

        ("VH_incl", 3),

        # QCD / photon / minimum-bias / resonance-like samples
        ("QCD_HT50toInf", 3),
        ("QCD_HT50tobb", 3),
        ("gamma", 3),
        ("gamma_V", 3),
        ("minbias", 3),
        ("upsilon_to_leptons", 3),
    ]
)


def free_gb(path: Path) -> float:
    usage = shutil.disk_usage(path)
    return usage.free / 1e9


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--local-dir", default="outputs/collide_selected_backgrounds")
    parser.add_argument("--min-free-gb", type=float, default=50.0)
    parser.add_argument(
        "--plan",
        choices=["balanced", "all"],
        default="balanced",
        help="balanced = safe expanded subset; all = try all files but stop before disk fills",
    )
    args = parser.parse_args()

    local_dir = Path(args.local_dir)
    local_dir.mkdir(parents=True, exist_ok=True)

    api = HfApi()
    files = api.list_repo_files(repo_id=REPO_ID, repo_type=REPO_TYPE)
    parquets = sorted([f for f in files if f.endswith(".parquet")])

    by_sample = defaultdict(list)
    for f in parquets:
        sample = f.split("/")[0]
        by_sample[sample].append(f)

    if args.plan == "balanced":
        target_counts = BALANCED_TARGETS
    else:
        # Try every parquet file in every sample. This will probably NOT finish on 300 GB,
        # but it can be resumed later because existing files are skipped.
        target_counts = OrderedDict(
            (sample, len(sorted(paths)))
            for sample, paths in sorted(by_sample.items())
            if sample != ".cache"
        )

    selected = []
    print(f"Plan: {args.plan}")
    print(f"Minimum free space to preserve: {args.min_free_gb:.1f} GB\n")

    for sample, requested in target_counts.items():
        available = sorted(by_sample.get(sample, []))
        n_take = min(requested, len(available))
        chosen = available[:n_take]
        selected.extend(chosen)
        print(
            f"{sample:45s} requested={requested:3d} "
            f"available={len(available):3d} target={len(chosen):3d}"
        )

    print(f"\nTotal target parquet files: {len(selected)}")
    print(f"Current free space: {free_gb(local_dir):.1f} GB\n")

    downloaded = 0
    skipped = 0
    stopped = False

    for i, filename in enumerate(selected, start=1):
        out_path = local_dir / filename

        if out_path.exists() and out_path.stat().st_size > 0:
            print(f"[{i}/{len(selected)}] exists, skipping: {filename}")
            skipped += 1
            continue

        current_free = free_gb(local_dir)
        if current_free < args.min_free_gb:
            print(
                f"\nSTOPPING: only {current_free:.1f} GB free, "
                f"below min-free-gb={args.min_free_gb:.1f}."
            )
            stopped = True
            break

        print(f"[{i}/{len(selected)}] downloading: {filename}")
        hf_hub_download(
            repo_id=REPO_ID,
            repo_type=REPO_TYPE,
            filename=filename,
            local_dir=str(local_dir),
        )
        downloaded += 1

    print("\nDone.")
    print(f"Skipped existing files: {skipped}")
    print(f"Downloaded new files: {downloaded}")
    print(f"Stopped early because of disk guard: {stopped}")
    print(f"Final free space: {free_gb(local_dir):.1f} GB")


if __name__ == "__main__":
    main()