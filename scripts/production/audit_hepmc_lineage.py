#!/usr/bin/env python3

from pathlib import Path
import gzip
import hashlib
import json
import os

import pandas as pd

REPO = Path(os.environ["HH4B_REPO"])
STORE = Path(os.environ["HH4B_STORE"])

HEPMC_DIR = STORE / "hepmc"
OUTDIR = REPO / "outputs/audits/hepmc_lineage_run2_2026_07_16"
OUTDIR.mkdir(parents=True, exist_ok=True)


def classify_process(name: str) -> str:
    low = name.lower()

    rules = [
        ("ggF_HH_4b", ["ggf", "hh4b"]),
        ("VBF_HH_4b", ["vbf", "hh4b"]),
        ("QCD_bbbb", ["qcd", "bbbb"]),
        ("Zbbbb", ["zbbbb"]),
        ("ttbb", ["ttbb"]),
        ("ttbar", ["ttbar"]),
        ("ZZ_4b", ["zz4b"]),
        ("ZH_4b", ["zh4b"]),
        ("Zbb", ["zbb"]),
    ]

    for process, tokens in rules:
        if all(token in low for token in tokens):
            return process

    return "unclassified"


def analysis_role(process: str) -> str:
    roles = {
        "ggF_HH_4b": "signal",
        "VBF_HH_4b": "signal",
        "ttbar": "physics_background_pilot",
        "QCD_bbbb": "ml_enrichment",
        "Zbbbb": "ml_enrichment",
        "ttbb": "ml_enrichment",
        "ZZ_4b": "validation",
        "ZH_4b": "validation",
        "Zbb": "development",
    }
    return roles.get(process, "manual_review")


def open_binary(path: Path):
    if path.suffix == ".gz":
        return gzip.open(path, "rb")
    return path.open("rb")


def inspect_hepmc(path: Path):
    digest = hashlib.sha256()
    n_events = 0
    first_lines = []

    with open_binary(path) as handle:
        for line_number, line in enumerate(handle):
            digest.update(line)

            if line_number < 8:
                first_lines.append(
                    line.decode("utf-8", errors="replace").rstrip()
                )

            # HepMC2 and HepMC3 ASCII event records begin with "E ".
            if line.startswith(b"E "):
                n_events += 1

    return {
        "sha256": digest.hexdigest(),
        "n_events": n_events,
        "header_preview": " | ".join(first_lines),
    }


paths = sorted(
    path
    for path in HEPMC_DIR.rglob("*")
    if path.is_file()
    and (
        path.name.endswith(".hepmc")
        or path.name.endswith(".hepmc.gz")
    )
)

rows = []

for index, path in enumerate(paths, start=1):
    print(f"[{index}/{len(paths)}] {path}")

    info = inspect_hepmc(path)
    process = classify_process(path.name)
    role = analysis_role(process)

    rows.append({
        "process": process,
        "analysis_role": role,
        "filename": path.name,
        "path": str(path),
        "size_GB": path.stat().st_size / 1.0e9,
        "n_events": info["n_events"],
        "sha256": info["sha256"],
        "sha256_short": info["sha256"][:16],
        "header_preview": info["header_preview"],
    })

inventory = pd.DataFrame(rows)

if inventory.empty:
    raise SystemExit(f"No HepMC files found under {HEPMC_DIR}")

inventory["exact_duplicate_group_size"] = (
    inventory.groupby("sha256")["sha256"].transform("size")
)

inventory["exact_duplicate_rank"] = (
    inventory.groupby("sha256")["path"].rank(
        method="first",
        ascending=True,
    ).astype(int)
)

inventory["is_exact_duplicate"] = (
    inventory["exact_duplicate_rank"] > 1
)

inventory["canonical_file"] = (
    inventory["exact_duplicate_rank"] == 1
)

inventory["production_action"] = "manual_review"

known_final_roles = {
    "signal",
    "physics_background_pilot",
    "ml_enrichment",
    "validation",
}

inventory.loc[
    inventory["canonical_file"]
    & inventory["analysis_role"].isin(known_final_roles)
    & (inventory["n_events"] >= 10000),
    "production_action",
] = "rerun_with_frozen_v2"

inventory.loc[
    inventory["canonical_file"]
    & (inventory["n_events"] < 10000),
    "production_action",
] = "development_only"

inventory.loc[
    inventory["is_exact_duplicate"],
    "production_action",
] = "exclude_exact_duplicate"

def assign_split(row):
    if row["production_action"] != "rerun_with_frozen_v2":
        return "not_assigned"

    token = (
        f"run2_13tev/{row['process']}/{row['sha256']}"
    ).encode()

    bucket = int(hashlib.sha256(token).hexdigest(), 16) % 10

    if bucket <= 5:
        return "train"
    if bucket <= 7:
        return "validation"
    return "test"

inventory["immutable_split"] = inventory.apply(
    assign_split,
    axis=1,
)

inventory.to_csv(
    OUTDIR / "hepmc_lineage_inventory.csv",
    index=False,
)
(
    OUTDIR / "hepmc_lineage_inventory.md"
).write_text(inventory.to_markdown(index=False) + "\n")

canonical = inventory[
    inventory["production_action"] == "rerun_with_frozen_v2"
].copy()

summary = (
    canonical.groupby(
        ["process", "analysis_role"],
        as_index=False,
    )
    .agg(
        n_unique_hepmc_files=("path", "count"),
        unique_generated_events=("n_events", "sum"),
        total_size_GB=("size_GB", "sum"),
        train_events=(
            "n_events",
            lambda values: 0,
        ),
    )
)

# Add split totals separately and robustly.
split_counts = (
    canonical.groupby(
        ["process", "immutable_split"],
        as_index=False,
    )["n_events"]
    .sum()
    .pivot(
        index="process",
        columns="immutable_split",
        values="n_events",
    )
    .fillna(0)
    .reset_index()
)

summary = summary.drop(columns=["train_events"]).merge(
    split_counts,
    on="process",
    how="left",
)

for split in ["train", "validation", "test"]:
    if split not in summary:
        summary[split] = 0
    summary = summary.rename(
        columns={split: f"{split}_events"}
    )

summary.to_csv(
    OUTDIR / "canonical_hepmc_summary.csv",
    index=False,
)
(
    OUTDIR / "canonical_hepmc_summary.md"
).write_text(summary.to_markdown(index=False) + "\n")

duplicates = inventory[
    inventory["exact_duplicate_group_size"] > 1
].copy()

duplicates.to_csv(
    OUTDIR / "exact_duplicate_hepmc_files.csv",
    index=False,
)
(
    OUTDIR / "exact_duplicate_hepmc_files.md"
).write_text(duplicates.to_markdown(index=False) + "\n")

rerun_manifest = canonical[
    [
        "process",
        "analysis_role",
        "path",
        "filename",
        "n_events",
        "size_GB",
        "sha256",
        "immutable_split",
    ]
].copy()

rerun_manifest["detector_card"] = (
    "cards/delphes/"
    "delphes_card_CMS_lpc_ak4ak8_run2_frozen_v2.tcl"
)
rerun_manifest["detector_card_sha256"] = (
    "1b2041245162de8360defdc502404e069"
    "6496c50773d4aa7f043798651a9517c"
)
rerun_manifest["campaign"] = "run2_13tev"

rerun_manifest.to_csv(
    OUTDIR / "canonical_frozen_v2_rerun_manifest.csv",
    index=False,
)
(
    OUTDIR / "canonical_frozen_v2_rerun_manifest.md"
).write_text(rerun_manifest.to_markdown(index=False) + "\n")

payload = {
    "n_hepmc_files_scanned": int(len(inventory)),
    "total_hepmc_events_before_deduplication": int(
        inventory["n_events"].sum()
    ),
    "n_exact_duplicate_files": int(
        inventory["is_exact_duplicate"].sum()
    ),
    "n_canonical_rerun_files": int(len(canonical)),
    "canonical_rerun_events": int(
        canonical["n_events"].sum()
    ),
    "canonical_rerun_size_GB": float(
        canonical["size_GB"].sum()
    ),
}

(OUTDIR / "summary.json").write_text(
    json.dumps(payload, indent=2) + "\n"
)

print("\n=== Canonical HepMC summary ===")
print(summary.to_string(index=False))

print("\n=== Exact duplicates ===")
if duplicates.empty:
    print("No byte-identical HepMC duplicates found.")
else:
    print(
        duplicates[
            [
                "process",
                "filename",
                "n_events",
                "sha256_short",
                "exact_duplicate_rank",
            ]
        ].to_string(index=False)
    )

print("\n=== Overall ===")
print(json.dumps(payload, indent=2))

print("\nWrote:", OUTDIR)
