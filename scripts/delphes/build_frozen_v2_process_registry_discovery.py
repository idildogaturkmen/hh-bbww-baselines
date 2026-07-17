#!/usr/bin/env python3

from __future__ import annotations

import csv
import hashlib
import json
import os
import re
from collections import Counter
from pathlib import Path


REPO = Path(os.environ["HH4B_REPO"]).resolve()
STORE = Path(os.environ["HH4B_STORE"]).resolve()
OUT = Path(os.environ["REGISTRY"]).resolve()

MG5_ROOT = STORE / "mg5"
HEPMC_ROOT = STORE / "hepmc"

PHASE1_MANIFEST = (
    REPO
    / "outputs/agent_runs/"
    "frozen_v2_phase1_reprocess_20260717/"
    "phase1_manifest.tsv"
)

FROZEN_CARD = (
    REPO
    / "cards/delphes/"
    "delphes_card_CMS_lpc_ak4ak8_run2_frozen_v2.tcl"
)

CARD_NAMES = [
    "proc_card_mg5.dat",
    "run_card.dat",
    "param_card.dat",
    "pythia8_card.dat",
    "madspin_card.dat",
]

RUN_KEYS = {
    "nevents",
    "iseed",
    "ebeam1",
    "ebeam2",
    "pdlabel",
    "lhaid",
    "ickkw",
    "xqcut",
    "maxjetflavor",
    "ptj",
    "ptjmax",
    "ptb",
    "ptbmax",
    "pta",
    "ptamax",
    "ptl",
    "ptlmax",
    "etaj",
    "etab",
    "etal",
    "drjj",
    "drbb",
    "drbj",
    "mmjj",
    "mmbb",
    "ihtmin",
    "ihtmax",
    "htjmin",
    "htjmax",
    "cut_decays",
    "auto_ptj_mjj",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()

    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)

    return digest.hexdigest()


def process_definition(path: Path) -> list[str]:
    if not path.is_file():
        return []

    keep = []

    for raw in path.read_text(errors="replace").splitlines():
        line = raw.strip()
        lower = line.lower()

        if (
            lower.startswith("import model")
            or lower.startswith("define ")
            or lower.startswith("generate ")
            or lower.startswith("add process ")
            or lower.startswith("output ")
        ):
            keep.append(line)

    return keep


def run_parameters(path: Path) -> dict[str, str]:
    if not path.is_file():
        return {}

    found: dict[str, str] = {}

    for raw in path.read_text(errors="replace").splitlines():
        line = re.split(r"[#!]", raw, maxsplit=1)[0].strip()

        if "=" not in line:
            continue

        left, right = [
            piece.strip()
            for piece in line.split("=", maxsplit=1)
        ]

        left_key = left.lower()
        right_key = right.lower()

        if right_key in RUN_KEYS:
            found[right_key] = left
        elif left_key in RUN_KEYS:
            found[left_key] = right

    return found


def infer_family(text: str) -> str:
    lower = text.lower()

    if "vbf" in lower and ("hh4b" in lower or "hh_" in lower):
        return "vbf_hh4b"

    if (
        ("ggf" in lower or "gghh" in lower)
        and ("hh4b" in lower or "hh_" in lower)
    ):
        return "ggf_hh4b"

    if "ttbb" in lower or "tt_b" in lower:
        return "ttbb"

    if "ttbar" in lower or "tt_" in lower:
        return "ttbar"

    if "zbbbb" in lower:
        return "zbbbb"

    if "qcd" in lower and "bbbb" in lower:
        return "qcd_bbbb"

    if "zh4b" in lower or "zh_4b" in lower:
        return "zh4b"

    if "zz4b" in lower or "zz_4b" in lower:
        return "zz4b"

    if "qcd" in lower or "hardqcd" in lower:
        return "inclusive_qcd"

    return "unresolved"


def phase_space_signature(parameters: dict[str, str]) -> str:
    keys = [
        "ihtmin",
        "ihtmax",
        "htjmin",
        "htjmax",
        "ptj",
        "ptjmax",
        "ptb",
        "ptbmax",
        "etaj",
        "etab",
        "drjj",
        "drbb",
        "drbj",
    ]

    parts = [
        f"{key}={parameters[key]}"
        for key in keys
        if key in parameters
    ]

    return "; ".join(parts)


def write_csv(
    path: Path,
    rows: list[dict[str, object]],
    fieldnames: list[str],
) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


OUT.mkdir(parents=True, exist_ok=True)

if not MG5_ROOT.is_dir():
    raise SystemExit(f"ERROR: missing MG5 root: {MG5_ROOT}")

if not HEPMC_ROOT.is_dir():
    raise SystemExit(f"ERROR: missing HepMC root: {HEPMC_ROOT}")

if not FROZEN_CARD.is_file():
    raise SystemExit(f"ERROR: missing frozen card: {FROZEN_CARD}")

frozen_card_sha = sha256(FROZEN_CARD)

expected_frozen_sha = (
    "1b2041245162de8360defdc502404e069"
    "6496c50773d4aa7f043798651a9517c"
)

if frozen_card_sha != expected_frozen_sha:
    raise SystemExit(
        "ERROR: frozen detector-card checksum changed:\n"
        f"expected={expected_frozen_sha}\n"
        f"actual={frozen_card_sha}"
    )

process_rows: list[dict[str, object]] = []

for proc_card in sorted(MG5_ROOT.rglob("Cards/proc_card_mg5.dat")):
    process_dir = proc_card.parent.parent
    relative_dir = process_dir.relative_to(MG5_ROOT)

    definition = process_definition(proc_card)
    parameters = run_parameters(
        process_dir / "Cards/run_card.dat"
    )

    combined_text = " ".join(
        [
            str(relative_dir),
            *definition,
        ]
    )

    card_hashes = {}

    for card_name in CARD_NAMES:
        card_path = process_dir / "Cards" / card_name

        if card_path.is_file():
            card_hashes[card_name] = sha256(card_path)

    process_rows.append(
        {
            "process_directory": str(process_dir),
            "relative_process_directory": str(relative_dir),
            "suggested_family": infer_family(combined_text),
            "process_definition": " | ".join(definition),
            "nevents": parameters.get("nevents", ""),
            "iseed": parameters.get("iseed", ""),
            "lhaid": parameters.get("lhaid", ""),
            "ickkw": parameters.get("ickkw", ""),
            "ihtmin": parameters.get("ihtmin", ""),
            "ihtmax": parameters.get("ihtmax", ""),
            "htjmin": parameters.get("htjmin", ""),
            "htjmax": parameters.get("htjmax", ""),
            "ptj": parameters.get("ptj", ""),
            "ptjmax": parameters.get("ptjmax", ""),
            "ptb": parameters.get("ptb", ""),
            "ptbmax": parameters.get("ptbmax", ""),
            "etaj": parameters.get("etaj", ""),
            "etab": parameters.get("etab", ""),
            "drjj": parameters.get("drjj", ""),
            "drbb": parameters.get("drbb", ""),
            "drbj": parameters.get("drbj", ""),
            "phase_space_signature": phase_space_signature(parameters),
            "run_parameters_json": json.dumps(
                parameters,
                sort_keys=True,
            ),
            "card_hashes_json": json.dumps(
                card_hashes,
                sort_keys=True,
            ),
        }
    )

process_fields = list(process_rows[0].keys()) if process_rows else [
    "process_directory",
    "relative_process_directory",
    "suggested_family",
]

write_csv(
    OUT / "process_cards_discovery.csv",
    process_rows,
    process_fields,
)

phase1_mapping: dict[str, dict[str, str]] = {}

if PHASE1_MANIFEST.is_file():
    with PHASE1_MANIFEST.open() as handle:
        reader = csv.DictReader(handle, delimiter="\t")

        for row in reader:
            input_path = str(Path(row["input_hepmc"]).resolve())
            phase1_mapping[input_path] = row

hepmc_rows: list[dict[str, object]] = []

for path in sorted(HEPMC_ROOT.rglob("*.hepmc")):
    resolved = str(path.resolve())
    phase1 = phase1_mapping.get(resolved)

    hepmc_rows.append(
        {
            "path": str(path),
            "relative_path": str(path.relative_to(HEPMC_ROOT)),
            "basename": path.name,
            "size_bytes": path.stat().st_size,
            "size_gib": path.stat().st_size / 1024**3,
            "suggested_family": infer_family(path.name),
            "used_in_phase1": bool(phase1),
            "phase1_process": phase1["process"] if phase1 else "",
            "phase1_shard": phase1["shard"] if phase1 else "",
            "phase1_split": phase1["split"] if phase1 else "",
            "phase1_tag": phase1["tag"] if phase1 else "",
        }
    )

hepmc_fields = list(hepmc_rows[0].keys()) if hepmc_rows else [
    "path",
    "relative_path",
    "basename",
]

write_csv(
    OUT / "retained_hepmc_inventory.csv",
    hepmc_rows,
    hepmc_fields,
)

registry_rows = []

for row in process_rows:
    relative = str(row["relative_process_directory"])
    family = str(row["suggested_family"])

    registry_rows.append(
        {
            "sample_id": relative.replace("/", "__"),
            "process_directory": relative,
            "physics_family": family,
            "intended_role": "UNRESOLVED",
            "physical_normalization_allowed": "NO_UNTIL_AUDITED",
            "overlap_group": "UNRESOLVED",
            "phase_space_axis": "UNRESOLVED",
            "phase_space_min": "",
            "phase_space_max": "",
            "exclusive_lower_edge": "",
            "exclusive_upper_edge": "",
            "cross_section_pb": "",
            "cross_section_source": "UNRESOLVED",
            "event_weight_formula": "UNRESOLVED",
            "detector_card_sha256": expected_frozen_sha,
            "truth_definition_status": "UNRESOLVED",
            "production_status": "DISCOVERED_ONLY",
            "decision_notes": "",
        }
    )

registry_fields = list(registry_rows[0].keys()) if registry_rows else [
    "sample_id",
    "process_directory",
    "physics_family",
]

write_csv(
    OUT / "canonical_sample_registry_template.csv",
    registry_rows,
    registry_fields,
)

family_counts = Counter(
    str(row["suggested_family"])
    for row in process_rows
)

hepmc_family_counts = Counter(
    str(row["suggested_family"])
    for row in hepmc_rows
)

report = {
    "status": "discovery_complete",
    "frozen_detector_card": str(FROZEN_CARD),
    "frozen_detector_card_sha256": frozen_card_sha,
    "n_process_directories": len(process_rows),
    "process_family_counts": dict(sorted(family_counts.items())),
    "n_retained_hepmc_files": len(hepmc_rows),
    "retained_hepmc_family_counts": dict(
        sorted(hepmc_family_counts.items())
    ),
    "phase1_hepmc_files_mapped": sum(
        bool(row["used_in_phase1"])
        for row in hepmc_rows
    ),
    "phase1_manifest": str(PHASE1_MANIFEST),
}

(OUT / "discovery_report.json").write_text(
    json.dumps(report, indent=2, sort_keys=True) + "\n"
)

print("PROCESS_CARD_ROWS:", len(process_rows))
print("RETAINED_HEPMC_FILES:", len(hepmc_rows))
print(
    "PHASE1_HEPMC_MAPPED:",
    report["phase1_hepmc_files_mapped"],
)
print("FROZEN_CARD_SHA256:", frozen_card_sha)

print()
print("=== PROCESS FAMILY COUNTS ===")

for family, count in sorted(family_counts.items()):
    print(f"{family:24s} {count}")

print()
print("=== RETAINED HEPMC FAMILY COUNTS ===")

for family, count in sorted(hepmc_family_counts.items()):
    print(f"{family:24s} {count}")

print()
print("=== PROCESS DEFINITIONS AND PHASE-SPACE CUTS ===")

for row in process_rows:
    print(
        f"{str(row['suggested_family']):20s} "
        f"{str(row['relative_process_directory']):55s} "
        f"{row['phase_space_signature']}"
    )

print()
print("GATE1_PROCESS_DISCOVERY_COMPLETE")
print("Wrote:", OUT / "discovery_report.json")
print("Wrote:", OUT / "process_cards_discovery.csv")
print("Wrote:", OUT / "retained_hepmc_inventory.csv")
print("Wrote:", OUT / "canonical_sample_registry_template.csv")
