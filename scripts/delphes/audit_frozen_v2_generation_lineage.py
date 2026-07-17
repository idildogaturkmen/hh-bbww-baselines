#!/usr/bin/env python3

from __future__ import annotations

import csv
import hashlib
import json
import os
import re
from pathlib import Path


REPO = Path(os.environ["HH4B_REPO"]).resolve()
STORE = Path(os.environ["HH4B_STORE"]).resolve()
OUT = Path(os.environ["LINEAGE_OUT"]).resolve()

MG5_ROOT = STORE / "mg5"
PHASE1_MANIFEST = (
    REPO
    / "outputs/agent_runs/"
    "frozen_v2_phase1_reprocess_20260717/"
    "phase1_manifest.tsv"
)

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
    "etaj",
    "etab",
    "drjj",
    "drbb",
    "drbj",
    "mmbb",
    "ihtmin",
    "ihtmax",
    "htjmin",
    "htjmax",
}

SEARCH_TERMS = [
    "qcd_bbbb_ak4ak8_extra50k",
    "qcd_bbbb_iht400to600",
    "ttbar_extra50k_ak4ak8",
    "zbbbb_ak4ak8_extra50k",
    "phase1_inputs_20260717",
    "ihtmin",
    "ihtmax",
]

TEXT_SUFFIXES = {
    ".txt",
    ".log",
    ".dat",
    ".json",
    ".csv",
    ".tsv",
    ".yaml",
    ".yml",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()

    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)

    return digest.hexdigest()


def process_definition(text: str) -> list[str]:
    keep = []

    for raw in text.splitlines():
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


def run_parameters(text: str) -> dict[str, str]:
    found: dict[str, str] = {}

    for raw in text.splitlines():
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

    if "zbbbb" in lower:
        return "zbbbb"

    if "zbb" in lower:
        return "zbb"

    if "ttbb" in lower or "tt_b" in lower:
        return "ttbb"

    if "ttbar" in lower or "tt_" in lower:
        return "ttbar"

    if "qcd" in lower and "bbbb" in lower:
        return "qcd_bbbb"

    if "vbf" in lower and "hh" in lower:
        return "vbf_hh4b"

    if ("ggf" in lower or "gghh" in lower) and "hh" in lower:
        return "ggf_hh4b"

    if "zh4b" in lower:
        return "zh4b"

    if "zz4b" in lower:
        return "zz4b"

    return "unresolved"


def numeric(value: str | None) -> float | None:
    if value is None:
        return None

    match = re.search(
        r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eEdD][-+]?\d+)?",
        value,
    )

    if not match:
        return None

    return float(
        match.group(0)
        .replace("D", "E")
        .replace("d", "e")
    )


def signature(parameters: dict[str, str]) -> str:
    ordered = [
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
        "mmbb",
    ]

    return "; ".join(
        f"{key}={parameters[key]}"
        for key in ordered
        if key in parameters
    )


def write_csv(
    path: Path,
    rows: list[dict[str, object]],
    fields: list[str],
) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


OUT.mkdir(parents=True, exist_ok=True)

if not MG5_ROOT.is_dir():
    raise SystemExit(f"ERROR: missing MG5 directory: {MG5_ROOT}")

if not PHASE1_MANIFEST.is_file():
    raise SystemExit(
        f"ERROR: missing Phase 1 manifest: {PHASE1_MANIFEST}"
    )

banner_rows: list[dict[str, object]] = []

for banner in sorted(MG5_ROOT.rglob("*_banner.txt")):
    text = banner.read_text(errors="replace")
    definitions = process_definition(text)
    parameters = run_parameters(text)

    combined = " ".join(
        [
            str(banner),
            *definitions,
        ]
    )

    banner_rows.append(
        {
            "banner_path": str(banner),
            "relative_banner_path": str(
                banner.relative_to(MG5_ROOT)
            ),
            "process_directory": str(
                banner.parents[2].relative_to(MG5_ROOT)
            )
            if len(banner.parents) >= 3
            else "",
            "run_directory": banner.parent.name,
            "suggested_family": infer_family(combined),
            "process_definition": " | ".join(definitions),
            "nevents": parameters.get("nevents", ""),
            "iseed": parameters.get("iseed", ""),
            "lhaid": parameters.get("lhaid", ""),
            "ickkw": parameters.get("ickkw", ""),
            "ihtmin": parameters.get("ihtmin", ""),
            "ihtmax": parameters.get("ihtmax", ""),
            "htjmin": parameters.get("htjmin", ""),
            "htjmax": parameters.get("htjmax", ""),
            "ptj": parameters.get("ptj", ""),
            "ptb": parameters.get("ptb", ""),
            "etaj": parameters.get("etaj", ""),
            "etab": parameters.get("etab", ""),
            "drjj": parameters.get("drjj", ""),
            "drbb": parameters.get("drbb", ""),
            "drbj": parameters.get("drbj", ""),
            "phase_space_signature": signature(parameters),
            "banner_sha256": sha256(banner),
            "size_bytes": banner.stat().st_size,
            "mtime_ns": banner.stat().st_mtime_ns,
        }
    )

banner_fields = (
    list(banner_rows[0].keys())
    if banner_rows
    else ["banner_path"]
)

write_csv(
    OUT / "mg5_run_banner_inventory.csv",
    banner_rows,
    banner_fields,
)

with PHASE1_MANIFEST.open() as handle:
    manifest_rows = list(
        csv.DictReader(handle, delimiter="\t")
    )

family_map = {
    "qcd_bbbb_general": "qcd_bbbb",
    "qcd_bbbb_iht400to600": "qcd_bbbb",
    "ttbar": "ttbar",
    "zbbbb": "zbbbb",
}

candidate_rows: list[dict[str, object]] = []

for manifest_row in manifest_rows:
    process = manifest_row["process"]
    wanted_family = family_map.get(process, "unresolved")

    candidates = [
        row
        for row in banner_rows
        if row["suggested_family"] == wanted_family
    ]

    if not candidates:
        candidate_rows.append(
            {
                "phase1_process": process,
                "phase1_shard": manifest_row["shard"],
                "phase1_tag": manifest_row["tag"],
                "input_hepmc": manifest_row["input_hepmc"],
                "candidate_banner": "",
                "candidate_family": "",
                "ihtmin": "",
                "ihtmax": "",
                "phase_space_signature": "",
                "lineage_status": "NO_FAMILY_MATCHING_BANNER",
            }
        )
        continue

    for candidate in candidates:
        candidate_rows.append(
            {
                "phase1_process": process,
                "phase1_shard": manifest_row["shard"],
                "phase1_tag": manifest_row["tag"],
                "input_hepmc": manifest_row["input_hepmc"],
                "candidate_banner": candidate["banner_path"],
                "candidate_family": candidate[
                    "suggested_family"
                ],
                "ihtmin": candidate["ihtmin"],
                "ihtmax": candidate["ihtmax"],
                "phase_space_signature": candidate[
                    "phase_space_signature"
                ],
                "lineage_status": "FAMILY_MATCH_ONLY_NOT_PROVEN",
            }
        )

candidate_fields = (
    list(candidate_rows[0].keys())
    if candidate_rows
    else ["phase1_process"]
)

write_csv(
    OUT / "phase1_banner_candidates.csv",
    candidate_rows,
    candidate_fields,
)

evidence_rows: list[dict[str, object]] = []

search_roots = [
    STORE / "metadata",
    STORE / "logs",
    MG5_ROOT,
]

for root in search_roots:
    if not root.exists():
        continue

    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue

        if path.suffix.lower() not in TEXT_SUFFIXES:
            continue

        if path.stat().st_size > 20 * 1024 * 1024:
            continue

        path_text = str(path).lower()
        path_matches = [
            term
            for term in SEARCH_TERMS
            if term.lower() in path_text
        ]

        content_matches: list[str] = []

        try:
            text = path.read_text(errors="replace")
            lower_text = text.lower()

            content_matches = [
                term
                for term in SEARCH_TERMS
                if term.lower() in lower_text
            ]
        except Exception:
            text = ""

        if not path_matches and not content_matches:
            continue

        evidence_rows.append(
            {
                "path": str(path),
                "relative_to_store": (
                    str(path.relative_to(STORE))
                    if path.is_relative_to(STORE)
                    else ""
                ),
                "size_bytes": path.stat().st_size,
                "path_matches": "|".join(path_matches),
                "content_matches": "|".join(content_matches),
            }
        )

evidence_fields = (
    list(evidence_rows[0].keys())
    if evidence_rows
    else ["path"]
)

write_csv(
    OUT / "lineage_evidence_file_matches.csv",
    evidence_rows,
    evidence_fields,
)

qcd_banners = [
    row
    for row in banner_rows
    if row["suggested_family"] == "qcd_bbbb"
]

has_qcd_general = any(
    numeric(str(row["ihtmin"])) in {0.0, None}
    and numeric(str(row["ihtmax"])) in {-1.0, None}
    for row in qcd_banners
)

has_qcd_400_600 = any(
    numeric(str(row["ihtmin"])) == 400.0
    and numeric(str(row["ihtmax"])) == 600.0
    for row in qcd_banners
)

report = {
    "status": "lineage_discovery_complete",
    "n_run_banners": len(banner_rows),
    "n_phase1_manifest_rows": len(manifest_rows),
    "n_banner_candidate_rows": len(candidate_rows),
    "n_evidence_file_matches": len(evidence_rows),
    "has_qcd_general_banner_evidence": has_qcd_general,
    "has_qcd_iht400to600_banner_evidence": has_qcd_400_600,
    "has_ttbar_banner_evidence": any(
        row["suggested_family"] == "ttbar"
        for row in banner_rows
    ),
    "has_zbbbb_banner_evidence": any(
        row["suggested_family"] == "zbbbb"
        for row in banner_rows
    ),
    "warning": (
        "Banner-family matches are candidates only. "
        "Exact HepMC-to-banner lineage is not proven by family matching."
    ),
}

(OUT / "lineage_discovery_report.json").write_text(
    json.dumps(report, indent=2, sort_keys=True) + "\n"
)

print("MG5_RUN_BANNERS:", len(banner_rows))
print("PHASE1_MANIFEST_ROWS:", len(manifest_rows))
print("EVIDENCE_FILE_MATCHES:", len(evidence_rows))

print()
print("=== RUN BANNERS ===")

for row in banner_rows:
    print(
        f"{str(row['suggested_family']):16s} "
        f"{str(row['relative_banner_path']):65s} "
        f"{row['phase_space_signature']}"
    )

print()
print("=== LINEAGE EVIDENCE STATUS ===")
print(
    "QCD general banner evidence:",
    has_qcd_general,
)
print(
    "QCD IHT 400-600 banner evidence:",
    has_qcd_400_600,
)
print(
    "ttbar banner evidence:",
    report["has_ttbar_banner_evidence"],
)
print(
    "Zbbbb banner evidence:",
    report["has_zbbbb_banner_evidence"],
)

print()
print("GATE1B_LINEAGE_DISCOVERY_COMPLETE")
print("Wrote:", OUT / "lineage_discovery_report.json")
print("Wrote:", OUT / "mg5_run_banner_inventory.csv")
print("Wrote:", OUT / "phase1_banner_candidates.csv")
print("Wrote:", OUT / "lineage_evidence_file_matches.csv")
