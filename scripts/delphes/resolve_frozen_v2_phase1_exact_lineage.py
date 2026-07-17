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
OUT = Path(os.environ["EXACT_LINEAGE_OUT"]).resolve()

MG5_ROOT = STORE / "mg5"

PHASE1 = (
    REPO
    / "outputs/agent_runs/"
    "frozen_v2_phase1_reprocess_20260717"
)

MANIFEST = PHASE1 / "phase1_manifest.tsv"
AUDIT_METADATA = (
    PHASE1
    / "final_audit_20260717"
    / "metadata"
)

EXPECTED_COUNTS = {
    "qcd_bbbb_general": 5,
    "qcd_bbbb_iht400to600": 10,
    "ttbar": 5,
    "zbbbb": 5,
}

EXPECTED_PROCESS_DIRECTORIES = {
    "qcd_bbbb_general": "QCD_bbbb_presel_smoke",
    "qcd_bbbb_iht400to600": "QCD_bbbb_presel_smoke",
    "ttbar": "TTbar_smoke",
    "zbbbb": "Zbbbb_presel_smoke",
}

EXPECTED_IHT = {
    "qcd_bbbb_general": (0.0, -1.0),
    "qcd_bbbb_iht400to600": (400.0, 600.0),
    "zbbbb": (0.0, -1.0),
}

RUN_KEYS = {
    "nevents",
    "iseed",
    "lhaid",
    "ickkw",
    "xqcut",
    "cut_decays",
    "ptj",
    "ptb",
    "etaj",
    "etab",
    "drjj",
    "drbb",
    "drbj",
    "ihtmin",
    "ihtmax",
    "htjmin",
    "htjmax",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()

    with path.open("rb") as handle:
        for block in iter(
            lambda: handle.read(4 * 1024 * 1024),
            b"",
        ):
            digest.update(block)

    return digest.hexdigest()


def numeric(value: str | None) -> float | None:
    if value is None:
        return None

    match = re.search(
        r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)"
        r"(?:[eEdD][-+]?\d+)?",
        str(value),
    )

    if not match:
        return None

    return float(
        match.group(0)
        .replace("D", "E")
        .replace("d", "e")
    )


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


def process_definition(text: str) -> str:
    lines = []

    for raw in text.splitlines():
        line = raw.strip()
        lower = line.lower()

        if (
            lower.startswith("import model")
            or lower.startswith("define ")
            or lower.startswith("generate ")
            or lower.startswith("add process ")
        ):
            lines.append(line)

    return " | ".join(lines)


def integrated_weight_pb(text: str) -> float | None:
    patterns = [
        r"Integrated\s+weight\s*\(pb\)\s*:\s*"
        r"([-+0-9.eEdD]+)",
        r"Cross[- ]section\s*\(pb\)\s*:\s*"
        r"([-+0-9.eEdD]+)",
    ]

    for pattern in patterns:
        match = re.search(pattern, text, flags=re.IGNORECASE)

        if match:
            return numeric(match.group(1))

    return None


def same_number(
    actual: float | None,
    expected: float,
    tolerance: float = 1.0e-9,
) -> bool:
    if actual is None:
        return False

    return abs(actual - expected) <= tolerance


OUT.mkdir(parents=True, exist_ok=True)

if not MANIFEST.is_file():
    raise SystemExit(f"ERROR: missing manifest: {MANIFEST}")

if not AUDIT_METADATA.is_dir():
    raise SystemExit(
        f"ERROR: missing audit metadata: {AUDIT_METADATA}"
    )

with MANIFEST.open() as handle:
    manifest_rows = list(
        csv.DictReader(handle, delimiter="\t")
    )

if len(manifest_rows) != 25:
    raise SystemExit(
        f"ERROR: expected 25 manifest rows, "
        f"found {len(manifest_rows)}"
    )

records: list[dict[str, object]] = []
seeds: list[int] = []

for index, row in enumerate(manifest_rows, start=1):
    process = row["process"]
    shard = int(row["shard"])
    split = row["split"]
    tag = row["tag"]
    expected_events = int(row["n_generated_expected"])

    input_path = Path(row["input_hepmc"]).resolve()
    basename = input_path.name

    match = re.fullmatch(
        r"(.+)_shard(\d{3})_pythia8\.hepmc",
        basename,
    )

    if not match:
        raise AssertionError(
            f"Unrecognized HepMC basename: {basename}"
        )

    campaign_stem = match.group(1)
    basename_shard = int(match.group(2))

    assert basename_shard == shard, (
        f"Manifest/basename shard mismatch for {tag}"
    )

    run_name = (
        f"run_{campaign_stem}_{shard:03d}"
    )

    banner_matches = [
        path
        for path in MG5_ROOT.rglob(
            f"{run_name}_tag_1_banner.txt"
        )
        if path.parent.name == run_name
    ]

    assert len(banner_matches) == 1, (
        f"Expected exactly one banner for {run_name}; "
        f"found {len(banner_matches)}:\n"
        + "\n".join(str(path) for path in banner_matches)
    )

    banner_path = banner_matches[0].resolve()
    banner_text = banner_path.read_text(errors="replace")
    parameters = run_parameters(banner_text)

    expected_process_directory = (
        EXPECTED_PROCESS_DIRECTORIES[process]
    )

    assert expected_process_directory in banner_path.parts, (
        f"Unexpected MG5 process directory for {tag}: "
        f"{banner_path}"
    )

    banner_nevents = numeric(parameters.get("nevents"))

    assert banner_nevents is not None, (
        f"Missing nevents in banner for {tag}"
    )

    assert int(round(banner_nevents)) == expected_events, (
        f"Banner event-count mismatch for {tag}: "
        f"{banner_nevents} != {expected_events}"
    )

    seed_value = numeric(parameters.get("iseed"))
    seed_int = (
        int(round(seed_value))
        if seed_value is not None
        else None
    )

    if seed_int is not None:
        seeds.append(seed_int)

    if process in EXPECTED_IHT:
        expected_ihtmin, expected_ihtmax = (
            EXPECTED_IHT[process]
        )

        actual_ihtmin = numeric(
            parameters.get("ihtmin")
        )
        actual_ihtmax = numeric(
            parameters.get("ihtmax")
        )

        assert same_number(
            actual_ihtmin,
            expected_ihtmin,
        ), (
            f"ihtmin mismatch for {tag}: "
            f"{actual_ihtmin} != {expected_ihtmin}"
        )

        assert same_number(
            actual_ihtmax,
            expected_ihtmax,
        ), (
            f"ihtmax mismatch for {tag}: "
            f"{actual_ihtmax} != {expected_ihtmax}"
        )
    else:
        actual_ihtmin = numeric(
            parameters.get("ihtmin")
        )
        actual_ihtmax = numeric(
            parameters.get("ihtmax")
        )

    metadata_path = (
        AUDIT_METADATA / f"{tag}_metadata.json"
    )

    assert metadata_path.is_file(), (
        f"Missing Phase 1 metadata for {tag}"
    )

    metadata = json.loads(metadata_path.read_text())

    assert metadata["process"] == process
    assert int(metadata["shard"]) == shard
    assert metadata["dataset_split"] == split
    assert (
        int(metadata["n_generated_expected"])
        == expected_events
    )
    assert (
        int(metadata["n_root_events"])
        == expected_events
    )

    assert input_path.is_file(), (
        f"Missing local HepMC input: {input_path}"
    )

    print(
        f"HASHING {index:02d}/25 "
        f"{process:24s} shard={shard:02d}",
        flush=True,
    )

    input_sha = sha256(input_path)

    assert (
        input_sha
        == metadata["input_hepmc_sha256"]
    ), (
        f"HepMC SHA256 mismatch for {tag}"
    )

    banner_xsec = integrated_weight_pb(banner_text)

    metadata_xsec_raw = metadata.get(
        "generator_cross_section_pb_median"
    )

    metadata_xsec = (
        float(metadata_xsec_raw)
        if metadata_xsec_raw is not None
        else None
    )

    if (
        banner_xsec is not None
        and metadata_xsec is not None
        and banner_xsec != 0
    ):
        xsec_relative_difference = abs(
            metadata_xsec - banner_xsec
        ) / abs(banner_xsec)
    else:
        xsec_relative_difference = None

    records.append(
        {
            "process": process,
            "shard": shard,
            "split": split,
            "tag": tag,
            "input_hepmc": str(input_path),
            "input_basename": basename,
            "input_size_bytes": input_path.stat().st_size,
            "input_hepmc_sha256": input_sha,
            "expected_run_name": run_name,
            "banner_path": str(banner_path),
            "banner_sha256": sha256(banner_path),
            "nevents": int(round(banner_nevents)),
            "iseed": seed_int if seed_int is not None else "",
            "lhaid": parameters.get("lhaid", ""),
            "ickkw": parameters.get("ickkw", ""),
            "cut_decays": parameters.get(
                "cut_decays",
                "",
            ),
            "ihtmin": (
                actual_ihtmin
                if actual_ihtmin is not None
                else ""
            ),
            "ihtmax": (
                actual_ihtmax
                if actual_ihtmax is not None
                else ""
            ),
            "ptj": parameters.get("ptj", ""),
            "ptb": parameters.get("ptb", ""),
            "etaj": parameters.get("etaj", ""),
            "etab": parameters.get("etab", ""),
            "drbb": parameters.get("drbb", ""),
            "drbj": parameters.get("drbj", ""),
            "process_definition": process_definition(
                banner_text
            ),
            "banner_integrated_weight_pb": (
                banner_xsec
                if banner_xsec is not None
                else ""
            ),
            "metadata_generator_xsec_pb": (
                metadata_xsec
                if metadata_xsec is not None
                else ""
            ),
            "xsec_relative_difference": (
                xsec_relative_difference
                if xsec_relative_difference is not None
                else ""
            ),
            "lineage_status": "EXACT_LINEAGE_VALID",
        }
    )

    print(
        f"VALID   {process:24s} "
        f"shard={shard:02d} "
        f"run={run_name}",
        flush=True,
    )

process_counts = Counter(
    record["process"]
    for record in records
)

assert dict(process_counts) == EXPECTED_COUNTS, (
    f"Unexpected process counts: {dict(process_counts)}"
)

assert len(
    {
        record["expected_run_name"]
        for record in records
    }
) == 25, "Run names are not unique"

assert len(
    {
        record["banner_sha256"]
        for record in records
    }
) == 25, "Banner hashes are not unique"

assert len(
    {
        record["input_hepmc_sha256"]
        for record in records
    }
) == 25, "HepMC hashes are not unique"

if seeds:
    assert len(seeds) == 25, (
        "Some but not all banners contain a seed"
    )

    assert len(set(seeds)) == 25, (
        "Generation seeds are not unique"
    )

fieldnames = list(records[0].keys())

with (
    OUT / "phase1_exact_generation_lineage.csv"
).open("w", newline="") as handle:
    writer = csv.DictWriter(
        handle,
        fieldnames=fieldnames,
    )

    writer.writeheader()
    writer.writerows(records)

xsec_comparisons = [
    float(record["xsec_relative_difference"])
    for record in records
    if record["xsec_relative_difference"] != ""
]

report = {
    "status": "exact_lineage_valid",
    "n_manifest_rows": len(manifest_rows),
    "n_exact_lineage_rows": len(records),
    "process_counts": dict(process_counts),
    "n_unique_run_names": len(
        {
            record["expected_run_name"]
            for record in records
        }
    ),
    "n_unique_banners": len(
        {
            record["banner_sha256"]
            for record in records
        }
    ),
    "n_unique_hepmc_hashes": len(
        {
            record["input_hepmc_sha256"]
            for record in records
        }
    ),
    "n_unique_seeds": len(set(seeds)),
    "n_xsec_comparisons": len(xsec_comparisons),
    "maximum_xsec_relative_difference": (
        max(xsec_comparisons)
        if xsec_comparisons
        else None
    ),
    "scientific_warning": (
        "Exact generation lineage does not make overlapping "
        "samples additive. qcd_bbbb_general includes the "
        "IHT 400-600 phase space."
    ),
}

(
    OUT / "phase1_exact_generation_lineage_report.json"
).write_text(
    json.dumps(report, indent=2, sort_keys=True)
    + "\n"
)

print()
print("GATE1C_EXACT_LINEAGE_VALID")
print("Rows:", len(records))
print("Unique run names:", report["n_unique_run_names"])
print("Unique banners:", report["n_unique_banners"])
print(
    "Unique HepMC hashes:",
    report["n_unique_hepmc_hashes"],
)
print("Unique seeds:", report["n_unique_seeds"])
print(
    "Cross-section comparisons:",
    report["n_xsec_comparisons"],
)
print(
    "Maximum cross-section relative difference:",
    report["maximum_xsec_relative_difference"],
)
print(
    "Wrote:",
    OUT / "phase1_exact_generation_lineage.csv",
)
print(
    "Wrote:",
    OUT
    / "phase1_exact_generation_lineage_report.json",
)
