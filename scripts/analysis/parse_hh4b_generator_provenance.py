#!/usr/bin/env python3
"""Parse generator provenance conservatively without assigning physical weights."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
import shutil
import tarfile
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


EXPECTED_EOS_CAMPAIGNS = 53
EXPECTED_ALL_CAMPAIGNS = 54
EXPECTED_PROVENANCE_FILES = 547

FLOAT_PATTERN = r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eEdD][+-]?\d+)?"

XSEC_PATTERNS = (
    re.compile(
        rf"Integrated\s+weight\s*\(\s*pb\s*\)\s*[:=]\s*({FLOAT_PATTERN})",
        flags=re.IGNORECASE,
    ),
    re.compile(
        rf"(?:cross[- ]?section|xsec|sigma)"
        rf"[^0-9+\-]{{0,80}}({FLOAT_PATTERN})\s*(pb|fb|nb|mb)\b",
        flags=re.IGNORECASE,
    ),
)

EVENT_PATTERNS = (
    re.compile(
        r"Number\s+of\s+Events\s*[:=]\s*(\d+)",
        flags=re.IGNORECASE,
    ),
    re.compile(
        r"(?:generated[_ ]events|events[_ ]generated|nevents)"
        r"[^0-9]{0,30}(\d+)",
        flags=re.IGNORECASE,
    ),
)

SUMW_PATTERNS = (
    re.compile(
        rf"(?:sum[_ ]of[_ ]weights|sum[_ ]weights|sumw)"
        rf"[^0-9+\-]{{0,40}}({FLOAT_PATTERN})",
        flags=re.IGNORECASE,
    ),
)

VERSION_PATTERNS = {
    "madgraph_version": re.compile(
        r"(?:MadGraph5_aMC@NLO|MG5_aMC|MadGraph)\s*(?:version|v)?\s*"
        r"([0-9]+(?:\.[0-9A-Za-z_-]+)+)",
        flags=re.IGNORECASE,
    ),
    "pythia_version": re.compile(
        r"Pythia\s*8(?:\s+version|\s+v)?\s*"
        r"([0-9]+(?:\.[0-9A-Za-z_-]+)+)",
        flags=re.IGNORECASE,
    ),
    "delphes_version": re.compile(
        r"Delphes\s*(?:version|v)?\s*"
        r"([0-9]+(?:\.[0-9A-Za-z_-]+)+)",
        flags=re.IGNORECASE,
    ),
}

RUN_CARD_INTEREST_KEYS = {
    "ebeam1",
    "ebeam2",
    "nevents",
    "iseed",
    "lhaid",
    "pdlabel",
    "ickkw",
    "xqcut",
    "ptj",
    "ptb",
    "pta",
    "ptl",
    "etaj",
    "etab",
    "etal",
    "drjj",
    "drbb",
    "drbj",
    "maxjetflavor",
    "dynamical_scale_choice",
    "fixed_ren_scale",
    "fixed_fac_scale",
    "event_norm",
}

JSON_KEY_CLASSES = {
    "cross_section": (
        "cross_section",
        "crosssection",
        "xsec",
        "sigma_pb",
        "integrated_weight_pb",
    ),
    "generated_events": (
        "generated_events",
        "events_generated",
        "nevents",
        "n_events",
        "event_count",
    ),
    "sum_generator_weights": (
        "sum_generator_weights",
        "sum_gen_weights",
        "sum_weights",
        "sumw",
    ),
    "beam_energy": (
        "beam_energy",
        "ebeam1",
        "ebeam2",
        "sqrt_s",
        "collision_energy",
    ),
    "pthat": (
        "pthat",
        "pthat_min",
        "pthat_max",
        "phase_space",
        "bin_low",
        "bin_high",
    ),
    "filter_efficiency": (
        "filter_efficiency",
        "filter_eff",
    ),
    "importance_weight": (
        "importance_weight",
        "sampling_weight",
        "sampling_fraction",
        "proposal_probability",
    ),
}

EVIDENCE_FIELDS = [
    "sample_class",
    "process_or_mode",
    "campaign",
    "field",
    "value",
    "unit",
    "confidence",
    "source_archive_path",
    "source_member_name",
    "source_line",
    "source_kind",
    "status",
]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def read_tsv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def write_tsv(
    path: Path,
    rows: Iterable[dict[str, object]],
    fields: list[str],
) -> None:
    materialized = list(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=fields,
            delimiter="\t",
            lineterminator="\n",
            extrasaction="ignore",
        )
        writer.writeheader()
        writer.writerows(materialized)


def parse_float(value: str) -> float:
    return float(value.replace("D", "E").replace("d", "e"))


def convert_xsec_to_pb(value: float, unit: str) -> float:
    factors = {
        "pb": 1.0,
        "fb": 1.0e-3,
        "nb": 1.0e3,
        "mb": 1.0e9,
    }
    normalized = unit.lower()
    if normalized not in factors:
        raise ValueError(f"unsupported cross-section unit: {unit}")
    return value * factors[normalized]


def normalize_key(key: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", key.lower()).strip("_")


def flatten_json(
    value: Any,
    prefix: str = "",
) -> Iterable[tuple[str, Any]]:
    if isinstance(value, dict):
        for key, item in value.items():
            normalized = normalize_key(str(key))
            child = f"{prefix}.{normalized}" if prefix else normalized
            yield from flatten_json(item, child)
    elif isinstance(value, list):
        for index, item in enumerate(value):
            child = f"{prefix}[{index}]"
            yield from flatten_json(item, child)
    else:
        yield prefix, value


def json_field_class(path: str) -> str | None:
    final = normalize_key(path.split(".")[-1].split("[")[0])
    for field_class, keys in JSON_KEY_CLASSES.items():
        if final in keys or any(final.endswith("_" + key) for key in keys):
            return field_class
    return None


def add_evidence(
    rows: list[dict[str, object]],
    *,
    sample_class: str,
    process: str,
    campaign: str,
    field: str,
    value: object,
    unit: str,
    confidence: str,
    archive_path: str,
    member_name: str,
    line: int | str,
    source_kind: str,
) -> None:
    rows.append(
        {
            "sample_class": sample_class,
            "process_or_mode": process,
            "campaign": campaign,
            "field": field,
            "value": value,
            "unit": unit or "none",
            "confidence": confidence,
            "source_archive_path": archive_path,
            "source_member_name": member_name,
            "source_line": line,
            "source_kind": source_kind,
            "status": "parsed_candidate_not_yet_authoritative",
        }
    )


def parse_run_card_line(line: str) -> tuple[str, str] | None:
    stripped = line.split("#", 1)[0].strip()
    if "=" not in stripped:
        return None
    left, right = [part.strip() for part in stripped.split("=", 1)]
    left_key = normalize_key(left)
    right_key = normalize_key(right)
    if right_key in RUN_CARD_INTEREST_KEYS:
        return right_key, left
    if left_key in RUN_CARD_INTEREST_KEYS:
        return left_key, right
    return None


def process_definition_line(line: str) -> bool:
    stripped = line.strip().lower()
    return stripped.startswith(
        (
            "generate ",
            "add process ",
            "decay ",
            "define ",
            "import model ",
        )
    )


def pythia_setting_line(line: str) -> bool:
    stripped = line.strip()
    if "=" not in stripped:
        return False
    lowered = stripped.lower()
    prefixes = (
        "beams:",
        "phasespace:",
        "hardqcd:",
        "higgs:",
        "pdf:",
        "tune:",
        "partonlevel:",
        "hadronlevel:",
        "23:",
        "24:",
        "25:",
        "6:",
    )
    return lowered.startswith(prefixes)


def consistent_numeric(
    values: list[float],
    relative_tolerance: float,
) -> tuple[bool, float | None, float | None, float | None]:
    finite = [value for value in values if math.isfinite(value)]
    if not finite:
        return False, None, None, None
    low = min(finite)
    high = max(finite)
    center = sum(finite) / len(finite)
    scale = max(abs(center), 1.0e-300)
    consistent = (high - low) / scale <= relative_tolerance
    return consistent, center, low, high


def parse_json_evidence(
    *,
    text: str,
    sample_class: str,
    process: str,
    campaign: str,
    archive_path: str,
    member_name: str,
    evidence_rows: list[dict[str, object]],
) -> None:
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        return

    for path, value in flatten_json(payload):
        field_class = json_field_class(path)
        if field_class is None:
            continue
        if isinstance(value, bool) or value is None:
            continue
        if isinstance(value, (int, float, str)):
            unit = "none"
            field = field_class
            serialized = value
            normalized_path = normalize_key(path)
            if field_class == "cross_section":
                if normalized_path.endswith("_fb"):
                    unit = "fb"
                elif normalized_path.endswith("_nb"):
                    unit = "nb"
                elif normalized_path.endswith("_mb"):
                    unit = "mb"
                else:
                    unit = "pb"
            elif field_class == "generated_events":
                unit = "events"
            elif field_class == "sum_generator_weights":
                unit = "weight"
            elif field_class == "beam_energy":
                unit = "GeV"
                field = path.split(".")[-1].split("[")[0]
            elif field_class == "filter_efficiency":
                unit = "dimensionless"
            elif field_class == "importance_weight":
                unit = "dimensionless"

            add_evidence(
                evidence_rows,
                sample_class=sample_class,
                process=process,
                campaign=campaign,
                field=field,
                value=serialized,
                unit=unit,
                confidence="high_json_key",
                archive_path=archive_path,
                member_name=member_name,
                line="json:" + path,
                source_kind="json",
            )


def source_policy(
    campaign_rows: list[dict[str, str]],
    legacy_decision: dict[str, str],
) -> list[dict[str, object]]:
    if len(campaign_rows) != EXPECTED_ALL_CAMPAIGNS:
        raise RuntimeError(
            f"campaign inventory rows={len(campaign_rows)}, "
            f"expected {EXPECTED_ALL_CAMPAIGNS}"
        )

    output: list[dict[str, object]] = []
    excluded = 0

    for row in campaign_rows:
        legacy = (
            row["process_or_mode"] == "ttbar_inclusive"
            and row["campaign"] == "parquet"
        )
        if legacy:
            excluded += 1
            include = False
            status = "excluded_unresolved_legacy_provenance"
            reason = (
                "Exact legacy proc_card and run_card were not recovered; "
                "physical weighting is forbidden. Provenance-complete EOS "
                "ttbar campaigns remain available."
            )
        else:
            include = True
            status = "included_pending_generator_and_theory_review"
            reason = (
                "EOS representative-bundle provenance is extracted; "
                "generator parsing and external reference-cross-section "
                "review remain required."
            )

        output.append(
            {
                "sample_class": row["sample_class"],
                "process_or_mode": row["process_or_mode"],
                "campaign": row["campaign"],
                "members": int(row["members"]),
                "generated_events_manifest": int(row["generated_events"]),
                "candidate_rows_metadata": int(row["candidate_rows_metadata"]),
                "train_members": int(row["train_members"]),
                "validation_members": int(row["validation_members"]),
                "final_evaluation_members": int(
                    row["final_evaluation_members"]
                ),
                "physical_normalization_included": include,
                "authoritative_physics_weight_authorized_now": False,
                "model_training_membership_changed": False,
                "candidate_content_opened": False,
                "reason": reason,
                "status": status,
            }
        )

    if excluded != 1:
        raise RuntimeError(
            f"excluded legacy campaign count={excluded}, expected 1"
        )

    if legacy_decision["legacy_sample_physics_weight_authorized"] != "False":
        raise RuntimeError("legacy decision unexpectedly authorizes weight")
    if legacy_decision["search_outcome"] != "legacy_provenance_partially_located":
        raise RuntimeError(
            "legacy search outcome changed unexpectedly: "
            + legacy_decision["search_outcome"]
        )

    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--campaign-inventory", type=Path, required=True)
    parser.add_argument("--provenance-archive", type=Path, required=True)
    parser.add_argument("--provenance-manifest", type=Path, required=True)
    parser.add_argument("--legacy-decision", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--source-commit", required=True)
    args = parser.parse_args()

    output = args.output.resolve()
    if output.exists():
        raise RuntimeError(f"output already exists: {output}")

    campaign_rows = read_tsv(args.campaign_inventory)
    manifest_rows = read_tsv(args.provenance_manifest)
    legacy_rows = read_tsv(args.legacy_decision)

    if len(manifest_rows) != EXPECTED_PROVENANCE_FILES:
        raise RuntimeError(
            f"provenance manifest rows={len(manifest_rows)}, "
            f"expected {EXPECTED_PROVENANCE_FILES}"
        )
    if len(legacy_rows) != 1:
        raise RuntimeError(
            f"legacy decision rows={len(legacy_rows)}, expected 1"
        )

    policy_rows = source_policy(campaign_rows, legacy_rows[0])

    output.mkdir(parents=True)
    try:
        manifest_map = {
            row["archive_path"]: row
            for row in manifest_rows
        }
        if len(manifest_map) != EXPECTED_PROVENANCE_FILES:
            raise RuntimeError("provenance archive paths are not unique")

        evidence_rows: list[dict[str, object]] = []
        process_rows: list[dict[str, object]] = []
        run_card_rows: list[dict[str, object]] = []
        pythia_rows: list[dict[str, object]] = []
        version_rows: list[dict[str, object]] = []
        qcd_rows: list[dict[str, object]] = []

        files_by_campaign: dict[
            tuple[str, str, str],
            list[dict[str, str]],
        ] = defaultdict(list)

        with tarfile.open(args.provenance_archive, mode="r:gz") as archive:
            members = {
                member.name: member
                for member in archive.getmembers()
            }
            if len(members) != EXPECTED_PROVENANCE_FILES:
                raise RuntimeError(
                    f"provenance archive members={len(members)}, "
                    f"expected {EXPECTED_PROVENANCE_FILES}"
                )

            if set(members) != set(manifest_map):
                raise RuntimeError(
                    "provenance archive and manifest path sets differ"
                )

            for archive_path in sorted(members):
                member = members[archive_path]
                manifest = manifest_map[archive_path]
                if not member.isfile():
                    raise RuntimeError(
                        f"nonregular provenance member: {archive_path}"
                    )

                handle = archive.extractfile(member)
                if handle is None:
                    raise RuntimeError(
                        f"could not read provenance member: {archive_path}"
                    )
                data = handle.read()
                if len(data) != int(manifest["size_bytes"]):
                    raise RuntimeError(
                        f"provenance member size mismatch: {archive_path}"
                    )
                if sha256_bytes(data) != manifest["sha256"]:
                    raise RuntimeError(
                        f"provenance member hash mismatch: {archive_path}"
                    )
                text = data.decode("utf-8")

                sample_class = manifest["sample_class"]
                process = manifest["process_or_mode"]
                campaign = manifest["campaign"]
                member_name = manifest["member_name"]
                artifact_types = manifest["artifact_types"]
                key = (sample_class, process, campaign)
                files_by_campaign[key].append(manifest)

                parse_json_evidence(
                    text=text,
                    sample_class=sample_class,
                    process=process,
                    campaign=campaign,
                    archive_path=archive_path,
                    member_name=member_name,
                    evidence_rows=evidence_rows,
                )

                lines = text.splitlines()
                for line_number, line in enumerate(lines, start=1):
                    stripped = line.strip()

                    if process_definition_line(stripped):
                        process_rows.append(
                            {
                                "sample_class": sample_class,
                                "process_or_mode": process,
                                "campaign": campaign,
                                "definition_line": stripped,
                                "source_archive_path": archive_path,
                                "source_member_name": member_name,
                                "source_line": line_number,
                                "status": "parsed_definition_candidate",
                            }
                        )

                    parsed_run = parse_run_card_line(line)
                    if parsed_run is not None:
                        setting, value = parsed_run
                        run_card_rows.append(
                            {
                                "sample_class": sample_class,
                                "process_or_mode": process,
                                "campaign": campaign,
                                "setting": setting,
                                "value": value,
                                "source_archive_path": archive_path,
                                "source_member_name": member_name,
                                "source_line": line_number,
                                "status": "parsed_run_card_candidate",
                            }
                        )
                        add_evidence(
                            evidence_rows,
                            sample_class=sample_class,
                            process=process,
                            campaign=campaign,
                            field=setting,
                            value=value,
                            unit=(
                                "GeV"
                                if setting in {"ebeam1", "ebeam2"}
                                else "events"
                                if setting == "nevents"
                                else "none"
                            ),
                            confidence="high_run_card",
                            archive_path=archive_path,
                            member_name=member_name,
                            line=line_number,
                            source_kind="run_card",
                        )

                    if pythia_setting_line(stripped):
                        setting, value = [
                            item.strip()
                            for item in stripped.split("=", 1)
                        ]
                        pythia_rows.append(
                            {
                                "sample_class": sample_class,
                                "process_or_mode": process,
                                "campaign": campaign,
                                "setting": setting,
                                "value": value,
                                "source_archive_path": archive_path,
                                "source_member_name": member_name,
                                "source_line": line_number,
                                "status": "parsed_pythia_setting_candidate",
                            }
                        )
                        lowered_setting = setting.lower()
                        if (
                            "pthat" in lowered_setting
                            or "phasespace:" in lowered_setting
                            or process == "qcd_hardqcd"
                        ):
                            qcd_rows.append(
                                {
                                    "sample_class": sample_class,
                                    "process_or_mode": process,
                                    "campaign": campaign,
                                    "field": setting,
                                    "value": value,
                                    "source_archive_path": archive_path,
                                    "source_member_name": member_name,
                                    "source_line": line_number,
                                    "status": "qcd_sampling_candidate",
                                }
                            )

                    for pattern_index, pattern in enumerate(
                        XSEC_PATTERNS,
                        start=1,
                    ):
                        for match in pattern.finditer(line):
                            raw_value = parse_float(match.group(1))
                            unit = (
                                "pb"
                                if pattern_index == 1
                                else match.group(2).lower()
                            )
                            value_pb = convert_xsec_to_pb(raw_value, unit)
                            add_evidence(
                                evidence_rows,
                                sample_class=sample_class,
                                process=process,
                                campaign=campaign,
                                field="generator_cross_section_pb",
                                value=f"{value_pb:.17g}",
                                unit="pb",
                                confidence=(
                                    "high_banner_generation_info"
                                    if pattern_index == 1
                                    else "medium_text_cross_section"
                                ),
                                archive_path=archive_path,
                                member_name=member_name,
                                line=line_number,
                                source_kind="text",
                            )

                    for pattern_index, pattern in enumerate(
                        EVENT_PATTERNS,
                        start=1,
                    ):
                        for match in pattern.finditer(line):
                            add_evidence(
                                evidence_rows,
                                sample_class=sample_class,
                                process=process,
                                campaign=campaign,
                                field="generated_events",
                                value=int(match.group(1)),
                                unit="events",
                                confidence=(
                                    "high_banner_generation_info"
                                    if pattern_index == 1
                                    else "medium_text_event_count"
                                ),
                                archive_path=archive_path,
                                member_name=member_name,
                                line=line_number,
                                source_kind="text",
                            )

                    for pattern in SUMW_PATTERNS:
                        for match in pattern.finditer(line):
                            add_evidence(
                                evidence_rows,
                                sample_class=sample_class,
                                process=process,
                                campaign=campaign,
                                field="sum_generator_weights",
                                value=f"{parse_float(match.group(1)):.17g}",
                                unit="weight",
                                confidence="medium_text_sumweights",
                                archive_path=archive_path,
                                member_name=member_name,
                                line=line_number,
                                source_kind="text",
                            )

                    for version_field, pattern in VERSION_PATTERNS.items():
                        for match in pattern.finditer(line):
                            version_rows.append(
                                {
                                    "sample_class": sample_class,
                                    "process_or_mode": process,
                                    "campaign": campaign,
                                    "field": version_field,
                                    "value": match.group(1),
                                    "source_archive_path": archive_path,
                                    "source_member_name": member_name,
                                    "source_line": line_number,
                                    "status": "parsed_version_candidate",
                                }
                            )

        eos_keys = {
            (
                row["sample_class"],
                row["process_or_mode"],
                row["campaign"],
            )
            for row in policy_rows
            if row["physical_normalization_included"] is True
        }
        if len(eos_keys) != EXPECTED_EOS_CAMPAIGNS:
            raise RuntimeError(
                f"included EOS campaigns={len(eos_keys)}, "
                f"expected {EXPECTED_EOS_CAMPAIGNS}"
            )
        if set(files_by_campaign) != eos_keys:
            missing = sorted(eos_keys - set(files_by_campaign))
            extra = sorted(set(files_by_campaign) - eos_keys)
            raise RuntimeError(
                "provenance campaign coverage mismatch\n"
                f"missing={missing}\nextra={extra}"
            )

        evidence_by_campaign: dict[
            tuple[str, str, str],
            list[dict[str, object]],
        ] = defaultdict(list)
        for row in evidence_rows:
            evidence_by_campaign[
                (
                    str(row["sample_class"]),
                    str(row["process_or_mode"]),
                    str(row["campaign"]),
                )
            ].append(row)

        process_count = Counter(
            (
                str(row["sample_class"]),
                str(row["process_or_mode"]),
                str(row["campaign"]),
            )
            for row in process_rows
        )
        run_count = Counter(
            (
                str(row["sample_class"]),
                str(row["process_or_mode"]),
                str(row["campaign"]),
            )
            for row in run_card_rows
        )
        pythia_count = Counter(
            (
                str(row["sample_class"]),
                str(row["process_or_mode"]),
                str(row["campaign"]),
            )
            for row in pythia_rows
        )
        qcd_count = Counter(
            (
                str(row["sample_class"]),
                str(row["process_or_mode"]),
                str(row["campaign"]),
            )
            for row in qcd_rows
        )

        campaign_parse_rows: list[dict[str, object]] = []
        xsec_rows: list[dict[str, object]] = []
        denominator_rows: list[dict[str, object]] = []
        unresolved_rows: list[dict[str, object]] = []

        for key in sorted(eos_keys):
            sample_class, process, campaign = key
            evidence = evidence_by_campaign[key]

            xsec_candidates: list[tuple[float, dict[str, object]]] = []
            event_candidates: list[tuple[int, dict[str, object]]] = []
            sumw_candidates: list[tuple[float, dict[str, object]]] = []
            ebeam1: list[float] = []
            ebeam2: list[float] = []

            for row in evidence:
                field = str(row["field"])
                value = row["value"]
                try:
                    if field == "generator_cross_section_pb":
                        xsec_candidates.append((float(value), row))
                    elif field == "cross_section":
                        numeric = float(value)
                        unit = str(row["unit"])
                        xsec_candidates.append(
                            (convert_xsec_to_pb(numeric, unit), row)
                        )
                    elif field == "generated_events":
                        event_candidates.append((int(float(value)), row))
                    elif field == "sum_generator_weights":
                        sumw_candidates.append((float(value), row))
                    elif normalize_key(field) == "ebeam1":
                        ebeam1.append(float(value))
                    elif normalize_key(field) == "ebeam2":
                        ebeam2.append(float(value))
                except (TypeError, ValueError):
                    continue

            high_xsecs = [
                value
                for value, row in xsec_candidates
                if str(row["confidence"]).startswith("high")
            ]
            xsec_values = high_xsecs or [
                value for value, _ in xsec_candidates
            ]
            xsec_consistent, xsec_center, xsec_low, xsec_high = (
                consistent_numeric(xsec_values, relative_tolerance=0.02)
            )

            event_values = [value for value, _ in event_candidates]
            event_unique = sorted(set(event_values))

            ebeam1_unique = sorted(set(ebeam1))
            ebeam2_unique = sorted(set(ebeam2))
            sqrt_s_candidates = sorted(
                {
                    first + second
                    for first in ebeam1_unique
                    for second in ebeam2_unique
                }
            )

            generator_xsec_status = (
                "candidate_consistent_within_2pct_not_authoritative"
                if xsec_values and xsec_consistent
                else "candidate_conflict_requires_review"
                if xsec_values
                else "not_located"
            )
            event_status = (
                "candidate_values_located_require_campaign_denominator_review"
                if event_values
                else "not_located"
            )
            sumw_status = (
                "candidate_values_located_require_signed_weight_review"
                if sumw_candidates
                else "not_located_do_not_substitute_event_count_yet"
            )
            beam_status = (
                "candidate_sqrt_s_located"
                if sqrt_s_candidates
                else "not_located"
            )
            process_status = (
                "definition_lines_located"
                if process_count[key] > 0
                else "not_located"
            )

            campaign_parse_rows.append(
                {
                    "sample_class": sample_class,
                    "process_or_mode": process,
                    "campaign": campaign,
                    "provenance_files": len(files_by_campaign[key]),
                    "process_definition_lines": process_count[key],
                    "run_card_settings": run_count[key],
                    "pythia_settings": pythia_count[key],
                    "qcd_sampling_candidates": qcd_count[key],
                    "generator_cross_section_candidate_count": len(
                        xsec_candidates
                    ),
                    "generator_cross_section_candidate_pb": (
                        f"{xsec_center:.17g}"
                        if xsec_center is not None
                        else ""
                    ),
                    "generator_cross_section_candidate_min_pb": (
                        f"{xsec_low:.17g}"
                        if xsec_low is not None
                        else ""
                    ),
                    "generator_cross_section_candidate_max_pb": (
                        f"{xsec_high:.17g}"
                        if xsec_high is not None
                        else ""
                    ),
                    "generator_cross_section_status": generator_xsec_status,
                    "generated_event_candidate_values": (
                        ",".join(str(value) for value in event_unique)
                        if event_unique
                        else ""
                    ),
                    "generated_event_status": event_status,
                    "sum_generator_weight_candidate_count": len(
                        sumw_candidates
                    ),
                    "sum_generator_weights_status": sumw_status,
                    "sqrt_s_candidate_GeV": (
                        ",".join(
                            f"{value:.17g}"
                            for value in sqrt_s_candidates
                        )
                        if sqrt_s_candidates
                        else ""
                    ),
                    "beam_energy_status": beam_status,
                    "process_definition_status": process_status,
                    "external_reference_cross_section_assigned": False,
                    "normalization_denominator_authorized": False,
                    "physical_weight_authorized": False,
                    "status": "generator_provenance_parsed_review_pending",
                }
            )

            for value, row in xsec_candidates:
                xsec_rows.append(
                    {
                        "sample_class": sample_class,
                        "process_or_mode": process,
                        "campaign": campaign,
                        "value_pb": f"{value:.17g}",
                        "confidence": row["confidence"],
                        "source_archive_path": row["source_archive_path"],
                        "source_member_name": row["source_member_name"],
                        "source_line": row["source_line"],
                        "authoritative_generator_cross_section": False,
                        "status": "candidate_requires_review",
                    }
                )

            for value, row in event_candidates:
                denominator_rows.append(
                    {
                        "sample_class": sample_class,
                        "process_or_mode": process,
                        "campaign": campaign,
                        "candidate_type": "generated_events",
                        "value": value,
                        "confidence": row["confidence"],
                        "source_archive_path": row["source_archive_path"],
                        "source_member_name": row["source_member_name"],
                        "source_line": row["source_line"],
                        "normalization_denominator_authorized": False,
                        "status": "candidate_requires_weight_strategy_review",
                    }
                )
            for value, row in sumw_candidates:
                denominator_rows.append(
                    {
                        "sample_class": sample_class,
                        "process_or_mode": process,
                        "campaign": campaign,
                        "candidate_type": "sum_generator_weights",
                        "value": f"{value:.17g}",
                        "confidence": row["confidence"],
                        "source_archive_path": row["source_archive_path"],
                        "source_member_name": row["source_member_name"],
                        "source_line": row["source_line"],
                        "normalization_denominator_authorized": False,
                        "status": "candidate_requires_signed_weight_review",
                    }
                )

            required_statuses = {
                "process_definition": process_status,
                "beam_energy": beam_status,
                "generator_cross_section": generator_xsec_status,
                "generated_events": event_status,
                "sum_generator_weights_or_proven_unweighted_denominator": (
                    sumw_status
                ),
                "decay_and_branching_fraction_convention": "manual_review_required",
                "filter_efficiency_convention": "manual_review_required",
                "external_reference_cross_section": "not_assigned",
                "stitching_and_overlap_policy": (
                    "manual_review_required"
                    if process.startswith("qcd_")
                    else "not_yet_classified"
                ),
            }
            for field, status in required_statuses.items():
                if status not in {
                    "definition_lines_located",
                    "candidate_sqrt_s_located",
                    "candidate_consistent_within_2pct_not_authoritative",
                    "candidate_values_located_require_campaign_denominator_review",
                }:
                    unresolved_rows.append(
                        {
                            "sample_class": sample_class,
                            "process_or_mode": process,
                            "campaign": campaign,
                            "field": field,
                            "current_status": status,
                            "blocking_for_physical_weight": True,
                            "required_next_action": (
                                "review parsed evidence and attach an "
                                "authoritative source or explicit convention"
                            ),
                        }
                    )

        write_tsv(
            output / "authoritative_source_policy.tsv",
            policy_rows,
            list(policy_rows[0]),
        )
        write_tsv(
            output / "campaign_generator_parse.tsv",
            campaign_parse_rows,
            list(campaign_parse_rows[0]),
        )
        write_tsv(
            output / "generator_provenance_evidence.tsv",
            evidence_rows,
            EVIDENCE_FIELDS,
        )
        write_tsv(
            output / "process_definition_candidates.tsv",
            process_rows,
            (
                list(process_rows[0])
                if process_rows
                else [
                    "sample_class",
                    "process_or_mode",
                    "campaign",
                    "definition_line",
                    "source_archive_path",
                    "source_member_name",
                    "source_line",
                    "status",
                ]
            ),
        )
        write_tsv(
            output / "run_card_setting_candidates.tsv",
            run_card_rows,
            (
                list(run_card_rows[0])
                if run_card_rows
                else [
                    "sample_class",
                    "process_or_mode",
                    "campaign",
                    "setting",
                    "value",
                    "source_archive_path",
                    "source_member_name",
                    "source_line",
                    "status",
                ]
            ),
        )
        write_tsv(
            output / "pythia_setting_candidates.tsv",
            pythia_rows,
            (
                list(pythia_rows[0])
                if pythia_rows
                else [
                    "sample_class",
                    "process_or_mode",
                    "campaign",
                    "setting",
                    "value",
                    "source_archive_path",
                    "source_member_name",
                    "source_line",
                    "status",
                ]
            ),
        )
        write_tsv(
            output / "generator_version_candidates.tsv",
            version_rows,
            (
                list(version_rows[0])
                if version_rows
                else [
                    "sample_class",
                    "process_or_mode",
                    "campaign",
                    "field",
                    "value",
                    "source_archive_path",
                    "source_member_name",
                    "source_line",
                    "status",
                ]
            ),
        )
        write_tsv(
            output / "generator_cross_section_candidates.tsv",
            xsec_rows,
            (
                list(xsec_rows[0])
                if xsec_rows
                else [
                    "sample_class",
                    "process_or_mode",
                    "campaign",
                    "value_pb",
                    "confidence",
                    "source_archive_path",
                    "source_member_name",
                    "source_line",
                    "authoritative_generator_cross_section",
                    "status",
                ]
            ),
        )
        write_tsv(
            output / "normalization_denominator_candidates.tsv",
            denominator_rows,
            (
                list(denominator_rows[0])
                if denominator_rows
                else [
                    "sample_class",
                    "process_or_mode",
                    "campaign",
                    "candidate_type",
                    "value",
                    "confidence",
                    "source_archive_path",
                    "source_member_name",
                    "source_line",
                    "normalization_denominator_authorized",
                    "status",
                ]
            ),
        )
        write_tsv(
            output / "qcd_importance_and_phase_space_candidates.tsv",
            qcd_rows,
            (
                list(qcd_rows[0])
                if qcd_rows
                else [
                    "sample_class",
                    "process_or_mode",
                    "campaign",
                    "field",
                    "value",
                    "source_archive_path",
                    "source_member_name",
                    "source_line",
                    "status",
                ]
            ),
        )
        write_tsv(
            output / "unresolved_physical_normalization_fields.tsv",
            unresolved_rows,
            (
                list(unresolved_rows[0])
                if unresolved_rows
                else [
                    "sample_class",
                    "process_or_mode",
                    "campaign",
                    "field",
                    "current_status",
                    "blocking_for_physical_weight",
                    "required_next_action",
                ]
            ),
        )

        included_policy = [
            row
            for row in policy_rows
            if row["physical_normalization_included"] is True
        ]
        excluded_policy = [
            row
            for row in policy_rows
            if row["physical_normalization_included"] is False
        ]

        summary = {
            "schema_version": 1,
            "status": (
                "hh4b_physical_normalization_generator_provenance_parse_pass"
            ),
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "source_commit": args.source_commit,
            "inputs": {
                "campaign_inventory": str(args.campaign_inventory),
                "campaign_inventory_sha256": sha256_file(
                    args.campaign_inventory
                ),
                "provenance_archive": str(args.provenance_archive),
                "provenance_archive_sha256": sha256_file(
                    args.provenance_archive
                ),
                "provenance_manifest": str(args.provenance_manifest),
                "provenance_manifest_sha256": sha256_file(
                    args.provenance_manifest
                ),
                "legacy_decision": str(args.legacy_decision),
                "legacy_decision_sha256": sha256_file(
                    args.legacy_decision
                ),
            },
            "source_policy": {
                "campaign_records": len(policy_rows),
                "included_eos_campaigns": len(included_policy),
                "excluded_legacy_campaigns": len(excluded_policy),
                "included_generated_events": sum(
                    int(row["generated_events_manifest"])
                    for row in included_policy
                ),
                "excluded_generated_events": sum(
                    int(row["generated_events_manifest"])
                    for row in excluded_policy
                ),
                "model_training_membership_changed": False,
            },
            "parsing": {
                "eos_campaigns": len(campaign_parse_rows),
                "provenance_files_verified": len(manifest_rows),
                "evidence_rows": len(evidence_rows),
                "process_definition_candidates": len(process_rows),
                "run_card_setting_candidates": len(run_card_rows),
                "pythia_setting_candidates": len(pythia_rows),
                "generator_version_candidates": len(version_rows),
                "generator_cross_section_candidates": len(xsec_rows),
                "normalization_denominator_candidates": len(
                    denominator_rows
                ),
                "qcd_importance_and_phase_space_candidates": len(qcd_rows),
                "unresolved_field_rows": len(unresolved_rows),
                "campaigns_with_xsec_candidates": sum(
                    bool(row["generator_cross_section_candidate_count"])
                    for row in campaign_parse_rows
                ),
                "campaigns_with_process_definition_lines": sum(
                    int(row["process_definition_lines"]) > 0
                    for row in campaign_parse_rows
                ),
                "campaigns_with_beam_energy_candidates": sum(
                    bool(row["sqrt_s_candidate_GeV"])
                    for row in campaign_parse_rows
                ),
            },
            "readiness": {
                "generator_provenance_parsed": True,
                "legacy_campaign_excluded_from_authoritative_physical_weight": True,
                "external_reference_cross_sections_assigned": 0,
                "normalization_denominators_authorized": 0,
                "qcd_stitching_resolved": False,
                "physics_normalization_ready": False,
            },
            "controls": {
                "candidate_parquet_files_opened": 0,
                "candidate_rows_read": 0,
                "validation_candidate_files_opened": 0,
                "evaluation_candidate_files_opened": 0,
                "event_payload_files_opened": 0,
                "event_payload_files_extracted": 0,
                "cross_sections_assigned": 0,
                "physical_yields_calculated": 0,
                "models_trained": 0,
                "thresholds_selected": 0,
            },
            "next_gate": (
                "review_generator_parse_and_freeze_campaign_equivalence_"
                "decay_filter_and_denominator_conventions"
            ),
        }

        (output / "summary.json").write_text(
            json.dumps(summary, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

        (output / "README.md").write_text(
            "# HH4b generator-provenance parse\n\n"
            "This checkpoint parses the 547 checksum-verified provenance "
            "files from all 53 EOS process-campaign bundles. It records "
            "candidate process definitions, run-card settings, Pythia "
            "settings, generator versions, generator cross sections, event "
            "counts, possible weight sums, and QCD phase-space information.\n\n"
            "All parsed quantities remain candidates. No generator cross "
            "section is promoted to an authoritative reference cross "
            "section, and no event-count candidate is promoted to a "
            "normalization denominator without a weight-strategy review.\n\n"
            "The unresolved 270k-event local legacy ttbar campaign is "
            "excluded from the authoritative physical-normalization source "
            "policy. This does not change model-training membership or "
            "modify candidate data. Provenance-complete EOS ttbar campaigns "
            "remain included pending review.\n\n"
            "No candidate Parquet, validation content, final-evaluation "
            "content, ROOT, HepMC, or LHE payload is opened. No physical "
            "yield or model threshold is calculated.\n",
            encoding="utf-8",
        )

        checksum_path = output / "SHA256SUMS"
        products = sorted(
            path
            for path in output.iterdir()
            if path.is_file() and path != checksum_path
        )
        checksum_path.write_text(
            "\n".join(
                f"{sha256_file(path)}  {path.name}"
                for path in products
            )
            + "\n",
            encoding="utf-8",
        )

        print(json.dumps(summary, indent=2, sort_keys=True))
        print()
        print("ALL_547_PROVENANCE_FILES_HASH_VERIFIED_PASS")
        print("ALL_53_EOS_CAMPAIGNS_PARSED_PASS")
        print("LEGACY_270K_TTBAR_AUTHORITATIVE_PHYSICAL_EXCLUSION_PASS")
        print("MODEL_TRAINING_MEMBERSHIP_UNCHANGED_PASS")
        print("GENERATOR_CROSS_SECTION_CANDIDATES_ONLY_PASS")
        print("NORMALIZATION_DENOMINATORS_REMAIN_UNAUTHORIZED_PASS")
        print("NO_EXTERNAL_REFERENCE_CROSS_SECTIONS_ASSIGNED")
        print("NO_CANDIDATE_PARQUET_OPENED")
        print("NO_VALIDATION_CANDIDATE_CONTENT_OPENED")
        print("NO_EVALUATION_CANDIDATE_CONTENT_OPENED")
        print("NO_PHYSICAL_YIELDS_CALCULATED")
        print("HH4B_GENERATOR_PROVENANCE_PARSE_PASS")
    except Exception:
        shutil.rmtree(output, ignore_errors=True)
        raise


if __name__ == "__main__":
    main()
