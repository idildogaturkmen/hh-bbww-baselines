#!/usr/bin/env python3
"""Review generator configurations source-aware, without authorizing weights."""

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


EXPECTED_PROVENANCE_FILES = 547
EXPECTED_EOS_CAMPAIGNS = 53
EXPECTED_ALL_CAMPAIGNS = 54
EXPECTED_STANDARD_CARD_CAMPAIGNS = 47
EXPECTED_NONSTANDARD_CAMPAIGNS = 6

VOLATILE_RUN_KEYS = {
    "iseed",
    "nevents",
    "python_seed",
}

VOLATILE_PYTHIA_KEYS = {
    "random:seed",
    "random:setseed",
    "main:numberofevents",
}

DECAY_SETTING_PATTERN = re.compile(
    r"^(?:\d+|[A-Za-z0-9_+\-]+):"
    r"(?:onmode|onifmatch|onifany|offifany|oneifany|maydecay)$",
    flags=re.IGNORECASE,
)

QCD_TOKEN_PATTERN = re.compile(
    r"(?:pthat|phase.?space|importance|sampling|bin[_ -]?"
    r"(?:low|high|index)|minimum_fraction|proposal|stitch|overlap)",
    flags=re.IGNORECASE,
)

FLOAT_PATTERN = re.compile(
    r"^[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eEdD][+-]?\d+)?$"
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


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


def normalize_key(value: str) -> str:
    return re.sub(r"[^a-z0-9:+\-]+", "_", value.lower()).strip("_")


def normalize_space(value: str) -> str:
    return " ".join(value.strip().split())


def strip_card_comment(line: str) -> str:
    positions = [
        position
        for marker in ("#", "!")
        if (position := line.find(marker)) >= 0
    ]
    if positions:
        line = line[: min(positions)]
    return line.strip()


def looks_like_key(value: str) -> bool:
    normalized = normalize_key(value)
    return bool(
        normalized
        and (
            re.fullmatch(r"[a-z][a-z0-9_:+\-]*", normalized)
            or re.fullmatch(
                r"[0-9]+:[a-z][a-z0-9_:+\-]*",
                normalized,
            )
        )
    )

def parse_assignment(line: str) -> tuple[str, str] | None:
    stripped = strip_card_comment(line)
    if not stripped or "=" not in stripped:
        return None

    left, right = [part.strip() for part in stripped.split("=", 1)]
    if not left or not right:
        return None

    if looks_like_key(right) and not looks_like_key(left):
        return normalize_key(right), normalize_space(left)

    if looks_like_key(left):
        return normalize_key(left), normalize_space(right)

    return None


def source_kind(member_name: str, artifact_types: str) -> str:
    lowered = member_name.lower()
    types = set(artifact_types.split(","))

    if lowered.endswith("/proc_card_mg5.dat") or "proc_card" in types:
        return "proc_card"
    if lowered.endswith("/run_card.dat") or "run_card" in types:
        return "run_card"
    if "pythia" in Path(lowered).name or "parton_shower_card" in types:
        return "pythia"
    if lowered.endswith(".json"):
        return "json"
    if "banner" in lowered or "generator_banner_or_lhe_init" in types:
        return "banner"
    if "generator_stdout_or_summary" in types:
        return "generator_log"
    return "other_provenance"


def process_line_kind(line: str) -> str | None:
    lowered = line.strip().lower()
    for prefix, kind in (
        ("import model ", "import_model"),
        ("define ", "definition"),
        ("generate ", "generate"),
        ("add process ", "add_process"),
        ("decay ", "madspin_or_decay_command"),
    ):
        if lowered.startswith(prefix):
            return kind
    return None


def flatten_json(
    value: Any,
    prefix: str = "",
) -> Iterable[tuple[str, Any]]:
    if isinstance(value, dict):
        for key, item in value.items():
            child = f"{prefix}.{key}" if prefix else str(key)
            yield from flatten_json(item, child)
    elif isinstance(value, list):
        for index, item in enumerate(value):
            yield from flatten_json(item, f"{prefix}[{index}]")
    else:
        yield prefix, value


def numeric(value: str) -> float | None:
    normalized = value.strip().replace("D", "E").replace("d", "e")
    if not FLOAT_PATTERN.fullmatch(normalized):
        return None
    try:
        return float(normalized)
    except ValueError:
        return None


def canonical_settings(
    rows: list[dict[str, object]],
    volatile: set[str],
) -> str:
    pairs = sorted(
        (
            normalize_key(str(row["setting"])),
            normalize_space(str(row["value"])),
        )
        for row in rows
        if normalize_key(str(row["setting"])) not in volatile
    )
    return "\n".join(f"{key}={value}" for key, value in pairs)


def campaign_key(row: dict[str, object]) -> tuple[str, str, str]:
    return (
        str(row["sample_class"]),
        str(row["process_or_mode"]),
        str(row["campaign"]),
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--campaign-inventory", type=Path, required=True)
    parser.add_argument("--provenance-archive", type=Path, required=True)
    parser.add_argument("--provenance-manifest", type=Path, required=True)
    parser.add_argument("--c1-campaign-parse", type=Path, required=True)
    parser.add_argument("--c1-source-policy", type=Path, required=True)
    parser.add_argument("--c1-xsec-candidates", type=Path, required=True)
    parser.add_argument("--c1-denominator-candidates", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--source-commit", required=True)
    args = parser.parse_args()

    output = args.output.resolve()
    if output.exists():
        raise RuntimeError(f"output already exists: {output}")

    campaign_inventory = read_tsv(args.campaign_inventory)
    manifest_rows = read_tsv(args.provenance_manifest)
    c1_campaign_rows = read_tsv(args.c1_campaign_parse)
    c1_source_policy = read_tsv(args.c1_source_policy)
    c1_xsec_rows = read_tsv(args.c1_xsec_candidates)
    c1_denominator_rows = read_tsv(args.c1_denominator_candidates)

    if len(campaign_inventory) != EXPECTED_ALL_CAMPAIGNS:
        raise RuntimeError(
            f"campaign inventory rows={len(campaign_inventory)}, "
            f"expected {EXPECTED_ALL_CAMPAIGNS}"
        )
    if len(manifest_rows) != EXPECTED_PROVENANCE_FILES:
        raise RuntimeError(
            f"provenance manifest rows={len(manifest_rows)}, "
            f"expected {EXPECTED_PROVENANCE_FILES}"
        )
    if len(c1_campaign_rows) != EXPECTED_EOS_CAMPAIGNS:
        raise RuntimeError(
            f"PN-c1 campaign rows={len(c1_campaign_rows)}, "
            f"expected {EXPECTED_EOS_CAMPAIGNS}"
        )
    if len(c1_source_policy) != EXPECTED_ALL_CAMPAIGNS:
        raise RuntimeError(
            f"PN-c1 source-policy rows={len(c1_source_policy)}, "
            f"expected {EXPECTED_ALL_CAMPAIGNS}"
        )

    included_keys = {
        (
            row["sample_class"],
            row["process_or_mode"],
            row["campaign"],
        )
        for row in c1_source_policy
        if row["physical_normalization_included"] == "True"
    }
    if len(included_keys) != EXPECTED_EOS_CAMPAIGNS:
        raise RuntimeError("PN-c1 included-source count mismatch")

    manifest_map = {row["archive_path"]: row for row in manifest_rows}
    if len(manifest_map) != EXPECTED_PROVENANCE_FILES:
        raise RuntimeError("provenance archive paths are not unique")

    output.mkdir(parents=True)
    try:
        file_rows: list[dict[str, object]] = []
        process_rows: list[dict[str, object]] = []
        run_rows: list[dict[str, object]] = []
        pythia_rows: list[dict[str, object]] = []
        decay_rows: list[dict[str, object]] = []
        qcd_rows: list[dict[str, object]] = []

        raw_by_campaign: dict[
            tuple[str, str, str],
            dict[str, list[str]],
        ] = defaultdict(lambda: defaultdict(list))

        with tarfile.open(args.provenance_archive, mode="r:gz") as archive:
            members = {
                member.name: member
                for member in archive.getmembers()
            }
            if len(members) != EXPECTED_PROVENANCE_FILES:
                raise RuntimeError(
                    f"archive members={len(members)}, "
                    f"expected {EXPECTED_PROVENANCE_FILES}"
                )
            if set(members) != set(manifest_map):
                raise RuntimeError(
                    "provenance archive and manifest path sets differ"
                )

            for archive_path in sorted(members):
                member = members[archive_path]
                manifest = manifest_map[archive_path]
                handle = archive.extractfile(member)
                if handle is None:
                    raise RuntimeError(
                        f"could not read provenance member: {archive_path}"
                    )
                data = handle.read()
                if len(data) != int(manifest["size_bytes"]):
                    raise RuntimeError(
                        f"size mismatch: {archive_path}"
                    )
                if sha256_bytes(data) != manifest["sha256"]:
                    raise RuntimeError(
                        f"SHA-256 mismatch: {archive_path}"
                    )
                text = data.decode("utf-8")
                kind = source_kind(
                    manifest["member_name"],
                    manifest["artifact_types"],
                )
                key = (
                    manifest["sample_class"],
                    manifest["process_or_mode"],
                    manifest["campaign"],
                )
                raw_by_campaign[key][kind].append(text)

                file_rows.append(
                    {
                        "sample_class": key[0],
                        "process_or_mode": key[1],
                        "campaign": key[2],
                        "archive_path": archive_path,
                        "member_name": manifest["member_name"],
                        "artifact_types": manifest["artifact_types"],
                        "source_kind": kind,
                        "size_bytes": len(data),
                        "sha256": manifest["sha256"],
                        "content_hash_verified": True,
                        "status": "source_aware_review_input",
                    }
                )

                if kind == "proc_card":
                    for line_number, line in enumerate(
                        text.splitlines(),
                        start=1,
                    ):
                        command_kind = process_line_kind(line)
                        if command_kind is None:
                            continue
                        command = normalize_space(line)
                        process_rows.append(
                            {
                                "sample_class": key[0],
                                "process_or_mode": key[1],
                                "campaign": key[2],
                                "command_kind": command_kind,
                                "command": command,
                                "source_archive_path": archive_path,
                                "source_member_name": manifest["member_name"],
                                "source_line": line_number,
                                "status": "source_aware_proc_card_command",
                            }
                        )

                if kind == "run_card":
                    for line_number, line in enumerate(
                        text.splitlines(),
                        start=1,
                    ):
                        assignment = parse_assignment(line)
                        if assignment is None:
                            continue
                        setting, value = assignment
                        run_rows.append(
                            {
                                "sample_class": key[0],
                                "process_or_mode": key[1],
                                "campaign": key[2],
                                "setting": setting,
                                "value": value,
                                "numeric_value": (
                                    ""
                                    if numeric(value) is None
                                    else f"{numeric(value):.17g}"
                                ),
                                "source_archive_path": archive_path,
                                "source_member_name": manifest["member_name"],
                                "source_line": line_number,
                                "status": "source_aware_run_card_setting",
                            }
                        )

                if kind == "pythia":
                    for line_number, line in enumerate(
                        text.splitlines(),
                        start=1,
                    ):
                        assignment = parse_assignment(line)
                        if assignment is None:
                            continue
                        setting, value = assignment
                        pythia_rows.append(
                            {
                                "sample_class": key[0],
                                "process_or_mode": key[1],
                                "campaign": key[2],
                                "setting": setting,
                                "value": value,
                                "source_archive_path": archive_path,
                                "source_member_name": manifest["member_name"],
                                "source_line": line_number,
                                "status": "source_aware_pythia_setting",
                            }
                        )
                        if DECAY_SETTING_PATTERN.fullmatch(setting):
                            decay_rows.append(
                                {
                                    "sample_class": key[0],
                                    "process_or_mode": key[1],
                                    "campaign": key[2],
                                    "particle_setting": setting,
                                    "value": value,
                                    "source_archive_path": archive_path,
                                    "source_member_name": manifest["member_name"],
                                    "source_line": line_number,
                                    "decay_or_branching_convention_authorized": False,
                                    "status": "forced_decay_candidate_requires_review",
                                }
                            )

                if (
                    key[1].startswith("qcd_")
                    or kind == "json"
                ):
                    if kind == "json":
                        try:
                            payload = json.loads(text)
                        except json.JSONDecodeError:
                            payload = None
                        if payload is not None:
                            for json_path, value in flatten_json(payload):
                                combined = f"{json_path} {value}"
                                if QCD_TOKEN_PATTERN.search(combined):
                                    qcd_rows.append(
                                        {
                                            "sample_class": key[0],
                                            "process_or_mode": key[1],
                                            "campaign": key[2],
                                            "field": json_path,
                                            "value": value,
                                            "source_archive_path": archive_path,
                                            "source_member_name": manifest["member_name"],
                                            "source_location": "json",
                                            "qcd_weight_or_stitching_authorized": False,
                                            "status": "qcd_sampling_or_overlap_candidate",
                                        }
                                    )
                    else:
                        for line_number, line in enumerate(
                            text.splitlines(),
                            start=1,
                        ):
                            if QCD_TOKEN_PATTERN.search(line):
                                qcd_rows.append(
                                    {
                                        "sample_class": key[0],
                                        "process_or_mode": key[1],
                                        "campaign": key[2],
                                        "field": "text_line",
                                        "value": normalize_space(line),
                                        "source_archive_path": archive_path,
                                        "source_member_name": manifest["member_name"],
                                        "source_location": line_number,
                                        "qcd_weight_or_stitching_authorized": False,
                                        "status": "qcd_sampling_or_overlap_candidate",
                                    }
                                )

        source_counts = Counter(str(row["source_kind"]) for row in file_rows)
        campaign_source_counts: dict[
            tuple[str, str, str],
            Counter[str],
        ] = defaultdict(Counter)
        for row in file_rows:
            campaign_source_counts[campaign_key(row)][
                str(row["source_kind"])
            ] += 1

        process_by_campaign: dict[
            tuple[str, str, str],
            list[dict[str, object]],
        ] = defaultdict(list)
        for row in process_rows:
            process_by_campaign[campaign_key(row)].append(row)

        run_by_campaign: dict[
            tuple[str, str, str],
            list[dict[str, object]],
        ] = defaultdict(list)
        for row in run_rows:
            run_by_campaign[campaign_key(row)].append(row)

        pythia_by_campaign: dict[
            tuple[str, str, str],
            list[dict[str, object]],
        ] = defaultdict(list)
        for row in pythia_rows:
            pythia_by_campaign[campaign_key(row)].append(row)

        decay_by_campaign = Counter(campaign_key(row) for row in decay_rows)
        qcd_by_campaign = Counter(campaign_key(row) for row in qcd_rows)

        c1_campaign_map = {
            (
                row["sample_class"],
                row["process_or_mode"],
                row["campaign"],
            ): row
            for row in c1_campaign_rows
        }
        inventory_map = {
            (
                row["sample_class"],
                row["process_or_mode"],
                row["campaign"],
            ): row
            for row in campaign_inventory
        }

        xsec_by_campaign: dict[
            tuple[str, str, str],
            list[dict[str, str]],
        ] = defaultdict(list)
        for row in c1_xsec_rows:
            xsec_by_campaign[
                (
                    row["sample_class"],
                    row["process_or_mode"],
                    row["campaign"],
                )
            ].append(row)

        denom_by_campaign: dict[
            tuple[str, str, str],
            list[dict[str, str]],
        ] = defaultdict(list)
        for row in c1_denominator_rows:
            denom_by_campaign[
                (
                    row["sample_class"],
                    row["process_or_mode"],
                    row["campaign"],
                )
            ].append(row)

        fingerprint_rows: list[dict[str, object]] = []
        beam_rows: list[dict[str, object]] = []
        xsec_review_rows: list[dict[str, object]] = []
        denominator_review_rows: list[dict[str, object]] = []
        convention_rows: list[dict[str, object]] = []

        for key in sorted(included_keys):
            sample_class, process, campaign = key
            process_commands = process_by_campaign[key]
            run_settings = run_by_campaign[key]
            pythia_settings = pythia_by_campaign[key]

            process_text = "\n".join(
                sorted(
                    normalize_space(str(row["command"]))
                    for row in process_commands
                )
            )
            run_text = canonical_settings(
                run_settings,
                VOLATILE_RUN_KEYS,
            )
            pythia_text = canonical_settings(
                pythia_settings,
                VOLATILE_PYTHIA_KEYS,
            )

            has_standard_cards = bool(
                process_commands and run_settings and pythia_settings
            )

            process_hash = sha256_text(process_text) if process_text else ""
            run_hash = sha256_text(run_text) if run_text else ""
            pythia_hash = sha256_text(pythia_text) if pythia_text else ""
            combined_hash = (
                sha256_text(
                    "\n".join((process_hash, run_hash, pythia_hash))
                )
                if has_standard_cards
                else ""
            )

            fingerprint_rows.append(
                {
                    "sample_class": sample_class,
                    "process_or_mode": process,
                    "campaign": campaign,
                    "proc_card_files": campaign_source_counts[key]["proc_card"],
                    "run_card_files": campaign_source_counts[key]["run_card"],
                    "pythia_files": campaign_source_counts[key]["pythia"],
                    "process_commands": len(process_commands),
                    "run_card_settings": len(run_settings),
                    "pythia_settings": len(pythia_settings),
                    "forced_decay_candidates": decay_by_campaign[key],
                    "qcd_sampling_candidates": qcd_by_campaign[key],
                    "process_fingerprint_sha256": process_hash,
                    "run_card_normalized_fingerprint_sha256": run_hash,
                    "pythia_normalized_fingerprint_sha256": pythia_hash,
                    "combined_configuration_fingerprint_sha256": combined_hash,
                    "standard_configuration_complete": has_standard_cards,
                    "campaign_equivalence_authorized": False,
                    "status": (
                        "standard_configuration_fingerprint_complete"
                        if has_standard_cards
                        else "nonstandard_or_incomplete_configuration_requires_review"
                    ),
                }
            )

            run_map: dict[str, list[str]] = defaultdict(list)
            for row in run_settings:
                run_map[str(row["setting"])].append(str(row["value"]))

            ebeam1 = sorted(
                {
                    value
                    for item in run_map.get("ebeam1", [])
                    if (value := numeric(item)) is not None
                }
            )
            ebeam2 = sorted(
                {
                    value
                    for item in run_map.get("ebeam2", [])
                    if (value := numeric(item)) is not None
                }
            )
            sqrt_s = sorted(
                {
                    left + right
                    for left in ebeam1
                    for right in ebeam2
                }
            )

            beam_rows.append(
                {
                    "sample_class": sample_class,
                    "process_or_mode": process,
                    "campaign": campaign,
                    "ebeam1_candidates_GeV": ",".join(
                        f"{value:.17g}" for value in ebeam1
                    ),
                    "ebeam2_candidates_GeV": ",".join(
                        f"{value:.17g}" for value in ebeam2
                    ),
                    "sqrt_s_candidates_GeV": ",".join(
                        f"{value:.17g}" for value in sqrt_s
                    ),
                    "path_energy_tag": inventory_map[key][
                        "energy_tag_inferred_from_path"
                    ],
                    "beam_energy_authorized": False,
                    "status": (
                        "source_aware_beam_candidate_located"
                        if sqrt_s
                        else "beam_energy_missing_from_available_configuration"
                    ),
                }
            )

            xsecs = xsec_by_campaign[key]
            values = [
                float(row["value_pb"])
                for row in xsecs
                if row["value_pb"]
            ]
            c1 = c1_campaign_map[key]
            if process == "qcd_hardqcd":
                xsec_status = (
                    "bin_dependent_candidates_do_not_average_or_promote"
                )
            elif not values:
                xsec_status = "generator_cross_section_not_located"
            elif max(values) - min(values) <= 0.02 * max(
                abs(sum(values) / len(values)),
                1.0e-300,
            ):
                xsec_status = (
                    "internally_consistent_generator_candidate_"
                    "external_reference_or_filter_review_pending"
                )
            else:
                xsec_status = "generator_candidates_conflict_require_review"

            xsec_review_rows.append(
                {
                    "sample_class": sample_class,
                    "process_or_mode": process,
                    "campaign": campaign,
                    "candidate_count": len(values),
                    "candidate_values_pb": ",".join(
                        f"{value:.17g}" for value in sorted(values)
                    ),
                    "c1_campaign_candidate_pb": c1[
                        "generator_cross_section_candidate_pb"
                    ],
                    "generator_cross_section_authorized": False,
                    "external_reference_cross_section_authorized": False,
                    "branching_fraction_convention_authorized": False,
                    "filter_efficiency_convention_authorized": False,
                    "status": xsec_status,
                }
            )

            denominator_candidates = denom_by_campaign[key]
            generated_candidates = [
                row["value"]
                for row in denominator_candidates
                if row["candidate_type"] == "generated_events"
            ]
            sumw_candidates = [
                row["value"]
                for row in denominator_candidates
                if row["candidate_type"] == "sum_generator_weights"
            ]
            run_nevents = sorted(
                set(run_map.get("nevents", []))
            )
            event_norm = sorted(
                set(run_map.get("event_norm", []))
            )

            if sumw_candidates:
                strategy = "signed_or_weighted_sumw_review_required"
            elif has_standard_cards:
                strategy = (
                    "prove_positive_unweighted_LO_then_manifest_event_count_"
                    "may_be_authorized"
                )
            else:
                strategy = (
                    "nonstandard_generation_requires_explicit_weight_"
                    "denominator_recovery"
                )

            denominator_review_rows.append(
                {
                    "sample_class": sample_class,
                    "process_or_mode": process,
                    "campaign": campaign,
                    "manifest_generated_events": int(
                        inventory_map[key]["generated_events"]
                    ),
                    "representative_run_nevents": ",".join(run_nevents),
                    "event_norm_settings": ",".join(event_norm),
                    "c1_generated_event_candidates": ",".join(
                        sorted(set(generated_candidates))
                    ),
                    "c1_sumw_candidates": ",".join(
                        sorted(set(sumw_candidates))
                    ),
                    "denominator_strategy_candidate": strategy,
                    "normalization_denominator_authorized": False,
                    "status": "weight_strategy_review_pending",
                }
            )

            convention_rows.append(
                {
                    "sample_class": sample_class,
                    "process_or_mode": process,
                    "campaign": campaign,
                    "standard_configuration_complete": has_standard_cards,
                    "beam_energy_candidate_located": bool(sqrt_s),
                    "process_definition_candidate_located": bool(
                        process_commands
                    ),
                    "pythia_configuration_candidate_located": bool(
                        pythia_settings
                    ),
                    "forced_decay_candidates": decay_by_campaign[key],
                    "qcd_sampling_candidates": qcd_by_campaign[key],
                    "generator_xsec_candidate_located": bool(values),
                    "sumw_candidate_located": bool(sumw_candidates),
                    "campaign_equivalence_authorized": False,
                    "decay_convention_authorized": False,
                    "filter_efficiency_authorized": False,
                    "denominator_authorized": False,
                    "physical_weight_authorized": False,
                    "status": "configuration_review_complete_freeze_pending",
                }
            )

        complete_fingerprints = [
            row
            for row in fingerprint_rows
            if row["standard_configuration_complete"] is True
        ]
        incomplete_fingerprints = [
            row
            for row in fingerprint_rows
            if row["standard_configuration_complete"] is False
        ]

        if len(complete_fingerprints) != EXPECTED_STANDARD_CARD_CAMPAIGNS:
            raise RuntimeError(
                "source-aware standard-card campaign count mismatch: "
                f"{len(complete_fingerprints)} != "
                f"{EXPECTED_STANDARD_CARD_CAMPAIGNS}"
            )
        if len(incomplete_fingerprints) != EXPECTED_NONSTANDARD_CAMPAIGNS:
            raise RuntimeError(
                "nonstandard campaign count mismatch: "
                f"{len(incomplete_fingerprints)} != "
                f"{EXPECTED_NONSTANDARD_CAMPAIGNS}"
            )

        grouped: dict[
            tuple[str, str],
            list[dict[str, object]],
        ] = defaultdict(list)
        for row in complete_fingerprints:
            grouped[
                (
                    str(row["process_or_mode"]),
                    str(
                        row[
                            "combined_configuration_fingerprint_sha256"
                        ]
                    ),
                )
            ].append(row)

        equivalence_rows: list[dict[str, object]] = []
        for (process, fingerprint), rows in sorted(grouped.items()):
            campaigns = sorted(str(row["campaign"]) for row in rows)
            equivalence_rows.append(
                {
                    "process_or_mode": process,
                    "configuration_fingerprint_sha256": fingerprint,
                    "campaign_count": len(campaigns),
                    "campaigns": ",".join(campaigns),
                    "candidate_equivalence_group_id": (
                        f"{process}__{fingerprint[:12]}"
                    ),
                    "all_members_same_process_label": True,
                    "configuration_equivalence_authorized": False,
                    "normalization_metadata_sharing_authorized": False,
                    "status": (
                        "multi_campaign_configuration_match_requires_freeze"
                        if len(campaigns) > 1
                        else "single_campaign_configuration_group"
                    ),
                }
            )

        missing_rows: list[dict[str, object]] = []
        for row in convention_rows:
            key = campaign_key(row)
            requirements = {
                "standard_configuration": row[
                    "standard_configuration_complete"
                ],
                "beam_energy": row[
                    "beam_energy_candidate_located"
                ],
                "process_definition": row[
                    "process_definition_candidate_located"
                ],
                "pythia_configuration": row[
                    "pythia_configuration_candidate_located"
                ],
                "generator_cross_section_provenance": row[
                    "generator_xsec_candidate_located"
                ],
                "decay_and_branching_fraction_convention": False,
                "filter_efficiency_convention": False,
                "normalization_denominator": False,
                "external_reference_cross_section": False,
            }
            if key[1].startswith("qcd_"):
                requirements["qcd_overlap_and_stitching"] = False
            for field, located_or_resolved in requirements.items():
                if located_or_resolved:
                    continue
                missing_rows.append(
                    {
                        "sample_class": key[0],
                        "process_or_mode": key[1],
                        "campaign": key[2],
                        "field": field,
                        "blocking_for_physical_weight": True,
                        "required_next_action": (
                            "freeze an explicit reviewed convention or "
                            "recover missing provenance"
                        ),
                    }
                )

        write_tsv(
            output / "source_aware_provenance_files.tsv",
            file_rows,
            list(file_rows[0]),
        )
        write_tsv(
            output / "process_card_commands.tsv",
            process_rows,
            list(process_rows[0]),
        )
        write_tsv(
            output / "run_card_settings_source_aware.tsv",
            run_rows,
            list(run_rows[0]),
        )
        write_tsv(
            output / "pythia_settings_source_aware.tsv",
            pythia_rows,
            list(pythia_rows[0]),
        )
        write_tsv(
            output / "forced_decay_candidates.tsv",
            decay_rows,
            (
                list(decay_rows[0])
                if decay_rows
                else [
                    "sample_class",
                    "process_or_mode",
                    "campaign",
                    "particle_setting",
                    "value",
                    "source_archive_path",
                    "source_member_name",
                    "source_line",
                    "decay_or_branching_convention_authorized",
                    "status",
                ]
            ),
        )
        write_tsv(
            output / "qcd_sampling_and_overlap_candidates.tsv",
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
                    "source_location",
                    "qcd_weight_or_stitching_authorized",
                    "status",
                ]
            ),
        )
        write_tsv(
            output / "campaign_configuration_fingerprints.tsv",
            fingerprint_rows,
            list(fingerprint_rows[0]),
        )
        write_tsv(
            output / "campaign_equivalence_candidates.tsv",
            equivalence_rows,
            list(equivalence_rows[0]),
        )
        write_tsv(
            output / "beam_energy_candidates_source_aware.tsv",
            beam_rows,
            list(beam_rows[0]),
        )
        write_tsv(
            output / "generator_cross_section_review.tsv",
            xsec_review_rows,
            list(xsec_review_rows[0]),
        )
        write_tsv(
            output / "normalization_denominator_review.tsv",
            denominator_review_rows,
            list(denominator_review_rows[0]),
        )
        write_tsv(
            output / "campaign_convention_status.tsv",
            convention_rows,
            list(convention_rows[0]),
        )
        write_tsv(
            output / "unresolved_configuration_review_fields.tsv",
            missing_rows,
            list(missing_rows[0]),
        )

        campaigns_with_beam = sum(
            bool(row["sqrt_s_candidates_GeV"])
            for row in beam_rows
        )
        campaigns_with_pythia = sum(
            int(row["pythia_settings"]) > 0
            for row in fingerprint_rows
        )
        campaigns_with_decay = sum(
            int(row["forced_decay_candidates"]) > 0
            for row in fingerprint_rows
        )
        qcd_campaigns_with_evidence = sum(
            int(row["qcd_sampling_candidates"]) > 0
            for row in fingerprint_rows
            if str(row["process_or_mode"]).startswith("qcd_")
        )

        summary = {
            "schema_version": 1,
            "status": (
                "hh4b_physical_normalization_configuration_review_pass"
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
                "pn_c1_campaign_parse": str(args.c1_campaign_parse),
                "pn_c1_campaign_parse_sha256": sha256_file(
                    args.c1_campaign_parse
                ),
            },
            "source_aware_parse": {
                "provenance_files_verified": len(file_rows),
                "eos_campaigns": len(fingerprint_rows),
                "standard_card_campaigns": len(complete_fingerprints),
                "nonstandard_or_incomplete_campaigns": len(
                    incomplete_fingerprints
                ),
                "proc_card_files": source_counts["proc_card"],
                "run_card_files": source_counts["run_card"],
                "pythia_files": source_counts["pythia"],
                "process_card_commands": len(process_rows),
                "run_card_settings": len(run_rows),
                "pythia_settings": len(pythia_rows),
                "forced_decay_candidates": len(decay_rows),
                "qcd_sampling_and_overlap_candidates": len(qcd_rows),
                "campaigns_with_beam_energy_candidates": campaigns_with_beam,
                "campaigns_with_pythia_settings": campaigns_with_pythia,
                "campaigns_with_forced_decay_candidates": campaigns_with_decay,
                "qcd_campaigns_with_sampling_or_overlap_evidence": (
                    qcd_campaigns_with_evidence
                ),
                "configuration_equivalence_candidate_groups": len(
                    equivalence_rows
                ),
                "unresolved_review_fields": len(missing_rows),
            },
            "readiness": {
                "source_aware_configuration_review_complete": True,
                "campaign_equivalence_authorized": 0,
                "decay_conventions_authorized": 0,
                "filter_efficiencies_authorized": 0,
                "normalization_denominators_authorized": 0,
                "external_reference_cross_sections_authorized": 0,
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
                "normalization_denominators_authorized": 0,
                "physical_yields_calculated": 0,
                "models_trained": 0,
                "thresholds_selected": 0,
            },
            "next_gate": (
                "freeze_campaign_equivalence_decay_filter_beam_and_"
                "denominator_conventions_fail_closed"
            ),
        }

        (output / "summary.json").write_text(
            json.dumps(summary, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

        (output / "README.md").write_text(
            "# HH4b source-aware generator-configuration review\n\n"
            "This checkpoint corrects the intentionally broad PN-c1 parse "
            "by interpreting each file according to its provenance type. "
            "Run-card assignments are read only from `run_card.dat`; Pythia "
            "assignments are read only from Pythia cards and logs; hard-"
            "process commands are read only from `proc_card_mg5.dat`.\n\n"
            "The review produces normalized configuration fingerprints and "
            "candidate campaign-equivalence groups while removing volatile "
            "event-count and random-seed fields from equivalence hashes. "
            "Equivalence is not yet authorized.\n\n"
            "Beam-energy, forced-decay, generator-cross-section, "
            "denominator, and QCD sampling evidence remain candidates. No "
            "external theory cross section, branching fraction, filter "
            "efficiency, denominator, event weight, yield, or threshold is "
            "assigned.\n",
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
        print("ALL_547_PROVENANCE_FILES_SOURCE_AWARE_HASH_VERIFIED_PASS")
        print("ALL_53_EOS_CAMPAIGNS_CONFIGURATION_REVIEWED_PASS")
        print("STANDARD_47_CARD_CAMPAIGNS_SOURCE_AWARE_PARSED_PASS")
        print("NONSTANDARD_6_CAMPAIGNS_FAIL_CLOSED_RECORDED_PASS")
        print("RUN_CARD_AND_PYTHIA_SOURCE_SEPARATION_PASS")
        print("CAMPAIGN_CONFIGURATION_FINGERPRINTS_PASS")
        print("CAMPAIGN_EQUIVALENCE_CANDIDATES_ONLY_PASS")
        print("NO_CAMPAIGN_EQUIVALENCE_AUTHORIZED")
        print("NO_NORMALIZATION_DENOMINATOR_AUTHORIZED")
        print("NO_EXTERNAL_REFERENCE_CROSS_SECTION_ASSIGNED")
        print("NO_CANDIDATE_PARQUET_OPENED")
        print("NO_VALIDATION_CANDIDATE_CONTENT_OPENED")
        print("NO_EVALUATION_CANDIDATE_CONTENT_OPENED")
        print("NO_PHYSICAL_YIELDS_CALCULATED")
        print("HH4B_SOURCE_AWARE_CONFIGURATION_REVIEW_PASS")
    except Exception:
        shutil.rmtree(output, ignore_errors=True)
        raise


if __name__ == "__main__":
    main()
