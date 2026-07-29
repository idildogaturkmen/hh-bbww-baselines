#!/usr/bin/env python3
# Freeze only conventions directly supported by exact generator cards.

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import shutil
import tarfile
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Any, Iterable


EXPECTED_ALL_CAMPAIGNS = 54
EXPECTED_EOS_CAMPAIGNS = 53
EXPECTED_STANDARD_CAMPAIGNS = 47
EXPECTED_NONSTANDARD_CAMPAIGNS = 6
EXPECTED_STANDARD_PROCESS_LABELS = 23
EXPECTED_EXACT_CARD_FILES = 188
EXPECTED_PROVENANCE_FILES = 547

RUN_VOLATILE_KEYS = {
    "run_tag",
    "nevents",
    "iseed",
    "python_seed",
}

PYTHIA8_VOLATILE_KEYS = {
    "main:numberofevents",
    "random:seed",
    "random:setseed",
    "hepmcoutput:file",
    "beams:lhef",
}

DECAY_KEY_PATTERN = re.compile(
    r"^[0-9]+:(?:onmode|onifmatch|onifany|offifany|oneifany|maydecay)$",
    flags=re.IGNORECASE,
)

QCD_KEY_PATTERN = re.compile(
    r"(?:pthat|ihtmin|ihtmax|importance|sampling|minimum_fraction|"
    r"proposal|bin[_ -]?(?:low|high|index)|stitch|overlap)",
    flags=re.IGNORECASE,
)

PHASE_SPACE_KEYS = {
    "ptj",
    "ptb",
    "pta",
    "ptl",
    "ptheavy",
    "xptj",
    "xptb",
    "xpta",
    "xptl",
    "etaj",
    "etab",
    "etaa",
    "etal",
    "drjj",
    "drbb",
    "drbj",
    "draa",
    "drll",
    "mmjj",
    "mmbb",
    "ihtmin",
    "ihtmax",
    "dsqrt_shat",
    "xqcut",
    "ickkw",
    "maxjetflavor",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


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


def normalize_space(value: str) -> str:
    return " ".join(value.strip().split())


def normalize_key(value: str) -> str:
    return re.sub(r"[^a-z0-9:+_\-]+", "_", value.lower()).strip("_")


def strip_comment(line: str) -> str:
    positions = [
        position
        for marker in ("#", "!")
        if (position := line.find(marker)) >= 0
    ]
    if positions:
        line = line[: min(positions)]
    return line.strip()


def canonical_proc_card(text: str) -> tuple[str, list[str]]:
    commands = []
    for line in text.splitlines():
        stripped = normalize_space(strip_comment(line))
        lowered = stripped.lower()
        if lowered.startswith(
            ("import model ", "define ", "generate ", "add process ", "decay ")
        ):
            commands.append(stripped)
    canonical = "\n".join(commands)
    return canonical, commands


def parse_run_card(text: str) -> dict[str, list[str]]:
    settings: dict[str, list[str]] = defaultdict(list)
    for line in text.splitlines():
        stripped = strip_comment(line)
        if not stripped or "=" not in stripped:
            continue
        value, key = [part.strip() for part in stripped.split("=", 1)]
        normalized_key = normalize_key(key)
        if normalized_key:
            settings[normalized_key].append(normalize_space(value))
    return dict(settings)


def canonical_run_card(
    settings: dict[str, list[str]],
) -> str:
    pairs = []
    for key in sorted(settings):
        if key in RUN_VOLATILE_KEYS:
            continue
        for value in sorted(settings[key]):
            pairs.append(f"{key}={value}")
    return "\n".join(pairs)


def parse_pythia8_card(text: str) -> dict[str, list[str]]:
    settings: dict[str, list[str]] = defaultdict(list)
    for line in text.splitlines():
        stripped = strip_comment(line)
        if not stripped or "=" not in stripped:
            continue
        key, value = [part.strip() for part in stripped.split("=", 1)]
        normalized_key = normalize_key(key)
        if normalized_key:
            settings[normalized_key].append(normalize_space(value))
    return dict(settings)


def canonical_pythia8_card(
    settings: dict[str, list[str]],
) -> str:
    pairs = []
    for key in sorted(settings):
        if key in PYTHIA8_VOLATILE_KEYS:
            continue
        for value in sorted(settings[key]):
            pairs.append(f"{key}={value}")
    return "\n".join(pairs)


def parse_pythia6_card(text: str) -> dict[str, list[str]]:
    settings: dict[str, list[str]] = defaultdict(list)
    for line in text.splitlines():
        stripped = strip_comment(line)
        if not stripped or "=" not in stripped:
            continue
        value, key = [part.strip() for part in stripped.split("=", 1)]
        normalized_key = normalize_key(key)
        if normalized_key:
            settings[normalized_key].append(normalize_space(value))
    return dict(settings)


def canonical_pythia6_card(
    settings: dict[str, list[str]],
) -> str:
    pairs = []
    for key in sorted(settings):
        for value in sorted(settings[key]):
            pairs.append(f"{key}={value}")
    return "\n".join(pairs)


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


def campaign_key(row: dict[str, object]) -> tuple[str, str, str]:
    return (
        str(row["sample_class"]),
        str(row["process_or_mode"]),
        str(row["campaign"]),
    )


def explicit_decay_commands(commands: list[str]) -> list[str]:
    output = []
    for command in commands:
        lowered = command.lower()
        if not lowered.startswith(("generate ", "add process ")):
            continue
        if "," not in command:
            continue
        suffix = command.split(",", 1)[1]
        if ">" in suffix:
            output.append(command)
    return output


def nondefault_phase_space(
    settings: dict[str, list[str]],
) -> list[str]:
    rows = []
    for key in sorted(PHASE_SPACE_KEYS):
        for value in settings.get(key, []):
            normalized = value.lower()
            if normalized in {
                "0",
                "0.0",
                "-1",
                "-1.0",
                "{}",
                "false",
                "none",
            }:
                continue
            rows.append(f"{key}={value}")
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--campaign-inventory", type=Path, required=True)
    parser.add_argument("--provenance-archive", type=Path, required=True)
    parser.add_argument("--provenance-manifest", type=Path, required=True)
    parser.add_argument("--source-policy", type=Path, required=True)
    parser.add_argument("--c2a-fingerprints", type=Path, required=True)
    parser.add_argument("--c2a-denominators", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--source-commit", required=True)
    args = parser.parse_args()

    output = args.output.resolve()
    if output.exists():
        raise RuntimeError(f"output already exists: {output}")

    inventory_rows = read_tsv(args.campaign_inventory)
    manifest_rows = read_tsv(args.provenance_manifest)
    source_policy_rows = read_tsv(args.source_policy)
    c2a_rows = read_tsv(args.c2a_fingerprints)
    denominator_input_rows = read_tsv(args.c2a_denominators)

    if len(inventory_rows) != EXPECTED_ALL_CAMPAIGNS:
        raise RuntimeError("campaign inventory row count mismatch")
    if len(manifest_rows) != EXPECTED_PROVENANCE_FILES:
        raise RuntimeError("provenance manifest row count mismatch")
    if len(source_policy_rows) != EXPECTED_ALL_CAMPAIGNS:
        raise RuntimeError("source policy row count mismatch")
    if len(c2a_rows) != EXPECTED_EOS_CAMPAIGNS:
        raise RuntimeError("PN-c2a campaign row count mismatch")
    if len(denominator_input_rows) != EXPECTED_EOS_CAMPAIGNS:
        raise RuntimeError("PN-c2a denominator row count mismatch")

    included_keys = {
        (
            row["sample_class"],
            row["process_or_mode"],
            row["campaign"],
        )
        for row in source_policy_rows
        if row["physical_normalization_included"] == "True"
    }
    if len(included_keys) != EXPECTED_EOS_CAMPAIGNS:
        raise RuntimeError("included EOS source count mismatch")

    standard_keys = {
        (
            row["sample_class"],
            row["process_or_mode"],
            row["campaign"],
        )
        for row in c2a_rows
        if row["standard_configuration_complete"] == "True"
    }
    nonstandard_keys = included_keys - standard_keys

    if len(standard_keys) != EXPECTED_STANDARD_CAMPAIGNS:
        raise RuntimeError("standard campaign count mismatch")
    if len(nonstandard_keys) != EXPECTED_NONSTANDARD_CAMPAIGNS:
        raise RuntimeError("nonstandard campaign count mismatch")
    if len({key[1] for key in standard_keys}) != EXPECTED_STANDARD_PROCESS_LABELS:
        raise RuntimeError("standard process-label count mismatch")

    manifest_map = {row["archive_path"]: row for row in manifest_rows}
    if len(manifest_map) != EXPECTED_PROVENANCE_FILES:
        raise RuntimeError("provenance archive paths are not unique")

    inventory_map = {
        (
            row["sample_class"],
            row["process_or_mode"],
            row["campaign"],
        ): row
        for row in inventory_rows
    }
    denominator_input_map = {
        (
            row["sample_class"],
            row["process_or_mode"],
            row["campaign"],
        ): row
        for row in denominator_input_rows
    }

    output.mkdir(parents=True)
    try:
        files_by_campaign: dict[
            tuple[str, str, str],
            dict[str, tuple[bytes, dict[str, str]]],
        ] = defaultdict(dict)
        json_by_campaign: dict[
            tuple[str, str, str],
            list[tuple[str, str]],
        ] = defaultdict(list)

        exact_card_rows: list[dict[str, object]] = []

        with tarfile.open(args.provenance_archive, mode="r:gz") as archive:
            members = {
                member.name: member
                for member in archive.getmembers()
            }
            if len(members) != EXPECTED_PROVENANCE_FILES:
                raise RuntimeError("provenance archive member count mismatch")
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
                        f"could not read provenance file: {archive_path}"
                    )
                data = handle.read()
                if len(data) != int(manifest["size_bytes"]):
                    raise RuntimeError(
                        f"size mismatch for provenance file: {archive_path}"
                    )
                if sha256_bytes(data) != manifest["sha256"]:
                    raise RuntimeError(
                        f"SHA-256 mismatch for provenance file: {archive_path}"
                    )

                key = (
                    manifest["sample_class"],
                    manifest["process_or_mode"],
                    manifest["campaign"],
                )
                basename = PurePosixPath(manifest["member_name"]).name
                parent_parts = PurePosixPath(
                    manifest["member_name"]
                ).parts
                under_cards = "cards" in parent_parts

                card_kind = ""
                if under_cards and basename == "proc_card_mg5.dat":
                    card_kind = "proc_card"
                elif under_cards and basename == "run_card.dat":
                    card_kind = "run_card"
                elif under_cards and basename == "pythia8_card_default.dat":
                    card_kind = "pythia8_card"
                elif under_cards and basename == "pythia_card_default.dat":
                    card_kind = "pythia6_card"

                if card_kind:
                    if card_kind in files_by_campaign[key]:
                        raise RuntimeError(
                            f"duplicate exact card kind for {key}: {card_kind}"
                        )
                    files_by_campaign[key][card_kind] = (data, manifest)
                    exact_card_rows.append(
                        {
                            "sample_class": key[0],
                            "process_or_mode": key[1],
                            "campaign": key[2],
                            "card_kind": card_kind,
                            "member_name": manifest["member_name"],
                            "archive_path": archive_path,
                            "size_bytes": len(data),
                            "sha256": manifest["sha256"],
                            "included_in_card_only_fingerprint": True,
                            "status": "exact_generator_card_hash_verified",
                        }
                    )

                if basename.endswith(".json"):
                    text = data.decode("utf-8")
                    json_by_campaign[key].append((archive_path, text))

        if len(exact_card_rows) != EXPECTED_EXACT_CARD_FILES:
            raise RuntimeError(
                f"exact card files={len(exact_card_rows)}, "
                f"expected {EXPECTED_EXACT_CARD_FILES}"
            )

        fingerprint_rows: list[dict[str, object]] = []
        beam_rows: list[dict[str, object]] = []
        decay_rows: list[dict[str, object]] = []
        filter_rows: list[dict[str, object]] = []
        denominator_rows: list[dict[str, object]] = []
        qcd_rows: list[dict[str, object]] = []
        nonstandard_rows: list[dict[str, object]] = []

        canonical_by_campaign: dict[
            tuple[str, str, str],
            dict[str, object],
        ] = {}

        for key in sorted(standard_keys):
            cards = files_by_campaign[key]
            required = {
                "proc_card",
                "run_card",
                "pythia8_card",
                "pythia6_card",
            }
            if set(cards) != required:
                raise RuntimeError(
                    f"exact card set mismatch for {key}: {sorted(cards)}"
                )

            proc_text = cards["proc_card"][0].decode("utf-8")
            run_text = cards["run_card"][0].decode("utf-8")
            pythia8_text = cards["pythia8_card"][0].decode("utf-8")
            pythia6_text = cards["pythia6_card"][0].decode("utf-8")

            proc_canonical, commands = canonical_proc_card(proc_text)
            run_settings = parse_run_card(run_text)
            run_canonical = canonical_run_card(run_settings)
            pythia8_settings = parse_pythia8_card(pythia8_text)
            pythia8_canonical = canonical_pythia8_card(pythia8_settings)
            pythia6_settings = parse_pythia6_card(pythia6_text)
            pythia6_canonical = canonical_pythia6_card(pythia6_settings)

            component_hashes = {
                "proc": sha256_text(proc_canonical),
                "run": sha256_text(run_canonical),
                "pythia8": sha256_text(pythia8_canonical),
                "pythia6": sha256_text(pythia6_canonical),
            }
            combined = sha256_text(
                "\n".join(
                    f"{name}:{component_hashes[name]}"
                    for name in sorted(component_hashes)
                )
            )

            canonical_by_campaign[key] = {
                "commands": commands,
                "run_settings": run_settings,
                "pythia8_settings": pythia8_settings,
                "pythia6_settings": pythia6_settings,
                "fingerprint": combined,
            }

            fingerprint_rows.append(
                {
                    "sample_class": key[0],
                    "process_or_mode": key[1],
                    "campaign": key[2],
                    "proc_card_canonical_sha256": component_hashes["proc"],
                    "run_card_operational_fields_removed_sha256": component_hashes[
                        "run"
                    ],
                    "pythia8_card_operational_fields_removed_sha256": (
                        component_hashes["pythia8"]
                    ),
                    "pythia6_card_canonical_sha256": component_hashes[
                        "pythia6"
                    ],
                    "card_only_combined_fingerprint_sha256": combined,
                    "excluded_run_fields": ",".join(sorted(RUN_VOLATILE_KEYS)),
                    "excluded_pythia8_fields": ",".join(
                        sorted(PYTHIA8_VOLATILE_KEYS)
                    ),
                    "logs_included": False,
                    "configuration_equivalence_authorized": False,
                    "status": "card_only_fingerprint_complete",
                }
            )

            lpp1 = sorted(set(run_settings.get("lpp1", [])))
            lpp2 = sorted(set(run_settings.get("lpp2", [])))
            ebeam1 = sorted(set(run_settings.get("ebeam1", [])))
            ebeam2 = sorted(set(run_settings.get("ebeam2", [])))
            path_tag = inventory_map[key]["energy_tag_inferred_from_path"]
            beam_authorized = (
                lpp1 == ["1"]
                and lpp2 == ["1"]
                and ebeam1 in (["6500"], ["6500.0"])
                and ebeam2 in (["6500"], ["6500.0"])
                and path_tag == "run2_13tev"
            )
            beam_rows.append(
                {
                    "sample_class": key[0],
                    "process_or_mode": key[1],
                    "campaign": key[2],
                    "lpp1": ",".join(lpp1),
                    "lpp2": ",".join(lpp2),
                    "ebeam1_GeV": ",".join(ebeam1),
                    "ebeam2_GeV": ",".join(ebeam2),
                    "sqrt_s_GeV": 13000 if beam_authorized else "",
                    "path_energy_tag": path_tag,
                    "beam_energy_convention_authorized": beam_authorized,
                    "status": (
                        "run_card_and_path_13tev_agree"
                        if beam_authorized
                        else "beam_convention_fail_closed"
                    ),
                }
            )

            me_decay = explicit_decay_commands(commands)
            shower_decay = []
            for setting, values in pythia8_settings.items():
                if DECAY_KEY_PATTERN.fullmatch(setting):
                    for value in values:
                        shower_decay.append(f"{setting}={value}")

            if me_decay and shower_decay:
                mode = "matrix_element_and_pythia_decay_controls_present"
            elif me_decay:
                mode = "explicit_matrix_element_decay_chain"
            elif shower_decay:
                mode = "explicit_pythia_decay_control"
            else:
                mode = "no_explicit_forced_decay_control_in_exact_cards"

            decay_rows.append(
                {
                    "sample_class": key[0],
                    "process_or_mode": key[1],
                    "campaign": key[2],
                    "matrix_element_decay_commands": " || ".join(me_decay),
                    "pythia_decay_settings": " || ".join(shower_decay),
                    "decay_control_mode": mode,
                    "numeric_branching_fraction_authorized": False,
                    "generator_xsec_branching_scope_authorized": False,
                    "status": "decay_scope_reviewed_branching_freeze_pending",
                }
            )

            cuts = nondefault_phase_space(run_settings)
            filter_rows.append(
                {
                    "sample_class": key[0],
                    "process_or_mode": key[1],
                    "campaign": key[2],
                    "nondefault_generation_phase_space_settings": " || ".join(
                        cuts
                    ),
                    "generator_reported_cross_section_scope": (
                        "configured_hard_process_and_run_card_phase_space"
                    ),
                    "external_inclusive_xsec_requires_acceptance_or_filter_proof": (
                        True
                    ),
                    "numeric_filter_efficiency_authorized": False,
                    "status": "generator_phase_space_scope_frozen_fail_closed",
                }
            )

            denominator_input = denominator_input_map[key]
            denominator_rows.append(
                {
                    "sample_class": key[0],
                    "process_or_mode": key[1],
                    "campaign": key[2],
                    "manifest_generated_events": int(
                        denominator_input["manifest_generated_events"]
                    ),
                    "representative_run_nevents": ",".join(
                        sorted(set(run_settings.get("nevents", [])))
                    ),
                    "event_norm": ",".join(
                        sorted(set(run_settings.get("event_norm", [])))
                    ),
                    "sumw_candidate": denominator_input[
                        "c1_sumw_candidates"
                    ],
                    "positive_unweighted_proof_complete": False,
                    "normalization_denominator_authorized": False,
                    "required_proof": (
                        "recover generator-level sumw or prove every generated "
                        "event has the same positive nominal weight before Ngen use"
                    ),
                    "status": "denominator_fail_closed_pending_weight_proof",
                }
            )

        process_campaigns: dict[str, set[tuple[str, str, str]]] = defaultdict(set)
        grouped: dict[
            tuple[str, str],
            list[tuple[str, str, str]],
        ] = defaultdict(list)

        for key, values in canonical_by_campaign.items():
            process_campaigns[key[1]].add(key)
            grouped[(key[1], str(values["fingerprint"]))].append(key)

        groups_per_process = Counter(process for process, _ in grouped)

        group_rows: list[dict[str, object]] = []
        campaign_freeze_rows: list[dict[str, object]] = []

        for (process, fingerprint), keys in sorted(grouped.items()):
            campaigns = sorted(key[2] for key in keys)
            process_has_single_group = groups_per_process[process] == 1
            group_rows.append(
                {
                    "process_or_mode": process,
                    "card_only_configuration_fingerprint_sha256": fingerprint,
                    "campaign_count": len(keys),
                    "campaigns": ",".join(campaigns),
                    "all_process_campaigns_covered": (
                        len(keys) == len(process_campaigns[process])
                    ),
                    "configuration_equivalence_authorized": (
                        process_has_single_group
                    ),
                    "normalization_configuration_sharing_authorized": (
                        process_has_single_group
                    ),
                    "cross_section_sharing_authorized": False,
                    "denominator_sharing_authorized": False,
                    "status": (
                        "card_only_process_equivalence_frozen"
                        if process_has_single_group
                        else "process_split_across_card_configurations"
                    ),
                }
            )
            for key in sorted(keys):
                campaign_freeze_rows.append(
                    {
                        "sample_class": key[0],
                        "process_or_mode": key[1],
                        "campaign": key[2],
                        "card_only_configuration_fingerprint_sha256": fingerprint,
                        "configuration_equivalence_authorized": (
                            process_has_single_group
                        ),
                        "equivalent_campaigns": ",".join(campaigns),
                        "cross_section_sharing_authorized": False,
                        "denominator_sharing_authorized": False,
                        "status": (
                            "card_only_equivalence_frozen"
                            if process_has_single_group
                            else "card_configuration_split_fail_closed"
                        ),
                    }
                )

        for key in sorted(nonstandard_keys):
            cards = files_by_campaign.get(key, {})
            blocker = (
                "ggF HH provenance bundle contains no exact proc/run/Pythia "
                "cards; recover generator configuration and weight denominator"
                if key[1] == "ggf_hh4b"
                else "importance-sampled hard-QCD provenance lacks exact "
                "proc/run cards; recover bin definitions and weight denominator"
            )
            nonstandard_rows.append(
                {
                    "sample_class": key[0],
                    "process_or_mode": key[1],
                    "campaign": key[2],
                    "exact_card_kinds_located": ",".join(sorted(cards)),
                    "path_energy_tag": inventory_map[key][
                        "energy_tag_inferred_from_path"
                    ],
                    "configuration_equivalence_authorized": False,
                    "beam_energy_convention_authorized": False,
                    "normalization_denominator_authorized": False,
                    "physical_weight_authorized": False,
                    "blocking_reason": blocker,
                    "status": "nonstandard_campaign_fail_closed",
                }
            )
            denominator_input = denominator_input_map[key]
            denominator_rows.append(
                {
                    "sample_class": key[0],
                    "process_or_mode": key[1],
                    "campaign": key[2],
                    "manifest_generated_events": int(
                        denominator_input["manifest_generated_events"]
                    ),
                    "representative_run_nevents": "",
                    "event_norm": "",
                    "sumw_candidate": denominator_input[
                        "c1_sumw_candidates"
                    ],
                    "positive_unweighted_proof_complete": False,
                    "normalization_denominator_authorized": False,
                    "required_proof": (
                        "recover explicit generator-weight denominator and "
                        "nonstandard generation configuration"
                    ),
                    "status": "nonstandard_denominator_fail_closed",
                }
            )

            if key[1].startswith("qcd_"):
                for archive_path, text in json_by_campaign.get(key, []):
                    try:
                        payload = json.loads(text)
                    except json.JSONDecodeError:
                        continue
                    for json_path, value in flatten_json(payload):
                        if QCD_KEY_PATTERN.search(f"{json_path} {value}"):
                            qcd_rows.append(
                                {
                                    "sample_class": key[0],
                                    "process_or_mode": key[1],
                                    "campaign": key[2],
                                    "field": json_path,
                                    "value": value,
                                    "source_archive_path": archive_path,
                                    "importance_weight_authorized": False,
                                    "stitching_authorized": False,
                                    "status": "structured_qcd_candidate_fail_closed",
                                }
                            )

        for key in sorted(standard_keys):
            if not key[1].startswith("qcd_"):
                continue
            run_settings = canonical_by_campaign[key]["run_settings"]
            for setting in sorted(run_settings):
                if not QCD_KEY_PATTERN.search(setting):
                    continue
                for value in run_settings[setting]:
                    qcd_rows.append(
                        {
                            "sample_class": key[0],
                            "process_or_mode": key[1],
                            "campaign": key[2],
                            "field": setting,
                            "value": value,
                            "source_archive_path": "exact_run_card.dat",
                            "importance_weight_authorized": False,
                            "stitching_authorized": False,
                            "status": "qcd_phase_space_candidate_fail_closed",
                        }
                    )

        unresolved_rows: list[dict[str, object]] = []
        for key in sorted(included_keys):
            is_standard = key in standard_keys
            fields = [
                (
                    "normalization_denominator",
                    "recover sumw or uniform-positive-weight proof",
                ),
                (
                    "branching_fraction_convention",
                    "review decay scope against chosen reference cross section",
                ),
                (
                    "filter_efficiency_or_acceptance",
                    "derive acceptance if an inclusive external cross section is used",
                ),
                (
                    "external_reference_cross_section",
                    "attach only after exact process and decay scope matching",
                ),
            ]
            if not is_standard:
                fields.extend(
                    [
                        (
                            "exact_generator_configuration",
                            "recover nonstandard generator configuration",
                        ),
                        (
                            "beam_energy",
                            "recover explicit nonstandard beam-energy provenance",
                        ),
                    ]
                )
            if key[1].startswith("qcd_"):
                fields.append(
                    (
                        "qcd_overlap_and_stitching",
                        "prove mutually exclusive phase space and importance weights",
                    )
                )
            for field, action in fields:
                unresolved_rows.append(
                    {
                        "sample_class": key[0],
                        "process_or_mode": key[1],
                        "campaign": key[2],
                        "field": field,
                        "blocking_for_physical_weight": True,
                        "required_next_action": action,
                    }
                )

        write_tsv(
            output / "exact_generator_card_inventory.tsv",
            exact_card_rows,
            list(exact_card_rows[0]),
        )
        write_tsv(
            output / "card_only_campaign_fingerprints.tsv",
            fingerprint_rows,
            list(fingerprint_rows[0]),
        )
        write_tsv(
            output / "card_only_equivalence_groups.tsv",
            group_rows,
            list(group_rows[0]),
        )
        write_tsv(
            output / "campaign_equivalence_freeze.tsv",
            campaign_freeze_rows,
            list(campaign_freeze_rows[0]),
        )
        write_tsv(
            output / "beam_energy_convention_freeze.tsv",
            beam_rows,
            list(beam_rows[0]),
        )
        write_tsv(
            output / "decay_and_branching_scope_review.tsv",
            decay_rows,
            list(decay_rows[0]),
        )
        write_tsv(
            output / "generator_phase_space_filter_review.tsv",
            filter_rows,
            list(filter_rows[0]),
        )
        write_tsv(
            output / "normalization_denominator_fail_closed.tsv",
            denominator_rows,
            list(denominator_rows[0]),
        )
        write_tsv(
            output / "nonstandard_campaign_blockers.tsv",
            nonstandard_rows,
            list(nonstandard_rows[0]),
        )
        write_tsv(
            output / "qcd_structured_provenance_candidates.tsv",
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
                    "importance_weight_authorized",
                    "stitching_authorized",
                    "status",
                ]
            ),
        )
        write_tsv(
            output / "unresolved_conventions.tsv",
            unresolved_rows,
            list(unresolved_rows[0]),
        )

        authorized_groups = sum(
            row["configuration_equivalence_authorized"] is True
            for row in group_rows
        )
        authorized_campaigns = sum(
            row["configuration_equivalence_authorized"] is True
            for row in campaign_freeze_rows
        )
        beam_authorized = sum(
            row["beam_energy_convention_authorized"] is True
            for row in beam_rows
        )
        process_splits = sum(
            row["status"] == "process_split_across_card_configurations"
            for row in group_rows
        )

        summary = {
            "schema_version": 1,
            "status": (
                "hh4b_physical_normalization_card_only_convention_freeze_pass"
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
                "pn_c2a_fingerprints": str(args.c2a_fingerprints),
                "pn_c2a_fingerprints_sha256": sha256_file(
                    args.c2a_fingerprints
                ),
            },
            "freeze": {
                "eos_campaigns": len(included_keys),
                "standard_card_campaigns": len(standard_keys),
                "nonstandard_campaigns": len(nonstandard_keys),
                "standard_process_labels": len(process_campaigns),
                "exact_generator_card_files": len(exact_card_rows),
                "card_only_equivalence_groups": len(group_rows),
                "configuration_equivalence_groups_authorized": (
                    authorized_groups
                ),
                "configuration_equivalent_campaigns_authorized": (
                    authorized_campaigns
                ),
                "processes_split_across_card_configurations": process_splits,
                "beam_energy_conventions_authorized": beam_authorized,
                "explicit_matrix_element_decay_campaigns": sum(
                    bool(row["matrix_element_decay_commands"])
                    for row in decay_rows
                ),
                "explicit_pythia_decay_campaigns": sum(
                    bool(row["pythia_decay_settings"])
                    for row in decay_rows
                ),
                "qcd_structured_candidate_rows": len(qcd_rows),
                "unresolved_convention_rows": len(unresolved_rows),
            },
            "readiness": {
                "card_only_configuration_freeze_complete": True,
                "beam_energy_authorized_campaigns": beam_authorized,
                "branching_fractions_authorized": 0,
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
                "branching_fraction_values_assigned": 0,
                "filter_efficiency_values_assigned": 0,
                "normalization_denominators_authorized": 0,
                "cross_sections_assigned": 0,
                "physical_yields_calculated": 0,
                "models_trained": 0,
                "thresholds_selected": 0,
            },
            "next_gate": (
                "recover_weight_denominators_and_nonstandard_ggf_qcd_"
                "provenance_before_reference_cross_section_registry"
            ),
        }

        (output / "summary.json").write_text(
            json.dumps(summary, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

        (output / "README.md").write_text(
            "# HH4b card-only configuration and convention freeze\n\n"
            "This checkpoint excludes generator logs from campaign-equivalence "
            "fingerprints and uses only exact process, run, Pythia 8, and "
            "legacy Pythia card content. Run event counts and random seeds, "
            "plus operational Pythia output and seed settings, are removed "
            "from the canonical comparison.\n\n"
            "Configuration equivalence is authorized only within an identical "
            "process label whose campaigns collapse to one card-only "
            "fingerprint. This authorization covers generator configuration "
            "only; it does not authorize sharing a cross section or a "
            "normalization denominator.\n\n"
            "A 13 TeV beam convention is authorized only where both run-card "
            "beam energies and the frozen source path agree. The six "
            "nonstandard ggF HH and hard-QCD campaigns remain fail-closed.\n\n"
            "No branching fraction, filter efficiency, normalization "
            "denominator, QCD stitching rule, external cross section, physical "
            "weight, yield, or model threshold is assigned.\n",
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
        print("ALL_547_PROVENANCE_FILES_REVERIFIED_PASS")
        print("EXACT_188_GENERATOR_CARD_FILES_VERIFIED_PASS")
        print("STANDARD_47_CAMPAIGNS_CARD_ONLY_FINGERPRINTED_PASS")
        print("NONSTANDARD_6_CAMPAIGNS_FAIL_CLOSED_PASS")
        print("GENERATOR_LOGS_EXCLUDED_FROM_EQUIVALENCE_PASS")
        print("CARD_ONLY_CONFIGURATION_EQUIVALENCE_FREEZE_PASS")
        print("BEAM_CONVENTION_FREEZE_FAIL_CLOSED_PASS")
        print("DECAY_AND_FILTER_SCOPE_REVIEW_PASS")
        print("NORMALIZATION_DENOMINATORS_REMAIN_UNAUTHORIZED_PASS")
        print("NO_EXTERNAL_REFERENCE_CROSS_SECTION_ASSIGNED")
        print("NO_CANDIDATE_PARQUET_OPENED")
        print("NO_VALIDATION_CANDIDATE_CONTENT_OPENED")
        print("NO_EVALUATION_CANDIDATE_CONTENT_OPENED")
        print("NO_PHYSICAL_YIELDS_CALCULATED")
        print("HH4B_CARD_ONLY_EQUIVALENCE_AND_CONVENTION_FREEZE_PASS")
    except Exception:
        shutil.rmtree(output, ignore_errors=True)
        raise


if __name__ == "__main__":
    main()
