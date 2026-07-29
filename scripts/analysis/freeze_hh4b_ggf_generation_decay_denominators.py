#!/usr/bin/env python3
"""Freeze ggF HH generation, decay, and equal-event denominator conventions."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import shutil
import tarfile
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable


GGF_CAMPAIGNS = {
    "ggf_hh4b_ml_ext20k_frozen_v2_20260716": {
        "members": 20,
        "generated_events": 20000,
        "expected_shards": set(range(0, 20)),
    },
    "ggf_hh4b_ml_ext80k_frozen_v2_20260721_v1": {
        "members": 80,
        "generated_events": 80000,
        "expected_shards": set(range(20, 100)),
    },
}

EXPECTED_PAYLOAD_SHA256 = (
    "93dc90edd396c5a61d502a0f349fdabcf1f5f167f4553ffd410c0adf818f7923"
)
EXPECTED_PAYLOAD_SIZE_BYTES = 10368061

PROC_MEMBER = "payload/mg5_template/Cards/proc_card_mg5.dat"
RUN_MEMBER = "payload/mg5_template/Cards/run_card.dat"
PARAM_MEMBER = "payload/mg5_template/Cards/param_card.dat"
VERSION_MEMBER = "payload/mg5_template/MGMEVersion.txt"
WORKER_MEMBER = (
    "payload/repo/scripts/delphes/run_ggf_hh4b_shard_transfer.sh"
)

TAG_RE = re.compile(
    r"HH4b_ggf_(?P<campaign>.+)_shard(?P<shard>\d+)_seed(?P<seed>\d+)$"
)


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


def active_card_lines(text: str) -> list[str]:
    lines: list[str] = []
    for raw in text.splitlines():
        line = raw.split("#", 1)[0].strip()
        if line:
            lines.append(" ".join(line.split()))
    return lines


def parse_assignment(text: str, key: str) -> str:
    pattern = re.compile(
        rf"^\s*(?P<value>[^#!=]+?)\s*=\s*{re.escape(key)}\b",
        flags=re.IGNORECASE | re.MULTILINE,
    )
    match = pattern.search(text)
    if match is None:
        return ""
    return match.group("value").strip().strip("'\"")


def parse_tag(tag: str) -> tuple[str, int, int]:
    match = TAG_RE.fullmatch(tag)
    if match is None:
        raise RuntimeError(f"could not parse ggF source tag: {tag!r}")
    return (
        match.group("campaign"),
        int(match.group("shard")),
        int(match.group("seed")),
    )


def extract_text_members(
    payload: Path,
) -> tuple[dict[str, str], list[dict[str, object]]]:
    required = (
        PROC_MEMBER,
        RUN_MEMBER,
        PARAM_MEMBER,
        VERSION_MEMBER,
        WORKER_MEMBER,
    )
    texts: dict[str, str] = {}
    manifest: list[dict[str, object]] = []

    with tarfile.open(payload, mode="r:gz") as archive:
        members = {
            member.name: member
            for member in archive.getmembers()
        }
        for name in required:
            if name not in members:
                raise RuntimeError(f"payload member absent: {name}")
            member = members[name]
            if not member.isfile():
                raise RuntimeError(f"payload member is not a file: {name}")
            handle = archive.extractfile(member)
            if handle is None:
                raise RuntimeError(f"could not read payload member: {name}")
            data = handle.read()
            if b"\x00" in data:
                raise RuntimeError(f"selected payload member is binary: {name}")
            text = data.decode("utf-8")
            texts[name] = text
            manifest.append(
                {
                    "payload_member": name,
                    "size_bytes": len(data),
                    "sha256": sha256_bytes(data),
                    "status": "selected_text_configuration_member_verified",
                }
            )

    return texts, manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--campaign-inventory", type=Path, required=True)
    parser.add_argument("--nonstandard-ledger", type=Path, required=True)
    parser.add_argument("--structured-metadata", type=Path, required=True)
    parser.add_argument("--c4a-status", type=Path, required=True)
    parser.add_argument("--payload", type=Path, required=True)
    parser.add_argument("--c4c-text", type=Path, required=True)
    parser.add_argument("--c4c-json", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--source-commit", required=True)
    args = parser.parse_args()

    if args.payload.stat().st_size != EXPECTED_PAYLOAD_SIZE_BYTES:
        raise RuntimeError("ggF payload size mismatch")
    if sha256_file(args.payload) != EXPECTED_PAYLOAD_SHA256:
        raise RuntimeError("ggF payload SHA-256 mismatch")

    c4c_text = args.c4c_text.read_text(encoding="utf-8")
    c4c_json = json.loads(args.c4c_json.read_text(encoding="utf-8"))

    required_report_tokens = (
        "representative_payload_sha_verified_for_both_campaigns: True",
        "submit_descriptions_referencing_exact_payload: 4",
        "import model /uscms_data/d3/iturkmen/hh4b_delphes/mg5_models/heft",
        "generate p p > h h, (h > b b~), (h > b b~)",
        "beam_energy_GeV_runtime: 13000",
        "cut_decays_runtime: False",
        "PN_C4C_GGF_PAYLOAD_CONFIGURATION_AUDIT_PASS",
    )
    for token in required_report_tokens:
        if token not in c4c_text:
            raise RuntimeError(
                f"required PN-c4c report evidence absent: {token}"
            )

    inventory_rows = [
        row
        for row in read_tsv(args.campaign_inventory)
        if row["process_or_mode"] == "ggf_hh4b"
    ]
    if len(inventory_rows) != 2:
        raise RuntimeError("expected two ggF campaign inventory rows")

    inventory_map = {
        row["campaign"]: row
        for row in inventory_rows
    }
    if set(inventory_map) != set(GGF_CAMPAIGNS):
        raise RuntimeError("ggF campaign inventory keys changed")

    status_rows = [
        row
        for row in read_tsv(args.c4a_status)
        if row["process_or_mode"] == "ggf_hh4b"
    ]
    if len(status_rows) != 2:
        raise RuntimeError("expected two ggF PN-c4a status rows")
    for row in status_rows:
        if row["denominator_metadata_coverage_complete"] != "True":
            raise RuntimeError(
                f"incomplete ggF metadata coverage: {row['campaign']}"
            )

    ledger_rows = [
        row
        for row in read_tsv(args.nonstandard_ledger)
        if row["process_or_mode"] == "ggf_hh4b"
    ]
    if len(ledger_rows) != 100:
        raise RuntimeError(
            f"ggF EOS ledger rows={len(ledger_rows)}, expected 100"
        )

    ledger_by_campaign: dict[str, list[dict[str, str]]] = defaultdict(list)
    ledger_tags: set[str] = set()
    for row in ledger_rows:
        campaign, shard, seed = parse_tag(row["source_tag"])
        if campaign != row["campaign"]:
            raise RuntimeError(
                f"source-tag campaign mismatch: {row['source_tag']}"
            )
        if row["structured_metadata_rows"] == "0":
            raise RuntimeError(
                f"ggF bundle lacks structured metadata: {row['source_tag']}"
            )
        if row["source_tag"] in ledger_tags:
            raise RuntimeError(
                f"duplicate ggF source tag: {row['source_tag']}"
            )
        ledger_tags.add(row["source_tag"])
        row = {
            **row,
            "parsed_shard": str(shard),
            "parsed_seed": str(seed),
        }
        ledger_by_campaign[campaign].append(row)

    structured_rows = [
        row
        for row in read_tsv(args.structured_metadata)
        if row["process_or_mode"] == "ggf_hh4b"
        and row["source_tag"] in ledger_tags
        and row["field"].split(".")[-1] in {
            "n_events",
            "generated_events",
        }
    ]

    n_events_by_tag: dict[str, set[int]] = defaultdict(set)
    for row in structured_rows:
        try:
            value = int(float(row["value"]))
        except ValueError as error:
            raise RuntimeError(
                f"noninteger ggF n_events value: {row}"
            ) from error
        n_events_by_tag[row["source_tag"]].add(value)

    missing_tags = sorted(ledger_tags - set(n_events_by_tag))
    if missing_tags:
        raise RuntimeError(
            "ggF source tags missing n_events evidence:\n"
            + "\n".join(missing_tags)
        )

    conflicting = {
        tag: values
        for tag, values in n_events_by_tag.items()
        if values != {1000}
    }
    if conflicting:
        raise RuntimeError(
            f"ggF source tags have conflicting event counts: {conflicting}"
        )

    payload_texts, payload_manifest = extract_text_members(args.payload)
    proc_text = payload_texts[PROC_MEMBER]
    run_text = payload_texts[RUN_MEMBER]
    version_text = payload_texts[VERSION_MEMBER].strip()
    worker_text = payload_texts[WORKER_MEMBER]

    proc_lines = active_card_lines(proc_text)
    model_lines = [
        line for line in proc_lines
        if line.lower().startswith("import model ")
    ]
    generate_lines = [
        line for line in proc_lines
        if line.lower().startswith("generate ")
    ]
    add_process_lines = [
        line for line in proc_lines
        if line.lower().startswith("add process ")
    ]

    expected_model = (
        "import model /uscms_data/d3/iturkmen/hh4b_delphes/mg5_models/heft"
    )
    expected_process = (
        "generate p p > h h, (h > b b~), (h > b b~)"
    )

    if model_lines != [expected_model]:
        raise RuntimeError(f"unexpected ggF model lines: {model_lines}")
    if generate_lines != [expected_process]:
        raise RuntimeError(
            f"unexpected ggF process lines: {generate_lines}"
        )
    if add_process_lines:
        raise RuntimeError(
            f"unexpected additional ggF processes: {add_process_lines}"
        )
    if "[qcd]" in expected_process.lower():
        raise RuntimeError("NLO process syntax is not authorized")
    if expected_process.count("(h > b b~)") != 2:
        raise RuntimeError("expected exactly two forced H->bb decays")

    if "6500.0 = ebeam1" not in worker_text:
        raise RuntimeError("13 TeV ebeam1 runtime override absent")
    if "6500.0 = ebeam2" not in worker_text:
        raise RuntimeError("13 TeV ebeam2 runtime override absent")
    if "False = cut_decays" not in worker_text:
        raise RuntimeError("cut_decays=False runtime override absent")
    if 'Number of unweighted events requested' not in worker_text:
        raise RuntimeError("unweighted-event request evidence absent")
    if './bin/generate_events "$RUN_NAME" -f' not in worker_text:
        raise RuntimeError("MG5 generate_events execution evidence absent")

    event_norm = parse_assignment(run_text, "event_norm")
    ebeam1_card = parse_assignment(run_text, "ebeam1")
    ebeam2_card = parse_assignment(run_text, "ebeam2")
    cut_decays_card = parse_assignment(run_text, "cut_decays")

    campaign_freeze_rows: list[dict[str, object]] = []
    shard_rows: list[dict[str, object]] = []

    for campaign, expected in GGF_CAMPAIGNS.items():
        inventory = inventory_map[campaign]
        rows = ledger_by_campaign[campaign]

        if int(inventory["members"]) != expected["members"]:
            raise RuntimeError(f"member count mismatch for {campaign}")
        if int(inventory["generated_events"]) != expected["generated_events"]:
            raise RuntimeError(f"event count mismatch for {campaign}")
        if len(rows) != expected["members"]:
            raise RuntimeError(f"ledger member count mismatch for {campaign}")

        observed_shards = {
            int(row["parsed_shard"])
            for row in rows
        }
        if observed_shards != expected["expected_shards"]:
            raise RuntimeError(
                f"shard coverage mismatch for {campaign}: "
                f"{sorted(observed_shards)}"
            )

        observed_seeds = {
            int(row["parsed_seed"])
            for row in rows
        }
        expected_seeds = {
            86000 + shard
            for shard in expected["expected_shards"]
        }
        if observed_seeds != expected_seeds:
            raise RuntimeError(f"seed coverage mismatch for {campaign}")

        total_n_events = sum(
            next(iter(n_events_by_tag[row["source_tag"]]))
            for row in rows
        )
        if total_n_events != expected["generated_events"]:
            raise RuntimeError(
                f"structured n_events sum mismatch for {campaign}"
            )

        campaign_freeze_rows.append(
            {
                "sample_class": "signal",
                "process_or_mode": "ggf_hh4b",
                "campaign": campaign,
                "members": expected["members"],
                "generated_events": expected["generated_events"],
                "per_shard_events": 1000,
                "generator_weight_convention": (
                    "LO_MG5_unweighted_equal_probability_events"
                ),
                "effective_generator_numerator": 1,
                "normalization_denominator_type": (
                    "full_campaign_generated_event_count"
                ),
                "normalization_denominator": expected["generated_events"],
                "normalization_denominator_authorized": True,
                "denominator_scope": (
                    "full_campaign_across_train_validation_and_evaluation_shards"
                ),
                "split_combination_policy": (
                    "do_not_renormalize_each_split_independently"
                ),
                "physical_weight_application_authorized": False,
                "status": "ggf_campaign_equal_event_denominator_frozen",
            }
        )

        for row in sorted(
            rows,
            key=lambda item: int(item["parsed_shard"]),
        ):
            shard_rows.append(
                {
                    "campaign": campaign,
                    "source_tag": row["source_tag"],
                    "shard": int(row["parsed_shard"]),
                    "seed": int(row["parsed_seed"]),
                    "n_events": 1000,
                    "remote_bundle_path": row["remote_bundle_path"],
                    "structured_metadata_proven": True,
                    "payload_sha256": EXPECTED_PAYLOAD_SHA256,
                    "payload_equivalence_scope": (
                        "four_submit_descriptions_plus_one_representative_"
                        "provenance_per_campaign"
                    ),
                    "status": "ggf_shard_denominator_evidence_frozen",
                }
            )

    convention_rows = [
        {
            "process_or_mode": "ggf_hh4b",
            "beam_energy_GeV": 13000,
            "generator_framework": "MadGraph5_aMC_at_LO_generate_events",
            "generator_version_text": version_text,
            "model": "HEFT",
            "process_definition": "p p > h h",
            "resonance_scope": "nonresonant_no_explicit_s_channel_resonance",
            "decay_definition": "(h > b b~) twice in process card",
            "generated_final_state": "HH_to_bbbb",
            "reference_cross_section_scope_required": "inclusive_pp_to_HH",
            "branching_fraction_convention": (
                "multiply_external_BR_H_to_bb_squared_exactly_once"
            ),
            "branching_fraction_power": 2,
            "numerical_branching_fraction_authorized": False,
            "additional_generator_filter": "none_identified",
            "generator_filter_efficiency_symbolic": 1,
            "kinematic_modeling_scope": (
                "LO_HEFT_ML_training_approximation_not_precision_ggF_modeling"
            ),
            "payload_sha256": EXPECTED_PAYLOAD_SHA256,
            "payload_equivalence_evidence": (
                "same_exact_payload_in_four_submit_descriptions_and_"
                "representative_provenance_for_both_campaigns"
            ),
            "per_shard_payload_sha_coverage_available": False,
            "event_norm_run_card": event_norm,
            "ebeam1_run_card": ebeam1_card,
            "ebeam2_run_card": ebeam2_card,
            "cut_decays_run_card": cut_decays_card,
            "runtime_ebeam1_GeV": 6500,
            "runtime_ebeam2_GeV": 6500,
            "runtime_cut_decays": False,
            "generation_decay_convention_authorized": True,
            "external_cross_section_authorized": False,
            "physical_weight_application_authorized": False,
            "status": "ggf_generation_and_decay_convention_frozen",
        }
    ]

    controls = {
        "root_files_opened": 0,
        "hepmc_files_opened": 0,
        "lhe_files_opened": 0,
        "parquet_files_opened": 0,
        "candidate_files_opened": 0,
        "validation_candidate_files_opened": 0,
        "evaluation_candidate_files_opened": 0,
        "numerical_branching_fractions_assigned": 0,
        "external_cross_sections_assigned": 0,
        "physical_weights_calculated": 0,
        "physical_yields_calculated": 0,
        "models_trained": 0,
        "thresholds_selected": 0,
    }

    args.output.mkdir(parents=True, exist_ok=True)

    write_tsv(
        args.output / "ggf_generation_decay_convention_freeze.tsv",
        convention_rows,
        list(convention_rows[0]),
    )
    write_tsv(
        args.output / "ggf_campaign_denominator_freeze.tsv",
        campaign_freeze_rows,
        list(campaign_freeze_rows[0]),
    )
    write_tsv(
        args.output / "ggf_shard_denominator_evidence.tsv",
        shard_rows,
        list(shard_rows[0]),
    )
    write_tsv(
        args.output / "ggf_payload_selected_member_manifest.tsv",
        payload_manifest,
        list(payload_manifest[0]),
    )

    shutil.copy2(
        args.c4c_text,
        args.output / "pn_c4c_ggf_payload_configuration_audit.txt",
    )
    shutil.copy2(
        args.c4c_json,
        args.output / "pn_c4c_ggf_payload_configuration_audit.json",
    )

    summary = {
        "schema_version": 1,
        "status": "hh4b_ggf_generation_decay_denominator_freeze_pass",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "source_commit": args.source_commit,
        "ggf": {
            "campaigns": 2,
            "source_bundles": 100,
            "generated_events": 100000,
            "campaign_denominators_authorized": 2,
            "generation_decay_convention_authorized": True,
            "branching_fraction_convention_frozen_symbolically": True,
            "numerical_branching_fraction_authorized": False,
            "external_cross_section_authorized": False,
            "physical_weight_application_authorized": False,
            "modeling_scope": (
                "LO_HEFT_ML_training_approximation_not_precision_ggF_modeling"
            ),
        },
        "readiness": {
            "ggf_denominator_gate_complete": True,
            "ggf_generation_decay_gate_complete": True,
            "qcd_importance_denominator_gate_complete": False,
            "qcd_stitching_authorized": False,
            "reference_cross_section_registry_complete": False,
            "physics_normalization_ready": False,
        },
        "controls": controls,
        "next_gate": (
            "freeze_qcd_importance_denominators_and_exclusive_stitching_policy"
        ),
    }

    (args.output / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    (args.output / "README.md").write_text(
        "# HH4b ggF generation, decay, and denominator freeze\n\n"
        "This checkpoint freezes the two nonstandard ggF HH campaigns as "
        "13 TeV LO HEFT samples generated with the process "
        "`p p > h h, (h > b b~), (h > b b~)`. The production payload is "
        "identical across the four submit descriptions and is matched by "
        "one checksum-frozen representative provenance JSON from each "
        "campaign.\n\n"
        "The samples contain 100 shards of 1000 unweighted events. Their "
        "effective generator numerator is one and their full-campaign "
        "denominators are 20,000 and 80,000. Split-specific independent "
        "renormalization is forbidden.\n\n"
        "Because the process card forces both Higgs bosons to `bb`, a future "
        "registry based on an inclusive `pp -> HH` reference cross section "
        "must multiply `BR(H -> bb)^2` exactly once. No numerical branching "
        "fraction or reference cross section is assigned here.\n\n"
        "The generated kinematics are explicitly classified as an LO HEFT "
        "ML-training approximation, not precision ggF HH modeling. No ROOT, "
        "HepMC, LHE, Parquet, candidate, validation, or evaluation payload "
        "is opened.\n",
        encoding="utf-8",
    )

    checksum_path = args.output / "SHA256SUMS"
    products = sorted(
        path
        for path in args.output.rglob("*")
        if path.is_file() and path != checksum_path
    )
    checksum_path.write_text(
        "\n".join(
            f"{sha256_file(path)}  {path.relative_to(args.output)}"
            for path in products
        )
        + "\n",
        encoding="utf-8",
    )

    print(json.dumps(summary, indent=2, sort_keys=True))
    print()
    print("BOTH_GGF_CAMPAIGNS_PAYLOAD_CONFIGURATION_FROZEN_PASS")
    print("ALL_100_GGF_SHARD_IDENTITIES_AND_EVENT_COUNTS_PASS")
    print("GGF_LO_UNWEIGHTED_EQUAL_EVENT_CONVENTION_PASS")
    print("TWO_GGF_FULL_CAMPAIGN_DENOMINATORS_FROZEN_PASS")
    print("GGF_FORCED_HBB_DECAY_SCOPE_FROZEN_PASS")
    print("EXTERNAL_BR_HBB_SQUARED_CONVENTION_FROZEN_PASS")
    print("GGF_HEFT_APPROXIMATION_SCOPE_RECORDED_PASS")
    print("NO_ROOT_HEPMC_LHE_PARQUET_OR_CANDIDATE_PAYLOAD_OPENED")
    print("NO_NUMERICAL_BRANCHING_FRACTION_ASSIGNED")
    print("NO_EXTERNAL_REFERENCE_CROSS_SECTION_ASSIGNED")
    print("NO_PHYSICAL_WEIGHTS_OR_YIELDS_CALCULATED")
    print("HH4B_GGF_GENERATION_DECAY_DENOMINATOR_FREEZE_PASS")


if __name__ == "__main__":
    main()
