#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path


EXPECTED_SOURCE_ROWS = 630
EXPECTED_QCD_MEMBERS = 261
EXPECTED_QCD_BACKGROUND_EVENTS = 2_510_000
EXPECTED_BUNDLE_MEMBERS = 342
EXPECTED_BUNDLE_EVENTS = 2_420_000
EXPECTED_BUNDLE_BACKGROUND_MEMBERS = 232
EXPECTED_BUNDLE_BACKGROUND_EVENTS = 2_220_000
EXPECTED_SIGNAL_MEMBERS = 110
EXPECTED_SIGNAL_EVENTS = 200_000
EXPECTED_LEGACY_TTBAR_MEMBERS = 27
EXPECTED_LEGACY_TTBAR_EVENTS = 270_000
EXPECTED_BACKGROUND_EVENTS = 5_000_000
EXPECTED_CAMPAIGNS = 49
EXPECTED_SPLITS = {"train": 256, "validation": 60, "test": 26}


def read_tsv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def write_tsv(
    path: Path,
    rows: list[dict[str, object]],
    fields: list[str],
) -> None:
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
        writer.writerows(rows)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def parse_bool(value: str) -> bool:
    normalized = value.strip().lower()
    if normalized == "true":
        return True
    if normalized == "false":
        return False
    raise ValueError(f"invalid boolean value: {value!r}")


def finite_float(value: str, label: str) -> float:
    parsed = float(value)
    if not math.isfinite(parsed):
        raise RuntimeError(f"{label} is non-finite: {value}")
    return parsed


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-registry", required=True)
    parser.add_argument("--member-candidates", required=True)
    parser.add_argument("--campaign-candidates", required=True)
    parser.add_argument("--legacy-registry", required=True)
    parser.add_argument("--ggf-transport-registry", required=True)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    source_path = Path(args.source_registry)
    member_path = Path(args.member_candidates)
    campaign_path = Path(args.campaign_candidates)
    legacy_path = Path(args.legacy_registry)
    ggf_transport_path = Path(args.ggf_transport_registry)
    out = Path(args.out)

    source_rows = read_tsv(source_path)
    member_rows = read_tsv(member_path)
    campaign_rows = read_tsv(campaign_path)
    legacy_rows = read_tsv(legacy_path)
    ggf_transport_rows = read_tsv(ggf_transport_path)

    if len(source_rows) != EXPECTED_SOURCE_ROWS:
        raise RuntimeError(
            f"source registry rows={len(source_rows)}, "
            f"expected {EXPECTED_SOURCE_ROWS}"
        )
    if len(member_rows) != EXPECTED_BUNDLE_MEMBERS:
        raise RuntimeError(
            f"bundle member rows={len(member_rows)}, "
            f"expected {EXPECTED_BUNDLE_MEMBERS}"
        )
    if len(campaign_rows) != EXPECTED_CAMPAIGNS:
        raise RuntimeError(
            f"campaign rows={len(campaign_rows)}, "
            f"expected {EXPECTED_CAMPAIGNS}"
        )
    if len(legacy_rows) != EXPECTED_LEGACY_TTBAR_MEMBERS:
        raise RuntimeError(
            f"legacy ttbar rows={len(legacy_rows)}, "
            f"expected {EXPECTED_LEGACY_TTBAR_MEMBERS}"
        )
    if len(ggf_transport_rows) != 100:
        raise RuntimeError(
            f"ggF nested transport rows={len(ggf_transport_rows)}, "
            "expected 100"
        )

    source_by_member = {
        row["member_id"]: row for row in source_rows
    }
    if len(source_by_member) != len(source_rows):
        raise RuntimeError("source registry member IDs are not unique")

    member_by_id = {
        row["member_id"]: row for row in member_rows
    }
    if len(member_by_id) != len(member_rows):
        raise RuntimeError("bundle denominator member IDs are not unique")

    legacy_by_id = {
        row["member_id"]: row for row in legacy_rows
    }
    if len(legacy_by_id) != len(legacy_rows):
        raise RuntimeError("legacy ttbar member IDs are not unique")

    qcd_source = [
        row
        for row in source_rows
        if row["process_or_mode"] == "qcd_hardqcd"
    ]
    non_qcd_source = [
        row
        for row in source_rows
        if row["process_or_mode"] != "qcd_hardqcd"
    ]

    if len(qcd_source) != EXPECTED_QCD_MEMBERS:
        raise RuntimeError(
            f"hard-QCD members={len(qcd_source)}, "
            f"expected {EXPECTED_QCD_MEMBERS}"
        )
    qcd_events = sum(
        int(row["generated_events"]) for row in qcd_source
    )
    if qcd_events != EXPECTED_QCD_BACKGROUND_EVENTS:
        raise RuntimeError(
            f"hard-QCD events={qcd_events}, "
            f"expected {EXPECTED_QCD_BACKGROUND_EVENTS}"
        )

    source_bundle_non_qcd = []
    source_legacy = []

    for row in non_qcd_source:
        locator = row["source_locator"].removeprefix("bundle:")
        bundle_backed = (
            "/bundles/" in locator
            and locator.endswith((".tar.gz", ".tgz", ".tar"))
        )
        if bundle_backed:
            source_bundle_non_qcd.append(row)
        else:
            source_legacy.append(row)

    if len(source_bundle_non_qcd) != EXPECTED_BUNDLE_MEMBERS:
        raise RuntimeError(
            f"source bundle-backed non-QCD members="
            f"{len(source_bundle_non_qcd)}, "
            f"expected {EXPECTED_BUNDLE_MEMBERS}"
        )
    if len(source_legacy) != EXPECTED_LEGACY_TTBAR_MEMBERS:
        raise RuntimeError(
            f"source legacy members={len(source_legacy)}, "
            f"expected {EXPECTED_LEGACY_TTBAR_MEMBERS}"
        )

    if {
        row["member_id"] for row in source_bundle_non_qcd
    } != set(member_by_id):
        missing = sorted(
            {
                row["member_id"]
                for row in source_bundle_non_qcd
            }
            - set(member_by_id)
        )
        extra = sorted(
            set(member_by_id)
            - {
                row["member_id"]
                for row in source_bundle_non_qcd
            }
        )
        raise RuntimeError(
            "bundle denominator/source identity closure failed: "
            f"missing={missing[:5]}, extra={extra[:5]}"
        )

    if {
        row["member_id"] for row in source_legacy
    } != set(legacy_by_id):
        raise RuntimeError(
            "legacy registry/source identity closure failed"
        )

    split_counts = Counter()
    bundle_events = 0
    signal_members = 0
    signal_events = 0
    background_members = 0
    background_events = 0
    signed_members = 0
    uniform_members = 0

    frozen_member_rows = []

    for row in member_rows:
        source = source_by_member[row["member_id"]]

        for member_key, source_key in (
            ("sample_class", "sample_class"),
            ("process_or_mode", "process_or_mode"),
            ("member_id", "member_id"),
            ("final_split", "final_split"),
            ("generated_events_frozen", "generated_events"),
            ("candidate_rows_metadata", "candidate_rows_metadata"),
        ):
            if row[member_key] != source[source_key]:
                raise RuntimeError(
                    f"member/source mismatch for {member_key}: "
                    f"{row[member_key]!r} != {source[source_key]!r}"
                )

        generated = int(row["generated_events_frozen"])
        entries = int(row["root_tree_entries"])
        weight_values = int(row["event_weight_values"])
        negative = int(row["negative_weight_events"])
        zero = int(row["zero_weight_events"])
        positive = int(row["positive_weight_events"])

        if entries != generated or weight_values != generated:
            raise RuntimeError(
                f"event-count closure failed for {row['member_id']}"
            )
        if negative + zero + positive != generated:
            raise RuntimeError(
                f"weight-sign closure failed for {row['member_id']}"
            )

        sumw = finite_float(
            row["sum_generator_weights_candidate"],
            f"{row['member_id']} sumw",
        )
        sumw2 = finite_float(
            row["sum_squared_generator_weights_candidate"],
            f"{row['member_id']} sumw2",
        )
        minimum = finite_float(
            row["min_generator_weight_candidate"],
            f"{row['member_id']} min weight",
        )
        maximum = finite_float(
            row["max_generator_weight_candidate"],
            f"{row['member_id']} max weight",
        )

        if sumw2 < 0.0:
            raise RuntimeError(
                f"negative sumw2 for {row['member_id']}"
            )
        if minimum > maximum:
            raise RuntimeError(
                f"weight range reversed for {row['member_id']}"
            )
        if parse_bool(
            row["normalization_denominator_authorized"]
        ):
            raise RuntimeError(
                "source step unexpectedly authorized a denominator"
            )
        if parse_bool(row["reference_cross_section_assigned"]):
            raise RuntimeError(
                "source step unexpectedly assigned a cross section"
            )
        if parse_bool(row["physical_weight_calculated"]):
            raise RuntimeError(
                "source step unexpectedly calculated physical weights"
            )

        split_counts[row["final_split"]] += 1
        bundle_events += generated

        if row["sample_class"] == "signal":
            signal_members += 1
            signal_events += generated
        elif row["sample_class"] == "background":
            background_members += 1
            background_events += generated
        else:
            raise RuntimeError(
                f"unknown sample class: {row['sample_class']}"
            )

        signed_members += negative > 0
        uniform_members += (
            row["uniform_event_weights"].lower() == "true"
        )

        frozen = dict(row)
        frozen.update(
            {
                "denominator_source_branch": "Event/Event.Weight",
                "denominator_value_status": (
                    "frozen_exact_full_member_signed_sumw"
                ),
                "denominator_value_frozen": True,
                "denominator_use_authorized_within_campaign": True,
                "cross_campaign_process_aggregation_authorized": False,
                "process_reference_cross_section_assigned": False,
                "physical_weight_application_authorized": False,
                "physical_yield_calculation_authorized": False,
            }
        )
        frozen_member_rows.append(frozen)

    if dict(split_counts) != EXPECTED_SPLITS:
        raise RuntimeError(
            f"bundle split counts={dict(split_counts)}, "
            f"expected {EXPECTED_SPLITS}"
        )
    if bundle_events != EXPECTED_BUNDLE_EVENTS:
        raise RuntimeError(
            f"bundle generated events={bundle_events}, "
            f"expected {EXPECTED_BUNDLE_EVENTS}"
        )
    if signal_members != EXPECTED_SIGNAL_MEMBERS:
        raise RuntimeError(
            f"signal members={signal_members}, "
            f"expected {EXPECTED_SIGNAL_MEMBERS}"
        )
    if signal_events != EXPECTED_SIGNAL_EVENTS:
        raise RuntimeError(
            f"signal events={signal_events}, "
            f"expected {EXPECTED_SIGNAL_EVENTS}"
        )
    if background_members != EXPECTED_BUNDLE_BACKGROUND_MEMBERS:
        raise RuntimeError(
            f"bundle background members={background_members}, "
            f"expected {EXPECTED_BUNDLE_BACKGROUND_MEMBERS}"
        )
    if background_events != EXPECTED_BUNDLE_BACKGROUND_EVENTS:
        raise RuntimeError(
            f"bundle background events={background_events}, "
            f"expected {EXPECTED_BUNDLE_BACKGROUND_EVENTS}"
        )

    members_by_campaign: dict[
        tuple[str, str, str],
        list[dict[str, str]],
    ] = defaultdict(list)
    for row in member_rows:
        members_by_campaign[
            (
                row["sample_class"],
                row["process_or_mode"],
                row["campaign"],
            )
        ].append(row)

    if len(members_by_campaign) != EXPECTED_CAMPAIGNS:
        raise RuntimeError(
            f"member-derived campaigns={len(members_by_campaign)}, "
            f"expected {EXPECTED_CAMPAIGNS}"
        )

    frozen_campaign_rows = []

    for row in campaign_rows:
        key = (
            row["sample_class"],
            row["process_or_mode"],
            row["campaign"],
        )
        members = members_by_campaign.get(key)
        if members is None:
            raise RuntimeError(
                f"campaign absent from member registry: {key}"
            )

        expected_members = len(members)
        expected_events = sum(
            int(member["generated_events_frozen"])
            for member in members
        )
        expected_sumw = math.fsum(
            float(member["sum_generator_weights_candidate"])
            for member in members
        )
        expected_sumw2 = math.fsum(
            float(
                member[
                    "sum_squared_generator_weights_candidate"
                ]
            )
            for member in members
        )

        if int(row["members"]) != expected_members:
            raise RuntimeError(
                f"campaign member count mismatch for {key}"
            )
        if int(row["generated_events"]) != expected_events:
            raise RuntimeError(
                f"campaign event count mismatch for {key}"
            )
        if not math.isclose(
            float(row["sum_generator_weights_candidate"]),
            expected_sumw,
            rel_tol=1e-14,
            abs_tol=1e-14,
        ):
            raise RuntimeError(
                f"campaign sumw mismatch for {key}"
            )
        if not math.isclose(
            float(
                row[
                    "sum_squared_generator_weights_candidate"
                ]
            ),
            expected_sumw2,
            rel_tol=1e-14,
            abs_tol=1e-14,
        ):
            raise RuntimeError(
                f"campaign sumw2 mismatch for {key}"
            )

        if row["all_member_denominators_recovered"].lower() != "true":
            raise RuntimeError(
                f"campaign denominator incomplete for {key}"
            )
        if parse_bool(
            row["normalization_denominator_authorized"]
        ):
            raise RuntimeError(
                f"source campaign unexpectedly authorized for {key}"
            )
        if parse_bool(row["reference_cross_section_assigned"]):
            raise RuntimeError(
                f"source campaign cross section assigned for {key}"
            )
        if parse_bool(row["physics_normalization_ready"]):
            raise RuntimeError(
                f"source campaign physics-ready for {key}"
            )

        frozen = dict(row)
        frozen.update(
            {
                "campaign_denominator_status": (
                    "frozen_exact_sum_of_all_member_Event.Weight"
                ),
                "campaign_denominator_value_frozen": True,
                "campaign_denominator_use_authorized": True,
                "cross_campaign_process_aggregation_authorized": False,
                "process_reference_cross_section_assigned": False,
                "physical_weight_application_authorized": False,
                "physical_yield_calculation_authorized": False,
            }
        )
        frozen_campaign_rows.append(frozen)

    campaign_keys = {
        (
            row["sample_class"],
            row["process_or_mode"],
            row["campaign"],
        )
        for row in campaign_rows
    }
    if campaign_keys != set(members_by_campaign):
        raise RuntimeError("campaign/member key closure failed")

    legacy_events = sum(
        int(row["generated_events_frozen"]) for row in legacy_rows
    )
    if legacy_events != EXPECTED_LEGACY_TTBAR_EVENTS:
        raise RuntimeError(
            f"legacy ttbar events={legacy_events}, "
            f"expected {EXPECTED_LEGACY_TTBAR_EVENTS}"
        )
    if {
        row["process_or_mode"] for row in legacy_rows
    } != {"ttbar_inclusive"}:
        raise RuntimeError(
            "legacy rows are not exclusively ttbar_inclusive"
        )
    if {
        row["source_kind"] for row in legacy_rows
    } != {"local_candidate_parquet"}:
        raise RuntimeError(
            "legacy rows are not exclusively local candidate Parquet"
        )
    if any(
        parse_bool(row["physical_normalization_authorized"])
        for row in legacy_rows
    ):
        raise RuntimeError(
            "legacy source unexpectedly authorized for normalization"
        )

    bundle_ttbar = [
        row
        for row in member_rows
        if row["process_or_mode"] == "ttbar_inclusive"
    ]
    bundle_ttbar_campaigns = defaultdict(list)
    for row in bundle_ttbar:
        bundle_ttbar_campaigns[row["campaign"]].append(row)

    bundle_ttbar_summary = []
    for campaign, rows in sorted(bundle_ttbar_campaigns.items()):
        bundle_ttbar_summary.append(
            {
                "process_or_mode": "ttbar_inclusive",
                "campaign": campaign,
                "members": len(rows),
                "generated_events": sum(
                    int(row["generated_events_frozen"])
                    for row in rows
                ),
                "sum_generator_weights_candidate": format(
                    math.fsum(
                        float(
                            row[
                                "sum_generator_weights_candidate"
                            ]
                        )
                        for row in rows
                    ),
                    ".17g",
                ),
                "campaign_denominator_frozen": True,
                "relationship_to_legacy_parquet": (
                    "unresolved_no_exact_source_identity_link"
                ),
                "cross_population_aggregation_authorized": False,
            }
        )

    source_candidate_paths = {
        row["member_id"]: row.get("candidate_path", "")
        for row in source_rows
    }
    bundle_ttbar_paths = {
        source_candidate_paths[row["member_id"]]
        for row in bundle_ttbar
        if source_candidate_paths[row["member_id"]]
    }
    legacy_paths = {
        row["candidate_path_recorded_only"]
        for row in legacy_rows
        if row["candidate_path_recorded_only"]
    }
    exact_path_overlap = sorted(
        bundle_ttbar_paths & legacy_paths
    )

    if set(member_by_id) & set(legacy_by_id):
        raise RuntimeError(
            "legacy and bundle-backed member IDs overlap"
        )

    legacy_frozen_rows = []
    for row in legacy_rows:
        frozen = dict(row)
        frozen.update(
            {
                "physical_role_status": (
                    "blocked_pending_original_generator_source_"
                    "recovery_or_formal_exclusion"
                ),
                "relationship_to_bundle_backed_ttbar": (
                    "unresolved_no_exact_source_identity_link"
                ),
                "denominator_value_frozen": False,
                "campaign_denominator_use_authorized": False,
                "cross_population_aggregation_authorized": False,
                "process_reference_cross_section_assigned": False,
                "physical_weight_application_authorized": False,
                "included_in_complete_5m_physical_normalization": False,
            }
        )
        legacy_frozen_rows.append(frozen)

    full_background_events = (
        qcd_events + background_events + legacy_events
    )
    if full_background_events != EXPECTED_BACKGROUND_EVENTS:
        raise RuntimeError(
            f"full background accounting={full_background_events}, "
            f"expected {EXPECTED_BACKGROUND_EVENTS}"
        )

    partition_rows = [
        {
            "population": "hard_qcd_background",
            "sample_class": "background",
            "members": len(qcd_source),
            "generated_events": qcd_events,
            "denominator_status": (
                "frozen_in_prior_qcd_denominator_checkpoint"
            ),
            "physical_role_status": (
                "secondary_direct_qcd_closure_projection"
            ),
            "included_in_complete_event_inventory": True,
            "physical_weight_application_authorized": False,
        },
        {
            "population": "bundle_backed_non_qcd_background",
            "sample_class": "background",
            "members": background_members,
            "generated_events": background_events,
            "denominator_status": (
                "frozen_exact_campaign_denominators_this_checkpoint"
            ),
            "physical_role_status": (
                "ordinary_mc_pending_cross_section_registry"
            ),
            "included_in_complete_event_inventory": True,
            "physical_weight_application_authorized": False,
        },
        {
            "population": "legacy_ttbar_parquet_background",
            "sample_class": "background",
            "members": len(legacy_rows),
            "generated_events": legacy_events,
            "denominator_status": "unresolved",
            "physical_role_status": (
                "blocked_pending_recovery_or_formal_exclusion"
            ),
            "included_in_complete_event_inventory": True,
            "physical_weight_application_authorized": False,
        },
        {
            "population": "bundle_backed_signal",
            "sample_class": "signal",
            "members": signal_members,
            "generated_events": signal_events,
            "denominator_status": (
                "frozen_exact_campaign_denominators_this_checkpoint"
            ),
            "physical_role_status": (
                "ordinary_mc_pending_cross_section_registry"
            ),
            "included_in_complete_event_inventory": True,
            "physical_weight_application_authorized": False,
        },
    ]

    source_checksums = [
        {
            "artifact_role": "full_source_registry",
            "path": str(source_path),
            "sha256": sha256_file(source_path),
        },
        {
            "artifact_role": "bundle_member_denominator_candidates",
            "path": str(member_path),
            "sha256": sha256_file(member_path),
        },
        {
            "artifact_role": "bundle_campaign_denominator_candidates",
            "path": str(campaign_path),
            "sha256": sha256_file(campaign_path),
        },
        {
            "artifact_role": "legacy_ttbar_registry",
            "path": str(legacy_path),
            "sha256": sha256_file(legacy_path),
        },
        {
            "artifact_role": "ggf_nested_transport_registry",
            "path": str(ggf_transport_path),
            "sha256": sha256_file(ggf_transport_path),
        },
    ]

    write_tsv(
        out / "frozen_member_denominator_registry.tsv",
        sorted(
            frozen_member_rows,
            key=lambda row: int(row["target_index"]),
        ),
        list(frozen_member_rows[0]),
    )
    write_tsv(
        out / "frozen_campaign_denominator_registry.tsv",
        sorted(
            frozen_campaign_rows,
            key=lambda row: (
                row["sample_class"],
                row["process_or_mode"],
                row["campaign"],
            ),
        ),
        list(frozen_campaign_rows[0]),
    )
    write_tsv(
        out / "legacy_ttbar_physical_role_registry.tsv",
        sorted(
            legacy_frozen_rows,
            key=lambda row: int(row["source_index"]),
        ),
        list(legacy_frozen_rows[0]),
    )
    write_tsv(
        out / "bundle_backed_ttbar_campaign_summary.tsv",
        bundle_ttbar_summary,
        (
            list(bundle_ttbar_summary[0])
            if bundle_ttbar_summary
            else [
                "process_or_mode",
                "campaign",
                "members",
                "generated_events",
                "sum_generator_weights_candidate",
                "campaign_denominator_frozen",
                "relationship_to_legacy_parquet",
                "cross_population_aggregation_authorized",
            ]
        ),
    )
    write_tsv(
        out / "full_sample_partition.tsv",
        partition_rows,
        list(partition_rows[0]),
    )
    write_tsv(
        out / "source_artifact_checksums.tsv",
        source_checksums,
        list(source_checksums[0]),
    )

    legacy_audit = {
        "schema_version": 1,
        "legacy_process": "ttbar_inclusive",
        "legacy_members": len(legacy_rows),
        "legacy_generated_events": legacy_events,
        "legacy_split_counts": dict(
            sorted(
                Counter(
                    row["final_split"] for row in legacy_rows
                ).items()
            )
        ),
        "legacy_candidate_rows_metadata": sum(
            int(row["candidate_rows_metadata"])
            for row in legacy_rows
        ),
        "bundle_backed_ttbar_members": len(bundle_ttbar),
        "bundle_backed_ttbar_generated_events": sum(
            int(row["generated_events_frozen"])
            for row in bundle_ttbar
        ),
        "bundle_backed_ttbar_campaigns": sorted(
            bundle_ttbar_campaigns
        ),
        "exact_candidate_path_overlap_count": len(
            exact_path_overlap
        ),
        "exact_candidate_path_overlap": exact_path_overlap,
        "member_id_overlap_count": 0,
        "original_generation_source_resolved": False,
        "relationship_to_bundle_backed_ttbar_resolved": False,
        "formal_exclusion_authorized": False,
        "physical_normalization_inclusion_authorized": False,
        "required_resolution": (
            "recover_original_generator_or_Delphes_source_and_"
            "Event.Weight_denominator_or_formally_exclude_with_"
            "documented_physics_impact"
        ),
    }
    (out / "legacy_ttbar_role_audit.json").write_text(
        json.dumps(legacy_audit, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    summary = {
        "schema_version": 1,
        "status": (
            "full_bundle_backed_denominator_checkpoint_frozen"
        ),
        "timestamp_utc": datetime.now(
            timezone.utc
        ).isoformat(),
        "source_commit": args.source_commit,
        "full_event_inventory": {
            "background_generated_events": full_background_events,
            "signal_generated_events": signal_events,
            "total_generated_events": (
                full_background_events + signal_events
            ),
        },
        "hard_qcd_background": {
            "members": len(qcd_source),
            "generated_events": qcd_events,
            "denominator_checkpoint_preexisting": True,
        },
        "bundle_backed_non_qcd": {
            "members": len(member_rows),
            "generated_events": bundle_events,
            "campaigns": len(campaign_rows),
            "split_counts": dict(sorted(split_counts.items())),
            "background_members": background_members,
            "background_generated_events": background_events,
            "signal_members": signal_members,
            "signal_generated_events": signal_events,
            "signed_weight_members": signed_members,
            "uniform_weight_members": uniform_members,
            "member_denominator_values_frozen": True,
            "campaign_denominator_values_frozen": True,
            "cross_campaign_process_aggregation_authorized": False,
        },
        "legacy_ttbar": legacy_audit,
        "authorization": {
            "bundle_campaign_denominator_use_authorized": True,
            "legacy_ttbar_denominator_use_authorized": False,
            "cross_campaign_process_aggregation_authorized": False,
            "process_reference_cross_section_assignment_authorized": False,
            "physical_weight_application_authorized": False,
            "physical_yield_calculation_authorized": False,
            "complete_5m_background_physical_normalization_authorized": False,
            "complete_200k_signal_physical_normalization_authorized": False,
        },
        "next_gate": (
            "targeted_recover_or_formally_exclude_legacy_ttbar_"
            "before_process_reference_cross_section_registry"
        ),
    }
    (out / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    report = [
        "HH4b full bundle-backed denominator freeze",
        "==========================================",
        "",
        "Full generated-event inventory:",
        f"  background: {full_background_events}",
        f"  signal: {signal_events}",
        f"  total: {full_background_events + signal_events}",
        "",
        "Hard-QCD background:",
        f"  members: {len(qcd_source)}",
        f"  generated events: {qcd_events}",
        "  denominator checkpoint: frozen previously",
        "",
        "Bundle-backed non-hard-QCD:",
        f"  members: {len(member_rows)}",
        f"  campaigns: {len(campaign_rows)}",
        f"  generated events: {bundle_events}",
        f"  train / validation / test: "
        f"{split_counts['train']} / "
        f"{split_counts['validation']} / "
        f"{split_counts['test']}",
        f"  background members/events: "
        f"{background_members} / {background_events}",
        f"  signal members/events: "
        f"{signal_members} / {signal_events}",
        f"  signed-weight members: {signed_members}",
        f"  uniform-weight members: {uniform_members}",
        "  member denominator values frozen: True",
        "  campaign denominator values frozen: True",
        "",
        "Legacy ttbar/Parquet:",
        f"  members: {len(legacy_rows)}",
        f"  generated events: {legacy_events}",
        f"  bundle-backed ttbar members: {len(bundle_ttbar)}",
        f"  bundle-backed ttbar campaigns: "
        f"{len(bundle_ttbar_campaigns)}",
        f"  exact candidate-path overlaps: "
        f"{len(exact_path_overlap)}",
        "  original generation source resolved: False",
        "  physical role resolved: False",
        "  physical normalization inclusion authorized: False",
        "",
        "Authorization:",
        "  bundle campaign denominator use: authorized",
        "  cross-campaign process aggregation: not authorized",
        "  process reference cross sections: not assigned",
        "  physical weights: not calculated",
        "  physical yields: not calculated",
        "  complete 5M-background normalization: not authorized",
        "  complete 200k-signal normalization: not authorized",
        "",
        "NEXT_GATE: targeted_recover_or_formally_exclude_legacy_ttbar_before_process_reference_cross_section_registry",
        "",
        "FULL_BUNDLE_BACKED_DENOMINATOR_FREEZE_PASS",
    ]
    (out / "report.txt").write_text(
        "\n".join(report) + "\n",
        encoding="utf-8",
    )
    (
        out / "FULL_BUNDLE_BACKED_DENOMINATOR_FREEZE_PASS"
    ).write_text(
        "",
        encoding="utf-8",
    )

    print("\n".join(report))


if __name__ == "__main__":
    main()
