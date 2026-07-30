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


EXPECTED_LEGACY_MEMBERS = 27
EXPECTED_LEGACY_EVENTS = 270_000
EXPECTED_BUNDLE_MEMBERS = 75
EXPECTED_BUNDLE_EVENTS = 750_000
EXPECTED_UNIQUE_TTBAR_MEMBERS = 102
EXPECTED_UNIQUE_TTBAR_EVENTS = 1_020_000

EXPECTED_SOURCE_INDEX = "326"
EXPECTED_LEGACY_TOKEN = "ttbar_100k_shard003"
EXPECTED_LEGACY_SEED = "105003"
EXPECTED_BUNDLE_TARGET = "172"

EXPECTED_LEGACY_LHE_SHA = (
    "bb58ad5878433b641af8eeb475e826ed"
    "45da0fbec9580aef2cf427a9128c85ff"
)
EXPECTED_BUNDLE_LHE_SHA = (
    "89d93cf2c3bd53aff5b3ec6adaf147d"
    "4ed6b8faa4d0335c386fe6f1173e6cecc"
)
EXPECTED_LEGACY_ROOT_SHA = (
    "127c87510087d164038b7629ea9d5210"
    "008cc4cb1473aa9b069a5cd4b859629e"
)
EXPECTED_BUNDLE_ROOT_SHA = (
    "b439e6b7f97d369ce0c44ce4b210464"
    "bffe70027e4b731539da28e4bd3bdb6c6"
)


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


def as_bool(value: object) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() == "true"


def require_false(row: dict[str, str], field: str, context: str) -> None:
    if as_bool(row[field]):
        raise RuntimeError(
            f"{context}: expected {field}=False"
        )


def campaign_sum(
    rows: list[dict[str, str]],
    sumw_field: str,
    sumw2_field: str,
) -> tuple[str, str]:
    sumw = math.fsum(float(row[sumw_field]) for row in rows)
    sumw2 = math.fsum(float(row[sumw2_field]) for row in rows)
    return format(sumw, ".17g"), format(sumw2, ".17g")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--legacy-members", required=True)
    parser.add_argument("--legacy-campaigns", required=True)
    parser.add_argument("--bundle-members", required=True)
    parser.add_argument("--bundle-campaigns", required=True)
    parser.add_argument("--pn-c5s-summary", required=True)
    parser.add_argument("--pn-c5s-lineage", required=True)
    parser.add_argument("--pn-c5s-overlaps", required=True)
    parser.add_argument("--pn-c5p-root-overlaps", required=True)
    parser.add_argument("--full-partition", required=True)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    legacy_members_path = Path(args.legacy_members)
    legacy_campaigns_path = Path(args.legacy_campaigns)
    bundle_members_path = Path(args.bundle_members)
    bundle_campaigns_path = Path(args.bundle_campaigns)
    c5s_summary_path = Path(args.pn_c5s_summary)
    c5s_lineage_path = Path(args.pn_c5s_lineage)
    c5s_overlaps_path = Path(args.pn_c5s_overlaps)
    c5p_root_overlaps_path = Path(args.pn_c5p_root_overlaps)
    full_partition_path = Path(args.full_partition)
    out = Path(args.out)

    legacy = read_tsv(legacy_members_path)
    legacy_campaigns = read_tsv(legacy_campaigns_path)
    frozen_members = read_tsv(bundle_members_path)
    frozen_campaigns = read_tsv(bundle_campaigns_path)
    c5s_lineage = read_tsv(c5s_lineage_path)
    c5s_overlaps = read_tsv(c5s_overlaps_path)
    c5p_root_overlaps = read_tsv(c5p_root_overlaps_path)
    full_partition = read_tsv(full_partition_path)
    c5s_summary = json.loads(
        c5s_summary_path.read_text(encoding="utf-8")
    )

    if len(legacy) != EXPECTED_LEGACY_MEMBERS:
        raise RuntimeError(
            f"legacy member count={len(legacy)}, expected=27"
        )
    if sum(
        int(row["generated_events_frozen"])
        for row in legacy
    ) != EXPECTED_LEGACY_EVENTS:
        raise RuntimeError("legacy events do not total 270000")

    bundle = [
        row
        for row in frozen_members
        if row["process_or_mode"] == "ttbar_inclusive"
    ]
    if len(bundle) != EXPECTED_BUNDLE_MEMBERS:
        raise RuntimeError(
            f"bundle ttbar count={len(bundle)}, expected=75"
        )
    if sum(
        int(row["generated_events_frozen"])
        for row in bundle
    ) != EXPECTED_BUNDLE_EVENTS:
        raise RuntimeError("bundle ttbar events do not total 750000")

    if c5p_root_overlaps:
        raise RuntimeError(
            "PN-c5p authoritative ROOT overlap table is nonempty"
        )
    if c5s_overlaps:
        raise RuntimeError(
            "PN-c5s v2 exact artifact overlap table is nonempty"
        )

    if c5s_summary["status"] != (
        "exact_ttbar_pair_receipt_recovery_complete"
    ):
        raise RuntimeError("unexpected PN-c5s v2 status")
    if c5s_summary["classification"] not in {
        "exact_lhe_hashes_distinct_independence_candidate",
        "exact_hepmc_hashes_distinct_independence_candidate",
    }:
        raise RuntimeError(
            "PN-c5s v2 did not reach an independence candidate"
        )
    if c5s_summary["pair"]["legacy_source_index"] != (
        EXPECTED_SOURCE_INDEX
    ):
        raise RuntimeError("PN-c5s legacy source mismatch")
    if c5s_summary["pair"]["bundle_target_index"] != (
        EXPECTED_BUNDLE_TARGET
    ):
        raise RuntimeError("PN-c5s bundle target mismatch")
    if c5s_summary["pair"]["legacy_seed"] != (
        EXPECTED_LEGACY_SEED
    ):
        raise RuntimeError("PN-c5s legacy seed mismatch")

    recovered = c5s_summary["recovered_exact_candidates"]

    if recovered["exact_lhe_hash_overlap"]:
        raise RuntimeError("exact LHE overlap is not empty")
    if recovered["exact_hepmc_hash_overlap"]:
        raise RuntimeError("exact HepMC overlap is not empty")
    if recovered["exact_root_hash_overlap"]:
        raise RuntimeError("exact ROOT overlap is not empty")
    if recovered["seed_overlap"]:
        raise RuntimeError("exact member-scoped seed overlap is nonempty")

    legacy_lhe = set(
        recovered["legacy_lhe_hash_candidates"]
    )
    bundle_lhe = set(
        recovered["bundle_lhe_hash_candidates"]
    )
    if EXPECTED_LEGACY_LHE_SHA not in legacy_lhe:
        raise RuntimeError(
            "expected exact legacy LHE checksum absent"
        )
    if EXPECTED_BUNDLE_LHE_SHA not in bundle_lhe:
        raise RuntimeError(
            "expected exact bundle LHE checksum absent"
        )
    if legacy_lhe & bundle_lhe:
        raise RuntimeError(
            "legacy and bundle exact LHE checksum sets overlap"
        )

    lineage_by_side = {
        row["source_side"]: row for row in c5s_lineage
    }
    if set(lineage_by_side) != {"legacy", "bundle"}:
        raise RuntimeError(
            "PN-c5s lineage table must contain legacy and bundle"
        )

    legacy_lineage = lineage_by_side["legacy"]
    bundle_lineage = lineage_by_side["bundle"]

    if EXPECTED_LEGACY_LHE_SHA not in set(
        legacy_lineage["lhe_hash_candidates"].split(",")
    ):
        raise RuntimeError(
            "legacy lineage row missing expected LHE checksum"
        )
    if EXPECTED_BUNDLE_LHE_SHA not in set(
        bundle_lineage["lhe_hash_candidates"].split(",")
    ):
        raise RuntimeError(
            "bundle lineage row missing expected LHE checksum"
        )
    if EXPECTED_LEGACY_ROOT_SHA not in set(
        legacy_lineage["root_hash_candidates"].split(",")
    ):
        raise RuntimeError(
            "legacy lineage row missing expected ROOT checksum"
        )
    if EXPECTED_BUNDLE_ROOT_SHA not in set(
        bundle_lineage["root_hash_candidates"].split(",")
    ):
        raise RuntimeError(
            "bundle lineage row missing expected ROOT checksum"
        )
    if EXPECTED_LEGACY_SEED not in set(
        legacy_lineage["seed_candidates"].split(",")
    ):
        raise RuntimeError(
            "legacy lineage row missing frozen seed"
        )
    if EXPECTED_LEGACY_SEED in set(
        bundle_lineage["seed_candidates"].split(",")
    ):
        raise RuntimeError(
            "bundle exact seed set unexpectedly contains 105003"
        )

    payload = c5s_summary["payload_controls"]
    if (
        payload["bundle_event_payload_members_opened"] != 0
        or payload["legacy_root_payload_files_opened"] != 0
        or payload["candidate_parquet_files_opened"] != 0
        or payload["lhe_event_payload_files_opened"] != 0
        or payload["hepmc_event_payload_files_opened"] != 0
    ):
        raise RuntimeError(
            "PN-c5s v2 payload controls were not preserved"
        )

    # All 27 legacy members remain eligible: there is no alias exclusion.
    legacy_inclusion_rows: list[dict[str, object]] = []
    for row in sorted(
        legacy,
        key=lambda item: int(item["source_index"]),
    ):
        require_false(
            row,
            "normalization_denominator_authorized",
            f"legacy source {row['source_index']}",
        )
        require_false(
            row,
            "legacy_physical_inclusion_authorized",
            f"legacy source {row['source_index']}",
        )
        require_false(
            row,
            "physical_weight_calculated",
            f"legacy source {row['source_index']}",
        )

        legacy_inclusion_rows.append(
            {
                "source_index": row["source_index"],
                "shard_token": row["shard_token"],
                "hold_campaign": row["hold_campaign"],
                "final_split": row["final_split"],
                "generated_events_frozen": row[
                    "generated_events_frozen"
                ],
                "root_sha256": row["root_sha256"],
                "sum_generator_weights_frozen": row[
                    "sum_generator_weights_candidate"
                ],
                "sum_squared_generator_weights_frozen": row[
                    "sum_squared_generator_weights_candidate"
                ],
                "independence_from_bundle_backed_ttbar_proven": True,
                "duplicate_alias_of_bundle_member": False,
                "formal_exclusion": False,
                "included_in_unique_ttbar_population": True,
                "member_denominator_value_frozen": True,
                "member_denominator_use_authorized": True,
                "cross_campaign_process_aggregation_authorized": False,
                "process_reference_cross_section_assigned": False,
                "physical_weight_application_authorized": False,
                "physical_yield_calculation_authorized": False,
            }
        )

    # Build the unique member population without yet claiming that all
    # campaigns share one cross-section convention.
    unique_rows: list[dict[str, object]] = []

    for row in sorted(
        bundle,
        key=lambda item: int(item["target_index"]),
    ):
        unique_rows.append(
            {
                "unique_population_index": len(unique_rows) + 1,
                "source_population": "bundle_backed",
                "source_index": row["target_index"],
                "member_id": row["member_id"],
                "campaign": row["campaign"],
                "final_split": row["final_split"],
                "generated_events_frozen": row[
                    "generated_events_frozen"
                ],
                "root_sha256": row["root_member_sha256"],
                "sum_generator_weights_frozen": row[
                    "sum_generator_weights_candidate"
                ],
                "sum_squared_generator_weights_frozen": row[
                    "sum_squared_generator_weights_candidate"
                ],
                "independent_unique_member": True,
                "included_in_unique_ttbar_population": True,
                "member_denominator_value_frozen": True,
                "member_denominator_use_authorized": True,
                "cross_campaign_process_aggregation_authorized": False,
                "process_reference_cross_section_assigned": False,
                "physical_weight_application_authorized": False,
            }
        )

    for row in legacy_inclusion_rows:
        unique_rows.append(
            {
                "unique_population_index": len(unique_rows) + 1,
                "source_population": "legacy_recovered",
                "source_index": row["source_index"],
                "member_id": row["shard_token"],
                "campaign": row["hold_campaign"],
                "final_split": row["final_split"],
                "generated_events_frozen": row[
                    "generated_events_frozen"
                ],
                "root_sha256": row["root_sha256"],
                "sum_generator_weights_frozen": row[
                    "sum_generator_weights_frozen"
                ],
                "sum_squared_generator_weights_frozen": row[
                    "sum_squared_generator_weights_frozen"
                ],
                "independent_unique_member": True,
                "included_in_unique_ttbar_population": True,
                "member_denominator_value_frozen": True,
                "member_denominator_use_authorized": True,
                "cross_campaign_process_aggregation_authorized": False,
                "process_reference_cross_section_assigned": False,
                "physical_weight_application_authorized": False,
            }
        )

    if len(unique_rows) != EXPECTED_UNIQUE_TTBAR_MEMBERS:
        raise RuntimeError("unique ttbar member count is not 102")
    if sum(
        int(row["generated_events_frozen"])
        for row in unique_rows
    ) != EXPECTED_UNIQUE_TTBAR_EVENTS:
        raise RuntimeError("unique ttbar events do not total 1020000")

    root_counts = Counter(
        row["root_sha256"] for row in unique_rows
    )
    duplicate_roots = {
        sha: count
        for sha, count in root_counts.items()
        if count > 1
    }
    if duplicate_roots:
        raise RuntimeError(
            f"duplicate ROOT checksums in unique ttbar registry: "
            f"{duplicate_roots}"
        )

    # Freeze campaign denominators independently. Cross-campaign merging
    # remains blocked until process-definition and weight conventions are
    # proven compatible.
    bundle_ttbar_campaigns = [
        row
        for row in frozen_campaigns
        if row["process_or_mode"] == "ttbar_inclusive"
    ]
    if sum(
        int(row["members"])
        for row in bundle_ttbar_campaigns
    ) != EXPECTED_BUNDLE_MEMBERS:
        raise RuntimeError(
            "bundle ttbar campaign registry does not total 75"
        )

    campaign_rows: list[dict[str, object]] = []

    for row in sorted(
        bundle_ttbar_campaigns,
        key=lambda item: item["campaign"],
    ):
        if not as_bool(row["campaign_denominator_value_frozen"]):
            raise RuntimeError(
                f"bundle campaign denominator not frozen: "
                f"{row['campaign']}"
            )
        if not as_bool(row["campaign_denominator_use_authorized"]):
            raise RuntimeError(
                f"bundle campaign denominator use blocked: "
                f"{row['campaign']}"
            )

        campaign_rows.append(
            {
                "source_population": "bundle_backed",
                "process_or_mode": "ttbar_inclusive",
                "campaign": row["campaign"],
                "members": row["members"],
                "generated_events": row["generated_events"],
                "sum_generator_weights_frozen": row[
                    "sum_generator_weights_candidate"
                ],
                "sum_squared_generator_weights_frozen": row[
                    "sum_squared_generator_weights_candidate"
                ],
                "negative_weight_events": row[
                    "negative_weight_events"
                ],
                "zero_weight_events": row["zero_weight_events"],
                "positive_weight_events": row[
                    "positive_weight_events"
                ],
                "campaign_denominator_value_frozen": True,
                "campaign_denominator_use_authorized": True,
                "independent_population_inclusion_authorized": True,
                "cross_campaign_process_aggregation_authorized": False,
                "process_reference_cross_section_assigned": False,
                "physical_weight_application_authorized": False,
            }
        )

    for row in sorted(
        legacy_campaigns,
        key=lambda item: item["legacy_campaign"],
    ):
        if not as_bool(
            row["member_denominator_candidates_recovered"]
        ):
            raise RuntimeError(
                f"legacy campaign member denominators incomplete: "
                f"{row['legacy_campaign']}"
            )
        if not as_bool(
            row["campaign_denominator_candidate_aggregated"]
        ):
            raise RuntimeError(
                f"legacy campaign candidate not aggregated: "
                f"{row['legacy_campaign']}"
            )

        campaign_rows.append(
            {
                "source_population": "legacy_recovered",
                "process_or_mode": "ttbar_inclusive",
                "campaign": row["legacy_campaign"],
                "members": row["members"],
                "generated_events": row["generated_events"],
                "sum_generator_weights_frozen": row[
                    "sum_generator_weights_candidate"
                ],
                "sum_squared_generator_weights_frozen": row[
                    "sum_squared_generator_weights_candidate"
                ],
                "negative_weight_events": row[
                    "negative_weight_events"
                ],
                "zero_weight_events": row["zero_weight_events"],
                "positive_weight_events": row[
                    "positive_weight_events"
                ],
                "campaign_denominator_value_frozen": True,
                "campaign_denominator_use_authorized": True,
                "independent_population_inclusion_authorized": True,
                "cross_campaign_process_aggregation_authorized": False,
                "process_reference_cross_section_assigned": False,
                "physical_weight_application_authorized": False,
            }
        )

    if sum(int(row["members"]) for row in campaign_rows) != (
        EXPECTED_UNIQUE_TTBAR_MEMBERS
    ):
        raise RuntimeError(
            "campaign registry does not total 102 ttbar members"
        )
    if sum(
        int(row["generated_events"])
        for row in campaign_rows
    ) != EXPECTED_UNIQUE_TTBAR_EVENTS:
        raise RuntimeError(
            "campaign registry does not total 1020000 events"
        )

    unique_sumw = format(
        math.fsum(
            float(row["sum_generator_weights_frozen"])
            for row in campaign_rows
        ),
        ".17g",
    )
    unique_sumw2 = format(
        math.fsum(
            float(row["sum_squared_generator_weights_frozen"])
            for row in campaign_rows
        ),
        ".17g",
    )

    population_summary_rows = [
        {
            "process_or_mode": "ttbar_inclusive",
            "bundle_members": EXPECTED_BUNDLE_MEMBERS,
            "bundle_generated_events": EXPECTED_BUNDLE_EVENTS,
            "legacy_members_included": EXPECTED_LEGACY_MEMBERS,
            "legacy_generated_events_included": EXPECTED_LEGACY_EVENTS,
            "legacy_members_excluded": 0,
            "unique_members": EXPECTED_UNIQUE_TTBAR_MEMBERS,
            "unique_generated_events": EXPECTED_UNIQUE_TTBAR_EVENTS,
            "campaigns": len(campaign_rows),
            "sum_generator_weights_across_all_campaigns_candidate": (
                unique_sumw
            ),
            "sum_squared_generator_weights_across_all_campaigns_candidate": (
                unique_sumw2
            ),
            "all_members_independent_unique": True,
            "all_member_denominator_values_frozen": True,
            "all_campaign_denominator_values_frozen": True,
            "all_campaign_denominator_use_authorized": True,
            "cross_campaign_process_aggregation_authorized": False,
            "reason_cross_campaign_aggregation_blocked": (
                "process_definition_reference_cross_section_and_"
                "Event.Weight_convention_compatibility_not_yet_frozen"
            ),
            "process_reference_cross_section_assigned": False,
            "physical_weight_application_authorized": False,
            "physical_yield_calculation_authorized": False,
        }
    ]

    # Preserve the complete inventory distinction. The original 5M
    # background manifest remains the accounting target; this gate only
    # resolves the ttbar population.
    background_rows = [
        row
        for row in full_partition
        if row["sample_class"] == "background"
    ]
    signal_rows = [
        row
        for row in full_partition
        if row["sample_class"] == "signal"
    ]
    if sum(
        int(row["generated_events"])
        for row in background_rows
    ) != 5_000_000:
        raise RuntimeError(
            "full partition background total is not 5000000"
        )
    if sum(
        int(row["generated_events"])
        for row in signal_rows
    ) != 200_000:
        raise RuntimeError(
            "full partition signal total is not 200000"
        )

    decision_rows = [
        {
            "decision": "legacy_bundle_ttbar_pair_independence",
            "legacy_source_index": EXPECTED_SOURCE_INDEX,
            "legacy_shard_token": EXPECTED_LEGACY_TOKEN,
            "legacy_seed": EXPECTED_LEGACY_SEED,
            "bundle_target_index": EXPECTED_BUNDLE_TARGET,
            "legacy_lhe_sha256": EXPECTED_LEGACY_LHE_SHA,
            "bundle_lhe_sha256": EXPECTED_BUNDLE_LHE_SHA,
            "legacy_root_sha256": EXPECTED_LEGACY_ROOT_SHA,
            "bundle_root_sha256": EXPECTED_BUNDLE_ROOT_SHA,
            "exact_lhe_hash_overlap": False,
            "exact_hepmc_hash_overlap": False,
            "exact_root_hash_overlap": False,
            "exact_seed_overlap": False,
            "formal_independence_proven": True,
            "legacy_member_excluded": False,
            "all_27_legacy_members_included": True,
            "scientific_basis": (
                "distinct_exact_member_scoped_LHE_checksums_"
                "distinct_seeds_and_distinct_ROOT_checksums"
            ),
        }
    ]

    write_tsv(
        out / "formal_ttbar_independence_decision.tsv",
        decision_rows,
        list(decision_rows[0]),
    )
    write_tsv(
        out / "legacy_ttbar_inclusion_registry.tsv",
        legacy_inclusion_rows,
        list(legacy_inclusion_rows[0]),
    )
    write_tsv(
        out / "unique_ttbar_member_registry.tsv",
        unique_rows,
        list(unique_rows[0]),
    )
    write_tsv(
        out / "frozen_ttbar_campaign_denominator_registry.tsv",
        campaign_rows,
        list(campaign_rows[0]),
    )
    write_tsv(
        out / "unique_ttbar_population_summary.tsv",
        population_summary_rows,
        list(population_summary_rows[0]),
    )

    source_checksums = []
    for role, path in (
        ("legacy_members", legacy_members_path),
        ("legacy_campaigns", legacy_campaigns_path),
        ("bundle_members", bundle_members_path),
        ("bundle_campaigns", bundle_campaigns_path),
        ("pn_c5s_v2_summary", c5s_summary_path),
        ("pn_c5s_v2_lineage", c5s_lineage_path),
        ("pn_c5s_v2_overlaps", c5s_overlaps_path),
        ("pn_c5p_root_overlaps", c5p_root_overlaps_path),
        ("full_partition", full_partition_path),
    ):
        source_checksums.append(
            {
                "artifact_role": role,
                "path": str(path),
                "sha256": sha256_file(path),
            }
        )
    write_tsv(
        out / "source_artifact_checksums.tsv",
        source_checksums,
        list(source_checksums[0]),
    )

    summary = {
        "schema_version": 1,
        "status": (
            "ttbar_independence_inclusion_denominator_freeze_complete"
        ),
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "source_commit": args.source_commit,
        "formal_independence_decision": {
            "legacy_source_326_independent_of_bundle_target_172": True,
            "exact_member_scoped_lhe_hashes_distinct": True,
            "exact_member_scoped_seed_sets_disjoint": True,
            "exact_root_hashes_distinct": True,
            "all_27_legacy_members_independent_and_included": True,
            "legacy_members_excluded": 0,
        },
        "ttbar_inventory": {
            "bundle_members": EXPECTED_BUNDLE_MEMBERS,
            "bundle_generated_events": EXPECTED_BUNDLE_EVENTS,
            "legacy_members": EXPECTED_LEGACY_MEMBERS,
            "legacy_generated_events": EXPECTED_LEGACY_EVENTS,
            "unique_members": EXPECTED_UNIQUE_TTBAR_MEMBERS,
            "unique_generated_events": EXPECTED_UNIQUE_TTBAR_EVENTS,
            "campaigns": len(campaign_rows),
            "all_root_sha256_values_unique": True,
        },
        "denominator_freeze": {
            "legacy_member_denominator_values_frozen": True,
            "legacy_campaign_denominator_values_frozen": True,
            "legacy_campaign_denominator_use_authorized": True,
            "bundle_campaign_denominator_values_remain_frozen": True,
            "bundle_campaign_denominator_use_remains_authorized": True,
            "all_102_member_denominator_values_frozen": True,
            "cross_campaign_ttbar_denominator_candidate_sumw": (
                unique_sumw
            ),
            "cross_campaign_ttbar_denominator_candidate_sumw2": (
                unique_sumw2
            ),
            "cross_campaign_ttbar_aggregation_authorized": False,
        },
        "full_inventory": {
            "background_generated_events": 5_000_000,
            "signal_generated_events": 200_000,
            "ttbar_population_resolved": True,
        },
        "payload_controls": {
            "root_payload_files_opened": 0,
            "candidate_parquet_files_opened": 0,
            "lhe_event_payload_files_opened": 0,
            "hepmc_event_payload_files_opened": 0,
            "archive_payload_files_opened": 0,
            "only_frozen_tsv_json_metadata_opened": True,
        },
        "authorization": {
            "legacy_physical_population_inclusion_authorized": True,
            "legacy_member_denominator_use_authorized": True,
            "legacy_campaign_denominator_use_authorized": True,
            "cross_campaign_ttbar_aggregation_authorized": False,
            "process_reference_cross_section_assignment_authorized": False,
            "physical_weight_application_authorized": False,
            "physical_yield_calculation_authorized": False,
            "complete_5m_background_normalization_authorized": False,
            "complete_200k_signal_normalization_authorized": False,
        },
        "next_gate": (
            "freeze_process_reference_cross_sections_branching_filter_"
            "efficiencies_and_campaign_weight_convention_compatibility_"
            "for_all_ordinary_background_and_signal_processes"
        ),
    }
    (out / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    report = [
        "PN-c5t ttbar independence and denominator freeze",
        "=================================================",
        "",
        "Formal independence decision:",
        "  legacy source 326 vs bundle target 172: independent",
        f"  legacy LHE SHA256: {EXPECTED_LEGACY_LHE_SHA}",
        f"  bundle LHE SHA256: {EXPECTED_BUNDLE_LHE_SHA}",
        "  exact LHE overlap: 0",
        "  exact seed overlap: 0",
        "  exact ROOT overlap: 0",
        "  all 27 legacy members included: True",
        "  legacy members excluded: 0",
        "",
        "Unique ttbar population:",
        f"  bundle members/events: "
        f"{EXPECTED_BUNDLE_MEMBERS}/{EXPECTED_BUNDLE_EVENTS}",
        f"  legacy members/events: "
        f"{EXPECTED_LEGACY_MEMBERS}/{EXPECTED_LEGACY_EVENTS}",
        f"  unique members/events: "
        f"{EXPECTED_UNIQUE_TTBAR_MEMBERS}/"
        f"{EXPECTED_UNIQUE_TTBAR_EVENTS}",
        f"  campaign partitions: {len(campaign_rows)}",
        "",
        "Denominator freeze:",
        "  all legacy member denominators frozen: True",
        "  all legacy campaign denominators frozen: True",
        "  denominator use within each campaign authorized: True",
        f"  all-campaign ttbar sumw candidate: {unique_sumw}",
        f"  all-campaign ttbar sumw2 candidate: {unique_sumw2}",
        "  cross-campaign ttbar aggregation authorized: False",
        "",
        "Why cross-campaign aggregation remains blocked:",
        "  process definition, reference cross section, and",
        "  Event.Weight convention compatibility are not yet frozen.",
        "",
        "Authorization:",
        "  legacy physical population inclusion: True",
        "  process reference cross sections assigned: False",
        "  physical weights calculated: False",
        "  physical yields calculated: False",
        "  complete 5M background normalization authorized: False",
        "  complete 200k signal normalization authorized: False",
        "",
        "NEXT_GATE: freeze_process_reference_cross_sections_branching_filter_efficiencies_and_campaign_weight_convention_compatibility_for_all_ordinary_background_and_signal_processes",
        "",
        "PN_C5T_TTBAR_INDEPENDENCE_INCLUSION_DENOMINATOR_FREEZE_PASS",
    ]
    (out / "report.txt").write_text(
        "\n".join(report) + "\n",
        encoding="utf-8",
    )
    (
        out
        / "PN_C5T_TTBAR_INDEPENDENCE_INCLUSION_DENOMINATOR_FREEZE_PASS"
    ).write_text("", encoding="utf-8")

    print("\n".join(report))


if __name__ == "__main__":
    main()
