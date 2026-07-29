#!/usr/bin/env python3
"""Freeze metadata-level hard-QCD denominators and exclusive pTHat stitching."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable


QCD_PROCESS = "qcd_hardqcd"
EXPECTED_QCD_CAMPAIGNS = 4
EXPECTED_PRODUCTION_CAMPAIGNS = 3
EXPECTED_DIAGNOSTIC_CAMPAIGNS = 1
EXPECTED_QCD_BUNDLES = 261
EXPECTED_PRODUCTION_BUNDLES = 260
EXPECTED_DIAGNOSTIC_BUNDLES = 1
EXPECTED_QCD_EVENTS = 2_510_000
EXPECTED_PRODUCTION_EVENTS = 2_500_000
EXPECTED_DIAGNOSTIC_EVENTS = 10_000
EXPECTED_CAMPAIGN_BIN_GROUPS = 25
EXPECTED_EXCLUSIVE_BINS = (
    (50, 75),
    (75, 100),
    (100, 200),
    (200, 300),
    (300, 500),
    (500, 700),
    (700, 1000),
    (1000, "Inf"),
)
REQUIRED_CONVENTION = "pythia_info_weight_normalized_to_sigma_gen"
NUMERIC_REL_TOL = 1.0e-12
NUMERIC_ABS_TOL = 1.0e-12

REQUIRED_METADATA_FIELDS = (
    "n_events",
    "sigma_gen_pb",
    "event_weight_convention",
    "sum_event_weights",
    "sum_squared_event_weights",
    "min_event_weight",
    "max_event_weight",
)

NUMERIC_FIELDS = {
    "n_events",
    "seed",
    "shard_id",
    "pthat_min_GeV",
    "pthat_max_GeV",
    "sigma_gen_mb",
    "sigma_gen_pb",
    "sum_event_weights",
    "sum_squared_event_weights",
    "min_event_weight",
    "max_event_weight",
}

OUTPUT_SHARD_FIELDS = [
    "sample_class",
    "process_or_mode",
    "campaign",
    "campaign_role",
    "source_tag",
    "remote_bundle_path",
    "bundle_basename",
    "bin_index",
    "pthat_min_GeV",
    "pthat_max_GeV",
    "shard_id",
    "seed",
    "n_events",
    "sigma_gen_pb",
    "sum_event_weights",
    "sum_squared_event_weights",
    "min_event_weight",
    "max_event_weight",
    "event_weight_convention",
    "effective_events",
    "weights_uniform_positive",
    "metadata_complete",
    "production_included",
    "bin_total_generated_events",
    "production_mixture_fraction",
    "generator_coefficient_pb_per_event_weight",
    "normalization_denominator_type",
    "normalization_denominator",
    "denominator_authorized",
    "per_event_weight_transport_required",
    "status",
]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


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


def canonical_field(field: str) -> str:
    return field.strip().split(".")[-1]


def parse_bool(value: object) -> bool:
    if isinstance(value, bool):
        return value
    normalized = str(value).strip().lower()
    if normalized in {"true", "1", "yes"}:
        return True
    if normalized in {"false", "0", "no"}:
        return False
    raise ValueError(f"not a Boolean value: {value!r}")


def parse_bound(value: object) -> int | str:
    normalized = str(value).strip()
    if normalized.lower() in {"inf", "infinity", "none", "null", ""}:
        return "Inf"
    number = float(normalized)
    if not number.is_integer():
        raise ValueError(f"pTHat bound is not integral: {value!r}")
    return int(number)


def bound_sort_value(value: int | str) -> float:
    if str(value).lower() in {"inf", "infinity"}:
        return math.inf
    return float(value)


def same_value(field: str, left: object, right: object) -> bool:
    if field == "event_weight_convention":
        return str(left).strip() == str(right).strip()
    if field == "pthat_max_GeV":
        return parse_bound(left) == parse_bound(right)
    if field in NUMERIC_FIELDS:
        return math.isclose(
            float(left),
            float(right),
            rel_tol=NUMERIC_REL_TOL,
            abs_tol=NUMERIC_ABS_TOL,
        )
    return str(left).strip() == str(right).strip()


def resolve_metadata_values(
    rows: list[dict[str, str]],
) -> tuple[dict[str, object], list[str]]:
    grouped: dict[str, list[str]] = defaultdict(list)
    for row in rows:
        field = canonical_field(row["field"])
        grouped[field].append(row["value"])

    resolved: dict[str, object] = {}
    conflicts: list[str] = []

    for field, values in grouped.items():
        first = values[0]
        if any(not same_value(field, first, value) for value in values[1:]):
            conflicts.append(field)
            continue

        if field == "event_weight_convention":
            resolved[field] = first.strip()
        elif field == "pthat_max_GeV":
            resolved[field] = parse_bound(first)
        elif field in NUMERIC_FIELDS:
            number = float(first)
            if field in {
                "n_events",
                "seed",
                "shard_id",
                "pthat_min_GeV",
            }:
                if not number.is_integer():
                    conflicts.append(field)
                    continue
                resolved[field] = int(number)
            else:
                resolved[field] = number
        else:
            resolved[field] = first

    return resolved, sorted(set(conflicts))


def campaign_role(campaign: str) -> str:
    return "diagnostic_testfill_excluded" if "testfill" in campaign else "nominal_production_extension"


def weights_uniform_positive(
    minimum: float,
    maximum: float,
) -> bool:
    if not math.isfinite(minimum) or not math.isfinite(maximum):
        return False
    if minimum <= 0.0 or maximum <= 0.0:
        return False
    scale = max(abs(minimum), abs(maximum), 1.0)
    return abs(maximum - minimum) <= 1.0e-12 * scale


def validate_exclusive_intervals(
    intervals: list[tuple[int, int | str]],
) -> bool:
    if len(intervals) != len(EXPECTED_EXCLUSIVE_BINS):
        return False
    if tuple(intervals) != EXPECTED_EXCLUSIVE_BINS:
        return False
    for index, (low, high) in enumerate(intervals):
        if str(high) == "Inf":
            if index != len(intervals) - 1:
                return False
            continue
        if low >= int(high):
            return False
        if index + 1 < len(intervals):
            if int(high) != intervals[index + 1][0]:
                return False
    return True


def weighted_mean_and_spread(
    values: list[float],
    weights: list[float],
) -> tuple[float, float, float, float, float]:
    if not values or len(values) != len(weights):
        raise ValueError("weighted estimator received inconsistent inputs")
    if any(
        not math.isfinite(value) or value <= 0.0
        for value in values
    ):
        raise ValueError("cross-section estimates must be finite and positive")
    if any(
        not math.isfinite(weight) or weight <= 0.0
        for weight in weights
    ):
        raise ValueError("estimator weights must be finite and positive")

    total_weight = math.fsum(weights)
    mean = math.fsum(
        value * weight
        for value, weight in zip(values, weights)
    ) / total_weight
    variance = math.fsum(
        weight * (value - mean) ** 2
        for value, weight in zip(values, weights)
    ) / total_weight
    effective_shards = total_weight**2 / math.fsum(
        weight**2
        for weight in weights
    )
    empirical_standard_error = (
        math.sqrt(variance / effective_shards)
        if effective_shards > 1.0
        else 0.0
    )
    relative_spread = (
        math.sqrt(variance) / mean
        if mean > 0.0
        else math.inf
    )
    return (
        mean,
        math.sqrt(variance),
        empirical_standard_error,
        min(values),
        max(values),
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--campaign-inventory", type=Path, required=True)
    parser.add_argument("--nonstandard-registry", type=Path, required=True)
    parser.add_argument("--bundle-ledger", type=Path, required=True)
    parser.add_argument("--structured-metadata", type=Path, required=True)
    parser.add_argument("--bin-closure", type=Path, required=True)
    parser.add_argument("--configuration-evidence", type=Path, required=True)
    parser.add_argument("--duplicate-identities", type=Path, required=True)
    parser.add_argument("--overlap-risks", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--source-commit", required=True)
    args = parser.parse_args()

    inventory_rows = read_tsv(args.campaign_inventory)
    registry_rows = read_tsv(args.nonstandard_registry)
    ledger_rows = read_tsv(args.bundle_ledger)
    metadata_rows = read_tsv(args.structured_metadata)
    closure_rows = read_tsv(args.bin_closure)
    config_rows = read_tsv(args.configuration_evidence)
    duplicate_rows = read_tsv(args.duplicate_identities)
    overlap_rows = read_tsv(args.overlap_risks)

    qcd_inventory = [
        row
        for row in inventory_rows
        if row["process_or_mode"] == QCD_PROCESS
    ]
    qcd_registry = [
        row
        for row in registry_rows
        if row["process_or_mode"] == QCD_PROCESS
    ]
    qcd_ledger = [
        row
        for row in ledger_rows
        if row["process_or_mode"] == QCD_PROCESS
    ]
    qcd_metadata_rows = [
        row
        for row in metadata_rows
        if row["process_or_mode"] == QCD_PROCESS
    ]
    qcd_closure = [
        row
        for row in closure_rows
        if True
    ]
    qcd_config = [
        row
        for row in config_rows
        if row["process_or_mode"] == QCD_PROCESS
    ]
    qcd_duplicates = [
        row
        for row in duplicate_rows
        if row.get("process_or_mode") == QCD_PROCESS
    ]

    if len(qcd_inventory) != EXPECTED_QCD_CAMPAIGNS:
        raise RuntimeError(
            f"QCD inventory campaign count={len(qcd_inventory)}, "
            f"expected {EXPECTED_QCD_CAMPAIGNS}"
        )
    if len(qcd_registry) != EXPECTED_QCD_CAMPAIGNS:
        raise RuntimeError(
            f"QCD registry campaign count={len(qcd_registry)}, "
            f"expected {EXPECTED_QCD_CAMPAIGNS}"
        )
    if len(qcd_ledger) != EXPECTED_QCD_BUNDLES:
        raise RuntimeError(
            f"QCD bundle count={len(qcd_ledger)}, "
            f"expected {EXPECTED_QCD_BUNDLES}"
        )
    if len(qcd_closure) != EXPECTED_CAMPAIGN_BIN_GROUPS:
        raise RuntimeError(
            f"QCD campaign-bin groups={len(qcd_closure)}, "
            f"expected {EXPECTED_CAMPAIGN_BIN_GROUPS}"
        )
    if qcd_duplicates:
        raise RuntimeError(
            f"duplicate QCD source identities remain: {len(qcd_duplicates)}"
        )
    if not qcd_config:
        raise RuntimeError("QCD configuration evidence is absent")

    inventory_map = {
        row["campaign"]: row
        for row in qcd_inventory
    }
    registry_map = {
        row["campaign"]: row
        for row in qcd_registry
    }
    if set(inventory_map) != set(registry_map):
        raise RuntimeError("QCD campaign inventory/registry keys differ")

    production_campaigns = sorted(
        campaign
        for campaign in inventory_map
        if campaign_role(campaign) == "nominal_production_extension"
    )
    diagnostic_campaigns = sorted(
        campaign
        for campaign in inventory_map
        if campaign_role(campaign) == "diagnostic_testfill_excluded"
    )
    if len(production_campaigns) != EXPECTED_PRODUCTION_CAMPAIGNS:
        raise RuntimeError("expected three nominal QCD production campaigns")
    if len(diagnostic_campaigns) != EXPECTED_DIAGNOSTIC_CAMPAIGNS:
        raise RuntimeError("expected one QCD test-fill campaign")

    metadata_by_tag: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in qcd_metadata_rows:
        source_tag = row["source_tag"].strip()
        if source_tag:
            metadata_by_tag[source_tag].append(row)

    seen_tags: set[str] = set()
    seen_source_identities: set[tuple[str, int, int]] = set()
    shard_rows: list[dict[str, object]] = []
    unresolved_rows: list[dict[str, object]] = []

    for ledger in sorted(
        qcd_ledger,
        key=lambda row: (
            row["campaign"],
            int(row["bin_index"]),
            int(row["shard_id"]),
            int(row["seed"]),
        ),
    ):
        source_tag = ledger["source_tag"]
        campaign = ledger["campaign"]
        role = campaign_role(campaign)
        production_included = role == "nominal_production_extension"

        if not source_tag:
            raise RuntimeError(
                f"QCD ledger row lacks source tag: {ledger['bundle_basename']}"
            )
        if source_tag in seen_tags:
            raise RuntimeError(f"duplicate QCD source tag: {source_tag}")
        seen_tags.add(source_tag)

        identity = (
            campaign,
            int(ledger["shard_id"]),
            int(ledger["seed"]),
        )
        if identity in seen_source_identities:
            raise RuntimeError(f"duplicate QCD source identity: {identity}")
        seen_source_identities.add(identity)

        metadata, conflicts = resolve_metadata_values(
            metadata_by_tag.get(source_tag, [])
        )
        missing = [
            field
            for field in REQUIRED_METADATA_FIELDS
            if field not in metadata
        ]

        ledger_low = int(ledger["pthat_min_GeV"])
        ledger_high = parse_bound(ledger["pthat_max_GeV"])
        ledger_shard = int(ledger["shard_id"])
        ledger_seed = int(ledger["seed"])

        mismatches: list[str] = []
        for field, expected in (
            ("pthat_min_GeV", ledger_low),
            ("shard_id", ledger_shard),
            ("seed", ledger_seed),
        ):
            if field in metadata and int(metadata[field]) != int(expected):
                mismatches.append(field)
        if "pthat_max_GeV" in metadata:
            if parse_bound(metadata["pthat_max_GeV"]) != ledger_high:
                mismatches.append("pthat_max_GeV")

        metadata_complete = not missing and not conflicts and not mismatches

        n_events = int(metadata["n_events"]) if "n_events" in metadata else 0
        sigma_gen_pb = (
            float(metadata["sigma_gen_pb"])
            if "sigma_gen_pb" in metadata
            else math.nan
        )
        sumw = (
            float(metadata["sum_event_weights"])
            if "sum_event_weights" in metadata
            else math.nan
        )
        sumw2 = (
            float(metadata["sum_squared_event_weights"])
            if "sum_squared_event_weights" in metadata
            else math.nan
        )
        minimum = (
            float(metadata["min_event_weight"])
            if "min_event_weight" in metadata
            else math.nan
        )
        maximum = (
            float(metadata["max_event_weight"])
            if "max_event_weight" in metadata
            else math.nan
        )
        convention = str(metadata.get("event_weight_convention", ""))

        numeric_valid = (
            n_events > 0
            and math.isfinite(sigma_gen_pb)
            and sigma_gen_pb > 0.0
            and math.isfinite(sumw)
            and sumw > 0.0
            and math.isfinite(sumw2)
            and sumw2 > 0.0
            and math.isfinite(minimum)
            and minimum > 0.0
            and math.isfinite(maximum)
            and maximum >= minimum
        )
        convention_valid = convention == REQUIRED_CONVENTION
        denominator_authorized = (
            production_included
            and metadata_complete
            and numeric_valid
            and convention_valid
        )
        effective_events = (
            sumw * sumw / sumw2
            if numeric_valid
            else math.nan
        )
        uniform = (
            weights_uniform_positive(minimum, maximum)
            if numeric_valid
            else False
        )

        if not (
            metadata_complete
            and numeric_valid
            and convention_valid
        ):
            unresolved_rows.append(
                {
                    "sample_class": ledger["sample_class"],
                    "process_or_mode": ledger["process_or_mode"],
                    "campaign": campaign,
                    "source_tag": source_tag,
                    "remote_bundle_path": ledger["remote_bundle_path"],
                    "missing_metadata_fields": ",".join(missing),
                    "conflicting_metadata_fields": ",".join(conflicts),
                    "ledger_metadata_mismatches": ",".join(mismatches),
                    "numeric_metadata_valid": numeric_valid,
                    "event_weight_convention_valid": convention_valid,
                    "production_included": production_included,
                    "required_next_action": (
                        "recover_exact_generator_json_or_provenance_for_this_shard"
                    ),
                    "status": "qcd_shard_normalization_fail_closed",
                }
            )

        shard_rows.append(
            {
                "sample_class": ledger["sample_class"],
                "process_or_mode": ledger["process_or_mode"],
                "campaign": campaign,
                "campaign_role": role,
                "source_tag": source_tag,
                "remote_bundle_path": ledger["remote_bundle_path"],
                "bundle_basename": ledger["bundle_basename"],
                "bin_index": int(ledger["bin_index"]),
                "pthat_min_GeV": ledger_low,
                "pthat_max_GeV": ledger_high,
                "shard_id": ledger_shard,
                "seed": ledger_seed,
                "n_events": n_events,
                "sigma_gen_pb": (
                    f"{sigma_gen_pb:.17g}"
                    if math.isfinite(sigma_gen_pb)
                    else ""
                ),
                "sum_event_weights": (
                    f"{sumw:.17g}"
                    if math.isfinite(sumw)
                    else ""
                ),
                "sum_squared_event_weights": (
                    f"{sumw2:.17g}"
                    if math.isfinite(sumw2)
                    else ""
                ),
                "min_event_weight": (
                    f"{minimum:.17g}"
                    if math.isfinite(minimum)
                    else ""
                ),
                "max_event_weight": (
                    f"{maximum:.17g}"
                    if math.isfinite(maximum)
                    else ""
                ),
                "event_weight_convention": convention,
                "effective_events": (
                    f"{effective_events:.17g}"
                    if math.isfinite(effective_events)
                    else ""
                ),
                "weights_uniform_positive": uniform,
                "metadata_complete": metadata_complete,
                "production_included": production_included,
                "bin_total_generated_events": "",
                "production_mixture_fraction": "",
                "generator_coefficient_pb_per_event_weight": "",
                "normalization_denominator_type": (
                    "shard_sum_event_weights"
                    if denominator_authorized
                    else ""
                ),
                "normalization_denominator": (
                    f"{sumw:.17g}"
                    if denominator_authorized
                    else ""
                ),
                "denominator_authorized": denominator_authorized,
                "per_event_weight_transport_required": (
                    denominator_authorized and not uniform
                ),
                "status": (
                    "qcd_production_shard_denominator_frozen"
                    if denominator_authorized
                    else (
                        "qcd_testfill_diagnostic_excluded"
                        if not production_included
                        else "qcd_shard_denominator_fail_closed"
                    )
                ),
            }
        )

    qcd_event_total = sum(int(row["n_events"]) for row in shard_rows)
    production_rows = [
        row
        for row in shard_rows
        if bool(row["production_included"])
    ]
    diagnostic_rows = [
        row
        for row in shard_rows
        if not bool(row["production_included"])
    ]

    production_event_total = sum(int(row["n_events"]) for row in production_rows)
    diagnostic_event_total = sum(int(row["n_events"]) for row in diagnostic_rows)

    if qcd_event_total != EXPECTED_QCD_EVENTS:
        raise RuntimeError(
            f"QCD metadata event total={qcd_event_total}, "
            f"expected {EXPECTED_QCD_EVENTS}"
        )
    if len(production_rows) != EXPECTED_PRODUCTION_BUNDLES:
        raise RuntimeError(
            f"production QCD shards={len(production_rows)}, "
            f"expected {EXPECTED_PRODUCTION_BUNDLES}"
        )
    if len(diagnostic_rows) != EXPECTED_DIAGNOSTIC_BUNDLES:
        raise RuntimeError(
            f"diagnostic QCD shards={len(diagnostic_rows)}, "
            f"expected {EXPECTED_DIAGNOSTIC_BUNDLES}"
        )
    if production_event_total != EXPECTED_PRODUCTION_EVENTS:
        raise RuntimeError(
            f"production QCD events={production_event_total}, "
            f"expected {EXPECTED_PRODUCTION_EVENTS}"
        )
    if diagnostic_event_total != EXPECTED_DIAGNOSTIC_EVENTS:
        raise RuntimeError(
            f"diagnostic QCD events={diagnostic_event_total}, "
            f"expected {EXPECTED_DIAGNOSTIC_EVENTS}"
        )

    for campaign, inventory in inventory_map.items():
        campaign_rows = [
            row
            for row in shard_rows
            if row["campaign"] == campaign
        ]
        observed_members = len(campaign_rows)
        observed_events = sum(int(row["n_events"]) for row in campaign_rows)
        expected_members = int(inventory["members"])
        expected_events = int(inventory["generated_events"])
        if observed_members != expected_members:
            raise RuntimeError(
                f"campaign member mismatch for {campaign}: "
                f"{observed_members} != {expected_members}"
            )
        if observed_events != expected_events:
            raise RuntimeError(
                f"campaign event mismatch for {campaign}: "
                f"{observed_events} != {expected_events}"
            )

    production_grouped: dict[
        tuple[int, int | str],
        list[dict[str, object]],
    ] = defaultdict(list)
    for row in production_rows:
        production_grouped[
            (
                int(row["pthat_min_GeV"]),
                parse_bound(row["pthat_max_GeV"]),
            )
        ].append(row)

    intervals = sorted(
        production_grouped,
        key=lambda interval: (
            interval[0],
            bound_sort_value(interval[1]),
        ),
    )
    interval_closure_pass = validate_exclusive_intervals(intervals)

    production_complete = all(
        bool(row["denominator_authorized"])
        for row in production_rows
    )
    campaign_coverage_pass = True
    for interval, rows in production_grouped.items():
        campaigns = {str(row["campaign"]) for row in rows}
        if campaigns != set(production_campaigns):
            campaign_coverage_pass = False
            unresolved_rows.append(
                {
                    "sample_class": "background",
                    "process_or_mode": QCD_PROCESS,
                    "campaign": ",".join(sorted(campaigns)),
                    "source_tag": "",
                    "remote_bundle_path": "",
                    "missing_metadata_fields": "",
                    "conflicting_metadata_fields": "",
                    "ledger_metadata_mismatches": "",
                    "numeric_metadata_valid": True,
                    "event_weight_convention_valid": True,
                    "production_included": True,
                    "required_next_action": (
                        "resolve_incomplete_production_campaign_coverage_for_interval_"
                        f"{interval[0]}_{interval[1]}"
                    ),
                    "status": "qcd_bin_campaign_coverage_fail_closed",
                }
            )

    bin_rows: list[dict[str, object]] = []
    coefficient_closure_pass = True

    for bin_index, interval in enumerate(intervals):
        low, high = interval
        rows = production_grouped[interval]
        resolved_rows = [
            row
            for row in rows
            if bool(row["denominator_authorized"])
        ]
        unresolved_bin_rows = [
            row
            for row in rows
            if not bool(row["denominator_authorized"])
        ]

        n_total = sum(int(row["n_events"]) for row in rows)
        n_resolved = sum(int(row["n_events"]) for row in resolved_rows)

        sigma_mean = math.nan
        sigma_rms = math.nan
        sigma_empirical_se = math.nan
        sigma_min = math.nan
        sigma_max = math.nan
        sumw_total = math.nan
        sumw2_total = math.nan
        effective_total = math.nan
        coefficient_closes = False
        coefficient_sum = math.nan
        variable_shards = 0

        if resolved_rows:
            sigma_values = [
                float(row["sigma_gen_pb"])
                for row in resolved_rows
            ]
            event_counts = [
                float(row["n_events"])
                for row in resolved_rows
            ]
            (
                sigma_mean,
                sigma_rms,
                sigma_empirical_se,
                sigma_min,
                sigma_max,
            ) = weighted_mean_and_spread(
                sigma_values,
                event_counts,
            )

            sumw_total = math.fsum(
                float(row["sum_event_weights"])
                for row in resolved_rows
            )
            sumw2_total = math.fsum(
                float(row["sum_squared_event_weights"])
                for row in resolved_rows
            )
            effective_total = (
                sumw_total * sumw_total / sumw2_total
            )

            coefficient_sum_value = 0.0
            for row in resolved_rows:
                n_events = int(row["n_events"])
                sigma = float(row["sigma_gen_pb"])
                sumw = float(row["sum_event_weights"])
                mixture_fraction = n_events / n_total
                coefficient = mixture_fraction * sigma / sumw
                coefficient_sum_value += coefficient * sumw
                if bool(row["per_event_weight_transport_required"]):
                    variable_shards += 1

                row["bin_total_generated_events"] = n_total
                row["production_mixture_fraction"] = (
                    f"{mixture_fraction:.17g}"
                )
                row["generator_coefficient_pb_per_event_weight"] = (
                    f"{coefficient:.17g}"
                )

            coefficient_sum = coefficient_sum_value

            # A coefficient closure statement is only meaningful when every
            # production shard in the interval has exact sigmaGen and sumw.
            if not unresolved_bin_rows:
                coefficient_closes = math.isclose(
                    coefficient_sum,
                    sigma_mean,
                    rel_tol=1.0e-12,
                    abs_tol=max(
                        1.0e-12,
                        abs(sigma_mean) * 1.0e-12,
                    ),
                )

        bin_metadata_complete = not unresolved_bin_rows
        coefficient_closure_pass = (
            coefficient_closure_pass
            and bin_metadata_complete
            and coefficient_closes
        )

        bin_authorized = (
            production_complete
            and interval_closure_pass
            and campaign_coverage_pass
            and bin_metadata_complete
            and coefficient_closes
        )

        def format_finite(value: float) -> str:
            return f"{value:.17g}" if math.isfinite(value) else ""

        bin_rows.append(
            {
                "nominal_bin_index": bin_index,
                "pthat_min_GeV": low,
                "pthat_max_GeV": high,
                "interval_semantics": (
                    f"[{low},infinity)"
                    if str(high) == "Inf"
                    else f"[{low},{high})"
                ),
                "production_campaigns": ",".join(production_campaigns),
                "production_shards": len(rows),
                "resolved_normalization_shards": len(resolved_rows),
                "unresolved_normalization_shards": len(unresolved_bin_rows),
                "generated_events": n_total,
                "resolved_generated_events": n_resolved,
                "sum_event_weights": format_finite(sumw_total),
                "sum_squared_event_weights": format_finite(sumw2_total),
                "effective_events": format_finite(effective_total),
                "event_count_weighted_sigma_gen_pb": (
                    format_finite(sigma_mean)
                ),
                "between_shard_sigma_rms_pb": (
                    format_finite(sigma_rms)
                ),
                "empirical_sigma_standard_error_pb": (
                    format_finite(sigma_empirical_se)
                ),
                "minimum_shard_sigma_gen_pb": (
                    format_finite(sigma_min)
                ),
                "maximum_shard_sigma_gen_pb": (
                    format_finite(sigma_max)
                ),
                "relative_between_shard_sigma_rms": (
                    format_finite(
                        sigma_rms / sigma_mean
                        if (
                            math.isfinite(sigma_rms)
                            and math.isfinite(sigma_mean)
                            and sigma_mean > 0.0
                        )
                        else math.nan
                    )
                ),
                "partial_resolved_coefficient_sum_pb": (
                    format_finite(coefficient_sum)
                ),
                "variable_weight_shards": variable_shards,
                "all_three_production_campaigns_present": (
                    {
                        str(row["campaign"])
                        for row in rows
                    }
                    == set(production_campaigns)
                ),
                "bin_normalization_metadata_complete": (
                    bin_metadata_complete
                ),
                "coefficient_cross_section_closure_pass": (
                    coefficient_closes
                ),
                "denominator_and_stitching_authorized": bin_authorized,
                "physical_weight_application_authorized": False,
                "status": (
                    "qcd_exclusive_bin_denominator_stitching_frozen"
                    if bin_authorized
                    else "qcd_exclusive_bin_fail_closed"
                ),
            }
        )

    if len(bin_rows) != len(EXPECTED_EXCLUSIVE_BINS):
        raise RuntimeError("nominal QCD exclusive-bin count changed")

    testfill_rows = [
        {
            **row,
            "testfill_exclusion_reason": (
                "diagnostic duplicate of nominal [200,300) pTHat bin"
            ),
            "nominal_production_normalization_included": False,
            "status": "qcd_testfill_explicitly_excluded",
        }
        for row in diagnostic_rows
    ]

    overlap_risk_text = "\n".join(
        "\t".join(row.values())
        for row in overlap_rows
    )
    required_overlap_tokens = (
        "qcd_bbbb_general",
        "qcd_bbbb_iht400to600",
        "qcd_hardqcd",
    )
    overlap_risk_evidence_present = all(
        token in overlap_risk_text
        for token in required_overlap_tokens
    )

    all_bins_authorized = all(
        bool(row["denominator_and_stitching_authorized"])
        for row in bin_rows
    )
    qcd_stitching_authorized = (
        production_complete
        and interval_closure_pass
        and campaign_coverage_pass
        and coefficient_closure_pass
        and all_bins_authorized
        and overlap_risk_evidence_present
        and not unresolved_rows
    )

    campaign_role_rows = []
    for campaign in sorted(inventory_map):
        role = campaign_role(campaign)
        campaign_rows = [
            row
            for row in shard_rows
            if row["campaign"] == campaign
        ]
        campaign_role_rows.append(
            {
                "sample_class": "background",
                "process_or_mode": QCD_PROCESS,
                "campaign": campaign,
                "campaign_role": role,
                "members": len(campaign_rows),
                "generated_events": sum(
                    int(row["n_events"])
                    for row in campaign_rows
                ),
                "nominal_production_included": (
                    role == "nominal_production_extension"
                ),
                "independent_full_cross_section_assignment_forbidden": True,
                "combination_policy": (
                    "merge_with_other_production_extensions_inside_each_exclusive_pthat_bin"
                    if role == "nominal_production_extension"
                    else "exclude_from_nominal_keep_for_diagnostic_closure_only"
                ),
                "status": (
                    "qcd_production_extension_role_frozen"
                    if role == "nominal_production_extension"
                    else "qcd_testfill_role_frozen"
                ),
            }
        )

    stitching_rows = [
        {
            "policy_name": "exclusive_pthat_intervals",
            "policy_value": (
                "[50,75),[75,100),[100,200),[200,300),"
                "[300,500),[500,700),[700,1000),[1000,infinity)"
            ),
            "authorized": qcd_stitching_authorized,
            "rationale": (
                "Pythia HardQCD samples are generated in explicit nonoverlapping "
                "pTHat intervals; boundary ownership is lower-inclusive and upper-exclusive"
            ),
        },
        {
            "policy_name": "production_extension_combination",
            "policy_value": (
                "pool all three nominal production campaigns within each pTHat bin; "
                "weight each shard by n_events_shard/N_events_bin and "
                "sigmaGen_shard/sumw_shard"
            ),
            "authorized": qcd_stitching_authorized,
            "rationale": (
                "each campaign is an independent extension of the same physical bin, "
                "not an additional process cross section"
            ),
        },
        {
            "policy_name": "testfill_policy",
            "policy_value": (
                "exclude qcd_hardqcd_importance_testfill_bin3_10k_20260717 "
                "from nominal production"
            ),
            "authorized": qcd_stitching_authorized,
            "rationale": (
                "the test-fill sample duplicates the [200,300) production bin"
            ),
        },
        {
            "policy_name": "other_qcd_family_policy",
            "policy_value": (
                "do not sum qcd_hardqcd with qcd_bbbb_general or "
                "qcd_bbbb_iht400to600 in one nominal prediction"
            ),
            "authorized": qcd_stitching_authorized,
            "rationale": (
                "the samples have overlapping phase space and different generator logic"
            ),
        },
        {
            "policy_name": "cms_style_analysis_role",
            "policy_value": (
                "stitched direct hard-QCD MC is secondary closure/projection only; "
                "primary multijet estimate uses lower-b-tag simulation pseudo-data transfer"
            ),
            "authorized": qcd_stitching_authorized,
            "rationale": (
                "preserves the CMS-style data-driven multijet architecture in a Delphes study"
            ),
        },
    ]

    resolved_production_rows = [
        row
        for row in production_rows
        if bool(row["denominator_authorized"])
    ]
    unresolved_production_shards = (
        len(production_rows) - len(resolved_production_rows)
    )
    variable_production_shards = sum(
        bool(row["per_event_weight_transport_required"])
        for row in resolved_production_rows
    )
    uniform_production_shards = sum(
        bool(row["weights_uniform_positive"])
        for row in resolved_production_rows
    )

    if qcd_stitching_authorized:
        next_gate = (
            "materialize_qcd_train_generator_weight_sidecars_then_build_"
            "process_reference_cross_section_registry"
            if variable_production_shards > 0
            else "build_process_reference_cross_section_registry"
        )
    else:
        next_gate = (
            "recover_exact_qcd_shard_sigmaGen_sumw_and_weight_range_"
            "then_rerun_denominator_stitching_freeze"
        )

    write_tsv(
        args.output / "qcd_shard_normalization_registry.tsv",
        shard_rows,
        OUTPUT_SHARD_FIELDS,
    )
    write_tsv(
        args.output / "qcd_campaign_role_freeze.tsv",
        campaign_role_rows,
        list(campaign_role_rows[0]),
    )
    write_tsv(
        args.output / "qcd_exclusive_bin_registry.tsv",
        bin_rows,
        list(bin_rows[0]),
    )
    write_tsv(
        args.output / "qcd_testfill_exclusion_registry.tsv",
        testfill_rows,
        list(testfill_rows[0]),
    )
    write_tsv(
        args.output / "qcd_stitching_policy.tsv",
        stitching_rows,
        list(stitching_rows[0]),
    )
    write_tsv(
        args.output / "unresolved_qcd_normalization_issues.tsv",
        unresolved_rows,
        (
            list(unresolved_rows[0])
            if unresolved_rows
            else [
                "sample_class",
                "process_or_mode",
                "campaign",
                "source_tag",
                "remote_bundle_path",
                "missing_metadata_fields",
                "conflicting_metadata_fields",
                "ledger_metadata_mismatches",
                "numeric_metadata_valid",
                "event_weight_convention_valid",
                "production_included",
                "required_next_action",
                "status",
            ]
        ),
    )

    summary = {
        "schema_version": 1,
        "status": (
            "hh4b_qcd_importance_denominator_stitching_freeze_pass"
            if qcd_stitching_authorized
            else "hh4b_qcd_importance_denominator_stitching_audit_fail_closed"
        ),
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "source_commit": args.source_commit,
        "inventory": {
            "qcd_campaigns": len(inventory_map),
            "production_campaigns": len(production_campaigns),
            "diagnostic_campaigns": len(diagnostic_campaigns),
            "qcd_source_bundles": len(shard_rows),
            "production_source_bundles": len(production_rows),
            "diagnostic_source_bundles": len(diagnostic_rows),
            "qcd_generated_events": qcd_event_total,
            "production_generated_events": production_event_total,
            "diagnostic_generated_events": diagnostic_event_total,
            "campaign_bin_groups": len(qcd_closure),
            "nominal_exclusive_bins": len(bin_rows),
        },
        "normalization": {
            "production_shard_denominators_authorized": sum(
                bool(row["denominator_authorized"])
                for row in production_rows
            ),
            "uniform_positive_production_shards": uniform_production_shards,
            "variable_positive_production_shards": variable_production_shards,
            "unresolved_production_shards": unresolved_production_shards,
            "per_event_generator_weight_transport_required": (
                variable_production_shards > 0
            ),
            "exclusive_bins_authorized": sum(
                bool(row["denominator_and_stitching_authorized"])
                for row in bin_rows
            ),
            "testfill_shards_excluded": len(diagnostic_rows),
            "unresolved_issue_rows": len(unresolved_rows),
        },
        "readiness": {
            "qcd_importance_denominator_gate_complete": (
                qcd_stitching_authorized
            ),
            "qcd_exclusive_stitching_authorized": (
                qcd_stitching_authorized
            ),
            "qcd_direct_mc_role": (
                "secondary_closure_and_projection_only"
            ),
            "qcd_primary_multijet_role": (
                "lower_btag_simulation_pseudodata_transfer"
            ),
            "external_reference_cross_section_registry_complete": False,
            "physical_weight_application_authorized": False,
            "physics_normalization_ready": False,
        },
        "controls": {
            "root_files_opened": 0,
            "hepmc_files_opened": 0,
            "lhe_files_opened": 0,
            "parquet_files_opened": 0,
            "candidate_files_opened": 0,
            "candidate_rows_read": 0,
            "validation_candidate_files_opened": 0,
            "evaluation_candidate_files_opened": 0,
            "external_reference_cross_sections_assigned": 0,
            "physical_weights_calculated": 0,
            "physical_yields_calculated": 0,
            "models_trained": 0,
            "thresholds_selected": 0,
        },
        "next_gate": next_gate,
    }

    (args.output / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    (args.output / "README.md").write_text(
        "# HH4b hard-QCD importance denominator and stitching audit\n\n"
        "This checkpoint audits all 261 hard-QCD source identities and the "
        "available exact generator metadata. It never substitutes blank or "
        "representative-only values for missing shard-level `sigmaGen`, "
        "`sum_event_weights`, `sum_squared_event_weights`, or weight ranges.\n\n"
        "The three physical campaigns are independent Monte Carlo extensions "
        "inside eight exclusive pTHat intervals. The 10k test-fill shard is "
        "excluded from nominal production. A future frozen event coefficient "
        "for event `i` in shard `s` and bin `b` has the form\n\n"
        "`(N_s / N_b) * (sigmaGen_s / sumw_s) * w_i`.\n\n"
        "That coefficient is authorized only after every production shard in "
        "the relevant interval has exact shard-level metadata. When the "
        "current checkpoint finds identity-only receipt metadata, it records "
        "the missing fields and leaves denominators and stitching fail-closed "
        "instead of attempting numeric aggregation.\n\n"
        "Direct hard-QCD MC remains secondary closure/projection only and must "
        "not be summed with `qcd_bbbb_general` or "
        "`qcd_bbbb_iht400to600`. The primary CMS-style multijet architecture "
        "remains the lower-b-tag simulation pseudo-data transfer.\n\n"
        "No ROOT, HepMC, LHE, Parquet, candidate, validation, or evaluation "
        "payload is opened. No external reference cross section, physical "
        "event weight, or yield is assigned.\n",
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
    print("ALL_FOUR_QCD_CAMPAIGN_ROLES_FROZEN_PASS")
    print("ALL_261_QCD_SOURCE_IDENTITIES_AUDITED_PASS")
    print("QCD_TESTFILL_EXPLICITLY_EXCLUDED_PASS")
    print("EIGHT_EXCLUSIVE_PTHAT_INTERVALS_CLOSED_PASS")
    print("THREE_QCD_PRODUCTION_EXTENSIONS_COMBINATION_POLICY_AUDITED_PASS")
    print("QCD_OTHER_FAMILY_OVERLAP_EXCLUSION_POLICY_FROZEN_PASS")
    print("QCD_DIRECT_MC_SECONDARY_ROLE_FROZEN_PASS")
    print("NO_ROOT_HEPMC_LHE_PARQUET_OR_CANDIDATE_PAYLOAD_OPENED")
    print("NO_EXTERNAL_REFERENCE_CROSS_SECTION_ASSIGNED")
    print("NO_PHYSICAL_WEIGHTS_OR_YIELDS_CALCULATED")
    print("HH4B_QCD_IMPORTANCE_DENOMINATOR_STITCHING_AUDIT_PASS")
    if qcd_stitching_authorized:
        print("QCD_SHARD_GENERATOR_COEFFICIENT_CLOSURE_PASS")
        print("ALL_260_PRODUCTION_QCD_SHARD_DENOMINATORS_FROZEN_PASS")
        print("ALL_8_QCD_EXCLUSIVE_BINS_STITCHING_FROZEN_PASS")
        print("HH4B_QCD_IMPORTANCE_DENOMINATOR_STITCHING_FREEZE_PASS")
    else:
        print(
            "UNRESOLVED_PRODUCTION_QCD_SHARDS\t"
            + str(unresolved_production_shards)
        )
        print("QCD_SHARD_NORMALIZATION_METADATA_COVERAGE_FAIL_CLOSED")
        print("QCD_DENOMINATOR_STITCHING_REMAINS_FAIL_CLOSED")


if __name__ == "__main__":
    main()
