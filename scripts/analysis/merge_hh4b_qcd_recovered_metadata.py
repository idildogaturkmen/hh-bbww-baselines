#!/usr/bin/env python3
"""Merge exact recovered hard-QCD shard metadata with the frozen source ledger."""

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
REQUIRED_CONVENTION = "pythia_info_weight_normalized_to_sigma_gen"
EXPECTED_LEDGER_ROWS = 261
EXPECTED_PRODUCTION_ROWS = 260
EXPECTED_DIAGNOSTIC_ROWS = 1
EXPECTED_RECOVERED_ROWS = 257
EXPECTED_RECOVERED_EVENTS = 2_470_000
EXPECTED_VARIABLE_RECOVERED = 169
EXPECTED_UNIFORM_RECOVERED = 88

NORMALIZATION_FIELDS = (
    "n_events",
    "sigma_gen_pb",
    "event_weight_convention",
    "sum_event_weights",
    "sum_squared_event_weights",
    "min_event_weight",
    "max_event_weight",
)
IDENTITY_FIELDS = (
    "shard_id",
    "seed",
    "pthat_min_GeV",
    "pthat_max_GeV",
)
ALL_REQUIRED_FIELDS = NORMALIZATION_FIELDS + IDENTITY_FIELDS
RECOVERED_PHYSICS_FIELDS = (
    "sigma_gen_pb",
    "event_weight_convention",
    "sum_event_weights",
    "sum_squared_event_weights",
    "min_event_weight",
    "max_event_weight",
)
NUMERIC_FIELDS = {
    "n_events",
    "sigma_gen_pb",
    "sum_event_weights",
    "sum_squared_event_weights",
    "min_event_weight",
    "max_event_weight",
    "shard_id",
    "seed",
    "pthat_min_GeV",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_tsv(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        if reader.fieldnames is None:
            raise RuntimeError(f"TSV has no header: {path}")
        return list(reader.fieldnames), list(reader)


def write_tsv(path: Path, rows: Iterable[dict[str, object]], fields: list[str]) -> None:
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


def parse_bound(value: object) -> int | str:
    normalized = str(value).strip()
    if normalized.lower() in {"inf", "infinity", "none", "null", ""}:
        return "Inf"
    number = float(normalized)
    if not number.is_integer():
        raise ValueError(f"pTHat bound is not integral: {value!r}")
    return int(number)


def same_value(field: str, left: object, right: object) -> bool:
    if field == "event_weight_convention":
        return str(left).strip() == str(right).strip()
    if field == "pthat_max_GeV":
        return parse_bound(left) == parse_bound(right)
    if field in NUMERIC_FIELDS:
        return math.isclose(
            float(left),
            float(right),
            rel_tol=1.0e-12,
            abs_tol=1.0e-12,
        )
    return str(left).strip() == str(right).strip()


def collect_values(
    rows: Iterable[dict[str, str]],
) -> tuple[
    dict[str, dict[str, list[str]]],
    dict[str, dict[str, str]],
    dict[str, list[str]],
]:
    grouped: dict[str, dict[str, list[str]]] = defaultdict(lambda: defaultdict(list))
    for row in rows:
        if row.get("process_or_mode", "").strip() != QCD_PROCESS:
            continue
        source_tag = row.get("source_tag", "").strip()
        field = canonical_field(row.get("field", ""))
        value = row.get("value", "")
        if source_tag and field and str(value).strip() != "":
            grouped[source_tag][field].append(str(value))

    resolved: dict[str, dict[str, str]] = {}
    conflicts: dict[str, list[str]] = {}
    for source_tag, field_map in grouped.items():
        tag_resolved: dict[str, str] = {}
        tag_conflicts: list[str] = []
        for field, values in field_map.items():
            first = values[0]
            if any(not same_value(field, first, value) for value in values[1:]):
                tag_conflicts.append(field)
            else:
                tag_resolved[field] = first
        resolved[source_tag] = tag_resolved
        if tag_conflicts:
            conflicts[source_tag] = sorted(set(tag_conflicts))
    return grouped, resolved, conflicts


def split_base_rows_for_recovered_tags(
    base_rows: Iterable[dict[str, str]],
    recovered_tags: set[str],
) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    """Supersede old partial required-field rows for exactly recovered tags.

    The recovered records passed fragment-level identity checks against the frozen
    bundle ledger. Older PN-c4a rows for those tags were partial discovery evidence,
    not authoritative normalization records. They are retained in a separate audit
    table rather than participating in duplicate-value resolution.
    """
    retained: list[dict[str, str]] = []
    superseded: list[dict[str, str]] = []
    for row in base_rows:
        source_tag = row.get("source_tag", "").strip()
        field = canonical_field(row.get("field", ""))
        if source_tag in recovered_tags and field in ALL_REQUIRED_FIELDS:
            archived = dict(row)
            archived["canonical_field"] = field
            archived["supersession_reason"] = (
                "superseded_by_exact_pn_c4f_small_bundle_recovery_after_ledger_identity_pass"
            )
            superseded.append(archived)
        else:
            retained.append(dict(row))
    return retained, superseded


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-structured-metadata", type=Path, required=True)
    parser.add_argument("--recovered-structured-metadata", type=Path, required=True)
    parser.add_argument("--recovered-wide", type=Path, required=True)
    parser.add_argument("--fragment-manifest", type=Path, required=True)
    parser.add_argument("--scaleout-summary", type=Path, required=True)
    parser.add_argument("--bundle-ledger", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--source-commit", required=True)
    args = parser.parse_args()

    base_fields, base_rows = read_tsv(args.base_structured_metadata)
    recovered_fields, recovered_rows = read_tsv(args.recovered_structured_metadata)
    _, wide_rows = read_tsv(args.recovered_wide)
    _, fragment_rows = read_tsv(args.fragment_manifest)
    _, ledger_rows_all = read_tsv(args.bundle_ledger)
    scaleout_summary = json.loads(args.scaleout_summary.read_text(encoding="utf-8"))

    for required in ("process_or_mode", "source_tag", "field", "value"):
        if required not in base_fields:
            raise RuntimeError(f"base structured metadata lacks {required}")
        if required not in recovered_fields:
            raise RuntimeError(f"recovered structured metadata lacks {required}")

    ledger_rows = [
        row for row in ledger_rows_all
        if row.get("process_or_mode", "").strip() == QCD_PROCESS
    ]
    if len(ledger_rows) != EXPECTED_LEDGER_ROWS:
        raise RuntimeError(f"QCD ledger rows={len(ledger_rows)}, expected {EXPECTED_LEDGER_ROWS}")

    ledger_by_tag = {row["source_tag"]: row for row in ledger_rows}
    if len(ledger_by_tag) != EXPECTED_LEDGER_ROWS:
        raise RuntimeError("QCD ledger source tags are not unique")

    production_ledger = [row for row in ledger_rows if "testfill" not in row["campaign"]]
    diagnostic_ledger = [row for row in ledger_rows if "testfill" in row["campaign"]]
    if len(production_ledger) != EXPECTED_PRODUCTION_ROWS:
        raise RuntimeError("QCD production ledger count changed")
    if len(diagnostic_ledger) != EXPECTED_DIAGNOSTIC_ROWS:
        raise RuntimeError("QCD diagnostic ledger count changed")

    if len(wide_rows) != EXPECTED_RECOVERED_ROWS:
        raise RuntimeError(f"recovered wide rows={len(wide_rows)}")
    recovered_tags = {row["source_tag"] for row in wide_rows}
    if len(recovered_tags) != EXPECTED_RECOVERED_ROWS:
        raise RuntimeError("recovered source tags are not unique")
    if not recovered_tags <= set(ledger_by_tag):
        raise RuntimeError("recovered source tags are not a subset of the QCD ledger")
    if any("testfill" in row["campaign"] for row in wide_rows):
        raise RuntimeError("diagnostic test-fill appeared in the 257 production recovery rows")

    recovered_events = sum(int(row["n_events"]) for row in wide_rows)
    variable_recovered = sum(
        row["per_event_generator_weight_transport_required"].strip().lower() == "true"
        for row in wide_rows
    )
    uniform_recovered = sum(
        row["weights_uniform_positive"].strip().lower() == "true"
        for row in wide_rows
    )
    if recovered_events != EXPECTED_RECOVERED_EVENTS:
        raise RuntimeError(f"recovered events={recovered_events}")
    if variable_recovered != EXPECTED_VARIABLE_RECOVERED:
        raise RuntimeError(f"variable recovered shards={variable_recovered}")
    if uniform_recovered != EXPECTED_UNIFORM_RECOVERED:
        raise RuntimeError(f"uniform recovered shards={uniform_recovered}")
    for row in wide_rows:
        source_tag = row["source_tag"]
        ledger = ledger_by_tag[source_tag]
        if row["identity_pass"].strip().lower() != "true":
            raise RuntimeError(f"identity failure in recovered row {source_tag}")
        if row["exact_metadata_recovered"].strip().lower() != "true":
            raise RuntimeError(f"metadata recovery not exact for {source_tag}")
        if row["normalization_denominator_authorized"].strip().lower() != "false":
            raise RuntimeError("recovery stage improperly authorized a denominator")
        if row["campaign"] != ledger["campaign"]:
            raise RuntimeError(f"recovered campaign mismatch for {source_tag}")
        if row["remote_bundle_path"] != ledger["remote_bundle_path"]:
            raise RuntimeError(f"recovered bundle-path mismatch for {source_tag}")
        if int(row["bin_index"]) != int(ledger["bin_index"]):
            raise RuntimeError(f"recovered bin-index mismatch for {source_tag}")
        if int(row["shard_id"]) != int(ledger["shard_id"]):
            raise RuntimeError(f"recovered shard mismatch for {source_tag}")
        if int(row["seed"]) != int(ledger["seed"]):
            raise RuntimeError(f"recovered seed mismatch for {source_tag}")
        if int(float(row["pthat_min_GeV"])) != int(float(ledger["pthat_min_GeV"])):
            raise RuntimeError(f"recovered pTHat minimum mismatch for {source_tag}")
        if parse_bound(row["pthat_max_GeV"]) != parse_bound(ledger["pthat_max_GeV"]):
            raise RuntimeError(f"recovered pTHat maximum mismatch for {source_tag}")
        if int(row["n_events"]) <= 0:
            raise RuntimeError(f"recovered n_events is nonpositive for {source_tag}")

    if len(fragment_rows) != EXPECTED_RECOVERED_ROWS:
        raise RuntimeError(f"fragment manifest rows={len(fragment_rows)}")
    if {row["source_tag"] for row in fragment_rows} != recovered_tags:
        raise RuntimeError("fragment manifest and recovered-wide source tags differ")

    expected_long_rows = EXPECTED_RECOVERED_ROWS * len(ALL_REQUIRED_FIELDS)
    if len(recovered_rows) != expected_long_rows:
        raise RuntimeError(
            f"recovered long rows={len(recovered_rows)}, expected {expected_long_rows}"
        )
    recovered_grouped, recovered_resolved, recovered_conflicts = collect_values(recovered_rows)
    if recovered_conflicts:
        raise RuntimeError(f"recovered metadata conflicts remain: {recovered_conflicts}")
    if set(recovered_resolved) != recovered_tags:
        raise RuntimeError("recovered long and wide source-tag sets differ")
    for source_tag in recovered_tags:
        observed_fields = set(recovered_resolved[source_tag])
        if observed_fields != set(ALL_REQUIRED_FIELDS):
            raise RuntimeError(
                f"recovered field set differs for {source_tag}: {sorted(observed_fields)}"
            )
        convention = recovered_resolved[source_tag]["event_weight_convention"]
        if convention != REQUIRED_CONVENTION:
            raise RuntimeError(f"unexpected convention for {source_tag}: {convention}")

    _, base_resolved, base_conflicts = collect_values(base_rows)
    if base_conflicts:
        raise RuntimeError(f"pre-existing QCD metadata conflicts: {base_conflicts}")
    preexisting_complete = {
        source_tag
        for source_tag, values in base_resolved.items()
        if set(ALL_REQUIRED_FIELDS) <= set(values)
    }
    preexisting_production = {
        tag for tag in preexisting_complete
        if "testfill" not in ledger_by_tag[tag]["campaign"]
    }
    preexisting_diagnostic = preexisting_complete - preexisting_production
    if len(preexisting_complete) != 4:
        raise RuntimeError(f"pre-existing complete QCD tags={len(preexisting_complete)}, expected 4")
    if len(preexisting_production) != 3 or len(preexisting_diagnostic) != 1:
        raise RuntimeError("pre-existing QCD completion split is not 3 production + 1 diagnostic")
    if recovered_tags & preexisting_complete:
        raise RuntimeError("recovered tags overlap pre-existing complete QCD tags")

    expected_recovered_tags = {
        row["source_tag"] for row in production_ledger
    } - preexisting_production
    if recovered_tags != expected_recovered_tags:
        missing = sorted(expected_recovered_tags - recovered_tags)
        extra = sorted(recovered_tags - expected_recovered_tags)
        raise RuntimeError(f"recovered target-set mismatch; missing={missing[:3]}, extra={extra[:3]}")

    combined_fields: list[str] = []
    for field in base_fields + recovered_fields:
        if field not in combined_fields:
            combined_fields.append(field)
    for mandatory in ("sample_class", "process_or_mode", "campaign", "source_tag", "field", "value"):
        if mandatory not in combined_fields:
            combined_fields.append(mandatory)

    retained_base_rows, superseded_base_rows = split_base_rows_for_recovered_tags(
        base_rows,
        recovered_tags,
    )
    unexpected_superseded_physics = [
        row
        for row in superseded_base_rows
        if row["canonical_field"] in RECOVERED_PHYSICS_FIELDS
        and str(row.get("value", "")).strip() != ""
    ]
    if unexpected_superseded_physics:
        preview = [
            (row.get("source_tag", ""), row["canonical_field"], row.get("value", ""))
            for row in unexpected_superseded_physics[:5]
        ]
        raise RuntimeError(
            "pre-existing PN-c4a rows unexpectedly contain recovered physics fields: "
            f"{preview}"
        )
    if not superseded_base_rows:
        raise RuntimeError("no partial base identity rows were superseded for recovered tags")

    combined_rows = retained_base_rows + [dict(row) for row in recovered_rows]
    combined_rows.sort(
        key=lambda row: (
            row.get("process_or_mode", ""),
            row.get("campaign", ""),
            row.get("source_tag", ""),
            canonical_field(row.get("field", "")),
            row.get("source_path", row.get("source_paths", "")),
            row.get("value", ""),
        )
    )

    _, merged_resolved, merged_conflicts = collect_values(combined_rows)
    if merged_conflicts:
        raise RuntimeError(f"merged QCD metadata conflicts: {merged_conflicts}")
    if set(ledger_by_tag) - set(merged_resolved):
        raise RuntimeError("some QCD ledger source tags have no merged metadata")

    coverage_rows: list[dict[str, object]] = []
    for ledger in sorted(
        ledger_rows,
        key=lambda row: (
            row["campaign"],
            int(row["bin_index"]),
            int(row["shard_id"]),
            int(row["seed"]),
        ),
    ):
        source_tag = ledger["source_tag"]
        values = merged_resolved[source_tag]
        missing = [field for field in ALL_REQUIRED_FIELDS if field not in values]
        if missing:
            raise RuntimeError(f"merged metadata missing {missing} for {source_tag}")

        n_events = int(float(values["n_events"]))
        if n_events <= 0:
            raise RuntimeError(f"nonpositive n_events for {source_tag}")
        if int(float(values["shard_id"])) != int(ledger["shard_id"]):
            raise RuntimeError(f"shard mismatch for {source_tag}")
        if int(float(values["seed"])) != int(ledger["seed"]):
            raise RuntimeError(f"seed mismatch for {source_tag}")
        if int(float(values["pthat_min_GeV"])) != int(ledger["pthat_min_GeV"]):
            raise RuntimeError(f"pTHat minimum mismatch for {source_tag}")
        if parse_bound(values["pthat_max_GeV"]) != parse_bound(ledger["pthat_max_GeV"]):
            raise RuntimeError(f"pTHat maximum mismatch for {source_tag}")
        if values["event_weight_convention"].strip() != REQUIRED_CONVENTION:
            raise RuntimeError(f"weight convention mismatch for {source_tag}")

        sigma = float(values["sigma_gen_pb"])
        sumw = float(values["sum_event_weights"])
        sumw2 = float(values["sum_squared_event_weights"])
        minimum = float(values["min_event_weight"])
        maximum = float(values["max_event_weight"])
        if not (
            math.isfinite(sigma) and sigma > 0.0
            and math.isfinite(sumw) and sumw > 0.0
            and math.isfinite(sumw2) and sumw2 > 0.0
            and math.isfinite(minimum) and minimum > 0.0
            and math.isfinite(maximum) and maximum >= minimum
        ):
            raise RuntimeError(f"invalid numeric metadata for {source_tag}")

        coverage_rows.append(
            {
                "sample_class": ledger["sample_class"],
                "process_or_mode": ledger["process_or_mode"],
                "campaign": ledger["campaign"],
                "campaign_role": (
                    "diagnostic_testfill_excluded"
                    if "testfill" in ledger["campaign"]
                    else "nominal_production_extension"
                ),
                "source_tag": source_tag,
                "remote_bundle_path": ledger["remote_bundle_path"],
                "bin_index": ledger["bin_index"],
                "pthat_min_GeV": ledger["pthat_min_GeV"],
                "pthat_max_GeV": ledger["pthat_max_GeV"],
                "shard_id": ledger["shard_id"],
                "seed": ledger["seed"],
                "n_events": n_events,
                "metadata_origin": (
                    "pn_c4f_exact_small_bundle_recovery"
                    if source_tag in recovered_tags
                    else "preexisting_exact_metadata"
                ),
                "all_required_fields_complete": True,
                "conflicting_fields": "",
                "identity_pass": True,
                "normalization_denominator_authorized": False,
                "status": "qcd_exact_metadata_merge_complete",
            }
        )

    total_qcd_events = sum(int(row["n_events"]) for row in coverage_rows)
    production_events = sum(
        int(row["n_events"])
        for row in coverage_rows
        if row["campaign_role"] == "nominal_production_extension"
    )
    diagnostic_events = total_qcd_events - production_events
    if total_qcd_events != 2_510_000:
        raise RuntimeError(f"merged QCD event total={total_qcd_events}")
    if production_events != 2_500_000:
        raise RuntimeError(f"merged production event total={production_events}")
    if diagnostic_events != 10_000:
        raise RuntimeError(f"merged diagnostic event total={diagnostic_events}")

    output = args.output
    output.mkdir(parents=True, exist_ok=False)
    write_tsv(
        output / "merged_structured_nonstandard_metadata_candidates.tsv",
        combined_rows,
        combined_fields,
    )
    write_tsv(
        output / "qcd_exact_metadata_coverage.tsv",
        coverage_rows,
        list(coverage_rows[0]),
    )
    superseded_fields: list[str] = []
    for row in superseded_base_rows:
        for field in row:
            if field not in superseded_fields:
                superseded_fields.append(field)
    write_tsv(
        output / "superseded_partial_base_metadata.tsv",
        sorted(
            superseded_base_rows,
            key=lambda row: (
                row.get("campaign", ""),
                row.get("source_tag", ""),
                row.get("canonical_field", ""),
                row.get("value", ""),
            ),
        ),
        superseded_fields,
    )

    summary = {
        "schema_version": 1,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "source_commit": args.source_commit,
        "status": "hh4b_qcd_exact_metadata_recovery_merge_pass",
        "inventory": {
            "qcd_source_bundles": len(ledger_rows),
            "production_source_bundles": len(production_ledger),
            "diagnostic_source_bundles": len(diagnostic_ledger),
            "preexisting_complete_qcd_shards": len(preexisting_complete),
            "preexisting_complete_production_shards": len(preexisting_production),
            "recovered_production_shards": len(recovered_tags),
            "merged_complete_qcd_shards": len(coverage_rows),
            "merged_complete_production_shards": len(production_ledger),
            "recovered_generated_events": recovered_events,
            "merged_qcd_generated_events": total_qcd_events,
            "merged_production_generated_events": production_events,
            "merged_diagnostic_generated_events": diagnostic_events,
            "recovered_variable_positive_shards": variable_recovered,
            "recovered_uniform_positive_shards": uniform_recovered,
            "retained_base_metadata_rows": len(retained_base_rows),
            "superseded_partial_base_metadata_rows": len(superseded_base_rows),
        },
        "readiness": {
            "all_261_qcd_metadata_records_complete": True,
            "all_260_production_qcd_metadata_records_complete": True,
            "metadata_conflicts": 0,
            "identity_mismatches": 0,
            "normalization_denominators_authorized": 0,
            "qcd_exclusive_stitching_authorized": False,
            "physical_weight_application_authorized": False,
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
        "scaleout_summary_sha256": sha256_file(args.scaleout_summary),
        "next_gate": "rerun_pn_c4e_qcd_denominator_and_stitching_freeze",
    }
    (output / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (output / "README.md").write_text(
        "# HH4b exact hard-QCD metadata recovery merge\n\n"
        "This checkpoint merges the exact small-provenance metadata recovered for "
        "257 previously unresolved physical hard-QCD shards with the four QCD shards "
        "that already had exact metadata: three nominal production shards and the "
        "excluded diagnostic test-fill shard.\n\n"
        "For the 257 recovered tags, older partial PN-c4a required-field rows are "
        "preserved in `superseded_partial_base_metadata.tsv` but do not participate in "
        "duplicate-value resolution. The recovered records independently passed exact "
        "campaign, bundle-path, bin, shard, seed, and pTHat identity checks against the "
        "frozen ledger. All 261 QCD source identities now have exact `n_events`, "
        "`sigma_gen_pb`, generator-weight sums, squared-weight sums, weight ranges, "
        "shard/seed identity, and pTHat bounds. All 260 physical production shards are "
        "metadata-complete. "
        "This merge authorizes no denominator, stitching coefficient, physical weight, "
        "yield, candidate read, model, or threshold. Those decisions remain delegated "
        "to the PN-c4e denominator-and-stitching freeze.\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    print("ALL_257_RECOVERED_QCD_SHARD_METADATA_RECORDS_MERGED_PASS")
    print("ALL_261_QCD_SOURCE_METADATA_RECORDS_COMPLETE_PASS")
    print("ALL_260_PRODUCTION_QCD_METADATA_RECORDS_COMPLETE_PASS")
    print("NO_NORMALIZATION_DENOMINATOR_OR_PHYSICAL_WEIGHT_AUTHORIZED")
    print("HH4B_QCD_EXACT_METADATA_RECOVERY_MERGE_PASS")


if __name__ == "__main__":
    main()
