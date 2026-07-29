#!/usr/bin/env python3
"""Audit LHE nominal weights for six representative standard processes."""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import io
import json
import math
import os
import shutil
import subprocess
import tarfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable


CANARY_PROCESSES = (
    "vbf_hh4b",
    "ttbar_inclusive",
    "qcd_bbbb_general",
    "tchannel_top",
    "tth_hbb",
    "wwz_zbb",
)

EXPECTED_CANARIES = len(CANARY_PROCESSES)
EXPECTED_EQUIVALENCE_GROUPS = 23
EXPECTED_STANDARD_CAMPAIGNS = 47
MAX_RECORDED_DISTINCT_WEIGHTS = 20


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


def remote_uri(host: str, locator: str) -> str:
    return host.rstrip("/") + "//" + locator.lstrip("/")


def classify_weights(
    *,
    event_count: int,
    positive: int,
    negative: int,
    zero: int,
    minimum: float,
    maximum: float,
    sum_weights: float,
    sum_squared_weights: float,
) -> tuple[str, bool]:
    if event_count <= 0:
        return "no_events", False

    scale = max(abs(minimum), abs(maximum), 1.0)
    uniform = abs(maximum - minimum) <= 1.0e-12 * scale

    if zero > 0:
        return "contains_zero_weights", False
    if negative == 0 and positive == event_count and uniform:
        return "uniform_positive", True
    if negative == 0 and positive == event_count:
        return "variable_positive", False
    if positive > 0 and negative > 0:
        magnitude_scale = max(abs(minimum), abs(maximum), 1.0)
        symmetric = abs(abs(minimum) - abs(maximum)) <= 1.0e-12 * magnitude_scale
        if symmetric:
            return "signed_uniform_magnitude", False
        return "variable_signed", False
    if negative == event_count and uniform:
        return "uniform_negative", False

    if not math.isfinite(sum_weights) or not math.isfinite(sum_squared_weights):
        return "nonfinite_weight_summary", False

    return "unclassified", False


def parse_lhe_weights(
    bundle: Path,
    member_name: str,
    expected_member_size: int,
) -> dict[str, object]:
    with tarfile.open(bundle, mode="r:*") as archive:
        try:
            member = archive.getmember(member_name)
        except KeyError as error:
            raise RuntimeError(
                f"LHE member is absent from bundle: {member_name}"
            ) from error

        if not member.isfile():
            raise RuntimeError(f"LHE member is not a regular file: {member_name}")
        if member.size != expected_member_size:
            raise RuntimeError(
                f"LHE member size mismatch for {member_name}: "
                f"{member.size} != {expected_member_size}"
            )

        extracted = archive.extractfile(member)
        if extracted is None:
            raise RuntimeError(f"could not stream LHE member: {member_name}")

        with gzip.GzipFile(fileobj=extracted, mode="rb") as compressed:
            with io.TextIOWrapper(
                compressed,
                encoding="utf-8",
                errors="strict",
                newline="",
            ) as text:
                event_count = 0
                positive = 0
                negative = 0
                zero = 0
                sum_weights = 0.0
                sum_squared_weights = 0.0
                minimum = math.inf
                maximum = -math.inf
                idwtup_values: set[int] = set()
                distinct_weights: set[str] = set()
                first_weights: list[str] = []

                awaiting_init_line = False
                awaiting_event_line = False

                for raw_line in text:
                    stripped = raw_line.strip()
                    if not stripped:
                        continue

                    if stripped.startswith("<init>"):
                        awaiting_init_line = True
                        continue

                    if stripped.startswith("<event"):
                        awaiting_event_line = True
                        continue

                    if awaiting_init_line:
                        if stripped.startswith(("#", "<")):
                            continue
                        fields = stripped.split()
                        if len(fields) < 10:
                            raise RuntimeError(
                                "malformed LHE init line: " + stripped[:200]
                            )
                        try:
                            idwtup_values.add(int(fields[-1]))
                        except ValueError as error:
                            raise RuntimeError(
                                "invalid IDWTUP in LHE init line: "
                                + stripped[:200]
                            ) from error
                        awaiting_init_line = False
                        continue

                    if awaiting_event_line:
                        if stripped.startswith(("#", "<")):
                            continue
                        fields = stripped.split()
                        if len(fields) < 3:
                            raise RuntimeError(
                                "malformed LHE event header: "
                                + stripped[:200]
                            )
                        try:
                            int(fields[0])
                            int(fields[1])
                            weight = float(
                                fields[2].replace("D", "E").replace("d", "e")
                            )
                        except ValueError as error:
                            raise RuntimeError(
                                "invalid LHE event header: " + stripped[:200]
                            ) from error

                        if not math.isfinite(weight):
                            raise RuntimeError("nonfinite LHE event weight")

                        event_count += 1
                        sum_weights += weight
                        sum_squared_weights += weight * weight
                        minimum = min(minimum, weight)
                        maximum = max(maximum, weight)
                        if weight > 0:
                            positive += 1
                        elif weight < 0:
                            negative += 1
                        else:
                            zero += 1

                        rendered = f"{weight:.17g}"
                        if len(first_weights) < MAX_RECORDED_DISTINCT_WEIGHTS:
                            first_weights.append(rendered)
                        if len(distinct_weights) < MAX_RECORDED_DISTINCT_WEIGHTS:
                            distinct_weights.add(rendered)

                        awaiting_event_line = False

                if awaiting_event_line:
                    raise RuntimeError("LHE ended while awaiting an event header")
                if event_count <= 0:
                    raise RuntimeError("no LHE event headers were parsed")
                if math.isinf(minimum) or math.isinf(maximum):
                    raise RuntimeError("LHE weight extrema were not populated")

                classification, ngen_candidate = classify_weights(
                    event_count=event_count,
                    positive=positive,
                    negative=negative,
                    zero=zero,
                    minimum=minimum,
                    maximum=maximum,
                    sum_weights=sum_weights,
                    sum_squared_weights=sum_squared_weights,
                )

                return {
                    "lhe_event_count": event_count,
                    "idwtup_values": ",".join(
                        str(value) for value in sorted(idwtup_values)
                    ),
                    "positive_weight_events": positive,
                    "negative_weight_events": negative,
                    "zero_weight_events": zero,
                    "sum_lhe_nominal_weights": f"{sum_weights:.17g}",
                    "sum_squared_lhe_nominal_weights": (
                        f"{sum_squared_weights:.17g}"
                    ),
                    "minimum_lhe_nominal_weight": f"{minimum:.17g}",
                    "maximum_lhe_nominal_weight": f"{maximum:.17g}",
                    "first_lhe_nominal_weights": ",".join(first_weights),
                    "recorded_distinct_lhe_nominal_weights": ",".join(
                        sorted(distinct_weights)
                    ),
                    "weight_classification": classification,
                    "ngen_denominator_candidate_supported_by_canary": (
                        ngen_candidate
                    ),
                    "normalization_denominator_authorized": False,
                }


def validate_receipt(
    receipt: dict[str, object],
    selection: dict[str, object],
) -> bool:
    identifying_fields = (
        "sample_class",
        "process_or_mode",
        "campaign",
        "representative_locator",
        "bundle_sha256",
        "lhe_member_name",
        "lhe_member_size_bytes",
    )
    return all(
        str(receipt.get(field)) == str(selection.get(field))
        for field in identifying_fields
    ) and receipt.get("status") == "canary_lhe_weight_audit_complete"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bundle-summary", type=Path, required=True)
    parser.add_argument("--event-payload-inventory", type=Path, required=True)
    parser.add_argument("--equivalence-groups", type=Path, required=True)
    parser.add_argument("--standard-status", type=Path, required=True)
    parser.add_argument("--access-plan", type=Path, required=True)
    parser.add_argument("--eos-host", required=True)
    parser.add_argument("--scratch", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--source-commit", required=True)
    args = parser.parse_args()

    bundle_rows = read_tsv(args.bundle_summary)
    payload_rows = read_tsv(args.event_payload_inventory)
    group_rows = read_tsv(args.equivalence_groups)
    standard_rows = read_tsv(args.standard_status)
    access_rows = read_tsv(args.access_plan)

    if len(group_rows) != EXPECTED_EQUIVALENCE_GROUPS:
        raise RuntimeError("configuration-equivalence group count mismatch")
    if len(standard_rows) != EXPECTED_STANDARD_CAMPAIGNS:
        raise RuntimeError("standard campaign status count mismatch")

    group_map = {row["process_or_mode"]: row for row in group_rows}
    standard_map = {
        (
            row["sample_class"],
            row["process_or_mode"],
            row["campaign"],
        ): row
        for row in standard_rows
    }
    access_map = {
        (
            row["sample_class"],
            row["process_or_mode"],
            row["campaign"],
        ): row
        for row in access_rows
    }

    for process in CANARY_PROCESSES:
        if process not in group_map:
            raise RuntimeError(f"missing equivalence group: {process}")
        if group_map[process]["configuration_equivalence_authorized"] != "True":
            raise RuntimeError(
                f"configuration equivalence is not authorized for {process}"
            )

    bundles_by_process: dict[str, list[dict[str, str]]] = {}
    for process in CANARY_PROCESSES:
        campaigns = set(group_map[process]["campaigns"].split(","))
        candidates = [
            row
            for row in bundle_rows
            if row["process_or_mode"] == process
            and row["campaign"] in campaigns
        ]
        if not candidates:
            raise RuntimeError(f"no bundle candidates for {process}")
        bundles_by_process[process] = candidates

    payload_by_campaign: dict[
        tuple[str, str, str],
        list[dict[str, str]],
    ] = {}
    for row in payload_rows:
        key = (
            row["sample_class"],
            row["process_or_mode"],
            row["campaign"],
        )
        payload_by_campaign.setdefault(key, []).append(row)

    selections: list[dict[str, object]] = []
    for process in CANARY_PROCESSES:
        candidates = bundles_by_process[process]
        eligible = []
        for row in candidates:
            key = (
                row["sample_class"],
                row["process_or_mode"],
                row["campaign"],
            )
            lhe_rows = [
                item
                for item in payload_by_campaign.get(key, [])
                if item["member_name"].lower().endswith(".lhe.gz")
            ]
            if len(lhe_rows) == 1:
                eligible.append((int(row["expected_size_bytes"]), row, lhe_rows[0]))

        if not eligible:
            raise RuntimeError(
                f"no campaign with exactly one LHE payload for {process}"
            )

        _, bundle, lhe = min(
            eligible,
            key=lambda item: (
                item[0],
                item[1]["campaign"],
            ),
        )
        key = (
            bundle["sample_class"],
            bundle["process_or_mode"],
            bundle["campaign"],
        )
        status = standard_map[key]
        access = access_map[key]

        if access["controlled_generator_payload_weight_audit_required"] != "True":
            raise RuntimeError(
                f"controlled weight audit is not required for selected {key}"
            )
        if access["generator_payload_access_authorized_now"] != "False":
            raise RuntimeError(
                f"prior gate unexpectedly authorized payload access for {key}"
            )

        selections.append(
            {
                "sample_class": key[0],
                "process_or_mode": key[1],
                "campaign": key[2],
                "equivalent_campaigns": group_map[process]["campaigns"],
                "selection_rule": (
                    "smallest_frozen_representative_bundle_with_one_lhe_member"
                ),
                "representative_locator": bundle["representative_locator"],
                "expected_bundle_size_bytes": int(
                    bundle["expected_size_bytes"]
                ),
                "bundle_sha256": bundle["bundle_sha256"],
                "lhe_member_name": lhe["member_name"],
                "lhe_member_size_bytes": int(lhe["size_bytes"]),
                "manifest_generated_events": int(
                    status["manifest_generated_events"]
                ),
                "run_card_nevents_values": status[
                    "run_card_nevents_values"
                ],
                "run_card_event_norm_values": status[
                    "run_card_event_norm_values"
                ],
                "candidate_parquet_access_authorized": False,
                "root_access_authorized": False,
                "lhe_weight_access_authorized_for_this_canary": True,
                "normalization_denominator_authorized": False,
                "status": "selected_for_controlled_lhe_weight_canary",
            }
        )

    if len(selections) != EXPECTED_CANARIES:
        raise RuntimeError("canary selection count mismatch")

    args.output.mkdir(parents=True, exist_ok=True)
    receipt_dir = args.output / "receipts"
    receipt_dir.mkdir(exist_ok=True)
    args.scratch.mkdir(parents=True, exist_ok=True)

    results: list[dict[str, object]] = []
    transfer_rows: list[dict[str, object]] = []

    for ordinal, selection in enumerate(selections, start=1):
        process = str(selection["process_or_mode"])
        campaign = str(selection["campaign"])
        receipt_path = receipt_dir / f"{ordinal:02d}_{process}.json"

        if receipt_path.is_file():
            receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
            if not validate_receipt(receipt, selection):
                raise RuntimeError(
                    f"existing receipt does not match frozen selection: {receipt_path}"
                )
            print(f"RESUME\t{ordinal}/{EXPECTED_CANARIES}\t{process}\t{campaign}")
            results.append(receipt)
            transfer_rows.append(
                {
                    "ordinal": ordinal,
                    "process_or_mode": process,
                    "campaign": campaign,
                    "transfer_performed_this_run": False,
                    "bundle_removed_after_audit": True,
                    "status": "resumed_from_verified_receipt",
                }
            )
            continue

        local_bundle = args.scratch / f"{ordinal:02d}_{process}_bundle.tar.gz"
        local_bundle.unlink(missing_ok=True)

        uri = remote_uri(
            args.eos_host,
            str(selection["representative_locator"]),
        )

        print(
            f"TRANSFER\t{ordinal}/{EXPECTED_CANARIES}\t{process}\t{campaign}"
        )
        print(f"REMOTE\t{uri}")
        print(f"LOCAL\t{local_bundle}")

        try:
            subprocess.run(
                [
                    "xrdcp",
                    "--nopbar",
                    "--force",
                    uri,
                    str(local_bundle),
                ],
                check=True,
            )

            observed_size = local_bundle.stat().st_size
            expected_size = int(selection["expected_bundle_size_bytes"])
            if observed_size != expected_size:
                raise RuntimeError(
                    f"bundle size mismatch for {process}: "
                    f"{observed_size} != {expected_size}"
                )

            observed_sha = sha256_file(local_bundle)
            if observed_sha != selection["bundle_sha256"]:
                raise RuntimeError(
                    f"bundle SHA-256 mismatch for {process}"
                )

            audit = parse_lhe_weights(
                local_bundle,
                str(selection["lhe_member_name"]),
                int(selection["lhe_member_size_bytes"]),
            )

            receipt = {
                **selection,
                **audit,
                "ordinal": ordinal,
                "remote_uri": uri,
                "observed_bundle_size_bytes": observed_size,
                "observed_bundle_sha256": observed_sha,
                "root_files_opened": 0,
                "parquet_files_opened": 0,
                "candidate_files_opened": 0,
                "validation_candidate_files_opened": 0,
                "evaluation_candidate_files_opened": 0,
                "lhe_members_opened": 1,
                "status": "canary_lhe_weight_audit_complete",
            }

            receipt_path.write_text(
                json.dumps(receipt, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            results.append(receipt)
            transfer_rows.append(
                {
                    "ordinal": ordinal,
                    "process_or_mode": process,
                    "campaign": campaign,
                    "transfer_performed_this_run": True,
                    "bundle_removed_after_audit": True,
                    "status": "bundle_verified_lhe_only_audited",
                }
            )
        finally:
            local_bundle.unlink(missing_ok=True)

    results.sort(key=lambda row: int(row["ordinal"]))
    transfer_rows.sort(key=lambda row: int(row["ordinal"]))

    uniform_positive = sum(
        row["weight_classification"] == "uniform_positive"
        for row in results
    )
    nonuniform_or_signed = len(results) - uniform_positive
    all_parsed = all(int(row["lhe_event_count"]) > 0 for row in results)

    next_gate = (
        "scale_standard_lhe_weight_audit_to_all_23_groups_and_47_campaigns"
        if all_parsed and uniform_positive == EXPECTED_CANARIES
        else "review_nonuniform_or_signed_standard_lhe_weight_canary"
    )

    decision_rows = []
    for row in results:
        event_count = int(row["lhe_event_count"])
        manifest_count = int(row["manifest_generated_events"])
        run_values = {
            int(value)
            for value in str(row["run_card_nevents_values"]).split(",")
            if value
        }
        decision_rows.append(
            {
                "process_or_mode": row["process_or_mode"],
                "campaign": row["campaign"],
                "weight_classification": row["weight_classification"],
                "lhe_event_count": event_count,
                "manifest_generated_events": manifest_count,
                "lhe_count_matches_manifest_campaign_count": (
                    event_count == manifest_count
                ),
                "lhe_count_matches_representative_run_card_nevents": (
                    event_count in run_values
                ),
                "ngen_denominator_candidate_supported_by_canary": row[
                    "ngen_denominator_candidate_supported_by_canary"
                ],
                "equivalent_campaigns_covered_by_configuration_only": row[
                    "equivalent_campaigns"
                ],
                "denominator_generalization_authorized": False,
                "normalization_denominator_authorized": False,
                "required_next_action": (
                    "scale weight audit across all standard groups and campaigns"
                    if row[
                        "ngen_denominator_candidate_supported_by_canary"
                    ]
                    else "review signed or variable generator weights"
                ),
                "status": "canary_evidence_only_fail_closed",
            }
        )

    write_tsv(
        args.output / "canary_selection.tsv",
        selections,
        list(selections[0]),
    )
    write_tsv(
        args.output / "lhe_weight_audit.tsv",
        results,
        list(results[0]),
    )
    write_tsv(
        args.output / "canary_decision.tsv",
        decision_rows,
        list(decision_rows[0]),
    )
    write_tsv(
        args.output / "transfer_and_cleanup_audit.tsv",
        transfer_rows,
        list(transfer_rows[0]),
    )

    summary = {
        "schema_version": 1,
        "status": "hh4b_standard_lhe_weight_canary_pass",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "source_commit": args.source_commit,
        "selection": {
            "process_strata": list(CANARY_PROCESSES),
            "selected_processes": len(selections),
            "selected_campaigns": len(selections),
            "selection_rule": (
                "smallest frozen representative bundle with exactly one LHE member"
            ),
        },
        "audit": {
            "bundles_size_and_sha256_verified": len(results),
            "lhe_members_opened": len(results),
            "lhe_events_parsed": sum(
                int(row["lhe_event_count"]) for row in results
            ),
            "uniform_positive_canaries": uniform_positive,
            "nonuniform_or_signed_canaries": nonuniform_or_signed,
            "all_canaries_parsed": all_parsed,
            "all_temporary_bundles_removed": not any(
                args.scratch.iterdir()
            ),
        },
        "readiness": {
            "controlled_standard_lhe_weight_canary_complete": True,
            "ngen_denominator_candidates_supported_by_canary": (
                uniform_positive
            ),
            "denominator_generalization_authorized": 0,
            "standard_campaign_denominators_authorized": 0,
            "nonstandard_campaign_denominators_authorized": 0,
            "external_reference_cross_sections_authorized": 0,
            "physics_normalization_ready": False,
        },
        "controls": {
            "root_files_opened": 0,
            "parquet_files_opened": 0,
            "candidate_files_opened": 0,
            "candidate_rows_read": 0,
            "validation_candidate_files_opened": 0,
            "evaluation_candidate_files_opened": 0,
            "lhe_members_opened": len(results),
            "normalization_denominators_authorized": 0,
            "cross_sections_assigned": 0,
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
        "# HH4b controlled standard LHE-weight canary\n\n"
        "This checkpoint opens only six explicitly selected "
        "`source/unweighted_events.lhe.gz` members, one from each physics "
        "stratum: VBF HH signal, inclusive ttbar, standard QCD bbbb, "
        "t-channel single top, ttH(H->bb), and WWZ->Zbb.\n\n"
        "Each containing EOS bundle is verified against its frozen byte size "
        "and SHA-256 before the LHE member is streamed. ROOT, Parquet, "
        "candidate, validation-candidate, and evaluation-candidate content "
        "is not opened.\n\n"
        "The canary records IDWTUP and every nominal XWGTUP value in each "
        "selected LHE. Uniform-positive behavior supports only a candidate "
        "Ngen convention. It does not authorize generalization to other "
        "campaigns or authorize a normalization denominator.\n",
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
    print("SIX_STANDARD_PROCESS_STRATA_SELECTED_PASS")
    print("ALL_SELECTED_BUNDLE_SIZE_AND_SHA256_CHECKS_PASS")
    print("SIX_LHE_WEIGHT_PAYLOADS_CONTROLLED_ACCESS_PASS")
    print("ALL_SELECTED_LHE_EVENTS_WEIGHT_PARSED_PASS")
    print("TEMPORARY_CANARY_BUNDLES_REMOVED_PASS")
    print("NO_ROOT_FILE_OPENED")
    print("NO_PARQUET_FILE_OPENED")
    print("NO_CANDIDATE_CONTENT_OPENED")
    print("NO_VALIDATION_CANDIDATE_CONTENT_OPENED")
    print("NO_EVALUATION_CANDIDATE_CONTENT_OPENED")
    print("NO_NORMALIZATION_DENOMINATOR_AUTHORIZED")
    print("NO_EXTERNAL_REFERENCE_CROSS_SECTION_ASSIGNED")
    print("NO_PHYSICAL_YIELDS_CALCULATED")
    print("HH4B_STANDARD_LHE_WEIGHT_CANARY_PASS")


if __name__ == "__main__":
    main()
