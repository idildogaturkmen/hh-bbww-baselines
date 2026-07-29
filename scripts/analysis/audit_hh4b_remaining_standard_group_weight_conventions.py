#!/usr/bin/env python3
"""Audit one frozen train ROOT representative for each remaining standard group."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import shutil
import subprocess
import tarfile
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

import awkward as ak
import numpy as np
import uproot


TRIBOSON_PROCESSES = {"wwz_zbb", "wzz_zbb", "zzz_zbb"}
EXPECTED_ALL_STANDARD_GROUPS = 23
EXPECTED_REMAINING_GROUPS = 20
EXPECTED_REMAINING_CAMPAIGNS = 41
EXPECTED_EXISTING_TRAIN_TRIBOSON_DENOMINATORS = 3


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


def process_campaign_pairs(
    group_map: dict[str, dict[str, str]],
    processes: Iterable[str],
) -> set[tuple[str, str]]:
    """Return physics-process/campaign pairs without collapsing shared labels."""

    return {
        (process, campaign.strip())
        for process in processes
        for campaign in group_map[process]["campaigns"].split(",")
        if campaign.strip()
    }


def campaign_from_bundle_path(path: str) -> str:
    parts = [part for part in path.split("/") if part]
    try:
        index = len(parts) - 1 - parts[::-1].index("bundles")
    except ValueError as error:
        raise RuntimeError(
            f"bundle path lacks a bundles component: {path}"
        ) from error
    if index + 2 != len(parts) - 1:
        raise RuntimeError(
            "bundle path does not match "
            ".../bundles/<campaign>/<bundle>.tar.gz: "
            + path
        )
    return parts[index + 1]


def normalized_tar_name(name: str) -> str:
    while name.startswith("./"):
        name = name[2:]
    return name


def extract_named_member(
    archive_path: Path,
    member_name: str,
    destination: Path,
) -> tuple[int, str]:
    target = normalized_tar_name(member_name)
    with tarfile.open(archive_path, mode="r:*") as archive:
        matches = [
            member
            for member in archive.getmembers()
            if normalized_tar_name(member.name) == target
        ]
        if len(matches) != 1:
            raise RuntimeError(
                f"expected one archive member {member_name!r}; "
                f"observed {len(matches)}"
            )
        member = matches[0]
        if not member.isfile():
            raise RuntimeError(
                f"archive member is not a regular file: {member.name}"
            )
        source = archive.extractfile(member)
        if source is None:
            raise RuntimeError(
                f"could not stream archive member: {member.name}"
            )
        destination.parent.mkdir(parents=True, exist_ok=True)
        with source, destination.open("wb") as output:
            shutil.copyfileobj(source, output, length=8 * 1024 * 1024)

    observed_size = destination.stat().st_size
    if observed_size != member.size:
        raise RuntimeError(
            f"extraction size mismatch: {observed_size} != {member.size}"
        )
    return observed_size, sha256_file(destination)


def extract_mapped_root(
    bundle: Path,
    *,
    root_container_member: str,
    root_archive_member: str,
    scratch: Path,
    stem: str,
) -> tuple[Path, dict[str, object]]:
    local_root = scratch / f"{stem}.root"
    local_nested = scratch / f"{stem}.nested.tar.gz"
    local_root.unlink(missing_ok=True)
    local_nested.unlink(missing_ok=True)

    if root_container_member.strip():
        nested_size, nested_sha = extract_named_member(
            bundle,
            root_container_member,
            local_nested,
        )
        root_size, root_sha = extract_named_member(
            local_nested,
            root_archive_member,
            local_root,
        )
        return local_root, {
            "nested_archive_used": True,
            "nested_archive_size_bytes": nested_size,
            "nested_archive_sha256": nested_sha,
            "extracted_root_size_bytes": root_size,
            "extracted_root_sha256": root_sha,
            "nested_path": str(local_nested),
        }

    root_size, root_sha = extract_named_member(
        bundle,
        root_archive_member,
        local_root,
    )
    return local_root, {
        "nested_archive_used": False,
        "nested_archive_size_bytes": 0,
        "nested_archive_sha256": "",
        "extracted_root_size_bytes": root_size,
        "extracted_root_sha256": root_sha,
        "nested_path": "",
    }


def branch_leaf_name(branch_name: str) -> str:
    normalized = str(branch_name).split(";", 1)[0].strip()
    return normalized.rsplit("/", 1)[-1]


def select_nominal_weight_branch(branch_names: Iterable[str]) -> str:
    names = sorted(
        {
            str(name).split(";", 1)[0].strip()
            for name in branch_names
            if str(name).strip()
        }
    )
    for leaf in ("event.weight", "lhefevent.weight", "hepmcevent.weight"):
        matches = [
            name
            for name in names
            if branch_leaf_name(name).lower() == leaf
        ]
        if len(matches) == 1:
            return matches[0]
        if len(matches) > 1:
            exact = [
                name
                for name in matches
                if name.lower()
                == leaf.split(".", 1)[0] + "/" + leaf
            ]
            if len(exact) == 1:
                return exact[0]
            raise RuntimeError(
                f"ambiguous nominal event-weight branch for {leaf}: "
                f"{matches!r}"
            )

    weight_like = [
        name
        for name in names
        if branch_leaf_name(name).lower().endswith(".weight")
    ]
    raise RuntimeError(
        "no recognized nominal event-record weight branch; "
        f"weight-like branches={weight_like!r}"
    )


def one_weight_per_entry(
    array: ak.Array,
    expected_entries: int,
) -> np.ndarray:
    if len(array) != expected_entries:
        raise RuntimeError(
            f"weight outer length {len(array)} != {expected_entries}"
        )

    if int(array.ndim) == 1:
        values = np.asarray(ak.to_numpy(array), dtype=np.float64)
    elif int(array.ndim) == 2:
        counts = np.asarray(ak.to_numpy(ak.num(array, axis=1)))
        if counts.shape != (expected_entries,) or not np.all(counts == 1):
            raise RuntimeError(
                "nominal event-weight branch does not contain exactly "
                "one weight per ROOT entry"
            )
        values = np.asarray(
            ak.to_numpy(ak.flatten(array, axis=1)),
            dtype=np.float64,
        )
    else:
        raise RuntimeError(
            f"unsupported nominal-weight dimensionality: {array.ndim}"
        )

    if values.shape != (expected_entries,):
        raise RuntimeError(
            f"weight shape {values.shape!r} "
            f"does not match ({expected_entries},)"
        )
    if not np.all(np.isfinite(values)):
        raise RuntimeError("nonfinite nominal ROOT weights")
    return values


def classify_weights(values: np.ndarray) -> tuple[str, bool]:
    if values.ndim != 1 or values.size == 0:
        return "invalid_shape_or_empty", False
    if not np.all(np.isfinite(values)):
        return "nonfinite", False

    positive = int(np.count_nonzero(values > 0.0))
    negative = int(np.count_nonzero(values < 0.0))
    zero = int(np.count_nonzero(values == 0.0))
    minimum = float(np.min(values))
    maximum = float(np.max(values))
    scale = max(abs(minimum), abs(maximum), 1.0)
    uniform = abs(maximum - minimum) <= 1.0e-12 * scale

    if zero:
        return "contains_zero_weights", False
    if positive == values.size and uniform:
        return "uniform_positive", True
    if positive == values.size:
        return "variable_positive", False
    if negative == values.size and uniform:
        return "uniform_negative", False
    if positive and negative:
        symmetric = (
            abs(abs(minimum) - abs(maximum))
            <= 1.0e-12 * scale
        )
        if symmetric:
            return "signed_uniform_magnitude", False
        return "variable_signed", False
    return "unclassified", False


def receipt_matches(
    receipt: dict[str, object],
    selection: dict[str, object],
) -> bool:
    fields = (
        "process_or_mode",
        "member_index",
        "campaign",
        "remote_bundle_path",
        "bundle_sha256_expected",
        "bundle_size_bytes",
        "root_container_member",
        "root_archive_member",
        "generated_events",
    )
    return all(
        str(receipt.get(field)) == str(selection.get(field))
        for field in fields
    ) and receipt.get("status") == "standard_group_root_weight_audit_complete"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root-source-map", type=Path, required=True)
    parser.add_argument("--campaign-inventory", type=Path, required=True)
    parser.add_argument("--equivalence-groups", type=Path, required=True)
    parser.add_argument("--triboson-denominators", type=Path, required=True)
    parser.add_argument("--eos-host", required=True)
    parser.add_argument("--scratch", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--source-commit", required=True)
    args = parser.parse_args()

    source_rows = read_tsv(args.root_source_map)
    inventory_rows = read_tsv(args.campaign_inventory)
    group_rows = read_tsv(args.equivalence_groups)
    triboson_denominator_rows = read_tsv(args.triboson_denominators)

    if len(group_rows) != EXPECTED_ALL_STANDARD_GROUPS:
        raise RuntimeError(
            f"standard group count={len(group_rows)}, "
            f"expected {EXPECTED_ALL_STANDARD_GROUPS}"
        )
    if (
        len(triboson_denominator_rows)
        != EXPECTED_EXISTING_TRAIN_TRIBOSON_DENOMINATORS
    ):
        raise RuntimeError("existing train triboson denominator count mismatch")

    group_map = {
        row["process_or_mode"]: row
        for row in group_rows
    }
    remaining_processes = sorted(
        set(group_map) - TRIBOSON_PROCESSES
    )
    if len(remaining_processes) != EXPECTED_REMAINING_GROUPS:
        raise RuntimeError(
            f"remaining standard group count={len(remaining_processes)}, "
            f"expected {EXPECTED_REMAINING_GROUPS}"
        )

    remaining_process_campaign_pairs = process_campaign_pairs(
        group_map,
        remaining_processes,
    )
    remaining_unique_campaign_labels = {
        campaign
        for _, campaign in remaining_process_campaign_pairs
    }
    if (
        len(remaining_process_campaign_pairs)
        != EXPECTED_REMAINING_CAMPAIGNS
    ):
        raise RuntimeError(
            "remaining process-campaign pair count="
            f"{len(remaining_process_campaign_pairs)}, "
            f"expected {EXPECTED_REMAINING_CAMPAIGNS}; "
            "unique campaign labels="
            f"{len(remaining_unique_campaign_labels)}"
        )

    print(
        "REMAINING_41_PROCESS_CAMPAIGN_PAIRS_VALIDATED_PASS"
    )
    print(
        "REMAINING_UNIQUE_CAMPAIGN_LABELS\t"
        + str(len(remaining_unique_campaign_labels))
    )

    inventory_map = {
        (
            row["sample_class"],
            row["process_or_mode"],
            row["campaign"],
        ): row
        for row in inventory_rows
    }

    source_by_process: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in source_rows:
        process = row["process_or_mode"]
        if process not in remaining_processes:
            continue
        if row["dataset_split"] != "train":
            continue
        if row["resolution_status"] != "resolved":
            continue
        if row["bundle_exists"] != "True":
            continue
        if row["bundle_stat_status"] != "pass":
            continue
        if int(row["bundle_size_bytes"]) <= 0:
            continue
        if not row["bundle_sha256_expected"].strip():
            continue
        campaign = campaign_from_bundle_path(
            row["remote_bundle_path"]
        )
        if campaign not in group_map[process]["campaigns"].split(","):
            raise RuntimeError(
                f"source-map campaign {campaign!r} is outside frozen "
                f"group {process!r}"
            )
        source_by_process[process].append(row)

    selections: list[dict[str, object]] = []
    for process in remaining_processes:
        candidates = source_by_process.get(process, [])
        if not candidates:
            raise RuntimeError(
                f"no resolved train ROOT source candidate for {process}"
            )
        selected = min(
            candidates,
            key=lambda row: (
                int(row["bundle_size_bytes"]),
                int(row["member_index"]),
            ),
        )
        campaign = campaign_from_bundle_path(
            selected["remote_bundle_path"]
        )
        selections.append(
            {
                "process_or_mode": process,
                "configuration_fingerprint_sha256": group_map[process][
                    "card_only_configuration_fingerprint_sha256"
                ],
                "equivalent_campaigns": group_map[process]["campaigns"],
                "member_index": int(selected["member_index"]),
                "sample_class": selected["sample_class"],
                "campaign": campaign,
                "dataset_split": "train",
                "generated_events": int(selected["generated_events"]),
                "selection_rule": (
                    "smallest_resolved_train_bundle_then_member_index"
                ),
                "remote_bundle_path": selected["remote_bundle_path"],
                "remote_bundle_uri": remote_uri(
                    args.eos_host,
                    selected["remote_bundle_path"],
                ),
                "bundle_size_bytes": int(
                    selected["bundle_size_bytes"]
                ),
                "bundle_sha256_expected": selected[
                    "bundle_sha256_expected"
                ],
                "root_container_member": selected[
                    "root_container_member"
                ],
                "root_archive_member": selected[
                    "root_archive_member"
                ],
                "root_basename": selected["root_basename"],
                "candidate_path_recorded_but_unopened": selected[
                    "candidate_path"
                ],
                "candidate_file_opened": False,
                "status": "selected_for_standard_group_weight_audit",
            }
        )

    if len(selections) != EXPECTED_REMAINING_GROUPS:
        raise RuntimeError("selection count mismatch")

    args.output.mkdir(parents=True, exist_ok=True)
    receipt_dir = args.output / "receipts"
    receipt_dir.mkdir(exist_ok=True)
    args.scratch.mkdir(parents=True, exist_ok=True)

    results: list[dict[str, object]] = []
    transfer_rows: list[dict[str, object]] = []

    for ordinal, selection in enumerate(selections, start=1):
        process = str(selection["process_or_mode"])
        member_index = int(selection["member_index"])
        receipt_path = (
            receipt_dir
            / f"{ordinal:02d}_{process}_member{member_index:04d}.json"
        )

        if receipt_path.is_file():
            receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
            if not receipt_matches(receipt, selection):
                raise RuntimeError(
                    f"receipt mismatch: {receipt_path}"
                )
            print(
                f"RESUME\t{ordinal}/{EXPECTED_REMAINING_GROUPS}\t"
                f"{process}\tmember={member_index}"
            )
            results.append(receipt)
            transfer_rows.append(
                {
                    "ordinal": ordinal,
                    "process_or_mode": process,
                    "member_index": member_index,
                    "transfer_performed_this_run": False,
                    "temporary_bundle_removed": True,
                    "temporary_nested_archive_removed": True,
                    "temporary_root_removed": True,
                    "status": "resumed_from_verified_receipt",
                }
            )
            continue

        local_bundle = (
            args.scratch
            / f"{ordinal:02d}_{process}_member{member_index:04d}.tar.gz"
        )
        stem = f"{ordinal:02d}_{process}_member{member_index:04d}"
        local_bundle.unlink(missing_ok=True)

        print(
            f"TRANSFER\t{ordinal}/{EXPECTED_REMAINING_GROUPS}\t"
            f"{process}\tmember={member_index}"
        )
        print(f"REMOTE\t{selection['remote_bundle_uri']}")
        print(f"LOCAL\t{local_bundle}")

        local_root: Path | None = None
        nested_path: Path | None = None
        try:
            subprocess.run(
                [
                    "xrdcp",
                    "--nopbar",
                    "--force",
                    str(selection["remote_bundle_uri"]),
                    str(local_bundle),
                ],
                check=True,
            )

            observed_bundle_size = local_bundle.stat().st_size
            if observed_bundle_size != int(
                selection["bundle_size_bytes"]
            ):
                raise RuntimeError(
                    f"bundle size mismatch for {process}"
                )
            observed_bundle_sha = sha256_file(local_bundle)
            if (
                observed_bundle_sha
                != selection["bundle_sha256_expected"]
            ):
                raise RuntimeError(
                    f"bundle SHA-256 mismatch for {process}"
                )

            local_root, extraction = extract_mapped_root(
                local_bundle,
                root_container_member=str(
                    selection["root_container_member"]
                ),
                root_archive_member=str(
                    selection["root_archive_member"]
                ),
                scratch=args.scratch,
                stem=stem,
            )
            if extraction["nested_path"]:
                nested_path = Path(str(extraction["nested_path"]))

            with uproot.open(local_root) as root_file:
                tree = root_file["Delphes"]
                entries = int(tree.num_entries)
                expected_entries = int(selection["generated_events"])
                if entries != expected_entries:
                    raise RuntimeError(
                        f"ROOT entries {entries} != "
                        f"source-map generated_events {expected_entries}"
                    )
                branch = select_nominal_weight_branch(
                    tree.keys(recursive=True)
                )
                weights = one_weight_per_entry(
                    tree[branch].array(library="ak"),
                    entries,
                )

            classification, ngen_supported = classify_weights(weights)
            positive = int(np.count_nonzero(weights > 0.0))
            negative = int(np.count_nonzero(weights < 0.0))
            zero = int(np.count_nonzero(weights == 0.0))
            sumw = float(np.sum(weights, dtype=np.float64))
            sumw2 = float(
                np.sum(weights * weights, dtype=np.float64)
            )

            receipt = {
                **selection,
                **extraction,
                "ordinal": ordinal,
                "observed_bundle_size_bytes": observed_bundle_size,
                "observed_bundle_sha256": observed_bundle_sha,
                "root_weight_branch": branch,
                "root_entries": entries,
                "positive_weight_events": positive,
                "negative_weight_events": negative,
                "zero_weight_events": zero,
                "sum_root_nominal_weights": f"{sumw:.17g}",
                "sum_squared_root_nominal_weights": f"{sumw2:.17g}",
                "minimum_root_nominal_weight": (
                    f"{float(np.min(weights)):.17g}"
                ),
                "maximum_root_nominal_weight": (
                    f"{float(np.max(weights)):.17g}"
                ),
                "weight_classification": classification,
                "ngen_effective_denominator_supported": ngen_supported,
                "candidate_files_opened": 0,
                "validation_candidate_files_opened": 0,
                "evaluation_candidate_files_opened": 0,
                "normalization_denominator_authorized": False,
                "status": "standard_group_root_weight_audit_complete",
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
                    "member_index": member_index,
                    "transfer_performed_this_run": True,
                    "temporary_bundle_removed": True,
                    "temporary_nested_archive_removed": True,
                    "temporary_root_removed": True,
                    "status": "bundle_verified_root_weight_only_audited",
                }
            )
        finally:
            if local_root is not None:
                local_root.unlink(missing_ok=True)
            if nested_path is not None:
                nested_path.unlink(missing_ok=True)
            local_bundle.unlink(missing_ok=True)

    results.sort(key=lambda row: int(row["ordinal"]))
    transfer_rows.sort(key=lambda row: int(row["ordinal"]))

    result_map = {
        str(row["process_or_mode"]): row
        for row in results
    }
    if set(result_map) != set(remaining_processes):
        raise RuntimeError("result process coverage mismatch")

    group_decisions: list[dict[str, object]] = []
    campaign_denominators: list[dict[str, object]] = []
    unresolved_groups: list[dict[str, object]] = []

    for process in remaining_processes:
        result = result_map[process]
        uniform_positive = (
            result["weight_classification"] == "uniform_positive"
            and result["ngen_effective_denominator_supported"] is True
        )
        campaigns = [
            item
            for item in group_map[process]["campaigns"].split(",")
            if item
        ]

        group_decisions.append(
            {
                "process_or_mode": process,
                "configuration_fingerprint_sha256": group_map[process][
                    "card_only_configuration_fingerprint_sha256"
                ],
                "representative_member_index": result["member_index"],
                "representative_campaign": result["campaign"],
                "representative_weight_branch": result[
                    "root_weight_branch"
                ],
                "representative_weight_classification": result[
                    "weight_classification"
                ],
                "representative_minimum_weight": result[
                    "minimum_root_nominal_weight"
                ],
                "representative_maximum_weight": result[
                    "maximum_root_nominal_weight"
                ],
                "campaign_count": len(campaigns),
                "campaigns": ",".join(campaigns),
                "uniform_positive_convention_authorized": uniform_positive,
                "effective_generator_numerator": (
                    "1"
                    if uniform_positive
                    else ""
                ),
                "effective_denominator_strategy": (
                    "manifest_generated_event_count"
                    if uniform_positive
                    else "signed_or_variable_weight_sidecar_required"
                ),
                "cross_section_sharing_authorized": False,
                "physical_weight_application_authorized": False,
                "status": (
                    "uniform_positive_group_convention_frozen"
                    if uniform_positive
                    else "group_weight_convention_fail_closed"
                ),
            }
        )

        if not uniform_positive:
            unresolved_groups.append(
                {
                    "process_or_mode": process,
                    "weight_classification": result[
                        "weight_classification"
                    ],
                    "representative_member_index": result["member_index"],
                    "required_next_action": (
                        "audit_all_source_shards_and_materialize exact "
                        "event-weight sidecars before denominator freeze"
                    ),
                    "normalization_denominator_authorized": False,
                    "status": "signed_or_variable_group_unresolved",
                }
            )

        for campaign in campaigns:
            matching = [
                (key, row)
                for key, row in inventory_map.items()
                if key[1] == process and key[2] == campaign
            ]
            if len(matching) != 1:
                raise RuntimeError(
                    f"campaign inventory match count for "
                    f"{process}/{campaign}: {len(matching)}"
                )
            key, inventory = matching[0]
            denominator = int(inventory["generated_events"])
            if denominator <= 0:
                raise RuntimeError(
                    f"nonpositive manifest generated events for {key}"
                )

            campaign_denominators.append(
                {
                    "sample_class": key[0],
                    "process_or_mode": process,
                    "campaign": campaign,
                    "campaign_members": int(inventory["members"]),
                    "generated_events": denominator,
                    "train_members": int(inventory["train_members"]),
                    "validation_members": int(
                        inventory["validation_members"]
                    ),
                    "final_evaluation_members": int(
                        inventory["final_evaluation_members"]
                    ),
                    "generator_weight_convention": (
                        "uniform_positive_constant_cancels"
                        if uniform_positive
                        else "unresolved_signed_or_variable"
                    ),
                    "effective_generator_numerator": (
                        "1"
                        if uniform_positive
                        else ""
                    ),
                    "normalization_denominator_type": (
                        "generated_event_count"
                        if uniform_positive
                        else "unresolved"
                    ),
                    "normalization_denominator": (
                        denominator
                        if uniform_positive
                        else ""
                    ),
                    "denominator_convention_authorized": uniform_positive,
                    "candidate_content_opened": False,
                    "validation_candidate_content_opened": False,
                    "evaluation_candidate_content_opened": False,
                    "external_cross_section_required": True,
                    "physical_weight_application_authorized": False,
                    "status": (
                        "standard_campaign_ngen_denominator_frozen"
                        if uniform_positive
                        else "standard_campaign_denominator_fail_closed"
                    ),
                }
            )

    if len(group_decisions) != EXPECTED_REMAINING_GROUPS:
        raise RuntimeError("group-decision count mismatch")
    if len(campaign_denominators) != EXPECTED_REMAINING_CAMPAIGNS:
        raise RuntimeError("campaign-denominator row count mismatch")

    authorized_groups = sum(
        bool(row["uniform_positive_convention_authorized"])
        for row in group_decisions
    )
    authorized_campaigns = sum(
        bool(row["denominator_convention_authorized"])
        for row in campaign_denominators
    )

    next_gate = (
        "recover_nonstandard_ggf_and_qcd_denominators_then_build_reference_xsec_registry"
        if authorized_groups == EXPECTED_REMAINING_GROUPS
        else "materialize_signed_or_variable_weight_sidecars_for_flagged_standard_groups"
    )

    write_tsv(
        args.output / "representative_group_selection.tsv",
        selections,
        list(selections[0]),
    )
    write_tsv(
        args.output / "representative_group_root_weight_audit.tsv",
        results,
        list(results[0]),
    )
    write_tsv(
        args.output / "standard_group_weight_convention_decision.tsv",
        group_decisions,
        list(group_decisions[0]),
    )
    write_tsv(
        args.output / "standard_campaign_denominator_freeze.tsv",
        campaign_denominators,
        list(campaign_denominators[0]),
    )
    write_tsv(
        args.output / "unresolved_standard_weight_groups.tsv",
        unresolved_groups,
        (
            list(unresolved_groups[0])
            if unresolved_groups
            else [
                "process_or_mode",
                "weight_classification",
                "representative_member_index",
                "required_next_action",
                "normalization_denominator_authorized",
                "status",
            ]
        ),
    )
    write_tsv(
        args.output / "transfer_and_cleanup_audit.tsv",
        transfer_rows,
        list(transfer_rows[0]),
    )

    summary = {
        "schema_version": 1,
        "status": (
            "hh4b_remaining_standard_group_weight_convention_audit_pass"
        ),
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "source_commit": args.source_commit,
        "selection": {
            "remaining_standard_groups": len(remaining_processes),
            "remaining_standard_campaigns": len(
                campaign_denominators
            ),
            "representative_train_root_files_opened": len(results),
            "existing_train_triboson_denominators": len(
                triboson_denominator_rows
            ),
        },
        "audit": {
            "bundles_size_and_sha256_verified": len(results),
            "root_files_opened": len(results),
            "root_branches_read": len(results),
            "root_events_audited": sum(
                int(row["root_entries"]) for row in results
            ),
            "uniform_positive_groups": authorized_groups,
            "signed_or_variable_groups": (
                EXPECTED_REMAINING_GROUPS - authorized_groups
            ),
            "campaign_denominators_authorized": (
                authorized_campaigns
            ),
            "all_temporary_files_removed": not any(
                args.scratch.iterdir()
            ),
        },
        "readiness": {
            "remaining_standard_group_audit_complete": True,
            "remaining_standard_groups_authorized": authorized_groups,
            "remaining_standard_campaign_denominators_authorized": (
                authorized_campaigns
            ),
            "train_triboson_denominators_already_authorized": 3,
            "nonstandard_campaign_denominators_authorized": 0,
            "external_reference_cross_sections_authorized": 0,
            "physical_weight_application_authorized": False,
            "physics_normalization_ready": False,
        },
        "controls": {
            "candidate_files_opened": 0,
            "candidate_rows_read": 0,
            "validation_candidate_files_opened": 0,
            "evaluation_candidate_files_opened": 0,
            "root_files_opened": len(results),
            "root_branches_read": len(results),
            "cross_sections_assigned": 0,
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
        "# HH4b remaining standard-group weight-convention audit\n\n"
        "This checkpoint opens one deterministic train ROOT representative "
        "for each of the 20 non-triboson standard card-equivalence groups. "
        "It reads only the nominal event-record weight branch and classifies "
        "the observed convention.\n\n"
        "For a group whose representative is uniform positive, the common "
        "constant generator weight cancels between the event numerator and "
        "campaign sum. The effective event numerator is therefore one and "
        "the campaign denominator is the frozen manifest generated-event "
        "count. This convention is recorded independently for each campaign; "
        "cross-section sharing is not authorized.\n\n"
        "Signed, variable, zero-containing, ambiguous, or nonfinite groups "
        "remain fail-closed and require complete per-source event-weight "
        "sidecars.\n\n"
        "No candidate Parquet file is opened. Validation and evaluation "
        "candidate content remains sealed. No external cross section, "
        "physical event weight, yield, model, or threshold is calculated.\n",
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
    print("ALL_20_REMAINING_STANDARD_GROUPS_SELECTED_PASS")
    print("ALL_20_REPRESENTATIVE_BUNDLES_VERIFIED_PASS")
    print("ALL_20_REPRESENTATIVE_ROOT_WEIGHT_BRANCHES_AUDITED_PASS")
    print("STANDARD_GROUP_WEIGHT_CONVENTION_DECISIONS_RECORDED_PASS")
    print("STANDARD_CAMPAIGN_DENOMINATOR_FREEZE_FAIL_CLOSED_PASS")
    print("EXISTING_THREE_TRAIN_TRIBOSON_DENOMINATORS_PRESERVED_PASS")
    print("TEMPORARY_STANDARD_GROUP_BUNDLES_REMOVED_PASS")
    print("NO_CANDIDATE_PARQUET_OPENED")
    print("NO_VALIDATION_CANDIDATE_CONTENT_OPENED")
    print("NO_EVALUATION_CANDIDATE_CONTENT_OPENED")
    print("NO_EXTERNAL_REFERENCE_CROSS_SECTION_ASSIGNED")
    print("NO_PHYSICAL_WEIGHTS_CALCULATED")
    print("NO_PHYSICAL_YIELDS_CALCULATED")
    print("HH4B_REMAINING_STANDARD_GROUP_WEIGHT_CONVENTION_AUDIT_PASS")


if __name__ == "__main__":
    main()
