#!/usr/bin/env python3
"""Audit signed nominal-weight transport in all used triboson ROOT shards."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
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


TRIBOSON_PROCESSES = ("wwz_zbb", "wzz_zbb", "zzz_zbb")
EXPECTED_BUNDLES = 15
EXPECTED_CAMPAIGNS = 6
EXPECTED_PROCESSES = 3
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


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


def require_columns(
    rows: list[dict[str, str]],
    required: set[str],
    label: str,
) -> None:
    if not rows:
        raise RuntimeError(f"{label} is empty")
    observed = set(rows[0])
    missing = sorted(required - observed)
    if missing:
        raise RuntimeError(
            f"{label} is missing required columns: {missing}; "
            f"observed columns={sorted(observed)}"
        )


def campaign_from_source_row(row: dict[str, str]) -> str:
    """Resolve the campaign from the frozen EOS bundle locator.

    The member-level ROOT source map does not contain a ``bundle_campaign``
    column. Its campaign is the directory immediately below ``bundles/`` in
    ``remote_bundle_path``. An explicit future column is accepted only when
    it agrees with the path-derived value.
    """

    remote_path = row.get("remote_bundle_path", "").strip()
    if not remote_path:
        raise RuntimeError("source-map row has an empty remote_bundle_path")

    parts = [part for part in remote_path.split("/") if part]
    try:
        bundles_index = len(parts) - 1 - parts[::-1].index("bundles")
    except ValueError as error:
        raise RuntimeError(
            "remote_bundle_path does not contain a bundles directory: "
            + remote_path
        ) from error

    if bundles_index + 2 != len(parts) - 1:
        raise RuntimeError(
            "remote_bundle_path is not in the expected "
            ".../bundles/<campaign>/<bundle>.tar.gz layout: "
            + remote_path
        )

    derived = parts[bundles_index + 1]
    explicit = row.get("bundle_campaign", "").strip()
    if explicit and explicit != derived:
        raise RuntimeError(
            "source-map campaign disagreement: "
            f"bundle_campaign={explicit!r}, path-derived={derived!r}"
        )
    return explicit or derived


def remote_uri(host: str, locator: str) -> str:
    return host.rstrip("/") + "//" + locator.lstrip("/")


def normalized_tar_name(name: str) -> str:
    while name.startswith("./"):
        name = name[2:]
    return name


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


def branch_leaf_name(branch_name: str) -> str:
    """Return the split-branch leaf without a ROOT cycle suffix."""

    normalized = str(branch_name).split(";", 1)[0].strip()
    return normalized.rsplit("/", 1)[-1]


def select_nominal_weight_branch(branch_names: Iterable[str]) -> str:
    """Select the nominal Delphes event weight deterministically.

    Uproot exposes split Delphes branches with names such as
    ``Event/Event.Weight``. A separate ``Weight/Weight.Weight`` collection
    may also be present and is not used as the nominal event header weight.
    The selector therefore ranks the event-record branch explicitly rather
    than relying on a generic ``.Weight`` suffix.
    """

    names = sorted(
        {
            str(name).split(";", 1)[0].strip()
            for name in branch_names
            if str(name).strip()
        }
    )

    preferred_leafs = (
        "event.weight",
        "lhefevent.weight",
        "hepmcevent.weight",
    )

    for preferred_leaf in preferred_leafs:
        matches = [
            name
            for name in names
            if branch_leaf_name(name).lower() == preferred_leaf
        ]
        if len(matches) == 1:
            return matches[0]
        if len(matches) > 1:
            exact_split = [
                name
                for name in matches
                if name.lower()
                == preferred_leaf.split(".", 1)[0]
                + "/"
                + preferred_leaf
            ]
            if len(exact_split) == 1:
                return exact_split[0]
            raise RuntimeError(
                "multiple branches match the preferred nominal event "
                f"weight leaf {preferred_leaf!r}: {matches!r}"
            )

    weight_like = [
        name
        for name in names
        if branch_leaf_name(name).lower().endswith(".weight")
    ]

    raise RuntimeError(
        "no recognized nominal event-record weight branch was found; "
        f"weight-like branches={weight_like!r}"
    )


def awkward_to_one_weight_per_entry(
    array: ak.Array,
    expected_entries: int,
) -> np.ndarray:
    if len(array) != expected_entries:
        raise RuntimeError(
            f"weight array outer length {len(array)} "
            f"does not match tree entries {expected_entries}"
        )

    ndim = int(array.ndim)
    if ndim == 1:
        values = np.asarray(ak.to_numpy(array), dtype=np.float64)
    elif ndim == 2:
        counts = np.asarray(ak.to_numpy(ak.num(array, axis=1)))
        if counts.shape != (expected_entries,) or not np.all(counts == 1):
            raise RuntimeError(
                "nominal weight branch does not contain exactly one "
                "weight per Delphes entry"
            )
        values = np.asarray(
            ak.to_numpy(ak.flatten(array, axis=1)),
            dtype=np.float64,
        )
    else:
        raise RuntimeError(
            f"unsupported nominal-weight array dimensionality: {ndim}"
        )

    if values.shape != (expected_entries,):
        raise RuntimeError(
            f"flattened weight shape {values.shape!r} "
            f"does not match ({expected_entries},)"
        )
    return values


def extract_root_member(
    bundle: Path,
    member_name: str,
    destination: Path,
) -> tuple[int, str]:
    target = normalized_tar_name(member_name)

    with tarfile.open(bundle, mode="r:*") as archive:
        matches = [
            member
            for member in archive.getmembers()
            if normalized_tar_name(member.name) == target
        ]
        if len(matches) != 1:
            raise RuntimeError(
                f"expected exactly one ROOT member {member_name!r}; "
                f"observed {len(matches)}"
            )
        member = matches[0]
        if not member.isfile():
            raise RuntimeError(
                f"ROOT archive member is not a regular file: {member.name}"
            )

        source = archive.extractfile(member)
        if source is None:
            raise RuntimeError(
                f"could not stream ROOT archive member: {member.name}"
            )

        destination.parent.mkdir(parents=True, exist_ok=True)
        with source, destination.open("wb") as output:
            shutil.copyfileobj(source, output, length=8 * 1024 * 1024)

    observed_size = destination.stat().st_size
    if observed_size != member.size:
        raise RuntimeError(
            f"ROOT extraction size mismatch: "
            f"{observed_size} != {member.size}"
        )
    return observed_size, sha256_file(destination)


def audit_root_weight(
    root_path: Path,
    expected_entries: int,
) -> dict[str, object]:
    with uproot.open(root_path) as root_file:
        try:
            tree = root_file["Delphes"]
        except KeyError as error:
            raise RuntimeError("Delphes tree is absent") from error

        entries = int(tree.num_entries)
        if entries != expected_entries:
            raise RuntimeError(
                f"Delphes entry count {entries} "
                f"does not match expected {expected_entries}"
            )

        branch_names = tree.keys(recursive=True)
        branch = select_nominal_weight_branch(branch_names)
        typename = str(tree[branch].typename)
        array = tree[branch].array(library="ak")
        values = awkward_to_one_weight_per_entry(array, entries)

    classification, ngen_supported = classify_weights(values)
    positive = int(np.count_nonzero(values > 0.0))
    negative = int(np.count_nonzero(values < 0.0))
    zero = int(np.count_nonzero(values == 0.0))

    return {
        "delphes_entries": entries,
        "nominal_weight_branch": branch,
        "nominal_weight_typename": typename,
        "positive_weight_events": positive,
        "negative_weight_events": negative,
        "zero_weight_events": zero,
        "sum_root_nominal_weights": f"{float(np.sum(values, dtype=np.float64)):.17g}",
        "sum_squared_root_nominal_weights": (
            f"{float(np.sum(values * values, dtype=np.float64)):.17g}"
        ),
        "minimum_root_nominal_weight": f"{float(np.min(values)):.17g}",
        "maximum_root_nominal_weight": f"{float(np.max(values)):.17g}",
        "root_weight_classification": classification,
        "ngen_denominator_candidate_supported": ngen_supported,
        "signed_weight_transport_observed": negative > 0,
    }


def receipt_matches(
    receipt: dict[str, object],
    selection: dict[str, object],
) -> bool:
    identifying = (
        "member_index",
        "sample_class",
        "process_or_mode",
        "campaign",
        "dataset_split",
        "generated_events",
        "remote_bundle_path",
        "bundle_sha256_expected",
        "bundle_size_bytes",
        "root_archive_member",
        "candidate_path",
    )
    return all(
        str(receipt.get(field)) == str(selection.get(field))
        for field in identifying
    ) and receipt.get("status") == "triboson_root_weight_audit_complete"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root-source-map", type=Path, required=True)
    parser.add_argument("--standard-status", type=Path, required=True)
    parser.add_argument("--canary-audit", type=Path, required=True)
    parser.add_argument("--eos-host", required=True)
    parser.add_argument("--scratch", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--source-commit", required=True)
    args = parser.parse_args()

    source_rows = read_tsv(args.root_source_map)
    status_rows = read_tsv(args.standard_status)
    canary_rows = read_tsv(args.canary_audit)

    require_columns(
        source_rows,
        {
            "member_index",
            "sample_class",
            "process_or_mode",
            "dataset_split",
            "generated_events",
            "candidate_rows",
            "candidate_path",
            "candidate_sha256",
            "remote_bundle_path",
            "bundle_sha256_expected",
            "bundle_exists",
            "bundle_size_bytes",
            "bundle_stat_status",
            "root_container_member",
            "root_archive_member",
            "root_basename",
            "archive_depth",
            "resolution_status",
        },
        "member-level ROOT source map",
    )
    require_columns(
        status_rows,
        {
            "sample_class",
            "process_or_mode",
            "campaign",
            "manifest_generated_events",
        },
        "standard campaign weight-proof status",
    )
    require_columns(
        canary_rows,
        {
            "process_or_mode",
            "weight_classification",
        },
        "standard LHE-weight canary audit",
    )

    triboson_status = [
        row
        for row in status_rows
        if row["process_or_mode"] in TRIBOSON_PROCESSES
    ]
    if len(triboson_status) != EXPECTED_CAMPAIGNS:
        raise RuntimeError(
            f"expected {EXPECTED_CAMPAIGNS} triboson campaign status rows, "
            f"observed {len(triboson_status)}"
        )

    canary_wwz = [
        row
        for row in canary_rows
        if row["process_or_mode"] == "wwz_zbb"
    ]
    if len(canary_wwz) != 1:
        raise RuntimeError("expected one WWZ LHE canary row")
    if canary_wwz[0]["weight_classification"] != "signed_uniform_magnitude":
        raise RuntimeError(
            "WWZ LHE canary is not the expected signed-uniform sample"
        )

    selected_by_locator: dict[
        tuple[str, str],
        dict[str, str],
    ] = {}

    for row in source_rows:
        if row["process_or_mode"] not in TRIBOSON_PROCESSES:
            continue

        if row["sample_class"] != "background":
            raise RuntimeError("triboson source row is not background")
        if row["resolution_status"] != "resolved":
            raise RuntimeError(
                f"unresolved triboson source row: {row['member_index']}"
            )
        if row["bundle_exists"] != "True":
            raise RuntimeError(
                f"triboson bundle is not proven present: {row['member_index']}"
            )
        if row["bundle_stat_status"] != "pass":
            raise RuntimeError(
                f"triboson bundle stat did not pass: {row['member_index']}"
            )
        if row["archive_depth"] != "1":
            raise RuntimeError(
                f"unexpected triboson archive depth: {row['archive_depth']}"
            )
        if row["root_container_member"].strip():
            raise RuntimeError(
                "triboson ROOT source unexpectedly has a nested container"
            )
        if not row["root_archive_member"].lower().endswith(".root"):
            raise RuntimeError("triboson ROOT member name is invalid")
        if not SHA256_RE.fullmatch(row["bundle_sha256_expected"]):
            raise RuntimeError("missing or invalid frozen bundle SHA-256")
        if int(row["bundle_size_bytes"]) <= 0:
            raise RuntimeError("invalid frozen bundle size")
        if int(row["generated_events"]) <= 0:
            raise RuntimeError("invalid expected generated-event count")

        key = (row["remote_bundle_path"], row["root_archive_member"])
        previous = selected_by_locator.get(key)
        if previous is not None and previous != row:
            raise RuntimeError(
                "duplicate triboson source locator has inconsistent rows"
            )
        selected_by_locator[key] = row

    raw_selections = list(selected_by_locator.values())
    if len(raw_selections) != EXPECTED_BUNDLES:
        raise RuntimeError(
            f"expected {EXPECTED_BUNDLES} triboson source bundles, "
            f"observed {len(raw_selections)}"
        )

    status_map = {
        (
            row["sample_class"],
            row["process_or_mode"],
            row["campaign"],
        ): row
        for row in triboson_status
    }

    grouped_source: dict[tuple[str, str, str], list[dict[str, str]]] = (
        defaultdict(list)
    )
    for row in raw_selections:
        campaign = campaign_from_source_row(row)
        key = (
            row["sample_class"],
            row["process_or_mode"],
            campaign,
        )
        grouped_source[key].append(row)

    if set(grouped_source) != set(status_map):
        raise RuntimeError(
            "triboson campaign coverage mismatch between source map and "
            "normalization status"
        )

    selections: list[dict[str, object]] = []
    for key, rows in sorted(grouped_source.items()):
        status = status_map[key]
        manifest_total = int(status["manifest_generated_events"])
        source_total = sum(int(row["generated_events"]) for row in rows)
        if source_total != manifest_total:
            raise RuntimeError(
                f"source-map total mismatch for {key}: "
                f"{source_total} != {manifest_total}"
            )

        for row in sorted(
            rows,
            key=lambda item: (
                item["dataset_split"],
                item["remote_bundle_path"],
            ),
        ):
            selections.append(
                {
                    "member_index": int(row["member_index"]),
                    "sample_class": row["sample_class"],
                    "process_or_mode": row["process_or_mode"],
                    "campaign": key[2],
                    "dataset_split": row["dataset_split"],
                    "generated_events": int(row["generated_events"]),
                    "candidate_rows_recorded_in_prior_source_map": int(
                        row["candidate_rows"]
                    ),
                    "candidate_path": row["candidate_path"],
                    "candidate_sha256": row["candidate_sha256"],
                    "remote_bundle_path": row["remote_bundle_path"],
                    "remote_bundle_uri": remote_uri(
                        args.eos_host,
                        row["remote_bundle_path"],
                    ),
                    "bundle_sha256_expected": row[
                        "bundle_sha256_expected"
                    ],
                    "bundle_size_bytes": int(row["bundle_size_bytes"]),
                    "root_archive_member": row["root_archive_member"],
                    "root_basename": row["root_basename"],
                    "campaign_manifest_generated_events": manifest_total,
                    "candidate_parquet_access_authorized": False,
                    "root_nominal_weight_access_authorized_for_this_audit": (
                        True
                    ),
                    "normalization_denominator_authorized": False,
                    "status": "selected_for_triboson_root_weight_audit",
                }
            )

    if len(selections) != EXPECTED_BUNDLES:
        raise RuntimeError("triboson selection count changed unexpectedly")

    args.output.mkdir(parents=True, exist_ok=True)
    receipt_dir = args.output / "receipts"
    receipt_dir.mkdir(exist_ok=True)
    args.scratch.mkdir(parents=True, exist_ok=True)

    results: list[dict[str, object]] = []
    transfer_rows: list[dict[str, object]] = []

    for ordinal, selection in enumerate(selections, start=1):
        process = str(selection["process_or_mode"])
        campaign = str(selection["campaign"])
        member_index = int(selection["member_index"])
        receipt_path = (
            receipt_dir
            / f"{ordinal:02d}_{process}_member{member_index:04d}.json"
        )

        if receipt_path.is_file():
            receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
            if not receipt_matches(receipt, selection):
                raise RuntimeError(
                    "existing receipt does not match frozen selection: "
                    f"{receipt_path}"
                )
            print(
                f"RESUME\t{ordinal}/{EXPECTED_BUNDLES}\t"
                f"{process}\t{campaign}\tmember={member_index}"
            )
            results.append(receipt)
            transfer_rows.append(
                {
                    "ordinal": ordinal,
                    "member_index": member_index,
                    "process_or_mode": process,
                    "campaign": campaign,
                    "transfer_performed_this_run": False,
                    "temporary_bundle_removed": True,
                    "temporary_root_removed": True,
                    "status": "resumed_from_verified_receipt",
                }
            )
            continue

        local_bundle = (
            args.scratch
            / f"{ordinal:02d}_{process}_member{member_index:04d}.tar.gz"
        )
        local_root = (
            args.scratch
            / f"{ordinal:02d}_{process}_member{member_index:04d}.root"
        )
        local_bundle.unlink(missing_ok=True)
        local_root.unlink(missing_ok=True)

        print(
            f"TRANSFER\t{ordinal}/{EXPECTED_BUNDLES}\t"
            f"{process}\t{campaign}\tmember={member_index}"
        )
        print(f"REMOTE\t{selection['remote_bundle_uri']}")
        print(f"LOCAL\t{local_bundle}")

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
            expected_bundle_size = int(selection["bundle_size_bytes"])
            if observed_bundle_size != expected_bundle_size:
                raise RuntimeError(
                    f"bundle size mismatch for member {member_index}: "
                    f"{observed_bundle_size} != {expected_bundle_size}"
                )

            observed_bundle_sha = sha256_file(local_bundle)
            if observed_bundle_sha != selection["bundle_sha256_expected"]:
                raise RuntimeError(
                    f"bundle SHA-256 mismatch for member {member_index}"
                )

            root_size, root_sha = extract_root_member(
                local_bundle,
                str(selection["root_archive_member"]),
                local_root,
            )
            weight_audit = audit_root_weight(
                local_root,
                int(selection["generated_events"]),
            )

            receipt = {
                **selection,
                **weight_audit,
                "ordinal": ordinal,
                "observed_bundle_size_bytes": observed_bundle_size,
                "observed_bundle_sha256": observed_bundle_sha,
                "extracted_root_size_bytes": root_size,
                "extracted_root_sha256": root_sha,
                "root_files_opened": 1,
                "root_branches_read": 1,
                "parquet_files_opened": 0,
                "candidate_files_opened": 0,
                "validation_candidate_files_opened": 0,
                "evaluation_candidate_files_opened": 0,
                "status": "triboson_root_weight_audit_complete",
            }
            receipt_path.write_text(
                json.dumps(receipt, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            results.append(receipt)
            transfer_rows.append(
                {
                    "ordinal": ordinal,
                    "member_index": member_index,
                    "process_or_mode": process,
                    "campaign": campaign,
                    "transfer_performed_this_run": True,
                    "temporary_bundle_removed": True,
                    "temporary_root_removed": True,
                    "status": "bundle_verified_root_weight_only_audited",
                }
            )
        finally:
            local_root.unlink(missing_ok=True)
            local_bundle.unlink(missing_ok=True)

    results.sort(key=lambda row: int(row["ordinal"]))
    transfer_rows.sort(key=lambda row: int(row["ordinal"]))

    campaign_groups: dict[
        tuple[str, str, str],
        list[dict[str, object]],
    ] = defaultdict(list)
    for row in results:
        key = (
            str(row["sample_class"]),
            str(row["process_or_mode"]),
            str(row["campaign"]),
        )
        campaign_groups[key].append(row)

    campaign_rows: list[dict[str, object]] = []
    for key, rows in sorted(campaign_groups.items()):
        status = status_map[key]
        entries = sum(int(row["delphes_entries"]) for row in rows)
        positive = sum(int(row["positive_weight_events"]) for row in rows)
        negative = sum(int(row["negative_weight_events"]) for row in rows)
        zero = sum(int(row["zero_weight_events"]) for row in rows)
        sumw = sum(float(row["sum_root_nominal_weights"]) for row in rows)
        sumw2 = sum(
            float(row["sum_squared_root_nominal_weights"])
            for row in rows
        )
        minimum = min(
            float(row["minimum_root_nominal_weight"]) for row in rows
        )
        maximum = max(
            float(row["maximum_root_nominal_weight"]) for row in rows
        )
        manifest = int(status["manifest_generated_events"])
        if entries != manifest:
            raise RuntimeError(
                f"campaign ROOT entry total mismatch for {key}: "
                f"{entries} != {manifest}"
            )
        if positive + negative + zero != entries:
            raise RuntimeError(
                f"campaign sign-count closure failed for {key}"
            )

        if negative > 0:
            denominator_strategy = "signed_sum_root_nominal_weights"
        elif zero == 0 and positive == entries:
            denominator_strategy = "positive_root_weight_sum_candidate"
        else:
            denominator_strategy = "unresolved_weight_strategy"

        campaign_rows.append(
            {
                "sample_class": key[0],
                "process_or_mode": key[1],
                "campaign": key[2],
                "source_bundle_count": len(rows),
                "manifest_generated_events": manifest,
                "root_entries": entries,
                "root_entries_match_manifest": True,
                "positive_weight_events": positive,
                "negative_weight_events": negative,
                "zero_weight_events": zero,
                "sum_root_nominal_weights_candidate": f"{sumw:.17g}",
                "sum_squared_root_nominal_weights": f"{sumw2:.17g}",
                "minimum_root_nominal_weight": f"{minimum:.17g}",
                "maximum_root_nominal_weight": f"{maximum:.17g}",
                "signed_weight_transport_observed": negative > 0,
                "denominator_strategy_candidate": denominator_strategy,
                "campaign_denominator_candidate_recovered": (
                    denominator_strategy != "unresolved_weight_strategy"
                ),
                "normalization_denominator_authorized": False,
                "candidate_event_sign_transport_proven": False,
                "physical_weight_application_authorized": False,
                "status": "campaign_root_weight_candidate_fail_closed",
            }
        )

    if len(campaign_rows) != EXPECTED_CAMPAIGNS:
        raise RuntimeError("triboson campaign aggregate count mismatch")

    process_groups: dict[str, list[dict[str, object]]] = defaultdict(list)
    for row in campaign_rows:
        process_groups[str(row["process_or_mode"])].append(row)

    process_rows: list[dict[str, object]] = []
    for process, rows in sorted(process_groups.items()):
        process_rows.append(
            {
                "process_or_mode": process,
                "campaign_count": len(rows),
                "campaigns": ",".join(
                    sorted(str(row["campaign"]) for row in rows)
                ),
                "root_entries": sum(int(row["root_entries"]) for row in rows),
                "positive_weight_events": sum(
                    int(row["positive_weight_events"]) for row in rows
                ),
                "negative_weight_events": sum(
                    int(row["negative_weight_events"]) for row in rows
                ),
                "zero_weight_events": sum(
                    int(row["zero_weight_events"]) for row in rows
                ),
                "sum_root_nominal_weights_candidate": (
                    f"{sum(float(row['sum_root_nominal_weights_candidate']) for row in rows):.17g}"
                ),
                "sum_squared_root_nominal_weights": (
                    f"{sum(float(row['sum_squared_root_nominal_weights']) for row in rows):.17g}"
                ),
                "all_campaign_entry_totals_closed": all(
                    bool(row["root_entries_match_manifest"]) for row in rows
                ),
                "any_signed_weight_transport_observed": any(
                    bool(row["signed_weight_transport_observed"])
                    for row in rows
                ),
                "normalization_denominator_authorized": False,
                "status": "process_root_weight_candidate_fail_closed",
            }
        )

    if len(process_rows) != EXPECTED_PROCESSES:
        raise RuntimeError("triboson process aggregate count mismatch")

    signed_campaigns = sum(
        bool(row["signed_weight_transport_observed"])
        for row in campaign_rows
    )
    recovered_campaign_candidates = sum(
        bool(row["campaign_denominator_candidate_recovered"])
        for row in campaign_rows
    )

    transport_decisions = []
    for row in campaign_rows:
        signed = bool(row["signed_weight_transport_observed"])
        transport_decisions.append(
            {
                "process_or_mode": row["process_or_mode"],
                "campaign": row["campaign"],
                "signed_weight_transport_observed_in_root": signed,
                "root_denominator_candidate_recovered": row[
                    "campaign_denominator_candidate_recovered"
                ],
                "candidate_files_opened": 0,
                "candidate_sign_transport_proven": False,
                "normalization_denominator_authorized": False,
                "physical_weight_application_authorized": False,
                "required_next_action": (
                    "materialize_or_join_root_event_sign_into_candidate_weight_layer"
                    if signed
                    else "retain_positive_weight_candidate_pending_standard_scaleout"
                ),
                "status": "transport_decision_recorded_fail_closed",
            }
        )

    next_gate = (
        "materialize_triboson_event_sign_then_scale_all_standard_campaign_weight_audit"
        if signed_campaigns > 0
        else "scale_all_47_standard_campaign_weight_audit"
    )

    write_tsv(
        args.output / "triboson_bundle_selection.tsv",
        selections,
        list(selections[0]),
    )
    write_tsv(
        args.output / "triboson_root_weight_audit.tsv",
        results,
        list(results[0]),
    )
    write_tsv(
        args.output / "triboson_campaign_weight_aggregate.tsv",
        campaign_rows,
        list(campaign_rows[0]),
    )
    write_tsv(
        args.output / "triboson_process_weight_aggregate.tsv",
        process_rows,
        list(process_rows[0]),
    )
    write_tsv(
        args.output / "triboson_candidate_weight_transport_decision.tsv",
        transport_decisions,
        list(transport_decisions[0]),
    )
    write_tsv(
        args.output / "transfer_and_cleanup_audit.tsv",
        transfer_rows,
        list(transfer_rows[0]),
    )

    summary = {
        "schema_version": 1,
        "status": "hh4b_triboson_signed_weight_transport_audit_pass",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "source_commit": args.source_commit,
        "selection": {
            "processes": list(TRIBOSON_PROCESSES),
            "selected_source_bundles": len(selections),
            "selected_campaigns": len(campaign_rows),
            "selected_processes": len(process_rows),
            "root_source_map_rows_used": len(selections),
        },
        "audit": {
            "bundles_size_and_sha256_verified": len(results),
            "root_files_opened": len(results),
            "root_branches_read": len(results),
            "root_entries_audited": sum(
                int(row["delphes_entries"]) for row in results
            ),
            "campaign_entry_totals_closed": sum(
                bool(row["root_entries_match_manifest"])
                for row in campaign_rows
            ),
            "campaign_denominator_candidates_recovered": (
                recovered_campaign_candidates
            ),
            "campaigns_with_signed_root_weights": signed_campaigns,
            "all_temporary_files_removed": not any(args.scratch.iterdir()),
        },
        "readiness": {
            "triboson_root_weight_transport_audit_complete": True,
            "triboson_campaign_denominator_candidates_recovered": (
                recovered_campaign_candidates
            ),
            "candidate_event_sign_transport_proven": 0,
            "triboson_denominators_authorized": 0,
            "standard_campaign_denominators_authorized": 0,
            "nonstandard_campaign_denominators_authorized": 0,
            "external_reference_cross_sections_authorized": 0,
            "physics_normalization_ready": False,
        },
        "controls": {
            "root_files_opened": len(results),
            "root_branches_read": len(results),
            "parquet_files_opened": 0,
            "candidate_files_opened": 0,
            "candidate_rows_read": 0,
            "validation_candidate_files_opened": 0,
            "evaluation_candidate_files_opened": 0,
            "normalization_denominators_authorized": 0,
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
        "# HH4b triboson signed-weight transport audit\n\n"
        "The standard LHE-weight canary found signed nominal weights in "
        "`wwz_zbb`. The representative WWZ source LHE contains 46,800 "
        "events while each reconstructed WWZ ROOT shard contains 5,200 "
        "events, so the full source-LHE sum cannot be used directly as the "
        "denominator of a reconstructed campaign.\n\n"
        "This checkpoint verifies and opens only the 15 ROOT source bundles "
        "already mapped to the six WWZ, WZZ, and ZZZ pilot/scaleout "
        "campaigns. From each Delphes tree it reads exactly one nominal "
        "event-weight branch and records signed sums, squared sums, and sign "
        "counts. It verifies that campaign ROOT-entry totals equal the "
        "manifest-generated event totals.\n\n"
        "Candidate Parquet content is not opened. ROOT-level denominator "
        "values remain candidates until per-event sign transport into the "
        "candidate weight layer is proven. No normalization denominator, "
        "reference cross section, physical event weight, yield, model, or "
        "threshold is authorized.\n",
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
    print("ALL_15_TRIBOSON_SOURCE_BUNDLES_VERIFIED_PASS")
    print("ALL_15_TRIBOSON_ROOT_WEIGHT_BRANCHES_AUDITED_PASS")
    print("ALL_6_TRIBOSON_CAMPAIGN_ROOT_ENTRY_TOTALS_PASS")
    print("TRIBOSON_SIGNED_WEIGHT_TRANSPORT_DECISION_RECORDED_PASS")
    print("TEMPORARY_TRIBOSON_BUNDLES_AND_ROOT_FILES_REMOVED_PASS")
    print("NO_PARQUET_FILE_OPENED")
    print("NO_CANDIDATE_CONTENT_OPENED")
    print("NO_VALIDATION_CANDIDATE_CONTENT_OPENED")
    print("NO_EVALUATION_CANDIDATE_CONTENT_OPENED")
    print("NO_NORMALIZATION_DENOMINATOR_AUTHORIZED")
    print("NO_EXTERNAL_REFERENCE_CROSS_SECTION_ASSIGNED")
    print("NO_PHYSICAL_WEIGHTS_CALCULATED")
    print("NO_PHYSICAL_YIELDS_CALCULATED")
    print("HH4B_TRIBOSON_SIGNED_WEIGHT_TRANSPORT_AUDIT_PASS")


if __name__ == "__main__":
    main()
