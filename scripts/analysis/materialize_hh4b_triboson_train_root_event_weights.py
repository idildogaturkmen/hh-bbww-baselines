#!/usr/bin/env python3
"""Materialize train-only triboson ROOT nominal-weight sidecars."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import shutil
import subprocess
import tarfile
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

import awkward as ak
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import uproot


TRIBOSON_PROCESSES = ("wwz_zbb", "wzz_zbb", "zzz_zbb")
EXPECTED_TRAIN_SOURCE_FILES = 12
EXPECTED_VALIDATION_SOURCE_FILES = 3
EXPECTED_TRAIN_CAMPAIGNS = 3
EXPECTED_TRAIN_ROOT_EVENTS = 40000
SIDECAR_SCHEMA_VERSION = 1


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
                f"expected one ROOT member {member_name!r}, "
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
                f"could not stream ROOT member: {member.name}"
            )
        destination.parent.mkdir(parents=True, exist_ok=True)
        with source, destination.open("wb") as output:
            shutil.copyfileobj(source, output, length=8 * 1024 * 1024)

    observed_size = destination.stat().st_size
    if observed_size != member.size:
        raise RuntimeError(
            f"ROOT extraction size mismatch: {observed_size} != {member.size}"
        )
    return observed_size, sha256_file(destination)


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
                f"ambiguous nominal event-weight branch for {leaf}: {matches}"
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
            f"nominal-weight shape {values.shape!r} "
            f"does not match ({expected_entries},)"
        )
    if not np.all(np.isfinite(values)):
        raise RuntimeError("nonfinite nominal ROOT weights")
    if np.any(values == 0.0):
        raise RuntimeError("zero nominal ROOT weights are not authorized")
    return values


def read_candidate_event_indices(
    candidate_path: Path,
    expected_rows: int,
) -> np.ndarray:
    parquet_file = pq.ParquetFile(candidate_path)
    names = set(parquet_file.schema_arrow.names)
    if "event" not in names:
        raise RuntimeError(
            f"candidate file lacks event column: {candidate_path}"
        )
    table = pq.read_table(candidate_path, columns=["event"])
    if table.num_rows != expected_rows:
        raise RuntimeError(
            f"candidate rows {table.num_rows} != {expected_rows}: "
            f"{candidate_path}"
        )
    events = np.asarray(
        table.column("event").combine_chunks().to_numpy(
            zero_copy_only=False
        ),
        dtype=np.int64,
    )
    if events.shape != (expected_rows,):
        raise RuntimeError("candidate event-index shape mismatch")
    if len(np.unique(events)) != len(events):
        raise RuntimeError(
            f"candidate event indices are not unique: {candidate_path}"
        )
    return events


def write_sidecar_atomic(
    output: Path,
    *,
    member_index: int,
    process: str,
    campaign: str,
    root_weight_branch: str,
    weights: np.ndarray,
) -> None:
    events = np.arange(weights.size, dtype=np.int64)
    signs = np.sign(weights).astype(np.int8)

    metadata = {
        b"hh4b_sidecar_schema_version": str(
            SIDECAR_SCHEMA_VERSION
        ).encode(),
        b"member_index": str(member_index).encode(),
        b"process_or_mode": process.encode(),
        b"campaign": campaign.encode(),
        b"dataset_split": b"train",
        b"root_weight_branch": root_weight_branch.encode(),
        b"meaning": (
            b"one nominal generator weight per Delphes ROOT entry"
        ),
    }

    schema = pa.schema(
        [
            pa.field("event", pa.int64(), nullable=False),
            pa.field(
                "generator_nominal_weight",
                pa.float64(),
                nullable=False,
            ),
            pa.field(
                "generator_weight_sign",
                pa.int8(),
                nullable=False,
            ),
        ],
        metadata=metadata,
    )
    table = pa.Table.from_arrays(
        [
            pa.array(events, type=pa.int64()),
            pa.array(weights, type=pa.float64()),
            pa.array(signs, type=pa.int8()),
        ],
        schema=schema,
    )

    temporary = output.with_name(
        f".{output.name}.{os.getpid()}.tmp"
    )
    temporary.unlink(missing_ok=True)
    try:
        pq.write_table(
            table,
            temporary,
            compression="zstd",
            version="2.6",
            write_statistics=True,
        )
        check = pq.read_table(temporary)
        if check.schema.names != table.schema.names:
            raise RuntimeError("sidecar schema round-trip mismatch")
        if check.num_rows != weights.size:
            raise RuntimeError("sidecar row-count round-trip mismatch")
        output.parent.mkdir(parents=True, exist_ok=True)
        temporary.replace(output)
    finally:
        temporary.unlink(missing_ok=True)


def receipt_matches(
    receipt: dict[str, object],
    selection: dict[str, object],
) -> bool:
    fields = (
        "member_index",
        "process_or_mode",
        "campaign",
        "candidate_path",
        "candidate_sha256",
        "remote_bundle_path",
        "bundle_sha256_expected",
        "root_archive_member",
        "generated_events",
        "candidate_rows",
    )
    return all(
        str(receipt.get(field)) == str(selection.get(field))
        for field in fields
    ) and receipt.get("status") == "train_root_weight_sidecar_complete"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root-source-map", type=Path, required=True)
    parser.add_argument("--c3c-root-audit", type=Path, required=True)
    parser.add_argument("--c3c-campaign-aggregate", type=Path, required=True)
    parser.add_argument("--eos-host", required=True)
    parser.add_argument("--scratch", type=Path, required=True)
    parser.add_argument("--sidecar-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--source-commit", required=True)
    args = parser.parse_args()

    source_rows = read_tsv(args.root_source_map)
    c3c_root_rows = read_tsv(args.c3c_root_audit)
    c3c_campaign_rows = read_tsv(args.c3c_campaign_aggregate)

    triboson_rows = [
        row
        for row in source_rows
        if row["process_or_mode"] in TRIBOSON_PROCESSES
    ]
    train_rows = [
        row for row in triboson_rows
        if row["dataset_split"] == "train"
    ]
    validation_rows = [
        row for row in triboson_rows
        if row["dataset_split"] == "validation"
    ]

    if len(train_rows) != EXPECTED_TRAIN_SOURCE_FILES:
        raise RuntimeError(
            f"expected {EXPECTED_TRAIN_SOURCE_FILES} train source rows, "
            f"observed {len(train_rows)}"
        )
    if len(validation_rows) != EXPECTED_VALIDATION_SOURCE_FILES:
        raise RuntimeError(
            f"expected {EXPECTED_VALIDATION_SOURCE_FILES} validation rows, "
            f"observed {len(validation_rows)}"
        )
    if any(
        row["dataset_split"] not in {"train", "validation"}
        for row in triboson_rows
    ):
        raise RuntimeError("unexpected triboson dataset split")

    c3c_root_map = {
        int(row["member_index"]): row
        for row in c3c_root_rows
    }

    c3c_train_campaign_map = {
        (
            row["sample_class"],
            row["process_or_mode"],
            row["campaign"],
        ): row
        for row in c3c_campaign_rows
        if "scaleout" in row["campaign"]
    }
    if len(c3c_train_campaign_map) != EXPECTED_TRAIN_CAMPAIGNS:
        raise RuntimeError("expected three train triboson aggregates")

    selections: list[dict[str, object]] = []
    for row in sorted(train_rows, key=lambda item: int(item["member_index"])):
        member_index = int(row["member_index"])
        if member_index not in c3c_root_map:
            raise RuntimeError(
                f"member {member_index} is absent from PN-c3c ROOT audit"
            )
        campaign = campaign_from_bundle_path(row["remote_bundle_path"])
        c3c = c3c_root_map[member_index]

        if c3c["campaign"] != campaign:
            raise RuntimeError(
                f"PN-c3c campaign mismatch for member {member_index}"
            )
        if c3c["nominal_weight_branch"] != "Event/Event.Weight":
            raise RuntimeError(
                f"unexpected PN-c3c weight branch for member {member_index}: "
                + c3c["nominal_weight_branch"]
            )
        if row["resolution_status"] != "resolved":
            raise RuntimeError(f"unresolved train source member {member_index}")
        if row["bundle_exists"] != "True":
            raise RuntimeError(f"bundle not proven present: {member_index}")
        if row["bundle_stat_status"] != "pass":
            raise RuntimeError(f"bundle stat failed: {member_index}")
        if row["archive_depth"] != "1":
            raise RuntimeError(
                f"unexpected triboson archive depth: {member_index}"
            )
        if row["root_container_member"].strip():
            raise RuntimeError(
                f"unexpected nested triboson ROOT: {member_index}"
            )
        if not row["candidate_path"].strip():
            raise RuntimeError(
                f"missing train candidate path: {member_index}"
            )
        if not row["candidate_sha256"].strip():
            raise RuntimeError(
                f"missing train candidate SHA-256: {member_index}"
            )

        sidecar_name = (
            f"member{member_index:04d}_{row['process_or_mode']}_"
            f"{Path(row['root_basename']).stem}_event_weights.parquet"
        )
        selections.append(
            {
                "member_index": member_index,
                "sample_class": row["sample_class"],
                "process_or_mode": row["process_or_mode"],
                "campaign": campaign,
                "dataset_split": "train",
                "generated_events": int(row["generated_events"]),
                "candidate_rows": int(row["candidate_rows"]),
                "candidate_path": row["candidate_path"],
                "candidate_sha256": row["candidate_sha256"],
                "remote_bundle_path": row["remote_bundle_path"],
                "remote_bundle_uri": remote_uri(
                    args.eos_host,
                    row["remote_bundle_path"],
                ),
                "bundle_size_bytes": int(row["bundle_size_bytes"]),
                "bundle_sha256_expected": row[
                    "bundle_sha256_expected"
                ],
                "root_archive_member": row["root_archive_member"],
                "root_basename": row["root_basename"],
                "c3c_root_sumw": c3c["sum_root_nominal_weights"],
                "c3c_root_sumw2": c3c[
                    "sum_squared_root_nominal_weights"
                ],
                "sidecar_path": str(args.sidecar_dir / sidecar_name),
                "validation_candidate_access_authorized": False,
                "source_candidate_modification_authorized": False,
                "status": "selected_train_only_weight_transport",
            }
        )

    if (
        sum(int(row["generated_events"]) for row in selections)
        != EXPECTED_TRAIN_ROOT_EVENTS
    ):
        raise RuntimeError("train triboson generated-event total mismatch")

    args.output.mkdir(parents=True, exist_ok=True)
    receipt_dir = args.output / "receipts"
    receipt_dir.mkdir(exist_ok=True)
    args.scratch.mkdir(parents=True, exist_ok=True)
    args.sidecar_dir.mkdir(parents=True, exist_ok=True)

    results: list[dict[str, object]] = []
    transfer_rows: list[dict[str, object]] = []

    for ordinal, selection in enumerate(selections, start=1):
        member_index = int(selection["member_index"])
        process = str(selection["process_or_mode"])
        campaign = str(selection["campaign"])
        receipt_path = (
            receipt_dir
            / f"{ordinal:02d}_{process}_member{member_index:04d}.json"
        )
        sidecar_path = Path(str(selection["sidecar_path"]))

        if receipt_path.is_file():
            receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
            if not receipt_matches(receipt, selection):
                raise RuntimeError(
                    f"receipt mismatch: {receipt_path}"
                )
            if not sidecar_path.is_file():
                raise RuntimeError(
                    f"receipt exists but sidecar is absent: {sidecar_path}"
                )
            if sha256_file(sidecar_path) != receipt["sidecar_sha256"]:
                raise RuntimeError(
                    f"sidecar SHA-256 mismatch on resume: {sidecar_path}"
                )
            print(
                f"RESUME\t{ordinal}/{EXPECTED_TRAIN_SOURCE_FILES}\t"
                f"{process}\tmember={member_index}"
            )
            results.append(receipt)
            transfer_rows.append(
                {
                    "ordinal": ordinal,
                    "member_index": member_index,
                    "process_or_mode": process,
                    "campaign": campaign,
                    "transfer_performed_this_run": False,
                    "candidate_opened_this_run": False,
                    "sidecar_reused": True,
                    "temporary_bundle_removed": True,
                    "temporary_root_removed": True,
                    "status": "resumed_from_verified_receipt",
                }
            )
            continue

        if sidecar_path.exists():
            raise RuntimeError(
                "sidecar exists without a matching receipt; "
                f"refusing to overwrite: {sidecar_path}"
            )

        candidate_path = Path(str(selection["candidate_path"]))
        if not candidate_path.is_file():
            raise RuntimeError(
                f"train candidate file is absent: {candidate_path}"
            )
        observed_candidate_sha = sha256_file(candidate_path)
        if observed_candidate_sha != selection["candidate_sha256"]:
            raise RuntimeError(
                f"candidate SHA-256 mismatch: {candidate_path}"
            )

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
            f"TRANSFER\t{ordinal}/{EXPECTED_TRAIN_SOURCE_FILES}\t"
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
            if observed_bundle_size != int(selection["bundle_size_bytes"]):
                raise RuntimeError(
                    f"bundle size mismatch for member {member_index}"
                )
            observed_bundle_sha = sha256_file(local_bundle)
            if (
                observed_bundle_sha
                != selection["bundle_sha256_expected"]
            ):
                raise RuntimeError(
                    f"bundle SHA-256 mismatch for member {member_index}"
                )

            root_size, root_sha = extract_root_member(
                local_bundle,
                str(selection["root_archive_member"]),
                local_root,
            )

            with uproot.open(local_root) as root_file:
                tree = root_file["Delphes"]
                entries = int(tree.num_entries)
                expected_entries = int(selection["generated_events"])
                if entries != expected_entries:
                    raise RuntimeError(
                        f"ROOT entries {entries} != {expected_entries}"
                    )
                branch = select_nominal_weight_branch(
                    tree.keys(recursive=True)
                )
                if branch != "Event/Event.Weight":
                    raise RuntimeError(
                        f"unexpected nominal branch: {branch}"
                    )
                weights = one_weight_per_entry(
                    tree[branch].array(library="ak"),
                    entries,
                )

            sumw = float(np.sum(weights, dtype=np.float64))
            sumw2 = float(
                np.sum(weights * weights, dtype=np.float64)
            )
            expected_sumw = float(selection["c3c_root_sumw"])
            expected_sumw2 = float(selection["c3c_root_sumw2"])

            if not math.isclose(
                sumw,
                expected_sumw,
                rel_tol=1.0e-13,
                abs_tol=1.0e-12,
            ):
                raise RuntimeError(
                    f"PN-c3c sumw mismatch for member {member_index}: "
                    f"{sumw} != {expected_sumw}"
                )
            if not math.isclose(
                sumw2,
                expected_sumw2,
                rel_tol=1.0e-13,
                abs_tol=1.0e-12,
            ):
                raise RuntimeError(
                    f"PN-c3c sumw2 mismatch for member {member_index}"
                )

            candidate_events = read_candidate_event_indices(
                candidate_path,
                int(selection["candidate_rows"]),
            )
            if candidate_events.size:
                if int(candidate_events.min()) < 0:
                    raise RuntimeError("negative candidate event index")
                if int(candidate_events.max()) >= entries:
                    raise RuntimeError(
                        "candidate event index exceeds ROOT entry range"
                    )

            selected_weights = weights[candidate_events]
            selected_positive = int(
                np.count_nonzero(selected_weights > 0.0)
            )
            selected_negative = int(
                np.count_nonzero(selected_weights < 0.0)
            )
            selected_zero = int(
                np.count_nonzero(selected_weights == 0.0)
            )
            if (
                selected_positive
                + selected_negative
                + selected_zero
                != candidate_events.size
            ):
                raise RuntimeError("candidate sign-count closure failed")
            if selected_zero:
                raise RuntimeError(
                    "selected candidate has a zero nominal weight"
                )

            write_sidecar_atomic(
                sidecar_path,
                member_index=member_index,
                process=process,
                campaign=campaign,
                root_weight_branch=branch,
                weights=weights,
            )
            sidecar_sha = sha256_file(sidecar_path)
            sidecar_size = sidecar_path.stat().st_size

            sidecar_check = pq.read_table(sidecar_path)
            sidecar_events = np.asarray(
                sidecar_check.column("event").combine_chunks().to_numpy(
                    zero_copy_only=False
                ),
                dtype=np.int64,
            )
            sidecar_weights = np.asarray(
                sidecar_check.column(
                    "generator_nominal_weight"
                ).combine_chunks().to_numpy(zero_copy_only=False),
                dtype=np.float64,
            )
            sidecar_signs = np.asarray(
                sidecar_check.column(
                    "generator_weight_sign"
                ).combine_chunks().to_numpy(zero_copy_only=False),
                dtype=np.int8,
            )

            if not np.array_equal(
                sidecar_events,
                np.arange(entries, dtype=np.int64),
            ):
                raise RuntimeError("sidecar event-index round-trip mismatch")
            if not np.array_equal(sidecar_weights, weights):
                raise RuntimeError("sidecar weight round-trip mismatch")
            if not np.array_equal(
                sidecar_signs,
                np.sign(weights).astype(np.int8),
            ):
                raise RuntimeError("sidecar sign round-trip mismatch")
            if candidate_events.size and not np.array_equal(
                sidecar_weights[candidate_events],
                selected_weights,
            ):
                raise RuntimeError(
                    "candidate-to-sidecar weight join mismatch"
                )

            receipt = {
                **selection,
                "ordinal": ordinal,
                "observed_candidate_sha256": observed_candidate_sha,
                "observed_bundle_size_bytes": observed_bundle_size,
                "observed_bundle_sha256": observed_bundle_sha,
                "extracted_root_size_bytes": root_size,
                "extracted_root_sha256": root_sha,
                "root_weight_branch": branch,
                "root_entries": entries,
                "positive_root_events": int(
                    np.count_nonzero(weights > 0.0)
                ),
                "negative_root_events": int(
                    np.count_nonzero(weights < 0.0)
                ),
                "zero_root_events": int(
                    np.count_nonzero(weights == 0.0)
                ),
                "sum_root_nominal_weights": f"{sumw:.17g}",
                "sum_squared_root_nominal_weights": f"{sumw2:.17g}",
                "candidate_rows_opened": int(candidate_events.size),
                "candidate_events_unique": True,
                "candidate_events_in_root_range": True,
                "candidate_positive_weight_rows": selected_positive,
                "candidate_negative_weight_rows": selected_negative,
                "candidate_zero_weight_rows": selected_zero,
                "candidate_to_sidecar_weight_join_exact": True,
                "sidecar_rows": entries,
                "sidecar_size_bytes": sidecar_size,
                "sidecar_sha256": sidecar_sha,
                "source_candidate_modified": False,
                "validation_candidate_files_opened": 0,
                "evaluation_candidate_files_opened": 0,
                "normalization_denominator_authorized": False,
                "status": "train_root_weight_sidecar_complete",
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
                    "candidate_opened_this_run": True,
                    "sidecar_reused": False,
                    "temporary_bundle_removed": True,
                    "temporary_root_removed": True,
                    "status": "train_candidate_join_and_sidecar_pass",
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

    if len(campaign_groups) != EXPECTED_TRAIN_CAMPAIGNS:
        raise RuntimeError("train campaign aggregate count mismatch")

    denominator_rows: list[dict[str, object]] = []
    for key, rows in sorted(campaign_groups.items()):
        root_entries = sum(int(row["root_entries"]) for row in rows)
        sumw = sum(
            float(row["sum_root_nominal_weights"]) for row in rows
        )
        sumw2 = sum(
            float(row["sum_squared_root_nominal_weights"])
            for row in rows
        )
        c3c = c3c_train_campaign_map[key]
        expected_entries = int(c3c["root_entries"])
        expected_sumw = float(
            c3c["sum_root_nominal_weights_candidate"]
        )
        expected_sumw2 = float(
            c3c["sum_squared_root_nominal_weights"]
        )

        if root_entries != expected_entries:
            raise RuntimeError(
                f"campaign entry mismatch for {key}: "
                f"{root_entries} != {expected_entries}"
            )
        if not math.isclose(
            sumw,
            expected_sumw,
            rel_tol=1.0e-13,
            abs_tol=1.0e-11,
        ):
            raise RuntimeError(
                f"campaign sumw mismatch for {key}: "
                f"{sumw} != {expected_sumw}"
            )
        if not math.isclose(
            sumw2,
            expected_sumw2,
            rel_tol=1.0e-13,
            abs_tol=1.0e-11,
        ):
            raise RuntimeError(
                f"campaign sumw2 mismatch for {key}"
            )

        candidate_rows = sum(
            int(row["candidate_rows_opened"]) for row in rows
        )
        candidate_negative = sum(
            int(row["candidate_negative_weight_rows"])
            for row in rows
        )

        denominator_rows.append(
            {
                "sample_class": key[0],
                "process_or_mode": key[1],
                "campaign": key[2],
                "dataset_split": "train",
                "source_root_files": len(rows),
                "root_entries": root_entries,
                "candidate_rows_joined": candidate_rows,
                "candidate_negative_weight_rows": candidate_negative,
                "denominator_type": "signed_sum_root_nominal_weights",
                "normalization_denominator": f"{sumw:.17g}",
                "sum_squared_root_nominal_weights": f"{sumw2:.17g}",
                "candidate_event_weight_transport_proven": True,
                "normalization_denominator_authorized": True,
                "validation_application_authorized": False,
                "physical_weight_application_authorized": False,
                "external_cross_section_required": True,
                "status": "train_triboson_denominator_frozen",
            }
        )

    registry_rows = []
    for row in results:
        registry_rows.append(
            {
                "member_index": row["member_index"],
                "sample_class": row["sample_class"],
                "process_or_mode": row["process_or_mode"],
                "campaign": row["campaign"],
                "dataset_split": "train",
                "candidate_path": row["candidate_path"],
                "candidate_sha256": row["candidate_sha256"],
                "candidate_rows": row["candidate_rows_opened"],
                "source_root_basename": row["root_basename"],
                "source_root_sha256": row["extracted_root_sha256"],
                "root_entries": row["root_entries"],
                "root_weight_branch": row["root_weight_branch"],
                "sidecar_path": row["sidecar_path"],
                "sidecar_sha256": row["sidecar_sha256"],
                "sidecar_rows": row["sidecar_rows"],
                "join_key": "event",
                "join_cardinality": "candidate_many_to_source_one_per_event",
                "join_exact": True,
                "source_candidate_modified": False,
                "status": "train_weight_sidecar_registered",
            }
        )

    validation_policy_rows = []
    for row in sorted(
        validation_rows,
        key=lambda item: int(item["member_index"]),
    ):
        validation_policy_rows.append(
            {
                "member_index": int(row["member_index"]),
                "process_or_mode": row["process_or_mode"],
                "campaign": campaign_from_bundle_path(
                    row["remote_bundle_path"]
                ),
                "dataset_split": "validation",
                "candidate_path": row["candidate_path"],
                "candidate_rows": int(row["candidate_rows"]),
                "candidate_file_opened": False,
                "sidecar_materialized": False,
                "normalization_denominator_authorized": False,
                "physical_weight_application_authorized": False,
                "status": "validation_candidate_remains_sealed",
            }
        )

    write_tsv(
        args.output / "train_source_selection.tsv",
        selections,
        list(selections[0]),
    )
    write_tsv(
        args.output / "train_weight_sidecar_registry.tsv",
        registry_rows,
        list(registry_rows[0]),
    )
    write_tsv(
        args.output / "train_triboson_denominator_freeze.tsv",
        denominator_rows,
        list(denominator_rows[0]),
    )
    write_tsv(
        args.output / "validation_seal_policy.tsv",
        validation_policy_rows,
        list(validation_policy_rows[0]),
    )
    write_tsv(
        args.output / "transfer_and_cleanup_audit.tsv",
        transfer_rows,
        list(transfer_rows[0]),
    )

    total_candidate_rows = sum(
        int(row["candidate_rows_opened"]) for row in results
    )
    total_candidate_negative = sum(
        int(row["candidate_negative_weight_rows"])
        for row in results
    )

    summary = {
        "schema_version": 1,
        "status": (
            "hh4b_triboson_train_weight_sidecar_and_denominator_freeze_pass"
        ),
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "source_commit": args.source_commit,
        "selection": {
            "train_source_files": len(results),
            "train_campaigns": len(denominator_rows),
            "validation_source_files_recorded_but_unopened": len(
                validation_policy_rows
            ),
            "train_root_events_materialized": sum(
                int(row["sidecar_rows"]) for row in results
            ),
            "train_candidate_files_opened": len(results),
            "train_candidate_rows_joined": total_candidate_rows,
        },
        "transport": {
            "sidecars_materialized": len(results),
            "sidecar_rows": sum(
                int(row["sidecar_rows"]) for row in results
            ),
            "candidate_to_sidecar_joins_exact": sum(
                bool(row["candidate_to_sidecar_weight_join_exact"])
                for row in results
            ),
            "candidate_negative_weight_rows": total_candidate_negative,
            "source_candidates_modified": 0,
            "all_temporary_files_removed": not any(
                args.scratch.iterdir()
            ),
        },
        "readiness": {
            "train_triboson_event_weight_transport_proven": True,
            "train_triboson_denominators_authorized": len(
                denominator_rows
            ),
            "validation_triboson_denominators_authorized": 0,
            "other_standard_campaign_denominators_authorized": 0,
            "nonstandard_campaign_denominators_authorized": 0,
            "external_reference_cross_sections_authorized": 0,
            "physical_weight_application_authorized": False,
            "physics_normalization_ready": False,
        },
        "controls": {
            "train_candidate_files_opened": len(results),
            "train_candidate_rows_read": total_candidate_rows,
            "source_candidate_files_modified": 0,
            "validation_candidate_files_opened": 0,
            "evaluation_candidate_files_opened": 0,
            "root_files_opened": len(results),
            "root_branches_read": len(results),
            "normalization_denominators_authorized": len(
                denominator_rows
            ),
            "cross_sections_assigned": 0,
            "physical_weights_calculated": 0,
            "physical_yields_calculated": 0,
            "models_trained": 0,
            "thresholds_selected": 0,
        },
        "next_gate": (
            "scale_weight_convention_audit_to_remaining_20_standard_"
            "configuration_groups"
        ),
    }

    (args.output / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    (args.output / "README.md").write_text(
        "# HH4b train-only triboson ROOT event-weight sidecars\n\n"
        "This checkpoint materializes one immutable sidecar row per Delphes "
        "ROOT event for the 12 train-only WWZ, WZZ, and ZZZ source shards. "
        "Each sidecar stores the ROOT event index, exact nominal generator "
        "weight, and sign from `Event/Event.Weight`.\n\n"
        "The existing train candidate Parquet files are checksum-verified "
        "and opened only to read their `event` indices. Every candidate index "
        "is proven unique, in range, and exactly joinable to its source "
        "event-weight sidecar. Source candidate files are never modified.\n\n"
        "The three validation candidate files remain unopened and receive no "
        "sidecars. Their denominators and physical-weight application remain "
        "unauthorized.\n\n"
        "For the three train scaleout campaigns, the complete sidecar sums "
        "reproduce the PN-c3c signed ROOT sums, so those campaign denominators "
        "are frozen. External cross sections and physical weights are still "
        "unauthorized.\n",
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
    print("ALL_12_TRIBOSON_TRAIN_SOURCE_BUNDLES_VERIFIED_PASS")
    print("ALL_12_TRIBOSON_TRAIN_ROOT_WEIGHT_SIDECARS_PASS")
    print("ALL_TRAIN_CANDIDATE_EVENT_JOINS_EXACT_PASS")
    print("THREE_TRAIN_TRIBOSON_DENOMINATORS_FROZEN_PASS")
    print("SOURCE_CANDIDATE_FILES_IMMUTABLE_PASS")
    print("ALL_3_VALIDATION_CANDIDATE_FILES_REMAIN_SEALED_PASS")
    print("TEMPORARY_TRAIN_BUNDLES_AND_ROOT_FILES_REMOVED_PASS")
    print("NO_EXTERNAL_REFERENCE_CROSS_SECTION_ASSIGNED")
    print("NO_PHYSICAL_WEIGHTS_CALCULATED")
    print("NO_PHYSICAL_YIELDS_CALCULATED")
    print(
        "HH4B_TRIBOSON_TRAIN_WEIGHT_SIDECAR_AND_"
        "DENOMINATOR_FREEZE_PASS"
    )


if __name__ == "__main__":
    main()
