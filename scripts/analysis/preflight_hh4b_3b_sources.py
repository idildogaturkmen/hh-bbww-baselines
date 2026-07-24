#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import shutil
from collections import Counter
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq


SOURCE_FIELD_TOKENS = (
    "root",
    "source",
    "input",
    "origin",
    "parent",
    "eos",
    "uri",
    "path",
    "file",
)

SPLIT_FIELD_CANDIDATES = (
    "dataset_split",
    "split",
    "sample_split",
    "registry_split",
)

BRANCH_PREFIXES = (
    "Jet",
    "Event",
    "MissingET",
    "GenParticle",
    "Electron",
    "Muon",
    "ScalarHT",
    "Track",
    "ParticleFlowCandidate",
)


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()

    with path.open("rb") as handle:
        for block in iter(
            lambda: handle.read(1024 * 1024),
            b"",
        ):
            digest.update(block)

    return digest.hexdigest()


def resolve(repo: Path, value: str) -> Path:
    path = Path(value).expanduser()

    if not path.is_absolute():
        path = repo / path

    return path.resolve()


def read_tsv(
    path: Path,
) -> tuple[list[str], list[dict[str, str]]]:
    with path.open(
        newline="",
        encoding="utf-8",
        errors="strict",
    ) as handle:
        reader = csv.DictReader(
            handle,
            delimiter="\t",
        )

        fields = list(reader.fieldnames or [])
        rows = list(reader)

    require(
        bool(fields),
        f"{path}: TSV header is missing",
    )

    return fields, rows


def write_tsv(
    path: Path,
    rows: list[dict[str, Any]],
    fields: list[str],
) -> None:
    with path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=fields,
            delimiter="\t",
            lineterminator="\n",
            extrasaction="ignore",
        )

        writer.writeheader()
        writer.writerows(rows)


def rootlike(value: str) -> bool:
    stripped = value.strip()
    lowered = stripped.lower()

    if not stripped:
        return False

    if lowered.startswith("root://"):
        return True

    return (
        re.search(
            r"\.root(?:$|[?#])",
            lowered,
        )
        is not None
    )


def local_existing_file(
    repo: Path,
    value: str,
) -> bool:
    stripped = value.strip()

    if (
        not stripped
        or stripped.startswith("root://")
    ):
        return False

    path = Path(stripped).expanduser()

    if not path.is_absolute():
        path = repo / path

    return path.is_file()


def resolve_split_field(
    fields: list[str],
) -> str | None:
    by_lower = {
        field.lower(): field
        for field in fields
    }

    for candidate in SPLIT_FIELD_CANDIDATES:
        if candidate in by_lower:
            return by_lower[candidate]

    fuzzy = [
        field
        for field in fields
        if "split" in field.lower()
    ]

    if len(fuzzy) == 1:
        return fuzzy[0]

    return None


def development_rows(
    *,
    fields: list[str],
    rows: list[dict[str, str]],
    development_splits: set[str],
) -> tuple[list[dict[str, str]], str | None]:
    split_field = resolve_split_field(
        fields
    )

    if split_field is None:
        return rows, None

    selected = [
        row
        for row in rows
        if row.get(split_field, "")
        .strip()
        .lower()
        in development_splits
    ]

    return selected, split_field


def inventory_table_fields(
    *,
    repo: Path,
    table_label: str,
    table_path: Path,
    fields: list[str],
    rows: list[dict[str, str]],
    split_field: str | None,
) -> list[dict[str, Any]]:
    inventory: list[dict[str, Any]] = []

    for field in fields:
        values = [
            row.get(field, "").strip()
            for row in rows
        ]

        nonempty = sum(
            bool(value)
            for value in values
        )

        rootlike_values = sum(
            rootlike(value)
            for value in values
        )

        remote_root_values = sum(
            value.startswith("root://")
            for value in values
        )

        local_existing_values = sum(
            local_existing_file(repo, value)
            for value in values
        )

        field_name_related = any(
            token in field.lower()
            for token in SOURCE_FIELD_TOKENS
        )

        if not (
            field_name_related
            or rootlike_values > 0
        ):
            continue

        inventory.append({
            "table_label":
                table_label,
            "table_path":
                str(table_path),
            "split_field":
                split_field or "",
            "development_rows":
                len(rows),
            "field_name":
                field,
            "nonempty_values":
                nonempty,
            "rootlike_values":
                rootlike_values,
            "remote_root_values":
                remote_root_values,
            "existing_local_files":
                local_existing_values,
            "complete_nonempty_coverage":
                nonempty == len(rows),
            "complete_rootlike_coverage":
                rootlike_values == len(rows),
        })

    return inventory


def metadata_keys(
    parquet_file: pq.ParquetFile,
) -> list[str]:
    raw = (
        parquet_file.metadata.metadata
        or {}
    )

    result = []

    for key in raw:
        try:
            result.append(
                key.decode("utf-8")
            )
        except UnicodeDecodeError:
            result.append(
                repr(key)
            )

    return sorted(result)


def frozen_builder_references(
    builder_text: str,
) -> list[dict[str, str]]:
    prefix_pattern = "|".join(
        re.escape(prefix)
        for prefix in BRANCH_PREFIXES
    )

    dotted_pattern = re.compile(
        rf"""(?P<quote>["'])
        (?P<branch>
            (?:{prefix_pattern})
            \.[A-Za-z0-9_]+
        )
        (?P=quote)""",
        flags=re.VERBOSE,
    )

    dotted = sorted({
        match.group("branch")
        for match in dotted_pattern.finditer(
            builder_text
        )
    })

    rows = [
        {
            "reference_type":
                "literal_dotted_branch",
            "reference":
                value,
        }
        for value in dotted
    ]

    for token in (
        "BTag",
        "Flavor",
        "FlavorPhys",
        "PT",
        "Eta",
        "Phi",
        "Mass",
        "Jet",
        "Event",
    ):
        if re.search(
            rf"\b{re.escape(token)}\b",
            builder_text,
        ):
            rows.append({
                "reference_type":
                    "source_token",
                "reference":
                    token,
            })

    rows.sort(
        key=lambda row: (
            row["reference_type"],
            row["reference"],
        )
    )

    return rows


def copy_checkpoint_file(
    source: Path,
    destination_dir: Path,
) -> None:
    require(
        source.is_file(),
        f"missing checkpoint source: {source}",
    )

    shutil.copy2(
        source,
        destination_dir / source.name,
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Inventory all available ROOT-provenance "
            "channels for the HH4b 3b-control workflow "
            "without assuming the candidate registries "
            "contain one complete ROOT-path field."
        )
    )

    parser.add_argument(
        "--config",
        required=True,
        type=Path,
    )

    args = parser.parse_args()

    repo = Path(__file__).resolve().parents[2]

    config_path = (
        args.config.resolve()
        if args.config.is_absolute()
        else (repo / args.config).resolve()
    )

    require(
        config_path.is_file(),
        f"missing configuration: {config_path}",
    )

    config = json.loads(
        config_path.read_text(
            encoding="utf-8",
        )
    )

    require(
        config.get("schema_version") == 2,
        "unsupported configuration schema",
    )

    require(
        config.get("test_access_allowed")
        is False,
        "test access must remain disabled",
    )

    source_commit = str(
        config["source_commit"]
    ).lower()

    require(
        re.fullmatch(
            r"[0-9a-f]{40}",
            source_commit,
        )
        is not None,
        "source_commit is not a full SHA",
    )

    output_dir = resolve(
        repo,
        config["output_dir"],
    )

    checkpoint_dir = resolve(
        repo,
        config["checkpoint_dir"],
    )

    output_tmp = output_dir.with_name(
        output_dir.name + "_incomplete"
    )

    checkpoint_tmp = checkpoint_dir.with_name(
        checkpoint_dir.name + "_incomplete"
    )

    for path in (
        output_dir,
        output_tmp,
        checkpoint_dir,
        checkpoint_tmp,
    ):
        require(
            not path.exists(),
            f"refusing to overwrite {path}",
        )

    manifest_path = resolve(
        repo,
        config[
            "development_manifest_path"
        ],
    )

    require(
        manifest_path.is_file(),
        f"missing manifest: {manifest_path}",
    )

    require(
        sha256_file(manifest_path)
        == config[
            "development_manifest_sha256"
        ],
        "development-manifest SHA-256 mismatch",
    )

    signal_registry_path = resolve(
        repo,
        config["signal_registry_path"],
    )

    background_registry_path = resolve(
        repo,
        config[
            "background_registry_path"
        ],
    )

    vbf_cache_path = resolve(
        repo,
        config[
            "vbf_cache_manifest_path"
        ],
    )

    builder_path = resolve(
        repo,
        config["frozen_builder_path"],
    )

    policy_path = resolve(
        repo,
        config["frozen_policy_path"],
    )

    for path in (
        signal_registry_path,
        background_registry_path,
        vbf_cache_path,
        builder_path,
        policy_path,
    ):
        require(
            path.is_file(),
            f"missing required input: {path}",
        )

    manifest_fields, manifest_rows = (
        read_tsv(manifest_path)
    )

    required_manifest_fields = {
        "member_index",
        "sample_class",
        "dataset_split",
        "candidate_rows",
        "local_path",
    }

    missing_manifest_fields = sorted(
        required_manifest_fields
        - set(manifest_fields)
    )

    require(
        not missing_manifest_fields,
        (
            "development manifest missing fields: "
            + ", ".join(
                missing_manifest_fields
            )
        ),
    )

    expected = config["expected"]

    require(
        len(manifest_rows)
        == int(
            expected[
                "development_members"
            ]
        ),
        (
            "unexpected development member count: "
            f"{len(manifest_rows)}"
        ),
    )

    development_splits = {
        value.lower()
        for value in config[
            "development_splits"
        ]
    }

    require(
        all(
            row["dataset_split"]
            .strip()
            .lower()
            in development_splits
            for row in manifest_rows
        ),
        "manifest contains a non-development split",
    )

    class_counts = Counter(
        row["sample_class"]
        for row in manifest_rows
    )

    split_counts = Counter(
        row["dataset_split"]
        for row in manifest_rows
    )

    require(
        class_counts["signal"]
        == int(
            expected[
                "signal_development_members"
            ]
        ),
        f"signal-member count mismatch: {class_counts}",
    )

    require(
        class_counts["background"]
        == int(
            expected[
                "background_development_members"
            ]
        ),
        (
            "background-member count mismatch: "
            f"{class_counts}"
        ),
    )

    require(
        split_counts["train"]
        == int(expected["train_members"]),
        f"train-member count mismatch: {split_counts}",
    )

    require(
        split_counts["validation"]
        == int(
            expected["validation_members"]
        ),
        (
            "validation-member count mismatch: "
            f"{split_counts}"
        ),
    )

    candidate_row_total = sum(
        int(row["candidate_rows"])
        for row in manifest_rows
    )

    require(
        candidate_row_total
        == int(
            expected[
                "development_candidate_rows"
            ]
        ),
        (
            "development candidate-row mismatch: "
            f"{candidate_row_total}"
        ),
    )

    auxiliary_tables = (
        (
            "canonical_development_manifest",
            manifest_path,
            manifest_fields,
            manifest_rows,
            "dataset_split",
        ),
    )

    table_inputs = [
        (
            "signal_registry",
            signal_registry_path,
        ),
        (
            "background_registry",
            background_registry_path,
        ),
        (
            "vbf_cache_manifest",
            vbf_cache_path,
        ),
    ]

    auxiliary_list = list(
        auxiliary_tables
    )

    for label, path in table_inputs:
        fields, rows = read_tsv(path)

        selected_rows, split_field = (
            development_rows(
                fields=fields,
                rows=rows,
                development_splits=development_splits,
            )
        )

        auxiliary_list.append((
            label,
            path,
            fields,
            selected_rows,
            split_field,
        ))

    field_inventory_rows: list[
        dict[str, Any]
    ] = []

    table_summary_rows: list[
        dict[str, Any]
    ] = []

    direct_root_candidates: list[
        dict[str, Any]
    ] = []

    for (
        label,
        path,
        fields,
        rows,
        split_field,
    ) in auxiliary_list:
        inventory = inventory_table_fields(
            repo=repo,
            table_label=label,
            table_path=path,
            fields=fields,
            rows=rows,
            split_field=split_field,
        )

        field_inventory_rows.extend(
            inventory
        )

        complete_fields = [
            row["field_name"]
            for row in inventory
            if row[
                "complete_rootlike_coverage"
            ]
        ]

        for field_name in complete_fields:
            direct_root_candidates.append({
                "table_label":
                    label,
                "table_path":
                    str(path),
                "field_name":
                    field_name,
                "development_rows":
                    len(rows),
            })

        table_summary_rows.append({
            "table_label":
                label,
            "table_path":
                str(path),
            "development_rows":
                len(rows),
            "field_count":
                len(fields),
            "split_field":
                split_field or "",
            "complete_root_fields_json":
                json.dumps(
                    complete_fields,
                    sort_keys=True,
                ),
        })

    candidate_schema_rows: list[
        dict[str, Any]
    ] = []

    nonzero_files_opened = 0
    zero_row_files_skipped = 0
    schema_count_counter: Counter[int] = (
        Counter()
    )
    source_root_schema_members = 0
    source_root_index_schema_members = 0
    analysis_sample_schema_members = 0
    metadata_key_counter: Counter[str] = (
        Counter()
    )

    for row in manifest_rows:
        candidate_rows = int(
            row["candidate_rows"]
        )

        source_path = Path(
            row["local_path"]
        ).expanduser().resolve()

        if candidate_rows == 0:
            zero_row_files_skipped += 1

            candidate_schema_rows.append({
                "member_index":
                    row["member_index"],
                "sample_class":
                    row["sample_class"],
                "dataset_split":
                    row["dataset_split"],
                "candidate_rows":
                    candidate_rows,
                "local_path":
                    str(source_path),
                "metadata_opened":
                    False,
                "schema_columns":
                    "",
                "has_source_root_column":
                    False,
                "has_source_root_index_column":
                    False,
                "has_analysis_sample_column":
                    False,
                "parquet_metadata_keys_json":
                    "[]",
            })

            continue

        require(
            source_path.is_file(),
            (
                "missing nonzero candidate Parquet: "
                f"{source_path}"
            ),
        )

        parquet_file = pq.ParquetFile(
            source_path
        )

        nonzero_files_opened += 1

        require(
            parquet_file.metadata.num_rows
            == candidate_rows,
            (
                f"{source_path.name}: expected "
                f"{candidate_rows} rows, metadata has "
                f"{parquet_file.metadata.num_rows}"
            ),
        )

        names = list(
            parquet_file.schema_arrow.names
        )

        schema_count_counter[len(names)] += 1

        has_source_root = (
            "source_root" in names
        )

        has_source_root_index = (
            "source_root_index" in names
        )

        has_analysis_sample = (
            "analysis_sample" in names
        )

        source_root_schema_members += int(
            has_source_root
        )

        source_root_index_schema_members += int(
            has_source_root_index
        )

        analysis_sample_schema_members += int(
            has_analysis_sample
        )

        keys = metadata_keys(
            parquet_file
        )

        for key in keys:
            metadata_key_counter[key] += 1

        candidate_schema_rows.append({
            "member_index":
                row["member_index"],
            "sample_class":
                row["sample_class"],
            "dataset_split":
                row["dataset_split"],
            "candidate_rows":
                candidate_rows,
            "local_path":
                str(source_path),
            "metadata_opened":
                True,
            "schema_columns":
                len(names),
            "has_source_root_column":
                has_source_root,
            "has_source_root_index_column":
                has_source_root_index,
            "has_analysis_sample_column":
                has_analysis_sample,
            "parquet_metadata_keys_json":
                json.dumps(keys),
        })

    require(
        nonzero_files_opened
        == int(
            expected[
                "nonzero_candidate_files"
            ]
        ),
        (
            "nonzero metadata-file count mismatch: "
            f"{nonzero_files_opened}"
        ),
    )

    require(
        zero_row_files_skipped
        == int(
            expected[
                "zero_row_candidate_files"
            ]
        ),
        (
            "zero-row file count mismatch: "
            f"{zero_row_files_skipped}"
        ),
    )

    require(
        schema_count_counter
        == Counter({
            72:
                int(
                    expected[
                        "canonical72_files"
                    ]
                ),
            75:
                int(
                    expected[
                        "canonical75_files"
                    ]
                ),
        }),
        (
            "observed schema-column counts changed: "
            f"{dict(schema_count_counter)}"
        ),
    )

    require(
        source_root_schema_members
        == int(
            expected[
                "candidate_files_with_source_root_column"
            ]
        ),
        (
            "source_root schema-member count changed: "
            f"{source_root_schema_members}"
        ),
    )

    require(
        source_root_index_schema_members
        == source_root_schema_members,
        (
            "source_root/source_root_index "
            "coverage mismatch"
        ),
    )

    require(
        analysis_sample_schema_members
        == source_root_schema_members,
        (
            "source_root/analysis_sample "
            "coverage mismatch"
        ),
    )

    builder_text = builder_path.read_text(
        encoding="utf-8",
        errors="strict",
    )

    builder_reference_rows = (
        frozen_builder_references(
            builder_text
        )
    )

    output_tmp.mkdir(
        parents=True,
        exist_ok=False,
    )

    table_summary_path = (
        output_tmp
        / "source_table_summary.tsv"
    )

    write_tsv(
        table_summary_path,
        table_summary_rows,
        [
            "table_label",
            "table_path",
            "development_rows",
            "field_count",
            "split_field",
            "complete_root_fields_json",
        ],
    )

    field_inventory_path = (
        output_tmp
        / "source_field_inventory.tsv"
    )

    write_tsv(
        field_inventory_path,
        field_inventory_rows,
        [
            "table_label",
            "table_path",
            "split_field",
            "development_rows",
            "field_name",
            "nonempty_values",
            "rootlike_values",
            "remote_root_values",
            "existing_local_files",
            "complete_nonempty_coverage",
            "complete_rootlike_coverage",
        ],
    )

    candidate_schema_path = (
        output_tmp
        / "candidate_provenance_schema_inventory.tsv"
    )

    write_tsv(
        candidate_schema_path,
        candidate_schema_rows,
        [
            "member_index",
            "sample_class",
            "dataset_split",
            "candidate_rows",
            "local_path",
            "metadata_opened",
            "schema_columns",
            "has_source_root_column",
            "has_source_root_index_column",
            "has_analysis_sample_column",
            "parquet_metadata_keys_json",
        ],
    )

    builder_reference_path = (
        output_tmp
        / "frozen_builder_branch_references.tsv"
    )

    write_tsv(
        builder_reference_path,
        builder_reference_rows,
        [
            "reference_type",
            "reference",
        ],
    )

    resolution_candidates = {
        "schema_version": 1,
        "direct_complete_root_field_candidates":
            direct_root_candidates,
        "candidate_parquet_provenance": {
            "members_with_source_root_column":
                source_root_schema_members,
            "members_with_source_root_index_column":
                source_root_index_schema_members,
            "members_with_analysis_sample_column":
                analysis_sample_schema_members,
            "event_values_read":
                0
        },
        "parquet_metadata_key_frequencies":
            dict(
                sorted(
                    metadata_key_counter.items()
                )
            ),
        "root_source_resolution_complete":
            False,
        "reason": (
            "No single registry field is assumed to "
            "be authoritative. Member-level values "
            "must be resolved from direct registry "
            "fields, candidate provenance columns, "
            "production receipts, and cache/source "
            "manifests."
        ),
        "next_gate":
            "build_member_level_root_source_resolution_map"
    }

    resolution_candidates_path = (
        output_tmp
        / "source_resolution_candidates.json"
    )

    resolution_candidates_path.write_text(
        json.dumps(
            resolution_candidates,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    summary = {
        "schema_version": 1,
        "status":
            "hh4b_3b_source_inventory_pass",
        "classification":
            "paper_quality_provenance_control_artifact",
        "analysis_name":
            config["analysis_name"],
        "source_commit":
            source_commit,
        "configuration":
            str(config_path),
        "configuration_sha256":
            sha256_file(config_path),
        "development_manifest":
            str(manifest_path),
        "development_manifest_sha256":
            sha256_file(manifest_path),
        "development_members":
            len(manifest_rows),
        "development_candidate_rows":
            candidate_row_total,
        "sample_class_member_counts":
            dict(class_counts),
        "dataset_split_member_counts":
            dict(split_counts),
        "nonzero_candidate_metadata_files_opened":
            nonzero_files_opened,
        "zero_row_candidate_files_skipped":
            zero_row_files_skipped,
        "observed_schema_column_counts":
            {
                str(key): value
                for key, value
                in sorted(
                    schema_count_counter.items()
                )
            },
        "members_with_source_root_schema":
            source_root_schema_members,
        "direct_complete_root_field_candidates":
            direct_root_candidates,
        "root_source_resolution_complete":
            False,
        "root_files_opened":
            0,
        "candidate_event_arrays_read":
            0,
        "candidate_provenance_values_read":
            0,
        "test_members_considered":
            0,
        "test_root_files_opened":
            0,
        "test_event_rows_read":
            0,
        "source_table_summary":
            str(
                output_dir
                / table_summary_path.name
            ),
        "source_field_inventory":
            str(
                output_dir
                / field_inventory_path.name
            ),
        "candidate_provenance_schema_inventory":
            str(
                output_dir
                / candidate_schema_path.name
            ),
        "source_resolution_candidates":
            str(
                output_dir
                / resolution_candidates_path.name
            ),
        "frozen_builder_branch_references":
            str(
                output_dir
                / builder_reference_path.name
            ),
        "next_gate":
            "build_member_level_root_source_resolution_map"
    }

    summary_path = (
        output_tmp
        / "hh4b_3b_source_inventory_summary.json"
    )

    summary_path.write_text(
        json.dumps(
            summary,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    output_checksum_files = sorted(
        path
        for path in output_tmp.iterdir()
        if path.is_file()
        and path.name != "SHA256SUMS"
    )

    (
        output_tmp / "SHA256SUMS"
    ).write_text(
        "\n".join(
            f"{sha256_file(path)}  {path.name}"
            for path in output_checksum_files
        )
        + "\n",
        encoding="utf-8",
    )

    output_tmp.rename(output_dir)

    checkpoint_tmp.mkdir(
        parents=True,
        exist_ok=False,
    )

    for source_name in (
        "hh4b_3b_source_inventory_summary.json",
        "source_table_summary.tsv",
        "source_field_inventory.tsv",
        "candidate_provenance_schema_inventory.tsv",
        "source_resolution_candidates.json",
        "frozen_builder_branch_references.tsv",
    ):
        copy_checkpoint_file(
            output_dir / source_name,
            checkpoint_tmp,
        )

    readme = f"""# HH4b 3b-control source inventory

## Purpose

This checkpoint inventories all available source-provenance channels
needed to reconstruct a three-b-tag control category from the frozen
HH4b development membership.

It deliberately does not assume that the candidate-output registries
contain one complete ROOT-path field.

## Failure corrected

The original preflight required a single signal-registry ROOT field with
complete coverage. No such field exists because ROOT provenance is
distributed across registries, candidate provenance columns, production
receipts, and cache/source manifests.

The absence of one complete registry field is not evidence that source
ROOT files are missing.

## Frozen membership

- Development members: {len(manifest_rows)}
- Signal members: {class_counts['signal']}
- Background members: {class_counts['background']}
- Train members: {split_counts['train']}
- Validation members: {split_counts['validation']}
- Development candidate rows: {candidate_row_total:,}
- Test members considered: 0

## Metadata findings

- Nonzero candidate Parquet metadata files opened:
  {nonzero_files_opened}
- Zero-row candidate files skipped:
  {zero_row_files_skipped}
- Files with 72 columns:
  {schema_count_counter[72]}
- Files with 75 columns:
  {schema_count_counter[75]}
- Candidate files whose schema contains `source_root`:
  {source_root_schema_members}
- Candidate event arrays read: 0
- ROOT files opened: 0

The 97 files with 75 columns contain the provenance columns
`source_root`, `source_root_index`, and `analysis_sample`.

## Interpretation

This checkpoint establishes the available evidence channels but does
not yet claim a complete member-level ROOT-source map.

The next stage must resolve each of the 581 development members using a
documented hierarchy:

1. direct member-level ROOT fields in authoritative manifests;
2. `source_root` provenance values in compatible candidate Parquets;
3. production receipts and materialization manifests;
4. cache/source mapping records.

No test source may be accessed.

## Reproducibility

- Source commit: `{source_commit}`
- Development manifest SHA-256:
  `{sha256_file(manifest_path)}`
- Frozen builder SHA-256:
  `{sha256_file(builder_path)}`
- Frozen policy SHA-256:
  `{sha256_file(policy_path)}`

## Next gate

`build_member_level_root_source_resolution_map`
"""

    (
        checkpoint_tmp / "README.md"
    ).write_text(
        readme,
        encoding="utf-8",
    )

    checkpoint_document = {
        "schema_version": 1,
        "status":
            "hh4b_3b_source_inventory_checkpoint",
        "classification":
            "paper_quality_provenance_control_artifact",
        "source_commit":
            source_commit,
        "development_members":
            len(manifest_rows),
        "members_with_source_root_schema":
            source_root_schema_members,
        "root_source_resolution_complete":
            False,
        "root_files_opened":
            0,
        "event_arrays_read":
            0,
        "test_members_considered":
            0,
        "next_gate":
            "build_member_level_root_source_resolution_map"
    }

    (
        checkpoint_tmp / "checkpoint.json"
    ).write_text(
        json.dumps(
            checkpoint_document,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    checkpoint_checksum_files = sorted(
        path
        for path in checkpoint_tmp.iterdir()
        if path.is_file()
        and path.name != "SHA256SUMS"
    )

    (
        checkpoint_tmp / "SHA256SUMS"
    ).write_text(
        "\n".join(
            f"{sha256_file(path)}  {path.name}"
            for path in checkpoint_checksum_files
        )
        + "\n",
        encoding="utf-8",
    )

    checkpoint_tmp.rename(
        checkpoint_dir
    )

    print("HH4B_3B_SOURCE_INVENTORY_PASS")
    print(
        "HH4B_DEVELOPMENT_MEMBERS="
        f"{len(manifest_rows)}"
    )
    print(
        "HH4B_SIGNAL_DEVELOPMENT_MEMBERS="
        f"{class_counts['signal']}"
    )
    print(
        "HH4B_BACKGROUND_DEVELOPMENT_MEMBERS="
        f"{class_counts['background']}"
    )
    print(
        "HH4B_NONZERO_CANDIDATE_METADATA_FILES_OPENED="
        f"{nonzero_files_opened}"
    )
    print(
        "HH4B_ZERO_ROW_CANDIDATE_FILES_SKIPPED="
        f"{zero_row_files_skipped}"
    )
    print(
        "HH4B_MEMBERS_WITH_SOURCE_ROOT_SCHEMA="
        f"{source_root_schema_members}"
    )
    print("HH4B_ROOT_SOURCE_RESOLUTION_COMPLETE=False")
    print("HH4B_ROOT_FILES_OPENED=0")
    print("HH4B_CANDIDATE_EVENT_ARRAYS_READ=0")
    print("HH4B_TEST_MEMBERS_CONSIDERED=0")
    print(
        "NEXT_GATE="
        "build_member_level_root_source_resolution_map"
    )
    print(
        "summary_json="
        f"{output_dir / summary_path.name}"
    )
    print(f"checkpoint={checkpoint_dir}")


if __name__ == "__main__":
    main()
