#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import subprocess
import tempfile
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow.parquet as pq
import yaml


REQUIRED_COLUMNS = [
    "event",
    "n_selected_bjets",
    "j1_pt",
    "j2_pt",
    "j3_pt",
    "j4_pt",
    "j1_eta",
    "j2_eta",
    "j3_eta",
    "j4_eta",
    "r_hh_125_120",
    "mhh",
]

STAGE_ORDER = [
    "candidate_denominator",
    "four_candidate_jets_pt40",
    "four_candidate_jets_abs_eta24",
    "analysis_region_rhh55",
    "signal_region_rhh30",
    "control_region_rhh30_55",
    "signal_region_low_mhh",
    "signal_region_high_mhh",
    "control_region_low_mhh",
    "control_region_high_mhh",
]


def require(
    condition: bool,
    message: str,
) -> None:
    if not condition:
        raise RuntimeError(message)


def sha256_file(
    path: Path,
) -> str:
    digest = hashlib.sha256()

    with path.open("rb") as handle:
        for block in iter(
            lambda: handle.read(1024 * 1024),
            b"",
        ):
            digest.update(block)

    return digest.hexdigest()


def load_json(
    path: Path,
) -> dict[str, Any]:
    document = json.loads(
        path.read_text(errors="replace")
    )

    require(
        isinstance(document, dict),
        f"{path}: JSON is not an object",
    )

    return document


def load_tsv(
    path: Path,
) -> list[dict[str, str]]:
    with path.open(
        newline="",
        errors="replace",
    ) as handle:
        return list(
            csv.DictReader(
                handle,
                delimiter="\t",
            )
        )


def xrootd_url(
    eos_host: str,
    lfn: str,
) -> str:
    return (
        eos_host.rstrip("/")
        + "//"
        + lfn.lstrip("/")
    )


def materialize_candidate(
    *,
    row: dict[str, str],
    eos_host: str,
    temporary: Path,
) -> Path:
    access = row["candidate_access"]
    source = row["candidate_parquet"]

    if access == "local_path":
        path = Path(source)

        require(
            path.is_file(),
            f"missing local candidate {path}",
        )

        return path

    require(
        access == "eos_lfn",
        f"unknown candidate access mode {access!r}",
    )

    require(
        source.startswith("/store/"),
        f"invalid EOS candidate LFN {source!r}",
    )

    destination = (
        temporary
        / (
            f"{int(row['global_member_index']):03d}_"
            "candidate.parquet"
        )
    )

    subprocess.run(
        [
            "xrdcp",
            "-f",
            "--nopbar",
            xrootd_url(
                eos_host,
                source,
            ),
            str(destination),
        ],
        check=True,
    )

    require(
        destination.is_file(),
        f"xrdcp did not create {destination}",
    )

    return destination


def column_numpy(
    table: Any,
    name: str,
) -> np.ndarray:
    column = table[name].combine_chunks()

    require(
        column.null_count == 0,
        f"column {name} contains null values",
    )

    return column.to_numpy(
        zero_copy_only=False
    )


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--registry",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--registry-summary",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--baseline",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--eos-host",
        required=True,
    )

    parser.add_argument(
        "--outdir",
        type=Path,
        required=True,
    )

    args = parser.parse_args()

    for path in (
        args.registry,
        args.registry_summary,
        args.baseline,
    ):
        require(
            path.is_file(),
            f"missing required input {path}",
        )

    require(
        not args.outdir.exists(),
        f"refusing to overwrite {args.outdir}",
    )

    registry = load_tsv(
        args.registry
    )

    summary = load_json(
        args.registry_summary
    )

    baseline = yaml.safe_load(
        args.baseline.read_text(
            errors="replace"
        )
    )

    require(
        isinstance(baseline, dict),
        "baseline YAML is not a mapping",
    )

    require(
        summary["status"] == "pass",
        "combined registry status is not pass",
    )

    require(
        int(summary["canonical_members"]) == 110,
        "combined registry does not contain 110 members",
    )

    require(
        int(summary["generated_events"]) == 200_000,
        "combined generated total is not 200000",
    )

    require(
        int(summary["candidate_rows"]) == 12_395,
        "combined candidate total is not 12395",
    )

    require(
        summary["physics_yield_authorized"] is False,
        "physics yield is unexpectedly authorized",
    )

    require(
        summary[
            "full_signal_background_cutflow_authorized"
        ]
        is False,
        "full signal/background cutflow unexpectedly authorized",
    )

    require(
        baseline["status"] == "frozen_reference",
        "baseline is not frozen",
    )

    require(
        baseline["baseline_name"]
        == "hh4b_cms_run2_resolved_rich_v2_reference_v1",
        "unexpected baseline name",
    )

    pt_threshold = float(
        baseline[
            "cuts"
        ][
            "four_candidate_jets_pt40"
        ][
            "threshold_gev"
        ]
    )

    eta_threshold = float(
        baseline[
            "cuts"
        ][
            "four_candidate_jets_abs_eta24"
        ][
            "threshold"
        ]
    )

    analysis_rhh = float(
        baseline[
            "regions"
        ][
            "analysis_region"
        ][
            "upper_bound"
        ]
    )

    signal_rhh = float(
        baseline[
            "regions"
        ][
            "signal_region"
        ][
            "upper_bound"
        ]
    )

    control_lower = float(
        baseline[
            "regions"
        ][
            "control_region"
        ][
            "lower_bound"
        ]
    )

    control_upper = float(
        baseline[
            "regions"
        ][
            "control_region"
        ][
            "upper_bound"
        ]
    )

    mhh_boundary = float(
        baseline[
            "categories"
        ][
            "low_mhh"
        ][
            "upper_bound_gev"
        ]
    )

    require(
        pt_threshold == 40.0,
        "unexpected pT threshold",
    )

    require(
        eta_threshold == 2.4,
        "unexpected eta threshold",
    )

    require(
        analysis_rhh == 55.0,
        "unexpected analysis-region boundary",
    )

    require(
        signal_rhh == 30.0,
        "unexpected signal-region boundary",
    )

    require(
        control_lower == 30.0
        and control_upper == 55.0,
        "unexpected control-region boundaries",
    )

    require(
        mhh_boundary == 450.0,
        "unexpected mHH boundary",
    )

    require(
        len(registry) == 110,
        f"registry has {len(registry)} rows, not 110",
    )

    expected_indices = {
        int(row["global_member_index"])
        for row in registry
    }

    require(
        expected_indices == set(range(110)),
        "global member indices are not 0 through 109",
    )

    generated_denominators: Counter[
        tuple[str, str]
    ] = Counter()

    candidate_denominators: Counter[
        tuple[str, str]
    ] = Counter()

    selected_counts: Counter[
        tuple[str, str, str]
    ] = Counter()

    processed_members: Counter[
        tuple[str, str]
    ] = Counter()

    skipped_test_members = 0
    skipped_test_generated = 0
    skipped_test_candidates = 0

    member_rows: list[
        dict[str, object]
    ] = []

    with tempfile.TemporaryDirectory(
        prefix="hh4b_signal_reference_cutflow.",
        dir="/tmp",
    ) as temporary_directory:
        temporary = Path(
            temporary_directory
        )

        for row in sorted(
            registry,
            key=lambda item: int(
                item["global_member_index"]
            ),
        ):
            mode = row["signal_mode"]
            split = row["immutable_split"]

            generated_events = int(
                row["generated_events"]
            )

            expected_candidate_rows = int(
                row["candidate_rows"]
            )

            if split == "test":
                require(
                    mode == "ggf_hh4b",
                    "non-ggF test member encountered",
                )

                require(
                    row["test_sealed"].lower()
                    == "true",
                    "ggF test member is not sealed",
                )

                skipped_test_members += 1
                skipped_test_generated += (
                    generated_events
                )
                skipped_test_candidates += (
                    expected_candidate_rows
                )

                continue

            require(
                split in {
                    "train",
                    "validation",
                },
                f"unexpected processed split {split!r}",
            )

            candidate_path = materialize_candidate(
                row=row,
                eos_host=args.eos_host,
                temporary=temporary,
            )

            actual_sha256 = sha256_file(
                candidate_path
            )

            require(
                actual_sha256
                == row[
                    "candidate_parquet_sha256"
                ],
                (
                    f"{row['member_id']}: "
                    "candidate SHA-256 mismatch"
                ),
            )

            parquet = pq.ParquetFile(
                candidate_path
            )

            actual_rows = int(
                parquet.metadata.num_rows
            )

            require(
                actual_rows
                == expected_candidate_rows,
                (
                    f"{row['member_id']}: "
                    f"rows={actual_rows}, "
                    f"expected={expected_candidate_rows}"
                ),
            )

            available_columns = set(
                parquet.schema_arrow.names
            )

            missing_columns = sorted(
                set(REQUIRED_COLUMNS)
                - available_columns
            )

            require(
                not missing_columns,
                (
                    f"{row['member_id']}: "
                    f"missing columns {missing_columns}"
                ),
            )

            table = pq.read_table(
                candidate_path,
                columns=REQUIRED_COLUMNS,
            )

            event = column_numpy(
                table,
                "event",
            )

            n_selected_bjets = column_numpy(
                table,
                "n_selected_bjets",
            )

            pt = np.column_stack([
                column_numpy(table, "j1_pt"),
                column_numpy(table, "j2_pt"),
                column_numpy(table, "j3_pt"),
                column_numpy(table, "j4_pt"),
            ])

            eta = np.column_stack([
                column_numpy(table, "j1_eta"),
                column_numpy(table, "j2_eta"),
                column_numpy(table, "j3_eta"),
                column_numpy(table, "j4_eta"),
            ])

            rhh = column_numpy(
                table,
                "r_hh_125_120",
            )

            mhh = column_numpy(
                table,
                "mhh",
            )

            require(
                len(np.unique(event))
                == actual_rows,
                (
                    f"{row['member_id']}: "
                    "event identifiers are not unique"
                ),
            )

            require(
                np.all(
                    n_selected_bjets >= 4
                ),
                (
                    f"{row['member_id']}: "
                    "candidate denominator contains "
                    "fewer than four selected b jets"
                ),
            )

            require(
                np.all(np.isfinite(pt)),
                f"{row['member_id']}: nonfinite jet pT",
            )

            require(
                np.all(np.isfinite(eta)),
                f"{row['member_id']}: nonfinite jet eta",
            )

            require(
                np.all(np.isfinite(rhh)),
                f"{row['member_id']}: nonfinite RHH",
            )

            require(
                np.all(np.isfinite(mhh)),
                f"{row['member_id']}: nonfinite mHH",
            )

            denominator = np.ones(
                actual_rows,
                dtype=bool,
            )

            pt40 = (
                denominator
                & np.all(
                    pt > pt_threshold,
                    axis=1,
                )
            )

            eta24 = (
                pt40
                & np.all(
                    np.abs(eta)
                    < eta_threshold,
                    axis=1,
                )
            )

            analysis_region = (
                eta24
                & (rhh < analysis_rhh)
            )

            signal_region = (
                eta24
                & (rhh < signal_rhh)
            )

            control_region = (
                eta24
                & (rhh >= control_lower)
                & (rhh < control_upper)
            )

            signal_low_mhh = (
                signal_region
                & (mhh < mhh_boundary)
            )

            signal_high_mhh = (
                signal_region
                & (mhh >= mhh_boundary)
            )

            control_low_mhh = (
                control_region
                & (mhh < mhh_boundary)
            )

            control_high_mhh = (
                control_region
                & (mhh >= mhh_boundary)
            )

            require(
                not np.any(
                    signal_region
                    & control_region
                ),
                (
                    f"{row['member_id']}: "
                    "signal and control regions overlap"
                ),
            )

            require(
                np.array_equal(
                    analysis_region,
                    signal_region
                    | control_region,
                ),
                (
                    f"{row['member_id']}: "
                    "analysis region does not equal "
                    "signal union control"
                ),
            )

            stage_masks = {
                "candidate_denominator":
                    denominator,
                "four_candidate_jets_pt40":
                    pt40,
                "four_candidate_jets_abs_eta24":
                    eta24,
                "analysis_region_rhh55":
                    analysis_region,
                "signal_region_rhh30":
                    signal_region,
                "control_region_rhh30_55":
                    control_region,
                "signal_region_low_mhh":
                    signal_low_mhh,
                "signal_region_high_mhh":
                    signal_high_mhh,
                "control_region_low_mhh":
                    control_low_mhh,
                "control_region_high_mhh":
                    control_high_mhh,
            }

            key = (
                mode,
                split,
            )

            generated_denominators[
                key
            ] += generated_events

            candidate_denominators[
                key
            ] += actual_rows

            processed_members[
                key
            ] += 1

            member_output: dict[
                str,
                object
            ] = {
                "global_member_index":
                    int(
                        row[
                            "global_member_index"
                        ]
                    ),
                "signal_mode":
                    mode,
                "immutable_split":
                    split,
                "member_id":
                    row["member_id"],
                "generated_events":
                    generated_events,
                "candidate_rows":
                    actual_rows,
            }

            for stage in STAGE_ORDER:
                selected = int(
                    np.count_nonzero(
                        stage_masks[stage]
                    )
                )

                selected_counts[
                    (
                        mode,
                        split,
                        stage,
                    )
                ] += selected

                member_output[
                    stage
                ] = selected

            member_rows.append(
                member_output
            )

            print(
                "HH4B_SIGNAL_CUTFLOW_PROGRESS="
                f"{int(row['global_member_index']) + 1}/110"
            )

    require(
        skipped_test_members == 3,
        (
            "expected three skipped sealed-test "
            f"members, found {skipped_test_members}"
        ),
    )

    require(
        skipped_test_generated == 3_000,
        (
            "expected 3000 skipped test events, "
            f"found {skipped_test_generated}"
        ),
    )

    require(
        skipped_test_candidates == 174,
        (
            "expected 174 skipped test candidates, "
            f"found {skipped_test_candidates}"
        ),
    )

    require(
        sum(processed_members.values()) == 107,
        "processed member total is not 107",
    )

    require(
        sum(generated_denominators.values())
        == 197_000,
        "processed generated-event total is not 197000",
    )

    require(
        sum(candidate_denominators.values())
        == 12_221,
        "processed candidate total is not 12221",
    )

    args.outdir.mkdir(
        parents=True,
        exist_ok=False,
    )

    member_output_path = (
        args.outdir
        / "hh4b_signal_reference_cutflow_members.tsv"
    )

    with member_output_path.open(
        "w",
        newline="",
    ) as handle:
        columns = list(
            member_rows[0]
        )

        writer = csv.DictWriter(
            handle,
            fieldnames=columns,
            delimiter="\t",
            lineterminator="\n",
        )

        writer.writeheader()
        writer.writerows(
            member_rows
        )

    cutflow_rows: list[
        dict[str, object]
    ] = []

    for mode, split in sorted(
        generated_denominators
    ):
        generated_total = (
            generated_denominators[
                (mode, split)
            ]
        )

        candidate_total = (
            candidate_denominators[
                (mode, split)
            ]
        )

        for stage in STAGE_ORDER:
            selected = selected_counts[
                (
                    mode,
                    split,
                    stage,
                )
            ]

            cutflow_rows.append({
                "signal_mode":
                    mode,
                "split":
                    split,
                "stage":
                    stage,
                "members":
                    processed_members[
                        (mode, split)
                    ],
                "generated_events":
                    generated_total,
                "candidate_denominator":
                    candidate_total,
                "selected_events":
                    selected,
                "efficiency_vs_generated":
                    selected
                    / generated_total,
                "efficiency_vs_candidate":
                    selected
                    / candidate_total,
            })

    cutflow_output_path = (
        args.outdir
        / "hh4b_signal_reference_cutflow.tsv"
    )

    with cutflow_output_path.open(
        "w",
        newline="",
    ) as handle:
        columns = list(
            cutflow_rows[0]
        )

        writer = csv.DictWriter(
            handle,
            fieldnames=columns,
            delimiter="\t",
            lineterminator="\n",
        )

        writer.writeheader()
        writer.writerows(
            cutflow_rows
        )

    output_summary = {
        "schema_version": 1,
        "status": "pass",
        "baseline_name":
            baseline["baseline_name"],
        "baseline_sha256":
            sha256_file(args.baseline),
        "registry_sha256":
            sha256_file(args.registry),
        "processed_splits": [
            "train",
            "validation",
        ],
        "processed_members":
            sum(processed_members.values()),
        "processed_generated_events":
            sum(
                generated_denominators.values()
            ),
        "processed_candidate_rows":
            sum(
                candidate_denominators.values()
            ),
        "sealed_test_accessed": False,
        "sealed_test_members_skipped":
            skipped_test_members,
        "sealed_test_generated_events_skipped":
            skipped_test_generated,
        "sealed_test_candidate_rows_skipped":
            skipped_test_candidates,
        "signal_modes_combined": False,
        "optimization_performed": False,
        "physical_yields_reported": False,
        "significance_reported": False,
        "cutflow":
            str(cutflow_output_path),
        "member_cutflow":
            str(member_output_path),
    }

    summary_output_path = (
        args.outdir
        / "hh4b_signal_reference_cutflow.json"
    )

    summary_output_path.write_text(
        json.dumps(
            output_summary,
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )

    print(
        "HH4B_SIGNAL_REFERENCE_CUTFLOW_VALID"
    )
    print(
        "HH4B_SIGNAL_REFERENCE_CUTFLOW_TRAIN_VALIDATION_ONLY"
    )
    print(
        "HH4B_SIGNAL_SEALED_TEST_NOT_ACCESSED"
    )
    print(
        "HH4B_SIGNAL_MODES_REPORTED_SEPARATELY"
    )
    print(
        "HH4B_SIGNAL_OPTIMIZATION_NOT_PERFORMED"
    )
    print(
        "HH4B_SIGNAL_PHYSICAL_YIELDS_NOT_REPORTED"
    )
    print(
        f"summary_json={summary_output_path}"
    )
    print(
        f"cutflow_tsv={cutflow_output_path}"
    )


if __name__ == "__main__":
    main()
