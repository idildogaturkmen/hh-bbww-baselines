#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow.parquet as pq


EXPECTED_FILES = 27
EXPECTED_ROWS = 998

REQUIRED_COLUMNS = [
    "sample",
    "event",
    "n_selected_bjets",
    "mbb1",
    "mbb2",
    "avg_mbb",
    "delta_mbb",
    "mhh",
    "drbb1",
    "drbb2",
    "pairing",
    "j1_pt",
    "j2_pt",
    "j3_pt",
    "j4_pt",
]

NUMERIC_COLUMNS = [
    "j1_pt",
    "j2_pt",
    "j3_pt",
    "j4_pt",
    "mbb1",
    "mbb2",
    "avg_mbb",
    "delta_mbb",
    "mhh",
    "drbb1",
    "drbb2",
]


def require(
    condition: bool,
    message: str,
) -> None:
    if not condition:
        raise RuntimeError(message)


def quantiles(
    values: np.ndarray,
) -> dict[str, float]:
    require(
        len(values) > 0,
        "cannot calculate quantiles of an empty array",
    )

    return {
        "minimum": float(np.min(values)),
        "q01": float(np.quantile(values, 0.01)),
        "q05": float(np.quantile(values, 0.05)),
        "median": float(np.quantile(values, 0.50)),
        "q95": float(np.quantile(values, 0.95)),
        "q99": float(np.quantile(values, 0.99)),
        "maximum": float(np.max(values)),
    }


def read_manifest(
    path: Path,
) -> list[dict[str, str]]:
    require(
        path.is_file(),
        f"missing source manifest {path}",
    )

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


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--manifest",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--outdir",
        type=Path,
        required=True,
    )

    arguments = parser.parse_args()

    manifest_path = arguments.manifest.resolve()
    outdir = arguments.outdir.resolve()

    require(
        not outdir.exists(),
        f"refusing to overwrite {outdir}",
    )

    manifest = read_manifest(
        manifest_path
    )

    local_members = [
        row
        for row in manifest
        if (
            row["availability"]
            == "local_candidate_parquet"
        )
    ]

    require(
        len(local_members) == EXPECTED_FILES,
        (
            f"expected {EXPECTED_FILES} local sources, "
            f"found {len(local_members)}"
        ),
    )

    values: dict[str, list[Any]] = {
        column: []
        for column in REQUIRED_COLUMNS
    }

    file_rows: list[dict[str, Any]] = []
    schema_orders: Counter[str] = Counter()
    event_keys: set[tuple[str, int]] = set()

    required_set = set(
        REQUIRED_COLUMNS
    )

    for manifest_row in local_members:
        path = Path(
            manifest_row[
                "local_candidate_parquet"
            ]
        ).resolve()

        require(
            path.is_file(),
            f"missing candidate Parquet {path}",
        )

        parquet_file = pq.ParquetFile(
            path
        )

        observed_columns = list(
            parquet_file.schema_arrow.names
        )

        observed_set = set(
            observed_columns
        )

        missing_columns = sorted(
            required_set
            - observed_set
        )

        extra_columns = sorted(
            observed_set
            - required_set
        )

        require(
            not missing_columns,
            (
                f"{path}: required columns are missing: "
                f"{missing_columns}"
            ),
        )

        require(
            not extra_columns,
            (
                f"{path}: unexpected columns are present: "
                f"{extra_columns}"
            ),
        )

        # Column order is allowed to differ. PyArrow returns only
        # the requested columns for the semantic audit.
        table = pq.read_table(
            path,
            columns=REQUIRED_COLUMNS,
        )

        schema_orders[
            " | ".join(
                observed_columns
            )
        ] += 1

        file_rows.append({
            "path":
                str(path),
            "rows":
                table.num_rows,
            "component":
                manifest_row["component"],
            "family":
                manifest_row["family"],
            "dataset_split":
                manifest_row[
                    "dataset_split"
                ],
            "observed_column_order":
                " | ".join(
                    observed_columns
                ),
        })

        for column in REQUIRED_COLUMNS:
            column_values = (
                table[column].to_pylist()
            )

            require(
                all(
                    value is not None
                    for value
                    in column_values
                ),
                (
                    f"{path}: null values found "
                    f"in {column}"
                ),
            )

            values[column].extend(
                column_values
            )

        for event in table[
            "event"
        ].to_pylist():
            key = (
                str(path),
                int(event),
            )

            require(
                key not in event_keys,
                (
                    "duplicate event index within "
                    f"candidate file: {key}"
                ),
            )

            event_keys.add(key)

    row_count = len(
        values["event"]
    )

    require(
        row_count == EXPECTED_ROWS,
        (
            f"expected {EXPECTED_ROWS} candidate rows, "
            f"found {row_count}"
        ),
    )

    for column in REQUIRED_COLUMNS:
        require(
            len(values[column])
            == row_count,
            (
                f"column {column} has inconsistent "
                "row count"
            ),
        )

    arrays: dict[str, np.ndarray] = {}

    for column in NUMERIC_COLUMNS:
        array = np.asarray(
            values[column],
            dtype=float,
        )

        require(
            np.all(
                np.isfinite(array)
            ),
            (
                f"nonfinite values found "
                f"in {column}"
            ),
        )

        arrays[column] = array

    n_selected_bjets = np.asarray(
        values[
            "n_selected_bjets"
        ],
        dtype=int,
    )

    j1 = arrays["j1_pt"]
    j2 = arrays["j2_pt"]
    j3 = arrays["j3_pt"]
    j4 = arrays["j4_pt"]

    mbb1 = arrays["mbb1"]
    mbb2 = arrays["mbb2"]
    avg_mbb = arrays["avg_mbb"]
    delta_mbb = arrays["delta_mbb"]
    mhh = arrays["mhh"]

    expected_avg_mbb = (
        mbb1 + mbb2
    ) / 2.0

    expected_abs_delta_mbb = np.abs(
        mbb1 - mbb2
    )

    expected_signed_delta_mbb = (
        mbb1 - mbb2
    )

    avg_mbb_residual = np.abs(
        avg_mbb
        - expected_avg_mbb
    )

    abs_delta_residual = np.abs(
        delta_mbb
        - expected_abs_delta_mbb
    )

    signed_delta_residual = np.abs(
        delta_mbb
        - expected_signed_delta_mbb
    )

    r_hh_direct = np.sqrt(
        (mbb1 - 125.0) ** 2
        + (mbb2 - 120.0) ** 2
    )

    r_hh_swapped = np.sqrt(
        (mbb2 - 125.0) ** 2
        + (mbb1 - 120.0) ** 2
    )

    r_hh_symmetric = np.minimum(
        r_hh_direct,
        r_hh_swapped,
    )

    all_pt_ge_25 = (
        (j1 >= 25.0)
        & (j2 >= 25.0)
        & (j3 >= 25.0)
        & (j4 >= 25.0)
    )

    all_pt_gt_30 = (
        (j1 > 30.0)
        & (j2 > 30.0)
        & (j3 > 30.0)
        & (j4 > 30.0)
    )

    all_pt_gt_40 = (
        (j1 > 40.0)
        & (j2 > 40.0)
        & (j3 > 40.0)
        & (j4 > 40.0)
    )

    all_pt_ge_40 = (
        (j1 >= 40.0)
        & (j2 >= 40.0)
        & (j3 >= 40.0)
        & (j4 >= 40.0)
    )

    pt_nonincreasing = (
        (j1 >= j2)
        & (j2 >= j3)
        & (j3 >= j4)
    )

    b4 = (
        n_selected_bjets >= 4
    )

    direct_signal = (
        r_hh_direct < 30.0
    )

    direct_analysis_region = (
        r_hh_direct < 55.0
    )

    symmetric_signal = (
        r_hh_symmetric < 30.0
    )

    symmetric_analysis_region = (
        r_hh_symmetric < 55.0
    )

    counts = {
        "all_rows":
            row_count,
        "n_selected_bjets_ge_4":
            int(np.sum(b4)),
        "all_four_pt_ge_25":
            int(
                np.sum(
                    all_pt_ge_25
                )
            ),
        "all_four_pt_gt_30":
            int(
                np.sum(
                    all_pt_gt_30
                )
            ),
        "all_four_pt_gt_40":
            int(
                np.sum(
                    all_pt_gt_40
                )
            ),
        "all_four_pt_ge_40":
            int(
                np.sum(
                    all_pt_ge_40
                )
            ),
        "jets_pt_nonincreasing":
            int(
                np.sum(
                    pt_nonincreasing
                )
            ),
        "direct_r_hh_lt_55":
            int(
                np.sum(
                    direct_analysis_region
                )
            ),
        "direct_r_hh_lt_30":
            int(
                np.sum(
                    direct_signal
                )
            ),
        "symmetric_r_hh_lt_55":
            int(
                np.sum(
                    symmetric_analysis_region
                )
            ),
        "symmetric_r_hh_lt_30":
            int(
                np.sum(
                    symmetric_signal
                )
            ),
        "pt40_and_direct_r_hh_lt_30":
            int(
                np.sum(
                    all_pt_gt_40
                    & direct_signal
                )
            ),
        "pt40_direct_r_hh_lt_30_low_mhh":
            int(
                np.sum(
                    all_pt_gt_40
                    & direct_signal
                    & (mhh < 450.0)
                )
            ),
        "pt40_direct_r_hh_lt_30_high_mhh":
            int(
                np.sum(
                    all_pt_gt_40
                    & direct_signal
                    & (mhh >= 450.0)
                )
            ),
    }

    summary = {
        "schema_version": 2,
        "status": "pass",
        "scope":
            (
                "27 local canonical legacy-ttbar "
                "candidate Parquets"
            ),
        "files_inspected":
            len(local_members),
        "candidate_rows":
            row_count,
        "required_column_set_valid":
            True,
        "column_order_required":
            False,
        "distinct_observed_column_orders":
            len(schema_orders),
        "observed_column_orders":
            dict(schema_orders),
        "samples":
            dict(
                Counter(
                    str(value)
                    for value
                    in values["sample"]
                )
            ),
        "pairing_values":
            dict(
                Counter(
                    str(value)
                    for value
                    in values["pairing"]
                )
            ),
        "n_selected_bjets_values": {
            str(key): count
            for key, count in sorted(
                Counter(
                    int(value)
                    for value
                    in values[
                        "n_selected_bjets"
                    ]
                ).items()
            )
        },
        "mbb_label_ordering": {
            "mbb1_greater_than_mbb2":
                int(
                    np.sum(
                        mbb1 > mbb2
                    )
                ),
            "mbb1_equal_to_mbb2":
                int(
                    np.sum(
                        mbb1 == mbb2
                    )
                ),
            "mbb1_less_than_mbb2":
                int(
                    np.sum(
                        mbb1 < mbb2
                    )
                ),
        },
        "derived_column_checks": {
            "avg_mbb_max_abs_residual":
                float(
                    np.max(
                        avg_mbb_residual
                    )
                ),
            "delta_mbb_abs_definition_max_residual":
                float(
                    np.max(
                        abs_delta_residual
                    )
                ),
            "delta_mbb_signed_definition_max_residual":
                float(
                    np.max(
                        signed_delta_residual
                    )
                ),
        },
        "jet_pt_quantiles": {
            "j1_pt":
                quantiles(j1),
            "j2_pt":
                quantiles(j2),
            "j3_pt":
                quantiles(j3),
            "j4_pt":
                quantiles(j4),
        },
        "r_hh_direct_quantiles":
            quantiles(
                r_hh_direct
            ),
        "r_hh_symmetric_quantiles":
            quantiles(
                r_hh_symmetric
            ),
        "mhh_quantiles":
            quantiles(mhh),
        "candidate_level_counts":
            counts,
        "missing_for_full_cms_object_baseline": [
            "jet_eta",
            "per_jet_btag_score",
            "electron_or_muon_information",
            "jet_phi",
            "jet_mass",
            "alternative_pairing_four_vectors",
            "higgs_candidate_pt",
        ],
        "reduced_candidate_cutflow_possible":
            True,
        "full_cms_object_cutflow_possible":
            False,
        "cms_cut_mapping_frozen":
            False,
    }

    outdir.mkdir(
        parents=True
    )

    summary_path = (
        outdir
        / "local_candidate_semantics_audit.json"
    )

    summary_path.write_text(
        json.dumps(
            summary,
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )

    files_path = (
        outdir
        / "local_candidate_files.tsv"
    )

    with files_path.open(
        "w",
        newline="",
    ) as handle:
        columns = [
            "path",
            "rows",
            "component",
            "family",
            "dataset_split",
            "observed_column_order",
        ]

        writer = csv.DictWriter(
            handle,
            fieldnames=columns,
            delimiter="\t",
            lineterminator="\n",
        )

        writer.writeheader()
        writer.writerows(
            file_rows
        )

    print(
        "HH4B_LOCAL_CANDIDATE_SEMANTICS_AUDIT_VALID"
    )
    print(
        f"FILES_INSPECTED={len(local_members)}"
    )
    print(
        f"CANDIDATE_ROWS={row_count}"
    )
    print(
        "COLUMN_SET_VALID"
    )
    print(
        "COLUMN_ORDER_NOT_REQUIRED"
    )
    print(
        "N_SELECTED_BJETS_VALUES="
        + json.dumps(
            summary[
                "n_selected_bjets_values"
            ],
            sort_keys=True,
        )
    )
    print(
        "PAIRING_VALUES="
        + json.dumps(
            summary[
                "pairing_values"
            ],
            sort_keys=True,
        )
    )
    print(
        "MBB_LABEL_ORDERING="
        + json.dumps(
            summary[
                "mbb_label_ordering"
            ],
            sort_keys=True,
        )
    )

    for key, value in counts.items():
        print(
            f"{key.upper()}={value}"
        )

    print(
        "AVG_MBB_MAX_ABS_RESIDUAL="
        f"{summary['derived_column_checks']['avg_mbb_max_abs_residual']}"
    )
    print(
        "DELTA_MBB_ABS_MAX_RESIDUAL="
        f"{summary['derived_column_checks']['delta_mbb_abs_definition_max_residual']}"
    )
    print(
        "DELTA_MBB_SIGNED_MAX_RESIDUAL="
        f"{summary['derived_column_checks']['delta_mbb_signed_definition_max_residual']}"
    )
    print(
        "REDUCED_CANDIDATE_CUTFLOW_POSSIBLE"
    )
    print(
        "FULL_CMS_OBJECT_CUTFLOW_NOT_POSSIBLE_FROM_CURRENT_SCHEMA"
    )
    print(
        "CMS_CUT_MAPPING_NOT_YET_FROZEN"
    )
    print(
        f"summary_json={summary_path}"
    )


if __name__ == "__main__":
    main()
