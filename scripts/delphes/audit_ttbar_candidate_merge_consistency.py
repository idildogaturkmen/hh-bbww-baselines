#!/usr/bin/env python3

from collections import Counter
from pathlib import Path
import csv
import json
import sys

import numpy as np
import pandas as pd


store = Path(sys.argv[1]).resolve()
output_dir = Path(sys.argv[2]).resolve()
output_dir.mkdir(parents=True, exist_ok=True)

campaigns = [
    "ttbar_100k",
    "ttbar_200k",
]


def region_counts(frame):
    result = {
        "rows": int(len(frame)),
        "rhh_lt80": None,
        "rhh_lt50": None,
        "rhh_lt30": None,
        "mhh_gt600": None,
        "mhh_gt800": None,
        "rhh_lt50_and_mhh_gt400": None,
    }

    rhh = None

    if {"mbb1", "mbb2"}.issubset(frame.columns):
        rhh = np.sqrt(
            np.square(
                frame["mbb1"].to_numpy(dtype=float) - 125.0
            )
            + np.square(
                frame["mbb2"].to_numpy(dtype=float) - 125.0
            )
        )

    elif "r_hh" in frame.columns:
        rhh = frame["r_hh"].to_numpy(dtype=float)

    if rhh is not None:
        result["rhh_lt80"] = int((rhh < 80).sum())
        result["rhh_lt50"] = int((rhh < 50).sum())
        result["rhh_lt30"] = int((rhh < 30).sum())

    if "mhh" in frame.columns:
        mhh = frame["mhh"].to_numpy(dtype=float)

        result["mhh_gt600"] = int((mhh > 600).sum())
        result["mhh_gt800"] = int((mhh > 800).sum())

        if rhh is not None:
            result["rhh_lt50_and_mhh_gt400"] = int(
                ((rhh < 50) & (mhh > 400)).sum()
            )

    return result


def choose_key_columns(left, right):
    candidates = [
        ["sample", "event"],
        ["sample", "event_id"],
        ["run", "event"],
        ["run_name", "event"],
        ["source_sample", "event"],
    ]

    for columns in candidates:
        if all(
            column in left.columns
            and column in right.columns
            for column in columns
        ):
            return columns

    return []


def keys_from_columns(frame, columns):
    normalized = frame[columns].copy()

    for column in columns:
        normalized[column] = normalized[column].astype(str)

    return [
        tuple(row)
        for row in normalized.itertuples(
            index=False,
            name=None,
        )
    ]


report = {
    "schema_version": 1,
    "status": "candidate_merge_reconciliation_complete",
    "campaigns": {},
}

for campaign in campaigns:
    manifest_path = (
        store
        / "metadata"
        / f"{campaign}_manifest.csv"
    )

    merged_path = (
        store
        / "parquet"
        / f"{campaign}_merged_hh4b_candidates.parquet"
    )

    assert manifest_path.is_file(), manifest_path
    assert merged_path.is_file(), merged_path

    with manifest_path.open(newline="") as handle:
        manifest = list(csv.DictReader(handle))

    assert manifest

    shard_frames = []

    for manifest_row in manifest:
        path = Path(
            manifest_row["candidate_parquet"]
        )

        assert path.is_file(), path

        frame = pd.read_parquet(path)
        frame = frame.copy()

        frame["_manifest_shard"] = str(
            manifest_row["shard"]
        )

        frame["_manifest_seed"] = int(
            manifest_row["seed"]
        )

        frame["_manifest_path"] = str(path)

        shard_frames.append(frame)

    per_shard = pd.concat(
        shard_frames,
        ignore_index=True,
        sort=False,
    )

    merged = pd.read_parquet(merged_path)

    key_columns = choose_key_columns(
        per_shard,
        merged,
    )

    comparison = {
        "campaign": campaign,
        "manifest_path": str(
            manifest_path.resolve()
        ),
        "merged_path": str(
            merged_path.resolve()
        ),
        "manifest_shards": len(manifest),
        "per_shard_columns": sorted(
            column
            for column in per_shard.columns
            if not column.startswith("_manifest_")
        ),
        "merged_columns": sorted(
            merged.columns.astype(str)
        ),
        "only_in_per_shard_columns": sorted(
            set(per_shard.columns)
            - set(merged.columns)
            - {
                "_manifest_shard",
                "_manifest_seed",
                "_manifest_path",
            }
        ),
        "only_in_merged_columns": sorted(
            set(merged.columns)
            - set(per_shard.columns)
        ),
        "per_shard_regions": region_counts(
            per_shard
        ),
        "merged_regions": region_counts(
            merged
        ),
        "key_columns": key_columns,
    }

    comparison["region_differences_merged_minus_shards"] = {
        key: (
            comparison["merged_regions"][key]
            - comparison["per_shard_regions"][key]
            if (
                comparison["merged_regions"][key]
                is not None
                and comparison["per_shard_regions"][key]
                is not None
            )
            else None
        )
        for key in comparison[
            "per_shard_regions"
        ]
    }

    if key_columns:
        shard_keys = keys_from_columns(
            per_shard,
            key_columns,
        )

        merged_keys = keys_from_columns(
            merged,
            key_columns,
        )

        shard_counter = Counter(shard_keys)
        merged_counter = Counter(merged_keys)

        comparison.update(
            {
                "per_shard_unique_keys": len(
                    shard_counter
                ),
                "merged_unique_keys": len(
                    merged_counter
                ),
                "per_shard_duplicate_rows_by_key": (
                    len(shard_keys)
                    - len(shard_counter)
                ),
                "merged_duplicate_rows_by_key": (
                    len(merged_keys)
                    - len(merged_counter)
                ),
                "rows_only_in_merged_by_key": sum(
                    (
                        merged_counter
                        - shard_counter
                    ).values()
                ),
                "rows_only_in_shards_by_key": sum(
                    (
                        shard_counter
                        - merged_counter
                    ).values()
                ),
            }
        )

        only_merged = list(
            (
                merged_counter
                - shard_counter
            ).elements()
        )

        only_shards = list(
            (
                shard_counter
                - merged_counter
            ).elements()
        )

        comparison[
            "first_keys_only_in_merged"
        ] = [
            list(value)
            for value in only_merged[:30]
        ]

        comparison[
            "first_keys_only_in_shards"
        ] = [
            list(value)
            for value in only_shards[:30]
        ]

    else:
        common_columns = sorted(
            set(per_shard.columns)
            & set(merged.columns)
        )

        common_columns = [
            column
            for column in common_columns
            if not column.startswith("_manifest_")
        ]

        assert common_columns

        shard_hashes = (
            pd.util.hash_pandas_object(
                per_shard[common_columns],
                index=False,
            )
            .astype(str)
            .tolist()
        )

        merged_hashes = (
            pd.util.hash_pandas_object(
                merged[common_columns],
                index=False,
            )
            .astype(str)
            .tolist()
        )

        shard_counter = Counter(shard_hashes)
        merged_counter = Counter(merged_hashes)

        comparison.update(
            {
                "comparison_mode": "common_column_row_hash",
                "common_hash_columns": common_columns,
                "rows_only_in_merged_by_hash": sum(
                    (
                        merged_counter
                        - shard_counter
                    ).values()
                ),
                "rows_only_in_shards_by_hash": sum(
                    (
                        shard_counter
                        - merged_counter
                    ).values()
                ),
            }
        )

    report["campaigns"][campaign] = comparison

report_path = (
    output_dir
    / "ttbar_candidate_merge_reconciliation.json"
)

report_path.write_text(
    json.dumps(
        report,
        indent=2,
        sort_keys=True,
    )
    + "\n"
)

for campaign, comparison in report[
    "campaigns"
].items():
    print()
    print("===", campaign, "===")

    print(
        "Per-shard:",
        comparison["per_shard_regions"],
    )

    print(
        "Merged:",
        comparison["merged_regions"],
    )

    print(
        "Differences:",
        comparison[
            "region_differences_merged_minus_shards"
        ],
    )

    print(
        "Key columns:",
        comparison["key_columns"],
    )

    if comparison["key_columns"]:
        print(
            "Rows only in merged:",
            comparison[
                "rows_only_in_merged_by_key"
            ],
        )

        print(
            "Rows only in shards:",
            comparison[
                "rows_only_in_shards_by_key"
            ],
        )

        print(
            "Merged duplicate keys:",
            comparison[
                "merged_duplicate_rows_by_key"
            ],
        )

print()
print("TTBAR_CANDIDATE_MERGE_RECONCILIATION_COMPLETE")
print("report:", report_path)
