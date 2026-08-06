#!/usr/bin/env python3
"""Prepare immutable fold/category tables for the HH→4b cut scan."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any
import hashlib
import json
import os

import numpy as np
import pandas as pd
import pyarrow.parquet as pq


HASH_KEY = "0123456789123456"
EXPECTED_PRIMARY_SOURCES = 441
EXPECTED_PRIMARY_RESOLVED_ROWS = 1_042_397

BASE_COLUMNS = [
    "source_uid",
    "group_id",
    "source_fold",
    "event_uid",
    "sample_class",
    "process_or_mode",
    "auxiliary_qcd",
    "physical_evaluation_eligible",
    "candidate_tagged_jet_count",
    "mbb1",
    "mbb2",
    "r_hh_125_125",
    "ht_candidate_jets",
    "h2_pt",
    "drbb1",
    "drbb2",
    "mhh",
    "abs_h_delta_eta",
    "comparison_weight_outer_fold_0",
    "comparison_weight_outer_fold_1",
    "comparison_weight_outer_fold_2",
    "comparison_weight_outer_fold_3",
    "comparison_weight_outer_fold_4",
    "resolved_selection_contribution_weight",
]

DERIVED_COLUMNS = [
    "max_abs_mbb_minus_125",
    "abs_mbb1_minus_125",
    "abs_mbb2_minus_125",
    "max_drbb",
]


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def clean(value: Any) -> str:
    text = str(value).strip()
    if text.lower() in {"", "nan", "none", "null"}:
        return ""
    return text


def truthy(value: Any) -> bool:
    return clean(value).lower() in {"true", "1", "yes", "y"}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def deterministic_event_order(values: pd.Series) -> np.ndarray:
    hashes = pd.util.hash_pandas_object(
        values.astype(str),
        index=False,
        hash_key=HASH_KEY,
    ).to_numpy(dtype=np.uint64)
    return np.argsort(hashes, kind="mergesort")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--mode", choices=["canary", "full"], required=True)
    parser.add_argument("--rows-per-source-category", type=int, default=64)
    parser.add_argument("--expected-repository-head", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config_path = Path(args.config).resolve()
    config = json.loads(config_path.read_text())
    repo = Path(config["repository"]["path"]).resolve()
    output_root = Path(args.output_root).resolve()
    require(not output_root.exists(), f"output root exists: {output_root}")
    output_root.mkdir(parents=True)

    actual_head = os.popen(f"git -C {repo} rev-parse HEAD").read().strip()
    require(
        actual_head == args.expected_repository_head,
        "repository head mismatch",
    )

    plan = pd.read_csv(
        config["inputs"]["production_plan"],
        sep="\t",
        dtype=str,
        keep_default_na=False,
    )
    manifest = pd.read_csv(
        config["inputs"]["source_worker_manifest"],
        sep="\t",
        dtype=str,
        keep_default_na=False,
    )
    authorization = pd.read_csv(
        config["inputs"]["physical_authorization_registry"],
        sep="\t",
        dtype=str,
        keep_default_na=False,
    )

    for name, frame in [
        ("plan", plan),
        ("manifest", manifest),
        ("authorization", authorization),
    ]:
        frame["production_row_index"] = pd.to_numeric(
            frame["production_row_index"], errors="raise"
        ).astype(int)
        require(len(frame) == 464, f"{name} row count changed")
        require(
            frame["production_row_index"].is_unique,
            f"{name} production index is not unique",
        )

    merged = (
        manifest.merge(
            plan[
                [
                    "production_row_index",
                    "source_uid",
                    "final_resolved_path",
                ]
            ],
            on=["production_row_index", "source_uid"],
            how="inner",
            validate="one_to_one",
        )
        .merge(
            authorization[
                [
                    "production_row_index",
                    "source_uid",
                    "physical_weight_application_authorized",
                ]
            ],
            on=["production_row_index", "source_uid"],
            how="inner",
            validate="one_to_one",
        )
    )
    merged["source_fold"] = pd.to_numeric(
        merged["optimized_fold_k5"], errors="raise"
    ).astype(int)
    merged["is_auxiliary"] = merged["auxiliary_qcd"].map(truthy)
    merged["is_authorized"] = merged[
        "physical_weight_application_authorized"
    ].map(truthy)
    primary = merged.loc[
        (~merged["is_auxiliary"]) & merged["is_authorized"]
    ].copy()
    require(len(primary) == EXPECTED_PRIMARY_SOURCES, "primary count changed")

    parts: dict[tuple[int, str], list[pd.DataFrame]] = {
        (fold, category): []
        for fold in range(5)
        for category in ("exact3tag", "ge4tag")
    }
    source_rows = []
    all_event_uids: set[str] = set()
    total_resolved_rows = 0
    total_lt3_rows = 0

    for source in primary.itertuples(index=False):
        path = Path(clean(source.final_resolved_path))
        require(path.is_file(), f"missing resolved source: {path}")
        schema = set(pq.read_schema(path).names)
        missing = sorted(set(BASE_COLUMNS) - schema)
        require(not missing, f"missing columns in {source.source_uid}: {missing}")
        frame = pd.read_parquet(path, columns=BASE_COLUMNS)
        total_resolved_rows += len(frame)
        require(
            len(frame) == int(source.broad_eligible_rows),
            f"row closure failed for {source.source_uid}",
        )
        require(
            frame["source_uid"].astype(str).map(clean).eq(
                clean(source.source_uid)
            ).all(),
            f"source UID mismatch in {source.source_uid}",
        )
        require(
            pd.to_numeric(frame["source_fold"], errors="raise")
            .astype(int)
            .eq(int(source.source_fold))
            .all(),
            f"fold mismatch in {source.source_uid}",
        )
        require(
            ~frame["auxiliary_qcd"].astype(bool).any(),
            f"auxiliary row entered primary source {source.source_uid}",
        )
        require(
            frame["physical_evaluation_eligible"].astype(bool).all(),
            f"ineligible row entered {source.source_uid}",
        )
        uids = frame["event_uid"].astype(str)
        require(uids.is_unique, f"duplicate UID inside {source.source_uid}")
        overlap = all_event_uids.intersection(uids.tolist())
        require(not overlap, f"global UID collision in {source.source_uid}")
        all_event_uids.update(uids.tolist())

        frame = frame.copy()
        frame["source_fold"] = pd.to_numeric(
            frame["source_fold"], errors="raise"
        ).astype(int)
        frame["candidate_tagged_jet_count"] = pd.to_numeric(
            frame["candidate_tagged_jet_count"], errors="raise"
        ).astype(int)
        for column in [
            "mbb1",
            "mbb2",
            "r_hh_125_125",
            "ht_candidate_jets",
            "h2_pt",
            "drbb1",
            "drbb2",
            "mhh",
            "abs_h_delta_eta",
            "resolved_selection_contribution_weight",
        ]:
            frame[column] = pd.to_numeric(frame[column], errors="raise")
            require(np.isfinite(frame[column].to_numpy(dtype=float)).all(),
                    f"nonfinite {column} in {source.source_uid}")

        for outer_fold in range(5):
            column = f"comparison_weight_outer_fold_{outer_fold}"
            frame[column] = pd.to_numeric(frame[column], errors="coerce")
            own = frame["source_fold"].to_numpy(dtype=int) == outer_fold
            require(
                frame.loc[own, column].isna().all(),
                f"own-fold sentinel changed in {source.source_uid}",
            )
            development = frame.loc[~own, column].to_numpy(dtype=float)
            require(np.isfinite(development).all(),
                    f"nonfinite development weight in {source.source_uid}")
            require((development >= 0.0).all(),
                    f"negative development weight in {source.source_uid}")

        frame["max_abs_mbb_minus_125"] = np.maximum(
            np.abs(frame["mbb1"].to_numpy(dtype=float) - 125.0),
            np.abs(frame["mbb2"].to_numpy(dtype=float) - 125.0),
        )
        frame["abs_mbb1_minus_125"] = np.abs(
            frame["mbb1"].to_numpy(dtype=float) - 125.0
        )
        frame["abs_mbb2_minus_125"] = np.abs(
            frame["mbb2"].to_numpy(dtype=float) - 125.0
        )
        frame["max_drbb"] = np.maximum(
            frame["drbb1"].to_numpy(dtype=float),
            frame["drbb2"].to_numpy(dtype=float),
        )

        tagged = frame["candidate_tagged_jet_count"].to_numpy(dtype=int)
        total_lt3_rows += int(np.count_nonzero(tagged < 3))
        for category, selector in [
            ("exact3tag", tagged == 3),
            ("ge4tag", tagged >= 4),
        ]:
            category_frame = frame.loc[selector].copy()
            available = len(category_frame)
            if args.mode == "canary" and available > 0:
                require(
                    args.rows_per_source_category > 0,
                    "canary rows-per-source-category must be positive",
                )
                order = deterministic_event_order(category_frame["event_uid"])
                category_frame = category_frame.iloc[
                    order[: min(args.rows_per_source_category, available)]
                ].copy()
            sampled = len(category_frame)
            if sampled:
                parts[(int(source.source_fold), category)].append(
                    category_frame
                )
            source_rows.append(
                {
                    "production_row_index": int(source.production_row_index),
                    "source_uid": clean(source.source_uid),
                    "group_id": clean(source.group_id),
                    "source_fold": int(source.source_fold),
                    "sample_class": clean(source.sample_class),
                    "category_id": category,
                    "available_category_rows": available,
                    "written_category_rows": sampled,
                    "resolved_path": str(path),
                }
            )

    require(
        total_resolved_rows == EXPECTED_PRIMARY_RESOLVED_ROWS,
        "authorized primary resolved-row count changed",
    )
    require(
        len(all_event_uids) == EXPECTED_PRIMARY_RESOLVED_ROWS,
        "global UID count changed",
    )

    output_rows = []
    written_total = 0
    for fold in range(5):
        for category in ("exact3tag", "ge4tag"):
            key = (fold, category)
            require(parts[key], f"empty fold/category table: {key}")
            table = pd.concat(parts[key], ignore_index=True)
            table = table.sort_values(
                ["source_uid", "event_uid"], kind="mergesort"
            ).reset_index(drop=True)
            path = output_root / f"fold_{fold}" / f"{category}.parquet"
            path.parent.mkdir(parents=True, exist_ok=True)
            table.to_parquet(
                path,
                index=False,
                engine="pyarrow",
                compression="zstd",
            )
            written_total += len(table)
            output_rows.append(
                {
                    "fold": fold,
                    "category_id": category,
                    "rows": len(table),
                    "signal_rows": int(
                        (table["sample_class"].astype(str) == "signal").sum()
                    ),
                    "background_rows": int(
                        (table["sample_class"].astype(str) == "background").sum()
                    ),
                    "unique_sources": int(table["source_uid"].nunique()),
                    "unique_groups": int(table["group_id"].nunique()),
                    "path": str(path),
                    "sha256": sha256(path),
                }
            )

    pd.DataFrame(source_rows).to_csv(
        output_root / "source_category_manifest.tsv",
        sep="\t",
        index=False,
    )
    pd.DataFrame(output_rows).to_csv(
        output_root / "fold_category_table_manifest.tsv",
        sep="\t",
        index=False,
    )

    summary = {
        "schema_version": 1,
        "status": "pass_fold_category_tables_prepared",
        "mode": args.mode,
        "repository_head": actual_head,
        "authorized_primary_sources": EXPECTED_PRIMARY_SOURCES,
        "authorized_primary_resolved_rows": EXPECTED_PRIMARY_RESOLVED_ROWS,
        "rows_with_candidate_tagged_jet_count_lt3": total_lt3_rows,
        "written_exact3tag_and_ge4tag_rows": written_total,
        "rows_per_source_category": (
            args.rows_per_source_category if args.mode == "canary" else None
        ),
        "fold_category_tables": output_rows,
        "validation_payloads_opened": 0,
        "test_payloads_opened": 0,
    }
    (output_root / "fold_category_preparation_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n"
    )

    checksum_lines = []
    for path in sorted(p for p in output_root.rglob("*") if p.is_file()):
        if path.name == "SHA256SUMS":
            continue
        checksum_lines.append(
            f"{sha256(path)}  ./{path.relative_to(output_root)}"
        )
    (output_root / "SHA256SUMS").write_text(
        "\n".join(checksum_lines) + "\n"
    )

    print("FOLD_CATEGORY_PREPARATION=PASS")
    print(f"MODE={args.mode}")
    print(f"AUTHORIZED_PRIMARY_SOURCES={EXPECTED_PRIMARY_SOURCES}/441")
    print(f"AUTHORIZED_PRIMARY_RESOLVED_ROWS={total_resolved_rows}")
    print(f"ROWS_WITH_LT3_TAGS={total_lt3_rows}")
    print(f"WRITTEN_CATEGORY_ROWS={written_total}")
    print("FOLD_CATEGORY_TABLES=10/10")
    print("VALIDATION_PAYLOADS_OPENED=0")
    print("TEST_PAYLOADS_OPENED=0")
    print(f"OUTPUT_ROOT={output_root}")
    print("RESULT=HH4B_MULTIVARIATE_CUT_FOLD_TABLE_PREPARATION_PASS")


if __name__ == "__main__":
    main()
