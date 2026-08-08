#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
import hashlib
import json

import numpy as np
import pandas as pd


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--nominal-root", required=True)
    p.add_argument("--draw-parquet", required=True)
    p.add_argument("--replica", type=int, required=True)
    p.add_argument("--output-root", required=True)
    p.add_argument("--summary-json", required=True)
    return p.parse_args()


def main() -> None:
    args = parse_args()

    nominal_root = Path(args.nominal_root).resolve()
    draw_path = Path(args.draw_parquet).resolve()
    output_root = Path(args.output_root).resolve()
    summary_path = Path(args.summary_json).resolve()

    require(args.replica >= 0, "replica must be nonnegative")
    require(nominal_root.is_dir(), "nominal fold root missing")
    require(draw_path.is_file(), "draw parquet missing")
    require(not output_root.exists(), "replica output root already exists")

    draws = pd.read_parquet(draw_path)
    required_draw = {"replica", "group_id", "multiplicity", "sample_class"}
    require(required_draw.issubset(draws.columns), "draw schema mismatch")

    replica = draws[
        pd.to_numeric(draws["replica"], errors="raise").astype(int)
        == args.replica
    ].copy()
    require(len(replica) > 0, "replica absent from draw registry")

    replica["group_id"] = replica["group_id"].astype(str)
    replica["multiplicity"] = pd.to_numeric(
        replica["multiplicity"], errors="raise"
    ).astype(int)
    require((replica["multiplicity"] >= 1).all(), "nonpositive multiplicity")
    require(
        not replica["group_id"].duplicated().any(),
        "duplicate group_id in sparse draw replica",
    )

    mult = replica.set_index("group_id")["multiplicity"].to_dict()

    tables = sorted(nominal_root.rglob("*.parquet"))
    require(len(tables) == 10, f"expected 10 nominal tables, found {len(tables)}")

    output_root.mkdir(parents=True)
    table_records = []
    all_materialized_uids = []

    for src in tables:
        rel = src.relative_to(nominal_root)
        require(len(rel.parts) == 2, f"unexpected table layout: {rel}")
        fold_dir, _ = rel.parts
        require(fold_dir.startswith("fold_"), f"unexpected fold dir: {fold_dir}")

        frame = pd.read_parquet(src)
        for col in ("group_id", "event_uid", "source_fold", "sample_class"):
            require(col in frame.columns, f"{src} missing {col}")

        frame = frame.copy()
        frame["group_id"] = frame["group_id"].astype(str)

        multiplicity = frame["group_id"].map(mult).fillna(0).astype(int)
        require((multiplicity >= 0).all(), "negative mapped multiplicity")

        max_mult = int(multiplicity.max()) if len(multiplicity) else 0
        chunks = []

        for copy_index in range(max_mult):
            keep = multiplicity > copy_index
            if not keep.any():
                continue
            chunk = frame.loc[keep].copy()
            chunk["event_uid"] = (
                chunk["event_uid"].astype(str)
                + f"::bootstrap_replica{args.replica:04d}"
                + f"::sourcecopy{copy_index:03d}"
            )
            chunks.append(chunk)

        out = (
            pd.concat(chunks, ignore_index=True)
            if chunks
            else frame.iloc[0:0].copy()
        )

        require(
            out["event_uid"].astype(str).is_unique,
            f"materialized event_uid collision in {rel}",
        )

        expected_rows = int(multiplicity.sum())
        require(len(out) == expected_rows, f"row multiplicity closure failed: {rel}")

        if len(out):
            expected_fold = int(fold_dir.split("_")[1])
            require(
                pd.to_numeric(out["source_fold"], errors="raise")
                .astype(int)
                .eq(expected_fold)
                .all(),
                f"source_fold changed in {rel}",
            )

        weight_columns = [
            c for c in frame.columns
            if c.startswith("comparison_weight_outer_fold_")
        ]
        if "resolved_selection_contribution_weight" in frame.columns:
            weight_columns.append("resolved_selection_contribution_weight")

        for col in weight_columns:
            values = pd.to_numeric(frame[col], errors="coerce").to_numpy(float)
            m = multiplicity.to_numpy(float)
            out_values = pd.to_numeric(out[col], errors="coerce").to_numpy(float)

            expected_sum = float(np.nansum(values * m))
            actual_sum = float(np.nansum(out_values))
            expected_sumw2 = float(np.nansum(np.square(values) * m))
            actual_sumw2 = float(np.nansum(np.square(out_values)))

            require(
                np.isclose(actual_sum, expected_sum, rtol=1e-12, atol=1e-9),
                f"sumw closure failed {rel} {col}",
            )
            require(
                np.isclose(actual_sumw2, expected_sumw2, rtol=1e-12, atol=1e-6),
                f"sumw2 closure failed {rel} {col}",
            )

        dst = output_root / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        out.to_parquet(dst, index=False)

        all_materialized_uids.extend(out["event_uid"].astype(str).tolist())

        table_records.append(
            {
                "relative_path": str(rel),
                "nominal_rows": int(len(frame)),
                "materialized_rows": int(len(out)),
                "nominal_unique_groups": int(frame["group_id"].nunique()),
                "materialized_unique_groups": int(out["group_id"].nunique()),
                "output_sha256": sha256(dst),
            }
        )

    require(
        len(all_materialized_uids) == len(set(all_materialized_uids)),
        "event_uid collision across all ten materialized tables",
    )

    summary = {
        "schema_version": 1,
        "status": "pass_source_copy_bootstrap_replica_materialization",
        "replica": args.replica,
        "method": "literal whole-source row-copy resampling",
        "per_event_weights_scaled_by_multiplicity": False,
        "source_rows_duplicated_by_multiplicity": True,
        "event_uid_bootstrap_suffix_applied": True,
        "source_fold_reassigned": False,
        "drawn_sparse_groups": int(len(replica)),
        "total_source_multiplicity": int(replica["multiplicity"].sum()),
        "table_count": len(table_records),
        "total_materialized_rows": int(
            sum(x["materialized_rows"] for x in table_records)
        ),
        "tables": table_records,
        "validation_payloads_opened": 0,
        "test_payloads_opened": 0,
    }
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")

    print("SOURCE_COPY_BOOTSTRAP_MATERIALIZATION=PASS")
    print(f"REPLICA={args.replica}")
    print(f"DRAWN_SPARSE_GROUPS={len(replica)}")
    print(f"TOTAL_SOURCE_MULTIPLICITY={int(replica['multiplicity'].sum())}")
    print(f"TABLE_COUNT={len(table_records)}")
    print(
        "TOTAL_MATERIALIZED_ROWS="
        f"{sum(x['materialized_rows'] for x in table_records)}"
    )
    print("PER_EVENT_WEIGHTS_SCALED_BY_MULTIPLICITY=FALSE")
    print("SOURCE_ROWS_DUPLICATED_BY_MULTIPLICITY=TRUE")
    print("SOURCE_FOLD_REASSIGNED=FALSE")
    print("SUMW_AND_SUMW2_CLOSURE=PASS")
    print("VALIDATION_PAYLOADS_OPENED=0")
    print("TEST_PAYLOADS_OPENED=0")


if __name__ == "__main__":
    main()
