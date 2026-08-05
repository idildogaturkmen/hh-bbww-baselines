#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path
import argparse
import hashlib
import importlib.util
import json
import os
import platform
import shutil
import sys
import time

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq


PRODUCTION_SCHEMA_VERSION = 1


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def clean(value) -> str:
    text = str(value).strip()
    if text.lower() in {"", "nan", "none", "null"}:
        return ""
    return text


def truthy(value) -> bool:
    return clean(value).lower() in {"true", "1", "yes", "y"}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def check_checksum_manifest(directory: Path) -> int:
    sums = directory / "SHA256SUMS"
    require(sums.is_file(), f"missing checksum manifest: {sums}")

    checked = 0
    for raw in sums.read_text(encoding="utf-8").splitlines():
        if not raw.strip():
            continue
        expected, relative = raw.split("  ", 1)
        path = directory / relative
        require(path.is_file(), f"missing checksummed file: {path}")
        require(sha256(path) == expected, f"checksum mismatch: {path}")
        checked += 1

    require(checked > 0, f"empty checksum manifest: {sums}")
    return checked


def read_weight_table(path: Path) -> pd.DataFrame:
    require(path.is_file(), f"missing weight table: {path}")

    if path.suffix.lower() == ".parquet":
        frame = pd.read_parquet(path)
    else:
        frame = pd.read_csv(path, sep="\t")

    required = {"event", "generator_nominal_weight"}
    require(required.issubset(frame.columns), f"bad weight table schema: {path}")

    frame = frame[["event", "generator_nominal_weight"]].copy()
    frame["event"] = pd.to_numeric(frame["event"], errors="raise").astype("int64")
    frame["generator_nominal_weight"] = pd.to_numeric(
        frame["generator_nominal_weight"], errors="raise"
    ).astype("float64")

    require(frame["event"].is_unique, f"duplicate event keys: {path}")
    require(
        np.isfinite(frame["generator_nominal_weight"]).all(),
        f"non-finite weights: {path}",
    )
    return frame


def compare_weights(observed, expected, label: str) -> None:
    observed = np.asarray(observed, dtype=np.float64)
    expected = np.asarray(expected, dtype=np.float64)

    require(observed.shape == expected.shape, f"{label}: shape mismatch")
    require(np.isfinite(observed).all(), f"{label}: non-finite observed weights")
    require(np.isfinite(expected).all(), f"{label}: non-finite expected weights")
    require(
        np.allclose(observed, expected, rtol=2e-6, atol=2e-9),
        (
            f"{label}: maximum absolute mismatch="
            f"{np.max(np.abs(observed - expected)):.17g}"
        ),
    )


def add_or_replace_columns(table: pa.Table, columns: dict[str, pa.Array]) -> pa.Table:
    result = table
    for name, array in columns.items():
        if name in result.column_names:
            result = result.set_column(result.column_names.index(name), name, array)
        else:
            result = result.append_column(name, array)
    return result


def load_extractor(path: Path):
    spec = importlib.util.spec_from_file_location("hh4b_broad_feature_extractor_v1", path)
    require(spec is not None and spec.loader is not None, f"cannot import extractor: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def validate_existing(output_path: Path, receipt_path: Path, row: pd.Series) -> bool:
    if not output_path.is_file() or not receipt_path.is_file():
        return False

    try:
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    except Exception:
        return False

    if receipt.get("status") != "pass":
        return False
    if receipt.get("source_uid") != clean(row["source_uid"]):
        return False
    if int(receipt.get("generated_events", -1)) != int(row["generated_events"]):
        return False
    if receipt.get("output_sha256") != sha256(output_path):
        return False

    metadata = pq.read_metadata(output_path)
    if metadata.num_rows != int(row["generated_events"]):
        return False

    return True


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--row-index", type=int, required=True)
    parser.add_argument("--readiness-manifest", type=Path, required=True)
    parser.add_argument("--extractor", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--work-root", type=Path, required=True)
    parser.add_argument("--expected-head", required=True)
    args = parser.parse_args()

    started = time.time()

    require(args.readiness_manifest.is_file(), "readiness manifest is missing")
    require(args.extractor.is_file(), "extractor is missing")

    readiness = pd.read_csv(args.readiness_manifest, sep="\t")
    require(len(readiness) == 464, f"readiness rows={len(readiness)}, expected 464")
    require(readiness["source_uid"].astype(str).nunique() == 464, "source_uid not unique")

    matches = readiness.loc[
        pd.to_numeric(readiness["production_row_index"], errors="raise").astype("int64")
        == args.row_index
    ]
    require(len(matches) == 1, f"row index {args.row_index} matched {len(matches)} rows")
    row = matches.iloc[0]

    source_uid = clean(row["source_uid"])
    expected_events = int(row["generated_events"])
    source_token = hashlib.sha256(source_uid.encode("utf-8")).hexdigest()[:16]
    stem = f"source_{args.row_index:04d}_{source_token}"

    parquet_dir = args.output_root / "parquet"
    receipt_dir = args.output_root / "receipts"
    parquet_dir.mkdir(parents=True, exist_ok=True)
    receipt_dir.mkdir(parents=True, exist_ok=True)
    args.work_root.mkdir(parents=True, exist_ok=True)

    output_path = parquet_dir / f"{stem}.parquet"
    receipt_path = receipt_dir / f"{stem}.json"

    if validate_existing(output_path, receipt_path, row):
        print(f"WORKER_STATUS=ALREADY_COMPLETE;row_index={args.row_index};source_uid={source_uid}")
        return

    work = args.work_root / f"{stem}_{os.getpid()}"
    shutil.rmtree(work, ignore_errors=True)
    work.mkdir(parents=True, exist_ok=True)

    output_tmp = parquet_dir / f".{stem}.parquet.tmp.{os.getpid()}"
    receipt_tmp = receipt_dir / f".{stem}.json.tmp.{os.getpid()}"

    output_tmp.unlink(missing_ok=True)
    receipt_tmp.unlink(missing_ok=True)

    try:
        extractor = load_extractor(args.extractor)
        extractor.MAX_EVENTS_PER_SOURCE = expected_events

        root_path, resolution_mode, resolution_chain = extractor.stage_source(row, work)
        table, feature_metadata = extractor.build_table(row, root_path)

        require(table.num_rows == expected_events, "feature table row closure failed")

        data = table.select(
            [
                "source_entry",
                "event_uid",
                "raw_event_weight_available",
                "raw_event_weight",
                "broad_event_eligible",
                "assignment_matchable",
                "n_selected_jets",
            ]
        ).to_pydict()

        source_entries = np.asarray(data["source_entry"], dtype=np.int64)
        require(
            np.array_equal(source_entries, np.arange(expected_events, dtype=np.int64)),
            "source_entry is not contiguous",
        )
        require(len(set(data["event_uid"])) == expected_events, "event_uid is not unique")

        raw_available = np.asarray(data["raw_event_weight_available"], dtype=bool)
        raw_weights = np.asarray(
            [np.nan if value is None else float(value) for value in data["raw_event_weight"]],
            dtype=np.float64,
        )

        transport_class = clean(row["generator_weight_transport_class"])
        transport_applicable = truthy(row["generator_weight_transport_applicable"])
        population_kind = clean(row["population_kind"])
        transport_id = clean(row["transport_id"])
        overlap_rows = 0
        fragment_files_checked = 0

        if not transport_applicable:
            require(
                transport_class == "not_applicable_auxiliary_qcd",
                "non-applicable source has unexpected transport class",
            )
            require(
                not truthy(row["physical_evaluation_eligible"]),
                "auxiliary QCD is physical-evaluation eligible",
            )

            nominal_array = pa.array([None] * expected_events, type=pa.float64())
            sign_array = pa.array([None] * expected_events, type=pa.int8())
            join_status = "excluded_by_auxiliary_qcd_contract"
        else:
            require(raw_available.all(), "primary source has unavailable raw ROOT weights")
            require(np.isfinite(raw_weights).all(), "primary source has non-finite raw ROOT weights")

            if transport_class in {"uniform_member_constant", "qcd_uniform_constant"}:
                constant = float(row["constant_generator_weight"])
                compare_weights(
                    raw_weights,
                    np.full(expected_events, constant, dtype=np.float64),
                    f"row {args.row_index} frozen constant",
                )
                join_status = "raw_root_weight_matches_frozen_constant"

            elif transport_class == "signed_exact_event_sidecar":
                sidecar = Path(clean(row["sidecar_or_fragment_path"]))
                require(sidecar.is_file(), f"missing signed sidecar: {sidecar}")
                require(
                    sha256(sidecar) == clean(row["sidecar_or_fragment_sha256"]),
                    f"signed sidecar checksum mismatch: {sidecar}",
                )

                weights = read_weight_table(sidecar)
                require(len(weights) == expected_events, "signed sidecar is not member-wide")
                selected = pd.DataFrame(
                    {"event": np.arange(expected_events, dtype=np.int64)}
                ).merge(weights, on="event", how="left", validate="one_to_one")
                require(
                    selected["generator_nominal_weight"].notna().all(),
                    "signed sidecar join is incomplete",
                )
                compare_weights(
                    raw_weights,
                    selected["generator_nominal_weight"].to_numpy(),
                    f"row {args.row_index} signed sidecar",
                )
                overlap_rows = expected_events
                join_status = "raw_root_weight_matches_member_wide_signed_sidecar"

            elif transport_class == "qcd_variable_frozen_fragment":
                fragment = Path(clean(row["sidecar_or_fragment_path"]))
                require(fragment.is_dir(), f"missing QCD fragment: {fragment}")
                sums = fragment / "SHA256SUMS"
                require(sums.is_file(), f"missing QCD fragment SHA256SUMS: {fragment}")
                require(
                    sha256(sums) == clean(row["sidecar_or_fragment_sha256"]),
                    f"QCD fragment manifest hash mismatch: {fragment}",
                )
                fragment_files_checked = check_checksum_manifest(fragment)
                require((fragment / "COMPLETE").is_file(), f"missing COMPLETE: {fragment}")

                fragment_weights = read_weight_table(
                    fragment / "train_generator_weight_sidecar.tsv"
                )
                overlap = pd.DataFrame(
                    {"event": np.arange(expected_events, dtype=np.int64)}
                ).merge(fragment_weights, on="event", how="inner", validate="one_to_one")
                overlap_rows = len(overlap)

                if overlap_rows:
                    indices = overlap["event"].to_numpy(dtype=np.int64)
                    compare_weights(
                        raw_weights[indices],
                        overlap["generator_nominal_weight"].to_numpy(),
                        f"row {args.row_index} QCD variable overlap",
                    )

                join_status = (
                    "raw_root_weight_plus_frozen_qcd_fragment_with_overlap"
                    if overlap_rows
                    else "raw_root_weight_plus_frozen_qcd_fragment_no_overlap"
                )

            else:
                raise RuntimeError(f"unsupported transport class: {transport_class}")

            nominal_array = pa.array(raw_weights, type=pa.float64())
            sign_array = pa.array(np.sign(raw_weights).astype(np.int8), type=pa.int8())

        enriched = add_or_replace_columns(
            table,
            {
                "production_schema_version": pa.array(
                    [PRODUCTION_SCHEMA_VERSION] * expected_events, type=pa.int16()
                ),
                "production_row_index": pa.array(
                    [args.row_index] * expected_events, type=pa.int32()
                ),
                "generator_weight_transport_applicable": pa.array(
                    [transport_applicable] * expected_events, type=pa.bool_()
                ),
                "population_kind": pa.array(
                    [population_kind] * expected_events, type=pa.string()
                ),
                "transport_id": pa.array([transport_id] * expected_events, type=pa.string()),
                "generator_weight_transport_class": pa.array(
                    [transport_class] * expected_events, type=pa.string()
                ),
                "generator_weight_provenance": pa.array(
                    [clean(row["generator_weight_provenance"])] * expected_events,
                    type=pa.string(),
                ),
                "generator_nominal_weight": nominal_array,
                "generator_weight_sign": sign_array,
                "generator_weight_join_status": pa.array(
                    [join_status] * expected_events, type=pa.string()
                ),
                "physical_weight_application_authorized": pa.array(
                    [False] * expected_events, type=pa.bool_()
                ),
            },
        )

        pq.write_table(
            enriched,
            output_tmp,
            compression="zstd",
            compression_level=9,
            use_dictionary=False,
            write_statistics=True,
            data_page_version="1.0",
            version="2.6",
        )

        readback = pq.read_table(
            output_tmp,
            columns=[
                "source_uid",
                "source_entry",
                "event_uid",
                "generator_weight_transport_applicable",
                "generator_nominal_weight",
                "physical_weight_application_authorized",
            ],
        ).to_pydict()

        require(len(readback["source_uid"]) == expected_events, "readback row closure failed")
        require(set(readback["source_uid"]) == {source_uid}, "readback source_uid drift")
        require(
            readback["source_entry"] == list(range(expected_events)),
            "readback source_entry drift",
        )
        require(len(set(readback["event_uid"])) == expected_events, "readback event_uid drift")
        require(
            not any(bool(value) for value in readback["physical_weight_application_authorized"]),
            "physical-weight authorization changed",
        )

        if transport_applicable:
            require(
                all(value is not None for value in readback["generator_nominal_weight"]),
                "primary generator weights missing after readback",
            )
        else:
            require(
                all(value is None for value in readback["generator_nominal_weight"]),
                "auxiliary QCD received physical generator weights",
            )

        output_hash = sha256(output_tmp)
        broad_rows = int(sum(bool(value) for value in data["broad_event_eligible"]))
        matchable_rows = int(sum(bool(value) for value in data["assignment_matchable"]))
        max_selected_jets = int(max(data["n_selected_jets"])) if expected_events else 0

        receipt = {
            "schema_version": 1,
            "status": "pass",
            "expected_head": args.expected_head,
            "production_row_index": args.row_index,
            "source_uid": source_uid,
            "sample_class": clean(row["sample_class"]),
            "process_or_mode": clean(row["process_or_mode"]),
            "workflow_population": clean(row["workflow_population"]),
            "generated_events": expected_events,
            "rows_materialized": enriched.num_rows,
            "output_path": str(output_path),
            "output_sha256": output_hash,
            "output_bytes": output_tmp.stat().st_size,
            "root_resolution_mode": resolution_mode,
            "root_resolution_chain": resolution_chain,
            "broad_eligible_rows": broad_rows,
            "assignment_matchable_rows": matchable_rows,
            "max_selected_jets": max_selected_jets,
            "generator_weight_transport_applicable": transport_applicable,
            "generator_weight_transport_class": transport_class,
            "generator_weight_join_status": join_status,
            "transport_overlap_rows": overlap_rows,
            "qcd_fragment_files_checked": fragment_files_checked,
            "physical_weight_application_authorized": False,
            "validation_payloads_opened": 0,
            "test_payloads_opened": 0,
            "models_trained": 0,
            "runtime": {
                "python": sys.version,
                "python_executable": sys.executable,
                "platform": platform.platform(),
                "numpy": np.__version__,
                "pandas": pd.__version__,
                "pyarrow": pa.__version__,
            },
            "feature_metadata": feature_metadata,
            "elapsed_seconds": time.time() - started,
        }

        receipt_tmp.write_text(
            json.dumps(receipt, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

        os.replace(output_tmp, output_path)
        os.replace(receipt_tmp, receipt_path)

        require(validate_existing(output_path, receipt_path, row), "final receipt validation failed")

        print(
            "WORKER_STATUS=PASS;"
            f"row_index={args.row_index};"
            f"events={expected_events};"
            f"broad_rows={broad_rows};"
            f"matchable_rows={matchable_rows};"
            f"transport_class={transport_class};"
            f"source_uid={source_uid}"
        )

    finally:
        output_tmp.unlink(missing_ok=True)
        receipt_tmp.unlink(missing_ok=True)
        shutil.rmtree(work, ignore_errors=True)


if __name__ == "__main__":
    main()
