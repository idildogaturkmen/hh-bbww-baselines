#!/usr/bin/env python3

from __future__ import annotations

import hashlib
import json
import re
import subprocess
import sys
from pathlib import Path

import pandas as pd
import uproot


EXPECTED_CARD_SHA256 = (
    "1b2041245162de8360defdc502404e069"
    "6496c50773d4aa7f043798651a9517c"
)

EXPECTED_CAMPAIGN = "frozen_v2_phase1_reprocess_20260717"

EXPECTED_PROCESS_SHARDS = {
    "qcd_bbbb_general": 5,
    "qcd_bbbb_iht400to600": 10,
    "ttbar": 5,
    "zbbbb": 5,
}

EXPECTED_GENERATED_BY_PROCESS = {
    "qcd_bbbb_general": 50_000,
    "qcd_bbbb_iht400to600": 100_000,
    "ttbar": 50_000,
    "zbbbb": 50_000,
}

REQUIRED_COLUMNS = {
    "event_uid",
    "source_process",
    "source_campaign",
    "source_shard",
    "dataset_split",
    "detector_card_sha256",
    "input_hepmc_sha256",
    "mbb1",
    "mbb2",
    "r_hh",
    "j1_pt",
    "j2_pt",
    "j3_pt",
    "j4_pt",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()

    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)

    return digest.hexdigest()


def remote_size(eos_host: str, path: str) -> int:
    result = subprocess.run(
        ["xrdfs", eos_host, "stat", path],
        check=True,
        capture_output=True,
        text=True,
    )

    match = re.search(r"Size:\s+(\d+)", result.stdout)

    if not match:
        raise RuntimeError(f"Could not parse remote size for {path}")

    return int(match.group(1))


def unique_value(frame: pd.DataFrame, column: str) -> object:
    values = frame[column].drop_duplicates().tolist()

    if len(values) != 1:
        raise AssertionError(
            f"{column} is not constant: {values[:10]}"
        )

    return values[0]


def main() -> None:
    if len(sys.argv) != 5:
        raise SystemExit(
            "Usage: audit_phase1.py "
            "MANIFEST AUDIT_DIR EOS_HOST EOS_CAMPAIGN"
        )

    manifest_path = Path(sys.argv[1])
    audit_dir = Path(sys.argv[2])
    eos_host = sys.argv[3]
    eos_campaign = sys.argv[4].rstrip("/")

    metadata_dir = audit_dir / "metadata"
    parquet_dir = audit_dir / "parquet"
    tables_dir = audit_dir / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)

    manifest = pd.read_csv(manifest_path, sep="\t")

    assert len(manifest) == 25, (
        f"Expected 25 manifest rows, found {len(manifest)}"
    )

    required_manifest_columns = {
        "process",
        "shard",
        "split",
        "n_generated_expected",
        "tag",
        "input_hepmc",
    }

    missing_manifest = required_manifest_columns - set(manifest.columns)

    assert not missing_manifest, (
        f"Missing manifest columns: {sorted(missing_manifest)}"
    )

    records: list[dict[str, object]] = []
    global_event_uids: list[pd.Series] = []
    reference_schema: list[str] | None = None

    for row in manifest.itertuples(index=False):
        process = str(row.process)
        shard = int(row.shard)
        split = str(row.split)
        n_expected = int(row.n_generated_expected)
        tag = str(row.tag)

        metadata_path = metadata_dir / f"{tag}_metadata.json"
        parquet_path = parquet_dir / f"{tag}_candidates.parquet"

        assert metadata_path.is_file(), (
            f"Missing metadata: {metadata_path}"
        )
        assert parquet_path.is_file(), (
            f"Missing Parquet: {parquet_path}"
        )

        metadata = json.loads(metadata_path.read_text())

        assert metadata["process"] == process
        assert int(metadata["shard"]) == shard
        assert metadata["dataset_split"] == split
        assert int(metadata["n_generated_expected"]) == n_expected
        assert int(metadata["n_root_events"]) == n_expected

        assert metadata["detector_card_sha256"] == EXPECTED_CARD_SHA256

        for hash_key in (
            "detector_card_sha256",
            "input_hepmc_sha256",
            "root_sha256",
            "candidate_parquet_sha256",
        ):
            value = str(metadata[hash_key])
            assert len(value) == 64, (
                f"Invalid {hash_key} for {tag}: {value}"
            )

        local_parquet_sha256 = sha256_file(parquet_path)

        assert (
            local_parquet_sha256
            == metadata["candidate_parquet_sha256"]
        ), f"Parquet checksum mismatch for {tag}"

        frame = pd.read_parquet(parquet_path)

        assert len(frame) == int(metadata["n_candidate_rows"]), (
            f"Candidate-row mismatch for {tag}"
        )

        metadata_columns = list(metadata["candidate_columns"])

        assert list(frame.columns) == metadata_columns, (
            f"Metadata/Parquet schema mismatch for {tag}"
        )

        if reference_schema is None:
            reference_schema = list(frame.columns)
        else:
            assert list(frame.columns) == reference_schema, (
                f"Cross-shard schema mismatch for {tag}"
            )

        missing_columns = REQUIRED_COLUMNS - set(frame.columns)

        assert not missing_columns, (
            f"Missing required columns for {tag}: "
            f"{sorted(missing_columns)}"
        )

        assert frame["event_uid"].is_unique, (
            f"Duplicate event_uid values within {tag}"
        )

        assert str(unique_value(frame, "source_process")) == process
        assert str(unique_value(frame, "source_campaign")) == EXPECTED_CAMPAIGN
        assert str(unique_value(frame, "source_shard")) == str(shard)
        assert str(unique_value(frame, "dataset_split")) == split

        assert (
            str(unique_value(frame, "detector_card_sha256"))
            == EXPECTED_CARD_SHA256
        )

        assert (
            str(unique_value(frame, "input_hepmc_sha256"))
            == metadata["input_hepmc_sha256"]
        )

        global_event_uids.append(frame["event_uid"].astype(str))

        root_remote_path = (
            f"{eos_campaign}/root/{process}/{tag}_delphes.root"
        )

        parquet_remote_path = (
            f"{eos_campaign}/parquet/{process}/"
            f"{tag}_candidates.parquet"
        )

        root_remote_size = remote_size(eos_host, root_remote_path)
        parquet_remote_size = remote_size(
            eos_host, parquet_remote_path
        )

        assert root_remote_size == int(metadata["root_size_bytes"]), (
            f"ROOT remote-size mismatch for {tag}"
        )

        assert parquet_remote_size == int(
            metadata["candidate_parquet_size_bytes"]
        ), f"Parquet remote-size mismatch for {tag}"

        root_url = f"{eos_host}/{root_remote_path}"

        with uproot.open(root_url) as root_file:
            root_entries = int(root_file["Delphes"].num_entries)

        assert root_entries == n_expected, (
            f"Remote ROOT entry mismatch for {tag}: "
            f"{root_entries} != {n_expected}"
        )

        records.append(
            {
                "process": process,
                "shard": shard,
                "split": split,
                "tag": tag,
                "n_generated": n_expected,
                "n_root_events": root_entries,
                "n_candidate_rows": len(frame),
                "candidate_efficiency": len(frame) / n_expected,
                "candidate_columns": len(frame.columns),
                "generator_cross_section_pb_median": metadata.get(
                    "generator_cross_section_pb_median"
                ),
                "root_size_bytes": root_remote_size,
                "parquet_size_bytes": parquet_remote_size,
                "detector_card_sha256": metadata[
                    "detector_card_sha256"
                ],
                "input_hepmc_sha256": metadata[
                    "input_hepmc_sha256"
                ],
                "root_sha256": metadata["root_sha256"],
                "candidate_parquet_sha256": metadata[
                    "candidate_parquet_sha256"
                ],
            }
        )

        print(
            f"VALID {process:24s} shard={shard:02d} "
            f"split={split:10s} candidates={len(frame):5d}"
        )

    all_event_uids = pd.concat(
        global_event_uids,
        ignore_index=True,
    )

    assert all_event_uids.is_unique, (
        "Duplicate event_uid values exist across Phase 1 shards"
    )

    summary = pd.DataFrame(records).sort_values(
        ["process", "shard"]
    )

    observed_process_shards = (
        summary.groupby("process")
        .size()
        .to_dict()
    )

    assert observed_process_shards == EXPECTED_PROCESS_SHARDS, (
        f"Unexpected shard counts: {observed_process_shards}"
    )

    observed_generated = (
        summary.groupby("process")["n_generated"]
        .sum()
        .to_dict()
    )

    assert observed_generated == EXPECTED_GENERATED_BY_PROCESS, (
        f"Unexpected generated-event counts: {observed_generated}"
    )

    assert int(summary["n_generated"].sum()) == 250_000
    assert int(summary["n_root_events"].sum()) == 250_000
    assert summary["candidate_columns"].nunique() == 1
    assert int(summary["candidate_columns"].iloc[0]) == 80
    assert summary["detector_card_sha256"].nunique() == 1

    split_summary = (
        summary.groupby(["process", "split"], as_index=False)
        .agg(
            shards=("shard", "count"),
            n_generated=("n_generated", "sum"),
            n_candidate_rows=("n_candidate_rows", "sum"),
            root_size_bytes=("root_size_bytes", "sum"),
            parquet_size_bytes=("parquet_size_bytes", "sum"),
        )
    )

    split_summary["candidate_efficiency"] = (
        split_summary["n_candidate_rows"]
        / split_summary["n_generated"]
    )

    process_summary = (
        summary.groupby("process", as_index=False)
        .agg(
            shards=("shard", "count"),
            n_generated=("n_generated", "sum"),
            n_root_events=("n_root_events", "sum"),
            n_candidate_rows=("n_candidate_rows", "sum"),
            root_size_bytes=("root_size_bytes", "sum"),
            parquet_size_bytes=("parquet_size_bytes", "sum"),
            xsec_min_pb=(
                "generator_cross_section_pb_median",
                "min",
            ),
            xsec_max_pb=(
                "generator_cross_section_pb_median",
                "max",
            ),
        )
    )

    process_summary["candidate_efficiency"] = (
        process_summary["n_candidate_rows"]
        / process_summary["n_generated"]
    )

    summary.to_csv(
        tables_dir / "phase1_shard_audit.csv",
        index=False,
    )

    split_summary.to_csv(
        tables_dir / "phase1_split_summary.csv",
        index=False,
    )

    process_summary.to_csv(
        tables_dir / "phase1_process_summary.csv",
        index=False,
    )

    report = {
        "status": "valid",
        "campaign": EXPECTED_CAMPAIGN,
        "n_shards": int(len(summary)),
        "n_generated": int(summary["n_generated"].sum()),
        "n_root_events": int(summary["n_root_events"].sum()),
        "n_candidate_rows": int(
            summary["n_candidate_rows"].sum()
        ),
        "candidate_columns": int(
            summary["candidate_columns"].iloc[0]
        ),
        "unique_event_uids": int(all_event_uids.nunique()),
        "detector_card_sha256": EXPECTED_CARD_SHA256,
        "process_shard_counts": observed_process_shards,
        "process_generated_counts": observed_generated,
    }

    (audit_dir / "phase1_audit_report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n"
    )

    print()
    print("=== PROCESS SUMMARY ===")
    print(process_summary.to_string(index=False))

    print()
    print("=== SPLIT SUMMARY ===")
    print(split_summary.to_string(index=False))

    print()
    print("PHASE1_FINAL_AUDIT_VALID")
    print("Shards:", report["n_shards"])
    print("Generated events:", report["n_generated"])
    print("ROOT events:", report["n_root_events"])
    print("Candidate rows:", report["n_candidate_rows"])
    print("Candidate columns:", report["candidate_columns"])
    print("Unique event_uids:", report["unique_event_uids"])
    print("Wrote:", audit_dir / "phase1_audit_report.json")


if __name__ == "__main__":
    main()
