#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import os
from pathlib import Path, PurePosixPath
import sys
import tempfile
from collections import Counter, defaultdict
from typing import Any


LCG_SETUP = Path(
    "/cvmfs/sft.cern.ch/lcg/views/"
    "LCG_106/x86_64-el9-gcc13-opt/setup.sh"
)


def ensure_environment() -> None:
    if importlib.util.find_spec("uproot") is not None:
        return

    if os.environ.get("HH4B_QCD_REGISTRY_AUDIT_LCG_ACTIVE") == "1":
        raise SystemExit(
            "ERROR: uproot unavailable after loading frozen LCG view"
        )

    if not LCG_SETUP.is_file():
        raise SystemExit(
            f"ERROR: missing frozen LCG setup: {LCG_SETUP}"
        )

    environment = dict(os.environ)
    environment[
        "HH4B_QCD_REGISTRY_AUDIT_LCG_ACTIVE"
    ] = "1"

    command = [
        "bash",
        "-c",
        'set +u; source "$1"; set -u; shift; exec "$@"',
        "qcd-registry-audit",
        str(LCG_SETUP),
        "python3",
        str(Path(__file__).resolve()),
        *sys.argv[1:],
    ]

    os.execvpe("bash", command, environment)


ensure_environment()


def load_core():
    path = Path(__file__).with_name(
        "audit_qcd_adaptive_campaign.py"
    )

    if not path.is_file():
        raise SystemExit(
            f"ERROR: missing audit core: {path}"
        )

    spec = importlib.util.spec_from_file_location(
        "qcd_adaptive_audit_core",
        path,
    )

    if spec is None or spec.loader is None:
        raise SystemExit(
            f"ERROR: cannot import audit core: {path}"
        )

    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)

    return module


CORE = load_core()


def read_registry(path: Path) -> list[dict[str, Any]]:
    with path.open(newline="") as handle:
        raw_rows = list(csv.DictReader(handle))

    if not raw_rows:
        raise ValueError("registry is empty")

    integer_fields = [
        "cluster_id",
        "bin_id",
        "pthat_min_GeV",
        "pthat_max_GeV",
        "shard_id",
        "seed",
        "n_events",
    ]

    rows: list[dict[str, Any]] = []

    for raw in raw_rows:
        row: dict[str, Any] = dict(raw)

        for field in integer_fields:
            row[field] = int(row[field])

        rows.append(row)

    identity_keys = {
        (
            row["campaign"],
            row["bin_id"],
            row["shard_id"],
        )
        for row in rows
    }

    if len(identity_keys) != len(rows):
        raise ValueError(
            "registry has duplicate campaign/bin/shard identities"
        )

    if len({row["seed"] for row in rows}) != len(rows):
        raise ValueError("registry seeds are not unique")

    if len(
        {row["remote_bundle"] for row in rows}
    ) != len(rows):
        raise ValueError("registry bundle paths are not unique")

    return rows


def write_csv(
    path: Path,
    rows: list[dict[str, Any]],
) -> None:
    import pandas as pd

    pd.DataFrame(rows).to_csv(path, index=False)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Stream, validate, and statistically combine the "
            "canonical multi-campaign adaptive-QCD registry."
        )
    )

    parser.add_argument(
        "--registry",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--proxy",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--eos-host",
        default="root://cmseos.fnal.gov",
    )

    parser.add_argument(
        "--expected-events",
        type=int,
        default=1_010_000,
    )

    parser.add_argument(
        "--expected-shards",
        type=int,
        default=111,
    )

    parser.add_argument(
        "--bootstrap",
        type=int,
        default=2000,
    )

    parser.add_argument(
        "--bootstrap-seed",
        type=int,
        default=1_010_000,
    )

    parser.add_argument(
        "--max-shards",
        type=int,
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if not args.proxy.is_file():
        raise SystemExit(
            f"ERROR: proxy is missing: {args.proxy}"
        )

    environment = dict(os.environ)
    environment["X509_USER_PROXY"] = str(
        args.proxy.resolve()
    )

    CORE.run_checked(
        [
            "openssl",
            "x509",
            "-in",
            str(args.proxy),
            "-noout",
            "-checkend",
            "3600",
        ],
        environment,
    )

    registry = read_registry(args.registry)

    if len(registry) != args.expected_shards:
        raise SystemExit(
            f"ERROR: expected {args.expected_shards} registry rows, "
            f"found {len(registry)}"
        )

    total_events = sum(
        int(row["n_events"])
        for row in registry
    )

    if total_events != args.expected_events:
        raise SystemExit(
            f"ERROR: expected {args.expected_events} events, "
            f"found {total_events}"
        )

    available = registry

    if args.max_shards is not None:
        if args.max_shards <= 0:
            raise SystemExit(
                "ERROR: --max-shards must be positive"
            )

        available = available[: args.max_shards]

    args.output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    shards = []
    failures: list[dict[str, Any]] = []

    for index, row in enumerate(
        available,
        start=1,
    ):
        receipt_path = Path(row["receipt_path"])

        if not receipt_path.is_file():
            raise SystemExit(
                f"ERROR: receipt missing: {receipt_path}"
            )

        receipt = CORE.read_json(receipt_path)

        if receipt.get("remote_bundle") != row[
            "remote_bundle"
        ]:
            raise SystemExit(
                "ERROR: registry and receipt bundle paths disagree "
                f"for {receipt_path}"
            )

        if receipt.get("payload_sha256") != row[
            "payload_sha256"
        ]:
            raise SystemExit(
                "ERROR: registry and receipt payload hashes disagree "
                f"for {receipt_path}"
            )

        if receipt.get(
            "delphes_card_sha256"
        ) != row["delphes_card_sha256"]:
            raise SystemExit(
                "ERROR: registry and receipt card hashes disagree "
                f"for {receipt_path}"
            )

        remote_bundle = PurePosixPath(
            row["remote_bundle"]
        )

        manifest = {
            "job_id": index - 1,
            "campaign": row["campaign"],
            "bin_id": row["bin_id"],
            "pthat_min_GeV": row[
                "pthat_min_GeV"
            ],
            "pthat_max_GeV": row[
                "pthat_max_GeV"
            ],
            "n_events": row["n_events"],
            "shard_id": row["shard_id"],
            "seed": row["seed"],
            "dataset_split": row[
                "dataset_split"
            ],
            "split_salt": row["split_salt"],
            "eos_directory": str(
                remote_bundle.parent
            ),
            "payload_sha256": row[
                "payload_sha256"
            ],
            "card_sha256": row[
                "delphes_card_sha256"
            ],
            "wrapper_sha256": receipt[
                "wrapper_sha256"
            ],
        }

        print(
            f"[{index}/{len(available)}] "
            f"{row['wave']} "
            f"bin={row['bin_id']} "
            f"shard={row['shard_id']} "
            f"split={row['dataset_split']}",
            flush=True,
        )

        try:
            with tempfile.TemporaryDirectory(
                prefix=(
                    f"qcd_registry_"
                    f"{index:03d}_"
                ),
                dir="/tmp",
            ) as temporary:
                stats = CORE.validate_shard(
                    manifest=manifest,
                    receipt_path=receipt_path,
                    receipt=receipt,
                    eos_host=args.eos_host,
                    environment=environment,
                    temp_root=Path(temporary),
                    cluster_id=row["cluster_id"],
                )

            stats.row["wave"] = row["wave"]
            stats.row["source"] = row["source"]
            stats.row["campaign"] = row[
                "campaign"
            ]

            stats.manifest["wave"] = row["wave"]
            stats.manifest["source"] = row[
                "source"
            ]

            shards.append(stats)

        except Exception as error:
            failures.append(
                {
                    "registry_index": index - 1,
                    "wave": row["wave"],
                    "campaign": row["campaign"],
                    "bin_id": row["bin_id"],
                    "shard_id": row["shard_id"],
                    "receipt_path": str(
                        receipt_path
                    ),
                    "error": str(error),
                }
            )

            break

    shard_rows = [
        item.row
        for item in shards
    ]

    write_csv(
        args.output_dir
        / "qcd_1p01m_shard_validation.csv",
        shard_rows,
    )

    checksum_fields = [
        "wave",
        "source",
        "campaign",
        "job_id",
        "bin_id",
        "shard_id",
        "remote_bundle",
        "receipt_adler32",
        "eos_adler32",
        "downloaded_adler32",
        "bundle_bytes",
        "bundle_sha256",
        "receipt_sha256",
        "all_checks_pass",
    ]

    checksum_rows = [
        {
            field: row[field]
            for field in checksum_fields
        }
        for row in shard_rows
    ]

    write_csv(
        args.output_dir
        / "qcd_1p01m_bundle_checksums.csv",
        checksum_rows,
    )

    complete = (
        len(shards) == len(registry)
        and not failures
        and args.max_shards is None
    )

    bin_rows = (
        CORE.aggregate_by_bin(shards)
        if shards
        else []
    )

    inclusive_rows = (
        CORE.inclusive_rows(bin_rows)
        if bin_rows
        else []
    )

    split_rows = (
        CORE.split_rows(shards)
        if shards
        else []
    )

    bootstrap_rows = (
        CORE.bootstrap_summary(
            shards,
            args.bootstrap,
            args.bootstrap_seed,
        )
        if complete
        else []
    )

    wave_bin_rows: list[dict[str, Any]] = []
    wave_inclusive_rows: list[dict[str, Any]] = []

    waves = sorted(
        {
            str(item.manifest["wave"])
            for item in shards
        }
    )

    for wave in waves:
        subset = [
            item
            for item in shards
            if item.manifest["wave"] == wave
        ]

        current_bin_rows = (
            CORE.aggregate_by_bin(subset)
        )

        current_inclusive_rows = (
            CORE.inclusive_rows(
                current_bin_rows
            )
        )

        wave_bin_rows.extend(
            {
                "wave": wave,
                **row,
            }
            for row in current_bin_rows
        )

        wave_inclusive_rows.extend(
            {
                "wave": wave,
                **row,
            }
            for row in current_inclusive_rows
        )

    write_csv(
        args.output_dir
        / "qcd_1p01m_bin_summary.csv",
        bin_rows,
    )

    write_csv(
        args.output_dir
        / "qcd_1p01m_inclusive_summary.csv",
        inclusive_rows,
    )

    write_csv(
        args.output_dir
        / "qcd_1p01m_split_summary.csv",
        split_rows,
    )

    write_csv(
        args.output_dir
        / "qcd_1p01m_bootstrap_summary.csv",
        bootstrap_rows,
    )

    write_csv(
        args.output_dir
        / "qcd_1p01m_wave_bin_summary.csv",
        wave_bin_rows,
    )

    write_csv(
        args.output_dir
        / "qcd_1p01m_wave_inclusive_summary.csv",
        wave_inclusive_rows,
    )

    events_by_bin: dict[str, int] = defaultdict(
        int
    )

    events_by_split: dict[str, int] = defaultdict(
        int
    )

    jobs_by_wave: Counter[str] = Counter()

    for row in registry:
        events_by_bin[
            str(row["bin_id"])
        ] += int(row["n_events"])

        events_by_split[
            row["dataset_split"]
        ] += int(row["n_events"])

        jobs_by_wave[row["wave"]] += 1

    report = {
        "schema_version": 1,
        "status": (
            "pass"
            if complete
            else "partial_pass"
            if shards and not failures
            else "fail"
        ),
        "registry": str(
            args.registry.resolve()
        ),
        "registry_sha256": CORE.sha256_file(
            args.registry
        ),
        "processed_shards": len(shards),
        "expected_shards": len(registry),
        "processed_events": sum(
            int(item.row["n_event_summary"])
            for item in shards
        ),
        "expected_events": total_events,
        "jobs_by_wave": dict(jobs_by_wave),
        "events_by_bin": dict(
            sorted(events_by_bin.items())
        ),
        "events_by_split": dict(
            sorted(events_by_split.items())
        ),
        "failures": failures,
        "streaming": {
            "one_bundle_at_a_time": True,
            "persistent_event_artifacts": False,
            "temporary_root": "/tmp",
        },
        "physical_weighting": {
            "normalization": (
                "all accepted shards are combined "
                "within each disjoint pTHat stratum"
            ),
            "never_normalized_per_shard": True,
            "waves_combined_before_stratum_weighting": True,
        },
        "splitting": {
            "whole_shard_assignment": True,
            "test_is_evaluation_only": True,
            "test_used_for_training": False,
        },
        "bootstrap": {
            "replicates": (
                args.bootstrap
                if complete
                else 0
            ),
            "seed": args.bootstrap_seed,
            "unit": (
                "whole shard within pTHat stratum"
            ),
        },
        "target_total_events": 5_000_000,
        "additional_events_to_target": (
            5_000_000 - total_events
        ),
        "additional_generation_authorized": False,
        "five_million_authorized": False,
        "artifacts": {
            "shard_validation": str(
                args.output_dir
                / "qcd_1p01m_shard_validation.csv"
            ),
            "bin_summary": str(
                args.output_dir
                / "qcd_1p01m_bin_summary.csv"
            ),
            "inclusive_summary": str(
                args.output_dir
                / "qcd_1p01m_inclusive_summary.csv"
            ),
            "split_summary": str(
                args.output_dir
                / "qcd_1p01m_split_summary.csv"
            ),
            "bootstrap_summary": str(
                args.output_dir
                / "qcd_1p01m_bootstrap_summary.csv"
            ),
            "wave_bin_summary": str(
                args.output_dir
                / "qcd_1p01m_wave_bin_summary.csv"
            ),
            "wave_inclusive_summary": str(
                args.output_dir
                / "qcd_1p01m_wave_inclusive_summary.csv"
            ),
        },
    }

    report_path = (
        args.output_dir
        / "qcd_1p01m_cross_layer_validation.json"
    )

    report_path.write_text(
        json.dumps(
            report,
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )

    print(
        f"status={report['status']} "
        f"shards={len(shards)}/{len(registry)} "
        f"events={report['processed_events']}/"
        f"{total_events}",
        flush=True,
    )

    print(
        f"report={report_path}",
        flush=True,
    )

    if failures:
        raise SystemExit(1)

    if complete:
        print(
            "QCD_ADAPTIVE_1P01M_"
            "CROSS_LAYER_AUDIT_VALID"
        )


if __name__ == "__main__":
    main()
