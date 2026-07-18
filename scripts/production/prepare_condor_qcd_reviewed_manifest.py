#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import tarfile
from collections import Counter, defaultdict
from typing import Any


EXPECTED_BRANCH = "delphes-hh4b-production"

EXPECTED_CARD_SHA256 = (
    "1b2041245162de8360defdc502404e069"
    "6496c50773d4aa7f043798651a9517c"
)

EXPECTED_TOTAL = 500_000
MAX_EVENTS_PER_JOB = 10_000
MAX_PROJECTED_TIB = 0.10

PTHAT_BINS = [
    (50, 75),
    (75, 100),
    (100, 200),
    (200, 300),
    (300, 500),
    (500, 700),
    (700, 1000),
    (1000, 0),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Prepare, but never submit, one Condor campaign "
            "from an exact reviewed QCD manifest."
        )
    )

    parser.add_argument("--campaign", required=True)
    parser.add_argument(
        "--expected-total",
        type=int,
        default=EXPECTED_TOTAL,
    )
    parser.add_argument(
        "--max-projected-tib",
        type=float,
        default=MAX_PROJECTED_TIB,
    )
    parser.add_argument(
        "--allow-no-new-test",
        action="store_true",
    )
    parser.add_argument(
        "--reviewed-manifest",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--reviewed-summary",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--authorization",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--previous-manifest",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--previous-shard-validation",
        type=Path,
        required=True,
    )
    parser.add_argument("--proxy", type=Path, required=True)
    parser.add_argument("--payload", type=Path, required=True)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--store", type=Path, required=True)
    parser.add_argument(
        "--eos-base",
        default=(
            f"/store/user/{os.environ.get('USER', '')}/"
            "hh4b_delphes/run2_13tev/frozen_v2"
        ),
    )

    return parser.parse_args()


def run(
    command: list[str],
    *,
    cwd: Path | None = None,
) -> str:
    result = subprocess.run(
        command,
        cwd=cwd,
        check=True,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )

    return result.stdout.strip()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()

    with path.open("rb") as handle:
        for block in iter(
            lambda: handle.read(8 * 1024 * 1024),
            b"",
        ):
            digest.update(block)

    return digest.hexdigest()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))

    if not rows:
        raise SystemExit(f"ERROR: empty CSV: {path}")

    return rows


def load_payload_manifest(
    payload: Path,
) -> dict[str, str]:
    with tarfile.open(payload, "r:gz") as archive:
        member = archive.getmember("payload/manifest.txt")
        extracted = archive.extractfile(member)

        if extracted is None:
            raise SystemExit(
                "ERROR: payload manifest is unreadable"
            )

        lines = (
            extracted.read()
            .decode("utf-8")
            .splitlines()
        )

    result: dict[str, str] = {}

    for line in lines:
        key, separator, value = line.partition("=")

        if not separator or not key or key in result:
            raise SystemExit(
                "ERROR: malformed payload manifest"
            )

        result[key] = value

    return result


def ensure_eos_absent(
    eos_directory: str,
    proxy: Path,
) -> None:
    environment = os.environ.copy()
    environment["X509_USER_PROXY"] = str(proxy)

    result = subprocess.run(
        [
            "xrdfs",
            "root://cmseos.fnal.gov",
            "stat",
            eos_directory,
        ],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        env=environment,
    )

    if result.returncode == 0:
        raise SystemExit(
            "ERROR: EOS campaign namespace already exists: "
            f"{eos_directory}"
        )

    diagnostic = result.stdout + result.stderr

    if (
        "No such file or directory" not in diagnostic
        and "[3011]" not in diagnostic
    ):
        raise SystemExit(
            "ERROR: EOS namespace preflight failed: "
            f"{diagnostic.strip()}"
        )


def load_prior_identities(
    store: Path,
) -> tuple[set[int], set[int]]:
    seeds: set[int] = set()
    shards: set[int] = set()

    submit_root = store / "condor_submit"

    if submit_root.is_dir():
        for path in submit_root.rglob("*manifest.csv"):
            try:
                rows = read_csv(path)
            except (
                OSError,
                csv.Error,
                UnicodeDecodeError,
                SystemExit,
            ):
                continue

            for row in rows:
                if row.get("seed") not in (None, ""):
                    seeds.add(int(row["seed"]))

                if row.get("shard_id") not in (None, ""):
                    shards.add(int(row["shard_id"]))

    receipt_root = store / "condor_return"

    if receipt_root.is_dir():
        for path in receipt_root.rglob("*_receipt.json"):
            try:
                receipt = json.loads(path.read_text())
            except (OSError, json.JSONDecodeError):
                continue

            if type(receipt.get("seed")) is int:
                seeds.add(int(receipt["seed"]))

            if type(receipt.get("shard_id")) is int:
                shards.add(int(receipt["shard_id"]))

    return seeds, shards


def projected_storage(
    validation_path: Path,
    reviewed_rows: list[dict[str, str]],
) -> tuple[float, dict[int, float]]:
    validation_rows = read_csv(validation_path)

    bytes_by_bin: dict[int, int] = defaultdict(int)
    events_by_bin: dict[int, int] = defaultdict(int)

    for row in validation_rows:
        bin_id = int(row["bin_id"])
        bundle_bytes = int(row["bundle_bytes"])

        n_events_raw = (
            row.get("n_event_summary")
            or row.get("n_events")
        )

        if n_events_raw in (None, ""):
            raise SystemExit(
                "ERROR: validation table lacks event counts"
            )

        bytes_by_bin[bin_id] += bundle_bytes
        events_by_bin[bin_id] += int(n_events_raw)

    if set(bytes_by_bin) != set(range(8)):
        raise SystemExit(
            "ERROR: storage reference lacks all eight bins"
        )

    bytes_per_event = {
        bin_id: bytes_by_bin[bin_id]
        / events_by_bin[bin_id]
        for bin_id in range(8)
    }

    projected = sum(
        int(row["n_events"])
        * bytes_per_event[int(row["bin_id"])]
        for row in reviewed_rows
    )

    return projected, bytes_per_event


def main() -> None:
    args = parse_args()

    repo = args.repo.resolve()
    store = args.store.resolve()
    reviewed_manifest = (
        args.reviewed_manifest.resolve()
    )
    reviewed_summary = (
        args.reviewed_summary.resolve()
    )
    authorization_path = (
        args.authorization.resolve()
    )
    previous_manifest = (
        args.previous_manifest.resolve()
    )
    previous_validation = (
        args.previous_shard_validation.resolve()
    )
    proxy = args.proxy.resolve()
    source_payload = args.payload.resolve()

    required_files = [
        reviewed_manifest,
        reviewed_summary,
        authorization_path,
        previous_manifest,
        previous_validation,
        proxy,
        source_payload,
    ]

    for path in required_files:
        if not path.is_file() or path.stat().st_size == 0:
            raise SystemExit(
                f"ERROR: missing required file: {path}"
            )

    if not re.fullmatch(
        r"[A-Za-z0-9][A-Za-z0-9_.-]+",
        args.campaign,
    ):
        raise SystemExit("ERROR: invalid campaign name")

    branch = run(
        ["git", "branch", "--show-current"],
        cwd=repo,
    )

    if branch != EXPECTED_BRANCH:
        raise SystemExit(
            f"ERROR: branch is {branch}, "
            f"expected {EXPECTED_BRANCH}"
        )

    if run(["git", "status", "--porcelain"], cwd=repo):
        raise SystemExit(
            "ERROR: repository worktree or index is dirty"
        )

    git_head = run(
        ["git", "rev-parse", "HEAD"],
        cwd=repo,
    )

    subprocess.run(
        [
            "openssl",
            "x509",
            "-in",
            str(proxy),
            "-noout",
            "-checkend",
            "3600",
        ],
        check=True,
        stdout=subprocess.DEVNULL,
    )

    manifest_sha = sha256_file(reviewed_manifest)
    summary_sha = sha256_file(reviewed_summary)

    authorization = json.loads(
        authorization_path.read_text()
    )

    if (
        authorization.get("status")
        != "authorized_for_single_submission_after_preflight"
    ):
        raise SystemExit(
            "ERROR: campaign is not authorized"
        )

    if authorization.get("campaign") != args.campaign:
        raise SystemExit(
            "ERROR: authorization campaign mismatch"
        )

    if (
        authorization.get("reviewed_manifest_sha256")
        != manifest_sha
    ):
        raise SystemExit(
            "ERROR: authorization manifest hash mismatch"
        )

    if (
        authorization.get("reviewed_summary_sha256")
        != summary_sha
    ):
        raise SystemExit(
            "ERROR: authorization summary hash mismatch"
        )

    if authorization.get("submission_count_authorized") != 1:
        raise SystemExit(
            "ERROR: authorization is not exactly once"
        )

    summary = json.loads(reviewed_summary.read_text())

    if (
        summary.get("status")
        != "proposal_not_authorized_for_submission"
    ):
        raise SystemExit(
            "ERROR: reviewed summary has wrong status"
        )

    if summary.get("campaign") != args.campaign:
        raise SystemExit(
            "ERROR: reviewed summary campaign mismatch"
        )

    if summary.get("total_events") != args.expected_total:
        raise SystemExit(
            "ERROR: reviewed summary event mismatch"
        )

    if summary.get("combined_test_bins") != list(range(8)):
        raise SystemExit(
            "ERROR: combined test set lacks a stratum"
        )

    reviewed_rows = read_csv(reviewed_manifest)
    previous_rows = read_csv(previous_manifest)

    required_columns = {
        "campaign",
        "job_id",
        "bin_id",
        "pthat_min_GeV",
        "pthat_max_GeV",
        "n_events",
        "shard_id",
        "seed",
        "dataset_split",
        "split_assignment_reason",
        "split_salt",
    }

    missing = required_columns - set(reviewed_rows[0])

    if missing:
        raise SystemExit(
            "ERROR: reviewed manifest missing columns: "
            f"{sorted(missing)}"
        )

    if len(reviewed_rows) != int(
        authorization["authorized_jobs"]
    ):
        raise SystemExit(
            "ERROR: reviewed job-count mismatch"
        )

    if int(
        authorization.get(
            "authorized_events",
            -1,
        )
    ) != args.expected_total:
        raise SystemExit(
            "ERROR: authorization event-count mismatch"
        )

    if sum(
        int(row["n_events"])
        for row in reviewed_rows
    ) != args.expected_total:
        raise SystemExit(
            "ERROR: reviewed manifest event total mismatch"
        )

    expected_job_ids = list(range(len(reviewed_rows)))
    observed_job_ids = [
        int(row["job_id"])
        for row in reviewed_rows
    ]

    if observed_job_ids != expected_job_ids:
        raise SystemExit(
            "ERROR: job IDs are not contiguous and ordered"
        )

    seeds: list[int] = []
    shards: list[int] = []
    split_salts: set[str] = set()
    forced_bins: set[int] = set()

    for row in reviewed_rows:
        if row["campaign"] != args.campaign:
            raise SystemExit(
                "ERROR: manifest campaign mismatch"
            )

        bin_id = int(row["bin_id"])
        n_events = int(row["n_events"])
        shard_id = int(row["shard_id"])
        seed = int(row["seed"])

        if bin_id not in range(8):
            raise SystemExit("ERROR: invalid bin")

        expected_low, expected_high = PTHAT_BINS[bin_id]

        if (
            int(row["pthat_min_GeV"]) != expected_low
            or int(row["pthat_max_GeV"]) != expected_high
        ):
            raise SystemExit(
                "ERROR: pTHat boundary mismatch"
            )

        if not 1 <= n_events <= MAX_EVENTS_PER_JOB:
            raise SystemExit(
                "ERROR: invalid events per job"
            )

        if seed != (
            1_200_000
            + bin_id * 10_000
            + shard_id
        ):
            raise SystemExit(
                "ERROR: seed formula mismatch"
            )

        if row["dataset_split"] not in {
            "train",
            "validation",
            "test",
        }:
            raise SystemExit(
                "ERROR: invalid dataset split"
            )

        if not re.fullmatch(
            r"[A-Za-z0-9_.-]+",
            row["split_salt"],
        ):
            raise SystemExit(
                "ERROR: invalid split salt"
            )

        if (
            row["split_assignment_reason"]
            == "predeclared_sealed_test_for_missing_stratum"
        ):
            if row["dataset_split"] != "test":
                raise SystemExit(
                    "ERROR: forced test assignment is not test"
                )

            forced_bins.add(bin_id)

        seeds.append(seed)
        shards.append(shard_id)
        split_salts.add(row["split_salt"])

    if len(seeds) != len(set(seeds)):
        raise SystemExit("ERROR: duplicate new seeds")

    if len(shards) != len(set(shards)):
        raise SystemExit("ERROR: duplicate new shards")

    if len(split_salts) != 1:
        raise SystemExit(
            "ERROR: multiple split salts in manifest"
        )

    if args.allow_no_new_test:
        if forced_bins:
            raise SystemExit(
                "ERROR: train/validation-only checkpoint "
                "contains forced test assignments"
            )
    elif forced_bins != {1, 4, 5, 6, 7}:
        raise SystemExit(
            "ERROR: forced test-bin set changed"
        )

    previous_test_bins = {
        int(row["bin_id"])
        for row in previous_rows
        if row["dataset_split"] == "test"
    }

    new_test_bins = {
        int(row["bin_id"])
        for row in reviewed_rows
        if row["dataset_split"] == "test"
    }

    if args.allow_no_new_test and new_test_bins:
        raise SystemExit(
            "ERROR: train/validation-only checkpoint "
            "contains new test shards"
        )

    if previous_test_bins | new_test_bins != set(range(8)):
        raise SystemExit(
            "ERROR: combined test coverage incomplete"
        )

    prior_seeds, prior_shards = load_prior_identities(
        store
    )

    if set(seeds) & prior_seeds:
        raise SystemExit(
            "ERROR: reviewed seed collides with prior production"
        )

    if set(shards) & prior_shards:
        raise SystemExit(
            "ERROR: reviewed shard collides with prior production"
        )

    projected_bytes, bytes_per_event = projected_storage(
        previous_validation,
        reviewed_rows,
    )

    projected_tib = projected_bytes / 2**40

    if projected_tib >= args.max_projected_tib:
        raise SystemExit(
            "ERROR: projected storage exceeds gate: "
            f"{projected_tib:.9f} TiB"
        )

    payload_sha = sha256_file(source_payload)
    wrapper = (
        repo
        / "scripts/production/"
        "run_qcd_importance_bundle.sh"
    )
    wrapper_sha = sha256_file(wrapper)

    card = (
        repo
        / "cards/delphes/"
        "delphes_card_CMS_lpc_ak4ak8_run2_frozen_v2.tcl"
    )
    card_sha = sha256_file(card)

    if card_sha != EXPECTED_CARD_SHA256:
        raise SystemExit(
            "ERROR: frozen-v2 card hash mismatch"
        )

    payload_manifest = load_payload_manifest(
        source_payload
    )

    if payload_manifest.get("git_head") != git_head:
        raise SystemExit(
            "ERROR: payload Git head differs from checkout"
        )

    if (
        payload_manifest.get("worker_wrapper_sha256")
        != wrapper_sha
    ):
        raise SystemExit(
            "ERROR: payload wrapper hash mismatch"
        )

    submit_dir = (
        store / "condor_submit" / args.campaign
    )
    log_dir = (
        store / "condor_logs" / args.campaign
    )
    return_root = (
        store / "condor_return" / args.campaign
    )
    return_dir = return_root / "receipts"

    for path in (
        submit_dir,
        log_dir,
        return_root,
    ):
        if path.exists():
            raise SystemExit(
                "ERROR: refusing to reuse local campaign path: "
                f"{path}"
            )

    eos_directory = (
        f"{args.eos_base}/bundles/{args.campaign}"
    )

    ensure_eos_absent(eos_directory, proxy)

    submit_dir.mkdir(parents=True)
    log_dir.mkdir(parents=True)
    return_dir.mkdir(parents=True)

    payload_name = (
        f"{args.campaign}_inputs_"
        f"{payload_sha[:16]}.tar.gz"
    )
    campaign_payload = submit_dir / payload_name

    shutil.copy2(source_payload, campaign_payload)

    if sha256_file(campaign_payload) != payload_sha:
        raise SystemExit(
            "ERROR: copied payload checksum mismatch"
        )

    campaign_payload.chmod(0o444)

    submit_file = (
        submit_dir / f"{args.campaign}.sub"
    )
    production_manifest = (
        submit_dir
        / f"{args.campaign}_manifest.csv"
    )
    production_summary = (
        submit_dir
        / f"{args.campaign}_summary.json"
    )

    split_salt = next(iter(split_salts))

    submit_file.write_text(
        "\n".join(
            [
                "universe = vanilla",
                f"executable = {wrapper}",
                (
                    "arguments = $(bin_id) $(shard_id) "
                    "$(n_events) "
                    f"{args.campaign} $(Cluster) "
                    f"{payload_name} {eos_directory} "
                    f"{payload_sha} $(dataset_split) "
                    f"{split_salt}"
                ),
                "",
                (
                    f"output = {log_dir}/"
                    f"{args.campaign}_$(Cluster)_"
                    "job$(job_id)_bin$(bin_id)_"
                    "shard$(shard_id).out"
                ),
                (
                    f"error = {log_dir}/"
                    f"{args.campaign}_$(Cluster)_"
                    "job$(job_id)_bin$(bin_id)_"
                    "shard$(shard_id).err"
                ),
                (
                    f"log = {log_dir}/"
                    f"{args.campaign}_$(Cluster).log"
                ),
                "",
                "should_transfer_files = YES",
                "when_to_transfer_output = ON_EXIT",
                (
                    "transfer_input_files = "
                    f"{campaign_payload}"
                ),
                "transfer_output_files = job_receipt.json",
                (
                    "transfer_output_remaps = "
                    "\"job_receipt.json = "
                    f"{return_dir}/"
                    f"{args.campaign}_$(Cluster)_"
                    "job$(job_id)_bin$(bin_id)_"
                    "shard$(shard_id)_receipt.json\""
                ),
                "",
                "request_cpus = 1",
                "request_memory = 4GB",
                "request_disk = 20GB",
                "",
                "getenv = False",
                'environment = "LC_ALL=C LANG=C"',
                "use_x509userproxy = True",
                f"x509userproxy = {proxy}",
                "",
                f'+JobBatchName = "{args.campaign}"',
                '+DesiredOS = "EL9"',
                (
                    '+QCDDatasetSplit = '
                    '"$(dataset_split)"'
                ),
                (
                    "on_exit_hold = "
                    "(ExitBySignal == True) || "
                    "(ExitCode != 0)"
                ),
                "",
                (
                    "queue "
                    "job_id,bin_id,shard_id,"
                    "n_events,dataset_split from ("
                ),
                *[
                    (
                        f"{row['job_id']} "
                        f"{row['bin_id']} "
                        f"{row['shard_id']} "
                        f"{row['n_events']} "
                        f"{row['dataset_split']}"
                    )
                    for row in reviewed_rows
                ],
                ")",
                "",
            ]
        )
    )

    extra_fields = [
        "payload_sha256",
        "wrapper_sha256",
        "card_sha256",
        "git_head",
        "eos_directory",
        "reviewed_manifest_sha256",
        "reviewed_summary_sha256",
        "authorization_sha256",
    ]

    manifest_fields = (
        list(reviewed_rows[0].keys())
        + extra_fields
    )

    authorization_sha = sha256_file(
        authorization_path
    )

    with production_manifest.open(
        "w",
        newline="",
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=manifest_fields,
            lineterminator="\n",
        )

        writer.writeheader()

        for row in reviewed_rows:
            writer.writerow(
                {
                    **row,
                    "payload_sha256": payload_sha,
                    "wrapper_sha256": wrapper_sha,
                    "card_sha256": card_sha,
                    "git_head": git_head,
                    "eos_directory": eos_directory,
                    "reviewed_manifest_sha256": (
                        manifest_sha
                    ),
                    "reviewed_summary_sha256": (
                        summary_sha
                    ),
                    "authorization_sha256": (
                        authorization_sha
                    ),
                }
            )

    split_jobs = Counter(
        row["dataset_split"]
        for row in reviewed_rows
    )
    split_events: dict[str, int] = defaultdict(int)

    for row in reviewed_rows:
        split_events[row["dataset_split"]] += int(
            row["n_events"]
        )

    prepared_summary: dict[str, Any] = {
        "schema_version": 1,
        "status": "prepared_not_submitted",
        "campaign": args.campaign,
        "total_jobs": len(reviewed_rows),
        "total_events": args.expected_total,
        "events_by_bin": summary["events_by_bin"],
        "jobs_by_bin": summary["jobs_by_bin"],
        "split_job_counts": dict(split_jobs),
        "split_event_counts": dict(split_events),
        "previous_test_bins": sorted(
            previous_test_bins
        ),
        "new_test_bins": sorted(new_test_bins),
        "combined_test_bins": list(range(8)),
        "forced_sealed_test_bins": sorted(
            forced_bins
        ),
        "reviewed_manifest": str(
            reviewed_manifest
        ),
        "reviewed_manifest_sha256": manifest_sha,
        "reviewed_summary_sha256": summary_sha,
        "authorization_sha256": authorization_sha,
        "payload_sha256": payload_sha,
        "wrapper_sha256": wrapper_sha,
        "card_sha256": card_sha,
        "git_head": git_head,
        "projected_compressed_bytes": (
            projected_bytes
        ),
        "projected_compressed_TiB": (
            projected_tib
        ),
        "storage_gate_TiB": args.max_projected_tib,
        "storage_gate_pass": True,
        "bytes_per_event_reference_by_bin": {
            str(key): bytes_per_event[key]
            for key in range(8)
        },
        "seed_reuse_count": 0,
        "shard_reuse_count": 0,
        "eos_directory": eos_directory,
        "eos_namespace_absent_at_preparation": True,
        "submit_file": str(submit_file),
        "production_manifest": str(
            production_manifest
        ),
        "submission_authorized_exactly_once": True,
        "submitted": False,
        "five_million_authorized": False,
    }

    production_summary.write_text(
        json.dumps(
            prepared_summary,
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )

    print("QCD_REVIEWED_MANIFEST_PREPARED")
    print("campaign:", args.campaign)
    print("jobs:", len(reviewed_rows))
    print("events:", args.expected_total)
    print(
        "projected compressed TiB:",
        f"{projected_tib:.9f}",
    )
    print("submit file:", submit_file)
    print(
        "production manifest:",
        production_manifest,
    )
    print(
        "production summary:",
        production_summary,
    )
    print("EOS directory:", eos_directory)
    print("NO_SUBMISSION_PERFORMED")


if __name__ == "__main__":
    main()
