#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import shlex
import subprocess


FAMILY_CONFIG = {
    "tth_hbb": {
        "shard_start": 2001,
        "seed_start": 952001,
    },
    "ttz_zbb": {
        "shard_start": 2101,
        "seed_start": 952101,
    },
    "tttt": {
        "shard_start": 2201,
        "seed_start": 952201,
    },
    "vbf_hbb": {
        "shard_start": 2301,
        "seed_start": 952301,
    },
}

EVENTS_PER_JOB = 10000
JOBS_PER_FAMILY = 10
TRAIN_JOBS_PER_FAMILY = 8
TOTAL_EVENTS = 400000


def command_output(command):
    result = subprocess.run(
        command,
        check=True,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )

    return result.stdout.strip()


def sha256_file(path):
    digest = hashlib.sha256()

    with path.open("rb") as handle:
        for block in iter(
            lambda: handle.read(
                8 * 1024 * 1024
            ),
            b"",
        ):
            digest.update(block)

    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--repo",
        required=True,
    )

    parser.add_argument(
        "--store",
        required=True,
    )

    parser.add_argument(
        "--source-pilot-campaign",
        required=True,
    )

    parser.add_argument(
        "--campaign",
        required=True,
    )

    parser.add_argument(
        "--proxy",
        required=True,
    )

    args = parser.parse_args()

    repo = Path(
        args.repo
    ).resolve()

    store = Path(
        args.store
    ).resolve()

    proxy = Path(
        args.proxy
    ).resolve()

    source_campaign = (
        args.source_pilot_campaign
    )

    campaign = args.campaign

    if not proxy.is_file():
        raise SystemExit(
            f"ERROR: missing proxy: {proxy}"
        )

    subprocess.run(
        [
            "openssl",
            "x509",
            "-in",
            str(proxy),
            "-noout",
            "-checkend",
            "43200",
        ],
        check=True,
    )

    dirty = command_output(
        [
            "git",
            "-C",
            str(repo),
            "status",
            "--porcelain",
        ]
    )

    if dirty:
        raise SystemExit(
            "ERROR: repository must be clean"
        )

    local_head = command_output(
        [
            "git",
            "-C",
            str(repo),
            "rev-parse",
            "HEAD",
        ]
    )

    remote_head = command_output(
        [
            "git",
            "-C",
            str(repo),
            "rev-parse",
            "origin/delphes-hh4b-production",
        ]
    )

    if local_head != remote_head:
        raise SystemExit(
            "ERROR: local and remote heads differ"
        )

    worker = (
        repo
        / "scripts"
        / "production"
        / "run_mg5_frozen_v2_bundle.sh"
    )

    if not os.access(
        worker,
        os.X_OK,
    ):
        raise SystemExit(
            f"ERROR: worker is not executable: {worker}"
        )

    source_submit_dir = (
        store
        / "condor_submit"
        / source_campaign
    )

    source_manifest = (
        source_submit_dir
        / f"{source_campaign}_manifest.csv"
    )

    source_receipt_dir = (
        store
        / "condor_return"
        / source_campaign
        / "receipts"
    )

    if not source_manifest.is_file():
        raise SystemExit(
            f"ERROR: missing source manifest: {source_manifest}"
        )

    if not source_receipt_dir.is_dir():
        raise SystemExit(
            f"ERROR: missing source receipts: {source_receipt_dir}"
        )

    with source_manifest.open(
        newline=""
    ) as handle:
        source_rows = list(
            csv.DictReader(handle)
        )

    source_by_family = {
        row["family"]: row
        for row in source_rows
    }

    if set(source_by_family) != set(
        FAMILY_CONFIG
    ):
        raise SystemExit(
            "ERROR: source pilot family set is incorrect"
        )

    validated = {}

    for family in FAMILY_CONFIG:
        source = source_by_family[
            family
        ]

        if int(source["events"]) != 10000:
            raise SystemExit(
                f"ERROR: source pilot is not 10K for {family}"
            )

        payload = Path(
            source["payload"]
        ).resolve()

        if (
            not payload.is_file()
            or payload.stat().st_size == 0
        ):
            raise SystemExit(
                f"ERROR: missing payload for {family}: {payload}"
            )

        observed_payload_sha = (
            sha256_file(payload)
        )

        if (
            observed_payload_sha
            != source["payload_sha256"]
        ):
            raise SystemExit(
                f"ERROR: payload checksum mismatch for {family}"
            )

        receipt_paths = sorted(
            source_receipt_dir.glob(
                f"{family}_*_receipt.json"
            )
        )

        if len(receipt_paths) != 1:
            raise SystemExit(
                f"ERROR: expected one pilot receipt for "
                f"{family}, found {len(receipt_paths)}"
            )

        receipt_path = receipt_paths[0]

        receipt = json.loads(
            receipt_path.read_text()
        )

        required = {
            "campaign": source_campaign,
            "family": family,
            "target_tag": source[
                "target_tag"
            ],
            "n_events": 10000,
            "stage": "complete_copied_and_verified",
            "exit_status": 0,
            "lhe_events": 10000,
            "hepmc_events": 10000,
            "root_events": 10000,
            "payload_sha256": source[
                "payload_sha256"
            ],
            "expected_payload_sha256": source[
                "payload_sha256"
            ],
        }

        for key, expected in required.items():
            observed = receipt.get(
                key
            )

            if observed != expected:
                raise SystemExit(
                    f"ERROR: pilot gate failed for "
                    f"{family}, {key}: "
                    f"{observed!r} != {expected!r}"
                )

        if not receipt.get(
            "remote_bundle",
            "",
        ).startswith("/store/user/"):
            raise SystemExit(
                f"ERROR: source pilot has no EOS bundle for {family}"
            )

        validated[family] = {
            "source": source,
            "payload": payload,
            "receipt": receipt,
            "receipt_path": receipt_path,
        }

    submit_dir = (
        store
        / "condor_submit"
        / campaign
    )

    log_dir = (
        store
        / "condor_logs"
        / campaign
    )

    return_root = (
        store
        / "condor_return"
        / campaign
    )

    receipt_dir = (
        return_root
        / "receipts"
    )

    for path in (
        submit_dir,
        log_dir,
        return_root,
    ):
        if path.exists():
            raise SystemExit(
                f"ERROR: refusing to reuse path: {path}"
            )

    submit_dir.mkdir(
        parents=True
    )

    log_dir.mkdir(
        parents=True
    )

    receipt_dir.mkdir(
        parents=True
    )

    eos_dir = (
        f"/store/user/{os.environ['USER']}/"
        "hh4b_delphes/run2_13tev/frozen_v2/"
        f"bundles/{campaign}"
    )

    manifest_rows = []
    submit_files = []

    for family, config in FAMILY_CONFIG.items():
        source = validated[
            family
        ]["source"]

        payload = validated[
            family
        ]["payload"]

        pilot_receipt = validated[
            family
        ]["receipt"]

        pilot_receipt_path = validated[
            family
        ]["receipt_path"]

        for offset in range(
            JOBS_PER_FAMILY
        ):
            shard_id = (
                config["shard_start"]
                + offset
            )

            seed = (
                config["seed_start"]
                + offset
            )

            dataset_split = (
                "train"
                if offset
                < TRAIN_JOBS_PER_FAMILY
                else "validation"
            )

            target_tag = (
                f"{family}_run2_frozen_v2_"
                f"waveb_scale400k_"
                f"shard{shard_id:04d}_"
                f"{EVENTS_PER_JOB}evt"
            )

            stem = (
                f"{campaign}_"
                f"{family}_"
                f"shard{shard_id:04d}"
            )

            submit_file = (
                submit_dir
                / f"{stem}.sub"
            )

            dry_run = (
                submit_dir
                / f"{stem}_dryrun.txt"
            )

            receipt_output = (
                receipt_dir
                / (
                    f"{family}_"
                    f"shard{shard_id:04d}_"
                    "$(Cluster)_receipt.json"
                )
            )

            arguments = " ".join([
                campaign,
                family,
                target_tag,
                str(shard_id),
                str(EVENTS_PER_JOB),
                str(seed),
                dataset_split,
                source["split_salt"],
                source["run_card_profile"],
                source["dataset_role"],
                payload.name,
                eos_dir,
                source["payload_sha256"],
                "$(Cluster)",
            ])

            submit_text = f"""universe = vanilla

executable = {worker}

arguments = {arguments}

output = {log_dir}/{family}_shard{shard_id:04d}_$(Cluster).out
error = {log_dir}/{family}_shard{shard_id:04d}_$(Cluster).err
log = {log_dir}/{family}_shard{shard_id:04d}_$(Cluster).log

should_transfer_files = YES
when_to_transfer_output = ON_EXIT

transfer_input_files = {payload}

transfer_output_files = job_receipt.json
transfer_output_remaps = "job_receipt.json = {receipt_output}"

request_cpus = 1
request_memory = 6GB
request_disk = 35GB

getenv = False
environment = "LC_ALL=C LANG=C"

use_x509userproxy = True
x509userproxy = {proxy}

+JobBatchName = "{stem}"
+DesiredOS = "EL9"

notification = Never

on_exit_hold = (ExitBySignal == True) || (ExitCode != 0)

queue 1
"""

            submit_file.write_text(
                submit_text
            )

            subprocess.run(
                [
                    "/bin/bash",
                    "-lc",
                    'condor_submit -dry-run "$1" "$2"',
                    "condor-submit-dry-run",
                    str(dry_run),
                    str(submit_file),
                ],
                check=True,
            )

            submit_files.append(
                submit_file
            )

            manifest_rows.append({
                "campaign": campaign,
                "family": family,
                "target_tag": target_tag,
                "shard_id": shard_id,
                "events": EVENTS_PER_JOB,
                "seed": seed,
                "dataset_split": dataset_split,
                "split_salt": source[
                    "split_salt"
                ],
                "run_card_profile": source[
                    "run_card_profile"
                ],
                "dataset_role": source[
                    "dataset_role"
                ],
                "accounting_class": "counted_generation",
                "physics_yield_authorized": False,
                "training_authorized": True,
                "count_toward_5m": True,
                "overlap_group": source[
                    "overlap_group"
                ],
                "normalization_rule": source[
                    "normalization_rule"
                ],
                "generator_cross_section_pb": "",
                "physics_cross_section_pb": "",
                "branching_fraction": 1.0,
                "filter_efficiency": 1.0,
                "generated_denominator": "",
                "payload": str(payload),
                "payload_sha256": source[
                    "payload_sha256"
                ],
                "payload_git_head": pilot_receipt[
                    "payload_git_head"
                ],
                "source_pilot_campaign": source_campaign,
                "source_pilot_target_tag": source[
                    "target_tag"
                ],
                "source_pilot_receipt": str(
                    pilot_receipt_path
                ),
                "source_pilot_remote_bundle": pilot_receipt[
                    "remote_bundle"
                ],
                "eos_directory": eos_dir,
                "submit_file": str(
                    submit_file
                ),
                "dry_run": str(
                    dry_run
                ),
            })

    if len(manifest_rows) != 40:
        raise SystemExit(
            "ERROR: scale-out did not produce 40 rows"
        )

    if sum(
        row["events"]
        for row in manifest_rows
    ) != TOTAL_EVENTS:
        raise SystemExit(
            "ERROR: scale-out total is not 400000"
        )

    manifest_path = (
        submit_dir
        / f"{campaign}_manifest.csv"
    )

    with manifest_path.open(
        "w",
        newline="",
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=list(
                manifest_rows[0]
            ),
            lineterminator="\n",
        )

        writer.writeheader()
        writer.writerows(
            manifest_rows
        )

    commands_path = (
        submit_dir
        / "submit_commands.sh"
    )

    commands = [
        "#!/usr/bin/env bash",
        "",
        "set -Eeuo pipefail",
        "",
    ]

    for submit_file in submit_files:
        commands.append(
            "condor_submit "
            + shlex.quote(
                str(submit_file)
            )
        )

    commands_path.write_text(
        "\n".join(commands)
        + "\n"
    )

    commands_path.chmod(
        0o755
    )

    summary = {
        "schema_version": 1,
        "status": (
            "waveb_priority_scale400k_"
            "prepared_and_dryrun_valid"
        ),
        "campaign": campaign,
        "orchestration_git_head": (
            local_head
        ),
        "source_pilot_campaign": (
            source_campaign
        ),
        "jobs": len(
            manifest_rows
        ),
        "events": sum(
            row["events"]
            for row in manifest_rows
        ),
        "events_per_family": {
            family: sum(
                row["events"]
                for row in manifest_rows
                if row["family"]
                == family
            )
            for family in FAMILY_CONFIG
        },
        "train_jobs": sum(
            row["dataset_split"]
            == "train"
            for row in manifest_rows
        ),
        "validation_jobs": sum(
            row["dataset_split"]
            == "validation"
            for row in manifest_rows
        ),
        "count_toward_5m": True,
        "training_authorized": True,
        "physics_yield_authorized": False,
        "final_test_authorized": False,
        "full_remaining_production_authorized": False,
        "eos_directory": eos_dir,
        "manifest": str(
            manifest_path
        ),
        "submit_commands": str(
            commands_path
        ),
    }

    summary_path = (
        submit_dir
        / "preparation_summary.json"
    )

    summary_path.write_text(
        json.dumps(
            summary,
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )

    print(
        "WAVEB_PRIORITY_SCALE400K_PREPARED_AND_DRYRUN_VALID"
    )

    print(
        f"campaign={campaign}"
    )

    print(
        f"jobs={len(manifest_rows)}"
    )

    print(
        "events="
        f"{sum(row['events'] for row in manifest_rows)}"
    )

    print("train_jobs=32")
    print("validation_jobs=8")
    print("count_toward_5m=True")
    print("physics_yield_authorized=False")
    print(f"eos_directory={eos_dir}")
    print(f"manifest={manifest_path}")
    print(f"submit_commands={commands_path}")


if __name__ == "__main__":
    main()
