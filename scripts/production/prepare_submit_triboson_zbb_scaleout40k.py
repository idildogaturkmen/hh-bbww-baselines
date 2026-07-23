#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import re
import shlex
import shutil
import subprocess
from pathlib import Path
from typing import Any


CAMPAIGN = (
    "final_background_triboson_zbb_scaleout40k_20260722_v1"
)

PILOT_AUDIT_TAG = (
    "final_background_triboson_pilot10k_final_audit_20260722_v1"
)

PILOT_AUDIT_FILENAME = (
    "triboson_pilot10k_final_audit.json"
)

BUILD_TAG = (
    "final_background_triboson_worker_v2_build_20260722_v1"
)

BUILD_FILENAME = (
    "triboson_worker_v2_build.json"
)

REFERENCE_CAMPAIGN = (
    "final_background_triboson_zbb_pilot10k_20260722_v1"
)

SPLIT_SALT = (
    "final-background-triboson-zbb-scaleout-v1"
)


def make_shards(
    *,
    family: str,
    shard_ids: list[int],
    seed_start: int,
    accepted_events: int,
    maximum_input_events: int,
    pilot_events: int,
    final_family_target_events: int,
) -> list[dict[str, Any]]:
    rows = []

    for offset, shard_id in enumerate(
        shard_ids,
    ):
        rows.append({
            "family": family,
            "target_tag": (
                f"{family}_run2_frozen_v2_"
                f"train_shard{shard_id}_"
                f"{accepted_events}"
            ),
            "shard_id": shard_id,
            "seed": seed_start + offset,
            "accepted_events": accepted_events,
            "maximum_input_events":
                maximum_input_events,
            "validated_pilot_events":
                pilot_events,
            "final_family_target_events":
                final_family_target_events,
        })

    return rows


SHARDS: list[dict[str, Any]] = []

SHARDS.extend(
    make_shards(
        family="wwz_zbb",
        shard_ids=[
            9901,
            9902,
            9903,
            9904,
        ],
        seed_start=997101,
        accepted_events=5200,
        maximum_input_events=46800,
        pilot_events=5200,
        final_family_target_events=26000,
    )
)

SHARDS.extend(
    make_shards(
        family="wzz_zbb",
        shard_ids=[
            9911,
            9912,
            9913,
            9914,
        ],
        seed_start=997201,
        accepted_events=3300,
        maximum_input_events=19800,
        pilot_events=3300,
        final_family_target_events=16500,
    )
)

SHARDS.extend(
    make_shards(
        family="zzz_zbb",
        shard_ids=[
            9921,
            9922,
            9923,
            9924,
        ],
        seed_start=997301,
        accepted_events=1500,
        maximum_input_events=7500,
        pilot_events=1500,
        final_family_target_events=7500,
    )
)


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
            lambda: handle.read(
                8 * 1024 * 1024
            ),
            b"",
        ):
            digest.update(block)

    return digest.hexdigest()


def submit_value(
    text: str,
    key: str,
) -> str:
    match = re.search(
        rf"(?m)^\s*{re.escape(key)}\s*=\s*(.*?)\s*$",
        text,
    )

    require(
        match is not None,
        f"reference submit file is missing {key}",
    )

    return match.group(1)


def run_shell(
    arguments: list[str | Path],
    *,
    environment: dict[str, str],
) -> str:
    command = shlex.join([
        str(argument)
        for argument in arguments
    ])

    result = subprocess.run(
        [
            "/bin/bash",
            "-c",
            command,
        ],
        check=True,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        env=environment,
    )

    return result.stdout


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--repo",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--store",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--submit",
        action="store_true",
    )

    arguments = parser.parse_args()

    repo = arguments.repo.resolve()
    store = arguments.store.resolve()

    pilot_audit_path = (
        repo
        / "outputs/agent_runs"
        / PILOT_AUDIT_TAG
        / PILOT_AUDIT_FILENAME
    )

    build_path = (
        repo
        / "outputs/agent_runs"
        / BUILD_TAG
        / BUILD_FILENAME
    )

    for path in (
        pilot_audit_path,
        build_path,
    ):
        require(
            path.is_file(),
            f"missing required input {path}",
        )

    pilot_audit = json.loads(
        pilot_audit_path.read_text()
    )

    build = json.loads(
        build_path.read_text()
    )

    required_pilot_audit = {
        "status": "pass",
        "triboson_pilot10k_final_audit_valid":
            True,
        "triboson_scaleout40k_submission_authorized":
            True,
        "triboson_pilot_events_validated":
            10000,
        "triboson_scaleout_events_remaining":
            40000,
        "validated_background_events_now":
            4960000,
        "background_events_remaining":
            40000,
        "final_background_target":
            5000000,
        "physics_yield_authorized":
            False,
    }

    for key, expected in (
        required_pilot_audit.items()
    ):
        observed = pilot_audit.get(key)

        require(
            observed == expected,
            (
                f"pilot audit mismatch for {key}: "
                f"{observed!r} != {expected!r}"
            ),
        )

    required_build = {
        "status": "pass",
        "worker_v2_shell_syntax_valid":
            True,
        "converter_compile_valid":
            True,
        "three_payloads_built":
            True,
        "physics_yield_authorized":
            False,
    }

    for key, expected in (
        required_build.items()
    ):
        observed = build.get(key)

        require(
            observed == expected,
            (
                f"worker build mismatch for {key}: "
                f"{observed!r} != {expected!r}"
            ),
        )

    require(
        len(SHARDS) == 12,
        "expected exactly twelve scale-out shards",
    )

    require(
        len({
            row["target_tag"]
            for row in SHARDS
        }) == 12,
        "scale-out target tags are not unique",
    )

    require(
        len({
            int(row["shard_id"])
            for row in SHARDS
        }) == 12,
        "scale-out shard IDs are not unique",
    )

    require(
        len({
            int(row["seed"])
            for row in SHARDS
        }) == 12,
        "scale-out MG5 seeds are not unique",
    )

    require(
        sum(
            int(row["accepted_events"])
            for row in SHARDS
        ) == 40000,
        "scale-out accepted target is not 40k",
    )

    expected_family_scaleout = {
        "wwz_zbb": 20800,
        "wzz_zbb": 13200,
        "zzz_zbb": 6000,
    }

    for family, expected_total in (
        expected_family_scaleout.items()
    ):
        observed_total = sum(
            int(row["accepted_events"])
            for row in SHARDS
            if row["family"] == family
        )

        require(
            observed_total == expected_total,
            (
                f"{family}: scale-out total "
                f"{observed_total} != "
                f"{expected_total}"
            ),
        )

    remaining_rows = {
        row["family"]: row
        for row in pilot_audit[
            "remaining_allocation"
        ]
    }

    require(
        set(remaining_rows)
        == set(expected_family_scaleout),
        "remaining-allocation family set mismatch",
    )

    for family, expected_total in (
        expected_family_scaleout.items()
    ):
        observed = int(
            remaining_rows[family][
                "remaining_scaleout_events"
            ]
        )

        require(
            observed == expected_total,
            (
                f"{family}: audited remaining "
                f"target {observed} != "
                f"{expected_total}"
            ),
        )

    build_rows = {
        row["family"]: row
        for row in build["payloads"]
    }

    require(
        set(build_rows)
        == set(expected_family_scaleout),
        "payload family set mismatch",
    )

    worker = Path(
        build["worker"]
    )

    require(
        worker.is_file(),
        f"missing worker {worker}",
    )

    reference_submit = (
        store
        / "condor_submit"
        / REFERENCE_CAMPAIGN
        / f"{REFERENCE_CAMPAIGN}.sub"
    )

    require(
        reference_submit.is_file(),
        (
            "missing reference submit file "
            f"{reference_submit}"
        ),
    )

    reference_text = (
        reference_submit.read_text(
            errors="replace"
        )
    )

    request_cpus = submit_value(
        reference_text,
        "request_cpus",
    )

    request_memory = submit_value(
        reference_text,
        "request_memory",
    )

    request_disk = submit_value(
        reference_text,
        "request_disk",
    )

    source_proxy_raw = os.environ.get(
        "X509_USER_PROXY",
        "",
    )

    require(
        source_proxy_raw,
        "X509_USER_PROXY is not set",
    )

    source_proxy = Path(
        source_proxy_raw
    ).resolve()

    require(
        source_proxy.is_file(),
        f"missing proxy {source_proxy}",
    )

    subprocess.run(
        [
            "openssl",
            "x509",
            "-in",
            str(source_proxy),
            "-noout",
            "-checkend",
            "21600",
        ],
        check=True,
        stdout=subprocess.DEVNULL,
    )

    submit_dir = (
        store
        / "condor_submit"
        / CAMPAIGN
    )

    log_dir = (
        store
        / "condor_logs"
        / CAMPAIGN
    )

    return_root = (
        store
        / "condor_return"
        / CAMPAIGN
    )

    receipt_dir = (
        return_root
        / "receipts"
    )

    outdir = (
        repo
        / "outputs/agent_runs"
        / "final_background_triboson_zbb_scaleout40k_submission_20260722_v1"
    )

    for path in (
        submit_dir,
        log_dir,
        return_root,
        outdir,
    ):
        require(
            not path.exists(),
            f"refusing to overwrite {path}",
        )

    submit_dir.mkdir(parents=True)
    log_dir.mkdir(parents=True)
    receipt_dir.mkdir(parents=True)
    outdir.mkdir(parents=True)

    credential_dir = (
        store
        / "condor_credentials"
    )

    credential_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    campaign_proxy = (
        credential_dir
        / (
            f"x509up_u{os.getuid()}_"
            f"{CAMPAIGN}"
        )
    )

    shutil.copy2(
        source_proxy,
        campaign_proxy,
    )

    campaign_proxy.chmod(0o600)

    environment = os.environ.copy()
    environment["X509_USER_PROXY"] = str(
        campaign_proxy
    )

    eos_dir = (
        f"/store/user/"
        f"{environment['USER']}/"
        "hh4b_delphes/run2_13tev/"
        "frozen_v2/bundles/"
        f"{CAMPAIGN}"
    )

    print(
        "===== PREPARE EOS DIRECTORY ====="
    )

    print(
        run_shell(
            [
                "xrdfs",
                "root://cmseos.fnal.gov",
                "mkdir",
                "-p",
                eos_dir,
            ],
            environment=environment,
        )
    )

    queue_rows: list[
        dict[str, Any]
    ] = []

    for shard in SHARDS:
        family = str(
            shard["family"]
        )

        payload_row = build_rows[
            family
        ]

        payload = Path(
            payload_row["payload"]
        )

        require(
            payload.is_file(),
            (
                f"{family}: missing payload "
                f"{payload}"
            ),
        )

        require(
            sha256_file(payload)
            == payload_row[
                "payload_sha256"
            ],
            (
                f"{family}: payload SHA "
                "mismatch"
            ),
        )

        require(
            payload_row[
                "exact_target_output"
            ]
            is True,
            (
                f"{family}: payload is not "
                "exact-target"
            ),
        )

        require(
            payload_row[
                "physics_yield_authorized"
            ]
            is False,
            (
                f"{family}: unexpected "
                "yield authorization"
            ),
        )

        queue_rows.append({
            **shard,
            "payload_path":
                str(payload),
            "payload_name":
                payload.name,
            "payload_sha256":
                payload_row[
                    "payload_sha256"
                ],
        })

    queue_path = (
        submit_dir
        / f"{CAMPAIGN}_queue.tsv"
    )

    queue_columns = [
        "family",
        "target_tag",
        "shard_id",
        "accepted_events",
        "maximum_input_events",
        "seed",
        "validated_pilot_events",
        "final_family_target_events",
        "payload_path",
        "payload_name",
        "payload_sha256",
    ]

    with queue_path.open(
        "w",
        newline="",
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=queue_columns,
            delimiter="\t",
            lineterminator="\n",
        )

        # HTCondor queue-from treats every line as a job.
        # This file must contain data rows only, with no TSV header.
        writer.writerows(
            queue_rows
        )

    submit_file = (
        submit_dir
        / f"{CAMPAIGN}.sub"
    )

    submit_text = f"""universe = vanilla

executable = {worker}

arguments = {CAMPAIGN} $(family) $(target_tag) $(shard_id) $(accepted_events) $(maximum_input_events) $(seed) train {SPLIT_SALT} preserve_template canonical_train $(payload_name) {eos_dir} $(payload_sha256) $(Cluster)

should_transfer_files = YES
when_to_transfer_output = ON_EXIT

transfer_input_files = $(payload_path)

transfer_output_files = job_receipt.json
transfer_output_remaps = "job_receipt.json = {receipt_dir}/$(family)_$(Cluster)_$(Process)_receipt.json"

output = {log_dir}/$(family)_$(Cluster)_$(Process).out
error = {log_dir}/$(family)_$(Cluster)_$(Process).err
log = {log_dir}/{CAMPAIGN}_$(Cluster).condor.log

request_cpus = {request_cpus}
request_memory = {request_memory}
request_disk = {request_disk}

notification = Never

use_x509userproxy = True
x509userproxy = {campaign_proxy}

+JobBatchName = "{CAMPAIGN}"
+CampaignName = "{CAMPAIGN}"
+DatasetFamily = "$(family)"
+DatasetRole = "canonical_train"
+DatasetSplit = "train"
+CountTowardBackgroundTarget = True
+AcceptedEventTarget = $(accepted_events)
+MaximumInclusiveInputEvents = $(maximum_input_events)
+ValidatedPilotFamilyEvents = $(validated_pilot_events)
+FinalTribosonFamilyTargetEvents = $(final_family_target_events)
+ExactTargetOutput = True
+InclusivePythiaZDecays = True
+AllZForcedToBB = False
+FilterRequirement = "at least one direct Z to bb"
+AllocationSemantics = "MC statistics proxy only"
+PhysicsYieldAuthorized = False

queue family, target_tag, shard_id, accepted_events, maximum_input_events, seed, validated_pilot_events, final_family_target_events, payload_path, payload_name, payload_sha256 from {queue_path}
"""

    submit_file.write_text(
        submit_text
    )

    dryrun_ads = (
        submit_dir
        / f"{CAMPAIGN}_dryrun.classads"
    )

    print(
        "===== CONDOR DRY RUN ====="
    )

    dryrun_output = run_shell(
        [
            "condor_submit",
            "-dry-run",
            dryrun_ads,
            submit_file,
        ],
        environment=environment,
    )

    print(dryrun_output)

    dryrun_job_matches = re.findall(
        r"(?m)^([0-9]+) job\(s\) dry-run to cluster",
        dryrun_output,
    )

    require(
        dryrun_job_matches == ["12"],
        (
            "Condor dry run did not contain exactly "
            f"12 jobs: {dryrun_job_matches}"
        ),
    )

    require(
        dryrun_ads.is_file()
        and dryrun_ads.stat().st_size > 0,
        (
            "Condor dry run produced "
            "no classads"
        ),
    )

    plan_path = (
        outdir
        / "triboson_scaleout40k_plan.json"
    )

    plan_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "status":
                    "prepared_and_dryrun_valid",
                "campaign": CAMPAIGN,
                "jobs": 12,
                "accepted_events_total":
                    40000,
                "validated_pilot_events":
                    10000,
                "final_triboson_events":
                    50000,
                "validated_background_events_before_scaleout":
                    4960000,
                "potential_validated_background_events_after_scaleout":
                    5000000,
                "dataset_split": "train",
                "dataset_role":
                    "canonical_train",
                "split_salt":
                    SPLIT_SALT,
                "count_toward_background_target":
                    True,
                "worker": str(worker),
                "worker_sha256":
                    build[
                        "worker_sha256"
                    ],
                "pilot_final_audit":
                    str(
                        pilot_audit_path
                    ),
                "submit_file":
                    str(submit_file),
                "queue_file":
                    str(queue_path),
                "shards":
                    queue_rows,
                "scaleout40k_submission_authorized":
                    True,
                "physics_yield_authorized":
                    False,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )

    if not arguments.submit:
        print()
        print(
            "TRIBOSON_SCALEOUT40K_DRYRUN_VALID"
        )
        print(
            "TRIBOSON_SCALEOUT40K_SUBMISSION_AUTHORIZED"
        )
        print(
            "RERUN_WITH_SUBMIT_FLAG_TO_SUBMIT"
        )
        return

    print(
        "===== SUBMIT TRIBOSON SCALEOUT 40K ====="
    )

    submission_output = run_shell(
        [
            "condor_submit",
            submit_file,
        ],
        environment=environment,
    )

    print(submission_output)

    cluster_matches = re.findall(
        r"cluster\s+([0-9]+)",
        submission_output,
        re.IGNORECASE,
    )

    require(
        len(cluster_matches) == 1,
        (
            "could not identify submitted "
            f"cluster: {cluster_matches}"
        ),
    )

    schedd_matches = re.findall(
        (
            r"Attempting to submit jobs to"
            r"\s+([A-Za-z0-9.-]+)"
        ),
        submission_output,
    )

    actual_schedd = (
        schedd_matches[-1]
        if schedd_matches
        else "system_selected_unknown"
    )

    cluster_id = int(
        cluster_matches[0]
    )

    cluster_path = (
        outdir
        / "triboson_scaleout40k_cluster.tsv"
    )

    with cluster_path.open(
        "w",
        newline="",
    ) as handle:
        columns = [
            "campaign",
            "schedd",
            "cluster_id",
            "jobs",
            "accepted_events_total",
            "count_toward_background_target",
            "submit_file",
        ]

        writer = csv.DictWriter(
            handle,
            fieldnames=columns,
            delimiter="\t",
            lineterminator="\n",
        )

        writer.writeheader()

        writer.writerow({
            "campaign": CAMPAIGN,
            "schedd": actual_schedd,
            "cluster_id": cluster_id,
            "jobs": 12,
            "accepted_events_total":
                40000,
            "count_toward_background_target":
                True,
            "submit_file":
                str(submit_file),
        })

    print()
    print(
        "WWZ_ZBB_SCALEOUT20800_SUBMITTED"
    )
    print(
        "WZZ_ZBB_SCALEOUT13200_SUBMITTED"
    )
    print(
        "ZZZ_ZBB_SCALEOUT6000_SUBMITTED"
    )
    print(
        "TRIBOSON_SCALEOUT_JOBS_SUBMITTED=12"
    )
    print(
        "TRIBOSON_SCALEOUT_ACCEPTED_EVENTS_SUBMITTED=40000"
    )
    print(
        "BACKGROUND_TARGET_EVENTS_IN_FLIGHT=5000000"
    )
    print(
        "NO_BACKGROUND_PHYSICS_YIELD_AUTHORIZATION_YET"
    )
    print(
        f"actual_schedd={actual_schedd}"
    )
    print(
        f"cluster_id={cluster_id}"
    )
    print(
        f"cluster_record={cluster_path}"
    )


if __name__ == "__main__":
    main()
