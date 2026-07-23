#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import shlex
import shutil
import subprocess
from pathlib import Path
from typing import Any


CAMPAIGN = "final_background_triboson_zbb_canary100_20260722_v1"
EVENTS_ACCEPTED = 100
MAXIMUM_INPUT_EVENTS = 1000

BUILD_TAG = "final_background_triboson_worker_v2_build_20260722_v1"
BUILD_FILENAME = "triboson_worker_v2_build.json"

SOURCE_SUBMIT_CAMPAIGN = "ggh_hbb_scaleout90k_20260722_v1"

CANARIES = [
    {
        "family": "wwz_zbb",
        "target_tag":
            "wwz_zbb_run2_frozen_v2_canary_shard9501_100",
        "shard_id": 9501,
        "seed": 995001,
    },
    {
        "family": "wzz_zbb",
        "target_tag":
            "wzz_zbb_run2_frozen_v2_canary_shard9502_100",
        "shard_id": 9502,
        "seed": 995002,
    },
    {
        "family": "zzz_zbb",
        "target_tag":
            "zzz_zbb_run2_frozen_v2_canary_shard9503_100",
        "shard_id": 9503,
        "seed": 995003,
    },
]


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def submit_value(text: str, key: str) -> str:
    match = re.search(
        rf"(?m)^\s*{re.escape(key)}\s*=\s*(.*?)\s*$",
        text,
    )

    require(
        match is not None,
        f"missing {key} in reference submit file",
    )

    return match.group(1)


def run_shell(
    arguments: list[str | Path],
    *,
    environment: dict[str, str],
) -> str:
    command = shlex.join(
        [str(argument) for argument in arguments]
    )

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

    build_path = (
        repo
        / "outputs/agent_runs"
        / BUILD_TAG
        / BUILD_FILENAME
    )

    require(
        build_path.is_file(),
        f"missing worker-v2 build summary {build_path}",
    )

    build = json.loads(
        build_path.read_text()
    )

    required_build = {
        "status": "pass",
        "worker_v2_shell_syntax_valid": True,
        "converter_compile_valid": True,
        "three_payloads_built": True,
        "canary_preparation_authorized": True,
        "condor_submission_authorized": False,
        "physics_yield_authorized": False,
    }

    for key, expected in required_build.items():
        observed = build.get(key)

        require(
            observed == expected,
            (
                f"build summary mismatch for {key}: "
                f"{observed!r} != {expected!r}"
            ),
        )

    worker = Path(
        build["worker"]
    )

    require(
        worker.is_file(),
        f"missing triboson worker {worker}",
    )

    payload_rows = {
        row["family"]: row
        for row in build["payloads"]
    }

    expected_families = {
        row["family"]
        for row in CANARIES
    }

    require(
        set(payload_rows) == expected_families,
        "payload and canary family sets differ",
    )

    require(
        len({
            row["shard_id"]
            for row in CANARIES
        }) == 3,
        "canary shard IDs are not unique",
    )

    require(
        len({
            row["seed"]
            for row in CANARIES
        }) == 3,
        "canary seeds are not unique",
    )

    source_submit = (
        store
        / "condor_submit"
        / SOURCE_SUBMIT_CAMPAIGN
        / f"{SOURCE_SUBMIT_CAMPAIGN}.sub"
    )

    require(
        source_submit.is_file(),
        f"missing reference submit file {source_submit}",
    )

    source_text = source_submit.read_text(
        errors="replace"
    )

    request_cpus = submit_value(
        source_text,
        "request_cpus",
    )

    request_memory = submit_value(
        source_text,
        "request_memory",
    )

    request_disk = submit_value(
        source_text,
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
        / "final_background_triboson_zbb_canary100_submission_20260722_v1"
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
        / f"x509up_u{os.getuid()}_{CAMPAIGN}"
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
        f"/store/user/{environment['USER']}/"
        "hh4b_delphes/run2_13tev/frozen_v2/"
        f"bundles/{CAMPAIGN}"
    )

    print("===== PREPARE EOS DIRECTORY =====")

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

    queue_rows: list[dict[str, Any]] = []

    for canary in CANARIES:
        family = str(
            canary["family"]
        )

        payload_row = payload_rows[family]

        payload = Path(
            payload_row["payload"]
        )

        require(
            payload.is_file(),
            f"{family}: missing payload {payload}",
        )

        require(
            payload_row["exact_target_output"] is True,
            f"{family}: payload is not exact-target",
        )

        require(
            payload_row["physics_yield_authorized"] is False,
            f"{family}: unexpected physics-yield authorization",
        )

        queue_rows.append({
            "family": family,
            "target_tag": canary["target_tag"],
            "shard_id": canary["shard_id"],
            "seed": canary["seed"],
            "payload_path": str(payload),
            "payload_name": payload.name,
            "payload_sha256": payload_row["payload_sha256"],
        })

    queue_path = (
        submit_dir
        / f"{CAMPAIGN}_queue.tsv"
    )

    with queue_path.open(
        "w",
        newline="",
    ) as handle:
        columns = [
            "family",
            "target_tag",
            "shard_id",
            "seed",
            "payload_path",
            "payload_name",
            "payload_sha256",
        ]

        writer = csv.DictWriter(
            handle,
            fieldnames=columns,
            delimiter="\t",
            lineterminator="\n",
        )

        for row in queue_rows:
            writer.writerow(row)

    submit_file = (
        submit_dir
        / f"{CAMPAIGN}.sub"
    )

    submit_text = f"""universe = vanilla

executable = {worker}

arguments = {CAMPAIGN} $(family) $(target_tag) $(shard_id) {EVENTS_ACCEPTED} {MAXIMUM_INPUT_EVENTS} $(seed) validation final-background-triboson-zbb-canary-v1 preserve_template qa_canary $(payload_name) {eos_dir} $(payload_sha256) $(Cluster)

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
+DatasetRole = "qa_canary"
+DatasetSplit = "validation"
+CountTowardBackgroundTarget = False
+AcceptedEventTarget = {EVENTS_ACCEPTED}
+MaximumInclusiveInputEvents = {MAXIMUM_INPUT_EVENTS}
+ExactTargetOutput = True
+InclusivePythiaZDecays = True
+AllZForcedToBB = False
+FilterRequirement = "at least one direct Z to bb"
+PhysicsYieldAuthorized = False

queue family, target_tag, shard_id, seed, payload_path, payload_name, payload_sha256 from {queue_path}
"""

    submit_file.write_text(
        submit_text
    )

    dryrun_ads = (
        submit_dir
        / f"{CAMPAIGN}_dryrun.classads"
    )

    print("===== CONDOR DRY RUN =====")

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

    require(
        dryrun_ads.is_file()
        and dryrun_ads.stat().st_size > 0,
        "Condor dry run did not produce classads",
    )

    authorization = {
        "schema_version": 1,
        "status": "prepared_and_dryrun_valid",
        "campaign": CAMPAIGN,
        "jobs": 3,
        "accepted_events_per_job": EVENTS_ACCEPTED,
        "maximum_input_events_per_job":
            MAXIMUM_INPUT_EVENTS,
        "accepted_canary_events_total": 300,
        "count_toward_background_target": False,
        "physics_yield_authorized": False,
        "worker": str(worker),
        "worker_sha256": build["worker_sha256"],
        "source_worker_build": str(build_path),
        "submit_file": str(submit_file),
        "queue_file": str(queue_path),
        "canary_submission_authorized": True,
    }

    authorization_path = (
        outdir
        / "triboson_canary_submission_authorization.json"
    )

    authorization_path.write_text(
        json.dumps(
            authorization,
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )

    if not arguments.submit:
        print()
        print("TRIBOSON_CANARY100_DRYRUN_VALID")
        print("TRIBOSON_CANARY_SUBMISSION_AUTHORIZED")
        print("RERUN_WITH_SUBMIT_FLAG_TO_SUBMIT")
        return

    print("===== SUBMIT THREE TRIBOSON CANARIES =====")

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
            "could not identify submitted cluster: "
            f"{cluster_matches}"
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
        / "triboson_canary_cluster.tsv"
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
            "accepted_events_per_job",
            "maximum_input_events_per_job",
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
            "jobs": 3,
            "accepted_events_per_job":
                EVENTS_ACCEPTED,
            "maximum_input_events_per_job":
                MAXIMUM_INPUT_EVENTS,
            "count_toward_background_target":
                False,
            "submit_file": str(submit_file),
        })

    print()
    print("WWZ_ZBB_CANARY100_SUBMITTED")
    print("WZZ_ZBB_CANARY100_SUBMITTED")
    print("ZZZ_ZBB_CANARY100_SUBMITTED")
    print("TRIBOSON_CANARY_JOBS_SUBMITTED=3")
    print("TRIBOSON_CANARY_ACCEPTED_EVENTS_COUNTED=0")
    print("NO_TRIBOSON_COUNTED_PRODUCTION_AUTHORIZED_YET")
    print(f"actual_schedd={actual_schedd}")
    print(f"cluster_id={cluster_id}")
    print(f"cluster_record={cluster_path}")


if __name__ == "__main__":
    main()
