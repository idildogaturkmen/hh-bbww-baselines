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


CAMPAIGN = "final_background_triboson_zbb_pilot10k_20260722_v1"

CANARY_AUDIT_TAG = (
    "final_background_triboson_canary100_final_audit_20260722_v1"
)

CANARY_AUDIT_FILENAME = (
    "triboson_canary100_final_audit.json"
)

BUILD_TAG = (
    "final_background_triboson_worker_v2_build_20260722_v1"
)

BUILD_FILENAME = "triboson_worker_v2_build.json"

SPLIT_SALT = "final-background-triboson-zbb-pilot-v1"

PILOTS: list[dict[str, Any]] = [
    {
        "family": "wwz_zbb",
        "target_tag":
            "wwz_zbb_run2_frozen_v2_pilot_validation_shard9601_5200",
        "shard_id": 9601,
        "seed": 996101,
        "accepted_events": 5200,
        "maximum_input_events": 46800,
        "final_family_target_events": 26000,
    },
    {
        "family": "wzz_zbb",
        "target_tag":
            "wzz_zbb_run2_frozen_v2_pilot_validation_shard9602_3300",
        "shard_id": 9602,
        "seed": 996102,
        "accepted_events": 3300,
        "maximum_input_events": 19800,
        "final_family_target_events": 16500,
    },
    {
        "family": "zzz_zbb",
        "target_tag":
            "zzz_zbb_run2_frozen_v2_pilot_validation_shard9603_1500",
        "shard_id": 9603,
        "seed": 996103,
        "accepted_events": 1500,
        "maximum_input_events": 7500,
        "final_family_target_events": 7500,
    },
]


def require(
    condition: bool,
    message: str,
) -> None:
    if not condition:
        raise RuntimeError(message)


def sha256_file(path: Path) -> str:
    import hashlib

    digest = hashlib.sha256()

    with path.open("rb") as handle:
        for block in iter(
            lambda: handle.read(8 * 1024 * 1024),
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

    audit_path = (
        repo
        / "outputs/agent_runs"
        / CANARY_AUDIT_TAG
        / CANARY_AUDIT_FILENAME
    )

    build_path = (
        repo
        / "outputs/agent_runs"
        / BUILD_TAG
        / BUILD_FILENAME
    )

    require(
        audit_path.is_file(),
        f"missing canary audit {audit_path}",
    )

    require(
        build_path.is_file(),
        f"missing worker build {build_path}",
    )

    audit = json.loads(
        audit_path.read_text()
    )

    build = json.loads(
        build_path.read_text()
    )

    required_audit = {
        "status": "pass",
        "triboson_canaries_final_audit_valid":
            True,
        "triboson_canary_events_counted":
            0,
        "triboson_pilot10k_submission_authorized":
            True,
        "triboson_scaleout40k_submission_authorized":
            False,
        "validated_background_events_now":
            4950000,
        "remaining_background_events":
            50000,
        "physics_yield_authorized":
            False,
    }

    for key, expected in required_audit.items():
        observed = audit.get(key)

        require(
            observed == expected,
            (
                f"canary audit mismatch for {key}: "
                f"{observed!r} != {expected!r}"
            ),
        )

    required_build = {
        "status": "pass",
        "worker_v2_shell_syntax_valid": True,
        "converter_compile_valid": True,
        "three_payloads_built": True,
        "physics_yield_authorized": False,
    }

    for key, expected in required_build.items():
        observed = build.get(key)

        require(
            observed == expected,
            (
                f"worker build mismatch for {key}: "
                f"{observed!r} != {expected!r}"
            ),
        )

    require(
        sum(
            int(row["accepted_events"])
            for row in PILOTS
        ) == 10000,
        "pilot accepted-event target is not 10k",
    )

    require(
        sum(
            int(row["final_family_target_events"])
            for row in PILOTS
        ) == 50000,
        "final triboson allocation is not 50k",
    )

    require(
        len({
            int(row["shard_id"])
            for row in PILOTS
        }) == 3,
        "pilot shard IDs are not unique",
    )

    require(
        len({
            int(row["seed"])
            for row in PILOTS
        }) == 3,
        "pilot seeds are not unique",
    )

    canary_rows = {
        row["family"]: row
        for row in audit["canaries"]
    }

    build_rows = {
        row["family"]: row
        for row in build["payloads"]
    }

    expected_families = {
        row["family"]
        for row in PILOTS
    }

    require(
        set(canary_rows) == expected_families,
        "canary-audit family set mismatch",
    )

    require(
        set(build_rows) == expected_families,
        "payload family set mismatch",
    )

    proxy_values: dict[str, float] = {}

    for family in expected_families:
        canary = canary_rows[family]

        processed = int(
            canary["processed_input_events"]
        )

        accepted = int(
            canary["accepted_events"]
        )

        xsec = float(
            canary["generator_xsec_pb"]
        )

        require(
            processed > 0,
            f"{family}: invalid processed count",
        )

        require(
            accepted == 100,
            f"{family}: wrong canary accepted count",
        )

        acceptance_proxy = (
            accepted / processed
        )

        proxy_values[family] = (
            xsec * acceptance_proxy
        )

    proxy_total = sum(
        proxy_values.values()
    )

    require(
        proxy_total > 0.0,
        "invalid triboson allocation proxy",
    )

    pilot_by_family = {
        row["family"]: row
        for row in PILOTS
    }

    allocation_rows: list[dict[str, Any]] = []

    for family in sorted(expected_families):
        pilot = pilot_by_family[family]
        canary = canary_rows[family]

        proxy_fraction = (
            proxy_values[family]
            / proxy_total
        )

        chosen_fraction = (
            int(
                pilot[
                    "final_family_target_events"
                ]
            )
            / 50000
        )

        require(
            abs(
                proxy_fraction
                - chosen_fraction
            ) < 0.03,
            (
                f"{family}: chosen allocation differs "
                "from canary proxy by at least 3%"
            ),
        )

        allocation_rows.append({
            "family": family,
            "canary_generator_xsec_pb":
                float(
                    canary["generator_xsec_pb"]
                ),
            "canary_processed_input_events":
                int(
                    canary[
                        "processed_input_events"
                    ]
                ),
            "canary_accepted_events":
                int(
                    canary["accepted_events"]
                ),
            "acceptance_proxy":
                int(
                    canary["accepted_events"]
                )
                / int(
                    canary[
                        "processed_input_events"
                    ]
                ),
            "filtered_xsec_proxy_pb":
                proxy_values[family],
            "proxy_fraction":
                proxy_fraction,
            "pilot_accepted_events":
                int(
                    pilot["accepted_events"]
                ),
            "pilot_maximum_input_events":
                int(
                    pilot[
                        "maximum_input_events"
                    ]
                ),
            "final_family_target_events":
                int(
                    pilot[
                        "final_family_target_events"
                    ]
                ),
            "chosen_fraction":
                chosen_fraction,
            "normalization_use_authorized":
                False,
        })

    worker = Path(
        build["worker"]
    )

    require(
        worker.is_file(),
        f"missing worker {worker}",
    )

    reference_campaign = (
        "final_background_triboson_zbb_canary100_20260722_v1"
    )

    reference_submit = (
        store
        / "condor_submit"
        / reference_campaign
        / f"{reference_campaign}.sub"
    )

    require(
        reference_submit.is_file(),
        f"missing reference submit {reference_submit}",
    )

    reference_text = reference_submit.read_text(
        errors="replace"
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
        / "final_background_triboson_zbb_pilot10k_submission_20260722_v1"
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

    for pilot in PILOTS:
        family = str(
            pilot["family"]
        )

        payload_row = build_rows[family]

        payload = Path(
            payload_row["payload"]
        )

        require(
            payload.is_file(),
            f"{family}: missing payload {payload}",
        )

        require(
            sha256_file(payload)
            == payload_row["payload_sha256"],
            f"{family}: payload SHA mismatch",
        )

        require(
            payload_row["exact_target_output"]
            is True,
            f"{family}: payload is not exact-target",
        )

        require(
            payload_row["physics_yield_authorized"]
            is False,
            f"{family}: unexpected yield authorization",
        )

        queue_rows.append({
            "family": family,
            "target_tag":
                pilot["target_tag"],
            "shard_id":
                pilot["shard_id"],
            "accepted_events":
                pilot["accepted_events"],
            "maximum_input_events":
                pilot["maximum_input_events"],
            "seed":
                pilot["seed"],
            "payload_path":
                str(payload),
            "payload_name":
                payload.name,
            "payload_sha256":
                payload_row["payload_sha256"],
            "final_family_target_events":
                pilot[
                    "final_family_target_events"
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
        "payload_path",
        "payload_name",
        "payload_sha256",
        "final_family_target_events",
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

        for row in queue_rows:
            writer.writerow(row)

    submit_file = (
        submit_dir
        / f"{CAMPAIGN}.sub"
    )

    submit_text = f"""universe = vanilla

executable = {worker}

arguments = {CAMPAIGN} $(family) $(target_tag) $(shard_id) $(accepted_events) $(maximum_input_events) $(seed) validation {SPLIT_SALT} preserve_template pilot_validation $(payload_name) {eos_dir} $(payload_sha256) $(Cluster)

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
+DatasetRole = "pilot_validation"
+DatasetSplit = "validation"
+CountTowardBackgroundTarget = True
+AcceptedEventTarget = $(accepted_events)
+MaximumInclusiveInputEvents = $(maximum_input_events)
+FinalTribosonFamilyTargetEvents = $(final_family_target_events)
+ExactTargetOutput = True
+InclusivePythiaZDecays = True
+AllZForcedToBB = False
+FilterRequirement = "at least one direct Z to bb"
+AllocationSemantics = "MC statistics proxy only"
+PhysicsYieldAuthorized = False

queue family, target_tag, shard_id, accepted_events, maximum_input_events, seed, payload_path, payload_name, payload_sha256, final_family_target_events from {queue_path}
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
        "Condor dry run produced no classads",
    )

    allocation_path = (
        outdir
        / "triboson_final50k_allocation_freeze.json"
    )

    allocation_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "status": "frozen",
                "allocation_basis":
                    (
                        "canary inclusive generator xsec "
                        "times observed filter acceptance"
                    ),
                "allocation_use":
                    "MC statistics allocation only",
                "normalization_use_authorized":
                    False,
                "final_total_events":
                    50000,
                "pilot_total_events":
                    10000,
                "rows":
                    allocation_rows,
                "physics_yield_authorized":
                    False,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )

    authorization_path = (
        outdir
        / "triboson_pilot10k_submission_authorization.json"
    )

    authorization_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "status":
                    "prepared_and_dryrun_valid",
                "campaign": CAMPAIGN,
                "jobs": 3,
                "accepted_events_total":
                    10000,
                "count_toward_background_target":
                    True,
                "validated_background_events_before_pilot":
                    4950000,
                "potential_validated_background_events_after_pilot":
                    4960000,
                "worker":
                    str(worker),
                "worker_sha256":
                    build["worker_sha256"],
                "canary_final_audit":
                    str(audit_path),
                "allocation_freeze":
                    str(allocation_path),
                "submit_file":
                    str(submit_file),
                "queue_file":
                    str(queue_path),
                "pilot_submission_authorized":
                    True,
                "scaleout40k_submission_authorized":
                    False,
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
        print("TRIBOSON_PILOT10K_DRYRUN_VALID")
        print("TRIBOSON_PILOT10K_SUBMISSION_AUTHORIZED")
        print("RERUN_WITH_SUBMIT_FLAG_TO_SUBMIT")
        return

    print("===== SUBMIT TRIBOSON PILOT 10K =====")

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
        / "triboson_pilot10k_cluster.tsv"
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
            "jobs": 3,
            "accepted_events_total": 10000,
            "count_toward_background_target":
                True,
            "submit_file": str(submit_file),
        })

    print()
    print("WWZ_ZBB_PILOT5200_SUBMITTED")
    print("WZZ_ZBB_PILOT3300_SUBMITTED")
    print("ZZZ_ZBB_PILOT1500_SUBMITTED")
    print("TRIBOSON_PILOT_JOBS_SUBMITTED=3")
    print("TRIBOSON_PILOT_ACCEPTED_EVENTS_SUBMITTED=10000")
    print("NO_TRIBOSON_SCALEOUT40K_SUBMISSION_AUTHORIZED_YET")
    print("NO_BACKGROUND_PHYSICS_YIELD_AUTHORIZATION_YET")
    print(f"actual_schedd={actual_schedd}")
    print(f"cluster_id={cluster_id}")
    print(f"cluster_record={cluster_path}")


if __name__ == "__main__":
    main()
