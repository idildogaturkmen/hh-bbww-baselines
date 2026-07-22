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


CAMPAIGN = "vbf_hh4b_sm_scaleout90k_20260722_v1"
FAMILY = "vbf_hh4b_sm"

AUDIT_TAG = "vbf_hh4b_sm_pilot10k_final_audit_20260722_v1"
AUDIT_FILENAME = "vbf_hh4b_pilot10k_final_audit.json"

SOURCE_CAMPAIGN = "vbf_hh4b_sm_pilot10k_20260722_v1"

PAYLOAD_SHA256 = (
    "adfdc4ac14341e195c31fe6dcd9a2ef1a4dd1e7c21f7bd87ab1e86dda305726a"
)

SHARDS = [
    {
        "target_tag":
            f"vbf_hh4b_sm_run2_frozen_v2_train_shard{shard}_10k",
        "shard_id": shard,
        "seed": 984000 + offset,
        "dataset_split": "train",
        "dataset_role": "canonical_train",
    }
    for offset, shard in enumerate(
        range(9401, 9409),
        start=1,
    )
]

SHARDS.append({
    "target_tag":
        "vbf_hh4b_sm_run2_frozen_v2_validation_shard9409_10k",
    "shard_id": 9409,
    "seed": 984009,
    "dataset_split": "validation",
    "dataset_role": "canonical_validation",
})


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def sha256_file(path: Path) -> str:
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
        f"missing {key} in source submit file",
    )

    return match.group(1)


def run_shell(
    arguments: list[str | Path],
    *,
    environment: dict[str, str] | None = None,
) -> str:
    command = shlex.join(
        [str(argument) for argument in arguments]
    )

    result = subprocess.run(
        [
            "/bin/bash",
            "-lc",
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
        / AUDIT_TAG
        / AUDIT_FILENAME
    )

    require(
        audit_path.is_file(),
        f"missing VBF pilot audit {audit_path}",
    )

    audit = json.loads(
        audit_path.read_text()
    )

    required_audit = {
        "status": "pass",
        "vbf_hh4b_pilot10k_final_audit_valid": True,
        "vbf_hh4b_scaleout90k_submission_authorized": True,
        "canonical_signal_target_events": 100000,
        "validated_pilot_events": 10000,
        "remaining_scaleout_events": 90000,
        "physics_yield_authorized": False,
    }

    for key, expected in required_audit.items():
        observed = audit.get(key)

        require(
            observed == expected,
            (
                f"audit mismatch for {key}: "
                f"{observed!r} != {expected!r}"
            ),
        )

    require(
        len(SHARDS) == 9,
        "expected exactly nine VBF scale-out shards",
    )

    require(
        sum(
            1
            for shard in SHARDS
            if shard["dataset_split"] == "train"
        )
        == 8,
        "expected eight train shards",
    )

    require(
        sum(
            1
            for shard in SHARDS
            if shard["dataset_split"] == "validation"
        )
        == 1,
        "expected one validation shard",
    )

    require(
        len({
            shard["shard_id"]
            for shard in SHARDS
        })
        == 9,
        "VBF shard IDs are not unique",
    )

    require(
        len({
            shard["seed"]
            for shard in SHARDS
        })
        == 9,
        "VBF MG5 seeds are not unique",
    )

    source_submit = (
        store
        / "condor_submit"
        / SOURCE_CAMPAIGN
        / f"{SOURCE_CAMPAIGN}.sub"
    )

    require(
        source_submit.is_file(),
        f"missing source submit file {source_submit}",
    )

    source_text = source_submit.read_text(
        errors="replace"
    )

    executable = Path(
        submit_value(
            source_text,
            "executable",
        )
    )

    payload = Path(
        submit_value(
            source_text,
            "transfer_input_files",
        )
    )

    source_proxy = Path(
        submit_value(
            source_text,
            "x509userproxy",
        )
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

    require(
        executable.is_file(),
        f"missing worker {executable}",
    )

    require(
        payload.is_file(),
        f"missing payload {payload}",
    )

    require(
        source_proxy.is_file(),
        f"missing source proxy {source_proxy}",
    )

    require(
        sha256_file(payload) == PAYLOAD_SHA256,
        "VBF payload differs from audited payload",
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
        / "vbf_hh4b_sm_scaleout90k_submission_20260722_v1"
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

    plan_path = (
        outdir
        / "vbf_scaleout90k_shard_plan.tsv"
    )

    with plan_path.open(
        "w",
        newline="",
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "target_tag",
                "shard_id",
                "seed",
                "dataset_split",
                "dataset_role",
            ],
            delimiter="\t",
            lineterminator="\n",
        )

        writer.writeheader()
        writer.writerows(SHARDS)

    queue_path = (
        submit_dir
        / f"{CAMPAIGN}_queue.tsv"
    )

    with queue_path.open(
        "w",
        newline="",
    ) as handle:
        for shard in SHARDS:
            handle.write(
                "\t".join([
                    str(shard["target_tag"]),
                    str(shard["shard_id"]),
                    str(shard["seed"]),
                    str(shard["dataset_split"]),
                    str(shard["dataset_role"]),
                ])
                + "\n"
            )

    submit_file = (
        submit_dir
        / f"{CAMPAIGN}.sub"
    )

    submit_text = f"""universe = vanilla

executable = {executable}

arguments = {CAMPAIGN} {FAMILY} $(target_tag) $(shard_id) 10000 $(seed) $(dataset_split) vbf-hh4b-sm-scaleout-v1 preserve_template $(dataset_role) {payload.name} {eos_dir} {PAYLOAD_SHA256} $(Cluster)

should_transfer_files = YES
when_to_transfer_output = ON_EXIT

transfer_input_files = {payload}

transfer_output_files = job_receipt.json
transfer_output_remaps = "job_receipt.json = {receipt_dir}/{FAMILY}_$(Cluster)_$(Process)_receipt.json"

output = {log_dir}/{FAMILY}_$(Cluster)_$(Process).out
error = {log_dir}/{FAMILY}_$(Cluster)_$(Process).err
log = {log_dir}/{FAMILY}_$(Cluster).condor.log

request_cpus = {request_cpus}
request_memory = {request_memory}
request_disk = {request_disk}

notification = Never

use_x509userproxy = True
x509userproxy = {campaign_proxy}

+JobBatchName = "{CAMPAIGN}_{FAMILY}"
+CampaignName = "{CAMPAIGN}"
+DatasetFamily = "{FAMILY}"
+DatasetRole = "$(dataset_role)"
+DatasetSplit = "$(dataset_split)"
+CountTowardSignalTarget = True
+CanonicalSignalTargetEvents = 100000
+PhysicsYieldAuthorized = False
+MatrixElementProcess = "p p -> h h j j, QCD=0"
+HiggsDecayStrategy = "Pythia force H->bb"

queue target_tag, shard_id, seed, dataset_split, dataset_role from {queue_path}
"""

    submit_file.write_text(
        submit_text
    )

    dryrun_ads = (
        submit_dir
        / f"{CAMPAIGN}_dryrun.classads"
    )

    dryrun_log = (
        submit_dir
        / f"{CAMPAIGN}_dryrun.log"
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

    dryrun_log.write_text(
        dryrun_output
    )

    require(
        dryrun_ads.is_file()
        and dryrun_ads.stat().st_size > 0,
        "Condor dry run did not produce classads",
    )

    authorization = {
        "schema_version": 1,
        "status": "prepared_and_dryrun_valid",
        "campaign": CAMPAIGN,
        "family": FAMILY,
        "jobs": 9,
        "events_per_job": 10000,
        "events_total": 90000,
        "train_events": 80000,
        "validation_events": 10000,
        "count_toward_signal_target": True,
        "canonical_signal_target_events": 100000,
        "physics_yield_authorized": False,
        "source_pilot_audit": str(audit_path),
        "payload": str(payload),
        "payload_sha256": PAYLOAD_SHA256,
        "worker": str(executable),
        "submit_file": str(submit_file),
        "shard_plan": str(plan_path),
        "dryrun_valid": True,
        "scaleout90k_submission_authorized": True,
    }

    authorization_path = (
        outdir
        / "vbf_scaleout90k_authorization.json"
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
        print("VBF_HH4B_SCALEOUT90K_DRYRUN_VALID")
        print("RERUN_WITH_SUBMIT_FLAG_TO_SUBMIT")
        return

    print("===== SUBMIT NINE VBF JOBS =====")

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
            "could not determine submitted cluster: "
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
        / "vbf_scaleout90k_cluster.tsv"
    )

    with cluster_path.open(
        "w",
        newline="",
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "campaign",
                "family",
                "schedd",
                "cluster_id",
                "jobs",
                "events_per_job",
                "events_total",
                "count_toward_signal_target",
                "submit_file",
            ],
            delimiter="\t",
            lineterminator="\n",
        )

        writer.writeheader()

        writer.writerow({
            "campaign": CAMPAIGN,
            "family": FAMILY,
            "schedd": actual_schedd,
            "cluster_id": cluster_id,
            "jobs": 9,
            "events_per_job": 10000,
            "events_total": 90000,
            "count_toward_signal_target": True,
            "submit_file": str(submit_file),
        })

    print()
    print("VBF_HH4B_SCALEOUT90K_SUBMITTED")
    print("VBF_HH4B_SCALEOUT_JOBS_SUBMITTED=9")
    print("VBF_HH4B_SCALEOUT_EVENTS_SUBMITTED=90000")
    print("VBF_HH4B_CANONICAL_TARGET_SUBMITTED=100000")
    print("NO_VBF_PHYSICS_YIELD_AUTHORIZATION_YET")
    print(f"actual_schedd={actual_schedd}")
    print(f"cluster_id={cluster_id}")
    print(f"cluster_record={cluster_path}")


if __name__ == "__main__":
    main()
