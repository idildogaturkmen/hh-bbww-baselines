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


AUDIT_TAG = "final_background_hbb_pilots10k_audit_20260722_v1"
AUDIT_FILENAME = "hbb_pilots10k_final_audit.json"

CONFIG: dict[str, dict[str, Any]] = {
    "ggh_hbb": {
        "source_campaign": "ggh_hbb_pilot10k_20260722_v1",
        "campaign": "ggh_hbb_scaleout90k_20260722_v1",
        "events_total": 90000,
        "canonical_family_target": 100000,
        "matrix_element": "g g -> h [noborn=QCD], loop_sm",
        "shards": [
            {
                "target_tag":
                    f"ggh_hbb_run2_frozen_v2_train_shard{shard}_10k",
                "shard_id": shard,
                "seed": 997000 + offset,
                "dataset_split": "train",
                "dataset_role": "canonical_train",
            }
            for offset, shard in enumerate(
                range(9701, 9709),
                start=1,
            )
        ] + [
            {
                "target_tag":
                    "ggh_hbb_run2_frozen_v2_validation_shard9709_10k",
                "shard_id": 9709,
                "seed": 997009,
                "dataset_split": "validation",
                "dataset_role": "canonical_validation",
            }
        ],
    },
    "bbh_hbb_4fs": {
        "source_campaign": "bbh_hbb_4fs_pilot10k_20260722_v1",
        "campaign": "bbh_hbb_4fs_scaleout40k_20260722_v1",
        "events_total": 40000,
        "canonical_family_target": 50000,
        "matrix_element": "p p -> b b~ h QCD=2 QED=1, 4FS",
        "shards": [
            {
                "target_tag":
                    f"bbh_hbb_4fs_run2_frozen_v2_train_shard{shard}_10k",
                "shard_id": shard,
                "seed": 998000 + offset,
                "dataset_split": "train",
                "dataset_role": "canonical_train",
            }
            for offset, shard in enumerate(
                range(9801, 9805),
                start=1,
            )
        ],
    },
}


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


def submit_value(text: str, key: str) -> str:
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

    audit_path = (
        repo
        / "outputs/agent_runs"
        / AUDIT_TAG
        / AUDIT_FILENAME
    )

    require(
        audit_path.is_file(),
        f"missing Hbb pilot audit {audit_path}",
    )

    audit = json.loads(
        audit_path.read_text()
    )

    required_audit = {
        "status": "pass",
        "hbb_pilots20k_final_audit_valid": True,
        "ggh_hbb_scaleout90k_submission_authorized": True,
        "bbh_hbb_4fs_scaleout40k_submission_authorized": True,
        "validated_pilot_events": 20000,
        "validated_background_events_after_pilots": 4820000,
        "remaining_hbb_events": 130000,
        "remaining_triboson_events": 50000,
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

    pilot_rows = {
        row["family"]: row
        for row in audit["pilots"]
    }

    require(
        set(pilot_rows) == set(CONFIG),
        "pilot audit and scale-out family sets differ",
    )

    total_jobs = sum(
        len(config["shards"])
        for config in CONFIG.values()
    )

    total_events = sum(
        int(config["events_total"])
        for config in CONFIG.values()
    )

    require(
        total_jobs == 13,
        f"expected 13 Hbb jobs, found {total_jobs}",
    )

    require(
        total_events == 130000,
        f"expected 130000 Hbb events, found {total_events}",
    )

    all_shard_ids = [
        int(shard["shard_id"])
        for config in CONFIG.values()
        for shard in config["shards"]
    ]

    all_seeds = [
        int(shard["seed"])
        for config in CONFIG.values()
        for shard in config["shards"]
    ]

    require(
        len(set(all_shard_ids)) == len(all_shard_ids),
        "Hbb scale-out shard IDs are not unique",
    )

    require(
        len(set(all_seeds)) == len(all_seeds),
        "Hbb scale-out seeds are not unique",
    )

    outdir = (
        repo
        / "outputs/agent_runs"
        / "final_background_hbb_scaleouts130k_submission_20260722_v1"
    )

    require(
        not outdir.exists(),
        f"refusing to overwrite {outdir}",
    )

    outdir.mkdir(parents=True)

    authorization_rows: list[dict[str, Any]] = []
    cluster_rows: list[dict[str, Any]] = []

    for family, config in CONFIG.items():
        print(
            f"===== PREPARING {family} SCALE-OUT =====",
            flush=True,
        )

        source_campaign = str(
            config["source_campaign"]
        )

        campaign = str(
            config["campaign"]
        )

        shards = list(
            config["shards"]
        )

        source_submit = (
            store
            / "condor_submit"
            / source_campaign
            / f"{source_campaign}.sub"
        )

        require(
            source_submit.is_file(),
            f"missing source submit {source_submit}",
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
            f"{family}: missing worker {executable}",
        )

        require(
            payload.is_file(),
            f"{family}: missing payload {payload}",
        )

        require(
            source_proxy.is_file(),
            f"{family}: missing proxy {source_proxy}",
        )

        expected_payload_sha = str(
            pilot_rows[family]["payload_sha256"]
        )

        observed_payload_sha = sha256_file(
            payload
        )

        require(
            observed_payload_sha == expected_payload_sha,
            f"{family}: payload differs from audited pilot payload",
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
            require(
                not path.exists(),
                f"refusing to overwrite {path}",
            )

        submit_dir.mkdir(parents=True)
        log_dir.mkdir(parents=True)
        receipt_dir.mkdir(parents=True)

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
            / f"x509up_u{os.getuid()}_{campaign}"
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
            f"bundles/{campaign}"
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

        queue_path = (
            submit_dir
            / f"{campaign}_queue.tsv"
        )

        with queue_path.open(
            "w",
            newline="",
        ) as handle:
            for shard in shards:
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

        shard_plan_path = (
            outdir
            / f"{campaign}_shard_plan.tsv"
        )

        with shard_plan_path.open(
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
            writer.writerows(shards)

        submit_file = (
            submit_dir
            / f"{campaign}.sub"
        )

        submit_text = f"""universe = vanilla

executable = {executable}

arguments = {campaign} {family} $(target_tag) $(shard_id) 10000 $(seed) $(dataset_split) final-background-hbb-scaleout-v1 preserve_template $(dataset_role) {payload.name} {eos_dir} {expected_payload_sha} $(Cluster)

should_transfer_files = YES
when_to_transfer_output = ON_EXIT

transfer_input_files = {payload}

transfer_output_files = job_receipt.json
transfer_output_remaps = "job_receipt.json = {receipt_dir}/{family}_$(Cluster)_$(Process)_receipt.json"

output = {log_dir}/{family}_$(Cluster)_$(Process).out
error = {log_dir}/{family}_$(Cluster)_$(Process).err
log = {log_dir}/{family}_$(Cluster).condor.log

request_cpus = {request_cpus}
request_memory = {request_memory}
request_disk = {request_disk}

notification = Never

use_x509userproxy = True
x509userproxy = {campaign_proxy}

+JobBatchName = "{campaign}_{family}"
+CampaignName = "{campaign}"
+DatasetFamily = "{family}"
+DatasetRole = "$(dataset_role)"
+DatasetSplit = "$(dataset_split)"
+CountTowardBackgroundTarget = True
+CanonicalBackgroundFamilyTargetEvents = {config["canonical_family_target"]}
+PhysicsYieldAuthorized = False
+MatrixElementProcess = "{config["matrix_element"]}"
+HiggsDecayStrategy = "Pythia force H->bb"

queue target_tag, shard_id, seed, dataset_split, dataset_role from {queue_path}
"""

        submit_file.write_text(
            submit_text
        )

        dryrun_ads = (
            submit_dir
            / f"{campaign}_dryrun.classads"
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

        require(
            dryrun_ads.is_file()
            and dryrun_ads.stat().st_size > 0,
            f"{family}: dry-run did not create classads",
        )

        authorization = {
            "schema_version": 1,
            "status": "prepared_and_dryrun_valid",
            "family": family,
            "campaign": campaign,
            "jobs": len(shards),
            "events_per_job": 10000,
            "events_total": int(config["events_total"]),
            "canonical_family_target":
                int(config["canonical_family_target"]),
            "count_toward_background_target": True,
            "physics_yield_authorized": False,
            "source_pilot_audit": str(audit_path),
            "payload": str(payload),
            "payload_sha256": expected_payload_sha,
            "worker": str(executable),
            "submit_file": str(submit_file),
            "shard_plan": str(shard_plan_path),
            "dryrun_valid": True,
            "scaleout_submission_authorized": True,
        }

        authorization_rows.append(
            authorization
        )

        if not arguments.submit:
            continue

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
                f"{family}: could not determine cluster: "
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

        cluster_rows.append({
            "family": family,
            "campaign": campaign,
            "schedd": actual_schedd,
            "cluster_id": int(cluster_matches[0]),
            "jobs": len(shards),
            "events_per_job": 10000,
            "events_total": int(config["events_total"]),
            "count_toward_background_target": True,
            "submit_file": str(submit_file),
        })

    authorization_path = (
        outdir
        / "hbb_scaleouts130k_authorizations.json"
    )

    authorization_path.write_text(
        json.dumps(
            authorization_rows,
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )

    if not arguments.submit:
        print()
        print("HBB_SCALEOUTS130K_DRYRUNS_VALID")
        print("RERUN_WITH_SUBMIT_FLAG_TO_SUBMIT")
        return

    require(
        len(cluster_rows) == 2,
        "expected two submitted Hbb scale-out clusters",
    )

    cluster_path = (
        outdir
        / "hbb_scaleouts130k_clusters.tsv"
    )

    with cluster_path.open(
        "w",
        newline="",
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=list(cluster_rows[0]),
            delimiter="\t",
            lineterminator="\n",
        )

        writer.writeheader()
        writer.writerows(cluster_rows)

    print()
    print("GGH_HBB_SCALEOUT90K_SUBMITTED")
    print("BBH_HBB_4FS_SCALEOUT40K_SUBMITTED")
    print("HBB_SCALEOUT_JOBS_SUBMITTED=13")
    print("HBB_SCALEOUT_EVENTS_SUBMITTED=130000")
    print("BACKGROUND_TARGET_EVENTS_IN_FLIGHT=4950000")
    print("BACKGROUND_EVENTS_STILL_BLOCKED_TRIBOSON=50000")
    print("NO_BACKGROUND_PHYSICS_YIELD_AUTHORIZATION_YET")
    print(f"cluster_index={cluster_path}")


if __name__ == "__main__":
    main()
