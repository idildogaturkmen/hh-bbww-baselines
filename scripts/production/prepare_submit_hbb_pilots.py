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


CONFIG = {
    "ggh_hbb": {
        "source_campaign": "ggh_hbb_canary100_20260722_v1",
        "campaign": "ggh_hbb_pilot10k_20260722_v1",
        "target_tag": "ggh_hbb_run2_frozen_v2_pilot_shard9601_10k",
        "shard_id": 9601,
        "seed": 996001,
    },
    "bbh_hbb_4fs": {
        "source_campaign": "bbh_hbb_4fs_canary100_20260722_v1",
        "campaign": "bbh_hbb_4fs_pilot10k_20260722_v1",
        "target_tag": "bbh_hbb_4fs_run2_frozen_v2_pilot_shard9602_10k",
        "shard_id": 9602,
        "seed": 996002,
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


def run(
    arguments: list[str | Path],
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
        env=os.environ.copy(),
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
        "--schedd",
        default="lpcschedd5.fnal.gov",
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
        / "final_background_hbb_canary_audit_20260722_v2"
        / "hbb_canary_final_audit.json"
    )

    require(
        audit_path.is_file(),
        f"missing audit {audit_path}",
    )

    audit = json.loads(
        audit_path.read_text()
    )

    required_audit = {
        "status": "pass",
        "hbb_canaries_final_audit_valid": True,
        "ggh_hbb_pilot10k_submission_authorized": True,
        "bbh_hbb_4fs_pilot10k_submission_authorized": True,
        "hbb_scaleout_submission_authorized": False,
        "physics_yield_authorized": False,
        "canary_events_counted_toward_background": 0,
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

    audit_rows = {
        row["family"]: row
        for row in audit["technical_subprocesses"]
    }

    outdir = (
        repo
        / "outputs/agent_runs"
        / "final_background_hbb_pilot10k_submission_20260722_v1"
    )

    require(
        not outdir.exists(),
        f"refusing to overwrite {outdir}",
    )

    outdir.mkdir(
        parents=True,
    )

    authorization_rows: list[dict[str, Any]] = []
    cluster_rows: list[dict[str, Any]] = []

    for family, config in CONFIG.items():
        print(
            f"===== PREPARING {family} =====",
            flush=True,
        )

        source_campaign = config["source_campaign"]
        campaign = config["campaign"]

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

        payload_sha = sha256_file(
            payload
        )

        audit_row = audit_rows[family]

        require(
            payload_sha
            == audit_row["payload_sha256"],
            f"{family}: payload differs from audited payload",
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

        credential_dir = (
            store
            / "condor_credentials"
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

        submit_dir.mkdir(
            parents=True,
        )
        log_dir.mkdir(
            parents=True,
        )
        receipt_dir.mkdir(
            parents=True,
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

        os.environ["X509_USER_PROXY"] = str(
            campaign_proxy
        )

        eos_dir = (
            f"/store/user/{os.environ['USER']}/"
            "hh4b_delphes/run2_13tev/frozen_v2/"
            f"bundles/{campaign}"
        )

        subprocess.run(
            [
                "xrdfs",
                "root://cmseos.fnal.gov",
                "mkdir",
                "-p",
                eos_dir,
            ],
            check=True,
        )

        submit_file = (
            submit_dir
            / f"{campaign}.sub"
        )

        dryrun_ads = (
            submit_dir
            / f"{campaign}_dryrun.classads"
        )

        dryrun_log = (
            submit_dir
            / f"{campaign}_dryrun.log"
        )

        authorization_path = (
            submit_dir
            / f"{campaign}_authorization.json"
        )

        split_salt = (
            "final-background-hbb-pilot-v1"
        )

        submit_text = f"""universe = vanilla

executable = {executable}

arguments = {campaign} {family} {config["target_tag"]} {config["shard_id"]} 10000 {config["seed"]} validation {split_salt} preserve_template pilot_validation {payload.name} {eos_dir} {payload_sha} $(Cluster)

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
+DatasetRole = "pilot_validation"
+CountTowardBackgroundTarget = True
+PhysicsYieldAuthorized = False
+HiggsDecayStrategy = "Pythia force H->bb"

queue 1
"""

        submit_file.write_text(
            submit_text
        )

        dryrun_output = run(
            [
                "condor_submit",
                "-dry-run",
                str(dryrun_ads),
                str(submit_file),
            ]
        )

        dryrun_log.write_text(
            dryrun_output
        )

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
            "target_tag": config["target_tag"],
            "events": 10000,
            "shard_id": config["shard_id"],
            "seed": config["seed"],
            "dataset_split": "validation",
            "dataset_role": "pilot_validation",
            "count_toward_background_target": True,
            "physics_yield_authorized": False,
            "source_canary_audit": str(audit_path),
            "payload": str(payload),
            "payload_sha256": payload_sha,
            "worker": str(executable),
            "submit_file": str(submit_file),
            "dryrun_valid": True,
            "pilot_submission_authorized": True,
            "scaleout_submission_authorized": False,
        }

        authorization_path.write_text(
            json.dumps(
                authorization,
                indent=2,
                sort_keys=True,
            )
            + "\n"
        )

        authorization_rows.append(
            authorization
        )

        print(
            f"{family}: dry-run PASS",
            flush=True,
        )

        if arguments.submit:
            submission_output = run(
                [
                    "condor_submit",
                    str(submit_file),
                ]
            )

            print(
                submission_output,
                flush=True,
            )

            matches = re.findall(
                r"cluster\s+([0-9]+)",
                submission_output,
                re.IGNORECASE,
            )

            require(
                len(matches) == 1,
                (
                    f"{family}: could not determine "
                    f"cluster from {matches}"
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
                else arguments.schedd
            )

            cluster_rows.append({
                "family": family,
                "campaign": campaign,
                "schedd": actual_schedd,
                "cluster_id": int(matches[0]),
                "events": 10000,
                "count_toward_background_target": True,
                "submit_file": str(submit_file),
            })

            print(
                f"{family}: submitted cluster={matches[0]}",
                flush=True,
            )

    authorization_index = (
        outdir
        / "pilot_authorizations.json"
    )

    authorization_index.write_text(
        json.dumps(
            authorization_rows,
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )

    if arguments.submit:
        cluster_index = (
            outdir
            / "pilot_clusters.tsv"
        )

        with cluster_index.open(
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

        require(
            len(cluster_rows) == 2,
            "expected two submitted pilot jobs",
        )

        print()
        print("GGH_HBB_PILOT10K_SUBMITTED")
        print("BBH_HBB_4FS_PILOT10K_SUBMITTED")
        print("HBB_COUNTED_PILOT_EVENTS_SUBMITTED=20000")
        print("NO_HBB_SCALEOUT_SUBMISSION_AUTHORIZED_YET")
        print(f"cluster_index={cluster_index}")
    else:
        print()
        print("HBB_PILOT10K_DRYRUNS_VALID")
        print("RERUN_WITH_SUBMIT_FLAG_TO_SUBMIT")


if __name__ == "__main__":
    main()
