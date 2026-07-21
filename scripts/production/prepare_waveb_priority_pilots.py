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


FAMILIES = (
    "tth_hbb",
    "ttz_zbb",
    "tttt",
    "vbf_hbb",
)

SHARD_IDS = {
    "tth_hbb": 1001,
    "ttz_zbb": 1002,
    "tttt": 1003,
    "vbf_hbb": 1004,
}


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
            lambda: handle.read(8 * 1024 * 1024),
            b"",
        ):
            digest.update(block)

    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--repo", required=True)
    parser.add_argument("--store", required=True)
    parser.add_argument("--plan", required=True)
    parser.add_argument("--campaign", required=True)
    parser.add_argument(
        "--source-canary-campaign",
        required=True,
    )
    parser.add_argument("--proxy", required=True)

    args = parser.parse_args()

    repo = Path(args.repo).resolve()
    store = Path(args.store).resolve()
    plan_path = Path(args.plan).resolve()
    proxy = Path(args.proxy).resolve()

    campaign = args.campaign
    source_campaign = args.source_canary_campaign

    if not plan_path.is_file():
        raise SystemExit(
            f"ERROR: missing Wave-B plan: {plan_path}"
        )

    if not proxy.is_file():
        raise SystemExit(
            f"ERROR: missing shared proxy: {proxy}"
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
            "ERROR: local and remote repository heads differ"
        )

    worker = (
        repo
        / "scripts"
        / "production"
        / "run_mg5_frozen_v2_bundle.sh"
    )

    if not os.access(worker, os.X_OK):
        raise SystemExit(
            f"ERROR: worker is not executable: {worker}"
        )

    with plan_path.open(newline="") as handle:
        plan_rows = list(
            csv.DictReader(
                handle,
                delimiter="\t",
            )
        )

    plan_by_family = {
        row["family"]: row
        for row in plan_rows
        if row["family"] in FAMILIES
    }

    if set(plan_by_family) != set(FAMILIES):
        raise SystemExit(
            "ERROR: priority families are incomplete in the plan"
        )

    source_submit_dir = (
        store
        / "condor_submit"
        / source_campaign
    )

    source_manifest_path = (
        source_submit_dir
        / f"{source_campaign}_manifest.csv"
    )

    source_receipt_dir = (
        store
        / "condor_return"
        / source_campaign
        / "receipts"
    )

    if not source_manifest_path.is_file():
        raise SystemExit(
            "ERROR: source canary manifest is missing"
        )

    if not source_receipt_dir.is_dir():
        raise SystemExit(
            "ERROR: source canary receipts are missing"
        )

    with source_manifest_path.open(
        newline=""
    ) as handle:
        source_rows = list(
            csv.DictReader(handle)
        )

    source_by_family = {
        row["family"]: row
        for row in source_rows
    }

    if set(source_by_family) != set(FAMILIES):
        raise SystemExit(
            "ERROR: source canary manifest family set is wrong"
        )

    validated_inputs = {}

    for family in FAMILIES:
        plan = plan_by_family[family]
        source = source_by_family[family]

        events = int(plan["events"])
        seed = int(plan["seed"])
        dataset_split = plan["dataset_split"]

        if events != 10000:
            raise SystemExit(
                f"ERROR: {family} pilot does not request 10000 events"
            )

        if seed <= 0:
            raise SystemExit(
                f"ERROR: invalid pilot seed for {family}"
            )

        if dataset_split not in {
            "train",
            "validation",
        }:
            raise SystemExit(
                f"ERROR: unauthorized split for {family}"
            )

        if plan["run_card_profile"] != "preserve_template":
            raise SystemExit(
                f"ERROR: wrong run-card profile for {family}"
            )

        if source["count_toward_5m"] != "False":
            raise SystemExit(
                f"ERROR: source {family} canary was not marked "
                "as infrastructure-only"
            )

        payload = Path(
            source["payload"]
        ).resolve()

        if (
            not payload.is_file()
            or payload.stat().st_size == 0
        ):
            raise SystemExit(
                f"ERROR: missing source payload for {family}: "
                f"{payload}"
            )

        observed_payload_sha = sha256_file(
            payload
        )

        expected_payload_sha = source[
            "payload_sha256"
        ]

        if observed_payload_sha != expected_payload_sha:
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
                f"ERROR: expected one source receipt for "
                f"{family}, found {len(receipt_paths)}"
            )

        receipt = json.loads(
            receipt_paths[0].read_text()
        )

        required_receipt_values = {
            "campaign": source_campaign,
            "family": family,
            "stage": "complete_copied_and_verified",
            "exit_status": 0,
            "n_events": 100,
            "lhe_events": 100,
            "hepmc_events": 100,
            "root_events": 100,
            "payload_sha256": expected_payload_sha,
            "expected_payload_sha256": expected_payload_sha,
        }

        for key, expected in required_receipt_values.items():
            observed = receipt.get(key)

            if observed != expected:
                raise SystemExit(
                    f"ERROR: source canary gate failed for "
                    f"{family}, field {key}: "
                    f"{observed!r} != {expected!r}"
                )

        if not receipt.get(
            "remote_bundle",
            "",
        ).startswith("/store/user/"):
            raise SystemExit(
                f"ERROR: source canary lacks EOS bundle for {family}"
            )

        validated_inputs[family] = {
            "plan": plan,
            "source": source,
            "receipt": receipt,
            "payload": payload,
            "payload_sha256": expected_payload_sha,
        }

    if len(
        {
            validated_inputs[family]["plan"]["seed"]
            for family in FAMILIES
        }
    ) != len(FAMILIES):
        raise SystemExit(
            "ERROR: pilot seeds are not unique"
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
        if path.exists():
            raise SystemExit(
                f"ERROR: refusing to reuse campaign path: {path}"
            )

    submit_dir.mkdir(parents=True)
    log_dir.mkdir(parents=True)
    receipt_dir.mkdir(parents=True)

    eos_dir = (
        f"/store/user/{os.environ['USER']}/"
        "hh4b_delphes/run2_13tev/frozen_v2/"
        f"bundles/{campaign}"
    )

    manifest_rows = []
    submit_files = []

    for family in FAMILIES:
        validated = validated_inputs[family]
        plan = validated["plan"]
        source = validated["source"]
        receipt = validated["receipt"]
        payload = validated["payload"]

        events = int(plan["events"])
        seed = int(plan["seed"])
        shard_id = SHARD_IDS[family]
        dataset_split = plan["dataset_split"]
        run_card_profile = plan[
            "run_card_profile"
        ]
        dataset_role = plan["dataset_role"]

        target_tag = (
            f"{family}_run2_frozen_v2_"
            f"waveb_pilot_shard{shard_id:04d}_"
            f"{events}evt"
        )

        submit_file = (
            submit_dir
            / f"{campaign}_{family}.sub"
        )

        dry_run = (
            submit_dir
            / f"{campaign}_{family}_dryrun.txt"
        )

        submit_text = f"""universe = vanilla

executable = {worker}

arguments = {campaign} {family} {target_tag} {shard_id} {events} {seed} {dataset_split} unified-background-5m-waveb-v1 {run_card_profile} {dataset_role} {payload.name} {eos_dir} {validated['payload_sha256']} $(Cluster)

output = {log_dir}/{family}_$(Cluster).out
error = {log_dir}/{family}_$(Cluster).err
log = {log_dir}/{family}_$(Cluster).log

should_transfer_files = YES
when_to_transfer_output = ON_EXIT

transfer_input_files = {payload}

transfer_output_files = job_receipt.json
transfer_output_remaps = "job_receipt.json = {receipt_dir}/{family}_$(Cluster)_receipt.json"

request_cpus = 1
request_memory = 6GB
request_disk = 35GB

getenv = False
environment = "LC_ALL=C LANG=C"

use_x509userproxy = True
x509userproxy = {proxy}

+JobBatchName = "{campaign}_{family}"
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
            "group": plan["group"],
            "target_tag": target_tag,
            "shard_id": shard_id,
            "events": events,
            "seed": seed,
            "dataset_split": dataset_split,
            "split_salt": "unified-background-5m-waveb-v1",
            "sampling_mode": plan["sampling_mode"],
            "forced_decay": plan["forced_decay"],
            "generator_enrichment": plan[
                "generator_enrichment"
            ],
            "run_card_profile": run_card_profile,
            "dataset_role": dataset_role,
            "overlap_group": plan["overlap_group"],
            "normalization_rule": plan[
                "normalization_rule"
            ],
            "payload": str(payload),
            "payload_sha256": validated[
                "payload_sha256"
            ],
            "payload_git_head": receipt[
                "payload_git_head"
            ],
            "source_canary_campaign": source_campaign,
            "source_canary_target_tag": source[
                "target_tag"
            ],
            "source_canary_remote_bundle": receipt[
                "remote_bundle"
            ],
            "source_canary_receipt": str(
                source_receipt_dir
                / (
                    f"{family}_"
                    f"{receipt['cluster_id']}"
                    "_receipt.json"
                )
            ),
            "eos_directory": eos_dir,
            "submit_file": str(submit_file),
            "dry_run": str(dry_run),
            "count_toward_5m": True,
        })

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

    commands_path.chmod(0o755)

    summary = {
        "schema_version": 1,
        "status": (
            "waveb_priority_pilots_"
            "prepared_and_dryrun_valid"
        ),
        "campaign": campaign,
        "orchestration_git_head": local_head,
        "source_canary_campaign": source_campaign,
        "pilot_jobs": len(
            manifest_rows
        ),
        "pilot_events": sum(
            row["events"]
            for row in manifest_rows
        ),
        "families": list(FAMILIES),
        "eos_directory": eos_dir,
        "manifest": str(manifest_path),
        "submit_commands": str(commands_path),
        "count_toward_5m": True,
        "large_scale_production_authorized": False,
        "final_test_authorized": False,
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
        "WAVEB_PRIORITY_PILOTS_PREPARED_AND_DRYRUN_VALID"
    )
    print(f"campaign={campaign}")
    print(f"source_canary_campaign={source_campaign}")
    print(f"pilot_jobs={len(manifest_rows)}")
    print(
        "pilot_events="
        f"{sum(row['events'] for row in manifest_rows)}"
    )
    print("count_toward_5m=True")
    print(f"eos_directory={eos_dir}")
    print(f"manifest={manifest_path}")
    print(f"submit_commands={commands_path}")


if __name__ == "__main__":
    main()
