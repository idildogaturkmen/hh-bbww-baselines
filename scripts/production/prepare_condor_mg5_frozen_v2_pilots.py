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


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()

    with path.open("rb") as handle:
        for block in iter(
            lambda: handle.read(8 * 1024 * 1024),
            b"",
        ):
            digest.update(block)

    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument("--spec", required=True)
    parser.add_argument("--repo", required=True)
    parser.add_argument("--store", required=True)
    parser.add_argument("--campaign", required=True)
    parser.add_argument("--proxy", required=True)

    args = parser.parse_args()

    specification = Path(args.spec).resolve()
    repo = Path(args.repo).resolve()
    store = Path(args.store).resolve()
    proxy = Path(args.proxy).resolve()
    campaign = args.campaign

    if not specification.is_file():
        raise SystemExit(
            f"ERROR: missing pilot specification: {specification}"
        )

    if not proxy.is_file():
        raise SystemExit(
            f"ERROR: missing proxy: {proxy}"
        )

    if subprocess.run(
        [
            "git",
            "-C",
            str(repo),
            "status",
            "--porcelain",
        ],
        check=True,
        text=True,
        capture_output=True,
    ).stdout.strip():
        raise SystemExit(
            "ERROR: repository must be clean before preparing production"
        )

    current_head = subprocess.run(
        [
            "git",
            "-C",
            str(repo),
            "rev-parse",
            "HEAD",
        ],
        check=True,
        text=True,
        capture_output=True,
    ).stdout.strip()

    remote_head = subprocess.run(
        [
            "git",
            "-C",
            str(repo),
            "rev-parse",
            "origin/delphes-hh4b-production",
        ],
        check=True,
        text=True,
        capture_output=True,
    ).stdout.strip()

    if current_head != remote_head:
        raise SystemExit(
            "ERROR: local and remote branch heads differ"
        )

    submit_dir = store / "condor_submit" / campaign
    log_dir = store / "condor_logs" / campaign
    return_dir = (
        store
        / "condor_return"
        / campaign
        / "receipts"
    )

    for path in (
        submit_dir,
        log_dir,
        return_dir.parent,
    ):
        if path.exists():
            raise SystemExit(
                f"ERROR: refusing to reuse campaign path: {path}"
            )

    submit_dir.mkdir(parents=True)
    log_dir.mkdir(parents=True)
    return_dir.mkdir(parents=True)

    builder = (
        repo
        / "scripts/production"
        / "prepare_mg5_frozen_v2_payload.sh"
    )

    worker = (
        repo
        / "scripts/production"
        / "run_mg5_frozen_v2_bundle.sh"
    )

    if not os.access(builder, os.X_OK):
        raise SystemExit(
            f"ERROR: payload builder is not executable: {builder}"
        )

    if not os.access(worker, os.X_OK):
        raise SystemExit(
            f"ERROR: worker is not executable: {worker}"
        )

    with specification.open(newline="") as handle:
        rows = list(csv.DictReader(handle))

    if len(rows) != 6:
        raise SystemExit(
            f"ERROR: expected six pilots, found {len(rows)}"
        )

    eos_dir = (
        f"/store/user/{os.environ['USER']}/hh4b_delphes/"
        f"run2_13tev/frozen_v2/bundles/{campaign}"
    )

    manifest_rows = []
    submit_files = []

    for row in rows:
        family = row["family"]
        process_dir = Path(row["process_dir"])
        shard_id = int(row["pilot_shard"])
        events = int(row["events"])
        seed = int(row["seed"])
        dataset_split = row["dataset_split"]
        profile = row["run_card_profile"]
        dataset_role = row["dataset_role"]

        tag = (
            f"{family}_run2_frozen_v2_"
            f"unified5m_pilot_shard{shard_id:03d}_"
            f"{events // 1000}k"
        )

        payload = (
            submit_dir
            / f"{family}_inputs.tar.gz"
        )

        subprocess.run(
            [
                str(builder),
                family,
                str(process_dir),
                str(payload),
            ],
            check=True,
            env={
                **os.environ,
                "HH4B_REPO": str(repo),
                "DELPHES_DIR": (
                    f"/uscms_data/d3/{os.environ['USER']}/"
                    "software/Delphes"
                ),
            },
        )

        payload_sha256 = sha256_file(payload)
        payload_name = payload.name

        submit_file = (
            submit_dir
            / f"{campaign}_{family}.sub"
        )

        batch_name = f"{campaign}_{family}"

        submit_text = f"""universe = vanilla

executable = {worker}

arguments = {campaign} {family} {tag} {shard_id} {events} {seed} {dataset_split} unified-background-5m-v1 {profile} {dataset_role} {payload_name} {eos_dir} {payload_sha256} $(Cluster)

output = {log_dir}/{family}_$(Cluster).out
error = {log_dir}/{family}_$(Cluster).err
log = {log_dir}/{family}_$(Cluster).log

should_transfer_files = YES
when_to_transfer_output = ON_EXIT

transfer_input_files = {payload}

transfer_output_files = job_receipt.json
transfer_output_remaps = "job_receipt.json = {return_dir}/{family}_$(Cluster)_receipt.json"

request_cpus = 1
request_memory = 6GB
request_disk = 35GB

getenv = False
environment = "LC_ALL=C LANG=C"

use_x509userproxy = True
x509userproxy = {proxy}

+JobBatchName = "{batch_name}"
+DesiredOS = "EL9"

on_exit_hold = (ExitBySignal == True) || (ExitCode != 0)

queue 1
"""

        submit_file.write_text(submit_text)
        submit_files.append(submit_file)

        dry_run = (
            submit_dir
            / f"{campaign}_{family}_dryrun.txt"
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

        manifest_rows.append({
            "campaign": campaign,
            "family": family,
            "target_tag": tag,
            "shard_id": shard_id,
            "events": events,
            "seed": seed,
            "dataset_split": dataset_split,
            "split_salt": "unified-background-5m-v1",
            "run_card_profile": profile,
            "dataset_role": dataset_role,
            "process_dir": str(process_dir),
            "payload": str(payload),
            "payload_sha256": payload_sha256,
            "git_head": current_head,
            "eos_directory": eos_dir,
            "submit_file": str(submit_file),
            "dry_run": str(dry_run),
        })

    manifest_path = (
        submit_dir
        / f"{campaign}_manifest.csv"
    )

    fieldnames = list(manifest_rows[0])

    with manifest_path.open(
        "w",
        newline="",
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=fieldnames,
            lineterminator="\n",
        )

        writer.writeheader()
        writer.writerows(manifest_rows)

    commands_path = (
        submit_dir
        / "submit_commands.sh"
    )

    commands = [
        "#!/usr/bin/env bash",
        "",
        "set -euo pipefail",
        "",
    ]

    for submit_file in submit_files:
        commands.append(
            "condor_submit "
            + shlex.quote(str(submit_file))
        )

    commands_path.write_text(
        "\n".join(commands) + "\n"
    )

    commands_path.chmod(0o755)

    summary = {
        "schema_version": 1,
        "status": (
            "unified_5m_wavea_pilots_prepared_and_dryrun_valid"
        ),
        "campaign": campaign,
        "git_head": current_head,
        "pilot_jobs": len(rows),
        "pilot_events": sum(
            int(row["events"])
            for row in rows
        ),
        "families": [
            row["family"]
            for row in rows
        ],
        "eos_directory": eos_dir,
        "manifest": str(manifest_path),
        "submit_commands": str(commands_path),
        "production_scope": (
            "six 10k validation pilots only"
        ),
        "full_wave_a_authorized": False,
        "next_step": (
            "submit six pilots once and validate all receipts"
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
        "UNIFIED_5M_WAVEA_PILOTS_PREPARED_AND_DRYRUN_VALID"
    )
    print("pilot_jobs=6")
    print("pilot_events=60000")
    print(f"submit_dir={submit_dir}")
    print(f"manifest={manifest_path}")
    print(f"submit_commands={commands_path}")


if __name__ == "__main__":
    main()
