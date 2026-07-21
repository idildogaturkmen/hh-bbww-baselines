#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import re
import shlex
import subprocess


EXPECTED = {
    "tth_hbb": {
        "process_basename": "WaveB_ttH_Hbb_smoke100",
        "dataset_split": "train",
        "dataset_role": "physical_stratum",
    },
    "ttz_zbb": {
        "process_basename": "WaveB_ttZ_Zbb_smoke100",
        "dataset_split": "validation",
        "dataset_role": "physical_stratum",
    },
    "tttt": {
        "process_basename": "WaveB_fourtop_smoke100",
        "dataset_split": "train",
        "dataset_role": "physical_reference",
    },
    "vbf_hbb": {
        "process_basename": "WaveB_EW_Hjj_Hbb_smoke100",
        "dataset_split": "validation",
        "dataset_role": "physical_reference",
    },
}

REQUIRED_FIELDS = {
    "family",
    "process_dir",
    "shard_id",
    "events",
    "seed",
    "dataset_split",
    "run_card_profile",
    "dataset_role",
}


def command_output(command: list[str]) -> str:
    result = subprocess.run(
        command,
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


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument("--spec", required=True)
    parser.add_argument("--repo", required=True)
    parser.add_argument("--store", required=True)
    parser.add_argument("--campaign", required=True)
    parser.add_argument("--proxy", required=True)

    args = parser.parse_args()

    spec = Path(args.spec).resolve()
    repo = Path(args.repo).resolve()
    store = Path(args.store).resolve()
    proxy = Path(args.proxy).resolve()
    campaign = args.campaign

    if not re.fullmatch(
        r"[A-Za-z0-9][A-Za-z0-9_.-]+",
        campaign,
    ):
        raise SystemExit(
            "ERROR: invalid campaign name"
        )

    if not spec.is_file():
        raise SystemExit(
            f"ERROR: missing specification: {spec}"
        )

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
            "7200",
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

    builder = (
        repo
        / "scripts"
        / "production"
        / "prepare_mg5_frozen_v2_payload.sh"
    )

    worker = (
        repo
        / "scripts"
        / "production"
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

    with spec.open(newline="") as handle:
        reader = csv.DictReader(handle)

        if reader.fieldnames is None:
            raise SystemExit(
                "ERROR: canary specification has no header"
            )

        missing_fields = (
            REQUIRED_FIELDS
            - set(reader.fieldnames)
        )

        if missing_fields:
            raise SystemExit(
                "ERROR: specification is missing fields: "
                + ", ".join(sorted(missing_fields))
            )

        rows = list(reader)

    if len(rows) != 4:
        raise SystemExit(
            f"ERROR: expected four canaries, found {len(rows)}"
        )

    normalized = []
    seen_families = set()
    seen_seeds = set()
    seen_shards = set()

    for row in rows:
        family = row["family"]

        if family not in EXPECTED:
            raise SystemExit(
                f"ERROR: unexpected family: {family}"
            )

        if family in seen_families:
            raise SystemExit(
                f"ERROR: duplicate family: {family}"
            )

        expected = EXPECTED[family]
        process_dir = Path(
            row["process_dir"]
        ).resolve()

        shard_id = int(row["shard_id"])
        events = int(row["events"])
        seed = int(row["seed"])
        dataset_split = row["dataset_split"]
        profile = row["run_card_profile"]
        dataset_role = row["dataset_role"]

        if events != 100:
            raise SystemExit(
                f"ERROR: {family} does not request 100 events"
            )

        if seed <= 0:
            raise SystemExit(
                f"ERROR: invalid seed for {family}"
            )

        if shard_id <= 0:
            raise SystemExit(
                f"ERROR: invalid shard ID for {family}"
            )

        if seed in seen_seeds:
            raise SystemExit(
                f"ERROR: duplicate seed: {seed}"
            )

        if shard_id in seen_shards:
            raise SystemExit(
                f"ERROR: duplicate shard ID: {shard_id}"
            )

        if dataset_split == "test":
            raise SystemExit(
                "ERROR: final test data are not authorized"
            )

        if (
            dataset_split
            != expected["dataset_split"]
        ):
            raise SystemExit(
                f"ERROR: wrong split for {family}"
            )

        if profile != "preserve_template":
            raise SystemExit(
                f"ERROR: wrong run-card profile for {family}"
            )

        if (
            dataset_role
            != expected["dataset_role"]
        ):
            raise SystemExit(
                f"ERROR: wrong dataset role for {family}"
            )

        if (
            process_dir.name
            != expected["process_basename"]
        ):
            raise SystemExit(
                f"ERROR: wrong process directory for {family}"
            )

        if not (
            process_dir
            / "bin"
            / "generate_events"
        ).is_file():
            raise SystemExit(
                f"ERROR: missing MG5 executable for {family}"
            )

        if not (
            process_dir
            / "Cards"
            / "proc_card_mg5.dat"
        ).is_file():
            raise SystemExit(
                f"ERROR: missing process card for {family}"
            )

        if not (
            process_dir
            / "Cards"
            / "run_card.dat"
        ).is_file():
            raise SystemExit(
                f"ERROR: missing run card for {family}"
            )

        seen_families.add(family)
        seen_seeds.add(seed)
        seen_shards.add(shard_id)

        normalized.append({
            "family": family,
            "process_dir": process_dir,
            "shard_id": shard_id,
            "events": events,
            "seed": seed,
            "dataset_split": dataset_split,
            "run_card_profile": profile,
            "dataset_role": dataset_role,
        })

    if seen_families != set(EXPECTED):
        raise SystemExit(
            "ERROR: canary family set is incomplete"
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

    for row in normalized:
        family = row["family"]
        process_dir = row["process_dir"]
        shard_id = row["shard_id"]
        events = row["events"]
        seed = row["seed"]
        dataset_split = row["dataset_split"]
        profile = row["run_card_profile"]
        dataset_role = row["dataset_role"]

        target_tag = (
            f"{family}_run2_frozen_v2_"
            f"waveb_canary_shard{shard_id:04d}_"
            "100evt"
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
                    f"/uscms_data/d3/"
                    f"{os.environ['USER']}/"
                    "software/Delphes"
                ),
            },
        )

        payload_sha256 = sha256_file(
            payload
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

arguments = {campaign} {family} {target_tag} {shard_id} {events} {seed} {dataset_split} unified-background-5m-waveb-v1 {profile} {dataset_role} {payload.name} {eos_dir} {payload_sha256} $(Cluster)

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
            "target_tag": target_tag,
            "shard_id": shard_id,
            "events": events,
            "seed": seed,
            "dataset_split": dataset_split,
            "split_salt": (
                "unified-background-5m-waveb-v1"
            ),
            "run_card_profile": profile,
            "dataset_role": dataset_role,
            "process_dir": str(process_dir),
            "payload": str(payload),
            "payload_sha256": payload_sha256,
            "payload_size_bytes": (
                payload.stat().st_size
            ),
            "git_head": local_head,
            "eos_directory": eos_dir,
            "submit_file": str(submit_file),
            "dry_run": str(dry_run),
            "count_toward_5m": False,
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
            "waveb_priority_canaries_"
            "prepared_and_dryrun_valid"
        ),
        "campaign": campaign,
        "git_head": local_head,
        "canary_jobs": len(
            manifest_rows
        ),
        "canary_events": sum(
            row["events"]
            for row in manifest_rows
        ),
        "families": sorted(
            seen_families
        ),
        "eos_directory": eos_dir,
        "manifest": str(
            manifest_path
        ),
        "submit_commands": str(
            commands_path
        ),
        "count_toward_5m": False,
        "full_production_authorized": False,
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
        "WAVEB_PRIORITY_CANARIES_PREPARED_AND_DRYRUN_VALID"
    )
    print(
        f"campaign={campaign}"
    )
    print(
        f"git_head={local_head}"
    )
    print(
        f"canary_jobs={len(manifest_rows)}"
    )
    print(
        "canary_events="
        f"{sum(row['events'] for row in manifest_rows)}"
    )
    print(
        f"eos_directory={eos_dir}"
    )
    print(
        f"manifest={manifest_path}"
    )
    print(
        f"submit_commands={commands_path}"
    )

    for row in manifest_rows:
        print(
            "payload="
            f"{row['family']} "
            f"{row['payload_size_bytes']} "
            f"{row['payload_sha256']}"
        )


if __name__ == "__main__":
    main()
