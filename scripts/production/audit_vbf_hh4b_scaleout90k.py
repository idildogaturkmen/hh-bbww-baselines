#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import math
import re
import shlex
import subprocess
import tarfile
import tempfile
from pathlib import Path
from typing import Any

import pandas as pd


EOS_HOST = "root://cmseos.fnal.gov"

CAMPAIGN = "vbf_hh4b_sm_scaleout90k_20260722_v1"
FAMILY = "vbf_hh4b_sm"
SCHEDD = "lpcschedd5.fnal.gov"
CLUSTER_ID = "59867184"

EVENTS_PER_SHARD = 10000
SPLIT_SALT = "vbf-hh4b-sm-scaleout-v1"

PAYLOAD_SHA256 = (
    "adfdc4ac14341e195c31fe6dcd9a2ef1a4dd1e7c21f7bd87ab1e86dda305726a"
)

PROCESS_CARD_SHA256 = (
    "a6f7ceb66905d566455dfdc45e4e689cfc6a17a15418c3e85794be5d4ccc8019"
)

DELPHES_CARD_SHA256 = (
    "1b2041245162de8360defdc502404e0696496c50773d4aa7f043798651a9517c"
)

REFERENCE_XSEC_PB = 0.00139031419


EXPECTED: list[dict[str, Any]] = [
    {
        "target_tag": (
            "vbf_hh4b_sm_run2_frozen_v2_"
            f"train_shard{shard_id}_10k"
        ),
        "shard_id": shard_id,
        "seed": 984000 + offset,
        "dataset_split": "train",
        "dataset_role": "canonical_train",
    }
    for offset, shard_id in enumerate(
        range(9401, 9409),
        start=1,
    )
]

EXPECTED.append({
    "target_tag": (
        "vbf_hh4b_sm_run2_frozen_v2_"
        "validation_shard9409_10k"
    ),
    "shard_id": 9409,
    "seed": 984009,
    "dataset_split": "validation",
    "dataset_role": "canonical_validation",
})


def load_module(
    path: Path,
    name: str,
) -> Any:
    specification = importlib.util.spec_from_file_location(
        name,
        path,
    )

    if (
        specification is None
        or specification.loader is None
    ):
        raise RuntimeError(
            f"could not load {path}"
        )

    module = importlib.util.module_from_spec(
        specification
    )

    specification.loader.exec_module(
        module
    )

    return module



def run_shell_output(
    arguments: list[str | Path],
) -> str:
    """Run an LPC shell wrapper through Bash."""

    command = shlex.join([
        str(argument)
        for argument in arguments
    ])

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
        "--outdir",
        type=Path,
        required=True,
    )

    arguments = parser.parse_args()

    repo = arguments.repo.resolve()
    store = arguments.store.resolve()
    outdir = arguments.outdir.resolve()

    pilot_module = load_module(
        repo
        / "scripts/production"
        / "audit_vbf_hh4b_pilot10k.py",
        "vbf_pilot_audit_helpers",
    )

    helpers = pilot_module.load_helpers(
        repo
    )

    require = helpers.require
    sha256_file = helpers.sha256_file
    adler32_file = helpers.adler32_file
    run_output = helpers.run_output
    find_unique = helpers.find_unique
    parse_lhe_events = helpers.parse_lhe_events
    run_card_value = helpers.run_card_value
    consistent_float = helpers.consistent_float

    safe_extract = pilot_module.safe_extract
    direct_hh4b_truth_count = (
        pilot_module.direct_hh4b_truth_count
    )
    read_event_xsec = (
        pilot_module.read_event_xsec
    )

    require(
        not outdir.exists(),
        f"refusing to overwrite {outdir}",
    )

    outdir.mkdir(parents=True)

    inspection_path = (
        repo
        / "outputs/agent_runs"
        / "vbf_hh4b_sm_scaleout90k_audit_input_inspection_20260723_v1"
        / "vbf_scaleout90k_receipts.json"
    )

    pilot_audit_path = (
        repo
        / "outputs/agent_runs"
        / "vbf_hh4b_sm_pilot10k_final_audit_20260722_v1"
        / "vbf_hh4b_pilot10k_final_audit.json"
    )

    submission_dir = (
        repo
        / "outputs/agent_runs"
        / "vbf_hh4b_sm_scaleout90k_submission_20260722_v1"
    )

    authorization_path = (
        submission_dir
        / "vbf_scaleout90k_authorization.json"
    )

    cluster_path = (
        submission_dir
        / "vbf_scaleout90k_cluster.tsv"
    )

    for path in (
        inspection_path,
        pilot_audit_path,
        authorization_path,
        cluster_path,
    ):
        require(
            path.is_file(),
            f"missing required input {path}",
        )

    inspection_rows = json.loads(
        inspection_path.read_text()
    )

    pilot_audit = json.loads(
        pilot_audit_path.read_text()
    )

    authorization = json.loads(
        authorization_path.read_text()
    )

    required_pilot_audit = {
        "status": "pass",
        "vbf_hh4b_pilot10k_final_audit_valid":
            True,
        "vbf_hh4b_scaleout90k_submission_authorized":
            True,
        "canonical_signal_target_events":
            100000,
        "validated_pilot_events":
            10000,
        "remaining_scaleout_events":
            90000,
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

    required_authorization = {
        "status": "prepared_and_dryrun_valid",
        "campaign": CAMPAIGN,
        "family": FAMILY,
        "jobs": 9,
        "events_per_job": 10000,
        "events_total": 90000,
        "train_events": 80000,
        "validation_events": 10000,
        "count_toward_signal_target": True,
        "canonical_signal_target_events":
            100000,
        "physics_yield_authorized": False,
        "dryrun_valid": True,
        "scaleout90k_submission_authorized":
            True,
    }

    for key, expected in (
        required_authorization.items()
    ):
        observed = authorization.get(key)

        require(
            observed == expected,
            (
                f"scale-out authorization mismatch "
                f"for {key}: "
                f"{observed!r} != {expected!r}"
            ),
        )

    with cluster_path.open(
        newline="",
    ) as handle:
        cluster_rows = list(
            csv.DictReader(
                handle,
                delimiter="\t",
            )
        )

    require(
        len(cluster_rows) == 1,
        "expected one VBF cluster record",
    )

    cluster_row = cluster_rows[0]

    require(
        cluster_row["campaign"] == CAMPAIGN,
        "cluster-record campaign mismatch",
    )

    require(
        cluster_row["family"] == FAMILY,
        "cluster-record family mismatch",
    )

    require(
        cluster_row["schedd"] == SCHEDD,
        "cluster-record schedd mismatch",
    )

    require(
        cluster_row["cluster_id"]
        == CLUSTER_ID,
        "cluster-record cluster mismatch",
    )

    require(
        int(cluster_row["jobs"]) == 9,
        "cluster-record job count mismatch",
    )

    require(
        int(cluster_row["events_total"])
        == 90000,
        "cluster-record event total mismatch",
    )

    require(
        len(inspection_rows) == 9,
        "expected nine inspected receipts",
    )

    receipts_by_tag = {
        row["target_tag"]: row
        for row in inspection_rows
    }

    expected_by_tag = {
        row["target_tag"]: row
        for row in EXPECTED
    }

    require(
        set(receipts_by_tag)
        == set(expected_by_tag),
        "VBF receipt target-tag set mismatch",
    )

    require(
        len(receipts_by_tag) == 9,
        "duplicate VBF receipt target tags",
    )

    require(
        len({
            int(row["shard_id"])
            for row in inspection_rows
        }) == 9,
        "duplicate VBF shard IDs",
    )

    require(
        len({
            int(row["seed"])
            for row in inspection_rows
        }) == 9,
        "duplicate VBF MG5 seeds",
    )

    require(
        len({
            int(row["pythia_seed"])
            for row in inspection_rows
        }) == 9,
        "duplicate VBF Pythia seeds",
    )

    require(
        len({
            str(row["remote_bundle"])
            for row in inspection_rows
        }) == 9,
        "duplicate VBF remote bundles",
    )

    active_output = run_shell_output([
        "condor_q",
        "-name",
        SCHEDD,
        CLUSTER_ID,
        "-af",
        "ProcId",
    ]).strip()

    require(
        active_output == "",
        (
            "VBF scale-out still has active jobs: "
            f"{active_output!r}"
        ),
    )

    history_output = run_shell_output([
        "condor_history",
        "-name",
        SCHEDD,
        CLUSTER_ID,
        "-limit",
        "20",
        "-af",
        "ProcId",
        "JobStatus",
        "ExitCode",
        "NumJobStarts",
    ])

    history_rows: dict[
        int,
        tuple[int, int, int],
    ] = {}

    for line in history_output.splitlines():
        fields = line.split()

        if len(fields) < 4:
            continue

        proc_id = int(fields[0])
        job_status = int(fields[1])
        exit_code = int(fields[2])
        starts = int(fields[3])

        history_rows[proc_id] = (
            job_status,
            exit_code,
            starts,
        )

    require(
        set(history_rows) == set(range(9)),
        (
            "VBF history process set mismatch: "
            f"{sorted(history_rows)}"
        ),
    )

    for proc_id, values in (
        history_rows.items()
    ):
        require(
            values == (4, 0, 1),
            (
                f"VBF process {proc_id} history "
                f"mismatch: {values}"
            ),
        )

    submit_file = (
        store
        / "condor_submit"
        / CAMPAIGN
        / f"{CAMPAIGN}.sub"
    )

    require(
        submit_file.is_file(),
        f"missing submit file {submit_file}",
    )

    submit_text = submit_file.read_text(
        errors="replace"
    )

    for fragment in (
        '+CountTowardSignalTarget = True',
        '+CanonicalSignalTargetEvents = 100000',
        '+PhysicsYieldAuthorized = False',
        '+DatasetRole = "$(dataset_role)"',
        '+DatasetSplit = "$(dataset_split)"',
        '+MatrixElementProcess = "p p -> h h j j, QCD=0"',
        '+HiggsDecayStrategy = "Pythia force H->bb"',
        SPLIT_SALT,
    ):
        require(
            fragment in submit_text,
            (
                "VBF submit metadata missing "
                f"{fragment}"
            ),
        )

    payload = (
        store
        / "condor_payloads"
        / "vbf_hh4b_sm_frozen_v2_20260722_v3"
        / "vbf_hh4b_sm_inputs.tar.gz"
    )

    require(
        payload.is_file(),
        f"missing payload {payload}",
    )

    require(
        sha256_file(payload)
        == PAYLOAD_SHA256,
        "local VBF payload hash mismatch",
    )

    audit_rows: list[
        dict[str, Any]
    ] = []

    for expected in sorted(
        EXPECTED,
        key=lambda row: int(
            row["shard_id"]
        ),
    ):
        target_tag = str(
            expected["target_tag"]
        )

        receipt = receipts_by_tag[
            target_tag
        ]

        seed = int(
            expected["seed"]
        )

        pythia_seed = (
            seed * 37 + 17
        ) % 900_000_000

        if pythia_seed == 0:
            pythia_seed = 1

        print(
            (
                "===== AUDITING VBF SHARD "
                f"{expected['shard_id']} ====="
            ),
            flush=True,
        )

        required_receipt = {
            "schema_version": 1,
            "campaign": CAMPAIGN,
            "family": FAMILY,
            "target_tag": target_tag,
            "shard_id":
                expected["shard_id"],
            "n_events":
                EVENTS_PER_SHARD,
            "seed": seed,
            "pythia_seed": pythia_seed,
            "dataset_split":
                expected["dataset_split"],
            "split_assignment_unit":
                "whole_shard",
            "split_salt":
                SPLIT_SALT,
            "run_card_profile":
                "preserve_template",
            "dataset_role":
                expected["dataset_role"],
            "cluster_id":
                CLUSTER_ID,
            "stage":
                "complete_copied_and_verified",
            "payload_sha256":
                PAYLOAD_SHA256,
            "expected_payload_sha256":
                PAYLOAD_SHA256,
            "delphes_card_sha256":
                DELPHES_CARD_SHA256,
            "lhe_events":
                EVENTS_PER_SHARD,
            "hepmc_events":
                EVENTS_PER_SHARD,
            "root_events":
                EVENTS_PER_SHARD,
            "exit_status": 0,
        }

        for key, value in (
            required_receipt.items()
        ):
            observed = receipt.get(key)

            require(
                observed == value,
                (
                    f"{target_tag}: receipt "
                    f"mismatch for {key}: "
                    f"{observed!r} != {value!r}"
                ),
            )

        remote_bundle = str(
            receipt["remote_bundle"]
        )

        require(
            remote_bundle.startswith(
                "/store/user/"
            ),
            (
                f"{target_tag}: invalid "
                "remote bundle"
            ),
        )

        with tempfile.TemporaryDirectory(
            prefix=(
                "vbf_scaleout90k_"
                f"{expected['shard_id']}_"
            ),
        ) as temporary:
            temporary_root = Path(
                temporary
            )

            local_bundle = (
                temporary_root
                / "bundle.tar.gz"
            )

            subprocess.run(
                [
                    "xrdcp",
                    "-f",
                    "--nopbar",
                    EOS_HOST
                    + "/"
                    + remote_bundle,
                    str(local_bundle),
                ],
                check=True,
            )

            local_sha256 = sha256_file(
                local_bundle
            )

            local_adler32 = adler32_file(
                local_bundle
            )

            local_size = (
                local_bundle.stat().st_size
            )

            require(
                local_sha256
                == receipt["bundle_sha256"],
                (
                    f"{target_tag}: bundle "
                    "SHA-256 mismatch"
                ),
            )

            require(
                local_adler32
                == str(
                    receipt["bundle_adler32"]
                ).lower(),
                (
                    f"{target_tag}: bundle "
                    "Adler-32 mismatch"
                ),
            )

            require(
                local_size
                == int(
                    receipt[
                        "bundle_size_bytes"
                    ]
                ),
                (
                    f"{target_tag}: bundle "
                    "size mismatch"
                ),
            )

            checksum_output = run_output([
                "xrdfs",
                EOS_HOST,
                "query",
                "checksum",
                remote_bundle,
            ])

            checksum_matches = re.findall(
                r"\b[0-9a-fA-F]{8}\b",
                checksum_output,
            )

            require(
                checksum_matches,
                (
                    f"{target_tag}: could not "
                    "parse EOS checksum"
                ),
            )

            require(
                checksum_matches[-1].lower()
                == local_adler32,
                (
                    f"{target_tag}: EOS/local "
                    "checksum mismatch"
                ),
            )

            stat_output = run_output([
                "xrdfs",
                EOS_HOST,
                "stat",
                remote_bundle,
            ])

            size_match = re.search(
                r"(?m)^Size:\s*([0-9]+)\s*$",
                stat_output,
            )

            require(
                size_match is not None,
                (
                    f"{target_tag}: could not "
                    "parse EOS size"
                ),
            )

            require(
                int(size_match.group(1))
                == local_size,
                (
                    f"{target_tag}: EOS/local "
                    "size mismatch"
                ),
            )

            extraction_dir = (
                temporary_root
                / "extracted"
            )

            extraction_dir.mkdir()

            with tarfile.open(
                local_bundle,
                "r:gz",
            ) as archive:
                safe_extract(
                    archive,
                    extraction_dir,
                    require,
                )

            proc_card = find_unique(
                extraction_dir,
                "proc_card_mg5.dat",
            )

            run_card = find_unique(
                extraction_dir,
                "run_card.dat",
            )

            root_file = find_unique(
                extraction_dir,
                "*_delphes.root",
            )

            event_summary = find_unique(
                extraction_dir,
                "*_event_summary.parquet",
            )

            candidates = find_unique(
                extraction_dir,
                "*_hh4b_candidates*.parquet",
            )

            source_lhe = find_unique(
                extraction_dir,
                "unweighted_events.lhe.gz",
            )

            banner = find_unique(
                extraction_dir,
                "generation_banner.txt",
            )

            provenance_path = find_unique(
                extraction_dir,
                "*_provenance.json",
            )

            require(
                sha256_file(proc_card)
                == PROCESS_CARD_SHA256,
                (
                    f"{target_tag}: process-card "
                    "hash mismatch"
                ),
            )

            for file_path, receipt_key in (
                (
                    root_file,
                    "root_sha256",
                ),
                (
                    event_summary,
                    "event_summary_sha256",
                ),
                (
                    candidates,
                    "candidate_sha256",
                ),
                (
                    source_lhe,
                    "source_lhe_sha256",
                ),
                (
                    banner,
                    "banner_sha256",
                ),
            ):
                require(
                    sha256_file(file_path)
                    == receipt[receipt_key],
                    (
                        f"{target_tag}: hash "
                        f"mismatch for {receipt_key}"
                    ),
                )

            proc_text = " ".join(
                proc_card.read_text(
                    errors="replace"
                ).split()
            )

            require(
                (
                    "generate p p > h h j j "
                    "QCD=0"
                )
                in proc_text,
                (
                    f"{target_tag}: expected "
                    "VBF HH process is missing"
                ),
            )

            run_text = run_card.read_text(
                errors="replace"
            )

            expected_run_values = {
                "nevents":
                    float(
                        EVENTS_PER_SHARD
                    ),
                "iseed": float(seed),
                "ebeam1": 6500.0,
                "ebeam2": 6500.0,
            }

            for key, expected_value in (
                expected_run_values.items()
            ):
                observed = run_card_value(
                    run_text,
                    key,
                )

                require(
                    math.isclose(
                        observed,
                        expected_value,
                        rel_tol=0.0,
                        abs_tol=1.0e-9,
                    ),
                    (
                        f"{target_tag}: run-card "
                        f"{key} mismatch: "
                        f"{observed} != "
                        f"{expected_value}"
                    ),
                )

            lhe_events = parse_lhe_events(
                source_lhe
            )

            require(
                len(lhe_events)
                == EVENTS_PER_SHARD,
                (
                    f"{target_tag}: LHE count "
                    f"{len(lhe_events)} != "
                    f"{EVENTS_PER_SHARD}"
                ),
            )

            stable_two_higgs = 0

            for event_index, particles in (
                enumerate(lhe_events)
            ):
                final_higgs_count = sum(
                    1
                    for pid, status
                    in particles
                    if (
                        pid == 25
                        and status == 1
                    )
                )

                require(
                    final_higgs_count == 2,
                    (
                        f"{target_tag}: LHE "
                        f"event {event_index} has "
                        f"{final_higgs_count} "
                        "stable Higgs bosons"
                    ),
                )

                stable_two_higgs += 1

            root_entries, truth_events = (
                direct_hh4b_truth_count(
                    root_file,
                    require,
                )
            )

            require(
                root_entries
                == EVENTS_PER_SHARD,
                (
                    f"{target_tag}: ROOT count "
                    f"{root_entries} != "
                    f"{EVENTS_PER_SHARD}"
                ),
            )

            require(
                truth_events
                == EVENTS_PER_SHARD,
                (
                    f"{target_tag}: forced "
                    "HH→4b truth mismatch: "
                    f"{truth_events}/"
                    f"{EVENTS_PER_SHARD}"
                ),
            )

            event_frame = pd.read_parquet(
                event_summary
            )

            candidate_frame = (
                pd.read_parquet(
                    candidates
                )
            )

            require(
                len(event_frame)
                == EVENTS_PER_SHARD,
                (
                    f"{target_tag}: "
                    "event-summary count mismatch"
                ),
            )

            require(
                len(candidate_frame)
                == int(
                    receipt[
                        "candidate_rows"
                    ]
                ),
                (
                    f"{target_tag}: "
                    "candidate-row mismatch"
                ),
            )

            xsec_column, summary_xsec = (
                read_event_xsec(
                    event_frame,
                    require,
                )
            )

            receipt_xsec = float(
                receipt[
                    "generator_xsec_pb"
                ]
            )

            provenance = json.loads(
                provenance_path.read_text()
            )

            provenance_xsec = float(
                provenance[
                    "generator_xsec_pb"
                ]
            )

            require(
                consistent_float(
                    receipt_xsec,
                    summary_xsec,
                ),
                (
                    f"{target_tag}: receipt/"
                    "event-summary xsec mismatch"
                ),
            )

            require(
                consistent_float(
                    receipt_xsec,
                    provenance_xsec,
                ),
                (
                    f"{target_tag}: receipt/"
                    "provenance xsec mismatch"
                ),
            )

            xsec_ratio = (
                receipt_xsec
                / REFERENCE_XSEC_PB
            )

            require(
                0.80
                <= xsec_ratio
                <= 1.20,
                (
                    f"{target_tag}: VBF xsec "
                    "outside frozen reference "
                    f"tolerance: {xsec_ratio}"
                ),
            )

            required_provenance = {
                "campaign": CAMPAIGN,
                "family": FAMILY,
                "n_events":
                    EVENTS_PER_SHARD,
                "seed": seed,
                "pythia_seed":
                    pythia_seed,
                "dataset_split":
                    expected[
                        "dataset_split"
                    ],
                "dataset_role":
                    expected[
                        "dataset_role"
                    ],
                "lhe_events":
                    EVENTS_PER_SHARD,
                "hepmc_events":
                    EVENTS_PER_SHARD,
                "root_events":
                    EVENTS_PER_SHARD,
            }

            for key, value in (
                required_provenance.items()
            ):
                observed = provenance.get(
                    key
                )

                require(
                    observed == value,
                    (
                        f"{target_tag}: "
                        "provenance mismatch "
                        f"for {key}: "
                        f"{observed!r} != "
                        f"{value!r}"
                    ),
                )

            audit_rows.append({
                "campaign": CAMPAIGN,
                "family": FAMILY,
                "cluster_id":
                    CLUSTER_ID,
                "target_tag":
                    target_tag,
                "shard_id":
                    expected["shard_id"],
                "seed": seed,
                "pythia_seed":
                    pythia_seed,
                "dataset_split":
                    expected[
                        "dataset_split"
                    ],
                "dataset_role":
                    expected[
                        "dataset_role"
                    ],
                "events":
                    EVENTS_PER_SHARD,
                "candidate_rows":
                    len(candidate_frame),
                "generator_xsec_pb":
                    receipt_xsec,
                "reference_xsec_pb":
                    REFERENCE_XSEC_PB,
                "xsec_to_reference_ratio":
                    xsec_ratio,
                "xsec_column":
                    xsec_column,
                "stable_two_higgs_lhe_events":
                    stable_two_higgs,
                "forced_hh4b_truth_events":
                    truth_events,
                "payload_sha256":
                    PAYLOAD_SHA256,
                "bundle_sha256":
                    local_sha256,
                "bundle_adler32":
                    local_adler32,
                "count_toward_signal_target":
                    True,
                "scaleout_final_audit_valid":
                    True,
                "physics_yield_authorized":
                    False,
            })

        print(
            f"{target_tag}: PASS",
            flush=True,
        )

    require(
        len(audit_rows) == 9,
        "expected nine audited VBF shards",
    )

    total_events = sum(
        int(row["events"])
        for row in audit_rows
    )

    train_events = sum(
        int(row["events"])
        for row in audit_rows
        if row["dataset_split"] == "train"
    )

    validation_events = sum(
        int(row["events"])
        for row in audit_rows
        if (
            row["dataset_split"]
            == "validation"
        )
    )

    truth_events = sum(
        int(
            row[
                "forced_hh4b_truth_events"
            ]
        )
        for row in audit_rows
    )

    require(
        total_events == 90000,
        "audited VBF scale-out total is not 90k",
    )

    require(
        train_events == 80000,
        "audited VBF train total is not 80k",
    )

    require(
        validation_events == 10000,
        (
            "audited VBF scale-out "
            "validation total is not 10k"
        ),
    )

    require(
        truth_events == 90000,
        (
            "audited VBF HH→4b truth "
            "total is not 90k"
        ),
    )

    xsecs = [
        float(
            row[
                "generator_xsec_pb"
            ]
        )
        for row in audit_rows
    ]

    summary = {
        "schema_version": 1,
        "status": "pass",
        "scaleout": audit_rows,
        "vbf_hh4b_scaleout90k_final_audit_valid":
            True,
        "validated_scaleout_events":
            90000,
        "validated_scaleout_train_events":
            80000,
        "validated_scaleout_validation_events":
            10000,
        "validated_scaleout_hh4b_truth_events":
            90000,
        "validated_pilot_events":
            10000,
        "canonical_signal_events_after_combination":
            100000,
        "canonical_signal_train_events_after_combination":
            80000,
        "canonical_signal_validation_events_after_combination":
            20000,
        "vbf_hh4b_canonical100k_registry_audit_authorized":
            True,
        "generator_xsec_min_pb":
            min(xsecs),
        "generator_xsec_max_pb":
            max(xsecs),
        "raw_generator_xsec_normalization_authorized":
            False,
        "physics_yield_authorized":
            False,
    }

    summary_path = (
        outdir
        / "vbf_hh4b_scaleout90k_final_audit.json"
    )

    summary_path.write_text(
        json.dumps(
            summary,
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )

    table_path = (
        outdir
        / "vbf_hh4b_scaleout90k_final_audit.tsv"
    )

    with table_path.open(
        "w",
        newline="",
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=list(
                audit_rows[0]
            ),
            delimiter="\t",
            lineterminator="\n",
        )

        writer.writeheader()
        writer.writerows(
            audit_rows
        )

    print()
    print(
        "VBF_HH4B_SCALEOUT90K_FINAL_AUDIT_VALID"
    )
    print(
        "VBF_HH4B_SCALEOUT_TRAIN80000_VALID"
    )
    print(
        "VBF_HH4B_SCALEOUT_VALIDATION10000_VALID"
    )
    print(
        "VBF_HH4B_SCALEOUT_HH4B_TRUTH90000_VALID"
    )
    print(
        "VBF_HH4B_CANONICAL100K_REGISTRY_AUDIT_AUTHORIZED"
    )
    print(
        "VBF_HH4B_SCALEOUT_EVENTS_VALIDATED=90000"
    )
    print(
        "NO_VBF_PHYSICS_YIELD_AUTHORIZATION_YET"
    )
    print(
        f"summary_json={summary_path}"
    )


if __name__ == "__main__":
    main()
