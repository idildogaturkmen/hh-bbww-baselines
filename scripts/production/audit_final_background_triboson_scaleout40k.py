#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import math
import re
import subprocess
import tarfile
import tempfile
from pathlib import Path
from typing import Any

import pandas as pd


EOS_HOST = "root://cmseos.fnal.gov"

CAMPAIGN = (
    "final_background_triboson_zbb_scaleout40k_20260722_v1"
)

CLUSTER_ID = "3615607"

INSPECTION_TAG = (
    "final_background_triboson_scaleout40k_"
    "audit_input_inspection_20260723_v1"
)

PILOT_AUDIT_TAG = (
    "final_background_triboson_pilot10k_"
    "final_audit_20260722_v1"
)

BUILD_TAG = (
    "final_background_triboson_worker_v2_build_20260722_v1"
)

SUBMISSION_TAG = (
    "final_background_triboson_zbb_scaleout40k_"
    "submission_20260722_v1"
)

CANARY_AUDIT_TAG = (
    "final_background_triboson_canary100_"
    "final_audit_20260722_v1"
)


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

    prepare_module = load_module(
        repo
        / "scripts/production"
        / "prepare_submit_triboson_zbb_scaleout40k.py",
        "triboson_scaleout_preparation",
    )

    pilot_module = load_module(
        repo
        / "scripts/production"
        / "audit_final_background_triboson_pilot10k.py",
        "triboson_pilot_audit",
    )

    triboson_helpers = pilot_module.load_module(
        repo
        / "scripts/production"
        / "audit_final_background_triboson_canaries.py",
        "triboson_canary_helpers",
    )

    hbb_helpers = triboson_helpers.load_helpers(
        repo
    )

    require = hbb_helpers.require
    sha256_file = hbb_helpers.sha256_file
    adler32_file = hbb_helpers.adler32_file
    run_output = hbb_helpers.run_output
    find_unique = hbb_helpers.find_unique
    run_card_value = hbb_helpers.run_card_value
    consistent_float = hbb_helpers.consistent_float

    safe_extract = triboson_helpers.safe_extract
    count_lhe_events = triboson_helpers.count_lhe_events
    direct_zbb_multiplicities = (
        triboson_helpers.direct_zbb_multiplicities
    )
    event_xsec = triboson_helpers.event_xsec
    process_configuration = triboson_helpers.EXPECTED

    expected_shards = prepare_module.SHARDS
    split_salt = prepare_module.SPLIT_SALT

    require(
        not outdir.exists(),
        f"refusing to overwrite {outdir}",
    )

    outdir.mkdir(parents=True)

    inspection_path = (
        repo
        / "outputs/agent_runs"
        / INSPECTION_TAG
        / "triboson_scaleout40k_receipts.json"
    )

    pilot_audit_path = (
        repo
        / "outputs/agent_runs"
        / PILOT_AUDIT_TAG
        / "triboson_pilot10k_final_audit.json"
    )

    build_path = (
        repo
        / "outputs/agent_runs"
        / BUILD_TAG
        / "triboson_worker_v2_build.json"
    )

    plan_path = (
        repo
        / "outputs/agent_runs"
        / SUBMISSION_TAG
        / "triboson_scaleout40k_plan.json"
    )

    cluster_path = (
        repo
        / "outputs/agent_runs"
        / SUBMISSION_TAG
        / "triboson_scaleout40k_cluster.tsv"
    )

    incident_path = (
        repo
        / "outputs/agent_runs"
        / SUBMISSION_TAG
        / "triboson_scaleout40k_header_row_incident.json"
    )

    canary_audit_path = (
        repo
        / "outputs/agent_runs"
        / CANARY_AUDIT_TAG
        / "triboson_canary100_final_audit.json"
    )

    for path in (
        inspection_path,
        pilot_audit_path,
        build_path,
        plan_path,
        cluster_path,
        incident_path,
        canary_audit_path,
    ):
        require(
            path.is_file(),
            f"missing required audit input {path}",
        )

    inspection = json.loads(
        inspection_path.read_text()
    )

    pilot_audit = json.loads(
        pilot_audit_path.read_text()
    )

    build = json.loads(
        build_path.read_text()
    )

    plan = json.loads(
        plan_path.read_text()
    )

    incident = json.loads(
        incident_path.read_text()
    )

    canary_audit = json.loads(
        canary_audit_path.read_text()
    )

    required_inspection = {
        "status": "audit_inputs_valid",
        "campaign": CAMPAIGN,
        "schedd": "lpcschedd4.fnal.gov",
        "cluster_id": int(CLUSTER_ID),
        "malformed_process_0_excluded": True,
        "malformed_process_0_counted_events": 0,
        "real_process_ids": list(range(1, 13)),
        "accepted_events_total": 40000,
        "accepted_events_by_family": {
            "wwz_zbb": 20800,
            "wzz_zbb": 13200,
            "zzz_zbb": 6000,
        },
        "events_validated_by_this_inspection": 0,
        "physics_yield_authorized": False,
    }

    for key, expected in required_inspection.items():
        observed = inspection.get(key)

        require(
            observed == expected,
            (
                f"inspection mismatch for {key}: "
                f"{observed!r} != {expected!r}"
            ),
        )

    required_pilot_audit = {
        "status": "pass",
        "triboson_pilot10k_final_audit_valid": True,
        "triboson_scaleout40k_submission_authorized": True,
        "triboson_pilot_events_validated": 10000,
        "triboson_scaleout_events_remaining": 40000,
        "validated_background_events_now": 4960000,
        "background_events_remaining": 40000,
        "final_background_target": 5000000,
        "physics_yield_authorized": False,
    }

    for key, expected in required_pilot_audit.items():
        observed = pilot_audit.get(key)

        require(
            observed == expected,
            (
                f"pilot audit mismatch for {key}: "
                f"{observed!r} != {expected!r}"
            ),
        )

    required_plan = {
        "status": "prepared_and_dryrun_valid",
        "campaign": CAMPAIGN,
        "jobs": 12,
        "accepted_events_total": 40000,
        "validated_pilot_events": 10000,
        "final_triboson_events": 50000,
        "validated_background_events_before_scaleout":
            4960000,
        "potential_validated_background_events_after_scaleout":
            5000000,
        "dataset_split": "train",
        "dataset_role": "canonical_train",
        "split_salt": split_salt,
        "count_toward_background_target": True,
        "scaleout40k_submission_authorized": True,
        "physics_yield_authorized": False,
    }

    for key, expected in required_plan.items():
        observed = plan.get(key)

        require(
            observed == expected,
            (
                f"scale-out plan mismatch for {key}: "
                f"{observed!r} != {expected!r}"
            ),
        )

    required_incident = {
        "status": "corrected_before_worker_execution",
        "campaign": CAMPAIGN,
        "schedd": "lpcschedd4.fnal.gov",
        "cluster_id": int(CLUSTER_ID),
        "intended_jobs": 12,
        "submitted_jobs_reported": 13,
        "malformed_process_id": 0,
        "malformed_process_num_job_starts": 0,
        "malformed_process_events_generated": 0,
        "malformed_process_counted_events": 0,
        "valid_process_ids": list(range(1, 13)),
        "production_events_validated_by_this_record": 0,
        "physics_yield_authorized": False,
    }

    for key, expected in required_incident.items():
        observed = incident.get(key)

        require(
            observed == expected,
            (
                f"incident mismatch for {key}: "
                f"{observed!r} != {expected!r}"
            ),
        )

    require(
        build.get("status") == "pass",
        "triboson worker build did not pass",
    )

    require(
        build.get("three_payloads_built") is True,
        "three triboson payloads were not built",
    )

    require(
        canary_audit.get(
            "triboson_canaries_final_audit_valid"
        )
        is True,
        "triboson canary audit did not pass",
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
        "expected one cluster record",
    )

    cluster_row = cluster_rows[0]

    require(
        cluster_row["campaign"] == CAMPAIGN,
        "cluster-record campaign mismatch",
    )

    require(
        cluster_row["schedd"]
        == "lpcschedd4.fnal.gov",
        "cluster-record schedd mismatch",
    )

    require(
        cluster_row["cluster_id"] == CLUSTER_ID,
        "cluster-record cluster mismatch",
    )

    require(
        int(cluster_row["jobs"]) == 12,
        "cluster-record job count mismatch",
    )

    require(
        int(
            cluster_row["accepted_events_total"]
        )
        == 40000,
        "cluster-record accepted total mismatch",
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
        '+DatasetRole = "canonical_train"',
        '+DatasetSplit = "train"',
        "+CountTowardBackgroundTarget = True",
        "+ExactTargetOutput = True",
        "+InclusivePythiaZDecays = True",
        "+AllZForcedToBB = False",
        '+FilterRequirement = "at least one direct Z to bb"',
        '+AllocationSemantics = "MC statistics proxy only"',
        "+PhysicsYieldAuthorized = False",
    ):
        require(
            fragment in submit_text,
            (
                "scale-out submit metadata missing "
                f"{fragment}"
            ),
        )

    require(
        len(expected_shards) == 12,
        "expected twelve scale-out shards",
    )

    expected_by_tag = {
        row["target_tag"]: row
        for row in expected_shards
    }

    inspection_rows = inspection[
        "receipts"
    ]

    require(
        len(inspection_rows) == 12,
        "inspection does not contain twelve receipts",
    )

    receipt_by_tag = {
        row["target_tag"]: row
        for row in inspection_rows
    }

    require(
        set(receipt_by_tag)
        == set(expected_by_tag),
        "expected/receipt membership mismatch",
    )

    build_by_family = {
        row["family"]: row
        for row in build["payloads"]
    }

    canary_by_family = {
        row["family"]: row
        for row in canary_audit["canaries"]
    }

    pilot_by_family = {
        row["family"]: row
        for row in pilot_audit["pilots"]
    }

    require(
        set(build_by_family)
        == {"wwz_zbb", "wzz_zbb", "zzz_zbb"},
        "payload family membership mismatch",
    )

    audit_rows: list[
        dict[str, Any]
    ] = []

    for expected in sorted(
        expected_shards,
        key=lambda row: int(
            row["shard_id"]
        ),
    ):
        family = str(
            expected["family"]
        )

        target_tag = str(
            expected["target_tag"]
        )

        receipt = receipt_by_tag[
            target_tag
        ]

        build_row = build_by_family[
            family
        ]

        process_row = process_configuration[
            family
        ]

        canary_row = canary_by_family[
            family
        ]

        pilot_row = pilot_by_family[
            family
        ]

        seed = int(
            expected["seed"]
        )

        pythia_seed = (
            seed * 37 + 17
        ) % 900_000_000

        if pythia_seed == 0:
            pythia_seed = 1

        accepted_events = int(
            expected["accepted_events"]
        )

        input_events = int(
            expected[
                "maximum_input_events"
            ]
        )

        processed_events = int(
            receipt[
                "processed_input_events"
            ]
        )

        rejected_events = int(
            receipt[
                "rejected_input_events"
            ]
        )

        print(
            (
                "===== AUDITING "
                f"{target_tag} ====="
            ),
            flush=True,
        )

        required_receipt = {
            "schema_version": 1,
            "campaign": CAMPAIGN,
            "family": family,
            "target_tag": target_tag,
            "shard_id":
                int(expected["shard_id"]),
            "n_events": accepted_events,
            "input_lhe_events_requested":
                input_events,
            "seed": seed,
            "pythia_seed": pythia_seed,
            "dataset_split": "train",
            "split_assignment_unit":
                "whole_shard",
            "split_salt": split_salt,
            "run_card_profile":
                "preserve_template",
            "dataset_role":
                "canonical_train",
            "cluster_id": CLUSTER_ID,
            "stage":
                "complete_copied_and_verified",
            "payload_sha256":
                build_row[
                    "payload_sha256"
                ],
            "expected_payload_sha256":
                build_row[
                    "payload_sha256"
                ],
            "lhe_events": input_events,
            "processed_input_events":
                processed_events,
            "rejected_input_events":
                rejected_events,
            "pythia_failures": 0,
            "accepted_events":
                accepted_events,
            "hepmc_events":
                accepted_events,
            "root_events":
                accepted_events,
            "filter_requirement":
                "at_least_one_direct_z_to_bb",
            "generator_xsec_semantics":
                "inclusive_before_zbb_filter",
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

        require(
            processed_events
            == rejected_events
            + accepted_events,
            (
                f"{target_tag}: filter "
                "accounting does not close"
            ),
        )

        require(
            accepted_events
            <= processed_events
            <= input_events,
            (
                f"{target_tag}: invalid "
                "processed input count"
            ),
        )

        payload = Path(
            build_row["payload"]
        )

        require(
            payload.is_file(),
            f"{family}: missing payload {payload}",
        )

        require(
            sha256_file(payload)
            == build_row[
                "payload_sha256"
            ],
            f"{family}: payload SHA mismatch",
        )

        remote_bundle = str(
            receipt["remote_bundle"]
        )

        require(
            remote_bundle.startswith(
                "/store/user/"
            ),
            f"{target_tag}: invalid EOS path",
        )

        with tempfile.TemporaryDirectory(
            prefix=(
                "triboson_scaleout40k_"
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

            local_sha = sha256_file(
                local_bundle
            )

            local_adler = adler32_file(
                local_bundle
            )

            local_size = (
                local_bundle.stat().st_size
            )

            require(
                local_sha
                == receipt[
                    "bundle_sha256"
                ],
                (
                    f"{target_tag}: bundle "
                    "SHA mismatch"
                ),
            )

            require(
                local_adler
                == str(
                    receipt[
                        "bundle_adler32"
                    ]
                ).lower(),
                (
                    f"{target_tag}: bundle "
                    "Adler mismatch"
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
                    f"{target_tag}: could "
                    "not parse EOS checksum"
                ),
            )

            require(
                checksum_matches[-1].lower()
                == local_adler,
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
                    f"{target_tag}: could "
                    "not parse EOS size"
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

            accounting_path = find_unique(
                extraction_dir,
                "*_filter_accounting.json",
            )

            require(
                sha256_file(proc_card)
                == process_row[
                    "proc_card_sha256"
                ],
                (
                    f"{target_tag}: process "
                    "card hash mismatch"
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
                (
                    accounting_path,
                    "filter_accounting_sha256",
                ),
            ):
                require(
                    sha256_file(file_path)
                    == receipt[
                        receipt_key
                    ],
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

            for fragment in process_row[
                "process_fragments"
            ]:
                require(
                    " ".join(
                        fragment.split()
                    )
                    in proc_text,
                    (
                        f"{target_tag}: missing "
                        f"process fragment {fragment}"
                    ),
                )

            run_text = run_card.read_text(
                errors="replace"
            )

            for key, value in {
                "nevents":
                    float(input_events),
                "iseed": float(seed),
                "ebeam1": 6500.0,
                "ebeam2": 6500.0,
            }.items():
                observed = run_card_value(
                    run_text,
                    key,
                )

                require(
                    math.isclose(
                        observed,
                        value,
                        rel_tol=0.0,
                        abs_tol=1.0e-9,
                    ),
                    (
                        f"{target_tag}: run-card "
                        f"{key} mismatch"
                    ),
                )

            lhe_count = count_lhe_events(
                source_lhe
            )

            require(
                lhe_count == input_events,
                (
                    f"{target_tag}: LHE count "
                    f"{lhe_count} != "
                    f"{input_events}"
                ),
            )

            multiplicities = (
                direct_zbb_multiplicities(
                    root_file
                )
            )

            require(
                len(multiplicities)
                == accepted_events,
                (
                    f"{target_tag}: ROOT event "
                    "count mismatch"
                ),
            )

            require(
                all(
                    value >= 1
                    for value in multiplicities
                ),
                (
                    f"{target_tag}: accepted "
                    "event without direct Z→bb"
                ),
            )

            require(
                max(multiplicities)
                <= process_row[
                    "maximum_zbb_multiplicity"
                ],
                (
                    f"{target_tag}: excessive "
                    "direct Z→bb multiplicity"
                ),
            )

            event_frame = pd.read_parquet(
                event_summary
            )

            candidate_frame = pd.read_parquet(
                candidates
            )

            require(
                len(event_frame)
                == accepted_events,
                (
                    f"{target_tag}: event "
                    "summary row mismatch"
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
                    f"{target_tag}: candidate "
                    "row mismatch"
                ),
            )

            accounting = json.loads(
                accounting_path.read_text()
            )

            required_accounting = {
                "campaign": CAMPAIGN,
                "family": family,
                "target_tag": target_tag,
                "requested_input_events":
                    input_events,
                "lhe_input_events":
                    input_events,
                "processed_input_events":
                    processed_events,
                "rejected_input_events":
                    rejected_events,
                "pythia_failures": 0,
                "accepted_events":
                    accepted_events,
                "pythia_seed":
                    pythia_seed,
                "filter_requirement":
                    "at_least_one_direct_z_to_bb",
                "all_z_forced_to_bb":
                    False,
                "inclusive_pythia_z_decays":
                    True,
                "generator_xsec_semantics":
                    "inclusive_before_zbb_filter",
                "physics_yield_authorized":
                    False,
            }

            for key, value in (
                required_accounting.items()
            ):
                observed = accounting.get(
                    key
                )

                require(
                    observed == value,
                    (
                        f"{target_tag}: filter "
                        "accounting mismatch for "
                        f"{key}: {observed!r} "
                        f"!= {value!r}"
                    ),
                )

            provenance = json.loads(
                provenance_path.read_text()
            )

            required_provenance = {
                "campaign": CAMPAIGN,
                "family": family,
                "target_tag": target_tag,
                "shard_id":
                    int(
                        expected[
                            "shard_id"
                        ]
                    ),
                "n_events":
                    accepted_events,
                "seed": seed,
                "pythia_seed":
                    pythia_seed,
                "dataset_split":
                    "train",
                "split_assignment_unit":
                    "whole_shard",
                "split_salt":
                    split_salt,
                "run_card_profile":
                    "preserve_template",
                "dataset_role":
                    "canonical_train",
                "lhe_events":
                    input_events,
                "hepmc_events":
                    accepted_events,
                "root_events":
                    accepted_events,
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
                        f"{observed!r} "
                        f"!= {value!r}"
                    ),
                )

            xsec_column, summary_xsec = (
                event_xsec(
                    event_frame
                )
            )

            receipt_xsec = float(
                receipt[
                    "generator_xsec_pb"
                ]
            )

            provenance_xsec = float(
                provenance[
                    "generator_xsec_pb"
                ]
            )

            accounting_xsec = float(
                accounting[
                    "generator_xsec_pb"
                ]
            )

            require(
                receipt_xsec > 0.0,
                (
                    f"{target_tag}: "
                    "nonpositive cross section"
                ),
            )

            for label, value in (
                (
                    "event-summary",
                    summary_xsec,
                ),
                (
                    "provenance",
                    provenance_xsec,
                ),
                (
                    "filter-accounting",
                    accounting_xsec,
                ),
            ):
                require(
                    consistent_float(
                        receipt_xsec,
                        value,
                    ),
                    (
                        f"{target_tag}: receipt/"
                        f"{label} xsec mismatch"
                    ),
                )

            canary_xsec = float(
                canary_row[
                    "generator_xsec_pb"
                ]
            )

            pilot_xsec = float(
                pilot_row[
                    "generator_xsec_pb"
                ]
            )

            canary_difference = abs(
                receipt_xsec
                - canary_xsec
            ) / canary_xsec

            pilot_difference = abs(
                receipt_xsec
                - pilot_xsec
            ) / pilot_xsec

            require(
                canary_difference < 0.15,
                (
                    f"{target_tag}: xsec differs "
                    "from canary by 15% or more"
                ),
            )

            require(
                pilot_difference < 0.15,
                (
                    f"{target_tag}: xsec differs "
                    "from pilot by 15% or more"
                ),
            )

            audit_rows.append({
                "family": family,
                "campaign": CAMPAIGN,
                "cluster_id":
                    CLUSTER_ID,
                "process_id":
                    int(
                        receipt[
                            "process_id"
                        ]
                    ),
                "target_tag":
                    target_tag,
                "shard_id":
                    int(
                        expected[
                            "shard_id"
                        ]
                    ),
                "seed": seed,
                "pythia_seed":
                    pythia_seed,
                "dataset_split":
                    "train",
                "dataset_role":
                    "canonical_train",
                "input_lhe_events":
                    input_events,
                "processed_input_events":
                    processed_events,
                "rejected_input_events":
                    rejected_events,
                "accepted_events":
                    accepted_events,
                "direct_zbb_truth_events":
                    accepted_events,
                "maximum_zbb_multiplicity":
                    max(multiplicities),
                "candidate_rows":
                    len(candidate_frame),
                "generator_xsec_pb":
                    receipt_xsec,
                "xsec_column":
                    xsec_column,
                "xsec_to_canary_relative_difference":
                    canary_difference,
                "xsec_to_pilot_relative_difference":
                    pilot_difference,
                "payload_sha256":
                    receipt[
                        "payload_sha256"
                    ],
                "bundle_sha256":
                    local_sha,
                "bundle_adler32":
                    local_adler,
                "count_toward_background_target":
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
        len(audit_rows) == 12,
        "expected twelve audited shards",
    )

    family_expected = {
        "wwz_zbb": 20800,
        "wzz_zbb": 13200,
        "zzz_zbb": 6000,
    }

    family_observed = {
        family: sum(
            int(
                row[
                    "accepted_events"
                ]
            )
            for row in audit_rows
            if row["family"] == family
        )
        for family in family_expected
    }

    require(
        family_observed
        == family_expected,
        (
            "audited family totals mismatch: "
            f"{family_observed}"
        ),
    )

    total_accepted = sum(
        int(
            row[
                "accepted_events"
            ]
        )
        for row in audit_rows
    )

    total_truth = sum(
        int(
            row[
                "direct_zbb_truth_events"
            ]
        )
        for row in audit_rows
    )

    require(
        total_accepted == 40000,
        "audited accepted total is not 40k",
    )

    require(
        total_truth == 40000,
        (
            "audited direct-Zbb truth total "
            "is not 40k"
        ),
    )

    summary = {
        "schema_version": 1,
        "status": "pass",
        "scaleout": audit_rows,
        "triboson_scaleout40k_final_audit_valid":
            True,
        "triboson_scaleout_events_validated":
            40000,
        "triboson_scaleout_events_by_family":
            family_observed,
        "triboson_scaleout_direct_zbb_truth_events":
            40000,
        "malformed_process_0_excluded":
            True,
        "malformed_process_0_counted_events":
            0,
        "validated_triboson_pilot_events":
            10000,
        "potential_canonical_triboson_events":
            50000,
        "triboson_final50k_registry_audit_authorized":
            True,
        "background_events_registry_validated_before":
            4960000,
        "background_events_registry_validated_now":
            4960000,
        "potential_background_events_after_unified_registry":
            5000000,
        "raw_generator_xsec_normalization_authorized":
            False,
        "physics_yield_authorized":
            False,
    }

    summary_path = (
        outdir
        / "triboson_scaleout40k_final_audit.json"
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
        / "triboson_scaleout40k_final_audit.tsv"
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
        "WWZ_ZBB_SCALEOUT20800_FINAL_AUDIT_VALID"
    )
    print(
        "WZZ_ZBB_SCALEOUT13200_FINAL_AUDIT_VALID"
    )
    print(
        "ZZZ_ZBB_SCALEOUT6000_FINAL_AUDIT_VALID"
    )
    print(
        "TRIBOSON_SCALEOUT40K_FINAL_AUDIT_VALID"
    )
    print(
        "TRIBOSON_SCALEOUT_DIRECT_ZBB_TRUTH40000_VALID"
    )
    print(
        "TRIBOSON_SCALEOUT_EVENTS_VALIDATED=40000"
    )
    print(
        "TRIBOSON_FINAL50K_REGISTRY_AUDIT_AUTHORIZED"
    )
    print(
        "BACKGROUND_EVENTS_REGISTRY_VALIDATED_NOW=4960000"
    )
    print(
        "NO_BACKGROUND_PHYSICS_YIELD_AUTHORIZATION_YET"
    )
    print(
        f"summary_json={summary_path}"
    )


if __name__ == "__main__":
    main()
