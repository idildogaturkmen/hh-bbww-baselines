#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import math
import re
import shutil
import subprocess
import tarfile
import tempfile
from pathlib import Path
from typing import Any

import pandas as pd


EOS_HOST = "root://cmseos.fnal.gov"

CAMPAIGN = (
    "final_background_triboson_zbb_pilot10k_20260722_v1"
)

CLUSTER_ID = "84919658"

SPLIT_SALT = (
    "final-background-triboson-zbb-pilot-v1"
)

EXPECTED: dict[str, dict[str, Any]] = {
    "wwz_zbb": {
        "target_tag":
            "wwz_zbb_run2_frozen_v2_pilot_validation_shard9601_5200",
        "shard_id": 9601,
        "seed": 996101,
        "accepted_events": 5200,
        "input_events": 46800,
        "processed_input_events": 34622,
        "rejected_input_events": 29422,
        "final_family_target_events": 26000,
    },
    "wzz_zbb": {
        "target_tag":
            "wzz_zbb_run2_frozen_v2_pilot_validation_shard9602_3300",
        "shard_id": 9602,
        "seed": 996102,
        "accepted_events": 3300,
        "input_events": 19800,
        "processed_input_events": 11767,
        "rejected_input_events": 8467,
        "final_family_target_events": 16500,
    },
    "zzz_zbb": {
        "target_tag":
            "zzz_zbb_run2_frozen_v2_pilot_validation_shard9603_1500",
        "shard_id": 9603,
        "seed": 996103,
        "accepted_events": 1500,
        "input_events": 7500,
        "processed_input_events": 3995,
        "rejected_input_events": 2495,
        "final_family_target_events": 7500,
    },
}


def load_module(
    path: Path,
    name: str,
) -> Any:
    specification = (
        importlib.util.spec_from_file_location(
            name,
            path,
        )
    )

    if (
        specification is None
        or specification.loader is None
    ):
        raise RuntimeError(
            f"could not load {path}"
        )

    module = (
        importlib.util.module_from_spec(
            specification
        )
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

    triboson_helpers = load_module(
        repo
        / "scripts/production"
        / "audit_final_background_triboson_canaries.py",
        "triboson_canary_audit_helpers",
    )

    hbb_helpers = (
        triboson_helpers.load_helpers(
            repo
        )
    )

    require = hbb_helpers.require
    sha256_file = hbb_helpers.sha256_file
    adler32_file = hbb_helpers.adler32_file
    run_output = hbb_helpers.run_output
    find_unique = hbb_helpers.find_unique
    run_card_value = hbb_helpers.run_card_value
    consistent_float = hbb_helpers.consistent_float

    safe_extract = (
        triboson_helpers.safe_extract
    )

    count_lhe_events = (
        triboson_helpers.count_lhe_events
    )

    direct_zbb_multiplicities = (
        triboson_helpers.direct_zbb_multiplicities
    )

    event_xsec = (
        triboson_helpers.event_xsec
    )

    process_configuration = (
        triboson_helpers.EXPECTED
    )

    require(
        not outdir.exists(),
        f"refusing to overwrite {outdir}",
    )

    outdir.mkdir(parents=True)

    canary_audit_path = (
        repo
        / "outputs/agent_runs"
        / "final_background_triboson_canary100_final_audit_20260722_v1"
        / "triboson_canary100_final_audit.json"
    )

    build_path = (
        repo
        / "outputs/agent_runs"
        / "final_background_triboson_worker_v2_build_20260722_v1"
        / "triboson_worker_v2_build.json"
    )

    allocation_path = (
        repo
        / "outputs/agent_runs"
        / "final_background_triboson_zbb_pilot10k_submission_20260722_v1"
        / "triboson_final50k_allocation_freeze.json"
    )

    for path in (
        canary_audit_path,
        build_path,
        allocation_path,
    ):
        require(
            path.is_file(),
            f"missing required input {path}",
        )

    canary_audit = json.loads(
        canary_audit_path.read_text()
    )

    build = json.loads(
        build_path.read_text()
    )

    allocation = json.loads(
        allocation_path.read_text()
    )

    require(
        canary_audit.get(
            "triboson_canaries_final_audit_valid"
        )
        is True,
        "triboson canary final audit is not valid",
    )

    require(
        canary_audit.get(
            "triboson_pilot10k_submission_authorized"
        )
        is True,
        "triboson pilot submission was not authorized",
    )

    require(
        build.get("status") == "pass",
        "triboson worker build did not pass",
    )

    require(
        allocation.get("status") == "frozen",
        "triboson allocation is not frozen",
    )

    require(
        allocation.get("final_total_events")
        == 50000,
        "final allocation is not 50k",
    )

    require(
        allocation.get("pilot_total_events")
        == 10000,
        "pilot allocation is not 10k",
    )

    require(
        allocation.get(
            "normalization_use_authorized"
        )
        is False,
        "allocation was incorrectly authorized for normalization",
    )

    build_rows = {
        row["family"]: row
        for row in build["payloads"]
    }

    canary_rows = {
        row["family"]: row
        for row in canary_audit["canaries"]
    }

    require(
        set(EXPECTED) == set(build_rows),
        "pilot/build family sets differ",
    )

    require(
        set(EXPECTED) == set(canary_rows),
        "pilot/canary family sets differ",
    )

    require(
        len({
            row["target_tag"]
            for row in EXPECTED.values()
        }) == 3,
        "duplicate expected pilot target tags",
    )

    require(
        len({
            row["shard_id"]
            for row in EXPECTED.values()
        }) == 3,
        "duplicate expected pilot shard IDs",
    )

    require(
        len({
            row["seed"]
            for row in EXPECTED.values()
        }) == 3,
        "duplicate expected pilot seeds",
    )

    require(
        sum(
            int(row["accepted_events"])
            for row in EXPECTED.values()
        ) == 10000,
        "pilot accepted total is not 10k",
    )

    require(
        sum(
            int(
                row[
                    "final_family_target_events"
                ]
            )
            for row in EXPECTED.values()
        ) == 50000,
        "final family targets do not sum to 50k",
    )

    receipt_dir = (
        store
        / "condor_return"
        / CAMPAIGN
        / "receipts"
    )

    receipt_paths = sorted(
        receipt_dir.glob("*.json")
    )

    require(
        len(receipt_paths) == 3,
        (
            "expected three pilot receipts, "
            f"found {len(receipt_paths)}"
        ),
    )

    receipts = [
        json.loads(path.read_text())
        for path in receipt_paths
    ]

    receipts_by_family = {
        receipt["family"]: receipt
        for receipt in receipts
    }

    require(
        set(receipts_by_family)
        == set(EXPECTED),
        "pilot receipt family set mismatch",
    )

    require(
        len(receipts_by_family)
        == len(receipts),
        "duplicate pilot receipt families",
    )

    submit_file = (
        store
        / "condor_submit"
        / CAMPAIGN
        / f"{CAMPAIGN}.sub"
    )

    require(
        submit_file.is_file(),
        f"missing pilot submit file {submit_file}",
    )

    submit_text = submit_file.read_text(
        errors="replace"
    )

    for fragment in (
        "+CountTowardBackgroundTarget = True",
        '+DatasetRole = "pilot_validation"',
        '+DatasetSplit = "validation"',
        "+ExactTargetOutput = True",
        "+InclusivePythiaZDecays = True",
        "+AllZForcedToBB = False",
        "+PhysicsYieldAuthorized = False",
    ):
        require(
            fragment in submit_text,
            (
                "pilot submit metadata missing "
                f"{fragment}"
            ),
        )

    audit_rows: list[dict[str, Any]] = []

    with tempfile.TemporaryDirectory(
        prefix="triboson_pilot10k_audit_",
    ) as temporary:
        temporary_root = Path(temporary)

        for family in sorted(EXPECTED):
            expected = EXPECTED[family]
            receipt = receipts_by_family[family]
            build_row = build_rows[family]
            process_row = (
                process_configuration[family]
            )
            canary_row = canary_rows[family]

            print(
                f"===== AUDITING {family} PILOT =====",
                flush=True,
            )

            expected_pythia_seed = (
                int(expected["seed"]) * 37 + 17
            ) % 900_000_000

            if expected_pythia_seed == 0:
                expected_pythia_seed = 1

            required_receipt = {
                "schema_version": 1,
                "campaign": CAMPAIGN,
                "family": family,
                "target_tag":
                    expected["target_tag"],
                "shard_id":
                    expected["shard_id"],
                "n_events":
                    expected["accepted_events"],
                "input_lhe_events_requested":
                    expected["input_events"],
                "seed":
                    expected["seed"],
                "pythia_seed":
                    expected_pythia_seed,
                "dataset_split":
                    "validation",
                "split_assignment_unit":
                    "whole_shard",
                "split_salt":
                    SPLIT_SALT,
                "run_card_profile":
                    "preserve_template",
                "dataset_role":
                    "pilot_validation",
                "cluster_id":
                    CLUSTER_ID,
                "stage":
                    "complete_copied_and_verified",
                "payload_sha256":
                    build_row["payload_sha256"],
                "expected_payload_sha256":
                    build_row["payload_sha256"],
                "lhe_events":
                    expected["input_events"],
                "processed_input_events":
                    expected[
                        "processed_input_events"
                    ],
                "rejected_input_events":
                    expected[
                        "rejected_input_events"
                    ],
                "pythia_failures": 0,
                "accepted_events":
                    expected["accepted_events"],
                "hepmc_events":
                    expected["accepted_events"],
                "root_events":
                    expected["accepted_events"],
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
                        f"{family}: receipt mismatch "
                        f"for {key}: "
                        f"{observed!r} != {value!r}"
                    ),
                )

            require(
                receipt[
                    "processed_input_events"
                ]
                == receipt[
                    "rejected_input_events"
                ]
                + receipt["accepted_events"],
                (
                    f"{family}: filter accounting "
                    "does not close"
                ),
            )

            require(
                receipt[
                    "processed_input_events"
                ]
                <= receipt[
                    "input_lhe_events_requested"
                ],
                (
                    f"{family}: processed more input "
                    "events than requested"
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
                == build_row["payload_sha256"],
                f"{family}: payload SHA mismatch",
            )

            remote_bundle = str(
                receipt["remote_bundle"]
            )

            require(
                remote_bundle.startswith(
                    "/store/user/"
                ),
                f"{family}: invalid remote bundle",
            )

            local_bundle = (
                temporary_root
                / f"{family}_pilot_bundle.tar.gz"
            )

            subprocess.run(
                [
                    "xrdcp",
                    "-f",
                    "--nopbar",
                    EOS_HOST + "/" + remote_bundle,
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
                == receipt["bundle_sha256"],
                f"{family}: bundle SHA mismatch",
            )

            require(
                local_adler
                == str(
                    receipt["bundle_adler32"]
                ).lower(),
                f"{family}: bundle Adler mismatch",
            )

            require(
                local_size
                == int(
                    receipt["bundle_size_bytes"]
                ),
                f"{family}: bundle size mismatch",
            )

            checksum_output = run_output(
                [
                    "xrdfs",
                    EOS_HOST,
                    "query",
                    "checksum",
                    remote_bundle,
                ]
            )

            checksum_matches = re.findall(
                r"\b[0-9a-fA-F]{8}\b",
                checksum_output,
            )

            require(
                checksum_matches,
                (
                    f"{family}: could not parse "
                    "EOS checksum"
                ),
            )

            require(
                checksum_matches[-1].lower()
                == local_adler,
                (
                    f"{family}: EOS/local "
                    "checksum mismatch"
                ),
            )

            stat_output = run_output(
                [
                    "xrdfs",
                    EOS_HOST,
                    "stat",
                    remote_bundle,
                ]
            )

            size_match = re.search(
                r"(?m)^Size:\s*([0-9]+)\s*$",
                stat_output,
            )

            require(
                size_match is not None,
                (
                    f"{family}: could not parse "
                    "EOS bundle size"
                ),
            )

            require(
                int(size_match.group(1))
                == local_size,
                (
                    f"{family}: EOS/local "
                    "bundle size mismatch"
                ),
            )

            extraction_dir = (
                temporary_root
                / family
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
                    f"{family}: process-card "
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
                (
                    accounting_path,
                    "filter_accounting_sha256",
                ),
            ):
                require(
                    sha256_file(file_path)
                    == receipt[receipt_key],
                    (
                        f"{family}: hash mismatch "
                        f"for {receipt_key}"
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
                    ) in proc_text,
                    (
                        f"{family}: missing process "
                        f"fragment {fragment}"
                    ),
                )

            run_text = run_card.read_text(
                errors="replace"
            )

            for key, value in {
                "nevents":
                    float(
                        expected["input_events"]
                    ),
                "iseed":
                    float(expected["seed"]),
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
                        f"{family}: run-card "
                        f"{key} mismatch"
                    ),
                )

            lhe_count = count_lhe_events(
                source_lhe
            )

            require(
                lhe_count
                == expected["input_events"],
                (
                    f"{family}: LHE count "
                    f"{lhe_count} != "
                    f"{expected['input_events']}"
                ),
            )

            multiplicities = (
                direct_zbb_multiplicities(
                    root_file
                )
            )

            require(
                len(multiplicities)
                == expected["accepted_events"],
                (
                    f"{family}: ROOT event "
                    "count mismatch"
                ),
            )

            require(
                all(
                    value >= 1
                    for value in multiplicities
                ),
                (
                    f"{family}: accepted event "
                    "without direct Z→bb"
                ),
            )

            require(
                max(multiplicities)
                <= process_row[
                    "maximum_zbb_multiplicity"
                ],
                (
                    f"{family}: excessive Zbb "
                    "multiplicity"
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
                == expected["accepted_events"],
                (
                    f"{family}: event-summary "
                    "row count mismatch"
                ),
            )

            require(
                len(candidate_frame)
                == int(
                    receipt["candidate_rows"]
                ),
                (
                    f"{family}: candidate-row "
                    "count mismatch"
                ),
            )

            accounting = json.loads(
                accounting_path.read_text()
            )

            required_accounting = {
                "campaign": CAMPAIGN,
                "family": family,
                "target_tag":
                    expected["target_tag"],
                "requested_input_events":
                    expected["input_events"],
                "lhe_input_events":
                    expected["input_events"],
                "processed_input_events":
                    expected[
                        "processed_input_events"
                    ],
                "rejected_input_events":
                    expected[
                        "rejected_input_events"
                    ],
                "pythia_failures": 0,
                "accepted_events":
                    expected["accepted_events"],
                "pythia_seed":
                    expected_pythia_seed,
                "filter_requirement":
                    "at_least_one_direct_z_to_bb",
                "all_z_forced_to_bb": False,
                "inclusive_pythia_z_decays": True,
                "generator_xsec_semantics":
                    "inclusive_before_zbb_filter",
                "physics_yield_authorized":
                    False,
            }

            for key, value in (
                required_accounting.items()
            ):
                require(
                    accounting.get(key) == value,
                    (
                        f"{family}: filter-accounting "
                        f"mismatch for {key}"
                    ),
                )

            provenance = json.loads(
                provenance_path.read_text()
            )

            required_provenance = {
                "campaign": CAMPAIGN,
                "family": family,
                "target_tag":
                    expected["target_tag"],
                "shard_id":
                    expected["shard_id"],
                "n_events":
                    expected["accepted_events"],
                "seed":
                    expected["seed"],
                "pythia_seed":
                    expected_pythia_seed,
                "dataset_split":
                    "validation",
                "split_assignment_unit":
                    "whole_shard",
                "split_salt":
                    SPLIT_SALT,
                "run_card_profile":
                    "preserve_template",
                "dataset_role":
                    "pilot_validation",
                "lhe_events":
                    expected["input_events"],
                "hepmc_events":
                    expected["accepted_events"],
                "root_events":
                    expected["accepted_events"],
            }

            for key, value in (
                required_provenance.items()
            ):
                require(
                    provenance.get(key) == value,
                    (
                        f"{family}: provenance "
                        f"mismatch for {key}: "
                        f"{provenance.get(key)!r} "
                        f"!= {value!r}"
                    ),
                )

            xsec_column, summary_xsec = (
                event_xsec(
                    event_frame
                )
            )

            receipt_xsec = float(
                receipt["generator_xsec_pb"]
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
                f"{family}: nonpositive xsec",
            )

            require(
                consistent_float(
                    receipt_xsec,
                    summary_xsec,
                ),
                (
                    f"{family}: receipt/summary "
                    "xsec mismatch"
                ),
            )

            require(
                consistent_float(
                    receipt_xsec,
                    provenance_xsec,
                ),
                (
                    f"{family}: receipt/provenance "
                    "xsec mismatch"
                ),
            )

            require(
                consistent_float(
                    receipt_xsec,
                    accounting_xsec,
                ),
                (
                    f"{family}: receipt/accounting "
                    "xsec mismatch"
                ),
            )

            canary_xsec = float(
                canary_row[
                    "generator_xsec_pb"
                ]
            )

            relative_difference = abs(
                receipt_xsec - canary_xsec
            ) / canary_xsec

            require(
                relative_difference < 0.15,
                (
                    f"{family}: pilot xsec differs "
                    "from audited canary by 15% or more"
                ),
            )

            audit_rows.append({
                "family": family,
                "campaign": CAMPAIGN,
                "cluster_id": CLUSTER_ID,
                "target_tag":
                    expected["target_tag"],
                "shard_id":
                    expected["shard_id"],
                "seed":
                    expected["seed"],
                "pythia_seed":
                    expected_pythia_seed,
                "dataset_split":
                    "validation",
                "dataset_role":
                    "pilot_validation",
                "input_lhe_events":
                    expected["input_events"],
                "processed_input_events":
                    expected[
                        "processed_input_events"
                    ],
                "rejected_input_events":
                    expected[
                        "rejected_input_events"
                    ],
                "accepted_events":
                    expected["accepted_events"],
                "direct_zbb_truth_events":
                    expected["accepted_events"],
                "maximum_zbb_multiplicity":
                    max(multiplicities),
                "candidate_rows":
                    len(candidate_frame),
                "generator_xsec_pb":
                    receipt_xsec,
                "xsec_column":
                    xsec_column,
                "payload_sha256":
                    receipt["payload_sha256"],
                "bundle_sha256":
                    local_sha,
                "bundle_adler32":
                    local_adler,
                "final_family_target_events":
                    expected[
                        "final_family_target_events"
                    ],
                "count_toward_background_target":
                    True,
                "pilot_final_audit_valid":
                    True,
                "physics_yield_authorized":
                    False,
            })

            local_bundle.unlink()

            shutil.rmtree(
                extraction_dir
            )

            print(
                f"{family}: PASS",
                flush=True,
            )

    require(
        len(audit_rows) == 3,
        "expected three audited pilot shards",
    )

    require(
        sum(
            int(row["accepted_events"])
            for row in audit_rows
        ) == 10000,
        "audited pilot total is not 10k",
    )

    remaining_rows = []

    for row in audit_rows:
        remaining = (
            int(
                row[
                    "final_family_target_events"
                ]
            )
            - int(row["accepted_events"])
        )

        require(
            remaining > 0,
            (
                f"{row['family']}: nonpositive "
                "remaining target"
            ),
        )

        remaining_rows.append({
            "family": row["family"],
            "final_family_target_events":
                row[
                    "final_family_target_events"
                ],
            "validated_pilot_events":
                row["accepted_events"],
            "remaining_scaleout_events":
                remaining,
        })

    require(
        sum(
            int(
                row[
                    "remaining_scaleout_events"
                ]
            )
            for row in remaining_rows
        ) == 40000,
        "remaining triboson target is not 40k",
    )

    summary = {
        "schema_version": 1,
        "status": "pass",
        "pilots": audit_rows,
        "remaining_allocation":
            remaining_rows,
        "triboson_pilot10k_final_audit_valid":
            True,
        "triboson_scaleout40k_submission_authorized":
            True,
        "triboson_pilot_events_validated":
            10000,
        "triboson_scaleout_events_remaining":
            40000,
        "validated_background_events_before_pilot":
            4950000,
        "validated_background_events_now":
            4960000,
        "background_events_remaining":
            40000,
        "final_background_target":
            5000000,
        "physics_yield_authorized":
            False,
    }

    summary_path = (
        outdir
        / "triboson_pilot10k_final_audit.json"
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
        / "triboson_pilot10k_final_audit.tsv"
    )

    with table_path.open(
        "w",
        newline="",
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=list(audit_rows[0]),
            delimiter="\t",
            lineterminator="\n",
        )

        writer.writeheader()
        writer.writerows(audit_rows)

    print()
    print("WWZ_ZBB_PILOT5200_FINAL_AUDIT_VALID")
    print("WZZ_ZBB_PILOT3300_FINAL_AUDIT_VALID")
    print("ZZZ_ZBB_PILOT1500_FINAL_AUDIT_VALID")
    print("TRIBOSON_PILOT10K_FINAL_AUDIT_VALID")
    print("TRIBOSON_SCALEOUT40K_SUBMISSION_AUTHORIZED")
    print("VALIDATED_BACKGROUND_EVENTS_NOW_4960000")
    print("BACKGROUND_EVENTS_REMAINING_TRIBOSON_40000")
    print("NO_BACKGROUND_PHYSICS_YIELD_AUTHORIZATION_YET")
    print(f"summary_json={summary_path}")


if __name__ == "__main__":
    main()
