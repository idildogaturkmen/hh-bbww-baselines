#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
import gzip
import importlib.util
import json
import math
import re
import subprocess
import tarfile
import tempfile
from pathlib import Path
from typing import Any

import awkward as ak
import pandas as pd
import uproot


EOS_HOST = "root://cmseos.fnal.gov"

CAMPAIGN = "final_background_triboson_zbb_canary100_20260722_v1"
ACCEPTED_EVENTS = 100
INPUT_EVENTS = 1000
SPLIT_SALT = "final-background-triboson-zbb-canary-v1"

EXPECTED = {
    "wwz_zbb": {
        "target_tag":
            "wwz_zbb_run2_frozen_v2_canary_shard9501_100",
        "shard_id": 9501,
        "seed": 995001,
        "maximum_zbb_multiplicity": 1,
        "payload_sha256":
            "2d47cf5ff90d6a714eb3480240e04baf81faf0eec8fae856d535fee645f35558",
        "proc_card_sha256":
            "8c958881d07936f68f8f06ab13447d47c7ac9a4b7d0661648f3aae0855f517c7",
        "process_fragments": [
            "generate p p > w+ w- z QCD=0",
        ],
        "reference_xsec_pb": 0.093218173,
    },
    "wzz_zbb": {
        "target_tag":
            "wzz_zbb_run2_frozen_v2_canary_shard9502_100",
        "shard_id": 9502,
        "seed": 995002,
        "maximum_zbb_multiplicity": 2,
        "payload_sha256":
            "7c36dca0c5e8fc3706b02d6b850f35dc114be7c18444252e45312808b47ad052",
        "proc_card_sha256":
            "305cbd1338e31b595156d6ba48775044b0aa53ce609752a324963bbbb3d58ce1",
        "process_fragments": [
            "generate p p > w+ z z QCD=0 @1",
            "add process p p > w- z z QCD=0 @2",
        ],
        "reference_xsec_pb": 0.0296974367,
    },
    "zzz_zbb": {
        "target_tag":
            "zzz_zbb_run2_frozen_v2_canary_shard9503_100",
        "shard_id": 9503,
        "seed": 995003,
        "maximum_zbb_multiplicity": 3,
        "payload_sha256":
            "b5b703f63f2157291e90abf1025cdb0fd3821ebbac6bb2ca762c70a968951980",
        "proc_card_sha256":
            "8a197704f1cdfe86f2edc477ea36e69b74c881fb6845b33002be9f586612b457",
        "process_fragments": [
            "generate p p > z z z QCD=0",
        ],
        "reference_xsec_pb": 0.010104354,
    },
}


def load_helpers(repo: Path) -> Any:
    helper_path = (
        repo
        / "scripts/production"
        / "audit_final_background_hbb_canaries.py"
    )

    specification = importlib.util.spec_from_file_location(
        "production_audit_helpers",
        helper_path,
    )

    if (
        specification is None
        or specification.loader is None
    ):
        raise RuntimeError(
            f"could not load {helper_path}"
        )

    module = importlib.util.module_from_spec(
        specification
    )
    specification.loader.exec_module(module)

    return module


def safe_extract(
    archive: tarfile.TarFile,
    destination: Path,
    require: Any,
) -> None:
    resolved_destination = destination.resolve()

    for member in archive.getmembers():
        resolved_target = (
            destination
            / member.name
        ).resolve()

        require(
            resolved_target == resolved_destination
            or resolved_destination
            in resolved_target.parents,
            f"unsafe archive member {member.name}",
        )

    archive.extractall(destination)


def count_lhe_events(path: Path) -> int:
    count = 0

    with gzip.open(
        path,
        "rt",
        errors="replace",
    ) as handle:
        for line in handle:
            if line.strip() == "<event>":
                count += 1

    return count


def direct_zbb_multiplicities(
    root_path: Path,
) -> list[int]:
    multiplicities: list[int] = []

    with uproot.open(root_path) as source:
        tree = source["Delphes"]

        for arrays in tree.iterate(
            [
                "Particle.PID",
                "Particle.M1",
                "Particle.M2",
            ],
            step_size=500,
            library="ak",
        ):
            for raw_pid, raw_m1, raw_m2 in zip(
                arrays["Particle.PID"],
                arrays["Particle.M1"],
                arrays["Particle.M2"],
            ):
                pids = [
                    int(value)
                    for value in ak.to_list(raw_pid)
                ]

                mothers1 = [
                    int(value)
                    for value in ak.to_list(raw_m1)
                ]

                mothers2 = [
                    int(value)
                    for value in ak.to_list(raw_m2)
                ]

                zbb_count = 0

                for z_index, pid in enumerate(pids):
                    if pid != 23:
                        continue

                    direct_children = []

                    for child_index, child_pid in enumerate(pids):
                        if (
                            mothers1[child_index] == z_index
                            or mothers2[child_index] == z_index
                        ):
                            direct_children.append(
                                child_pid
                            )

                    if (
                        5 in direct_children
                        and -5 in direct_children
                    ):
                        zbb_count += 1

                multiplicities.append(
                    zbb_count
                )

    return multiplicities


def event_xsec(
    frame: pd.DataFrame,
) -> tuple[str, float]:
    for column in (
        "event_cross_section_pb",
        "generator_xsec_pb",
        "cross_section_pb",
    ):
        if column not in frame.columns:
            continue

        values = pd.to_numeric(
            frame[column],
            errors="coerce",
        ).dropna()

        if len(values) > 0:
            return column, float(
                values.median()
            )

    raise RuntimeError(
        "no recognized cross-section column"
    )


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

    args = parser.parse_args()

    repo = args.repo.resolve()
    store = args.store.resolve()
    outdir = args.outdir.resolve()

    helpers = load_helpers(repo)

    require = helpers.require
    sha256_file = helpers.sha256_file
    adler32_file = helpers.adler32_file
    run_output = helpers.run_output
    find_unique = helpers.find_unique
    run_card_value = helpers.run_card_value
    consistent_float = helpers.consistent_float

    require(
        not outdir.exists(),
        f"refusing to overwrite {outdir}",
    )

    outdir.mkdir(parents=True)

    build_path = (
        repo
        / "outputs/agent_runs"
        / "final_background_triboson_worker_v2_build_20260722_v1"
        / "triboson_worker_v2_build.json"
    )

    inspection_path = (
        repo
        / "outputs/agent_runs"
        / "final_background_triboson_canary100_audit_input_inspection_20260722_v1"
        / "triboson_canary_receipts.json"
    )

    require(
        build_path.is_file(),
        f"missing build summary {build_path}",
    )

    require(
        inspection_path.is_file(),
        f"missing receipt inspection {inspection_path}",
    )

    build = json.loads(
        build_path.read_text()
    )

    require(
        build.get("status") == "pass",
        "triboson worker build did not pass",
    )

    build_rows = {
        row["family"]: row
        for row in build["payloads"]
    }

    receipts = json.loads(
        inspection_path.read_text()
    )

    receipts_by_family = {
        row["family"]: row
        for row in receipts
    }

    require(
        set(build_rows) == set(EXPECTED),
        "build family set mismatch",
    )

    require(
        set(receipts_by_family) == set(EXPECTED),
        "receipt family set mismatch",
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
        "+CountTowardBackgroundTarget = False",
        "+ExactTargetOutput = True",
        "+InclusivePythiaZDecays = True",
        "+AllZForcedToBB = False",
        '+DatasetRole = "qa_canary"',
        "+PhysicsYieldAuthorized = False",
    ):
        require(
            fragment in submit_text,
            f"submit metadata missing {fragment}",
        )

    audit_rows: list[dict[str, Any]] = []

    with tempfile.TemporaryDirectory(
        prefix="triboson_canary_final_audit_",
    ) as temporary:
        temporary_root = Path(temporary)

        for family, expected in EXPECTED.items():
            print(
                f"===== AUDITING {family} CANARY =====",
                flush=True,
            )

            receipt = receipts_by_family[family]
            build_row = build_rows[family]

            expected_pythia_seed = (
                int(expected["seed"]) * 37 + 17
            ) % 900_000_000

            if expected_pythia_seed == 0:
                expected_pythia_seed = 1

            required_receipt = {
                "campaign": CAMPAIGN,
                "family": family,
                "target_tag":
                    expected["target_tag"],
                "shard_id":
                    expected["shard_id"],
                "n_events":
                    ACCEPTED_EVENTS,
                "input_lhe_events_requested":
                    INPUT_EVENTS,
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
                    "qa_canary",
                "stage":
                    "complete_copied_and_verified",
                "payload_sha256":
                    expected["payload_sha256"],
                "expected_payload_sha256":
                    expected["payload_sha256"],
                "lhe_events":
                    INPUT_EVENTS,
                "accepted_events":
                    ACCEPTED_EVENTS,
                "hepmc_events":
                    ACCEPTED_EVENTS,
                "root_events":
                    ACCEPTED_EVENTS,
                "pythia_failures": 0,
                "filter_requirement":
                    "at_least_one_direct_z_to_bb",
                "generator_xsec_semantics":
                    "inclusive_before_zbb_filter",
                "exit_status": 0,
            }

            for key, value in required_receipt.items():
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
                build_row["payload_sha256"]
                == expected["payload_sha256"],
                f"{family}: build payload mismatch",
            )

            payload = Path(
                build_row["payload"]
            )

            require(
                payload.is_file(),
                f"{family}: missing payload",
            )

            require(
                sha256_file(payload)
                == expected["payload_sha256"],
                f"{family}: local payload hash mismatch",
            )

            remote_bundle = str(
                receipt["remote_bundle"]
            )

            local_bundle = (
                temporary_root
                / f"{family}_bundle.tar.gz"
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

            bundle_sha = sha256_file(
                local_bundle
            )

            bundle_adler = adler32_file(
                local_bundle
            )

            require(
                bundle_sha
                == receipt["bundle_sha256"],
                f"{family}: bundle SHA mismatch",
            )

            require(
                bundle_adler
                == str(
                    receipt["bundle_adler32"]
                ).lower(),
                f"{family}: bundle Adler mismatch",
            )

            require(
                local_bundle.stat().st_size
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
                f"{family}: could not parse EOS checksum",
            )

            require(
                checksum_matches[-1].lower()
                == bundle_adler,
                f"{family}: EOS checksum mismatch",
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

            filter_accounting_path = find_unique(
                extraction_dir,
                "*_filter_accounting.json",
            )

            require(
                sha256_file(proc_card)
                == expected["proc_card_sha256"],
                f"{family}: process-card hash mismatch",
            )

            require(
                sha256_file(root_file)
                == receipt["root_sha256"],
                f"{family}: ROOT hash mismatch",
            )

            require(
                sha256_file(event_summary)
                == receipt["event_summary_sha256"],
                f"{family}: event-summary hash mismatch",
            )

            require(
                sha256_file(candidates)
                == receipt["candidate_sha256"],
                f"{family}: candidate hash mismatch",
            )

            require(
                sha256_file(source_lhe)
                == receipt["source_lhe_sha256"],
                f"{family}: LHE hash mismatch",
            )

            require(
                sha256_file(banner)
                == receipt["banner_sha256"],
                f"{family}: banner hash mismatch",
            )

            require(
                sha256_file(filter_accounting_path)
                == receipt["filter_accounting_sha256"],
                f"{family}: filter-accounting hash mismatch",
            )

            proc_text = " ".join(
                proc_card.read_text(
                    errors="replace"
                ).split()
            )

            for fragment in expected[
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
                "nevents": float(INPUT_EVENTS),
                "iseed": float(expected["seed"]),
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
                    f"{family}: run-card {key} mismatch",
                )

            lhe_count = count_lhe_events(
                source_lhe
            )

            require(
                lhe_count == INPUT_EVENTS,
                (
                    f"{family}: LHE count "
                    f"{lhe_count} != {INPUT_EVENTS}"
                ),
            )

            multiplicities = (
                direct_zbb_multiplicities(
                    root_file
                )
            )

            require(
                len(multiplicities)
                == ACCEPTED_EVENTS,
                f"{family}: ROOT event-count mismatch",
            )

            require(
                all(
                    value >= 1
                    for value in multiplicities
                ),
                (
                    f"{family}: accepted ROOT event "
                    "without direct Z→bb"
                ),
            )

            require(
                max(multiplicities)
                <= expected[
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
                == ACCEPTED_EVENTS,
                (
                    f"{family}: event-summary "
                    "row-count mismatch"
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

            filter_accounting = json.loads(
                filter_accounting_path.read_text()
            )

            required_filter_accounting = {
                "campaign": CAMPAIGN,
                "family": family,
                "target_tag":
                    expected["target_tag"],
                "requested_input_events":
                    INPUT_EVENTS,
                "lhe_input_events":
                    INPUT_EVENTS,
                "processed_input_events":
                    receipt["processed_input_events"],
                "rejected_input_events":
                    receipt["rejected_input_events"],
                "pythia_failures": 0,
                "accepted_events":
                    ACCEPTED_EVENTS,
                "maximum_zbb_multiplicity":
                    receipt[
                        "maximum_zbb_multiplicity"
                    ],
                "pythia_seed":
                    expected_pythia_seed,
                "filter_requirement":
                    "at_least_one_direct_z_to_bb",
                "all_z_forced_to_bb": False,
                "inclusive_pythia_z_decays": True,
                "generator_xsec_semantics":
                    "inclusive_before_zbb_filter",
                "physics_yield_authorized": False,
            }

            for key, value in (
                required_filter_accounting.items()
            ):
                observed = filter_accounting.get(
                    key
                )

                require(
                    observed == value,
                    (
                        f"{family}: filter-accounting "
                        f"mismatch for {key}"
                    ),
                )

            require(
                receipt["processed_input_events"]
                == receipt["rejected_input_events"]
                + receipt["accepted_events"],
                f"{family}: filter accounting does not close",
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
                    ACCEPTED_EVENTS,
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
                    "qa_canary",
                "lhe_events":
                    INPUT_EVENTS,
                "hepmc_events":
                    ACCEPTED_EVENTS,
                "root_events":
                    ACCEPTED_EVENTS,
            }

            for key, value in required_provenance.items():
                observed = provenance.get(key)

                require(
                    observed == value,
                    (
                        f"{family}: provenance "
                        f"mismatch for {key}: "
                        f"{observed!r} != {value!r}"
                    ),
                )

            xsec_column, summary_xsec = event_xsec(
                event_frame
            )

            receipt_xsec = float(
                receipt["generator_xsec_pb"]
            )

            provenance_xsec = float(
                provenance["generator_xsec_pb"]
            )

            filter_xsec = float(
                filter_accounting[
                    "generator_xsec_pb"
                ]
            )

            require(
                receipt_xsec > 0.0,
                f"{family}: nonpositive cross section",
            )

            require(
                consistent_float(
                    receipt_xsec,
                    summary_xsec,
                ),
                f"{family}: summary xsec mismatch",
            )

            require(
                consistent_float(
                    receipt_xsec,
                    provenance_xsec,
                ),
                f"{family}: provenance xsec mismatch",
            )

            require(
                consistent_float(
                    receipt_xsec,
                    filter_xsec,
                ),
                f"{family}: filter xsec mismatch",
            )

            relative_difference = abs(
                receipt_xsec
                - float(
                    expected["reference_xsec_pb"]
                )
            ) / float(
                expected["reference_xsec_pb"]
            )

            require(
                relative_difference < 0.15,
                (
                    f"{family}: xsec differs from "
                    "parton diagnostic by over 15%"
                ),
            )

            audit_rows.append({
                "family": family,
                "target_tag":
                    expected["target_tag"],
                "shard_id":
                    expected["shard_id"],
                "seed":
                    expected["seed"],
                "pythia_seed":
                    expected_pythia_seed,
                "input_lhe_events":
                    INPUT_EVENTS,
                "processed_input_events":
                    receipt["processed_input_events"],
                "rejected_input_events":
                    receipt["rejected_input_events"],
                "accepted_events":
                    ACCEPTED_EVENTS,
                "direct_zbb_truth_events":
                    ACCEPTED_EVENTS,
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
                    bundle_sha,
                "bundle_adler32":
                    bundle_adler,
                "count_toward_background_target":
                    False,
                "canary_final_audit_valid":
                    True,
                "physics_yield_authorized":
                    False,
            })

            print(
                f"{family}: PASS",
                flush=True,
            )

    require(
        len(audit_rows) == 3,
        "expected three audited canaries",
    )

    summary = {
        "schema_version": 1,
        "status": "pass",
        "canaries": audit_rows,
        "wwz_zbb_canary100_final_audit_valid":
            True,
        "wzz_zbb_canary100_final_audit_valid":
            True,
        "zzz_zbb_canary100_final_audit_valid":
            True,
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

    summary_path = (
        outdir
        / "triboson_canary100_final_audit.json"
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
        / "triboson_canary100_final_audit.tsv"
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
    print("WWZ_ZBB_CANARY100_FINAL_AUDIT_VALID")
    print("WZZ_ZBB_CANARY100_FINAL_AUDIT_VALID")
    print("ZZZ_ZBB_CANARY100_FINAL_AUDIT_VALID")
    print("TRIBOSON_CANARIES_FINAL_AUDIT_VALID")
    print("TRIBOSON_PILOT10K_SUBMISSION_AUTHORIZED")
    print("NO_TRIBOSON_SCALEOUT40K_SUBMISSION_AUTHORIZED_YET")
    print("TRIBOSON_CANARY_EVENTS_COUNTED=0")
    print("VALIDATED_BACKGROUND_EVENTS_NOW_4950000")
    print(f"summary_json={summary_path}")


if __name__ == "__main__":
    main()
