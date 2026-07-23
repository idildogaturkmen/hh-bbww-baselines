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
EVENTS_PER_SHARD = 10000
SPLIT_SALT = "final-background-hbb-scaleout-v1"

CAMPAIGNS = {
    "ggh_hbb": {
        "campaign": "ggh_hbb_scaleout90k_20260722_v1",
        "cluster_id": "84919107",
        "expected_events": 90000,
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
        "campaign": "bbh_hbb_4fs_scaleout40k_20260722_v1",
        "cluster_id": "84919108",
        "expected_events": 40000,
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


def load_module(
    path: Path,
    module_name: str,
) -> Any:
    specification = importlib.util.spec_from_file_location(
        module_name,
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
    specification.loader.exec_module(module)

    return module


def safe_extract(
    archive: tarfile.TarFile,
    destination: Path,
    require: Any,
) -> None:
    destination_resolved = destination.resolve()

    for member in archive.getmembers():
        target = (
            destination
            / member.name
        ).resolve()

        require(
            target == destination_resolved
            or destination_resolved in target.parents,
            f"unsafe archive member {member.name}",
        )

    archive.extractall(destination)


def submit_value(
    text: str,
    key: str,
    require: Any,
) -> str:
    match = re.search(
        rf"(?m)^\s*{re.escape(key)}\s*=\s*(.*?)\s*$",
        text,
    )

    require(
        match is not None,
        f"submit file is missing {key}",
    )

    return match.group(1)


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

    pilot_module_path = (
        repo
        / "scripts/production"
        / "audit_final_background_hbb_pilots10k.py"
    )

    canary_module_path = (
        repo
        / "scripts/production"
        / "audit_final_background_hbb_canaries.py"
    )

    pilot = load_module(
        pilot_module_path,
        "hbb_pilot_audit_helpers",
    )

    canary = load_module(
        canary_module_path,
        "hbb_canary_audit_helpers",
    )

    require = canary.require
    sha256_file = canary.sha256_file
    adler32_file = canary.adler32_file
    run_output = canary.run_output
    find_unique = canary.find_unique
    run_card_value = canary.run_card_value
    consistent_float = canary.consistent_float

    require(
        not outdir.exists(),
        f"refusing to overwrite {outdir}",
    )

    outdir.mkdir(parents=True)

    pilot_audit_path = (
        repo
        / "outputs/agent_runs"
        / "final_background_hbb_pilots10k_audit_20260722_v1"
        / "hbb_pilots10k_final_audit.json"
    )

    require(
        pilot_audit_path.is_file(),
        f"missing Hbb pilot audit {pilot_audit_path}",
    )

    pilot_audit = json.loads(
        pilot_audit_path.read_text()
    )

    require(
        pilot_audit.get(
            "hbb_pilots20k_final_audit_valid"
        )
        is True,
        "Hbb pilot audit is not valid",
    )

    pilot_rows = {
        row["family"]: row
        for row in pilot_audit["pilots"]
    }

    require(
        set(pilot_rows) == set(CAMPAIGNS),
        "pilot and scale-out family sets differ",
    )

    all_expected_shards = [
        shard
        for config in CAMPAIGNS.values()
        for shard in config["shards"]
    ]

    require(
        len(all_expected_shards) == 13,
        "expected exactly 13 Hbb scale-out shards",
    )

    require(
        len({
            shard["target_tag"]
            for shard in all_expected_shards
        }) == 13,
        "duplicate expected target tags",
    )

    require(
        len({
            shard["shard_id"]
            for shard in all_expected_shards
        }) == 13,
        "duplicate expected shard IDs",
    )

    require(
        len({
            shard["seed"]
            for shard in all_expected_shards
        }) == 13,
        "duplicate expected seeds",
    )

    audit_rows: list[dict[str, Any]] = []

    for family, config in CAMPAIGNS.items():
        campaign = str(
            config["campaign"]
        )

        cluster_id = str(
            config["cluster_id"]
        )

        expected_shards = {
            shard["target_tag"]: shard
            for shard in config["shards"]
        }

        receipt_dir = (
            store
            / "condor_return"
            / campaign
            / "receipts"
        )

        receipt_paths = sorted(
            receipt_dir.glob("*.json")
        )

        require(
            len(receipt_paths)
            == len(expected_shards),
            (
                f"{family}: expected {len(expected_shards)} "
                f"receipts, found {len(receipt_paths)}"
            ),
        )

        receipts = [
            json.loads(path.read_text())
            for path in receipt_paths
        ]

        receipts_by_tag = {
            receipt["target_tag"]: receipt
            for receipt in receipts
        }

        require(
            set(receipts_by_tag)
            == set(expected_shards),
            f"{family}: receipt target-tag set mismatch",
        )

        require(
            len(receipts_by_tag)
            == len(receipts),
            f"{family}: duplicate receipt target tags",
        )

        submit_file = (
            store
            / "condor_submit"
            / campaign
            / f"{campaign}.sub"
        )

        require(
            submit_file.is_file(),
            f"{family}: missing submit file {submit_file}",
        )

        submit_text = submit_file.read_text(
            errors="replace"
        )

        for fragment in (
            '+CountTowardBackgroundTarget = True',
            '+PhysicsYieldAuthorized = False',
            '+HiggsDecayStrategy = "Pythia force H->bb"',
        ):
            require(
                fragment in submit_text,
                f"{family}: submit metadata missing {fragment}",
            )

        payload = Path(
            submit_value(
                submit_text,
                "transfer_input_files",
                require,
            )
        )

        require(
            payload.is_file(),
            f"{family}: missing payload {payload}",
        )

        expected_payload_sha = str(
            pilot_rows[family]["payload_sha256"]
        )

        require(
            sha256_file(payload)
            == expected_payload_sha,
            f"{family}: payload differs from audited pilot payload",
        )

        expected_process = " ".join(
            str(
                canary.CONFIG[family]["process_text"]
            ).split()
        )

        expected_proc_sha = str(
            canary.CONFIG[family]["proc_card_sha256"]
        )

        with tempfile.TemporaryDirectory(
            prefix=f"hbb_scaleout_audit_{family}_",
        ) as temporary:
            temporary_root = Path(temporary)

            for target_tag in sorted(expected_shards):
                expected = expected_shards[target_tag]
                receipt = receipts_by_tag[target_tag]

                print(
                    f"===== AUDITING {family} {target_tag} =====",
                    flush=True,
                )

                expected_pythia_seed = (
                    int(expected["seed"]) * 37 + 17
                ) % 900_000_000

                if expected_pythia_seed == 0:
                    expected_pythia_seed = 1

                required_receipt = {
                    "campaign": campaign,
                    "family": family,
                    "target_tag": target_tag,
                    "shard_id":
                        int(expected["shard_id"]),
                    "n_events": EVENTS_PER_SHARD,
                    "seed": int(expected["seed"]),
                    "pythia_seed":
                        expected_pythia_seed,
                    "dataset_split":
                        expected["dataset_split"],
                    "split_assignment_unit":
                        "whole_shard",
                    "split_salt": SPLIT_SALT,
                    "run_card_profile":
                        "preserve_template",
                    "dataset_role":
                        expected["dataset_role"],
                    "cluster_id": cluster_id,
                    "stage":
                        "complete_copied_and_verified",
                    "payload_sha256":
                        expected_payload_sha,
                    "expected_payload_sha256":
                        expected_payload_sha,
                    "delphes_card_sha256":
                        pilot.EXPECTED_CARD_SHA256,
                    "lhe_events":
                        EVENTS_PER_SHARD,
                    "hepmc_events":
                        EVENTS_PER_SHARD,
                    "root_events":
                        EVENTS_PER_SHARD,
                    "exit_status": 0,
                }

                for key, value in required_receipt.items():
                    observed = receipt.get(key)

                    require(
                        observed == value,
                        (
                            f"{target_tag}: receipt mismatch "
                            f"for {key}: "
                            f"{observed!r} != {value!r}"
                        ),
                    )

                remote_bundle = str(
                    receipt.get("remote_bundle", "")
                )

                require(
                    remote_bundle.startswith(
                        "/store/user/"
                    ),
                    f"{target_tag}: invalid remote bundle",
                )

                local_bundle = (
                    temporary_root
                    / f"{target_tag}_bundle.tar.gz"
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
                    f"{target_tag}: bundle SHA mismatch",
                )

                require(
                    bundle_adler
                    == str(
                        receipt["bundle_adler32"]
                    ).lower(),
                    f"{target_tag}: bundle Adler mismatch",
                )

                require(
                    local_bundle.stat().st_size
                    == int(
                        receipt["bundle_size_bytes"]
                    ),
                    f"{target_tag}: bundle size mismatch",
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
                        f"{target_tag}: could not parse "
                        "EOS checksum"
                    ),
                )

                require(
                    checksum_matches[-1].lower()
                    == bundle_adler,
                    (
                        f"{target_tag}: EOS/local "
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
                        f"{target_tag}: could not parse "
                        "EOS bundle size"
                    ),
                )

                require(
                    int(size_match.group(1))
                    == local_bundle.stat().st_size,
                    (
                        f"{target_tag}: EOS/local "
                        "size mismatch"
                    ),
                )

                extraction_dir = (
                    temporary_root
                    / target_tag
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
                    == expected_proc_sha,
                    f"{target_tag}: process-card hash mismatch",
                )

                require(
                    sha256_file(root_file)
                    == receipt["root_sha256"],
                    f"{target_tag}: ROOT hash mismatch",
                )

                require(
                    sha256_file(event_summary)
                    == receipt["event_summary_sha256"],
                    (
                        f"{target_tag}: event-summary "
                        "hash mismatch"
                    ),
                )

                require(
                    sha256_file(candidates)
                    == receipt["candidate_sha256"],
                    (
                        f"{target_tag}: candidate "
                        "hash mismatch"
                    ),
                )

                require(
                    sha256_file(source_lhe)
                    == receipt["source_lhe_sha256"],
                    (
                        f"{target_tag}: source-LHE "
                        "hash mismatch"
                    ),
                )

                require(
                    sha256_file(banner)
                    == receipt["banner_sha256"],
                    f"{target_tag}: banner hash mismatch",
                )

                proc_text_raw = proc_card.read_text(
                    errors="replace"
                )

                proc_text = " ".join(
                    proc_text_raw.split()
                )

                require(
                    expected_process in proc_text,
                    (
                        f"{target_tag}: expected process "
                        "definition missing"
                    ),
                )

                if family == "ggh_hbb":
                    require(
                        "import model loop_sm"
                        in proc_text,
                        (
                            f"{target_tag}: loop_sm "
                            "declaration missing"
                        ),
                    )

                if family == "bbh_hbb_4fs":
                    proton_lines = [
                        line.strip()
                        for line
                        in proc_text_raw.splitlines()
                        if line.strip().startswith(
                            "define p ="
                        )
                    ]

                    require(
                        proton_lines,
                        (
                            f"{target_tag}: proton "
                            "definition missing"
                        ),
                    )

                    proton_tokens = set(
                        proton_lines[0]
                        .split("=")[1]
                        .split()
                    )

                    require(
                        "b" not in proton_tokens
                        and "b~" not in proton_tokens,
                        (
                            f"{target_tag}: 4FS proton "
                            "definition contains b"
                        ),
                    )

                run_text = run_card.read_text(
                    errors="replace"
                )

                expected_run_values = {
                    "nevents":
                        float(EVENTS_PER_SHARD),
                    "iseed":
                        float(expected["seed"]),
                    "ebeam1": 6500.0,
                    "ebeam2": 6500.0,
                }

                for key, value in expected_run_values.items():
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

                if family == "bbh_hbb_4fs":
                    for key, value in {
                        "ptb": 20.0,
                        "etab": 2.7,
                        "drbb": 0.4,
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

                lhe_events = pilot.audit_lhe(
                    source_lhe,
                    family,
                    require,
                )

                root_entries, hbb_truth_events = (
                    pilot.count_direct_hbb_truth(
                        root_file,
                        require,
                    )
                )

                require(
                    root_entries
                    == EVENTS_PER_SHARD,
                    (
                        f"{target_tag}: ROOT event "
                        "count mismatch"
                    ),
                )

                require(
                    hbb_truth_events
                    == EVENTS_PER_SHARD,
                    (
                        f"{target_tag}: direct H→bb "
                        "truth mismatch"
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
                    == EVENTS_PER_SHARD,
                    (
                        f"{target_tag}: event-summary "
                        "row count mismatch"
                    ),
                )

                require(
                    len(candidate_frame)
                    == int(
                        receipt["candidate_rows"]
                    ),
                    (
                        f"{target_tag}: candidate "
                        "row count mismatch"
                    ),
                )

                xsec_column, summary_xsec = (
                    pilot.event_xsec(
                        event_frame,
                        require,
                    )
                )

                receipt_xsec = float(
                    receipt["generator_xsec_pb"]
                )

                provenance = json.loads(
                    provenance_path.read_text()
                )

                provenance_xsec = float(
                    provenance["generator_xsec_pb"]
                )

                require(
                    float(
                        canary.CONFIG[family][
                            "xsec_min_pb"
                        ]
                    )
                    < receipt_xsec
                    < float(
                        canary.CONFIG[family][
                            "xsec_max_pb"
                        ]
                    ),
                    (
                        f"{target_tag}: generator "
                        "cross section outside range"
                    ),
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

                required_provenance = {
                    "campaign": campaign,
                    "family": family,
                    "target_tag": target_tag,
                    "shard_id":
                        int(expected["shard_id"]),
                    "n_events":
                        EVENTS_PER_SHARD,
                    "seed": int(expected["seed"]),
                    "pythia_seed":
                        expected_pythia_seed,
                    "dataset_split":
                        expected["dataset_split"],
                    "split_assignment_unit":
                        "whole_shard",
                    "split_salt": SPLIT_SALT,
                    "run_card_profile":
                        "preserve_template",
                    "dataset_role":
                        expected["dataset_role"],
                    "lhe_events":
                        EVENTS_PER_SHARD,
                    "hepmc_events":
                        EVENTS_PER_SHARD,
                    "root_events":
                        EVENTS_PER_SHARD,
                }

                for key, value in required_provenance.items():
                    observed = provenance.get(key)

                    require(
                        observed == value,
                        (
                            f"{target_tag}: provenance "
                            f"mismatch for {key}: "
                            f"{observed!r} != {value!r}"
                        ),
                    )

                audit_rows.append({
                    "family": family,
                    "campaign": campaign,
                    "cluster_id": cluster_id,
                    "target_tag": target_tag,
                    "shard_id":
                        int(expected["shard_id"]),
                    "seed": int(expected["seed"]),
                    "pythia_seed":
                        expected_pythia_seed,
                    "dataset_split":
                        expected["dataset_split"],
                    "dataset_role":
                        expected["dataset_role"],
                    "events":
                        EVENTS_PER_SHARD,
                    "lhe_events": lhe_events,
                    "root_events": root_entries,
                    "direct_hbb_truth_events":
                        hbb_truth_events,
                    "candidate_rows":
                        len(candidate_frame),
                    "generator_xsec_pb":
                        receipt_xsec,
                    "xsec_column": xsec_column,
                    "payload_sha256":
                        receipt["payload_sha256"],
                    "bundle_sha256":
                        bundle_sha,
                    "bundle_adler32":
                        bundle_adler,
                    "count_toward_background_target":
                        True,
                    "final_audit_valid": True,
                    "physics_yield_authorized":
                        False,
                })

                local_bundle.unlink()
                subprocess.run(
                    [
                        "rm",
                        "-rf",
                        str(extraction_dir),
                    ],
                    check=True,
                )

                print(
                    f"{target_tag}: PASS",
                    flush=True,
                )

    require(
        len(audit_rows) == 13,
        "expected 13 validated scale-out rows",
    )

    require(
        sum(
            int(row["events"])
            for row in audit_rows
        ) == 130000,
        "validated Hbb scale-out total is not 130k",
    )

    require(
        sum(
            int(row["events"])
            for row in audit_rows
            if row["family"] == "ggh_hbb"
        ) == 90000,
        "validated ggH scale-out total is not 90k",
    )

    require(
        sum(
            int(row["events"])
            for row in audit_rows
            if row["family"] == "bbh_hbb_4fs"
        ) == 40000,
        "validated bbH scale-out total is not 40k",
    )

    summary = {
        "schema_version": 1,
        "status": "pass",
        "shards": audit_rows,
        "ggh_hbb_scaleout90k_final_audit_valid":
            True,
        "bbh_hbb_4fs_scaleout40k_final_audit_valid":
            True,
        "hbb_scaleouts130k_final_audit_valid":
            True,
        "validated_hbb_pilot_events":
            20000,
        "validated_hbb_scaleout_events":
            130000,
        "validated_background_events_before_hbb":
            4800000,
        "validated_background_events_now":
            4950000,
        "remaining_background_events":
            50000,
        "remaining_background_family":
            "triboson_zbb",
        "physics_yield_authorized":
            False,
    }

    summary_path = (
        outdir
        / "hbb_scaleouts130k_final_audit.json"
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
        / "hbb_scaleouts130k_final_audit.tsv"
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
    print("GGH_HBB_SCALEOUT90K_FINAL_AUDIT_VALID")
    print("BBH_HBB_4FS_SCALEOUT40K_FINAL_AUDIT_VALID")
    print("HBB_SCALEOUTS130K_FINAL_AUDIT_VALID")
    print("VALIDATED_BACKGROUND_EVENTS_NOW_4950000")
    print("BACKGROUND_EVENTS_REMAINING_TRIBOSON_50000")
    print("NO_BACKGROUND_PHYSICS_YIELD_AUTHORIZATION_YET")
    print(f"summary_json={summary_path}")


if __name__ == "__main__":
    main()
