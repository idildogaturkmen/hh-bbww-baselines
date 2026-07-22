#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import math
import os
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

CAMPAIGN = "vbf_hh4b_sm_pilot10k_20260722_v1"
FAMILY = "vbf_hh4b_sm"
TARGET_TAG = "vbf_hh4b_sm_run2_frozen_v2_pilot_shard9301_10k"

SHARD_ID = 9301
EVENTS = 10000
SEED = 983001
PYTHIA_SEED = 36371054
CLUSTER_ID = "59862617"

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


def load_helpers(repo: Path) -> Any:
    helper_path = (
        repo
        / "scripts/production"
        / "audit_final_background_hbb_canaries.py"
    )

    if not helper_path.is_file():
        raise RuntimeError(
            f"missing audit helper module {helper_path}"
        )

    specification = importlib.util.spec_from_file_location(
        "hbb_audit_helpers",
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


def direct_hh4b_truth_count(
    root_file: Path,
    require: Any,
) -> tuple[int, int]:
    total_events = 0
    valid_events = 0

    with uproot.open(root_file) as source:
        tree = source["Delphes"]

        total_entries = int(tree.num_entries)

        for arrays in tree.iterate(
            [
                "Particle.PID",
                "Particle.M1",
                "Particle.M2",
            ],
            step_size=500,
            library="ak",
        ):
            for pids_raw, m1_raw, m2_raw in zip(
                arrays["Particle.PID"],
                arrays["Particle.M1"],
                arrays["Particle.M2"],
            ):
                total_events += 1

                pids = [
                    int(value)
                    for value in ak.to_list(pids_raw)
                ]

                mothers1 = [
                    int(value)
                    for value in ak.to_list(m1_raw)
                ]

                mothers2 = [
                    int(value)
                    for value in ak.to_list(m2_raw)
                ]

                higgs_decay_products: dict[int, set[int]] = {}

                for index, pid in enumerate(pids):
                    if pid not in (5, -5):
                        continue

                    mother_indices = {
                        mothers1[index],
                        mothers2[index],
                    }

                    for mother in mother_indices:
                        if (
                            0 <= mother < len(pids)
                            and pids[mother] == 25
                        ):
                            higgs_decay_products.setdefault(
                                mother,
                                set(),
                            ).add(pid)

                hbb_higgs_count = sum(
                    1
                    for daughters
                    in higgs_decay_products.values()
                    if {5, -5}.issubset(daughters)
                )

                if hbb_higgs_count >= 2:
                    valid_events += 1

    require(
        total_events == total_entries,
        (
            "ROOT iteration mismatch: "
            f"{total_events} != {total_entries}"
        ),
    )

    return total_entries, valid_events


def read_event_xsec(
    frame: pd.DataFrame,
    require: Any,
) -> tuple[str, float]:
    possible_columns = (
        "event_cross_section_pb",
        "generator_xsec_pb",
        "cross_section_pb",
    )

    for column in possible_columns:
        if column in frame.columns:
            values = pd.to_numeric(
                frame[column],
                errors="coerce",
            ).dropna()

            require(
                len(values) > 0,
                f"cross-section column {column} is empty",
            )

            return column, float(values.median())

    raise RuntimeError(
        "event summary contains no recognized "
        "cross-section column"
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

    arguments = parser.parse_args()

    repo = arguments.repo.resolve()
    store = arguments.store.resolve()
    outdir = arguments.outdir.resolve()

    helpers = load_helpers(repo)

    require = helpers.require
    sha256_file = helpers.sha256_file
    adler32_file = helpers.adler32_file
    run_output = helpers.run_output
    find_unique = helpers.find_unique
    parse_lhe_events = helpers.parse_lhe_events
    run_card_value = helpers.run_card_value
    consistent_float = helpers.consistent_float

    require(
        not outdir.exists(),
        f"refusing to overwrite {outdir}",
    )

    outdir.mkdir(parents=True)

    receipt_dir = (
        store
        / "condor_return"
        / CAMPAIGN
        / "receipts"
    )

    receipts = list(
        receipt_dir.glob("*.json")
    )

    require(
        len(receipts) == 1,
        f"expected one receipt, found {receipts}",
    )

    receipt_path = receipts[0]

    receipt = json.loads(
        receipt_path.read_text()
    )

    required_receipt = {
        "schema_version": 1,
        "campaign": CAMPAIGN,
        "family": FAMILY,
        "target_tag": TARGET_TAG,
        "shard_id": SHARD_ID,
        "n_events": EVENTS,
        "seed": SEED,
        "pythia_seed": PYTHIA_SEED,
        "dataset_split": "validation",
        "split_assignment_unit": "whole_shard",
        "split_salt": "vbf-hh4b-sm-pilot-v1",
        "run_card_profile": "preserve_template",
        "dataset_role": "pilot_validation",
        "cluster_id": CLUSTER_ID,
        "stage": "complete_copied_and_verified",
        "payload_sha256": PAYLOAD_SHA256,
        "expected_payload_sha256": PAYLOAD_SHA256,
        "delphes_card_sha256": DELPHES_CARD_SHA256,
        "lhe_events": EVENTS,
        "hepmc_events": EVENTS,
        "root_events": EVENTS,
        "exit_status": 0,
    }

    for key, expected in required_receipt.items():
        observed = receipt.get(key)

        require(
            observed == expected,
            (
                f"receipt mismatch for {key}: "
                f"{observed!r} != {expected!r}"
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

    required_submit_fragments = (
        '+CountTowardSignalTarget = True',
        '+DatasetRole = "pilot_validation"',
        '+MatrixElementProcess = "p p -> h h j j, QCD=0"',
        '+HiggsDecayStrategy = "Pythia force H->bb"',
    )

    for fragment in required_submit_fragments:
        require(
            fragment in submit_text,
            f"submit metadata missing: {fragment}",
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
        sha256_file(payload) == PAYLOAD_SHA256,
        "local payload hash mismatch",
    )

    remote_bundle = str(
        receipt.get("remote_bundle", "")
    )

    require(
        remote_bundle.startswith("/store/user/"),
        "invalid remote bundle path",
    )

    with tempfile.TemporaryDirectory(
        prefix="vbf_hh4b_pilot_audit_",
    ) as temporary:
        temporary_root = Path(temporary)

        local_bundle = (
            temporary_root
            / "vbf_hh4b_pilot_bundle.tar.gz"
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

        local_sha256 = sha256_file(local_bundle)
        local_adler32 = adler32_file(local_bundle)
        local_size = local_bundle.stat().st_size

        require(
            local_sha256
            == receipt["bundle_sha256"],
            "bundle SHA-256 mismatch",
        )

        require(
            local_adler32
            == str(receipt["bundle_adler32"]).lower(),
            "bundle Adler-32 mismatch",
        )

        require(
            local_size
            == int(receipt["bundle_size_bytes"]),
            "bundle size mismatch",
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
            "could not parse EOS checksum",
        )

        require(
            checksum_matches[-1].lower()
            == local_adler32,
            "EOS and local checksums differ",
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
            "could not parse EOS size",
        )

        require(
            int(size_match.group(1))
            == local_size,
            "EOS and local bundle sizes differ",
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
            "process-card hash mismatch",
        )

        require(
            sha256_file(root_file)
            == receipt["root_sha256"],
            "ROOT hash mismatch",
        )

        require(
            sha256_file(event_summary)
            == receipt["event_summary_sha256"],
            "event-summary hash mismatch",
        )

        require(
            sha256_file(candidates)
            == receipt["candidate_sha256"],
            "candidate hash mismatch",
        )

        require(
            sha256_file(source_lhe)
            == receipt["source_lhe_sha256"],
            "source-LHE hash mismatch",
        )

        require(
            sha256_file(banner)
            == receipt["banner_sha256"],
            "banner hash mismatch",
        )

        proc_text = " ".join(
            proc_card.read_text(
                errors="replace",
            ).split()
        )

        require(
            "generate p p > h h j j QCD=0"
            in proc_text,
            "expected VBF HH process is missing",
        )

        run_text = run_card.read_text(
            errors="replace"
        )

        expected_run_values = {
            "nevents": float(EVENTS),
            "iseed": float(SEED),
            "ebeam1": 6500.0,
            "ebeam2": 6500.0,
        }

        for key, expected in expected_run_values.items():
            observed = run_card_value(
                run_text,
                key,
            )

            require(
                math.isclose(
                    observed,
                    expected,
                    rel_tol=0.0,
                    abs_tol=1.0e-9,
                ),
                (
                    f"run-card {key} mismatch: "
                    f"{observed} != {expected}"
                ),
            )

        lhe_events = parse_lhe_events(
            source_lhe
        )

        require(
            len(lhe_events) == EVENTS,
            (
                "LHE event count mismatch: "
                f"{len(lhe_events)} != {EVENTS}"
            ),
        )

        for event_index, particles in enumerate(
            lhe_events
        ):
            final_higgs_count = sum(
                1
                for pid, status in particles
                if pid == 25 and status == 1
            )

            require(
                final_higgs_count == 2,
                (
                    f"LHE event {event_index} has "
                    f"{final_higgs_count} stable Higgs bosons"
                ),
            )

        root_entries, hh4b_truth_events = (
            direct_hh4b_truth_count(
                root_file,
                require,
            )
        )

        require(
            root_entries == EVENTS,
            (
                "ROOT event count mismatch: "
                f"{root_entries} != {EVENTS}"
            ),
        )

        require(
            hh4b_truth_events == EVENTS,
            (
                "forced HH->4b truth mismatch: "
                f"{hh4b_truth_events}/{EVENTS}"
            ),
        )

        event_frame = pd.read_parquet(
            event_summary
        )

        candidate_frame = pd.read_parquet(
            candidates
        )

        require(
            len(event_frame) == EVENTS,
            (
                "event-summary row mismatch: "
                f"{len(event_frame)} != {EVENTS}"
            ),
        )

        require(
            len(candidate_frame)
            == int(receipt["candidate_rows"]),
            "candidate-row count mismatch",
        )

        xsec_column, summary_xsec = (
            read_event_xsec(
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
            consistent_float(
                receipt_xsec,
                summary_xsec,
            ),
            "receipt/event-summary xsec mismatch",
        )

        require(
            consistent_float(
                receipt_xsec,
                provenance_xsec,
            ),
            "receipt/provenance xsec mismatch",
        )

        xsec_ratio = (
            receipt_xsec
            / REFERENCE_XSEC_PB
        )

        require(
            0.80 <= xsec_ratio <= 1.20,
            (
                "VBF cross section outside frozen "
                f"reference tolerance: ratio={xsec_ratio}"
            ),
        )

        required_provenance = {
            "campaign": CAMPAIGN,
            "family": FAMILY,
            "n_events": EVENTS,
            "seed": SEED,
            "pythia_seed": PYTHIA_SEED,
            "dataset_split": "validation",
            "dataset_role": "pilot_validation",
            "lhe_events": EVENTS,
            "hepmc_events": EVENTS,
            "root_events": EVENTS,
        }

        for key, expected in required_provenance.items():
            observed = provenance.get(key)

            require(
                observed == expected,
                (
                    f"provenance mismatch for {key}: "
                    f"{observed!r} != {expected!r}"
                ),
            )

        row = {
            "campaign": CAMPAIGN,
            "family": FAMILY,
            "cluster_id": CLUSTER_ID,
            "events": EVENTS,
            "candidate_rows": len(candidate_frame),
            "generator_xsec_pb": receipt_xsec,
            "reference_xsec_pb": REFERENCE_XSEC_PB,
            "xsec_to_reference_ratio": xsec_ratio,
            "xsec_column": xsec_column,
            "stable_two_higgs_lhe_events": EVENTS,
            "forced_hh4b_truth_events": hh4b_truth_events,
            "payload_sha256": PAYLOAD_SHA256,
            "bundle_sha256": local_sha256,
            "bundle_adler32": local_adler32,
            "count_toward_signal_target": True,
            "pilot_final_audit_valid": True,
            "scaleout90k_submission_authorized": True,
            "physics_yield_authorized": False,
        }

    summary = {
        "schema_version": 1,
        "status": "pass",
        "pilot": row,
        "vbf_hh4b_pilot10k_final_audit_valid": True,
        "vbf_hh4b_scaleout90k_submission_authorized": True,
        "canonical_signal_target_events": 100000,
        "validated_pilot_events": 10000,
        "remaining_scaleout_events": 90000,
        "physics_yield_authorized": False,
    }

    summary_path = (
        outdir
        / "vbf_hh4b_pilot10k_final_audit.json"
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
        / "vbf_hh4b_pilot10k_final_audit.tsv"
    )

    with table_path.open(
        "w",
        newline="",
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=list(row),
            delimiter="\t",
            lineterminator="\n",
        )

        writer.writeheader()
        writer.writerow(row)

    print()
    print("VBF_HH4B_PILOT10K_FINAL_AUDIT_VALID")
    print("VBF_HH4B_SCALEOUT90K_SUBMISSION_AUTHORIZED")
    print("NO_VBF_PHYSICS_YIELD_AUTHORIZATION_YET")
    print(f"summary_json={summary_path}")


if __name__ == "__main__":
    main()
