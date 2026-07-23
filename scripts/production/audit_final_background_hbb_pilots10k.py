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
from typing import Any, Iterator

import awkward as ak
import pandas as pd
import uproot


EOS_HOST = "root://cmseos.fnal.gov"

PILOTS = {
    "ggh_hbb": {
        "campaign": "ggh_hbb_pilot10k_20260722_v1",
        "target_tag": "ggh_hbb_run2_frozen_v2_pilot_shard9601_10k",
        "shard_id": 9601,
        "seed": 996001,
        "cluster_id": "3615067",
        "scaleout_events": 90000,
    },
    "bbh_hbb_4fs": {
        "campaign": "bbh_hbb_4fs_pilot10k_20260722_v1",
        "target_tag":
            "bbh_hbb_4fs_run2_frozen_v2_pilot_shard9602_10k",
        "shard_id": 9602,
        "seed": 996002,
        "cluster_id": "3615069",
        "scaleout_events": 40000,
    },
}

EVENTS = 10000
PILOT_SPLIT_SALT = "final-background-hbb-pilot-v1"
EXPECTED_CARD_SHA256 = (
    "1b2041245162de8360defdc502404e0696496c50773d4aa7f043798651a9517c"
)


def load_canary_helpers(repo: Path) -> Any:
    helper_path = (
        repo
        / "scripts/production"
        / "audit_final_background_hbb_canaries.py"
    )

    if not helper_path.is_file():
        raise RuntimeError(
            f"missing Hbb canary audit helper {helper_path}"
        )

    specification = importlib.util.spec_from_file_location(
        "hbb_canary_audit_helpers",
        helper_path,
    )

    if (
        specification is None
        or specification.loader is None
    ):
        raise RuntimeError(
            f"could not load helper module {helper_path}"
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


def iter_lhe_events(
    path: Path,
) -> Iterator[list[tuple[int, int]]]:
    with gzip.open(
        path,
        "rt",
        errors="replace",
    ) as handle:
        in_event = False
        event_lines: list[str] = []

        for line in handle:
            stripped = line.strip()

            if stripped == "<event>":
                in_event = True
                event_lines = []
                continue

            if stripped == "</event>":
                if not event_lines:
                    raise RuntimeError(
                        f"empty LHE event in {path}"
                    )

                header = event_lines[0].split()

                if not header:
                    raise RuntimeError(
                        f"missing LHE event header in {path}"
                    )

                nup = int(header[0])

                particles: list[tuple[int, int]] = []

                for particle_line in event_lines[1:1 + nup]:
                    fields = particle_line.split()

                    if len(fields) < 2:
                        raise RuntimeError(
                            f"malformed LHE line: {particle_line}"
                        )

                    particles.append(
                        (
                            int(fields[0]),
                            int(fields[1]),
                        )
                    )

                if len(particles) != nup:
                    raise RuntimeError(
                        "LHE particle-count mismatch: "
                        f"{len(particles)} != {nup}"
                    )

                yield particles

                in_event = False
                event_lines = []
                continue

            if in_event and stripped:
                event_lines.append(stripped)


def audit_lhe(
    path: Path,
    family: str,
    require: Any,
) -> int:
    event_count = 0

    for event_index, particles in enumerate(
        iter_lhe_events(path)
    ):
        event_count += 1

        incoming_ids = [
            pid
            for pid, status in particles
            if status == -1
        ]

        final_ids = [
            pid
            for pid, status in particles
            if status == 1
        ]

        require(
            final_ids.count(25) == 1,
            (
                f"{family}: LHE event {event_index} "
                "does not contain exactly one stable Higgs"
            ),
        )

        if family == "ggh_hbb":
            require(
                len(incoming_ids) == 2
                and all(pid == 21 for pid in incoming_ids),
                (
                    f"{family}: LHE event {event_index} "
                    "is not gg initiated"
                ),
            )

        elif family == "bbh_hbb_4fs":
            require(
                all(abs(pid) != 5 for pid in incoming_ids),
                (
                    f"{family}: LHE event {event_index} "
                    "contains an incoming b quark"
                ),
            )

            require(
                5 in final_ids and -5 in final_ids,
                (
                    f"{family}: LHE event {event_index} "
                    "does not contain the final-state b pair"
                ),
            )

        else:
            raise RuntimeError(
                f"unsupported family {family}"
            )

    require(
        event_count == EVENTS,
        (
            f"{family}: LHE event count mismatch: "
            f"{event_count} != {EVENTS}"
        ),
    )

    return event_count


def count_direct_hbb_truth(
    root_file: Path,
    require: Any,
) -> tuple[int, int]:
    processed = 0
    valid = 0

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
                processed += 1

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

                has_b = False
                has_bbar = False

                for index, pid in enumerate(pids):
                    if pid not in (5, -5):
                        continue

                    mother_indices = {
                        mothers1[index],
                        mothers2[index],
                    }

                    from_higgs = any(
                        0 <= mother < len(pids)
                        and pids[mother] == 25
                        for mother in mother_indices
                    )

                    if not from_higgs:
                        continue

                    if pid == 5:
                        has_b = True
                    elif pid == -5:
                        has_bbar = True

                if has_b and has_bbar:
                    valid += 1

    require(
        processed == total_entries,
        (
            "ROOT iteration count mismatch: "
            f"{processed} != {total_entries}"
        ),
    )

    return total_entries, valid


def event_xsec(
    frame: pd.DataFrame,
    require: Any,
) -> tuple[str, float]:
    possible_columns = (
        "event_cross_section_pb",
        "generator_xsec_pb",
        "cross_section_pb",
    )

    for column in possible_columns:
        if column not in frame.columns:
            continue

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
        "event summary lacks a recognized "
        "cross-section column"
    )


def audit_family(
    *,
    family: str,
    config: dict[str, Any],
    canary_config: dict[str, Any],
    canary_row: dict[str, Any],
    store: Path,
    temporary_root: Path,
    helpers: Any,
) -> dict[str, Any]:
    require = helpers.require
    sha256_file = helpers.sha256_file
    adler32_file = helpers.adler32_file
    run_output = helpers.run_output
    find_unique = helpers.find_unique
    run_card_value = helpers.run_card_value
    consistent_float = helpers.consistent_float
    parse_manifest = helpers.parse_manifest_from_payload

    campaign = str(config["campaign"])

    receipt_dir = (
        store
        / "condor_return"
        / campaign
        / "receipts"
    )

    receipts = list(
        receipt_dir.glob("*.json")
    )

    require(
        len(receipts) == 1,
        f"{family}: expected one receipt, found {receipts}",
    )

    receipt_path = receipts[0]

    receipt = json.loads(
        receipt_path.read_text()
    )

    expected_pythia_seed = (
        int(config["seed"]) * 37 + 17
    ) % 900_000_000

    if expected_pythia_seed == 0:
        expected_pythia_seed = 1

    required_receipt = {
        "campaign": campaign,
        "family": family,
        "target_tag": config["target_tag"],
        "shard_id": int(config["shard_id"]),
        "n_events": EVENTS,
        "seed": int(config["seed"]),
        "pythia_seed": expected_pythia_seed,
        "dataset_split": "validation",
        "split_assignment_unit": "whole_shard",
        "split_salt": PILOT_SPLIT_SALT,
        "run_card_profile": "preserve_template",
        "dataset_role": "pilot_validation",
        "cluster_id": str(config["cluster_id"]),
        "stage": "complete_copied_and_verified",
        "payload_sha256": canary_row["payload_sha256"],
        "expected_payload_sha256": canary_row["payload_sha256"],
        "delphes_card_sha256": EXPECTED_CARD_SHA256,
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
                f"{family}: receipt mismatch for {key}: "
                f"{observed!r} != {expected!r}"
            ),
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

    required_submit_fragments = (
        '+CountTowardBackgroundTarget = True',
        '+DatasetRole = "pilot_validation"',
        '+PhysicsYieldAuthorized = False',
        '+HiggsDecayStrategy = "Pythia force H->bb"',
    )

    for fragment in required_submit_fragments:
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

    require(
        sha256_file(payload)
        == canary_row["payload_sha256"],
        f"{family}: payload differs from audited canary payload",
    )

    payload_manifest = parse_manifest(
        payload
    )

    require(
        payload_manifest.get("family") == family,
        f"{family}: payload manifest family mismatch",
    )

    worker_sha = payload_manifest.get(
        "worker_wrapper_sha256",
        "",
    )

    require(
        re.fullmatch(
            r"[0-9a-f]{64}",
            worker_sha,
        )
        is not None,
        f"{family}: invalid worker hash in payload manifest",
    )

    converter_sha = payload_manifest.get(
        "seeded_converter_sha256",
        "",
    )

    require(
        re.fullmatch(
            r"[0-9a-f]{64}",
            converter_sha,
        )
        is not None,
        f"{family}: invalid converter hash in payload manifest",
    )

    remote_bundle = str(
        receipt.get("remote_bundle", "")
    )

    require(
        remote_bundle.startswith("/store/user/"),
        f"{family}: invalid remote bundle path",
    )

    local_bundle = (
        temporary_root
        / f"{family}_pilot10k_bundle.tar.gz"
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

    bundle_sha256 = sha256_file(
        local_bundle
    )

    bundle_adler32 = adler32_file(
        local_bundle
    )

    require(
        bundle_sha256
        == receipt["bundle_sha256"],
        f"{family}: bundle SHA-256 mismatch",
    )

    require(
        bundle_adler32
        == str(receipt["bundle_adler32"]).lower(),
        f"{family}: bundle Adler-32 mismatch",
    )

    require(
        local_bundle.stat().st_size
        == int(receipt["bundle_size_bytes"]),
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
        == bundle_adler32,
        f"{family}: EOS/local checksum mismatch",
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
        f"{family}: could not parse EOS size",
    )

    require(
        int(size_match.group(1))
        == local_bundle.stat().st_size,
        f"{family}: EOS/local size mismatch",
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

    require(
        sha256_file(proc_card)
        == str(canary_config["proc_card_sha256"]),
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
        f"{family}: source-LHE hash mismatch",
    )

    require(
        sha256_file(banner)
        == receipt["banner_sha256"],
        f"{family}: banner hash mismatch",
    )

    proc_text_raw = proc_card.read_text(
        errors="replace"
    )

    proc_text = " ".join(
        proc_text_raw.split()
    )

    expected_process = " ".join(
        str(canary_config["process_text"]).split()
    )

    require(
        expected_process in proc_text,
        f"{family}: expected process definition missing",
    )

    if family == "ggh_hbb":
        require(
            "import model loop_sm" in proc_text,
            "ggh_hbb: loop_sm declaration missing",
        )

    elif family == "bbh_hbb_4fs":
        proton_lines = [
            line.strip()
            for line in proc_text_raw.splitlines()
            if line.strip().startswith("define p =")
        ]

        require(
            len(proton_lines) >= 1,
            "bbh_hbb_4fs: proton definition missing",
        )

        proton_tokens = set(
            proton_lines[0].split("=")[1].split()
        )

        require(
            "b" not in proton_tokens
            and "b~" not in proton_tokens,
            "bbh_hbb_4fs: proton definition contains b",
        )

    run_text = run_card.read_text(
        errors="replace"
    )

    expected_run_values = {
        "nevents": float(EVENTS),
        "iseed": float(config["seed"]),
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
                f"{family}: run-card {key} mismatch: "
                f"{observed} != {expected}"
            ),
        )

    if family == "bbh_hbb_4fs":
        required_bbh_values = {
            "ptb": 20.0,
            "etab": 2.7,
            "drbb": 0.4,
        }

        for key, expected in required_bbh_values.items():
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
                    f"{family}: run-card {key} mismatch: "
                    f"{observed} != {expected}"
                ),
            )

    lhe_events = audit_lhe(
        source_lhe,
        family,
        require,
    )

    root_entries, hbb_truth_events = (
        count_direct_hbb_truth(
            root_file,
            require,
        )
    )

    require(
        root_entries == EVENTS,
        (
            f"{family}: ROOT event count mismatch: "
            f"{root_entries} != {EVENTS}"
        ),
    )

    require(
        hbb_truth_events == EVENTS,
        (
            f"{family}: direct H->bb truth mismatch: "
            f"{hbb_truth_events}/{EVENTS}"
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
            f"{family}: event-summary rows mismatch: "
            f"{len(event_frame)} != {EVENTS}"
        ),
    )

    require(
        len(candidate_frame)
        == int(receipt["candidate_rows"]),
        f"{family}: candidate-row count mismatch",
    )

    xsec_column, summary_xsec = event_xsec(
        event_frame,
        require,
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
        float(canary_config["xsec_min_pb"])
        < receipt_xsec
        < float(canary_config["xsec_max_pb"]),
        f"{family}: generator cross section outside expected range",
    )

    require(
        consistent_float(
            receipt_xsec,
            summary_xsec,
        ),
        f"{family}: receipt/event-summary xsec mismatch",
    )

    require(
        consistent_float(
            receipt_xsec,
            provenance_xsec,
        ),
        f"{family}: receipt/provenance xsec mismatch",
    )

    required_provenance = {
        "campaign": campaign,
        "family": family,
        "n_events": EVENTS,
        "seed": int(config["seed"]),
        "pythia_seed": expected_pythia_seed,
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
                f"{family}: provenance mismatch for {key}: "
                f"{observed!r} != {expected!r}"
            ),
        )

    return {
        "family": family,
        "campaign": campaign,
        "cluster_id": str(config["cluster_id"]),
        "events": EVENTS,
        "candidate_rows": len(candidate_frame),
        "generator_xsec_pb": receipt_xsec,
        "xsec_column": xsec_column,
        "lhe_events": lhe_events,
        "root_events": root_entries,
        "direct_hbb_truth_events": hbb_truth_events,
        "payload_sha256": receipt["payload_sha256"],
        "bundle_sha256": bundle_sha256,
        "bundle_adler32": bundle_adler32,
        "count_toward_background_target": True,
        "pilot_final_audit_valid": True,
        "scaleout_events_authorized":
            int(config["scaleout_events"]),
        "physics_yield_authorized": False,
    }


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

    helpers = load_canary_helpers(repo)
    require = helpers.require

    require(
        not outdir.exists(),
        f"refusing to overwrite {outdir}",
    )

    outdir.mkdir(parents=True)

    canary_audit_path = (
        repo
        / "outputs/agent_runs"
        / "final_background_hbb_canary_audit_20260722_v2"
        / "hbb_canary_final_audit.json"
    )

    require(
        canary_audit_path.is_file(),
        f"missing canary audit {canary_audit_path}",
    )

    canary_audit = json.loads(
        canary_audit_path.read_text()
    )

    require(
        canary_audit.get(
            "hbb_canaries_final_audit_valid"
        )
        is True,
        "Hbb canary final audit is not valid",
    )

    canary_rows = {
        row["family"]: row
        for row
        in canary_audit["technical_subprocesses"]
    }

    require(
        set(canary_rows) == set(PILOTS),
        "canary and pilot family sets differ",
    )

    rows: list[dict[str, Any]] = []

    with tempfile.TemporaryDirectory(
        prefix="hbb_pilot10k_audit_",
    ) as temporary:
        temporary_root = Path(temporary)

        for family, config in PILOTS.items():
            print(
                f"===== AUDITING {family} 10K PILOT =====",
                flush=True,
            )

            row = audit_family(
                family=family,
                config=config,
                canary_config=helpers.CONFIG[family],
                canary_row=canary_rows[family],
                store=store,
                temporary_root=temporary_root,
                helpers=helpers,
            )

            rows.append(row)

            print(
                f"{family}: PASS",
                flush=True,
            )

    require(
        len(rows) == 2,
        "expected two validated Hbb pilots",
    )

    summary = {
        "schema_version": 1,
        "status": "pass",
        "pilots": rows,
        "hbb_pilots20k_final_audit_valid": True,
        "ggh_hbb_scaleout90k_submission_authorized": True,
        "bbh_hbb_4fs_scaleout40k_submission_authorized": True,
        "validated_pilot_events": 20000,
        "validated_background_events_before_pilots": 4800000,
        "validated_background_events_after_pilots": 4820000,
        "remaining_background_events": 180000,
        "remaining_hbb_events": 130000,
        "remaining_triboson_events": 50000,
        "physics_yield_authorized": False,
    }

    summary_path = (
        outdir
        / "hbb_pilots10k_final_audit.json"
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
        / "hbb_pilots10k_final_audit.tsv"
    )

    with table_path.open(
        "w",
        newline="",
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=list(rows[0]),
            delimiter="\t",
            lineterminator="\n",
        )

        writer.writeheader()
        writer.writerows(rows)

    print()
    print("GGH_HBB_PILOT10K_FINAL_AUDIT_VALID")
    print("BBH_HBB_4FS_PILOT10K_FINAL_AUDIT_VALID")
    print("HBB_PILOTS20K_FINAL_AUDIT_VALID")
    print("GGH_HBB_SCALEOUT90K_SUBMISSION_AUTHORIZED")
    print("BBH_HBB_4FS_SCALEOUT40K_SUBMISSION_AUTHORIZED")
    print("VALIDATED_BACKGROUND_EVENTS_NOW_4820000")
    print("BACKGROUND_EVENTS_REMAINING_180000")
    print("NO_BACKGROUND_PHYSICS_YIELD_AUTHORIZATION_YET")
    print(f"summary_json={summary_path}")


if __name__ == "__main__":
    main()
