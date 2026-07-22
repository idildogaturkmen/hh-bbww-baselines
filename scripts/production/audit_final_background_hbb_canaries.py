#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import math
import os
import re
import subprocess
import tarfile
import tempfile
import zlib
from pathlib import Path
from typing import Any

import awkward as ak
import pandas as pd
import uproot


EOS_HOST = "root://cmseos.fnal.gov"

EXPECTED_CARD_SHA256 = (
    "1b2041245162de8360defdc502404e0696496c50773d4aa7f043798651a9517c"
)

EXPECTED_WORKER_SHA256 = (
    "f8b40a0cb35a268cf636f0a43e260a7a261c628fe3553a8f98d880e343f78e80"
)

CONFIG = {
    "ggh_hbb": {
        "campaign": "ggh_hbb_canary100_20260722_v1",
        "seed": 994001,
        "shard_id": 9401,
        "payload_sha256":
            "cb60c53327010f8725f09f0d2db53e3d72e6157673495dc7d65f40052651bd05",
        "proc_card_sha256":
            "a201ab063ea83a16e3e90ec66f11f55db0297478ed8a55d9b2f995a0b5e93519",
        "run_card_sha256":
            "a3f2edacbf682ec40083be6bed8914eefed3532f00df48c0e522f3d6332fd3c0",
        "process_text": "generate g g > h [noborn=QCD]",
        "xsec_min_pb": 5.0,
        "xsec_max_pb": 25.0,
    },
    "bbh_hbb_4fs": {
        "campaign": "bbh_hbb_4fs_canary100_20260722_v1",
        "seed": 994002,
        "shard_id": 9402,
        "payload_sha256":
            "3908923321cc860836f7cad2829c6da0f74601945ae640f4fc1487206e247687",
        "proc_card_sha256":
            "4d378340dd98cba49d8afc1d21cde9da35eb8fabb5470bdfbd55e30709a1208d",
        "run_card_sha256":
            "51f1ae9a6e2209723fc3683a8bbbfe64569a1f621dbedffe18f8e89bca118217",
        "process_text": "generate p p > b b~ h QCD=2 QED=1",
        "xsec_min_pb": 0.005,
        "xsec_max_pb": 0.1,
    },
}


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()

    with path.open("rb") as handle:
        for block in iter(
            lambda: handle.read(8 * 1024 * 1024),
            b"",
        ):
            digest.update(block)

    return digest.hexdigest()


def adler32_file(path: Path) -> str:
    checksum = 1

    with path.open("rb") as handle:
        for block in iter(
            lambda: handle.read(8 * 1024 * 1024),
            b"",
        ):
            checksum = zlib.adler32(
                block,
                checksum,
            )

    return f"{checksum & 0xFFFFFFFF:08x}"


def run_output(arguments: list[str]) -> str:
    result = subprocess.run(
        arguments,
        check=True,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )

    return result.stdout


def find_unique(
    root: Path,
    pattern: str,
) -> Path:
    matches = list(root.rglob(pattern))

    require(
        len(matches) == 1,
        f"expected one {pattern} under {root}, found {matches}",
    )

    return matches[0]


def parse_manifest_from_payload(
    payload: Path,
) -> dict[str, str]:
    with tarfile.open(payload, "r:gz") as archive:
        members = [
            member
            for member in archive.getmembers()
            if member.name.endswith("/manifest.txt")
        ]

        require(
            len(members) == 1,
            f"expected one manifest in {payload}",
        )

        extracted = archive.extractfile(members[0])
        require(extracted is not None, "could not read payload manifest")

        text = extracted.read().decode(
            "utf-8",
            errors="replace",
        )

    values: dict[str, str] = {}

    for line in text.splitlines():
        if "=" not in line:
            continue

        key, value = line.split("=", 1)
        values[key.strip()] = value.strip()

    return values


def parse_lhe_events(
    path: Path,
) -> list[list[tuple[int, int]]]:
    events: list[list[tuple[int, int]]] = []

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
                require(
                    event_lines,
                    f"empty LHE event in {path}",
                )

                header = event_lines[0].split()
                require(
                    header,
                    f"missing LHE header in {path}",
                )

                nup = int(header[0])
                particles: list[tuple[int, int]] = []

                for particle_line in event_lines[1:1 + nup]:
                    fields = particle_line.split()

                    require(
                        len(fields) >= 2,
                        f"malformed particle line: {particle_line}",
                    )

                    particles.append(
                        (
                            int(fields[0]),
                            int(fields[1]),
                        )
                    )

                require(
                    len(particles) == nup,
                    f"LHE particle count mismatch in {path}",
                )

                events.append(particles)
                in_event = False
                continue

            if in_event and stripped:
                event_lines.append(stripped)

    return events


def run_card_value(
    text: str,
    key: str,
) -> float:
    pattern = re.compile(
        rf"(?m)^\s*([^#!\n=]+?)\s*=\s*{re.escape(key)}\b"
    )

    matches = pattern.findall(text)

    require(
        len(matches) == 1,
        f"expected one active {key} value, found {matches}",
    )

    return float(matches[0].strip())


def direct_hbb_event_count(
    root_file: Path,
) -> tuple[int, int]:
    with uproot.open(root_file) as source:
        tree = source["Delphes"]

        arrays = tree.arrays(
            [
                "Particle.PID",
                "Particle.M1",
                "Particle.M2",
            ],
            library="ak",
        )

        total = int(tree.num_entries)

    valid = 0

    for pids_raw, m1_raw, m2_raw in zip(
        arrays["Particle.PID"],
        arrays["Particle.M1"],
        arrays["Particle.M2"],
    ):
        pids = ak.to_list(pids_raw)
        m1 = ak.to_list(m1_raw)
        m2 = ak.to_list(m2_raw)

        has_b = False
        has_bbar = False

        for index, pid in enumerate(pids):
            if pid not in (5, -5):
                continue

            mother_indices = {
                int(m1[index]),
                int(m2[index]),
            }

            from_higgs = any(
                0 <= mother < len(pids)
                and int(pids[mother]) == 25
                for mother in mother_indices
            )

            if not from_higgs:
                continue

            if pid == 5:
                has_b = True

            if pid == -5:
                has_bbar = True

        if has_b and has_bbar:
            valid += 1

    return total, valid


def consistent_float(
    first: float,
    second: float,
) -> bool:
    return math.isclose(
        first,
        second,
        rel_tol=1.0e-6,
        abs_tol=1.0e-12,
    )


def audit_family(
    family: str,
    config: dict[str, Any],
    repo: Path,
    store: Path,
    temporary_root: Path,
) -> dict[str, Any]:
    campaign = str(config["campaign"])

    receipt_dir = (
        store
        / "condor_return"
        / campaign
        / "receipts"
    )

    receipt_files = list(
        receipt_dir.glob("*.json")
    )

    require(
        len(receipt_files) == 1,
        f"{family}: expected one receipt, found {receipt_files}",
    )

    receipt_path = receipt_files[0]
    receipt = json.loads(
        receipt_path.read_text()
    )

    required_receipt_values = {
        "campaign": campaign,
        "family": family,
        "n_events": 100,
        "seed": int(config["seed"]),
        "shard_id": int(config["shard_id"]),
        "dataset_split": "validation",
        "dataset_role": "qa_canary",
        "exit_status": 0,
        "stage": "complete_copied_and_verified",
        "payload_sha256": str(config["payload_sha256"]),
        "expected_payload_sha256": str(config["payload_sha256"]),
        "delphes_card_sha256": EXPECTED_CARD_SHA256,
        "lhe_events": 100,
        "hepmc_events": 100,
        "root_events": 100,
    }

    for key, expected in required_receipt_values.items():
        observed = receipt.get(key)

        require(
            observed == expected,
            (
                f"{family}: receipt mismatch for {key}: "
                f"{observed!r} != {expected!r}"
            ),
        )

    expected_pythia_seed = (
        int(config["seed"]) * 37 + 17
    ) % 900_000_000

    if expected_pythia_seed == 0:
        expected_pythia_seed = 1

    require(
        receipt.get("pythia_seed") == expected_pythia_seed,
        f"{family}: wrong Pythia seed",
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

    require(
        '+CountTowardBackgroundTarget = False'
        in submit_text,
        f"{family}: canary count-zero classad missing",
    )

    require(
        "qa_canary" in submit_text,
        f"{family}: QA canary role missing from submit file",
    )

    payload = (
        store
        / "condor_payloads"
        / "final_background_hbb_force_payloads_20260722_v1"
        / f"{family}_inputs.tar.gz"
    )

    require(
        payload.is_file(),
        f"{family}: missing local payload {payload}",
    )

    require(
        sha256_file(payload)
        == str(config["payload_sha256"]),
        f"{family}: local payload hash mismatch",
    )

    payload_manifest = parse_manifest_from_payload(
        payload
    )

    require(
        payload_manifest.get("family") == family,
        f"{family}: payload manifest family mismatch",
    )

    require(
        payload_manifest.get("worker_wrapper_sha256")
        == EXPECTED_WORKER_SHA256,
        f"{family}: payload worker hash mismatch",
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
        f"{family}: invalid converter hash in payload",
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
        / f"{family}_bundle.tar.gz"
    )

    xrd_url = (
        EOS_HOST
        + "/"
        + remote_bundle
    )

    subprocess.run(
        [
            "xrdcp",
            "-f",
            "--nopbar",
            xrd_url,
            str(local_bundle),
        ],
        check=True,
    )

    local_bundle_sha = sha256_file(
        local_bundle
    )

    local_bundle_adler = adler32_file(
        local_bundle
    )

    require(
        local_bundle_sha
        == receipt.get("bundle_sha256"),
        f"{family}: bundle SHA-256 mismatch",
    )

    require(
        local_bundle_adler
        == str(receipt.get("bundle_adler32", "")).lower(),
        f"{family}: bundle Adler-32 mismatch",
    )

    require(
        local_bundle.stat().st_size
        == int(receipt.get("bundle_size_bytes", -1)),
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
        == local_bundle_adler,
        f"{family}: EOS checksum differs from local bundle",
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
        f"{family}: EOS size differs from local bundle",
    )

    extract_dir = temporary_root / family
    extract_dir.mkdir()

    with tarfile.open(
        local_bundle,
        "r:gz",
    ) as archive:
        archive.extractall(extract_dir)

    proc_card = find_unique(
        extract_dir,
        "proc_card_mg5.dat",
    )

    run_card = find_unique(
        extract_dir,
        "run_card.dat",
    )

    root_file = find_unique(
        extract_dir,
        "*_delphes.root",
    )

    event_summary = find_unique(
        extract_dir,
        "*_event_summary.parquet",
    )

    candidates = find_unique(
        extract_dir,
        "*_hh4b_candidates_v2.parquet",
    )

    source_lhe = find_unique(
        extract_dir,
        "unweighted_events.lhe.gz",
    )

    banner = find_unique(
        extract_dir,
        "generation_banner.txt",
    )

    provenance_path = find_unique(
        extract_dir,
        "*_provenance.json",
    )

    require(
        sha256_file(proc_card)
        == str(config["proc_card_sha256"]),
        f"{family}: process-card hash mismatch",
    )

    require(
        sha256_file(root_file)
        == receipt.get("root_sha256"),
        f"{family}: ROOT hash mismatch",
    )

    require(
        sha256_file(event_summary)
        == receipt.get("event_summary_sha256"),
        f"{family}: event-summary hash mismatch",
    )

    require(
        sha256_file(candidates)
        == receipt.get("candidate_sha256"),
        f"{family}: candidate hash mismatch",
    )

    require(
        sha256_file(source_lhe)
        == receipt.get("source_lhe_sha256"),
        f"{family}: source-LHE hash mismatch",
    )

    require(
        sha256_file(banner)
        == receipt.get("banner_sha256"),
        f"{family}: banner hash mismatch",
    )

    proc_text = proc_card.read_text(
        errors="replace"
    )

    run_text = run_card.read_text(
        errors="replace"
    )

    expected_run_values = {
        "nevents": 100.0,
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

    normalized_proc = " ".join(
        proc_text.split()
    )

    expected_process = " ".join(
        str(config["process_text"]).split()
    )

    require(
        expected_process in normalized_proc,
        f"{family}: expected process definition missing",
    )

    if family == "ggh_hbb":
        require(
            "import model loop_sm" in normalized_proc,
            "ggh_hbb: loop_sm model declaration missing",
        )

    if family == "bbh_hbb_4fs":
        proton_lines = [
            line.strip()
            for line in proc_text.splitlines()
            if line.strip().startswith("define p =")
        ]

        require(
            proton_lines,
            "bbh_hbb_4fs: proton definition missing",
        )

        proton_tokens = set(
            proton_lines[0].split("=")[1].split()
        )

        require(
            "b" not in proton_tokens
            and "b~" not in proton_tokens,
            "bbh_hbb_4fs: incoming proton definition contains b",
        )

        require(
            math.isclose(
                run_card_value(run_text, "ptb"),
                20.0,
            ),
            "bbh_hbb_4fs: ptb is not 20",
        )

        require(
            math.isclose(
                run_card_value(run_text, "etab"),
                2.7,
            ),
            "bbh_hbb_4fs: etab is not 2.7",
        )

        require(
            math.isclose(
                run_card_value(run_text, "drbb"),
                0.4,
            ),
            "bbh_hbb_4fs: drbb is not 0.4",
        )

    lhe_events = parse_lhe_events(
        source_lhe
    )

    require(
        len(lhe_events) == 100,
        f"{family}: source LHE does not contain 100 events",
    )

    for index, particles in enumerate(lhe_events):
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
            25 in final_ids,
            f"{family}: event {index} lacks final-state Higgs",
        )

        if family == "ggh_hbb":
            require(
                len(incoming_ids) == 2
                and all(pid == 21 for pid in incoming_ids),
                f"ggh_hbb: event {index} is not gg initiated",
            )

        if family == "bbh_hbb_4fs":
            require(
                all(abs(pid) != 5 for pid in incoming_ids),
                f"bbh_hbb_4fs: event {index} has incoming b",
            )

            require(
                5 in final_ids
                and -5 in final_ids,
                f"bbh_hbb_4fs: event {index} lacks final b pair",
            )

    root_entries, hbb_truth_events = (
        direct_hbb_event_count(root_file)
    )

    require(
        root_entries == 100,
        f"{family}: ROOT entry count is not 100",
    )

    require(
        hbb_truth_events == 100,
        (
            f"{family}: only {hbb_truth_events}/100 events "
            "contain direct H->bb truth"
        ),
    )

    event_frame = pd.read_parquet(
        event_summary
    )

    candidate_frame = pd.read_parquet(
        candidates
    )

    require(
        len(event_frame) == 100,
        f"{family}: event-summary row count is not 100",
    )

    require(
        len(candidate_frame)
        == int(receipt.get("candidate_rows", -1)),
        f"{family}: candidate-row count mismatch",
    )

    require(
        len(candidate_frame) >= 0,
        f"{family}: invalid candidate count",
    )

    require(
        "event_cross_section_pb"
        in event_frame.columns,
        f"{family}: event summary lacks cross section",
    )

    summary_xsec = float(
        event_frame[
            "event_cross_section_pb"
        ].median()
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
        float(config["xsec_min_pb"])
        < receipt_xsec
        < float(config["xsec_max_pb"]),
        f"{family}: cross section outside expected range",
    )

    require(
        consistent_float(
            receipt_xsec,
            summary_xsec,
        ),
        f"{family}: receipt/event-summary cross section mismatch",
    )

    require(
        consistent_float(
            receipt_xsec,
            provenance_xsec,
        ),
        f"{family}: receipt/provenance cross section mismatch",
    )

    provenance_required = {
        "campaign": campaign,
        "family": family,
        "n_events": 100,
        "seed": int(config["seed"]),
        "pythia_seed": expected_pythia_seed,
        "dataset_split": "validation",
        "dataset_role": "qa_canary",
        "lhe_events": 100,
        "hepmc_events": 100,
        "root_events": 100,
    }

    for key, expected in provenance_required.items():
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
        "cluster_id": int(receipt["cluster_id"]),
        "seed": int(receipt["seed"]),
        "pythia_seed": int(receipt["pythia_seed"]),
        "events": 100,
        "candidate_rows": int(
            receipt["candidate_rows"]
        ),
        "generator_xsec_pb": receipt_xsec,
        "payload_sha256": receipt["payload_sha256"],
        "bundle_sha256": local_bundle_sha,
        "bundle_adler32": local_bundle_adler,
        "direct_hbb_truth_events": hbb_truth_events,
        "count_toward_background_target": False,
        "technical_audit_valid": True,
        "pilot_submission_authorized": True,
        "scaleout_submission_authorized": False,
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

    require(
        not outdir.exists(),
        f"refusing to overwrite {outdir}",
    )

    outdir.mkdir(
        parents=True,
    )

    rows: list[dict[str, Any]] = []

    with tempfile.TemporaryDirectory(
        prefix="hbb_canary_audit_",
    ) as temporary:
        temporary_root = Path(temporary)

        for family, config in CONFIG.items():
            print(
                f"===== AUDITING {family} =====",
                flush=True,
            )

            row = audit_family(
                family,
                config,
                repo,
                store,
                temporary_root,
            )

            rows.append(row)

            print(
                f"{family}: PASS",
                flush=True,
            )

    require(
        len({row["seed"] for row in rows}) == 2,
        "MG5 seeds are not unique",
    )

    require(
        len({row["pythia_seed"] for row in rows}) == 2,
        "Pythia seeds are not unique",
    )

    summary = {
        "schema_version": 1,
        "status": "pass",
        "technical_subprocesses": rows,
        "canary_events_total": 200,
        "canary_events_counted_toward_background": 0,
        "hbb_canaries_final_audit_valid": True,
        "ggh_hbb_pilot10k_submission_authorized": True,
        "bbh_hbb_4fs_pilot10k_submission_authorized": True,
        "hbb_scaleout_submission_authorized": False,
        "physics_yield_authorized": False,
    }

    summary_path = (
        outdir
        / "hbb_canary_final_audit.json"
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
        / "hbb_canary_final_audit.tsv"
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
    print("GGH_HBB_CANARY100_FINAL_AUDIT_VALID")
    print("BBH_HBB_4FS_CANARY100_FINAL_AUDIT_VALID")
    print("HBB_CANARIES_FINAL_AUDIT_VALID")
    print("GGH_HBB_PILOT10K_SUBMISSION_AUTHORIZED")
    print("BBH_HBB_4FS_PILOT10K_SUBMISSION_AUTHORIZED")
    print("NO_HBB_SCALEOUT_SUBMISSION_AUTHORIZED_YET")
    print(f"summary_json={summary_path}")


if __name__ == "__main__":
    main()
