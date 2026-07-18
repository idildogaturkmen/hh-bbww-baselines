#!/usr/bin/env python3

"""Stream and validate a multi-shard inclusive-QCD adaptive campaign.

Only one compressed EOS bundle and its extracted contents exist locally at a
time.  The persistent outputs contain checks, checksums, and sufficient
statistics, never duplicated HepMC or ROOT products.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import math
import os
import re
import shutil
import subprocess
import sys
import tarfile
import tempfile
import zlib
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Any, Iterable

import numpy as np
import pandas as pd


EXPECTED_BINS = [
    (0, 50, 75),
    (1, 75, 100),
    (2, 100, 200),
    (3, 200, 300),
    (4, 300, 500),
    (5, 500, 700),
    (6, 700, 1000),
    (7, 1000, 0),
]
REGIONS = [
    "generated",
    "atleast_4j",
    "exactly_2b",
    "exactly_3b",
    "atleast_4b",
    "hh_candidate",
    "rhh_lt80",
    "rhh_lt50",
]
ZERO_COUNT_CONFIDENCE = 0.95
ALLOWED_TOP_LEVEL = {"root", "hepmc", "parquet", "logs", "metadata"}
DEFAULT_LCG_SETUP = Path(
    "/cvmfs/sft.cern.ch/lcg/views/LCG_106/x86_64-el9-gcc13-opt/setup.sh"
)


@dataclass
class ShardStats:
    manifest: dict[str, Any]
    row: dict[str, Any]
    regions: dict[str, dict[str, float | int]]
    checks: dict[str, bool]


def utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def ensure_analysis_environment() -> None:
    """Re-exec under the frozen worker LCG view when login Python lacks uproot."""
    if importlib.util.find_spec("uproot") is not None:
        return
    if os.environ.get("HH4B_QCD_AUDIT_LCG_ACTIVE") == "1":
        raise SystemExit("ERROR: uproot is unavailable after loading the frozen LCG view")
    if not DEFAULT_LCG_SETUP.is_file():
        raise SystemExit(f"ERROR: missing frozen LCG setup: {DEFAULT_LCG_SETUP}")
    environment = dict(os.environ)
    environment["HH4B_QCD_AUDIT_LCG_ACTIVE"] = "1"
    command = [
        "bash",
        "-c",
        'set +u; source "$1"; set -u; shift; exec "$@"',
        "qcd-adaptive-audit",
        str(DEFAULT_LCG_SETUP),
        "python3",
        str(Path(__file__).resolve()),
        *sys.argv[1:],
    ]
    os.execvpe("bash", command, environment)


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def file_hashes(path: Path) -> tuple[str, str]:
    sha = hashlib.sha256()
    adler = 1
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            sha.update(chunk)
            adler = zlib.adler32(chunk, adler)
    return sha.hexdigest(), f"{adler & 0xFFFFFFFF:08x}"


def zero_count_upper(n_trials: int, confidence: float = ZERO_COUNT_CONFIDENCE) -> float:
    if n_trials <= 0:
        raise ValueError("zero-count bound requires a positive trial count")
    return -math.expm1(math.log1p(-confidence) / n_trials)


def normalize_pthat_max(value: Any) -> int:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return 0
    return int(value)


def expected_tag(row: dict[str, Any]) -> str:
    high = int(row["pthat_max_GeV"])
    label = (
        f"pthat{int(row['pthat_min_GeV'])}toInf"
        if high == 0
        else f"pthat{int(row['pthat_min_GeV'])}to{high}"
    )
    return (
        f"{row['campaign']}_bin{int(row['bin_id']):02d}_{label}_"
        f"shard{int(row['shard_id']):03d}_seed{int(row['seed'])}"
    )


def run_checked(command: list[str], environment: dict[str, str]) -> str:
    result = subprocess.run(
        command,
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        env=environment,
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"command failed ({result.returncode}): {' '.join(command)}\n{result.stdout}"
        )
    return result.stdout.strip()


def load_manifest(path: Path) -> list[dict[str, Any]]:
    with path.open(newline="") as handle:
        raw_rows = list(csv.DictReader(handle))
    if not raw_rows:
        raise ValueError("campaign manifest is empty")
    integer_columns = [
        "job_id",
        "bin_id",
        "pthat_min_GeV",
        "pthat_max_GeV",
        "n_events",
        "shard_id",
        "seed",
    ]
    rows: list[dict[str, Any]] = []
    for raw in raw_rows:
        row: dict[str, Any] = dict(raw)
        for name in integer_columns:
            row[name] = int(row[name])
        rows.append(row)
    rows.sort(key=lambda item: item["job_id"])
    campaigns = {row["campaign"] for row in rows}
    if len(campaigns) != 1:
        raise ValueError(f"manifest has multiple campaigns: {sorted(campaigns)}")
    if [row["job_id"] for row in rows] != list(range(len(rows))):
        raise ValueError("manifest job IDs are not consecutive from zero")
    if len({row["shard_id"] for row in rows}) != len(rows):
        raise ValueError("manifest shard IDs are not unique")
    if len({row["seed"] for row in rows}) != len(rows):
        raise ValueError("manifest seeds are not unique")
    if any(row["n_events"] <= 0 or row["n_events"] > 10_000 for row in rows):
        raise ValueError("manifest contains an invalid per-job event count")
    if any(row["dataset_split"] not in {"train", "validation", "test"} for row in rows):
        raise ValueError("manifest contains an invalid whole-shard split")
    boundaries = {
        (row["bin_id"], row["pthat_min_GeV"], row["pthat_max_GeV"])
        for row in rows
    }
    if boundaries != set(EXPECTED_BINS):
        raise ValueError(f"manifest pTHat boundaries differ from {EXPECTED_BINS}")
    return rows


def load_receipts(directory: Path) -> dict[tuple[int, int], tuple[Path, dict[str, Any]]]:
    result: dict[tuple[int, int], tuple[Path, dict[str, Any]]] = {}
    for path in sorted(directory.glob("*_receipt.json")):
        receipt = read_json(path)
        key = (int(receipt["bin_id"]), int(receipt["shard_id"]))
        if key in result:
            raise ValueError(f"duplicate receipt for bin/shard {key}")
        result[key] = (path, receipt)
    return result


def safe_extract(bundle: Path, destination: Path) -> set[str]:
    destination = destination.resolve()
    files: set[str] = set()
    with tarfile.open(bundle, "r:gz") as archive:
        normalized: dict[str, tarfile.TarInfo] = {}
        for member in archive.getmembers():
            name = member.name
            while name.startswith("./"):
                name = name[2:]
            path = PurePosixPath(name)
            canonical = str(path)
            if canonical == "." and member.isdir():
                continue
            if not name or path.is_absolute() or ".." in path.parts:
                raise ValueError(f"unsafe tar member: {member.name}")
            if canonical in normalized:
                raise ValueError(f"duplicate tar member: {member.name}")
            if not (member.isfile() or member.isdir()):
                raise ValueError(f"unsupported tar member type: {member.name}")
            if not path.parts or path.parts[0] not in ALLOWED_TOP_LEVEL:
                raise ValueError(f"unexpected tar path: {member.name}")
            normalized[canonical] = member
        for name, member in normalized.items():
            target = (destination / name).resolve()
            if target != destination and destination not in target.parents:
                raise ValueError(f"tar target escapes extraction root: {member.name}")
            if member.isdir():
                target.mkdir(parents=True, exist_ok=True)
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            source = archive.extractfile(member)
            if source is None:
                raise ValueError(f"cannot read tar member: {member.name}")
            with source, target.open("wb") as output:
                shutil.copyfileobj(source, output, length=8 * 1024 * 1024)
            files.add(name)
    return files


def count_hepmc_events(path: Path) -> int:
    with path.open("rt", errors="replace") as handle:
        return sum(1 for line in handle if line.startswith("E "))


def require_equal(checks: dict[str, bool], name: str, observed: Any, expected: Any) -> None:
    checks[name] = observed == expected


def validate_shard(
    manifest: dict[str, Any],
    receipt_path: Path,
    receipt: dict[str, Any],
    eos_host: str,
    environment: dict[str, str],
    temp_root: Path,
    cluster_id: int,
) -> ShardStats:
    import uproot

    tag = expected_tag(manifest)
    expected_remote = f"{manifest['eos_directory']}/{tag}_bundle.tar.gz"
    checks: dict[str, bool] = {}
    require_equal(checks, "receipt_campaign", receipt.get("campaign"), manifest["campaign"])
    require_equal(checks, "receipt_bin_id", int(receipt.get("bin_id", -1)), manifest["bin_id"])
    require_equal(checks, "receipt_shard_id", int(receipt.get("shard_id", -1)), manifest["shard_id"])
    require_equal(checks, "receipt_seed", int(receipt.get("seed", -1)), manifest["seed"])
    require_equal(checks, "receipt_n_events", int(receipt.get("n_events", -1)), manifest["n_events"])
    require_equal(checks, "receipt_cluster", str(receipt.get("cluster_id")), str(cluster_id))
    require_equal(checks, "receipt_exit_zero", int(receipt.get("exit_status", -1)), 0)
    require_equal(
        checks,
        "receipt_complete",
        receipt.get("stage"),
        "complete_copied_and_verified",
    )
    require_equal(checks, "receipt_remote_bundle", receipt.get("remote_bundle"), expected_remote)
    require_equal(checks, "receipt_dataset_split", receipt.get("dataset_split"), manifest["dataset_split"])
    require_equal(checks, "receipt_split_unit", receipt.get("split_assignment_unit"), "whole_shard")
    require_equal(checks, "receipt_split_salt", receipt.get("split_salt"), manifest["split_salt"])
    require_equal(checks, "receipt_payload_hash", receipt.get("payload_sha256"), manifest["payload_sha256"])
    require_equal(checks, "receipt_expected_payload_hash", receipt.get("expected_payload_sha256"), manifest["payload_sha256"])
    require_equal(checks, "receipt_card_hash", receipt.get("delphes_card_sha256"), manifest["card_sha256"])
    require_equal(checks, "receipt_wrapper_hash", receipt.get("wrapper_sha256"), manifest["wrapper_sha256"])
    receipt_adler = str(receipt.get("adler32", "")).lower()
    checks["receipt_adler_format"] = bool(re.fullmatch(r"[0-9a-f]{8}", receipt_adler))
    if not all(checks.values()):
        failed = sorted(name for name, passed in checks.items() if not passed)
        raise ValueError(f"receipt validation failed for {tag}: {failed}")

    checksum_output = run_checked(
        ["xrdfs", eos_host, "query", "checksum", expected_remote], environment
    )
    eos_adler = checksum_output.split()[-1].lower()
    if not re.fullmatch(r"[0-9a-f]{8}", eos_adler):
        raise ValueError(f"bad EOS checksum response for {expected_remote}: {checksum_output}")
    checks["eos_adler_matches_receipt"] = eos_adler == receipt_adler

    local_bundle = temp_root / f"{tag}_bundle.tar.gz"
    run_checked(
        [
            "xrdcp",
            "--nopbar",
            "--cksum",
            "adler32:print",
            f"{eos_host}//{expected_remote.lstrip('/')}",
            str(local_bundle),
        ],
        environment,
    )
    bundle_sha256, downloaded_adler = file_hashes(local_bundle)
    checks["downloaded_adler_matches_eos"] = downloaded_adler == eos_adler
    checks["downloaded_adler_matches_receipt"] = downloaded_adler == receipt_adler

    extracted = temp_root / "extracted"
    extracted.mkdir()
    archive_files = safe_extract(local_bundle, extracted)
    required_paths = {
        "hepmc": extracted / "hepmc" / f"{tag}.hepmc",
        "root": extracted / "root" / f"{tag}_delphes.root",
        "events": extracted / "parquet" / f"{tag}_event_summary.parquet",
        "candidates": extracted / "parquet" / f"{tag}_hh4b_candidates_v2.parquet",
        "generator": extracted / "metadata" / f"{tag}_generator.json",
        "provenance": extracted / "metadata" / f"{tag}_provenance.json",
        "compile": extracted / "logs" / f"{tag}_compile_arguments.txt",
    }
    for name, path in required_paths.items():
        checks[f"archive_has_{name}"] = path.is_file()
    checks["archive_only_expected_top_levels"] = all(
        PurePosixPath(name).parts[0] in ALLOWED_TOP_LEVEL for name in archive_files
    )
    if not all(checks.values()):
        failed = sorted(name for name, passed in checks.items() if not passed)
        raise ValueError(f"archive/checksum validation failed for {tag}: {failed}")

    metadata = read_json(required_paths["generator"])
    provenance = read_json(required_paths["provenance"])
    events = pd.read_parquet(required_paths["events"])
    candidates = pd.read_parquet(required_paths["candidates"])
    checks["event_parquet_readable"] = True
    checks["candidate_parquet_readable"] = True
    required_event_columns = {
        "event",
        "event_weight",
        "n_jet_pt30_eta25",
        "n_bjet_pt30_eta25",
    }
    missing = required_event_columns - set(events.columns)
    if missing:
        raise ValueError(f"event summary lacks columns {sorted(missing)} for {tag}")
    n_events = len(events)
    event_ids_float = events["event"].to_numpy(dtype=float)
    jet_counts = events["n_jet_pt30_eta25"].to_numpy(dtype=float)
    bjet_counts = events["n_bjet_pt30_eta25"].to_numpy(dtype=float)
    raw_weights = events["event_weight"].to_numpy(dtype=float)
    checks["event_rows_requested"] = n_events == manifest["n_events"]
    checks["event_values_finite"] = bool(
        np.isfinite(event_ids_float).all()
        and np.isfinite(jet_counts).all()
        and np.isfinite(bjet_counts).all()
        and np.isfinite(raw_weights).all()
    )
    checks["event_ids_integral"] = bool(np.equal(event_ids_float, np.floor(event_ids_float)).all())
    checks["jet_counts_integral"] = bool(
        np.equal(jet_counts, np.floor(jet_counts)).all()
        and np.equal(bjet_counts, np.floor(bjet_counts)).all()
    )
    event_ids = event_ids_float.astype(np.int64)
    checks["event_ids_complete_unique"] = bool(
        len(np.unique(event_ids)) == n_events and set(event_ids) == set(range(n_events))
    )
    checks["jet_counts_physical"] = bool(
        (jet_counts >= 0).all()
        and (bjet_counts >= 0).all()
        and (bjet_counts <= jet_counts).all()
    )
    checks["raw_weights_positive"] = bool((raw_weights > 0).all())

    if "event" in candidates.columns:
        candidate_values = candidates["event"].to_numpy(dtype=float)
        checks["candidate_ids_finite_integral"] = bool(
            np.isfinite(candidate_values).all()
            and np.equal(candidate_values, np.floor(candidate_values)).all()
        )
        candidate_ids = set(candidate_values.astype(np.int64))
    elif len(candidates) == 0:
        checks["candidate_ids_finite_integral"] = True
        candidate_ids = set()
    else:
        raise ValueError(f"nonempty candidate Parquet lacks event for {tag}")
    checks["candidate_ids_unique"] = len(candidate_ids) == len(candidates)
    checks["candidate_ids_subset"] = candidate_ids.issubset(set(event_ids))
    if len(candidates):
        checks["candidate_rhh_finite"] = bool(
            "r_hh" in candidates.columns
            and np.isfinite(candidates["r_hh"].to_numpy(dtype=float)).all()
        )
        rhh80_ids = set(candidates.loc[candidates["r_hh"] < 80, "event"].astype(int))
        rhh50_ids = set(candidates.loc[candidates["r_hh"] < 50, "event"].astype(int))
    else:
        checks["candidate_rhh_finite"] = True
        rhh80_ids: set[int] = set()
        rhh50_ids: set[int] = set()

    indicators = {
        "generated": np.ones(n_events, dtype=bool),
        "atleast_4j": jet_counts >= 4,
        "exactly_2b": bjet_counts == 2,
        "exactly_3b": bjet_counts == 3,
        "atleast_4b": bjet_counts >= 4,
        "hh_candidate": np.asarray([event in candidate_ids for event in event_ids], dtype=bool),
        "rhh_lt80": np.asarray([event in rhh80_ids for event in event_ids], dtype=bool),
        "rhh_lt50": np.asarray([event in rhh50_ids for event in event_ids], dtype=bool),
    }
    checks["candidate_count_matches_atleast_4b"] = int(indicators["hh_candidate"].sum()) == int(
        indicators["atleast_4b"].sum()
    )

    require_equal(checks, "metadata_n_events", int(metadata.get("n_events", -1)), n_events)
    require_equal(
        checks,
        "metadata_attempt_accounting",
        int(metadata.get("n_attempts", -1)),
        n_events + int(metadata.get("n_failed_attempts", -1)),
    )
    require_equal(checks, "metadata_seed", int(metadata.get("seed", -1)), manifest["seed"])
    require_equal(checks, "metadata_pthat_min", int(metadata.get("pthat_min_GeV", -1)), manifest["pthat_min_GeV"])
    require_equal(
        checks,
        "metadata_pthat_max",
        normalize_pthat_max(metadata.get("pthat_max_GeV")),
        manifest["pthat_max_GeV"],
    )
    require_equal(checks, "metadata_process", metadata.get("process"), "HardQCD:all")
    require_equal(checks, "metadata_physics_role", metadata.get("physics_role"), "inclusive_QCD_importance_stratum")
    require_equal(
        checks,
        "metadata_weight_convention",
        metadata.get("event_weight_convention"),
        "pythia_info_weight_normalized_to_sigma_gen",
    )
    for key, observed in [
        ("sum_event_weights", raw_weights.sum()),
        ("sum_squared_event_weights", np.square(raw_weights).sum()),
        ("min_event_weight", raw_weights.min()),
        ("max_event_weight", raw_weights.max()),
    ]:
        checks[f"metadata_{key}"] = math.isclose(
            float(metadata.get(key, math.nan)), float(observed), rel_tol=2e-6, abs_tol=1e-9
        )
    sigma_pb = float(metadata.get("sigma_gen_pb", math.nan))
    sigma_err_pb = float(metadata.get("sigma_err_pb", math.nan))
    checks["cross_section_positive_finite"] = math.isfinite(sigma_pb) and sigma_pb > 0
    checks["cross_section_error_nonnegative_finite"] = math.isfinite(sigma_err_pb) and sigma_err_pb >= 0

    hepmc_sha256 = sha256_file(required_paths["hepmc"])
    root_sha256 = sha256_file(required_paths["root"])
    hepmc_events = count_hepmc_events(required_paths["hepmc"])
    with uproot.open(required_paths["root"]) as root_file:
        root_events = int(root_file["Delphes"].num_entries)
    require_equal(checks, "hepmc_count", hepmc_events, n_events)
    require_equal(checks, "root_count", root_events, n_events)
    require_equal(checks, "provenance_tag", provenance.get("tag"), tag)
    require_equal(checks, "provenance_role", provenance.get("physics_role"), "inclusive_QCD_importance_stratum")
    require_equal(checks, "provenance_bin", int(provenance.get("bin_id", -1)), manifest["bin_id"])
    require_equal(checks, "provenance_shard", int(provenance.get("shard_id", -1)), manifest["shard_id"])
    require_equal(checks, "provenance_seed", int(provenance.get("seed", -1)), manifest["seed"])
    require_equal(checks, "provenance_split", provenance.get("dataset_split"), manifest["dataset_split"])
    require_equal(checks, "provenance_split_unit", provenance.get("split_assignment_unit"), "whole_shard")
    require_equal(checks, "provenance_split_salt", provenance.get("split_salt"), manifest["split_salt"])
    require_equal(checks, "provenance_payload", provenance.get("payload_sha256"), manifest["payload_sha256"])
    require_equal(checks, "provenance_card", provenance.get("delphes_card_sha256"), manifest["card_sha256"])
    require_equal(checks, "provenance_wrapper", provenance.get("worker_wrapper_sha256"), manifest["wrapper_sha256"])
    require_equal(checks, "provenance_hepmc_hash", provenance.get("hepmc_sha256"), hepmc_sha256)
    require_equal(checks, "provenance_root_hash", provenance.get("root_sha256"), root_sha256)
    require_equal(checks, "provenance_root_count", int(provenance.get("root_events", -1)), root_events)
    require_equal(checks, "provenance_candidate_count", int(provenance.get("candidate_rows", -1)), len(candidates))
    require_equal(checks, "provenance_generator_record", provenance.get("generator"), metadata)
    require_equal(checks, "provenance_lcg_hash", provenance.get("lcg_setup_sha256"), receipt.get("lcg_setup_sha256"))
    require_equal(checks, "provenance_pythia_config", provenance.get("pythia_configuration"), receipt.get("pythia_configuration"))
    require_equal(checks, "provenance_hepmc_config", provenance.get("hepmc3_configuration"), receipt.get("hepmc3_configuration"))
    compile_lines = required_paths["compile"].read_text().splitlines()
    compile_args = [line.split("=", 1)[1] for line in compile_lines if line.startswith("compile_arg[") and "=" in line]
    checks["compile_arguments_present"] = bool(compile_args)
    checks["compile_arguments_have_no_shell_script"] = not any(".sh" in arg for arg in compile_args)
    if not all(checks.values()):
        failed = sorted(name for name, passed in checks.items() if not passed)
        raise ValueError(f"data/provenance validation failed for {tag}: {failed}")

    raw2 = np.square(raw_weights)
    regions: dict[str, dict[str, float | int]] = {}
    for name, indicator in indicators.items():
        selected = raw_weights[indicator]
        selected2 = raw2[indicator]
        regions[name] = {
            "count": int(indicator.sum()),
            "raw_sum": float(selected.sum()),
            "raw2_sum": float(selected2.sum()),
            "min_raw": float(selected.min()) if len(selected) else 0.0,
            "max_raw": float(selected.max()) if len(selected) else 0.0,
        }
    row = {
        "job_id": manifest["job_id"],
        "bin_id": manifest["bin_id"],
        "shard_id": manifest["shard_id"],
        "seed": manifest["seed"],
        "dataset_split": manifest["dataset_split"],
        "tag": tag,
        "n_requested": manifest["n_events"],
        "n_hepmc": hepmc_events,
        "n_root": root_events,
        "n_event_summary": n_events,
        "n_candidates": len(candidates),
        "sigma_gen_pb": sigma_pb,
        "sigma_err_pb": sigma_err_pb,
        "raw_weight_sum": float(raw_weights.sum()),
        "raw_weight2_sum": float(raw2.sum()),
        "raw_weight_min": float(raw_weights.min()),
        "raw_weight_max": float(raw_weights.max()),
        "bundle_bytes": local_bundle.stat().st_size,
        "receipt_sha256": sha256_file(receipt_path),
        "bundle_sha256": bundle_sha256,
        "receipt_adler32": receipt_adler,
        "eos_adler32": eos_adler,
        "downloaded_adler32": downloaded_adler,
        "remote_bundle": expected_remote,
        "hepmc_sha256": hepmc_sha256,
        "root_sha256": root_sha256,
        "candidate_parquet_is_empty": len(candidates) == 0,
        "empty_candidate_parquet_is_valid": True,
        "all_checks_pass": True,
        "check_count": len(checks),
    }
    for region, values in regions.items():
        row[f"n_{region}"] = values["count"]
    return ShardStats(manifest=manifest, row=row, regions=regions, checks=checks)


def aggregate_region(shards: list[ShardStats], region: str) -> dict[str, Any]:
    if not shards:
        raise ValueError("cannot aggregate an empty shard collection")
    n_generated = sum(int(item.row["n_event_summary"]) for item in shards)
    sigma_pb = sum(
        float(item.row["sigma_gen_pb"]) * int(item.row["n_event_summary"])
        for item in shards
    ) / n_generated
    sigma_err_pb = math.sqrt(
        sum(
            (float(item.row["sigma_err_pb"]) * int(item.row["n_event_summary"])) ** 2
            for item in shards
        )
    ) / n_generated
    raw_sum = sum(float(item.row["raw_weight_sum"]) for item in shards)
    raw2_sum = sum(float(item.row["raw_weight2_sum"]) for item in shards)
    selected_count = sum(int(item.regions[region]["count"]) for item in shards)
    selected_raw_sum = sum(float(item.regions[region]["raw_sum"]) for item in shards)
    selected_raw2_sum = sum(float(item.regions[region]["raw2_sum"]) for item in shards)
    selected_min_raw = min(
        (
            float(item.regions[region]["min_raw"])
            for item in shards
            if int(item.regions[region]["count"]) > 0
        ),
        default=0.0,
    )
    selected_max_raw = max(float(item.regions[region]["max_raw"]) for item in shards)
    efficiency = selected_count / n_generated
    probability = selected_raw_sum / raw_sum
    yield_pb = sigma_pb * probability
    scale = sigma_pb / raw_sum
    if n_generated > 1:
        residual2 = selected_raw2_sum * (1.0 - probability) ** 2 + (
            raw2_sum - selected_raw2_sum
        ) * probability**2
        variance_pb2 = n_generated / (n_generated - 1.0) * scale**2 * residual2
    else:
        variance_pb2 = 0.0
    sum_physical2 = scale**2 * selected_raw2_sum
    ess = yield_pb**2 / sum_physical2 if sum_physical2 > 0 else 0.0
    max_physical = scale * selected_max_raw if selected_count else 0.0
    min_physical = scale * selected_min_raw if selected_count else 0.0
    max_fraction = max_physical / yield_pb if yield_pb > 0 else 0.0
    bundle_bytes = sum(int(item.row["bundle_bytes"]) for item in shards)
    return {
        "n_shards": len(shards),
        "n_generated": n_generated,
        "n_selected": selected_count,
        "unweighted_efficiency": efficiency,
        "efficiency_one_sided_95_upper_if_zero": zero_count_upper(n_generated)
        if selected_count == 0
        else math.nan,
        "sigma_gen_pb_event_count_weighted": sigma_pb,
        "sigma_err_pb_independent_combination": sigma_err_pb,
        "weighted_yield_pb": yield_pb,
        "conditional_mc_variance_pb2": variance_pb2,
        "conditional_mc_standard_error_pb": math.sqrt(max(variance_pb2, 0.0)),
        "effective_sample_size": ess,
        "mean_single_event_weight_pb": yield_pb / selected_count if selected_count else math.nan,
        "minimum_single_event_weight_pb": min_physical if selected_count else math.nan,
        "maximum_single_event_weight_pb": max_physical,
        "maximum_single_event_weight_fraction": max_fraction,
        "raw_weight_sum": raw_sum,
        "selected_raw_weight_sum": selected_raw_sum,
        "selected_raw_weight2_sum": selected_raw2_sum,
        "bundle_bytes": bundle_bytes,
        "bytes_per_generated_event": bundle_bytes / n_generated,
        "bytes_per_selected_event": bundle_bytes / selected_count if selected_count else math.nan,
    }


def aggregate_by_bin(shards: list[ShardStats]) -> list[dict[str, Any]]:
    by_bin: dict[int, list[ShardStats]] = defaultdict(list)
    for shard in shards:
        by_bin[int(shard.row["bin_id"])].append(shard)
    rows: list[dict[str, Any]] = []
    for bin_id, low, high in EXPECTED_BINS:
        group = by_bin.get(bin_id, [])
        if not group:
            continue
        for region in REGIONS:
            rows.append(
                {
                    "bin_id": bin_id,
                    "pthat_min_GeV": low,
                    "pthat_max_GeV": high,
                    "region": region,
                    **aggregate_region(group, region),
                }
            )
    variance_totals = {
        region: sum(
            float(row["conditional_mc_variance_pb2"])
            for row in rows
            if row["region"] == region
        )
        for region in REGIONS
    }
    for row in rows:
        total = variance_totals[str(row["region"])]
        row["conditional_mc_variance_fraction_of_inclusive"] = (
            float(row["conditional_mc_variance_pb2"]) / total if total > 0 else 0.0
        )
    return rows


def inclusive_rows(bin_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for region in REGIONS:
        selected = [row for row in bin_rows if row["region"] == region]
        weighted_yield = sum(float(row["weighted_yield_pb"]) for row in selected)
        variance = sum(float(row["conditional_mc_variance_pb2"]) for row in selected)
        sum_weight2 = sum(
            float(row["selected_raw_weight2_sum"])
            * (float(row["sigma_gen_pb_event_count_weighted"]) / float(row["raw_weight_sum"])) ** 2
            for row in selected
        )
        max_weight = max(
            (float(row["maximum_single_event_weight_pb"]) for row in selected),
            default=0.0,
        )
        n_generated = sum(int(row["n_generated"]) for row in selected)
        n_selected = sum(int(row["n_selected"]) for row in selected)
        bundle_bytes = sum(int(row["bundle_bytes"]) for row in selected)
        rows.append(
            {
                "region": region,
                "n_strata": len(selected),
                "n_generated": n_generated,
                "n_selected": n_selected,
                "weighted_yield_pb": weighted_yield,
                "conditional_mc_variance_pb2": variance,
                "conditional_mc_standard_error_pb": math.sqrt(max(variance, 0.0)),
                "effective_sample_size": weighted_yield**2 / sum_weight2 if sum_weight2 else 0.0,
                "maximum_single_event_weight_pb": max_weight,
                "maximum_single_event_weight_fraction": max_weight / weighted_yield if weighted_yield else 0.0,
                "bundle_bytes": bundle_bytes,
                "bytes_per_generated_event": bundle_bytes / n_generated if n_generated else math.nan,
                "bytes_per_selected_event": bundle_bytes / n_selected if n_selected else math.nan,
            }
        )
    return rows


def split_rows(shards: list[ShardStats]) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for split in ("train", "validation", "test"):
        subset = [item for item in shards if item.row["dataset_split"] == split]
        for row in aggregate_by_bin(subset):
            result.append({"dataset_split": split, **row})
    return result


def bootstrap_summary(
    shards: list[ShardStats], n_bootstrap: int, seed: int
) -> list[dict[str, Any]]:
    if n_bootstrap <= 0:
        return []
    rng = np.random.default_rng(seed)
    by_bin: dict[int, list[ShardStats]] = defaultdict(list)
    for item in shards:
        by_bin[int(item.row["bin_id"])].append(item)
    values: dict[str, list[float]] = defaultdict(list)
    for _ in range(n_bootstrap):
        sampled_by_bin: dict[int, list[ShardStats]] = {}
        for bin_id, group in by_bin.items():
            indices = rng.integers(0, len(group), size=len(group))
            sampled_by_bin[bin_id] = [group[int(index)] for index in indices]
        for region in REGIONS:
            inclusive = 0.0
            for bin_id, group in sampled_by_bin.items():
                metric = aggregate_region(group, region)["weighted_yield_pb"]
                inclusive += float(metric)
                if region in {"hh_candidate", "rhh_lt80", "rhh_lt50"}:
                    values[f"bin{bin_id}_{region}_weighted_yield_pb"].append(float(metric))
            values[f"inclusive_{region}_weighted_yield_pb"].append(inclusive)
    nominal_bins = aggregate_by_bin(shards)
    nominal_inclusive = inclusive_rows(nominal_bins)
    nominal: dict[str, float] = {
        f"inclusive_{row['region']}_weighted_yield_pb": float(row["weighted_yield_pb"])
        for row in nominal_inclusive
    }
    nominal.update(
        {
            f"bin{row['bin_id']}_{row['region']}_weighted_yield_pb": float(row["weighted_yield_pb"])
            for row in nominal_bins
            if row["region"] in {"hh_candidate", "rhh_lt80", "rhh_lt50"}
        }
    )
    rows: list[dict[str, Any]] = []
    for metric, samples in sorted(values.items()):
        array = np.asarray(samples, dtype=float)
        rows.append(
            {
                "metric": metric,
                "n_bootstrap": n_bootstrap,
                "nominal": nominal[metric],
                "bootstrap_mean": float(array.mean()),
                "bootstrap_std": float(array.std(ddof=1)) if len(array) > 1 else 0.0,
                "bootstrap_p16": float(np.percentile(array, 16)),
                "bootstrap_p50": float(np.percentile(array, 50)),
                "bootstrap_p84": float(np.percentile(array, 84)),
                "relative_std": float(array.std(ddof=1) / nominal[metric])
                if len(array) > 1 and nominal[metric] != 0
                else math.nan,
                "resampling_unit": "whole shard within each pTHat stratum",
            }
        )
    return rows


def pilot_comparison(
    bin_rows: list[dict[str, Any]], pilot_summary_path: Path | None
) -> list[dict[str, Any]]:
    if pilot_summary_path is None:
        return []
    pilot = pd.read_csv(pilot_summary_path).set_index("bin_id")
    rows: list[dict[str, Any]] = []
    for adaptive in bin_rows:
        bin_id = int(adaptive["bin_id"])
        region = str(adaptive["region"])
        old = pilot.loc[bin_id]
        mappings = {
            "n_selected": f"n_{region}",
            "unweighted_efficiency": f"eff_{region}",
            "weighted_yield_pb": f"weighted_yield_{region}_pb",
            "conditional_mc_variance_pb2": f"conditional_mc_variance_{region}_pb2",
            "effective_sample_size": f"effective_sample_size_{region}",
            "maximum_single_event_weight_fraction": f"maximum_single_event_weight_fraction_{region}",
            "bytes_per_selected_event": f"bytes_per_{region}",
        }
        for adaptive_name, pilot_name in mappings.items():
            if pilot_name not in pilot.columns:
                continue
            new_value = float(adaptive[adaptive_name])
            old_value = float(old[pilot_name])
            rows.append(
                {
                    "bin_id": bin_id,
                    "region": region,
                    "metric": adaptive_name,
                    "pilot_80k": old_value,
                    "adaptive_500k": new_value,
                    "adaptive_minus_pilot": new_value - old_value,
                    "adaptive_over_pilot": new_value / old_value if old_value != 0 else math.nan,
                }
            )
    return rows


def write_csv(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    frame = pd.DataFrame(list(rows))
    frame.to_csv(path, index=False)


def main() -> None:
    ensure_analysis_environment()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--receipt-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--proxy", type=Path, required=True)
    parser.add_argument("--cluster-id", type=int, required=True)
    parser.add_argument("--eos-host", default="root://cmseos.fnal.gov")
    parser.add_argument("--require-complete", action="store_true")
    parser.add_argument("--max-shards", type=int)
    parser.add_argument("--bootstrap", type=int, default=1000)
    parser.add_argument("--bootstrap-seed", type=int, default=500000)
    parser.add_argument("--pilot-summary", type=Path)
    parser.add_argument(
        "--temp-root",
        type=Path,
        default=Path(
            os.environ.get(
                "HH4B_QCD_AUDIT_TMPDIR",
                "/tmp",
            )
        ),
        help=(
            "Directory used for one-bundle-at-a-time temporary "
            "downloads and extraction"
        ),
    )
    args = parser.parse_args()

    if args.max_shards is not None and args.max_shards <= 0:
        parser.error("--max-shards must be positive")
    if not args.proxy.is_file():
        parser.error(f"proxy is missing: {args.proxy}")
    if not args.receipt_dir.is_dir():
        parser.error(f"receipt directory is missing: {args.receipt_dir}")

    args.temp_root.mkdir(
        parents=True,
        exist_ok=True,
    )

    if not os.access(
        args.temp_root,
        os.W_OK | os.X_OK,
    ):
        parser.error(
            f"temporary directory is not writable: {args.temp_root}"
        )

    environment = dict(os.environ)
    environment["X509_USER_PROXY"] = str(args.proxy.resolve())
    run_checked(["openssl", "x509", "-in", str(args.proxy), "-noout", "-checkend", "3600"], environment)

    manifest_rows = load_manifest(args.manifest)
    receipts = load_receipts(args.receipt_dir)
    manifest_by_key = {
        (int(row["bin_id"]), int(row["shard_id"])): row for row in manifest_rows
    }
    unexpected_receipts = sorted(set(receipts) - set(manifest_by_key))
    if unexpected_receipts:
        raise SystemExit(f"ERROR: receipts not present in manifest: {unexpected_receipts}")
    missing_receipts = sorted(set(manifest_by_key) - set(receipts))
    if args.require_complete and missing_receipts:
        raise SystemExit(f"ERROR: missing {len(missing_receipts)} receipts")
    available = [row for row in manifest_rows if (row["bin_id"], row["shard_id"]) in receipts]
    if args.max_shards is not None:
        available = available[: args.max_shards]
    if not available:
        raise SystemExit("ERROR: no completed receipts are available for validation")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    shards: list[ShardStats] = []
    failures: list[dict[str, Any]] = []
    for index, manifest in enumerate(available, start=1):
        key = (manifest["bin_id"], manifest["shard_id"])
        receipt_path, receipt = receipts[key]
        print(
            f"[{index}/{len(available)}] validating job {manifest['job_id']} "
            f"bin {manifest['bin_id']} shard {manifest['shard_id']}",
            flush=True,
        )
        try:
            with tempfile.TemporaryDirectory(
                prefix=f"qcd_adaptive_job{manifest['job_id']}_", dir=str(args.temp_root.resolve())
            ) as temporary:
                shards.append(
                    validate_shard(
                        manifest,
                        receipt_path,
                        receipt,
                        args.eos_host,
                        environment,
                        Path(temporary),
                        args.cluster_id,
                    )
                )
        except Exception as error:  # preserve compact failure evidence, then stop
            failures.append(
                {
                    "job_id": manifest["job_id"],
                    "bin_id": manifest["bin_id"],
                    "shard_id": manifest["shard_id"],
                    "error": str(error),
                }
            )
            break

    shard_rows = [item.row for item in shards]
    write_csv(args.output_dir / "qcd_adaptive_shard_validation.csv", shard_rows)
    checksum_fields = [
        "job_id",
        "bin_id",
        "shard_id",
        "remote_bundle",
        "receipt_adler32",
        "eos_adler32",
        "downloaded_adler32",
        "bundle_bytes",
        "bundle_sha256",
        "receipt_sha256",
        "all_checks_pass",
    ]
    write_csv(
        args.output_dir / "qcd_adaptive_bundle_checksums.csv",
        [{name: row[name] for name in checksum_fields} for row in shard_rows],
    )

    complete = not missing_receipts and len(shards) == len(manifest_rows) and not failures
    bin_rows = aggregate_by_bin(shards) if shards else []
    inclusive = inclusive_rows(bin_rows) if bin_rows else []
    splits = split_rows(shards) if shards else []
    bootstrap = (
        bootstrap_summary(shards, args.bootstrap, args.bootstrap_seed)
        if complete
        else []
    )
    comparison = (
        pilot_comparison(bin_rows, args.pilot_summary) if complete else []
    )
    write_csv(args.output_dir / "qcd_adaptive_bin_summary.csv", bin_rows)
    write_csv(args.output_dir / "qcd_adaptive_inclusive_summary.csv", inclusive)
    write_csv(args.output_dir / "qcd_adaptive_split_summary.csv", splits)
    write_csv(args.output_dir / "qcd_adaptive_bootstrap_summary.csv", bootstrap)
    write_csv(args.output_dir / "qcd_adaptive_vs_pilot80k.csv", comparison)

    processed_events = sum(int(item.row["n_event_summary"]) for item in shards)
    test_bins = sorted(
        {int(item.row["bin_id"]) for item in shards if item.row["dataset_split"] == "test"}
    )
    report = {
        "schema_version": 1,
        "created_utc": utc_now(),
        "status": "pass" if complete else "partial_pass" if shards and not failures else "fail",
        "campaign": manifest_rows[0]["campaign"],
        "cluster_id": args.cluster_id,
        "manifest": str(args.manifest),
        "receipt_directory": str(args.receipt_dir),
        "eos_host": args.eos_host,
        "streaming": {
            "one_bundle_at_a_time": True,
            "temporary_root": str(args.temp_root.resolve()),
            "persistent_event_artifacts": False,
            "persistent_outputs_are_compact_sufficient_statistics": True,
        },
        "campaign_checks": {
            "manifest_jobs": len(manifest_rows),
            "manifest_events": sum(int(row["n_events"]) for row in manifest_rows),
            "manifest_total_is_exactly_500000": sum(int(row["n_events"]) for row in manifest_rows) == 500_000,
            "all_jobs_at_most_10000": all(int(row["n_events"]) <= 10_000 for row in manifest_rows),
            "manifest_seed_unique": len({row["seed"] for row in manifest_rows}) == len(manifest_rows),
            "manifest_shard_unique": len({row["shard_id"] for row in manifest_rows}) == len(manifest_rows),
            "whole_shard_split": all(row["dataset_split"] in {"train", "validation", "test"} for row in manifest_rows),
            "receipts_complete": not missing_receipts,
            "all_processed_shards_pass": not failures and len(shards) == len(available),
        },
        "processed_shards": len(shards),
        "processed_events": processed_events,
        "expected_shards": len(manifest_rows),
        "expected_events": sum(int(row["n_events"]) for row in manifest_rows),
        "missing_receipt_keys": missing_receipts,
        "failures": failures,
        "shard_checks": [
            {
                "job_id": item.row["job_id"],
                "bin_id": item.row["bin_id"],
                "shard_id": item.row["shard_id"],
                "tag": item.row["tag"],
                "checks": item.checks,
                "status": "pass" if all(item.checks.values()) else "fail",
            }
            for item in shards
        ],
        "zero_candidate_policy": {
            "empty_candidate_parquet_is_valid": True,
            "physical_contribution_never_set_to_zero_from_zero_candidates": True,
            "one_sided_confidence": ZERO_COUNT_CONFIDENCE,
            "upper_bound_definition": "1 - (1-confidence)^(1/n), exact Clopper-Pearson for zero successes",
        },
        "physical_weighting": {
            "sample": "inclusive pTHat-stratified Pythia8 HardQCD",
            "stratum_cross_section": "event-count-weighted mean of independent shard sigma_gen_pb estimates",
            "per_event_weight": "sigma_stratum * raw_event_weight / sum_raw_event_weights_over_all_accepted_events_in_that_stratum_and_dataset",
            "normalization_is_never_per_shard": True,
            "strata_are_disjoint": True,
        },
        "sample_roles": {
            "physics_weighted_inference": "inclusive pTHat-stratified HardQCD",
            "ml_enrichment_only": ["QCD_bbbb", "Zbbbb", "ttbb", "other targeted samples"],
            "ml_enrichment_excluded_from_physical_normalization": True,
        },
        "splitting": {
            "assignment_unit": "whole shard",
            "final_test_used_for_allocation_or_tuning": False,
            "final_test_use": "evaluation only",
            "test_strata_present": test_bins,
            "test_has_all_eight_strata": test_bins == list(range(8)),
        },
        "bootstrap": {
            "replicates": args.bootstrap if complete else 0,
            "seed": args.bootstrap_seed,
            "resampling_unit": "whole shard within each pTHat stratum",
        },
        "artifacts": {
            "shard_validation_csv": str(args.output_dir / "qcd_adaptive_shard_validation.csv"),
            "bundle_checksums_csv": str(args.output_dir / "qcd_adaptive_bundle_checksums.csv"),
            "bin_summary_csv": str(args.output_dir / "qcd_adaptive_bin_summary.csv"),
            "inclusive_summary_csv": str(args.output_dir / "qcd_adaptive_inclusive_summary.csv"),
            "split_summary_csv": str(args.output_dir / "qcd_adaptive_split_summary.csv"),
            "bootstrap_summary_csv": str(args.output_dir / "qcd_adaptive_bootstrap_summary.csv"),
            "pilot_comparison_csv": str(args.output_dir / "qcd_adaptive_vs_pilot80k.csv"),
        },
    }
    report_path = args.output_dir / "qcd_adaptive_cross_layer_validation.json"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(
        f"status={report['status']} shards={len(shards)}/{len(manifest_rows)} "
        f"events={processed_events}/{report['expected_events']}",
        flush=True,
    )
    print(f"report={report_path}", flush=True)
    if failures or (args.require_complete and not complete):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
