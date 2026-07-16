#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
import tarfile
import zlib
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Any

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


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def adler32_file(path: Path) -> str:
    checksum = 1
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            checksum = zlib.adler32(chunk, checksum)
    return f"{checksum & 0xFFFFFFFF:08x}"


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def normalize_pthat_max(value: Any) -> int:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return 0
    return int(value)


def deterministic_split(tag: str, event: int, salt: str) -> str:
    payload = f"{salt}\0{tag}\0{event}".encode()
    value = int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")
    fraction = value / 2**64
    if fraction < 0.70:
        return "train"
    if fraction < 0.85:
        return "validation"
    return "test"


def conditional_selection_variance_pb2(
    physical_weights_pb: np.ndarray, indicator: np.ndarray
) -> float:
    n_events = len(physical_weights_pb)
    total_weight = float(physical_weights_pb.sum())
    if n_events <= 1 or total_weight <= 0:
        return 0.0
    selected_fraction = float(physical_weights_pb[indicator].sum()) / total_weight
    residuals = physical_weights_pb * (
        indicator.astype(float) - selected_fraction
    )
    return float(n_events / (n_events - 1.0) * np.square(residuals).sum())


def allocation_fractions(scores: np.ndarray, floor_fraction: float) -> np.ndarray:
    if len(scores) == 0:
        return scores
    if floor_fraction < 0 or floor_fraction * len(scores) >= 1:
        raise ValueError("allocation floor must be nonnegative and sum to less than one")
    finite_scores = np.where(np.isfinite(scores) & (scores > 0), scores, 0.0)
    if finite_scores.sum() == 0:
        finite_scores = np.ones(len(scores), dtype=float)
    residual = 1.0 - floor_fraction * len(scores)
    return floor_fraction + residual * finite_scores / finite_scores.sum()


def integer_allocation(fractions: np.ndarray, total: int) -> np.ndarray:
    raw = fractions * total
    result = np.floor(raw).astype(int)
    remainder = total - int(result.sum())
    order = np.argsort(-(raw - result), kind="stable")
    result[order[:remainder]] += 1
    return result


def effective_sample_size(weights: list[float]) -> float:
    if not weights:
        return 0.0
    array = np.asarray(weights, dtype=float)
    denominator = float(np.square(array).sum())
    if denominator == 0:
        return 0.0
    return float(array.sum() ** 2 / denominator)


def maximum_weight_fraction(weights: list[float]) -> float:
    if not weights:
        return 0.0
    total = float(sum(weights))
    if total == 0:
        return 0.0
    return float(max(weights) / total)


def count_hepmc_events(path: Path) -> int:
    with path.open("rt", errors="replace") as handle:
        return sum(1 for line in handle if line.startswith("E "))


def data_bytes_for_tag(base: Path, tag: str) -> int:
    paths = [
        base / "hepmc" / f"{tag}.hepmc",
        base / "root" / f"{tag}_delphes.root",
        base / "parquet" / f"{tag}_event_summary.parquet",
        base / "parquet" / f"{tag}_hh4b_candidates_v2.parquet",
        base / "metadata" / f"{tag}_generator.json",
        base / "metadata" / f"{tag}_provenance.json",
    ]
    return sum(path.stat().st_size for path in paths if path.is_file())


def load_manifest(path: Path | None) -> dict[int, dict[str, str]]:
    if path is None:
        return {}
    with path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    result = {}
    for row in rows:
        bin_id = int(row["bin_id"])
        if bin_id in result:
            raise SystemExit(f"ERROR: duplicate bin {bin_id} in campaign manifest")
        result[bin_id] = row
    return result


def load_receipts(directory: Path | None) -> dict[int, tuple[Path, dict[str, Any]]]:
    if directory is None:
        return {}
    result = {}
    for path in sorted(directory.glob("*_receipt.json")):
        receipt = read_json(path)
        bin_id = int(receipt["bin_id"])
        if bin_id in result:
            raise SystemExit(f"ERROR: duplicate receipt for bin {bin_id}")
        result[bin_id] = (path, receipt)
    return result


def load_eos_checksums(path: Path | None) -> dict[int, dict[str, str]]:
    if path is None:
        return {}
    with path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    result = {}
    for row in rows:
        bin_id = int(row["bin_id"])
        if bin_id in result:
            raise SystemExit(f"ERROR: duplicate bin {bin_id} in EOS checksum manifest")
        result[bin_id] = row
    return result


def require_file(path: Path) -> None:
    if not path.is_file():
        raise SystemExit(f"ERROR: missing {path}")


def check_equal(checks: dict[str, bool], name: str, left: Any, right: Any) -> None:
    checks[name] = left == right


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Audit an extracted eight-stratum QCD importance pilot."
    )
    parser.add_argument("pilot_directory", help="Directory containing extracted output trees")
    parser.add_argument(
        "--bundle-dir",
        type=Path,
        help="Directory containing the downloaded per-bin bundle tarballs",
    )
    parser.add_argument(
        "--receipt-dir",
        type=Path,
        help="Directory containing returned Condor job receipts",
    )
    parser.add_argument(
        "--campaign-manifest",
        type=Path,
        help="Immutable CSV manifest created with the Condor submit file",
    )
    parser.add_argument(
        "--eos-checksum-manifest",
        type=Path,
        help="CSV produced while independently querying and downloading EOS bundles",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Audit output directory (default: pilot directory)",
    )
    parser.add_argument(
        "--split-salt",
        default="qcd-importance-frozen-v1",
        help="Stable salt for deterministic event-level train/validation/test splits",
    )
    parser.add_argument(
        "--allocation-total",
        type=int,
        default=500_000,
        help="Event total for integer adaptive-allocation recommendations",
    )
    parser.add_argument(
        "--allocation-floor-fraction",
        type=float,
        default=0.025,
        help="Minimum fraction assigned to every pTHat stratum",
    )
    parser.add_argument(
        "--local-only",
        action="store_true",
        help="Run legacy local-file checks without producing a production cross-layer pass",
    )
    args = parser.parse_args()

    if args.allocation_total <= 0:
        parser.error("--allocation-total must be positive")

    cross_layer_paths = [
        args.bundle_dir,
        args.receipt_dir,
        args.campaign_manifest,
        args.eos_checksum_manifest,
    ]
    if any(value is not None for value in cross_layer_paths) and not all(
        value is not None for value in cross_layer_paths
    ):
        parser.error(
            "--bundle-dir, --receipt-dir, --campaign-manifest, and "
            "--eos-checksum-manifest must be supplied together"
        )
    if args.local_only and any(value is not None for value in cross_layer_paths):
        parser.error("--local-only cannot be combined with cross-layer inputs")
    if not args.local_only and not all(value is not None for value in cross_layer_paths):
        parser.error(
            "production audit requires all cross-layer inputs; use --local-only only for "
            "legacy local pilots"
        )

    base = Path(args.pilot_directory)
    output_dir = args.output_dir or base
    output_dir.mkdir(parents=True, exist_ok=True)

    metadata_files = sorted((base / "metadata").glob("*_generator.json"))
    if not metadata_files:
        raise SystemExit(f"ERROR: no generator metadata under {base}")

    manifest = load_manifest(args.campaign_manifest)
    receipts = load_receipts(args.receipt_dir)
    eos_checksums = load_eos_checksums(args.eos_checksum_manifest)
    cross_layer_enabled = all(value is not None for value in cross_layer_paths)

    rows: list[dict[str, Any]] = []
    split_frames: list[pd.DataFrame] = []
    bin_audits: list[dict[str, Any]] = []
    region_weights: dict[str, list[float]] = {region: [] for region in REGIONS}
    region_counts: dict[str, int] = {region: 0 for region in REGIONS}
    region_yields_pb: dict[str, float] = {region: 0.0 for region in REGIONS}
    region_variances_pb2: dict[str, float] = {region: 0.0 for region in REGIONS}

    for inferred_bin_id, metadata_path in enumerate(metadata_files):
        metadata = read_json(metadata_path)
        tag = metadata_path.name.removesuffix("_generator.json")
        provenance_path = base / "metadata" / f"{tag}_provenance.json"
        event_path = base / "parquet" / f"{tag}_event_summary.parquet"
        candidate_path = base / "parquet" / f"{tag}_hh4b_candidates_v2.parquet"
        hepmc_path = base / "hepmc" / f"{tag}.hepmc"
        root_path = base / "root" / f"{tag}_delphes.root"

        for required in [event_path, candidate_path]:
            require_file(required)

        events = pd.read_parquet(event_path)
        candidates = pd.read_parquet(candidate_path)
        required_event_columns = {
            "event",
            "event_weight",
            "n_jet_pt30_eta25",
            "n_bjet_pt30_eta25",
        }
        missing_event_columns = required_event_columns - set(events.columns)
        if missing_event_columns:
            raise SystemExit(
                f"ERROR: {event_path} lacks columns {sorted(missing_event_columns)}"
            )

        n_events = len(events)
        if n_events == 0:
            raise SystemExit(f"ERROR: zero generated events in {event_path}")

        event_numbers = events["event"].to_numpy(dtype=float)
        jet_counts = events["n_jet_pt30_eta25"].to_numpy(dtype=float)
        bjet_counts = events["n_bjet_pt30_eta25"].to_numpy(dtype=float)
        for name, values in [
            ("event", event_numbers),
            ("n_jet_pt30_eta25", jet_counts),
            ("n_bjet_pt30_eta25", bjet_counts),
        ]:
            if not np.isfinite(values).all():
                raise SystemExit(f"ERROR: nonfinite {name} value in {event_path}")
            if not np.equal(values, np.floor(values)).all():
                raise SystemExit(f"ERROR: nonintegral {name} value in {event_path}")
            if (values < 0).any():
                raise SystemExit(f"ERROR: negative {name} value in {event_path}")
        if (bjet_counts > jet_counts).any():
            raise SystemExit(f"ERROR: b-jet count exceeds jet count in {event_path}")
        event_numbers = event_numbers.astype(np.int64)

        if "event" in candidates.columns:
            candidate_numbers = candidates["event"].to_numpy(dtype=float)
            if not np.isfinite(candidate_numbers).all():
                raise SystemExit(f"ERROR: nonfinite candidate event ID: {candidate_path}")
            if not np.equal(candidate_numbers, np.floor(candidate_numbers)).all():
                raise SystemExit(f"ERROR: nonintegral candidate event ID: {candidate_path}")
            if (candidate_numbers < 0).any():
                raise SystemExit(f"ERROR: negative candidate event ID: {candidate_path}")
            candidate_event_ids = set(candidate_numbers.astype(np.int64))
        elif len(candidates) == 0:
            candidate_event_ids = set()
        else:
            raise SystemExit(f"ERROR: nonempty candidate table has no event column: {candidate_path}")

        if len(candidates):
            if "r_hh" not in candidates.columns:
                raise SystemExit(
                    f"ERROR: nonempty candidate table has no r_hh column: {candidate_path}"
                )
            if not np.isfinite(candidates["r_hh"].to_numpy(dtype=float)).all():
                raise SystemExit(
                    f"ERROR: nonfinite r_hh value in candidate table: {candidate_path}"
                )

        event_ids = set(event_numbers)
        if not candidate_event_ids.issubset(event_ids):
            raise SystemExit(f"ERROR: candidate event IDs are not a subset of {event_path}")

        split_frame = pd.DataFrame(
            {
                "tag": tag,
                "event": event_numbers,
                "split": [
                    deterministic_split(tag, int(event), args.split_salt)
                    for event in event_numbers
                ],
            }
        )
        split_frames.append(split_frame)
        split_by_event = dict(zip(split_frame["event"], split_frame["split"]))
        allocation_event_ids = {
            event for event, split in split_by_event.items() if split != "test"
        }
        allocation_mask = np.asarray(
            [event in allocation_event_ids for event in event_numbers], dtype=bool
        )
        rhh80_event_ids = set(
            candidates.loc[candidates["r_hh"] < 80, "event"].astype(int)
        ) if len(candidates) else set()
        rhh50_event_ids = set(
            candidates.loc[candidates["r_hh"] < 50, "event"].astype(int)
        ) if len(candidates) else set()
        region_indicators = {
            "generated": np.ones(n_events, dtype=bool),
            "atleast_4j": jet_counts >= 4,
            "exactly_2b": bjet_counts == 2,
            "exactly_3b": bjet_counts == 3,
            "atleast_4b": bjet_counts >= 4,
            "hh_candidate": np.asarray(
                [event in candidate_event_ids for event in event_numbers], dtype=bool
            ),
            "rhh_lt80": np.asarray(
                [event in rhh80_event_ids for event in event_numbers], dtype=bool
            ),
            "rhh_lt50": np.asarray(
                [event in rhh50_event_ids for event in event_numbers], dtype=bool
            ),
        }
        counts = {
            region: int(indicator.sum())
            for region, indicator in region_indicators.items()
        }
        allocation_counts = {
            "generated": int(allocation_mask.sum()),
            "atleast_4b": int(
                (allocation_mask & region_indicators["atleast_4b"]).sum()
            ),
            "rhh_lt80": int(
                (allocation_mask & region_indicators["rhh_lt80"]).sum()
            ),
        }

        sigma_pb = float(metadata["sigma_gen_pb"])
        if not math.isfinite(sigma_pb) or sigma_pb <= 0:
            raise SystemExit(f"ERROR: invalid generated cross section in {metadata_path}")
        sigma_err_pb = float(metadata["sigma_err_pb"])
        if not math.isfinite(sigma_err_pb) or sigma_err_pb < 0:
            raise SystemExit(f"ERROR: invalid generated cross-section error in {metadata_path}")

        raw_event_weights = events["event_weight"].to_numpy(dtype=float)
        if not np.isfinite(raw_event_weights).all() or (raw_event_weights <= 0).any():
            raise SystemExit(f"ERROR: invalid generator event weight in {event_path}")
        weight_convention = metadata.get("event_weight_convention")
        weight_metadata_matches = True
        if weight_convention == "pythia_info_weight_normalized_to_sigma_gen":
            for key, observed in [
                ("sum_event_weights", raw_event_weights.sum()),
                ("sum_squared_event_weights", np.square(raw_event_weights).sum()),
                ("min_event_weight", raw_event_weights.min()),
                ("max_event_weight", raw_event_weights.max()),
            ]:
                expected = float(metadata[key])
                weight_metadata_matches &= math.isclose(
                    expected, float(observed), rel_tol=2e-6, abs_tol=1e-9
                )
            physical_weights_pb = sigma_pb * raw_event_weights / raw_event_weights.sum()
        elif weight_convention is None and np.allclose(
            raw_event_weights, 1.0, rtol=0.0, atol=1e-12
        ):
            weight_convention = "legacy_unit_weights_sigma_over_n"
            physical_weights_pb = np.full(n_events, sigma_pb / n_events)
        else:
            raise SystemExit(
                f"ERROR: unsupported or missing event-weight convention in {metadata_path}"
            )
        if not weight_metadata_matches:
            raise SystemExit(
                f"ERROR: generator weight aggregates disagree with Parquet in {metadata_path}"
            )

        for region in REGIONS:
            count = counts[region]
            indicator = region_indicators[region]
            selected_weights = physical_weights_pb[indicator]
            region_counts[region] += count
            region_yields_pb[region] += float(selected_weights.sum())
            region_variances_pb2[region] += conditional_selection_variance_pb2(
                physical_weights_pb, indicator
            )
            region_weights[region].extend(selected_weights.tolist())

        p_tail_ml = (allocation_counts["rhh_lt80"] + 0.5) / (
            allocation_counts["generated"] + 1.0
        )
        allocation_raw_weights = raw_event_weights[allocation_mask]
        allocation_tail = region_indicators["rhh_lt80"][allocation_mask]
        mean_raw_weight = float(allocation_raw_weights.mean())
        p_tail_physics = (
            float(allocation_raw_weights[allocation_tail].sum()) + 0.5 * mean_raw_weight
        ) / (float(allocation_raw_weights.sum()) + mean_raw_weight)
        weighted_residual_second_moment = float(
            np.mean(
                np.square(allocation_raw_weights)
                * np.square(allocation_tail.astype(float) - p_tail_physics)
            )
        )

        bin_id = inferred_bin_id
        match = re.search(r"_bin([0-9]{2})_", tag)
        if match:
            bin_id = int(match.group(1))

        bundle_path = (
            args.bundle_dir / f"{tag}_bundle.tar.gz" if args.bundle_dir else None
        )
        stored_bytes = (
            bundle_path.stat().st_size
            if bundle_path is not None and bundle_path.is_file()
            else data_bytes_for_tag(base, tag)
        )

        row: dict[str, Any] = {
            "bin_id": bin_id,
            "tag": tag,
            "pthat_min_GeV": int(metadata["pthat_min_GeV"]),
            "pthat_max_GeV": normalize_pthat_max(metadata.get("pthat_max_GeV")),
            "seed": int(metadata["seed"]),
            "n_generated": n_events,
            "sigma_gen_pb": sigma_pb,
            "sigma_err_pb": sigma_err_pb,
            "event_weight_convention": weight_convention,
            "physical_weight_pb_mean": float(physical_weights_pb.mean()),
            "physical_weight_pb_min": float(physical_weights_pb.min()),
            "physical_weight_pb_max": float(physical_weights_pb.max()),
            "stored_bytes": stored_bytes,
            "bytes_per_generated_event": stored_bytes / n_events,
            "bytes_per_hh_candidate": stored_bytes / counts["hh_candidate"]
            if counts["hh_candidate"]
            else np.nan,
            "bytes_per_rhh_lt80": stored_bytes / counts["rhh_lt80"]
            if counts["rhh_lt80"]
            else np.nan,
            "n_train": int((split_frame["split"] == "train").sum()),
            "n_validation": int((split_frame["split"] == "validation").sum()),
            "n_test": int((split_frame["split"] == "test").sum()),
            "allocation_n_train_plus_validation": allocation_counts["generated"],
            "allocation_n_atleast_4b": allocation_counts["atleast_4b"],
            "allocation_n_rhh_lt80": allocation_counts["rhh_lt80"],
            "physics_neyman_score": sigma_pb
            * math.sqrt(weighted_residual_second_moment)
            / mean_raw_weight,
            "ml_tail_score": math.sqrt(p_tail_ml),
        }
        for region in REGIONS:
            row[f"n_{region}"] = counts[region]
            row[f"eff_{region}"] = counts[region] / n_events
            row[f"weighted_yield_{region}_pb"] = float(
                physical_weights_pb[region_indicators[region]].sum()
            )
            row[f"conditional_mc_variance_{region}_pb2"] = (
                conditional_selection_variance_pb2(
                    physical_weights_pb, region_indicators[region]
                )
            )

        checks: dict[str, bool] = {
            "metadata_event_count": int(metadata["n_events"]) == n_events,
            "metadata_attempt_accounting": int(metadata["n_attempts"])
            == n_events + int(metadata["n_failed_attempts"]),
            "event_ids_unique": events["event"].is_unique,
            "event_ids_complete": event_ids == set(range(n_events)),
            "jet_and_bjet_counts_valid": bool(
                np.isfinite(jet_counts).all()
                and np.equal(jet_counts, np.floor(jet_counts)).all()
                and np.isfinite(bjet_counts).all()
                and np.equal(bjet_counts, np.floor(bjet_counts)).all()
                and (jet_counts >= 0).all()
                and (bjet_counts >= 0).all()
                and (bjet_counts <= jet_counts).all()
            ),
            "generator_event_weights_valid": bool(
                np.isfinite(raw_event_weights).all() and (raw_event_weights > 0).all()
            ),
            "generator_weight_metadata_matches": weight_metadata_matches,
            "candidate_ids_unique": len(candidate_event_ids) == len(candidates),
            "candidate_ids_subset": candidate_event_ids.issubset(event_ids),
            "candidate_r_hh_finite": len(candidates) == 0
            or bool(np.isfinite(candidates["r_hh"].to_numpy(dtype=float)).all()),
            "candidate_count_matches_atleast_4b": counts["hh_candidate"]
            == counts["atleast_4b"],
        }

        details: dict[str, Any] = {
            "tag": tag,
            "bin_id": bin_id,
            "checks": checks,
        }

        if cross_layer_enabled:
            import uproot

            for required in [
                provenance_path,
                hepmc_path,
                root_path,
                bundle_path,
            ]:
                require_file(required)
            if bin_id not in manifest:
                raise SystemExit(f"ERROR: bin {bin_id} absent from campaign manifest")
            if bin_id not in receipts:
                raise SystemExit(f"ERROR: bin {bin_id} has no returned receipt")
            if bin_id not in eos_checksums:
                raise SystemExit(f"ERROR: bin {bin_id} has no independent EOS checksum row")

            provenance = read_json(provenance_path)
            receipt_path, receipt = receipts[bin_id]
            manifest_row = manifest[bin_id]
            eos_row = eos_checksums[bin_id]
            compile_log = base / "logs" / f"{tag}_compile_arguments.txt"
            require_file(compile_log)
            expected_archive_files = {
                f"hepmc/{tag}.hepmc": hepmc_path,
                f"root/{tag}_delphes.root": root_path,
                f"parquet/{tag}_event_summary.parquet": event_path,
                f"parquet/{tag}_hh4b_candidates_v2.parquet": candidate_path,
                f"metadata/{tag}_generator.json": metadata_path,
                f"metadata/{tag}_provenance.json": provenance_path,
                f"logs/{tag}_compile_arguments.txt": base
                / "logs"
                / f"{tag}_compile_arguments.txt",
            }
            with tarfile.open(bundle_path, "r:gz") as archive:
                archive_members: dict[str, list[tarfile.TarInfo]] = {}
                for member in archive.getmembers():
                    normalized_name = member.name
                    while normalized_name.startswith("./"):
                        normalized_name = normalized_name[2:]
                    normalized_name = str(PurePosixPath(normalized_name))
                    archive_members.setdefault(normalized_name, []).append(member)
                archive_files_match = True
                for relative_name, audited_path in expected_archive_files.items():
                    members = archive_members.get(relative_name, [])
                    if len(members) != 1 or not members[0].isfile():
                        archive_files_match = False
                        continue
                    extracted = archive.extractfile(members[0])
                    if extracted is None:
                        archive_files_match = False
                        continue
                    digest = hashlib.sha256()
                    for chunk in iter(lambda: extracted.read(8 * 1024 * 1024), b""):
                        digest.update(chunk)
                    if digest.hexdigest() != sha256_file(audited_path):
                        archive_files_match = False

            local_adler32 = adler32_file(bundle_path)
            hepmc_sha256 = sha256_file(hepmc_path)
            root_sha256 = sha256_file(root_path)
            root_events = int(uproot.open(root_path)["Delphes"].num_entries)
            compile_arguments = [
                line.split("=", 1)[1]
                for line in compile_log.read_text().splitlines()
                if line.startswith("compile_arg[") and "=" in line
            ]
            compile_executables = [
                line.split("=", 1)[1]
                for line in compile_log.read_text().splitlines()
                if line.startswith("compile_executable=")
            ]

            manifest_pthat_max = int(manifest_row["pthat_max_GeV"])
            pthat_label = (
                f"pthat{manifest_row['pthat_min_GeV']}toInf"
                if manifest_pthat_max == 0
                else f"pthat{manifest_row['pthat_min_GeV']}to{manifest_pthat_max}"
            )
            expected_tag = (
                f"{manifest_row['campaign']}_bin{bin_id:02d}_{pthat_label}_"
                f"shard{int(manifest_row['shard_id']):03d}_seed{manifest_row['seed']}"
            )

            check_equal(checks, "manifest_tag", expected_tag, tag)
            check_equal(checks, "manifest_campaign", manifest_row["campaign"], receipt["campaign"])
            check_equal(checks, "manifest_bin_id", int(manifest_row["bin_id"]), bin_id)
            check_equal(checks, "receipt_bin_id", int(receipt["bin_id"]), bin_id)
            check_equal(checks, "manifest_seed", int(manifest_row["seed"]), int(metadata["seed"]))
            check_equal(
                checks,
                "manifest_pthat_min",
                int(manifest_row["pthat_min_GeV"]),
                int(metadata["pthat_min_GeV"]),
            )
            check_equal(
                checks,
                "manifest_pthat_max",
                int(manifest_row["pthat_max_GeV"]),
                normalize_pthat_max(metadata.get("pthat_max_GeV")),
            )
            check_equal(checks, "manifest_n_events", int(manifest_row["n_events"]), n_events)
            check_equal(checks, "receipt_exit_zero", int(receipt["exit_status"]), 0)
            check_equal(
                checks,
                "receipt_complete",
                receipt["stage"],
                "complete_copied_and_verified",
            )
            check_equal(checks, "receipt_seed", int(receipt["seed"]), int(metadata["seed"]))
            check_equal(checks, "receipt_event_count", int(receipt["n_events"]), n_events)
            check_equal(checks, "bundle_adler32", receipt["adler32"].lower(), local_adler32)
            check_equal(
                checks,
                "current_eos_adler32",
                eos_row["eos_adler32"].lower(),
                local_adler32,
            )
            check_equal(
                checks,
                "acquisition_receipt_adler32",
                eos_row["receipt_adler32"].lower(),
                receipt["adler32"].lower(),
            )
            check_equal(
                checks,
                "acquisition_local_adler32",
                eos_row["local_adler32"].lower(),
                local_adler32,
            )
            check_equal(
                checks,
                "acquisition_bundle_bytes",
                int(eos_row["bundle_bytes"]),
                stored_bytes,
            )
            check_equal(checks, "acquisition_status", eos_row["status"], "pass")
            check_equal(
                checks,
                "acquisition_receipt_path",
                Path(eos_row["receipt"]).resolve(),
                receipt_path.resolve(),
            )
            check_equal(
                checks,
                "acquisition_remote_bundle",
                eos_row["remote_bundle"],
                receipt["remote_bundle"],
            )
            check_equal(
                checks,
                "acquisition_local_bundle",
                Path(eos_row["local_bundle"]).resolve(),
                bundle_path.resolve(),
            )
            check_equal(
                checks,
                "bundle_remote_basename",
                Path(receipt["remote_bundle"]).name,
                bundle_path.name,
            )
            checks["bundle_files_equal_audited_files"] = archive_files_match
            check_equal(checks, "hepmc_sha256", provenance["hepmc_sha256"], hepmc_sha256)
            check_equal(checks, "hepmc_event_count", count_hepmc_events(hepmc_path), n_events)
            check_equal(checks, "root_sha256", provenance["root_sha256"], root_sha256)
            check_equal(checks, "root_event_count", root_events, n_events)
            check_equal(
                checks,
                "provenance_candidate_count",
                int(provenance["candidate_rows"]),
                len(candidates),
            )
            check_equal(checks, "provenance_generator", provenance["generator"], metadata)
            check_equal(
                checks,
                "payload_hash_manifest_receipt",
                manifest_row["payload_sha256"],
                receipt["payload_sha256"],
            )
            check_equal(
                checks,
                "payload_hash_receipt_provenance",
                receipt["payload_sha256"],
                provenance["payload_sha256"],
            )
            check_equal(
                checks,
                "payload_hash_expected",
                receipt["payload_sha256"],
                receipt["expected_payload_sha256"],
            )
            check_equal(
                checks,
                "card_hash_manifest_receipt",
                manifest_row["card_sha256"],
                receipt["delphes_card_sha256"],
            )
            check_equal(
                checks,
                "card_hash_receipt_provenance",
                receipt["delphes_card_sha256"],
                provenance["delphes_card_sha256"],
            )
            check_equal(
                checks,
                "wrapper_hash_manifest_receipt",
                manifest_row["wrapper_sha256"],
                receipt["wrapper_sha256"],
            )
            check_equal(
                checks,
                "wrapper_hash_receipt_provenance",
                receipt["wrapper_sha256"],
                provenance["worker_wrapper_sha256"],
            )
            check_equal(
                checks,
                "lcg_hash_receipt_provenance",
                receipt["lcg_setup_sha256"],
                provenance["lcg_setup_sha256"],
            )
            check_equal(
                checks,
                "pythia_configuration",
                receipt["pythia_configuration"],
                provenance["pythia_configuration"],
            )
            check_equal(
                checks,
                "hepmc3_configuration",
                receipt["hepmc3_configuration"],
                provenance["hepmc3_configuration"],
            )
            checks["compiler_arguments_have_no_shell_script"] = not any(
                ".sh" in argument for argument in compile_arguments
            )
            checks["compiler_arguments_present"] = bool(
                compile_arguments and len(compile_executables) == 1
            )
            details.update(
                {
                    "bundle": str(bundle_path),
                    "bundle_bytes": stored_bytes,
                    "bundle_adler32": local_adler32,
                    "receipt": str(receipt_path),
                    "remote_bundle": receipt["remote_bundle"],
                }
            )

        details["status"] = "pass" if all(checks.values()) else "fail"
        bin_audits.append(details)
        rows.append(row)

    result = pd.DataFrame(rows).sort_values("pthat_min_GeV").reset_index(drop=True)
    observed_bins = [
        (int(row.bin_id), int(row.pthat_min_GeV), int(row.pthat_max_GeV))
        for row in result.itertuples()
    ]
    boundaries_pass = observed_bins == EXPECTED_BINS

    physics_fractions = allocation_fractions(
        result["physics_neyman_score"].to_numpy(dtype=float),
        args.allocation_floor_fraction,
    )
    ml_fractions = allocation_fractions(
        result["ml_tail_score"].to_numpy(dtype=float),
        args.allocation_floor_fraction,
    )
    result["physics_allocation_fraction"] = physics_fractions
    result["physics_allocation_events"] = integer_allocation(
        physics_fractions, args.allocation_total
    )
    result["ml_tail_allocation_fraction"] = ml_fractions
    result["ml_tail_allocation_events"] = integer_allocation(
        ml_fractions, args.allocation_total
    )

    summary_path = output_dir / "qcd_importance_pilot_summary.csv"
    result.to_csv(summary_path, index=False)

    split_manifest = pd.concat(split_frames, ignore_index=True)
    split_manifest_path = output_dir / "qcd_importance_split_manifest.parquet"
    split_manifest.to_parquet(split_manifest_path, index=False)
    split_counts = (
        split_manifest.groupby(["tag", "split"], sort=True)
        .size()
        .rename("n_events")
        .reset_index()
    )
    split_counts_path = output_dir / "qcd_importance_split_counts.csv"
    split_counts.to_csv(split_counts_path, index=False)

    weighted_summary: dict[str, dict[str, Any]] = {}
    for region in REGIONS:
        variance = region_variances_pb2[region]
        weighted_summary[region] = {
            "unweighted_count": region_counts[region],
            "weighted_yield_pb": region_yields_pb[region],
            "conditional_mc_variance_pb2": variance,
            "conditional_mc_standard_error_pb": math.sqrt(max(variance, 0.0)),
            "effective_sample_size": effective_sample_size(region_weights[region]),
            "maximum_single_event_weight_fraction": maximum_weight_fraction(
                region_weights[region]
            ),
        }

    total_events = int(result["n_generated"].sum())
    total_bytes = int(result["stored_bytes"].sum())
    bytes_per_generated = total_bytes / total_events
    storage_projections = {}
    for events in [500_000, 1_000_000, 5_000_000]:
        projected_bytes = bytes_per_generated * events
        storage_projections[str(events)] = {
            "projected_bytes": projected_bytes,
            "projected_GB_decimal": projected_bytes / 1e9,
            "projected_TiB": projected_bytes / 2**40,
        }

    campaign_checks = {
        "eight_expected_bins_with_disjoint_boundaries": boundaries_pass,
        "unique_seeds": result["seed"].is_unique,
        "all_bin_checks_pass": all(item["status"] == "pass" for item in bin_audits),
        "manifest_has_exact_expected_bins": set(manifest) == set(range(8))
        if cross_layer_enabled
        else True,
        "receipts_have_exact_expected_bins": set(receipts) == set(range(8))
        if cross_layer_enabled
        else True,
        "eos_checksum_manifest_has_exact_expected_bins": set(eos_checksums)
        == set(range(8))
        if cross_layer_enabled
        else True,
        "allocation_final_test_excluded": True,
        "physics_allocation_sums_to_target": int(result["physics_allocation_events"].sum())
        == args.allocation_total,
        "ml_allocation_sums_to_target": int(result["ml_tail_allocation_events"].sum())
        == args.allocation_total,
    }
    checks_pass = all(campaign_checks.values())
    status = (
        "pass"
        if checks_pass and cross_layer_enabled
        else "local_only_pass"
        if checks_pass and args.local_only
        else "fail"
    )

    report = {
        "schema_version": 1,
        "created_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "status": status,
        "inputs": {
            "pilot_directory": str(base),
            "bundle_directory": str(args.bundle_dir) if args.bundle_dir else None,
            "receipt_directory": str(args.receipt_dir) if args.receipt_dir else None,
            "campaign_manifest": str(args.campaign_manifest)
            if args.campaign_manifest
            else None,
            "eos_checksum_manifest": str(args.eos_checksum_manifest)
            if args.eos_checksum_manifest
            else None,
            "cross_layer_validation_enabled": cross_layer_enabled,
            "local_only": args.local_only,
        },
        "campaign_checks": campaign_checks,
        "bin_boundaries": [
            {"bin_id": bin_id, "pthat_min_GeV": low, "pthat_max_GeV": high}
            for bin_id, low, high in observed_bins
        ],
        "weighting": {
            "definition": (
                "physical_weight_j_pb = sigma_gen_pb * raw_event_weight_j / "
                "sum_raw_event_weights_in_stratum; this reduces to sigma_gen_pb / "
                "n_generated for unit weights"
            ),
            "strata_are_disjoint": boundaries_pass,
            "variance_definition": (
                "conditional ratio-estimator Monte Carlo variance from the empirical "
                "weighted residual second moment; generator cross-section uncertainty "
                "is reported separately"
            ),
        },
        "splitting": {
            "algorithm": "SHA256(split_salt, tag, event), 70% train / 15% validation / 15% test",
            "salt": args.split_salt,
            "allocation_scope": "train_plus_validation",
            "final_test_used_for_allocation": False,
            "manifest": str(split_manifest_path),
            "counts": str(split_counts_path),
        },
        "weighted_and_unweighted_summary": weighted_summary,
        "storage": {
            "measured_total_bytes": total_bytes,
            "measured_total_generated_events": total_events,
            "measured_bytes_per_generated_event": bytes_per_generated,
            "basis": "downloaded compressed bundle bytes"
            if cross_layer_enabled
            else "uncompressed produced data-file bytes",
            "equal_stratified_projections": storage_projections,
        },
        "adaptive_recommendations": {
            "event_total": args.allocation_total,
            "minimum_fraction_per_bin": args.allocation_floor_fraction,
            "physics": {
                "objective": "Neyman allocation for r_HH<80 target-region cross-section variance",
                "score": (
                    "sigma_i * sqrt(mean(raw_weight^2 * (indicator-p_i)^2)) / "
                    "mean(raw_weight), using a Jeffreys-smoothed weighted p_i"
                ),
                "events_by_bin": {
                    str(int(row.bin_id)): int(row.physics_allocation_events)
                    for row in result.itertuples()
                },
            },
            "ml_tail": {
                "objective": "retain broad r_HH<80 tail examples while preserving every stratum",
                "score": "sqrt(Jeffreys-smoothed p_i(r_HH<80))",
                "events_by_bin": {
                    str(int(row.bin_id)): int(row.ml_tail_allocation_events)
                    for row in result.itertuples()
                },
            },
            "final_test_split_used": False,
        },
        "bins": bin_audits,
        "artifacts": {
            "summary_csv": str(summary_path),
            "split_manifest_parquet": str(split_manifest_path),
            "split_counts_csv": str(split_counts_path),
        },
    }
    report_path = output_dir / "qcd_importance_cross_layer_validation.json"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")

    display_columns = [
        "bin_id",
        "pthat_min_GeV",
        "pthat_max_GeV",
        "n_generated",
        "sigma_gen_pb",
        "n_atleast_4j",
        "n_exactly_2b",
        "n_exactly_3b",
        "n_atleast_4b",
        "n_rhh_lt80",
        "n_rhh_lt50",
        "physics_allocation_events",
        "ml_tail_allocation_events",
    ]
    print("\n===== QCD IMPORTANCE PILOT =====")
    print(
        result[display_columns].to_string(
            index=False, float_format=lambda value: f"{value:.6g}"
        )
    )
    print(f"\nCross-layer status: {status}")
    print(f"Wrote: {summary_path}")
    print(f"Wrote: {report_path}")
    print(f"Wrote: {split_manifest_path}")

    if not checks_pass:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
