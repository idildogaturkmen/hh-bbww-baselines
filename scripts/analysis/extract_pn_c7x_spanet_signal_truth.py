#!/usr/bin/env python3
"""Extract symmetry-safe train-signal truth labels for PN-c7x.

Only the 87 exact train-signal members in the frozen 441-source registry are
eligible.  Validation/test payloads and background ROOT files are never opened.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tarfile
import tempfile
from typing import Any

import numpy as np
import pandas as pd

from scripts.analysis.pn_c7_ml_common import (
    CHECKPOINTS,
    artifact_rows,
    require,
    require_group_fold_integrity,
    require_train_only,
    seal_checkpoint,
    sha256,
    verify_checkpoint,
    write_json,
    write_tsv,
)
from scripts.analysis.pn_c7x_spanet_common import (
    adler32,
    massless_pair_pt,
    resolve_truth_partition,
    safe_archive_member,
)


REPO = Path(__file__).resolve().parents[2]
ROOT_REGISTRY_CHECKPOINT = (
    REPO
    / "docs/checkpoints/hh4b_physical_normalization_train_threeb_root_registry_and_manifest_freeze_20260731_v5"
)
ROOT_REGISTRY = ROOT_REGISTRY_CHECKPOINT / "exact_441_train_root_source_registry.tsv"
ROOT_REGISTRY_SHA256 = "1d0c1b50c82579d597cc4f75d0de5465be2d857a085d7c5f1bb0f1783493cb78"
C7S = CHECKPOINTS["c7s"]
DEFAULT_OUTPUT = Path(
    "/uscms_data/d3/iturkmen/hh4b_delphes/harvest_checkpoints/"
    "pn_c7x_signal_truth_labels_20260803_v1"
)
TRANSPORT_ID_PATTERN = re.compile(r"^[A-Za-z0-9_.-]+$")
ROOT_BRANCH_OPTIONS = {
    "jet_pt": ("Jet.PT", "Jet/Jet.PT"),
    "jet_eta": ("Jet.Eta", "Jet/Jet.Eta"),
    "jet_phi": ("Jet.Phi", "Jet/Jet.Phi"),
    "particle_pid": ("Particle.PID", "Particle/Particle.PID"),
    "particle_d1": ("Particle.D1", "Particle/Particle.D1"),
    "particle_d2": ("Particle.D2", "Particle/Particle.D2"),
    "particle_pt": ("Particle.PT", "Particle/Particle.PT"),
    "particle_eta": ("Particle.Eta", "Particle/Particle.Eta"),
    "particle_phi": ("Particle.Phi", "Particle/Particle.Phi"),
}


def _find_branch(tree: Any, options: tuple[str, ...]) -> str:
    keys = set(tree.keys())
    for option in options:
        if option in keys:
            return option
    final = {option.split("/")[-1].lower() for option in options}
    matches = [key for key in tree.keys() if key.split("/")[-1].lower() in final]
    require(len(matches) == 1, f"ROOT branch resolution failed for {options}: {matches}")
    return matches[0]


def verify_bundle(bundle: Path, registry_row: pd.Series) -> dict[str, Any]:
    expected_size = int(registry_row.source_size_bytes)
    observed_size = bundle.stat().st_size
    require(observed_size == expected_size, f"bundle size drift for {registry_row.transport_id}")
    kind = str(registry_row.source_checksum_kind)
    expected = str(registry_row.source_checksum).lower()
    if kind == "bundle_adler32_size":
        observed = adler32(bundle)
        algorithm = "adler32"
    elif kind == "bundle_sha256":
        observed = sha256(bundle)
        algorithm = "sha256"
    else:
        raise RuntimeError(f"unsupported bundle checksum kind: {kind}")
    require(observed == expected, f"bundle checksum drift for {registry_row.transport_id}")
    return {
        "source_checksum_kind": kind,
        "checksum_algorithm": algorithm,
        "checksum_expected": expected,
        "checksum_observed": observed,
        "size_expected": expected_size,
        "size_observed": observed_size,
    }


def extract_exact_root(bundle: Path, registry_row: pd.Series, scratch: Path) -> tuple[Path, str]:
    expected_root = safe_archive_member(str(registry_row.root_archive_member))
    root_path = scratch / Path(expected_root).name
    require(not root_path.exists(), f"scratch collision: {root_path}")
    layout = str(registry_row.layout_rule)
    if layout == "standard_bundle_root_target_tag":
        with tarfile.open(bundle, "r:gz") as outer:
            matches = [
                member
                for member in outer.getmembers()
                if member.isfile() and member.name.lstrip("./") == expected_root.lstrip("./")
            ]
            require(len(matches) == 1, f"exact direct ROOT member not found once: {expected_root}")
            safe_archive_member(matches[0].name)
            source = outer.extractfile(matches[0])
            require(source is not None, "failed to open exact direct ROOT archive member")
            with root_path.open("xb") as destination:
                shutil.copyfileobj(source, destination, length=8 * 1024 * 1024)
    elif layout == "nested_reconstruction_bundle_source_root_basename":
        reconstruction = scratch / "reconstruction.tar.gz"
        require(not reconstruction.exists(), f"scratch collision: {reconstruction}")
        with tarfile.open(bundle, "r:gz") as outer:
            candidates = [
                member
                for member in outer.getmembers()
                if member.isfile() and member.name.lstrip("./").startswith("products/")
                and member.name.endswith("_reconstruction.tar.gz")
            ]
            require(len(candidates) == 1, f"expected one reconstruction archive, found {len(candidates)}")
            safe_archive_member(candidates[0].name)
            source = outer.extractfile(candidates[0])
            require(source is not None, "failed to open nested reconstruction archive")
            with reconstruction.open("xb") as destination:
                shutil.copyfileobj(source, destination, length=8 * 1024 * 1024)
        with tarfile.open(reconstruction, "r:gz") as inner:
            matches = [
                member
                for member in inner.getmembers()
                if member.isfile() and member.name.lstrip("./") == expected_root.lstrip("./")
            ]
            require(len(matches) == 1, f"exact nested ROOT member not found once: {expected_root}")
            safe_archive_member(matches[0].name)
            source = inner.extractfile(matches[0])
            require(source is not None, "failed to open exact nested ROOT archive member")
            with root_path.open("xb") as destination:
                shutil.copyfileobj(source, destination, length=8 * 1024 * 1024)
    else:
        raise RuntimeError(f"unsupported frozen bundle layout rule: {layout}")
    require(root_path.name == str(registry_row.root_basename), "ROOT basename contract drift")
    return root_path, sha256(root_path)


def label_source(root_path: Path, rows: pd.DataFrame, registry_row: pd.Series) -> tuple[pd.DataFrame, dict[str, Any]]:
    try:
        import uproot
        from uproot.source.file import MemmapSource
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "PN-c7x truth extraction requires uproot/awkward in "
            "/tmp/hh4b_pn_c7t_baseline_env_20260802_v1/bin/python"
        ) from exc

    require(len(rows) == int(registry_row.fourb_candidate_rows), "candidate row count drift")
    require(not rows["event"].duplicated().any(), "duplicate signal event within source")
    start = int(rows["event"].min())
    stop = int(rows["event"].max()) + 1
    output_rows: list[dict[str, Any]] = []
    with uproot.open(root_path, handler=MemmapSource) as root_file:
        require("Delphes" in root_file, "missing Delphes tree")
        tree = root_file["Delphes"]
        branches = {key: _find_branch(tree, options) for key, options in ROOT_BRANCH_OPTIONS.items()}
        require(start >= 0 and stop <= int(tree.num_entries), "candidate event index outside ROOT tree")
        arrays = tree.arrays(list(branches.values()), entry_start=start, entry_stop=stop, library="ak")
        for row in rows.sort_values("event").itertuples(index=False):
            entry = int(row.event)
            offset = entry - start
            values = {key: np.asarray(arrays[name][offset]) for key, name in branches.items()}
            raw_indices = tuple(int(getattr(row, f"j{position}_raw_index")) for position in range(1, 5))
            for position, raw_index in enumerate(raw_indices, start=1):
                require(
                    np.isclose(float(getattr(row, f"j{position}_pt")), float(values["jet_pt"][raw_index]), rtol=0.0, atol=1e-3),
                    f"candidate/ROOT jet pT mismatch for {registry_row.transport_id} event {entry}",
                )
            result = resolve_truth_partition(
                values["particle_pid"],
                values["particle_d1"],
                values["particle_d2"],
                values["particle_eta"],
                values["particle_phi"],
                values["jet_eta"],
                values["jet_phi"],
                raw_indices,
                dr_max=0.4,
            )
            truth_pts = sorted(
                massless_pair_pt(pair, values["particle_pt"], values["particle_phi"])
                for pair in result.daughter_pairs[:2]
            )
            output_rows.append({
                "registry_group_id": str(row.registry_group_id),
                "registry_member_index": int(row.registry_member_index),
                "registry_oof_fold": int(row.registry_oof_fold),
                "transport_id": str(row.transport_id),
                "event": entry,
                "candidate_row_index": int(row.candidate_row_index),
                "n_selected_jets": int(row.n_selected_jets),
                "n_extra_selected_jets": int(row.n_extra_selected_jets),
                "mhh": float(row.mhh),
                "truth_status": result.status,
                "truth_partition_label": result.label,
                "unique_higgs_daughter_pairs": result.unique_higgs_daughter_pairs,
                "compatible_partitions": result.compatible_partitions,
                "truth_higgs_pt_low": truth_pts[0] if len(truth_pts) == 2 else np.nan,
                "truth_higgs_pt_high": truth_pts[1] if len(truth_pts) == 2 else np.nan,
                "assignment_loss_mask": result.status == "matchable",
                "candidate_jet_mask_count": 4,
            })
    result_frame = pd.DataFrame(output_rows)
    status_counts = result_frame["truth_status"].value_counts().sort_index().to_dict()
    return result_frame, {
        "transport_id": str(registry_row.transport_id),
        "registry_member_index": int(rows["registry_member_index"].iloc[0]),
        "oof_fold": int(rows["registry_oof_fold"].iloc[0]),
        "root_entries": int(tree.num_entries),
        "candidate_rows": len(result_frame),
        "status_counts": {str(key): int(value) for key, value in status_counts.items()},
        "matchable_rows": int(result_frame["assignment_loss_mask"].sum()),
        "root_sha256": sha256(root_path),
        "uproot_source": "uproot.source.file.MemmapSource",
    }


def _completion_paths(source_dir: Path, transport_id: str) -> tuple[Path, Path]:
    require(bool(TRANSPORT_ID_PATTERN.fullmatch(transport_id)), f"unsafe transport ID: {transport_id}")
    return source_dir / f"{transport_id}.parquet", source_dir / f"{transport_id}.json"


def process_source(
    registry_row: pd.Series,
    rows: pd.DataFrame,
    source_dir: Path,
    scratch_root: Path,
    local_bundle: Path | None,
) -> dict[str, Any]:
    transport_id = str(registry_row.transport_id)
    parquet_path, json_path = _completion_paths(source_dir, transport_id)
    require(not parquet_path.exists() and not json_path.exists(), f"source output already exists: {transport_id}")
    with tempfile.TemporaryDirectory(prefix=f"pn_c7x_{transport_id}_", dir=scratch_root) as temporary:
        scratch = Path(temporary)
        if local_bundle is None:
            bundle = scratch / Path(str(registry_row.remote_bundle_path)).name
            command = ["xrdcp", "--nopbar", str(registry_row.remote_bundle_uri), str(bundle)]
            completed = subprocess.run(command, check=False, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
            require(completed.returncode == 0, f"xrdcp failed for {transport_id}: {completed.stdout[-2000:]}")
            transfer_mode = "xrdcp_frozen_remote_bundle"
        else:
            bundle = local_bundle
            transfer_mode = "caller_supplied_canary_bundle"
        bundle_evidence = verify_bundle(bundle, registry_row)
        root_path, root_digest = extract_exact_root(bundle, registry_row, scratch)
        labelled, root_evidence = label_source(root_path, rows, registry_row)
        partial_parquet = parquet_path.with_suffix(".parquet.partial")
        partial_json = json_path.with_suffix(".json.partial")
        require(not partial_parquet.exists() and not partial_json.exists(), f"preserved partial output blocks {transport_id}")
        labelled.to_parquet(partial_parquet, index=False, compression="zstd")
        record = {
            **root_evidence,
            **bundle_evidence,
            "remote_bundle_uri": str(registry_row.remote_bundle_uri),
            "root_archive_member": str(registry_row.root_archive_member),
            "layout_rule": str(registry_row.layout_rule),
            "root_sha256": root_digest,
            "parquet_sha256": sha256(partial_parquet),
            "transfer_mode": transfer_mode,
            "validation_payload_files_opened": 0,
            "test_or_evaluation_payload_files_opened": 0,
        }
        write_json(partial_json, record)
        os.rename(partial_parquet, parquet_path)
        os.rename(partial_json, json_path)
    return record


def load_completed(source_dir: Path, registry_row: pd.Series) -> dict[str, Any] | None:
    parquet_path, json_path = _completion_paths(source_dir, str(registry_row.transport_id))
    if not parquet_path.exists() and not json_path.exists():
        return None
    require(parquet_path.is_file() and json_path.is_file(), f"incomplete source output: {registry_row.transport_id}")
    record = json.loads(json_path.read_text())
    require(record["parquet_sha256"] == sha256(parquet_path), f"resume checksum drift: {parquet_path}")
    require(int(record["candidate_rows"]) == int(registry_row.fourb_candidate_rows), "resume row-count drift")
    return record


def prepare_staging(output: Path, resume: bool) -> Path:
    require(output.is_absolute(), "output must be absolute")
    require(output.parent.is_dir(), f"output parent does not exist: {output.parent}")
    require(not output.exists(), f"refusing to overwrite checkpoint: {output}")
    staging = output.with_name(f".{output.name}.staging")
    if staging.exists():
        require(resume, f"preserved staging exists; pass --resume after review: {staging}")
        require(not (staging / "COMPLETE").exists(), "cannot resume a completed staging directory")
    else:
        require(not resume, f"--resume requested but staging does not exist: {staging}")
        staging.mkdir()
    return staging


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--scratch-root", type=Path, default=Path("/tmp"))
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--only-transport-id")
    parser.add_argument("--local-bundle", type=Path)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    require(args.workers >= 1, "workers must be positive")
    require(args.scratch_root.is_dir(), f"scratch root missing: {args.scratch_root}")
    require(sha256(ROOT_REGISTRY) == ROOT_REGISTRY_SHA256, "exact 441-source registry identity drift")
    c7s_evidence = verify_checkpoint("c7s")

    registry = pd.read_csv(ROOT_REGISTRY, sep="\t")
    require(len(registry) == 441, "root registry member count drift")
    require(not registry["validation_or_test_payload_opened"].astype(bool).any(), "registry records validation/test access")
    signal_registry = registry[registry["sample_class"].eq("signal")].copy().sort_values("transport_row_index")
    require(len(signal_registry) == 87, "signal source count drift")
    training = pd.read_parquet(C7S / "tables/train_fourb_model_development.parquet")
    require_train_only(training, "registry_final_split", "c7s development table")
    require_group_fold_integrity(training)
    signal_rows = training[training["registry_sample_class"].eq("signal")].copy()
    require(len(signal_rows) == 9337, "signal candidate count drift")
    require(set(signal_rows["registry_group_id"]) == set(signal_registry["transport_id"]), "signal source coverage drift")
    if args.only_transport_id:
        signal_registry = signal_registry[signal_registry["transport_id"].eq(args.only_transport_id)]
        require(len(signal_registry) == 1, f"unknown or duplicate transport ID: {args.only_transport_id}")
    require((args.local_bundle is None) or len(signal_registry) == 1, "--local-bundle requires exactly one source")
    require(args.smoke or len(signal_registry) == 87, "partial extraction requires --smoke")

    staging = prepare_staging(args.output, args.resume)
    source_dir = staging / "source_truth"
    source_dir.mkdir(exist_ok=True)
    completed_records: dict[str, dict[str, Any]] = {}
    pending: list[tuple[pd.Series, pd.DataFrame]] = []
    for _, source in signal_registry.iterrows():
        previous = load_completed(source_dir, source) if args.resume else None
        if previous is not None:
            completed_records[str(source.transport_id)] = previous
            continue
        rows = signal_rows[signal_rows["registry_group_id"].eq(source.transport_id)].copy()
        pending.append((source, rows))

    with ThreadPoolExecutor(max_workers=min(args.workers, max(1, len(pending)))) as executor:
        futures = {
            executor.submit(process_source, source, rows, source_dir, args.scratch_root, args.local_bundle): str(source.transport_id)
            for source, rows in pending
        }
        for future in as_completed(futures):
            transport_id = futures[future]
            record = future.result()
            completed_records[transport_id] = record
            print(f"completed {len(completed_records)}/{len(signal_registry)} {transport_id}", flush=True)

    require(len(completed_records) == len(signal_registry), "source extraction did not complete")
    labelled_frames = []
    source_audit = []
    for _, source in signal_registry.iterrows():
        transport_id = str(source.transport_id)
        parquet_path, _ = _completion_paths(source_dir, transport_id)
        frame = pd.read_parquet(parquet_path)
        require(len(frame) == int(source.fourb_candidate_rows), f"final source row-count drift: {transport_id}")
        labelled_frames.append(frame)
        record = completed_records[transport_id]
        source_audit.append({
            "transport_id": transport_id,
            "registry_member_index": record["registry_member_index"],
            "oof_fold": record["oof_fold"],
            "candidate_rows": record["candidate_rows"],
            "matchable_rows": record["matchable_rows"],
            "truth_status_counts_json": json.dumps(record["status_counts"], sort_keys=True),
            "source_checksum_kind": record["source_checksum_kind"],
            "source_checksum_observed": record["checksum_observed"],
            "source_size_bytes": record["size_observed"],
            "root_sha256": record["root_sha256"],
            "source_truth_parquet_sha256": record["parquet_sha256"],
            "status": "source_truth_extraction_pass",
        })
    combined = pd.concat(labelled_frames, ignore_index=True)
    require(not combined.duplicated(["registry_group_id", "event"]).any(), "duplicate combined signal event")
    expected_rows = int(signal_registry["fourb_candidate_rows"].sum())
    require(len(combined) == expected_rows, "combined truth row-count drift")
    combined.to_parquet(staging / "train_signal_truth_labels.parquet", index=False, compression="zstd")
    write_tsv(staging / "source_truth_extraction_audit.tsv", source_audit)
    status_counts = combined["truth_status"].value_counts().sort_index()
    status_rows = [
        {
            "truth_status": str(status),
            "rows": int(count),
            "fraction_of_signal_rows": float(count / len(combined)),
            "assignment_loss_enabled": status == "matchable",
        }
        for status, count in status_counts.items()
    ]
    write_tsv(staging / "truth_status_summary.tsv", status_rows)
    jet_rows = []
    for column in ("n_selected_jets", "n_extra_selected_jets", "candidate_jet_mask_count"):
        for value, count in combined[column].value_counts().sort_index().items():
            jet_rows.append({"variable": column, "value": int(value), "rows": int(count), "fraction": float(count / len(combined))})
    write_tsv(staging / "jet_multiplicity_and_mask_distribution.tsv", jet_rows)
    source_evidence = [
        {"path": str(ROOT_REGISTRY), "bytes": ROOT_REGISTRY.stat().st_size, "sha256": sha256(ROOT_REGISTRY), "role": "exact_train_source_registry"},
        *[
            {"path": item["checkpoint_root"] + "/" + item["relative_path"], "bytes": item["bytes"], "sha256": item["sha256"], "role": "sealed_c7s_input"}
            for item in c7s_evidence
        ],
    ]
    write_tsv(staging / "source_evidence_manifest.tsv", source_evidence)
    summary = {
        "schema_version": 1,
        "status": "pn_c7x_signal_truth_smoke_pass" if args.smoke else "pn_c7x_signal_truth_labels_pass",
        "created_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "signal_sources": len(signal_registry),
        "signal_rows": len(combined),
        "truth_status_counts": {str(key): int(value) for key, value in status_counts.items()},
        "matchable_rows": int(combined["assignment_loss_mask"].sum()),
        "assignment_loss_background_rows": 0,
        "background_rows_assignment_loss_masked": 21368 if not args.smoke else None,
        "candidate_jet_mask": "exactly_four_real_candidate_jets_for_every_row",
        "truth_match_dr_max": 0.4,
        "uproot_local_source": "MemmapSource",
        "root_registry_sha256": ROOT_REGISTRY_SHA256,
        "validation_payload_files_opened": 0,
        "test_or_evaluation_payload_files_opened": 0,
        "observed_data_opened": False,
        "full_run2_prediction": False,
    }
    write_json(staging / "summary.json", summary)
    (staging / "README.md").write_text(
        "# PN-c7x train-signal truth labels\n\nChecksum-bound truth extraction for the candidate-four "
        "single-head SPA-Net study.  Delphes simulation only; train signal only; no validation, test, "
        "observed data, or background ROOT payload was opened.\n"
    )
    (staging / "RUN_CONTRACT.txt").write_text(
        "Only the exact 87 train-signal source members are eligible. Bundle checksum and size are verified "
        "before exact nested ROOT extraction. Duplicate Higgs copies are collapsed by b-daughter identity; "
        "only a unique symmetry-distinct four-jet partition within deltaR<0.4 is labelled. Background and "
        "unmatched/ambiguous signal assignment losses remain masked. Uproot local reads use MemmapSource.\n"
    )
    write_tsv(staging / "artifact_manifest.tsv", artifact_rows(staging, exclude={"artifact_manifest.tsv", "SHA256SUMS", "COMPLETE"}))
    manifest_sha = seal_checkpoint(
        staging,
        args.output,
        "pn_c7x_signal_truth_smoke_pass" if args.smoke else "pn_c7x_signal_truth_labels_pass",
    )
    print(json.dumps({"checkpoint": str(args.output), "manifest_sha256": manifest_sha, **summary}, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
