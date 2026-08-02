#!/usr/bin/env python3
"""Freeze final train-only HH4b baseline inputs after PN-c7q/PN-c7r."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import stat
import sys
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

REPO = Path("/uscms_data/d3/iturkmen/repos/hh-bbww-baselines")
sys.path.insert(0, str(REPO / "scripts/analysis"))
from hh4b_bdt_v1_common import (  # noqa: E402
    MASS_AWARE_FEATURES,
    MASS_PLANE_BLIND_FEATURES,
    add_derived_features,
    assign_member_folds,
    build_hierarchical_weights,
    feature_quality_rows,
)

C7Q = Path("/uscms_data/d3/iturkmen/hh4b_delphes/harvest_checkpoints/pn_c7q_full441_train_physical_tables_20260802_v1")
C7R = Path("/uscms_data/d3/iturkmen/hh4b_delphes/harvest_checkpoints/pn_c7r_threeb_fourb_multijet_transfer_20260802_v1")
PINNED = {
    C7Q / "SHA256SUMS": "4356554c5cb927f2c1daa1c412f2797ad398f31df5c27525a084748da7eda273",
    C7R / "SHA256SUMS": "a663fe73d23db0e2ee856dd041916a22008c020c9b11a7eb59e3dd14acbd19c6",
    REPO / "configs/baselines/hh4b_expanded_cut_bdt_protocol_v2.json": "cc56783a4a5a19619c9dd69c1c82878a47cdf6ee6b3db24c43ca11f3a60fa636",
}


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def sha_manifest(path: Path) -> dict[str, str]:
    result: dict[str, str] = {}
    for line in path.read_text().splitlines():
        digest, rel = line.split(None, 1)
        rel = rel.lstrip("*")
        if rel.startswith("./"):
            rel = rel[2:]
        require(rel not in result, f"duplicate checksum path: {rel}")
        result[rel] = digest
    return result


def verify_checkpoint(checkpoint: Path, expected_manifest_sha: str) -> None:
    require(sha256(checkpoint / "SHA256SUMS") == expected_manifest_sha, f"checkpoint manifest mismatch: {checkpoint}")
    for rel, digest in sha_manifest(checkpoint / "SHA256SUMS").items():
        path = checkpoint / rel
        require(path.is_file() and sha256(path) == digest, f"checkpoint member mismatch: {path}")


def read_tsv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def write_tsv(path: Path, rows: list[dict[str, Any]]) -> None:
    require(bool(rows), f"empty TSV: {path}")
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), delimiter="\t", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n")


def prepare_members(source_rows: list[dict[str, str]], family_map: dict[str, str]) -> list[dict[str, Any]]:
    members: list[dict[str, Any]] = []
    for row in source_rows:
        sample = row["sample_class"]
        process = row["process_or_mode"]
        stratum = process if sample == "signal" else family_map[process]
        members.append({
            "member_index": int(row["production_row_index"]),
            "sample_class": sample,
            "stratum": stratum,
            "candidate_rows": int(row["fourb_rows"]),
            "transport_id": row["transport_id"],
            "process_or_mode": process,
        })
    require(len(members) == 441 and len({m["member_index"] for m in members}) == 441, "member registry drift")
    return members


def add_registry_fields(frame: pd.DataFrame, family_map: dict[str, str], folds: dict[int, int]) -> pd.DataFrame:
    result = add_derived_features(frame)
    result["registry_member_index"] = result["production_row_index"].astype(np.int64)
    result["registry_group_id"] = result["transport_id"]
    result["registry_training_target"] = (result["sample_class"] == "signal").astype(np.int8)
    result["registry_process_or_mode"] = result["process_or_mode"]
    result["registry_sample_class"] = result["sample_class"]
    result["registry_final_split"] = "train"
    result["registry_background_family"] = result["process_or_mode"].map(family_map).fillna("")
    result["registry_signal_mode"] = np.where(result["sample_class"] == "signal", result["process_or_mode"], "")
    result["registry_stratum"] = np.where(
        result["sample_class"] == "signal", result["registry_signal_mode"], result["registry_background_family"]
    )
    result["registry_oof_fold"] = result["registry_member_index"].map(folds).astype(np.int8)
    return result


def build(staging: Path) -> dict[str, Any]:
    verify_checkpoint(C7Q, PINNED[C7Q / "SHA256SUMS"])
    verify_checkpoint(C7R, PINNED[C7R / "SHA256SUMS"])
    config_path = REPO / "configs/baselines/hh4b_expanded_cut_bdt_protocol_v2.json"
    require(sha256(config_path) == PINNED[config_path], "frozen BDT protocol checksum mismatch")
    config = json.loads(config_path.read_text())
    family_map = config["weighting"]["background_family_mapping"]
    signal_modes = tuple(config["weighting"]["signal_mode_mapping"])
    background_families = tuple(sorted(set(family_map.values())))

    source_rows = read_tsv(C7Q / "source_materialization_registry.tsv")
    members = prepare_members(source_rows, family_map)
    fold_result = assign_member_folds(members, n_folds=5)

    threeb = pq.read_table(C7Q / "tables/train_exactly3b_promoted_run2_physical.parquet").to_pandas()
    fourb = pq.read_table(C7Q / "tables/train_at_least4b_run2_physical.parquet").to_pandas()
    require(len(threeb) == 108678 and len(fourb) == 30705, "input row-count drift")
    fourb = add_registry_fields(fourb, family_map, fold_result.assignment)
    weight_result = build_hierarchical_weights(
        fourb["registry_member_index"].to_numpy(), members,
        signal_modes=signal_modes, background_families=background_families,
        tolerance=1e-10,
    )
    fourb["development_hierarchical_weight"] = weight_result.weights
    require(np.isclose(fourb["development_hierarchical_weight"].mean(), 1.0, atol=1e-10, rtol=0), "training weight mean drift")

    quality = feature_quality_rows(fourb, MASS_AWARE_FEATURES)
    require(all(row["status"] == "pass" for row in quality), "frozen feature quality failure")

    factors = read_tsv(C7R / "transfer_factor_registry.tsv")
    nominal = {
        row["mhh_category"]: float(row["transfer_factor"])
        for row in factors if row["factor_scheme"] == "cms_cr_nominal"
    }
    require(set(nominal) == {"inclusive_mhh", "low_mhh", "high_mhh"}, "nominal transfer-factor category drift")
    threeb_qcd = threeb.loc[threeb["population_kind"] == "hard_qcd"].copy()
    require(len(threeb_qcd) == 576, "three-b transfer template row-count drift")
    threeb_qcd = add_registry_fields(threeb_qcd, family_map, fold_result.assignment)
    threeb_qcd["analysis_population_role"] = "primary_transferred_multijet_template"
    threeb_qcd["primary_projection_physical_weight_inclusive"] = (
        threeb_qcd["run2_candidate_physical_weight"] * nominal["inclusive_mhh"]
    )
    category_factor = np.where(threeb_qcd["mhh"].to_numpy() < 450.0, nominal["low_mhh"], nominal["high_mhh"])
    threeb_qcd["primary_projection_physical_weight_mhh_category"] = (
        threeb_qcd["run2_candidate_physical_weight"].to_numpy() * category_factor
    )

    fourb_primary = fourb.loc[
        (fourb["sample_class"] == "signal")
        | ((fourb["sample_class"] == "background") & (fourb["population_kind"] == "ordinary"))
    ].copy()
    fourb_primary["analysis_population_role"] = np.where(
        fourb_primary["sample_class"] == "signal", "fourb_signal", "fourb_ordinary_background"
    )
    fourb_primary["primary_projection_physical_weight_inclusive"] = fourb_primary["run2_candidate_physical_weight"]
    fourb_primary["primary_projection_physical_weight_mhh_category"] = fourb_primary["run2_candidate_physical_weight"]
    primary = pd.concat([fourb_primary, threeb_qcd], ignore_index=True, sort=False)
    require(len(primary) == 31225, "primary projection row-count drift")
    require(primary[list(MASS_AWARE_FEATURES)].notna().all().all(), "primary projection feature null")

    fourb["analysis_population_role"] = np.where(
        fourb["sample_class"] == "signal", "fourb_signal",
        np.where(fourb["population_kind"] == "hard_qcd", "secondary_direct_qcd", "fourb_ordinary_background"),
    )
    fourb["direct_projection_physical_weight"] = fourb["run2_candidate_physical_weight"]

    table_dir = staging / "tables"
    table_dir.mkdir()
    paths = {
        "fourb_training": table_dir / "train_fourb_model_development.parquet",
        "primary_projection": table_dir / "train_primary_physical_projection.parquet",
        "direct_projection": table_dir / "train_direct_qcd_secondary_projection.parquet",
    }
    pq.write_table(pa.Table.from_pandas(fourb, preserve_index=False), paths["fourb_training"], compression="zstd")
    pq.write_table(pa.Table.from_pandas(primary, preserve_index=False), paths["primary_projection"], compression="zstd")
    pq.write_table(pa.Table.from_pandas(fourb, preserve_index=False), paths["direct_projection"], compression="zstd")
    write_tsv(staging / "member_fold_registry.tsv", [
        {**member, "oof_fold": fold_result.assignment[member["member_index"]]} for member in members
    ])
    write_tsv(staging / "member_weight_audit.tsv", weight_result.member_rows)
    write_tsv(staging / "weight_summary.tsv", weight_result.summary_rows)
    write_tsv(staging / "feature_quality.tsv", quality)
    write_tsv(staging / "feature_sets.tsv", [
        {"feature_set": "mass_aware_34", "position": i, "feature": feature}
        for i, feature in enumerate(MASS_AWARE_FEATURES)
    ] + [
        {"feature_set": "mass_plane_blind_30", "position": i, "feature": feature}
        for i, feature in enumerate(MASS_PLANE_BLIND_FEATURES)
    ])

    artifacts = []
    for role, path in paths.items():
        metadata = pq.read_metadata(path)
        artifacts.append({
            "role": role, "path": path.relative_to(staging).as_posix(), "bytes": path.stat().st_size,
            "sha256": sha256(path), "rows": metadata.num_rows, "columns": metadata.num_columns,
            "split": "train", "validation_opened": False, "test_opened": False,
        })
    write_tsv(staging / "table_artifact_manifest.tsv", artifacts)
    write_tsv(staging / "source_evidence_manifest.tsv", [
        {"path": str(path), "sha256": digest, "role": "pinned_frozen_input"}
        for path, digest in PINNED.items()
    ])
    summary = {
        "schema_version": 1,
        "status": "pn_c7s_final_train_baseline_inputs_freeze_pass",
        "created_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "split": "train",
        "source_members": 441,
        "fourb_model_rows": len(fourb),
        "primary_projection_rows": len(primary),
        "primary_transferred_multijet_rows": len(threeb_qcd),
        "direct_qcd_secondary_rows": int((fourb["population_kind"] == "hard_qcd").sum()),
        "oof_folds": 5,
        "fold_algorithm": fold_result.algorithm,
        "development_weighting": config["weighting"]["name"],
        "development_weight_mean": float(fourb["development_hierarchical_weight"].mean()),
        "mass_aware_features": list(MASS_AWARE_FEATURES),
        "mass_plane_blind_features": list(MASS_PLANE_BLIND_FEATURES),
        "nominal_transfer_factors": nominal,
        "validation_payload_files_opened": 0,
        "test_or_evaluation_payload_files_opened": 0,
        "baseline_development_authorized": True,
        "validation_evaluation_authorized_at_this_gate": False,
        "table_artifacts": artifacts,
        "next_gate": "run cut, BDT, dense-DNN, and LBN-DNN train-only frozen baselines in order",
    }
    write_json(staging / "summary.json", summary)
    (staging / "RUN_CONTRACT.txt").write_text(
        "The four-b model table uses the frozen 34-feature mass-aware and 30-feature mass-plane-blind "
        "sets, five deterministic source-group OOF folds, and split-local hierarchical development "
        "weights. The primary physical projection replaces direct four-b QCD with the PN-c7r "
        "three-b transfer template; direct QCD remains a secondary projection. Validation and test "
        "remain sealed.\n"
    )
    (staging / "README.md").write_text(
        "# PN-c7s final train baseline inputs\n\nImmutable normalized train tables and frozen "
        "model-development metadata for the cut, BDT, dense-DNN, and LBN-DNN baseline sequence.\n"
    )
    return summary


def freeze(staging: Path, output: Path) -> str:
    (staging / "COMPLETE").write_text("pn_c7s_final_train_baseline_inputs_freeze_pass\n")
    files = sorted(path for path in staging.rglob("*") if path.is_file() and path.name != "SHA256SUMS")
    (staging / "SHA256SUMS").write_text(
        "\n".join(f"{sha256(path)}  ./{path.relative_to(staging).as_posix()}" for path in files) + "\n"
    )
    digest = sha256(staging / "SHA256SUMS")
    os.rename(staging, output)
    for path in sorted(output.rglob("*"), reverse=True):
        path.chmod(0o444 if path.is_file() else 0o555)
    output.chmod(0o555)
    return digest


def run(output: Path) -> dict[str, Any]:
    require(output.is_absolute() and output.parent.is_dir(), "invalid absolute output path")
    require(not output.exists(), f"refusing to overwrite output: {output}")
    staging = output.with_name(f".{output.name}.staging")
    require(not staging.exists(), f"refusing to overwrite preserved staging: {staging}")
    staging.mkdir()
    try:
        summary = build(staging)
        digest = freeze(staging, output)
        return {"output": str(output), "sha256sums_sha256": digest, **summary}
    except Exception as exc:
        try:
            write_json(staging / "FAILURE.json", {
                "status": "pn_c7s_failed_preserved_staging", "error": str(exc),
                "created_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
                "traceback": traceback.format_exc(),
            })
        except Exception:
            pass
        raise


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(run(args.output), indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
