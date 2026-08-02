#!/usr/bin/env python3
"""Materialize the sealed PN-c7q full-train physical-weight tables.

This program is intentionally specific to the frozen PN-c7p full-441 harvest and
the already-sealed Run-2 coefficient/selected-four-b checkpoints.  It fails
closed on every identity, checksum, schema, event join, normalization, and
prior-four-b-closure assertion.  Validation and test inputs are not accepted.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import shutil
import stat
import sys
import traceback
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq


REPO = Path("/uscms_data/d3/iturkmen/repos/hh-bbww-baselines")
HARVEST = Path(
    "/uscms_data/d3/iturkmen/hh4b_delphes/harvest_checkpoints/"
    "pn_c7p_full441_train_production_59954551_20260801_v2"
)
COEFFICIENT_CHECKPOINT = REPO / (
    "docs/checkpoints/"
    "hh4b_physical_normalization_run2_coefficient_analytical_closure_freeze_20260731_v2"
)
SIDECAR_CHECKPOINT = REPO / (
    "docs/checkpoints/"
    "hh4b_physical_normalization_train_candidate_physical_weight_sidecar_materialization_20260731_v1"
)
FOURB_CLOSURE_CHECKPOINT = REPO / (
    "docs/checkpoints/"
    "hh4b_physical_normalization_train_fourb_selected_yield_closure_20260731_v1"
)
FOURB_SIDECAR = Path(
    "/uscms/home/iturkmen/physical_weight_sidecars/"
    "train_candidate_run2_physical_weights_20260731_v1/"
    "train_candidate_run2_physical_weights.parquet"
)

PINNED_SHA256 = {
    HARVEST / "SHA256SUMS": "2a04797e1d718c80b47b70873f3406aa7abc866dd963b051ee280e11694dd291",
    COEFFICIENT_CHECKPOINT / "SHA256SUMS": "12a4ae0e2e5bf480963138913224d3aa32d974c159ae175af9e0d550528c495c",
    SIDECAR_CHECKPOINT / "SHA256SUMS": "95aa22f017d8bf5837a38fbd9e26c447075d360132f83af2bbd6754febf96a05",
    FOURB_CLOSURE_CHECKPOINT / "SHA256SUMS": "5d3add3b8532d9f172165beaaa5de511c2f9dc23ed6c54ca36aaa14c1bbcbd6a",
    FOURB_SIDECAR: "d5315ed70bb0298b4fc67e6e8c41437a7457b9b28b10fea2982d95b96275e2d3",
}

EXPECTED_HARVEST_STATUS = "pn_c7p_full441_train_production_formal_scientific_harvest_pass"
EXPECTED_THREEB_COLUMNS = [
    "sample", "event", "n_selected_jets", "n_selected_bjets",
    "n_extra_selected_jets", "n_extra_selected_bjets", "ht_selected_jets",
    "ht_selected_bjets", "ht_candidate_jets", "pairing",
    "pairing_combo_bjet_ranks", "pairing_score_125_125", "higgs_ordering",
    "mbb1", "mbb2", "avg_mbb", "delta_mbb", "r_hh", "r_hh_125_125",
    "r_hh_125_120", "mhh", "hh_pt", "hh_eta", "hh_phi", "h1_pt",
    "h1_eta", "h1_phi", "h2_pt", "h2_eta", "h2_phi", "h_delta_eta",
    "h_delta_phi", "h_delta_r", "h_pt_balance", "drbb1", "drbb2",
    "j1_pt", "j2_pt", "j3_pt", "j4_pt", "j1_eta", "j1_phi",
    "j1_mass", "j1_btag", "j1_flavor", "j1_raw_index",
    "j1_selected_index", "j1_bjet_rank", "j2_eta", "j2_phi", "j2_mass",
    "j2_btag", "j2_flavor", "j2_raw_index", "j2_selected_index",
    "j2_bjet_rank", "j3_eta", "j3_phi", "j3_mass", "j3_btag",
    "j3_flavor", "j3_raw_index", "j3_selected_index", "j3_bjet_rank",
    "j4_eta", "j4_phi", "j4_mass", "j4_btag", "j4_flavor",
    "j4_raw_index", "j4_selected_index", "j4_bjet_rank",
    "candidate_category", "promoted_jet_rule", "promoted_jet_raw_index",
    "promoted_jet_selected_index", "promoted_jet_pt", "promoted_jet_eta",
    "promoted_jet_phi", "promoted_jet_mass", "promoted_jet_btag",
    "promoted_jet_flavor", "n_selected_untagged_jets",
]
FOURB_COMMON_COLUMNS = EXPECTED_THREEB_COLUMNS[:72]
FOURB_OPTIONAL_SIGNAL_COLUMNS = ["analysis_sample", "source_root", "source_root_index"]

REGIONS = [
    ("baseline_all_candidates", lambda d: np.ones(len(d), dtype=bool)),
    ("cms_reference_sr_rhh125120_lt30", lambda d: d["r_hh_125_120"].to_numpy() < 30.0),
    (
        "cms_reference_cr_rhh125120_ge30_lt55",
        lambda d: (d["r_hh_125_120"].to_numpy() >= 30.0)
        & (d["r_hh_125_120"].to_numpy() < 55.0),
    ),
    (
        "cms_reference_outside_rhh125120_ge55",
        lambda d: d["r_hh_125_120"].to_numpy() >= 55.0,
    ),
    ("optimized_nominal_sr_rhh125120_lt34", lambda d: d["r_hh_125_120"].to_numpy() < 34.0),
    ("higher_purity_sr_rhh125120_lt31p5", lambda d: d["r_hh_125_120"].to_numpy() < 31.5),
    ("higher_efficiency_sr_rhh125120_lt35p5", lambda d: d["r_hh_125_120"].to_numpy() < 35.5),
]
NESTED_REGIONS = [
    (0, "baseline_all_candidates"),
    (1, "higher_efficiency_sr_rhh125120_lt35p5"),
    (2, "optimized_nominal_sr_rhh125120_lt34"),
    (3, "higher_purity_sr_rhh125120_lt31p5"),
    (4, "cms_reference_sr_rhh125120_lt30"),
]


def fail(message: str) -> None:
    raise RuntimeError(message)


def require(condition: bool, message: str) -> None:
    if not condition:
        fail(message)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def read_tsv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def write_tsv(path: Path, rows: list[dict[str, Any]], fields: list[str] | None = None) -> None:
    require(bool(rows) or bool(fields), f"cannot infer empty TSV schema: {path}")
    names = fields or list(rows[0])
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=names, delimiter="\t", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n")


def parse_sha256sums(path: Path) -> dict[str, str]:
    result: dict[str, str] = {}
    for raw in path.read_text().splitlines():
        digest, rel = raw.split(None, 1)
        rel = rel.lstrip("*")
        if rel.startswith("./"):
            rel = rel[2:]
        require(rel not in result, f"duplicate checksum entry in {path}: {rel}")
        result[rel] = digest
    return result


def verify_pinned_inputs() -> dict[str, str]:
    observed: dict[str, str] = {}
    for path, expected in PINNED_SHA256.items():
        require(path.is_file(), f"missing pinned input: {path}")
        digest = sha256(path)
        require(digest == expected, f"pinned checksum mismatch: {path}: {digest} != {expected}")
        observed[str(path)] = digest
    for checkpoint in (COEFFICIENT_CHECKPOINT, SIDECAR_CHECKPOINT, FOURB_CLOSURE_CHECKPOINT):
        manifest = parse_sha256sums(checkpoint / "SHA256SUMS")
        for rel, expected in manifest.items():
            path = checkpoint / rel
            require(path.is_file(), f"checkpoint member missing: {path}")
            require(sha256(path) == expected, f"checkpoint member checksum mismatch: {path}")
    return observed


def verify_harvest_member(path: Path, manifest: dict[str, str]) -> None:
    rel = path.relative_to(HARVEST).as_posix()
    require(rel in manifest, f"harvest member absent from frozen checksum manifest: {rel}")
    require(sha256(path) == manifest[rel], f"harvest member checksum mismatch: {rel}")


def coefficient_maps() -> tuple[dict[str, dict[str, str]], dict[str, dict[str, str]]]:
    ordinary_rows = read_tsv(COEFFICIENT_CHECKPOINT / "ordinary_process_run2_coefficient_registry.tsv")
    qcd_rows = read_tsv(COEFFICIENT_CHECKPOINT / "hard_qcd_shard_run2_coefficient_registry.tsv")
    ordinary = {row["process_or_mode"]: row for row in ordinary_rows}
    qcd = {row["source_tag"]: row for row in qcd_rows}
    require(len(ordinary) == len(ordinary_rows), "ordinary coefficient process keys are not unique")
    require(len(qcd) == len(qcd_rows), "hard-QCD coefficient source keys are not unique")
    for row in ordinary_rows + qcd_rows:
        require(row["coefficient_construction_authorized"] == "True", "unauthorized coefficient row")
        require(row["run2_analytical_yield_closure_pass"] == "True", "coefficient closure failed")
    return ordinary, qcd


def coefficient_for(
    receipt: dict[str, Any], registry: dict[str, str], ordinary: dict[str, dict[str, str]], qcd: dict[str, dict[str, str]]
) -> tuple[float, float, float]:
    if receipt["population_kind"] == "hard_qcd":
        row = qcd.get(receipt["transport_id"])
    else:
        row = ordinary.get(receipt["process_or_mode"])
    require(row is not None, f"missing coefficient row for {receipt['transport_id']}")
    cross = float(row["cross_section_coefficient_pb_per_generator_weight"])
    lumi = float(row["run2_luminosity_pb_inverse"])
    run2 = float(row["run2_yield_coefficient_per_generator_weight"])
    require(math.isclose(cross * lumi, run2, rel_tol=2e-15, abs_tol=1e-12), "coefficient*lumi closure failed")
    require(registry["population_kind"] == receipt["population_kind"], "population registry mismatch")
    require(registry["sample_class"] == receipt["sample_class"], "sample registry mismatch")
    require(registry["process_or_mode"] == receipt["process_or_mode"], "process registry mismatch")
    return cross, lumi, run2


def constant_series(value: Any, length: int, dtype: Any | None = None) -> pd.Series:
    if dtype is None:
        return pd.Series([value] * length)
    return pd.Series(np.full(length, value, dtype=dtype))


def attach_provenance(
    frame: pd.DataFrame,
    receipt: dict[str, Any],
    registry: dict[str, str],
    category: str,
    category_sha: str,
    weight_sha: str,
) -> pd.DataFrame:
    n = len(frame)
    result = frame.reset_index(drop=True).copy()
    result["candidate_row_index"] = np.arange(n, dtype=np.int64)
    result["production_row_index"] = np.full(n, int(receipt["production_row_index"]), dtype=np.int64)
    result["population_kind"] = constant_series(receipt["population_kind"], n)
    result["sample_class"] = constant_series(receipt["sample_class"], n)
    result["process_or_mode"] = constant_series(receipt["process_or_mode"], n)
    result["campaign"] = constant_series(registry["campaign"], n)
    result["production_campaign"] = constant_series(receipt["campaign"], n)
    result["transport_id"] = constant_series(receipt["transport_id"], n)
    result["source_index_or_target_index"] = np.full(
        n, int(registry["source_index_or_target_index"]), dtype=np.int64
    )
    result["transport_class"] = constant_series(receipt["transport_class"], n)
    result["source_access_mode"] = constant_series(receipt["source_access_mode"], n)
    result["final_split"] = constant_series("train", n)
    result["physical_category"] = constant_series(category, n)
    result["source_payload_sha256"] = constant_series(registry["source_payload_sha256"], n)
    result["input_source_checksum_kind"] = constant_series(receipt["source_checksum_kind"], n)
    result["input_source_checksum_expected"] = constant_series(receipt["source_checksum_expected"], n)
    result["input_root_sha256"] = constant_series(receipt["root_sha256"], n)
    result["input_category_fragment_sha256"] = constant_series(category_sha, n)
    result["input_generator_weight_fragment_sha256"] = constant_series(weight_sha, n)
    return result


def validate_candidate_invariants(
    frame: pd.DataFrame, receipt: dict[str, Any], registry: dict[str, str], category: str
) -> None:
    require(len(frame) == int(receipt[f"{category}_rows"]), f"{category} row-count mismatch")
    if len(frame) == 0:
        return
    require(frame["sample"].nunique(dropna=False) == 1, f"mixed sample values in {category}")
    require(bool(frame["sample"].iloc[0]), f"empty sample identity in {category}")
    require(not frame["event"].duplicated().any(), f"duplicate event within {category} source")
    require((frame["event"] >= 0).all(), f"negative event index in {category}")
    require((frame["event"] < int(receipt["generated_events"])).all(), f"out-of-range event in {category}")
    if category == "threeb":
        require(list(frame.columns) == EXPECTED_THREEB_COLUMNS, "three-b schema drift")
        expected_sample_argument = (
            f"{receipt['process_or_mode']}::{registry['campaign']}::"
            f"{receipt['production_row_index']}"
        )
        require(
            frame["sample"].iloc[0] == expected_sample_argument,
            "three-b frozen sample_argument identity mismatch",
        )
        require((frame["n_selected_bjets"] == 3).all(), "three-b multiplicity contract failed")
        require((frame["candidate_category"] == "exactly_3b_plus_highest_pt_untagged").all(), "category drift")
        require((frame["promoted_jet_rule"] == "highest_pt_selected_untagged").all(), "promotion rule drift")
        require((frame["n_selected_untagged_jets"] >= 1).all(), "missing promotion candidate")
    else:
        allowed = [FOURB_COMMON_COLUMNS, FOURB_COMMON_COLUMNS + FOURB_OPTIONAL_SIGNAL_COLUMNS]
        require(list(frame.columns) in allowed, "four-b schema drift")
        require((frame["n_selected_bjets"] >= 4).all(), "four-b multiplicity contract failed")


def materialize_tables(staging: Path, harvest_manifest: dict[str, str]) -> tuple[pd.DataFrame, pd.DataFrame, list[dict[str, Any]]]:
    required_harvest = [
        HARVEST / "harvest_record.json",
        HARVEST / "file_inventory.tsv",
        HARVEST / "process_source_summary.tsv",
        HARVEST / "packet/unified_train_generator_weight_transport_registry.tsv",
        HARVEST / "packet/baseline_gate_sequence.tsv",
        HARVEST / "packet/paper_style_contract.json",
        HARVEST / "packet/paper_table_contract.tsv",
        HARVEST / "packet/paper_figure_contract.tsv",
    ]
    for path in required_harvest:
        verify_harvest_member(path, harvest_manifest)

    harvest_record = json.loads((HARVEST / "harvest_record.json").read_text())
    require(harvest_record["status"] == EXPECTED_HARVEST_STATUS, "harvest status is not sealed pass")
    require(harvest_record["successful_source_rows"] == 441, "source count drift")
    require(harvest_record["threeb_rows"] == 108678, "three-b count drift")
    require(harvest_record["fourb_rows"] == 30705, "four-b count drift")
    require(harvest_record["validation_payload_files_opened"] == 0, "validation payload was opened")
    require(harvest_record["test_or_evaluation_payload_files_opened"] == 0, "test payload was opened")
    require(harvest_record["physical_weights_calculated"] == 0, "PN-c7p unexpectedly normalized")

    registry_rows = read_tsv(HARVEST / "packet/unified_train_generator_weight_transport_registry.tsv")
    registry_by_id = {row["transport_id"]: row for row in registry_rows}
    require(len(registry_rows) == 441 and len(registry_by_id) == 441, "transport registry identity failure")
    require({row["final_split"] for row in registry_rows} == {"train"}, "non-train registry row encountered")
    require({row["validation_or_test_payload_opened"] for row in registry_rows} == {"False"}, "sealed split opened")

    ordinary_coeff, qcd_coeff = coefficient_maps()
    sidecar = pq.read_table(FOURB_SIDECAR).to_pandas()
    require(len(sidecar) == 30705, "four-b sidecar row-count drift")
    require(not sidecar.duplicated(["transport_id", "event"]).any(), "four-b sidecar identity duplicates")
    sidecar_groups = {key: value.sort_values("candidate_row_index").reset_index(drop=True) for key, value in sidecar.groupby("transport_id", sort=False)}

    threeb_parts: list[pd.DataFrame] = []
    fourb_parts: list[pd.DataFrame] = []
    source_rows: list[dict[str, Any]] = []
    global_threeb_identities: set[tuple[str, int]] = set()
    global_fourb_identities: set[tuple[str, int]] = set()

    for index in range(441):
        receipt_path = HARVEST / f"receipts/row_{index:03d}_receipt.json"
        threeb_path = HARVEST / f"parquet/row_{index:03d}_threeb.parquet"
        weight_path = HARVEST / f"parquet/row_{index:03d}_weights.parquet"
        for path in (receipt_path, threeb_path, weight_path):
            verify_harvest_member(path, harvest_manifest)
        receipt = json.loads(receipt_path.read_text())
        require(receipt["status"] == "success", f"non-success receipt row {index}")
        require(receipt["production_row_index"] == index, f"receipt row identity mismatch {index}")
        require(receipt["validation_payload_files_opened"] == 0, "validation payload receipt flag")
        require(receipt["test_or_evaluation_payload_files_opened"] == 0, "test payload receipt flag")
        require(receipt["physical_weights_calculated"] == 0, "receipt contains premature weights")
        registry = registry_by_id.get(receipt["transport_id"])
        require(registry is not None, f"receipt transport absent from registry: {index}")
        require(receipt["source_live_checksum_verified"] is True, "live source checksum was not verified")
        require(receipt["threeb_output_sha256"] == harvest_manifest[threeb_path.relative_to(HARVEST).as_posix()], "three-b receipt SHA mismatch")
        require(receipt["weight_output_sha256"] == harvest_manifest[weight_path.relative_to(HARVEST).as_posix()], "weight receipt SHA mismatch")

        candidate_sha = registry["candidate_sha256"]
        candidate_path = HARVEST / f"parquet/candidate_{candidate_sha}.parquet"
        verify_harvest_member(candidate_path, harvest_manifest)
        require(sha256(candidate_path) == candidate_sha, "candidate content-address mismatch")

        threeb = pq.read_table(threeb_path).to_pandas()
        weights = pq.read_table(weight_path).to_pandas()
        fourb = pq.read_table(candidate_path).to_pandas()
        validate_candidate_invariants(threeb, receipt, registry, "threeb")
        validate_candidate_invariants(fourb, receipt, registry, "fourb")
        require(list(weights.columns) == ["transport_id", "event", "generator_nominal_weight", "generator_weight_provenance"], "weight schema drift")
        require(len(weights) == len(threeb), "three-b/weight row-count mismatch")
        require(not weights["event"].duplicated().any(), "duplicate weight events")
        require(set(weights["transport_id"]) == ({receipt["transport_id"]} if len(weights) else set()), "weight transport mismatch")
        require(set(weights["event"]) == set(threeb["event"]), "three-b/weight event-set mismatch")
        fourb_events = set(fourb["event"]) if len(fourb) else set()
        require(set(threeb["event"]).isdisjoint(fourb_events), "three-b/four-b event overlap")

        cross, lumi, run2 = coefficient_for(receipt, registry, ordinary_coeff, qcd_coeff)
        joined = threeb.merge(
            weights[["event", "generator_nominal_weight", "generator_weight_provenance"]],
            on="event",
            how="left",
            validate="one_to_one",
            sort=False,
        )
        require(not joined["generator_nominal_weight"].isna().any(), "three-b generator weight join failed")
        joined = attach_provenance(
            joined, receipt, registry, "exactly3b_promoted", receipt["threeb_output_sha256"], receipt["weight_output_sha256"]
        )
        joined["generator_weight_sign"] = np.sign(joined["generator_nominal_weight"].to_numpy()).astype(np.int8)
        joined["cross_section_coefficient_pb_per_generator_weight"] = np.full(len(joined), cross, dtype=np.float64)
        joined["generator_cross_section_contribution_pb"] = joined["generator_nominal_weight"].to_numpy(dtype=np.float64) * cross
        joined["run2_luminosity_pb_inverse"] = np.full(len(joined), lumi, dtype=np.float64)
        joined["run2_yield_coefficient_per_generator_weight"] = np.full(len(joined), run2, dtype=np.float64)
        joined["run2_candidate_physical_weight"] = joined["generator_nominal_weight"].to_numpy(dtype=np.float64) * run2
        threeb_parts.append(joined)

        existing = sidecar_groups.get(receipt["transport_id"], sidecar.iloc[0:0]).reset_index(drop=True)
        require(len(existing) == len(fourb), f"four-b sidecar/source count mismatch row {index}")
        if len(fourb):
            require(np.array_equal(existing["candidate_row_index"].to_numpy(), np.arange(len(fourb))), "candidate row index drift")
            require(np.array_equal(existing["event"].to_numpy(), fourb["event"].to_numpy()), "four-b event ordering drift")
            for field, value in (
                ("population_kind", receipt["population_kind"]),
                ("sample_class", receipt["sample_class"]),
                ("process_or_mode", receipt["process_or_mode"]),
                ("campaign", registry["campaign"]),
                ("transport_id", receipt["transport_id"]),
            ):
                require((existing[field] == value).all(), f"four-b sidecar {field} mismatch")
            require((existing["source_index_or_target_index"] == int(registry["source_index_or_target_index"])).all(), "four-b source index mismatch")
            require(np.array_equal(existing["generator_weight_sign"].to_numpy(), np.sign(existing["generator_nominal_weight"]).astype(np.int8)), "sidecar sign mismatch")
            require(np.allclose(existing["cross_section_coefficient_pb_per_generator_weight"], cross, rtol=0, atol=0), "sidecar cross coefficient mismatch")
            require(np.allclose(existing["run2_luminosity_pb_inverse"], lumi, rtol=0, atol=0), "sidecar luminosity mismatch")
            require(np.allclose(existing["run2_yield_coefficient_per_generator_weight"], run2, rtol=2e-15, atol=1e-12), "sidecar yield coefficient mismatch")
            require(
                np.allclose(
                    existing["run2_candidate_physical_weight"],
                    existing["generator_nominal_weight"] * existing["run2_yield_coefficient_per_generator_weight"],
                    rtol=2e-15,
                    atol=1e-12,
                ),
                "sidecar physical-weight formula mismatch",
            )
            four_joined = attach_provenance(fourb, receipt, registry, "at_least4b", candidate_sha, "")
            for field in (
                "generator_nominal_weight", "generator_weight_sign",
                "cross_section_coefficient_pb_per_generator_weight",
                "generator_cross_section_contribution_pb", "run2_luminosity_pb_inverse",
                "run2_yield_coefficient_per_generator_weight", "run2_candidate_physical_weight",
            ):
                four_joined[field] = existing[field].to_numpy()
            fourb_parts.append(four_joined)

        threeb_ids = {(receipt["transport_id"], int(event)) for event in threeb["event"]}
        fourb_ids = {(receipt["transport_id"], int(event)) for event in fourb_events}
        require(global_threeb_identities.isdisjoint(threeb_ids), "global duplicate three-b identity")
        require(global_fourb_identities.isdisjoint(fourb_ids), "global duplicate four-b identity")
        global_threeb_identities.update(threeb_ids)
        global_fourb_identities.update(fourb_ids)
        source_rows.append(
            {
                "production_row_index": index,
                "population_kind": receipt["population_kind"],
                "sample_class": receipt["sample_class"],
                "process_or_mode": receipt["process_or_mode"],
                "campaign": registry["campaign"],
                "transport_id": receipt["transport_id"],
                "source_index_or_target_index": int(registry["source_index_or_target_index"]),
                "transport_class": receipt["transport_class"],
                "source_access_mode": receipt["source_access_mode"],
                "generated_events": int(receipt["generated_events"]),
                "threeb_rows": len(threeb),
                "fourb_rows": len(fourb),
                "cross_section_coefficient_pb_per_generator_weight": cross,
                "run2_luminosity_pb_inverse": lumi,
                "run2_yield_coefficient_per_generator_weight": run2,
                "source_payload_sha256": registry["source_payload_sha256"],
                "source_checksum_kind": receipt["source_checksum_kind"],
                "source_checksum_expected": receipt["source_checksum_expected"],
                "root_sha256": receipt["root_sha256"],
                "threeb_fragment_sha256": receipt["threeb_output_sha256"],
                "generator_weight_fragment_sha256": receipt["weight_output_sha256"],
                "fourb_candidate_sha256": candidate_sha,
                "event_overlap": len(threeb_ids & fourb_ids),
                "validation_payload_opened": False,
                "test_payload_opened": False,
            }
        )

    require(global_threeb_identities.isdisjoint(global_fourb_identities), "global three-b/four-b overlap")
    threeb_all = pd.concat(threeb_parts, ignore_index=True, sort=False)
    fourb_all = pd.concat(fourb_parts, ignore_index=True, sort=False)
    require(len(threeb_all) == 108678, "assembled three-b count mismatch")
    require(len(fourb_all) == 30705, "assembled four-b count mismatch")
    require(len(source_rows) == 441, "assembled source registry count mismatch")

    # Reorder four-b rows to the exact sealed sidecar row order and prove the join is exact.
    fourb_all = fourb_all.set_index(["transport_id", "candidate_row_index"], drop=False)
    side_keys = pd.MultiIndex.from_frame(sidecar[["transport_id", "candidate_row_index"]])
    require(fourb_all.index.is_unique, "assembled four-b table keys are not unique")
    require(set(fourb_all.index) == set(side_keys), "assembled four-b/sidecar key-set mismatch")
    fourb_all = fourb_all.loc[side_keys].reset_index(drop=True)
    require(np.array_equal(fourb_all["event"].to_numpy(), sidecar["event"].to_numpy()), "four-b sidecar event mismatch")
    require(
        np.array_equal(
            fourb_all["run2_candidate_physical_weight"].to_numpy(),
            sidecar["run2_candidate_physical_weight"].to_numpy(),
        ),
        "four-b physical sidecar weights changed",
    )

    table_dir = staging / "tables"
    table_dir.mkdir()
    pq.write_table(pa.Table.from_pandas(threeb_all, preserve_index=False), table_dir / "train_exactly3b_promoted_run2_physical.parquet", compression="zstd")
    pq.write_table(pa.Table.from_pandas(fourb_all, preserve_index=False), table_dir / "train_at_least4b_run2_physical.parquet", compression="zstd")
    write_tsv(staging / "source_materialization_registry.tsv", source_rows)
    write_json(staging / "schemas.json", {
        "threeb": {field.name: str(field.type) for field in pa.Table.from_pandas(threeb_all, preserve_index=False).schema},
        "fourb": {field.name: str(field.type) for field in pa.Table.from_pandas(fourb_all, preserve_index=False).schema},
    })
    return threeb_all, fourb_all, source_rows


def yield_stats(frame: pd.DataFrame) -> dict[str, Any]:
    weights = frame["run2_candidate_physical_weight"].to_numpy(dtype=np.float64)
    total = float(weights.sum(dtype=np.float64))
    sum2 = float(np.square(weights).sum(dtype=np.float64))
    sumabs = float(np.abs(weights).sum(dtype=np.float64))
    effective = float(total * total / sum2) if sum2 > 0 else 0.0
    return {
        "raw_candidate_rows": int(len(frame)),
        "signed_physical_yield": total,
        "sum_squared_physical_weights": sum2,
        "sum_absolute_physical_weights": sumabs,
        "effective_events": effective,
        "negative_weight_rows": int((weights < 0).sum()),
        "zero_weight_rows": int((weights == 0).sum()),
        "positive_weight_rows": int((weights > 0).sum()),
        "cancellation_fraction": float(1.0 - abs(total) / sumabs) if sumabs else 0.0,
    }


def region_masks(frame: pd.DataFrame) -> dict[str, np.ndarray]:
    masks = {name: np.asarray(fn(frame), dtype=bool) for name, fn in REGIONS}
    require(all(len(mask) == len(frame) for mask in masks.values()), "region mask size mismatch")
    return masks


def write_weighted_closures(staging: Path, category_frames: dict[str, pd.DataFrame]) -> dict[str, Any]:
    region_rows: list[dict[str, Any]] = []
    process_rows: list[dict[str, Any]] = []
    source_rows: list[dict[str, Any]] = []
    nested_rows: list[dict[str, Any]] = []
    category_summary: dict[str, Any] = {}

    for category, frame in category_frames.items():
        masks = region_masks(frame)
        mhh_masks = {
            "inclusive_mhh": np.ones(len(frame), dtype=bool),
            "low_mhh": frame["mhh"].to_numpy() < 450.0,
            "high_mhh": frame["mhh"].to_numpy() >= 450.0,
        }
        if category == "threeb":
            btag_masks = {"exactly3b": frame["n_selected_bjets"].to_numpy() == 3}
        else:
            btags = frame["n_selected_bjets"].to_numpy()
            btag_masks = {
                "inclusive_ge4b": btags >= 4,
                "exactly4b": btags == 4,
                "ge5b": btags >= 5,
            }
        for region, region_mask in masks.items():
            for mhh_name, mhh_mask in mhh_masks.items():
                for btag_name, btag_mask in btag_masks.items():
                    base = region_mask & mhh_mask & btag_mask
                    for sample_class in ("background", "signal"):
                        selected = frame.loc[base & (frame["sample_class"].to_numpy() == sample_class)]
                        region_rows.append({
                            "physical_category": category,
                            "region": region,
                            "mhh_category": mhh_name,
                            "btag_category": btag_name,
                            "sample_class": sample_class,
                            **yield_stats(selected),
                            "yield_scope": "train_partition_contribution_only",
                        })
                    selected = frame.loc[base]
                    for keys, group in selected.groupby(["population_kind", "sample_class", "process_or_mode"], sort=True):
                        population, sample_class, process = keys
                        process_rows.append({
                            "physical_category": category,
                            "region": region,
                            "mhh_category": mhh_name,
                            "btag_category": btag_name,
                            "population_kind": population,
                            "sample_class": sample_class,
                            "process_or_mode": process,
                            **yield_stats(group),
                            "yield_scope": "train_partition_contribution_only",
                        })
        baseline_yields: dict[str, float] = {}
        for cut_order, region in NESTED_REGIONS:
            for sample_class in ("signal", "background"):
                selected = frame.loc[masks[region] & (frame["sample_class"].to_numpy() == sample_class)]
                stats = yield_stats(selected)
                if cut_order == 0:
                    baseline_yields[sample_class] = stats["signed_physical_yield"]
                baseline = baseline_yields[sample_class]
                nested_rows.append({
                    "physical_category": category,
                    "cut_order": cut_order,
                    "region": region,
                    "sample_class": sample_class,
                    **stats,
                    "efficiency_relative_to_baseline_yield": stats["signed_physical_yield"] / baseline if baseline else 0.0,
                    "yield_scope": "train_partition_contribution_only",
                })
        for keys, group in frame.groupby(
            ["population_kind", "sample_class", "process_or_mode", "campaign", "transport_id"], sort=True
        ):
            population, sample_class, process, campaign, transport = keys
            source_rows.append({
                "physical_category": category,
                "population_kind": population,
                "sample_class": sample_class,
                "process_or_mode": process,
                "campaign": campaign,
                "transport_id": transport,
                **yield_stats(group),
                "yield_scope": "train_partition_contribution_only",
            })
        category_summary[category] = {
            "rows": len(frame),
            "sources_with_candidates": int(frame["transport_id"].nunique()),
            "class": {sample: yield_stats(frame.loc[frame["sample_class"] == sample]) for sample in ("background", "signal")},
        }

    closure_dir = staging / "weighted_closures"
    closure_dir.mkdir()
    write_tsv(closure_dir / "train_class_region_cutflow.tsv", region_rows)
    write_tsv(closure_dir / "train_process_region_cutflow.tsv", process_rows)
    write_tsv(closure_dir / "train_source_yield_closure.tsv", source_rows)
    write_tsv(closure_dir / "train_nested_cutflow.tsv", nested_rows)
    return category_summary


def compare_numeric(observed: float, expected: float, label: str, *, rtol: float = 2e-12, atol: float = 1e-12) -> float:
    difference = abs(observed - expected)
    require(math.isclose(observed, expected, rel_tol=rtol, abs_tol=atol), f"prior four-b closure mismatch {label}: {observed} != {expected}")
    return difference


def reproduce_prior_fourb_closure(staging: Path, fourb: pd.DataFrame) -> dict[str, Any]:
    masks = region_masks(fourb)
    prior_nested = read_tsv(FOURB_CLOSURE_CHECKPOINT / "train_fourb_nested_cutflow.tsv")
    reproduction: list[dict[str, Any]] = []
    max_yield_difference = 0.0
    max_sum2_difference = 0.0
    old_to_new_region = {"baseline_all_fourb_candidates": "baseline_all_candidates"}
    for row in prior_nested:
        region = old_to_new_region.get(row["region"], row["region"])
        selected = fourb.loc[masks[region] & (fourb["sample_class"].to_numpy() == row["sample_class"])]
        stats = yield_stats(selected)
        require(stats["raw_candidate_rows"] == int(row["raw_candidate_rows"]), "prior nested raw-count mismatch")
        yield_diff = compare_numeric(stats["signed_physical_yield"], float(row["signed_physical_yield"]), f"nested/{region}/{row['sample_class']}")
        sum2_diff = compare_numeric(stats["sum_squared_physical_weights"], float(row["sum_squared_physical_weights"]), f"nested-sum2/{region}/{row['sample_class']}")
        max_yield_difference = max(max_yield_difference, yield_diff)
        max_sum2_difference = max(max_sum2_difference, sum2_diff)
        reproduction.append({
            "comparison": "nested_cutflow",
            "region": row["region"],
            "mhh_category": "inclusive_mhh",
            "btag_category": "inclusive_ge4b",
            "sample_class": row["sample_class"],
            "process_or_mode": "all",
            "expected_raw_rows": int(row["raw_candidate_rows"]),
            "observed_raw_rows": stats["raw_candidate_rows"],
            "expected_signed_yield": float(row["signed_physical_yield"]),
            "observed_signed_yield": stats["signed_physical_yield"],
            "absolute_yield_difference": yield_diff,
            "absolute_sum2_difference": sum2_diff,
            "pass": True,
        })

    prior_sources = read_tsv(FOURB_CLOSURE_CHECKPOINT / "train_fourb_source_yield_closure.tsv")
    grouped = {
        key: group
        for key, group in fourb.groupby(
            ["population_kind", "sample_class", "process_or_mode", "campaign", "transport_id"], sort=False
        )
    }
    require(len(grouped) == len(prior_sources), "prior source-group count mismatch")
    for row in prior_sources:
        key = tuple(row[field] for field in ("population_kind", "sample_class", "process_or_mode", "campaign", "transport_id"))
        require(key in grouped, f"prior source key missing: {key}")
        stats = yield_stats(grouped[key])
        require(stats["raw_candidate_rows"] == int(row["raw_candidate_rows"]), "prior source raw-count mismatch")
        yield_diff = compare_numeric(stats["signed_physical_yield"], float(row["signed_physical_yield"]), f"source/{key[-1]}")
        sum2_diff = compare_numeric(stats["sum_squared_physical_weights"], float(row["sum_squared_physical_weights"]), f"source-sum2/{key[-1]}")
        max_yield_difference = max(max_yield_difference, yield_diff)
        max_sum2_difference = max(max_sum2_difference, sum2_diff)

    write_tsv(staging / "fourb_prior_closure_reproduction.tsv", reproduction)
    return {
        "nested_rows_compared": len(prior_nested),
        "source_rows_compared": len(prior_sources),
        "max_absolute_signed_yield_difference": max_yield_difference,
        "max_absolute_sum_squared_weight_difference": max_sum2_difference,
        "status": "exact_identity_and_tolerance_closure_pass",
    }


def parquet_manifest(staging: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in sorted((staging / "tables").glob("*.parquet")):
        metadata = pq.read_metadata(path)
        rows.append({
            "path": path.relative_to(staging).as_posix(),
            "bytes": path.stat().st_size,
            "sha256": sha256(path),
            "rows": metadata.num_rows,
            "columns": metadata.num_columns,
            "row_groups": metadata.num_row_groups,
            "compression": "zstd",
            "split": "train",
            "validation_opened": False,
            "test_opened": False,
        })
    return rows


def freeze_checkpoint(staging: Path, output: Path) -> str:
    (staging / "COMPLETE").write_text("pn_c7q_full441_train_physical_tables_pass\n")
    paths = [path for path in staging.rglob("*") if path.is_file() and path.name != "SHA256SUMS"]
    lines = [f"{sha256(path)}  ./{path.relative_to(staging).as_posix()}" for path in sorted(paths)]
    (staging / "SHA256SUMS").write_text("\n".join(lines) + "\n")
    manifest_digest = sha256(staging / "SHA256SUMS")
    os.rename(staging, output)
    for path in sorted(output.rglob("*"), reverse=True):
        if path.is_file():
            path.chmod(stat.S_IRUSR | stat.S_IRGRP | stat.S_IROTH)
        elif path.is_dir():
            path.chmod(stat.S_IRUSR | stat.S_IXUSR | stat.S_IRGRP | stat.S_IXGRP | stat.S_IROTH | stat.S_IXOTH)
    output.chmod(stat.S_IRUSR | stat.S_IXUSR | stat.S_IRGRP | stat.S_IXGRP | stat.S_IROTH | stat.S_IXOTH)
    return manifest_digest


def run(output: Path) -> dict[str, Any]:
    require(output.is_absolute(), "output checkpoint path must be absolute")
    require(not output.exists(), f"refusing to overwrite output checkpoint: {output}")
    staging = output.with_name(f".{output.name}.staging")
    require(not staging.exists(), f"refusing to overwrite preserved staging: {staging}")
    require(output.parent.is_dir(), f"output parent does not exist: {output.parent}")
    staging.mkdir(mode=0o755)
    try:
        pinned = verify_pinned_inputs()
        harvest_manifest = parse_sha256sums(HARVEST / "SHA256SUMS")
        threeb, fourb, sources = materialize_tables(staging, harvest_manifest)
        category_summary = write_weighted_closures(staging, {"threeb": threeb, "fourb": fourb})
        fourb_reproduction = reproduce_prior_fourb_closure(staging, fourb)
        table_manifest = parquet_manifest(staging)
        write_tsv(staging / "table_artifact_manifest.tsv", table_manifest)
        source_manifest = [
            {"path": path, "sha256": digest, "role": "pinned_frozen_input"}
            for path, digest in sorted(pinned.items())
        ]
        write_tsv(staging / "source_evidence_manifest.tsv", source_manifest)
        summary = {
            "schema_version": 1,
            "status": "pn_c7q_full441_train_physical_tables_pass",
            "created_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "input_harvest": str(HARVEST),
            "output_checkpoint": str(output),
            "source_rows": len(sources),
            "threeb_rows": len(threeb),
            "fourb_rows": len(fourb),
            "threeb_fourb_composite_identity_overlap": 0,
            "run2_luminosity_pb_inverse": 138000.0,
            "physical_weights_authorized_at_this_gate": True,
            "physical_yields_authorized_at_this_gate": True,
            "candidate_multiplicity_division_applied": False,
            "split": "train",
            "validation_payload_files_opened": 0,
            "test_or_evaluation_payload_files_opened": 0,
            "category_summary": category_summary,
            "fourb_prior_closure_reproduction": fourb_reproduction,
            "table_artifacts": table_manifest,
            "paper_geometry": {
                "sqrt_s_TeV": 13,
                "run2_luminosity_fb_inverse": 138,
                "cms_reference_sr": "r_hh_125_120 < 30 GeV",
                "cms_reference_cr": "30 <= r_hh_125_120 < 55 GeV",
                "optimized_nominal_sr": "r_hh_125_120 < 34 GeV",
                "mhh_boundary_GeV": 450,
            },
            "next_gate": "PN-c7r frozen train-only three-b-to-four-b multijet transfer closure and systematics",
            "baseline_development_authorized": False,
        }
        write_json(staging / "summary.json", summary)
        (staging / "RUN_CONTRACT.txt").write_text(
            "PN-c7q materializes only the sealed train partition. It applies the frozen Run-2 "
            "coefficient registries row-by-row, copies and revalidates the sealed four-b physical "
            "sidecar, does not divide event weights by candidate multiplicity, and does not open "
            "validation or test payloads. Baseline development remains gated on PN-c7r.\n"
        )
        (staging / "README.md").write_text(
            "# PN-c7q full-441 train physical tables\n\n"
            "This immutable checkpoint contains normalized, provenance-complete train-only "
            "exactly-three-b promoted and at-least-four-b candidate tables. The weighted closure "
            "tables use the frozen CMS-inspired mass geometry. The prior four-b sidecar and "
            "selected-yield closure are reproduced before publication. Validation and test remain sealed.\n"
        )
        manifest_digest = freeze_checkpoint(staging, output)
        return {"output": str(output), "sha256sums_sha256": manifest_digest, **summary}
    except Exception as exc:
        failure = {
            "status": "pn_c7q_materialization_failed_preserved_staging",
            "created_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "error": str(exc),
            "traceback": traceback.format_exc(),
        }
        try:
            write_json(staging / "FAILURE.json", failure)
        except Exception:
            pass
        raise


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = run(args.output)
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
