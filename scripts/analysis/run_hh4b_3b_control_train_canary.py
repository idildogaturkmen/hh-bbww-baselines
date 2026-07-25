#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
from collections import Counter
from pathlib import Path
from typing import Any

import awkward as ak

MATPLOTLIB_CONFIG_DIR = (
    Path(tempfile.gettempdir())
    / f"hh4b_3b_train_canary_matplotlib_{os.getpid()}"
)
os.environ["MPLCONFIGDIR"] = str(MATPLOTLIB_CONFIG_DIR)

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import uproot

import validate_hh4b_4b_real_parity_canaries as common


OUTPUT_TABLES = (
    "threeb_train_canary_selection.tsv",
    "threeb_train_canary_file_summary.tsv",
    "threeb_candidate_integrity.tsv",
    "fourb_candidate_integrity.tsv",
    "threeb_fourb_disjointness.tsv",
    "unweighted_candidate_row_counts.tsv",
    "promoted_jet_summary.tsv",
    "promoted_jet_candidate_position.tsv",
    "candidate_schema_audit.tsv",
    "extraction_access_audit.tsv",
    "plot_inventory.tsv",
    "summary.json",
)

PLOT_NAMES = (
    "background_candidate_rows_by_process.png",
    "background_mbb_plane_threeb.png",
    "background_mbb_plane_fourb.png",
    "background_rhh_threeb_fourb.png",
    "background_mhh_threeb_fourb.png",
    "background_selected_jet_multiplicity_threeb.png",
    "background_promoted_jet_pt.png",
    "background_promoted_jet_eta.png",
    "background_promoted_jet_flavor.png",
    "background_promoted_jet_candidate_position.png",
    "signal_candidate_rows_by_mode.png",
    "signal_mbb_plane_threeb.png",
    "signal_mbb_plane_fourb.png",
)

TITLE_PREFIX = "Unweighted train-canary QA — not a physics prediction"

SELECTION_FIELDS = [
    "canary_index",
    "member_index",
    "sample_class",
    "process_family",
    "process_or_mode",
    "dataset_split",
    "existing_canonical_4b_rows",
    "selection_reasons",
    "selection_policy",
    "resolution_method",
    "layout_rule",
    "remote_bundle_path",
    "root_container_member",
    "root_archive_member",
]

FILE_SUMMARY_FIELDS = [
    "member_index",
    "sample_class",
    "process_or_mode",
    "dataset_split",
    "selection_reasons",
    "resolution_method",
    "layout_rule",
    "remote_bundle_path",
    "root_container_member",
    "root_archive_member",
    "delphes_num_entries",
    "existing_canonical_4b_rows",
    "threeb_output_rows",
    "fourb_output_rows",
    "threeb_schema_columns",
    "fourb_schema_columns",
    "threeb_duplicate_events",
    "fourb_duplicate_events",
    "threeb_fourb_overlap_events",
    "threeb_integrity_failures",
    "fourb_integrity_failures",
    "reconstruction_status",
]

INTEGRITY_FIELDS = [
    "member_index",
    "sample_class",
    "process_or_mode",
    "check_name",
    "rows_checked",
    "failure_count",
    "integrity_status",
]

DISJOINTNESS_FIELDS = [
    "member_index",
    "sample_class",
    "process_or_mode",
    "threeb_unique_events",
    "fourb_unique_events",
    "intersection_size",
    "intersection_events_json",
    "disjointness_status",
]

ROW_COUNT_FIELDS = [
    "member_index",
    "process_or_mode",
    "sample_class",
    "candidate_category",
    "unweighted_candidate_rows",
    "counting_scope",
    "physics_yield_interpretation",
]

PROMOTED_SUMMARY_FIELDS = [
    "member_index",
    "process_or_mode",
    "sample_class",
    "threeb_rows",
    "promoted_pt_min",
    "promoted_pt_mean",
    "promoted_pt_max",
    "promoted_eta_mean",
    "promoted_btag_max",
    "promoted_flavor_counts_json",
    "candidate_category",
]

PROMOTED_POSITION_FIELDS = [
    "member_index",
    "process_or_mode",
    "sample_class",
    "candidate_position",
    "total_threeb_rows",
    "promoted_rows",
]

SCHEMA_AUDIT_FIELDS = [
    "member_index",
    "mode",
    "observed_column_count",
    "expected_column_count",
    "ordered_columns_match",
    "threeb_provenance_contract_match",
    "schema_status",
]

EXTRACTION_AUDIT_FIELDS = [
    "member_index",
    "remote_bundle_path",
    "remote_bundle_uri",
    "expected_bundle_size_bytes",
    "xrdcp_attempted",
    "xrdcp_return_code",
    "xrdcp_status",
    "xrdcp_failure_text",
    "local_bundle_size_bytes",
    "bundle_size_match",
    "outer_archive_members_inspected",
    "outer_requested_member",
    "outer_exact_matches",
    "nested_archive_used",
    "nested_archive_members_inspected",
    "nested_requested_member",
    "nested_exact_matches",
    "root_file_extracted",
    "root_arrays_opened_for_integrity",
    "root_branches_read_json",
    "threeb_builder_invoked",
    "threeb_builder_return_code",
    "fourb_builder_invoked",
    "fourb_builder_return_code",
    "threeb_output_retained_runtime",
    "fourb_output_retained_runtime",
    "root_file_deleted",
    "nested_archive_deleted",
    "outer_bundle_deleted",
    "temporary_files_remaining",
    "failure_reason",
    "access_status",
]

PLOT_INVENTORY_FIELDS = [
    "plot_path",
    "sample_scope",
    "plot_name",
    "title",
    "fixed_binning",
    "unweighted_rows_used",
    "file_size_bytes",
    "plot_status",
]


def load_builder_contract(path: Path) -> Any:
    spec = importlib.util.spec_from_file_location("hh4b_builder_contract", path)
    common.require(
        spec is not None and spec.loader is not None,
        "cannot import builder contract",
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def select_canaries(
    source_rows: list[dict[str, str]],
    config: dict[str, Any],
) -> list[dict[str, Any]]:
    train_rows = [
        row
        for row in source_rows
        if row["dataset_split"] == config["allowed_dataset_split"]
    ]
    selected_reasons: dict[str, set[str]] = {}
    for rule in config["selection_rules"]:
        matches = [
            row
            for row in train_rows
            if all(
                common.condition_matches(row, condition)
                for condition in rule["all"]
            )
        ]
        common.require(
            bool(matches),
            f"selection rule {rule['rule_id']} has no train match",
        )
        chosen = min(matches, key=lambda row: int(row["member_index"]))
        selected_reasons.setdefault(chosen["member_index"], set()).add(
            str(rule["rule_id"])
        )

    by_member = {row["member_index"]: row for row in train_rows}
    selected: list[dict[str, Any]] = []
    for canary_index, member_index in enumerate(
        sorted(selected_reasons, key=int),
        start=1,
    ):
        row = dict(by_member[member_index])
        row.update(
            {
                "canary_index": canary_index,
                "existing_canonical_4b_rows": row["candidate_rows"],
                "selection_reasons": json.dumps(
                    sorted(selected_reasons[member_index]),
                    separators=(",", ":"),
                ),
                "selection_policy": config["selection_policy"],
            }
        )
        selected.append(row)
    common.require(
        len(selected) >= int(config["minimum_unique_canaries"]),
        "deduplicated canary set is too small",
    )
    common.require(
        all(row["dataset_split"] == "train" for row in selected),
        "non-train canary selected",
    )
    return selected


def make_integrity_row(
    *,
    canary: dict[str, Any],
    check_name: str,
    checks: list[bool],
) -> dict[str, Any]:
    failures = sum(not value for value in checks)
    return {
        "member_index": canary["member_index"],
        "sample_class": canary["sample_class"],
        "process_or_mode": canary["process_or_mode"],
        "check_name": check_name,
        "rows_checked": len(checks),
        "failure_count": failures,
        "integrity_status": "pass" if failures == 0 else "fail",
    }


def exact_scalar_equal(left: Any, right: Any) -> bool:
    if pd.isna(left) or pd.isna(right):
        return bool(pd.isna(left) and pd.isna(right))
    return bool(left == right)


def threeb_integrity(
    *,
    canary: dict[str, Any],
    frame: pd.DataFrame,
    arrays: Any,
    config: dict[str, Any],
) -> list[dict[str, Any]]:
    check_results: dict[str, list[bool]] = {
        "n_selected_bjets_equals_3": [],
        "n_selected_jets_at_least_4": [],
        "n_selected_untagged_jets_at_least_1": [],
        "candidate_category_constant": [],
        "promoted_jet_rule_constant": [],
        "promoted_jet_btag_nonpositive": [],
        "exactly_one_promoted_bjet_rank": [],
        "tagged_bjet_ranks_are_0_1_2": [],
        "promoted_raw_index_matches_candidate": [],
        "promoted_selected_index_matches_candidate": [],
        "promoted_pt_matches_candidate": [],
        "promoted_eta_matches_candidate": [],
        "promoted_phi_matches_candidate": [],
        "promoted_mass_matches_candidate": [],
        "promoted_btag_matches_candidate": [],
        "promoted_flavor_matches_candidate": [],
        "promoted_pt_is_highest_selected_untagged": [],
        "selected_untagged_count_matches_root": [],
    }
    provenance_fields = (
        ("raw_index", "promoted_jet_raw_index"),
        ("selected_index", "promoted_jet_selected_index"),
        ("pt", "promoted_jet_pt"),
        ("eta", "promoted_jet_eta"),
        ("phi", "promoted_jet_phi"),
        ("mass", "promoted_jet_mass"),
        ("btag", "promoted_jet_btag"),
        ("flavor", "promoted_jet_flavor"),
    )
    provenance_checks = {
        "raw_index": "promoted_raw_index_matches_candidate",
        "selected_index": "promoted_selected_index_matches_candidate",
        "pt": "promoted_pt_matches_candidate",
        "eta": "promoted_eta_matches_candidate",
        "phi": "promoted_phi_matches_candidate",
        "mass": "promoted_mass_matches_candidate",
        "btag": "promoted_btag_matches_candidate",
        "flavor": "promoted_flavor_matches_candidate",
    }
    jet_pt_min = float(config["reconstruction_parameters"]["jet_pt_min"])
    jet_eta_max = float(config["reconstruction_parameters"]["jet_eta_max"])
    btag_min = float(config["reconstruction_parameters"]["btag_min"])

    for _, row in frame.iterrows():
        check_results["n_selected_bjets_equals_3"].append(
            int(row["n_selected_bjets"]) == 3
        )
        check_results["n_selected_jets_at_least_4"].append(
            int(row["n_selected_jets"]) >= 4
        )
        check_results["n_selected_untagged_jets_at_least_1"].append(
            int(row["n_selected_untagged_jets"]) >= 1
        )
        check_results["candidate_category_constant"].append(
            row["candidate_category"]
            == "exactly_3b_plus_highest_pt_untagged"
        )
        check_results["promoted_jet_rule_constant"].append(
            row["promoted_jet_rule"]
            == "highest_pt_selected_untagged"
        )
        check_results["promoted_jet_btag_nonpositive"].append(
            float(row["promoted_jet_btag"]) <= btag_min
        )
        ranks = [
            int(row[f"j{position}_bjet_rank"])
            for position in range(1, 5)
        ]
        promoted_positions = [
            position
            for position, rank in enumerate(ranks, start=1)
            if rank == -1
        ]
        one_promoted = len(promoted_positions) == 1
        check_results["exactly_one_promoted_bjet_rank"].append(one_promoted)
        check_results["tagged_bjet_ranks_are_0_1_2"].append(
            sorted(rank for rank in ranks if rank != -1) == [0, 1, 2]
        )
        for field, promoted_field in provenance_fields:
            matches = False
            if one_promoted:
                position = promoted_positions[0]
                matches = exact_scalar_equal(
                    row[f"j{position}_{field}"],
                    row[promoted_field],
                )
            check_results[provenance_checks[field]].append(matches)

        event = int(row["event"])
        pts = ak.to_numpy(arrays["Jet.PT"][event])
        etas = ak.to_numpy(arrays["Jet.Eta"][event])
        btags = ak.to_numpy(arrays["Jet.BTag"][event])
        selected_mask = (pts > jet_pt_min) & (np.abs(etas) < jet_eta_max)
        untagged_mask = selected_mask & (btags <= btag_min)
        untagged_pts = pts[untagged_mask]
        check_results["promoted_pt_is_highest_selected_untagged"].append(
            bool(
                len(untagged_pts) >= 1
                and float(row["promoted_jet_pt"])
                >= float(np.max(untagged_pts))
            )
        )
        check_results["selected_untagged_count_matches_root"].append(
            int(row["n_selected_untagged_jets"])
            == int(np.count_nonzero(untagged_mask))
        )

    return [
        make_integrity_row(
            canary=canary,
            check_name=check_name,
            checks=checks,
        )
        for check_name, checks in check_results.items()
    ]


def fourb_integrity(
    *,
    canary: dict[str, Any],
    frame: pd.DataFrame,
    threeb_only_columns: set[str],
) -> list[dict[str, Any]]:
    return [
        make_integrity_row(
            canary=canary,
            check_name="n_selected_bjets_at_least_4",
            checks=[
                int(value) >= 4
                for value in frame["n_selected_bjets"].tolist()
            ],
        ),
        make_integrity_row(
            canary=canary,
            check_name="n_selected_jets_at_least_4",
            checks=[
                int(value) >= 4
                for value in frame["n_selected_jets"].tolist()
            ],
        ),
        make_integrity_row(
            canary=canary,
            check_name="threeb_only_columns_absent",
            checks=[not bool(set(frame.columns) & threeb_only_columns)],
        ),
    ]


def schema_audit(
    *,
    member_index: str,
    mode: str,
    observed: list[str],
    expected: list[str],
    provenance_columns: list[str],
) -> dict[str, Any]:
    ordered_match = observed == expected
    if mode == "threeb-control":
        provenance_match = observed[-len(provenance_columns) :] == (
            provenance_columns
        )
    else:
        provenance_match = not bool(
            set(observed) & set(provenance_columns)
        )
    return {
        "member_index": member_index,
        "mode": mode,
        "observed_column_count": len(observed),
        "expected_column_count": len(expected),
        "ordered_columns_match": ordered_match,
        "threeb_provenance_contract_match": provenance_match,
        "schema_status": (
            "pass" if ordered_match and provenance_match else "fail"
        ),
    }


def promoted_summary(
    canary: dict[str, Any],
    frame: pd.DataFrame,
) -> dict[str, Any]:
    if frame.empty:
        return {
            "member_index": canary["member_index"],
            "process_or_mode": canary["process_or_mode"],
            "sample_class": canary["sample_class"],
            "threeb_rows": 0,
            "promoted_pt_min": "",
            "promoted_pt_mean": "",
            "promoted_pt_max": "",
            "promoted_eta_mean": "",
            "promoted_btag_max": "",
            "promoted_flavor_counts_json": "{}",
            "candidate_category": "exactly_3b_plus_highest_pt_untagged",
        }
    flavor_counts = Counter(
        int(value) for value in frame["promoted_jet_flavor"]
    )
    return {
        "member_index": canary["member_index"],
        "process_or_mode": canary["process_or_mode"],
        "sample_class": canary["sample_class"],
        "threeb_rows": len(frame),
        "promoted_pt_min": float(frame["promoted_jet_pt"].min()),
        "promoted_pt_mean": float(frame["promoted_jet_pt"].mean()),
        "promoted_pt_max": float(frame["promoted_jet_pt"].max()),
        "promoted_eta_mean": float(frame["promoted_jet_eta"].mean()),
        "promoted_btag_max": float(frame["promoted_jet_btag"].max()),
        "promoted_flavor_counts_json": json.dumps(
            dict(sorted(flavor_counts.items())),
            separators=(",", ":"),
        ),
        "candidate_category": "exactly_3b_plus_highest_pt_untagged",
    }


def promoted_positions(frame: pd.DataFrame) -> list[int]:
    positions: list[int] = []
    for _, row in frame.iterrows():
        matches = [
            position
            for position in range(1, 5)
            if int(row[f"j{position}_bjet_rank"]) == -1
        ]
        if len(matches) == 1:
            positions.append(matches[0])
    return positions


def add_plot_metadata(
    frame: pd.DataFrame,
    canary: dict[str, Any],
) -> pd.DataFrame:
    enriched = frame.copy()
    enriched["_member_index"] = int(canary["member_index"])
    enriched["_process_or_mode"] = canary["process_or_mode"]
    enriched["_sample_class"] = canary["sample_class"]
    return enriched


def empty_file_summary(canary: dict[str, Any]) -> dict[str, Any]:
    return {
        "member_index": canary["member_index"],
        "sample_class": canary["sample_class"],
        "process_or_mode": canary["process_or_mode"],
        "dataset_split": canary["dataset_split"],
        "selection_reasons": canary["selection_reasons"],
        "resolution_method": canary["resolution_method"],
        "layout_rule": canary["layout_rule"],
        "remote_bundle_path": canary["remote_bundle_path"],
        "root_container_member": canary["root_container_member"],
        "root_archive_member": canary["root_archive_member"],
        "delphes_num_entries": "",
        "existing_canonical_4b_rows": canary[
            "existing_canonical_4b_rows"
        ],
        "threeb_output_rows": "",
        "fourb_output_rows": "",
        "threeb_schema_columns": "",
        "fourb_schema_columns": "",
        "threeb_duplicate_events": "",
        "fourb_duplicate_events": "",
        "threeb_fourb_overlap_events": "",
        "threeb_integrity_failures": "",
        "fourb_integrity_failures": "",
        "reconstruction_status": "fail",
    }


def process_canary(
    *,
    repo: Path,
    output_tmp: Path,
    canary: dict[str, Any],
    canary_number: int,
    canary_total: int,
    config: dict[str, Any],
    builder_path: Path,
    builder_contract: Any,
) -> dict[str, Any]:
    member_index = canary["member_index"]
    print(
        (
            f"[{canary_number}/{canary_total}] member={member_index} "
            f"process={canary['process_or_mode']} "
            f"size={canary['bundle_size_bytes']}"
        ),
        flush=True,
    )
    temp_dir = Path(
        tempfile.mkdtemp(
            prefix=f"hh4b_3b_train_canary_{member_index}_",
            dir=config["temporary_root"],
        )
    )
    outer_path = temp_dir / "outer_bundle.tar.gz"
    nested_path = temp_dir / "nested_reconstruction.tar.gz"
    root_path = temp_dir / canary["root_basename"]
    parquet_dir = output_tmp / "parquets" / f"member_{int(member_index):04d}"
    parquet_dir.mkdir(parents=True, exist_ok=False)
    threeb_output = parquet_dir / "threeb_control.parquet"
    fourb_output = parquet_dir / "fourb_parity.parquet"
    file_row = empty_file_summary(canary)
    threeb_integrity_rows: list[dict[str, Any]] = []
    fourb_integrity_rows: list[dict[str, Any]] = []
    disjointness_row: dict[str, Any] | None = None
    count_rows: list[dict[str, Any]] = []
    promoted_summary_row: dict[str, Any] | None = None
    promoted_position_rows: list[dict[str, Any]] = []
    schema_rows: list[dict[str, Any]] = []
    threeb_plot_frame: pd.DataFrame | None = None
    fourb_plot_frame: pd.DataFrame | None = None
    access: dict[str, Any] = {
        "member_index": member_index,
        "remote_bundle_path": canary["remote_bundle_path"],
        "remote_bundle_uri": canary["remote_bundle_uri"],
        "expected_bundle_size_bytes": canary["bundle_size_bytes"],
        "xrdcp_attempted": True,
        "xrdcp_return_code": "",
        "xrdcp_status": "not_completed",
        "xrdcp_failure_text": "",
        "local_bundle_size_bytes": "",
        "bundle_size_match": False,
        "outer_archive_members_inspected": 0,
        "outer_requested_member": (
            canary["root_container_member"]
            or canary["root_archive_member"]
        ),
        "outer_exact_matches": 0,
        "nested_archive_used": bool(canary["root_container_member"]),
        "nested_archive_members_inspected": 0,
        "nested_requested_member": (
            canary["root_archive_member"]
            if canary["root_container_member"]
            else ""
        ),
        "nested_exact_matches": 0,
        "root_file_extracted": False,
        "root_arrays_opened_for_integrity": False,
        "root_branches_read_json": "[]",
        "threeb_builder_invoked": False,
        "threeb_builder_return_code": "",
        "fourb_builder_invoked": False,
        "fourb_builder_return_code": "",
        "threeb_output_retained_runtime": False,
        "fourb_output_retained_runtime": False,
        "root_file_deleted": False,
        "nested_archive_deleted": False,
        "outer_bundle_deleted": False,
        "temporary_files_remaining": "",
        "failure_reason": "",
        "access_status": "fail",
    }
    try:
        download = subprocess.run(
            [
                "xrdcp",
                "--nopbar",
                "--force",
                canary["remote_bundle_uri"],
                str(outer_path),
            ],
            check=False,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=int(config["xrdcp_timeout_seconds"]),
        )
        access["xrdcp_return_code"] = download.returncode
        if download.returncode != 0:
            failure_text = (download.stderr or download.stdout).strip()
            access["xrdcp_failure_text"] = re.sub(
                r"\s+",
                " ",
                failure_text,
            )
            raise RuntimeError(
                f"xrdcp failed with return code {download.returncode}"
            )
        access["xrdcp_status"] = "pass"
        local_size = outer_path.stat().st_size
        access["local_bundle_size_bytes"] = local_size
        access["bundle_size_match"] = (
            local_size == int(canary["bundle_size_bytes"])
        )
        common.require(
            access["bundle_size_match"],
            "downloaded bundle size mismatch",
        )
        if canary["root_container_member"]:
            count, matches = common.extract_exact_member(
                archive_path=outer_path,
                requested_member=canary["root_container_member"],
                destination_path=nested_path,
            )
            access["outer_archive_members_inspected"] = count
            access["outer_exact_matches"] = matches
            count, matches = common.extract_exact_member(
                archive_path=nested_path,
                requested_member=canary["root_archive_member"],
                destination_path=root_path,
            )
            access["nested_archive_members_inspected"] = count
            access["nested_exact_matches"] = matches
        else:
            count, matches = common.extract_exact_member(
                archive_path=outer_path,
                requested_member=canary["root_archive_member"],
                destination_path=root_path,
            )
            access["outer_archive_members_inspected"] = count
            access["outer_exact_matches"] = matches
        common.require(root_path.is_file(), "ROOT file was not extracted")
        access["root_file_extracted"] = True

        stable_sample = (
            f"{canary['process_or_mode']}_train_canary_member{member_index}"
        )
        access["threeb_builder_invoked"] = True
        threeb_run = common.run_builder(
            repo=repo,
            builder_path=builder_path,
            root_path=root_path,
            output_path=threeb_output,
            sample=stable_sample,
            parameters=config["reconstruction_parameters"],
            timeout_seconds=int(config["builder_timeout_seconds"]),
            mode="threeb-control",
        )
        access["threeb_builder_return_code"] = threeb_run.returncode
        common.require(
            threeb_run.returncode == 0,
            "threeb builder failed: "
            + re.sub(
                r"\s+",
                " ",
                (threeb_run.stderr or threeb_run.stdout).strip(),
            ),
        )
        common.require(threeb_output.is_file(), "threeb output is missing")
        access["threeb_output_retained_runtime"] = True

        access["fourb_builder_invoked"] = True
        fourb_run = common.run_builder(
            repo=repo,
            builder_path=builder_path,
            root_path=root_path,
            output_path=fourb_output,
            sample=stable_sample,
            parameters=config["reconstruction_parameters"],
            timeout_seconds=int(config["builder_timeout_seconds"]),
            mode="fourb-parity",
        )
        access["fourb_builder_return_code"] = fourb_run.returncode
        common.require(
            fourb_run.returncode == 0,
            "fourb builder failed: "
            + re.sub(
                r"\s+",
                " ",
                (fourb_run.stderr or fourb_run.stdout).strip(),
            ),
        )
        common.require(fourb_output.is_file(), "fourb output is missing")
        access["fourb_output_retained_runtime"] = True

        threeb_frame = pd.read_parquet(threeb_output)
        fourb_frame = pd.read_parquet(fourb_output)
        with uproot.open(root_path) as root_file:
            tree = root_file["Delphes"]
            arrays = tree.arrays(
                list(config["required_root_branches"]),
                library="ak",
            )
            delphes_num_entries = int(tree.num_entries)
        access["root_arrays_opened_for_integrity"] = True
        access["root_branches_read_json"] = json.dumps(
            config["required_root_branches"],
            separators=(",", ":"),
        )
        common.require(
            delphes_num_entries == int(canary["generated_events"]),
            "Delphes entry count differs from source map",
        )

        schema_rows = [
            schema_audit(
                member_index=member_index,
                mode="threeb-control",
                observed=list(threeb_frame.columns),
                expected=list(builder_contract.THREEB_CANDIDATE_COLUMNS),
                provenance_columns=list(
                    builder_contract.THREEB_PROVENANCE_COLUMNS
                ),
            ),
            schema_audit(
                member_index=member_index,
                mode="fourb-parity",
                observed=list(fourb_frame.columns),
                expected=list(builder_contract.FROZEN_CANDIDATE_COLUMNS),
                provenance_columns=list(
                    builder_contract.THREEB_PROVENANCE_COLUMNS
                ),
            ),
        ]
        threeb_integrity_rows = threeb_integrity(
            canary=canary,
            frame=threeb_frame,
            arrays=arrays,
            config=config,
        )
        fourb_integrity_rows = fourb_integrity(
            canary=canary,
            frame=fourb_frame,
            threeb_only_columns=set(
                builder_contract.THREEB_PROVENANCE_COLUMNS
            ),
        )
        threeb_duplicates = int(
            threeb_frame["event"].duplicated(keep=False).sum()
        )
        fourb_duplicates = int(
            fourb_frame["event"].duplicated(keep=False).sum()
        )
        threeb_events = {
            int(value) for value in threeb_frame["event"].tolist()
        }
        fourb_events = {
            int(value) for value in fourb_frame["event"].tolist()
        }
        overlap = sorted(threeb_events & fourb_events)
        disjointness_row = {
            "member_index": member_index,
            "sample_class": canary["sample_class"],
            "process_or_mode": canary["process_or_mode"],
            "threeb_unique_events": len(threeb_events),
            "fourb_unique_events": len(fourb_events),
            "intersection_size": len(overlap),
            "intersection_events_json": json.dumps(overlap),
            "disjointness_status": "pass" if not overlap else "fail",
        }
        count_rows = [
            {
                "member_index": member_index,
                "process_or_mode": canary["process_or_mode"],
                "sample_class": canary["sample_class"],
                "candidate_category": "existing_canonical_fourb_registry",
                "unweighted_candidate_rows": int(
                    canary["existing_canonical_4b_rows"]
                ),
                "counting_scope": "source_map_registry_diagnostic",
                "physics_yield_interpretation": "none",
            },
            {
                "member_index": member_index,
                "process_or_mode": canary["process_or_mode"],
                "sample_class": canary["sample_class"],
                "candidate_category": "threeb-control",
                "unweighted_candidate_rows": len(threeb_frame),
                "counting_scope": "reconstructed_train_canary_qa",
                "physics_yield_interpretation": "none",
            },
            {
                "member_index": member_index,
                "process_or_mode": canary["process_or_mode"],
                "sample_class": canary["sample_class"],
                "candidate_category": "fourb-parity",
                "unweighted_candidate_rows": len(fourb_frame),
                "counting_scope": "reconstructed_train_canary_qa",
                "physics_yield_interpretation": "none",
            },
        ]
        promoted_summary_row = promoted_summary(canary, threeb_frame)
        position_counts = Counter(promoted_positions(threeb_frame))
        promoted_position_rows = [
            {
                "member_index": member_index,
                "process_or_mode": canary["process_or_mode"],
                "sample_class": canary["sample_class"],
                "candidate_position": position,
                "total_threeb_rows": len(threeb_frame),
                "promoted_rows": position_counts[position],
            }
            for position in config["plot_binning"][
                "promoted_candidate_positions"
            ]
        ]
        threeb_failures = sum(
            int(row["failure_count"]) for row in threeb_integrity_rows
        )
        fourb_failures = sum(
            int(row["failure_count"]) for row in fourb_integrity_rows
        )
        schema_failures = sum(
            row["schema_status"] != "pass" for row in schema_rows
        )
        passed = (
            threeb_failures == 0
            and fourb_failures == 0
            and schema_failures == 0
            and threeb_duplicates == 0
            and fourb_duplicates == 0
            and not overlap
        )
        file_row.update(
            {
                "delphes_num_entries": delphes_num_entries,
                "threeb_output_rows": len(threeb_frame),
                "fourb_output_rows": len(fourb_frame),
                "threeb_schema_columns": len(threeb_frame.columns),
                "fourb_schema_columns": len(fourb_frame.columns),
                "threeb_duplicate_events": threeb_duplicates,
                "fourb_duplicate_events": fourb_duplicates,
                "threeb_fourb_overlap_events": len(overlap),
                "threeb_integrity_failures": threeb_failures,
                "fourb_integrity_failures": fourb_failures,
                "reconstruction_status": "pass" if passed else "fail",
            }
        )
        threeb_plot_frame = add_plot_metadata(threeb_frame, canary)
        fourb_plot_frame = add_plot_metadata(fourb_frame, canary)
        access["access_status"] = "pass"
    except subprocess.TimeoutExpired as error:
        access["failure_reason"] = (
            f"subprocess timed out after {error.timeout} seconds"
        )
    except Exception as error:
        access["failure_reason"] = str(error)
    finally:
        shutil.rmtree(temp_dir)
        common.require(
            not temp_dir.exists(),
            f"temporary directory remains: {temp_dir}",
        )
        access["root_file_deleted"] = not root_path.exists()
        access["nested_archive_deleted"] = not nested_path.exists()
        access["outer_bundle_deleted"] = not outer_path.exists()
        access["temporary_files_remaining"] = 0

    print(
        (
            f"[{canary_number}/{canary_total}] member={member_index} "
            f"status={file_row['reconstruction_status']} "
            f"threeb={file_row['threeb_output_rows']} "
            f"fourb={file_row['fourb_output_rows']} "
            f"failure={access['failure_reason']!r}"
        ),
        flush=True,
    )
    return {
        "file_row": file_row,
        "threeb_integrity_rows": threeb_integrity_rows,
        "fourb_integrity_rows": fourb_integrity_rows,
        "disjointness_row": disjointness_row,
        "count_rows": count_rows,
        "promoted_summary_row": promoted_summary_row,
        "promoted_position_rows": promoted_position_rows,
        "schema_rows": schema_rows,
        "threeb_plot_frame": threeb_plot_frame,
        "fourb_plot_frame": fourb_plot_frame,
        "access_row": access,
    }


def concatenate_frames(frames: list[pd.DataFrame]) -> pd.DataFrame:
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def create_plots(
    *,
    output_tmp: Path,
    config: dict[str, Any],
    threeb_frames: list[pd.DataFrame],
    fourb_frames: list[pd.DataFrame],
) -> list[dict[str, Any]]:
    plots_dir = output_tmp / "plots"
    plots_dir.mkdir(parents=True, exist_ok=False)
    threeb = concatenate_frames(threeb_frames)
    fourb = concatenate_frames(fourb_frames)
    background_threeb = threeb[threeb["_sample_class"] == "background"]
    background_fourb = fourb[fourb["_sample_class"] == "background"]
    signal_threeb = threeb[threeb["_sample_class"] == "signal"]
    signal_fourb = fourb[fourb["_sample_class"] == "signal"]
    inventory: list[dict[str, Any]] = []

    def save(
        *,
        fig: Any,
        name: str,
        scope: str,
        title: str,
        binning: str,
        rows_used: int,
    ) -> None:
        path = plots_dir / name
        fig.tight_layout()
        fig.savefig(path, dpi=140)
        plt.close(fig)
        status = "pass" if path.is_file() and path.stat().st_size > 0 else "fail"
        inventory.append(
            {
                "plot_path": f"plots/{name}",
                "sample_scope": scope,
                "plot_name": name,
                "title": title,
                "fixed_binning": binning,
                "unweighted_rows_used": rows_used,
                "file_size_bytes": path.stat().st_size if path.exists() else 0,
                "plot_status": status,
            }
        )

    def row_count_plot(
        three: pd.DataFrame,
        four: pd.DataFrame,
        *,
        name: str,
        scope: str,
        label: str,
    ) -> None:
        processes = sorted(
            set(three["_process_or_mode"]) | set(four["_process_or_mode"])
        )
        three_counts = [
            int((three["_process_or_mode"] == process).sum())
            for process in processes
        ]
        four_counts = [
            int((four["_process_or_mode"] == process).sum())
            for process in processes
        ]
        x = np.arange(len(processes))
        fig, ax = plt.subplots(figsize=(max(8, len(processes) * 0.8), 5))
        ax.bar(x - 0.2, three_counts, 0.4, label=f"3b (N={len(three)})")
        ax.bar(x + 0.2, four_counts, 0.4, label=f"4b (N={len(four)})")
        ax.set_xticks(x)
        ax.set_xticklabels(processes, rotation=45, ha="right")
        ax.set_ylabel("Raw candidate rows")
        title = f"{TITLE_PREFIX}\n{label}"
        ax.set_title(title)
        ax.legend()
        save(
            fig=fig,
            name=name,
            scope=scope,
            title=title,
            binning="categorical process_or_mode; raw row counts",
            rows_used=len(three) + len(four),
        )

    def mbb_plane(
        frame: pd.DataFrame,
        *,
        name: str,
        scope: str,
        label: str,
    ) -> None:
        mbb = config["plot_binning"]["mbb"]
        fig, ax = plt.subplots(figsize=(6, 5))
        histogram = ax.hist2d(
            frame["mbb1"],
            frame["mbb2"],
            bins=int(mbb["bins"]),
            range=[
                [float(mbb["min"]), float(mbb["max"])],
                [float(mbb["min"]), float(mbb["max"])],
            ],
            cmap="viridis",
        )
        fig.colorbar(histogram[3], ax=ax, label="Raw candidate rows")
        ax.set_xlabel("mbb1 [GeV]")
        ax.set_ylabel("mbb2 [GeV]")
        title = f"{TITLE_PREFIX}\n{label} (N={len(frame)})"
        ax.set_title(title)
        save(
            fig=fig,
            name=name,
            scope=scope,
            title=title,
            binning=(
                f"{mbb['bins']}x{mbb['bins']} bins; "
                f"axes [{mbb['min']}, {mbb['max']}] GeV"
            ),
            rows_used=len(frame),
        )

    row_count_plot(
        background_threeb,
        background_fourb,
        name="background_candidate_rows_by_process.png",
        scope="background_only",
        label="Background candidate rows by process",
    )
    mbb_plane(
        background_threeb,
        name="background_mbb_plane_threeb.png",
        scope="background_only",
        label="Background exactly-3b mbb plane",
    )
    mbb_plane(
        background_fourb,
        name="background_mbb_plane_fourb.png",
        scope="background_only",
        label="Background fourb-parity mbb plane",
    )

    for variable, plot_name, label, bin_key in (
        (
            "r_hh",
            "background_rhh_threeb_fourb.png",
            "Background R_HH",
            "rhh",
        ),
        (
            "mhh",
            "background_mhh_threeb_fourb.png",
            "Background mHH",
            "mhh",
        ),
    ):
        bins = config["plot_binning"][bin_key]
        edges = np.linspace(
            float(bins["min"]),
            float(bins["max"]),
            int(bins["bins"]) + 1,
        )
        fig, ax = plt.subplots(figsize=(7, 5))
        ax.hist(
            background_threeb[variable],
            bins=edges,
            histtype="step",
            linewidth=1.8,
            label=f"3b (N={len(background_threeb)})",
        )
        ax.hist(
            background_fourb[variable],
            bins=edges,
            histtype="step",
            linewidth=1.8,
            label=f"4b (N={len(background_fourb)})",
        )
        ax.set_xlabel(f"{label} [GeV]")
        ax.set_ylabel("Raw candidate rows")
        title = f"{TITLE_PREFIX}\n{label}"
        ax.set_title(title)
        ax.legend()
        save(
            fig=fig,
            name=plot_name,
            scope="background_only",
            title=title,
            binning=(
                f"{bins['bins']} bins; "
                f"[{bins['min']}, {bins['max']}]"
            ),
            rows_used=len(background_threeb) + len(background_fourb),
        )

    multiplicity_edges = np.asarray(
        config["plot_binning"]["selected_jet_multiplicity_edges"],
        dtype=float,
    )
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.hist(
        background_threeb["n_selected_jets"],
        bins=multiplicity_edges,
        histtype="step",
        linewidth=1.8,
        label=f"3b (N={len(background_threeb)})",
    )
    ax.set_xlabel("Selected jet multiplicity")
    ax.set_ylabel("Raw candidate rows")
    title = f"{TITLE_PREFIX}\nBackground selected-jet multiplicity (3b)"
    ax.set_title(title)
    ax.legend()
    save(
        fig=fig,
        name="background_selected_jet_multiplicity_threeb.png",
        scope="background_only",
        title=title,
        binning=json.dumps(multiplicity_edges.tolist()),
        rows_used=len(background_threeb),
    )

    for column, plot_name, label, key in (
        (
            "promoted_jet_pt",
            "background_promoted_jet_pt.png",
            "Promoted jet pT [GeV]",
            "promoted_jet_pt",
        ),
        (
            "promoted_jet_eta",
            "background_promoted_jet_eta.png",
            "Promoted jet eta",
            "promoted_jet_eta",
        ),
    ):
        bins = config["plot_binning"][key]
        edges = np.linspace(
            float(bins["min"]),
            float(bins["max"]),
            int(bins["bins"]) + 1,
        )
        fig, ax = plt.subplots(figsize=(7, 5))
        ax.hist(
            background_threeb[column],
            bins=edges,
            histtype="stepfilled",
            alpha=0.65,
            label=f"3b background (N={len(background_threeb)})",
        )
        ax.set_xlabel(label)
        ax.set_ylabel("Raw candidate rows")
        title = f"{TITLE_PREFIX}\nBackground {label}"
        ax.set_title(title)
        ax.legend()
        save(
            fig=fig,
            name=plot_name,
            scope="background_only",
            title=title,
            binning=(
                f"{bins['bins']} bins; "
                f"[{bins['min']}, {bins['max']}]"
            ),
            rows_used=len(background_threeb),
        )

    fixed_flavors = [
        int(value)
        for value in config["plot_binning"]["promoted_flavor_values"]
    ]
    observed_flavors = [
        int(value) for value in background_threeb["promoted_jet_flavor"]
    ]
    flavor_counts = Counter(observed_flavors)
    flavor_labels = [str(value) for value in fixed_flavors] + ["other"]
    flavor_values = [flavor_counts[value] for value in fixed_flavors]
    flavor_values.append(
        sum(
            count
            for value, count in flavor_counts.items()
            if value not in fixed_flavors
        )
    )
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.bar(flavor_labels, flavor_values)
    ax.set_xlabel("Promoted Jet.Flavor")
    ax.set_ylabel("Raw candidate rows")
    title = (
        f"{TITLE_PREFIX}\nBackground promoted-jet flavor "
        f"(N={len(background_threeb)})"
    )
    ax.set_title(title)
    save(
        fig=fig,
        name="background_promoted_jet_flavor.png",
        scope="background_only",
        title=title,
        binning=f"fixed categories {flavor_labels}",
        rows_used=len(background_threeb),
    )

    background_positions = promoted_positions(background_threeb)
    position_values = [
        int(value)
        for value in config["plot_binning"][
            "promoted_candidate_positions"
        ]
    ]
    position_counts = Counter(background_positions)
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.bar(
        [str(value) for value in position_values],
        [position_counts[value] for value in position_values],
    )
    ax.set_xlabel("Candidate-jet position")
    ax.set_ylabel("Raw candidate rows")
    title = (
        f"{TITLE_PREFIX}\nBackground promoted-jet candidate position "
        f"(N={len(background_threeb)})"
    )
    ax.set_title(title)
    save(
        fig=fig,
        name="background_promoted_jet_candidate_position.png",
        scope="background_only",
        title=title,
        binning=f"fixed categories {position_values}",
        rows_used=len(background_threeb),
    )

    row_count_plot(
        signal_threeb,
        signal_fourb,
        name="signal_candidate_rows_by_mode.png",
        scope="signal_only",
        label="Signal reconstruction QA candidate rows by mode",
    )
    mbb_plane(
        signal_threeb,
        name="signal_mbb_plane_threeb.png",
        scope="signal_only",
        label="Signal reconstruction QA exactly-3b mbb plane",
    )
    mbb_plane(
        signal_fourb,
        name="signal_mbb_plane_fourb.png",
        scope="signal_only",
        label="Signal reconstruction QA fourb-parity mbb plane",
    )
    common.require(
        {row["plot_name"] for row in inventory} == set(PLOT_NAMES),
        "plot inventory differs from requested plots",
    )
    return inventory


def checkpoint_readme(summary: dict[str, Any]) -> str:
    return f"""# HH4b exactly-3b control train canary

## Purpose

This checkpoint validates the first bounded real-Delphes reconstruction of
the exactly-three-tag HH4b control category. A deterministic train-only
canary set was reconstructed in both `threeb-control` and `fourb-parity`
modes from each identical ROOT member.

## Result

- Status: `{summary['status']}`
- Canaries selected and passed: {summary['canaries_passed']}
- Background canaries: {summary['background_canaries']}
- Signal reconstruction-QA canaries: {summary['signal_canaries']}
- Exactly-3b candidate rows: {summary['threeb_candidate_rows']}
- Four-b parity candidate rows: {summary['fourb_candidate_rows']}
- Three-b integrity failures: {summary['threeb_integrity_failures']}
- Four-b integrity failures: {summary['fourb_integrity_failures']}
- Duplicate event rows: {summary['threeb_duplicate_event_rows'] + summary['fourb_duplicate_event_rows']}
- Three-b/four-b event overlap: {summary['total_threeb_fourb_event_overlap']}
- QA plots written: {summary['plots_written']}
- Temporary files remaining: {summary['temporary_files_remaining']}

Every promoted jet matches its candidate-jet record and is the highest-pT
selected untagged jet according to the six authorized ROOT branches. The
three-b and four-b event categories are exactly disjoint within every source
member. Event-level Parquets remain only in the ignored runtime directory and
are not included in this checkpoint.

## Scientific interpretation boundary

All counts and plots are raw, unweighted train-canary reconstruction QA. They
are not physical yields or a background prediction. No cross-section or
luminosity normalization, 3b/4b ratio, transfer factor, closure prediction,
significance, optimized signal region, or signal-based bin optimization was
performed. Signal and background QA plots are separate.

## Safety record

- Validation members considered: {summary['validation_members_considered']}
- Test members considered: {summary['test_members_considered']}
- Weighted events calculated: {summary['weighted_events_calculated']}
- Physics yields calculated: {summary['physics_yields_calculated']}
- Transfer factors calculated: {summary['transfer_factors_calculated']}
- Closure predictions calculated: {summary['closure_predictions_calculated']}
- Significances calculated: {summary['significances_calculated']}

## Reproducibility

- Source commit: `{summary['source_commit']}`
- Source-map SHA-256: `{summary['source_map_sha256']}`
- Builder SHA-256: `{summary['builder_sha256']}`
- Configuration SHA-256: `{summary['configuration_sha256']}`
- Runner SHA-256: `{summary['runner_sha256']}`

## Next gate

`{summary['next_gate']}`
"""


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Run bounded train-only HH4b exactly-3b and four-b canary "
            "reconstruction with integrity and unweighted QA outputs."
        )
    )
    parser.add_argument("--config", required=True, type=Path)
    args = parser.parse_args()

    repo = Path(__file__).resolve().parents[2]
    runner_path = Path(__file__).resolve()
    config_path = (
        args.config.resolve()
        if args.config.is_absolute()
        else (repo / args.config).resolve()
    )
    config = common.read_json_object(config_path)
    common.require(config.get("schema_version") == 1, "unsupported config")
    common.require(
        config.get("allowed_dataset_split") == "train",
        "only train members are authorized",
    )
    common.require(
        config.get("validation_access_allowed") is False,
        "validation access must be disabled",
    )
    common.require(
        config.get("test_access_allowed") is False,
        "test access must be disabled",
    )
    source_commit = str(config["source_commit"]).lower()
    head = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repo,
        check=True,
        text=True,
        stdout=subprocess.PIPE,
    ).stdout.strip()
    common.require(head == source_commit, "HEAD differs from source commit")

    for value in config["prerequisite_checkpoint_dirs"]:
        common.verify_checkpoint(common.resolve(repo, value))
    parity_summary = common.read_json_object(
        common.resolve(repo, config["real_parity_summary_path"])
    )
    common.require(
        parity_summary["status"] == "hh4b_4b_real_delphes_parity_pass",
        "real parity checkpoint did not pass",
    )
    for field in (
        "total_parity_mismatches",
        "validation_canaries",
        "test_canaries",
        "threeb_candidates_reconstructed",
    ):
        common.require(
            int(parity_summary[field]) == 0,
            f"real parity precondition failed: {field}",
        )

    source_map_path = common.resolve(repo, config["source_map_path"])
    common.require(
        common.sha256_file(source_map_path) == config["source_map_sha256"],
        "source-map SHA-256 mismatch",
    )
    protected_paths: list[str] = []
    for value, expected_hash in config["protected_files"].items():
        path = common.resolve(repo, value)
        common.require(
            common.sha256_file(path) == expected_hash,
            f"protected file hash mismatch: {value}",
        )
        protected_paths.append(value)
    protected_diff = subprocess.run(
        ["git", "diff", "--exit-code", "--", *protected_paths],
        cwd=repo,
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    common.require(
        protected_diff.returncode == 0,
        "a protected builder or policy has uncommitted changes",
    )
    builder_path = common.resolve(repo, config["builder_path"])
    builder_contract = load_builder_contract(builder_path)

    output_dir = common.resolve(repo, config["output_dir"])
    checkpoint_dir = common.resolve(repo, config["checkpoint_dir"])
    output_tmp = output_dir.with_name(output_dir.name + "_incomplete")
    checkpoint_tmp = checkpoint_dir.with_name(
        checkpoint_dir.name + "_incomplete"
    )
    for path in (output_dir, output_tmp, checkpoint_dir, checkpoint_tmp):
        common.require(not path.exists(), f"refusing to overwrite {path}")
    temporary_root = common.resolve(repo, config["temporary_root"])
    common.require(temporary_root.is_dir(), "temporary root is missing")
    config["temporary_root"] = str(temporary_root)

    source_fields, source_rows = common.read_tsv(source_map_path)
    required_fields = {
        "member_index",
        "sample_class",
        "process_family",
        "process_or_mode",
        "dataset_split",
        "generated_events",
        "candidate_rows",
        "resolution_method",
        "layout_rule",
        "remote_bundle_path",
        "remote_bundle_uri",
        "bundle_size_bytes",
        "root_container_member",
        "root_archive_member",
        "root_basename",
        "resolution_status",
        "ambiguity_count",
    }
    common.require(
        required_fields <= set(source_fields),
        "source map is missing required fields",
    )
    common.require(
        all(
            row["resolution_status"] == "resolved"
            and int(row["ambiguity_count"]) == 0
            for row in source_rows
        ),
        "source map contains unresolved or ambiguous members",
    )
    canaries = select_canaries(source_rows, config)
    selection_rows = [
        {field: row.get(field, "") for field in SELECTION_FIELDS}
        for row in canaries
    ]

    output_tmp.mkdir(parents=True, exist_ok=False)
    common.write_tsv(
        output_tmp / "threeb_train_canary_selection.tsv",
        selection_rows,
        SELECTION_FIELDS,
    )
    results = []
    for canary_number, canary in enumerate(canaries, start=1):
        results.append(
            process_canary(
                repo=repo,
                output_tmp=output_tmp,
                canary=canary,
                canary_number=canary_number,
                canary_total=len(canaries),
                config=config,
                builder_path=builder_path,
                builder_contract=builder_contract,
            )
        )

    file_rows = [result["file_row"] for result in results]
    threeb_integrity_rows = [
        row
        for result in results
        for row in result["threeb_integrity_rows"]
    ]
    fourb_integrity_rows = [
        row
        for result in results
        for row in result["fourb_integrity_rows"]
    ]
    disjointness_rows = [
        result["disjointness_row"]
        for result in results
        if result["disjointness_row"] is not None
    ]
    count_rows = [
        row for result in results for row in result["count_rows"]
    ]
    promoted_summary_rows = [
        result["promoted_summary_row"]
        for result in results
        if result["promoted_summary_row"] is not None
    ]
    promoted_position_rows = [
        row
        for result in results
        for row in result["promoted_position_rows"]
    ]
    schema_rows = [
        row for result in results for row in result["schema_rows"]
    ]
    access_rows = [result["access_row"] for result in results]
    threeb_plot_frames = [
        result["threeb_plot_frame"]
        for result in results
        if result["threeb_plot_frame"] is not None
    ]
    fourb_plot_frames = [
        result["fourb_plot_frame"]
        for result in results
        if result["fourb_plot_frame"] is not None
    ]
    plot_inventory = create_plots(
        output_tmp=output_tmp,
        config=config,
        threeb_frames=threeb_plot_frames,
        fourb_frames=fourb_plot_frames,
    )
    if MATPLOTLIB_CONFIG_DIR.exists():
        shutil.rmtree(MATPLOTLIB_CONFIG_DIR)
    common.require(
        not MATPLOTLIB_CONFIG_DIR.exists(),
        "matplotlib temporary configuration remains",
    )

    canaries_passed = sum(
        row["reconstruction_status"] == "pass" for row in file_rows
    )
    threeb_rows_total = sum(
        int(row["threeb_output_rows"] or 0) for row in file_rows
    )
    fourb_rows_total = sum(
        int(row["fourb_output_rows"] or 0) for row in file_rows
    )
    background_members = {
        row["member_index"]
        for row in canaries
        if row["sample_class"] == "background"
    }
    signal_members = {
        row["member_index"]
        for row in canaries
        if row["sample_class"] == "signal"
    }
    summary: dict[str, Any] = {
        "schema_version": 1,
        "status": "hh4b_3b_control_train_canary_fail",
        "source_commit": source_commit,
        "configuration": str(config_path),
        "configuration_sha256": common.sha256_file(config_path),
        "runner": str(runner_path),
        "runner_sha256": common.sha256_file(runner_path),
        "source_map_path": str(source_map_path),
        "source_map_sha256": common.sha256_file(source_map_path),
        "builder_path": str(builder_path),
        "builder_sha256": common.sha256_file(builder_path),
        "canaries_selected": len(canaries),
        "canaries_processed": len(file_rows),
        "canaries_passed": canaries_passed,
        "canaries_failed": len(file_rows) - canaries_passed,
        "background_canaries": len(background_members),
        "signal_canaries": len(signal_members),
        "train_canaries": sum(
            row["dataset_split"] == "train" for row in canaries
        ),
        "validation_canaries": 0,
        "test_canaries": 0,
        "direct_layout_canaries": sum(
            not bool(row["root_container_member"]) for row in canaries
        ),
        "nested_layout_canaries": sum(
            bool(row["root_container_member"]) for row in canaries
        ),
        "existing_zero_4b_row_background_canaries": sum(
            row["sample_class"] == "background"
            and int(row["existing_canonical_4b_rows"]) == 0
            for row in canaries
        ),
        "existing_nonzero_4b_row_background_canaries": sum(
            row["sample_class"] == "background"
            and int(row["existing_canonical_4b_rows"]) > 0
            for row in canaries
        ),
        "root_files_extracted": sum(
            bool(row["root_file_extracted"]) for row in access_rows
        ),
        "root_files_opened_in_threeb_mode": sum(
            row["threeb_builder_return_code"] == 0 for row in access_rows
        ),
        "root_files_opened_in_fourb_mode": sum(
            row["fourb_builder_return_code"] == 0 for row in access_rows
        ),
        "root_files_opened_for_integrity": sum(
            bool(row["root_arrays_opened_for_integrity"])
            for row in access_rows
        ),
        "threeb_builder_invocations": sum(
            bool(row["threeb_builder_invoked"]) for row in access_rows
        ),
        "fourb_builder_invocations": sum(
            bool(row["fourb_builder_invoked"]) for row in access_rows
        ),
        "threeb_candidate_rows": threeb_rows_total,
        "fourb_candidate_rows": fourb_rows_total,
        "background_threeb_candidate_rows": sum(
            int(row["threeb_output_rows"] or 0)
            for row in file_rows
            if row["member_index"] in background_members
        ),
        "background_fourb_candidate_rows": sum(
            int(row["fourb_output_rows"] or 0)
            for row in file_rows
            if row["member_index"] in background_members
        ),
        "signal_threeb_candidate_rows": sum(
            int(row["threeb_output_rows"] or 0)
            for row in file_rows
            if row["member_index"] in signal_members
        ),
        "signal_fourb_candidate_rows": sum(
            int(row["fourb_output_rows"] or 0)
            for row in file_rows
            if row["member_index"] in signal_members
        ),
        "threeb_integrity_checks": sum(
            int(row["rows_checked"]) for row in threeb_integrity_rows
        ),
        "threeb_integrity_failures": sum(
            int(row["failure_count"]) for row in threeb_integrity_rows
        ),
        "fourb_integrity_checks": sum(
            int(row["rows_checked"]) for row in fourb_integrity_rows
        ),
        "fourb_integrity_failures": sum(
            int(row["failure_count"]) for row in fourb_integrity_rows
        ),
        "threeb_duplicate_event_rows": sum(
            int(row["threeb_duplicate_events"] or 0) for row in file_rows
        ),
        "fourb_duplicate_event_rows": sum(
            int(row["fourb_duplicate_events"] or 0) for row in file_rows
        ),
        "total_threeb_fourb_event_overlap": sum(
            int(row["intersection_size"]) for row in disjointness_rows
        ),
        "threeb_schema_mismatches": sum(
            row["mode"] == "threeb-control"
            and row["schema_status"] != "pass"
            for row in schema_rows
        ),
        "fourb_schema_mismatches": sum(
            row["mode"] == "fourb-parity"
            and row["schema_status"] != "pass"
            for row in schema_rows
        ),
        "plots_requested": len(PLOT_NAMES),
        "plots_written": sum(
            row["plot_status"] == "pass" for row in plot_inventory
        ),
        "plots_failed": sum(
            row["plot_status"] != "pass" for row in plot_inventory
        ),
        "weighted_events_calculated": 0,
        "physics_yields_calculated": 0,
        "cross_section_normalizations_calculated": 0,
        "transfer_factors_calculated": 0,
        "threeb_fourb_ratios_calculated": 0,
        "closure_predictions_calculated": 0,
        "significances_calculated": 0,
        "signal_regions_optimized": 0,
        "validation_members_considered": 0,
        "test_members_considered": 0,
        "temporary_files_remaining": sum(
            int(row["temporary_files_remaining"]) for row in access_rows
        ),
        "event_level_parquets_committed": 0,
        "gate_errors": [],
        "next_gate": config["next_gate"],
    }

    gate_errors: list[str] = []

    def gate(condition: bool, message: str) -> None:
        if not condition:
            gate_errors.append(message)

    gate(summary["canaries_selected"] >= 12, "fewer than 12 canaries")
    gate(
        summary["canaries_processed"] == summary["canaries_selected"],
        "not every selected canary was processed",
    )
    gate(
        summary["canaries_passed"] == summary["canaries_selected"],
        "not every selected canary passed",
    )
    gate(summary["canaries_failed"] == 0, "one or more canaries failed")
    gate(
        summary["train_canaries"] == summary["canaries_selected"],
        "non-train canary selected",
    )
    gate(summary["validation_canaries"] == 0, "validation canary selected")
    gate(summary["test_canaries"] == 0, "test canary selected")
    gate(summary["background_canaries"] >= 9, "background coverage too small")
    gate(summary["signal_canaries"] >= 3, "signal QA coverage too small")
    gate(summary["direct_layout_canaries"] >= 1, "direct layout uncovered")
    gate(summary["nested_layout_canaries"] >= 2, "nested layouts uncovered")
    gate(
        summary["existing_zero_4b_row_background_canaries"] >= 1,
        "existing zero-row background uncovered",
    )
    gate(
        summary["existing_nonzero_4b_row_background_canaries"] >= 1,
        "existing nonzero-row background uncovered",
    )
    for field in (
        "threeb_builder_invocations",
        "fourb_builder_invocations",
        "root_files_opened_in_threeb_mode",
        "root_files_opened_in_fourb_mode",
    ):
        gate(
            summary[field] == summary["canaries_selected"],
            f"{field} differs from selected canaries",
        )
    gate(summary["threeb_candidate_rows"] > 0, "no threeb candidates")
    gate(
        summary["background_threeb_candidate_rows"] > 0,
        "no background threeb candidates",
    )
    for field in (
        "threeb_integrity_failures",
        "fourb_integrity_failures",
        "threeb_duplicate_event_rows",
        "fourb_duplicate_event_rows",
        "total_threeb_fourb_event_overlap",
        "threeb_schema_mismatches",
        "fourb_schema_mismatches",
        "plots_failed",
        "weighted_events_calculated",
        "physics_yields_calculated",
        "cross_section_normalizations_calculated",
        "transfer_factors_calculated",
        "threeb_fourb_ratios_calculated",
        "closure_predictions_calculated",
        "significances_calculated",
        "signal_regions_optimized",
        "validation_members_considered",
        "test_members_considered",
        "temporary_files_remaining",
    ):
        gate(summary[field] == 0, f"{field} is nonzero")
    gate(
        summary["plots_written"] == summary["plots_requested"],
        "not every requested plot was written",
    )
    gate(
        all(row["access_status"] == "pass" for row in access_rows),
        "one or more access records failed",
    )
    gate(
        len(disjointness_rows) == summary["canaries_selected"],
        "disjointness coverage incomplete",
    )
    summary["gate_errors"] = gate_errors
    if not gate_errors:
        summary["status"] = "hh4b_3b_control_train_canary_pass"

    common.write_tsv(
        output_tmp / "threeb_train_canary_file_summary.tsv",
        file_rows,
        FILE_SUMMARY_FIELDS,
    )
    common.write_tsv(
        output_tmp / "threeb_candidate_integrity.tsv",
        threeb_integrity_rows,
        INTEGRITY_FIELDS,
    )
    common.write_tsv(
        output_tmp / "fourb_candidate_integrity.tsv",
        fourb_integrity_rows,
        INTEGRITY_FIELDS,
    )
    common.write_tsv(
        output_tmp / "threeb_fourb_disjointness.tsv",
        disjointness_rows,
        DISJOINTNESS_FIELDS,
    )
    common.write_tsv(
        output_tmp / "unweighted_candidate_row_counts.tsv",
        count_rows,
        ROW_COUNT_FIELDS,
    )
    common.write_tsv(
        output_tmp / "promoted_jet_summary.tsv",
        promoted_summary_rows,
        PROMOTED_SUMMARY_FIELDS,
    )
    common.write_tsv(
        output_tmp / "promoted_jet_candidate_position.tsv",
        promoted_position_rows,
        PROMOTED_POSITION_FIELDS,
    )
    common.write_tsv(
        output_tmp / "candidate_schema_audit.tsv",
        schema_rows,
        SCHEMA_AUDIT_FIELDS,
    )
    common.write_tsv(
        output_tmp / "extraction_access_audit.tsv",
        access_rows,
        EXTRACTION_AUDIT_FIELDS,
    )
    common.write_tsv(
        output_tmp / "plot_inventory.tsv",
        plot_inventory,
        PLOT_INVENTORY_FIELDS,
    )
    (output_tmp / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    output_tmp.rename(output_dir)

    print(json.dumps(summary, indent=2, sort_keys=True))
    if gate_errors:
        raise RuntimeError(
            "threeb train-canary gate failed: " + "; ".join(gate_errors)
        )

    checkpoint_tmp.mkdir(parents=True, exist_ok=False)
    for name in OUTPUT_TABLES:
        shutil.copy2(output_dir / name, checkpoint_tmp / name)
    shutil.copytree(output_dir / "plots", checkpoint_tmp / "plots")
    (checkpoint_tmp / "README.md").write_text(
        checkpoint_readme(summary),
        encoding="utf-8",
    )
    checkpoint = {
        "schema_version": 1,
        "classification": (
            "paper_quality_real_delphes_threeb_train_canary_checkpoint"
        ),
        "status": summary["status"],
        "source_commit": source_commit,
        "canaries_selected": summary["canaries_selected"],
        "canaries_passed": summary["canaries_passed"],
        "threeb_candidate_rows": summary["threeb_candidate_rows"],
        "fourb_candidate_rows": summary["fourb_candidate_rows"],
        "threeb_integrity_failures": 0,
        "fourb_integrity_failures": 0,
        "total_threeb_fourb_event_overlap": 0,
        "plots_written": summary["plots_written"],
        "validation_members_considered": 0,
        "test_members_considered": 0,
        "weighted_events_calculated": 0,
        "physics_yields_calculated": 0,
        "transfer_factors_calculated": 0,
        "threeb_fourb_ratios_calculated": 0,
        "closure_predictions_calculated": 0,
        "significances_calculated": 0,
        "temporary_files_remaining": 0,
        "event_level_parquets_committed": 0,
        "next_gate": summary["next_gate"],
    }
    (checkpoint_tmp / "checkpoint.json").write_text(
        json.dumps(checkpoint, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    checksum_lines = []
    for path in sorted(
        checkpoint_tmp.rglob("*"),
        key=lambda item: item.relative_to(checkpoint_tmp).as_posix(),
    ):
        if path.is_file() and path.name != "SHA256SUMS":
            relative = path.relative_to(checkpoint_tmp).as_posix()
            checksum_lines.append(
                f"{common.sha256_file(path)}  {relative}"
            )
    (checkpoint_tmp / "SHA256SUMS").write_text(
        "\n".join(checksum_lines) + "\n",
        encoding="utf-8",
    )
    checkpoint_tmp.rename(checkpoint_dir)


if __name__ == "__main__":
    main()
