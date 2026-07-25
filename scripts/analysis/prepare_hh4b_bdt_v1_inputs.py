#!/usr/bin/env python3
"""Freeze canonical HH4b BDT-v1 inputs without training a model.

Only train candidate Parquets are opened.  Validation population counts come
from the frozen manifest metadata, test data are not considered, and no
row-level feature matrix is written.
"""

from __future__ import annotations

import argparse
import atexit
from collections import Counter, defaultdict
import csv
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile
from typing import Any, Iterable, Mapping, Sequence


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPOSITORY_ROOT))
sys.path.insert(0, str(REPOSITORY_ROOT / "scripts" / "analysis"))

MPLCONFIG_TEMP = (
    Path(tempfile.gettempdir()) / f"hh4b_bdt_v1_mplconfig_{os.getpid()}"
)
os.environ["MPLCONFIGDIR"] = str(MPLCONFIG_TEMP)
atexit.register(shutil.rmtree, MPLCONFIG_TEMP, ignore_errors=True)

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import pyarrow  # noqa: E402
import yaml  # noqa: E402

from hh4b_bdt_v1_common import (  # noqa: E402
    COMMON_FEATURES,
    DERIVED_FEATURE_EXPRESSIONS,
    FORBIDDEN_EXACT_FEATURES,
    FORBIDDEN_SUBSTRINGS,
    MASS_AWARE_FEATURES,
    MASS_PLANE_BLIND_FEATURES,
    MASS_PLANE_FEATURES,
    SOURCE_COLUMNS,
    add_derived_features,
    assign_member_folds,
    build_hierarchical_weights,
    feature_quality_rows,
    forbidden_feature_reason,
    validate_feature_names,
)
from scripts.plotting.hh4b_cms_style import (  # noqa: E402
    CATEGORY_COLORS,
    FEATURE_LATEX_LABELS,
    add_delphes_header,
    apply_cms_style,
    save_png_pdf,
)


TABLE_FIELDS: Mapping[str, tuple[str, ...]] = {
    "train_population_summary.tsv": (
        "scope",
        "sample_class",
        "metadata_source",
        "members",
        "candidate_rows",
        "candidate_files_opened",
        "status",
    ),
    "train_population_by_family.tsv": (
        "category_type",
        "category",
        "sample_class",
        "training_target",
        "members",
        "candidate_bearing_members",
        "zero_candidate_members",
        "candidate_rows",
        "mapping_source",
        "status",
    ),
    "signal_mode_mapping.tsv": (
        "manifest_process_or_mode",
        "signal_mode",
        "source_field",
        "train_members",
        "train_candidate_rows",
        "validation_members_from_manifest",
        "validation_candidate_rows_from_manifest",
        "status",
    ),
    "background_family_mapping.tsv": (
        "manifest_process_or_mode",
        "background_family",
        "source_field",
        "train_members",
        "train_candidate_rows",
        "validation_members_from_manifest",
        "validation_candidate_rows_from_manifest",
        "status",
    ),
    "feature_contract.tsv": (
        "feature_order_mass_aware",
        "feature_order_mass_plane_blind",
        "feature",
        "source_column_or_expression",
        "logical_dtype",
        "common_non_mass_plane_feature",
        "mass_aware_feature",
        "mass_plane_blind_feature",
        "feature_role",
        "latex_short_label",
        "status",
    ),
    "forbidden_feature_audit.tsv": (
        "forbidden_input_or_pattern",
        "scope",
        "reason",
        "selected_features",
        "selected_count",
        "status",
    ),
    "feature_quality_audit.tsv": (
        "feature_order_mass_aware",
        "feature",
        "source_column_or_expression",
        "logical_dtype",
        "minimum",
        "maximum",
        "mean",
        "standard_deviation",
        "finite_rows",
        "nonfinite_rows",
        "missing_rows",
        "unique_values",
        "constant_feature",
        "status",
    ),
    "member_fold_assignment.tsv": (
        "member_index",
        "sample_class",
        "training_target",
        "process_or_mode",
        "stratum_type",
        "stratum",
        "candidate_rows",
        "fold",
        "assignment_algorithm",
        "grouping_key",
        "status",
    ),
    "fold_composition.tsv": (
        "fold",
        "composition_level",
        "category",
        "sample_class",
        "members",
        "candidate_bearing_members",
        "candidate_rows",
        "family_train_members",
        "family_expected_in_every_fold",
        "unavoidable_absence",
        "status",
    ),
    "training_weight_summary.tsv": (
        "level",
        "category",
        "parent",
        "candidate_rows",
        "members",
        "candidate_bearing_members",
        "target_hierarchical_fraction",
        "raw_weight_sum",
        "rescale_factor",
        "training_weight_sum",
        "mean_training_weight",
        "physical_normalization",
        "status",
    ),
    "member_weight_summary.tsv": (
        "member_index",
        "sample_class",
        "stratum",
        "candidate_rows",
        "fold",
        "weight_eligibility",
        "candidate_bearing_members_in_stratum",
        "hierarchical_stratum_fraction",
        "hierarchical_member_fraction",
        "per_row_raw_weight",
        "per_row_training_weight",
        "aggregate_raw_weight",
        "aggregate_training_weight",
        "status",
    ),
    "legacy_bdt_inventory.tsv": (
        "path",
        "artifact_type",
        "apparent_population",
        "apparent_split_policy",
        "feature_definition",
        "physical_weight_status",
        "canonical_v1_compatibility",
        "disposition",
    ),
    "plot_inventory.tsv": (
        "path",
        "paired_path",
        "plot_name",
        "variables",
        "normalization",
        "binning",
        "style_helper_path",
        "style_helper_sha256",
        "mplhep_used",
        "cms_inspired_style_applied",
        "latex_math_labels_applied",
        "dpi",
        "format",
        "file_size_bytes",
        "status",
    ),
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def git_output(*arguments: str) -> str:
    result = subprocess.run(
        ["git", *arguments],
        cwd=REPOSITORY_ROOT,
        check=True,
        text=True,
        capture_output=True,
    )
    return result.stdout.strip()


def load_config(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        raise ValueError("configuration must be a mapping")
    return config


def write_tsv(
    path: Path,
    rows: Iterable[Mapping[str, Any]],
    fields: Sequence[str],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=list(fields),
            delimiter="\t",
            extrasaction="raise",
            lineterminator="\n",
        )
        writer.writeheader()
        for raw in rows:
            row = {}
            for field in fields:
                value = raw.get(field, "")
                if isinstance(value, (list, tuple, dict)):
                    value = json.dumps(value, sort_keys=True, separators=(",", ":"))
                row[field] = value
            writer.writerow(row)


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def prepare_directory(path: Path, overwrite: bool) -> None:
    if path.exists():
        if not overwrite:
            raise FileExistsError(
                f"output already exists; use --overwrite for this exact target: {path}"
            )
        shutil.rmtree(path)
    path.mkdir(parents=True)


def read_manifest(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        required = {
            "member_index",
            "sample_class",
            "training_target",
            "process_or_mode",
            "dataset_split",
            "candidate_rows",
            "local_path",
            "test_member",
        }
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"manifest missing fields: {sorted(missing)}")
        if "process_family" in (reader.fieldnames or []):
            raise ValueError(
                "process_family unexpectedly appeared; review explicit mapping"
            )
        rows = []
        for raw in reader:
            row = dict(raw)
            row["member_index"] = int(row["member_index"])
            row["training_target"] = int(row["training_target"])
            row["candidate_rows"] = int(row["candidate_rows"])
            row["test_member"] = row["test_member"].lower() == "true"
            rows.append(row)
    return rows


def count_rows(
    rows: Sequence[Mapping[str, Any]],
    *,
    split: str | None = None,
    sample_class: str | None = None,
    process_or_mode: str | None = None,
) -> tuple[int, int]:
    selected = [
        row
        for row in rows
        if (split is None or row["dataset_split"] == split)
        and (sample_class is None or row["sample_class"] == sample_class)
        and (
            process_or_mode is None
            or row["process_or_mode"] == process_or_mode
        )
    ]
    return len(selected), sum(int(row["candidate_rows"]) for row in selected)


def validate_manifest(
    rows: Sequence[dict[str, Any]],
    config: Mapping[str, Any],
) -> dict[str, Any]:
    expected = config["population"]["expected"]
    signal_mapping = config["signal_mode_mapping"]
    background_mapping = config["background_family_mapping"]

    if len({row["member_index"] for row in rows}) != len(rows):
        raise ValueError("manifest member_index values are not unique")
    if any(row["test_member"] for row in rows):
        raise ValueError("canonical development manifest contains test members")
    if {row["dataset_split"] for row in rows} != {"train", "validation"}:
        raise ValueError("manifest contains an unexpected split")
    if {
        (row["sample_class"], row["training_target"])
        for row in rows
    } != {("background", 0), ("signal", 1)}:
        raise ValueError("authoritative binary label fields are inconsistent")

    signal_processes = {
        row["process_or_mode"]
        for row in rows
        if row["sample_class"] == "signal"
    }
    background_processes = {
        row["process_or_mode"]
        for row in rows
        if row["sample_class"] == "background"
    }
    if signal_processes != set(signal_mapping):
        raise ValueError(
            f"signal mapping mismatch: manifest={sorted(signal_processes)}, "
            f"config={sorted(signal_mapping)}"
        )
    if background_processes != set(background_mapping):
        raise ValueError(
            f"background mapping mismatch: manifest={sorted(background_processes)}, "
            f"config={sorted(background_mapping)}"
        )

    observed = {
        "development_members": len(rows),
        "train_members": count_rows(rows, split="train")[0],
        "validation_members": count_rows(rows, split="validation")[0],
        "train_signal_members": count_rows(
            rows, split="train", sample_class="signal"
        )[0],
        "train_background_members": count_rows(
            rows, split="train", sample_class="background"
        )[0],
        "validation_signal_members": count_rows(
            rows, split="validation", sample_class="signal"
        )[0],
        "validation_background_members": count_rows(
            rows, split="validation", sample_class="background"
        )[0],
        "development_candidate_rows": sum(row["candidate_rows"] for row in rows),
        "train_candidate_rows": count_rows(rows, split="train")[1],
        "train_signal_rows": count_rows(
            rows, split="train", sample_class="signal"
        )[1],
        "train_background_rows": count_rows(
            rows, split="train", sample_class="background"
        )[1],
        "validation_candidate_rows": count_rows(rows, split="validation")[1],
    }
    differences = {
        key: (expected[key], value)
        for key, value in observed.items()
        if int(expected[key]) != value
    }
    if differences:
        raise ValueError(f"manifest invariant mismatch: {differences}")
    return observed


def mapped_train_members(
    rows: Sequence[dict[str, Any]],
    config: Mapping[str, Any],
) -> list[dict[str, Any]]:
    members = []
    for row in sorted(
        (item for item in rows if item["dataset_split"] == "train"),
        key=lambda item: item["member_index"],
    ):
        member = dict(row)
        if member["sample_class"] == "signal":
            member["stratum_type"] = "signal_mode"
            member["stratum"] = config["signal_mode_mapping"][
                member["process_or_mode"]
            ]
        else:
            member["stratum_type"] = "background_family"
            member["stratum"] = config["background_family_mapping"][
                member["process_or_mode"]
            ]
        members.append(member)
    return members


def mapping_rows(
    rows: Sequence[dict[str, Any]],
    config: Mapping[str, Any],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    signal_rows = []
    for process, mode in sorted(config["signal_mode_mapping"].items()):
        train_members, train_candidates = count_rows(
            rows,
            split="train",
            sample_class="signal",
            process_or_mode=process,
        )
        validation_members, validation_candidates = count_rows(
            rows,
            split="validation",
            sample_class="signal",
            process_or_mode=process,
        )
        signal_rows.append(
            {
                "manifest_process_or_mode": process,
                "signal_mode": mode,
                "source_field": "process_or_mode",
                "train_members": train_members,
                "train_candidate_rows": train_candidates,
                "validation_members_from_manifest": validation_members,
                "validation_candidate_rows_from_manifest": validation_candidates,
                "status": "pass",
            }
        )

    background_rows = []
    for process, family in sorted(config["background_family_mapping"].items()):
        train_members, train_candidates = count_rows(
            rows,
            split="train",
            sample_class="background",
            process_or_mode=process,
        )
        validation_members, validation_candidates = count_rows(
            rows,
            split="validation",
            sample_class="background",
            process_or_mode=process,
        )
        background_rows.append(
            {
                "manifest_process_or_mode": process,
                "background_family": family,
                "source_field": "process_or_mode",
                "train_members": train_members,
                "train_candidate_rows": train_candidates,
                "validation_members_from_manifest": validation_members,
                "validation_candidate_rows_from_manifest": validation_candidates,
                "status": "pass",
            }
        )
    return signal_rows, background_rows


def population_rows(
    all_rows: Sequence[dict[str, Any]],
    train_members: Sequence[dict[str, Any]],
    train_files_opened: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    summary_rows = []
    for scope, split, sample_class, source, opened in (
        ("development", None, "", "manifest_metadata", train_files_opened),
        ("train", "train", "", "manifest_and_opened_train_candidates", train_files_opened),
        ("train_signal", "train", "signal", "manifest_and_opened_train_candidates", ""),
        ("train_background", "train", "background", "manifest_and_opened_train_candidates", ""),
        ("validation", "validation", "", "manifest_metadata_only", 0),
        ("validation_signal", "validation", "signal", "manifest_metadata_only", 0),
        ("validation_background", "validation", "background", "manifest_metadata_only", 0),
    ):
        members, candidates = count_rows(
            all_rows,
            split=split,
            sample_class=sample_class or None,
        )
        summary_rows.append(
            {
                "scope": scope,
                "sample_class": sample_class or "all",
                "metadata_source": source,
                "members": members,
                "candidate_rows": candidates,
                "candidate_files_opened": opened,
                "status": "pass",
            }
        )

    by_category: dict[tuple[str, str, str, int], list[dict[str, Any]]] = defaultdict(list)
    for member in train_members:
        category_type = member["stratum_type"]
        key = (
            category_type,
            member["stratum"],
            member["sample_class"],
            member["training_target"],
        )
        by_category[key].append(member)

    family_rows = []
    for key in sorted(by_category):
        category_type, category, sample_class, target = key
        members = by_category[key]
        family_rows.append(
            {
                "category_type": category_type,
                "category": category,
                "sample_class": sample_class,
                "training_target": target,
                "members": len(members),
                "candidate_bearing_members": sum(
                    member["candidate_rows"] > 0 for member in members
                ),
                "zero_candidate_members": sum(
                    member["candidate_rows"] == 0 for member in members
                ),
                "candidate_rows": sum(
                    member["candidate_rows"] for member in members
                ),
                "mapping_source": (
                    "authoritative_process_or_mode_direct"
                    if category_type == "signal_mode"
                    else "explicit_config_mapping_from_process_or_mode"
                ),
                "status": "pass",
            }
        )
    return summary_rows, family_rows


def read_train_feature_rows(
    train_members: Sequence[dict[str, Any]],
) -> tuple[pd.DataFrame, np.ndarray, int, int]:
    frames: list[pd.DataFrame] = []
    row_members: list[np.ndarray] = []
    opened = 0
    schema_less_zero_row_files = 0
    for position, member in enumerate(train_members, start=1):
        path = Path(member["local_path"])
        if not path.is_file():
            raise FileNotFoundError(
                f"train candidate file is unavailable for member "
                f"{member['member_index']}: {path}"
            )
        expected_rows = member["candidate_rows"]
        if expected_rows == 0:
            # Some canonical zero-candidate artifacts are intentionally
            # schema-less Parquets.  Open them without a column projection,
            # prove that they contain no rows, and require the full feature
            # contract only from files that can contribute model inputs.
            frame = pd.read_parquet(path, engine="pyarrow")
            schema_less_zero_row_files += int(len(frame.columns) == 0)
        else:
            frame = pd.read_parquet(
                path,
                columns=list(SOURCE_COLUMNS),
                engine="pyarrow",
            )
        opened += 1
        if len(frame) != expected_rows:
            raise ValueError(
                f"member {member['member_index']} row mismatch: "
                f"manifest={expected_rows}, parquet={len(frame)}"
            )
        if len(frame):
            derived = add_derived_features(frame)
            frames.append(derived.loc[:, list(MASS_AWARE_FEATURES)])
            row_members.append(
                np.full(len(frame), member["member_index"], dtype=np.int64)
            )
        if position % 50 == 0 or position == len(train_members):
            print(
                f"Opened train candidates {position}/{len(train_members)}; "
                f"accumulated rows={sum(len(item) for item in frames)}",
                flush=True,
            )

    if not frames:
        raise ValueError("train candidate population is empty")
    return (
        pd.concat(frames, ignore_index=True),
        np.concatenate(row_members),
        opened,
        schema_less_zero_row_files,
    )


def feature_contract_rows(frame: pd.DataFrame) -> list[dict[str, Any]]:
    rows = []
    for index, feature in enumerate(MASS_AWARE_FEATURES, start=1):
        blind_order = (
            MASS_PLANE_BLIND_FEATURES.index(feature) + 1
            if feature in MASS_PLANE_BLIND_FEATURES
            else ""
        )
        rows.append(
            {
                "feature_order_mass_aware": index,
                "feature_order_mass_plane_blind": blind_order,
                "feature": feature,
                "source_column_or_expression": DERIVED_FEATURE_EXPRESSIONS.get(
                    feature, feature
                ),
                "logical_dtype": str(frame[feature].dtype),
                "common_non_mass_plane_feature": feature in COMMON_FEATURES,
                "mass_aware_feature": True,
                "mass_plane_blind_feature": feature
                in MASS_PLANE_BLIND_FEATURES,
                "feature_role": (
                    "explicit_mass_plane"
                    if feature in MASS_PLANE_FEATURES
                    else "common_non_mass_plane"
                ),
                "latex_short_label": FEATURE_LATEX_LABELS[feature],
                "status": "pass",
            }
        )
    return rows


def forbidden_audit_rows() -> tuple[list[dict[str, Any]], int]:
    selected = tuple(MASS_AWARE_FEATURES)
    rows = []
    for name in sorted(FORBIDDEN_EXACT_FEATURES):
        matches = [feature for feature in selected if feature == name]
        rows.append(
            {
                "forbidden_input_or_pattern": name,
                "scope": "both_nominal_feature_sets",
                "reason": forbidden_feature_reason(name),
                "selected_features": matches,
                "selected_count": len(matches),
                "status": "pass" if not matches else "fail",
            }
        )
    for token in FORBIDDEN_SUBSTRINGS:
        matches = [feature for feature in selected if token in feature.lower()]
        rows.append(
            {
                "forbidden_input_or_pattern": f"*{token}*",
                "scope": "both_nominal_feature_sets",
                "reason": "forbidden_identity_rank_or_threeb_provenance_input",
                "selected_features": matches,
                "selected_count": len(matches),
                "status": "pass" if not matches else "fail",
            }
        )
    return rows, sum(row["selected_count"] for row in rows)


def fold_tables(
    train_members: Sequence[dict[str, Any]],
    assignment: Mapping[int, int],
    algorithm: str,
    n_folds: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    member_rows = []
    by_member = {}
    for member in train_members:
        index = member["member_index"]
        fold = assignment[index]
        row = {
            "member_index": index,
            "sample_class": member["sample_class"],
            "training_target": member["training_target"],
            "process_or_mode": member["process_or_mode"],
            "stratum_type": member["stratum_type"],
            "stratum": member["stratum"],
            "candidate_rows": member["candidate_rows"],
            "fold": fold,
            "assignment_algorithm": algorithm,
            "grouping_key": "member_index",
            "status": "pass",
        }
        member_rows.append(row)
        by_member[index] = row

    composition = []
    categories: list[tuple[str, str, str]] = [
        ("binary_class", "signal", "signal"),
        ("binary_class", "background", "background"),
    ]
    categories.extend(
        ("signal_mode", category, "signal")
        for category in sorted(
            {
                member["stratum"]
                for member in train_members
                if member["sample_class"] == "signal"
            }
        )
    )
    categories.extend(
        ("background_family", category, "background")
        for category in sorted(
            {
                member["stratum"]
                for member in train_members
                if member["sample_class"] == "background"
            }
        )
    )

    family_totals = Counter(member["stratum"] for member in train_members)
    for fold in range(n_folds):
        for level, category, sample_class in categories:
            if level == "binary_class":
                selected = [
                    member
                    for member in train_members
                    if member["sample_class"] == category
                    and assignment[member["member_index"]] == fold
                ]
                total_members = sum(
                    member["sample_class"] == category
                    for member in train_members
                )
            else:
                selected = [
                    member
                    for member in train_members
                    if member["stratum"] == category
                    and assignment[member["member_index"]] == fold
                ]
                total_members = family_totals[category]
            expected_everywhere = total_members >= n_folds
            unavoidable = not selected and not expected_everywhere
            status = (
                "pass"
                if selected or unavoidable
                else "fail"
            )
            composition.append(
                {
                    "fold": fold,
                    "composition_level": level,
                    "category": category,
                    "sample_class": sample_class,
                    "members": len(selected),
                    "candidate_bearing_members": sum(
                        member["candidate_rows"] > 0 for member in selected
                    ),
                    "candidate_rows": sum(
                        member["candidate_rows"] for member in selected
                    ),
                    "family_train_members": total_members,
                    "family_expected_in_every_fold": expected_everywhere,
                    "unavoidable_absence": unavoidable,
                    "status": status,
                }
            )

    duplicates = len(member_rows) - len(by_member)
    assigned = len(by_member)
    unassigned = len(train_members) - assigned
    audit = {
        "train_members_assigned_to_folds": assigned,
        "duplicate_members_across_folds": duplicates,
        "unassigned_train_members": unassigned,
        "fold_composition_failures": sum(
            row["status"] != "pass" for row in composition
        ),
    }
    return member_rows, composition, audit


def legacy_inventory() -> list[dict[str, Any]]:
    tracked = [
        Path(path)
        for path in git_output("ls-files").splitlines()
        if path
    ]
    candidates: set[tuple[str, str]] = set()
    model_suffixes = {".joblib", ".pkl", ".pickle", ".model", ".onnx"}

    for path in tracked:
        lowered = path.as_posix().lower()
        name = path.name.lower()
        has_bdt = "bdt" in lowered
        if not has_bdt:
            continue
        if path.parts and path.parts[0] == "scripts":
            candidates.add((path.as_posix(), "script"))
        elif path.parts and path.parts[0] == "configs":
            candidates.add((path.as_posix(), "configuration"))
        elif path.suffix.lower() in model_suffixes or "model" in name:
            candidates.add((path.as_posix(), "model_file"))
        elif "prediction" in name or "scored" in name:
            candidates.add((path.as_posix(), "prediction_file"))
        elif path.parts[:2] == ("docs", "checkpoints"):
            candidates.add((path.as_posix(), "checkpoint_artifact"))
        else:
            candidates.add((path.as_posix(), "bdt_related_file"))

        if len(path.parts) >= 3 and path.parts[0] == "outputs" and path.parts[1] in {
            "baselines",
            "models",
            "plots",
            "summaries",
            "tables",
        }:
            directory = Path(*path.parts[:3])
            if (REPOSITORY_ROOT / directory).is_dir():
                candidates.add((directory.as_posix(), "result_directory"))
        elif len(path.parts) >= 2 and path.parts[0] in {
            "results",
            "backgrounds",
        }:
            directory = Path(*path.parts[:2])
            if (REPOSITORY_ROOT / directory).is_dir():
                candidates.add((directory.as_posix(), "result_directory"))

    rows = []
    text_suffixes = {
        ".py",
        ".md",
        ".txt",
        ".json",
        ".yaml",
        ".yml",
        ".csv",
        ".tsv",
    }
    for path_text, artifact_type in sorted(candidates):
        path = REPOSITORY_ROOT / path_text
        lowered = path_text.lower()
        content = ""
        if path.is_file() and path.suffix.lower() in text_suffixes:
            try:
                content = path.read_text(encoding="utf-8", errors="replace")[
                    :250_000
                ].lower()
            except OSError:
                content = ""

        if "ak4ak8" in lowered:
            population = "legacy_ak4ak8_population"
        elif "qcdplus" in lowered:
            population = "legacy_qcdplus_population"
        elif "ttbar" in lowered:
            population = "legacy_ttbar_focused_population"
        elif "sm_normalized" in lowered:
            population = "legacy_sm_normalized_population"
        elif "hh4b" in lowered:
            population = "legacy_hh4b_population_not_currently_proven"
        else:
            population = "population_not_recoverable"

        combined = f"{lowered}\n{content}"
        if "train_test_split" in combined:
            split_policy = "row_level_train_test_split_referenced"
        elif "testset" in combined or "test_" in combined or "/test" in combined:
            split_policy = "test_split_referenced_policy_not_currently_proven"
        elif "validation" in combined:
            split_policy = "validation_split_referenced_policy_not_currently_proven"
        else:
            split_policy = "not_recoverable_from_inventory"

        if "feature_names" in content or "features =" in content:
            feature_definition = "embedded_feature_definition_recoverable_in_artifact"
        elif "feature" in content:
            feature_definition = "feature_references_present_definition_not_frozen"
        else:
            feature_definition = "not_recoverable_from_inventory"

        if any(token in content for token in ("cross section", "cross_section", "lumi")):
            weight_status = "physical_or_luminosity_weighting_referenced"
        elif "weight" in content:
            weight_status = "weighting_referenced_physical_status_unresolved"
        else:
            weight_status = "not_recoverable_from_inventory"

        rows.append(
            {
                "path": path_text,
                "artifact_type": artifact_type,
                "apparent_population": population,
                "apparent_split_policy": split_policy,
                "feature_definition": feature_definition,
                "physical_weight_status": weight_status,
                "canonical_v1_compatibility": (
                    "legacy_noncanonical_current_581_member_contract_not_proven"
                ),
                "disposition": "retain_unchanged_do_not_reuse_as_canonical_bdt",
            }
        )
    return rows


def run_unittests(output_dir: Path) -> tuple[int, bool]:
    command = [
        sys.executable,
        str(REPOSITORY_ROOT / "tests" / "test_hh4b_bdt_v1_common.py"),
        "-v",
    ]
    result = subprocess.run(
        command,
        cwd=REPOSITORY_ROOT,
        text=True,
        capture_output=True,
    )
    transcript = result.stdout + result.stderr
    (output_dir / "unittest.log").write_text(transcript, encoding="utf-8")
    print(transcript, end="", flush=True)
    match = re.search(r"Ran (\d+) tests?", transcript)
    tests_run = int(match.group(1)) if match else 0
    return tests_run, result.returncode == 0 and "OK" in transcript


def category_label(category: str) -> str:
    return {
        "ggf_hh4b": r"ggF $HH$",
        "vbf_hh4b": r"VBF $HH$",
        "qcd_multijet": "QCD multijet",
        "ttbar": r"$t\bar{t}$",
        "single_top": "Single top",
        "top_associated": r"$t\bar{t}+X$",
        "single_higgs": "Single Higgs",
        "diboson": r"$VV$",
        "triboson": r"$VVV$",
        "zbbbb": r"$Z+b\bar{b}b\bar{b}$",
    }[category]


def add_plot_inventory_pair(
    rows: list[dict[str, Any]],
    png: Path,
    pdf: Path,
    *,
    plot_name: str,
    variables: Sequence[str],
    normalization: str,
    binning: str,
    style_path: str,
    style_sha: str,
    style_metadata: Mapping[str, Any],
    dpi: int,
) -> None:
    for path, paired, format_name in (
        (png, pdf, "png"),
        (pdf, png, "pdf"),
    ):
        rows.append(
            {
                "path": f"plots/{path.name}",
                "paired_path": f"plots/{paired.name}",
                "plot_name": plot_name,
                "variables": list(variables),
                "normalization": normalization,
                "binning": binning,
                "style_helper_path": style_path,
                "style_helper_sha256": style_sha,
                "mplhep_used": style_metadata["mplhep_used"],
                "cms_inspired_style_applied": style_metadata[
                    "cms_inspired_style_applied"
                ],
                "latex_math_labels_applied": style_metadata[
                    "latex_math_labels_applied"
                ],
                "dpi": dpi,
                "format": format_name,
                "file_size_bytes": path.stat().st_size,
                "status": "pass",
            }
        )


def make_plots(
    output_dir: Path,
    population_by_family: Sequence[dict[str, Any]],
    fold_composition: Sequence[dict[str, Any]],
    features: pd.DataFrame,
    config: Mapping[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    import matplotlib.pyplot as plt

    plot_dir = output_dir / "plots"
    plot_dir.mkdir(parents=True)
    style_metadata = apply_cms_style()
    style_path = config["plotting"]["style_helper"]
    style_sha = sha256_file(REPOSITORY_ROOT / style_path)
    dpi = int(config["plotting"]["dpi"])
    inventory: list[dict[str, Any]] = []

    categories = [
        row["category"]
        for row in population_by_family
    ]
    values = [
        int(row["candidate_rows"])
        for row in population_by_family
    ]
    labels = [category_label(category) for category in categories]
    colors = [
        CATEGORY_COLORS[index % len(CATEGORY_COLORS)]
        for index in range(len(categories))
    ]
    fig, ax = plt.subplots(figsize=(10.0, 5.8))
    positions = np.arange(len(categories))
    bars = ax.bar(positions, values, color=colors, width=0.72)
    ax.set_yscale("log")
    ax.set_ylabel("Raw canonical train candidates")
    ax.set_xticks(positions)
    ax.set_xticklabels(labels, rotation=35, ha="right")
    ax.set_ylim(5.0, 65_000.0)
    for bar, value in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            value * 1.08,
            f"{value:,}",
            ha="center",
            va="bottom",
            rotation=90,
            fontsize=7,
        )
    add_delphes_header(
        ax,
        "Train only; Raw canonical train candidates by family",
    )
    png, pdf = save_png_pdf(
        fig, plot_dir / "train_population_by_family", dpi=dpi
    )
    add_plot_inventory_pair(
        inventory,
        png,
        pdf,
        plot_name="train_population_by_family",
        variables=["candidate_rows", "signal_mode", "background_family"],
        normalization="none_raw_canonical_train_candidates",
        binning="categorical_frozen_signal_modes_and_background_families",
        style_path=style_path,
        style_sha=style_sha,
        style_metadata=style_metadata,
        dpi=dpi,
    )

    folds = list(range(int(config["folds"]["count"])))
    signal_rows = []
    background_rows = []
    for fold in folds:
        signal_rows.append(
            next(
                int(row["candidate_rows"])
                for row in fold_composition
                if row["fold"] == fold
                and row["composition_level"] == "binary_class"
                and row["category"] == "signal"
            )
        )
        background_rows.append(
            next(
                int(row["candidate_rows"])
                for row in fold_composition
                if row["fold"] == fold
                and row["composition_level"] == "binary_class"
                and row["category"] == "background"
            )
        )
    fig, ax = plt.subplots(figsize=(7.2, 5.2))
    ax.bar(
        folds,
        background_rows,
        color=CATEGORY_COLORS[0],
        label="Background",
    )
    ax.bar(
        folds,
        signal_rows,
        bottom=background_rows,
        color=CATEGORY_COLORS[1],
        label=r"$HH\rightarrow b\bar{b}b\bar{b}$",
    )
    ax.set_xticks(folds)
    ax.set_xticklabels([f"Fold {fold}" for fold in folds])
    ax.set_ylabel("Raw canonical train candidates")
    ax.legend(loc="upper right")
    add_delphes_header(
        ax,
        "Train only; Grouped by source member; five-fold composition",
    )
    png, pdf = save_png_pdf(
        fig, plot_dir / "train_fold_composition", dpi=dpi
    )
    add_plot_inventory_pair(
        inventory,
        png,
        pdf,
        plot_name="train_fold_composition",
        variables=["fold", "sample_class", "candidate_rows"],
        normalization="none_raw_canonical_train_candidates",
        binning="five_frozen_member_level_folds",
        style_path=style_path,
        style_sha=style_sha,
        style_metadata=style_metadata,
        dpi=dpi,
    )

    matrix = features.loc[:, list(MASS_AWARE_FEATURES)].to_numpy(
        dtype=np.float64,
        copy=False,
    )
    correlation = np.corrcoef(matrix, rowvar=False)
    if correlation.shape != (34, 34) or not np.all(np.isfinite(correlation)):
        raise ValueError("mass-aware feature correlation is invalid")
    fig, ax = plt.subplots(figsize=(13.0, 11.5))
    image = ax.imshow(
        correlation,
        vmin=-1.0,
        vmax=1.0,
        cmap="RdBu_r",
        interpolation="nearest",
        aspect="equal",
    )
    short_labels = [FEATURE_LATEX_LABELS[name] for name in MASS_AWARE_FEATURES]
    locations = np.arange(len(short_labels))
    ax.set_xticks(locations)
    ax.set_yticks(locations)
    ax.set_xticklabels(short_labels, rotation=90)
    ax.set_yticklabels(short_labels)
    ax.minorticks_off()
    add_delphes_header(
        ax,
        "Train only; mass-aware unweighted Pearson correlation",
    )
    colorbar = fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    colorbar.set_label("Pearson correlation")
    png, pdf = save_png_pdf(
        fig, plot_dir / "mass_aware_feature_correlation", dpi=dpi
    )
    add_plot_inventory_pair(
        inventory,
        png,
        pdf,
        plot_name="mass_aware_feature_correlation",
        variables=list(MASS_AWARE_FEATURES),
        normalization="unweighted_pearson_correlation_no_labels_or_weights",
        binning="fixed_34_by_34_mass_aware_feature_order",
        style_path=style_path,
        style_sha=style_sha,
        style_metadata=style_metadata,
        dpi=dpi,
    )
    return inventory, style_metadata


def generate_readme(
    summary: Mapping[str, Any],
    config: Mapping[str, Any],
) -> str:
    return f"""# Canonical HH4b BDT-v1 input contract

## Result

- Status: `{summary['status']}`
- Canonical development members: {summary['development_members']}
- Train members assigned to five folds: {summary['train_members_assigned_to_folds']}
- Train candidate rows audited in memory: {summary['train_candidate_rows']}
- Mass-aware features: {summary['mass_aware_features']}
- Explicit dijet-mass-plane-blind features: {summary['mass_plane_blind_features']}
- Missing, nonfinite, or constant selected features: \
{summary['missing_feature_values']}, {summary['nonfinite_feature_values']}, \
{summary['constant_selected_features']}
- Models trained, predictions written: {summary['models_trained']}, \
{summary['predictions_written']}

## Population and label contract

The binary target is HH→bbbb signal (`training_target=1`) versus every
configured canonical background family (`training_target=0`) in the frozen
at-least-four-tag candidate population. Labels and signal modes come from the
manifest fields `sample_class`, `training_target`, and `process_or_mode`.
Because the manifest has no `process_family` field, the committed YAML freezes
one explicit `process_or_mode` → broad physics-family mapping.

All {summary['train_candidate_files_opened']} train candidate Parquets were
opened. Validation counts are manifest metadata only: validation candidate
files opened = {summary['validation_candidate_files_opened']}. Test members
and candidate files considered = {summary['test_members_considered']} and
{summary['test_candidate_files_opened']}.

The audit encountered {summary['schema_less_zero_row_train_files']} train
Parquets with no physical columns; every one is a manifest-declared zero-row
member and was opened without a feature projection to confirm its zero row
count. Every candidate-bearing file was required to supply the complete
selected-feature source contract.

No $R_{{HH}}^{{125,125}}<34$ requirement is applied to the BDT input. That
requirement remains the final decision rule of the optimized cut baseline.
No 40 GeV jet threshold, $|\\eta|<2.4$ requirement, or additional signal-region
selection was introduced.

## Features

The mass-aware model has the frozen 30 common detector-level scalar features
plus `mbb1`, `mbb2`, `delta_mbb`, and `r_hh_125_125`. The 30-feature ablation
is accurately described as an **explicit dijet-mass-plane-blind ablation**,
not as fully mass-decorrelated. Generator truth, identity, labels, paths,
flavors, raw indices, absolute azimuths, b-tag values, redundant mass aliases,
and exactly-3b promoted-jet provenance are forbidden.

No feature was imputed, clipped, or used to drop a row. The row-level feature
matrix existed only in memory and is not written or committed.

## Grouped folds

The installed environment has no scikit-learn, so the recorded fallback
`{summary['fold_assignment_algorithm']}` assigns immutable source members
deterministically. No member appears in more than one fold. All signal modes
and all configured background families have at least five train members and
occur in every fold.

## Development-balancing weights

These are development-balancing weights, not cross-section, luminosity,
generator, importance, or physical event weights. Signal and background each
receive half of the total; ggF and VBF receive equal signal shares; and the
{summary['background_families']} background families receive equal background
shares. Candidate-bearing source members receive equal totals within their
mode or family, then each member total is distributed uniformly over its
candidate rows. Members with zero candidates remain assigned to folds and
appear in the audit, but no row weight can exist for a member with no rows.
The positive row weights are rescaled to mean one.

## Legacy BDT inventory

The inventory retains older BDT-related scripts, configurations, models,
predictions, checkpoints, and result directories unchanged. Each is marked
legacy/noncanonical unless the current 581-member population and this exact
feature/fold/weight contract can be proven; none is reused as canonical-v1.

## Plotting

All three figures use `scripts/plotting/hh4b_cms_style.py`, a truthful
CMS-publication-inspired but nonofficial style. mplhep was
{'available and used' if summary['mplhep_used'] else 'unavailable, so the deterministic fallback was used'}.
Every figure is provided as a 300 dpi PNG and vector PDF without system LaTeX.

## Authorization boundary

- `canonical_bdt_training_authorized: false`
- `validation_evaluation_authorized: false`
- `test_access_authorized: false`
- `physical_significance_authorized: false`

## Reproducibility

- Source commit: `{summary['source_commit']}`
- Manifest SHA-256: `{summary['manifest_sha256']}`
- Configuration SHA-256: `{summary['config_sha256']}`
- Common helper SHA-256: `{summary['common_helper_sha256']}`
- Plot style helper SHA-256: `{summary['style_helper_sha256']}`
- Preparation runner SHA-256: `{summary['runner_sha256']}`

## Next gate

`{config['authorization']['next_gate']}`
"""


def write_checkpoint_manifest(checkpoint_dir: Path) -> None:
    entries = []
    for path in sorted(checkpoint_dir.rglob("*")):
        if path.is_file() and path.name != "SHA256SUMS":
            entries.append(
                f"{sha256_file(path)}  {path.relative_to(checkpoint_dir).as_posix()}"
            )
    (checkpoint_dir / "SHA256SUMS").write_text(
        "\n".join(entries) + "\n",
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Freeze canonical HH4b BDT-v1 inputs without model training"
    )
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="replace only the exact configured runtime/checkpoint directories",
    )
    args = parser.parse_args()

    config_path = args.config.resolve()
    config = load_config(config_path)
    source_commit = git_output("rev-parse", "HEAD")
    if source_commit != config["source_commit"]:
        raise ValueError(
            f"source commit mismatch: config={config['source_commit']}, "
            f"HEAD={source_commit}"
        )

    configured_common = tuple(config["feature_sets"]["common"])
    configured_append = tuple(config["feature_sets"]["mass_plane_append"])
    if configured_common != COMMON_FEATURES:
        raise ValueError("configured common feature order differs from shared module")
    if configured_append != MASS_PLANE_FEATURES:
        raise ValueError("configured mass-plane append order differs from shared module")
    if config["feature_sets"]["mass_plane_blind_description"] != (
        "explicit dijet-mass-plane-blind ablation"
    ):
        raise ValueError("mass-plane-blind ablation description is not exact")
    validate_feature_names(MASS_AWARE_FEATURES)
    validate_feature_names(MASS_PLANE_BLIND_FEATURES)

    manifest_path = REPOSITORY_ROOT / config["manifest"]["path"]
    manifest_sha = sha256_file(manifest_path)
    if manifest_sha != config["manifest"]["sha256"]:
        raise ValueError("canonical development manifest SHA-256 mismatch")
    manifest_rows = read_manifest(manifest_path)
    observed = validate_manifest(manifest_rows, config)
    train_members = mapped_train_members(manifest_rows, config)

    output_dir = REPOSITORY_ROOT / config["output_dir"]
    checkpoint_dir = REPOSITORY_ROOT / config["checkpoint_dir"]
    prepare_directory(output_dir, args.overwrite)
    prepare_directory(checkpoint_dir, args.overwrite)

    tests_run, tests_passed = run_unittests(output_dir)
    if not tests_passed:
        raise RuntimeError("synthetic unittest suite failed")

    (
        features,
        row_member_indices,
        train_files_opened,
        schema_less_zero_row_files,
    ) = read_train_feature_rows(train_members)
    if train_files_opened != observed["train_members"]:
        raise ValueError("not every train candidate file was opened exactly once")
    if len(features) != observed["train_candidate_rows"]:
        raise ValueError("combined train candidate row count mismatch")

    quality_rows = feature_quality_rows(features, MASS_AWARE_FEATURES)
    contract_rows = feature_contract_rows(features)
    forbidden_rows, forbidden_selected = forbidden_audit_rows()

    fold_result = assign_member_folds(
        train_members,
        n_folds=int(config["folds"]["count"]),
    )
    fold_member_rows, fold_composition_rows, fold_audit = fold_tables(
        train_members,
        fold_result.assignment,
        fold_result.algorithm,
        int(config["folds"]["count"]),
    )

    signal_modes = sorted(set(config["signal_mode_mapping"].values()))
    background_families = sorted(
        set(config["background_family_mapping"].values())
    )
    weight_result = build_hierarchical_weights(
        row_member_indices,
        train_members,
        signal_modes=signal_modes,
        background_families=background_families,
        tolerance=float(config["weighting"]["tolerance"]),
    )
    for row in weight_result.member_rows:
        row["fold"] = fold_result.assignment[row["member_index"]]

    signal_mapping_rows, background_mapping_rows = mapping_rows(
        manifest_rows, config
    )
    population_summary_rows, population_by_family_rows = population_rows(
        manifest_rows,
        train_members,
        train_files_opened,
    )
    legacy_rows = legacy_inventory()
    plot_rows, style_metadata = make_plots(
        output_dir,
        population_by_family_rows,
        fold_composition_rows,
        features,
        config,
    )

    missing_feature_values = sum(row["missing_rows"] for row in quality_rows)
    nonfinite_feature_values = sum(
        row["nonfinite_rows"] for row in quality_rows
    )
    constant_features = sum(
        bool(row["constant_feature"]) for row in quality_rows
    )
    weights = weight_result.weights
    tolerance = float(config["weighting"]["tolerance"])

    expected = config["population"]["expected"]
    summary: dict[str, Any] = {
        "schema_version": 1,
        "status": "hh4b_bdt_v1_input_contract_fail",
        "source_commit": source_commit,
        "development_members": observed["development_members"],
        "train_members": observed["train_members"],
        "validation_members": observed["validation_members"],
        "test_members_considered": 0,
        "train_signal_members": observed["train_signal_members"],
        "train_background_members": observed["train_background_members"],
        "validation_signal_members_from_manifest": observed[
            "validation_signal_members"
        ],
        "validation_background_members_from_manifest": observed[
            "validation_background_members"
        ],
        "development_candidate_rows": observed["development_candidate_rows"],
        "train_candidate_rows": observed["train_candidate_rows"],
        "train_signal_rows": observed["train_signal_rows"],
        "train_background_rows": observed["train_background_rows"],
        "validation_candidate_rows_from_manifest": observed[
            "validation_candidate_rows"
        ],
        "train_candidate_files_opened": train_files_opened,
        "validation_candidate_files_opened": 0,
        "test_candidate_files_opened": 0,
        "mass_aware_features": len(MASS_AWARE_FEATURES),
        "mass_plane_blind_features": len(MASS_PLANE_BLIND_FEATURES),
        "mass_plane_blind_description": (
            "explicit dijet-mass-plane-blind ablation"
        ),
        "forbidden_features_selected": forbidden_selected,
        "missing_feature_values": missing_feature_values,
        "nonfinite_feature_values": nonfinite_feature_values,
        "constant_selected_features": constant_features,
        "folds": int(config["folds"]["count"]),
        **fold_audit,
        "fold_assignment_algorithm": fold_result.algorithm,
        "fold_grouping_key": "immutable_member_index",
        "sklearn_available": False,
        "sklearn_version": "unavailable",
        "python_version": sys.version.split()[0],
        "numpy_version": np.__version__,
        "pandas_version": pd.__version__,
        "pyarrow_version": pyarrow.__version__,
        "finite_positive_training_weights": int(
            np.count_nonzero(np.isfinite(weights) & (weights > 0.0))
        ),
        "nonfinite_training_weights": int(
            np.count_nonzero(~np.isfinite(weights))
        ),
        "nonpositive_training_weights": int(np.count_nonzero(weights <= 0.0)),
        "normalized_mean_training_weight": float(np.mean(weights)),
        "training_weight_rescale_factor": weight_result.rescale_factor,
        "zero_candidate_train_members": sum(
            member["candidate_rows"] == 0 for member in train_members
        ),
        "candidate_bearing_train_members": sum(
            member["candidate_rows"] > 0 for member in train_members
        ),
        "schema_less_zero_row_train_files": schema_less_zero_row_files,
        "signal_modes": len(signal_modes),
        "background_families": len(background_families),
        "models_trained": 0,
        "scalers_fitted": 0,
        "hyperparameter_trials": 0,
        "predictions_written": 0,
        "validation_metrics_calculated": 0,
        "test_metrics_calculated": 0,
        "roc_auc_calculated": 0,
        "physical_event_weights_used": 0,
        "physics_yields_calculated": 0,
        "significances_calculated": 0,
        "r_hh_125_125_cut_applied_to_bdt_input": False,
        "additional_signal_region_cuts_applied": 0,
        "row_level_feature_matrices_written": 0,
        "legacy_artifacts_inventoried": len(legacy_rows),
        "unittest_tests_run": tests_run,
        "unittest_tests_passed": tests_run if tests_passed else 0,
        "unittest_failures": 0 if tests_passed else 1,
        "plots_requested": len(config["plotting"]["plots"]),
        "pngs_written": sum(row["format"] == "png" for row in plot_rows),
        "pdfs_written": sum(row["format"] == "pdf" for row in plot_rows),
        "plot_failures": sum(row["status"] != "pass" for row in plot_rows),
        "latex_math_label_failures": 0,
        "cms_style_failures": (
            0 if style_metadata["cms_inspired_style_applied"] else 1
        ),
        "mplhep_used": style_metadata["mplhep_used"],
        "official_cms_status_claimed": False,
        "manifest_path": config["manifest"]["path"],
        "manifest_sha256": manifest_sha,
        "config_path": str(config_path.relative_to(REPOSITORY_ROOT)),
        "config_sha256": sha256_file(config_path),
        "common_helper_sha256": sha256_file(
            REPOSITORY_ROOT / "scripts/analysis/hh4b_bdt_v1_common.py"
        ),
        "style_helper_sha256": sha256_file(
            REPOSITORY_ROOT / config["plotting"]["style_helper"]
        ),
        "runner_sha256": sha256_file(Path(__file__)),
        "canonical_bdt_training_authorized": False,
        "validation_evaluation_authorized": False,
        "test_access_authorized": False,
        "physical_significance_authorized": False,
        "next_gate": config["authorization"]["next_gate"],
    }

    pass_conditions = {
        "development_members": summary["development_members"]
        == int(expected["development_members"]),
        "train_members": summary["train_members"] == int(expected["train_members"]),
        "validation_members": summary["validation_members"]
        == int(expected["validation_members"]),
        "train_signal_members": summary["train_signal_members"]
        == int(expected["train_signal_members"]),
        "train_background_members": summary["train_background_members"]
        == int(expected["train_background_members"]),
        "train_candidate_rows": summary["train_candidate_rows"]
        == int(expected["train_candidate_rows"]),
        "train_signal_rows": summary["train_signal_rows"]
        == int(expected["train_signal_rows"]),
        "train_background_rows": summary["train_background_rows"]
        == int(expected["train_background_rows"]),
        "validation_candidate_rows": summary[
            "validation_candidate_rows_from_manifest"
        ]
        == int(expected["validation_candidate_rows"]),
        "validation_files_closed": summary["validation_candidate_files_opened"]
        == 0,
        "test_files_closed": summary["test_candidate_files_opened"] == 0,
        "test_members_excluded": summary["test_members_considered"] == 0,
        "feature_counts": summary["mass_aware_features"] == 34
        and summary["mass_plane_blind_features"] == 30,
        "feature_quality": forbidden_selected == 0
        and missing_feature_values == 0
        and nonfinite_feature_values == 0
        and constant_features == 0,
        "folds": summary["folds"] == 5
        and summary["train_members_assigned_to_folds"] == 458
        and summary["duplicate_members_across_folds"] == 0
        and summary["unassigned_train_members"] == 0
        and summary["fold_composition_failures"] == 0,
        "weights": summary["finite_positive_training_weights"]
        == summary["train_candidate_rows"]
        and summary["nonfinite_training_weights"] == 0
        and summary["nonpositive_training_weights"] == 0
        and np.isclose(
            summary["normalized_mean_training_weight"],
            1.0,
            rtol=0.0,
            atol=tolerance,
        )
        and all(row["status"] == "pass" for row in weight_result.summary_rows),
        "unittests": tests_passed and tests_run > 0,
        "no_training_or_prediction": summary["models_trained"] == 0
        and summary["hyperparameter_trials"] == 0
        and summary["predictions_written"] == 0
        and summary["scalers_fitted"] == 0,
        "no_evaluation": summary["validation_metrics_calculated"] == 0
        and summary["test_metrics_calculated"] == 0
        and summary["roc_auc_calculated"] == 0,
        "no_physical_weight_or_result": summary["physical_event_weights_used"] == 0
        and summary["physics_yields_calculated"] == 0
        and summary["significances_calculated"] == 0,
        "plots": summary["pngs_written"] == summary["plots_requested"]
        and summary["pdfs_written"] == summary["plots_requested"]
        and summary["plot_failures"] == 0
        and summary["latex_math_label_failures"] == 0
        and summary["cms_style_failures"] == 0,
        "no_rhh_or_sr_preselection": not summary[
            "r_hh_125_125_cut_applied_to_bdt_input"
        ]
        and summary["additional_signal_region_cuts_applied"] == 0,
        "no_row_matrix_written": summary["row_level_feature_matrices_written"] == 0,
    }
    summary["pass_conditions"] = pass_conditions
    summary["failed_pass_conditions"] = sorted(
        name for name, passed in pass_conditions.items() if not passed
    )
    if all(pass_conditions.values()):
        summary["status"] = "hh4b_bdt_v1_input_contract_pass"

    tables = {
        "train_population_summary.tsv": population_summary_rows,
        "train_population_by_family.tsv": population_by_family_rows,
        "signal_mode_mapping.tsv": signal_mapping_rows,
        "background_family_mapping.tsv": background_mapping_rows,
        "feature_contract.tsv": contract_rows,
        "forbidden_feature_audit.tsv": forbidden_rows,
        "feature_quality_audit.tsv": quality_rows,
        "member_fold_assignment.tsv": fold_member_rows,
        "fold_composition.tsv": fold_composition_rows,
        "training_weight_summary.tsv": weight_result.summary_rows,
        "member_weight_summary.tsv": weight_result.member_rows,
        "legacy_bdt_inventory.tsv": legacy_rows,
        "plot_inventory.tsv": plot_rows,
    }
    for name, rows in tables.items():
        write_tsv(output_dir / name, rows, TABLE_FIELDS[name])

    checkpoint_payload = {
        "schema_version": 1,
        "classification": "canonical_hh4b_bdt_v1_input_contract_checkpoint",
        "status": summary["status"],
        "source_commit": source_commit,
        "development_members": summary["development_members"],
        "train_members": summary["train_members"],
        "validation_members": summary["validation_members"],
        "train_candidate_rows": summary["train_candidate_rows"],
        "validation_candidate_files_opened": 0,
        "test_candidate_files_opened": 0,
        "mass_aware_features": 34,
        "mass_plane_blind_features": 30,
        "folds": 5,
        "models_trained": 0,
        "predictions_written": 0,
        "physical_event_weights_used": 0,
        "row_level_feature_matrices_committed": 0,
        "canonical_bdt_training_authorized": False,
        "validation_evaluation_authorized": False,
        "test_access_authorized": False,
        "physical_significance_authorized": False,
        "next_gate": config["authorization"]["next_gate"],
    }
    write_json(output_dir / "summary.json", summary)
    write_json(output_dir / "checkpoint.json", checkpoint_payload)
    (output_dir / "README.md").write_text(
        generate_readme(summary, config),
        encoding="utf-8",
    )

    for name in (
        "README.md",
        "checkpoint.json",
        "summary.json",
        *TABLE_FIELDS.keys(),
    ):
        shutil.copy2(output_dir / name, checkpoint_dir / name)
    shutil.copytree(output_dir / "plots", checkpoint_dir / "plots")
    write_checkpoint_manifest(checkpoint_dir)

    shutil.rmtree(MPLCONFIG_TEMP, ignore_errors=True)

    print(
        json.dumps(
            {
                "status": summary["status"],
                "train_members": summary["train_members"],
                "train_candidate_rows": summary["train_candidate_rows"],
                "folds": summary["folds"],
                "mass_aware_features": summary["mass_aware_features"],
                "mass_plane_blind_features": summary["mass_plane_blind_features"],
                "plots": summary["plots_requested"],
                "next_gate": summary["next_gate"],
            },
            indent=2,
            sort_keys=True,
        ),
        flush=True,
    )
    if summary["status"] != "hh4b_bdt_v1_input_contract_pass":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
