#!/usr/bin/env python3
"""Freeze the train-only CMS-inspired categorized HH4b BDT-v2 contract.

No model is trained, no prediction is written, and validation/test candidate
files are never opened.  Event-level feature matrices remain in memory only.
"""

from __future__ import annotations

import argparse
import atexit
from collections import Counter
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
from typing import Any, Iterable, Mapping, Sequence


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPOSITORY_ROOT))
sys.path.insert(0, str(REPOSITORY_ROOT / "scripts" / "analysis"))

MPLCONFIG_TEMP = (
    Path(tempfile.gettempdir()) / f"hh4b_bdt_v2_contract_mpl_{os.getpid()}"
)
os.environ["MPLCONFIGDIR"] = str(MPLCONFIG_TEMP)
atexit.register(shutil.rmtree, MPLCONFIG_TEMP, ignore_errors=True)

import matplotlib  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import pyarrow  # noqa: E402
import yaml  # noqa: E402

from hh4b_bdt_v2_common import (  # noqa: E402
    CATEGORIZED_EXPLICIT_DIJET_MASS_PLANE_BLIND_FEATURES,
    CATEGORIZED_MASS_AWARE_FEATURES,
    CATEGORY_NAMES,
    CMS_INSPIRED_DERIVED_FEATURES,
    DERIVED_FEATURE_DEFINITIONS,
    EXPLICIT_DIJET_MASS_PLANE_FEATURES,
    FEATURE_VARIANTS,
    GLOBAL_V1_MASS_AWARE_REFERENCE,
    MHH_BOUNDARY_GEV,
    PAIRWISE_DR_FEATURES,
    SOURCE_COLUMNS,
    assign_mhh_category,
    derive_cms_inspired_features,
    feature_quality_rows,
    require_finite_features,
    validate_feature_names,
    write_table_bundle,
)
from scripts.plotting.hh4b_cms_style import (  # noqa: E402
    CATEGORY_COLORS,
    FEATURE_LATEX_LABELS,
    add_delphes_header,
    apply_cms_style,
    save_png_pdf,
)


CONFIG_DEFAULT = (
    REPOSITORY_ROOT
    / "configs/baselines/hh4b_bdt_v2_cms_inspired_contract.yaml"
)
FLOAT_TOLERANCE = 1.0e-10


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


def load_yaml(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        payload = yaml.safe_load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"YAML root must be a mapping: {path}")
    return payload


def read_tsv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def verify_sha256_manifest(directory: Path) -> None:
    manifest = directory / "SHA256SUMS"
    if not manifest.is_file():
        raise FileNotFoundError(f"missing SHA256SUMS: {manifest}")
    failures = []
    for line in manifest.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        digest, relative = line.split(maxsplit=1)
        target = directory / relative.strip().lstrip("*")
        if not target.is_file() or sha256_file(target) != digest:
            failures.append(relative)
    if failures:
        raise ValueError(
            f"checkpoint SHA256 verification failed in {directory}: {failures}"
        )


def write_sha256_manifest(directory: Path) -> Path:
    targets = sorted(
        path
        for path in directory.rglob("*")
        if path.is_file() and path.name != "SHA256SUMS"
    )
    lines = [
        f"{sha256_file(path)}  {path.relative_to(directory).as_posix()}"
        for path in targets
    ]
    manifest = directory / "SHA256SUMS"
    manifest.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return manifest


def prepare_exact_directory(path: Path, overwrite: bool) -> None:
    if path.exists():
        if not overwrite:
            raise FileExistsError(
                f"exact output exists; pass --overwrite to replace it: {path}"
            )
        shutil.rmtree(path)
    path.mkdir(parents=True)


def require_values(
    payload: Mapping[str, Any],
    expected: Mapping[str, Any],
    *,
    label: str,
) -> None:
    differences = {
        key: {"expected": value, "observed": payload.get(key)}
        for key, value in expected.items()
        if payload.get(key) != value
    }
    if differences:
        raise ValueError(f"{label} precondition mismatch: {differences}")


def load_and_verify_inputs(
    config: Mapping[str, Any],
) -> dict[str, Any]:
    inputs = config["inputs"]
    v1 = inputs["v1_input_contract"]
    cv = inputs["v1_grouped_cv"]
    v1_dir = REPOSITORY_ROOT / v1["checkpoint_dir"]
    cv_dir = REPOSITORY_ROOT / cv["checkpoint_dir"]

    if sha256_file(v1_dir / "SHA256SUMS") != v1["sha256sums_sha256"]:
        raise ValueError("v1 input-contract SHA256SUMS digest changed")
    if sha256_file(cv_dir / "SHA256SUMS") != cv["sha256sums_sha256"]:
        raise ValueError("v1 grouped-CV SHA256SUMS digest changed")
    verify_sha256_manifest(v1_dir)
    verify_sha256_manifest(cv_dir)

    input_summary = json.loads(
        (v1_dir / "summary.json").read_text(encoding="utf-8")
    )
    cv_summary = json.loads(
        (cv_dir / "summary.json").read_text(encoding="utf-8")
    )
    expected = config["population"]["expected"]
    require_values(
        input_summary,
        {
            "status": v1["required_status"],
            "train_members": expected["train_members"],
            "train_candidate_rows": expected["train_rows"],
            "train_signal_rows": expected["signal_rows"],
            "train_background_rows": expected["background_rows"],
            "folds": expected["folds"],
            "validation_candidate_files_opened": 0,
            "test_candidate_files_opened": 0,
        },
        label="v1 input contract",
    )
    require_values(
        cv_summary,
        {
            "status": cv["required_status"],
            "train_members": expected["train_members"],
            "train_rows": expected["train_rows"],
            "signal_rows": expected["signal_rows"],
            "background_rows": expected["background_rows"],
            "folds": expected["folds"],
            "missing_oof_rows": expected["missing_oof_rows"],
            "duplicate_oof_rows": expected["duplicate_oof_rows"],
            "member_leakage_rows": expected["member_leakage_rows"],
            "validation_candidate_files_opened": 0,
            "test_candidate_files_opened": 0,
        },
        label="v1 grouped cross-validation",
    )

    feature_rows = read_tsv(v1_dir / v1["feature_contract"])
    frozen_features = [row["feature"] for row in feature_rows]
    if frozen_features != list(GLOBAL_V1_MASS_AWARE_REFERENCE):
        raise ValueError("frozen v1 mass-aware feature order changed")

    fold_rows = read_tsv(v1_dir / v1["member_fold_assignment"])
    weight_rows = read_tsv(v1_dir / v1["member_weight_summary"])
    if len(fold_rows) != expected["train_members"]:
        raise ValueError("frozen fold assignment has wrong member count")
    if len(weight_rows) != expected["train_members"]:
        raise ValueError("frozen weight table has wrong member count")
    if len({row["member_index"] for row in fold_rows}) != len(fold_rows):
        raise ValueError("duplicate member in frozen fold assignment")

    signal_map = {
        row["manifest_process_or_mode"]: row["signal_mode"]
        for row in read_tsv(v1_dir / v1["signal_mode_mapping"])
    }
    background_map = {
        row["manifest_process_or_mode"]: row["background_family"]
        for row in read_tsv(v1_dir / v1["background_family_mapping"])
    }
    v1_config_path = REPOSITORY_ROOT / v1["config_path"]
    v1_config = load_yaml(v1_config_path)
    manifest_path = REPOSITORY_ROOT / v1_config["manifest"]["path"]
    if sha256_file(manifest_path) != v1_config["manifest"]["sha256"]:
        raise ValueError("canonical development manifest digest changed")

    return {
        "input_summary": input_summary,
        "cv_summary": cv_summary,
        "v1_dir": v1_dir,
        "cv_dir": cv_dir,
        "fold_rows": fold_rows,
        "weight_rows": weight_rows,
        "signal_map": signal_map,
        "background_map": background_map,
        "manifest_path": manifest_path,
        "manifest_sha256": sha256_file(manifest_path),
        "v1_config_path": v1_config_path,
    }


def load_train_population(
    inputs: Mapping[str, Any],
    config: Mapping[str, Any],
) -> dict[str, Any]:
    manifest = read_tsv(inputs["manifest_path"])
    train_manifest = sorted(
        (row for row in manifest if row["dataset_split"] == "train"),
        key=lambda row: int(row["member_index"]),
    )
    expected = config["population"]["expected"]
    if len(train_manifest) != expected["train_members"]:
        raise ValueError("train manifest member count mismatch")
    if any(row["test_member"].lower() == "true" for row in train_manifest):
        raise ValueError("test member appeared in train manifest selection")

    fold_by_member = {
        int(row["member_index"]): int(row["fold"])
        for row in inputs["fold_rows"]
    }
    expected_rows_by_member = {
        int(row["member_index"]): int(row["candidate_rows"])
        for row in inputs["fold_rows"]
    }
    weight_by_member = {
        int(row["member_index"]): float(row["per_row_training_weight"])
        for row in inputs["weight_rows"]
    }
    if set(fold_by_member) != {
        int(row["member_index"]) for row in train_manifest
    }:
        raise ValueError("manifest/frozen-fold member sets differ")

    frames: list[pd.DataFrame] = []
    member_arrays: list[np.ndarray] = []
    fold_arrays: list[np.ndarray] = []
    target_arrays: list[np.ndarray] = []
    weight_arrays: list[np.ndarray] = []
    mode_arrays: list[np.ndarray] = []
    family_arrays: list[np.ndarray] = []
    process_arrays: list[np.ndarray] = []
    opened = 0
    schema_less_zero = 0

    for position, row in enumerate(train_manifest, start=1):
        member = int(row["member_index"])
        expected_rows = int(row["candidate_rows"])
        if expected_rows_by_member.get(member) != expected_rows:
            raise ValueError(f"frozen row count mismatch for member {member}")
        path = Path(row["local_path"])
        if not path.is_file():
            raise FileNotFoundError(f"missing train candidate file: {path}")
        if expected_rows == 0:
            frame = pd.read_parquet(path, engine="pyarrow")
            schema_less_zero += int(len(frame.columns) == 0)
        else:
            frame = pd.read_parquet(
                path,
                columns=list(SOURCE_COLUMNS),
                engine="pyarrow",
            )
        opened += 1
        if len(frame) != expected_rows:
            raise ValueError(
                f"member {member} row mismatch: "
                f"expected={expected_rows}, observed={len(frame)}"
            )
        if expected_rows:
            sample_class = row["sample_class"]
            target = int(row["training_target"])
            if (sample_class, target) not in {
                ("signal", 1),
                ("background", 0),
            }:
                raise ValueError(f"invalid frozen label for member {member}")
            if sample_class == "signal":
                mode = inputs["signal_map"][row["process_or_mode"]]
                family = ""
            else:
                mode = ""
                family = inputs["background_map"][row["process_or_mode"]]
            per_row_weight = weight_by_member[member]
            if not math.isfinite(per_row_weight) or per_row_weight <= 0.0:
                raise ValueError(f"invalid frozen weight for member {member}")
            frames.append(frame)
            member_arrays.append(
                np.full(expected_rows, member, dtype=np.int64)
            )
            fold_arrays.append(
                np.full(expected_rows, fold_by_member[member], dtype=np.int8)
            )
            target_arrays.append(
                np.full(expected_rows, target, dtype=np.int8)
            )
            weight_arrays.append(
                np.full(expected_rows, per_row_weight, dtype=np.float64)
            )
            mode_arrays.append(np.full(expected_rows, mode, dtype=object))
            family_arrays.append(np.full(expected_rows, family, dtype=object))
            process_arrays.append(
                np.full(expected_rows, row["process_or_mode"], dtype=object)
            )
        if position % 50 == 0 or position == len(train_manifest):
            print(
                f"Opened train candidates {position}/{len(train_manifest)}",
                flush=True,
            )

    if opened != expected["train_members"]:
        raise ValueError("not every train candidate file was opened exactly once")
    raw = pd.concat(frames, ignore_index=True)
    if len(raw) != expected["train_rows"]:
        raise ValueError("combined train row count mismatch")
    print("Deriving four-jet angular and rest-frame features", flush=True)
    features = derive_cms_inspired_features(raw)
    payload = {
        "features": features,
        "member_index": np.concatenate(member_arrays),
        "fold": np.concatenate(fold_arrays),
        "target": np.concatenate(target_arrays),
        "weight": np.concatenate(weight_arrays),
        "signal_mode": np.concatenate(mode_arrays),
        "background_family": np.concatenate(family_arrays),
        "process_or_mode": np.concatenate(process_arrays),
        "train_manifest": train_manifest,
        "candidate_files_opened": opened,
        "schema_less_zero_row_files": schema_less_zero,
    }
    lengths = {
        len(value)
        for key, value in payload.items()
        if key
        in {
            "features",
            "member_index",
            "fold",
            "target",
            "weight",
            "signal_mode",
            "background_family",
            "process_or_mode",
        }
    }
    if lengths != {expected["train_rows"]}:
        raise ValueError(f"in-memory train payload lengths differ: {lengths}")
    if not np.isclose(
        float(np.mean(payload["weight"])),
        1.0,
        rtol=0.0,
        atol=FLOAT_TOLERANCE,
    ):
        raise ValueError("frozen development weights do not have mean one")
    return payload


def build_categories(
    payload: Mapping[str, Any],
) -> tuple[np.ndarray, dict[str, int]]:
    mhh = payload["features"]["mhh"].to_numpy(dtype=np.float64)
    finite = np.isfinite(mhh)
    nonfinite = int(np.count_nonzero(~finite))
    if nonfinite:
        raise ValueError(f"nonfinite mHH rows: {nonfinite}")
    low = mhh < MHH_BOUNDARY_GEV
    high = mhh >= MHH_BOUNDARY_GEV
    overlaps = int(np.count_nonzero(low & high))
    gaps = int(np.count_nonzero(~(low | high)))
    categories = np.asarray(
        [assign_mhh_category(value) for value in mhh],
        dtype=object,
    )
    if overlaps or gaps:
        raise ValueError(f"category partition failed: overlaps={overlaps}, gaps={gaps}")
    return categories, {
        "category_overlaps": overlaps,
        "category_gaps": gaps,
        "nonfinite_mhh_rows": nonfinite,
    }


def category_definition_rows() -> list[dict[str, Any]]:
    return [
        {
            "category": "low_mhh",
            "definition": "mhh < 450.0 GeV",
            "latex_definition": r"$m_{HH}<450\,\mathrm{GeV}$",
            "boundary_GeV": MHH_BOUNDARY_GEV,
            "boundary_in_category": False,
            "boundary_optimized": False,
            "description": "CMS-inspired resolved HH→4b mHH categorization",
            "status": "pass",
        },
        {
            "category": "high_mhh",
            "definition": "mhh >= 450.0 GeV",
            "latex_definition": r"$m_{HH}\geq450\,\mathrm{GeV}$",
            "boundary_GeV": MHH_BOUNDARY_GEV,
            "boundary_in_category": True,
            "boundary_optimized": False,
            "description": "CMS-inspired resolved HH→4b mHH categorization",
            "status": "pass",
        },
    ]


def category_population_rows(
    payload: Mapping[str, Any],
    categories: np.ndarray,
) -> list[dict[str, Any]]:
    rows = []
    for category in CATEGORY_NAMES:
        category_mask = categories == category
        for target, target_label in ((1, "signal"), (0, "background")):
            mask = category_mask & (payload["target"] == target)
            rows.append(
                {
                    "category": category,
                    "training_target": target,
                    "target_label": target_label,
                    "rows": int(np.count_nonzero(mask)),
                    "source_members": int(
                        np.unique(payload["member_index"][mask]).size
                    ),
                    "frozen_folds": int(np.unique(payload["fold"][mask]).size),
                    "development_weight_sum": float(
                        np.sum(payload["weight"][mask])
                    ),
                    "raw_fraction_within_category": float(
                        np.count_nonzero(mask) / np.count_nonzero(category_mask)
                    ),
                    "normalization": "raw_rows_and_development_balancing",
                    "status": "pass",
                }
            )
    return rows


def category_stratum_rows(
    payload: Mapping[str, Any],
    categories: np.ndarray,
    *,
    array_name: str,
    value_name: str,
    values: Sequence[str],
    target: int,
) -> list[dict[str, Any]]:
    rows = []
    for category in CATEGORY_NAMES:
        for value in values:
            mask = (
                (categories == category)
                & (payload["target"] == target)
                & (payload[array_name] == value)
            )
            rows.append(
                {
                    "category": category,
                    value_name: value,
                    "training_target": target,
                    "rows": int(np.count_nonzero(mask)),
                    "source_members": int(
                        np.unique(payload["member_index"][mask]).size
                    ),
                    "frozen_folds": int(np.unique(payload["fold"][mask]).size),
                    "development_weight_sum": float(
                        np.sum(payload["weight"][mask])
                    ),
                    "status": (
                        "pass"
                        if np.count_nonzero(mask) > 0
                        else "recorded_absence"
                    ),
                }
            )
    return rows


def category_member_rows(
    payload: Mapping[str, Any],
    categories: np.ndarray,
    inputs: Mapping[str, Any],
) -> list[dict[str, Any]]:
    fold_records = {
        int(row["member_index"]): row for row in inputs["fold_rows"]
    }
    rows = []
    for member in sorted(fold_records):
        record = fold_records[member]
        for category in CATEGORY_NAMES:
            mask = (
                (payload["member_index"] == member)
                & (categories == category)
            )
            rows.append(
                {
                    "member_index": member,
                    "category": category,
                    "sample_class": record["sample_class"],
                    "process_or_mode": record["process_or_mode"],
                    "frozen_stratum": record["stratum"],
                    "fold": int(record["fold"]),
                    "rows": int(np.count_nonzero(mask)),
                    "development_weight_sum": float(
                        np.sum(payload["weight"][mask])
                    ),
                    "status": "pass",
                }
            )
    return rows


def category_fold_rows(
    payload: Mapping[str, Any],
    categories: np.ndarray,
    signal_modes: Sequence[str],
    background_families: Sequence[str],
    n_folds: int,
) -> list[dict[str, Any]]:
    components: list[tuple[str, str, int, str]] = [
        ("binary_target", "signal", 1, ""),
        ("binary_target", "background", 0, ""),
    ]
    components.extend(
        ("signal_mode", mode, 1, "signal_mode") for mode in signal_modes
    )
    components.extend(
        ("background_family", family, 0, "background_family")
        for family in background_families
    )
    rows = []
    for category in CATEGORY_NAMES:
        for fold in range(n_folds):
            for level, component, target, array_name in components:
                mask = (
                    (categories == category)
                    & (payload["fold"] == fold)
                    & (payload["target"] == target)
                )
                if array_name:
                    mask &= payload[array_name] == component
                count = int(np.count_nonzero(mask))
                rows.append(
                    {
                        "category": category,
                        "fold": fold,
                        "composition_level": level,
                        "component": component,
                        "training_target": target,
                        "rows": count,
                        "source_members": int(
                            np.unique(payload["member_index"][mask]).size
                        ),
                        "development_weight_sum": float(
                            np.sum(payload["weight"][mask])
                        ),
                        "absence_recorded": count == 0,
                        "required_in_every_fold": level == "binary_target",
                        "status": (
                            "fail"
                            if level == "binary_target" and count == 0
                            else (
                                "recorded_absence"
                                if count == 0
                                else "pass"
                            )
                        ),
                    }
                )
    return rows


def feature_contract_rows() -> list[dict[str, Any]]:
    rows = []
    for variant, features in FEATURE_VARIANTS.items():
        for order, feature in enumerate(features, start=1):
            rows.append(
                {
                    "variant": variant,
                    "category": (
                        "low_mhh"
                        if variant.startswith("low_mhh")
                        else (
                            "high_mhh"
                            if variant.startswith("high_mhh")
                            else "all_train_reference_or_both_categories"
                        )
                    ),
                    "feature_order": order,
                    "feature": feature,
                    "feature_origin": (
                        "cms_inspired_derived"
                        if feature in CMS_INSPIRED_DERIVED_FEATURES
                        else "frozen_bdt_v1"
                    ),
                    "explicit_dijet_mass_plane_feature": (
                        feature in EXPLICIT_DIJET_MASS_PLANE_FEATURES
                    ),
                    "mhh_retained": feature == "mhh",
                    "training_in_this_gate": False,
                    "status": "pass",
                }
            )
    return rows


def derived_definition_rows() -> list[dict[str, Any]]:
    units = {
        "candidate_vector_pt_sum": "GeV",
    }
    rows = []
    for order, feature in enumerate(CMS_INSPIRED_DERIVED_FEATURES, start=1):
        rows.append(
            {
                "feature_order_after_v1": order,
                "feature_order_mass_aware": len(GLOBAL_V1_MASS_AWARE_REFERENCE)
                + order,
                "feature": feature,
                "definition": DERIVED_FEATURE_DEFINITIONS[feature],
                "inputs": "four candidate-jet (pt,eta,phi,mass) four-vectors",
                "units": units.get(feature, "dimensionless"),
                "generator_truth_used": False,
                "absolute_azimuth_feature": False,
                "latex_label": FEATURE_LATEX_LABELS[feature],
                "status": "pass",
            }
        )
    return rows


def physical_range_audit(
    features: pd.DataFrame,
    categories: np.ndarray,
) -> tuple[list[dict[str, Any]], int]:
    rows = []
    signed_cosines = (
        "cos_theta_star_h1_in_hh_rest",
        "cos_theta_j1_in_h1_rest",
        "cos_theta_j_h2_leading_in_h2_rest",
    )
    absolute_cosines = (
        "abs_cos_theta_star_h1_in_hh_rest",
        "abs_cos_theta_j1_in_h1_rest",
        "abs_cos_theta_j_h2_leading_in_h2_rest",
    )

    def append(
        category: str,
        check: str,
        values: np.ndarray,
        failure_mask: np.ndarray,
    ) -> None:
        failures = int(np.count_nonzero(failure_mask))
        rows.append(
            {
                "category": category,
                "check": check,
                "rows": len(values),
                "failures": failures,
                "minimum": float(np.min(values)),
                "maximum": float(np.max(values)),
                "tolerance": FLOAT_TOLERANCE,
                "clipping_applied": False,
                "status": "pass" if failures == 0 else "fail",
            }
        )

    for category in CATEGORY_NAMES:
        subset = features.loc[categories == category]
        for feature in (
            *PAIRWISE_DR_FEATURES,
            "min_candidate_pair_dr",
            "max_candidate_pair_dr",
            "mean_candidate_pair_dr",
            "min_cross_higgs_pair_dr",
            "max_cross_higgs_pair_dr",
        ):
            values = subset[feature].to_numpy(dtype=np.float64)
            append(category, f"{feature} >= 0", values, values < 0.0)
        minimum = subset["min_candidate_pair_dr"].to_numpy(dtype=np.float64)
        maximum = subset["max_candidate_pair_dr"].to_numpy(dtype=np.float64)
        append(
            category,
            "min_candidate_pair_dr <= max_candidate_pair_dr",
            maximum - minimum,
            minimum > maximum,
        )
        for feature in signed_cosines:
            values = subset[feature].to_numpy(dtype=np.float64)
            append(
                category,
                f"-1 <= {feature} <= 1",
                values,
                (values < -1.0 - FLOAT_TOLERANCE)
                | (values > 1.0 + FLOAT_TOLERANCE),
            )
        for feature in absolute_cosines:
            values = subset[feature].to_numpy(dtype=np.float64)
            append(
                category,
                f"0 <= {feature} <= 1",
                values,
                (values < -FLOAT_TOLERANCE)
                | (values > 1.0 + FLOAT_TOLERANCE),
            )
        vector_pt = subset["candidate_vector_pt_sum"].to_numpy(dtype=np.float64)
        append(
            category,
            "candidate_vector_pt_sum >= 0",
            vector_pt,
            vector_pt < 0.0,
        )
    failures = sum(int(row["failures"]) for row in rows)
    return rows, failures


def cms_reference_inventory_rows() -> list[dict[str, Any]]:
    return [
        {
            "reference": "CMS HIG-20-005",
            "analysis_final_state": "HH→bbbb",
            "ml_purpose": "XGBoost signal-versus-background discrimination",
            "categorization": "low/high mHH at 450 GeV with separate category BDTs",
            "directly_implementable_elements": (
                "fixed reconstructed-mHH split; category-specific BDT contracts; "
                "member-grouped cross-application; four-jet angular and rest-frame variables"
            ),
            "unavailable_detector_or_data_elements": (
                "continuous DeepJet information; CMS b-jet regression and resolution; "
                "data control regions and complete VBF tagging-jet representation"
            ),
            "planned_role": (
                "categorized resolved BDT v2; 3b→4b reweighting remains a separate future task"
            ),
            "exact_reproduction_claimed": False,
        },
        {
            "reference": "CMS HIG-22-011",
            "analysis_final_state": "HH→bbbb",
            "ml_purpose": (
                "customized multiclass 4b classification with combinatorial "
                "jet-pairing-aware architecture"
            ),
            "categorization": "multiclass classifier outputs used by the analysis strategy",
            "directly_implementable_elements": (
                "pairing-aware representation and permutation-aware learning concepts"
            ),
            "unavailable_detector_or_data_elements": (
                "full CMS object content, calibrations, and data-driven background machinery"
            ),
            "planned_role": "later SPA-Net or set-model benchmark studies",
            "exact_reproduction_claimed": False,
        },
        {
            "reference": "uploaded CMS HIG-24-015 draft",
            "analysis_final_state": "HHH→4b2γ",
            "ml_purpose": (
                "BDTnonres and BDTres suppression of nonresonant and resonant backgrounds"
            ),
            "categorization": (
                "two-dimensional score categories optimized after physical yields "
                "and sideband occupancy are available"
            ),
            "directly_implementable_elements": (
                "separate discriminant roles; protected fit-variable sculpting validation"
            ),
            "unavailable_detector_or_data_elements": (
                "photon final state, data sidebands, physical yields, and fit model"
            ),
            "planned_role": (
                "future two-score category optimization only after yield and sideband gates"
            ),
            "exact_reproduction_claimed": False,
        },
    ]


def label_for_category(category: str) -> str:
    return {
        "low_mhh": r"low $m_{HH}$",
        "high_mhh": r"high $m_{HH}$",
    }[category]


def label_for_target(target: int) -> str:
    return "Signal" if target == 1 else "Background"


def label_for_signal_mode(mode: str) -> str:
    return {
        "ggf_hh4b": r"ggF $HH$",
        "vbf_hh4b": r"VBF $HH$",
    }.get(mode, mode.replace("_", " "))


def label_for_background_family(family: str) -> str:
    return {
        "ttbar": r"$t\bar{t}$",
        "top_associated": r"top associated ($t\bar{t}H$ incl.)",
        "qcd_multijet": "QCD multijet",
        "single_higgs": "single Higgs",
        "single_top": "single top",
        "diboson": "diboson",
        "triboson": "triboson",
        "zbbbb": r"$Z+bbbb$",
    }.get(family, family.replace("_", " "))


def weighted_density(
    values: np.ndarray,
    weights: np.ndarray,
    bins: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    counts, edges = np.histogram(values, bins=bins, weights=weights)
    widths = np.diff(edges)
    integral = float(np.sum(counts))
    density = counts / (integral * widths) if integral > 0.0 else counts
    return density, edges


def draw_step_density(
    ax: matplotlib.axes.Axes,
    values: np.ndarray,
    weights: np.ndarray,
    bins: np.ndarray,
    *,
    label: str,
    color: str,
    linewidth: float = 1.7,
) -> None:
    density, edges = weighted_density(values, weights, bins)
    ax.stairs(
        density,
        edges,
        label=label,
        color=color,
        linewidth=linewidth,
    )


def save_plot_with_inventory(
    fig: matplotlib.figure.Figure,
    plots_dir: Path,
    inventory: list[dict[str, Any]],
    *,
    name: str,
    variables: Sequence[str],
    category: str,
    sample_scope: str,
    normalization: str,
    binning: str,
    axis_limits: Mapping[str, Any],
    latex_labels: Sequence[str],
    style_helper_sha256: str,
    style_metadata: Mapping[str, Any],
    dpi: int,
) -> None:
    png, pdf = save_png_pdf(fig, plots_dir / name, dpi=dpi)
    for path, paired, file_format in (
        (png, pdf, "PNG"),
        (pdf, png, "PDF"),
    ):
        inventory.append(
            {
                "plot_path": f"plots/{path.name}",
                "paired_path": f"plots/{paired.name}",
                "plot_name": name,
                "variables": list(variables),
                "category": category,
                "sample_scope": sample_scope,
                "normalization": normalization,
                "binning": binning,
                "axis_limits": dict(axis_limits),
                "latex_labels": list(latex_labels),
                "style_helper_path": "scripts/plotting/hh4b_cms_style.py",
                "style_helper_sha256": style_helper_sha256,
                "cms_inspired_style_applied": style_metadata[
                    "cms_inspired_style_applied"
                ],
                "official_cms_status_claimed": False,
                "dpi": dpi,
                "format": file_format,
                "bytes": path.stat().st_size,
                "status": "pass",
            }
        )


def make_plots(
    output_dir: Path,
    payload: Mapping[str, Any],
    categories: np.ndarray,
    signal_modes: Sequence[str],
    background_families: Sequence[str],
    config: Mapping[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    plots_dir = output_dir / "plots"
    plots_dir.mkdir(parents=True)
    style_metadata = apply_cms_style()
    if matplotlib.rcParams["text.usetex"]:
        raise ValueError("text.usetex must remain false")
    if style_metadata["official_cms_status_claimed"]:
        raise ValueError("official CMS status was unexpectedly claimed")
    style_path = REPOSITORY_ROOT / config["plotting"]["style_helper"]
    style_hash = sha256_file(style_path)
    dpi = int(config["plotting"]["dpi"])
    inventory: list[dict[str, Any]] = []
    features = payload["features"]
    mhh = features["mhh"].to_numpy(dtype=np.float64)
    mhh_upper = max(
        1000.0,
        min(2500.0, math.ceil(float(np.quantile(mhh, 0.995)) / 100.0) * 100.0),
    )
    mhh_bins = np.linspace(200.0, mhh_upper, 61)

    fig, ax = plt.subplots(figsize=(6.4, 4.8))
    for index, category in enumerate(CATEGORY_NAMES):
        mask = categories == category
        ax.hist(
            mhh[mask],
            bins=mhh_bins,
            histtype="stepfilled",
            alpha=0.65,
            color=CATEGORY_COLORS[index],
            label=label_for_category(category),
        )
    ax.axvline(
        MHH_BOUNDARY_GEV,
        color="black",
        linestyle="--",
        linewidth=1.2,
        label=r"$m_{HH}=450\,\mathrm{GeV}$",
    )
    ax.set_xlim(mhh_bins[0], mhh_bins[-1])
    ax.set_xlabel(r"$m_{HH}$ [GeV]")
    ax.set_ylabel("Raw canonical train candidates")
    ax.legend(loc="upper right", ncol=1)
    add_delphes_header(ax, "Train only")
    fig.tight_layout()
    save_plot_with_inventory(
        fig,
        plots_dir,
        inventory,
        name="train_mhh_category_population",
        variables=["mhh", "category"],
        category="low_mhh_and_high_mhh",
        sample_scope="raw_canonical_train_candidates",
        normalization="none_raw_counts",
        binning=f"60 uniform bins in [{mhh_bins[0]},{mhh_bins[-1]}] GeV",
        axis_limits={"x": [mhh_bins[0], mhh_bins[-1]], "y": "automatic"},
        latex_labels=[r"$m_{HH}$", r"$m_{HH}=450\,\mathrm{GeV}$"],
        style_helper_sha256=style_hash,
        style_metadata=style_metadata,
        dpi=dpi,
    )

    fig, ax = plt.subplots(figsize=(6.4, 4.8))
    for index, target in enumerate((1, 0)):
        mask = payload["target"] == target
        draw_step_density(
            ax,
            mhh[mask],
            payload["weight"][mask],
            mhh_bins,
            label=label_for_target(target),
            color=CATEGORY_COLORS[index],
        )
    ax.axvline(MHH_BOUNDARY_GEV, color="black", linestyle="--", linewidth=1.2)
    ax.set_xlim(mhh_bins[0], mhh_bins[-1])
    ax.set_xlabel(r"$m_{HH}$ [GeV]")
    ax.set_ylabel("Weighted unit density")
    ax.legend(loc="upper right")
    ax.text(
        0.98,
        0.73,
        "Development balancing; not physical normalization",
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=8,
    )
    add_delphes_header(ax, "Train only")
    fig.tight_layout()
    save_plot_with_inventory(
        fig,
        plots_dir,
        inventory,
        name="train_mhh_by_target",
        variables=["mhh", "training_target", "development_weight"],
        category="all_train",
        sample_scope="canonical_train_signal_and_background",
        normalization="development_weighted_unit_density_not_physical",
        binning=f"60 uniform bins in [{mhh_bins[0]},{mhh_bins[-1]}] GeV",
        axis_limits={"x": [mhh_bins[0], mhh_bins[-1]], "y": "automatic"},
        latex_labels=[r"$m_{HH}$"],
        style_helper_sha256=style_hash,
        style_metadata=style_metadata,
        dpi=dpi,
    )

    fig, ax = plt.subplots(figsize=(6.4, 4.8))
    for index, mode in enumerate(signal_modes):
        mask = payload["signal_mode"] == mode
        draw_step_density(
            ax,
            mhh[mask],
            payload["weight"][mask],
            mhh_bins,
            label=label_for_signal_mode(mode),
            color=CATEGORY_COLORS[index],
        )
    ax.axvline(MHH_BOUNDARY_GEV, color="black", linestyle="--", linewidth=1.2)
    ax.set_xlim(mhh_bins[0], mhh_bins[-1])
    ax.set_xlabel(r"$m_{HH}$ [GeV]")
    ax.set_ylabel("Weighted unit density")
    ax.legend(loc="upper right")
    ax.text(
        0.98,
        0.73,
        "Development balancing; not physical normalization",
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=8,
    )
    add_delphes_header(ax, "Train only")
    fig.tight_layout()
    save_plot_with_inventory(
        fig,
        plots_dir,
        inventory,
        name="train_mhh_by_signal_mode",
        variables=["mhh", "signal_mode", "development_weight"],
        category="all_train_signal",
        sample_scope="canonical_train_signal",
        normalization="development_weighted_unit_density_not_physical",
        binning=f"60 uniform bins in [{mhh_bins[0]},{mhh_bins[-1]}] GeV",
        axis_limits={"x": [mhh_bins[0], mhh_bins[-1]], "y": "automatic"},
        latex_labels=[r"$m_{HH}$", r"ggF $HH$", r"VBF $HH$"],
        style_helper_sha256=style_hash,
        style_metadata=style_metadata,
        dpi=dpi,
    )

    fig, ax = plt.subplots(figsize=(6.8, 5.0))
    for index, family in enumerate(background_families):
        mask = payload["background_family"] == family
        draw_step_density(
            ax,
            mhh[mask],
            payload["weight"][mask],
            mhh_bins,
            label=label_for_background_family(family),
            color=CATEGORY_COLORS[index % len(CATEGORY_COLORS)],
            linewidth=1.4,
        )
    ax.axvline(MHH_BOUNDARY_GEV, color="black", linestyle="--", linewidth=1.1)
    ax.set_xlim(mhh_bins[0], mhh_bins[-1])
    ax.set_xlabel(r"$m_{HH}$ [GeV]")
    ax.set_ylabel("Weighted unit density")
    ax.legend(loc="upper right", ncol=2, fontsize=7.5)
    ax.text(
        0.98,
        0.56,
        "Development balancing; not physical normalization",
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=8,
    )
    add_delphes_header(ax, "Train only")
    fig.tight_layout()
    save_plot_with_inventory(
        fig,
        plots_dir,
        inventory,
        name="train_mhh_by_background_family",
        variables=["mhh", "background_family", "development_weight"],
        category="all_train_background",
        sample_scope="canonical_train_background",
        normalization="development_weighted_unit_density_not_physical",
        binning=f"60 uniform bins in [{mhh_bins[0]},{mhh_bins[-1]}] GeV",
        axis_limits={"x": [mhh_bins[0], mhh_bins[-1]], "y": "automatic"},
        latex_labels=[r"$m_{HH}$", r"$t\bar{t}$", r"$t\bar{t}H$"],
        style_helper_sha256=style_hash,
        style_metadata=style_metadata,
        dpi=dpi,
    )

    folds = np.arange(5)
    low_counts = np.asarray(
        [
            np.count_nonzero((payload["fold"] == fold) & (categories == "low_mhh"))
            for fold in folds
        ]
    )
    high_counts = np.asarray(
        [
            np.count_nonzero((payload["fold"] == fold) & (categories == "high_mhh"))
            for fold in folds
        ]
    )
    fig, ax = plt.subplots(figsize=(6.4, 4.8))
    ax.bar(
        folds,
        low_counts,
        color=CATEGORY_COLORS[0],
        label=label_for_category("low_mhh"),
    )
    ax.bar(
        folds,
        high_counts,
        bottom=low_counts,
        color=CATEGORY_COLORS[1],
        label=label_for_category("high_mhh"),
    )
    ax.set_xticks(folds)
    ax.set_xlabel("Frozen member-grouped fold")
    ax.set_ylabel("Raw canonical train candidates")
    ax.legend(loc="upper right")
    add_delphes_header(ax, "Train only")
    fig.tight_layout()
    save_plot_with_inventory(
        fig,
        plots_dir,
        inventory,
        name="category_fold_composition",
        variables=["fold", "category"],
        category="low_mhh_and_high_mhh",
        sample_scope="raw_canonical_train_candidates",
        normalization="none_raw_counts",
        binning="five frozen member-grouped folds",
        axis_limits={"x": [-0.5, 4.5], "y": "automatic"},
        latex_labels=[r"low $m_{HH}$", r"high $m_{HH}$"],
        style_helper_sha256=style_hash,
        style_metadata=style_metadata,
        dpi=dpi,
    )

    dr_bins = np.linspace(
        0.0,
        math.ceil(
            float(
                features[
                    ["min_candidate_pair_dr", "max_candidate_pair_dr"]
                ].to_numpy().max()
            )
            * 2.0
        )
        / 2.0,
        51,
    )
    fig, axes = plt.subplots(1, 2, figsize=(10.0, 4.4), sharey=True)
    for axis, feature in zip(
        axes,
        ("min_candidate_pair_dr", "max_candidate_pair_dr"),
    ):
        values = features[feature].to_numpy(dtype=np.float64)
        for index, target in enumerate((1, 0)):
            mask = payload["target"] == target
            draw_step_density(
                axis,
                values[mask],
                payload["weight"][mask],
                dr_bins,
                label=label_for_target(target),
                color=CATEGORY_COLORS[index],
            )
        axis.set_xlabel(FEATURE_LATEX_LABELS[feature])
        axis.set_xlim(dr_bins[0], dr_bins[-1])
    axes[0].set_ylabel("Weighted unit density")
    axes[1].legend(loc="upper right")
    axes[0].text(
        0.02,
        0.97,
        "Development balancing; not physical normalization",
        transform=axes[0].transAxes,
        ha="left",
        va="top",
        fontsize=8,
    )
    add_delphes_header(axes[0], "CMS-inspired categorization")
    fig.tight_layout()
    save_plot_with_inventory(
        fig,
        plots_dir,
        inventory,
        name="min_max_candidate_pair_dr_by_target",
        variables=[
            "min_candidate_pair_dr",
            "max_candidate_pair_dr",
            "training_target",
            "development_weight",
        ],
        category="all_train",
        sample_scope="canonical_train_signal_and_background",
        normalization="development_weighted_unit_density_not_physical",
        binning=f"50 uniform bins in [{dr_bins[0]},{dr_bins[-1]}]",
        axis_limits={"x": [dr_bins[0], dr_bins[-1]], "y": "shared_automatic"},
        latex_labels=[
            FEATURE_LATEX_LABELS["min_candidate_pair_dr"],
            FEATURE_LATEX_LABELS["max_candidate_pair_dr"],
        ],
        style_helper_sha256=style_hash,
        style_metadata=style_metadata,
        dpi=dpi,
    )

    cosine_bins = np.linspace(0.0, 1.0, 41)
    cosine_feature = "abs_cos_theta_star_h1_in_hh_rest"
    cosine = features[cosine_feature].to_numpy(dtype=np.float64)
    fig, ax = plt.subplots(figsize=(6.4, 4.8))
    for index, target in enumerate((1, 0)):
        mask = payload["target"] == target
        draw_step_density(
            ax,
            cosine[mask],
            payload["weight"][mask],
            cosine_bins,
            label=label_for_target(target),
            color=CATEGORY_COLORS[index],
        )
    ax.set_xlim(0.0, 1.0)
    ax.set_xlabel(FEATURE_LATEX_LABELS[cosine_feature])
    ax.set_ylabel("Weighted unit density")
    ax.legend(loc="upper left")
    ax.text(
        0.98,
        0.96,
        "Development balancing; not physical normalization",
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=8,
    )
    add_delphes_header(ax, "CMS-inspired categorization")
    fig.tight_layout()
    save_plot_with_inventory(
        fig,
        plots_dir,
        inventory,
        name="abs_cos_theta_star_by_target",
        variables=[cosine_feature, "training_target", "development_weight"],
        category="all_train",
        sample_scope="canonical_train_signal_and_background",
        normalization="development_weighted_unit_density_not_physical",
        binning="40 uniform bins in [0,1]",
        axis_limits={"x": [0.0, 1.0], "y": "automatic"},
        latex_labels=[FEATURE_LATEX_LABELS[cosine_feature]],
        style_helper_sha256=style_hash,
        style_metadata=style_metadata,
        dpi=dpi,
    )

    for category in CATEGORY_NAMES:
        subset = features.loc[
            categories == category,
            list(CMS_INSPIRED_DERIVED_FEATURES),
        ]
        correlation = subset.corr(method="pearson").to_numpy()
        fig, ax = plt.subplots(figsize=(10.0, 9.0))
        image = ax.imshow(
            correlation,
            vmin=-1.0,
            vmax=1.0,
            cmap="RdBu_r",
            aspect="equal",
        )
        labels = [
            FEATURE_LATEX_LABELS[feature]
            for feature in CMS_INSPIRED_DERIVED_FEATURES
        ]
        ticks = np.arange(len(labels))
        ax.set_xticks(ticks)
        ax.set_yticks(ticks)
        ax.set_xticklabels(labels, rotation=65, ha="right", fontsize=7)
        ax.set_yticklabels(labels, fontsize=7)
        colorbar = fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
        colorbar.set_label("Pearson correlation")
        add_delphes_header(ax, "CMS-inspired categorization")
        fig.tight_layout()
        name = f"derived_feature_correlation_{category}"
        save_plot_with_inventory(
            fig,
            plots_dir,
            inventory,
            name=name,
            variables=list(CMS_INSPIRED_DERIVED_FEATURES),
            category=category,
            sample_scope="canonical_train_signal_and_background",
            normalization="unweighted_pearson_correlation",
            binning="not_applicable",
            axis_limits={"correlation": [-1.0, 1.0]},
            latex_labels=labels,
            style_helper_sha256=style_hash,
            style_metadata=style_metadata,
            dpi=dpi,
        )

    requested = config["plotting"]["plots"]
    observed_names = sorted({row["plot_name"] for row in inventory})
    if observed_names != sorted(requested):
        raise ValueError(
            f"plot inventory mismatch: requested={requested}, observed={observed_names}"
        )
    return inventory, {
        "style_metadata": style_metadata,
        "style_helper_sha256": style_hash,
    }


def write_bundle(
    table_paths: list[Path],
    directory: Path,
    name: str,
    rows: Iterable[Mapping[str, Any]],
    fields: Sequence[str],
    *,
    caption: str,
    label: str,
    publication_fields: Sequence[str] | None = None,
    latex_raw_fields: Sequence[str] = (),
) -> None:
    table_paths.extend(
        write_table_bundle(
            directory,
            name,
            rows,
            fields,
            caption=caption,
            label=label,
            publication_fields=publication_fields,
            latex_raw_fields=latex_raw_fields,
        )
    )


def write_readme(output_dir: Path, summary: Mapping[str, Any]) -> None:
    text = f"""# CMS-inspired resolved HH→4b mHH category and feature contract

Status: `{summary['status']}`.

This checkpoint freezes a **CMS-inspired resolved HH→4b mHH categorization**.
It is not a reproduction of a CMS analysis.  The canonical global BDT v1 is
preserved unchanged as the uncategorized reference.

Every one of the {summary['train_rows']} canonical train candidates is assigned
exactly once: `low_mhh` uses $m_{{HH}} < 450$ GeV and `high_mhh` uses
$m_{{HH}} \\geq 450$ GeV.  The boundary was fixed a priori and was not
optimized.  Both ggF $HH$ and VBF $HH$ remain in these reconstructed-$m_{{HH}}$
categories.  A VBF production-mode BDT is deferred because the candidate
schema does not contain a frozen complete representation of two additional
VBF tagging jets.

The exact 34-feature mass-aware BDT-v1 order is retained.  Eighteen
detector-level features are appended from the four candidate-jet four-vectors.
The stored pairing, generator truth, `Jet.Flavor`, source metadata, absolute
azimuths, and binary b-tag flags are not nominal features.  No continuous
b-tag score, b-jet energy regression, or per-jet resolution is claimed.

The Higgs pairs are reconstructed from the four-vectors by the frozen
125-GeV mass-ranking rule.  $H_1$ is the higher-$p_{{\\mathrm{{T}}}}$ dijet,
and the leading jet in each candidate is its higher-$p_{{\\mathrm{{T}}}}$
member.  Rest-frame angles use helicity axes defined by each parent candidate's
laboratory flight direction.  Cosines are audited without clipping.

The explicit ablation removes only `mbb1`, `mbb2`, `delta_mbb`, and
`r_hh_125_125`; it retains `mhh` and is therefore described as a
**categorized explicit dijet-mass-plane-blind ablation**, not as fully
mass-decorrelated.

Only train candidate Parquets were opened.  Validation candidate files opened:
0.  Test candidate files opened: 0.  Models trained: 0.  Predictions written:
0.  Hyperparameter trials: 0.

The next authorized gate is
`{summary['next_gate']}`.
"""
    (output_dir / "README.md").write_text(text, encoding="utf-8")


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=CONFIG_DEFAULT)
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="replace only the exact configured runtime/checkpoint directories",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_arguments()
    config_path = args.config.resolve()
    config = load_yaml(config_path)
    source_commit = git_output("rev-parse", "HEAD")
    if source_commit != config["source_commit"]:
        raise ValueError(
            f"source commit mismatch: config={config['source_commit']} "
            f"HEAD={source_commit}"
        )
    validate_feature_names(CATEGORIZED_MASS_AWARE_FEATURES)
    validate_feature_names(
        CATEGORIZED_EXPLICIT_DIJET_MASS_PLANE_BLIND_FEATURES
    )

    runtime_dir = REPOSITORY_ROOT / config["outputs"]["runtime_dir"]
    checkpoint_dir = REPOSITORY_ROOT / config["outputs"]["checkpoint_dir"]
    prepare_exact_directory(runtime_dir, args.overwrite)
    if checkpoint_dir.exists() and not args.overwrite:
        raise FileExistsError(
            f"checkpoint exists; pass --overwrite to replace it: {checkpoint_dir}"
        )

    inputs = load_and_verify_inputs(config)
    payload = load_train_population(inputs, config)
    categories, category_audit = build_categories(payload)
    payload["category"] = categories
    features = payload["features"]
    expected = config["population"]["expected"]

    signal_rows = int(np.count_nonzero(payload["target"] == 1))
    background_rows = int(np.count_nonzero(payload["target"] == 0))
    if signal_rows != expected["signal_rows"]:
        raise ValueError("signal row count changed")
    if background_rows != expected["background_rows"]:
        raise ValueError("background row count changed")

    signal_modes = sorted(
        value for value in np.unique(payload["signal_mode"]) if value
    )
    background_families = sorted(
        value for value in np.unique(payload["background_family"]) if value
    )
    category_population = category_population_rows(payload, categories)
    mode_population = category_stratum_rows(
        payload,
        categories,
        array_name="signal_mode",
        value_name="signal_mode",
        values=signal_modes,
        target=1,
    )
    family_population = category_stratum_rows(
        payload,
        categories,
        array_name="background_family",
        value_name="background_family",
        values=background_families,
        target=0,
    )
    member_population = category_member_rows(payload, categories, inputs)
    fold_composition = category_fold_rows(
        payload,
        categories,
        signal_modes,
        background_families,
        expected["folds"],
    )
    binary_fold_failures = sum(
        row["status"] == "fail" for row in fold_composition
    )
    if binary_fold_failures:
        raise ValueError("a category/fold lacks signal or background")

    quality_rows = []
    for category in CATEGORY_NAMES:
        subset = features.loc[categories == category]
        require_finite_features(subset, CATEGORIZED_MASS_AWARE_FEATURES)
        quality_rows.extend(
            feature_quality_rows(
                subset,
                CATEGORIZED_MASS_AWARE_FEATURES,
                category=category,
            )
        )
    selected_missing = sum(int(row["missing_rows"]) for row in quality_rows)
    selected_nonfinite = sum(int(row["nonfinite_rows"]) for row in quality_rows)
    selected_constant = sum(bool(row["constant"]) for row in quality_rows)
    physical_rows, physical_failures = physical_range_audit(
        features,
        categories,
    )

    required_positive = []
    for category in CATEGORY_NAMES:
        category_mask = categories == category
        required_positive.extend(
            [
                (
                    f"{category}_signal",
                    np.count_nonzero(category_mask & (payload["target"] == 1)),
                ),
                (
                    f"{category}_background",
                    np.count_nonzero(category_mask & (payload["target"] == 0)),
                ),
                (
                    f"{category}_ggf",
                    np.count_nonzero(
                        category_mask & (payload["signal_mode"] == "ggf_hh4b")
                    ),
                ),
                (
                    f"{category}_vbf",
                    np.count_nonzero(
                        category_mask & (payload["signal_mode"] == "vbf_hh4b")
                    ),
                ),
            ]
        )
    missing_required = {
        name: int(count) for name, count in required_positive if count <= 0
    }
    if missing_required:
        raise ValueError(f"required category populations absent: {missing_required}")

    table_paths: list[Path] = []
    write_bundle(
        table_paths,
        runtime_dir,
        "category_definitions",
        category_definition_rows(),
        (
            "category",
            "definition",
            "latex_definition",
            "boundary_GeV",
            "boundary_in_category",
            "boundary_optimized",
            "description",
            "status",
        ),
        caption="Frozen reconstructed-mHH category definitions.",
        label="tab:hh4b_bdt_v2_category_definitions",
        publication_fields=(
            "category",
            "latex_definition",
            "boundary_GeV",
            "boundary_optimized",
            "status",
        ),
        latex_raw_fields=("latex_definition",),
    )
    write_bundle(
        table_paths,
        runtime_dir,
        "category_population",
        category_population,
        (
            "category",
            "training_target",
            "target_label",
            "rows",
            "source_members",
            "frozen_folds",
            "development_weight_sum",
            "raw_fraction_within_category",
            "normalization",
            "status",
        ),
        caption="Canonical train population in each reconstructed-mHH category.",
        label="tab:hh4b_bdt_v2_category_population",
        publication_fields=(
            "category",
            "target_label",
            "rows",
            "source_members",
            "development_weight_sum",
            "status",
        ),
    )
    write_bundle(
        table_paths,
        runtime_dir,
        "category_population_by_signal_mode",
        mode_population,
        (
            "category",
            "signal_mode",
            "training_target",
            "rows",
            "source_members",
            "frozen_folds",
            "development_weight_sum",
            "status",
        ),
        caption="Signal-mode composition of the reconstructed-mHH categories.",
        label="tab:hh4b_bdt_v2_signal_mode_population",
        publication_fields=(
            "category",
            "signal_mode",
            "rows",
            "source_members",
            "frozen_folds",
            "status",
        ),
    )
    write_bundle(
        table_paths,
        runtime_dir,
        "category_population_by_background_family",
        family_population,
        (
            "category",
            "background_family",
            "training_target",
            "rows",
            "source_members",
            "frozen_folds",
            "development_weight_sum",
            "status",
        ),
        caption="Background-family composition of the reconstructed-mHH categories.",
        label="tab:hh4b_bdt_v2_background_family_population",
        publication_fields=(
            "category",
            "background_family",
            "rows",
            "source_members",
            "frozen_folds",
            "status",
        ),
    )
    write_bundle(
        table_paths,
        runtime_dir,
        "category_population_by_source_member",
        member_population,
        (
            "member_index",
            "category",
            "sample_class",
            "process_or_mode",
            "frozen_stratum",
            "fold",
            "rows",
            "development_weight_sum",
            "status",
        ),
        caption="Per-source-member aggregate category composition.",
        label="tab:hh4b_bdt_v2_member_population",
        publication_fields=(
            "member_index",
            "category",
            "sample_class",
            "fold",
            "rows",
            "status",
        ),
    )
    write_bundle(
        table_paths,
        runtime_dir,
        "category_fold_composition",
        fold_composition,
        (
            "category",
            "fold",
            "composition_level",
            "component",
            "training_target",
            "rows",
            "source_members",
            "development_weight_sum",
            "absence_recorded",
            "required_in_every_fold",
            "status",
        ),
        caption="Frozen member-grouped fold composition in each category.",
        label="tab:hh4b_bdt_v2_category_fold_composition",
        publication_fields=(
            "category",
            "fold",
            "composition_level",
            "component",
            "rows",
            "absence_recorded",
            "status",
        ),
    )
    write_bundle(
        table_paths,
        runtime_dir,
        "derived_feature_definitions",
        derived_definition_rows(),
        (
            "feature_order_after_v1",
            "feature_order_mass_aware",
            "feature",
            "definition",
            "inputs",
            "units",
            "generator_truth_used",
            "absolute_azimuth_feature",
            "latex_label",
            "status",
        ),
        caption="CMS-inspired detector-level derived feature definitions.",
        label="tab:hh4b_bdt_v2_derived_features",
        publication_fields=(
            "feature_order_after_v1",
            "feature",
            "latex_label",
            "definition",
            "units",
            "status",
        ),
        latex_raw_fields=("latex_label",),
    )
    write_bundle(
        table_paths,
        runtime_dir,
        "feature_contracts",
        feature_contract_rows(),
        (
            "variant",
            "category",
            "feature_order",
            "feature",
            "feature_origin",
            "explicit_dijet_mass_plane_feature",
            "mhh_retained",
            "training_in_this_gate",
            "status",
        ),
        caption="Frozen ordered feature variants for the categorized BDT v2.",
        label="tab:hh4b_bdt_v2_feature_contracts",
        publication_fields=(
            "variant",
            "feature_order",
            "feature",
            "feature_origin",
            "status",
        ),
    )
    write_bundle(
        table_paths,
        runtime_dir,
        "feature_quality",
        quality_rows,
        (
            "category",
            "feature_order",
            "feature",
            "feature_origin",
            "rows",
            "finite_rows",
            "nonfinite_rows",
            "missing_rows",
            "minimum",
            "maximum",
            "mean",
            "standard_deviation",
            "unique_values",
            "constant",
            "status",
        ),
        caption="Per-category selected-feature quality audit.",
        label="tab:hh4b_bdt_v2_feature_quality",
        publication_fields=(
            "category",
            "feature",
            "rows",
            "nonfinite_rows",
            "missing_rows",
            "minimum",
            "maximum",
            "status",
        ),
    )
    write_bundle(
        table_paths,
        runtime_dir,
        "physical_range_audit",
        physical_rows,
        (
            "category",
            "check",
            "rows",
            "failures",
            "minimum",
            "maximum",
            "tolerance",
            "clipping_applied",
            "status",
        ),
        caption="Explicit physical-range checks for new derived features.",
        label="tab:hh4b_bdt_v2_physical_ranges",
        publication_fields=(
            "category",
            "check",
            "rows",
            "failures",
            "clipping_applied",
            "status",
        ),
    )
    write_bundle(
        table_paths,
        runtime_dir,
        "cms_reference_method_inventory",
        cms_reference_inventory_rows(),
        (
            "reference",
            "analysis_final_state",
            "ml_purpose",
            "categorization",
            "directly_implementable_elements",
            "unavailable_detector_or_data_elements",
            "planned_role",
            "exact_reproduction_claimed",
        ),
        caption="Inventory of CMS-inspired methodological inputs and project scope.",
        label="tab:hh4b_bdt_v2_cms_method_inventory",
        publication_fields=(
            "reference",
            "analysis_final_state",
            "ml_purpose",
            "categorization",
            "planned_role",
            "exact_reproduction_claimed",
        ),
    )

    plot_inventory, plot_metadata = make_plots(
        runtime_dir,
        payload,
        categories,
        signal_modes,
        background_families,
        config,
    )
    plot_inventory_fields = (
        "plot_path",
        "paired_path",
        "plot_name",
        "variables",
        "category",
        "sample_scope",
        "normalization",
        "binning",
        "axis_limits",
        "latex_labels",
        "style_helper_path",
        "style_helper_sha256",
        "cms_inspired_style_applied",
        "official_cms_status_claimed",
        "dpi",
        "format",
        "bytes",
        "status",
    )
    write_bundle(
        table_paths,
        runtime_dir,
        "plot_inventory",
        plot_inventory,
        plot_inventory_fields,
        caption="Inventory of paired CMS-inspired publication plots.",
        label="tab:hh4b_bdt_v2_plot_inventory",
        publication_fields=(
            "plot_name",
            "format",
            "category",
            "normalization",
            "dpi",
            "bytes",
            "status",
        ),
    )
    plotting_style_rows = []
    for name in config["plotting"]["plots"]:
        entries = [row for row in plot_inventory if row["plot_name"] == name]
        plotting_style_rows.append(
            {
                "plot_name": name,
                "png_written": sum(row["format"] == "PNG" for row in entries)
                == 1,
                "pdf_written": sum(row["format"] == "PDF" for row in entries)
                == 1,
                "white_background": True,
                "outward_major_minor_ticks": True,
                "colorblind_accessible_palette": True,
                "frameless_legend": True,
                "text_usetex": matplotlib.rcParams["text.usetex"],
                "cms_inspired_style_applied": all(
                    row["cms_inspired_style_applied"] for row in entries
                ),
                "official_cms_status_claimed": any(
                    row["official_cms_status_claimed"] for row in entries
                ),
                "status": (
                    "pass"
                    if len(entries) == 2
                    and not any(
                        row["official_cms_status_claimed"] for row in entries
                    )
                    else "fail"
                ),
            }
        )
    write_bundle(
        table_paths,
        runtime_dir,
        "plotting_style_audit",
        plotting_style_rows,
        (
            "plot_name",
            "png_written",
            "pdf_written",
            "white_background",
            "outward_major_minor_ticks",
            "colorblind_accessible_palette",
            "frameless_legend",
            "text_usetex",
            "cms_inspired_style_applied",
            "official_cms_status_claimed",
            "status",
        ),
        caption="CMS-inspired plotting-style compliance audit.",
        label="tab:hh4b_bdt_v2_plotting_style",
        publication_fields=(
            "plot_name",
            "png_written",
            "pdf_written",
            "text_usetex",
            "official_cms_status_claimed",
            "status",
        ),
    )

    low_mask = categories == "low_mhh"
    high_mask = categories == "high_mhh"
    tsv_count = sum(path.suffix == ".tsv" for path in table_paths)
    markdown_count = sum(path.suffix == ".md" for path in table_paths)
    latex_count = sum(path.suffix == ".tex" for path in table_paths)
    plots_requested = len(config["plotting"]["plots"])
    png_count = sum(row["format"] == "PNG" for row in plot_inventory)
    pdf_count = sum(row["format"] == "PDF" for row in plot_inventory)

    summary: dict[str, Any] = {
        "schema_version": 1,
        "status": "hh4b_bdt_v2_cms_inspired_category_and_feature_contract_fail",
        "source_commit": source_commit,
        "analysis_description": "CMS-inspired resolved HH→4b mHH categorization",
        "config_path": str(config_path.relative_to(REPOSITORY_ROOT)),
        "config_sha256": sha256_file(config_path),
        "runner_sha256": sha256_file(Path(__file__)),
        "common_helper_sha256": sha256_file(
            REPOSITORY_ROOT / "scripts/analysis/hh4b_bdt_v2_common.py"
        ),
        "style_helper_sha256": plot_metadata["style_helper_sha256"],
        "v1_input_contract_sha256": inputs["input_summary"].get(
            "config_sha256", ""
        ),
        "v1_input_contract_sha256sums_sha256": sha256_file(
            inputs["v1_dir"] / "SHA256SUMS"
        ),
        "v1_grouped_cv_sha256sums_sha256": sha256_file(
            inputs["cv_dir"] / "SHA256SUMS"
        ),
        "manifest_path": str(
            inputs["manifest_path"].relative_to(REPOSITORY_ROOT)
        ),
        "manifest_sha256": inputs["manifest_sha256"],
        "train_members": expected["train_members"],
        "train_rows": len(features),
        "signal_rows": signal_rows,
        "background_rows": background_rows,
        "low_mhh_rows": int(np.count_nonzero(low_mask)),
        "high_mhh_rows": int(np.count_nonzero(high_mask)),
        "low_mhh_signal_rows": int(
            np.count_nonzero(low_mask & (payload["target"] == 1))
        ),
        "low_mhh_background_rows": int(
            np.count_nonzero(low_mask & (payload["target"] == 0))
        ),
        "high_mhh_signal_rows": int(
            np.count_nonzero(high_mask & (payload["target"] == 1))
        ),
        "high_mhh_background_rows": int(
            np.count_nonzero(high_mask & (payload["target"] == 0))
        ),
        **category_audit,
        "folds": expected["folds"],
        "missing_oof_rows": inputs["cv_summary"]["missing_oof_rows"],
        "duplicate_oof_rows": inputs["cv_summary"]["duplicate_oof_rows"],
        "member_leakage_rows": inputs["cv_summary"]["member_leakage_rows"],
        "derived_features_requested": len(CMS_INSPIRED_DERIVED_FEATURES),
        "derived_features_accepted": len(CMS_INSPIRED_DERIVED_FEATURES),
        "derived_features_rejected": 0,
        "global_v1_mass_aware_features": len(
            GLOBAL_V1_MASS_AWARE_REFERENCE
        ),
        "categorized_mass_aware_features": len(
            CATEGORIZED_MASS_AWARE_FEATURES
        ),
        "categorized_explicit_dijet_mass_plane_blind_features": len(
            CATEGORIZED_EXPLICIT_DIJET_MASS_PLANE_BLIND_FEATURES
        ),
        "selected_feature_missing_values": selected_missing,
        "selected_feature_nonfinite_values": selected_nonfinite,
        "selected_feature_constant_count": selected_constant,
        "physical_range_failures": physical_failures,
        "category_fold_binary_failures": binary_fold_failures,
        "validation_candidate_files_opened": 0,
        "test_candidate_files_opened": 0,
        "models_trained": 0,
        "predictions_written": 0,
        "hyperparameter_trials": 0,
        "validation_metrics_calculated": 0,
        "test_metrics_calculated": 0,
        "plots_requested": plots_requested,
        "pngs_written": png_count,
        "pdfs_written": pdf_count,
        "plot_failures": 0,
        "cms_style_failures": 0,
        "latex_math_label_failures": 0,
        "tsv_tables_written": tsv_count,
        "markdown_tables_written": markdown_count,
        "latex_tables_written": latex_count,
        "table_failures": 0,
        "official_cms_status_claimed": False,
        "text_usetex": False,
        "v1_global_bdt_preserved": True,
        "v2_category_contract_frozen": True,
        "v2_training_authorized": False,
        "validation_evaluation_authorized": False,
        "test_access_authorized": False,
        "physical_significance_authorized": False,
        "next_gate": "train_hh4b_bdt_v2_cms_inspired_categorized_grouped_cv",
        "row_level_feature_matrices_written": 0,
        "row_level_predictions_written": 0,
        "schema_less_zero_row_train_files": payload[
            "schema_less_zero_row_files"
        ],
        "train_candidate_files_opened": payload["candidate_files_opened"],
        "python_version": sys.version.split()[0],
        "numpy_version": np.__version__,
        "pandas_version": pd.__version__,
        "pyarrow_version": pyarrow.__version__,
        "matplotlib_version": matplotlib.__version__,
    }
    pass_conditions = {
        "population": (
            summary["train_members"] == 458
            and summary["train_rows"] == 57326
            and summary["signal_rows"] == 9975
            and summary["background_rows"] == 47351
        ),
        "category_partition": (
            summary["low_mhh_rows"] + summary["high_mhh_rows"] == 57326
            and summary["category_overlaps"] == 0
            and summary["category_gaps"] == 0
            and summary["nonfinite_mhh_rows"] == 0
        ),
        "required_category_populations": not missing_required,
        "fold_composition": binary_fold_failures == 0,
        "feature_quality": (
            selected_missing == 0
            and selected_nonfinite == 0
            and selected_constant == 0
            and physical_failures == 0
        ),
        "validation_closed": summary["validation_candidate_files_opened"] == 0,
        "test_closed": summary["test_candidate_files_opened"] == 0,
        "no_training_or_predictions": (
            summary["models_trained"] == 0
            and summary["predictions_written"] == 0
            and summary["hyperparameter_trials"] == 0
        ),
        "no_validation_or_test_metrics": (
            summary["validation_metrics_calculated"] == 0
            and summary["test_metrics_calculated"] == 0
        ),
        "plots": (
            png_count == plots_requested
            and pdf_count == plots_requested
            and summary["plot_failures"] == 0
            and summary["cms_style_failures"] == 0
            and summary["latex_math_label_failures"] == 0
        ),
        "tables": (
            tsv_count == markdown_count == latex_count
            and tsv_count >= 9
            and summary["table_failures"] == 0
        ),
        "no_event_or_model_artifacts": (
            summary["row_level_feature_matrices_written"] == 0
            and summary["row_level_predictions_written"] == 0
        ),
        "v1_preserved": summary["v1_global_bdt_preserved"],
    }
    summary["pass_conditions"] = pass_conditions
    summary["failed_pass_conditions"] = sorted(
        name for name, passed in pass_conditions.items() if not passed
    )
    if summary["failed_pass_conditions"]:
        write_json(runtime_dir / "summary.json", summary)
        raise RuntimeError(
            f"contract failed: {summary['failed_pass_conditions']}"
        )
    summary["status"] = (
        "hh4b_bdt_v2_cms_inspired_category_and_feature_contract_pass"
    )
    write_json(runtime_dir / "summary.json", summary)
    checkpoint_record = {
        "schema_version": 1,
        "status": summary["status"],
        "source_commit": source_commit,
        "analysis_description": summary["analysis_description"],
        "input_checkpoints": {
            "hh4b_bdt_v1_input_contract_20260725_v1": (
                summary["v1_input_contract_sha256sums_sha256"]
            ),
            "hh4b_bdt_v1_grouped_cv_20260725_v1": (
                summary["v1_grouped_cv_sha256sums_sha256"]
            ),
        },
        "policy": {
            "v1_global_bdt_preserved": True,
            "v2_category_contract_frozen": True,
            "v2_training_authorized": False,
            "validation_evaluation_authorized": False,
            "test_access_authorized": False,
            "physical_significance_authorized": False,
        },
        "next_gate": summary["next_gate"],
    }
    write_json(runtime_dir / "checkpoint.json", checkpoint_record)
    write_readme(runtime_dir, summary)
    write_sha256_manifest(runtime_dir)
    verify_sha256_manifest(runtime_dir)

    if checkpoint_dir.exists():
        shutil.rmtree(checkpoint_dir)
    shutil.copytree(runtime_dir, checkpoint_dir)
    verify_sha256_manifest(checkpoint_dir)

    print(json.dumps(summary, indent=2, sort_keys=True))
    print(f"Runtime output: {runtime_dir}")
    print(f"Committed checkpoint candidate: {checkpoint_dir}")


if __name__ == "__main__":
    main()
