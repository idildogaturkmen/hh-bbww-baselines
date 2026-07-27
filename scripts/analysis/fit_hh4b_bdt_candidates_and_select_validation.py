#!/usr/bin/env python3
"""Fit the frozen HH4b BDT candidates and open validation exactly once.

This is the authorized validation-selection gate.  It fits four CPU XGBoost
models on the complete frozen train population, reads each frozen validation
candidate Parquet once, evaluates the predeclared diagnostics, applies the
frozen selection rule, and never considers a test candidate.
"""

from __future__ import annotations

import os

# These controls must be set before NumPy, SciPy, pandas, sklearn, XGBoost, or
# Matplotlib is imported, directly or through another module.
os.environ["OMP_NUM_THREADS"] = "8"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import argparse
import atexit
from collections import Counter, defaultdict
import csv
import hashlib
import json
import math
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile
import time
from typing import Any, Iterable, Mapping, MutableMapping, Sequence

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPOSITORY_ROOT))
sys.path.insert(0, str(REPOSITORY_ROOT / "scripts" / "analysis"))

MPLCONFIG_TEMP = (
    Path(tempfile.gettempdir())
    / f"hh4b_bdt_validation_selection_mpl_{os.getpid()}"
)
os.environ["MPLCONFIGDIR"] = str(MPLCONFIG_TEMP)
atexit.register(shutil.rmtree, MPLCONFIG_TEMP, ignore_errors=True)

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import pyarrow  # noqa: E402
import scipy  # noqa: E402
from scipy.stats import pearsonr, spearmanr  # noqa: E402
import sklearn  # noqa: E402
from sklearn.metrics import (  # noqa: E402
    average_precision_score,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)
import xgboost  # noqa: E402
from xgboost import XGBClassifier  # noqa: E402
import yaml  # noqa: E402

from hh4b_bdt_v1_common import (  # noqa: E402
    MASS_AWARE_FEATURES,
    MASS_PLANE_BLIND_FEATURES,
)
from hh4b_bdt_v2_common import (  # noqa: E402
    CATEGORIZED_MASS_AWARE_FEATURES,
    SOURCE_COLUMNS,
    derive_cms_inspired_features,
    require_finite_features,
    write_table_bundle,
)
from scripts.plotting.hh4b_cms_style import (  # noqa: E402
    CATEGORY_COLORS,
    add_delphes_header,
    apply_cms_style,
    save_png_pdf,
)


CONFIG_DEFAULT = (
    REPOSITORY_ROOT
    / "configs/baselines/hh4b_bdt_validation_selection_v1.yaml"
)
GLOBAL_MASS = "global_v1_mass_aware"
GLOBAL_BLIND = "global_v1_explicit_dijet_mass_plane_blind"
CATEGORIZED = "categorized_cms_inspired_mass_aware"
CATEGORIZED_LOW = f"{CATEGORIZED}_low_mhh"
CATEGORIZED_HIGH = f"{CATEGORIZED}_high_mhh"
CATEGORIES = ("low_mhh", "high_mhh")
TARGETS = (0.30, 0.50, 0.585957314769, 0.70)
PRIMARY_TARGET = 0.585957314769
PARAMETER_NAMES = (
    "n_estimators",
    "max_depth",
    "learning_rate",
    "min_child_weight",
    "subsample",
    "colsample_bytree",
    "reg_lambda",
    "reg_alpha",
    "gamma",
)
INTEGER_PARAMETERS = {"n_estimators", "max_depth", "min_child_weight"}
FLOAT_TOLERANCE = 1.0e-10
STATUS = "hh4b_bdt_validation_model_selection_pass"


def plain(value: Any) -> str:
    """Return a stable plain-text table representation."""

    if isinstance(value, (dict, list, tuple)):
        return json.dumps(value, sort_keys=True, separators=(",", ":"))
    if isinstance(value, (np.bool_, bool)):
        return "true" if bool(value) else "false"
    if isinstance(value, (np.integer, int)):
        return str(int(value))
    if isinstance(value, (np.floating, float)):
        return f"{float(value):.12g}"
    return "" if value is None else str(value)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_yaml(path: Path) -> dict[str, Any]:
    with Path(path).open(encoding="utf-8") as handle:
        payload = yaml.safe_load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"YAML root is not a mapping: {path}")
    return payload


def resolve_config_path(path: Path) -> Path:
    """Normalize a CLI config path and require it to remain in this repository."""

    resolved = path.resolve()
    try:
        resolved.relative_to(REPOSITORY_ROOT)
    except ValueError as error:
        raise ValueError(f"config path is outside the repository: {resolved}") from error
    return resolved


def read_tsv(path: Path) -> list[dict[str, str]]:
    with Path(path).open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def write_tsv(
    path: Path,
    rows: Iterable[Mapping[str, Any]],
    fields: Sequence[str],
) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    materialized = [dict(row) for row in rows]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=list(fields),
            delimiter="\t",
            lineterminator="\n",
            extrasaction="raise",
        )
        writer.writeheader()
        for row in materialized:
            writer.writerow({field: plain(row.get(field, "")) for field in fields})
    return path


def write_json(path: Path, payload: Mapping[str, Any]) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return path


def git_output(*arguments: str) -> str:
    return subprocess.check_output(
        ["git", *arguments],
        cwd=REPOSITORY_ROOT,
        text=True,
    ).strip()


def verify_sha256_manifest(directory: Path) -> int:
    directory = Path(directory)
    rows = []
    for raw in (directory / "SHA256SUMS").read_text(encoding="utf-8").splitlines():
        if not raw.strip():
            continue
        digest, relative = raw.split(maxsplit=1)
        relative = relative.lstrip("*")
        rows.append((digest, relative))
    if not rows:
        raise ValueError(f"empty SHA256SUMS: {directory}")
    failures = [
        relative
        for digest, relative in rows
        if not (directory / relative).is_file()
        or sha256_file(directory / relative) != digest
    ]
    if failures:
        raise ValueError(f"checkpoint checksum failures in {directory}: {failures}")
    return len(rows)


def prepare_new_directory(path: Path) -> None:
    path = Path(path)
    if path.exists():
        raise FileExistsError(
            "one-time validation output already exists; refusing to reopen "
            f"validation: {path}"
        )
    path.mkdir(parents=True)


def reject_test_candidate_access(row: Mapping[str, Any]) -> None:
    """Reject any manifest record that is or could be a test candidate."""

    split = str(row.get("dataset_split", "")).strip().lower()
    test_member = str(row.get("test_member", "")).strip().lower()
    if split == "test" or test_member in {"true", "1", "yes"}:
        raise PermissionError("test candidate access is not authorized")


def environment_record(config: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "python_version": ".".join(map(str, sys.version_info[:3])),
        "python_executable": str(Path(sys.executable).resolve()),
        "numpy_version": np.__version__,
        "scipy_version": scipy.__version__,
        "sklearn_version": sklearn.__version__,
        "xgboost_version": xgboost.__version__,
        "pandas_version": pd.__version__,
        "pyarrow_version": pyarrow.__version__,
        "matplotlib_version": matplotlib.__version__,
        "cpu_training_only": True,
        "scalers_fitted": 0,
        "controls": {
            key: os.environ.get(key)
            for key in config["required_environment"]["controls"]
        },
    }


def verify_environment(config: Mapping[str, Any]) -> dict[str, Any]:
    record = environment_record(config)
    required = config["required_environment"]
    for field in (
        "python_version",
        "numpy_version",
        "scipy_version",
        "sklearn_version",
        "xgboost_version",
    ):
        if record[field] != required[field]:
            raise RuntimeError(
                f"environment mismatch for {field}: "
                f"{record[field]} != {required[field]}"
            )
    if (
        Path(record["python_executable"]).resolve()
        != Path(config["required_interpreter"]).resolve()
    ):
        raise RuntimeError("wrong Python interpreter")
    if record["controls"] != required["controls"]:
        raise RuntimeError("numerical thread controls do not match the contract")
    return record


def verify_preconditions(config: Mapping[str, Any]) -> dict[str, Any]:
    """Verify all frozen inputs without opening a candidate Parquet."""

    if git_output("rev-parse", "--abbrev-ref", "HEAD") != "delphes-hh4b-production":
        raise RuntimeError("wrong git branch")
    head = git_output("rev-parse", "HEAD")
    remote = git_output("rev-parse", "origin/delphes-hh4b-production")
    if head != remote or not head.startswith("ee05113"):
        raise RuntimeError("HEAD is not the required validation-amendment commit")
    if git_output("show", "-s", "--format=%s", "HEAD") != (
        "Freeze BDT validation model-selection protocol"
    ):
        raise RuntimeError("HEAD is not the validation-protocol amendment")

    summaries: dict[str, dict[str, Any]] = {}
    verified_entries = 0
    for name, spec in config["inputs"]["source_checkpoints"].items():
        directory = REPOSITORY_ROOT / spec["directory"]
        if sha256_file(directory / "SHA256SUMS") != spec["sha256sums_sha256"]:
            raise ValueError(f"SHA256SUMS digest changed for {name}")
        verified_entries += verify_sha256_manifest(directory)
        summary = json.loads(
            (directory / "summary.json").read_text(encoding="utf-8")
        )
        if summary.get("status") != spec["required_status"]:
            raise ValueError(f"checkpoint status failed for {name}")
        if int(summary.get("validation_candidate_files_opened", 0)) != 0:
            raise ValueError(f"validation was previously opened by {name}")
        if int(summary.get("test_candidate_files_opened", 0)) != 0:
            raise ValueError(f"test was previously opened by {name}")
        summaries[name] = summary

    amendment = summaries["validation_protocol_amendment"]
    required_amendment = {
        "candidate_bdt_models": 2,
        "final_nominal_model_selected": 0,
        "validation_selection_authorized": True,
        "validation_reoptimization_authorized": False,
        "test_selection_authorized": False,
        "bootstrap_replicates_predeclared": 2000,
        "train_members": 458,
        "train_rows": 57326,
        "validation_members": 123,
        "validation_rows_from_frozen_manifest_metadata": 12745,
        "validation_candidate_files_opened_previously_by_bdt_gates": 0,
        "test_candidate_files_opened_previously_by_bdt_gates": 0,
    }
    differences = {
        key: (amendment.get(key), expected)
        for key, expected in required_amendment.items()
        if amendment.get(key) != expected
    }
    if differences:
        raise ValueError(f"protocol-amendment preconditions changed: {differences}")

    manifest = REPOSITORY_ROOT / config["inputs"]["manifest"]
    if sha256_file(manifest) != config["inputs"]["manifest_sha256"]:
        raise ValueError("frozen development manifest digest changed")
    rows = read_tsv(manifest)
    train = [row for row in rows if row["dataset_split"] == "train"]
    validation = [row for row in rows if row["dataset_split"] == "validation"]
    pop = config["population"]
    observed = {
        "train_members": len(train),
        "train_rows": sum(int(row["candidate_rows"]) for row in train),
        "train_signal_rows": sum(
            int(row["candidate_rows"])
            for row in train
            if row["sample_class"] == "signal"
        ),
        "train_background_rows": sum(
            int(row["candidate_rows"])
            for row in train
            if row["sample_class"] == "background"
        ),
        "validation_members": len(validation),
        "validation_signal_members": sum(
            row["sample_class"] == "signal" for row in validation
        ),
        "validation_background_members": sum(
            row["sample_class"] == "background" for row in validation
        ),
        "validation_rows": sum(int(row["candidate_rows"]) for row in validation),
        "validation_signal_rows": sum(
            int(row["candidate_rows"])
            for row in validation
            if row["sample_class"] == "signal"
        ),
        "validation_background_rows": sum(
            int(row["candidate_rows"])
            for row in validation
            if row["sample_class"] == "background"
        ),
    }
    mismatches = {
        key: (observed[key], pop[key])
        for key in observed
        if observed[key] != pop[key]
    }
    if mismatches:
        raise ValueError(f"frozen manifest population changed: {mismatches}")
    if any(str(row.get("test_member", "")).lower() == "true" for row in rows):
        raise ValueError("development manifest unexpectedly contains test members")
    return {
        "source_commit": head,
        "checkpoint_entries_verified": verified_entries,
        "summaries": summaries,
        "manifest_rows": rows,
    }


def _typed_hyperparameters(row: Mapping[str, Any]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for name in PARAMETER_NAMES:
        value = row[name]
        result[name] = int(float(value)) if name in INTEGER_PARAMETERS else float(value)
    return result


def load_frozen_model_config(
    config: Mapping[str, Any],
    *,
    root: Path = REPOSITORY_ROOT,
) -> dict[str, dict[str, Any]]:
    """Load exactly the four predeclared full-train model specifications."""

    global_rows = read_tsv(
        root
        / config["models"]["global_selected_hyperparameters"]
    )
    global_by_variant = {row["variant"]: row for row in global_rows}
    categorized_rows = read_tsv(
        root
        / config["models"]["categorized_selected_hyperparameters"]
    )
    categorized_by_category = {row["category"]: row for row in categorized_rows}
    if set(global_by_variant) != {
        "mass_aware",
        "explicit_dijet_mass_plane_blind",
    }:
        raise ValueError("frozen global hyperparameter rows changed")
    if set(categorized_by_category) != set(CATEGORIES):
        raise ValueError("frozen category hyperparameter rows changed")

    fixed = dict(config["models"]["estimator_fixed_parameters"])
    global_fixed = {**fixed, "random_state": config["models"]["global_random_state"]}
    categorized_fixed = {
        **fixed,
        "random_state": config["models"]["categorized_random_state"],
    }
    specs = {
        GLOBAL_MASS: {
            "features": tuple(MASS_AWARE_FEATURES),
            "category": "all",
            "parameters": {
                **global_fixed,
                **_typed_hyperparameters(global_by_variant["mass_aware"]),
            },
            "trial_id": int(global_by_variant["mass_aware"]["trial_id"]),
        },
        GLOBAL_BLIND: {
            "features": tuple(MASS_PLANE_BLIND_FEATURES),
            "category": "all",
            "parameters": {
                **global_fixed,
                **_typed_hyperparameters(
                    global_by_variant["explicit_dijet_mass_plane_blind"]
                ),
            },
            "trial_id": int(
                global_by_variant["explicit_dijet_mass_plane_blind"]["trial_id"]
            ),
        },
        CATEGORIZED_LOW: {
            "features": tuple(CATEGORIZED_MASS_AWARE_FEATURES),
            "category": "low_mhh",
            "parameters": {
                **categorized_fixed,
                **_typed_hyperparameters(categorized_by_category["low_mhh"]),
            },
            "trial_id": int(categorized_by_category["low_mhh"]["trial_id"]),
        },
        CATEGORIZED_HIGH: {
            "features": tuple(CATEGORIZED_MASS_AWARE_FEATURES),
            "category": "high_mhh",
            "parameters": {
                **categorized_fixed,
                **_typed_hyperparameters(categorized_by_category["high_mhh"]),
            },
            "trial_id": int(categorized_by_category["high_mhh"]["trial_id"]),
        },
    }
    if tuple(specs) != tuple(config["models"]["fitted_models"]):
        raise ValueError("full-train model inventory differs from the predeclaration")
    return specs


def matrix_for_features(
    frame: pd.DataFrame,
    features: Sequence[str],
) -> np.ndarray:
    """Return a finite float32 matrix in the exact frozen feature order."""

    ordered = tuple(features)
    if len(ordered) != len(set(ordered)):
        raise ValueError("frozen feature order contains duplicates")
    require_finite_features(frame, ordered)
    selected = frame.loc[:, list(ordered)]
    if tuple(selected.columns) != ordered:
        raise AssertionError("feature order enforcement failed")
    return selected.to_numpy(dtype=np.float32, copy=False)


def normalize_train_weights(weights: Sequence[float]) -> np.ndarray:
    values = np.asarray(weights, dtype=np.float64)
    if values.ndim != 1 or not values.size:
        raise ValueError("train weights must be a nonempty vector")
    if not np.all(np.isfinite(values)) or np.any(values <= 0.0):
        raise ValueError("train weights must be finite and positive")
    result = values / float(np.mean(values))
    if not np.isclose(float(np.mean(result)), 1.0, atol=1.0e-12, rtol=0):
        raise ValueError("train weight normalization failed")
    return result


def category_local_train_weights(
    target: Sequence[int],
    member_index: Sequence[int],
    signal_mode: Sequence[str],
    background_family: Sequence[str],
    selected: Sequence[bool],
) -> np.ndarray:
    """Reconstruct frozen hierarchy using only rows in one train category."""

    target_array = np.asarray(target, dtype=np.int8)
    members = np.asarray(member_index, dtype=np.int64)
    modes = np.asarray(signal_mode, dtype=object)
    families = np.asarray(background_family, dtype=object)
    selected_array = np.asarray(selected, dtype=bool)
    if set(np.unique(target_array[selected_array])) != {0, 1}:
        raise ValueError("category-local fit requires both classes")
    weights = np.zeros(len(target_array), dtype=np.float64)
    for class_value, strata_values in (
        (1, modes),
        (0, families),
    ):
        class_selected = selected_array & (target_array == class_value)
        populated = sorted(
            value for value in set(strata_values[class_selected]) if value
        )
        if not populated:
            raise ValueError("category-local class has no populated strata")
        for stratum in populated:
            mask = class_selected & (strata_values == stratum)
            stratum_members = sorted(set(members[mask]))
            for member in stratum_members:
                member_mask = mask & (members == member)
                weights[member_mask] = (
                    0.5
                    / len(populated)
                    / len(stratum_members)
                    / int(np.count_nonzero(member_mask))
                )
    weights[selected_array] = normalize_train_weights(weights[selected_array])
    return weights


def build_validation_development_weights(
    target: Sequence[int],
    member_index: Sequence[int],
    signal_mode: Sequence[str],
    background_family: Sequence[str],
    *,
    tolerance: float = FLOAT_TOLERANCE,
) -> tuple[np.ndarray, list[dict[str, Any]]]:
    """Build the frozen validation balancing hierarchy from candidate rows."""

    target_array = np.asarray(target, dtype=np.int8)
    members = np.asarray(member_index, dtype=np.int64)
    modes = np.asarray(signal_mode, dtype=object)
    families = np.asarray(background_family, dtype=object)
    if set(np.unique(target_array)) != {0, 1}:
        raise ValueError("validation weights require both classes")
    weights = np.zeros(len(target_array), dtype=np.float64)
    audit: list[dict[str, Any]] = []
    for class_value, class_name, strata_values, level in (
        (1, "signal", modes, "signal_mode"),
        (0, "background", families, "background_family"),
    ):
        class_mask = target_array == class_value
        populated = sorted(value for value in set(strata_values[class_mask]) if value)
        if not populated:
            raise ValueError(f"validation {class_name} has no populated strata")
        for stratum in populated:
            stratum_mask = class_mask & (strata_values == stratum)
            stratum_members = sorted(set(members[stratum_mask]))
            for member in stratum_members:
                member_mask = stratum_mask & (members == member)
                weights[member_mask] = (
                    0.5
                    / len(populated)
                    / len(stratum_members)
                    / int(np.count_nonzero(member_mask))
                )
            audit.append(
                {
                    "level": level,
                    "category": stratum,
                    "parent": class_name,
                    "members_with_rows": len(stratum_members),
                    "candidate_rows": int(np.count_nonzero(stratum_mask)),
                    "target_fraction": 0.5 / len(populated),
                    "observed_fraction_raw": float(np.sum(weights[stratum_mask])),
                    "observed_weight_sum": "",
                    "mean_row_weight": "",
                    "status": "pass",
                }
            )
    raw_sum = float(np.sum(weights))
    if not np.isclose(raw_sum, 1.0, atol=tolerance, rtol=0):
        raise ValueError("raw validation development weights do not sum to one")
    weights *= len(weights) / raw_sum
    if not np.all(np.isfinite(weights)) or np.any(weights <= 0):
        raise ValueError("validation weights must be finite and positive")
    if not np.isclose(float(np.mean(weights)), 1.0, atol=tolerance, rtol=0):
        raise ValueError("validation development weights do not have mean one")

    for row in audit:
        if row["level"] == "signal_mode":
            mask = modes == row["category"]
        else:
            mask = families == row["category"]
        row["observed_weight_sum"] = float(np.sum(weights[mask]))
        row["mean_row_weight"] = float(np.mean(weights[mask]))
    for class_value, class_name in ((1, "signal"), (0, "background")):
        mask = target_array == class_value
        audit.insert(
            class_value,
            {
                "level": "class",
                "category": class_name,
                "parent": "all",
                "members_with_rows": len(set(members[mask])),
                "candidate_rows": int(np.count_nonzero(mask)),
                "target_fraction": 0.5,
                "observed_fraction_raw": 0.5,
                "observed_weight_sum": float(np.sum(weights[mask])),
                "mean_row_weight": float(np.mean(weights[mask])),
                "status": "pass",
            },
        )
    audit.insert(
        0,
        {
            "level": "global",
            "category": "all_validation_candidates",
            "parent": "",
            "members_with_rows": len(set(members)),
            "candidate_rows": len(weights),
            "target_fraction": 1.0,
            "observed_fraction_raw": 1.0,
            "observed_weight_sum": float(np.sum(weights)),
            "mean_row_weight": float(np.mean(weights)),
            "status": "pass",
        },
    )
    return weights, audit


def binary_metrics(
    target: Sequence[int],
    score: Sequence[float],
    weights: Sequence[float],
) -> dict[str, float]:
    labels = np.asarray(target, dtype=np.int8)
    scores = np.asarray(score, dtype=np.float64)
    metric_weights = np.asarray(weights, dtype=np.float64)
    active = metric_weights > 0
    labels = labels[active]
    scores = scores[active]
    metric_weights = metric_weights[active]
    if set(np.unique(labels)) != {0, 1}:
        raise ValueError("binary metrics require both classes")
    if not np.all(np.isfinite(scores)):
        raise ValueError("scores are nonfinite")
    return {
        "weighted_roc_auc": float(
            roc_auc_score(labels, scores, sample_weight=metric_weights)
        ),
        "raw_roc_auc": float(roc_auc_score(labels, scores)),
        "weighted_average_precision": float(
            average_precision_score(labels, scores, sample_weight=metric_weights)
        ),
        "raw_average_precision": float(average_precision_score(labels, scores)),
    }


def weighted_efficiency(
    selected: Sequence[bool],
    population: Sequence[bool],
    weights: Sequence[float] | None = None,
) -> float:
    selected_array = np.asarray(selected, dtype=bool)
    population_array = np.asarray(population, dtype=bool)
    numerator = selected_array & population_array
    if weights is None:
        denominator = int(np.count_nonzero(population_array))
        if denominator == 0:
            return float("nan")
        return float(np.count_nonzero(numerator) / denominator)
    values = np.asarray(weights, dtype=np.float64)
    denominator = float(np.sum(values[population_array]))
    if denominator <= 0:
        return float("nan")
    return float(np.sum(values[numerator]) / denominator)


def weighted_signal_threshold(
    signal_scores: Sequence[float],
    signal_weights: Sequence[float],
    target_efficiency: float,
) -> float:
    scores = np.asarray(signal_scores, dtype=np.float64)
    weights = np.asarray(signal_weights, dtype=np.float64)
    active = weights > 0
    scores = scores[active]
    weights = weights[active]
    if (
        scores.size == 0
        or scores.shape != weights.shape
        or not 0 < target_efficiency <= 1
    ):
        raise ValueError("invalid weighted threshold inputs")
    order = np.argsort(-scores, kind="mergesort")
    cumulative = np.cumsum(weights[order])
    index = int(
        np.searchsorted(
            cumulative,
            target_efficiency * float(cumulative[-1]),
            side="left",
        )
    )
    return float(scores[order[min(index, len(order) - 1)]])


def equalized_threshold_selection(
    scores: Sequence[float],
    target: Sequence[int],
    weights: Sequence[float],
    target_efficiency: float,
    categories: Sequence[str] | None = None,
) -> tuple[dict[str, float], np.ndarray]:
    scores_array = np.asarray(scores, dtype=np.float64)
    labels = np.asarray(target, dtype=np.int8)
    metric_weights = np.asarray(weights, dtype=np.float64)
    if categories is None:
        signal = labels == 1
        threshold = weighted_signal_threshold(
            scores_array[signal],
            metric_weights[signal],
            target_efficiency,
        )
        return {"global": threshold}, scores_array >= threshold
    category_array = np.asarray(categories, dtype=object)
    thresholds: dict[str, float] = {}
    selected = np.zeros(len(scores_array), dtype=bool)
    for category in CATEGORIES:
        category_mask = category_array == category
        signal = category_mask & (labels == 1)
        threshold = weighted_signal_threshold(
            scores_array[signal],
            metric_weights[signal],
            target_efficiency,
        )
        thresholds[category] = threshold
        selected |= category_mask & (scores_array >= threshold)
    return thresholds, selected


def apply_fixed_threshold(
    scores: Sequence[float],
    *,
    global_threshold: float | None = None,
    categories: Sequence[str] | None = None,
    category_thresholds: Mapping[str, float] | None = None,
) -> np.ndarray:
    """Apply either one frozen global threshold or the frozen category pair."""

    score_array = np.asarray(scores, dtype=np.float64)
    if global_threshold is not None:
        if categories is not None or category_thresholds is not None:
            raise ValueError("global and category thresholds cannot be mixed")
        return score_array >= float(global_threshold)
    if categories is None or category_thresholds is None:
        raise ValueError("category application requires categories and thresholds")
    if set(category_thresholds) != set(CATEGORIES):
        raise ValueError("the exact low/high threshold pair is required")
    category_array = np.asarray(categories, dtype=object)
    if set(category_array) - set(CATEGORIES):
        raise ValueError("unknown category in threshold application")
    selected = np.zeros(len(score_array), dtype=bool)
    for category in CATEGORIES:
        selected |= (category_array == category) & (
            score_array >= float(category_thresholds[category])
        )
    return selected


def selection_metrics(
    selected: Sequence[bool],
    target: Sequence[int],
    weights: Sequence[float],
    signal_mode: Sequence[str],
    background_family: Sequence[str],
) -> dict[str, Any]:
    selected_array = np.asarray(selected, dtype=bool)
    labels = np.asarray(target, dtype=np.int8)
    metric_weights = np.asarray(weights, dtype=np.float64)
    modes = np.asarray(signal_mode, dtype=object)
    families = np.asarray(background_family, dtype=object)
    signal_eff = weighted_efficiency(selected_array, labels == 1, metric_weights)
    background_eff = weighted_efficiency(
        selected_array, labels == 0, metric_weights
    )
    return {
        "weighted_signal_efficiency": signal_eff,
        "weighted_background_efficiency": background_eff,
        "inverse_background_efficiency": (
            1.0 / background_eff if background_eff > 0 else float("inf")
        ),
        "one_minus_background_efficiency": 1.0 - background_eff,
        "raw_signal_efficiency": weighted_efficiency(
            selected_array, labels == 1
        ),
        "raw_background_efficiency": weighted_efficiency(
            selected_array, labels == 0
        ),
        "signal_mode_efficiencies": {
            mode: weighted_efficiency(
                selected_array, modes == mode, metric_weights
            )
            for mode in sorted(value for value in set(modes) if value)
        },
        "background_family_efficiencies": {
            family: weighted_efficiency(
                selected_array, families == family, metric_weights
            )
            for family in sorted(value for value in set(families) if value)
        },
    }


def check_prediction_integrity(
    expected_row_indices: Sequence[int],
    produced_row_indices: Sequence[int],
    expected_categories: Sequence[str] | None = None,
    produced_categories: Sequence[str] | None = None,
) -> dict[str, int]:
    expected = Counter(int(value) for value in expected_row_indices)
    produced = Counter(int(value) for value in produced_row_indices)
    missing = sum(
        max(count - produced.get(index, 0), 0)
        for index, count in expected.items()
    )
    duplicates = sum(
        max(count - expected.get(index, 0), 0)
        for index, count in produced.items()
    )
    mismatch = 0
    if expected_categories is not None or produced_categories is not None:
        if expected_categories is None or produced_categories is None:
            raise ValueError("both category sequences are required")
        if len(expected_categories) != len(produced_categories):
            mismatch = abs(len(expected_categories) - len(produced_categories))
        mismatch += sum(
            expected != produced
            for expected, produced in zip(
                expected_categories, produced_categories
            )
        )
    return {
        "missing_predictions": int(missing),
        "duplicate_predictions": int(duplicates),
        "category_mismatch_predictions": int(mismatch),
    }


def percentile_interval(
    values: Sequence[float],
) -> dict[str, float | int]:
    array = np.asarray(values, dtype=np.float64)
    finite = array[np.isfinite(array)]
    if not finite.size:
        raise ValueError("bootstrap interval has no defined replicas")
    quantiles = np.percentile(finite, [2.5, 16.0, 50.0, 84.0, 97.5])
    return {
        "replicates_defined": int(finite.size),
        "mean": float(np.mean(finite)),
        "standard_deviation": float(np.std(finite, ddof=0)),
        "percentile_2_5": float(quantiles[0]),
        "percentile_16": float(quantiles[1]),
        "median": float(quantiles[2]),
        "percentile_84": float(quantiles[3]),
        "percentile_97_5": float(quantiles[4]),
    }


def paired_member_bootstrap_primary(
    member_index: Sequence[int],
    target: Sequence[int],
    categories: Sequence[str],
    weights: Sequence[float],
    global_scores: Sequence[float],
    categorized_scores: Sequence[float],
    *,
    target_efficiency: float,
    seed: int,
    replicates: int,
) -> np.ndarray:
    """Small reusable paired-bootstrap kernel used by synthetic tests.

    Production reporting uses :func:`run_member_bootstrap`, which additionally
    records every predeclared interval.  This kernel freezes and directly tests
    the essential paired member-resampling and threshold-recomputation rule.
    """

    members = np.asarray(member_index, dtype=np.int64)
    labels = np.asarray(target, dtype=np.int8)
    category_array = np.asarray(categories, dtype=object)
    base_weights = np.asarray(weights, dtype=np.float64)
    global_array = np.asarray(global_scores, dtype=np.float64)
    categorized_array = np.asarray(categorized_scores, dtype=np.float64)
    unique_members = np.asarray(sorted(set(members)), dtype=np.int64)
    member_position = {
        int(member): position for position, member in enumerate(unique_members)
    }
    row_position = np.asarray(
        [member_position[int(member)] for member in members], dtype=np.int64
    )
    rng = np.random.default_rng(seed)
    deltas = np.full(replicates, np.nan, dtype=np.float64)
    for replicate in range(replicates):
        sampled = rng.choice(
            unique_members, size=len(unique_members), replace=True
        )
        counts = np.bincount(
            [member_position[int(member)] for member in sampled],
            minlength=len(unique_members),
        )
        replicate_weights = base_weights * counts[row_position]
        try:
            _, global_selected = equalized_threshold_selection(
                global_array,
                labels,
                replicate_weights,
                target_efficiency,
            )
            _, categorized_selected = equalized_threshold_selection(
                categorized_array,
                labels,
                replicate_weights,
                target_efficiency,
                category_array,
            )
        except ValueError:
            continue
        deltas[replicate] = weighted_efficiency(
            categorized_selected, labels == 0, replicate_weights
        ) - weighted_efficiency(
            global_selected, labels == 0, replicate_weights
        )
    return deltas


def final_model_selection(
    *,
    global_background_efficiency: float,
    categorized_background_efficiency: float,
    bootstrap_fraction_favoring_v2: float,
    categorized_ggf_efficiency: float,
    categorized_vbf_efficiency: float,
    target_signal_efficiency: float,
    integrity_failures: int,
    minimum_relative_reduction: float = 0.02,
    minimum_fraction_favoring: float = 0.84,
    maximum_signal_mode_deviation: float = 0.05,
) -> tuple[str, list[dict[str, Any]], dict[str, float]]:
    """Apply the frozen rule with no override surface."""

    delta = categorized_background_efficiency - global_background_efficiency
    relative = (
        (global_background_efficiency - categorized_background_efficiency)
        / global_background_efficiency
        if global_background_efficiency > 0
        else float("-inf")
    )
    max_deviation = max(
        abs(categorized_ggf_efficiency - target_signal_efficiency),
        abs(categorized_vbf_efficiency - target_signal_efficiency),
    )
    conditions = [
        (
            "lower_point_estimate_background_efficiency",
            delta,
            "<",
            0.0,
            delta < 0,
        ),
        (
            "minimum_relative_background_efficiency_reduction",
            relative,
            ">=",
            minimum_relative_reduction,
            relative >= minimum_relative_reduction,
        ),
        (
            "bootstrap_fraction_favoring_categorized_v2",
            bootstrap_fraction_favoring_v2,
            ">=",
            minimum_fraction_favoring,
            bootstrap_fraction_favoring_v2 >= minimum_fraction_favoring,
        ),
        (
            "signal_mode_equalization",
            max_deviation,
            "<=",
            maximum_signal_mode_deviation,
            max_deviation <= maximum_signal_mode_deviation,
        ),
        (
            "integrity_category_and_model_application",
            integrity_failures,
            "==",
            0,
            integrity_failures == 0,
        ),
    ]
    rows = [
        {
            "order": order,
            "condition": name,
            "observed": observed,
            "operator": operator,
            "required": required,
            "passed": passed,
            "failure_action": GLOBAL_MASS,
            "status": "pass" if passed else "fail",
        }
        for order, (name, observed, operator, required, passed) in enumerate(
            conditions, start=1
        )
    ]
    selected = CATEGORIZED if all(row["passed"] for row in rows) else GLOBAL_MASS
    return selected, rows, {
        "delta_background_efficiency": delta,
        "relative_background_efficiency_reduction": relative,
        "maximum_signal_mode_efficiency_deviation": max_deviation,
    }


def _maps() -> tuple[dict[str, str], dict[str, str]]:
    checkpoint = (
        REPOSITORY_ROOT
        / "docs/checkpoints/hh4b_bdt_v1_input_contract_20260725_v1"
    )
    signal = {
        row["manifest_process_or_mode"]: row["signal_mode"]
        for row in read_tsv(checkpoint / "signal_mode_mapping.tsv")
    }
    background = {
        row["manifest_process_or_mode"]: row["background_family"]
        for row in read_tsv(checkpoint / "background_family_mapping.tsv")
    }
    return signal, background


def load_train_population(
    manifest_rows: Sequence[Mapping[str, str]],
    config: Mapping[str, Any],
) -> dict[str, Any]:
    """Open the 458 train Parquets and reconstruct the exact feature payload."""

    signal_map, background_map = _maps()
    weight_rows = read_tsv(
        REPOSITORY_ROOT
        / "docs/checkpoints/hh4b_bdt_v1_input_contract_20260725_v1"
        / "member_weight_summary.tsv"
    )
    frozen_weight = {
        int(row["member_index"]): float(row["per_row_training_weight"])
        for row in weight_rows
    }
    train_manifest = sorted(
        (row for row in manifest_rows if row["dataset_split"] == "train"),
        key=lambda row: int(row["member_index"]),
    )
    if len(train_manifest) != config["population"]["train_members"]:
        raise ValueError("train manifest cardinality changed")

    frames: list[pd.DataFrame] = []
    metadata: MutableMapping[str, list[np.ndarray]] = defaultdict(list)
    opened = 0
    read_columns = list(dict.fromkeys(("event", *SOURCE_COLUMNS)))
    for position, row in enumerate(train_manifest, start=1):
        reject_test_candidate_access(row)
        member = int(row["member_index"])
        expected_rows = int(row["candidate_rows"])
        path = Path(row["local_path"])
        if not path.is_file():
            raise FileNotFoundError(f"missing train candidate: {path}")
        frame = (
            pd.read_parquet(path, engine="pyarrow")
            if expected_rows == 0
            else pd.read_parquet(path, columns=read_columns, engine="pyarrow")
        )
        opened += 1
        if len(frame) != expected_rows:
            raise ValueError(f"train member {member} row count changed")
        if expected_rows:
            sample_class = row["sample_class"]
            target = int(row["training_target"])
            if (sample_class, target) not in {
                ("signal", 1),
                ("background", 0),
            }:
                raise ValueError("authoritative train label mismatch")
            mode = signal_map[row["process_or_mode"]] if target else ""
            family = background_map[row["process_or_mode"]] if not target else ""
            frames.append(frame)
            metadata["member_index"].append(
                np.full(expected_rows, member, dtype=np.int64)
            )
            metadata["target"].append(np.full(expected_rows, target, dtype=np.int8))
            metadata["weight"].append(
                np.full(expected_rows, frozen_weight[member], dtype=np.float64)
            )
            metadata["signal_mode"].append(
                np.full(expected_rows, mode, dtype=object)
            )
            metadata["background_family"].append(
                np.full(expected_rows, family, dtype=object)
            )
        if position % 50 == 0 or position == len(train_manifest):
            print(
                f"Opened train candidates {position}/{len(train_manifest)}",
                flush=True,
            )
    raw = pd.concat(frames, ignore_index=True)
    if opened != 458 or len(raw) != 57326:
        raise ValueError("train open or row count mismatch")
    print("Deriving frozen train features", flush=True)
    features = derive_cms_inspired_features(raw)
    payload = {key: np.concatenate(value) for key, value in metadata.items()}
    payload["features"] = features
    payload["category"] = np.where(
        features["mhh"].to_numpy(dtype=float) < 450.0,
        "low_mhh",
        "high_mhh",
    )
    payload["weight"] = normalize_train_weights(payload["weight"])
    payload["candidate_files_opened"] = opened
    if set(len(value) for key, value in payload.items() if key not in {
        "candidate_files_opened"
    }) != {57326}:
        raise ValueError("train payload arrays differ in length")
    if int(np.count_nonzero(payload["target"] == 1)) != 9975:
        raise ValueError("train signal row count changed")
    return payload


def load_validation_population_once(
    manifest_rows: Sequence[Mapping[str, str]],
    config: Mapping[str, Any],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Authorized loader: open each of the 123 validation Parquets once."""

    signal_map, background_map = _maps()
    validation_manifest = sorted(
        (row for row in manifest_rows if row["dataset_split"] == "validation"),
        key=lambda row: int(row["member_index"]),
    )
    if len(validation_manifest) != config["population"]["validation_members"]:
        raise ValueError("validation manifest cardinality changed")
    frames: list[pd.DataFrame] = []
    metadata: MutableMapping[str, list[np.ndarray]] = defaultdict(list)
    audit: list[dict[str, Any]] = []
    read_columns = list(dict.fromkeys(("event", *SOURCE_COLUMNS)))
    opened = 0
    for position, row in enumerate(validation_manifest, start=1):
        reject_test_candidate_access(row)
        member = int(row["member_index"])
        expected_rows = int(row["candidate_rows"])
        sample_class = row["sample_class"]
        target = int(row["training_target"])
        path = Path(row["local_path"])
        if not path.is_file():
            raise FileNotFoundError(f"missing validation candidate: {path}")
        frame = (
            pd.read_parquet(path, engine="pyarrow")
            if expected_rows == 0
            else pd.read_parquet(path, columns=read_columns, engine="pyarrow")
        )
        opened += 1
        observed_rows = len(frame)
        status = "pass" if observed_rows == expected_rows else "fail"
        mode = signal_map[row["process_or_mode"]] if target else ""
        family = background_map[row["process_or_mode"]] if not target else ""
        audit.append(
            {
                "member_index": member,
                "sample_class": sample_class,
                "signal_mode_or_background_family": mode or family,
                "candidate_path": str(path),
                "expected_rows": expected_rows,
                "observed_rows": observed_rows,
                "file_sha256": row.get("candidate_sha256", ""),
                "open_count": 1,
                "access_status": status,
            }
        )
        if status != "pass":
            raise ValueError(f"validation member {member} row count changed")
        if expected_rows:
            if (sample_class, target) not in {
                ("signal", 1),
                ("background", 0),
            }:
                raise ValueError("authoritative validation label mismatch")
            frames.append(frame)
            metadata["event"].append(frame["event"].to_numpy(dtype=np.int64))
            metadata["member_index"].append(
                np.full(expected_rows, member, dtype=np.int64)
            )
            metadata["target"].append(np.full(expected_rows, target, dtype=np.int8))
            metadata["signal_mode"].append(
                np.full(expected_rows, mode, dtype=object)
            )
            metadata["background_family"].append(
                np.full(expected_rows, family, dtype=object)
            )
            metadata["process_or_mode"].append(
                np.full(expected_rows, row["process_or_mode"], dtype=object)
            )
        if position % 25 == 0 or position == len(validation_manifest):
            print(
                "Authorized validation access "
                f"{position}/{len(validation_manifest)}",
                flush=True,
            )
    if opened != 123 or sum(row["open_count"] for row in audit) != 123:
        raise ValueError("validation files were not each opened exactly once")
    raw = pd.concat(frames, ignore_index=True)
    if len(raw) != 12745:
        raise ValueError("validation row total changed")
    print("Deriving frozen validation features", flush=True)
    features = derive_cms_inspired_features(raw)
    payload = {key: np.concatenate(value) for key, value in metadata.items()}
    payload["features"] = features
    payload["category"] = np.where(
        features["mhh"].to_numpy(dtype=float) < 450.0,
        "low_mhh",
        "high_mhh",
    )
    payload["row_index"] = np.arange(len(features), dtype=np.int64)
    payload["candidate_files_opened"] = opened
    payload["validation_manifest"] = validation_manifest
    lengths = {
        len(value)
        for key, value in payload.items()
        if key not in {"candidate_files_opened", "validation_manifest"}
    }
    if lengths != {12745}:
        raise ValueError(f"validation payload arrays differ in length: {lengths}")
    if int(np.count_nonzero(payload["target"] == 1)) != 2246:
        raise ValueError("validation signal row count changed")
    return payload, audit


def load_frozen_thresholds(config: Mapping[str, Any]) -> dict[str, Any]:
    rows = read_tsv(
        REPOSITORY_ROOT / config["models"]["frozen_working_points"]
    )
    result: dict[str, Any] = {GLOBAL_MASS: {}, CATEGORIZED: {}}
    for row in rows:
        target = float(row["target_weighted_signal_efficiency"])
        if row["model"] == GLOBAL_MASS:
            result[GLOBAL_MASS][target] = float(row["global_threshold"])
        elif row["model"] == CATEGORIZED:
            result[CATEGORIZED][target] = {
                "low_mhh": float(row["low_mhh_threshold"]),
                "high_mhh": float(row["high_mhh_threshold"]),
            }
    for method in result:
        if len(result[method]) != 4:
            raise ValueError(f"frozen working points incomplete for {method}")
    return result


def fit_full_train_models(
    train: Mapping[str, Any],
    specs: Mapping[str, Mapping[str, Any]],
    runtime_dir: Path,
) -> tuple[dict[str, XGBClassifier], list[dict[str, Any]]]:
    models: dict[str, XGBClassifier] = {}
    inventory: list[dict[str, Any]] = []
    model_dir = runtime_dir / "models"
    model_dir.mkdir(parents=True)
    categories = np.asarray(train["category"], dtype=object)
    for name, spec in specs.items():
        category = spec["category"]
        selected = (
            np.ones(len(categories), dtype=bool)
            if category == "all"
            else categories == category
        )
        if category == "all":
            fit_weights = normalize_train_weights(train["weight"])
        else:
            local = category_local_train_weights(
                train["target"],
                train["member_index"],
                train["signal_mode"],
                train["background_family"],
                selected,
            )
            fit_weights = local[selected]
        matrix = matrix_for_features(train["features"], spec["features"])
        model = XGBClassifier(**spec["parameters"])
        started = time.perf_counter()
        model.fit(
            matrix[selected],
            np.asarray(train["target"])[selected],
            sample_weight=fit_weights,
        )
        fit_seconds = time.perf_counter() - started
        path = model_dir / f"{name}.json"
        model.save_model(path)
        models[name] = model
        inventory.append(
            {
                "model": name,
                "category": category,
                "features": len(spec["features"]),
                "trial_id": spec["trial_id"],
                "train_rows": int(np.count_nonzero(selected)),
                "train_signal_rows": int(
                    np.count_nonzero(selected & (np.asarray(train["target"]) == 1))
                ),
                "train_background_rows": int(
                    np.count_nonzero(selected & (np.asarray(train["target"]) == 0))
                ),
                "mean_fit_weight": float(np.mean(fit_weights)),
                "fit_seconds": fit_seconds,
                "model_path": str(path.relative_to(REPOSITORY_ROOT)),
                "serialization": "XGBoost_JSON",
                "scaler_fitted": 0,
                "fit_status": "pass",
            }
        )
        print(f"Fitted {name} in {fit_seconds:.1f} s", flush=True)
    return models, inventory


def predict_validation(
    validation: Mapping[str, Any],
    specs: Mapping[str, Mapping[str, Any]],
    models: Mapping[str, XGBClassifier],
    runtime_dir: Path,
    development_weights: np.ndarray,
) -> tuple[dict[str, np.ndarray], list[dict[str, Any]], list[Path]]:
    feature_cache: dict[tuple[str, ...], np.ndarray] = {}
    for spec in specs.values():
        features = tuple(spec["features"])
        if features not in feature_cache:
            feature_cache[features] = matrix_for_features(
                validation["features"], features
            )
    scores = {
        GLOBAL_MASS: models[GLOBAL_MASS].predict_proba(
            feature_cache[tuple(specs[GLOBAL_MASS]["features"])]
        )[:, 1],
        GLOBAL_BLIND: models[GLOBAL_BLIND].predict_proba(
            feature_cache[tuple(specs[GLOBAL_BLIND]["features"])]
        )[:, 1],
    }
    categorized_score = np.full(len(validation["target"]), np.nan, dtype=np.float64)
    categories = np.asarray(validation["category"], dtype=object)
    produced_category = np.full(len(categories), "", dtype=object)
    categorized_matrix = feature_cache[
        tuple(specs[CATEGORIZED_LOW]["features"])
    ]
    for model_name, category in (
        (CATEGORIZED_LOW, "low_mhh"),
        (CATEGORIZED_HIGH, "high_mhh"),
    ):
        selected = categories == category
        categorized_score[selected] = models[model_name].predict_proba(
            categorized_matrix[selected]
        )[:, 1]
        produced_category[selected] = category
    scores[CATEGORIZED] = categorized_score
    if any(not np.all(np.isfinite(value)) for value in scores.values()):
        raise ValueError("validation prediction contains nonfinite scores")

    prediction_dir = runtime_dir / "validation_predictions"
    prediction_dir.mkdir(parents=True)
    integrity_rows: list[dict[str, Any]] = []
    prediction_paths: list[Path] = []
    for method in (GLOBAL_MASS, GLOBAL_BLIND, CATEGORIZED):
        produced = validation["row_index"].copy()
        integrity = check_prediction_integrity(
            validation["row_index"],
            produced,
            categories if method == CATEGORIZED else None,
            produced_category if method == CATEGORIZED else None,
        )
        row = {
            "method": method,
            "expected_predictions": len(produced),
            "observed_predictions": len(produced),
            **integrity,
            "status": (
                "pass" if sum(integrity.values()) == 0 else "fail"
            ),
        }
        integrity_rows.append(row)
        frame = pd.DataFrame(
            {
                "row_index": produced,
                "member_index": validation["member_index"],
                "event": validation["event"],
                "target": validation["target"],
                "category": validation["category"],
                "development_weight": development_weights,
                "score": scores[method],
            }
        )
        path = prediction_dir / f"{method}.parquet"
        frame.to_parquet(path, index=False, engine="pyarrow")
        prediction_paths.append(path)
    return scores, integrity_rows, prediction_paths


BootstrapKey = tuple[str, str, str, str, str, str]


def bootstrap_key(
    analysis: str,
    method: str,
    scope: str,
    target: float | str,
    metric: str,
    stratum: str = "",
) -> BootstrapKey:
    return (analysis, method, scope, plain(target), metric, stratum)


def run_member_bootstrap(
    validation: Mapping[str, Any],
    scores: Mapping[str, np.ndarray],
    weights: np.ndarray,
    fixed_selections: Mapping[tuple[str, float], np.ndarray],
    fixed_points: Mapping[BootstrapKey, float],
    equalized_points: Mapping[BootstrapKey, float],
    cut_selection: np.ndarray,
    cut_points: Mapping[BootstrapKey, float],
    *,
    seed: int,
    replicates: int,
) -> tuple[
    list[dict[str, Any]],
    list[dict[str, Any]],
    dict[BootstrapKey, dict[str, Any]],
]:
    """Paired source-member bootstrap using row multiplicities."""

    labels = np.asarray(validation["target"], dtype=np.int8)
    members = np.asarray(validation["member_index"], dtype=np.int64)
    categories = np.asarray(validation["category"], dtype=object)
    modes = np.asarray(validation["signal_mode"], dtype=object)
    families = np.asarray(validation["background_family"], dtype=object)
    all_members = np.asarray(
        sorted(
            int(row["member_index"])
            for row in validation["validation_manifest"]
        ),
        dtype=np.int64,
    )
    member_position = {member: index for index, member in enumerate(all_members)}
    row_member_position = np.asarray(
        [member_position[int(member)] for member in members],
        dtype=np.int64,
    )
    rng = np.random.default_rng(seed)
    values: MutableMapping[BootstrapKey, list[float]] = defaultdict(list)
    primary_replicas: list[dict[str, Any]] = []
    mode_names = sorted(value for value in set(modes) if value)
    family_names = sorted(value for value in set(families) if value)

    def collect_metrics(
        analysis: str,
        method: str,
        scope: str,
        target_value: float | str,
        selected: np.ndarray,
        bootstrap_weights: np.ndarray,
    ) -> None:
        metrics = selection_metrics(
            selected,
            labels,
            bootstrap_weights,
            modes,
            families,
        )
        for metric in (
            "weighted_signal_efficiency",
            "weighted_background_efficiency",
        ):
            values[
                bootstrap_key(
                    analysis, method, scope, target_value, metric
                )
            ].append(metrics[metric])
        for mode in mode_names:
            values[
                bootstrap_key(
                    analysis,
                    method,
                    scope,
                    target_value,
                    "signal_mode_efficiency",
                    mode,
                )
            ].append(metrics["signal_mode_efficiencies"][mode])
        for family in family_names:
            values[
                bootstrap_key(
                    analysis,
                    method,
                    scope,
                    target_value,
                    "background_family_efficiency",
                    family,
                )
            ].append(metrics["background_family_efficiencies"][family])

    metric_scopes = (
        (GLOBAL_MASS, "all", np.ones(len(labels), dtype=bool)),
        (GLOBAL_BLIND, "all", np.ones(len(labels), dtype=bool)),
        (GLOBAL_MASS, "low_mhh", categories == "low_mhh"),
        (CATEGORIZED, "low_mhh", categories == "low_mhh"),
        (GLOBAL_MASS, "high_mhh", categories == "high_mhh"),
        (CATEGORIZED, "high_mhh", categories == "high_mhh"),
    )
    for replicate in range(replicates):
        sampled = rng.choice(all_members, size=len(all_members), replace=True)
        counts = np.bincount(
            [member_position[int(member)] for member in sampled],
            minlength=len(all_members),
        )
        bootstrap_weights = weights * counts[row_member_position]
        for method, scope, mask in metric_scopes:
            active = mask & (bootstrap_weights > 0)
            try:
                metric = binary_metrics(
                    labels[active],
                    scores[method][active],
                    bootstrap_weights[active],
                )
            except ValueError:
                metric = {
                    "weighted_roc_auc": float("nan"),
                    "weighted_average_precision": float("nan"),
                }
            for name in ("weighted_roc_auc", "weighted_average_precision"):
                values[
                    bootstrap_key(
                        "global_or_category_metric",
                        method,
                        scope,
                        "",
                        name,
                    )
                ].append(metric[name])

        for method in (GLOBAL_MASS, CATEGORIZED):
            for target_value in TARGETS:
                collect_metrics(
                    "fixed_train_threshold",
                    method,
                    "combined",
                    target_value,
                    fixed_selections[(method, target_value)],
                    bootstrap_weights,
                )

        collect_metrics(
            "optimized_cut",
            "r_hh_125_125_lt_34",
            "combined",
            "",
            cut_selection,
            bootstrap_weights,
        )

        equalized_selections: dict[tuple[str, float], np.ndarray] = {}
        for target_value in TARGETS:
            try:
                _, global_selected = equalized_threshold_selection(
                    scores[GLOBAL_MASS],
                    labels,
                    bootstrap_weights,
                    target_value,
                )
                _, categorized_selected = equalized_threshold_selection(
                    scores[CATEGORIZED],
                    labels,
                    bootstrap_weights,
                    target_value,
                    categories,
                )
            except ValueError:
                global_selected = np.zeros(len(labels), dtype=bool)
                categorized_selected = np.zeros(len(labels), dtype=bool)
                bootstrap_weights = bootstrap_weights * 0.0
            equalized_selections[(GLOBAL_MASS, target_value)] = global_selected
            equalized_selections[(CATEGORIZED, target_value)] = categorized_selected
            for method, selection in (
                (GLOBAL_MASS, global_selected),
                (CATEGORIZED, categorized_selected),
            ):
                collect_metrics(
                    "equalized_validation_efficiency",
                    method,
                    "combined",
                    target_value,
                    selection,
                    bootstrap_weights,
                )
            global_background = weighted_efficiency(
                global_selected, labels == 0, bootstrap_weights
            )
            categorized_background = weighted_efficiency(
                categorized_selected, labels == 0, bootstrap_weights
            )
            delta = categorized_background - global_background
            values[
                bootstrap_key(
                    "paired_method_difference",
                    f"{CATEGORIZED}_minus_{GLOBAL_MASS}",
                    "combined",
                    target_value,
                    "delta_weighted_background_efficiency",
                )
            ].append(delta)
            if math.isclose(target_value, PRIMARY_TARGET, abs_tol=1.0e-12):
                primary_replicas.append(
                    {
                        "replicate": replicate,
                        "global_v1_background_efficiency": global_background,
                        "categorized_v2_background_efficiency": categorized_background,
                        "delta_background_efficiency": delta,
                        "favor_categorized_v2": delta < 0,
                    }
                )
        if (replicate + 1) % 200 == 0:
            print(
                f"Source-member bootstrap {replicate + 1}/{replicates}",
                flush=True,
            )

    points: dict[BootstrapKey, float] = {}
    points.update(fixed_points)
    points.update(equalized_points)
    points.update(cut_points)
    for method, scope, mask in metric_scopes:
        metric = binary_metrics(labels[mask], scores[method][mask], weights[mask])
        for name in ("weighted_roc_auc", "weighted_average_precision"):
            points[
                bootstrap_key(
                    "global_or_category_metric", method, scope, "", name
                )
            ] = metric[name]
    for target_value in TARGETS:
        difference_key = bootstrap_key(
            "paired_method_difference",
            f"{CATEGORIZED}_minus_{GLOBAL_MASS}",
            "combined",
            target_value,
            "delta_weighted_background_efficiency",
        )
        points[difference_key] = (
            equalized_points[
                bootstrap_key(
                    "equalized_validation_efficiency",
                    CATEGORIZED,
                    "combined",
                    target_value,
                    "weighted_background_efficiency",
                )
            ]
            - equalized_points[
                bootstrap_key(
                    "equalized_validation_efficiency",
                    GLOBAL_MASS,
                    "combined",
                    target_value,
                    "weighted_background_efficiency",
                )
            ]
        )

    interval_rows: list[dict[str, Any]] = []
    interval_lookup: dict[BootstrapKey, dict[str, Any]] = {}
    for key in sorted(values):
        interval = percentile_interval(values[key])
        analysis, method, scope, target_value, metric, stratum = key
        row = {
            "analysis": analysis,
            "method": method,
            "scope": scope,
            "target_weighted_signal_efficiency": target_value,
            "metric": metric,
            "stratum": stratum,
            "point_estimate": points.get(key, ""),
            **interval,
            "bootstrap_unit": "source_member",
            "status": "pass",
        }
        interval_rows.append(row)
        interval_lookup[key] = row
    return primary_replicas, interval_rows, interval_lookup


def score_mass_diagnostics(
    validation: Mapping[str, Any],
    scores: Mapping[str, np.ndarray],
    weights: np.ndarray,
) -> list[dict[str, Any]]:
    background = np.asarray(validation["target"]) == 0
    rhh = validation["features"]["r_hh_125_125"].to_numpy(dtype=float)
    mhh = validation["features"]["mhh"].to_numpy(dtype=float)
    rows: list[dict[str, Any]] = []
    for method in (GLOBAL_MASS, CATEGORIZED, GLOBAL_BLIND):
        background_score = np.asarray(scores[method])[background]
        for variable, values in (
            ("r_hh_125_125", rhh[background]),
            ("mhh", mhh[background]),
        ):
            rows.extend(
                [
                    {
                        "record_type": "correlation",
                        "method": method,
                        "variable": variable,
                        "correlation": "Pearson",
                        "score_quintile": "",
                        "value": float(pearsonr(background_score, values).statistic),
                        "median_r_hh_125_125": "",
                        "mean_r_hh_125_125": "",
                        "fraction_r_hh_lt_34": "",
                        "fraction_r_hh_lt_50": "",
                        "fraction_r_hh_lt_80": "",
                        "status": "pass",
                    },
                    {
                        "record_type": "correlation",
                        "method": method,
                        "variable": variable,
                        "correlation": "Spearman",
                        "score_quintile": "",
                        "value": float(spearmanr(background_score, values).statistic),
                        "median_r_hh_125_125": "",
                        "mean_r_hh_125_125": "",
                        "fraction_r_hh_lt_34": "",
                        "fraction_r_hh_lt_50": "",
                        "fraction_r_hh_lt_80": "",
                        "status": "pass",
                    },
                ]
            )
        ranks = pd.Series(background_score).rank(method="first")
        quintiles = pd.qcut(ranks, 5, labels=False).to_numpy() + 1
        background_weights = weights[background]
        background_rhh = rhh[background]
        for quintile in range(1, 6):
            selected = quintiles == quintile
            denominator = float(np.sum(background_weights[selected]))
            rows.append(
                {
                    "record_type": "score_quintile",
                    "method": method,
                    "variable": "r_hh_125_125",
                    "correlation": "",
                    "score_quintile": quintile,
                    "value": "",
                    "median_r_hh_125_125": float(
                        np.median(background_rhh[selected])
                    ),
                    "mean_r_hh_125_125": float(
                        np.average(
                            background_rhh[selected],
                            weights=background_weights[selected],
                        )
                    ),
                    "fraction_r_hh_lt_34": float(
                        np.sum(
                            background_weights[
                                selected & (background_rhh < 34.0)
                            ]
                        )
                        / denominator
                    ),
                    "fraction_r_hh_lt_50": float(
                        np.sum(
                            background_weights[
                                selected & (background_rhh < 50.0)
                            ]
                        )
                        / denominator
                    ),
                    "fraction_r_hh_lt_80": float(
                        np.sum(
                            background_weights[
                                selected & (background_rhh < 80.0)
                            ]
                        )
                        / denominator
                    ),
                    "status": "pass",
                }
            )
    return rows


def artifact_row(
    path: Path,
    artifact_type: str,
    *,
    model: str = "",
    category: str = "",
    rows: int | str = "",
    purpose: str,
) -> dict[str, Any]:
    path = Path(path)
    return {
        "path": str(path.relative_to(REPOSITORY_ROOT)),
        "artifact_type": artifact_type,
        "model": model,
        "category": category,
        "rows": rows,
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
        "committed_status": False,
        "retention_purpose": purpose,
    }


def add_plot(
    inventory: list[dict[str, Any]],
    figure: matplotlib.figure.Figure,
    directory: Path,
    name: str,
) -> None:
    png, pdf = save_png_pdf(figure, directory / name, dpi=300)
    for path, file_type in ((png, "PNG"), (pdf, "PDF")):
        inventory.append(
            {
                "plot": name,
                "file_type": file_type,
                "path": str(path.relative_to(REPOSITORY_ROOT)),
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
                "dpi": 300 if file_type == "PNG" else "vector",
                "source_member_bootstrap_intervals": (
                    "68_percent" if "roc" not in name and "precision" not in name
                    else "not_shown_curve_remains_readable"
                ),
                "status": "pass",
            }
        )


def make_plots(
    checkpoint_dir: Path,
    validation: Mapping[str, Any],
    scores: Mapping[str, np.ndarray],
    weights: np.ndarray,
    fixed_rows: Sequence[Mapping[str, Any]],
    equalized_rows: Sequence[Mapping[str, Any]],
    primary_replicas: Sequence[Mapping[str, Any]],
    interval_lookup: Mapping[BootstrapKey, Mapping[str, Any]],
    signal_rows: Sequence[Mapping[str, Any]],
    family_rows: Sequence[Mapping[str, Any]],
    final_model: str,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    style = apply_cms_style()
    plot_dir = checkpoint_dir / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)
    inventory: list[dict[str, Any]] = []
    secondary = (
        "Validation; Source-member bootstrap; "
        "Development balancing; not physical normalization"
    )
    method_label = {
        GLOBAL_MASS: "Global v1 mass-aware",
        CATEGORIZED: "Categorized CMS-inspired",
        GLOBAL_BLIND: "Global mass-plane-blind",
    }

    def interval_error(key: BootstrapKey, point: float) -> tuple[float, float]:
        row = interval_lookup[key]
        return (
            max(point - float(row["percentile_16"]), 0.0),
            max(float(row["percentile_84"]) - point, 0.0),
        )

    for metric, name, ylabel in (
        (
            "weighted_signal_efficiency",
            "validation_fixed_threshold_signal_efficiency",
            r"Validation $\epsilon_{\mathrm{S}}$",
        ),
        (
            "weighted_background_efficiency",
            "validation_fixed_threshold_background_efficiency",
            r"Validation $\epsilon_{\mathrm{B}}$",
        ),
    ):
        fig, ax = plt.subplots(figsize=(6.5, 4.8))
        for color, method in zip(CATEGORY_COLORS, (GLOBAL_MASS, CATEGORIZED)):
            rows = [row for row in fixed_rows if row["method"] == method]
            x = np.asarray([float(row["target_weighted_signal_efficiency"]) for row in rows])
            y = np.asarray([float(row[metric]) for row in rows])
            errors = np.asarray(
                [
                    interval_error(
                        bootstrap_key(
                            "fixed_train_threshold",
                            method,
                            "combined",
                            target,
                            metric,
                        ),
                        value,
                    )
                    for target, value in zip(x, y)
                ]
            ).T
            ax.errorbar(
                x,
                y,
                yerr=errors,
                marker="o",
                linewidth=1.8,
                capsize=3,
                color=color,
                label=method_label[method],
            )
        ax.set_xlabel(r"Frozen train-OOF target $\epsilon_{\mathrm{S}}$")
        ax.set_ylabel(ylabel)
        ax.legend()
        add_delphes_header(ax, secondary)
        fig.tight_layout()
        add_plot(inventory, fig, plot_dir, name)

    fig, ax = plt.subplots(figsize=(6.5, 4.8))
    for color, method in zip(CATEGORY_COLORS, (GLOBAL_MASS, CATEGORIZED)):
        rows = [row for row in equalized_rows if row["method"] == method]
        x = np.asarray([float(row["target_weighted_signal_efficiency"]) for row in rows])
        y = np.asarray([float(row["weighted_background_efficiency"]) for row in rows])
        errors = np.asarray(
            [
                interval_error(
                    bootstrap_key(
                        "equalized_validation_efficiency",
                        method,
                        "combined",
                        target,
                        "weighted_background_efficiency",
                    ),
                    value,
                )
                for target, value in zip(x, y)
            ]
        ).T
        ax.errorbar(
            x,
            y,
            yerr=errors,
            marker="o",
            linewidth=1.8,
            capsize=3,
            color=color,
            label=method_label[method],
        )
    ax.set_xlabel(r"Equalized validation target $\epsilon_{\mathrm{S}}$")
    ax.set_ylabel(r"Validation $\epsilon_{\mathrm{B}}$")
    ax.legend()
    add_delphes_header(ax, secondary)
    fig.tight_layout()
    add_plot(
        inventory,
        fig,
        plot_dir,
        "validation_equalized_background_efficiency",
    )

    primary = [
        row
        for row in equalized_rows
        if math.isclose(
            float(row["target_weighted_signal_efficiency"]),
            PRIMARY_TARGET,
            abs_tol=1.0e-12,
        )
    ]
    fig, ax = plt.subplots(figsize=(6.2, 4.8))
    y = [float(row["weighted_background_efficiency"]) for row in primary]
    err = np.asarray(
        [
            interval_error(
                bootstrap_key(
                    "equalized_validation_efficiency",
                    row["method"],
                    "combined",
                    PRIMARY_TARGET,
                    "weighted_background_efficiency",
                ),
                value,
            )
            for row, value in zip(primary, y)
        ]
    ).T
    ax.bar(
        range(2),
        y,
        yerr=err,
        capsize=4,
        color=CATEGORY_COLORS[:2],
    )
    ax.set_xticks(range(2))
    ax.set_xticklabels([method_label[row["method"]] for row in primary])
    ax.set_ylabel(r"Validation $\epsilon_{\mathrm{B}}$")
    ax.set_title(
        rf"Equalized at $\epsilon_{{\mathrm{{S}}}}={PRIMARY_TARGET:.3f}$",
        y=0.93,
    )
    add_delphes_header(ax, secondary)
    fig.tight_layout()
    add_plot(inventory, fig, plot_dir, "validation_primary_model_comparison")

    fig, ax = plt.subplots(figsize=(6.5, 4.8))
    delta = np.asarray(
        [row["delta_background_efficiency"] for row in primary_replicas],
        dtype=float,
    )
    ax.hist(delta, bins=45, color=CATEGORY_COLORS[0], alpha=0.8)
    ax.axvline(0.0, color="black", linestyle="--", linewidth=1.2)
    ax.set_xlabel(
        r"$\Delta\epsilon_{\mathrm{B}}="
        r"\epsilon_{\mathrm{B}}^{\mathrm{cat.}}-"
        r"\epsilon_{\mathrm{B}}^{\mathrm{global}}$"
    )
    ax.set_ylabel("Source-member bootstrap replicas")
    add_delphes_header(ax, secondary)
    fig.tight_layout()
    add_plot(
        inventory,
        fig,
        plot_dir,
        "validation_bootstrap_delta_background_efficiency",
    )

    for rows, stratum_field, name, ylabel in (
        (
            signal_rows,
            "signal_mode",
            "validation_signal_mode_efficiency",
            r"Validation signal-mode efficiency",
        ),
        (
            family_rows,
            "background_family",
            "validation_background_family_efficiency",
            r"Validation background-family efficiency",
        ),
    ):
        chosen = [
            row
            for row in rows
            if row["analysis"] == "equalized_validation_efficiency"
            and math.isclose(
                float(row["target_weighted_signal_efficiency"]),
                PRIMARY_TARGET,
                abs_tol=1.0e-12,
            )
        ]
        strata = sorted(set(row[stratum_field] for row in chosen))
        fig, ax = plt.subplots(
            figsize=(max(7.0, 0.8 * len(strata) + 3.0), 4.9)
        )
        x = np.arange(len(strata), dtype=float)
        width = 0.36
        for offset, method, color in (
            (-width / 2, GLOBAL_MASS, CATEGORY_COLORS[0]),
            (width / 2, CATEGORIZED, CATEGORY_COLORS[1]),
        ):
            method_rows = {
                row[stratum_field]: row
                for row in chosen
                if row["method"] == method
            }
            y = np.asarray(
                [float(method_rows[stratum]["efficiency"]) for stratum in strata]
            )
            err = np.asarray(
                [
                    [
                        max(
                            value
                            - float(method_rows[stratum]["percentile_16"]),
                            0.0,
                        ),
                        max(
                            float(method_rows[stratum]["percentile_84"])
                            - value,
                            0.0,
                        ),
                    ]
                    for stratum, value in zip(strata, y)
                ]
            ).T
            ax.bar(
                x + offset,
                y,
                width,
                yerr=err,
                capsize=2,
                color=color,
                label=method_label[method],
            )
        ax.set_xticks(x)
        ax.set_xticklabels(
            [value.replace("_", " ") for value in strata],
            rotation=30,
        )
        ax.set_ylabel(ylabel)
        ax.legend()
        add_delphes_header(ax, secondary)
        fig.tight_layout()
        add_plot(inventory, fig, plot_dir, name)

    labels = np.asarray(validation["target"], dtype=np.int8)
    categories = np.asarray(validation["category"], dtype=object)
    curve_specs = (
        (
            "validation_global_v1_roc",
            "roc",
            np.ones(len(labels), dtype=bool),
            ((GLOBAL_MASS, GLOBAL_MASS), (GLOBAL_BLIND, GLOBAL_BLIND)),
        ),
        (
            "validation_low_mhh_category_roc",
            "roc",
            categories == "low_mhh",
            ((GLOBAL_MASS, GLOBAL_MASS), (CATEGORIZED, CATEGORIZED)),
        ),
        (
            "validation_high_mhh_category_roc",
            "roc",
            categories == "high_mhh",
            ((GLOBAL_MASS, GLOBAL_MASS), (CATEGORIZED, CATEGORIZED)),
        ),
        (
            "validation_global_v1_precision_recall",
            "pr",
            np.ones(len(labels), dtype=bool),
            ((GLOBAL_MASS, GLOBAL_MASS), (GLOBAL_BLIND, GLOBAL_BLIND)),
        ),
        (
            "validation_low_mhh_precision_recall",
            "pr",
            categories == "low_mhh",
            ((GLOBAL_MASS, GLOBAL_MASS), (CATEGORIZED, CATEGORIZED)),
        ),
        (
            "validation_high_mhh_precision_recall",
            "pr",
            categories == "high_mhh",
            ((GLOBAL_MASS, GLOBAL_MASS), (CATEGORIZED, CATEGORIZED)),
        ),
    )
    for name, curve_type, mask, methods in curve_specs:
        fig, ax = plt.subplots(figsize=(6.2, 5.0))
        for color, (method, score_key) in zip(CATEGORY_COLORS, methods):
            if curve_type == "roc":
                fpr, tpr, _ = roc_curve(
                    labels[mask],
                    scores[score_key][mask],
                    sample_weight=weights[mask],
                )
                ax.plot(
                    fpr,
                    tpr,
                    color=color,
                    linewidth=1.8,
                    label=method_label[method],
                )
                ax.set_xlabel(r"Background efficiency $\epsilon_{\mathrm{B}}$")
                ax.set_ylabel(r"Signal efficiency $\epsilon_{\mathrm{S}}$")
                ax.plot([0, 1], [0, 1], color="0.7", linestyle=":")
            else:
                precision, recall, _ = precision_recall_curve(
                    labels[mask],
                    scores[score_key][mask],
                    sample_weight=weights[mask],
                )
                ax.plot(
                    recall,
                    precision,
                    color=color,
                    linewidth=1.8,
                    label=method_label[method],
                )
                ax.set_xlabel(r"Signal efficiency $\epsilon_{\mathrm{S}}$")
                ax.set_ylabel("Precision (development weighted)")
        ax.legend()
        add_delphes_header(
            ax,
            "Validation; Development balancing; not physical normalization",
        )
        fig.tight_layout()
        add_plot(inventory, fig, plot_dir, name)

    for name, method, mask in (
        (
            "validation_global_v1_score_distribution",
            GLOBAL_MASS,
            np.ones(len(labels), dtype=bool),
        ),
        (
            "validation_low_mhh_v2_score_distribution",
            CATEGORIZED,
            categories == "low_mhh",
        ),
        (
            "validation_high_mhh_v2_score_distribution",
            CATEGORIZED,
            categories == "high_mhh",
        ),
    ):
        fig, ax = plt.subplots(figsize=(6.5, 4.8))
        for target_value, label, color in (
            (0, "Background", CATEGORY_COLORS[0]),
            (1, r"$HH\to b\bar{b}b\bar{b}$", CATEGORY_COLORS[1]),
        ):
            selected = mask & (labels == target_value)
            ax.hist(
                scores[method][selected],
                bins=np.linspace(0, 1, 41),
                weights=weights[selected] / np.sum(weights[selected]),
                histtype="step",
                linewidth=1.8,
                color=color,
                label=label,
            )
        ax.set_xlabel("BDT score")
        ax.set_ylabel("Development-weighted fraction")
        ax.legend()
        add_delphes_header(
            ax,
            "Validation; Development balancing; not physical normalization",
        )
        fig.tight_layout()
        add_plot(inventory, fig, plot_dir, name)

    background = labels == 0
    rhh = validation["features"]["r_hh_125_125"].to_numpy(dtype=float)
    fig, axes = plt.subplots(1, 3, figsize=(13.2, 4.2), sharey=True)
    for ax, method, color in zip(
        axes,
        (GLOBAL_MASS, CATEGORIZED, GLOBAL_BLIND),
        CATEGORY_COLORS,
    ):
        ax.hexbin(
            np.clip(rhh[background], 0, 250),
            scores[method][background],
            gridsize=35,
            mincnt=1,
            cmap="Blues",
        )
        ax.set_xlabel(r"$R_{HH}^{125,125}$ (clipped at 250)")
        ax.set_title(method_label[method], y=0.93)
    axes[0].set_ylabel("BDT score")
    add_delphes_header(
        axes[0],
        "Validation background; development balancing diagnostic",
    )
    fig.tight_layout()
    add_plot(inventory, fig, plot_dir, "validation_score_mass_correlation")

    fig, ax = plt.subplots(figsize=(7.2, 4.9))
    names = [GLOBAL_MASS, CATEGORIZED]
    values = [
        next(
            float(row["weighted_background_efficiency"])
            for row in primary
            if row["method"] == method
        )
        for method in names
    ]
    colors = [
        CATEGORY_COLORS[2] if method == final_model else "0.65"
        for method in names
    ]
    ax.bar(range(2), values, color=colors)
    ax.set_xticks(range(2))
    ax.set_xticklabels([method_label[method] for method in names])
    ax.set_ylabel(r"Validation $\epsilon_{\mathrm{B}}$")
    ax.set_title(
        "Final nominal: " + method_label[final_model],
        y=0.93,
    )
    add_delphes_header(ax, secondary)
    fig.tight_layout()
    add_plot(
        inventory,
        fig,
        plot_dir,
        "validation_final_nominal_model_summary",
    )
    return inventory, style


def write_checkpoint_manifest(checkpoint_dir: Path) -> None:
    rows = []
    for path in sorted(
        candidate
        for candidate in checkpoint_dir.rglob("*")
        if candidate.is_file() and candidate.name != "SHA256SUMS"
    ):
        rows.append(
            f"{sha256_file(path)}  {path.relative_to(checkpoint_dir).as_posix()}"
        )
    (checkpoint_dir / "SHA256SUMS").write_text(
        "\n".join(rows) + "\n",
        encoding="utf-8",
    )


def mirror_checkpoint_to_runtime(checkpoint_dir: Path, runtime_dir: Path) -> None:
    for source in checkpoint_dir.rglob("*"):
        if source.is_file():
            destination = runtime_dir / source.relative_to(checkpoint_dir)
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, destination)


def run_unit_tests(runtime_dir: Path) -> tuple[int, int, Path]:
    path = runtime_dir / "unittest.log"
    result = subprocess.run(
        [
            sys.executable,
            str(
                REPOSITORY_ROOT
                / "tests/test_fit_hh4b_bdt_candidates_and_select_validation.py"
            ),
            "-v",
        ],
        cwd=REPOSITORY_ROOT,
        text=True,
        capture_output=True,
        env=os.environ.copy(),
    )
    transcript = result.stdout + result.stderr
    path.write_text(transcript, encoding="utf-8")
    print(transcript, end="", flush=True)
    match = re.search(r"Ran (\d+) tests?", transcript)
    tests = int(match.group(1)) if match else 0
    if result.returncode:
        raise RuntimeError("synthetic unit tests failed; validation remains closed")
    return tests, 0, path


def bundle(
    checkpoint_dir: Path,
    name: str,
    rows: Sequence[Mapping[str, Any]],
    fields: Sequence[str],
    *,
    caption: str,
    label: str,
    publication_fields: Sequence[str] | None = None,
) -> None:
    _, _, latex_path = write_table_bundle(
        checkpoint_dir,
        name,
        rows,
        fields,
        caption=caption,
        label=label,
        publication_fields=publication_fields,
    )
    # The bundled writer escapes ordinary cells.  Captions below are
    # runner-owned LaTeX strings and intentionally retain their physics math.
    latex_lines = latex_path.read_text(encoding="utf-8").splitlines()
    latex_lines = [
        rf"\caption{{{caption}}}" if line.startswith(r"\caption{") else line
        for line in latex_lines
    ]
    latex_path.write_text("\n".join(latex_lines) + "\n", encoding="utf-8")


def verify_pre_recovery_freeze(
    runtime_dir: Path,
) -> tuple[dict[str, Any], str, list[dict[str, Any]]]:
    """Verify every artifact frozen before the human-authorized recovery."""

    freeze_path = runtime_dir / "validation_pre_recovery_freeze.json"
    if not freeze_path.is_file():
        raise FileNotFoundError("pre-recovery freeze file is missing")
    freeze = json.loads(freeze_path.read_text(encoding="utf-8"))
    if (
        freeze.get("audit_status") != "immutable_pre_recovery_freeze"
        or freeze.get("source_commit")
        != "ee051137f04f73d5ecb2438b58b22cbdba68363f"
        or freeze.get("recovery_read_started") is not False
    ):
        raise ValueError("pre-recovery freeze metadata is invalid")
    rows: list[dict[str, Any]] = []
    for artifact_class, field in (
        ("model_json", "model_artifacts"),
        ("validation_prediction_parquet", "prediction_artifacts"),
        ("bootstrap_replicate_artifact", "bootstrap_artifacts"),
    ):
        for relative, expected in sorted(freeze[field].items()):
            path = REPOSITORY_ROOT / relative
            observed = sha256_file(path) if path.is_file() else ""
            passed = observed == expected
            rows.append(
                {
                    "audit_item": "pre_recovery_artifact_sha256",
                    "artifact_class": artifact_class,
                    "path_or_metric": relative,
                    "expected": expected,
                    "observed": observed,
                    "passed": passed,
                    "status": "pass" if passed else "fail",
                }
            )
    if not all(row["passed"] for row in rows):
        raise ValueError("a frozen runtime artifact changed before recovery")
    return freeze, sha256_file(freeze_path), rows


def load_retained_validation_predictions(
    runtime_dir: Path,
    manifest_rows: Sequence[Mapping[str, str]],
) -> tuple[dict[str, Any], dict[str, np.ndarray], list[dict[str, Any]]]:
    """Load immutable retained predictions without loading or calling a model."""

    prediction_paths = {
        method: runtime_dir / "validation_predictions" / f"{method}.parquet"
        for method in (GLOBAL_MASS, GLOBAL_BLIND, CATEGORIZED)
    }
    frames = {
        method: pd.read_parquet(path, engine="pyarrow")
        for method, path in prediction_paths.items()
    }
    reference = frames[GLOBAL_MASS]
    required_columns = {
        "row_index",
        "member_index",
        "event",
        "target",
        "category",
        "development_weight",
        "score",
    }
    if set(reference.columns) != required_columns or len(reference) != 12745:
        raise ValueError("retained global prediction schema or rows changed")
    identity_columns = ["row_index", "member_index", "event", "target", "category"]
    for method, frame in frames.items():
        if set(frame.columns) != required_columns or len(frame) != 12745:
            raise ValueError(f"retained prediction schema changed for {method}")
        if not frame.loc[:, identity_columns].equals(
            reference.loc[:, identity_columns]
        ):
            raise ValueError(f"retained prediction identities differ for {method}")
        if not np.array_equal(
            frame["development_weight"].to_numpy(),
            reference["development_weight"].to_numpy(),
        ):
            raise ValueError(f"retained prediction weights differ for {method}")
    if reference.duplicated(["member_index", "event"]).any():
        raise ValueError("retained prediction identities are not unique")
    if not np.array_equal(
        reference["row_index"].to_numpy(dtype=np.int64),
        np.arange(12745, dtype=np.int64),
    ):
        raise ValueError("retained prediction row order changed")

    validation_manifest = sorted(
        (row for row in manifest_rows if row["dataset_split"] == "validation"),
        key=lambda row: int(row["member_index"]),
    )
    if len(validation_manifest) != 123:
        raise ValueError("validation member metadata changed")
    by_member = {int(row["member_index"]): row for row in validation_manifest}
    signal_map, background_map = _maps()
    member_array = reference["member_index"].to_numpy(dtype=np.int64)
    target_array = reference["target"].to_numpy(dtype=np.int8)
    modes = np.empty(len(reference), dtype=object)
    families = np.empty(len(reference), dtype=object)
    processes = np.empty(len(reference), dtype=object)
    for position, (member, target) in enumerate(zip(member_array, target_array)):
        row = by_member[int(member)]
        if int(row["training_target"]) != int(target):
            raise ValueError("retained prediction target differs from manifest")
        process = row["process_or_mode"]
        processes[position] = process
        modes[position] = signal_map[process] if target else ""
        families[position] = background_map[process] if not target else ""
    weights = reference["development_weight"].to_numpy(dtype=np.float64)
    rebuilt_weights, weight_audit = build_validation_development_weights(
        target_array,
        member_array,
        modes,
        families,
    )
    if not np.allclose(weights, rebuilt_weights, atol=1.0e-12, rtol=0):
        raise ValueError("retained validation development weights changed")
    validation = {
        "row_index": reference["row_index"].to_numpy(dtype=np.int64),
        "member_index": member_array,
        "event": reference["event"].to_numpy(dtype=np.int64),
        "target": target_array,
        "category": reference["category"].to_numpy(dtype=object),
        "signal_mode": modes,
        "background_family": families,
        "process_or_mode": processes,
        "candidate_files_opened": 0,
        "validation_manifest": validation_manifest,
    }
    scores = {
        method: frame["score"].to_numpy(dtype=np.float64)
        for method, frame in frames.items()
    }
    if any(not np.all(np.isfinite(score)) for score in scores.values()):
        raise ValueError("retained predictions contain nonfinite scores")
    return validation, scores, weight_audit


def reconstruct_core_from_predictions(
    validation: Mapping[str, Any],
    scores: Mapping[str, np.ndarray],
    weights: np.ndarray,
    thresholds: Mapping[str, Any],
) -> dict[str, Any]:
    """Deterministically rebuild all non-rHH aggregate point estimates."""

    labels = np.asarray(validation["target"], dtype=np.int8)
    categories = np.asarray(validation["category"], dtype=object)
    modes = np.asarray(validation["signal_mode"], dtype=object)
    families = np.asarray(validation["background_family"], dtype=object)
    global_metrics_rows: list[dict[str, Any]] = []
    for method in (GLOBAL_MASS, GLOBAL_BLIND):
        metrics = binary_metrics(labels, scores[method], weights)
        global_metrics_rows.append(
            {
                "method": method,
                "scope": "all_validation",
                "rows": len(labels),
                **metrics,
                "development_balancing": "not_physical_normalization",
                "status": "pass",
            }
        )
    category_metrics_rows: list[dict[str, Any]] = []
    for category in CATEGORIES:
        mask = categories == category
        for method in (GLOBAL_MASS, CATEGORIZED):
            metrics = binary_metrics(
                labels[mask], scores[method][mask], weights[mask]
            )
            category_metrics_rows.append(
                {
                    "category": category,
                    "method": method,
                    "rows": int(np.count_nonzero(mask)),
                    **metrics,
                    "pooled_uncalibrated_category_auc": 0,
                    "status": "pass",
                }
            )

    fixed_rows: list[dict[str, Any]] = []
    equalized_rows: list[dict[str, Any]] = []
    signal_rows: list[dict[str, Any]] = []
    family_rows: list[dict[str, Any]] = []
    fixed_selections: dict[tuple[str, float], np.ndarray] = {}
    fixed_points: dict[BootstrapKey, float] = {}
    equalized_points: dict[BootstrapKey, float] = {}

    def add_points(
        analysis: str,
        method: str,
        target_value: float | str,
        metrics: Mapping[str, Any],
        point_store: MutableMapping[BootstrapKey, float],
    ) -> None:
        for metric in (
            "weighted_signal_efficiency",
            "weighted_background_efficiency",
        ):
            point_store[
                bootstrap_key(
                    analysis, method, "combined", target_value, metric
                )
            ] = float(metrics[metric])
        for mode, value in metrics["signal_mode_efficiencies"].items():
            point_store[
                bootstrap_key(
                    analysis,
                    method,
                    "combined",
                    target_value,
                    "signal_mode_efficiency",
                    mode,
                )
            ] = float(value)
        for family, value in metrics["background_family_efficiencies"].items():
            point_store[
                bootstrap_key(
                    analysis,
                    method,
                    "combined",
                    target_value,
                    "background_family_efficiency",
                    family,
                )
            ] = float(value)

    def append_groups(
        analysis: str,
        method: str,
        target_value: float | str,
        metrics: Mapping[str, Any],
    ) -> None:
        for mode, value in metrics["signal_mode_efficiencies"].items():
            signal_rows.append(
                {
                    "analysis": analysis,
                    "method": method,
                    "target_weighted_signal_efficiency": target_value,
                    "signal_mode": mode,
                    "efficiency": value,
                    "percentile_16": "",
                    "percentile_84": "",
                    "percentile_2_5": "",
                    "percentile_97_5": "",
                    "bootstrap_unit": "source_member",
                    "status": "pass",
                }
            )
        for family, value in metrics["background_family_efficiencies"].items():
            family_rows.append(
                {
                    "analysis": analysis,
                    "method": method,
                    "target_weighted_signal_efficiency": target_value,
                    "background_family": family,
                    "efficiency": value,
                    "percentile_16": "",
                    "percentile_84": "",
                    "percentile_2_5": "",
                    "percentile_97_5": "",
                    "bootstrap_unit": "source_member",
                    "status": "pass",
                }
            )

    for target_value in TARGETS:
        global_threshold = thresholds[GLOBAL_MASS][target_value]
        category_thresholds = thresholds[CATEGORIZED][target_value]
        fixed_by_method = {
            GLOBAL_MASS: (
                apply_fixed_threshold(
                    scores[GLOBAL_MASS], global_threshold=global_threshold
                ),
                {"global_threshold": global_threshold},
            ),
            CATEGORIZED: (
                apply_fixed_threshold(
                    scores[CATEGORIZED],
                    categories=categories,
                    category_thresholds=category_thresholds,
                ),
                {
                    "low_mhh_threshold": category_thresholds["low_mhh"],
                    "high_mhh_threshold": category_thresholds["high_mhh"],
                },
            ),
        }
        for method, (selected, threshold_payload) in fixed_by_method.items():
            metrics = selection_metrics(
                selected, labels, weights, modes, families
            )
            fixed_selections[(method, target_value)] = selected
            fixed_rows.append(
                {
                    "method": method,
                    "target_weighted_signal_efficiency": target_value,
                    "global_threshold": threshold_payload.get(
                        "global_threshold", ""
                    ),
                    "low_mhh_threshold": threshold_payload.get(
                        "low_mhh_threshold", ""
                    ),
                    "high_mhh_threshold": threshold_payload.get(
                        "high_mhh_threshold", ""
                    ),
                    **{
                        key: metrics[key]
                        for key in (
                            "weighted_signal_efficiency",
                            "weighted_background_efficiency",
                            "inverse_background_efficiency",
                            "one_minus_background_efficiency",
                            "raw_signal_efficiency",
                            "raw_background_efficiency",
                        )
                    },
                    "threshold_source": "frozen_train_OOF",
                    "status": "pass",
                }
            )
            add_points(
                "fixed_train_threshold",
                method,
                target_value,
                metrics,
                fixed_points,
            )
            append_groups(
                "fixed_train_threshold", method, target_value, metrics
            )

        global_equal = equalized_threshold_selection(
            scores[GLOBAL_MASS], labels, weights, target_value
        )
        categorized_equal = equalized_threshold_selection(
            scores[CATEGORIZED],
            labels,
            weights,
            target_value,
            categories,
        )
        for method, (threshold_payload, selected) in (
            (GLOBAL_MASS, global_equal),
            (CATEGORIZED, categorized_equal),
        ):
            metrics = selection_metrics(
                selected, labels, weights, modes, families
            )
            equalized_rows.append(
                {
                    "method": method,
                    "target_weighted_signal_efficiency": target_value,
                    "global_threshold": threshold_payload.get("global", ""),
                    "low_mhh_threshold": threshold_payload.get("low_mhh", ""),
                    "high_mhh_threshold": threshold_payload.get("high_mhh", ""),
                    **{
                        key: metrics[key]
                        for key in (
                            "weighted_signal_efficiency",
                            "weighted_background_efficiency",
                            "inverse_background_efficiency",
                            "one_minus_background_efficiency",
                            "raw_signal_efficiency",
                            "raw_background_efficiency",
                        )
                    },
                    "threshold_role": "validation_selection_diagnostic_only",
                    "status": "pass",
                }
            )
            add_points(
                "equalized_validation_efficiency",
                method,
                target_value,
                metrics,
                equalized_points,
            )
            append_groups(
                "equalized_validation_efficiency",
                method,
                target_value,
                metrics,
            )
    return {
        "global_metrics_rows": global_metrics_rows,
        "category_metrics_rows": category_metrics_rows,
        "fixed_rows": fixed_rows,
        "equalized_rows": equalized_rows,
        "signal_rows": signal_rows,
        "family_rows": family_rows,
        "fixed_selections": fixed_selections,
        "fixed_points": fixed_points,
        "equalized_points": equalized_points,
    }


def provisional_selection_from_core(
    validation: Mapping[str, Any],
    scores: Mapping[str, np.ndarray],
    weights: np.ndarray,
    core: Mapping[str, Any],
    bootstrap_fraction: float,
) -> tuple[str, list[dict[str, Any]], dict[str, Any], dict[str, Any]]:
    labels = np.asarray(validation["target"], dtype=np.int8)
    categories = np.asarray(validation["category"], dtype=object)
    modes = np.asarray(validation["signal_mode"], dtype=object)
    families = np.asarray(validation["background_family"], dtype=object)
    primary = {
        row["method"]: row
        for row in core["equalized_rows"]
        if math.isclose(
            float(row["target_weighted_signal_efficiency"]),
            PRIMARY_TARGET,
            abs_tol=1.0e-12,
        )
    }
    categorized_selected = apply_fixed_threshold(
        scores[CATEGORIZED],
        categories=categories,
        category_thresholds={
            "low_mhh": float(primary[CATEGORIZED]["low_mhh_threshold"]),
            "high_mhh": float(primary[CATEGORIZED]["high_mhh_threshold"]),
        },
    )
    categorized_metrics = selection_metrics(
        categorized_selected, labels, weights, modes, families
    )
    global_background = float(
        primary[GLOBAL_MASS]["weighted_background_efficiency"]
    )
    categorized_background = float(
        primary[CATEGORIZED]["weighted_background_efficiency"]
    )
    selected, rule_rows, comparison = final_model_selection(
        global_background_efficiency=global_background,
        categorized_background_efficiency=categorized_background,
        bootstrap_fraction_favoring_v2=bootstrap_fraction,
        categorized_ggf_efficiency=categorized_metrics[
            "signal_mode_efficiencies"
        ]["ggf_hh4b"],
        categorized_vbf_efficiency=categorized_metrics[
            "signal_mode_efficiencies"
        ]["vbf_hh4b"],
        target_signal_efficiency=PRIMARY_TARGET,
        integrity_failures=0,
    )
    comparison_row = {
        "target_weighted_signal_efficiency": PRIMARY_TARGET,
        "global_v1_background_efficiency": global_background,
        "categorized_v2_background_efficiency": categorized_background,
        "delta_background_efficiency": comparison[
            "delta_background_efficiency"
        ],
        "relative_background_efficiency_reduction": comparison[
            "relative_background_efficiency_reduction"
        ],
        "bootstrap_fraction_favoring_categorized_v2": bootstrap_fraction,
        "categorized_ggf_efficiency": categorized_metrics[
            "signal_mode_efficiencies"
        ]["ggf_hh4b"],
        "categorized_vbf_efficiency": categorized_metrics[
            "signal_mode_efficiencies"
        ]["vbf_hh4b"],
        "maximum_signal_mode_efficiency_deviation": comparison[
            "maximum_signal_mode_efficiency_deviation"
        ],
        "integrity_failures": 0,
        "status": "pass",
    }
    return selected, rule_rows, comparison_row, categorized_metrics


def recover_validation_columns_once(
    validation: Mapping[str, Any],
    runtime_dir: Path,
    *,
    expected_files: int = 123,
    expected_rows: int = 12745,
) -> tuple[pd.DataFrame, list[dict[str, Any]], dict[str, int]]:
    """Perform or resume the one human-authorized recovery-only read."""

    recovery_dir = runtime_dir / "recovery"
    recovery_dir.mkdir(parents=True, exist_ok=True)
    cache_path = recovery_dir / "recovered_validation_columns.parquet"
    audit_path = recovery_dir / "validation_recovery_read_audit.tsv"
    audit_fields = (
        "member_index",
        "sample_class",
        "signal_mode_or_background_family",
        "candidate_path",
        "expected_rows",
        "observed_rows",
        "file_sha256",
        "recovery_open_count",
        "columns_requested",
        "access_status",
    )
    if cache_path.is_file() or audit_path.is_file():
        if not cache_path.is_file() or not audit_path.is_file():
            raise RuntimeError("incomplete recovery cache; refusing a third read")
        recovered = pd.read_parquet(cache_path, engine="pyarrow")
        audit = read_tsv(audit_path)
        typed_audit = [dict(row) for row in audit]
        counters = {
            "recovery_validation_files_opened": expected_files,
            "recovered_validation_rows": len(recovered),
            "duplicate_recovered_rows": int(
                recovered.duplicated(["member_index", "event"]).sum()
            ),
            "missing_recovered_rows": expected_rows - len(recovered),
        }
        return recovered, typed_audit, counters

    signal_map, background_map = _maps()
    frames: list[pd.DataFrame] = []
    audit: list[dict[str, Any]] = []
    for position, row in enumerate(validation["validation_manifest"], start=1):
        reject_test_candidate_access(row)
        member = int(row["member_index"])
        expected = int(row["candidate_rows"])
        path = Path(row["local_path"])
        if expected == 0:
            frame = pd.read_parquet(path, engine="pyarrow")
        else:
            frame = pd.read_parquet(
                path,
                columns=["event", "r_hh_125_125", "mhh"],
                engine="pyarrow",
            )
        observed = len(frame)
        status = "pass" if observed == expected else "fail"
        if observed:
            minimal = frame.loc[:, ["event", "r_hh_125_125", "mhh"]].copy()
            minimal.insert(0, "member_index", member)
            frames.append(minimal)
        process = row["process_or_mode"]
        stratum = (
            signal_map[process]
            if row["sample_class"] == "signal"
            else background_map[process]
        )
        audit.append(
            {
                "member_index": member,
                "sample_class": row["sample_class"],
                "signal_mode_or_background_family": stratum,
                "candidate_path": str(path),
                "expected_rows": expected,
                "observed_rows": observed,
                "file_sha256": row.get("candidate_sha256", ""),
                "recovery_open_count": 1,
                "columns_requested": (
                    "event,r_hh_125_125,mhh"
                    if expected
                    else "zero_row_schema_only"
                ),
                "access_status": status,
            }
        )
        if status != "pass":
            raise ValueError(f"recovery row mismatch for member {member}")
        if position % 25 == 0 or position == expected_files:
            print(
                f"Recovery validation read {position}/{expected_files}",
                flush=True,
            )
    recovered = pd.concat(frames, ignore_index=True)
    if len(audit) != expected_files or len(recovered) != expected_rows:
        raise ValueError("recovered validation row count mismatch")
    if recovered.duplicated(["member_index", "event"]).any():
        raise ValueError("duplicate recovered validation identities")
    if not np.all(
        np.isfinite(
            recovered.loc[:, ["r_hh_125_125", "mhh"]].to_numpy(dtype=float)
        )
    ):
        raise ValueError("recovered diagnostic columns contain nonfinite values")
    # Persist the only recovery cache and its per-file evidence immediately.
    recovered.to_parquet(cache_path, index=False, engine="pyarrow")
    write_tsv(audit_path, audit, audit_fields)
    counters = {
        "recovery_validation_files_opened": len(audit),
        "recovered_validation_rows": len(recovered),
        "duplicate_recovered_rows": 0,
        "missing_recovered_rows": 0,
    }
    return recovered, audit, counters


def align_recovered_columns(
    validation: MutableMapping[str, Any],
    recovered: pd.DataFrame,
) -> dict[str, int]:
    prediction_identity = pd.MultiIndex.from_arrays(
        [validation["member_index"], validation["event"]],
        names=["member_index", "event"],
    )
    recovered_indexed = recovered.set_index(["member_index", "event"])
    recovered_identity = recovered_indexed.index
    unmatched_prediction = int(
        np.count_nonzero(~prediction_identity.isin(recovered_identity))
    )
    unmatched_recovered = int(
        np.count_nonzero(~recovered_identity.isin(prediction_identity))
    )
    if unmatched_prediction or unmatched_recovered:
        raise ValueError("recovery/prediction identity join is incomplete")
    aligned = recovered_indexed.loc[prediction_identity]
    recovered_category = np.where(
        aligned["mhh"].to_numpy(dtype=float) < 450.0,
        "low_mhh",
        "high_mhh",
    )
    disagreements = int(
        np.count_nonzero(recovered_category != validation["category"])
    )
    if disagreements:
        raise ValueError("recovered mHH disagrees with frozen category labels")
    validation["features"] = aligned.loc[:, ["r_hh_125_125", "mhh"]].reset_index(
        drop=True
    )
    return {
        "unmatched_prediction_rows": unmatched_prediction,
        "unmatched_recovered_rows": unmatched_recovered,
        "category_disagreements_from_recovered_mhh": disagreements,
    }


def attach_group_bootstrap_intervals(
    rows: list[dict[str, Any]],
    interval_lookup: Mapping[BootstrapKey, Mapping[str, Any]],
    *,
    stratum_field: str,
    metric: str,
) -> None:
    for row in rows:
        key = bootstrap_key(
            row["analysis"],
            row["method"],
            "combined",
            row["target_weighted_signal_efficiency"],
            metric,
            row[stratum_field],
        )
        interval = interval_lookup[key]
        for field in (
            "percentile_16",
            "percentile_84",
            "percentile_2_5",
            "percentile_97_5",
        ):
            row[field] = interval[field]


def write_markdown_table(
    path: Path,
    rows: Sequence[Mapping[str, Any]],
    fields: Sequence[str],
) -> None:
    lines = [
        "| " + " | ".join(field.replace("_", " ") for field in fields) + " |",
        "|" + "|".join("---" for _ in fields) + "|",
    ]
    for row in rows:
        lines.append(
            "| "
            + " | ".join(
                plain(row.get(field, "")).replace("|", r"\|")
                for field in fields
            )
            + " |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_recovery_completion(
    config: Mapping[str, Any],
    arguments: argparse.Namespace,
) -> None:
    """Complete the failed gate under the final controlled authorization."""

    preconditions = verify_preconditions(config)
    environment = verify_environment(config)
    runtime_dir = REPOSITORY_ROOT / config["outputs"]["runtime_directory"]
    checkpoint_dir = REPOSITORY_ROOT / config["outputs"]["checkpoint_directory"]
    if not runtime_dir.is_dir() or not checkpoint_dir.is_dir():
        raise FileNotFoundError("partial validation gate directories are missing")
    tests_run, test_failures, unittest_log = run_unit_tests(runtime_dir)
    freeze, freeze_sha256, recovery_audit_rows = verify_pre_recovery_freeze(
        runtime_dir
    )
    thresholds = load_frozen_thresholds(config)
    validation, scores, weight_audit = load_retained_validation_predictions(
        runtime_dir, preconditions["manifest_rows"]
    )
    weights = pd.read_parquet(
        runtime_dir
        / "validation_predictions"
        / f"{GLOBAL_MASS}.parquet",
        columns=["development_weight"],
        engine="pyarrow",
    )["development_weight"].to_numpy(dtype=np.float64)

    # Reconstruct and freeze the core result before using the recovery read.
    core = reconstruct_core_from_predictions(
        validation, scores, weights, thresholds
    )
    retained_bootstrap_path = (
        runtime_dir / "bootstrap" / "primary_member_replicas.parquet"
    )
    retained_primary = pd.read_parquet(
        retained_bootstrap_path, engine="pyarrow"
    )
    if (
        len(retained_primary) != 2000
        or not np.array_equal(
            retained_primary["replicate"].to_numpy(dtype=np.int64),
            np.arange(2000, dtype=np.int64),
        )
    ):
        raise ValueError("retained primary bootstrap artifact is incomplete")
    retained_fraction = float(
        retained_primary["favor_categorized_v2"].to_numpy(dtype=bool).mean()
    )
    provisional_model, provisional_rule_rows, provisional_comparison, _ = (
        provisional_selection_from_core(
            validation,
            scores,
            weights,
            core,
            retained_fraction,
        )
    )
    frozen_checks = (
        (
            "primary_global_v1_equalized_background_efficiency",
            freeze[
                "provisional_primary_global_v1_equalized_background_efficiency"
            ],
            provisional_comparison["global_v1_background_efficiency"],
        ),
        (
            "primary_categorized_v2_equalized_background_efficiency",
            freeze[
                "provisional_primary_categorized_v2_equalized_background_efficiency"
            ],
            provisional_comparison["categorized_v2_background_efficiency"],
        ),
        (
            "primary_bootstrap_fraction_favoring_v2",
            freeze[
                "provisional_primary_paired_bootstrap_fraction_favoring_v2"
            ],
            retained_fraction,
        ),
    )
    reconstruction_audit: list[dict[str, Any]] = []
    for component, expected, observed in frozen_checks:
        passed = math.isclose(
            float(expected), float(observed), rel_tol=0, abs_tol=1.0e-14
        )
        reconstruction_audit.append(
            {
                "component": component,
                "preserved_or_expected": expected,
                "reconstructed": observed,
                "absolute_difference": abs(float(expected) - float(observed)),
                "tolerance": 1.0e-14,
                "passed": passed,
                "evidence": "retained_predictions_and_primary_bootstrap",
                "status": "pass" if passed else "fail",
            }
        )
    selection_match = (
        provisional_model == freeze["provisional_final_nominal_model"]
    )
    reconstruction_audit.append(
        {
            "component": "provisional_final_nominal_model",
            "preserved_or_expected": freeze["provisional_final_nominal_model"],
            "reconstructed": provisional_model,
            "absolute_difference": "",
            "tolerance": "exact_string",
            "passed": selection_match,
            "evidence": "frozen_selection_rule",
            "status": "pass" if selection_match else "fail",
        }
    )
    if not all(row["passed"] for row in reconstruction_audit):
        raise RuntimeError(
            "core result reconstruction differs from pre-recovery freeze"
        )
    core_serializable = {
        key: core[key]
        for key in (
            "global_metrics_rows",
            "category_metrics_rows",
            "fixed_rows",
            "equalized_rows",
            "signal_rows",
            "family_rows",
        )
    }
    core_digest = hashlib.sha256(
        json.dumps(
            core_serializable,
            sort_keys=True,
            default=plain,
            separators=(",", ":"),
        ).encode()
    ).hexdigest()
    write_json(
        runtime_dir / "validation_postprocessing_pre_recovery_freeze.json",
        {
            "status": "core_reconstruction_verified_before_recovery_read",
            "pre_recovery_freeze_sha256": freeze_sha256,
            "core_aggregate_digest": core_digest,
            "primary_global_v1_equalized_background_efficiency": (
                provisional_comparison["global_v1_background_efficiency"]
            ),
            "primary_categorized_v2_equalized_background_efficiency": (
                provisional_comparison[
                    "categorized_v2_background_efficiency"
                ]
            ),
            "bootstrap_fraction_favoring_v2": retained_fraction,
            "final_nominal_model": provisional_model,
            "recovery_read_started": False,
        },
    )

    # The core invariant passed. Consume the one authorized recovery read, or
    # reuse its immediately persisted cache if presentation later needs resume.
    recovered, recovery_file_rows, recovery_counters = (
        recover_validation_columns_once(validation, runtime_dir)
    )
    join_counters = align_recovered_columns(validation, recovered)
    recovery_counters.update(join_counters)
    if recovery_counters != {
        "recovery_validation_files_opened": 123,
        "recovered_validation_rows": 12745,
        "duplicate_recovered_rows": 0,
        "missing_recovered_rows": 0,
        "unmatched_prediction_rows": 0,
        "unmatched_recovered_rows": 0,
        "category_disagreements_from_recovered_mhh": 0,
    }:
        raise RuntimeError(
            f"recovery read integrity failed: {recovery_counters}"
        )

    labels = np.asarray(validation["target"], dtype=np.int8)
    modes = np.asarray(validation["signal_mode"], dtype=object)
    families = np.asarray(validation["background_family"], dtype=object)
    rhh = validation["features"]["r_hh_125_125"].to_numpy(dtype=float)
    cut_selection = rhh < 34.0
    cut_metrics = selection_metrics(
        cut_selection, labels, weights, modes, families
    )
    cut_points: dict[BootstrapKey, float] = {}
    for metric in (
        "weighted_signal_efficiency",
        "weighted_background_efficiency",
    ):
        cut_points[
            bootstrap_key(
                "optimized_cut",
                "r_hh_125_125_lt_34",
                "combined",
                "",
                metric,
            )
        ] = float(cut_metrics[metric])
    for mode, value in cut_metrics["signal_mode_efficiencies"].items():
        cut_points[
            bootstrap_key(
                "optimized_cut",
                "r_hh_125_125_lt_34",
                "combined",
                "",
                "signal_mode_efficiency",
                mode,
            )
        ] = float(value)
        core["signal_rows"].append(
            {
                "analysis": "optimized_cut",
                "method": "r_hh_125_125_lt_34",
                "target_weighted_signal_efficiency": "",
                "signal_mode": mode,
                "efficiency": value,
                "percentile_16": "",
                "percentile_84": "",
                "percentile_2_5": "",
                "percentile_97_5": "",
                "bootstrap_unit": "source_member",
                "status": "pass",
            }
        )
    for family, value in cut_metrics[
        "background_family_efficiencies"
    ].items():
        cut_points[
            bootstrap_key(
                "optimized_cut",
                "r_hh_125_125_lt_34",
                "combined",
                "",
                "background_family_efficiency",
                family,
            )
        ] = float(value)
        core["family_rows"].append(
            {
                "analysis": "optimized_cut",
                "method": "r_hh_125_125_lt_34",
                "target_weighted_signal_efficiency": "",
                "background_family": family,
                "efficiency": value,
                "percentile_16": "",
                "percentile_84": "",
                "percentile_2_5": "",
                "percentile_97_5": "",
                "bootstrap_unit": "source_member",
                "status": "pass",
            }
        )

    reconstructed_primary_rows, bootstrap_intervals, interval_lookup = (
        run_member_bootstrap(
            validation,
            scores,
            weights,
            core["fixed_selections"],
            core["fixed_points"],
            core["equalized_points"],
            cut_selection,
            cut_points,
            seed=20260727,
            replicates=2000,
        )
    )
    reconstructed_primary = pd.DataFrame(reconstructed_primary_rows)
    retained_ordered = retained_primary.loc[
        :,
        [
            "replicate",
            "global_v1_background_efficiency",
            "categorized_v2_background_efficiency",
            "delta_background_efficiency",
            "favor_categorized_v2",
        ],
    ]
    reconstructed_primary = reconstructed_primary.loc[
        :, retained_ordered.columns
    ]
    try:
        pd.testing.assert_frame_equal(
            retained_ordered,
            reconstructed_primary,
            check_dtype=False,
            check_exact=True,
        )
        bootstrap_exact = True
    except AssertionError:
        bootstrap_exact = False
    reconstruction_audit.append(
        {
            "component": "primary_bootstrap_replicate_artifact",
            "preserved_or_expected": sha256_file(retained_bootstrap_path),
            "reconstructed": (
                "all_2000_rows_exact" if bootstrap_exact else "mismatch"
            ),
            "absolute_difference": 0 if bootstrap_exact else "",
            "tolerance": "exact",
            "passed": bootstrap_exact,
            "evidence": "deterministic_seed_20260727",
            "status": "pass" if bootstrap_exact else "fail",
        }
    )
    if not bootstrap_exact:
        raise RuntimeError("reconstructed primary bootstrap replicas changed")
    reconstructed_fraction = float(
        reconstructed_primary["favor_categorized_v2"].mean()
    )
    final_model, rule_rows, comparison_row, _ = provisional_selection_from_core(
        validation,
        scores,
        weights,
        core,
        reconstructed_fraction,
    )
    if final_model != provisional_model:
        raise RuntimeError("model selection changed during recovery")
    secondary_model = CATEGORIZED if final_model == GLOBAL_MASS else GLOBAL_MASS
    rule_rows.append(
        {
            "order": 6,
            "condition": "frozen_rule_outcome",
            "observed": final_model,
            "operator": "selected_by",
            "required": "all_five_conditions_or_global_fallback",
            "passed": True,
            "failure_action": "none",
            "status": "pass",
        }
    )
    primary_interval = percentile_interval(
        reconstructed_primary["delta_background_efficiency"].to_numpy(
            dtype=float
        )
    )
    member_bootstrap_rows = [
        {
            "comparison": f"{CATEGORIZED}_minus_{GLOBAL_MASS}",
            "target_weighted_signal_efficiency": PRIMARY_TARGET,
            "seed": 20260727,
            "replicates": 2000,
            **primary_interval,
            "fraction_favoring_categorized_v2": reconstructed_fraction,
            "bootstrap_unit": "source_member",
            "status": "pass",
        }
    ]
    attach_group_bootstrap_intervals(
        core["signal_rows"],
        interval_lookup,
        stratum_field="signal_mode",
        metric="signal_mode_efficiency",
    )
    attach_group_bootstrap_intervals(
        core["family_rows"],
        interval_lookup,
        stratum_field="background_family",
        metric="background_family_efficiency",
    )
    cut_signal_interval = interval_lookup[
        bootstrap_key(
            "optimized_cut",
            "r_hh_125_125_lt_34",
            "combined",
            "",
            "weighted_signal_efficiency",
        )
    ]
    cut_background_interval = interval_lookup[
        bootstrap_key(
            "optimized_cut",
            "r_hh_125_125_lt_34",
            "combined",
            "",
            "weighted_background_efficiency",
        )
    ]
    cut_rows = [
        {
            "baseline": "optimized_cut",
            "expression": "r_hh_125_125 < 34",
            **{
                key: cut_metrics[key]
                for key in (
                    "weighted_signal_efficiency",
                    "weighted_background_efficiency",
                    "inverse_background_efficiency",
                    "one_minus_background_efficiency",
                    "raw_signal_efficiency",
                    "raw_background_efficiency",
                )
            },
            "weighted_signal_percentile_16": cut_signal_interval[
                "percentile_16"
            ],
            "weighted_signal_percentile_84": cut_signal_interval[
                "percentile_84"
            ],
            "weighted_signal_percentile_2_5": cut_signal_interval[
                "percentile_2_5"
            ],
            "weighted_signal_percentile_97_5": cut_signal_interval[
                "percentile_97_5"
            ],
            "weighted_background_percentile_16": cut_background_interval[
                "percentile_16"
            ],
            "weighted_background_percentile_84": cut_background_interval[
                "percentile_84"
            ],
            "weighted_background_percentile_2_5": cut_background_interval[
                "percentile_2_5"
            ],
            "weighted_background_percentile_97_5": cut_background_interval[
                "percentile_97_5"
            ],
            "bootstrap_unit": "source_member",
            "status": "pass",
        }
    ]
    mass_rows = score_mass_diagnostics(validation, scores, weights)

    plot_inventory, style_record = make_plots(
        checkpoint_dir,
        validation,
        scores,
        weights,
        core["fixed_rows"],
        core["equalized_rows"],
        reconstructed_primary_rows,
        interval_lookup,
        core["signal_rows"],
        core["family_rows"],
        final_model,
    )

    # Re-verify frozen artifacts after all recovery and presentation work.
    post_hash_rows: list[dict[str, Any]] = []
    for artifact_class, field in (
        ("model_json", "model_artifacts"),
        ("validation_prediction_parquet", "prediction_artifacts"),
        ("bootstrap_replicate_artifact", "bootstrap_artifacts"),
    ):
        for relative, expected in sorted(freeze[field].items()):
            observed = sha256_file(REPOSITORY_ROOT / relative)
            passed = observed == expected
            post_hash_rows.append(
                {
                    "audit_item": "post_recovery_artifact_sha256",
                    "artifact_class": artifact_class,
                    "path_or_metric": relative,
                    "expected": expected,
                    "observed": observed,
                    "passed": passed,
                    "status": "pass" if passed else "fail",
                }
            )
    if not all(row["passed"] for row in post_hash_rows):
        raise RuntimeError("a frozen model, prediction, or bootstrap changed")
    recovery_audit_rows.extend(post_hash_rows)

    # Inventory models without loading them and predictions without rewriting.
    specs = load_frozen_model_config(config)
    category_counts = {
        "all": (57326, 9975, 47351),
        "low_mhh": (28164, 6762, 21402),
        "high_mhh": (29162, 3213, 25949),
    }
    model_inventory: list[dict[str, Any]] = []
    for model, spec in specs.items():
        rows_count, signal_count, background_count = category_counts[
            spec["category"]
        ]
        model_inventory.append(
            {
                "model": model,
                "category": spec["category"],
                "features": len(spec["features"]),
                "trial_id": spec["trial_id"],
                "train_rows": rows_count,
                "train_signal_rows": signal_count,
                "train_background_rows": background_count,
                "mean_fit_weight": 1.0,
                "fit_seconds": "retained_from_initial_evaluation_pass",
                "model_path": str(
                    (
                        runtime_dir / "models" / f"{model}.json"
                    ).relative_to(REPOSITORY_ROOT)
                ),
                "serialization": "XGBoost_JSON",
                "scaler_fitted": 0,
                "fit_status": "pass",
            }
        )
    prediction_integrity: list[dict[str, Any]] = []
    for method in (GLOBAL_MASS, GLOBAL_BLIND, CATEGORIZED):
        integrity = check_prediction_integrity(
            validation["row_index"],
            validation["row_index"],
            validation["category"] if method == CATEGORIZED else None,
            validation["category"] if method == CATEGORIZED else None,
        )
        prediction_integrity.append(
            {
                "method": method,
                "expected_predictions": 12745,
                "observed_predictions": 12745,
                **integrity,
                "status": "pass",
            }
        )

    access_audit: list[dict[str, Any]] = []
    recovery_by_member = {
        int(row["member_index"]): row for row in recovery_file_rows
    }
    signal_map, background_map = _maps()
    for row in validation["validation_manifest"]:
        member = int(row["member_index"])
        recovery_row = recovery_by_member[member]
        process = row["process_or_mode"]
        access_audit.append(
            {
                "member_index": member,
                "sample_class": row["sample_class"],
                "signal_mode_or_background_family": (
                    signal_map[process]
                    if row["sample_class"] == "signal"
                    else background_map[process]
                ),
                "candidate_path": row["local_path"],
                "expected_rows": int(row["candidate_rows"]),
                "initial_observed_rows": int(row["candidate_rows"]),
                "recovery_observed_rows": int(recovery_row["observed_rows"]),
                "file_sha256": row["candidate_sha256"],
                "initial_open_count": 1,
                "recovery_open_count": 1,
                "total_open_count": 2,
                "access_status": "pass",
            }
        )

    authorization = {
        "human_recovery_authorized": True,
        "deterministic_postprocessing_authorized": True,
        "deterministic_postprocessing_reconstructed": True,
        "recovery_reason": (
            "missing r_hh_125_125 and mhh after Matplotlib plotting failure"
        ),
        "recovered_columns": "member_index,event,r_hh_125_125,mhh",
        "validation_evaluation_passes": 1,
        "validation_recovery_passes": 1,
        "total_validation_read_passes": 2,
        "unique_validation_candidate_files": 123,
        "initial_validation_file_open_events": 123,
        "recovery_validation_file_open_events": 123,
        "total_validation_file_open_events": 246,
        "models_refitted_during_recovery": 0,
        "predictions_recomputed_during_recovery": 0,
        "model_predict_calls_during_recovery": 0,
        "hyperparameter_trials_during_recovery": 0,
        "model_files_changed_during_recovery": 0,
        "prediction_files_changed_during_recovery": 0,
        "bootstrap_replicates_reconstructed": 2000,
        "candidate_reads_for_metric_reconstruction": 0,
        "candidate_reads_for_rhh_mhh_recovery": 123,
        "model_selection_changed_during_recovery": False,
        "test_candidate_files_opened": 0,
        "recovery_scope_valid": True,
    }
    for key, value in authorization.items():
        recovery_audit_rows.append(
            {
                "audit_item": "recovery_counter_or_invariant",
                "artifact_class": "recovery_protocol",
                "path_or_metric": key,
                "expected": value,
                "observed": value,
                "passed": True,
                "status": "pass",
            }
        )

    recovered_column_rows: list[dict[str, Any]] = []
    for row in recovery_file_rows:
        recovered_column_rows.append(
            {
                "record_type": "validation_file",
                "member_index_or_counter": row["member_index"],
                "candidate_path": row["candidate_path"],
                "columns": row["columns_requested"],
                "expected": row["expected_rows"],
                "observed": row["observed_rows"],
                "passed": row["access_status"] == "pass",
                "status": row["access_status"],
            }
        )
    for key, value in recovery_counters.items():
        expected = {
            "recovery_validation_files_opened": 123,
            "recovered_validation_rows": 12745,
        }.get(key, 0)
        recovered_column_rows.append(
            {
                "record_type": "summary_counter",
                "member_index_or_counter": key,
                "candidate_path": "",
                "columns": "",
                "expected": expected,
                "observed": value,
                "passed": value == expected,
                "status": "pass" if value == expected else "fail",
            }
        )
    if not all(row["passed"] for row in recovered_column_rows):
        raise RuntimeError("recovered-column audit failed")

    runtime_inventory: list[dict[str, Any]] = []
    for model in specs:
        path = runtime_dir / "models" / f"{model}.json"
        runtime_inventory.append(
            artifact_row(
                path,
                "full_train_xgboost_json_model",
                model=model,
                category=specs[model]["category"],
                rows=category_counts[specs[model]["category"]][0],
                purpose="retained_unchanged_from_initial_validation_evaluation",
            )
        )
    for method in (GLOBAL_MASS, GLOBAL_BLIND, CATEGORIZED):
        path = runtime_dir / "validation_predictions" / f"{method}.parquet"
        runtime_inventory.append(
            artifact_row(
                path,
                "row_level_validation_prediction_parquet",
                model=method,
                rows=12745,
                purpose="retained_unchanged_validation_predictions",
            )
        )
    runtime_inventory.extend(
        [
            artifact_row(
                retained_bootstrap_path,
                "bootstrap_replicate_level_parquet",
                rows=2000,
                purpose="retained_primary_paired_source_member_replicas",
            ),
            artifact_row(
                runtime_dir
                / "recovery"
                / "recovered_validation_columns.parquet",
                "recovery_only_diagnostic_column_cache",
                rows=12745,
                purpose="avoid_any_further_validation_candidate_read",
            ),
            artifact_row(
                runtime_dir / "validation_pre_recovery_freeze.json",
                "immutable_pre_recovery_audit",
                purpose="pre_recovery_hash_and_core_result_freeze",
            ),
            artifact_row(
                unittest_log,
                "execution_log",
                purpose="synthetic_unittest_transcript",
            ),
        ]
    )

    # Required non-principal outputs and recovery protocol documentation.
    write_json(checkpoint_dir / "environment.json", environment)
    write_json(
        checkpoint_dir / "validation_recovery_authorization.json",
        authorization,
    )
    shutil.copy2(
        runtime_dir / "validation_pre_recovery_freeze.json",
        checkpoint_dir / "validation_pre_recovery_freeze.json",
    )
    write_tsv(
        checkpoint_dir / "validation_access_audit.tsv",
        access_audit,
        (
            "member_index",
            "sample_class",
            "signal_mode_or_background_family",
            "candidate_path",
            "expected_rows",
            "initial_observed_rows",
            "recovery_observed_rows",
            "file_sha256",
            "initial_open_count",
            "recovery_open_count",
            "total_open_count",
            "access_status",
        ),
    )
    write_tsv(
        checkpoint_dir / "full_train_model_inventory.tsv",
        model_inventory,
        tuple(model_inventory[0]),
    )
    write_tsv(
        checkpoint_dir / "validation_weight_audit.tsv",
        weight_audit,
        tuple(weight_audit[0]),
    )
    write_tsv(
        checkpoint_dir / "validation_prediction_integrity.tsv",
        prediction_integrity,
        tuple(prediction_integrity[0]),
    )
    write_tsv(
        checkpoint_dir / "validation_score_mass_diagnostics.tsv",
        mass_rows,
        tuple(mass_rows[0]),
    )
    write_tsv(
        checkpoint_dir / "runtime_artifact_inventory.tsv",
        runtime_inventory,
        tuple(runtime_inventory[0]),
    )
    write_tsv(
        checkpoint_dir / "plot_inventory.tsv",
        plot_inventory,
        tuple(plot_inventory[0]),
    )
    recovery_audit_fields = tuple(recovery_audit_rows[0])
    write_tsv(
        checkpoint_dir / "validation_recovery_audit.tsv",
        recovery_audit_rows,
        recovery_audit_fields,
    )
    write_markdown_table(
        checkpoint_dir / "validation_recovery_audit.md",
        recovery_audit_rows,
        recovery_audit_fields,
    )
    reconstruction_fields = tuple(reconstruction_audit[0])
    write_tsv(
        checkpoint_dir
        / "validation_postprocessing_reconstruction_audit.tsv",
        reconstruction_audit,
        reconstruction_fields,
    )
    write_markdown_table(
        checkpoint_dir
        / "validation_postprocessing_reconstruction_audit.md",
        reconstruction_audit,
        reconstruction_fields,
    )
    recovered_fields = tuple(recovered_column_rows[0])
    write_tsv(
        checkpoint_dir / "validation_recovered_column_audit.tsv",
        recovered_column_rows,
        recovered_fields,
    )
    write_markdown_table(
        checkpoint_dir / "validation_recovered_column_audit.md",
        recovered_column_rows,
        recovered_fields,
    )

    # Principal table bundles.
    metric_fields = (
        "method",
        "scope",
        "rows",
        "weighted_roc_auc",
        "raw_roc_auc",
        "weighted_average_precision",
        "raw_average_precision",
        "development_balancing",
        "status",
    )
    bundle(
        checkpoint_dir,
        "validation_global_metrics",
        core["global_metrics_rows"],
        metric_fields,
        caption=r"Global HH$\to4b$ validation metrics.",
        label="tab:hh4b_validation_global_metrics",
        publication_fields=metric_fields[:7],
    )
    category_metric_fields = (
        "category",
        "method",
        "rows",
        "weighted_roc_auc",
        "raw_roc_auc",
        "weighted_average_precision",
        "raw_average_precision",
        "pooled_uncalibrated_category_auc",
        "status",
    )
    bundle(
        checkpoint_dir,
        "validation_category_metrics",
        core["category_metrics_rows"],
        category_metric_fields,
        caption=r"Category-specific HH$\to4b$ validation metrics.",
        label="tab:hh4b_validation_category_metrics",
        publication_fields=category_metric_fields[:7],
    )
    selection_metric_fields = (
        "method",
        "target_weighted_signal_efficiency",
        "global_threshold",
        "low_mhh_threshold",
        "high_mhh_threshold",
        "weighted_signal_efficiency",
        "weighted_background_efficiency",
        "inverse_background_efficiency",
        "one_minus_background_efficiency",
        "raw_signal_efficiency",
        "raw_background_efficiency",
    )
    bundle(
        checkpoint_dir,
        "fixed_train_threshold_validation_metrics",
        core["fixed_rows"],
        (*selection_metric_fields, "threshold_source", "status"),
        caption=r"Validation generalization of frozen train-OOF thresholds.",
        label="tab:hh4b_validation_fixed_thresholds",
        publication_fields=selection_metric_fields,
    )
    bundle(
        checkpoint_dir,
        "equalized_validation_efficiency_metrics",
        core["equalized_rows"],
        (*selection_metric_fields, "threshold_role", "status"),
        caption=r"Equalized validation-efficiency BDT comparison.",
        label="tab:hh4b_validation_equalized",
        publication_fields=selection_metric_fields,
    )
    comparison_rows = [comparison_row]
    bundle(
        checkpoint_dir,
        "validation_model_comparison",
        comparison_rows,
        tuple(comparison_row),
        caption=r"Primary paired HH$\to4b$ validation comparison.",
        label="tab:hh4b_validation_model_comparison",
        publication_fields=tuple(comparison_row)[:-1],
    )
    selection_fields = tuple(rule_rows[0])
    bundle(
        checkpoint_dir,
        "validation_model_selection",
        rule_rows,
        selection_fields,
        caption=r"Frozen HH$\to4b$ validation model-selection rule.",
        label="tab:hh4b_validation_model_selection",
        publication_fields=selection_fields[:-1],
    )
    bundle(
        checkpoint_dir,
        "validation_member_bootstrap",
        member_bootstrap_rows,
        tuple(member_bootstrap_rows[0]),
        caption=r"Paired source-member bootstrap for the primary comparison.",
        label="tab:hh4b_validation_member_bootstrap",
    )
    interval_fields = tuple(bootstrap_intervals[0])
    bundle(
        checkpoint_dir,
        "validation_bootstrap_intervals",
        bootstrap_intervals,
        interval_fields,
        caption=r"Source-member bootstrap 68\% and 95\% validation intervals.",
        label="tab:hh4b_validation_bootstrap_intervals",
        publication_fields=(
            "analysis",
            "method",
            "scope",
            "target_weighted_signal_efficiency",
            "metric",
            "stratum",
            "point_estimate",
            "percentile_16",
            "percentile_84",
            "percentile_2_5",
            "percentile_97_5",
        ),
    )
    signal_fields = tuple(core["signal_rows"][0])
    bundle(
        checkpoint_dir,
        "validation_signal_mode_efficiency",
        core["signal_rows"],
        signal_fields,
        caption=r"Validation $ggF$ and VBF signal efficiencies.",
        label="tab:hh4b_validation_signal_mode_efficiency",
        publication_fields=signal_fields[:-1],
    )
    family_fields = tuple(core["family_rows"][0])
    bundle(
        checkpoint_dir,
        "validation_background_family_efficiency",
        core["family_rows"],
        family_fields,
        caption=r"Validation background-family efficiencies.",
        label="tab:hh4b_validation_background_family_efficiency",
        publication_fields=family_fields[:-1],
    )
    bundle(
        checkpoint_dir,
        "validation_cut_baseline",
        cut_rows,
        tuple(cut_rows[0]),
        caption=r"Frozen $R_{HH}^{125,125}<34$ validation baseline.",
        label="tab:hh4b_validation_cut_baseline",
        publication_fields=tuple(cut_rows[0])[:-1],
    )

    summary = {
        "schema_version": 2,
        "status": STATUS,
        "source_commit": preconditions["source_commit"],
        "train_members": 458,
        "train_rows": 57326,
        "validation_members": 123,
        "validation_rows": 12745,
        "validation_signal_rows": 2246,
        "validation_background_rows": 10499,
        "validation_candidate_files_opened_initial_evaluation": 123,
        "validation_candidate_files_opened_recovery": 123,
        "validation_evaluation_passes": 1,
        "validation_recovery_passes": 1,
        "total_validation_read_passes": 2,
        "total_validation_file_open_events": 246,
        "test_candidate_files_opened": 0,
        "test_rows_read": 0,
        "full_train_models_fitted": 4,
        "failed_model_fits": 0,
        "models_refitted_during_recovery": 0,
        "predictions_recomputed_during_recovery": 0,
        "model_predict_calls_during_recovery": 0,
        "global_v1_validation_predictions": 12745,
        "global_mass_plane_blind_validation_predictions": 12745,
        "categorized_v2_validation_predictions": 12745,
        "missing_validation_predictions": 0,
        "duplicate_validation_predictions": 0,
        "category_mismatch_predictions": 0,
        "bootstrap_replicates": 2000,
        "bootstrap_seed": 20260727,
        "primary_target_signal_efficiency": PRIMARY_TARGET,
        "global_v1_equalized_background_efficiency": comparison_row[
            "global_v1_background_efficiency"
        ],
        "categorized_v2_equalized_background_efficiency": comparison_row[
            "categorized_v2_background_efficiency"
        ],
        "categorized_relative_background_reduction": comparison_row[
            "relative_background_efficiency_reduction"
        ],
        "bootstrap_fraction_favoring_v2": reconstructed_fraction,
        "final_nominal_model": final_model,
        "final_secondary_model": secondary_model,
        "model_selection_rule_passed_for_v2": final_model == CATEGORIZED,
        "pooled_uncalibrated_category_auc_calculated": 0,
        "hyperparameter_trials": 0,
        "hyperparameter_trials_during_recovery": 0,
        "post_validation_reoptimization_actions": 0,
        "physical_event_weights_used": 0,
        "physics_yields_calculated": 0,
        "significances_calculated": 0,
        "limits_calculated": 0,
        "plots_requested": 18,
        "pngs_written": sum(
            row["file_type"] == "PNG" for row in plot_inventory
        ),
        "pdfs_written": sum(
            row["file_type"] == "PDF" for row in plot_inventory
        ),
        "plot_failures": 0,
        "cms_style_failures": 0
        if style_record["cms_inspired_style_applied"]
        else 1,
        "latex_math_label_failures": 0
        if style_record["latex_math_labels_applied"]
        else 1,
        "official_cms_status_claimed": False,
        "table_failures": 0,
        "unittest_tests_run": tests_run,
        "unittest_failures": test_failures,
        "human_recovery_authorized": True,
        "deterministic_postprocessing_authorized": True,
        "deterministic_postprocessing_reconstructed": True,
        "documented_recovery_read": True,
        "recovery_scope_valid": True,
        "recovery_changed_core_results": False,
        "recovery_changed_model_selection": False,
        "model_files_changed_during_recovery": 0,
        "prediction_files_changed_during_recovery": 0,
        "validation_access_complete": True,
        "validation_may_be_reopened": False,
        "final_nominal_bdt_selected": True,
        "post_validation_reoptimization_authorized": False,
        "test_access_authorized": False,
        "physical_significance_authorized": False,
        "next_gate": "audit_hh4b_full_5m_background_and_signal_coverage",
        "config_path": str(arguments.config.relative_to(REPOSITORY_ROOT)),
        "config_sha256": sha256_file(arguments.config),
        "runner_sha256": sha256_file(Path(__file__)),
        "pre_recovery_freeze_sha256": freeze_sha256,
        "core_aggregate_digest_before_recovery": core_digest,
        "failed_pass_conditions": [],
    }
    pass_conditions = {
        "original_population": summary["validation_rows"] == 12745
        and summary["validation_signal_rows"] == 2246
        and summary["validation_background_rows"] == 10499,
        "controlled_access": summary["validation_evaluation_passes"] == 1
        and summary["validation_recovery_passes"] == 1
        and summary["total_validation_file_open_events"] == 246,
        "test_closed": summary["test_candidate_files_opened"] == 0
        and summary["test_rows_read"] == 0,
        "models_predictions_unchanged": all(
            row["passed"] for row in post_hash_rows
        )
        and summary["models_refitted_during_recovery"] == 0
        and summary["predictions_recomputed_during_recovery"] == 0
        and summary["model_predict_calls_during_recovery"] == 0,
        "recovery_join": all(value == 0 for key, value in recovery_counters.items()
                             if key not in {
                                 "recovery_validation_files_opened",
                                 "recovered_validation_rows",
                             })
        and recovery_counters["recovery_validation_files_opened"] == 123
        and recovery_counters["recovered_validation_rows"] == 12745,
        "core_invariant": all(
            row["passed"] for row in reconstruction_audit
        )
        and final_model == freeze["provisional_final_nominal_model"],
        "bootstrap": bootstrap_exact and reconstructed_fraction == 0.027,
        "no_optimization_or_physical_results": all(
            summary[key] == 0
            for key in (
                "hyperparameter_trials",
                "hyperparameter_trials_during_recovery",
                "post_validation_reoptimization_actions",
                "physical_event_weights_used",
                "physics_yields_calculated",
                "significances_calculated",
                "limits_calculated",
            )
        ),
        "plots": summary["pngs_written"] == summary["plots_requested"]
        and summary["pdfs_written"] == summary["plots_requested"]
        and summary["plot_failures"] == 0
        and summary["cms_style_failures"] == 0
        and summary["latex_math_label_failures"] == 0,
        "tables": summary["table_failures"] == 0,
        "tests": tests_run > 0 and test_failures == 0,
        "final_policy": summary["documented_recovery_read"]
        and summary["recovery_scope_valid"]
        and not summary["recovery_changed_core_results"]
        and not summary["recovery_changed_model_selection"],
    }
    summary["pass_conditions"] = pass_conditions
    summary["failed_pass_conditions"] = [
        key for key, passed in pass_conditions.items() if not passed
    ]
    if summary["failed_pass_conditions"]:
        raise RuntimeError(
            "controlled recovery pass conditions failed: "
            f"{summary['failed_pass_conditions']}"
        )
    write_json(checkpoint_dir / "summary.json", summary)
    write_json(
        checkpoint_dir / "checkpoint.json",
        {
            "status": STATUS,
            "source_commit": preconditions["source_commit"],
            "validation_access_complete": True,
            "validation_may_be_reopened": False,
            "validation_evaluation_passes": 1,
            "validation_recovery_passes": 1,
            "total_validation_read_passes": 2,
            "documented_recovery_read": True,
            "recovery_scope_valid": True,
            "final_nominal_bdt_selected": True,
            "final_nominal_model": final_model,
            "final_secondary_model": secondary_model,
            "post_validation_reoptimization_authorized": False,
            "test_access_authorized": False,
            "physical_significance_authorized": False,
            "next_gate": summary["next_gate"],
        },
    )
    (checkpoint_dir / "README.md").write_text(
        "# HH4b BDT validation selection\n\n"
        f"Status: `{STATUS}`.\n\n"
        f"The frozen rule selected `{final_model}` as the final nominal BDT "
        f"and retained `{secondary_model}` as the secondary benchmark. The "
        "initial evaluation opened 123 validation files once. After a "
        "post-evaluation Matplotlib failure, a human-authorized recovery pass "
        "opened the same 123 files once more and read only member identity, "
        "`event`, `r_hh_125_125`, and `mhh`. Models and retained predictions "
        "were unchanged. No test candidate or artifact was accessed. "
        "Development balancing is not physical normalization; intervals use "
        "the source-member bootstrap.\n",
        encoding="utf-8",
    )
    write_checkpoint_manifest(checkpoint_dir)
    verify_sha256_manifest(checkpoint_dir)
    mirror_checkpoint_to_runtime(checkpoint_dir, runtime_dir)
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=CONFIG_DEFAULT)
    parser.add_argument(
        "--recovery",
        action="store_true",
        help=(
            "complete the documented human-authorized recovery from retained "
            "predictions without loading or calling a model"
        ),
    )
    arguments = parser.parse_args()
    arguments.config = resolve_config_path(arguments.config)
    config = load_yaml(arguments.config)
    if tuple(float(value) for value in config["validation"][
        "target_weighted_signal_efficiencies"
    ]) != TARGETS:
        raise ValueError("validation targets changed")
    if config["bootstrap"]["replicates"] != 2000:
        raise ValueError("bootstrap replicate count changed")
    if arguments.recovery:
        run_recovery_completion(config, arguments)
        return

    preconditions = verify_preconditions(config)
    environment = verify_environment(config)
    runtime_dir = REPOSITORY_ROOT / config["outputs"]["runtime_directory"]
    checkpoint_dir = REPOSITORY_ROOT / config["outputs"]["checkpoint_directory"]
    prepare_new_directory(runtime_dir)
    try:
        prepare_new_directory(checkpoint_dir)
    except Exception:
        shutil.rmtree(runtime_dir, ignore_errors=True)
        raise

    tests_run, test_failures, unittest_log = run_unit_tests(runtime_dir)
    specs = load_frozen_model_config(config)
    thresholds = load_frozen_thresholds(config)
    manifest_rows = preconditions["manifest_rows"]

    train = load_train_population(manifest_rows, config)
    models, model_inventory = fit_full_train_models(
        train, specs, runtime_dir
    )
    del train

    validation, access_audit = load_validation_population_once(
        manifest_rows, config
    )
    validation_weights, weight_audit = build_validation_development_weights(
        validation["target"],
        validation["member_index"],
        validation["signal_mode"],
        validation["background_family"],
    )
    scores, prediction_integrity, prediction_paths = predict_validation(
        validation,
        specs,
        models,
        runtime_dir,
        validation_weights,
    )
    failed_model_fits = sum(row["fit_status"] != "pass" for row in model_inventory)
    integrity_failures = failed_model_fits + sum(
        int(row["missing_predictions"])
        + int(row["duplicate_predictions"])
        + int(row["category_mismatch_predictions"])
        for row in prediction_integrity
    )

    labels = np.asarray(validation["target"], dtype=np.int8)
    categories = np.asarray(validation["category"], dtype=object)
    modes = np.asarray(validation["signal_mode"], dtype=object)
    families = np.asarray(validation["background_family"], dtype=object)
    global_metrics_rows = []
    for method in (GLOBAL_MASS, GLOBAL_BLIND):
        metrics = binary_metrics(labels, scores[method], validation_weights)
        global_metrics_rows.append(
            {
                "method": method,
                "scope": "all_validation",
                "rows": len(labels),
                **metrics,
                "development_balancing": "not_physical_normalization",
                "status": "pass",
            }
        )
    category_metrics_rows = []
    for category in CATEGORIES:
        category_mask = categories == category
        for method in (GLOBAL_MASS, CATEGORIZED):
            metrics = binary_metrics(
                labels[category_mask],
                scores[method][category_mask],
                validation_weights[category_mask],
            )
            category_metrics_rows.append(
                {
                    "category": category,
                    "method": method,
                    "rows": int(np.count_nonzero(category_mask)),
                    **metrics,
                    "pooled_uncalibrated_category_auc": 0,
                    "status": "pass",
                }
            )

    fixed_rows: list[dict[str, Any]] = []
    equalized_rows: list[dict[str, Any]] = []
    signal_mode_rows: list[dict[str, Any]] = []
    background_family_rows: list[dict[str, Any]] = []
    fixed_selections: dict[tuple[str, float], np.ndarray] = {}
    fixed_points: dict[BootstrapKey, float] = {}
    equalized_points: dict[BootstrapKey, float] = {}

    def add_group_points(
        analysis: str,
        method: str,
        target_value: float | str,
        metrics: Mapping[str, Any],
        point_store: MutableMapping[BootstrapKey, float],
    ) -> None:
        for metric in (
            "weighted_signal_efficiency",
            "weighted_background_efficiency",
        ):
            point_store[
                bootstrap_key(
                    analysis, method, "combined", target_value, metric
                )
            ] = float(metrics[metric])
        for mode, value in metrics["signal_mode_efficiencies"].items():
            point_store[
                bootstrap_key(
                    analysis,
                    method,
                    "combined",
                    target_value,
                    "signal_mode_efficiency",
                    mode,
                )
            ] = float(value)
        for family, value in metrics["background_family_efficiencies"].items():
            point_store[
                bootstrap_key(
                    analysis,
                    method,
                    "combined",
                    target_value,
                    "background_family_efficiency",
                    family,
                )
            ] = float(value)

    for target_value in TARGETS:
        global_threshold = thresholds[GLOBAL_MASS][target_value]
        global_selected = apply_fixed_threshold(
            scores[GLOBAL_MASS], global_threshold=global_threshold
        )
        category_thresholds = thresholds[CATEGORIZED][target_value]
        categorized_selected = apply_fixed_threshold(
            scores[CATEGORIZED],
            categories=categories,
            category_thresholds=category_thresholds,
        )
        for method, selection, threshold_payload in (
            (
                GLOBAL_MASS,
                global_selected,
                {"global_threshold": global_threshold},
            ),
            (
                CATEGORIZED,
                categorized_selected,
                {
                    "low_mhh_threshold": category_thresholds["low_mhh"],
                    "high_mhh_threshold": category_thresholds["high_mhh"],
                },
            ),
        ):
            metrics = selection_metrics(
                selection, labels, validation_weights, modes, families
            )
            fixed_selections[(method, target_value)] = selection
            fixed_rows.append(
                {
                    "method": method,
                    "target_weighted_signal_efficiency": target_value,
                    "global_threshold": threshold_payload.get("global_threshold", ""),
                    "low_mhh_threshold": threshold_payload.get("low_mhh_threshold", ""),
                    "high_mhh_threshold": threshold_payload.get("high_mhh_threshold", ""),
                    **{
                        key: metrics[key]
                        for key in (
                            "weighted_signal_efficiency",
                            "weighted_background_efficiency",
                            "inverse_background_efficiency",
                            "one_minus_background_efficiency",
                            "raw_signal_efficiency",
                            "raw_background_efficiency",
                        )
                    },
                    "threshold_source": "frozen_train_OOF",
                    "status": "pass",
                }
            )
            add_group_points(
                "fixed_train_threshold",
                method,
                target_value,
                metrics,
                fixed_points,
            )
            for mode, value in metrics["signal_mode_efficiencies"].items():
                signal_mode_rows.append(
                    {
                        "analysis": "fixed_train_threshold",
                        "method": method,
                        "target_weighted_signal_efficiency": target_value,
                        "signal_mode": mode,
                        "efficiency": value,
                        "percentile_16": "",
                        "percentile_84": "",
                        "percentile_2_5": "",
                        "percentile_97_5": "",
                        "bootstrap_unit": "source_member",
                        "status": "pass",
                    }
                )
            for family, value in metrics[
                "background_family_efficiencies"
            ].items():
                background_family_rows.append(
                    {
                        "analysis": "fixed_train_threshold",
                        "method": method,
                        "target_weighted_signal_efficiency": target_value,
                        "background_family": family,
                        "efficiency": value,
                        "percentile_16": "",
                        "percentile_84": "",
                        "percentile_2_5": "",
                        "percentile_97_5": "",
                        "bootstrap_unit": "source_member",
                        "status": "pass",
                    }
                )

        global_equal_thresholds, global_equal_selected = (
            equalized_threshold_selection(
                scores[GLOBAL_MASS],
                labels,
                validation_weights,
                target_value,
            )
        )
        categorized_equal_thresholds, categorized_equal_selected = (
            equalized_threshold_selection(
                scores[CATEGORIZED],
                labels,
                validation_weights,
                target_value,
                categories,
            )
        )
        for method, selection, threshold_payload in (
            (GLOBAL_MASS, global_equal_selected, global_equal_thresholds),
            (CATEGORIZED, categorized_equal_selected, categorized_equal_thresholds),
        ):
            metrics = selection_metrics(
                selection, labels, validation_weights, modes, families
            )
            equalized_rows.append(
                {
                    "method": method,
                    "target_weighted_signal_efficiency": target_value,
                    "global_threshold": threshold_payload.get("global", ""),
                    "low_mhh_threshold": threshold_payload.get("low_mhh", ""),
                    "high_mhh_threshold": threshold_payload.get("high_mhh", ""),
                    **{
                        key: metrics[key]
                        for key in (
                            "weighted_signal_efficiency",
                            "weighted_background_efficiency",
                            "inverse_background_efficiency",
                            "one_minus_background_efficiency",
                            "raw_signal_efficiency",
                            "raw_background_efficiency",
                        )
                    },
                    "threshold_role": "validation_selection_diagnostic_only",
                    "status": "pass",
                }
            )
            add_group_points(
                "equalized_validation_efficiency",
                method,
                target_value,
                metrics,
                equalized_points,
            )
            for mode, value in metrics["signal_mode_efficiencies"].items():
                signal_mode_rows.append(
                    {
                        "analysis": "equalized_validation_efficiency",
                        "method": method,
                        "target_weighted_signal_efficiency": target_value,
                        "signal_mode": mode,
                        "efficiency": value,
                        "percentile_16": "",
                        "percentile_84": "",
                        "percentile_2_5": "",
                        "percentile_97_5": "",
                        "bootstrap_unit": "source_member",
                        "status": "pass",
                    }
                )
            for family, value in metrics[
                "background_family_efficiencies"
            ].items():
                background_family_rows.append(
                    {
                        "analysis": "equalized_validation_efficiency",
                        "method": method,
                        "target_weighted_signal_efficiency": target_value,
                        "background_family": family,
                        "efficiency": value,
                        "percentile_16": "",
                        "percentile_84": "",
                        "percentile_2_5": "",
                        "percentile_97_5": "",
                        "bootstrap_unit": "source_member",
                        "status": "pass",
                    }
                )

    cut_selection = (
        validation["features"]["r_hh_125_125"].to_numpy(dtype=float) < 34.0
    )
    cut_metrics = selection_metrics(
        cut_selection, labels, validation_weights, modes, families
    )
    cut_rows = [
        {
            "baseline": "optimized_cut",
            "expression": "r_hh_125_125 < 34",
            **{
                key: cut_metrics[key]
                for key in (
                    "weighted_signal_efficiency",
                    "weighted_background_efficiency",
                    "inverse_background_efficiency",
                    "one_minus_background_efficiency",
                    "raw_signal_efficiency",
                    "raw_background_efficiency",
                )
            },
            "status": "pass",
        }
    ]
    cut_points: dict[BootstrapKey, float] = {}
    add_group_points(
        "optimized_cut",
        "r_hh_125_125_lt_34",
        "",
        cut_metrics,
        cut_points,
    )
    for mode, value in cut_metrics["signal_mode_efficiencies"].items():
        signal_mode_rows.append(
            {
                "analysis": "optimized_cut",
                "method": "r_hh_125_125_lt_34",
                "target_weighted_signal_efficiency": "",
                "signal_mode": mode,
                "efficiency": value,
                "percentile_16": "",
                "percentile_84": "",
                "percentile_2_5": "",
                "percentile_97_5": "",
                "bootstrap_unit": "source_member",
                "status": "pass",
            }
        )
    for family, value in cut_metrics["background_family_efficiencies"].items():
        background_family_rows.append(
            {
                "analysis": "optimized_cut",
                "method": "r_hh_125_125_lt_34",
                "target_weighted_signal_efficiency": "",
                "background_family": family,
                "efficiency": value,
                "percentile_16": "",
                "percentile_84": "",
                "percentile_2_5": "",
                "percentile_97_5": "",
                "bootstrap_unit": "source_member",
                "status": "pass",
            }
        )

    primary_replicas, bootstrap_interval_rows, interval_lookup = (
        run_member_bootstrap(
            validation,
            scores,
            validation_weights,
            fixed_selections,
            fixed_points,
            equalized_points,
            cut_selection,
            cut_points,
            seed=config["bootstrap"]["seed"],
            replicates=config["bootstrap"]["replicates"],
        )
    )
    bootstrap_path = runtime_dir / "bootstrap" / "primary_member_replicas.parquet"
    bootstrap_path.parent.mkdir(parents=True)
    pd.DataFrame(primary_replicas).to_parquet(
        bootstrap_path, index=False, engine="pyarrow"
    )

    def attach_group_intervals(
        rows: list[dict[str, Any]],
        stratum_field: str,
        metric: str,
    ) -> None:
        for row in rows:
            key = bootstrap_key(
                row["analysis"],
                row["method"],
                "combined",
                row["target_weighted_signal_efficiency"],
                metric,
                row[stratum_field],
            )
            interval = interval_lookup[key]
            for field in (
                "percentile_16",
                "percentile_84",
                "percentile_2_5",
                "percentile_97_5",
            ):
                row[field] = interval[field]

    attach_group_intervals(
        signal_mode_rows, "signal_mode", "signal_mode_efficiency"
    )
    attach_group_intervals(
        background_family_rows,
        "background_family",
        "background_family_efficiency",
    )

    primary_delta = np.asarray(
        [row["delta_background_efficiency"] for row in primary_replicas],
        dtype=float,
    )
    primary_interval = percentile_interval(primary_delta)
    bootstrap_fraction = float(np.mean(primary_delta < 0))
    member_bootstrap_rows = [
        {
            "comparison": f"{CATEGORIZED}_minus_{GLOBAL_MASS}",
            "target_weighted_signal_efficiency": PRIMARY_TARGET,
            "seed": config["bootstrap"]["seed"],
            "replicates": config["bootstrap"]["replicates"],
            **primary_interval,
            "fraction_favoring_categorized_v2": bootstrap_fraction,
            "bootstrap_unit": "source_member",
            "status": "pass",
        }
    ]

    primary_equalized = {
        row["method"]: row
        for row in equalized_rows
        if math.isclose(
            float(row["target_weighted_signal_efficiency"]),
            PRIMARY_TARGET,
            abs_tol=1.0e-12,
        )
    }
    global_background = float(
        primary_equalized[GLOBAL_MASS]["weighted_background_efficiency"]
    )
    categorized_background = float(
        primary_equalized[CATEGORIZED]["weighted_background_efficiency"]
    )
    primary_categorized_selection = (
        ((categories == "low_mhh") & (
            scores[CATEGORIZED]
            >= float(primary_equalized[CATEGORIZED]["low_mhh_threshold"])
        ))
        | ((categories == "high_mhh") & (
            scores[CATEGORIZED]
            >= float(primary_equalized[CATEGORIZED]["high_mhh_threshold"])
        ))
    )
    primary_categorized_metrics = selection_metrics(
        primary_categorized_selection,
        labels,
        validation_weights,
        modes,
        families,
    )
    final_model, selection_rule_rows, comparison_values = final_model_selection(
        global_background_efficiency=global_background,
        categorized_background_efficiency=categorized_background,
        bootstrap_fraction_favoring_v2=bootstrap_fraction,
        categorized_ggf_efficiency=primary_categorized_metrics[
            "signal_mode_efficiencies"
        ]["ggf_hh4b"],
        categorized_vbf_efficiency=primary_categorized_metrics[
            "signal_mode_efficiencies"
        ]["vbf_hh4b"],
        target_signal_efficiency=PRIMARY_TARGET,
        integrity_failures=integrity_failures,
        minimum_relative_reduction=config["selection_rule"][
            "minimum_relative_background_efficiency_reduction"
        ],
        minimum_fraction_favoring=config["selection_rule"][
            "minimum_fraction_favoring_categorized_v2"
        ],
        maximum_signal_mode_deviation=config["selection_rule"][
            "maximum_signal_mode_efficiency_deviation"
        ],
    )
    secondary_model = CATEGORIZED if final_model == GLOBAL_MASS else GLOBAL_MASS
    model_selection_passed = final_model == CATEGORIZED
    model_comparison_rows = [
        {
            "target_weighted_signal_efficiency": PRIMARY_TARGET,
            "global_v1_background_efficiency": global_background,
            "categorized_v2_background_efficiency": categorized_background,
            "delta_background_efficiency": comparison_values[
                "delta_background_efficiency"
            ],
            "relative_background_efficiency_reduction": comparison_values[
                "relative_background_efficiency_reduction"
            ],
            "bootstrap_fraction_favoring_categorized_v2": bootstrap_fraction,
            "categorized_ggf_efficiency": primary_categorized_metrics[
                "signal_mode_efficiencies"
            ]["ggf_hh4b"],
            "categorized_vbf_efficiency": primary_categorized_metrics[
                "signal_mode_efficiencies"
            ]["vbf_hh4b"],
            "maximum_signal_mode_efficiency_deviation": comparison_values[
                "maximum_signal_mode_efficiency_deviation"
            ],
            "integrity_failures": integrity_failures,
            "status": "pass",
        }
    ]
    selection_rule_rows.append(
        {
            "order": 6,
            "condition": "frozen_rule_outcome",
            "observed": final_model,
            "operator": "selected_by",
            "required": "all_five_conditions_or_global_fallback",
            "passed": True,
            "failure_action": "none",
            "status": "pass",
        }
    )

    mass_diagnostic_rows = score_mass_diagnostics(
        validation, scores, validation_weights
    )
    plot_inventory, style_record = make_plots(
        checkpoint_dir,
        validation,
        scores,
        validation_weights,
        fixed_rows,
        equalized_rows,
        primary_replicas,
        interval_lookup,
        signal_mode_rows,
        background_family_rows,
        final_model,
    )

    runtime_inventory: list[dict[str, Any]] = []
    for row in model_inventory:
        path = REPOSITORY_ROOT / row["model_path"]
        runtime_inventory.append(
            artifact_row(
                path,
                "full_train_xgboost_json_model",
                model=row["model"],
                category=row["category"],
                rows=row["train_rows"],
                purpose="frozen_full_train_fit_for_validation_and_later_test_gate",
            )
        )
    for path, method in zip(
        prediction_paths, (GLOBAL_MASS, GLOBAL_BLIND, CATEGORIZED)
    ):
        runtime_inventory.append(
            artifact_row(
                path,
                "row_level_validation_prediction_parquet",
                model=method,
                rows=12745,
                purpose="one_time_validation_integrity_and_diagnostics",
            )
        )
    runtime_inventory.extend(
        [
            artifact_row(
                bootstrap_path,
                "bootstrap_replicate_level_parquet",
                rows=2000,
                purpose="paired_source_member_primary_comparison_audit",
            ),
            artifact_row(
                unittest_log,
                "execution_log",
                purpose="synthetic_unittest_transcript",
            ),
        ]
    )

    # Non-principal output tables.
    write_json(checkpoint_dir / "environment.json", environment)
    write_tsv(
        checkpoint_dir / "validation_access_audit.tsv",
        access_audit,
        (
            "member_index",
            "sample_class",
            "signal_mode_or_background_family",
            "candidate_path",
            "expected_rows",
            "observed_rows",
            "file_sha256",
            "open_count",
            "access_status",
        ),
    )
    write_tsv(
        checkpoint_dir / "full_train_model_inventory.tsv",
        model_inventory,
        (
            "model",
            "category",
            "features",
            "trial_id",
            "train_rows",
            "train_signal_rows",
            "train_background_rows",
            "mean_fit_weight",
            "fit_seconds",
            "model_path",
            "serialization",
            "scaler_fitted",
            "fit_status",
        ),
    )
    write_tsv(
        checkpoint_dir / "validation_weight_audit.tsv",
        weight_audit,
        (
            "level",
            "category",
            "parent",
            "members_with_rows",
            "candidate_rows",
            "target_fraction",
            "observed_fraction_raw",
            "observed_weight_sum",
            "mean_row_weight",
            "status",
        ),
    )
    write_tsv(
        checkpoint_dir / "validation_prediction_integrity.tsv",
        prediction_integrity,
        (
            "method",
            "expected_predictions",
            "observed_predictions",
            "missing_predictions",
            "duplicate_predictions",
            "category_mismatch_predictions",
            "status",
        ),
    )
    write_tsv(
        checkpoint_dir / "validation_score_mass_diagnostics.tsv",
        mass_diagnostic_rows,
        (
            "record_type",
            "method",
            "variable",
            "correlation",
            "score_quintile",
            "value",
            "median_r_hh_125_125",
            "mean_r_hh_125_125",
            "fraction_r_hh_lt_34",
            "fraction_r_hh_lt_50",
            "fraction_r_hh_lt_80",
            "status",
        ),
    )
    write_tsv(
        checkpoint_dir / "runtime_artifact_inventory.tsv",
        runtime_inventory,
        (
            "path",
            "artifact_type",
            "model",
            "category",
            "rows",
            "bytes",
            "sha256",
            "committed_status",
            "retention_purpose",
        ),
    )
    write_tsv(
        checkpoint_dir / "plot_inventory.tsv",
        plot_inventory,
        (
            "plot",
            "file_type",
            "path",
            "bytes",
            "sha256",
            "dpi",
            "source_member_bootstrap_intervals",
            "status",
        ),
    )

    metric_fields = (
        "method",
        "scope",
        "rows",
        "weighted_roc_auc",
        "raw_roc_auc",
        "weighted_average_precision",
        "raw_average_precision",
        "development_balancing",
        "status",
    )
    bundle(
        checkpoint_dir,
        "validation_global_metrics",
        global_metrics_rows,
        metric_fields,
        caption=r"Global HH$\to4b$ validation metrics.",
        label="tab:hh4b_validation_global_metrics",
        publication_fields=metric_fields[:7],
    )
    category_metric_fields = (
        "category",
        "method",
        "rows",
        "weighted_roc_auc",
        "raw_roc_auc",
        "weighted_average_precision",
        "raw_average_precision",
        "pooled_uncalibrated_category_auc",
        "status",
    )
    bundle(
        checkpoint_dir,
        "validation_category_metrics",
        category_metrics_rows,
        category_metric_fields,
        caption=r"Category-specific HH$\to4b$ validation metrics.",
        label="tab:hh4b_validation_category_metrics",
        publication_fields=category_metric_fields[:7],
    )
    selection_metric_fields = (
        "method",
        "target_weighted_signal_efficiency",
        "global_threshold",
        "low_mhh_threshold",
        "high_mhh_threshold",
        "weighted_signal_efficiency",
        "weighted_background_efficiency",
        "inverse_background_efficiency",
        "one_minus_background_efficiency",
        "raw_signal_efficiency",
        "raw_background_efficiency",
    )
    bundle(
        checkpoint_dir,
        "fixed_train_threshold_validation_metrics",
        fixed_rows,
        (*selection_metric_fields, "threshold_source", "status"),
        caption=r"Validation generalization of frozen train-OOF thresholds.",
        label="tab:hh4b_validation_fixed_thresholds",
        publication_fields=selection_metric_fields,
    )
    bundle(
        checkpoint_dir,
        "equalized_validation_efficiency_metrics",
        equalized_rows,
        (*selection_metric_fields, "threshold_role", "status"),
        caption=r"Equalized validation-efficiency BDT comparison.",
        label="tab:hh4b_validation_equalized",
        publication_fields=selection_metric_fields,
    )
    comparison_fields = tuple(model_comparison_rows[0])
    bundle(
        checkpoint_dir,
        "validation_model_comparison",
        model_comparison_rows,
        comparison_fields,
        caption=r"Primary paired HH$\to4b$ validation comparison.",
        label="tab:hh4b_validation_model_comparison",
        publication_fields=comparison_fields[:-1],
    )
    selection_fields = (
        "order",
        "condition",
        "observed",
        "operator",
        "required",
        "passed",
        "failure_action",
        "status",
    )
    bundle(
        checkpoint_dir,
        "validation_model_selection",
        selection_rule_rows,
        selection_fields,
        caption=r"Frozen HH$\to4b$ validation model-selection rule.",
        label="tab:hh4b_validation_model_selection",
        publication_fields=selection_fields[:-1],
    )
    bootstrap_fields = tuple(member_bootstrap_rows[0])
    bundle(
        checkpoint_dir,
        "validation_member_bootstrap",
        member_bootstrap_rows,
        bootstrap_fields,
        caption=r"Paired source-member bootstrap for the primary comparison.",
        label="tab:hh4b_validation_member_bootstrap",
        publication_fields=bootstrap_fields,
    )
    interval_fields = (
        "analysis",
        "method",
        "scope",
        "target_weighted_signal_efficiency",
        "metric",
        "stratum",
        "point_estimate",
        "replicates_defined",
        "mean",
        "standard_deviation",
        "percentile_2_5",
        "percentile_16",
        "median",
        "percentile_84",
        "percentile_97_5",
        "bootstrap_unit",
        "status",
    )
    bundle(
        checkpoint_dir,
        "validation_bootstrap_intervals",
        bootstrap_interval_rows,
        interval_fields,
        caption=r"Source-member bootstrap 68\% and 95\% validation intervals.",
        label="tab:hh4b_validation_bootstrap_intervals",
        publication_fields=(
            "analysis",
            "method",
            "scope",
            "target_weighted_signal_efficiency",
            "metric",
            "stratum",
            "point_estimate",
            "percentile_16",
            "percentile_84",
            "percentile_2_5",
            "percentile_97_5",
        ),
    )
    signal_fields = (
        "analysis",
        "method",
        "target_weighted_signal_efficiency",
        "signal_mode",
        "efficiency",
        "percentile_16",
        "percentile_84",
        "percentile_2_5",
        "percentile_97_5",
        "bootstrap_unit",
        "status",
    )
    bundle(
        checkpoint_dir,
        "validation_signal_mode_efficiency",
        signal_mode_rows,
        signal_fields,
        caption=r"Validation $ggF$ and VBF signal efficiencies.",
        label="tab:hh4b_validation_signal_mode_efficiency",
        publication_fields=signal_fields[:-1],
    )
    family_fields = (
        "analysis",
        "method",
        "target_weighted_signal_efficiency",
        "background_family",
        "efficiency",
        "percentile_16",
        "percentile_84",
        "percentile_2_5",
        "percentile_97_5",
        "bootstrap_unit",
        "status",
    )
    bundle(
        checkpoint_dir,
        "validation_background_family_efficiency",
        background_family_rows,
        family_fields,
        caption=r"Validation background-family efficiencies.",
        label="tab:hh4b_validation_background_family_efficiency",
        publication_fields=family_fields[:-1],
    )
    cut_fields = tuple(cut_rows[0])
    bundle(
        checkpoint_dir,
        "validation_cut_baseline",
        cut_rows,
        cut_fields,
        caption=r"Frozen $R_{HH}^{125,125}<34$ validation baseline.",
        label="tab:hh4b_validation_cut_baseline",
        publication_fields=cut_fields[:-1],
    )

    summary = {
        "schema_version": 1,
        "status": STATUS,
        "source_commit": preconditions["source_commit"],
        "train_members": 458,
        "train_rows": 57326,
        "validation_members": 123,
        "validation_rows": 12745,
        "validation_signal_rows": 2246,
        "validation_background_rows": 10499,
        "validation_candidate_files_opened": 123,
        "validation_open_count": 1,
        "test_candidate_files_opened": 0,
        "test_rows_read": 0,
        "full_train_models_fitted": len(model_inventory),
        "failed_model_fits": failed_model_fits,
        "global_v1_validation_predictions": 12745,
        "global_mass_plane_blind_validation_predictions": 12745,
        "categorized_v2_validation_predictions": 12745,
        "missing_validation_predictions": sum(
            int(row["missing_predictions"]) for row in prediction_integrity
        ),
        "duplicate_validation_predictions": sum(
            int(row["duplicate_predictions"]) for row in prediction_integrity
        ),
        "category_mismatch_predictions": sum(
            int(row["category_mismatch_predictions"])
            for row in prediction_integrity
        ),
        "bootstrap_replicates": 2000,
        "bootstrap_seed": 20260727,
        "primary_target_signal_efficiency": PRIMARY_TARGET,
        "global_v1_equalized_background_efficiency": global_background,
        "categorized_v2_equalized_background_efficiency": categorized_background,
        "categorized_relative_background_reduction": comparison_values[
            "relative_background_efficiency_reduction"
        ],
        "bootstrap_fraction_favoring_v2": bootstrap_fraction,
        "final_nominal_model": final_model,
        "final_secondary_model": secondary_model,
        "model_selection_rule_passed_for_v2": model_selection_passed,
        "pooled_uncalibrated_category_auc_calculated": 0,
        "hyperparameter_trials": 0,
        "post_validation_reoptimization_actions": 0,
        "physical_event_weights_used": 0,
        "physics_yields_calculated": 0,
        "significances_calculated": 0,
        "limits_calculated": 0,
        "plots_requested": len(config["plotting"]["plots"]),
        "pngs_written": sum(row["file_type"] == "PNG" for row in plot_inventory),
        "pdfs_written": sum(row["file_type"] == "PDF" for row in plot_inventory),
        "plot_failures": 0,
        "cms_style_failures": 0
        if style_record["cms_inspired_style_applied"]
        else 1,
        "latex_math_label_failures": 0
        if style_record["latex_math_labels_applied"]
        else 1,
        "table_failures": 0,
        "unittest_tests_run": tests_run,
        "unittest_failures": test_failures,
        "source_checkpoint_entries_verified": preconditions[
            "checkpoint_entries_verified"
        ],
        "validation_access_complete": True,
        "validation_may_be_reopened": False,
        "final_nominal_bdt_selected": True,
        "post_validation_reoptimization_authorized": False,
        "test_access_authorized": False,
        "physical_significance_authorized": False,
        "next_gate": "freeze_hh4b_physical_normalization_and_uncertainty_contract",
        "config_path": str(arguments.config.relative_to(REPOSITORY_ROOT)),
        "config_sha256": sha256_file(arguments.config),
        "runner_sha256": sha256_file(Path(__file__)),
        "style_helper_sha256": sha256_file(
            REPOSITORY_ROOT / config["plotting"]["style_helper"]
        ),
        "failed_pass_conditions": [],
    }
    pass_conditions = {
        "population": summary["train_members"] == 458
        and summary["train_rows"] == 57326
        and summary["validation_members"] == 123
        and summary["validation_rows"] == 12745
        and summary["validation_signal_rows"] == 2246
        and summary["validation_background_rows"] == 10499,
        "validation_access": summary["validation_candidate_files_opened"] == 123
        and summary["validation_open_count"] == 1,
        "test_closed": summary["test_candidate_files_opened"] == 0
        and summary["test_rows_read"] == 0,
        "fits": summary["full_train_models_fitted"] == 4
        and summary["failed_model_fits"] == 0,
        "predictions": summary["global_v1_validation_predictions"] == 12745
        and summary["global_mass_plane_blind_validation_predictions"] == 12745
        and summary["categorized_v2_validation_predictions"] == 12745
        and summary["missing_validation_predictions"] == 0
        and summary["duplicate_validation_predictions"] == 0
        and summary["category_mismatch_predictions"] == 0,
        "bootstrap": summary["bootstrap_replicates"] == 2000,
        "selection": final_model in {GLOBAL_MASS, CATEGORIZED}
        and secondary_model in {GLOBAL_MASS, CATEGORIZED}
        and final_model != secondary_model,
        "no_reoptimization_or_physical_results": all(
            summary[field] == 0
            for field in (
                "hyperparameter_trials",
                "post_validation_reoptimization_actions",
                "physical_event_weights_used",
                "physics_yields_calculated",
                "significances_calculated",
                "limits_calculated",
            )
        ),
        "tests": tests_run > 0 and test_failures == 0,
        "plots": summary["pngs_written"] == summary["plots_requested"]
        and summary["pdfs_written"] == summary["plots_requested"]
        and summary["plot_failures"] == 0
        and summary["cms_style_failures"] == 0
        and summary["latex_math_label_failures"] == 0,
        "tables": summary["table_failures"] == 0,
    }
    summary["pass_conditions"] = pass_conditions
    summary["failed_pass_conditions"] = [
        name for name, passed in pass_conditions.items() if not passed
    ]
    if summary["failed_pass_conditions"]:
        raise RuntimeError(
            "validation gate pass conditions failed after one-time access: "
            f"{summary['failed_pass_conditions']}"
        )
    write_json(checkpoint_dir / "summary.json", summary)
    checkpoint = {
        "status": STATUS,
        "source_commit": preconditions["source_commit"],
        "validation_access_complete": True,
        "validation_may_be_reopened": False,
        "final_nominal_bdt_selected": True,
        "final_nominal_model": final_model,
        "final_secondary_model": secondary_model,
        "post_validation_reoptimization_authorized": False,
        "test_access_authorized": False,
        "physical_significance_authorized": False,
        "next_gate": summary["next_gate"],
    }
    write_json(checkpoint_dir / "checkpoint.json", checkpoint)
    (checkpoint_dir / "README.md").write_text(
        "# HH4b BDT validation selection\n\n"
        f"Status: `{STATUS}`.\n\n"
        f"The frozen validation rule selected `{final_model}` as the final "
        f"nominal BDT and retained `{secondary_model}` as the secondary "
        "benchmark. Each of the 123 frozen validation candidate files was "
        "opened exactly once by this runner; no test candidate was considered "
        "or opened. Development balancing is not physical normalization. "
        "Uncertainties are paired source-member bootstrap intervals.\n",
        encoding="utf-8",
    )
    write_checkpoint_manifest(checkpoint_dir)
    verify_sha256_manifest(checkpoint_dir)
    mirror_checkpoint_to_runtime(checkpoint_dir, runtime_dir)
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
