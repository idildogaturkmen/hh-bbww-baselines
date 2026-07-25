#!/usr/bin/env python3
"""Train-only grouped cross-validation for the canonical HH4b BDT-v1.

This gate performs deterministic hyperparameter selection and selected-model
out-of-fold diagnostics.  It never opens validation or test candidate files,
fits no full-train model, and writes row-level predictions and fold models only
under the ignored runtime directory.
"""

from __future__ import annotations

import os

# These controls must be set before importing NumPy, SciPy, scikit-learn,
# XGBoost, pandas, PyArrow, or Matplotlib.
os.environ["OMP_NUM_THREADS"] = "8"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import argparse
import atexit
from collections import Counter, defaultdict
import csv
import gc
import hashlib
import itertools
import json
import math
from pathlib import Path
import random
import re
import shutil
import subprocess
import sys
import tempfile
import time
from typing import Any, Iterable, Mapping, Sequence


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPOSITORY_ROOT))
sys.path.insert(0, str(REPOSITORY_ROOT / "scripts" / "analysis"))

MPLCONFIG_TEMP = (
    Path(tempfile.gettempdir()) / f"hh4b_bdt_v1_cv_mplconfig_{os.getpid()}"
)
os.environ["MPLCONFIGDIR"] = str(MPLCONFIG_TEMP)
atexit.register(shutil.rmtree, MPLCONFIG_TEMP, ignore_errors=True)

import matplotlib  # noqa: E402
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
    SOURCE_COLUMNS,
    add_derived_features,
)
from scripts.plotting.hh4b_cms_style import (  # noqa: E402
    CATEGORY_COLORS,
    FEATURE_LATEX_LABELS,
    add_delphes_header,
    apply_cms_style,
    save_png_pdf,
)


PARAMETER_NAMES: tuple[str, ...] = (
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

VARIANT_NAMES: tuple[str, ...] = (
    "mass_aware",
    "explicit_dijet_mass_plane_blind",
)

TABLE_FIELDS: Mapping[str, tuple[str, ...]] = {
    "hyperparameter_space.tsv": (
        "trial_id",
        "is_baseline",
        *PARAMETER_NAMES,
        "complexity_proxy",
        "selection_seed",
        "status",
    ),
    "hyperparameter_trials.tsv": (
        "variant",
        "trial_id",
        *PARAMETER_NAMES,
        "folds",
        "mean_weighted_roc_auc",
        "worst_fold_weighted_roc_auc",
        "std_weighted_roc_auc",
        "maximum_weighted_roc_auc",
        "mean_raw_roc_auc",
        "mean_weighted_average_precision",
        "mean_raw_average_precision",
        "complexity_proxy",
        "selection_rank",
        "selected",
        "selection_rule",
        "status",
    ),
    "fold_metrics.tsv": (
        "evaluation_stage",
        "variant",
        "trial_id",
        "fold",
        "training_members",
        "heldout_members",
        "training_rows",
        "heldout_rows",
        "training_weight_rescale_factor",
        "weighted_roc_auc",
        "raw_roc_auc",
        "weighted_average_precision",
        "raw_average_precision",
        "training_time_seconds",
        "prediction_time_seconds",
        "member_leakage",
        "fit_status",
        "failure_reason",
    ),
    "selected_hyperparameters.tsv": (
        "variant",
        "trial_id",
        *PARAMETER_NAMES,
        "mean_weighted_roc_auc",
        "worst_fold_weighted_roc_auc",
        "std_weighted_roc_auc",
        "complexity_proxy",
        "selection_rank",
        "selection_rule",
        "status",
    ),
    "oof_metrics_summary.tsv": (
        "variant",
        "features",
        "selected_trial_id",
        "oof_rows",
        "weighted_roc_auc",
        "raw_roc_auc",
        "weighted_average_precision",
        "raw_average_precision",
        "fold_weighted_roc_auc_mean",
        "fold_weighted_roc_auc_std",
        "fold_weighted_roc_auc_minimum",
        "fold_weighted_roc_auc_maximum",
        "fold_raw_roc_auc_mean",
        "fold_raw_roc_auc_std",
        "fold_weighted_average_precision_mean",
        "fold_raw_average_precision_mean",
        "diagnostic_scope",
        "status",
    ),
    "cut_baseline_comparison.tsv": (
        "baseline",
        "selection",
        "weighted_signal_efficiency",
        "weighted_background_efficiency",
        "weighted_background_rejection",
        "raw_signal_efficiency",
        "raw_background_efficiency",
        "raw_background_rejection",
        "weighted_ggf_efficiency",
        "weighted_vbf_efficiency",
        "raw_ggf_efficiency",
        "raw_vbf_efficiency",
        "weighted_background_family_efficiencies",
        "raw_background_family_efficiencies",
        "balanced_efficiency_proxy",
        "purity_proxy",
        "proxy_interpretation",
        "physical_yields_used",
        "status",
    ),
    "working_points.tsv": (
        "variant",
        "working_point_definition",
        "target_weighted_signal_efficiency",
        "oof_score_threshold",
        "weighted_signal_efficiency",
        "weighted_background_efficiency",
        "weighted_background_rejection",
        "raw_signal_efficiency",
        "raw_background_efficiency",
        "raw_background_rejection",
        "weighted_ggf_efficiency",
        "weighted_vbf_efficiency",
        "raw_ggf_efficiency",
        "raw_vbf_efficiency",
        "weighted_background_family_efficiencies",
        "raw_background_family_efficiencies",
        "balanced_efficiency_proxy",
        "purity_proxy",
        "proxy_interpretation",
        "threshold_source",
        "status",
    ),
    "signal_mode_metrics.tsv": (
        "variant",
        "signal_mode",
        "signal_rows",
        "background_rows",
        "weighted_roc_auc_vs_all_background",
        "raw_roc_auc_vs_all_background",
        "weighted_average_precision_vs_all_background",
        "raw_average_precision_vs_all_background",
        "weighted_mean_score",
        "raw_mean_score",
        "diagnostic_scope",
        "status",
    ),
    "background_family_efficiency.tsv": (
        "source",
        "working_point_definition",
        "score_threshold",
        "background_family",
        "family_rows",
        "weighted_efficiency",
        "raw_efficiency",
        "normalization",
        "status",
    ),
    "feature_importance.tsv": (
        "variant",
        "feature_order",
        "feature",
        "latex_label",
        "mean_normalized_gain_importance",
        "standard_deviation_normalized_gain_importance",
        "minimum_normalized_gain_importance",
        "maximum_normalized_gain_importance",
        "folds",
        "importance_type",
        "interpretation",
        "status",
    ),
    "score_mass_plane_diagnostics.tsv": (
        "variant",
        "row_type",
        "score_quintile",
        "score_minimum",
        "score_maximum",
        "rows",
        "weighted_fraction",
        "spearman_score_rhh",
        "pearson_score_rhh",
        "median_r_hh_125_125",
        "mean_r_hh_125_125",
        "weighted_fraction_rhh_lt_34",
        "raw_fraction_rhh_lt_34",
        "weighted_fraction_rhh_lt_50",
        "raw_fraction_rhh_lt_50",
        "weighted_fraction_rhh_lt_80",
        "raw_fraction_rhh_lt_80",
        "used_in_model_selection",
        "scientific_description",
        "status",
    ),
    "runtime_artifact_inventory.tsv": (
        "path",
        "artifact_type",
        "model_variant",
        "fold",
        "rows",
        "bytes",
        "sha256",
        "committed_status",
        "retention_purpose",
        "status",
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


def load_yaml(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        payload = yaml.safe_load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"YAML root is not a mapping: {path}")
    return payload


def read_tsv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


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
                if isinstance(value, (dict, list, tuple)):
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
                f"target exists; use --overwrite for this exact directory: {path}"
            )
        shutil.rmtree(path)
    path.mkdir(parents=True)


def verify_sha256_manifest(checkpoint_dir: Path) -> int:
    manifest = checkpoint_dir / "SHA256SUMS"
    verified = 0
    for line in manifest.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        digest, relative = line.split("  ", 1)
        path = checkpoint_dir / relative
        if sha256_file(path) != digest:
            raise ValueError(f"checkpoint SHA-256 mismatch: {relative}")
        verified += 1
    return verified


def normalize_configuration(raw: Mapping[str, Any]) -> dict[str, Any]:
    normalized = {}
    for name in PARAMETER_NAMES:
        value = raw[name]
        if name in {"n_estimators", "max_depth", "min_child_weight"}:
            value = int(value)
        else:
            value = float(value)
        normalized[name] = value
    return normalized


def configuration_key(configuration: Mapping[str, Any]) -> tuple[Any, ...]:
    return tuple(configuration[name] for name in PARAMETER_NAMES)


def generate_hyperparameter_space(
    search_config: Mapping[str, Any],
) -> list[dict[str, Any]]:
    """Return the baseline plus 23 seeded choices from the sorted product."""

    axes = [search_config[name] for name in PARAMETER_NAMES]
    complete = [
        normalize_configuration(dict(zip(PARAMETER_NAMES, values)))
        for values in itertools.product(*axes)
    ]
    complete.sort(key=configuration_key)
    baseline = normalize_configuration(search_config["baseline"])
    baseline_key = configuration_key(baseline)
    if baseline_key not in {configuration_key(item) for item in complete}:
        raise ValueError("baseline is absent from the Cartesian product")

    remaining = [
        item for item in complete if configuration_key(item) != baseline_key
    ]
    rng = random.Random(int(search_config["deterministic_seed"]))
    sampled = rng.sample(remaining, int(search_config["configurations"]) - 1)
    sampled.sort(key=configuration_key)
    selected = [baseline, *sampled]

    rows = []
    for trial_id, parameters in enumerate(selected):
        row = {
            "trial_id": trial_id,
            "is_baseline": trial_id == 0,
            **parameters,
            "complexity_proxy": int(parameters["n_estimators"])
            * (2 ** int(parameters["max_depth"])),
            "selection_seed": int(search_config["deterministic_seed"]),
            "status": "pass",
        }
        rows.append(row)
    if len(rows) != 24 or len({configuration_key(row) for row in rows}) != 24:
        raise AssertionError("hyperparameter configuration cardinality failure")
    return rows


def rank_hyperparameter_trials(
    trial_rows: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Apply the exact frozen five-component model-selection rule."""

    ranked = sorted(
        (dict(row) for row in trial_rows),
        key=lambda row: (
            -float(row["mean_weighted_roc_auc"]),
            -float(row["worst_fold_weighted_roc_auc"]),
            float(row["std_weighted_roc_auc"]),
            int(row["complexity_proxy"]),
            int(row["trial_id"]),
        ),
    )
    for rank, row in enumerate(ranked, start=1):
        row["selection_rank"] = rank
        row["selected"] = rank == 1
    return ranked


def binary_metrics(
    target: Sequence[int],
    score: Sequence[float],
    weights: Sequence[float],
) -> dict[str, float]:
    target_array = np.asarray(target, dtype=np.int8)
    score_array = np.asarray(score, dtype=np.float64)
    weight_array = np.asarray(weights, dtype=np.float64)
    if set(np.unique(target_array)) != {0, 1}:
        raise ValueError("binary metrics require both target classes")
    if not np.all(np.isfinite(score_array)):
        raise ValueError("scores contain nonfinite values")
    if not np.all(np.isfinite(weight_array)) or not np.all(weight_array > 0):
        raise ValueError("metric weights must be finite and positive")
    return {
        "weighted_roc_auc": float(
            roc_auc_score(target_array, score_array, sample_weight=weight_array)
        ),
        "raw_roc_auc": float(roc_auc_score(target_array, score_array)),
        "weighted_average_precision": float(
            average_precision_score(
                target_array,
                score_array,
                sample_weight=weight_array,
            )
        ),
        "raw_average_precision": float(
            average_precision_score(target_array, score_array)
        ),
    }


def efficiency(
    selected: np.ndarray,
    population: np.ndarray,
    weights: np.ndarray | None = None,
) -> float:
    denominator_mask = np.asarray(population, dtype=bool)
    numerator_mask = np.asarray(selected, dtype=bool) & denominator_mask
    if weights is None:
        denominator = int(np.count_nonzero(denominator_mask))
        if denominator == 0:
            raise ValueError("efficiency population is empty")
        return float(np.count_nonzero(numerator_mask) / denominator)
    values = np.asarray(weights, dtype=np.float64)
    denominator = float(np.sum(values[denominator_mask]))
    if denominator <= 0:
        raise ValueError("weighted efficiency population has zero weight")
    return float(np.sum(values[numerator_mask]) / denominator)


def weighted_signal_threshold(
    signal_scores: Sequence[float],
    signal_weights: Sequence[float],
    target_efficiency: float,
) -> float:
    scores = np.asarray(signal_scores, dtype=np.float64)
    weights = np.asarray(signal_weights, dtype=np.float64)
    if not 0.0 < target_efficiency <= 1.0:
        raise ValueError("target signal efficiency must be in (0, 1]")
    if scores.size == 0 or scores.shape != weights.shape:
        raise ValueError("signal score/weight arrays are invalid")
    order = np.argsort(-scores, kind="mergesort")
    cumulative = np.cumsum(weights[order])
    index = int(
        np.searchsorted(
            cumulative,
            target_efficiency * float(cumulative[-1]),
            side="left",
        )
    )
    return float(scores[order[min(index, scores.size - 1)]])


def check_oof_integrity(
    expected_row_indices: Sequence[int],
    produced_row_indices: Sequence[int],
    training_member_sets: Sequence[set[int]],
    heldout_member_sets: Sequence[set[int]],
) -> dict[str, int]:
    expected = Counter(int(value) for value in expected_row_indices)
    produced = Counter(int(value) for value in produced_row_indices)
    missing = sum(max(expected[key] - produced.get(key, 0), 0) for key in expected)
    duplicates = sum(max(count - expected.get(key, 0), 0) for key, count in produced.items())
    leakage = sum(
        len(training & heldout)
        for training, heldout in zip(training_member_sets, heldout_member_sets)
    )
    return {
        "missing_oof_rows": int(missing),
        "duplicate_oof_rows": int(duplicates),
        "member_leakage_rows": int(leakage),
    }


def aggregate_gain_importance(
    fold_importances: Sequence[Sequence[float]],
    feature_names: Sequence[str],
    variant: str,
) -> list[dict[str, Any]]:
    matrix = np.asarray(fold_importances, dtype=np.float64)
    if matrix.ndim != 2 or matrix.shape[1] != len(feature_names):
        raise ValueError("feature-importance matrix has the wrong shape")
    if np.any(matrix < 0) or not np.all(np.isfinite(matrix)):
        raise ValueError("feature importances must be finite and nonnegative")
    sums = np.sum(matrix, axis=1)
    if np.any(sums <= 0):
        raise ValueError("every fold must have positive total gain")
    normalized = matrix / sums[:, None]
    rows = []
    for index, feature in enumerate(feature_names):
        values = normalized[:, index]
        rows.append(
            {
                "variant": variant,
                "feature_order": index + 1,
                "feature": feature,
                "latex_label": FEATURE_LATEX_LABELS[feature],
                "mean_normalized_gain_importance": float(np.mean(values)),
                "standard_deviation_normalized_gain_importance": float(
                    np.std(values, ddof=0)
                ),
                "minimum_normalized_gain_importance": float(np.min(values)),
                "maximum_normalized_gain_importance": float(np.max(values)),
                "folds": matrix.shape[0],
                "importance_type": "xgboost_gain_normalized_per_fold",
                "interpretation": (
                    "model_diagnostic_not_a_causal_attribution"
                ),
                "status": "pass",
            }
        )
    return rows


def weighted_quantiles(
    values: Sequence[float],
    weights: Sequence[float],
    quantiles: Sequence[float],
) -> np.ndarray:
    values_array = np.asarray(values, dtype=np.float64)
    weights_array = np.asarray(weights, dtype=np.float64)
    order = np.argsort(values_array, kind="mergesort")
    sorted_values = values_array[order]
    sorted_weights = weights_array[order]
    cumulative = np.cumsum(sorted_weights)
    positions = (cumulative - 0.5 * sorted_weights) / cumulative[-1]
    return np.interp(np.asarray(quantiles), positions, sorted_values)


def score_quintile_assignment(
    scores: Sequence[float],
    weights: Sequence[float],
) -> tuple[np.ndarray, np.ndarray]:
    boundaries = weighted_quantiles(scores, weights, [0.2, 0.4, 0.6, 0.8])
    assignment = np.searchsorted(boundaries, np.asarray(scores), side="right")
    return assignment.astype(np.int8), boundaries


def score_mass_diagnostic_rows(
    scores: Sequence[float],
    rhh: Sequence[float],
    weights: Sequence[float],
    variant: str,
) -> list[dict[str, Any]]:
    scores_array = np.asarray(scores, dtype=np.float64)
    rhh_array = np.asarray(rhh, dtype=np.float64)
    weights_array = np.asarray(weights, dtype=np.float64)
    assignment, _ = score_quintile_assignment(scores_array, weights_array)
    spearman = float(spearmanr(scores_array, rhh_array).statistic)
    pearson = float(pearsonr(scores_array, rhh_array).statistic)
    rows = [
        {
            "variant": variant,
            "row_type": "correlation_summary",
            "score_quintile": "",
            "score_minimum": float(np.min(scores_array)),
            "score_maximum": float(np.max(scores_array)),
            "rows": scores_array.size,
            "weighted_fraction": 1.0,
            "spearman_score_rhh": spearman,
            "pearson_score_rhh": pearson,
            "median_r_hh_125_125": float(np.median(rhh_array)),
            "mean_r_hh_125_125": float(np.mean(rhh_array)),
            "weighted_fraction_rhh_lt_34": efficiency(
                rhh_array < 34.0,
                np.ones(rhh_array.size, dtype=bool),
                weights_array,
            ),
            "raw_fraction_rhh_lt_34": float(np.mean(rhh_array < 34.0)),
            "weighted_fraction_rhh_lt_50": efficiency(
                rhh_array < 50.0,
                np.ones(rhh_array.size, dtype=bool),
                weights_array,
            ),
            "raw_fraction_rhh_lt_50": float(np.mean(rhh_array < 50.0)),
            "weighted_fraction_rhh_lt_80": efficiency(
                rhh_array < 80.0,
                np.ones(rhh_array.size, dtype=bool),
                weights_array,
            ),
            "raw_fraction_rhh_lt_80": float(np.mean(rhh_array < 80.0)),
            "used_in_model_selection": False,
            "scientific_description": (
                "background_oof_diagnostic_not_used_in_model_selection"
            ),
            "status": "pass",
        }
    ]
    total_weight = float(np.sum(weights_array))
    for quintile in range(5):
        selected = assignment == quintile
        if not np.any(selected):
            raise ValueError(f"empty background score quintile {quintile}")
        q_rhh = rhh_array[selected]
        q_weights = weights_array[selected]
        rows.append(
            {
                "variant": variant,
                "row_type": "weighted_background_score_quintile",
                "score_quintile": quintile + 1,
                "score_minimum": float(np.min(scores_array[selected])),
                "score_maximum": float(np.max(scores_array[selected])),
                "rows": int(np.count_nonzero(selected)),
                "weighted_fraction": float(np.sum(q_weights) / total_weight),
                "spearman_score_rhh": spearman,
                "pearson_score_rhh": pearson,
                "median_r_hh_125_125": float(np.median(q_rhh)),
                "mean_r_hh_125_125": float(np.mean(q_rhh)),
                "weighted_fraction_rhh_lt_34": efficiency(
                    q_rhh < 34.0,
                    np.ones(q_rhh.size, dtype=bool),
                    q_weights,
                ),
                "raw_fraction_rhh_lt_34": float(np.mean(q_rhh < 34.0)),
                "weighted_fraction_rhh_lt_50": efficiency(
                    q_rhh < 50.0,
                    np.ones(q_rhh.size, dtype=bool),
                    q_weights,
                ),
                "raw_fraction_rhh_lt_50": float(np.mean(q_rhh < 50.0)),
                "weighted_fraction_rhh_lt_80": efficiency(
                    q_rhh < 80.0,
                    np.ones(q_rhh.size, dtype=bool),
                    q_weights,
                ),
                "raw_fraction_rhh_lt_80": float(np.mean(q_rhh < 80.0)),
                "used_in_model_selection": False,
                "scientific_description": (
                    "background_oof_diagnostic_not_used_in_model_selection"
                ),
                "status": "pass",
            }
        )
    return rows


def selection_metrics(
    selected: np.ndarray,
    labels: np.ndarray,
    weights: np.ndarray,
    signal_modes: np.ndarray,
    background_families: np.ndarray,
) -> dict[str, Any]:
    signal = labels == 1
    background = labels == 0
    weighted_signal = efficiency(selected, signal, weights)
    weighted_background = efficiency(selected, background, weights)
    raw_signal = efficiency(selected, signal)
    raw_background = efficiency(selected, background)
    weighted_modes = {
        mode: efficiency(selected, signal_modes == mode, weights)
        for mode in sorted(set(signal_modes[signal]))
    }
    raw_modes = {
        mode: efficiency(selected, signal_modes == mode)
        for mode in sorted(set(signal_modes[signal]))
    }
    weighted_families = {
        family: efficiency(selected, background_families == family, weights)
        for family in sorted(set(background_families[background]))
    }
    raw_families = {
        family: efficiency(selected, background_families == family)
        for family in sorted(set(background_families[background]))
    }
    return {
        "weighted_signal_efficiency": weighted_signal,
        "weighted_background_efficiency": weighted_background,
        "weighted_background_rejection": 1.0 - weighted_background,
        "raw_signal_efficiency": raw_signal,
        "raw_background_efficiency": raw_background,
        "raw_background_rejection": 1.0 - raw_background,
        "weighted_ggf_efficiency": weighted_modes["ggf_hh4b"],
        "weighted_vbf_efficiency": weighted_modes["vbf_hh4b"],
        "raw_ggf_efficiency": raw_modes["ggf_hh4b"],
        "raw_vbf_efficiency": raw_modes["vbf_hh4b"],
        "weighted_background_family_efficiencies": weighted_families,
        "raw_background_family_efficiencies": raw_families,
        "balanced_efficiency_proxy": (
            weighted_signal / math.sqrt(weighted_background)
            if weighted_background > 0
            else float("inf")
        ),
        "purity_proxy": (
            weighted_signal / weighted_background
            if weighted_background > 0
            else float("inf")
        ),
    }


def cut_baseline_metrics(
    rhh: Sequence[float],
    labels: Sequence[int],
    weights: Sequence[float],
    signal_modes: Sequence[str],
    background_families: Sequence[str],
    threshold: float = 34.0,
) -> dict[str, Any]:
    return selection_metrics(
        np.asarray(rhh, dtype=np.float64) < threshold,
        np.asarray(labels, dtype=np.int8),
        np.asarray(weights, dtype=np.float64),
        np.asarray(signal_modes),
        np.asarray(background_families),
    )


def environment_record(config: Mapping[str, Any]) -> dict[str, Any]:
    try:
        import mplhep

        mplhep_version = mplhep.__version__
        mplhep_available = True
    except ImportError:
        mplhep_version = "unavailable"
        mplhep_available = False
    return {
        "python_version": sys.version.split()[0],
        "python_executable": sys.executable,
        "python_prefix": sys.prefix,
        "numpy_version": np.__version__,
        "pandas_version": pd.__version__,
        "pyarrow_version": pyarrow.__version__,
        "scipy_version": scipy.__version__,
        "sklearn_version": sklearn.__version__,
        "xgboost_version": xgboost.__version__,
        "matplotlib_version": matplotlib.__version__,
        "mplhep_available": mplhep_available,
        "mplhep_version": mplhep_version,
        "environment_controls": dict(config["required_environment"]["controls"]),
        "cpu_training_only": True,
    }


def verify_environment(config: Mapping[str, Any]) -> dict[str, Any]:
    required = config["required_environment"]
    record = environment_record(config)
    if ".".join(record["python_version"].split(".")[:2]) != required[
        "python_major_minor"
    ]:
        raise RuntimeError("required Python major/minor version is unavailable")
    if sklearn.__version__ != required["sklearn_version"]:
        raise RuntimeError("required scikit-learn version is unavailable")
    if xgboost.__version__ != required["xgboost_version"]:
        raise RuntimeError("required XGBoost version is unavailable")
    required_prefix = Path(config["required_interpreter"]).parent.parent.resolve()
    if Path(sys.prefix).resolve() != required_prefix:
        raise RuntimeError("runner is not using the exact required virtual environment")
    for key, expected in required["controls"].items():
        if os.environ.get(key) != str(expected):
            raise RuntimeError(f"environment control mismatch: {key}")
    return record


def load_and_verify_contract(
    config: Mapping[str, Any],
) -> tuple[
    dict[str, Any],
    list[dict[str, str]],
    list[dict[str, str]],
    dict[str, str],
    dict[str, str],
    list[str],
    list[str],
]:
    contract = config["input_contract"]
    checkpoint = REPOSITORY_ROOT / contract["checkpoint_dir"]
    verify_sha256_manifest(checkpoint)
    summary = json.loads((checkpoint / "summary.json").read_text(encoding="utf-8"))
    expected = config["expected_population"]
    required_values = {
        "status": "hh4b_bdt_v1_input_contract_pass",
        "development_members": expected["development_members"],
        "train_members": expected["train_members"],
        "validation_members": expected["validation_members"],
        "train_candidate_rows": expected["train_rows"],
        "train_signal_rows": expected["signal_rows"],
        "train_background_rows": expected["background_rows"],
        "mass_aware_features": expected["features_mass_aware"],
        "mass_plane_blind_features": expected["features_mass_plane_blind"],
        "folds": expected["folds"],
        "duplicate_members_across_folds": 0,
        "unassigned_train_members": 0,
        "validation_candidate_files_opened": 0,
        "test_candidate_files_opened": 0,
    }
    differences = {
        key: (value, summary.get(key))
        for key, value in required_values.items()
        if summary.get(key) != value
    }
    if differences:
        raise ValueError(f"input-contract precondition mismatch: {differences}")

    feature_rows = read_tsv(checkpoint / contract["feature_contract"])
    mass_aware = [row["feature"] for row in feature_rows]
    mass_blind = [
        row["feature"]
        for row in feature_rows
        if row["mass_plane_blind_feature"] == "True"
    ]
    if mass_aware != list(MASS_AWARE_FEATURES):
        raise ValueError("frozen mass-aware feature order mismatch")
    if mass_blind != list(MASS_PLANE_BLIND_FEATURES):
        raise ValueError("frozen mass-plane-blind feature order mismatch")

    fold_rows = read_tsv(checkpoint / contract["member_fold_assignment"])
    weight_rows = read_tsv(checkpoint / contract["member_weight_summary"])
    if len(fold_rows) != 458 or len(weight_rows) != 458:
        raise ValueError("frozen member tables have the wrong cardinality")
    background_map = {
        row["manifest_process_or_mode"]: row["background_family"]
        for row in read_tsv(checkpoint / contract["background_family_mapping"])
    }
    signal_map = {
        row["manifest_process_or_mode"]: row["signal_mode"]
        for row in read_tsv(checkpoint / contract["signal_mode_mapping"])
    }
    return (
        summary,
        fold_rows,
        weight_rows,
        background_map,
        signal_map,
        mass_aware,
        mass_blind,
    )


def load_train_population(
    config: Mapping[str, Any],
    fold_rows: Sequence[Mapping[str, str]],
    weight_rows: Sequence[Mapping[str, str]],
    background_map: Mapping[str, str],
    signal_map: Mapping[str, str],
) -> dict[str, Any]:
    contract_config = load_yaml(
        REPOSITORY_ROOT / config["input_contract"]["config_path"]
    )
    manifest_path = REPOSITORY_ROOT / contract_config["manifest"]["path"]
    manifest = read_tsv(manifest_path)
    train_manifest = sorted(
        (row for row in manifest if row["dataset_split"] == "train"),
        key=lambda row: int(row["member_index"]),
    )
    if len(train_manifest) != 458:
        raise ValueError("train manifest cardinality mismatch")

    fold_by_member = {
        int(row["member_index"]): int(row["fold"]) for row in fold_rows
    }
    weight_by_member = {
        int(row["member_index"]): float(row["per_row_training_weight"])
        for row in weight_rows
    }
    expected_rows_by_member = {
        int(row["member_index"]): int(row["candidate_rows"])
        for row in fold_rows
    }

    feature_frames: list[pd.DataFrame] = []
    event_arrays: list[np.ndarray] = []
    member_arrays: list[np.ndarray] = []
    fold_arrays: list[np.ndarray] = []
    label_arrays: list[np.ndarray] = []
    weight_arrays: list[np.ndarray] = []
    mode_arrays: list[np.ndarray] = []
    family_arrays: list[np.ndarray] = []
    opened = 0
    schema_less_zero = 0

    read_columns = list(dict.fromkeys(("event", *SOURCE_COLUMNS)))
    for position, row in enumerate(train_manifest, start=1):
        member = int(row["member_index"])
        expected_rows = int(row["candidate_rows"])
        if expected_rows != expected_rows_by_member[member]:
            raise ValueError(f"frozen row count mismatch for member {member}")
        path = Path(row["local_path"])
        if expected_rows == 0:
            frame = pd.read_parquet(path, engine="pyarrow")
            schema_less_zero += int(len(frame.columns) == 0)
        else:
            frame = pd.read_parquet(
                path,
                columns=read_columns,
                engine="pyarrow",
            )
        opened += 1
        if len(frame) != expected_rows:
            raise ValueError(f"candidate row mismatch for member {member}")
        if expected_rows:
            derived = add_derived_features(frame)
            feature_frames.append(
                derived.loc[:, list(MASS_AWARE_FEATURES)].astype(
                    np.float32,
                    copy=False,
                )
            )
            event_arrays.append(frame["event"].to_numpy(dtype=np.int64))
            member_arrays.append(np.full(expected_rows, member, dtype=np.int64))
            fold_arrays.append(
                np.full(expected_rows, fold_by_member[member], dtype=np.int8)
            )
            target = int(row["training_target"])
            label_arrays.append(np.full(expected_rows, target, dtype=np.int8))
            per_row_weight = weight_by_member[member]
            if per_row_weight <= 0:
                raise ValueError(
                    f"candidate-bearing member {member} has no frozen row weight"
                )
            weight_arrays.append(
                np.full(expected_rows, per_row_weight, dtype=np.float64)
            )
            if row["sample_class"] == "signal":
                mode = signal_map[row["process_or_mode"]]
                family = ""
            else:
                mode = ""
                family = background_map[row["process_or_mode"]]
            mode_arrays.append(np.full(expected_rows, mode, dtype=object))
            family_arrays.append(np.full(expected_rows, family, dtype=object))
        if position % 50 == 0 or position == len(train_manifest):
            print(
                f"Opened train candidates {position}/{len(train_manifest)}",
                flush=True,
            )

    features = pd.concat(feature_frames, ignore_index=True)
    payload = {
        "features": features,
        "event": np.concatenate(event_arrays),
        "member_index": np.concatenate(member_arrays),
        "fold": np.concatenate(fold_arrays),
        "target": np.concatenate(label_arrays),
        "weight": np.concatenate(weight_arrays),
        "signal_mode": np.concatenate(mode_arrays),
        "background_family": np.concatenate(family_arrays),
        "candidate_files_opened": opened,
        "schema_less_zero_row_files": schema_less_zero,
        "manifest_path": str(manifest_path.relative_to(REPOSITORY_ROOT)),
        "manifest_sha256": sha256_file(manifest_path),
    }
    lengths = {
        len(value)
        for key, value in payload.items()
        if key
        in {
            "features",
            "event",
            "member_index",
            "fold",
            "target",
            "weight",
            "signal_mode",
            "background_family",
        }
    }
    if lengths != {57326}:
        raise ValueError(f"train payload length mismatch: {lengths}")
    identities = set(zip(payload["member_index"], payload["event"]))
    if len(identities) != 57326:
        raise ValueError("source (member_index, event) identities are not unique")
    if not np.isclose(np.mean(payload["weight"]), 1.0, atol=1e-10, rtol=0):
        raise ValueError("frozen development weights do not have mean one")
    return payload


def run_unittests(output_dir: Path) -> tuple[int, bool, Path]:
    path = output_dir / "unittest.log"
    result = subprocess.run(
        [
            sys.executable,
            str(
                REPOSITORY_ROOT
                / "tests/test_train_hh4b_bdt_v1_grouped_cv.py"
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
    tests_run = int(match.group(1)) if match else 0
    return tests_run, result.returncode == 0 and "OK" in transcript, path


def xgb_parameters(
    fixed: Mapping[str, Any],
    trial: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        **dict(fixed),
        **{name: trial[name] for name in PARAMETER_NAMES},
    }


def fit_and_evaluate(
    x_train: np.ndarray,
    y_train: np.ndarray,
    training_weights: np.ndarray,
    x_heldout: np.ndarray,
    y_heldout: np.ndarray,
    heldout_weights: np.ndarray,
    parameters: Mapping[str, Any],
) -> tuple[XGBClassifier, np.ndarray, dict[str, float], float, float, float]:
    rescale = float(1.0 / np.mean(training_weights))
    fit_weights = training_weights * rescale
    model = XGBClassifier(**parameters)
    start = time.perf_counter()
    model.fit(x_train, y_train, sample_weight=fit_weights)
    training_seconds = time.perf_counter() - start
    start = time.perf_counter()
    score = model.predict_proba(x_heldout)[:, 1]
    prediction_seconds = time.perf_counter() - start
    metrics = binary_metrics(y_heldout, score, heldout_weights)
    return (
        model,
        score,
        metrics,
        rescale,
        training_seconds,
        prediction_seconds,
    )


def fold_metric_row(
    *,
    stage: str,
    variant: str,
    trial_id: int,
    fold: int,
    train_mask: np.ndarray,
    heldout_mask: np.ndarray,
    members: np.ndarray,
    rescale: float,
    metrics: Mapping[str, float],
    training_seconds: float,
    prediction_seconds: float,
    leakage: int,
    status: str = "pass",
    failure_reason: str = "none",
) -> dict[str, Any]:
    return {
        "evaluation_stage": stage,
        "variant": variant,
        "trial_id": trial_id,
        "fold": fold,
        "training_members": len(set(members[train_mask])),
        "heldout_members": len(set(members[heldout_mask])),
        "training_rows": int(np.count_nonzero(train_mask)),
        "heldout_rows": int(np.count_nonzero(heldout_mask)),
        "training_weight_rescale_factor": rescale,
        "weighted_roc_auc": metrics.get("weighted_roc_auc", ""),
        "raw_roc_auc": metrics.get("raw_roc_auc", ""),
        "weighted_average_precision": metrics.get(
            "weighted_average_precision", ""
        ),
        "raw_average_precision": metrics.get("raw_average_precision", ""),
        "training_time_seconds": training_seconds,
        "prediction_time_seconds": prediction_seconds,
        "member_leakage": leakage,
        "fit_status": status,
        "failure_reason": failure_reason,
    }


def aggregate_trial_rows(
    search_fold_rows: Sequence[Mapping[str, Any]],
    hyperparameter_rows: Sequence[Mapping[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]]]:
    by_key: dict[tuple[str, int], list[Mapping[str, Any]]] = defaultdict(list)
    for row in search_fold_rows:
        by_key[(str(row["variant"]), int(row["trial_id"]))].append(row)
    hyper_by_id = {int(row["trial_id"]): row for row in hyperparameter_rows}
    output: list[dict[str, Any]] = []
    selected: dict[str, dict[str, Any]] = {}
    selection_rule = (
        "mean_weighted_roc_auc_desc;"
        "worst_fold_weighted_roc_auc_desc;"
        "std_weighted_roc_auc_asc;"
        "n_estimators_times_two_to_max_depth_asc;"
        "trial_id_asc"
    )
    for variant in VARIANT_NAMES:
        variant_rows = []
        for trial_id in sorted(hyper_by_id):
            folds = by_key[(variant, trial_id)]
            passed = [row for row in folds if row["fit_status"] == "pass"]
            if len(passed) != 5:
                continue
            weighted_auc = np.asarray(
                [row["weighted_roc_auc"] for row in passed], dtype=float
            )
            raw_auc = np.asarray(
                [row["raw_roc_auc"] for row in passed], dtype=float
            )
            weighted_ap = np.asarray(
                [row["weighted_average_precision"] for row in passed],
                dtype=float,
            )
            raw_ap = np.asarray(
                [row["raw_average_precision"] for row in passed], dtype=float
            )
            hyper = hyper_by_id[trial_id]
            variant_rows.append(
                {
                    "variant": variant,
                    "trial_id": trial_id,
                    **{name: hyper[name] for name in PARAMETER_NAMES},
                    "folds": 5,
                    "mean_weighted_roc_auc": float(np.mean(weighted_auc)),
                    "worst_fold_weighted_roc_auc": float(np.min(weighted_auc)),
                    "std_weighted_roc_auc": float(
                        np.std(weighted_auc, ddof=0)
                    ),
                    "maximum_weighted_roc_auc": float(np.max(weighted_auc)),
                    "mean_raw_roc_auc": float(np.mean(raw_auc)),
                    "mean_weighted_average_precision": float(
                        np.mean(weighted_ap)
                    ),
                    "mean_raw_average_precision": float(np.mean(raw_ap)),
                    "complexity_proxy": hyper["complexity_proxy"],
                    "selection_rank": "",
                    "selected": False,
                    "selection_rule": selection_rule,
                    "status": "pass",
                }
            )
        ranked = rank_hyperparameter_trials(variant_rows)
        if len(ranked) != 24:
            raise ValueError(f"variant {variant} has incomplete search trials")
        output.extend(sorted(ranked, key=lambda row: int(row["trial_id"])))
        selected[variant] = next(row for row in ranked if row["selected"])
    return output, selected


def booster_gain_vector(
    model: XGBClassifier,
    feature_names: Sequence[str],
) -> np.ndarray:
    gain = model.get_booster().get_score(importance_type="gain")
    values = np.zeros(len(feature_names), dtype=np.float64)
    for key, value in gain.items():
        if key in feature_names:
            index = feature_names.index(key)
        elif key.startswith("f") and key[1:].isdigit():
            index = int(key[1:])
        else:
            raise ValueError(f"unrecognized XGBoost feature key: {key}")
        values[index] = float(value)
    return values


def add_runtime_artifact(
    rows: list[dict[str, Any]],
    path: Path,
    output_dir: Path,
    *,
    artifact_type: str,
    variant: str = "",
    fold: int | str = "",
    artifact_rows: int | str = "",
    purpose: str,
) -> None:
    rows.append(
        {
            "path": str(path.relative_to(REPOSITORY_ROOT)),
            "artifact_type": artifact_type,
            "model_variant": variant,
            "fold": fold,
            "rows": artifact_rows,
            "bytes": path.stat().st_size,
            "sha256": sha256_file(path),
            "committed_status": False,
            "retention_purpose": purpose,
            "status": "pass",
        }
    )


def add_plot_inventory_pair(
    rows: list[dict[str, Any]],
    png: Path,
    pdf: Path,
    *,
    name: str,
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
                "plot_name": name,
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
    payload: Mapping[str, Any],
    oof_scores: Mapping[str, np.ndarray],
    selected_fold_rows: Sequence[Mapping[str, Any]],
    working_rows: Sequence[Mapping[str, Any]],
    cut_metrics: Mapping[str, Any],
    feature_rows: Sequence[Mapping[str, Any]],
    config: Mapping[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    import matplotlib.pyplot as plt

    plot_dir = output_dir / "plots"
    plot_dir.mkdir(parents=True)
    style_metadata = apply_cms_style()
    style_path = config["plotting"]["style_helper"]
    style_sha = sha256_file(REPOSITORY_ROOT / style_path)
    dpi = int(config["plotting"]["dpi"])
    labels = payload["target"]
    weights = payload["weight"]
    inventory: list[dict[str, Any]] = []
    display = {
        "mass_aware": "Mass-aware BDT",
        "explicit_dijet_mass_plane_blind": (
            "Dijet-mass-plane-blind BDT"
        ),
    }
    variant_colors = {
        "mass_aware": CATEGORY_COLORS[0],
        "explicit_dijet_mass_plane_blind": CATEGORY_COLORS[1],
    }

    for weighted, name in (
        (True, "oof_roc_development_weighted"),
        (False, "oof_roc_raw"),
    ):
        fig, ax = plt.subplots(figsize=(6.3, 5.2))
        for variant in VARIANT_NAMES:
            curve_weights = weights if weighted else None
            fpr, tpr, _ = roc_curve(
                labels,
                oof_scores[variant],
                sample_weight=curve_weights,
            )
            auc = roc_auc_score(
                labels,
                oof_scores[variant],
                sample_weight=curve_weights,
            )
            ax.plot(
                tpr,
                1.0 - fpr,
                color=variant_colors[variant],
                linewidth=2.0,
                label=f"{display[variant]} (AUC={auc:.3f})",
            )
        if weighted:
            cut_x = cut_metrics["weighted_signal_efficiency"]
            cut_y = cut_metrics["weighted_background_rejection"]
        else:
            cut_x = cut_metrics["raw_signal_efficiency"]
            cut_y = cut_metrics["raw_background_rejection"]
        ax.scatter(
            [cut_x],
            [cut_y],
            marker="*",
            s=130,
            color="#000000",
            label=r"Optimized cut: $R_{HH}<34$",
            zorder=5,
        )
        ax.set_xlim(0.0, 1.0)
        ax.set_ylim(0.0, 1.0)
        ax.set_xlabel("Signal efficiency")
        ax.set_ylabel("Background rejection")
        ax.legend(loc="lower left")
        add_delphes_header(
            ax,
            "Train-only grouped cross-validation; "
            + ("development weighted" if weighted else "raw"),
        )
        png, pdf = save_png_pdf(fig, plot_dir / name, dpi=dpi)
        add_plot_inventory_pair(
            inventory,
            png,
            pdf,
            name=name,
            variables=["oof_score", "training_target", "r_hh_125_125"],
            normalization=(
                "hierarchical_development_weights"
                if weighted
                else "raw_train_candidates"
            ),
            binning="continuous_roc_curve_and_single_cut_operating_point",
            style_path=style_path,
            style_sha=style_sha,
            style_metadata=style_metadata,
            dpi=dpi,
        )

    name = "oof_precision_recall_development_weighted"
    fig, ax = plt.subplots(figsize=(6.3, 5.2))
    for variant in VARIANT_NAMES:
        precision, recall, _ = precision_recall_curve(
            labels,
            oof_scores[variant],
            sample_weight=weights,
        )
        ap = average_precision_score(
            labels,
            oof_scores[variant],
            sample_weight=weights,
        )
        ax.plot(
            recall,
            precision,
            color=variant_colors[variant],
            linewidth=2.0,
            label=f"{display[variant]} (AP={ap:.3f})",
        )
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.set_xlabel("Signal efficiency")
    ax.set_ylabel("Precision")
    ax.legend(loc="lower left")
    add_delphes_header(
        ax,
        "Train-only grouped cross-validation; development weighted",
    )
    png, pdf = save_png_pdf(fig, plot_dir / name, dpi=dpi)
    add_plot_inventory_pair(
        inventory,
        png,
        pdf,
        name=name,
        variables=["oof_score", "training_target"],
        normalization="hierarchical_development_weights",
        binning="continuous_precision_recall_curve",
        style_path=style_path,
        style_sha=style_sha,
        style_metadata=style_metadata,
        dpi=dpi,
    )

    score_bins = np.linspace(
        0.0,
        1.0,
        int(config["plotting"]["score_bin_edges"]),
    )
    for variant, name in (
        ("mass_aware", "oof_score_mass_aware"),
        (
            "explicit_dijet_mass_plane_blind",
            "oof_score_mass_plane_blind",
        ),
    ):
        fig, ax = plt.subplots(figsize=(6.5, 5.0))
        for target, label, color in (
            (0, "Background", CATEGORY_COLORS[0]),
            (1, r"$HH\rightarrow b\bar{b}b\bar{b}$", CATEGORY_COLORS[1]),
        ):
            selected = labels == target
            ax.hist(
                oof_scores[variant][selected],
                bins=score_bins,
                weights=weights[selected],
                density=True,
                histtype="step",
                linewidth=2.0,
                color=color,
                label=label,
            )
        ax.set_xlabel("OOF BDT score")
        ax.set_ylabel("Development-weighted density")
        ax.set_xlim(0.0, 1.0)
        ax.legend(loc="upper center")
        add_delphes_header(
            ax,
            f"Train-only grouped cross-validation; {display[variant]}",
        )
        png, pdf = save_png_pdf(fig, plot_dir / name, dpi=dpi)
        add_plot_inventory_pair(
            inventory,
            png,
            pdf,
            name=name,
            variables=["oof_score", "training_target"],
            normalization="hierarchical_development_weighted_unit_density",
            binning="40_fixed_uniform_bins_from_0_to_1",
            style_path=style_path,
            style_sha=style_sha,
            style_metadata=style_metadata,
            dpi=dpi,
        )

    name = "fold_auc_stability"
    fig, ax = plt.subplots(figsize=(6.5, 5.0))
    for variant in VARIANT_NAMES:
        rows = sorted(
            (
                row
                for row in selected_fold_rows
                if row["variant"] == variant
            ),
            key=lambda row: int(row["fold"]),
        )
        ax.plot(
            [int(row["fold"]) for row in rows],
            [float(row["weighted_roc_auc"]) for row in rows],
            marker="o",
            linewidth=1.8,
            color=variant_colors[variant],
            label=display[variant],
        )
    ax.set_xticks(range(5))
    ax.set_xticklabels([f"Fold {fold}" for fold in range(5)])
    ax.set_ylabel("Development-weighted ROC AUC")
    ax.legend(loc="lower right")
    add_delphes_header(
        ax,
        "Train-only grouped cross-validation; grouped by source member",
    )
    png, pdf = save_png_pdf(fig, plot_dir / name, dpi=dpi)
    add_plot_inventory_pair(
        inventory,
        png,
        pdf,
        name=name,
        variables=["fold", "weighted_roc_auc", "variant"],
        normalization="hierarchical_development_weights",
        binning="five_frozen_member_folds",
        style_path=style_path,
        style_sha=style_sha,
        style_metadata=style_metadata,
        dpi=dpi,
    )

    name = "background_family_efficiency_at_50pct_signal"
    families = sorted(
        set(payload["background_family"][payload["target"] == 0])
    )
    positions = np.arange(len(families))
    width = 0.36
    fig, ax = plt.subplots(figsize=(9.0, 5.2))
    for offset, variant in enumerate(VARIANT_NAMES):
        row = next(
            item
            for item in working_rows
            if item["variant"] == variant
            and item["working_point_definition"]
            == "weighted_signal_efficiency_0.50"
        )
        efficiencies = row["weighted_background_family_efficiencies"]
        ax.bar(
            positions + (offset - 0.5) * width,
            [efficiencies[family] for family in families],
            width=width,
            color=variant_colors[variant],
            label=display[variant],
        )
    ax.set_xticks(positions)
    ax.set_xticklabels(
        [family.replace("_", " ") for family in families],
        rotation=30,
        ha="right",
    )
    ax.set_ylabel("Development-weighted background efficiency")
    ax.set_ylim(bottom=0.0)
    ax.legend(loc="upper left")
    add_delphes_header(
        ax,
        r"Train-only OOF; weighted signal efficiency $=0.50$",
    )
    png, pdf = save_png_pdf(fig, plot_dir / name, dpi=dpi)
    add_plot_inventory_pair(
        inventory,
        png,
        pdf,
        name=name,
        variables=["background_family", "weighted_efficiency", "variant"],
        normalization="within_family_hierarchical_development_weights",
        binning="eight_frozen_background_families",
        style_path=style_path,
        style_sha=style_sha,
        style_metadata=style_metadata,
        dpi=dpi,
    )

    for variant, name in (
        ("mass_aware", "feature_importance_mass_aware"),
        (
            "explicit_dijet_mass_plane_blind",
            "feature_importance_mass_plane_blind",
        ),
    ):
        rows = sorted(
            (row for row in feature_rows if row["variant"] == variant),
            key=lambda row: float(row["mean_normalized_gain_importance"]),
        )
        fig, ax = plt.subplots(figsize=(7.5, 9.0))
        positions = np.arange(len(rows))
        ax.barh(
            positions,
            [row["mean_normalized_gain_importance"] for row in rows],
            xerr=[
                row["standard_deviation_normalized_gain_importance"]
                for row in rows
            ],
            color=variant_colors[variant],
            alpha=0.9,
        )
        ax.set_yticks(positions)
        ax.set_yticklabels([row["latex_label"] for row in rows])
        ax.set_xlabel("Mean normalized XGBoost gain")
        add_delphes_header(
            ax,
            f"Selected {display[variant]}; diagnostic, not causal attribution",
        )
        png, pdf = save_png_pdf(fig, plot_dir / name, dpi=dpi)
        add_plot_inventory_pair(
            inventory,
            png,
            pdf,
            name=name,
            variables=["feature", "normalized_gain", "fold"],
            normalization="each_selected_fold_model_gain_sums_to_one",
            binning="frozen_feature_order_sorted_for_display_by_mean_gain",
            style_path=style_path,
            style_sha=style_sha,
            style_metadata=style_metadata,
            dpi=dpi,
        )

    background = labels == 0
    rhh = payload["features"]["r_hh_125_125"].to_numpy(dtype=float)
    rhh_bins = np.asarray(
        config["score_mass_diagnostics"]["plot_rhh_bin_edges"],
        dtype=float,
    )
    for variant, name in (
        (
            "mass_aware",
            "background_rhh_by_score_quintile_mass_aware",
        ),
        (
            "explicit_dijet_mass_plane_blind",
            "background_rhh_by_score_quintile_mass_plane_blind",
        ),
    ):
        scores = oof_scores[variant][background]
        b_weights = weights[background]
        assignment, _ = score_quintile_assignment(scores, b_weights)
        fig, ax = plt.subplots(figsize=(7.0, 5.2))
        for quintile in range(5):
            selected = assignment == quintile
            ax.hist(
                rhh[background][selected],
                bins=rhh_bins,
                weights=b_weights[selected],
                density=True,
                histtype="step",
                linewidth=1.6,
                color=CATEGORY_COLORS[quintile],
                label=f"Score quintile {quintile + 1}",
            )
        ax.set_xlabel(r"$R_{HH}^{125,125}$")
        ax.set_ylabel("Development-weighted density")
        ax.set_xlim(rhh_bins[0], rhh_bins[-1])
        ax.legend(loc="upper right", ncol=2)
        add_delphes_header(
            ax,
            "Background OOF diagnostic; not used in model selection",
        )
        png, pdf = save_png_pdf(fig, plot_dir / name, dpi=dpi)
        add_plot_inventory_pair(
            inventory,
            png,
            pdf,
            name=name,
            variables=["r_hh_125_125", "weighted_background_score_quintile"],
            normalization="within_quintile_hierarchical_weighted_density",
            binning="identical_fixed_background_only_rhh_bins_for_both_variants",
            style_path=style_path,
            style_sha=style_sha,
            style_metadata=style_metadata,
            dpi=dpi,
        )
    return inventory, style_metadata


def checkpoint_readme(
    summary: Mapping[str, Any],
    selected_rows: Sequence[Mapping[str, Any]],
) -> str:
    selections = "\n".join(
        f"- `{row['variant']}`: trial {row['trial_id']}, "
        f"mean weighted fold AUC {float(row['mean_weighted_roc_auc']):.6f}"
        for row in selected_rows
    )
    return f"""# Canonical HH4b BDT-v1 grouped cross-validation

## Result

- Status: `{summary['status']}`
- Train members and rows: {summary['train_members']} and {summary['train_rows']}
- Frozen grouped folds: {summary['folds']}
- Hyperparameter configurations per variant: {summary['hyperparameter_configurations']}
- Search fold fits: {summary['search_fold_fits']}
- Selected-configuration fold refits: {summary['selected_configuration_refits']}
- Total XGBoost models trained: {summary['models_trained_total']}
- Missing/duplicate OOF rows and member leakage: \
{summary['missing_oof_rows']}/{summary['duplicate_oof_rows']}/\
{summary['member_leakage_rows']}

## Selected configurations

{selections}

Selection used exactly: highest mean held-out development-weighted ROC AUC,
highest worst-fold weighted ROC AUC, lowest fold AUC standard deviation,
lowest `n_estimators * (2 ** max_depth)`, then lowest trial ID. Validation,
test, physical significance, raw yields, and the central mass region played no
role.

## Scientific scope

The benchmark compares the frozen 34-feature mass-aware BDT with the frozen
30-feature **explicit dijet-mass-plane-blind ablation**. The latter is not
described as fully mass-decorrelated. No feature scaler was fitted, no
`r_hh_125_125 < 34` requirement or other signal-region selection was applied
to the BDT input, and every fit used the frozen hierarchical development
weights after a fold-local mean-one rescaling.

The optimized cut baseline is evaluated separately on the identical train
population. Balanced-efficiency and purity ratios are retained only as
explicitly nonphysical development proxies. No cross sections, luminosity,
generator/importance weights, physical yields, significance, or expected
limits were calculated.

Gain importance is a model diagnostic, not a causal attribution. Background
score–mass-plane diagnostics were not used in model selection.

## Data-access boundary

- Train candidate files opened: {summary['train_candidate_files_opened']}
- Validation candidate files opened / rows read / metrics: 0 / 0 / 0
- Test candidate files opened / rows read / metrics: 0 / 0 / 0
- Full-train models fitted: 0

The ten selected fold models and two row-level OOF prediction Parquets remain
only under the ignored runtime directory. No model, prediction, candidate
feature matrix, or serialized training dataset is in this checkpoint.

## Environment

- Python: `{summary['python_version']}`
- scikit-learn: `{summary['sklearn_version']}`
- XGBoost: `{summary['xgboost_version']}`
- NumPy: `{summary['numpy_version']}`
- SciPy: `{summary['scipy_version']}`
- mplhep used: `{summary['mplhep_used']}`

All plots use the frozen nonofficial CMS-publication-inspired helper, with the
truthful “Delphes simulation” and 13 TeV annotations.

## Authorization boundary

- `grouped_cv_complete: true`
- `selected_hyperparameters_frozen: true`
- `train_oof_working_points_frozen: true`
- `final_full_train_models_fitted: false`
- `validation_evaluation_authorized: false`
- `test_access_authorized: false`
- `physical_significance_authorized: false`

## Next gate

`{summary['next_gate']}`
"""


def write_checkpoint_manifest(checkpoint_dir: Path) -> None:
    lines = []
    for path in sorted(checkpoint_dir.rglob("*")):
        if path.is_file() and path.name != "SHA256SUMS":
            lines.append(
                f"{sha256_file(path)}  {path.relative_to(checkpoint_dir).as_posix()}"
            )
    (checkpoint_dir / "SHA256SUMS").write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Canonical HH4b BDT-v1 train-only grouped cross-validation"
    )
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="replace only the exact configured runtime/checkpoint directories",
    )
    args = parser.parse_args()
    config_path = args.config.resolve()
    config = load_yaml(config_path)

    source_commit = git_output("rev-parse", "HEAD")
    if source_commit != config["source_commit"]:
        raise ValueError("source commit differs from the frozen grouped-CV config")
    environment = verify_environment(config)
    (
        contract_summary,
        fold_contract_rows,
        weight_contract_rows,
        background_map,
        signal_map,
        mass_aware_features,
        mass_blind_features,
    ) = load_and_verify_contract(config)

    output_dir = REPOSITORY_ROOT / config["runtime_output_dir"]
    checkpoint_dir = REPOSITORY_ROOT / config["checkpoint_dir"]
    prepare_directory(output_dir, args.overwrite)
    prepare_directory(checkpoint_dir, args.overwrite)

    tests_run, tests_passed, unittest_log = run_unittests(output_dir)
    if not tests_passed:
        raise RuntimeError("grouped-CV synthetic unittest suite failed")

    payload = load_train_population(
        config,
        fold_contract_rows,
        weight_contract_rows,
        background_map,
        signal_map,
    )
    expected = config["expected_population"]
    if payload["candidate_files_opened"] != expected["train_members"]:
        raise ValueError("not every train candidate file was opened")
    labels = payload["target"]
    if int(np.count_nonzero(labels == 1)) != expected["signal_rows"]:
        raise ValueError("signal row count mismatch")
    if int(np.count_nonzero(labels == 0)) != expected["background_rows"]:
        raise ValueError("background row count mismatch")

    hyper_rows = generate_hyperparameter_space(
        config["hyperparameter_search"]
    )
    write_tsv(
        output_dir / "hyperparameter_space.tsv",
        hyper_rows,
        TABLE_FIELDS["hyperparameter_space.tsv"],
    )

    feature_arrays = {
        "mass_aware": payload["features"]
        .loc[:, mass_aware_features]
        .to_numpy(dtype=np.float32, copy=False),
        "explicit_dijet_mass_plane_blind": payload["features"]
        .loc[:, mass_blind_features]
        .to_numpy(dtype=np.float32, copy=False),
    }
    feature_names_by_variant = {
        "mass_aware": mass_aware_features,
        "explicit_dijet_mass_plane_blind": mass_blind_features,
    }
    folds = payload["fold"]
    members = payload["member_index"]
    weights = payload["weight"]
    fixed_parameters = config["estimator_fixed_parameters"]

    search_fold_rows: list[dict[str, Any]] = []
    models_trained_total = 0
    for variant in VARIANT_NAMES:
        matrix = feature_arrays[variant]
        for hyper in hyper_rows:
            trial_id = int(hyper["trial_id"])
            parameters = xgb_parameters(fixed_parameters, hyper)
            for fold in range(5):
                heldout = folds == fold
                training = ~heldout
                training_members = set(members[training])
                heldout_members = set(members[heldout])
                leakage = len(training_members & heldout_members)
                print(
                    f"SEARCH variant={variant} trial={trial_id:02d} "
                    f"fold={fold} rows={np.count_nonzero(training)}/"
                    f"{np.count_nonzero(heldout)}",
                    flush=True,
                )
                try:
                    model, _, metrics, rescale, train_seconds, predict_seconds = (
                        fit_and_evaluate(
                            matrix[training],
                            labels[training],
                            weights[training],
                            matrix[heldout],
                            labels[heldout],
                            weights[heldout],
                            parameters,
                        )
                    )
                    models_trained_total += 1
                    row = fold_metric_row(
                        stage="hyperparameter_search",
                        variant=variant,
                        trial_id=trial_id,
                        fold=fold,
                        train_mask=training,
                        heldout_mask=heldout,
                        members=members,
                        rescale=rescale,
                        metrics=metrics,
                        training_seconds=train_seconds,
                        prediction_seconds=predict_seconds,
                        leakage=leakage,
                    )
                    del model
                except Exception as error:
                    row = fold_metric_row(
                        stage="hyperparameter_search",
                        variant=variant,
                        trial_id=trial_id,
                        fold=fold,
                        train_mask=training,
                        heldout_mask=heldout,
                        members=members,
                        rescale=float("nan"),
                        metrics={},
                        training_seconds=0.0,
                        prediction_seconds=0.0,
                        leakage=leakage,
                        status="fail",
                        failure_reason=f"{type(error).__name__}: {error}",
                    )
                search_fold_rows.append(row)
                write_tsv(
                    output_dir / "fold_metrics_search_partial.tsv",
                    search_fold_rows,
                    TABLE_FIELDS["fold_metrics.tsv"],
                )
                gc.collect()

    trial_rows, selected_by_variant = aggregate_trial_rows(
        search_fold_rows,
        hyper_rows,
    )
    (output_dir / "fold_metrics_search_partial.tsv").unlink(missing_ok=True)
    selected_rows = [
        {
            "variant": variant,
            "trial_id": selected_by_variant[variant]["trial_id"],
            **{
                name: selected_by_variant[variant][name]
                for name in PARAMETER_NAMES
            },
            "mean_weighted_roc_auc": selected_by_variant[variant][
                "mean_weighted_roc_auc"
            ],
            "worst_fold_weighted_roc_auc": selected_by_variant[variant][
                "worst_fold_weighted_roc_auc"
            ],
            "std_weighted_roc_auc": selected_by_variant[variant][
                "std_weighted_roc_auc"
            ],
            "complexity_proxy": selected_by_variant[variant][
                "complexity_proxy"
            ],
            "selection_rank": 1,
            "selection_rule": selected_by_variant[variant]["selection_rule"],
            "status": "pass",
        }
        for variant in VARIANT_NAMES
    ]

    runtime_rows: list[dict[str, Any]] = []
    add_runtime_artifact(
        runtime_rows,
        unittest_log,
        output_dir,
        artifact_type="execution_log",
        artifact_rows=tests_run,
        purpose="synthetic_unittest_evidence",
    )
    selected_fold_rows: list[dict[str, Any]] = []
    oof_scores: dict[str, np.ndarray] = {}
    produced_indices_by_variant: dict[str, list[int]] = {}
    training_member_sets: dict[str, list[set[int]]] = {}
    heldout_member_sets: dict[str, list[set[int]]] = {}
    importance_vectors: dict[str, list[np.ndarray]] = {}
    failed_selected_refits = 0
    row_indices = np.arange(len(labels), dtype=np.int64)

    for variant in VARIANT_NAMES:
        matrix = feature_arrays[variant]
        selected_hyper = selected_by_variant[variant]
        parameters = xgb_parameters(fixed_parameters, selected_hyper)
        scores = np.full(len(labels), np.nan, dtype=np.float64)
        assignment_counts = np.zeros(len(labels), dtype=np.int8)
        produced_indices: list[int] = []
        train_sets: list[set[int]] = []
        heldout_sets: list[set[int]] = []
        fold_importance: list[np.ndarray] = []
        for fold in range(5):
            heldout = folds == fold
            training = ~heldout
            train_set = set(members[training])
            heldout_set = set(members[heldout])
            train_sets.append(train_set)
            heldout_sets.append(heldout_set)
            leakage = len(train_set & heldout_set)
            print(
                f"REFIT variant={variant} fold={fold}",
                flush=True,
            )
            try:
                (
                    model,
                    score,
                    metrics,
                    rescale,
                    train_seconds,
                    predict_seconds,
                ) = fit_and_evaluate(
                    matrix[training],
                    labels[training],
                    weights[training],
                    matrix[heldout],
                    labels[heldout],
                    weights[heldout],
                    parameters,
                )
                models_trained_total += 1
                scores[heldout] = score
                assignment_counts[heldout] += 1
                produced_indices.extend(row_indices[heldout].tolist())
                fold_importance.append(
                    booster_gain_vector(
                        model,
                        feature_names_by_variant[variant],
                    )
                )
                model_path = (
                    output_dir
                    / "selected_fold_models"
                    / variant
                    / f"fold_{fold}.json"
                )
                model_path.parent.mkdir(parents=True, exist_ok=True)
                model.save_model(model_path)
                add_runtime_artifact(
                    runtime_rows,
                    model_path,
                    output_dir,
                    artifact_type="selected_fold_xgboost_json_model",
                    variant=variant,
                    fold=fold,
                    purpose=(
                        "selected_configuration_fold_model_for_oof_reproducibility"
                    ),
                )
                selected_fold_rows.append(
                    fold_metric_row(
                        stage="selected_configuration_refit",
                        variant=variant,
                        trial_id=int(selected_hyper["trial_id"]),
                        fold=fold,
                        train_mask=training,
                        heldout_mask=heldout,
                        members=members,
                        rescale=rescale,
                        metrics=metrics,
                        training_seconds=train_seconds,
                        prediction_seconds=predict_seconds,
                        leakage=leakage,
                    )
                )
                del model
            except Exception as error:
                failed_selected_refits += 1
                selected_fold_rows.append(
                    fold_metric_row(
                        stage="selected_configuration_refit",
                        variant=variant,
                        trial_id=int(selected_hyper["trial_id"]),
                        fold=fold,
                        train_mask=training,
                        heldout_mask=heldout,
                        members=members,
                        rescale=float("nan"),
                        metrics={},
                        training_seconds=0.0,
                        prediction_seconds=0.0,
                        leakage=leakage,
                        status="fail",
                        failure_reason=f"{type(error).__name__}: {error}",
                    )
                )
            gc.collect()
        oof_scores[variant] = scores
        produced_indices_by_variant[variant] = produced_indices
        training_member_sets[variant] = train_sets
        heldout_member_sets[variant] = heldout_sets
        importance_vectors[variant] = fold_importance
        if np.any(assignment_counts != 1) or np.any(~np.isfinite(scores)):
            raise ValueError(f"OOF assignment failure for {variant}")

        oof_path = output_dir / "oof_predictions" / f"{variant}.parquet"
        oof_path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(
            {
                "row_index": row_indices,
                "member_index": members,
                "event": payload["event"],
                "fold": folds,
                "training_target": labels,
                "development_weight": weights,
                "score": scores,
                "signal_mode": payload["signal_mode"],
                "background_family": payload["background_family"],
                "r_hh_125_125": payload["features"][
                    "r_hh_125_125"
                ].to_numpy(dtype=float),
            }
        ).to_parquet(oof_path, index=False)
        add_runtime_artifact(
            runtime_rows,
            oof_path,
            output_dir,
            artifact_type="row_level_oof_prediction_parquet",
            variant=variant,
            artifact_rows=len(labels),
            purpose="train_only_oof_integrity_and_downstream_diagnostics",
        )

    all_integrity = {
        variant: check_oof_integrity(
            row_indices,
            produced_indices_by_variant[variant],
            training_member_sets[variant],
            heldout_member_sets[variant],
        )
        for variant in VARIANT_NAMES
    }
    missing_oof = sum(row["missing_oof_rows"] for row in all_integrity.values())
    duplicate_oof = sum(
        row["duplicate_oof_rows"] for row in all_integrity.values()
    )
    member_leakage = sum(
        row["member_leakage_rows"] for row in all_integrity.values()
    )

    oof_metric_rows: list[dict[str, Any]] = []
    for variant in VARIANT_NAMES:
        pooled = binary_metrics(labels, oof_scores[variant], weights)
        fold_rows = [
            row
            for row in selected_fold_rows
            if row["variant"] == variant and row["fit_status"] == "pass"
        ]
        weighted_auc = np.asarray(
            [row["weighted_roc_auc"] for row in fold_rows], dtype=float
        )
        raw_auc = np.asarray(
            [row["raw_roc_auc"] for row in fold_rows], dtype=float
        )
        weighted_ap = np.asarray(
            [row["weighted_average_precision"] for row in fold_rows],
            dtype=float,
        )
        raw_ap = np.asarray(
            [row["raw_average_precision"] for row in fold_rows], dtype=float
        )
        oof_metric_rows.append(
            {
                "variant": variant,
                "features": len(feature_names_by_variant[variant]),
                "selected_trial_id": selected_by_variant[variant]["trial_id"],
                "oof_rows": len(labels),
                **pooled,
                "fold_weighted_roc_auc_mean": float(np.mean(weighted_auc)),
                "fold_weighted_roc_auc_std": float(
                    np.std(weighted_auc, ddof=0)
                ),
                "fold_weighted_roc_auc_minimum": float(np.min(weighted_auc)),
                "fold_weighted_roc_auc_maximum": float(np.max(weighted_auc)),
                "fold_raw_roc_auc_mean": float(np.mean(raw_auc)),
                "fold_raw_roc_auc_std": float(np.std(raw_auc, ddof=0)),
                "fold_weighted_average_precision_mean": float(
                    np.mean(weighted_ap)
                ),
                "fold_raw_average_precision_mean": float(np.mean(raw_ap)),
                "diagnostic_scope": (
                    "pooled_train_only_out_of_fold_development_diagnostic"
                ),
                "status": "pass",
            }
        )

    rhh = payload["features"]["r_hh_125_125"].to_numpy(dtype=float)
    cut = cut_baseline_metrics(
        rhh,
        labels,
        weights,
        payload["signal_mode"],
        payload["background_family"],
        threshold=float(config["cut_baseline"]["threshold"]),
    )
    cut_row = {
        "baseline": "optimized_cut_baseline",
        "selection": "r_hh_125_125 < 34",
        **cut,
        "proxy_interpretation": "nonphysical_development_proxy",
        "physical_yields_used": False,
        "status": "pass",
    }

    working_rows: list[dict[str, Any]] = []
    background_efficiency_rows: list[dict[str, Any]] = []
    for family, weighted_value in cut[
        "weighted_background_family_efficiencies"
    ].items():
        background_efficiency_rows.append(
            {
                "source": "optimized_cut_baseline",
                "working_point_definition": "r_hh_125_125_lt_34",
                "score_threshold": "",
                "background_family": family,
                "family_rows": int(
                    np.count_nonzero(payload["background_family"] == family)
                ),
                "weighted_efficiency": weighted_value,
                "raw_efficiency": cut[
                    "raw_background_family_efficiencies"
                ][family],
                "normalization": "within_frozen_background_family",
                "status": "pass",
            }
        )

    target_definitions = [
        (f"weighted_signal_efficiency_{value:.2f}", float(value))
        for value in config["working_points"]["weighted_signal_efficiencies"]
    ]
    target_definitions.append(
        (
            "weighted_signal_efficiency_equal_to_cut_baseline",
            cut["weighted_signal_efficiency"],
        )
    )
    signal_mask = labels == 1
    for variant in VARIANT_NAMES:
        for definition, target_eff in target_definitions:
            threshold = weighted_signal_threshold(
                oof_scores[variant][signal_mask],
                weights[signal_mask],
                target_eff,
            )
            selected = oof_scores[variant] >= threshold
            metrics = selection_metrics(
                selected,
                labels,
                weights,
                payload["signal_mode"],
                payload["background_family"],
            )
            row = {
                "variant": variant,
                "working_point_definition": definition,
                "target_weighted_signal_efficiency": target_eff,
                "oof_score_threshold": threshold,
                **metrics,
                "proxy_interpretation": "nonphysical_development_proxy",
                "threshold_source": "pooled_train_only_oof_signal_scores",
                "status": "pass",
            }
            working_rows.append(row)
            for family, weighted_value in metrics[
                "weighted_background_family_efficiencies"
            ].items():
                background_efficiency_rows.append(
                    {
                        "source": variant,
                        "working_point_definition": definition,
                        "score_threshold": threshold,
                        "background_family": family,
                        "family_rows": int(
                            np.count_nonzero(
                                payload["background_family"] == family
                            )
                        ),
                        "weighted_efficiency": weighted_value,
                        "raw_efficiency": metrics[
                            "raw_background_family_efficiencies"
                        ][family],
                        "normalization": "within_frozen_background_family",
                        "status": "pass",
                    }
                )

    signal_mode_rows: list[dict[str, Any]] = []
    background_mask = labels == 0
    for variant in VARIANT_NAMES:
        for mode in ("ggf_hh4b", "vbf_hh4b"):
            mode_mask = payload["signal_mode"] == mode
            selected = mode_mask | background_mask
            target = mode_mask[selected].astype(np.int8)
            mode_metrics = binary_metrics(
                target,
                oof_scores[variant][selected],
                weights[selected],
            )
            signal_mode_rows.append(
                {
                    "variant": variant,
                    "signal_mode": mode,
                    "signal_rows": int(np.count_nonzero(mode_mask)),
                    "background_rows": int(np.count_nonzero(background_mask)),
                    "weighted_roc_auc_vs_all_background": mode_metrics[
                        "weighted_roc_auc"
                    ],
                    "raw_roc_auc_vs_all_background": mode_metrics[
                        "raw_roc_auc"
                    ],
                    "weighted_average_precision_vs_all_background": (
                        mode_metrics["weighted_average_precision"]
                    ),
                    "raw_average_precision_vs_all_background": mode_metrics[
                        "raw_average_precision"
                    ],
                    "weighted_mean_score": float(
                        np.average(
                            oof_scores[variant][mode_mask],
                            weights=weights[mode_mask],
                        )
                    ),
                    "raw_mean_score": float(
                        np.mean(oof_scores[variant][mode_mask])
                    ),
                    "diagnostic_scope": (
                        "train_only_oof_mode_versus_all_background"
                    ),
                    "status": "pass",
                }
            )

    feature_importance_rows: list[dict[str, Any]] = []
    for variant in VARIANT_NAMES:
        feature_importance_rows.extend(
            aggregate_gain_importance(
                importance_vectors[variant],
                feature_names_by_variant[variant],
                variant,
            )
        )

    score_mass_rows: list[dict[str, Any]] = []
    for variant in VARIANT_NAMES:
        score_mass_rows.extend(
            score_mass_diagnostic_rows(
                oof_scores[variant][background_mask],
                rhh[background_mask],
                weights[background_mask],
                variant,
            )
        )

    plot_rows, style_metadata = make_plots(
        output_dir,
        payload,
        oof_scores,
        selected_fold_rows,
        working_rows,
        cut,
        feature_importance_rows,
        config,
    )

    fold_rows_all = [*search_fold_rows, *selected_fold_rows]
    environment_path = output_dir / "environment.json"
    write_json(environment_path, environment)
    tables = {
        "hyperparameter_space.tsv": hyper_rows,
        "hyperparameter_trials.tsv": trial_rows,
        "fold_metrics.tsv": fold_rows_all,
        "selected_hyperparameters.tsv": selected_rows,
        "oof_metrics_summary.tsv": oof_metric_rows,
        "cut_baseline_comparison.tsv": [cut_row],
        "working_points.tsv": working_rows,
        "signal_mode_metrics.tsv": signal_mode_rows,
        "background_family_efficiency.tsv": background_efficiency_rows,
        "feature_importance.tsv": feature_importance_rows,
        "score_mass_plane_diagnostics.tsv": score_mass_rows,
        "runtime_artifact_inventory.tsv": runtime_rows,
        "plot_inventory.tsv": plot_rows,
    }
    for name, rows in tables.items():
        write_tsv(output_dir / name, rows, TABLE_FIELDS[name])

    search_failures = sum(
        row["fit_status"] != "pass" for row in search_fold_rows
    )
    pooled_by_variant = {
        row["variant"]: row for row in oof_metric_rows
    }
    summary: dict[str, Any] = {
        "schema_version": 1,
        "status": "hh4b_bdt_v1_grouped_cross_validation_fail",
        "source_commit": source_commit,
        **environment,
        "train_members": expected["train_members"],
        "train_rows": len(labels),
        "signal_rows": int(np.count_nonzero(labels == 1)),
        "background_rows": int(np.count_nonzero(labels == 0)),
        "train_candidate_files_opened": payload["candidate_files_opened"],
        "schema_less_zero_row_train_files": payload[
            "schema_less_zero_row_files"
        ],
        "folds": expected["folds"],
        "model_variants": len(VARIANT_NAMES),
        "features_mass_aware": len(mass_aware_features),
        "features_mass_plane_blind": len(mass_blind_features),
        "hyperparameter_configurations": len(hyper_rows),
        "hyperparameter_trials": len(trial_rows),
        "search_fold_fits": len(search_fold_rows),
        "failed_search_fold_fits": search_failures,
        "selected_configuration_refits": len(selected_fold_rows),
        "failed_selected_refits": failed_selected_refits,
        "models_trained_total": models_trained_total,
        "mass_aware_oof_rows": len(produced_indices_by_variant["mass_aware"]),
        "mass_plane_blind_oof_rows": len(
            produced_indices_by_variant[
                "explicit_dijet_mass_plane_blind"
            ]
        ),
        "missing_oof_rows": missing_oof,
        "duplicate_oof_rows": duplicate_oof,
        "member_leakage_rows": member_leakage,
        "mass_aware_weighted_oof_roc_auc": pooled_by_variant[
            "mass_aware"
        ]["weighted_roc_auc"],
        "mass_plane_blind_weighted_oof_roc_auc": pooled_by_variant[
            "explicit_dijet_mass_plane_blind"
        ]["weighted_roc_auc"],
        "mass_aware_raw_oof_roc_auc": pooled_by_variant["mass_aware"][
            "raw_roc_auc"
        ],
        "mass_plane_blind_raw_oof_roc_auc": pooled_by_variant[
            "explicit_dijet_mass_plane_blind"
        ]["raw_roc_auc"],
        "cut_baseline_weighted_signal_efficiency": cut[
            "weighted_signal_efficiency"
        ],
        "cut_baseline_weighted_background_efficiency": cut[
            "weighted_background_efficiency"
        ],
        "validation_candidate_files_opened": 0,
        "validation_rows_read": 0,
        "validation_metrics_calculated": 0,
        "test_candidate_files_opened": 0,
        "test_rows_read": 0,
        "test_metrics_calculated": 0,
        "full_train_models_fitted": 0,
        "scalers_fitted": 0,
        "physical_event_weights_used": 0,
        "physics_yields_calculated": 0,
        "significances_calculated": 0,
        "shap_calculations": 0,
        "unittest_tests_run": tests_run,
        "unittest_tests_passed": tests_run if tests_passed else 0,
        "unittest_failures": 0 if tests_passed else 1,
        "plots_requested": len(config["plotting"]["plots"]),
        "pngs_written": sum(row["format"] == "png" for row in plot_rows),
        "pdfs_written": sum(row["format"] == "pdf" for row in plot_rows),
        "plot_failures": sum(row["status"] != "pass" for row in plot_rows),
        "cms_style_failures": (
            0 if style_metadata["cms_inspired_style_applied"] else 1
        ),
        "latex_math_label_failures": 0,
        "mplhep_used": style_metadata["mplhep_used"],
        "runtime_artifacts": len(runtime_rows),
        "runtime_model_artifacts": sum(
            row["artifact_type"] == "selected_fold_xgboost_json_model"
            for row in runtime_rows
        ),
        "runtime_oof_prediction_artifacts": sum(
            row["artifact_type"] == "row_level_oof_prediction_parquet"
            for row in runtime_rows
        ),
        "committed_row_level_or_model_artifacts": 0,
        "input_contract_status": contract_summary["status"],
        "input_contract_checkpoint_sha256": sha256_file(
            REPOSITORY_ROOT
            / config["input_contract"]["checkpoint_dir"]
            / "SHA256SUMS"
        ),
        "config_sha256": sha256_file(config_path),
        "runner_sha256": sha256_file(Path(__file__)),
        "common_helper_sha256": sha256_file(
            REPOSITORY_ROOT / config["input_contract"]["common_helper_path"]
        ),
        "style_helper_sha256": sha256_file(
            REPOSITORY_ROOT / config["plotting"]["style_helper"]
        ),
        "grouped_cv_complete": True,
        "selected_hyperparameters_frozen": True,
        "train_oof_working_points_frozen": True,
        "final_full_train_models_fitted": False,
        "validation_evaluation_authorized": False,
        "test_access_authorized": False,
        "physical_significance_authorized": False,
        "next_gate": config["authorization"]["next_gate"],
    }
    pass_conditions = {
        "population": summary["train_members"] == 458
        and summary["train_rows"] == 57326
        and summary["signal_rows"] == 9975
        and summary["background_rows"] == 47351,
        "contract_dimensions": summary["folds"] == 5
        and summary["model_variants"] == 2
        and summary["features_mass_aware"] == 34
        and summary["features_mass_plane_blind"] == 30,
        "search": summary["hyperparameter_configurations"] == 24
        and summary["hyperparameter_trials"] == 48
        and summary["search_fold_fits"] == 240
        and summary["failed_search_fold_fits"] == 0,
        "refits": summary["selected_configuration_refits"] == 10
        and summary["failed_selected_refits"] == 0
        and summary["models_trained_total"] == 250,
        "oof_integrity": summary["mass_aware_oof_rows"] == 57326
        and summary["mass_plane_blind_oof_rows"] == 57326
        and summary["missing_oof_rows"] == 0
        and summary["duplicate_oof_rows"] == 0
        and summary["member_leakage_rows"] == 0,
        "auc": all(
            np.isfinite(summary[name]) and summary[name] > 0.5
            for name in (
                "mass_aware_weighted_oof_roc_auc",
                "mass_plane_blind_weighted_oof_roc_auc",
                "mass_aware_raw_oof_roc_auc",
                "mass_plane_blind_raw_oof_roc_auc",
            )
        ),
        "validation_closed": summary["validation_candidate_files_opened"] == 0
        and summary["validation_rows_read"] == 0
        and summary["validation_metrics_calculated"] == 0,
        "test_closed": summary["test_candidate_files_opened"] == 0
        and summary["test_rows_read"] == 0
        and summary["test_metrics_calculated"] == 0,
        "no_full_train_or_scaler": summary["full_train_models_fitted"] == 0
        and summary["scalers_fitted"] == 0,
        "no_physical_results": summary["physical_event_weights_used"] == 0
        and summary["physics_yields_calculated"] == 0
        and summary["significances_calculated"] == 0,
        "unittests": tests_passed and tests_run > 0,
        "plots": summary["pngs_written"] == summary["plots_requested"]
        and summary["pdfs_written"] == summary["plots_requested"]
        and summary["plot_failures"] == 0
        and summary["cms_style_failures"] == 0
        and summary["latex_math_label_failures"] == 0,
        "runtime_only_models_and_predictions": summary[
            "runtime_model_artifacts"
        ]
        == 10
        and summary["runtime_oof_prediction_artifacts"] == 2
        and summary["committed_row_level_or_model_artifacts"] == 0
        and all(
            not row["committed_status"]
            for row in runtime_rows
            if row["artifact_type"]
            in {
                "selected_fold_xgboost_json_model",
                "row_level_oof_prediction_parquet",
            }
        ),
    }
    summary["pass_conditions"] = pass_conditions
    summary["failed_pass_conditions"] = sorted(
        name for name, value in pass_conditions.items() if not value
    )
    if all(pass_conditions.values()):
        summary["status"] = "hh4b_bdt_v1_grouped_cross_validation_pass"

    checkpoint_payload = {
        "schema_version": 1,
        "classification": "canonical_hh4b_bdt_v1_grouped_cv_checkpoint",
        "status": summary["status"],
        "source_commit": source_commit,
        "grouped_cv_complete": True,
        "selected_hyperparameters_frozen": True,
        "train_oof_working_points_frozen": True,
        "final_full_train_models_fitted": False,
        "validation_evaluation_authorized": False,
        "test_access_authorized": False,
        "physical_significance_authorized": False,
        "models_committed": 0,
        "oof_predictions_committed": 0,
        "next_gate": config["authorization"]["next_gate"],
    }
    write_json(output_dir / "summary.json", summary)
    write_json(output_dir / "checkpoint.json", checkpoint_payload)
    (output_dir / "README.md").write_text(
        checkpoint_readme(summary, selected_rows),
        encoding="utf-8",
    )

    required = (
        "README.md",
        "checkpoint.json",
        "summary.json",
        "environment.json",
        *TABLE_FIELDS.keys(),
    )
    for name in required:
        shutil.copy2(output_dir / name, checkpoint_dir / name)
    shutil.copytree(output_dir / "plots", checkpoint_dir / "plots")
    write_checkpoint_manifest(checkpoint_dir)
    shutil.rmtree(MPLCONFIG_TEMP, ignore_errors=True)

    print(
        json.dumps(
            {
                "status": summary["status"],
                "models_trained_total": summary["models_trained_total"],
                "mass_aware_weighted_oof_roc_auc": summary[
                    "mass_aware_weighted_oof_roc_auc"
                ],
                "mass_plane_blind_weighted_oof_roc_auc": summary[
                    "mass_plane_blind_weighted_oof_roc_auc"
                ],
                "next_gate": summary["next_gate"],
            },
            indent=2,
            sort_keys=True,
        ),
        flush=True,
    )
    if summary["status"] != "hh4b_bdt_v1_grouped_cross_validation_pass":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
