#!/usr/bin/env bash
set -Eeuo pipefail

REPO="/uscms_data/d3/iturkmen/repos/hh-bbww-baselines"
BRANCH="delphes-hh4b-production"
CUT_AMENDMENT_HEAD="1cd21990ae07eeb503b858c1e57e7121b6a0e875"
EXPECTED_HEAD="91d8c03d63fcece89b51fd44179df11fdc1d675a"
EXPECTED_HEAD_PARENT="1cd21990ae07eeb503b858c1e57e7121b6a0e875"
EXPECTED_HEAD_SUBJECT="docs: freeze Track B Phase 2 inference namespace closure"
EXPECTED_DESCENDANT_FILE_COUNT="82"
EXPECTED_DESCENDANT_PREFIX="docs/checkpoints/track_b_phase2_inference_namespace_closure_20260806_v1/"

NULL_AUDIT_ROOT="/uscms_data/d3/iturkmen/hh4b_delphes/baselines/comparison_weight_null_pattern_audit_v1_20260806T172806Z"
NULL_AUDIT_SUMMARY="$NULL_AUDIT_ROOT/comparison_weight_null_pattern_audit_summary.json"

TARGET_AUDIT_ROOT="/uscms_data/d3/iturkmen/hh4b_delphes/baselines/multivariate_cut_target_feasibility_audit_v1_20260806T173911Z"
TARGET_AUDIT_SUMMARY="$TARGET_AUDIT_ROOT/multivariate_cut_target_feasibility_audit_summary.json"

COMMON="$REPO/docs/checkpoints/hh4b_train_common_table_full_output_freeze_20260805_v1"
AUTH="$REPO/docs/checkpoints/hh4b_train_run2_138fb_physical_authorization_freeze_20260806_v1"
BASE="$REPO/docs/checkpoints/hh4b_train_multivariate_cut_and_figure_contract_20260806_v1"
CLAR="$REPO/docs/checkpoints/hh4b_train_multivariate_cut_implementation_clarification_20260806_v1"
AMEND="$REPO/docs/checkpoints/hh4b_train_multivariate_cut_category_endpoint_amendment_20260806_v1"

PLAN="$COMMON/evidence/production_manifest/full_464_production_plan.tsv"
MANIFEST="$COMMON/evidence/production_manifest/full_464_source_worker_manifest.tsv"
AUTH_REGISTRY="$AUTH/evidence/authorization/physical_weight_authorization_registry_464.tsv"
BASE_CONTRACT="$BASE/multivariate_nested_oof_cut_contract.json"
CLAR_CONTRACT="$CLAR/multivariate_cut_implementation_clarification.json"
AMEND_CONTRACT="$AMEND/multivariate_cut_category_endpoint_amendment.json"

RUNTIME="/uscms_data/d3/iturkmen/hh4b_delphes/runtime/hh4b_broad_ml_py39_v1"
PYTHON="$RUNTIME/bin/python"

STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
OUT="/uscms_data/d3/iturkmen/hh4b_delphes/baselines/multivariate_cut_category_conditional_diagnostic_v2_${STAMP}"
LOG="$OUT/multivariate_cut_category_conditional_diagnostic_v2.log"

fail() {
    echo "ERROR: $*"
    echo "RESULT=MULTIVARIATE_CUT_CATEGORY_CONDITIONAL_DIAGNOSTIC_FAILED"
    return 1
}

run_canary() {
    echo "============================================================"
    echo "MULTIVARIATE CUT CATEGORY-CONDITIONAL DIAGNOSTIC V2"
    echo "============================================================"
    echo "EXPECTED_HEAD=$EXPECTED_HEAD"
    echo "OUT=$OUT"

    cd "$REPO"
    git fetch origin "$BRANCH"

    CURRENT_BRANCH="$(git branch --show-current)"
    LOCAL_HEAD="$(git rev-parse HEAD)"
    REMOTE_HEAD="$(git rev-parse "origin/$BRANCH")"
    AHEAD_COUNT="$(git rev-list --count "origin/$BRANCH..HEAD")"
    BEHIND_COUNT="$(git rev-list --count "HEAD..origin/$BRANCH")"
    STATUS_TEXT="$(git status --short)"

    HEAD_PARENT="$(git rev-parse "${EXPECTED_HEAD}^")"
    HEAD_SUBJECT="$(git show -s --format='%s' "$EXPECTED_HEAD")"
    DESCENDANT_PATHS="$(
        git diff             --name-only             "$CUT_AMENDMENT_HEAD..$EXPECTED_HEAD"             | sed '/^$/d'
    )"
    DESCENDANT_FILE_COUNT="$(
        printf '%s\n' "$DESCENDANT_PATHS"             | awk 'NF { count += 1 } END { print count + 0 }'
    )"
    UNEXPECTED_DESCENDANT_PATHS="$(
        printf '%s\n' "$DESCENDANT_PATHS"             | awk -v prefix="$EXPECTED_DESCENDANT_PREFIX" '
                NF > 0 && index($0, prefix) != 1 { print }
            '
    )"

    echo "CURRENT_BRANCH=$CURRENT_BRANCH"
    echo "LOCAL_HEAD=$LOCAL_HEAD"
    echo "REMOTE_HEAD=$REMOTE_HEAD"
    echo "AHEAD_COUNT=$AHEAD_COUNT"
    echo "BEHIND_COUNT=$BEHIND_COUNT"
    echo "CUT_AMENDMENT_HEAD=$CUT_AMENDMENT_HEAD"
    echo "HEAD_PARENT=$HEAD_PARENT"
    echo "EXPECTED_HEAD_PARENT=$EXPECTED_HEAD_PARENT"
    echo "HEAD_SUBJECT=$HEAD_SUBJECT"
    echo "EXPECTED_HEAD_SUBJECT=$EXPECTED_HEAD_SUBJECT"
    echo "DESCENDANT_FILE_COUNT=$DESCENDANT_FILE_COUNT"
    echo "EXPECTED_DESCENDANT_FILE_COUNT=$EXPECTED_DESCENDANT_FILE_COUNT"

    [[ "$CURRENT_BRANCH" == "$BRANCH" ]] || {
        fail "wrong branch"
        return 1
    }
    [[ "$LOCAL_HEAD" == "$EXPECTED_HEAD" ]] || {
        fail "unexpected local HEAD"
        return 1
    }
    [[ "$REMOTE_HEAD" == "$EXPECTED_HEAD" ]] || {
        fail "unexpected remote HEAD"
        return 1
    }
    [[ "$AHEAD_COUNT" -eq 0 && "$BEHIND_COUNT" -eq 0 ]] || {
        fail "local and remote are not synchronized"
        return 1
    }
    git merge-base --is-ancestor "$CUT_AMENDMENT_HEAD" "$EXPECTED_HEAD" || {
        fail "current head is not a descendant of the cut amendment"
        return 1
    }
    [[ "$HEAD_PARENT" == "$EXPECTED_HEAD_PARENT" ]] || {
        fail "unexpected descendant commit parent"
        return 1
    }
    [[ "$HEAD_SUBJECT" == "$EXPECTED_HEAD_SUBJECT" ]] || {
        fail "unexpected descendant commit subject"
        return 1
    }
    [[ "$DESCENDANT_FILE_COUNT" == "$EXPECTED_DESCENDANT_FILE_COUNT" ]] || {
        fail "unexpected descendant changed-file count"
        return 1
    }
    [[ -z "$UNEXPECTED_DESCENDANT_PATHS" ]] || {
        echo "UNEXPECTED_DESCENDANT_PATHS:"
        printf '%s\n' "$UNEXPECTED_DESCENDANT_PATHS"
        fail "descendant commit modified paths outside the Track B checkpoint"
        return 1
    }
    git diff --quiet         "$CUT_AMENDMENT_HEAD"         "$EXPECTED_HEAD"         --         docs/checkpoints/hh4b_train_multivariate_cut_and_figure_contract_20260806_v1         docs/checkpoints/hh4b_train_multivariate_cut_implementation_clarification_20260806_v1         docs/checkpoints/hh4b_train_multivariate_cut_category_endpoint_amendment_20260806_v1 || {
        fail "protected cut-contract paths changed in descendant commit"
        return 1
    }
    [[ -z "$STATUS_TEXT" ]] || {
        printf '%s\n' "$STATUS_TEXT"
        fail "worktree is not clean"
        return 1
    }

    for checkpoint in "$COMMON" "$AUTH" "$BASE" "$CLAR" "$AMEND"; do
        (
            cd "$checkpoint"
            sha256sum -c SHA256SUMS
        ) || {
            fail "checkpoint checksum verification failed: $checkpoint"
            return 1
        }
    done

    for audit_root in "$NULL_AUDIT_ROOT" "$TARGET_AUDIT_ROOT"; do
        (
            cd "$audit_root"
            sha256sum -c SHA256SUMS
        ) || {
            fail "external audit checksum verification failed: $audit_root"
            return 1
        }
    done

    for path in \
        "$PLAN" \
        "$MANIFEST" \
        "$AUTH_REGISTRY" \
        "$BASE_CONTRACT" \
        "$CLAR_CONTRACT" \
        "$AMEND_CONTRACT" \
        "$NULL_AUDIT_SUMMARY" \
        "$TARGET_AUDIT_SUMMARY" \
        "$PYTHON"
    do
        [[ -f "$path" ]] || {
            fail "required input missing: $path"
            return 1
        }
        echo "PASS_INPUT=$path"
    done

    echo "DESCENDANT_TRACK_B_ONLY_SCOPE=PASS"
    echo "PROTECTED_CUT_CONTRACT_PATHS_UNCHANGED=PASS"
    echo "ALL_INPUT_CHECKSUMS=PASS"
    echo "CATEGORY_ENDPOINT_AMENDMENT=PASS"
    echo "WORKTREE_CLEAN=PASS"

    PLAN_PATH="$PLAN" \
    MANIFEST_PATH="$MANIFEST" \
    AUTH_REGISTRY_PATH="$AUTH_REGISTRY" \
    BASE_CONTRACT_PATH="$BASE_CONTRACT" \
    CLAR_CONTRACT_PATH="$CLAR_CONTRACT" \
    AMEND_CONTRACT_PATH="$AMEND_CONTRACT" \
    NULL_AUDIT_SUMMARY_PATH="$NULL_AUDIT_SUMMARY" \
    TARGET_AUDIT_SUMMARY_PATH="$TARGET_AUDIT_SUMMARY" \
    EXPECTED_HEAD_VALUE="$EXPECTED_HEAD" \
    OUTPUT_ROOT="$OUT" \
    "$PYTHON" - <<'PY'
from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from pathlib import Path
from typing import Any
import hashlib
import json
import math
import os
import time

import numpy as np
import pandas as pd
import pyarrow.parquet as pq


TARGET_SIGNAL_EFFICIENCY = 0.585957
OUTER_FOLD = 0
ROWS_PER_SOURCE_CATEGORY = 64
HASH_KEY = "0123456789123456"
QUANTILE_PROBABILITIES = np.asarray(
    [0.125, 0.25, 0.375, 0.5, 0.625, 0.75, 0.875],
    dtype=float,
)
BEAM_WIDTH = 16
REFINEMENT_STARTS = 4
MAX_COORDINATE_PASSES = 2

BASE_COLUMNS = [
    "source_uid",
    "group_id",
    "source_fold",
    "event_uid",
    "sample_class",
    "process_or_mode",
    "auxiliary_qcd",
    "physical_evaluation_eligible",
    "candidate_tagged_jet_count",
    "mbb1",
    "mbb2",
    "r_hh_125_125",
    "ht_candidate_jets",
    "h2_pt",
    "drbb1",
    "drbb2",
    "mhh",
    "abs_h_delta_eta",
    "comparison_weight_outer_fold_0",
    "resolved_selection_contribution_weight",
]


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def clean(value: Any) -> str:
    text = str(value).strip()
    if text.lower() in {"", "nan", "none", "null"}:
        return ""
    return text


def truthy(value: Any) -> bool:
    return clean(value).lower() in {"true", "1", "yes", "y"}


def canonical_sha256(payload: Any) -> str:
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def threshold_json(thresholds: dict[str, float]) -> str:
    return json.dumps(
        {key: float(thresholds[key]) for key in sorted(thresholds)},
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def deterministic_event_order(values: pd.Series) -> np.ndarray:
    hashes = pd.util.hash_pandas_object(
        values.astype(str),
        index=False,
        hash_key=HASH_KEY,
    ).to_numpy(dtype=np.uint64)
    return np.argsort(hashes, kind="mergesort")


@dataclass(frozen=True)
class CutSpec:
    variable: str
    operator: str


@dataclass(frozen=True)
class Configuration:
    category_family_id: str
    category_id: str
    structure_id: str
    cuts: tuple[CutSpec, ...]


@dataclass(frozen=True)
class Metric:
    signal_efficiency: float
    background_efficiency: float
    selected_signal_rows: int
    selected_background_rows: int
    selected_signal_weight: float
    selected_background_weight: float
    feasible: bool


def category_mask(frame: pd.DataFrame, category_id: str) -> np.ndarray:
    tagged = frame["candidate_tagged_jet_count"].to_numpy(dtype=np.int64)
    if category_id == "exact3tag":
        return tagged == 3
    if category_id == "ge4tag":
        return tagged >= 4
    raise RuntimeError(f"unknown category: {category_id}")


def selection_mask(
    frame: pd.DataFrame,
    configuration: Configuration,
    thresholds: dict[str, float],
) -> np.ndarray:
    mask = np.ones(len(frame), dtype=bool)
    for cut in configuration.cuts:
        values = frame[cut.variable].to_numpy(dtype=float)
        threshold = float(thresholds[cut.variable])
        if cut.operator == "<":
            mask &= values < threshold
        elif cut.operator == ">":
            mask &= values > threshold
        else:
            raise RuntimeError(f"unsupported operator: {cut.operator}")
    return mask


def weighted_metric(
    frame: pd.DataFrame,
    mask: np.ndarray,
    weight_column: str,
) -> Metric:
    classes = frame["sample_class"].to_numpy(dtype=str)
    weights = frame[weight_column].to_numpy(dtype=float)

    require(np.isfinite(weights).all(), "nonfinite comparison weight")
    require((weights >= 0.0).all(), "negative comparison weight")

    signal = classes == "signal"
    background = classes == "background"
    total_signal = float(weights[signal].sum())
    total_background = float(weights[background].sum())
    require(total_signal > 0.0, "nonpositive category signal denominator")
    require(total_background > 0.0, "nonpositive category background denominator")

    selected_signal_weight = float(weights[mask & signal].sum())
    selected_background_weight = float(weights[mask & background].sum())
    signal_efficiency = selected_signal_weight / total_signal
    background_efficiency = selected_background_weight / total_background

    require(math.isfinite(signal_efficiency), "nonfinite signal efficiency")
    require(
        math.isfinite(background_efficiency),
        "nonfinite background efficiency",
    )

    return Metric(
        signal_efficiency=signal_efficiency,
        background_efficiency=background_efficiency,
        selected_signal_rows=int(np.count_nonzero(mask & signal)),
        selected_background_rows=int(np.count_nonzero(mask & background)),
        selected_signal_weight=selected_signal_weight,
        selected_background_weight=selected_background_weight,
        feasible=signal_efficiency >= TARGET_SIGNAL_EFFICIENCY,
    )


def metric_rank(
    metric: Metric,
    thresholds: dict[str, float],
) -> tuple[Any, ...]:
    threshold_tuple = tuple(
        (key, float(thresholds[key])) for key in sorted(thresholds)
    )
    if metric.feasible:
        return (
            0,
            metric.background_efficiency,
            metric.signal_efficiency - TARGET_SIGNAL_EFFICIENCY,
            threshold_tuple,
        )
    return (
        1,
        -metric.signal_efficiency,
        metric.background_efficiency,
        threshold_tuple,
    )


def configuration_rank(
    metric: Metric,
    configuration: Configuration,
) -> tuple[Any, ...]:
    if metric.feasible:
        return (
            0,
            metric.background_efficiency,
            metric.signal_efficiency - TARGET_SIGNAL_EFFICIENCY,
            len(configuration.cuts),
            configuration.structure_id,
        )
    return (
        1,
        -metric.signal_efficiency,
        metric.background_efficiency,
        len(configuration.cuts),
        configuration.structure_id,
    )


def exact_coordinate_candidates(
    frame: pd.DataFrame,
    configuration: Configuration,
    current: dict[str, float],
    coordinate: CutSpec,
    weight_column: str,
) -> tuple[dict[str, float], Metric]:
    base_mask = np.ones(len(frame), dtype=bool)
    for cut in configuration.cuts:
        if cut.variable == coordinate.variable:
            continue
        values = frame[cut.variable].to_numpy(dtype=float)
        threshold = float(current[cut.variable])
        if cut.operator == "<":
            base_mask &= values < threshold
        elif cut.operator == ">":
            base_mask &= values > threshold
        else:
            raise RuntimeError(f"unsupported operator: {cut.operator}")

    values = frame[coordinate.variable].to_numpy(dtype=float)
    classes = frame["sample_class"].to_numpy(dtype=str)
    weights = frame[weight_column].to_numpy(dtype=float)
    signal = classes == "signal"
    background = classes == "background"
    total_signal = float(weights[signal].sum())
    total_background = float(weights[background].sum())

    active_indices = np.flatnonzero(base_mask)
    require(
        active_indices.size > 0,
        f"no rows survive fixed coordinates for {coordinate.variable}",
    )

    order = np.argsort(values[active_indices], kind="mergesort")
    sorted_indices = active_indices[order]
    sorted_x = values[sorted_indices]
    sorted_signal_weight = np.where(
        signal[sorted_indices], weights[sorted_indices], 0.0
    )
    sorted_background_weight = np.where(
        background[sorted_indices], weights[sorted_indices], 0.0
    )
    sorted_signal_rows = signal[sorted_indices].astype(np.int64)
    sorted_background_rows = background[sorted_indices].astype(np.int64)

    unique_values, first_positions, counts = np.unique(
        sorted_x,
        return_index=True,
        return_counts=True,
    )
    last_positions = first_positions + counts - 1
    cumulative_signal_weight = np.cumsum(sorted_signal_weight)
    cumulative_background_weight = np.cumsum(sorted_background_weight)
    cumulative_signal_rows = np.cumsum(sorted_signal_rows)
    cumulative_background_rows = np.cumsum(sorted_background_rows)

    if coordinate.operator == "<":
        selected_signal_weight = cumulative_signal_weight[last_positions]
        selected_background_weight = cumulative_background_weight[last_positions]
        selected_signal_rows = cumulative_signal_rows[last_positions]
        selected_background_rows = cumulative_background_rows[last_positions]
        candidate_thresholds = np.nextafter(unique_values, np.inf)
    elif coordinate.operator == ">":
        total_active_signal_weight = cumulative_signal_weight[-1]
        total_active_background_weight = cumulative_background_weight[-1]
        total_active_signal_rows = cumulative_signal_rows[-1]
        total_active_background_rows = cumulative_background_rows[-1]

        before = first_positions - 1
        signal_before = np.where(
            before >= 0,
            cumulative_signal_weight[np.maximum(before, 0)],
            0.0,
        )
        background_before = np.where(
            before >= 0,
            cumulative_background_weight[np.maximum(before, 0)],
            0.0,
        )
        signal_rows_before = np.where(
            before >= 0,
            cumulative_signal_rows[np.maximum(before, 0)],
            0,
        )
        background_rows_before = np.where(
            before >= 0,
            cumulative_background_rows[np.maximum(before, 0)],
            0,
        )
        selected_signal_weight = total_active_signal_weight - signal_before
        selected_background_weight = (
            total_active_background_weight - background_before
        )
        selected_signal_rows = total_active_signal_rows - signal_rows_before
        selected_background_rows = (
            total_active_background_rows - background_rows_before
        )
        candidate_thresholds = np.nextafter(unique_values, -np.inf)
    else:
        raise RuntimeError(
            f"unsupported coordinate operator: {coordinate.operator}"
        )

    signal_efficiency = selected_signal_weight / total_signal
    background_efficiency = selected_background_weight / total_background
    feasible = signal_efficiency >= TARGET_SIGNAL_EFFICIENCY

    best_index = None
    best_rank = None
    for index in range(len(candidate_thresholds)):
        candidate_threshold_map = dict(current)
        candidate_threshold_map[coordinate.variable] = float(
            candidate_thresholds[index]
        )
        candidate_metric = Metric(
            signal_efficiency=float(signal_efficiency[index]),
            background_efficiency=float(background_efficiency[index]),
            selected_signal_rows=int(selected_signal_rows[index]),
            selected_background_rows=int(selected_background_rows[index]),
            selected_signal_weight=float(selected_signal_weight[index]),
            selected_background_weight=float(selected_background_weight[index]),
            feasible=bool(feasible[index]),
        )
        rank = metric_rank(candidate_metric, candidate_threshold_map)
        if best_rank is None or rank < best_rank:
            best_rank = rank
            best_index = index

    require(best_index is not None, "coordinate scan returned no result")
    best_thresholds = dict(current)
    best_thresholds[coordinate.variable] = float(
        candidate_thresholds[best_index]
    )
    best_metric = weighted_metric(
        frame,
        selection_mask(frame, configuration, best_thresholds),
        weight_column,
    )
    return best_thresholds, best_metric


def optimize_configuration(
    frame: pd.DataFrame,
    configuration: Configuration,
    weight_column: str,
) -> tuple[dict[str, float], Metric, dict[str, Any]]:
    require(len(frame) > 0, "empty category optimization frame")
    require(
        set(frame["sample_class"].astype(str)) == {"signal", "background"},
        "category optimization frame lacks a class",
    )

    grids: dict[str, list[float]] = {}
    for cut in configuration.cuts:
        values = frame[cut.variable].to_numpy(dtype=float)
        require(np.isfinite(values).all(), f"nonfinite values in {cut.variable}")
        quantiles = np.quantile(values, QUANTILE_PROBABILITIES)
        if cut.operator == "<":
            candidates = [
                *np.nextafter(quantiles, np.inf).tolist(),
                float(np.nextafter(values.max(), np.inf)),
            ]
        elif cut.operator == ">":
            candidates = [
                *np.nextafter(quantiles, -np.inf).tolist(),
                float(np.nextafter(values.min(), -np.inf)),
            ]
        else:
            raise RuntimeError(f"unsupported operator: {cut.operator}")
        grids[cut.variable] = sorted({float(value) for value in candidates})

    variables = [cut.variable for cut in configuration.cuts]
    coarse_results: list[
        tuple[tuple[Any, ...], dict[str, float], Metric]
    ] = []
    for values in product(*(grids[variable] for variable in variables)):
        thresholds = {
            variable: float(value)
            for variable, value in zip(variables, values)
        }
        metric = weighted_metric(
            frame,
            selection_mask(frame, configuration, thresholds),
            weight_column,
        )
        coarse_results.append(
            (metric_rank(metric, thresholds), thresholds, metric)
        )

    coarse_results.sort(key=lambda item: item[0])
    beam = coarse_results[:BEAM_WIDTH]
    starts = beam[:REFINEMENT_STARTS]
    require(starts, "optimizer produced no starting configurations")

    refined_results: list[
        tuple[tuple[Any, ...], dict[str, float], Metric, int]
    ] = []
    for _, initial_thresholds, initial_metric in starts:
        current_thresholds = dict(initial_thresholds)
        current_metric = initial_metric
        passes_completed = 0

        for _ in range(MAX_COORDINATE_PASSES):
            improved = False
            for coordinate in configuration.cuts:
                candidate_thresholds, candidate_metric = (
                    exact_coordinate_candidates(
                        frame,
                        configuration,
                        current_thresholds,
                        coordinate,
                        weight_column,
                    )
                )
                if metric_rank(
                    candidate_metric, candidate_thresholds
                ) < metric_rank(current_metric, current_thresholds):
                    current_thresholds = candidate_thresholds
                    current_metric = candidate_metric
                    improved = True
            passes_completed += 1
            if not improved:
                break

        refined_results.append(
            (
                metric_rank(current_metric, current_thresholds),
                current_thresholds,
                current_metric,
                passes_completed,
            )
        )

    refined_results.sort(key=lambda item: item[0])
    _, best_thresholds, best_metric, passes_completed = refined_results[0]
    require(
        best_metric.feasible,
        f"configuration did not reach conditional target: "
        f"{configuration.category_family_id}",
    )

    return best_thresholds, best_metric, {
        "grid_sizes": {
            variable: len(grids[variable]) for variable in variables
        },
        "coarse_candidates": len(coarse_results),
        "beam_width_used": len(beam),
        "refinement_starts_used": len(starts),
        "coordinate_passes_completed": passes_completed,
        "open_boundary_seed_included": True,
    }


def metric_to_dict(metric: Metric) -> dict[str, Any]:
    return {
        "signal_efficiency": metric.signal_efficiency,
        "background_efficiency": metric.background_efficiency,
        "selected_signal_rows": metric.selected_signal_rows,
        "selected_background_rows": metric.selected_background_rows,
        "selected_signal_weight": metric.selected_signal_weight,
        "selected_background_weight": metric.selected_background_weight,
        "feasible": metric.feasible,
    }


output_root = Path(os.environ["OUTPUT_ROOT"])
expected_head = os.environ["EXPECTED_HEAD_VALUE"]

base_contract = json.loads(Path(os.environ["BASE_CONTRACT_PATH"]).read_text())
clarification = json.loads(Path(os.environ["CLAR_CONTRACT_PATH"]).read_text())
amendment = json.loads(Path(os.environ["AMEND_CONTRACT_PATH"]).read_text())
null_audit = json.loads(
    Path(os.environ["NULL_AUDIT_SUMMARY_PATH"]).read_text()
)
target_audit = json.loads(
    Path(os.environ["TARGET_AUDIT_SUMMARY_PATH"]).read_text()
)

require(
    base_contract["status"] == "pass_frozen_before_multivariate_scan",
    "base contract status changed",
)
require(
    clarification["status"] == "pass_frozen_before_canary",
    "clarification status changed",
)
require(
    amendment["status"]
    == "pass_frozen_before_category_conditional_canary",
    "category-endpoint amendment status changed",
)
require(
    null_audit["status"]
    == "pass_exact_fold_local_null_sentinel_confirmed",
    "null-pattern audit did not pass",
)
require(
    target_audit["status"] == "pass_target_feasibility_diagnosed",
    "target-feasibility audit did not pass",
)
require(
    target_audit["full_population"][
        "all_five_outer_splits_target_feasible"
    ] is False,
    "old endpoint infeasibility evidence changed",
)
require(
    amendment["primary_endpoint"][
        "target_category_conditional_signal_efficiency"
    ] == TARGET_SIGNAL_EFFICIENCY,
    "category-conditional target changed",
)
require(
    amendment["category_program"]["categories_are_mutually_exclusive"]
    is True,
    "category exclusivity changed",
)

sentinel_ids = amendment["canary_v3_contract"][
    "sentinel_category_family_configurations"
]
require(len(sentinel_ids) == 6, "sentinel configuration count changed")

configuration_registry = {
    row["category_family_id"]: Configuration(
        category_family_id=row["category_family_id"],
        category_id=row["category_id"],
        structure_id=row["structure_id"],
        cuts=tuple(
            CutSpec(variable=cut["variable"], operator=cut["operator"])
            for cut in row["continuous_cuts"]
        ),
    )
    for row in amendment["category_program"][
        "category_family_configurations"
    ]
}
require(
    set(sentinel_ids).issubset(configuration_registry),
    "sentinel configuration missing from amendment registry",
)
sentinel_configurations = [
    configuration_registry[configuration_id]
    for configuration_id in sentinel_ids
]
require(
    {configuration.category_id for configuration in sentinel_configurations}
    == {"exact3tag", "ge4tag"},
    "both categories are not represented",
)

plan = pd.read_csv(
    os.environ["PLAN_PATH"], sep="\t", dtype=str, keep_default_na=False
)
manifest = pd.read_csv(
    os.environ["MANIFEST_PATH"], sep="\t", dtype=str, keep_default_na=False
)
authorization = pd.read_csv(
    os.environ["AUTH_REGISTRY_PATH"],
    sep="\t",
    dtype=str,
    keep_default_na=False,
)

for name, frame in [
    ("plan", plan),
    ("manifest", manifest),
    ("authorization", authorization),
]:
    frame["production_row_index"] = pd.to_numeric(
        frame["production_row_index"], errors="raise"
    ).astype(int)
    require(len(frame) == 464, f"{name} row count changed")
    require(
        frame["production_row_index"].is_unique,
        f"{name} production index is not unique",
    )

merged = (
    manifest.merge(
        plan[["production_row_index", "source_uid", "final_resolved_path"]],
        on=["production_row_index", "source_uid"],
        how="inner",
        validate="one_to_one",
    )
    .merge(
        authorization[
            [
                "production_row_index",
                "source_uid",
                "physical_weight_application_authorized",
            ]
        ],
        on=["production_row_index", "source_uid"],
        how="inner",
        validate="one_to_one",
    )
)
require(len(merged) == 464, "merged source registry row count changed")
merged["source_fold"] = pd.to_numeric(
    merged["optimized_fold_k5"], errors="raise"
).astype(int)
merged["is_auxiliary"] = merged["auxiliary_qcd"].map(truthy)
merged["is_authorized"] = merged[
    "physical_weight_application_authorized"
].map(truthy)
primary = merged.loc[
    (~merged["is_auxiliary"]) & merged["is_authorized"]
].copy()
require(len(primary) == 441, "authorized primary source count changed")
require(
    set(primary["sample_class"]) == {"signal", "background"},
    "unexpected sample classes",
)
require(
    set(primary["source_fold"]) == set(range(5)),
    "five-fold source coverage changed",
)

sample_parts = []
sample_manifest_rows = []
for row in primary.itertuples(index=False):
    path = Path(clean(row.final_resolved_path))
    require(path.is_file(), f"missing resolved Parquet: {path}")
    schema_names = set(pq.read_schema(path).names)
    require(
        set(BASE_COLUMNS).issubset(schema_names),
        f"canary columns missing: {row.source_uid}",
    )

    frame = pd.read_parquet(path, columns=BASE_COLUMNS)
    require(
        len(frame) == int(row.broad_eligible_rows),
        f"resolved row closure failed: {row.source_uid}",
    )
    require(
        frame["source_uid"].astype(str).map(clean).eq(clean(row.source_uid)).all(),
        f"source UID changed inside {row.source_uid}",
    )
    require(
        frame["group_id"].astype(str).map(clean).eq(clean(row.group_id)).all(),
        f"group ID changed inside {row.source_uid}",
    )
    require(
        pd.to_numeric(frame["source_fold"], errors="raise")
        .astype(int)
        .eq(int(row.source_fold))
        .all(),
        f"source fold changed inside {row.source_uid}",
    )
    require(
        frame["sample_class"].astype(str).eq(clean(row.sample_class)).all(),
        f"sample class changed inside {row.source_uid}",
    )
    require(
        ~frame["auxiliary_qcd"].astype(bool).any(),
        f"auxiliary row entered canary: {row.source_uid}",
    )
    require(
        frame["physical_evaluation_eligible"].astype(bool).all(),
        f"physically ineligible row entered canary: {row.source_uid}",
    )

    tagged = pd.to_numeric(
        frame["candidate_tagged_jet_count"], errors="raise"
    ).astype(int)
    frame = frame.copy()
    frame["candidate_tagged_jet_count"] = tagged

    sampled_count = 0
    for category_id, category_selector in [
        ("exact3tag", tagged == 3),
        ("ge4tag", tagged >= 4),
    ]:
        category_frame = frame.loc[category_selector].copy()
        if len(category_frame) == 0:
            sample_manifest_rows.append(
                {
                    "production_row_index": int(row.production_row_index),
                    "source_uid": clean(row.source_uid),
                    "group_id": clean(row.group_id),
                    "source_fold": int(row.source_fold),
                    "sample_class": clean(row.sample_class),
                    "category_id": category_id,
                    "available_category_rows": 0,
                    "sampled_category_rows": 0,
                    "resolved_path": str(path),
                }
            )
            continue
        order = deterministic_event_order(category_frame["event_uid"])
        keep = order[: min(ROWS_PER_SOURCE_CATEGORY, len(order))]
        sampled = category_frame.iloc[keep].copy()
        sampled["canary_category_id"] = category_id
        sample_parts.append(sampled)
        sampled_count += len(sampled)
        sample_manifest_rows.append(
            {
                "production_row_index": int(row.production_row_index),
                "source_uid": clean(row.source_uid),
                "group_id": clean(row.group_id),
                "source_fold": int(row.source_fold),
                "sample_class": clean(row.sample_class),
                "category_id": category_id,
                "available_category_rows": len(category_frame),
                "sampled_category_rows": len(sampled),
                "resolved_path": str(path),
            }
        )

require(sample_parts, "category-stratified canary sample is empty")
sample = pd.concat(sample_parts, ignore_index=True)
sample["source_fold"] = pd.to_numeric(
    sample["source_fold"], errors="raise"
).astype(int)
sample["candidate_tagged_jet_count"] = pd.to_numeric(
    sample["candidate_tagged_jet_count"], errors="raise"
).astype(int)

numeric_columns = [
    "mbb1",
    "mbb2",
    "r_hh_125_125",
    "ht_candidate_jets",
    "h2_pt",
    "drbb1",
    "drbb2",
    "mhh",
    "abs_h_delta_eta",
    "resolved_selection_contribution_weight",
]
for column in numeric_columns:
    sample[column] = pd.to_numeric(sample[column], errors="raise").astype(float)
    require(
        np.isfinite(sample[column].to_numpy()).all(),
        f"nonfinite sampled values in {column}",
    )

sample["comparison_weight_outer_fold_0"] = pd.to_numeric(
    sample["comparison_weight_outer_fold_0"], errors="coerce"
).astype(float)
sample["abs_mbb1_minus_125"] = np.abs(sample["mbb1"] - 125.0)
sample["abs_mbb2_minus_125"] = np.abs(sample["mbb2"] - 125.0)
sample["max_abs_mbb_minus_125"] = np.maximum(
    sample["abs_mbb1_minus_125"], sample["abs_mbb2_minus_125"]
)
sample["max_drbb"] = np.maximum(sample["drbb1"], sample["drbb2"])

require(
    sample["event_uid"].astype(str).is_unique,
    "sampled event UIDs are not globally unique",
)
require(
    (
        (sample["canary_category_id"] == "exact3tag")
        == (sample["candidate_tagged_jet_count"] == 3)
    ).all(),
    "exact3tag category assignment mismatch",
)
require(
    (
        (sample["canary_category_id"] == "ge4tag")
        == (sample["candidate_tagged_jet_count"] >= 4)
    ).all(),
    "ge4tag category assignment mismatch",
)

outer = sample.loc[sample["source_fold"] == OUTER_FOLD].copy()
development = sample.loc[sample["source_fold"] != OUTER_FOLD].copy()
require(
    set(development["source_fold"]) == {1, 2, 3, 4},
    "inner development folds are incomplete",
)
require(set(outer["source_fold"]) == {0}, "outer canary fold is incorrect")
require(
    development["comparison_weight_outer_fold_0"].notna().all(),
    "development contains null fold-0 comparison weights",
)
require(
    np.isfinite(
        development["comparison_weight_outer_fold_0"].to_numpy(dtype=float)
    ).all(),
    "development comparison weights are nonfinite",
)
require(
    (
        development["comparison_weight_outer_fold_0"].to_numpy(dtype=float)
        >= 0.0
    ).all(),
    "development comparison weights are negative",
)
require(
    outer["comparison_weight_outer_fold_0"].isna().all(),
    "outer fold comparison-weight null sentinel failed",
)

fold_group_sets = {
    fold: set(sample.loc[sample["source_fold"] == fold, "group_id"].astype(str))
    for fold in range(5)
}
for left in range(5):
    for right in range(left + 1, 5):
        require(
            fold_group_sets[left].isdisjoint(fold_group_sets[right]),
            f"group leakage between folds {left} and {right}",
        )

support_rows = []
for fold in range(5):
    fold_frame = sample.loc[sample["source_fold"] == fold]
    for category_id in ["exact3tag", "ge4tag"]:
        category_frame = fold_frame.loc[
            fold_frame["canary_category_id"] == category_id
        ]
        for sample_class in ["signal", "background"]:
            class_frame = category_frame.loc[
                category_frame["sample_class"] == sample_class
            ]
            support_rows.append(
                {
                    "source_fold": fold,
                    "category_id": category_id,
                    "sample_class": sample_class,
                    "sources": class_frame["source_uid"].nunique(),
                    "groups": class_frame["group_id"].nunique(),
                    "rows": len(class_frame),
                    "comparison_weight_sum": (
                        ""
                        if fold == OUTER_FOLD
                        else float(
                            class_frame[
                                "comparison_weight_outer_fold_0"
                            ].sum()
                        )
                    ),
                    "physical_weight_sum": float(
                        class_frame[
                            "resolved_selection_contribution_weight"
                        ].sum()
                    ),
                }
            )
            require(
                len(class_frame) > 0,
                f"empty canary cell fold={fold} category={category_id} "
                f"class={sample_class}",
            )

sample_manifest = pd.DataFrame(sample_manifest_rows).sort_values(
    ["production_row_index", "category_id"], kind="mergesort"
)
sample_manifest.to_csv(
    output_root / "diagnostic_source_category_sample_manifest.tsv",
    sep="\t",
    index=False,
)
pd.DataFrame(support_rows).to_csv(
    output_root / "diagnostic_fold_category_class_support.tsv",
    sep="\t",
    index=False,
)


def run_nested_once() -> dict[str, Any]:
    category_results = {}
    inner_records = []
    pooled_records = []
    outer_category_masks = {}

    for category_id in ["exact3tag", "ge4tag"]:
        category_configurations = [
            configuration
            for configuration in sentinel_configurations
            if configuration.category_id == category_id
        ]
        require(
            len(category_configurations) == 3,
            f"sentinel configuration count changed in {category_id}",
        )

        category_development = development.loc[
            development["canary_category_id"] == category_id
        ].copy()
        category_outer = outer.loc[
            outer["canary_category_id"] == category_id
        ].copy()
        require(len(category_development) > 0, f"empty development {category_id}")
        require(len(category_outer) > 0, f"empty outer {category_id}")

        pooled_masks = {
            configuration.category_family_id: pd.Series(
                False, index=category_development.index, dtype=bool
            )
            for configuration in category_configurations
        }

        for inner_held_out_fold in [1, 2, 3, 4]:
            train = category_development.loc[
                category_development["source_fold"] != inner_held_out_fold
            ].copy()
            held_out = category_development.loc[
                category_development["source_fold"] == inner_held_out_fold
            ].copy()
            require(
                set(train["source_fold"])
                == ({1, 2, 3, 4} - {inner_held_out_fold}),
                "inner training fold set changed",
            )
            require(
                set(held_out["source_fold"]) == {inner_held_out_fold},
                "inner held-out fold changed",
            )
            require(
                set(train["group_id"]).isdisjoint(set(held_out["group_id"])),
                "inner group leakage",
            )
            require(
                set(train["group_id"]).isdisjoint(set(category_outer["group_id"])),
                "outer groups entered inner optimization",
            )

            for configuration in category_configurations:
                started = time.time()
                thresholds, train_metric, optimizer_audit = (
                    optimize_configuration(
                        train,
                        configuration,
                        "comparison_weight_outer_fold_0",
                    )
                )
                held_out_mask = selection_mask(
                    held_out, configuration, thresholds
                )
                held_out_metric = weighted_metric(
                    held_out,
                    held_out_mask,
                    "comparison_weight_outer_fold_0",
                )
                recomputed_mask = selection_mask(
                    held_out,
                    configuration,
                    json.loads(threshold_json(thresholds)),
                )
                require(
                    np.array_equal(held_out_mask, recomputed_mask),
                    "inner held-out selected mask was not reproducible",
                )
                pooled_masks[configuration.category_family_id].loc[
                    held_out.index
                ] = held_out_mask
                inner_records.append(
                    {
                        "category_id": category_id,
                        "inner_held_out_fold": inner_held_out_fold,
                        "category_family_id": configuration.category_family_id,
                        "structure_id": configuration.structure_id,
                        "continuous_variables": [
                            cut.variable for cut in configuration.cuts
                        ],
                        "thresholds": {
                            key: float(value)
                            for key, value in sorted(thresholds.items())
                        },
                        "training_metric": metric_to_dict(train_metric),
                        "held_out_metric": metric_to_dict(held_out_metric),
                        "optimizer_audit": optimizer_audit,
                        "elapsed_seconds": time.time() - started,
                    }
                )

        candidates = []
        ordered_development = category_development.sort_index()
        for configuration in category_configurations:
            mask_series = pooled_masks[
                configuration.category_family_id
            ].sort_index()
            require(
                mask_series.index.equals(ordered_development.index),
                "pooled inner mask index mismatch",
            )
            pooled_metric = weighted_metric(
                ordered_development,
                mask_series.to_numpy(dtype=bool),
                "comparison_weight_outer_fold_0",
            )
            pooled_record = {
                "category_id": category_id,
                "category_family_id": configuration.category_family_id,
                "structure_id": configuration.structure_id,
                "continuous_variables": [
                    cut.variable for cut in configuration.cuts
                ],
                "pooled_inner_oof_metric": metric_to_dict(pooled_metric),
            }
            pooled_records.append(pooled_record)
            candidates.append(
                (
                    configuration_rank(pooled_metric, configuration),
                    configuration,
                    pooled_metric,
                )
            )

        feasible = [
            candidate for candidate in candidates if candidate[2].feasible
        ]
        category_target_reached = bool(feasible)
        ranked_candidates = feasible if feasible else candidates
        ranked_candidates.sort(key=lambda item: item[0])
        chosen_configuration = ranked_candidates[0][1]

        open_control_mask = np.ones(
            len(category_development),
            dtype=bool,
        )
        open_control_metric = weighted_metric(
            category_development,
            open_control_mask,
            "comparison_weight_outer_fold_0",
        )
        require(
            abs(open_control_metric.signal_efficiency - 1.0) < 1.0e-12,
            f"open-control signal efficiency is not unity in {category_id}",
        )
        require(
            abs(open_control_metric.background_efficiency - 1.0) < 1.0e-12,
            f"open-control background efficiency is not unity in {category_id}",
        )

        refit_thresholds, refit_metric, refit_audit = optimize_configuration(
            category_development,
            chosen_configuration,
            "comparison_weight_outer_fold_0",
        )
        outer_mask = selection_mask(
            category_outer, chosen_configuration, refit_thresholds
        )
        outer_mask_recomputed = selection_mask(
            category_outer,
            chosen_configuration,
            json.loads(threshold_json(refit_thresholds)),
        )
        require(
            np.array_equal(outer_mask, outer_mask_recomputed),
            "outer category selected mask was not reproducible",
        )

        classes = category_outer["sample_class"].to_numpy(dtype=str)
        physical = category_outer[
            "resolved_selection_contribution_weight"
        ].to_numpy(dtype=float)
        require(np.isfinite(physical).all(), "nonfinite outer physical weight")
        signal = classes == "signal"
        background = classes == "background"

        pooled_category_metrics = [
            row["pooled_inner_oof_metric"]
            for row in pooled_records
            if row["category_id"] == category_id
        ]
        require(
            len(pooled_category_metrics) == 3,
            f"pooled sentinel metric count changed in {category_id}",
        )

        metrics = {
            "category_id": category_id,
            "category_target_reached_by_any_sentinel": (
                category_target_reached
            ),
            "maximum_pooled_inner_oof_signal_efficiency": max(
                metric["signal_efficiency"]
                for metric in pooled_category_metrics
            ),
            "minimum_pooled_inner_oof_signal_efficiency": min(
                metric["signal_efficiency"]
                for metric in pooled_category_metrics
            ),
            "open_control_metric": metric_to_dict(open_control_metric),
            "diagnostic_fallback_used": not category_target_reached,
            "chosen_category_family_id": (
                chosen_configuration.category_family_id
            ),
            "chosen_structure_id": chosen_configuration.structure_id,
            "refit_thresholds": {
                key: float(value)
                for key, value in sorted(refit_thresholds.items())
            },
            "refit_development_metric": metric_to_dict(refit_metric),
            "refit_optimizer_audit": refit_audit,
            "outer_rows": len(category_outer),
            "outer_selected_rows": int(np.count_nonzero(outer_mask)),
            "outer_signal_rows": int(np.count_nonzero(signal)),
            "outer_background_rows": int(np.count_nonzero(background)),
            "outer_selected_signal_rows": int(
                np.count_nonzero(outer_mask & signal)
            ),
            "outer_selected_background_rows": int(
                np.count_nonzero(outer_mask & background)
            ),
            "signal_yield_pre": float(physical[signal].sum()),
            "background_yield_pre": float(physical[background].sum()),
            "signal_yield_post": float(physical[outer_mask & signal].sum()),
            "background_yield_post": float(
                physical[outer_mask & background].sum()
            ),
            "signal_sumw2_post": float(
                np.square(physical[outer_mask & signal]).sum()
            ),
            "background_sumw2_post": float(
                np.square(physical[outer_mask & background]).sum()
            ),
        }
        for key, value in metrics.items():
            if isinstance(value, float):
                require(math.isfinite(value), f"nonfinite category metric {key}")
        require(metrics["signal_yield_pre"] > 0.0, "nonpositive signal yield")
        require(
            metrics["background_yield_pre"] > 0.0,
            "nonpositive background yield",
        )
        require(metrics["signal_sumw2_post"] >= 0.0, "negative signal sumw2")
        require(
            metrics["background_sumw2_post"] >= 0.0,
            "negative background sumw2",
        )

        category_results[category_id] = metrics
        outer_category_masks[category_id] = set(
            category_outer.loc[outer_mask, "event_uid"].astype(str)
        )

    require(
        outer_category_masks["exact3tag"].isdisjoint(
            outer_category_masks["ge4tag"]
        ),
        "selected category event sets overlap",
    )

    combined = {
        "signal_yield_pre": sum(
            category_results[category]["signal_yield_pre"]
            for category in ["exact3tag", "ge4tag"]
        ),
        "background_yield_pre": sum(
            category_results[category]["background_yield_pre"]
            for category in ["exact3tag", "ge4tag"]
        ),
        "signal_yield_post": sum(
            category_results[category]["signal_yield_post"]
            for category in ["exact3tag", "ge4tag"]
        ),
        "background_yield_post": sum(
            category_results[category]["background_yield_post"]
            for category in ["exact3tag", "ge4tag"]
        ),
        "signal_sumw2_post": sum(
            category_results[category]["signal_sumw2_post"]
            for category in ["exact3tag", "ge4tag"]
        ),
        "background_sumw2_post": sum(
            category_results[category]["background_sumw2_post"]
            for category in ["exact3tag", "ge4tag"]
        ),
        "selected_event_sets_disjoint": True,
    }
    for key, value in combined.items():
        if isinstance(value, float):
            require(math.isfinite(value), f"nonfinite combined metric {key}")
    combined["signal_physical_efficiency_within_ge3tag"] = (
        combined["signal_yield_post"] / combined["signal_yield_pre"]
    )
    combined["background_physical_efficiency_within_ge3tag"] = (
        combined["background_yield_post"] / combined["background_yield_pre"]
    )

    return {
        "inner_records": inner_records,
        "pooled_records": pooled_records,
        "category_results": category_results,
        "combined_outer_result": combined,
    }


first = run_nested_once()
second = run_nested_once()


def deterministic_view(result: dict[str, Any]) -> dict[str, Any]:
    copied = json.loads(json.dumps(result))
    for row in copied["inner_records"]:
        row.pop("elapsed_seconds", None)
    return copied


first_view = deterministic_view(first)
second_view = deterministic_view(second)
first_sha = canonical_sha256(first_view)
second_sha = canonical_sha256(second_view)
require(first_sha == second_sha, "deterministic rerun checksum mismatch")
require(first_view == second_view, "deterministic rerun payload mismatch")

inner_rows = []
for row in first["inner_records"]:
    inner_rows.append(
        {
            "category_id": row["category_id"],
            "inner_held_out_fold": row["inner_held_out_fold"],
            "category_family_id": row["category_family_id"],
            "structure_id": row["structure_id"],
            "continuous_variables": ",".join(row["continuous_variables"]),
            "thresholds_json": json.dumps(
                row["thresholds"], sort_keys=True, separators=(",", ":")
            ),
            "training_signal_efficiency": row["training_metric"][
                "signal_efficiency"
            ],
            "training_background_efficiency": row["training_metric"][
                "background_efficiency"
            ],
            "held_out_signal_efficiency": row["held_out_metric"][
                "signal_efficiency"
            ],
            "held_out_background_efficiency": row["held_out_metric"][
                "background_efficiency"
            ],
            "held_out_selected_signal_rows": row["held_out_metric"][
                "selected_signal_rows"
            ],
            "held_out_selected_background_rows": row["held_out_metric"][
                "selected_background_rows"
            ],
            "optimizer_audit_json": json.dumps(
                row["optimizer_audit"],
                sort_keys=True,
                separators=(",", ":"),
            ),
            "elapsed_seconds_first_run": row["elapsed_seconds"],
        }
    )
pd.DataFrame(inner_rows).to_csv(
    output_root / "diagnostic_inner_crossfit_results.tsv",
    sep="\t",
    index=False,
)

pooled_rows = []
for row in first["pooled_records"]:
    metric = row["pooled_inner_oof_metric"]
    chosen = (
        row["category_family_id"]
        == first["category_results"][row["category_id"]][
            "chosen_category_family_id"
        ]
    )
    pooled_rows.append(
        {
            "category_id": row["category_id"],
            "category_family_id": row["category_family_id"],
            "structure_id": row["structure_id"],
            "continuous_variables": ",".join(row["continuous_variables"]),
            "pooled_inner_oof_signal_efficiency": metric[
                "signal_efficiency"
            ],
            "pooled_inner_oof_background_efficiency": metric[
                "background_efficiency"
            ],
            "pooled_inner_oof_selected_signal_rows": metric[
                "selected_signal_rows"
            ],
            "pooled_inner_oof_selected_background_rows": metric[
                "selected_background_rows"
            ],
            "pooled_inner_oof_feasible": metric["feasible"],
            "chosen_within_category": chosen,
        }
    )
pd.DataFrame(pooled_rows).sort_values(
    [
        "category_id",
        "pooled_inner_oof_feasible",
        "pooled_inner_oof_background_efficiency",
        "structure_id",
    ],
    ascending=[True, False, True, True],
    kind="mergesort",
).to_csv(
    output_root / "diagnostic_pooled_inner_oof_metrics.tsv",
    sep="\t",
    index=False,
)

(output_root / "diagnostic_outer_category_evaluation.json").write_text(
    json.dumps(
        {
            "category_results": first["category_results"],
            "combined_outer_result": first["combined_outer_result"],
        },
        indent=2,
        sort_keys=True,
        allow_nan=False,
    )
    + "\n"
)

summary = {
    "schema_version": 1,
    "status": "pass_category_conditional_target_diagnostic_v2_not_physics_result",
    "repository_head": expected_head,
    "canary_scope": {
        "outer_fold": OUTER_FOLD,
        "sentinel_category_family_configurations": sentinel_ids,
        "rows_per_source_category_cap": ROWS_PER_SOURCE_CATEGORY,
        "all_authorized_primary_sources_considered": 441,
        "sampled_sources": int(sample["source_uid"].nunique()),
        "sampled_groups": int(sample["group_id"].nunique()),
        "sampled_rows": len(sample),
        "sampled_development_rows": len(development),
        "sampled_outer_rows": len(outer),
        "production_full_scan_rows": 1_042_397,
        "production_category_family_configurations": 54,
    },
    "category_protocol": {
        "categories": ["exact3tag", "ge4tag"],
        "categories_mutually_exclusive": True,
        "conditional_target": TARGET_SIGNAL_EFFICIENCY,
        "family_selected_separately_per_category": True,
        "retain_both_categories": True,
        "selected_outer_event_sets_disjoint": True,
    },
    "true_nested_protocol": {
        "inner_held_out_folds": [1, 2, 3, 4],
        "inner_training_folds_per_fit": 3,
        "family_selected_from_cross_fitted_inner_predictions": True,
        "chosen_family_refit_on_all_outer_development_folds": True,
        "outer_fold_used_for_selection": False,
    },
    "determinism": {
        "reruns": 2,
        "first_payload_sha256": first_sha,
        "second_payload_sha256": second_sha,
        "identical": True,
    },
    "category_results": first["category_results"],
    "combined_outer_result": first["combined_outer_result"],
    "acceptance_gates": {
        "all_four_inner_folds_present_in_both_categories": True,
        "no_group_id_crosses_a_fold": True,
        "threshold_training_excludes_inner_held_out_rows": True,
        "outer_fold_zero_excluded_from_all_selection": True,
        "all_six_sentinel_configurations_exercised": True,
        "all_three_mass_families_exercised_in_each_category": True,
        "category_conditional_target_reached_exact3tag": (
            first["category_results"]["exact3tag"][
                "category_target_reached_by_any_sentinel"
            ]
        ),
        "category_conditional_target_reached_ge4tag": (
            first["category_results"]["ge4tag"][
                "category_target_reached_by_any_sentinel"
            ]
        ),
        "all_cut_operators_match_contract": True,
        "all_tie_breaks_deterministic": True,
        "identical_rerun_output_checksum": True,
        "selected_row_masks_recomputed_from_raw_columns": True,
        "comparison_weight_closure_finite": True,
        "outer_comparison_weight_null_sentinel": True,
        "physical_yield_and_sumw2_finite": True,
        "selected_category_event_sets_disjoint": True,
        "validation_payloads_opened": 0,
        "test_payloads_opened": 0,
        "repository_modified": False,
    },
    "interpretation": (
        "Diagnostic evidence only. This run records whether the bounded "
        "cross-fitted sentinel configurations reach the category-conditional "
        "target, while proving that the open-control endpoint is feasible. "
        "Diagnostic winners, thresholds, yields, and efficiencies may not be "
        "used as physics results, may not alter the 54-configuration production "
        "contract, and may not enter paper figures."
    ),
    "next": (
        "use the recorded pooled cross-fitted efficiencies to choose a "
        "predeclared canary-only acceptance repair or a production operating-"
        "point calibration; do not rerun v3 blindly"
    ),
}
(output_root / "multivariate_cut_category_conditional_diagnostic_summary.json").write_text(
    json.dumps(summary, indent=2, sort_keys=True, allow_nan=False) + "\n"
)

print("CATEGORY_CONDITIONAL_DIAGNOSTIC_V2_STATUS=PASS")
print("CATEGORY_ENDPOINT_AMENDMENT=PASS")
print("DESCENDANT_TRACK_B_ONLY_SCOPE=PASS")
print("PROTECTED_CUT_CONTRACT_PATHS_UNCHANGED=PASS")
print("AUTHORIZED_PRIMARY_SOURCES_CONSIDERED=441/441")
print(f"SAMPLED_SOURCES={summary['canary_scope']['sampled_sources']}")
print(f"SAMPLED_GROUPS={summary['canary_scope']['sampled_groups']}")
print(f"SAMPLED_ROWS={summary['canary_scope']['sampled_rows']}")
print(
    "SAMPLED_DEVELOPMENT_ROWS="
    f"{summary['canary_scope']['sampled_development_rows']}"
)
print(
    "SAMPLED_OUTER_ROWS="
    f"{summary['canary_scope']['sampled_outer_rows']}"
)
print("PRODUCTION_FULL_SCAN_ROWS=1042397")
print("PRODUCTION_CATEGORY_FAMILY_CONFIGURATIONS=54")
print("OUTER_FOLD=0")
print("INNER_HELD_OUT_FOLDS=1,2,3,4")
print("INNER_TRAINING_FOLDS_PER_FIT=3")
print("CATEGORIES=exact3tag,ge4tag")
print("CATEGORIES_MUTUALLY_EXCLUSIVE=TRUE")
print("CATEGORY_CONDITIONAL_TARGET=0.585957")
print("SENTINEL_CONFIGURATIONS_EXERCISED=6/6")
print("MASS_FAMILIES_PER_CATEGORY=3/3")
for category_id in ["exact3tag", "ge4tag"]:
    category_result = first["category_results"][category_id]
    prefix = category_id.upper()
    print(
        f"{prefix}_OPEN_CONTROL_SIGNAL_EFFICIENCY="
        f"{category_result['open_control_metric']['signal_efficiency']:.12g}"
    )
    print(
        f"{prefix}_MAX_POOLED_INNER_OOF_SIGNAL_EFFICIENCY="
        f"{category_result['maximum_pooled_inner_oof_signal_efficiency']:.12g}"
    )
    print(
        f"{prefix}_MIN_POOLED_INNER_OOF_SIGNAL_EFFICIENCY="
        f"{category_result['minimum_pooled_inner_oof_signal_efficiency']:.12g}"
    )
    print(
        f"{prefix}_ANY_SENTINEL_TARGET_REACHED="
        f"{str(category_result['category_target_reached_by_any_sentinel']).upper()}"
    )
    print(
        f"{prefix}_DIAGNOSTIC_FALLBACK_USED="
        f"{str(category_result['diagnostic_fallback_used']).upper()}"
    )
    print(
        f"{prefix}_DIAGNOSTIC_FAMILY="
        f"{category_result['chosen_category_family_id']}"
    )
print("FAMILY_SELECTED_FROM_CROSS_FITTED_INNER_PREDICTIONS=TRUE")
print("OUTER_FOLD_USED_FOR_SELECTION=FALSE")
print(f"DETERMINISTIC_PAYLOAD_SHA256={first_sha}")
print("DETERMINISTIC_RERUN=PASS")
print("SELECTED_ROW_MASK_RECOMPUTATION=PASS")
print("COMPARISON_WEIGHT_SELECTION_NONNEGATIVE=PASS")
print("OUTER_COMPARISON_WEIGHT_NULL_SENTINEL=PASS")
print("PHYSICAL_YIELD_AND_SUMW2_FINITE=PASS")
print("SELECTED_CATEGORY_EVENT_SETS_DISJOINT=PASS")
print("DIAGNOSTIC_PHYSICS_RESULT=FALSE")
print("VALIDATION_PAYLOADS_OPENED=0")
print("TEST_PAYLOADS_OPENED=0")
print("REPOSITORY_MODIFIED=FALSE")
print("RESULT=MULTIVARIATE_CUT_CATEGORY_CONDITIONAL_DIAGNOSTIC_V2_PASS")
print("NEXT=SELECT_EVIDENCE_BASED_CANARY_ACCEPTANCE_REPAIR")
PY
}

mkdir -p "$(dirname "$OUT")"

RUN_RC=0
if [[ -e "$OUT" ]]; then
    echo "ERROR: output-root collision: $OUT"
    RUN_RC=output_root_collision
else
    mkdir "$OUT"
    set +e
    run_canary 2>&1 | tee "$LOG"
    RUN_RC=${PIPESTATUS[0]}
    set -e
fi

if [[ "$RUN_RC" == "0" ]]; then
    (
        cd "$OUT"
        find . -type f ! -name SHA256SUMS -print0 \
            | LC_ALL=C sort -z \
            | xargs -0 sha256sum > SHA256SUMS
        sha256sum -c SHA256SUMS
    )

    echo "DIAGNOSTIC_V2_OUTPUT_CHECKSUMS=PASS"
    echo "OUTPUT_ROOT=$OUT"
    echo "RESULT=MULTIVARIATE_CUT_CATEGORY_CONDITIONAL_DIAGNOSTIC_V2_PASS"
    echo "NEXT=SELECT_EVIDENCE_BASED_CANARY_ACCEPTANCE_REPAIR"
else
    echo "OUTPUT_ROOT=$OUT"
    echo "RUN_RC=$RUN_RC"
    echo "RESULT=MULTIVARIATE_CUT_CATEGORY_CONDITIONAL_DIAGNOSTIC_FAILED"
fi

if [[ "$RUN_RC" == "0" ]]; then
    true
else
    false
fi
