#!/usr/bin/env python3
"""Deterministic nested-OOF cut optimizer for resolved HH→4b.

This module contains only train-side optimization primitives. It never opens
validation or test payloads and never assigns physical meaning to canary-only
thresholds. Selection uses nonnegative fold-local comparison weights; signed
physical weights are used only for yield and sumw2 evaluation.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence
import hashlib
import json
import math

import numpy as np
import pandas as pd


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
class OptimizationSettings:
    target_signal_efficiency: float
    quantile_count: int
    beam_width: int
    refinement_starts: int
    max_coordinate_passes: int
    minimum_inner_selected_signal_rows: int
    minimum_inner_selected_background_rows: int
    minimum_pooled_background_rows: int
    minimum_pooled_background_neff: float


@dataclass(frozen=True)
class Metric:
    signal_efficiency: float
    background_efficiency: float
    selected_signal_rows: int
    selected_background_rows: int
    selected_signal_weight: float
    selected_background_weight: float
    selected_signal_sumw2: float
    selected_background_sumw2: float
    selected_signal_neff: float
    selected_background_neff: float
    feasible: bool


@dataclass(frozen=True)
class OptimizationResult:
    thresholds: dict[str, float]
    metric: Metric
    audit: dict[str, Any]


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def canonical_sha256(payload: Any) -> str:
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def threshold_json(thresholds: Mapping[str, float]) -> str:
    return json.dumps(
        {key: float(thresholds[key]) for key in sorted(thresholds)},
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def effective_count(weights: np.ndarray) -> float:
    weights = np.asarray(weights, dtype=float)
    if weights.size == 0:
        return 0.0
    sumw = float(weights.sum())
    sumw2 = float(np.square(weights).sum())
    if sumw2 <= 0.0:
        return 0.0
    return float((sumw * sumw) / sumw2)


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
    thresholds: Mapping[str, float],
) -> np.ndarray:
    mask = np.ones(len(frame), dtype=bool)
    for cut in configuration.cuts:
        values = frame[cut.variable].to_numpy(dtype=float)
        require(
            np.isfinite(values).all(),
            f"nonfinite values in {cut.variable}",
        )
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
    target_signal_efficiency: float,
) -> Metric:
    classes = frame["sample_class"].astype(str).to_numpy()
    weights = frame[weight_column].to_numpy(dtype=float)

    require(len(mask) == len(frame), "selection-mask length mismatch")
    require(np.isfinite(weights).all(), "nonfinite comparison weight")
    require((weights >= 0.0).all(), "negative comparison weight")

    signal = classes == "signal"
    background = classes == "background"
    require(signal.any(), "category frame lacks signal")
    require(background.any(), "category frame lacks background")

    total_signal = float(weights[signal].sum())
    total_background = float(weights[background].sum())
    require(total_signal > 0.0, "nonpositive signal denominator")
    require(total_background > 0.0, "nonpositive background denominator")

    selected_signal_weights = weights[mask & signal]
    selected_background_weights = weights[mask & background]
    selected_signal_weight = float(selected_signal_weights.sum())
    selected_background_weight = float(selected_background_weights.sum())
    signal_efficiency = selected_signal_weight / total_signal
    background_efficiency = selected_background_weight / total_background

    require(math.isfinite(signal_efficiency), "nonfinite signal efficiency")
    require(
        math.isfinite(background_efficiency),
        "nonfinite background efficiency",
    )

    return Metric(
        signal_efficiency=float(signal_efficiency),
        background_efficiency=float(background_efficiency),
        selected_signal_rows=int(np.count_nonzero(mask & signal)),
        selected_background_rows=int(np.count_nonzero(mask & background)),
        selected_signal_weight=selected_signal_weight,
        selected_background_weight=selected_background_weight,
        selected_signal_sumw2=float(np.square(selected_signal_weights).sum()),
        selected_background_sumw2=float(
            np.square(selected_background_weights).sum()
        ),
        selected_signal_neff=effective_count(selected_signal_weights),
        selected_background_neff=effective_count(
            selected_background_weights
        ),
        feasible=bool(signal_efficiency >= target_signal_efficiency),
    )


def physical_yield_summary(
    frame: pd.DataFrame,
    mask: np.ndarray,
    physical_weight_column: str,
) -> dict[str, Any]:
    classes = frame["sample_class"].astype(str).to_numpy()
    weights = frame[physical_weight_column].to_numpy(dtype=float)
    require(np.isfinite(weights).all(), "nonfinite physical weight")
    require(len(mask) == len(frame), "physical mask length mismatch")

    result: dict[str, Any] = {}
    for sample_class in ("signal", "background"):
        class_mask = classes == sample_class
        selected = mask & class_mask
        class_weights = weights[class_mask]
        selected_weights = weights[selected]
        result[sample_class] = {
            "total_rows": int(np.count_nonzero(class_mask)),
            "selected_rows": int(np.count_nonzero(selected)),
            "total_signed_yield": float(class_weights.sum()),
            "selected_signed_yield": float(selected_weights.sum()),
            "total_sumw2": float(np.square(class_weights).sum()),
            "selected_sumw2": float(np.square(selected_weights).sum()),
            "negative_weight_rows": int(
                np.count_nonzero(class_weights < 0.0)
            ),
            "selected_negative_weight_rows": int(
                np.count_nonzero(selected_weights < 0.0)
            ),
        }
    return result


def metric_to_dict(metric: Metric) -> dict[str, Any]:
    return asdict(metric)


def metric_rank(
    metric: Metric,
    thresholds: Mapping[str, float],
    target_signal_efficiency: float,
) -> tuple[Any, ...]:
    threshold_tuple = tuple(
        (key, float(thresholds[key])) for key in sorted(thresholds)
    )
    if metric.feasible:
        return (
            0,
            metric.background_efficiency,
            metric.signal_efficiency - target_signal_efficiency,
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
    target_signal_efficiency: float,
) -> tuple[Any, ...]:
    if metric.feasible:
        return (
            0,
            metric.background_efficiency,
            metric.signal_efficiency - target_signal_efficiency,
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


def support_gate(
    metric: Metric,
    settings: OptimizationSettings,
    *,
    pooled: bool,
) -> tuple[bool, list[str]]:
    failures: list[str] = []
    if metric.selected_signal_rows < settings.minimum_inner_selected_signal_rows:
        failures.append("selected_signal_rows")
    if (
        metric.selected_background_rows
        < settings.minimum_inner_selected_background_rows
    ):
        failures.append("selected_background_rows")
    if pooled:
        if (
            metric.selected_background_rows
            < settings.minimum_pooled_background_rows
        ):
            failures.append("pooled_background_rows")
        if (
            metric.selected_background_neff
            < settings.minimum_pooled_background_neff
        ):
            failures.append("pooled_background_neff")
    return (not failures), failures


def _quantile_probabilities(count: int) -> np.ndarray:
    require(count >= 1, "quantile_count must be positive")
    return np.arange(1, count + 1, dtype=float) / float(count + 1)


def _open_boundary(values: np.ndarray, operator: str) -> float:
    if operator == "<":
        return float(np.nextafter(values.max(), np.inf))
    if operator == ">":
        return float(np.nextafter(values.min(), -np.inf))
    raise RuntimeError(f"unsupported operator: {operator}")


def build_threshold_grids(
    frame: pd.DataFrame,
    configuration: Configuration,
    settings: OptimizationSettings,
) -> dict[str, list[float]]:
    probabilities = _quantile_probabilities(settings.quantile_count)
    grids: dict[str, list[float]] = {}
    for cut in configuration.cuts:
        values = frame[cut.variable].to_numpy(dtype=float)
        require(np.isfinite(values).all(), f"nonfinite {cut.variable}")
        quantiles = np.quantile(values, probabilities)
        if cut.operator == "<":
            candidates = np.nextafter(quantiles, np.inf).tolist()
        elif cut.operator == ">":
            candidates = np.nextafter(quantiles, -np.inf).tolist()
        else:
            raise RuntimeError(f"unsupported operator: {cut.operator}")
        candidates.append(_open_boundary(values, cut.operator))
        grids[cut.variable] = sorted({float(value) for value in candidates})
    return grids


def deterministic_beam_seeds(
    configuration: Configuration,
    grids: Mapping[str, Sequence[float]],
    beam_width: int,
) -> list[dict[str, float]]:
    """Create a deterministic bounded sample of the quantile product grid.

    The contract explicitly forbids claiming an exhaustive Cartesian search.
    This design evaluates at most ``beam_width`` predeclared product-grid seeds,
    always includes the fully open point, and then performs exact empirical
    coordinate refinement from the best predeclared starts.
    """

    require(beam_width >= 1, "beam_width must be positive")
    variables = [cut.variable for cut in configuration.cuts]
    operators = {cut.variable: cut.operator for cut in configuration.cuts}
    primes = (1, 31, 47, 59)
    offsets = (0, 11, 23, 37)
    require(len(variables) <= len(primes), "too many cut coordinates")

    open_seed: dict[str, float] = {}
    for variable in variables:
        grid = list(grids[variable])
        require(grid, f"empty grid for {variable}")
        open_seed[variable] = (
            max(grid) if operators[variable] == "<" else min(grid)
        )

    seeds = [open_seed]
    seen = {threshold_json(open_seed)}
    seed_index = 1
    max_attempts = max(beam_width * 20, 100)
    attempts = 0
    while len(seeds) < beam_width and attempts < max_attempts:
        candidate: dict[str, float] = {}
        for coordinate_index, variable in enumerate(variables):
            grid = list(grids[variable])
            nonopen_size = max(len(grid) - 1, 1)
            index = (
                seed_index * primes[coordinate_index]
                + offsets[coordinate_index]
            ) % nonopen_size
            if operators[variable] == ">":
                # For a greater-than cut, the open point is the first item.
                # Shift sampled indices away from that open boundary.
                index = min(index + 1, len(grid) - 1)
            candidate[variable] = float(grid[index])
        key = threshold_json(candidate)
        if key not in seen:
            seeds.append(candidate)
            seen.add(key)
        seed_index += 1
        attempts += 1

    require(seeds, "no deterministic beam seeds produced")
    return seeds


def exact_coordinate_candidate(
    frame: pd.DataFrame,
    configuration: Configuration,
    current: Mapping[str, float],
    coordinate: CutSpec,
    weight_column: str,
    target_signal_efficiency: float,
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
    classes = frame["sample_class"].astype(str).to_numpy()
    weights = frame[weight_column].to_numpy(dtype=float)
    require(np.isfinite(values).all(), f"nonfinite {coordinate.variable}")
    require(np.isfinite(weights).all(), "nonfinite comparison weight")
    require((weights >= 0.0).all(), "negative comparison weight")

    signal = classes == "signal"
    background = classes == "background"
    total_signal = float(weights[signal].sum())
    total_background = float(weights[background].sum())
    require(total_signal > 0.0, "nonpositive signal denominator")
    require(total_background > 0.0, "nonpositive background denominator")

    active_indices = np.flatnonzero(base_mask)
    require(
        active_indices.size > 0,
        f"no rows survive fixed coordinates for {coordinate.variable}",
    )
    order = np.argsort(values[active_indices], kind="mergesort")
    sorted_indices = active_indices[order]
    sorted_x = values[sorted_indices]
    sorted_weights = weights[sorted_indices]
    sorted_signal = signal[sorted_indices]
    sorted_background = background[sorted_indices]

    unique_values, first_positions, counts = np.unique(
        sorted_x,
        return_index=True,
        return_counts=True,
    )
    last_positions = first_positions + counts - 1
    cumulative_signal_weight = np.cumsum(
        np.where(sorted_signal, sorted_weights, 0.0)
    )
    cumulative_background_weight = np.cumsum(
        np.where(sorted_background, sorted_weights, 0.0)
    )
    cumulative_signal_sumw2 = np.cumsum(
        np.where(sorted_signal, np.square(sorted_weights), 0.0)
    )
    cumulative_background_sumw2 = np.cumsum(
        np.where(sorted_background, np.square(sorted_weights), 0.0)
    )
    cumulative_signal_rows = np.cumsum(sorted_signal.astype(np.int64))
    cumulative_background_rows = np.cumsum(
        sorted_background.astype(np.int64)
    )

    if coordinate.operator == "<":
        selected_signal_weight = cumulative_signal_weight[last_positions]
        selected_background_weight = cumulative_background_weight[
            last_positions
        ]
        selected_signal_sumw2 = cumulative_signal_sumw2[last_positions]
        selected_background_sumw2 = cumulative_background_sumw2[
            last_positions
        ]
        selected_signal_rows = cumulative_signal_rows[last_positions]
        selected_background_rows = cumulative_background_rows[last_positions]
        candidate_thresholds = np.nextafter(unique_values, np.inf)
    elif coordinate.operator == ">":
        before = first_positions - 1

        def subtract_before(cumulative: np.ndarray) -> np.ndarray:
            prior = np.where(
                before >= 0,
                cumulative[np.maximum(before, 0)],
                0,
            )
            return cumulative[-1] - prior

        selected_signal_weight = subtract_before(cumulative_signal_weight)
        selected_background_weight = subtract_before(
            cumulative_background_weight
        )
        selected_signal_sumw2 = subtract_before(cumulative_signal_sumw2)
        selected_background_sumw2 = subtract_before(
            cumulative_background_sumw2
        )
        selected_signal_rows = subtract_before(cumulative_signal_rows)
        selected_background_rows = subtract_before(
            cumulative_background_rows
        )
        candidate_thresholds = np.nextafter(unique_values, -np.inf)
    else:
        raise RuntimeError(f"unsupported operator: {coordinate.operator}")

    signal_efficiency = selected_signal_weight / total_signal
    background_efficiency = selected_background_weight / total_background

    best_index: int | None = None
    best_rank: tuple[Any, ...] | None = None
    for index in range(len(candidate_thresholds)):
        signal_sumw = float(selected_signal_weight[index])
        background_sumw = float(selected_background_weight[index])
        signal_sumw2 = float(selected_signal_sumw2[index])
        background_sumw2 = float(selected_background_sumw2[index])
        metric = Metric(
            signal_efficiency=float(signal_efficiency[index]),
            background_efficiency=float(background_efficiency[index]),
            selected_signal_rows=int(selected_signal_rows[index]),
            selected_background_rows=int(selected_background_rows[index]),
            selected_signal_weight=signal_sumw,
            selected_background_weight=background_sumw,
            selected_signal_sumw2=signal_sumw2,
            selected_background_sumw2=background_sumw2,
            selected_signal_neff=(
                signal_sumw * signal_sumw / signal_sumw2
                if signal_sumw2 > 0.0
                else 0.0
            ),
            selected_background_neff=(
                background_sumw * background_sumw / background_sumw2
                if background_sumw2 > 0.0
                else 0.0
            ),
            feasible=bool(
                signal_efficiency[index] >= target_signal_efficiency
            ),
        )
        thresholds = dict(current)
        thresholds[coordinate.variable] = float(candidate_thresholds[index])
        rank = metric_rank(metric, thresholds, target_signal_efficiency)
        if best_rank is None or rank < best_rank:
            best_rank = rank
            best_index = index

    require(best_index is not None, "coordinate scan returned no result")
    thresholds = dict(current)
    thresholds[coordinate.variable] = float(candidate_thresholds[best_index])
    metric = weighted_metric(
        frame,
        selection_mask(frame, configuration, thresholds),
        weight_column,
        target_signal_efficiency,
    )
    return thresholds, metric


def optimize_configuration(
    frame: pd.DataFrame,
    configuration: Configuration,
    weight_column: str,
    settings: OptimizationSettings,
) -> OptimizationResult:
    require(len(frame) > 0, "empty optimization frame")
    require(
        set(frame["sample_class"].astype(str)) == {"signal", "background"},
        "optimization frame lacks one class",
    )
    require(1 <= len(configuration.cuts) <= 4, "invalid cut count")

    grids = build_threshold_grids(frame, configuration, settings)
    seeds = deterministic_beam_seeds(
        configuration,
        grids,
        settings.beam_width,
    )
    evaluated: list[tuple[tuple[Any, ...], dict[str, float], Metric]] = []
    for thresholds in seeds:
        metric = weighted_metric(
            frame,
            selection_mask(frame, configuration, thresholds),
            weight_column,
            settings.target_signal_efficiency,
        )
        evaluated.append(
            (
                metric_rank(
                    metric,
                    thresholds,
                    settings.target_signal_efficiency,
                ),
                dict(thresholds),
                metric,
            )
        )
    evaluated.sort(key=lambda item: item[0])
    beam = evaluated[: settings.beam_width]
    starts = beam[: settings.refinement_starts]
    require(starts, "optimizer produced no refinement starts")

    refined: list[
        tuple[tuple[Any, ...], dict[str, float], Metric, int]
    ] = []
    for _, initial_thresholds, initial_metric in starts:
        thresholds = dict(initial_thresholds)
        metric = initial_metric
        passes_completed = 0
        for _ in range(settings.max_coordinate_passes):
            improved = False
            for coordinate in configuration.cuts:
                candidate_thresholds, candidate_metric = (
                    exact_coordinate_candidate(
                        frame,
                        configuration,
                        thresholds,
                        coordinate,
                        weight_column,
                        settings.target_signal_efficiency,
                    )
                )
                if metric_rank(
                    candidate_metric,
                    candidate_thresholds,
                    settings.target_signal_efficiency,
                ) < metric_rank(
                    metric,
                    thresholds,
                    settings.target_signal_efficiency,
                ):
                    thresholds = candidate_thresholds
                    metric = candidate_metric
                    improved = True
            passes_completed += 1
            if not improved:
                break
        refined.append(
            (
                metric_rank(
                    metric,
                    thresholds,
                    settings.target_signal_efficiency,
                ),
                thresholds,
                metric,
                passes_completed,
            )
        )

    refined.sort(key=lambda item: item[0])
    _, thresholds, metric, passes_completed = refined[0]
    require(
        metric.feasible,
        "configuration training fit did not reach target: "
        f"{configuration.category_family_id}",
    )
    return OptimizationResult(
        thresholds={key: float(value) for key, value in thresholds.items()},
        metric=metric,
        audit={
            "algorithm": (
                "deterministic bounded quantile-product beam seeds plus "
                "exact empirical coordinate refinement"
            ),
            "exhaustive_cartesian_claim": False,
            "grid_sizes": {
                variable: len(values) for variable, values in grids.items()
            },
            "beam_seed_count": len(seeds),
            "beam_width_limit": settings.beam_width,
            "refinement_starts_used": len(starts),
            "max_coordinate_passes": settings.max_coordinate_passes,
            "coordinate_passes_completed": passes_completed,
            "fully_open_seed_included": True,
        },
    )


def load_amended_configurations(path: str | Path) -> list[Configuration]:
    payload = json.loads(Path(path).read_text())
    rows = payload["category_program"]["category_family_configurations"]
    configurations = [
        Configuration(
            category_family_id=row["category_family_id"],
            category_id=row["category_id"],
            structure_id=row["structure_id"],
            cuts=tuple(
                CutSpec(
                    variable=cut["variable"],
                    operator=cut["operator"],
                )
                for cut in row["continuous_cuts"]
            ),
        )
        for row in rows
    ]
    require(len(configurations) == 54, "amended configuration count changed")
    require(
        len({row.category_family_id for row in configurations}) == 54,
        "category-family IDs are not unique",
    )
    return configurations


def configuration_registry(
    configurations: Iterable[Configuration],
) -> dict[str, Configuration]:
    registry = {row.category_family_id: row for row in configurations}
    require(len(registry) > 0, "empty configuration registry")
    return registry


def structure_registry(
    configurations: Iterable[Configuration],
) -> dict[tuple[str, str], Configuration]:
    registry = {
        (row.category_id, row.structure_id): row for row in configurations
    }
    require(len(registry) == 54, "structure registry count changed")
    return registry
