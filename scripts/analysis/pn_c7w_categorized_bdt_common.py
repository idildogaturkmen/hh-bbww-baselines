#!/usr/bin/env python3
"""Common nested-category and paired-bootstrap utilities for PN-c7w."""

from __future__ import annotations

import itertools
import json
import math
from typing import Any, Iterable

import numpy as np
import pandas as pd

from pn_c7_bootstrap_common import group_count, group_sum, member_positions
from pn_c7_ml_common import (
    asimov_za,
    bootstrap_quantile_summary,
    correlated_multijet_profile_za,
    require,
)


STAGE1_EFFICIENCIES = (0.65, 0.75, 0.85, 0.90)
CATEGORY_COUNTS = (2, 3, 4, 5, 6)
BOUNDARY_QUANTILES = tuple(float(value) for value in np.linspace(0.10, 0.90, 9))
SPECIALIZED_STRATA = frozenset({"single_higgs", "single_top", "top_associated", "ttbar"})
COMPLEMENTARY_STRATA = frozenset({"qcd_multijet", "zbbbb", "diboson", "triboson"})
SUPPORT_CONTRACTS: dict[str, dict[str, float]] = {
    "nominal": {
        "background_rows": 50, "background_sources": 10, "background_neff": 2.0,
        "signal_rows": 25, "transferred_rows": 5, "transferred_sources": 3,
    },
    "strict_a": {
        "background_rows": 75, "background_sources": 12, "background_neff": 3.0,
        "signal_rows": 35, "transferred_rows": 7, "transferred_sources": 4,
    },
    "strict_b": {
        "background_rows": 100, "background_sources": 15, "background_neff": 4.0,
        "signal_rows": 50, "transferred_rows": 10, "transferred_sources": 5,
    },
}


def weighted_quantile(values: np.ndarray, weights: np.ndarray, quantiles: Iterable[float]) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64)
    quantiles_array = np.asarray(tuple(quantiles), dtype=np.float64)
    require(values.ndim == weights.ndim == 1 and len(values) == len(weights), "weighted quantile shape drift")
    require(len(values) > 0 and bool(np.isfinite(values).all()), "invalid weighted quantile values")
    require(bool((weights > 0.0).all()) and math.isfinite(float(weights.sum())), "invalid weighted quantile weights")
    require(bool(((quantiles_array > 0.0) & (quantiles_array < 1.0)).all()), "invalid quantile request")
    order = np.argsort(values, kind="mergesort")
    ordered_values = values[order]
    ordered_weights = weights[order]
    cumulative = (np.cumsum(ordered_weights) - 0.5 * ordered_weights) / ordered_weights.sum()
    return np.interp(quantiles_array, cumulative, ordered_values)


def assign_categories(stage1: np.ndarray, stage2: np.ndarray, threshold: float,
                      boundaries: Iterable[float]) -> np.ndarray:
    stage1 = np.asarray(stage1, dtype=np.float64)
    stage2 = np.asarray(stage2, dtype=np.float64)
    boundaries_array = np.asarray(tuple(boundaries), dtype=np.float64)
    require(stage1.shape == stage2.shape and bool(np.isfinite(stage1).all()), "category score shape/finite drift")
    require(bool(np.isfinite(stage2).all()), "nonfinite stage-2 category score")
    require(bool((np.diff(boundaries_array) > 0.0).all()) if len(boundaries_array) > 1 else True,
            "unordered category boundaries")
    categories = np.zeros(len(stage1), dtype=np.int8)
    high = stage1 >= float(threshold)
    categories[high] = np.searchsorted(boundaries_array, stage2[high], side="right").astype(np.int8) + 1
    require(set(np.unique(categories)).issubset(set(range(len(boundaries_array) + 2))), "category index drift")
    return categories


def support_reasons(row: dict[str, Any], contract: dict[str, float]) -> list[str]:
    reasons = []
    for field, description in (
        ("background_rows", "background rows"),
        ("background_sources", "background sources"),
        ("background_neff", "background Neff"),
        ("signal_rows", "signal rows"),
        ("transferred_rows", "transferred-QCD rows"),
        ("transferred_sources", "transferred-QCD sources"),
    ):
        if not math.isfinite(float(row[field])) or float(row[field]) < float(contract[field]):
            reasons.append(f"{description} below {contract[field]:g}")
    if not math.isfinite(float(row["signal_yield"])) or float(row["signal_yield"]) <= 0.0:
        reasons.append("nonpositive signal yield")
    return reasons


def _neff(weights: np.ndarray) -> float:
    total = float(np.asarray(weights, dtype=np.float64).sum())
    sum2 = float(np.square(np.asarray(weights, dtype=np.float64)).sum())
    return total * total / sum2 if sum2 > 0.0 else 0.0


def evaluate_categories(
    projection: pd.DataFrame,
    direct: pd.DataFrame,
    projection_categories: np.ndarray,
    direct_categories: np.ndarray,
    factor_values: Iterable[float],
    factor_stat_relative: float,
    *,
    contract_name: str = "nominal",
    require_category_support_for_systematic: bool = True,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Evaluate exclusive categories and one shared correlated multijet nuisance."""

    contract = SUPPORT_CONTRACTS[contract_name]
    factor_values_array = np.asarray(tuple(factor_values), dtype=np.float64)
    require(len(factor_values_array) == 3 and bool((factor_values_array > 0).all()), "factor contract drift")
    nominal_factor = float(factor_values_array[0])
    projection_categories = np.asarray(projection_categories, dtype=np.int8)
    direct_categories = np.asarray(direct_categories, dtype=np.int8)
    require(len(projection_categories) == len(projection), "projection category alignment drift")
    require(len(direct_categories) == len(direct), "direct category alignment drift")
    category_count = int(max(projection_categories.max(initial=0), direct_categories.max(initial=0)) + 1)
    require(set(np.unique(projection_categories)) == set(range(category_count)), "empty projection category")

    target = projection["registry_training_target"].to_numpy(dtype=np.int8)
    signal = target == 1
    ordinary = projection["analysis_population_role"].astype(str).to_numpy() == "fourb_ordinary_background"
    transferred = projection["analysis_population_role"].astype(str).to_numpy() == "primary_transferred_multijet_template"
    weight = projection["primary_projection_physical_weight_inclusive"].to_numpy(dtype=np.float64)
    direct_qcd = direct["population_kind"].astype(str).to_numpy() == "hard_qcd"
    direct_weight = direct["direct_projection_physical_weight"].to_numpy(dtype=np.float64)
    rows: list[dict[str, Any]] = []
    for category in range(category_count):
        selected = projection_categories == category
        selected_signal = selected & signal
        selected_background = selected & ~signal
        selected_transferred = selected & transferred
        selected_direct = (direct_categories == category) & direct_qcd
        signal_weights = weight[selected_signal]
        background_weights = weight[selected_background]
        ordinary_yield = float(weight[selected & ordinary].sum())
        transferred_yield = float(weight[selected_transferred].sum())
        direct_yield = float(direct_weight[selected_direct].sum())
        signal_yield = float(signal_weights.sum())
        background_yield = float(background_weights.sum())
        row = {
            "category": category,
            "signal_rows": int(selected_signal.sum()),
            "background_rows": int(selected_background.sum()),
            "background_sources": int(projection.loc[selected_background, "registry_group_id"].nunique()),
            "background_neff": _neff(background_weights),
            "signal_yield": signal_yield,
            "signal_neff": _neff(signal_weights),
            "ordinary_background_yield": ordinary_yield,
            "transferred_qcd_yield": transferred_yield,
            "transferred_rows": int(selected_transferred.sum()),
            "transferred_sources": int(projection.loc[selected_transferred, "registry_group_id"].nunique()),
            "direct_qcd_closure_yield": direct_yield,
            "direct_qcd_rows": int(selected_direct.sum()),
            "direct_qcd_sources": int(direct.loc[selected_direct, "registry_group_id"].nunique()),
            "direct_qcd_neff": _neff(direct_weight[selected_direct]),
            "background_yield": background_yield,
            "signal_over_background": signal_yield / background_yield if background_yield > 0.0 else float("nan"),
            "signal_over_sqrt_background": signal_yield / math.sqrt(background_yield) if background_yield > 0.0 else float("nan"),
            "transferred_qcd_fraction": transferred_yield / background_yield if background_yield > 0.0 else float("nan"),
            "direct_qcd_closure_ratio": direct_yield / transferred_yield if transferred_yield > 0.0 else float("nan"),
            "nominal_asimov_ZA_contribution": asimov_za(signal_yield, background_yield),
        }
        reasons = support_reasons(row, contract)
        row["support_pass"] = not reasons
        row["support_failure_reason"] = "; ".join(reasons)
        row["minimum_support_margin"] = min(
            float(row[field]) / float(contract[field])
            for field in ("background_rows", "background_sources", "background_neff", "signal_rows",
                          "transferred_rows", "transferred_sources")
        )
        rows.append(row)

    signal_yields = np.asarray([row["signal_yield"] for row in rows], dtype=np.float64)
    ordinary_yields = np.asarray([row["ordinary_background_yield"] for row in rows], dtype=np.float64)
    qcd_yields = np.asarray([row["transferred_qcd_yield"] for row in rows], dtype=np.float64)
    direct_total = float(sum(row["direct_qcd_closure_yield"] for row in rows))
    qcd_total = float(qcd_yields.sum())
    envelope_relative = float(np.max(np.abs(factor_values_array / nominal_factor - 1.0)))
    nonclosure = abs(qcd_total - direct_total) / abs(direct_total) if direct_total > 0.0 else float("nan")
    nuisance = math.sqrt(float(factor_stat_relative) ** 2 + envelope_relative ** 2 + nonclosure ** 2) if math.isfinite(nonclosure) else float("nan")
    joint_nominal = float(math.sqrt(sum(float(row["nominal_asimov_ZA_contribution"]) ** 2 for row in rows)))
    support_failure_reasons = [f"category {row['category']}: {row['support_failure_reason']}" for row in rows if not row["support_pass"]]
    if not math.isfinite(nonclosure):
        support_failure_reasons.append("shared direct-QCD nonclosure undefined")
    systematic_failure_reasons = list(support_failure_reasons if require_category_support_for_systematic else [])
    if not math.isfinite(nonclosure) and "shared direct-QCD nonclosure undefined" not in systematic_failure_reasons:
        systematic_failure_reasons.append("shared direct-QCD nonclosure undefined")
    joint_systematic = float("nan")
    if not systematic_failure_reasons:
        joint_systematic = correlated_multijet_profile_za(
            signal_yields, ordinary_yields, qcd_yields, np.full(category_count, nuisance, dtype=np.float64)
        )
    for row in rows:
        row["shared_qcd_relative_nonclosure"] = nonclosure
        row["shared_multijet_relative_nuisance"] = nuisance
        row["systematic_aware_ZA_standalone_contribution"] = (
            correlated_multijet_profile_za(
                np.asarray([row["signal_yield"]]), np.asarray([row["ordinary_background_yield"]]),
                np.asarray([row["transferred_qcd_yield"]]), np.asarray([nuisance]),
            ) if math.isfinite(nuisance) and (row["support_pass"] or not require_category_support_for_systematic) else float("nan")
        )
        row["contract_name"] = contract_name
    summary = {
        "category_count": category_count,
        "combined_nominal_asimov_ZA": joint_nominal,
        "combined_systematic_aware_asimov_ZA": joint_systematic,
        "signal_yield": float(signal_yields.sum()),
        "background_yield": float((ordinary_yields + qcd_yields).sum()),
        "signal_over_background": float(signal_yields.sum() / (ordinary_yields + qcd_yields).sum()),
        "signal_over_sqrt_background": float(signal_yields.sum() / math.sqrt((ordinary_yields + qcd_yields).sum())),
        "background_neff": _neff(weight[target == 0]),
        "signal_neff": _neff(weight[target == 1]),
        "transferred_qcd_fraction": float(qcd_total / (ordinary_yields + qcd_yields).sum()),
        "direct_qcd_closure_ratio": float(direct_total / qcd_total) if qcd_total > 0.0 else float("nan"),
        "shared_qcd_relative_nonclosure": nonclosure,
        "shared_multijet_relative_nuisance": nuisance,
        "support_pass": not support_failure_reasons,
        "support_failure_reason": "; ".join(support_failure_reasons),
        "systematic_evaluation_pass": not systematic_failure_reasons,
        "systematic_evaluation_failure_reason": "; ".join(systematic_failure_reasons),
        "minimum_support_margin": min(float(row["minimum_support_margin"]) for row in rows),
        "contract_name": contract_name,
    }
    return summary, rows


def boundary_instability(
    stage1: np.ndarray,
    stage2: np.ndarray,
    signal: np.ndarray,
    weights: np.ndarray,
    folds: np.ndarray,
    threshold: float,
    quantiles: tuple[float, ...],
    pooled_boundaries: np.ndarray,
) -> float:
    if not quantiles:
        return 0.0
    scale = max(float(np.ptp(stage2)), 1.0e-9)
    fold_values = []
    for fold in sorted(np.unique(folds)):
        selected = signal & (stage1 >= threshold) & (folds == fold)
        if selected.sum() < len(quantiles) + 2:
            return float("inf")
        fold_values.append(weighted_quantile(stage2[selected], weights[selected], quantiles))
    return float(np.median(np.abs(np.asarray(fold_values) - pooled_boundaries[None, :])) / scale)


def optimize_staircase(
    training: pd.DataFrame,
    projection: pd.DataFrame,
    direct: pd.DataFrame,
    train_stage1: np.ndarray,
    train_stage2_by_efficiency: dict[float, np.ndarray],
    projection_stage1: np.ndarray,
    projection_stage2_by_efficiency: dict[float, np.ndarray],
    thresholds: dict[float, float],
    factor_values: Iterable[float],
    factor_stat_relative: float,
    *,
    category_counts: Iterable[int] = CATEGORY_COUNTS,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Enumerate the frozen staircase grid and return candidates and per-category rows."""

    signal_train = training["registry_training_target"].to_numpy(dtype=np.int8) == 1
    development_weight = training["development_hierarchical_weight"].to_numpy(dtype=np.float64)
    folds = training["registry_oof_fold"].to_numpy(dtype=np.int8)
    candidates: list[dict[str, Any]] = []
    category_rows: list[dict[str, Any]] = []
    for efficiency in STAGE1_EFFICIENCIES:
        threshold = float(thresholds[efficiency])
        train_stage2 = np.asarray(train_stage2_by_efficiency[efficiency], dtype=np.float64)
        projection_stage2 = np.asarray(projection_stage2_by_efficiency[efficiency], dtype=np.float64)
        high_signal = signal_train & (train_stage1 >= threshold)
        require(int(high_signal.sum()) >= 100, f"insufficient high-stage signal at efficiency {efficiency}")
        quantile_values = dict(zip(
            BOUNDARY_QUANTILES,
            weighted_quantile(train_stage2[high_signal], development_weight[high_signal], BOUNDARY_QUANTILES),
        ))
        for category_count in category_counts:
            boundary_number = int(category_count) - 2
            combinations = [()] if boundary_number == 0 else itertools.combinations(BOUNDARY_QUANTILES, boundary_number)
            for quantile_combo_raw in combinations:
                quantile_combo = tuple(float(value) for value in quantile_combo_raw)
                boundaries = np.asarray([quantile_values[value] for value in quantile_combo], dtype=np.float64)
                if len(boundaries) > 1 and not bool((np.diff(boundaries) > 0.0).all()):
                    continue
                projection_categories = assign_categories(projection_stage1, projection_stage2, threshold, boundaries)
                direct_categories = assign_categories(train_stage1, train_stage2, threshold, boundaries)
                summary, rows = evaluate_categories(
                    projection, direct, projection_categories, direct_categories,
                    factor_values, factor_stat_relative, contract_name="nominal",
                )
                instability = boundary_instability(
                    train_stage1, train_stage2, signal_train, development_weight, folds,
                    threshold, quantile_combo, boundaries,
                )
                candidate_id = f"e{efficiency:.2f}_n{category_count}_q" + "-".join(f"{value:.2f}" for value in quantile_combo)
                record = {
                    "candidate_id": candidate_id,
                    "stage1_target_signal_efficiency": efficiency,
                    "stage1_threshold": threshold,
                    "category_count": int(category_count),
                    "boundary_quantiles_json": json.dumps(quantile_combo, separators=(",", ":")),
                    "stage2_boundaries_json": json.dumps(boundaries.tolist(), separators=(",", ":")),
                    "boundary_instability": instability,
                    **summary,
                }
                candidates.append(record)
                for row in rows:
                    category_rows.append({"candidate_id": candidate_id, **row})
    return pd.DataFrame(candidates), pd.DataFrame(category_rows)


def select_candidates(candidates: pd.DataFrame) -> tuple[pd.Series, dict[int, pd.Series]]:
    supported = candidates[candidates.support_pass & np.isfinite(candidates.combined_systematic_aware_asimov_ZA)].copy()
    require(not supported.empty, "no supported categorized-BDT candidate")
    supported = supported.sort_values(
        ["combined_systematic_aware_asimov_ZA", "combined_nominal_asimov_ZA", "category_count", "boundary_instability", "candidate_id"],
        ascending=[False, False, True, True, True], kind="mergesort",
    )
    selected = supported.iloc[0]
    by_count: dict[int, pd.Series] = {}
    for count in CATEGORY_COUNTS:
        sub = supported[supported.category_count == count]
        require(not sub.empty, f"no supported candidate for category count {count}")
        by_count[count] = sub.iloc[0]
    return selected, by_count


def bootstrap_fixed_categories(
    projection: pd.DataFrame,
    direct: pd.DataFrame,
    projection_categories: np.ndarray,
    direct_categories: np.ndarray,
    members: pd.DataFrame,
    draws: np.ndarray,
    factor_values: Iterable[float],
    factor_stat_relative: float,
    *,
    label: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Recompute fixed exhaustive categories on every common source draw."""

    factor_values_array = np.asarray(tuple(factor_values), dtype=np.float64)
    nominal_factor = float(factor_values_array[0])
    envelope_relative = float(np.max(np.abs(factor_values_array / nominal_factor - 1.0)))
    members = members.reset_index(drop=True)
    n_members = len(members)
    projection_pos = member_positions(projection, members, f"{label} projection")
    direct_pos = member_positions(direct, members, f"{label} direct")
    category_count = int(np.max(projection_categories) + 1)
    target = projection["registry_training_target"].to_numpy(dtype=np.int8)
    signal = target == 1
    ordinary = projection["analysis_population_role"].astype(str).to_numpy() == "fourb_ordinary_background"
    transferred = projection["analysis_population_role"].astype(str).to_numpy() == "primary_transferred_multijet_template"
    weight = projection["primary_projection_physical_weight_inclusive"].to_numpy(dtype=np.float64)
    direct_qcd = direct["population_kind"].astype(str).to_numpy() == "hard_qcd"
    direct_weight = direct["direct_projection_physical_weight"].to_numpy(dtype=np.float64)

    groups: list[dict[str, Any]] = []
    for category in range(category_count):
        selected = projection_categories == category
        selected_direct = direct_categories == category
        masks = {
            "signal": selected & signal,
            "ordinary": selected & ordinary,
            "qcd": selected & transferred,
            "background": selected & ~signal,
        }
        groups.append({
            "signal_yield": group_sum(weight, masks["signal"], projection_pos, n_members),
            "signal_sumw2": group_sum(np.square(weight), masks["signal"], projection_pos, n_members),
            "signal_rows": group_count(masks["signal"], projection_pos, n_members),
            "ordinary_yield": group_sum(weight, masks["ordinary"], projection_pos, n_members),
            "qcd_yield": group_sum(weight, masks["qcd"], projection_pos, n_members),
            "background_sumw2": group_sum(np.square(weight), masks["background"], projection_pos, n_members),
            "background_rows": group_count(masks["background"], projection_pos, n_members),
            "qcd_rows": group_count(masks["qcd"], projection_pos, n_members),
            "direct_yield": group_sum(direct_weight, selected_direct & direct_qcd, direct_pos, n_members),
            "direct_sumw2": group_sum(np.square(direct_weight), selected_direct & direct_qcd, direct_pos, n_members),
            "direct_rows": group_count(selected_direct & direct_qcd, direct_pos, n_members),
            "signal_presence": group_count(masks["signal"], projection_pos, n_members) > 0,
            "background_presence": group_count(masks["background"], projection_pos, n_members) > 0,
            "qcd_presence": group_count(masks["qcd"], projection_pos, n_members) > 0,
            "direct_presence": group_count(selected_direct & direct_qcd, direct_pos, n_members) > 0,
        })

    combined_rows: list[dict[str, Any]] = []
    category_rows: list[dict[str, Any]] = []
    contract = SUPPORT_CONTRACTS["nominal"]
    for replica_id, multiplicities in enumerate(draws):
        evaluated: list[dict[str, Any]] = []
        for category, group in enumerate(groups):
            signal_yield = float(multiplicities @ group["signal_yield"])
            ordinary_yield = float(multiplicities @ group["ordinary_yield"])
            qcd_yield = float(multiplicities @ group["qcd_yield"])
            background_yield = ordinary_yield + qcd_yield
            signal_sumw2 = float(multiplicities @ group["signal_sumw2"])
            background_sumw2 = float(multiplicities @ group["background_sumw2"])
            direct_yield = float(multiplicities @ group["direct_yield"])
            direct_sumw2 = float(multiplicities @ group["direct_sumw2"])
            row = {
                "label": label, "replica_id": replica_id, "category": category,
                "signal_yield": signal_yield, "ordinary_background_yield": ordinary_yield,
                "transferred_qcd_yield": qcd_yield, "background_yield": background_yield,
                "signal_over_background": signal_yield / background_yield if background_yield > 0.0 else float("nan"),
                "signal_over_sqrt_background": signal_yield / math.sqrt(background_yield) if background_yield > 0.0 else float("nan"),
                "background_neff": background_yield * background_yield / background_sumw2 if background_sumw2 > 0.0 else float("nan"),
                "signal_neff": signal_yield * signal_yield / signal_sumw2 if signal_sumw2 > 0.0 else float("nan"),
                "transferred_qcd_fraction": qcd_yield / background_yield if background_yield > 0.0 else float("nan"),
                "direct_qcd_closure_ratio": direct_yield / qcd_yield if qcd_yield > 0.0 else float("nan"),
                "signal_rows": int(multiplicities @ group["signal_rows"]),
                "background_rows": int(multiplicities @ group["background_rows"]),
                "background_sources": int(np.count_nonzero(multiplicities[group["background_presence"]])),
                "transferred_rows": int(multiplicities @ group["qcd_rows"]),
                "transferred_sources": int(np.count_nonzero(multiplicities[group["qcd_presence"]])),
                "direct_qcd_closure_yield": direct_yield,
                "direct_qcd_neff": direct_yield * direct_yield / direct_sumw2 if direct_sumw2 > 0.0 else float("nan"),
                "direct_qcd_rows": int(multiplicities @ group["direct_rows"]),
                "direct_qcd_sources": int(np.count_nonzero(multiplicities[group["direct_presence"]])),
                "nominal_asimov_ZA_contribution": asimov_za(signal_yield, background_yield),
            }
            reasons = support_reasons(row, contract)
            row["support_pass"] = not reasons
            row["support_failure_reason"] = "; ".join(reasons)
            row["minimum_support_margin"] = min(
                float(row[field]) / float(contract[field])
                for field in ("background_rows", "background_sources", "background_neff", "signal_rows",
                              "transferred_rows", "transferred_sources")
            )
            evaluated.append(row)
        qcd_total = float(sum(row["transferred_qcd_yield"] for row in evaluated))
        direct_total = float(sum(row["direct_qcd_closure_yield"] for row in evaluated))
        nonclosure = abs(qcd_total - direct_total) / abs(direct_total) if direct_total > 0.0 else float("nan")
        nuisance = math.sqrt(float(factor_stat_relative) ** 2 + envelope_relative ** 2 + nonclosure ** 2) if math.isfinite(nonclosure) else float("nan")
        # The nominal support contract selects the frozen boundaries.  It is a
        # selection-stability diagnostic under resampling, not a reason to
        # censor evaluation of an already frozen configuration.  Evaluation
        # fails only when the shared nuisance or the statistic is undefined.
        reasons: list[str] = []
        if not math.isfinite(nuisance):
            reasons.append("shared multijet nuisance undefined")
        nominal_za = float(math.sqrt(sum(row["nominal_asimov_ZA_contribution"] ** 2 for row in evaluated)))
        systematic_za = float("nan")
        if not reasons:
            systematic_za = correlated_multijet_profile_za(
                np.asarray([row["signal_yield"] for row in evaluated]),
                np.asarray([row["ordinary_background_yield"] for row in evaluated]),
                np.asarray([row["transferred_qcd_yield"] for row in evaluated]),
                np.full(category_count, nuisance),
            )
        for row in evaluated:
            row["shared_qcd_relative_nonclosure"] = nonclosure
            row["shared_multijet_relative_nuisance"] = nuisance
            row["systematic_aware_ZA_standalone_contribution"] = (
                correlated_multijet_profile_za(
                    np.asarray([row["signal_yield"]]), np.asarray([row["ordinary_background_yield"]]),
                    np.asarray([row["transferred_qcd_yield"]]), np.asarray([nuisance]),
                ) if math.isfinite(nuisance) else float("nan")
            )
            category_rows.append(row)
        total_signal = float(sum(row["signal_yield"] for row in evaluated))
        total_background = float(sum(row["background_yield"] for row in evaluated))
        signal_sumw2 = float(sum(multiplicities @ group["signal_sumw2"] for group in groups))
        background_sumw2 = float(sum(
            multiplicities @ group["background_sumw2"] for group in groups
        ))
        combined_rows.append({
            "label": label, "replica_id": replica_id,
            "combined_nominal_asimov_ZA": nominal_za,
            "combined_systematic_aware_asimov_ZA": systematic_za,
            "signal_yield": total_signal, "background_yield": total_background,
            "signal_over_background": total_signal / total_background if total_background > 0.0 else float("nan"),
            "signal_over_sqrt_background": total_signal / math.sqrt(total_background) if total_background > 0.0 else float("nan"),
            "background_neff": total_background * total_background / background_sumw2 if background_sumw2 > 0.0 else float("nan"),
            "signal_neff": total_signal * total_signal / signal_sumw2 if signal_sumw2 > 0.0 else float("nan"),
            "transferred_qcd_fraction": qcd_total / total_background if total_background > 0.0 else float("nan"),
            "direct_qcd_closure_ratio": direct_total / qcd_total if qcd_total > 0.0 else float("nan"),
            "shared_qcd_relative_nonclosure": nonclosure,
            "shared_multijet_relative_nuisance": nuisance,
            "minimum_support_margin": min(float(row["minimum_support_margin"]) for row in evaluated),
            "valid": not reasons,
            "failure_reason": "; ".join(reasons),
        })
    return pd.DataFrame(combined_rows), pd.DataFrame(category_rows)


def summarize_replicas(frame: pd.DataFrame, metrics: Iterable[str], keys: list[str]) -> pd.DataFrame:
    rows = []
    for key_values, group in frame.groupby(keys, sort=False, dropna=False):
        if not isinstance(key_values, tuple):
            key_values = (key_values,)
        base = dict(zip(keys, key_values))
        for metric in metrics:
            reasons = group.get("failure_reason", pd.Series([""] * len(group))).astype(str).tolist()
            values = group[metric].to_numpy(dtype=np.float64)
            if np.isfinite(values).any():
                summary = bootstrap_quantile_summary(values, invalid_reasons=reasons)
            else:
                counts: dict[str, int] = {}
                for reason in reasons:
                    key = reason or "nonfinite metric"
                    counts[key] = counts.get(key, 0) + 1
                summary = {
                    "bootstrap_mean": float("nan"), "bootstrap_median": float("nan"),
                    "bootstrap_standard_deviation": float("nan"), "bootstrap_p16": float("nan"),
                    "bootstrap_p84": float("nan"), "bootstrap_p2p5": float("nan"),
                    "bootstrap_p97p5": float("nan"), "valid_replicas": 0,
                    "invalid_replicas": len(values),
                    "invalid_reason_counts_json": json.dumps(counts, sort_keys=True, separators=(",", ":")),
                }
            rows.append({**base, "metric": metric, **summary})
    return pd.DataFrame(rows)
