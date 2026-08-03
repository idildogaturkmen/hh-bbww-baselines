#!/usr/bin/env python3
"""Paired source-member bootstrap evaluation for the pn-c7 ML studies."""

from __future__ import annotations

import itertools
import json
import math
from typing import Any, Iterable

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from pn_c7_ml_common import (
    asimov_za,
    asimov_za_with_uncertainty,
    bootstrap_quantile_summary,
    require,
    require_columns,
)


BOOTSTRAP_METRICS = (
    "weighted_auc",
    "unweighted_auc",
    "signal_efficiency",
    "background_efficiency",
    "background_rejection",
    "selected_signal_yield",
    "selected_background_yield",
    "signal_over_background",
    "signal_over_sqrt_background",
    "background_effective_events",
    "signal_effective_events",
    "nominal_asimov_ZA",
    "systematic_aware_asimov_ZA",
    "transferred_qcd_fraction",
    "direct_qcd_closure_ratio",
    "score_domain_qcd_nonclosure",
)


def member_positions(frame: pd.DataFrame, members: pd.DataFrame, label: str) -> np.ndarray:
    """Map every event row to its source-member column in the common registry."""

    require_columns(frame, ["registry_member_index", "registry_group_id"], label)
    require_columns(members, ["member_index", "transport_id"], "member registry")
    by_index = {int(value): pos for pos, value in enumerate(members["member_index"].astype(int))}
    by_transport = {str(value): pos for pos, value in enumerate(members["transport_id"].astype(str))}
    positions = np.empty(len(frame), dtype=np.int32)
    for row, (member_index, group_id) in enumerate(
        zip(frame["registry_member_index"].astype(int), frame["registry_group_id"].astype(str))
    ):
        require(member_index in by_index, f"{label} references unknown member index: {member_index}")
        require(group_id in by_transport, f"{label} references unknown source group: {group_id}")
        require(by_index[member_index] == by_transport[group_id], f"{label} member/group identity drift")
        positions[row] = by_index[member_index]
    return positions


def group_sum(values: np.ndarray, mask: np.ndarray, positions: np.ndarray, size: int) -> np.ndarray:
    return np.bincount(
        positions[np.asarray(mask, dtype=bool)],
        weights=np.asarray(values, dtype=np.float64)[np.asarray(mask, dtype=bool)],
        minlength=size,
    ).astype(np.float64)


def group_count(mask: np.ndarray, positions: np.ndarray, size: int) -> np.ndarray:
    return np.bincount(positions[np.asarray(mask, dtype=bool)], minlength=size).astype(np.float64)


def _ratio(numerator: np.ndarray, denominator: np.ndarray) -> np.ndarray:
    result = np.full_like(np.asarray(numerator, dtype=np.float64), np.nan)
    good = np.isfinite(numerator) & np.isfinite(denominator) & (denominator > 0.0)
    result[good] = np.asarray(numerator, dtype=np.float64)[good] / np.asarray(denominator, dtype=np.float64)[good]
    return result


def _neff(sumw: np.ndarray, sumw2: np.ndarray) -> np.ndarray:
    return _ratio(np.square(sumw), sumw2)


def _support_reasons(
    transferred_rows: float,
    transferred_sources: int,
    transferred_neff: float,
    direct_rows: float,
    direct_sources: int,
    value: float,
) -> list[str]:
    reasons = []
    if direct_rows <= 0:
        reasons.append("zero direct-QCD closure rows")
    if direct_sources < 3:
        reasons.append("fewer than 3 direct-QCD source groups")
    if transferred_rows < 5:
        reasons.append("fewer than 5 transferred-QCD rows")
    if transferred_sources < 3:
        reasons.append("fewer than 3 transferred-QCD source groups")
    if not math.isfinite(transferred_neff) or transferred_neff < 2.0:
        reasons.append("transferred-QCD effective statistics below 2")
    if not math.isfinite(value):
        reasons.append("nonfinite systematics-aware ZA")
    elif value <= 0.0:
        reasons.append("nonpositive systematics-aware ZA")
    return reasons


def evaluate_frozen_baselines(
    inputs: dict[str, Any],
    draws: np.ndarray,
    *,
    model_order: Iterable[str] = ("cut", "bdt", "dense_dnn", "lbn_dnn"),
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Evaluate frozen operating points for every common source bootstrap draw."""

    training = inputs["training"]
    projection = inputs["projection"]
    direct = inputs["direct"]
    training_scores = inputs["training_scores"]
    projection_scores = inputs["projection_scores"]
    members = inputs["members"].reset_index(drop=True)
    metrics = inputs["metrics"].set_index("baseline")
    require(draws.shape[1] == len(members), "common bootstrap/member alignment drift")

    n_members = len(members)
    train_pos = member_positions(training, members, "training table")
    projection_pos = member_positions(projection, members, "primary projection")
    direct_pos = member_positions(direct, members, "direct closure projection")
    labels = training["registry_training_target"].to_numpy(dtype=np.int8)
    signal_train = labels == 1
    development_weight = training["development_hierarchical_weight"].to_numpy(dtype=np.float64)
    signal_projection = projection["registry_training_target"].to_numpy(dtype=np.int8) == 1
    ordinary = projection["analysis_population_role"].astype(str).to_numpy() == "fourb_ordinary_background"
    transferred = projection["analysis_population_role"].astype(str).to_numpy() == "primary_transferred_multijet_template"
    direct_qcd = direct["population_kind"].astype(str).to_numpy() == "hard_qcd"
    physical = projection["primary_projection_physical_weight_inclusive"].to_numpy(dtype=np.float64)
    qcd_base_weight = projection["run2_candidate_physical_weight"].to_numpy(dtype=np.float64)
    direct_weight = direct["direct_projection_physical_weight"].to_numpy(dtype=np.float64)

    factor = inputs["factor"]
    inclusive = factor[factor["mhh_category"] == "inclusive_mhh"]
    nominal_row = inclusive[inclusive["factor_scheme"] == "cms_cr_nominal"].iloc[0]
    nominal_factor = float(nominal_row["transfer_factor"])
    factor_stat = float(nominal_row["transfer_factor_relative_statistical_uncertainty"])
    factor_values = inclusive["transfer_factor"].astype(float).to_numpy()

    replica_rows: list[dict[str, Any]] = []
    nominal_rows: list[dict[str, Any]] = []
    evaluation_draws = np.vstack([np.ones((1, n_members), dtype=np.int16), draws])
    for model in model_order:
        train_score = training_scores[f"{model}_score"].to_numpy(dtype=np.float64)
        projection_score = projection_scores[f"{model}_score"].to_numpy(dtype=np.float64)
        if model == "cut":
            selection = training["r_hh_125_120"].to_numpy(dtype=np.float64) < 34.0
            projection_selection = projection["r_hh_125_120"].to_numpy(dtype=np.float64) < 34.0
            threshold: float | str = r"R_HH(125,120) < 34 GeV"
        else:
            threshold = float(metrics.loc[model, "operating_threshold"])
            selection = train_score >= threshold
            projection_selection = projection_score >= threshold
        direct_selection = selection

        train_signal_total = group_sum(development_weight, signal_train, train_pos, n_members)
        train_background_total = group_sum(development_weight, ~signal_train, train_pos, n_members)
        train_signal_selected = group_sum(development_weight, selection & signal_train, train_pos, n_members)
        train_background_selected = group_sum(development_weight, selection & ~signal_train, train_pos, n_members)
        signal_yield_group = group_sum(physical, projection_selection & signal_projection, projection_pos, n_members)
        signal_sumw2_group = group_sum(np.square(physical), projection_selection & signal_projection, projection_pos, n_members)
        background_yield_group = group_sum(physical, projection_selection & ~signal_projection, projection_pos, n_members)
        background_sumw2_group = group_sum(np.square(physical), projection_selection & ~signal_projection, projection_pos, n_members)
        ordinary_yield_group = group_sum(physical, projection_selection & ordinary, projection_pos, n_members)
        qcd_base_group = group_sum(qcd_base_weight, projection_selection & transferred, projection_pos, n_members)
        qcd_sumw2_group = group_sum(np.square(physical), projection_selection & transferred, projection_pos, n_members)
        transferred_row_group = group_count(projection_selection & transferred, projection_pos, n_members)
        direct_yield_group = group_sum(direct_weight, direct_selection & direct_qcd, direct_pos, n_members)
        direct_sumw2_group = group_sum(np.square(direct_weight), direct_selection & direct_qcd, direct_pos, n_members)
        direct_row_group = group_count(direct_selection & direct_qcd, direct_pos, n_members)

        signal_efficiency = _ratio(evaluation_draws @ train_signal_selected, evaluation_draws @ train_signal_total)
        background_efficiency = _ratio(evaluation_draws @ train_background_selected, evaluation_draws @ train_background_total)
        signal_yield = evaluation_draws @ signal_yield_group
        background_yield = evaluation_draws @ background_yield_group
        ordinary_yield = evaluation_draws @ ordinary_yield_group
        qcd_base = evaluation_draws @ qcd_base_group
        qcd_yield = qcd_base * nominal_factor
        direct_yield = evaluation_draws @ direct_yield_group
        signal_sumw2 = evaluation_draws @ signal_sumw2_group
        background_sumw2 = evaluation_draws @ background_sumw2_group
        qcd_sumw2 = evaluation_draws @ qcd_sumw2_group
        direct_sumw2 = evaluation_draws @ direct_sumw2_group
        transferred_rows = evaluation_draws @ transferred_row_group
        direct_rows = evaluation_draws @ direct_row_group
        transferred_neff = _neff(qcd_yield, qcd_sumw2)
        direct_neff = _neff(direct_yield, direct_sumw2)
        signal_neff = _neff(signal_yield, signal_sumw2)
        background_neff = _neff(background_yield, background_sumw2)

        for eval_index, multiplicities in enumerate(evaluation_draws):
            replica_id = eval_index - 1
            train_multiplicity = multiplicities[train_pos].astype(np.float64)
            weighted_auc = float(roc_auc_score(labels, train_score, sample_weight=development_weight * train_multiplicity))
            unweighted_auc = float(roc_auc_score(labels, train_score, sample_weight=train_multiplicity))
            qcd = float(qcd_yield[eval_index])
            direct_truth = float(direct_yield[eval_index])
            alternatives = qcd_base[eval_index] * factor_values
            envelope = (
                float(np.max(np.abs(alternatives - qcd)) / abs(qcd)) if qcd != 0.0 else float("nan")
            )
            nonclosure = abs(qcd - direct_truth) / abs(direct_truth) if direct_truth != 0.0 else float("nan")
            sigma = (
                abs(qcd) * math.sqrt(factor_stat**2 + envelope**2 + nonclosure**2)
                if math.isfinite(envelope) and math.isfinite(nonclosure) else float("nan")
            )
            raw_systematic = (
                asimov_za_with_uncertainty(float(signal_yield[eval_index]), float(background_yield[eval_index]), sigma)
                if math.isfinite(sigma) and sigma > 0.0 else float("nan")
            )
            selected_transferred_positions = np.flatnonzero(transferred_row_group > 0)
            selected_direct_positions = np.flatnonzero(direct_row_group > 0)
            transferred_sources = int(np.count_nonzero(multiplicities[selected_transferred_positions]))
            direct_sources = int(np.count_nonzero(multiplicities[selected_direct_positions]))
            reasons = _support_reasons(
                float(transferred_rows[eval_index]), transferred_sources, float(transferred_neff[eval_index]),
                float(direct_rows[eval_index]), direct_sources, raw_systematic,
            )
            system_value = raw_systematic if not reasons else float("nan")
            row = {
                "model": model,
                "replica_id": replica_id,
                "operating_threshold": str(threshold),
                "weighted_auc": weighted_auc,
                "unweighted_auc": unweighted_auc,
                "signal_efficiency": float(signal_efficiency[eval_index]),
                "background_efficiency": float(background_efficiency[eval_index]),
                "background_rejection": float(1.0 - background_efficiency[eval_index]),
                "selected_signal_yield": float(signal_yield[eval_index]),
                "selected_background_yield": float(background_yield[eval_index]),
                "ordinary_background_yield": float(ordinary_yield[eval_index]),
                "transferred_qcd_yield": qcd,
                "direct_qcd_closure_yield": direct_truth,
                "signal_over_background": float(signal_yield[eval_index] / background_yield[eval_index]) if background_yield[eval_index] > 0 else float("nan"),
                "signal_over_sqrt_background": float(signal_yield[eval_index] / math.sqrt(background_yield[eval_index])) if background_yield[eval_index] > 0 else float("nan"),
                "background_effective_events": float(background_neff[eval_index]),
                "signal_effective_events": float(signal_neff[eval_index]),
                "transferred_qcd_effective_events": float(transferred_neff[eval_index]),
                "direct_qcd_effective_events": float(direct_neff[eval_index]),
                "nominal_asimov_ZA": asimov_za(float(signal_yield[eval_index]), float(background_yield[eval_index])),
                "raw_systematic_aware_asimov_ZA": raw_systematic,
                "systematic_aware_asimov_ZA": system_value,
                "transferred_qcd_fraction": float(qcd / background_yield[eval_index]) if background_yield[eval_index] > 0 else float("nan"),
                "direct_qcd_closure_ratio": float(qcd / direct_truth) if direct_truth > 0 else float("nan"),
                "score_domain_qcd_nonclosure": nonclosure,
                "multijet_systematic_absolute": sigma,
                "selected_transferred_qcd_rows": int(transferred_rows[eval_index]),
                "selected_transferred_qcd_sources": transferred_sources,
                "selected_direct_qcd_closure_rows": int(direct_rows[eval_index]),
                "selected_direct_qcd_closure_sources": direct_sources,
                "systematic_support_pass": not reasons,
                "systematic_failure_reason": "; ".join(reasons),
            }
            if replica_id < 0:
                nominal_rows.append(row)
            else:
                replica_rows.append(row)
    return pd.DataFrame(nominal_rows), pd.DataFrame(replica_rows)


def summarize_metric_replicas(nominal: pd.DataFrame, replicas: pd.DataFrame) -> pd.DataFrame:
    rows = []
    nominal_index = nominal.set_index("model")
    for model, group in replicas.groupby("model", sort=False):
        for metric in BOOTSTRAP_METRICS:
            reasons = group["systematic_failure_reason"].astype(str).tolist() if metric == "systematic_aware_asimov_ZA" else [""] * len(group)
            summary = bootstrap_quantile_summary(group[metric].to_numpy(dtype=float), invalid_reasons=reasons)
            rows.append({
                "model": model,
                "metric": metric,
                "nominal_full_oof": float(nominal_index.loc[model, metric]),
                **summary,
                "uncertainty_format": "median_asymmetric_p16_p84",
                "bootstrap_unit": "source_member",
            })
    return pd.DataFrame(rows)


def paired_model_differences(
    nominal: pd.DataFrame,
    replicas: pd.DataFrame,
    *,
    model_order: Iterable[str] = ("cut", "bdt", "dense_dnn", "lbn_dnn"),
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Calculate all pairwise model-minus-reference intervals on aligned draws."""

    order = list(model_order)
    nominal_index = nominal.set_index("model")
    replica_index = replicas.set_index(["model", "replica_id"]).sort_index()
    summary_rows = []
    replica_rows = []
    for reference, model in itertools.combinations(order, 2):
        ref = replica_index.loc[reference]
        candidate = replica_index.loc[model]
        require(ref.index.equals(candidate.index), f"paired replica-ID alignment drift: {model}/{reference}")
        for metric in BOOTSTRAP_METRICS:
            difference = candidate[metric].to_numpy(dtype=float) - ref[metric].to_numpy(dtype=float)
            finite = np.isfinite(difference)
            summary = bootstrap_quantile_summary(difference)
            summary_rows.append({
                "comparison": f"{model}_minus_{reference}",
                "model": model,
                "reference_model": reference,
                "metric": metric,
                "nominal_difference": float(nominal_index.loc[model, metric] - nominal_index.loc[reference, metric]),
                "bootstrap_median_difference": summary["bootstrap_median"],
                "bootstrap_p16_difference": summary["bootstrap_p16"],
                "bootstrap_p84_difference": summary["bootstrap_p84"],
                "bootstrap_p2p5_difference": summary["bootstrap_p2p5"],
                "bootstrap_p97p5_difference": summary["bootstrap_p97p5"],
                "valid_replicas": summary["valid_replicas"],
                "invalid_replicas": summary["invalid_replicas"],
                "fraction_valid_difference_greater_than_zero": float(np.mean(difference[finite] > 0.0)),
                "interval_includes_zero_68": bool(summary["bootstrap_p16"] <= 0.0 <= summary["bootstrap_p84"]),
                "interval_includes_zero_95": bool(summary["bootstrap_p2p5"] <= 0.0 <= summary["bootstrap_p97p5"]),
            })
            replica_rows.extend({
                "comparison": f"{model}_minus_{reference}",
                "model": model,
                "reference_model": reference,
                "metric": metric,
                "replica_id": int(replica_id),
                "difference": float(value) if math.isfinite(value) else float("nan"),
                "valid": bool(math.isfinite(value)),
            } for replica_id, value in zip(ref.index, difference))
    return pd.DataFrame(summary_rows), pd.DataFrame(replica_rows)


def bootstrap_validity_summary(replicas: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for model, group in replicas.groupby("model", sort=False):
        reasons: dict[str, int] = {}
        for reason in group.loc[~group["systematic_support_pass"], "systematic_failure_reason"].astype(str):
            for item in filter(None, reason.split("; ")):
                reasons[item] = reasons.get(item, 0) + 1
        rows.append({
            "model": model,
            "replicas": len(group),
            "valid_systematics_replicas": int(group["systematic_support_pass"].sum()),
            "invalid_systematics_replicas": int((~group["systematic_support_pass"]).sum()),
            "valid_fraction": float(group["systematic_support_pass"].mean()),
            "zero_direct_support_fraction": float(group["systematic_failure_reason"].str.contains("zero direct").mean()),
            "failure_reason_counts_json": json.dumps(reasons, sort_keys=True, separators=(",", ":")),
        })
    return pd.DataFrame(rows)
