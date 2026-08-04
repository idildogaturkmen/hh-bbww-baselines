#!/usr/bin/env python3
"""Torch-free source-member bootstrap evaluation for PN-c7x assignment."""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import pandas as pd

from scripts.analysis.pn_c7_ml_common import bootstrap_quantile_summary, require, require_columns


ASSIGNMENT_METRICS = (
    "matchable_event_efficiency",
    "ambiguity_fraction",
    "unmatched_fraction",
    "learned_exact_event_pairing_accuracy",
    "learned_weighted_exact_event_pairing_accuracy",
    "learned_per_higgs_pairing_accuracy",
    "learned_per_jet_partner_accuracy",
    "geometric_exact_event_pairing_accuracy",
    "geometric_weighted_exact_event_pairing_accuracy",
    "geometric_per_higgs_pairing_accuracy",
    "geometric_per_jet_partner_accuracy",
    "learned_mass_residual_MAE_GeV",
    "geometric_mass_residual_MAE_GeV",
    "learned_mass_residual_RMS_GeV",
    "geometric_mass_residual_RMS_GeV",
)

PAIRING_BIN_CONTRACTS = {
    "mHH_GeV": (np.asarray([0.0, 400.0, 600.0, 800.0, 1200.0, np.inf]), "$m_{HH}$ [GeV]"),
    "truth_Higgs_pT_GeV": (np.asarray([0.0, 100.0, 200.0, 300.0, 500.0, np.inf]), "Truth Higgs $p_T$ [GeV]"),
    "selected_jet_multiplicity": (np.asarray([4.0, 5.0, 6.0, 7.0, np.inf]), "Selected-jet multiplicity"),
    "extra_jet_activity": (np.asarray([0.0, 1.0, 2.0, 3.0, np.inf]), "Extra selected jets"),
}


def member_positions(frame: pd.DataFrame, members: pd.DataFrame, label: str) -> np.ndarray:
    require_columns(frame, ["registry_member_index", "registry_group_id"], label)
    require_columns(members, ["member_index", "transport_id"], "member registry")
    by_index = {int(value): position for position, value in enumerate(members.member_index.astype(int))}
    by_group = {str(value): position for position, value in enumerate(members.transport_id.astype(str))}
    positions = np.empty(len(frame), dtype=np.int32)
    for row, (member_index, group_id) in enumerate(zip(frame.registry_member_index, frame.registry_group_id)):
        member_index = int(member_index)
        group_id = str(group_id)
        require(member_index in by_index and group_id in by_group, f"unknown source identity in {label}")
        require(by_index[member_index] == by_group[group_id], f"member/group identity drift in {label}")
        positions[row] = by_index[member_index]
    return positions


def group_sum(values: np.ndarray, mask: np.ndarray, positions: np.ndarray, size: int) -> np.ndarray:
    selected = np.asarray(mask, dtype=bool)
    return np.bincount(
        positions[selected], weights=np.asarray(values, dtype=np.float64)[selected], minlength=size
    ).astype(np.float64)


def ratio(numerator: np.ndarray, denominator: np.ndarray) -> np.ndarray:
    result = np.full(len(numerator), np.nan, dtype=np.float64)
    valid = np.isfinite(numerator) & np.isfinite(denominator) & (denominator > 0.0)
    result[valid] = numerator[valid] / denominator[valid]
    return result


def evaluate_assignment_bootstrap(
    predictions: pd.DataFrame,
    members: pd.DataFrame,
    draws: np.ndarray,
) -> pd.DataFrame:
    signal = predictions[predictions.registry_sample_class.eq("signal")].copy().reset_index(drop=True)
    require(len(signal) == 9337, "assignment signal row-count drift")
    require(not signal[["truth_status", "truth_partition_label"]].isna().any().any(), "assignment truth incomplete")
    members = members.reset_index(drop=True)
    require(draws.shape[1] == len(members), "assignment bootstrap/member alignment drift")
    positions = member_positions(signal, members, "assignment predictions")
    size = len(members)
    evaluation_draws = np.vstack((np.ones((1, size), dtype=np.int16), draws))
    all_rows = np.ones(len(signal), dtype=bool)
    matchable = signal.assignment_loss_mask.astype(bool).to_numpy()
    ambiguous = signal.truth_status.eq("ambiguous_match").to_numpy()
    unmatched = signal.truth_status.eq("unmatched").to_numpy()
    truth = signal.truth_partition_label.to_numpy(dtype=np.int8)
    learned = signal.predicted_partition_label.to_numpy(dtype=np.int8)
    geometric = signal.frozen_geometric_partition_label.to_numpy(dtype=np.int8)
    learned_correct = matchable & (learned == truth)
    geometric_correct = matchable & (geometric == truth)
    weights = signal.development_hierarchical_weight.to_numpy(dtype=np.float64)

    all_count = evaluation_draws @ group_sum(np.ones(len(signal)), all_rows, positions, size)
    match_count = evaluation_draws @ group_sum(np.ones(len(signal)), matchable, positions, size)
    ambiguous_count = evaluation_draws @ group_sum(np.ones(len(signal)), ambiguous, positions, size)
    unmatched_count = evaluation_draws @ group_sum(np.ones(len(signal)), unmatched, positions, size)
    learned_count = evaluation_draws @ group_sum(np.ones(len(signal)), learned_correct, positions, size)
    geometric_count = evaluation_draws @ group_sum(np.ones(len(signal)), geometric_correct, positions, size)
    match_weight = evaluation_draws @ group_sum(weights, matchable, positions, size)
    learned_weight = evaluation_draws @ group_sum(weights, learned_correct, positions, size)
    geometric_weight = evaluation_draws @ group_sum(weights, geometric_correct, positions, size)
    learned_mae = evaluation_draws @ group_sum(
        signal.learned_pair_mean_absolute_mass_residual.to_numpy(dtype=np.float64), matchable, positions, size
    )
    geometric_mae = evaluation_draws @ group_sum(
        signal.frozen_pair_mean_absolute_mass_residual.to_numpy(dtype=np.float64), matchable, positions, size
    )
    learned_mse = evaluation_draws @ group_sum(
        signal.learned_pair_mean_squared_mass_residual.to_numpy(dtype=np.float64), matchable, positions, size
    )
    geometric_mse = evaluation_draws @ group_sum(
        signal.frozen_pair_mean_squared_mass_residual.to_numpy(dtype=np.float64), matchable, positions, size
    )

    learned_accuracy = ratio(learned_count, match_count)
    geometric_accuracy = ratio(geometric_count, match_count)
    values = {
        "matchable_event_efficiency": ratio(match_count, all_count),
        "ambiguity_fraction": ratio(ambiguous_count, all_count),
        "unmatched_fraction": ratio(unmatched_count, all_count),
        "learned_exact_event_pairing_accuracy": learned_accuracy,
        "learned_weighted_exact_event_pairing_accuracy": ratio(learned_weight, match_weight),
        "learned_per_higgs_pairing_accuracy": learned_accuracy,
        "learned_per_jet_partner_accuracy": learned_accuracy,
        "geometric_exact_event_pairing_accuracy": geometric_accuracy,
        "geometric_weighted_exact_event_pairing_accuracy": ratio(geometric_weight, match_weight),
        "geometric_per_higgs_pairing_accuracy": geometric_accuracy,
        "geometric_per_jet_partner_accuracy": geometric_accuracy,
        "learned_mass_residual_MAE_GeV": ratio(learned_mae, match_count),
        "geometric_mass_residual_MAE_GeV": ratio(geometric_mae, match_count),
        "learned_mass_residual_RMS_GeV": np.sqrt(ratio(learned_mse, match_count)),
        "geometric_mass_residual_RMS_GeV": np.sqrt(ratio(geometric_mse, match_count)),
    }
    rows = []
    for evaluation_index in range(len(evaluation_draws)):
        rows.append({
            "replica_id": evaluation_index - 1,
            **{metric: float(metric_values[evaluation_index]) for metric, metric_values in values.items()},
            "signal_rows": int(all_count[evaluation_index]),
            "matchable_rows": int(match_count[evaluation_index]),
            "ambiguous_rows": int(ambiguous_count[evaluation_index]),
            "unmatched_rows": int(unmatched_count[evaluation_index]),
        })
    return pd.DataFrame(rows)


def summarize_assignment_replicas(evaluated: pd.DataFrame) -> pd.DataFrame:
    nominal = evaluated[evaluated.replica_id == -1].iloc[0]
    replicas = evaluated[evaluated.replica_id >= 0]
    rows = []
    for metric in ASSIGNMENT_METRICS:
        rows.append({
            "metric": metric,
            "nominal_full_oof": float(nominal[metric]),
            **bootstrap_quantile_summary(replicas[metric].to_numpy(dtype=np.float64)),
            "uncertainty_format": "median_asymmetric_p16_p84",
            "bootstrap_unit": "source_member",
        })
    return pd.DataFrame(rows)


def assignment_paired_differences(evaluated: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    pairs = (
        ("exact_event_pairing_accuracy", "learned_exact_event_pairing_accuracy", "geometric_exact_event_pairing_accuracy"),
        ("weighted_exact_event_pairing_accuracy", "learned_weighted_exact_event_pairing_accuracy", "geometric_weighted_exact_event_pairing_accuracy"),
        ("mass_residual_MAE_GeV", "learned_mass_residual_MAE_GeV", "geometric_mass_residual_MAE_GeV"),
        ("mass_residual_RMS_GeV", "learned_mass_residual_RMS_GeV", "geometric_mass_residual_RMS_GeV"),
    )
    summary_rows = []
    replica_rows = []
    for metric, learned, geometric in pairs:
        difference = evaluated[learned].to_numpy(dtype=np.float64) - evaluated[geometric].to_numpy(dtype=np.float64)
        nominal_difference = float(difference[evaluated.replica_id.to_numpy() == -1][0])
        replica_difference = difference[evaluated.replica_id.to_numpy() >= 0]
        summary = bootstrap_quantile_summary(replica_difference)
        finite = np.isfinite(replica_difference)
        summary_rows.append({
            "comparison": "single_head_learned_minus_frozen_geometric",
            "metric": metric,
            "nominal_difference": nominal_difference,
            "bootstrap_median_difference": summary["bootstrap_median"],
            "bootstrap_p16_difference": summary["bootstrap_p16"],
            "bootstrap_p84_difference": summary["bootstrap_p84"],
            "bootstrap_p2p5_difference": summary["bootstrap_p2p5"],
            "bootstrap_p97p5_difference": summary["bootstrap_p97p5"],
            "valid_replicas": summary["valid_replicas"],
            "invalid_replicas": summary["invalid_replicas"],
            "fraction_valid_difference_greater_than_zero": float(np.mean(replica_difference[finite] > 0.0)),
            "interval_includes_zero_68": bool(summary["bootstrap_p16"] <= 0.0 <= summary["bootstrap_p84"]),
            "interval_includes_zero_95": bool(summary["bootstrap_p2p5"] <= 0.0 <= summary["bootstrap_p97p5"]),
        })
        replica_rows.extend({
            "comparison": "single_head_learned_minus_frozen_geometric",
            "metric": metric,
            "replica_id": int(replica_id),
            "difference": float(value),
            "valid": bool(math.isfinite(value)),
        } for replica_id, value in zip(evaluated.loc[evaluated.replica_id >= 0, "replica_id"], replica_difference))
    return pd.DataFrame(summary_rows), pd.DataFrame(replica_rows)


def _bin_label(low: float, high: float) -> str:
    if math.isinf(high):
        return f">={low:g}"
    return f"[{low:g},{high:g})"


def evaluate_binned_pairing(
    predictions: pd.DataFrame,
    members: pd.DataFrame,
    draws: np.ndarray,
) -> pd.DataFrame:
    signal = predictions[
        predictions.registry_sample_class.eq("signal") & predictions.assignment_loss_mask.astype(bool)
    ].copy().reset_index(drop=True)
    positions = member_positions(signal, members.reset_index(drop=True), "binned assignment predictions")
    learned_correct = (
        signal.predicted_partition_label.to_numpy(dtype=np.int8)
        == signal.truth_partition_label.to_numpy(dtype=np.int8)
    )
    geometric_correct = (
        signal.frozen_geometric_partition_label.to_numpy(dtype=np.int8)
        == signal.truth_partition_label.to_numpy(dtype=np.int8)
    )
    evaluation_draws = np.vstack((np.ones((1, len(members)), dtype=np.int16), draws))
    rows = []
    for variable, (edges, axis_label) in PAIRING_BIN_CONTRACTS.items():
        if variable == "truth_Higgs_pT_GeV":
            values = np.concatenate((
                signal.truth_higgs_pt_low.to_numpy(dtype=np.float64),
                signal.truth_higgs_pt_high.to_numpy(dtype=np.float64),
            ))
            local_positions = np.concatenate((positions, positions))
            local_learned = np.concatenate((learned_correct, learned_correct))
            local_geometric = np.concatenate((geometric_correct, geometric_correct))
        else:
            column = {
                "mHH_GeV": "mhh",
                "selected_jet_multiplicity": "n_selected_jets",
                "extra_jet_activity": "n_extra_selected_jets",
            }[variable]
            values = signal[column].to_numpy(dtype=np.float64)
            local_positions = positions
            local_learned = learned_correct
            local_geometric = geometric_correct
        for bin_index, (low, high) in enumerate(zip(edges[:-1], edges[1:])):
            selected = (values >= low) & (values < high)
            denominator_group = group_sum(np.ones(len(values)), selected, local_positions, len(members))
            denominator = evaluation_draws @ denominator_group
            for model, correct in (("single_head_learned", local_learned), ("frozen_geometric", local_geometric)):
                numerator_group = group_sum(np.ones(len(values)), selected & correct, local_positions, len(members))
                accuracy = ratio(evaluation_draws @ numerator_group, denominator)
                for evaluation_index, value in enumerate(accuracy):
                    rows.append({
                        "variable": variable,
                        "axis_label": axis_label,
                        "bin_index": bin_index,
                        "bin_low": float(low),
                        "bin_high": float(high),
                        "bin_label": _bin_label(float(low), float(high)),
                        "model": model,
                        "replica_id": evaluation_index - 1,
                        "pairing_accuracy": float(value),
                        "observations": int(denominator[evaluation_index]),
                        "valid": bool(math.isfinite(value)),
                    })
    return pd.DataFrame(rows)


def summarize_binned_pairing(evaluated: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    summary_rows = []
    paired_rows = []
    keys = ["variable", "axis_label", "bin_index", "bin_low", "bin_high", "bin_label"]
    for key_values, bin_frame in evaluated.groupby(keys, sort=False, dropna=False):
        base = dict(zip(keys, key_values))
        for model, model_frame in bin_frame.groupby("model", sort=False):
            nominal = model_frame.loc[model_frame.replica_id == -1, "pairing_accuracy"].iloc[0]
            replicas = model_frame.loc[model_frame.replica_id >= 0].sort_values("replica_id")
            summary_rows.append({
                **base,
                "model": model,
                "nominal_full_oof": float(nominal),
                **bootstrap_quantile_summary(replicas.pairing_accuracy.to_numpy(dtype=np.float64)),
            })
        learned = bin_frame[(bin_frame.model == "single_head_learned")].sort_values("replica_id")
        geometric = bin_frame[(bin_frame.model == "frozen_geometric")].sort_values("replica_id")
        require(learned.replica_id.reset_index(drop=True).equals(geometric.replica_id.reset_index(drop=True)), "binned paired replica drift")
        differences = learned.pairing_accuracy.to_numpy(dtype=np.float64) - geometric.pairing_accuracy.to_numpy(dtype=np.float64)
        nominal_difference = float(differences[learned.replica_id.to_numpy() == -1][0])
        replica_difference = differences[learned.replica_id.to_numpy() >= 0]
        summary = bootstrap_quantile_summary(replica_difference)
        finite = np.isfinite(replica_difference)
        paired_rows.append({
            **base,
            "comparison": "single_head_learned_minus_frozen_geometric",
            "nominal_difference": nominal_difference,
            "bootstrap_median_difference": summary["bootstrap_median"],
            "bootstrap_p16_difference": summary["bootstrap_p16"],
            "bootstrap_p84_difference": summary["bootstrap_p84"],
            "bootstrap_p2p5_difference": summary["bootstrap_p2p5"],
            "bootstrap_p97p5_difference": summary["bootstrap_p97p5"],
            "valid_replicas": summary["valid_replicas"],
            "invalid_replicas": summary["invalid_replicas"],
            "fraction_valid_difference_greater_than_zero": float(np.mean(replica_difference[finite] > 0.0)),
            "interval_includes_zero_68": bool(summary["bootstrap_p16"] <= 0.0 <= summary["bootstrap_p84"]),
        })
    return pd.DataFrame(summary_rows), pd.DataFrame(paired_rows)
