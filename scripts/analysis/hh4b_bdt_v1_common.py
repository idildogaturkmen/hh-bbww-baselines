#!/usr/bin/env python3
"""Shared contracts for the canonical HH4b BDT-v1 input preparation.

This module deliberately contains no model-training code.  It freezes the
feature ordering, derived scalar construction, grouped fold assignment, and
development-only hierarchical row weighting used by later benchmark gates.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd


COMMON_FEATURES: tuple[str, ...] = (
    "n_selected_jets",
    "n_selected_bjets",
    "ht_selected_jets",
    "ht_selected_bjets",
    "ht_candidate_jets",
    "mhh",
    "hh_pt",
    "abs_hh_eta",
    "h1_pt",
    "abs_h1_eta",
    "h2_pt",
    "abs_h2_eta",
    "abs_h_delta_eta",
    "abs_h_delta_phi",
    "h_delta_r",
    "h_pt_balance",
    "drbb1",
    "drbb2",
    "j1_pt",
    "abs_j1_eta",
    "j1_mass",
    "j2_pt",
    "abs_j2_eta",
    "j2_mass",
    "j3_pt",
    "abs_j3_eta",
    "j3_mass",
    "j4_pt",
    "abs_j4_eta",
    "j4_mass",
)

MASS_PLANE_FEATURES: tuple[str, ...] = (
    "mbb1",
    "mbb2",
    "delta_mbb",
    "r_hh_125_125",
)

MASS_AWARE_FEATURES: tuple[str, ...] = COMMON_FEATURES + MASS_PLANE_FEATURES
MASS_PLANE_BLIND_FEATURES: tuple[str, ...] = COMMON_FEATURES

DERIVED_FEATURE_EXPRESSIONS: Mapping[str, str] = {
    "abs_hh_eta": "abs(hh_eta)",
    "abs_h1_eta": "abs(h1_eta)",
    "abs_h2_eta": "abs(h2_eta)",
    "abs_h_delta_eta": "abs(h_delta_eta)",
    "abs_h_delta_phi": "abs(h_delta_phi)",
    "abs_j1_eta": "abs(j1_eta)",
    "abs_j2_eta": "abs(j2_eta)",
    "abs_j3_eta": "abs(j3_eta)",
    "abs_j4_eta": "abs(j4_eta)",
}

DERIVED_FEATURE_SOURCES: Mapping[str, str] = {
    name: expression[4:-1]
    for name, expression in DERIVED_FEATURE_EXPRESSIONS.items()
}

SOURCE_COLUMNS: tuple[str, ...] = tuple(
    dict.fromkeys(
        DERIVED_FEATURE_SOURCES.get(name, name)
        for name in MASS_AWARE_FEATURES
    )
)

FORBIDDEN_EXACT_FEATURES: frozenset[str] = frozenset(
    {
        "sample",
        "event",
        "sample_class",
        "training_target",
        "class_label",
        "process",
        "process_or_mode",
        "process_family",
        "dataset_split",
        "split",
        "member_index",
        "source_member",
        "source_path",
        "registered_path",
        "local_path",
        "campaign",
        "campaign_id",
        "candidate_path",
        "candidate_file_path",
        "candidate_sha256",
        "candidate_file_checksum",
        "j1_flavor",
        "j2_flavor",
        "j3_flavor",
        "j4_flavor",
        "pairing",
        "pairing_combo_bjet_ranks",
        "higgs_ordering",
        "source_root",
        "source_root_index",
        "analysis_sample",
        "hh_phi",
        "h1_phi",
        "h2_phi",
        "j1_phi",
        "j2_phi",
        "j3_phi",
        "j4_phi",
        "j1_btag",
        "j2_btag",
        "j3_btag",
        "j4_btag",
        "r_hh",
        "r_hh_125_120",
        "avg_mbb",
        "pairing_score_125_125",
    }
)

FORBIDDEN_SUBSTRINGS: tuple[str, ...] = (
    "_raw_index",
    "_selected_index",
    "_bjet_rank",
    "promoted_jet",
    "promoted_candidate",
    "threeb",
)


@dataclass(frozen=True)
class FoldAssignmentResult:
    """A deterministic member-level fold assignment and its algorithm label."""

    assignment: dict[int, int]
    algorithm: str


@dataclass(frozen=True)
class WeightResult:
    """Row weights and the metadata needed to audit their hierarchy."""

    weights: np.ndarray
    raw_weights: np.ndarray
    rescale_factor: float
    member_rows: list[dict[str, Any]]
    summary_rows: list[dict[str, Any]]


def add_derived_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Return a copy with the frozen deterministic absolute-value features."""

    missing = sorted(set(DERIVED_FEATURE_SOURCES.values()) - set(frame.columns))
    if missing:
        raise ValueError(f"missing derived-feature source columns: {missing}")

    result = frame.copy()
    for target, source in DERIVED_FEATURE_SOURCES.items():
        result[target] = np.abs(result[source].to_numpy())
    return result


def forbidden_feature_reason(name: str) -> str | None:
    """Return the frozen rejection reason for a prohibited input name."""

    if name in FORBIDDEN_EXACT_FEATURES:
        return "explicitly_forbidden_input"
    lowered = name.lower()
    if lowered.startswith("jet.") or "generator" in lowered or "gen_" in lowered:
        return "generator_or_raw_detector_collection_input"
    if any(token in lowered for token in FORBIDDEN_SUBSTRINGS):
        return "forbidden_identity_rank_or_threeb_provenance_input"
    return None


def validate_feature_names(features: Sequence[str]) -> None:
    """Reject duplicates and every frozen forbidden input."""

    duplicates = sorted(
        name for name, count in Counter(features).items() if count > 1
    )
    if duplicates:
        raise ValueError(f"duplicate selected features: {duplicates}")

    forbidden = {
        name: reason
        for name in features
        if (reason := forbidden_feature_reason(name)) is not None
    }
    if forbidden:
        raise ValueError(f"forbidden selected features: {forbidden}")


def feature_quality_rows(
    frame: pd.DataFrame,
    features: Sequence[str],
) -> list[dict[str, Any]]:
    """Audit the selected feature values without modifying any row."""

    validate_feature_names(features)
    missing_columns = sorted(set(features) - set(frame.columns))
    if missing_columns:
        raise ValueError(f"missing selected feature columns: {missing_columns}")

    rows: list[dict[str, Any]] = []
    for order, feature in enumerate(features, start=1):
        series = frame[feature]
        numeric = pd.to_numeric(series, errors="coerce")
        values = numeric.to_numpy(dtype=np.float64, copy=False)
        missing_mask = series.isna().to_numpy()
        finite_mask = np.isfinite(values)
        missing_rows = int(np.count_nonzero(missing_mask))
        nonfinite_rows = int(np.count_nonzero(~finite_mask & ~missing_mask))
        finite_values = values[finite_mask]
        unique_values = int(pd.Series(values).nunique(dropna=False))
        constant = unique_values <= 1

        rows.append(
            {
                "feature_order_mass_aware": order,
                "feature": feature,
                "source_column_or_expression": DERIVED_FEATURE_EXPRESSIONS.get(
                    feature, feature
                ),
                "logical_dtype": str(series.dtype),
                "minimum": (
                    float(np.min(finite_values)) if finite_values.size else ""
                ),
                "maximum": (
                    float(np.max(finite_values)) if finite_values.size else ""
                ),
                "mean": (
                    float(np.mean(finite_values)) if finite_values.size else ""
                ),
                "standard_deviation": (
                    float(np.std(finite_values, ddof=0))
                    if finite_values.size
                    else ""
                ),
                "finite_rows": int(finite_values.size),
                "nonfinite_rows": nonfinite_rows,
                "missing_rows": missing_rows,
                "unique_values": unique_values,
                "constant_feature": constant,
                "status": (
                    "pass"
                    if missing_rows == 0
                    and nonfinite_rows == 0
                    and not constant
                    else "fail"
                ),
            }
        )
    return rows


def assign_member_folds(
    members: Sequence[Mapping[str, Any]],
    *,
    n_folds: int = 5,
) -> FoldAssignmentResult:
    """Assign immutable members with a deterministic stratified greedy rule.

    Members are stratified by signal mode or background family.  Within every
    stratum, larger candidate populations are placed first into the fold with
    the smallest current stratum row count, then stratum member count, class
    row count, total row count, and finally fold index.  This distributes
    zero-row members as well as positive-row members and never splits a member.
    """

    if n_folds < 2:
        raise ValueError("n_folds must be at least two")
    if not members:
        raise ValueError("at least one member is required")

    required = {"member_index", "sample_class", "stratum", "candidate_rows"}
    normalized: list[dict[str, Any]] = []
    seen: set[int] = set()
    for raw in members:
        missing = required - set(raw)
        if missing:
            raise ValueError(f"member record missing fields: {sorted(missing)}")
        member_index = int(raw["member_index"])
        if member_index in seen:
            raise ValueError(f"duplicate member_index: {member_index}")
        seen.add(member_index)
        candidate_rows = int(raw["candidate_rows"])
        if candidate_rows < 0:
            raise ValueError("candidate_rows must be nonnegative")
        sample_class = str(raw["sample_class"])
        if sample_class not in {"signal", "background"}:
            raise ValueError(f"unsupported sample_class: {sample_class}")
        normalized.append(
            {
                "member_index": member_index,
                "sample_class": sample_class,
                "stratum": str(raw["stratum"]),
                "candidate_rows": candidate_rows,
            }
        )

    by_stratum: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for member in normalized:
        by_stratum[
            (member["sample_class"], member["stratum"])
        ].append(member)

    fold_total_rows = [0] * n_folds
    fold_class_rows = {
        sample_class: [0] * n_folds
        for sample_class in ("signal", "background")
    }
    fold_class_members = {
        sample_class: [0] * n_folds
        for sample_class in ("signal", "background")
    }
    fold_total_members = [0] * n_folds
    assignment: dict[int, int] = {}

    for stratum_key in sorted(by_stratum):
        sample_class, _ = stratum_key
        stratum_rows = [0] * n_folds
        stratum_members = [0] * n_folds
        ordered = sorted(
            by_stratum[stratum_key],
            key=lambda row: (-row["candidate_rows"], row["member_index"]),
        )
        for member in ordered:
            if member["candidate_rows"] == 0:
                key = lambda candidate_fold: (
                    stratum_members[candidate_fold],
                    fold_class_members[sample_class][candidate_fold],
                    fold_total_members[candidate_fold],
                    stratum_rows[candidate_fold],
                    fold_class_rows[sample_class][candidate_fold],
                    fold_total_rows[candidate_fold],
                    candidate_fold,
                )
            else:
                key = lambda candidate_fold: (
                    stratum_rows[candidate_fold],
                    stratum_members[candidate_fold],
                    fold_class_rows[sample_class][candidate_fold],
                    fold_total_rows[candidate_fold],
                    fold_class_members[sample_class][candidate_fold],
                    fold_total_members[candidate_fold],
                    candidate_fold,
                )
            fold = min(
                range(n_folds),
                key=key,
            )
            member_rows = member["candidate_rows"]
            assignment[member["member_index"]] = fold
            stratum_rows[fold] += member_rows
            stratum_members[fold] += 1
            fold_class_rows[sample_class][fold] += member_rows
            fold_class_members[sample_class][fold] += 1
            fold_total_rows[fold] += member_rows
            fold_total_members[fold] += 1

    if len(assignment) != len(normalized):
        raise AssertionError("internal fold-assignment cardinality failure")

    return FoldAssignmentResult(
        assignment=assignment,
        algorithm=(
            "deterministic_stratum_greedy_largest_candidate_count_first_v1"
        ),
    )


def build_hierarchical_weights(
    row_member_indices: Sequence[int],
    members: Sequence[Mapping[str, Any]],
    *,
    signal_modes: Sequence[str],
    background_families: Sequence[str],
    tolerance: float = 1.0e-10,
) -> WeightResult:
    """Build nonphysical hierarchical development-balancing row weights.

    The hierarchy gives each class half the total, each signal mode half of the
    signal class, each configured background family an equal share of the
    background class, and each candidate-bearing member an equal share inside
    its stratum.  Members with zero candidates remain in folds and audits but
    cannot carry a row weight because they contribute no row.
    """

    member_by_index: dict[int, dict[str, Any]] = {}
    for raw in members:
        member = dict(raw)
        index = int(member["member_index"])
        if index in member_by_index:
            raise ValueError(f"duplicate member_index: {index}")
        member["candidate_rows"] = int(member["candidate_rows"])
        member_by_index[index] = member

    row_members = np.asarray(row_member_indices, dtype=np.int64)
    if row_members.ndim != 1 or row_members.size == 0:
        raise ValueError("row_member_indices must be a nonempty 1D sequence")
    unknown = sorted(set(row_members.tolist()) - set(member_by_index))
    if unknown:
        raise ValueError(f"row references unknown members: {unknown}")

    observed_counts = Counter(int(value) for value in row_members)
    for index, member in member_by_index.items():
        expected = int(member["candidate_rows"])
        observed = observed_counts.get(index, 0)
        if observed != expected:
            raise ValueError(
                f"member {index} row mismatch: expected {expected}, observed {observed}"
            )

    signal_modes = tuple(signal_modes)
    background_families = tuple(background_families)
    if not signal_modes or not background_families:
        raise ValueError("signal modes and background families must be nonempty")

    stratum_fraction: dict[tuple[str, str], float] = {}
    for mode in signal_modes:
        stratum_fraction[("signal", mode)] = 0.5 / len(signal_modes)
    for family in background_families:
        stratum_fraction[("background", family)] = (
            0.5 / len(background_families)
        )

    members_by_stratum: dict[tuple[str, str], list[int]] = defaultdict(list)
    for index, member in member_by_index.items():
        key = (str(member["sample_class"]), str(member["stratum"]))
        if key not in stratum_fraction:
            raise ValueError(f"member {index} has unconfigured stratum {key}")
        members_by_stratum[key].append(index)

    raw_member_weight: dict[int, float] = {}
    candidate_bearing_by_stratum: dict[tuple[str, str], list[int]] = {}
    for key in sorted(stratum_fraction):
        positive = sorted(
            index
            for index in members_by_stratum.get(key, [])
            if member_by_index[index]["candidate_rows"] > 0
        )
        if not positive:
            raise ValueError(f"stratum has no candidate-bearing members: {key}")
        candidate_bearing_by_stratum[key] = positive
        per_member = stratum_fraction[key] / len(positive)
        for index in members_by_stratum[key]:
            raw_member_weight[index] = per_member if index in positive else 0.0

    raw = np.empty(row_members.size, dtype=np.float64)
    for position, member_index in enumerate(row_members):
        member_index = int(member_index)
        member_rows = member_by_index[member_index]["candidate_rows"]
        raw[position] = raw_member_weight[member_index] / member_rows

    raw_sum = float(np.sum(raw))
    if not np.isclose(raw_sum, 1.0, rtol=0.0, atol=tolerance):
        raise ValueError(f"raw hierarchical weights do not sum to one: {raw_sum}")
    rescale = float(row_members.size / raw_sum)
    weights = raw * rescale

    if not np.all(np.isfinite(weights)) or not np.all(weights > 0.0):
        raise ValueError("training weights must be finite and positive")
    if not np.isclose(
        float(np.mean(weights)), 1.0, rtol=0.0, atol=tolerance
    ):
        raise ValueError("normalized training weight mean is not one")

    member_rows: list[dict[str, Any]] = []
    for index in sorted(member_by_index):
        member = member_by_index[index]
        key = (str(member["sample_class"]), str(member["stratum"]))
        count = member["candidate_rows"]
        raw_aggregate = raw_member_weight[index]
        member_rows.append(
            {
                "member_index": index,
                "sample_class": key[0],
                "stratum": key[1],
                "candidate_rows": count,
                "weight_eligibility": (
                    "candidate_bearing_member"
                    if count > 0
                    else "no_candidate_rows_not_row_weightable"
                ),
                "candidate_bearing_members_in_stratum": len(
                    candidate_bearing_by_stratum[key]
                ),
                "hierarchical_stratum_fraction": stratum_fraction[key],
                "hierarchical_member_fraction": raw_aggregate,
                "per_row_raw_weight": (
                    raw_aggregate / count if count > 0 else 0.0
                ),
                "per_row_training_weight": (
                    raw_aggregate * rescale / count if count > 0 else 0.0
                ),
                "aggregate_raw_weight": raw_aggregate,
                "aggregate_training_weight": raw_aggregate * rescale,
                "status": "pass",
            }
        )

    summary_rows: list[dict[str, Any]] = []

    def append_summary(
        level: str,
        category: str,
        parent: str,
        indices: Iterable[int],
        target_fraction: float,
    ) -> None:
        indices = list(indices)
        selected = np.isin(row_members, indices)
        raw_total = float(np.sum(raw[selected]))
        final_total = float(np.sum(weights[selected]))
        summary_rows.append(
            {
                "level": level,
                "category": category,
                "parent": parent,
                "candidate_rows": int(np.count_nonzero(selected)),
                "members": len(indices),
                "candidate_bearing_members": sum(
                    member_by_index[index]["candidate_rows"] > 0
                    for index in indices
                ),
                "target_hierarchical_fraction": target_fraction,
                "raw_weight_sum": raw_total,
                "rescale_factor": rescale,
                "training_weight_sum": final_total,
                "mean_training_weight": (
                    float(np.mean(weights[selected])) if np.any(selected) else ""
                ),
                "physical_normalization": "none_development_balancing_only",
                "status": (
                    "pass"
                    if np.isclose(
                        raw_total, target_fraction, rtol=0.0, atol=tolerance
                    )
                    else "fail"
                ),
            }
        )

    all_indices = sorted(member_by_index)
    append_summary("global", "all_train_candidates", "", all_indices, 1.0)
    for sample_class in ("signal", "background"):
        indices = [
            index
            for index, member in member_by_index.items()
            if member["sample_class"] == sample_class
        ]
        append_summary("class", sample_class, "all", indices, 0.5)
    for mode in signal_modes:
        key = ("signal", mode)
        append_summary(
            "signal_mode",
            mode,
            "signal",
            members_by_stratum[key],
            stratum_fraction[key],
        )
    for family in background_families:
        key = ("background", family)
        append_summary(
            "background_family",
            family,
            "background",
            members_by_stratum[key],
            stratum_fraction[key],
        )

    if any(row["status"] != "pass" for row in summary_rows):
        raise ValueError("hierarchical aggregate balancing failed")

    return WeightResult(
        weights=weights,
        raw_weights=raw,
        rescale_factor=rescale,
        member_rows=member_rows,
        summary_rows=summary_rows,
    )
