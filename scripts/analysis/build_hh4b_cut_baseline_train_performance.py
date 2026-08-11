#!/usr/bin/env python3
"""Build publication-grade train-only HH4b cut performance and comparisons.

The primary generalization estimate is reconstructed from the ten frozen nested
outer-OOF winners.  The frozen deployment cut and historical R_HH < 34 cut are
evaluated as fixed-selection diagnostics on the same mutually exclusive
exact3tag/ge4tag outer-fold tables.  Their finite-MC comparison reuses the
frozen 2,000-replica paired source-group bootstrap draw plan exactly.

Validation and test payloads are never read by this program.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import subprocess
import sys
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd


BRANCH = "delphes-hh4b-production"
LUMINOSITY_FB_INVERSE = 138.0
LUMINOSITY_PB_INVERSE = 138000.0
CATEGORIES = ("exact3tag", "ge4tag")
OUTER_FOLDS = (0, 1, 2, 3, 4)
SAMPLE_CLASSES = ("signal", "background")

EXPECTED_WINNERS_SHA256 = "6c9d5cbccb12893b93242e03b131b6da5ce693f1f7920af68055670f6d38cfec"
EXPECTED_DEPLOYMENT_SHA256 = "1d6b7e007dbf8a4ffc68ba45b455d47737b0551ba2689bfa66b8e861a1cff1e6"
EXPECTED_HISTORICAL_FREEZE_SHA256 = "17a8565b3848646bf056fea9f17ca7efe1a0c0f691dd7b0ce7e167acbda823a0"
EXPECTED_HISTORICAL_EVALUATION_SHA256 = "53318fa304df2010502af0c991851ee2ca9179f955978c7bb8cd5a19872d0bd7"
EXPECTED_DRAW_MANIFEST_SHA256 = "5d8946dee33da6f3480dd889af8a3322b9a879b663cde5ac9348f034f93918db"
EXPECTED_DRAW_SHA256 = "3ac59a86fefc9e21275df73c945cd08e2b24642f02acc2b6459cf3c0f62c929b"
EXPECTED_FOLD_PAYLOAD_SUMS_SHA256 = "7ca7c572cdbf3e2b7ef0806ff69526ee62b407073cbcd11b89dc96c9123b805f"

NESTED_ROLE = "primary_train_only_generalization_estimate_pooled_nested_outer_oof"
FIXED_NOMINAL_ROLE = "secondary_fixed_frozen_deployment_selection_train_diagnostic"
HISTORICAL_ROLE = "fixed_historical_comparator_on_category_matched_train_universe"

COUNT_FIELDS = (
    "total_rows",
    "selected_rows",
    "negative_weight_rows",
    "selected_negative_weight_rows",
)
SUM_FIELDS = (
    "total_signed_yield",
    "selected_signed_yield",
    "total_sumw2",
    "selected_sumw2",
)
CLASS_FIELDS = COUNT_FIELDS + SUM_FIELDS
BOOTSTRAP_METRICS = (
    "signal_physical_efficiency",
    "background_physical_efficiency",
    "background_rejection",
    "signal_selected_signed_yield",
    "background_selected_signed_yield",
    "signal_selected_sumw2",
    "background_selected_sumw2",
    "signal_selected_effective_events",
    "background_selected_effective_events",
    "signal_finite_mc_relative_uncertainty",
    "background_finite_mc_relative_uncertainty",
    "signal_over_background",
    "asimov_significance_stat_only",
)
DISTRIBUTION_SPECS = {
    "r_hh_125_125": (0.0, 300.0, 60, r"$R_{HH}(125,125)$"),
    "mhh": (0.0, 3000.0, 60, r"$m_{HH}$ [GeV]"),
    "h2_pt": (0.0, 1500.0, 60, r"$p_T(H_2)$ [GeV]"),
    "ht_candidate_jets": (0.0, 3000.0, 60, r"$H_T^{\mathrm{cand.}}$ [GeV]"),
    "max_drbb": (0.0, 6.5, 52, r"$\max\Delta R_{bb}$"),
    "abs_h_delta_eta": (0.0, 12.0, 48, r"$|\Delta\eta(H_1,H_2)|$"),
}


class PerformanceError(RuntimeError):
    """Fail-closed train-performance contract violation."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise PerformanceError(message)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_json(value: Any) -> str:
    return json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n"


def write_json(path: Path, value: Any) -> None:
    path.write_text(canonical_json(value), encoding="utf-8")


def tsv_value(value: Any) -> str:
    if isinstance(value, bool):
        return "True" if value else "False"
    if isinstance(value, float):
        require(math.isfinite(value), "non-finite TSV value")
        return repr(value)
    if isinstance(value, (dict, list, tuple)):
        return json.dumps(value, separators=(",", ":"), sort_keys=True, allow_nan=False)
    return str(value)


def write_tsv(path: Path, fields: Sequence[str], rows: Iterable[Mapping[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=fields,
            delimiter="\t",
            lineterminator="\n",
            extrasaction="raise",
        )
        writer.writeheader()
        for row in rows:
            writer.writerow({field: tsv_value(row[field]) for field in fields})


def effective_events(signed_sum: float, sumw2: float) -> float:
    require(sumw2 >= 0.0 and math.isfinite(sumw2), "invalid sumw2")
    require(math.isfinite(signed_sum), "invalid signed sum")
    return 0.0 if sumw2 == 0.0 else signed_sum * signed_sum / sumw2


def relative_mc_uncertainty(signed_sum: float, sumw2: float) -> float:
    require(sumw2 >= 0.0 and math.isfinite(sumw2), "invalid sumw2")
    require(signed_sum != 0.0 and math.isfinite(signed_sum), "zero/invalid signed sum")
    return math.sqrt(sumw2) / abs(signed_sum)


def asimov_significance(signal_yield: float, background_yield: float) -> float:
    """Evaluate the exact stat-only formula used by the frozen comparator."""

    require(signal_yield >= 0.0 and background_yield > 0.0, "invalid yields for Asimov Z")
    radicand = 2.0 * (
        (signal_yield + background_yield) * math.log1p(signal_yield / background_yield)
        - signal_yield
    )
    require(radicand >= 0.0 and math.isfinite(radicand), "invalid Asimov radicand")
    return math.sqrt(radicand)


def zero_class_summary() -> dict[str, int | float]:
    return {field: 0 if field in COUNT_FIELDS else 0.0 for field in CLASS_FIELDS}


def add_class_summary(
    target: dict[str, int | float], source: Mapping[str, Any]
) -> None:
    for field in COUNT_FIELDS:
        target[field] = int(target[field]) + int(source[field])
    for field in SUM_FIELDS:
        target[field] = float(target[field]) + float(source[field])


def combine_summaries(
    summaries: Iterable[Mapping[str, Mapping[str, Any]]]
) -> dict[str, dict[str, int | float]]:
    result = {sample_class: zero_class_summary() for sample_class in SAMPLE_CLASSES}
    for summary in summaries:
        for sample_class in SAMPLE_CLASSES:
            add_class_summary(result[sample_class], summary[sample_class])
    return result


def summarize_frame(frame: pd.DataFrame, selected: np.ndarray) -> dict[str, dict[str, int | float]]:
    require(len(frame) == len(selected), "selection-mask length mismatch")
    require(selected.dtype == np.bool_, "selection mask is not Boolean")
    classes = frame["sample_class"].astype(str).to_numpy()
    weights = frame["resolved_selection_contribution_weight"].to_numpy(dtype=float)
    require(np.isfinite(weights).all(), "non-finite physical weight")
    result: dict[str, dict[str, int | float]] = {}
    for sample_class in SAMPLE_CLASSES:
        class_mask = classes == sample_class
        chosen = class_mask & selected
        total_weights = weights[class_mask]
        selected_weights = weights[chosen]
        result[sample_class] = {
            "total_rows": int(np.count_nonzero(class_mask)),
            "selected_rows": int(np.count_nonzero(chosen)),
            "negative_weight_rows": int(np.count_nonzero(total_weights < 0.0)),
            "selected_negative_weight_rows": int(np.count_nonzero(selected_weights < 0.0)),
            "total_signed_yield": float(total_weights.sum()),
            "selected_signed_yield": float(selected_weights.sum()),
            "total_sumw2": float(np.square(total_weights).sum()),
            "selected_sumw2": float(np.square(selected_weights).sum()),
        }
    return result


def performance_row(
    selection_id: str,
    evaluation_role: str,
    scope: str,
    outer_fold: int | str,
    summary: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    row: dict[str, Any] = {
        "selection_id": selection_id,
        "evaluation_role": evaluation_role,
        "scope": scope,
        "outer_fold": outer_fold,
        "sqrt_s_tev": 13.0,
        "luminosity_fb_inverse": LUMINOSITY_FB_INVERSE,
        "luminosity_pb_inverse_internal": LUMINOSITY_PB_INVERSE,
        "yield_interpretation": "Run-2 expected-yield projection",
    }
    for sample_class in SAMPLE_CLASSES:
        values = summary[sample_class]
        for field in CLASS_FIELDS:
            row[f"{sample_class}_{field}"] = values[field]
        total_yield = float(values["total_signed_yield"])
        selected_yield = float(values["selected_signed_yield"])
        total_sumw2 = float(values["total_sumw2"])
        selected_sumw2 = float(values["selected_sumw2"])
        require(total_yield > 0.0 and selected_yield > 0.0, "non-positive class yield")
        row[f"{sample_class}_physical_efficiency"] = selected_yield / total_yield
        row[f"{sample_class}_raw_efficiency"] = (
            int(values["selected_rows"]) / int(values["total_rows"])
        )
        row[f"{sample_class}_total_effective_events"] = effective_events(
            total_yield, total_sumw2
        )
        row[f"{sample_class}_selected_effective_events"] = effective_events(
            selected_yield, selected_sumw2
        )
        row[f"{sample_class}_finite_mc_relative_uncertainty"] = relative_mc_uncertainty(
            selected_yield, selected_sumw2
        )
    signal = float(row["signal_selected_signed_yield"])
    background = float(row["background_selected_signed_yield"])
    background_efficiency = float(row["background_physical_efficiency"])
    require(background_efficiency > 0.0, "non-positive background efficiency")
    row["background_rejection"] = 1.0 / background_efficiency
    row["signal_over_background"] = signal / background
    row["asimov_significance_stat_only"] = asimov_significance(signal, background)
    row["systematics_included"] = False
    row["official_cms_result"] = False
    row["validation_payloads_opened"] = 0
    row["test_payloads_opened"] = 0
    return row


PERFORMANCE_FIELDS = (
    "selection_id",
    "evaluation_role",
    "scope",
    "outer_fold",
    "sqrt_s_tev",
    "luminosity_fb_inverse",
    "luminosity_pb_inverse_internal",
    "yield_interpretation",
) + tuple(
    f"{sample_class}_{field}"
    for sample_class in SAMPLE_CLASSES
    for field in (
        *CLASS_FIELDS,
        "physical_efficiency",
        "raw_efficiency",
        "total_effective_events",
        "selected_effective_events",
        "finite_mc_relative_uncertainty",
    )
) + (
    "background_rejection",
    "signal_over_background",
    "asimov_significance_stat_only",
    "systematics_included",
    "official_cms_result",
    "validation_payloads_opened",
    "test_payloads_opened",
)


def verify_repository(repo: Path) -> str:
    require(repo.resolve() == repo, "repository path is not canonical")
    branch = subprocess.check_output(["git", "branch", "--show-current"], cwd=repo, text=True).strip()
    require(branch == BRANCH, f"unexpected branch: {branch}")
    head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo, text=True).strip()
    remote = subprocess.check_output(
        ["git", "rev-parse", f"origin/{BRANCH}"], cwd=repo, text=True
    ).strip()
    require(head == remote, f"local/remote HEAD mismatch: {head} != {remote}")
    return head


def verify_hash(path: Path, expected: str) -> None:
    require(path.is_file() and not path.is_symlink(), f"missing/non-regular input: {path}")
    actual = sha256_file(path)
    require(actual == expected, f"SHA-256 mismatch for {path}: {actual}")


def verify_checkpoint(checkpoint: Path) -> None:
    require(checkpoint.is_dir() and not checkpoint.is_symlink(), f"invalid checkpoint: {checkpoint}")
    sums = checkpoint / "SHA256SUMS"
    require(sums.is_file() and not sums.is_symlink(), f"missing checkpoint SHA256SUMS: {checkpoint}")
    check = subprocess.run(
        ["sha256sum", "-c", "SHA256SUMS"],
        cwd=checkpoint,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    require(check.returncode == 0, f"checkpoint checksum failure: {check.stdout}{check.stderr}")


def verify_all1000_gate(checkpoint: Path) -> Path:
    verify_checkpoint(checkpoint)
    summary_path = checkpoint / "evidence/aggregation/all1000_stability_summary.json"
    require(summary_path.is_file(), "all1000 summary missing from checkpoint")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    require(summary.get("status") == "pass_all1000_predeclared_selection_stability_aggregation",
            "all1000 aggregation status mismatch")
    require(summary.get("fold_level_winner_count") == 10000, "all1000 fold-winner count mismatch")
    require(summary.get("replica_category_result_count") == 2000,
            "all1000 replica/category count mismatch")
    require(summary.get("ranked_structure_result_count") == 270000,
            "all1000 ranked-result count mismatch")
    require(summary.get("nominal_deployment_candidate_changed") is False,
            "all1000 summary says nominal candidate changed")
    require(summary.get("validation_payloads_opened") == 0 and summary.get("test_payloads_opened") == 0,
            "all1000 checkpoint sealed-data count mismatch")
    return summary_path


def load_nested_winners(path: Path) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    require(payload.get("status") == "pass_outer_fold_category_family_selection",
            "nested winner status mismatch")
    require(payload.get("winner_count") == 10, "nested winner count mismatch")
    require(payload.get("outer_fold_results_used_for_selection") is False,
            "outer results were marked as selection inputs")
    require(payload.get("validation_payloads_opened") == 0 and payload.get("test_payloads_opened") == 0,
            "nested winner sealed-data count mismatch")
    winners = payload["winners"]
    identities = {(int(row["outer_fold"]), str(row["category_id"])) for row in winners}
    require(
        identities == {(fold, category) for fold in OUTER_FOLDS for category in CATEGORIES},
        "nested winner fold/category coverage mismatch",
    )
    require(all(row["outer_fold_used_for_selection"] is False for row in winners),
            "outer fold used for nested selection")
    return sorted(winners, key=lambda row: (int(row["outer_fold"]), str(row["category_id"])))


def load_deployment(path: Path) -> dict[str, dict[str, float]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    require(payload.get("status") == "train_only_deployment_candidate_frozen",
            "deployment candidate status mismatch")
    require(payload.get("nominal_selection_is_changed_by_stability_escalation") is False,
            "deployment candidate was marked changed by stability")
    require(payload.get("validation_payloads_opened") == 0 and payload.get("test_payloads_opened") == 0,
            "deployment candidate sealed-data count mismatch")
    result = {
        category: {
            str(key): float(value)
            for key, value in payload["categories"][category][
                "deployment_thresholds_coordinatewise_median"
            ].items()
        }
        for category in CATEGORIES
    }
    expected = {
        "exact3tag": {
            "ht_candidate_jets": 176.5458068847656,
            "r_hh_125_125": 36.40814019639858,
        },
        "ge4tag": {
            "abs_h_delta_eta": 6.904302164473993,
            "mhh": 164.73708096689654,
            "r_hh_125_125": 33.92808917804956,
        },
    }
    require(result == expected, "frozen nominal deployment thresholds mismatch")
    return result


def apply_cuts(frame: pd.DataFrame, cuts: Sequence[Mapping[str, str]],
               thresholds: Mapping[str, float]) -> np.ndarray:
    mask = np.ones(len(frame), dtype=bool)
    require({str(cut["variable"]) for cut in cuts} == set(thresholds),
            "cut/threshold coordinate mismatch")
    for cut in cuts:
        variable = str(cut["variable"])
        values = frame[variable].to_numpy(dtype=float)
        require(np.isfinite(values).all(), f"non-finite values in {variable}")
        if cut["operator"] == "<":
            mask &= values < float(thresholds[variable])
        elif cut["operator"] == ">":
            mask &= values > float(thresholds[variable])
        else:
            raise PerformanceError(f"unsupported cut operator: {cut['operator']}")
    return mask


def fixed_nominal_mask(
    frame: pd.DataFrame, category: str, deployment: Mapping[str, Mapping[str, float]]
) -> np.ndarray:
    thresholds = deployment[category]
    cuts = [{"variable": "r_hh_125_125", "operator": "<"}]
    if category == "exact3tag":
        cuts.append({"variable": "ht_candidate_jets", "operator": ">"})
    else:
        cuts.extend([
            {"variable": "mhh", "operator": ">"},
            {"variable": "abs_h_delta_eta", "operator": "<"},
        ])
    return apply_cuts(frame, cuts, thresholds)


def summaries_close(
    left: Mapping[str, Mapping[str, Any]], right: Mapping[str, Mapping[str, Any]]
) -> bool:
    for sample_class in SAMPLE_CLASSES:
        for field in COUNT_FIELDS:
            if int(left[sample_class][field]) != int(right[sample_class][field]):
                return False
        for field in SUM_FIELDS:
            if not math.isclose(
                float(left[sample_class][field]),
                float(right[sample_class][field]),
                rel_tol=2.0e-13,
                abs_tol=1.0e-10,
            ):
                return False
    return True


def source_summaries(
    frame: pd.DataFrame,
    selected: np.ndarray,
    selection_id: str,
    category: str,
) -> list[dict[str, Any]]:
    work = frame[["source_uid", "sample_class", "source_fold",
                  "resolved_selection_contribution_weight"]].copy()
    work["selected"] = selected
    rows: list[dict[str, Any]] = []
    for (source_uid, sample_class, source_fold), group in work.groupby(
        ["source_uid", "sample_class", "source_fold"], sort=True, observed=True
    ):
        weights = group["resolved_selection_contribution_weight"].to_numpy(dtype=float)
        chosen = group["selected"].to_numpy(dtype=bool)
        selected_weights = weights[chosen]
        rows.append({
            "selection_id": selection_id,
            "scope": category,
            "source_uid": str(source_uid),
            "sample_class": str(sample_class),
            "source_fold": int(source_fold),
            "total_rows": len(group),
            "selected_rows": int(np.count_nonzero(chosen)),
            "negative_weight_rows": int(np.count_nonzero(weights < 0.0)),
            "selected_negative_weight_rows": int(np.count_nonzero(selected_weights < 0.0)),
            "total_signed_yield": float(weights.sum()),
            "selected_signed_yield": float(selected_weights.sum()),
            "total_sumw2": float(np.square(weights).sum()),
            "selected_sumw2": float(np.square(selected_weights).sum()),
        })
    return rows


def add_combined_source_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    frame = pd.DataFrame(rows)
    combined: list[dict[str, Any]] = []
    for (selection_id, source_uid, sample_class, source_fold), group in frame.groupby(
        ["selection_id", "source_uid", "sample_class", "source_fold"],
        sort=True,
        observed=True,
    ):
        combined.append({
            "selection_id": selection_id,
            "scope": "combined",
            "source_uid": source_uid,
            "sample_class": sample_class,
            "source_fold": int(source_fold),
            **{
                field: int(group[field].sum()) if field.endswith("rows") else float(group[field].sum())
                for field in (
                    "total_rows", "selected_rows", "negative_weight_rows",
                    "selected_negative_weight_rows", "total_signed_yield",
                    "selected_signed_yield", "total_sumw2", "selected_sumw2",
                )
            },
        })
    return rows + combined


def accumulate_distributions(
    accumulator: dict[tuple[str, str, str, str], dict[str, Any]],
    frame: pd.DataFrame,
    selected: np.ndarray,
    category: str,
) -> None:
    classes = frame["sample_class"].astype(str).to_numpy()
    weights = frame["resolved_selection_contribution_weight"].to_numpy(dtype=float)
    for variable, (minimum, maximum, bins, label) in DISTRIBUTION_SPECS.items():
        values = frame[variable].to_numpy(dtype=float)
        require(np.isfinite(values).all(), f"non-finite distribution values: {variable}")
        edges = np.linspace(minimum, maximum, bins + 1)
        for sample_class in SAMPLE_CLASSES:
            class_mask = classes == sample_class
            for stage, stage_mask in (("preselection", class_mask),
                                      ("postselection", class_mask & selected)):
                chosen_values = values[stage_mask]
                chosen_weights = weights[stage_mask]
                key = (category, sample_class, stage, variable)
                if key not in accumulator:
                    accumulator[key] = {
                        "edges": edges,
                        "label": label,
                        "rows": np.zeros(bins, dtype=np.int64),
                        "signed_yield": np.zeros(bins, dtype=float),
                        "sumw2": np.zeros(bins, dtype=float),
                        "underflow_rows": 0,
                        "overflow_rows": 0,
                        "underflow_signed_yield": 0.0,
                        "overflow_signed_yield": 0.0,
                    }
                target = accumulator[key]
                target["rows"] += np.histogram(chosen_values, bins=edges)[0]
                target["signed_yield"] += np.histogram(
                    chosen_values, bins=edges, weights=chosen_weights
                )[0]
                target["sumw2"] += np.histogram(
                    chosen_values, bins=edges, weights=np.square(chosen_weights)
                )[0]
                underflow = chosen_values < minimum
                overflow = chosen_values >= maximum
                target["underflow_rows"] += int(np.count_nonzero(underflow))
                target["overflow_rows"] += int(np.count_nonzero(overflow))
                target["underflow_signed_yield"] += float(chosen_weights[underflow].sum())
                target["overflow_signed_yield"] += float(chosen_weights[overflow].sum())


def distribution_rows(
    accumulator: Mapping[tuple[str, str, str, str], Mapping[str, Any]]
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for (category, sample_class, stage, variable), values in sorted(accumulator.items()):
        edges = values["edges"]
        for index in range(len(edges) - 1):
            rows.append({
                "category_id": category,
                "sample_class": sample_class,
                "selection_stage": stage,
                "selection_id": "fixed_nominal_deployment_cut",
                "variable": variable,
                "axis_label_latex": values["label"],
                "bin_index": index,
                "bin_low_inclusive": float(edges[index]),
                "bin_high_exclusive": float(edges[index + 1]),
                "rows": int(values["rows"][index]),
                "signed_yield": float(values["signed_yield"][index]),
                "sumw2": float(values["sumw2"][index]),
                "underflow_rows_distribution_total": int(values["underflow_rows"]),
                "overflow_rows_distribution_total": int(values["overflow_rows"]),
                "underflow_signed_yield_distribution_total": float(
                    values["underflow_signed_yield"]
                ),
                "overflow_signed_yield_distribution_total": float(
                    values["overflow_signed_yield"]
                ),
                "validation_payloads_opened": 0,
                "test_payloads_opened": 0,
            })
    expected = sum(spec[2] for spec in DISTRIBUTION_SPECS.values()) * 2 * 2 * 2
    require(len(rows) == expected, "pre/post distribution row count mismatch")
    return rows


def bootstrap_performance_rows(
    draws: pd.DataFrame, source_rows: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    source = pd.DataFrame(source_rows)
    output: list[dict[str, Any]] = []
    require(draws["replica"].nunique() == 2000, "bootstrap replica count mismatch")
    for scope in (*CATEGORIES, "combined"):
        for selection_id, role in (
            ("historical_rhh125125_lt34", HISTORICAL_ROLE),
            ("fixed_nominal_deployment_cut", FIXED_NOMINAL_ROLE),
        ):
            selected_source = source[
                (source["scope"] == scope) & (source["selection_id"] == selection_id)
            ]
            require(not selected_source.empty, f"empty bootstrap source metrics: {scope}/{selection_id}")
            merged = draws[["replica", "source_uid", "sample_class", "multiplicity"]].merge(
                selected_source.drop(columns=["scope", "selection_id", "source_fold"]),
                on=["source_uid", "sample_class"],
                how="left",
                validate="many_to_one",
            )
            for field in (
                "total_rows", "selected_rows", "negative_weight_rows",
                "selected_negative_weight_rows", "total_signed_yield",
                "selected_signed_yield", "total_sumw2", "selected_sumw2",
            ):
                merged[field] = merged[field].fillna(0.0) * merged["multiplicity"]
            grouped = merged.groupby(["replica", "sample_class"], sort=True, observed=True)[
                ["total_rows", "selected_rows", "negative_weight_rows",
                 "selected_negative_weight_rows", "total_signed_yield",
                 "selected_signed_yield", "total_sumw2", "selected_sumw2"]
            ].sum()
            for replica in range(2000):
                summary: dict[str, dict[str, Any]] = {}
                for sample_class in SAMPLE_CLASSES:
                    values = grouped.loc[(replica, sample_class)]
                    summary[sample_class] = {
                        "total_rows": int(values["total_rows"]),
                        "selected_rows": int(values["selected_rows"]),
                        "negative_weight_rows": int(values["negative_weight_rows"]),
                        "selected_negative_weight_rows": int(
                            values["selected_negative_weight_rows"]
                        ),
                        "total_signed_yield": float(values["total_signed_yield"]),
                        "selected_signed_yield": float(values["selected_signed_yield"]),
                        "total_sumw2": float(values["total_sumw2"]),
                        "selected_sumw2": float(values["selected_sumw2"]),
                    }
                row = performance_row(selection_id, role, scope, "bootstrap", summary)
                row["replica"] = replica
                output.append(row)
    require(len(output) == 12000, "bootstrap performance row count mismatch")
    return output


def quantile(values: np.ndarray, probability: float) -> float:
    return float(np.quantile(values, probability, method="linear"))


def paired_difference_rows(
    bootstrap_rows: list[dict[str, Any]], nominal_rows: Sequence[Mapping[str, Any]]
) -> list[dict[str, Any]]:
    nominal_lookup = {
        (str(row["selection_id"]), str(row["scope"])): row
        for row in nominal_rows
        if row["outer_fold"] == "pooled"
        and row["selection_id"] in {
            "historical_rhh125125_lt34", "fixed_nominal_deployment_cut"
        }
    }
    frame = pd.DataFrame(bootstrap_rows)
    output: list[dict[str, Any]] = []
    for scope in (*CATEGORIES, "combined"):
        historical = frame[
            (frame["scope"] == scope)
            & (frame["selection_id"] == "historical_rhh125125_lt34")
        ].sort_values("replica")
        nominal = frame[
            (frame["scope"] == scope)
            & (frame["selection_id"] == "fixed_nominal_deployment_cut")
        ].sort_values("replica")
        require(
            historical["replica"].tolist() == nominal["replica"].tolist() == list(range(2000)),
            f"paired bootstrap replica mismatch: {scope}",
        )
        for metric in BOOTSTRAP_METRICS:
            difference = nominal[metric].to_numpy(dtype=float) - historical[metric].to_numpy(dtype=float)
            ratio = nominal[metric].to_numpy(dtype=float) / historical[metric].to_numpy(dtype=float)
            require(np.isfinite(difference).all() and np.isfinite(ratio).all(),
                    f"non-finite paired comparison: {scope}/{metric}")
            nominal_difference = float(nominal_lookup[("fixed_nominal_deployment_cut", scope)][metric]) - float(
                nominal_lookup[("historical_rhh125125_lt34", scope)][metric]
            )
            output.append({
                "scope": scope,
                "metric": metric,
                "difference_definition": "fixed_nominal_deployment_cut_minus_historical_rhh125125_lt34",
                "nominal_difference": nominal_difference,
                "difference_mean": float(np.mean(difference)),
                "difference_std": float(np.std(difference, ddof=1)),
                "difference_p2p5": quantile(difference, 0.025),
                "difference_p16": quantile(difference, 0.16),
                "difference_median": quantile(difference, 0.50),
                "difference_p84": quantile(difference, 0.84),
                "difference_p97p5": quantile(difference, 0.975),
                "ratio_mean": float(np.mean(ratio)),
                "ratio_std": float(np.std(ratio, ddof=1)),
                "ratio_p2p5": quantile(ratio, 0.025),
                "ratio_p16": quantile(ratio, 0.16),
                "ratio_median": quantile(ratio, 0.50),
                "ratio_p84": quantile(ratio, 0.84),
                "ratio_p97p5": quantile(ratio, 0.975),
                "bootstrap_replicas": 2000,
                "bootstrap_seed": 20260727,
                "bootstrap_unit": "source_group_within_sample_class_and_process_or_mode_strata",
                "selection_reoptimization_inside_bootstrap": False,
                "validation_payloads_opened": 0,
                "test_payloads_opened": 0,
            })
    require(len(output) == len(BOOTSTRAP_METRICS) * 3, "paired-summary row count mismatch")
    return output


def build(
    repo: Path,
    output: Path,
    all1000_checkpoint: Path,
    fold_payload: Path,
    draw_path: Path,
) -> dict[str, Any]:
    repository_head = verify_repository(repo)
    all1000_summary_path = verify_all1000_gate(all1000_checkpoint)

    winners_path = repo / (
        "docs/checkpoints/hh4b_train_multivariate_cut_train_only_nested_oof_aggregation_20260807_v1/"
        "evidence/outer_fold_category_winners.json"
    )
    deployment_path = repo / (
        "docs/checkpoints/hh4b_train_multivariate_cut_train_only_deployment_candidate_20260807_v1/"
        "train_only_deployment_candidate.json"
    )
    historical_freeze = repo / (
        "docs/checkpoints/hh4b_train_historical_rhh125125_lt34_freeze_20260806_v1/"
        "historical_rhh34_freeze_contract.json"
    )
    historical_evaluation = repo / (
        "docs/checkpoints/hh4b_train_historical_rhh125125_lt34_freeze_20260806_v1/"
        "evidence/historical_rhh34_evaluation_contract.json"
    )
    draw_manifest = repo / (
        "docs/checkpoints/hh4b_train_historical_rhh125125_lt34_freeze_20260806_v1/"
        "external_artifact_manifest.json"
    )
    for path, expected in (
        (winners_path, EXPECTED_WINNERS_SHA256),
        (deployment_path, EXPECTED_DEPLOYMENT_SHA256),
        (historical_freeze, EXPECTED_HISTORICAL_FREEZE_SHA256),
        (historical_evaluation, EXPECTED_HISTORICAL_EVALUATION_SHA256),
        (draw_manifest, EXPECTED_DRAW_MANIFEST_SHA256),
        (fold_payload / "SHA256SUMS", EXPECTED_FOLD_PAYLOAD_SUMS_SHA256),
        (draw_path, EXPECTED_DRAW_SHA256),
    ):
        verify_hash(path, expected)
    fold_check = subprocess.run(
        ["sha256sum", "-c", "SHA256SUMS"],
        cwd=fold_payload,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    require(fold_check.returncode == 0, f"fold payload checksum failure: {fold_check.stdout}{fold_check.stderr}")

    historical_contract = json.loads(historical_evaluation.read_text(encoding="utf-8"))
    require(historical_contract.get("threshold_scan_performed") is False,
            "historical contract indicates a threshold scan")
    require(historical_contract.get("thresholds_evaluated") == [34.0],
            "historical threshold contract mismatch")
    require(historical_contract.get("validation_payloads_opened") == 0
            and historical_contract.get("test_payloads_opened") == 0,
            "historical comparator sealed-data count mismatch")
    draw_contract = json.loads(draw_manifest.read_text(encoding="utf-8"))
    require(Path(draw_contract["path"]).resolve() == draw_path,
            "bootstrap draw path differs from frozen external manifest")
    require(draw_contract["sha256"] == EXPECTED_DRAW_SHA256, "draw manifest SHA mismatch")

    winners = load_nested_winners(winners_path)
    deployment = load_deployment(deployment_path)
    winner_lookup = {(int(row["outer_fold"]), str(row["category_id"])): row for row in winners}

    performance_rows: list[dict[str, Any]] = []
    nested_by_identity: dict[tuple[int, str], dict[str, dict[str, Any]]] = {}
    fixed_by_selection_identity: dict[tuple[str, int, str], dict[str, dict[str, Any]]] = {}
    source_rows: list[dict[str, Any]] = []
    seen_event_uids: set[str] = set()
    fold_input_rows: list[dict[str, Any]] = []
    reproduction_rows: list[dict[str, Any]] = []
    distributions: dict[tuple[str, str, str, str], dict[str, Any]] = {}

    needed_columns = [
        "source_uid", "group_id", "source_fold", "event_uid", "sample_class",
        "process_or_mode", "auxiliary_qcd", "physical_evaluation_eligible",
        "candidate_tagged_jet_count", "r_hh_125_125", "ht_candidate_jets",
        "mhh", "abs_h_delta_eta", "h2_pt", "max_drbb", "max_abs_mbb_minus_125",
        "abs_mbb1_minus_125", "abs_mbb2_minus_125",
        "resolved_selection_contribution_weight",
    ]
    for fold in OUTER_FOLDS:
        for category in CATEGORIES:
            path = fold_payload / f"fold_tables/fold_{fold}/{category}.parquet"
            frame = pd.read_parquet(path, columns=needed_columns)
            require(len(frame) > 0, f"empty fold table: {path}")
            require(frame["source_fold"].eq(fold).all(), f"source-fold mismatch: {path}")
            require(frame["auxiliary_qcd"].eq(False).all(), f"auxiliary QCD leakage: {path}")
            require(frame["physical_evaluation_eligible"].eq(True).all(),
                    f"ineligible physical rows: {path}")
            require(set(frame["sample_class"].astype(str)) == set(SAMPLE_CLASSES),
                    f"sample-class coverage mismatch: {path}")
            if category == "exact3tag":
                require(frame["candidate_tagged_jet_count"].eq(3).all(),
                        f"exact3tag mapping mismatch: {path}")
            else:
                require(frame["candidate_tagged_jet_count"].ge(4).all(),
                        f"ge4tag mapping mismatch: {path}")
            event_uids = set(frame["event_uid"].astype(str))
            require(len(event_uids) == len(frame), f"duplicate events within {path}")
            require(seen_event_uids.isdisjoint(event_uids), f"event overlap across fold/category tables: {path}")
            seen_event_uids.update(event_uids)
            fold_input_rows.append({
                "outer_fold": fold,
                "category_id": category,
                "path": str(path),
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
                "rows": len(frame),
                "unique_source_uids": frame["source_uid"].nunique(),
                "validation_payloads_opened": 0,
                "test_payloads_opened": 0,
            })

            winner = winner_lookup[(fold, category)]
            winner_cuts = json.loads(winner["cuts_json"])
            winner_thresholds = {
                str(key): float(value)
                for key, value in json.loads(winner["refit_thresholds_json"]).items()
            }
            nested_mask = apply_cuts(frame, winner_cuts, winner_thresholds)
            reproduced_nested = summarize_frame(frame, nested_mask)
            frozen_nested = winner["outer_physical_evaluation_json"]
            if isinstance(frozen_nested, str):
                frozen_nested = json.loads(frozen_nested)
            require(summaries_close(reproduced_nested, frozen_nested),
                    f"nested outer-OOF physical reproduction mismatch: fold {fold}/{category}")
            nested_by_identity[(fold, category)] = frozen_nested
            reproduction_rows.append({
                "outer_fold": fold,
                "category_id": category,
                "structure_id": winner["structure_id"],
                "thresholds_json": winner_thresholds,
                "frozen_outer_physical_sha256": hashlib.sha256(
                    json.dumps(frozen_nested, sort_keys=True, separators=(",", ":")).encode()
                ).hexdigest(),
                "row_counts_exact": True,
                "signed_yields_close_rtol_2em13": True,
                "sumw2_close_rtol_2em13": True,
                "outer_fold_used_for_selection": False,
                "validation_payloads_opened": 0,
                "test_payloads_opened": 0,
            })

            historical_mask = frame["r_hh_125_125"].to_numpy(dtype=float) < 34.0
            nominal_mask = fixed_nominal_mask(frame, category, deployment)
            accumulate_distributions(distributions, frame, nominal_mask, category)
            for selection_id, mask in (
                ("historical_rhh125125_lt34", historical_mask),
                ("fixed_nominal_deployment_cut", nominal_mask),
            ):
                summary = summarize_frame(frame, mask)
                fixed_by_selection_identity[(selection_id, fold, category)] = summary
                source_rows.extend(source_summaries(frame, mask, selection_id, category))

    require(len(seen_event_uids) == sum(int(row["rows"]) for row in fold_input_rows),
            "global event uniqueness closure failed")
    require(sum(int(row["rows"]) for row in fold_input_rows) == 139383,
            "frozen category/fold row total mismatch")

    for fold in OUTER_FOLDS:
        for category in CATEGORIES:
            performance_rows.append(performance_row(
                "nested_outer_oof", NESTED_ROLE, category, fold,
                nested_by_identity[(fold, category)],
            ))
        performance_rows.append(performance_row(
            "nested_outer_oof", NESTED_ROLE, "combined", fold,
            combine_summaries(nested_by_identity[(fold, category)] for category in CATEGORIES),
        ))
    for category in CATEGORIES:
        performance_rows.append(performance_row(
            "nested_outer_oof", NESTED_ROLE, category, "pooled",
            combine_summaries(nested_by_identity[(fold, category)] for fold in OUTER_FOLDS),
        ))
    performance_rows.append(performance_row(
        "nested_outer_oof", NESTED_ROLE, "combined", "pooled",
        combine_summaries(nested_by_identity.values()),
    ))

    for selection_id, role in (
        ("historical_rhh125125_lt34", HISTORICAL_ROLE),
        ("fixed_nominal_deployment_cut", FIXED_NOMINAL_ROLE),
    ):
        for fold in OUTER_FOLDS:
            for category in CATEGORIES:
                performance_rows.append(performance_row(
                    selection_id, role, category, fold,
                    fixed_by_selection_identity[(selection_id, fold, category)],
                ))
            performance_rows.append(performance_row(
                selection_id, role, "combined", fold,
                combine_summaries(
                    fixed_by_selection_identity[(selection_id, fold, category)]
                    for category in CATEGORIES
                ),
            ))
        for category in CATEGORIES:
            performance_rows.append(performance_row(
                selection_id, role, category, "pooled",
                combine_summaries(
                    fixed_by_selection_identity[(selection_id, fold, category)]
                    for fold in OUTER_FOLDS
                ),
            ))
        performance_rows.append(performance_row(
            selection_id, role, "combined", "pooled",
            combine_summaries(
                fixed_by_selection_identity[(selection_id, fold, category)]
                for fold in OUTER_FOLDS for category in CATEGORIES
            ),
        ))
    require(len(performance_rows) == 54, "train performance row count mismatch")

    source_rows = add_combined_source_rows(source_rows)
    require(len({row["source_uid"] for row in source_rows}) == 351,
            "category-matched source count mismatch")
    draws = pd.read_parquet(draw_path)
    require(
        list(draws.columns)
        == ["replica", "bootstrap_source_index", "source_uid", "sample_class",
            "process_or_mode", "multiplicity"],
        "bootstrap draw schema mismatch",
    )
    require(len(draws) == 567306 and draws["replica"].nunique() == 2000,
            "bootstrap draw dimensions mismatch")
    require(draws["multiplicity"].ge(1).all(), "non-positive bootstrap multiplicity")
    multiplicities = draws.groupby("replica", sort=True)["multiplicity"].sum()
    require(multiplicities.eq(441).all(), "bootstrap multiplicity closure mismatch")
    bootstrap_rows = bootstrap_performance_rows(draws, source_rows)
    paired_rows = paired_difference_rows(bootstrap_rows, performance_rows)
    fixed_distribution_rows = distribution_rows(distributions)

    output.mkdir(parents=True)
    tables = output / "tables"
    figure_data = output / "figure_data"
    manifests = output / "manifests"
    tables.mkdir()
    figure_data.mkdir()
    manifests.mkdir()

    write_tsv(tables / "train_performance_metrics.tsv", PERFORMANCE_FIELDS, performance_rows)
    pooled_rows = [row for row in performance_rows if row["outer_fold"] == "pooled"]
    fold_rows = [row for row in performance_rows if row["outer_fold"] != "pooled"]
    write_tsv(figure_data / "pooled_train_performance_metrics.tsv", PERFORMANCE_FIELDS, pooled_rows)
    write_tsv(figure_data / "fold_train_performance_metrics.tsv", PERFORMANCE_FIELDS, fold_rows)

    fold_input_fields = (
        "outer_fold", "category_id", "path", "bytes", "sha256", "rows",
        "unique_source_uids", "validation_payloads_opened", "test_payloads_opened",
    )
    write_tsv(manifests / "fold_table_inventory.tsv", fold_input_fields, fold_input_rows)
    reproduction_fields = (
        "outer_fold", "category_id", "structure_id", "thresholds_json",
        "frozen_outer_physical_sha256", "row_counts_exact",
        "signed_yields_close_rtol_2em13", "sumw2_close_rtol_2em13",
        "outer_fold_used_for_selection", "validation_payloads_opened", "test_payloads_opened",
    )
    write_tsv(tables / "nested_outer_oof_reproduction_audit.tsv", reproduction_fields,
              reproduction_rows)

    source_fields = (
        "selection_id", "scope", "source_uid", "sample_class", "source_fold",
        "total_rows", "selected_rows", "negative_weight_rows",
        "selected_negative_weight_rows", "total_signed_yield", "selected_signed_yield",
        "total_sumw2", "selected_sumw2",
    )
    write_tsv(tables / "fixed_selection_source_metrics.tsv", source_fields, source_rows)

    bootstrap_fields = ("replica",) + PERFORMANCE_FIELDS
    write_tsv(tables / "paired_bootstrap_replica_metrics.tsv", bootstrap_fields, bootstrap_rows)
    paired_fields = tuple(paired_rows[0])
    write_tsv(tables / "paired_bootstrap_difference_summary.tsv", paired_fields, paired_rows)
    write_tsv(figure_data / "paired_bootstrap_difference_summary.tsv", paired_fields, paired_rows)
    distribution_fields = tuple(fixed_distribution_rows[0])
    write_tsv(
        figure_data / "pre_post_nominal_variable_distributions.tsv",
        distribution_fields,
        fixed_distribution_rows,
    )

    pooled_lookup = {
        (row["selection_id"], row["scope"]): row for row in pooled_rows
    }
    summary = {
        "schema_version": 1,
        "status": "pass_hh4b_cut_baseline_train_performance_and_fixed_comparator",
        "repository_head_at_execution": repository_head,
        "sqrt_s_tev": 13.0,
        "luminosity_fb_inverse": LUMINOSITY_FB_INVERSE,
        "luminosity_pb_inverse_internal": LUMINOSITY_PB_INVERSE,
        "yield_interpretation": "Run-2 expected-yield projection",
        "official_cms_result": False,
        "primary_generalization_estimate": "pooled_nested_outer_oof",
        "fixed_deployment_cut_called_primary_generalization_estimate": False,
        "nested_outer_oof_fold_category_reproduction": "pass_10_of_10",
        "mutually_exclusive_category_combination_verified": True,
        "category_matched_event_rows": len(seen_event_uids),
        "category_matched_source_groups": 351,
        "historical_comparator_threshold_scan_performed": False,
        "historical_comparator_threshold": {"r_hh_125_125_lt": 34.0},
        "pre_post_nominal_variable_distribution_rows": len(fixed_distribution_rows),
        "frozen_nominal_deployment_thresholds_unchanged": deployment,
        "selection_stability_all1000_checkpoint": str(all1000_checkpoint.relative_to(repo)),
        "selection_stability_all1000_summary_sha256": sha256_file(all1000_summary_path),
        "paired_metric_bootstrap": {
            "draw_sha256": EXPECTED_DRAW_SHA256,
            "replicas": 2000,
            "seed": 20260727,
            "source_groups_in_draw_universe": 441,
            "strata": 23,
            "paired_across_fixed_selections": True,
            "selection_reoptimization_inside_bootstrap": False,
        },
        "pooled_metrics": {
            selection: {
                scope: {
                    metric: pooled_lookup[(selection, scope)][metric]
                    for metric in (
                        "signal_physical_efficiency", "background_physical_efficiency",
                        "background_rejection", "signal_selected_signed_yield",
                        "background_selected_signed_yield", "signal_selected_sumw2",
                        "background_selected_sumw2", "signal_selected_effective_events",
                        "background_selected_effective_events",
                        "signal_finite_mc_relative_uncertainty",
                        "background_finite_mc_relative_uncertainty",
                        "signal_over_background", "asimov_significance_stat_only",
                    )
                }
                for scope in (*CATEGORIES, "combined")
            }
            for selection in (
                "nested_outer_oof", "historical_rhh125125_lt34",
                "fixed_nominal_deployment_cut",
            )
        },
        "systematics_included": False,
        "validation_payloads_opened": 0,
        "test_payloads_opened": 0,
        "next": "freeze_master_train_only_cut_baseline_before_validation_authorization",
    }
    summary_path = output / "train_performance_summary.json"
    write_json(summary_path, summary)

    require(
        sha256_file(tables / "paired_bootstrap_difference_summary.tsv")
        == sha256_file(figure_data / "paired_bootstrap_difference_summary.tsv"),
        "paired-bootstrap figure-data copy mismatch",
    )
    files_before_manifest = sorted(
        path for path in output.rglob("*")
        if path.is_file() and path.name not in {"artifact_manifest.json", "SHA256SUMS"}
    )
    manifest = {
        "schema_version": 1,
        "status": "pass_hh4b_cut_baseline_train_performance_manifest",
        "implementation": {
            "path": str(Path(__file__).resolve().relative_to(repo)),
            "sha256": sha256_file(Path(__file__).resolve()),
        },
        "repository_head_at_execution": repository_head,
        "inputs": {
            str(path.relative_to(repo)): sha256_file(path)
            for path in (
                winners_path, deployment_path, historical_freeze,
                historical_evaluation, draw_manifest, all1000_summary_path,
            )
        },
        "external_inputs": {
            str(fold_payload / "SHA256SUMS"): EXPECTED_FOLD_PAYLOAD_SUMS_SHA256,
            str(draw_path): EXPECTED_DRAW_SHA256,
        },
        "outputs": {
            str(path.relative_to(output)): {
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
            for path in files_before_manifest
        },
        "validation_payloads_opened": 0,
        "test_payloads_opened": 0,
    }
    write_json(manifests / "artifact_manifest.json", manifest)
    checksum_paths = sorted(path for path in output.rglob("*") if path.is_file())
    with (output / "SHA256SUMS").open("w", encoding="utf-8", newline="") as handle:
        for path in checksum_paths:
            handle.write(f"{sha256_file(path)}  ./{path.relative_to(output).as_posix()}\n")
    return summary


def self_check() -> None:
    signal = 417.46273293569186
    background = 20240622849.775902
    require(math.isclose(asimov_significance(signal, background), 0.0029343085318647984,
                         rel_tol=2.0e-15), "historical Asimov benchmark mismatch")
    require(math.isclose(effective_events(10.0, 4.0), 25.0), "N_eff self-check mismatch")
    require(math.isclose(relative_mc_uncertainty(10.0, 4.0), 0.2),
            "relative-uncertainty self-check mismatch")
    print("TRAIN_PERFORMANCE_METRIC_SELF_CHECK=PASS")
    print("HISTORICAL_ASIMOV_BENCHMARK=PASS")
    print("VALIDATION_PAYLOADS_OPENED=0")
    print("TEST_PAYLOADS_OPENED=0")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--repo",
        type=Path,
        default=Path("/uscms_data/d3/iturkmen/repos/hh-bbww-baselines"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts/hh4b_cut_baseline/train_performance"),
    )
    parser.add_argument(
        "--all1000-checkpoint",
        type=Path,
        default=Path(
            "docs/checkpoints/"
            "hh4b_train_multivariate_cut_selection_stability_all1000_aggregation_20260810_v1"
        ),
    )
    parser.add_argument(
        "--fold-payload",
        type=Path,
        default=Path(
            "/uscms_data/d3/iturkmen/hh4b_delphes/baselines/"
            "multivariate_cut_full_train_scan_v9_20260807T051625Z/build/payload"
        ),
    )
    parser.add_argument(
        "--bootstrap-draws",
        type=Path,
        default=Path(
            "/uscms_data/d3/iturkmen/hh4b_delphes/baselines/"
            "hh4b_train_historical_rhh125125_lt34_v4_20260806T055823Z/"
            "source_group_bootstrap_draw_counts.parquet"
        ),
    )
    parser.add_argument("--self-check", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if args.self_check:
        self_check()
        return 0
    repo = args.repo.resolve()
    output = (args.output_dir if args.output_dir.is_absolute() else repo / args.output_dir).absolute()
    require(output.is_relative_to(repo), "output directory must be in repository")
    require(not output.exists() and not output.is_symlink(), f"output already exists: {output}")
    checkpoint = (
        args.all1000_checkpoint
        if args.all1000_checkpoint.is_absolute()
        else repo / args.all1000_checkpoint
    ).resolve()
    require(checkpoint.is_relative_to(repo), "all1000 checkpoint must be in repository")
    try:
        summary = build(
            repo,
            output,
            checkpoint,
            args.fold_payload.resolve(),
            args.bootstrap_draws.resolve(),
        )
    except Exception:
        if output.is_dir() and not output.is_symlink():
            (output / "BUILD_FAILED_DO_NOT_USE.txt").write_text(
                "STATUS=FAIL\nDO_NOT_USE_PARTIAL_OUTPUT=TRUE\n"
                "VALIDATION_PAYLOADS_OPENED=0\nTEST_PAYLOADS_OPENED=0\n",
                encoding="utf-8",
            )
        raise
    print("HH4B_CUT_BASELINE_TRAIN_PERFORMANCE=PASS")
    print(f"OUTPUT_DIR={output}")
    print(f"REPOSITORY_HEAD={summary['repository_head_at_execution']}")
    print("PRIMARY_GENERALIZATION_ESTIMATE=POOLED_NESTED_OUTER_OOF")
    print("HISTORICAL_FIXED_COMPARATOR=R_HH_125_125_LT_34")
    print("PAIRED_METRIC_BOOTSTRAP_REPLICAS=2000")
    print("SELECTION_REOPTIMIZATION_INSIDE_METRIC_BOOTSTRAP=FALSE")
    print("VALIDATION_PAYLOADS_OPENED=0")
    print("TEST_PAYLOADS_OPENED=0")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except PerformanceError as exc:
        print(f"HH4B_CUT_BASELINE_TRAIN_PERFORMANCE=FAIL\nERROR={exc}", file=sys.stderr)
        print("VALIDATION_PAYLOADS_OPENED=0\nTEST_PAYLOADS_OPENED=0", file=sys.stderr)
        raise SystemExit(1) from exc
