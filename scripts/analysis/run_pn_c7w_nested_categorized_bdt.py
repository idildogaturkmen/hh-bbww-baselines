#!/usr/bin/env python3
"""Run the frozen train-only nested HIG-24-015-inspired categorized BDT."""

from __future__ import annotations

import argparse
from collections import Counter
import json
import math
import platform
import shutil
import subprocess
import sys
import time
import traceback
from pathlib import Path
from typing import Any, Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import sklearn
import xgboost
from xgboost import XGBClassifier

from hh4b_plot_style import apply_hh4b_paper_style
from pn_c7_bootstrap_common import group_sum, member_positions
from pn_c7_ml_common import (
    artifact_rows,
    bootstrap_quantile_summary,
    parse_sha256sums,
    prepare_staging,
    require,
    seal_checkpoint,
    sha256,
    source_member_bootstrap_draws,
    threshold_at_efficiency,
    write_json,
    write_tsv,
)
from pn_c7w_categorized_bdt_common import (
    CATEGORY_COUNTS,
    COMPLEMENTARY_STRATA,
    SPECIALIZED_STRATA,
    STAGE1_EFFICIENCIES,
    SUPPORT_CONTRACTS,
    assign_categories,
    bootstrap_fixed_categories,
    evaluate_categories,
    optimize_staircase,
    select_candidates,
    summarize_replicas,
)


REPO = Path("/uscms_data/d3/iturkmen/repos/hh-bbww-baselines")
CHECKPOINT_ROOT = Path("/uscms_data/d3/iturkmen/hh4b_delphes/harvest_checkpoints")
C7S = CHECKPOINT_ROOT / "pn_c7s_final_train_baseline_inputs_20260802_v1"
C7R = CHECKPOINT_ROOT / "pn_c7r_threeb_fourb_multijet_transfer_20260802_v1"
C7T = CHECKPOINT_ROOT / "pn_c7t_reestablished_train_baselines_20260802_v1"
C7V = CHECKPOINT_ROOT / "pn_c7v_jhep_results_and_figures_20260803_v4"
MANIFEST_SHA = {
    "c7s": "9885748add80bfd44cb01947e5ae21b58e08f28396c55cf1fdc7934063309ae1",
    "c7r": "a663fe73d23db0e2ee856dd041916a22008c020c9b11a7eb59e3dd14acbd19c6",
    "c7t": "2de28666cdd9e055747b7e2d3df70968bad3a8d8d0613bf3d5c6e05a1d6fc016",
    "c7v": "40228a4ad86337d1a505c6ec64611ecf0a84c3f1c5a49adc73ff5f50bf776676",
}
CONFIG = REPO / "configs/baselines/hh4b_expanded_cut_bdt_protocol_v2.json"
CONFIG_SHA = "cc56783a4a5a19619c9dd69c1c82878a47cdf6ee6b3db24c43ca11f3a60fa636"
AMENDMENT = REPO / "docs/paper/jhep_hh4b_ml/HIG_24_015_IMPLEMENTATION_AMENDMENT.md"
TRAIN_PATH = C7S / "tables/train_fourb_model_development.parquet"
PROJECTION_PATH = C7S / "tables/train_primary_physical_projection.parquet"
DIRECT_PATH = C7S / "tables/train_direct_qcd_secondary_projection.parquet"
BOOTSTRAP_SEED = 20260802
BOOTSTRAP_REPLICATES = 1000
COMMON_REGISTRY_SHA = "37218e5531018ea2a79473bfbee43df657b309cf06d89751509e74b586d9dc29"
VARIANTS = ("mass_aware", "mass_plane_blind")
DISPLAY = {"mass_aware": "Mass-aware categorized BDT", "mass_plane_blind": "Mass-plane-blind categorized BDT"}
COLORS = {"mass_aware": "#0072B2", "mass_plane_blind": "#D55E00", "inclusive_bdt": "#009E73"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--paper-output", type=Path, required=True)
    parser.add_argument("--inspection-report", type=Path)
    parser.add_argument("--smoke", action="store_true")
    return parser.parse_args()


def verify_checkpoint(path: Path, expected_sha: str) -> list[dict[str, Any]]:
    manifest = path / "SHA256SUMS"
    require(sha256(manifest) == expected_sha, f"checkpoint manifest identity drift: {path}")
    rows = []
    for relative, expected in parse_sha256sums(manifest).items():
        member = path / relative
        require(member.is_file() and sha256(member) == expected, f"checkpoint member drift: {member}")
        rows.append({"checkpoint": path.name, "relative_path": relative, "sha256": expected, "bytes": member.stat().st_size})
    return rows


def load_inputs() -> dict[str, Any]:
    evidence = []
    for tag, path in (("c7s", C7S), ("c7r", C7R), ("c7t", C7T), ("c7v", C7V)):
        evidence.extend(verify_checkpoint(path, MANIFEST_SHA[tag]))
    require(sha256(CONFIG) == CONFIG_SHA, "frozen BDT protocol checksum drift")
    require(AMENDMENT.is_file(), "categorized-BDT implementation amendment missing")
    training = pq.read_table(TRAIN_PATH).to_pandas()
    projection = pq.read_table(PROJECTION_PATH).to_pandas()
    direct = pq.read_table(DIRECT_PATH).to_pandas()
    members = pd.read_csv(C7S / "member_fold_registry.tsv", sep="\t")
    require((len(training), len(projection), len(direct), len(members)) == (30705, 31225, 30705, 441), "c7s count drift")
    require(set(training.registry_final_split) == {"train"} and set(projection.registry_final_split) == {"train"}, "non-train payload")
    require(training[["registry_group_id", "event"]].reset_index(drop=True).equals(
        direct[["registry_group_id", "event"]].reset_index(drop=True)
    ), "training/direct row identity drift")
    strata = set(training.loc[training.registry_sample_class == "background", "registry_stratum"].astype(str))
    require(strata == SPECIALIZED_STRATA | COMPLEMENTARY_STRATA, "background taxonomy drift")
    specialized = training.registry_stratum.astype(str).isin(SPECIALIZED_STRATA) & (training.registry_training_target == 0)
    complementary = training.registry_stratum.astype(str).isin(COMPLEMENTARY_STRATA) & (training.registry_training_target == 0)
    require(int(specialized.sum()) == 15615 and training.loc[specialized, "registry_group_id"].nunique() == 130,
            "specialized taxonomy count drift")
    require(int(complementary.sum()) == 5753 and training.loc[complementary, "registry_group_id"].nunique() == 58,
            "complementary taxonomy count drift")
    config = json.loads(CONFIG.read_text())
    summary = json.loads((C7S / "summary.json").read_text())
    features = {
        "mass_aware": list(summary["mass_aware_features"]),
        "mass_plane_blind": list(summary["mass_plane_blind_features"]),
    }
    require(features["mass_aware"] == config["models"]["global_mass_aware"]["features"], "34-feature order drift")
    require(features["mass_plane_blind"] == config["models"]["global_mass_plane_blind"]["features"], "30-feature order drift")
    factor = pd.read_csv(C7R / "transfer_factor_registry.tsv", sep="\t")
    inclusive = factor[factor.mhh_category == "inclusive_mhh"]
    nominal = inclusive[inclusive.factor_scheme == "cms_cr_nominal"].iloc[0]
    factor_values = [float(nominal.transfer_factor)] + [
        float(value) for value in inclusive.transfer_factor if float(value) != float(nominal.transfer_factor)
    ]
    require(len(factor_values) == 3, "inclusive factor scheme drift")
    return {
        "training": training, "projection": projection, "direct": direct, "members": members,
        "features": features, "config": config, "factor_values": factor_values,
        "factor_stat_relative": float(nominal.transfer_factor_relative_statistical_uncertainty),
        "evidence": evidence,
    }


def bdt_parameters(config: dict[str, Any], *, seed: int, smoke: bool) -> dict[str, Any]:
    params = dict(config["models"]["global_mass_aware"]["parameters"])
    params.update(config["estimator_fixed_parameters"])
    params["random_state"] = int(seed)
    if smoke:
        params["n_estimators"] = 8
        params["n_jobs"] = min(4, int(params.get("n_jobs", 4)))
    return params


def fit_bdt(frame: pd.DataFrame, mask: np.ndarray, features: list[str], target: np.ndarray,
            weight: np.ndarray, config: dict[str, Any], *, seed: int, smoke: bool) -> XGBClassifier:
    require(int(mask.sum()) > 100 and set(np.unique(target[mask])) == {0, 1}, "invalid BDT fit population")
    model = XGBClassifier(**bdt_parameters(config, seed=seed, smoke=smoke))
    model.fit(frame.loc[mask, features].to_numpy(dtype=np.float32), target[mask], sample_weight=weight[mask])
    return model


def predict(model: XGBClassifier, frame: pd.DataFrame, mask: np.ndarray, features: list[str]) -> np.ndarray:
    result = np.full(len(frame), np.nan, dtype=np.float64)
    result[mask] = model.predict_proba(frame.loc[mask, features].to_numpy(dtype=np.float32))[:, 1]
    return result


def inner_nested_scores(inputs: dict[str, Any], variant: str, outer_fold: int, *, smoke: bool) -> dict[str, Any]:
    training = inputs["training"]
    projection = inputs["projection"]
    features = inputs["features"][variant]
    config = inputs["config"]
    folds = training.registry_oof_fold.to_numpy(dtype=np.int8)
    projection_folds = projection.registry_oof_fold.to_numpy(dtype=np.int8)
    target = training.registry_training_target.to_numpy(dtype=np.int8)
    weights = training.development_hierarchical_weight.to_numpy(dtype=np.float64)
    outer_train = folds != outer_fold
    projection_outer_train = projection_folds != outer_fold
    stage1_train = np.full(len(training), np.nan)
    stage1_projection = np.full(len(projection), np.nan)
    inner_folds = sorted(int(value) for value in np.unique(folds[outer_train]))
    for inner_fold in inner_folds:
        fit = outer_train & (folds != inner_fold)
        held = outer_train & (folds == inner_fold)
        projected = projection_outer_train & (projection_folds == inner_fold)
        model = fit_bdt(training, fit, features, target, weights, config,
                        seed=20260725 + outer_fold * 100 + inner_fold, smoke=smoke)
        stage1_train[held] = model.predict_proba(training.loc[held, features].to_numpy(np.float32))[:, 1]
        stage1_projection[projected] = model.predict_proba(projection.loc[projected, features].to_numpy(np.float32))[:, 1]
    require(np.isfinite(stage1_train[outer_train]).all() and np.isfinite(stage1_projection[projection_outer_train]).all(),
            "inner stage-1 OOF coverage failure")
    thresholds = {
        efficiency: threshold_at_efficiency(stage1_train[outer_train], target[outer_train] == 1,
                                            weights[outer_train], efficiency)
        for efficiency in STAGE1_EFFICIENCIES
    }
    stage2_train_by_efficiency = {}
    stage2_projection_by_efficiency = {}
    specialized_or_signal = (target == 1) | training.registry_stratum.astype(str).isin(SPECIALIZED_STRATA).to_numpy()
    for efficiency in STAGE1_EFFICIENCIES:
        threshold = thresholds[efficiency]
        train_scores = np.full(len(training), np.nan)
        projection_scores = np.full(len(projection), np.nan)
        for inner_fold in inner_folds:
            held = outer_train & (folds == inner_fold)
            projected = projection_outer_train & (projection_folds == inner_fold)
            fit = outer_train & (folds != inner_fold) & specialized_or_signal & (stage1_train >= threshold)
            model = fit_bdt(training, fit, features, target, weights, config,
                            seed=20261725 + outer_fold * 1000 + int(efficiency * 100) * 10 + inner_fold,
                            smoke=smoke)
            train_scores[held] = model.predict_proba(training.loc[held, features].to_numpy(np.float32))[:, 1]
            projection_scores[projected] = model.predict_proba(projection.loc[projected, features].to_numpy(np.float32))[:, 1]
        require(np.isfinite(train_scores[outer_train]).all() and np.isfinite(projection_scores[projection_outer_train]).all(),
                "inner stage-2 OOF coverage failure")
        stage2_train_by_efficiency[efficiency] = train_scores
        stage2_projection_by_efficiency[efficiency] = projection_scores
    return {
        "outer_train": outer_train, "projection_outer_train": projection_outer_train,
        "stage1_train": stage1_train, "stage1_projection": stage1_projection,
        "stage2_train_by_efficiency": stage2_train_by_efficiency,
        "stage2_projection_by_efficiency": stage2_projection_by_efficiency,
        "thresholds": thresholds,
    }


def nested_outer_training(inputs: dict[str, Any], models: Path, *, smoke: bool) -> dict[str, Any]:
    training = inputs["training"]
    projection = inputs["projection"]
    direct = inputs["direct"]
    folds = training.registry_oof_fold.to_numpy(dtype=np.int8)
    projection_folds = projection.registry_oof_fold.to_numpy(dtype=np.int8)
    target = training.registry_training_target.to_numpy(dtype=np.int8)
    weights = training.development_hierarchical_weight.to_numpy(dtype=np.float64)
    config = inputs["config"]
    candidate_frames = []
    candidate_category_frames = []
    selection_rows = []
    artifact_rows_output = []
    predictions: dict[str, dict[int, dict[str, np.ndarray]]] = {
        variant: {
            count: {
                "train_stage1": np.full(len(training), np.nan),
                "train_stage2": np.full(len(training), np.nan),
                "train_category": np.full(len(training), -1, dtype=np.int16),
                "projection_stage1": np.full(len(projection), np.nan),
                "projection_stage2": np.full(len(projection), np.nan),
                "projection_category": np.full(len(projection), -1, dtype=np.int16),
            } for count in CATEGORY_COUNTS
        } for variant in VARIANTS
    }
    for variant_index, variant in enumerate(VARIANTS):
        features = inputs["features"][variant]
        for outer_fold in range(5):
            print(f"{variant}: nested outer fold {outer_fold}", flush=True)
            nested = inner_nested_scores(inputs, variant, outer_fold, smoke=smoke)
            outer_train = nested["outer_train"]
            projection_outer_train = nested["projection_outer_train"]
            candidates, candidate_categories = optimize_staircase(
                training.loc[outer_train].reset_index(drop=True),
                projection.loc[projection_outer_train].reset_index(drop=True),
                direct.loc[outer_train].reset_index(drop=True),
                nested["stage1_train"][outer_train],
                {efficiency: values[outer_train] for efficiency, values in nested["stage2_train_by_efficiency"].items()},
                nested["stage1_projection"][projection_outer_train],
                {efficiency: values[projection_outer_train] for efficiency, values in nested["stage2_projection_by_efficiency"].items()},
                nested["thresholds"], inputs["factor_values"], inputs["factor_stat_relative"],
            )
            candidates.insert(0, "outer_fold", outer_fold); candidates.insert(0, "variant", variant)
            candidate_categories.insert(0, "outer_fold", outer_fold); candidate_categories.insert(0, "variant", variant)
            candidate_frames.append(candidates); candidate_category_frames.append(candidate_categories)
            selected, by_count = select_candidates(candidates)
            selection_rows.append({
                "variant": variant, "outer_fold": outer_fold, "selection_role": "outer_selected",
                "candidate_id": selected.candidate_id, "category_count": int(selected.category_count),
                "stage1_target_signal_efficiency": float(selected.stage1_target_signal_efficiency),
                "stage1_threshold": float(selected.stage1_threshold),
                "stage2_boundaries_json": selected.stage2_boundaries_json,
                "boundary_quantiles_json": selected.boundary_quantiles_json,
                "inner_nominal_ZA": float(selected.combined_nominal_asimov_ZA),
                "inner_systematic_ZA": float(selected.combined_systematic_aware_asimov_ZA),
                "boundary_instability": float(selected.boundary_instability),
                "outer_held_used_for_selection": False,
            })
            fit_outer = folds != outer_fold
            held_outer = folds == outer_fold
            projected_outer = projection_folds == outer_fold
            stage1_model = fit_bdt(training, fit_outer, features, target, weights, config,
                                   seed=20262725 + variant_index * 10000 + outer_fold, smoke=smoke)
            stage1_fit = stage1_model.predict_proba(training.loc[fit_outer, features].to_numpy(np.float32))[:, 1]
            stage1_held = stage1_model.predict_proba(training.loc[held_outer, features].to_numpy(np.float32))[:, 1]
            stage1_projection = stage1_model.predict_proba(projection.loc[projected_outer, features].to_numpy(np.float32))[:, 1]
            stage1_path = models / f"{variant}_outer{outer_fold}_stage1.json"
            stage1_model.save_model(stage1_path)
            artifact_rows_output.append({"variant": variant, "outer_fold": outer_fold, "stage": "stage1", "category_count": "all",
                                         "path": f"models/{stage1_path.name}", "sha256": sha256(stage1_path)})
            specialized_or_signal = (target == 1) | training.registry_stratum.astype(str).isin(SPECIALIZED_STRATA).to_numpy()
            for count, candidate in by_count.items():
                threshold = float(candidate.stage1_threshold)
                boundaries = np.asarray(json.loads(candidate.stage2_boundaries_json), dtype=np.float64)
                stage2_fit_mask = fit_outer.copy()
                stage2_fit_mask[fit_outer] &= specialized_or_signal[fit_outer] & (stage1_fit >= threshold)
                stage2_model = fit_bdt(training, stage2_fit_mask, features, target, weights, config,
                                       seed=20263725 + variant_index * 10000 + outer_fold * 10 + count, smoke=smoke)
                train_stage2 = stage2_model.predict_proba(training.loc[held_outer, features].to_numpy(np.float32))[:, 1]
                projection_stage2 = stage2_model.predict_proba(projection.loc[projected_outer, features].to_numpy(np.float32))[:, 1]
                target_store = predictions[variant][count]
                target_store["train_stage1"][held_outer] = stage1_held
                target_store["train_stage2"][held_outer] = train_stage2
                target_store["train_category"][held_outer] = assign_categories(stage1_held, train_stage2, threshold, boundaries)
                target_store["projection_stage1"][projected_outer] = stage1_projection
                target_store["projection_stage2"][projected_outer] = projection_stage2
                target_store["projection_category"][projected_outer] = assign_categories(stage1_projection, projection_stage2, threshold, boundaries)
                stage2_path = models / f"{variant}_outer{outer_fold}_stage2_n{count}.json"
                stage2_model.save_model(stage2_path)
                artifact_rows_output.append({"variant": variant, "outer_fold": outer_fold, "stage": "stage2", "category_count": count,
                                             "path": f"models/{stage2_path.name}", "sha256": sha256(stage2_path)})
                selection_rows.append({
                    "variant": variant, "outer_fold": outer_fold, "selection_role": f"best_supported_n{count}",
                    "candidate_id": candidate.candidate_id, "category_count": count,
                    "stage1_target_signal_efficiency": float(candidate.stage1_target_signal_efficiency),
                    "stage1_threshold": threshold, "stage2_boundaries_json": candidate.stage2_boundaries_json,
                    "boundary_quantiles_json": candidate.boundary_quantiles_json,
                    "inner_nominal_ZA": float(candidate.combined_nominal_asimov_ZA),
                    "inner_systematic_ZA": float(candidate.combined_systematic_aware_asimov_ZA),
                    "boundary_instability": float(candidate.boundary_instability),
                    "outer_held_used_for_selection": False,
                })
    for variant in VARIANTS:
        for count in CATEGORY_COUNTS:
            for field, values in predictions[variant][count].items():
                if "category" in field:
                    require(bool((values >= 0).all()), f"OOF category coverage failure: {variant}/{count}/{field}")
                else:
                    require(bool(np.isfinite(values).all()), f"OOF score coverage failure: {variant}/{count}/{field}")
    return {
        "candidates": pd.concat(candidate_frames, ignore_index=True),
        "candidate_categories": pd.concat(candidate_category_frames, ignore_index=True),
        "selections": pd.DataFrame(selection_rows),
        "model_artifacts": pd.DataFrame(artifact_rows_output),
        "predictions": predictions,
    }


def configuration_label(variant: str, count: int) -> str:
    role = "fixed5" if count == 5 else f"scan_n{count}"
    return f"{variant}_{role}_n{count}"


def build_outer_selected_predictions(inputs: dict[str, Any], nested: dict[str, Any]) -> tuple[dict[str, dict[str, np.ndarray]], dict[str, dict[int, int]]]:
    """Stitch only the independently selected configuration for each outer fold.

    Choosing one modal count after all outer folds would allow a source to affect
    the configuration used when that source is held out.  This function instead
    applies the count selected inside each fold's outer-training partition to
    that fold and nothing else.
    """

    training_folds = inputs["training"].registry_oof_fold.to_numpy(dtype=np.int8)
    projection_folds = inputs["projection"].registry_oof_fold.to_numpy(dtype=np.int8)
    result: dict[str, dict[str, np.ndarray]] = {}
    counts_by_fold: dict[str, dict[int, int]] = {}
    selected = nested["selections"][nested["selections"].selection_role == "outer_selected"]
    for variant in VARIANTS:
        result[variant] = {
            "train_stage1": np.full(len(training_folds), np.nan),
            "train_stage2": np.full(len(training_folds), np.nan),
            "train_category": np.full(len(training_folds), -1, dtype=np.int16),
            "projection_stage1": np.full(len(projection_folds), np.nan),
            "projection_stage2": np.full(len(projection_folds), np.nan),
            "projection_category": np.full(len(projection_folds), -1, dtype=np.int16),
        }
        counts_by_fold[variant] = {}
        rows = selected[selected.variant == variant]
        require(len(rows) == 5 and rows.outer_fold.nunique() == 5, f"outer-selected row drift: {variant}")
        for row in rows.itertuples():
            fold = int(row.outer_fold)
            count = int(row.category_count)
            counts_by_fold[variant][fold] = count
            source = nested["predictions"][variant][count]
            train_mask = training_folds == fold
            projection_mask = projection_folds == fold
            for field in ("train_stage1", "train_stage2", "train_category"):
                result[variant][field][train_mask] = source[field][train_mask]
            for field in ("projection_stage1", "projection_stage2", "projection_category"):
                result[variant][field][projection_mask] = source[field][projection_mask]
        for field, values in result[variant].items():
            if "category" in field:
                require(bool((values >= 0).all()), f"outer-selected category coverage failure: {variant}/{field}")
            else:
                require(bool(np.isfinite(values).all()), f"outer-selected score coverage failure: {variant}/{field}")
    return result, counts_by_fold


def evaluate_all_configurations(inputs: dict[str, Any], nested: dict[str, Any], draws: np.ndarray,
                                *, smoke: bool) -> dict[str, Any]:
    projection = inputs["projection"]
    direct = inputs["direct"]
    members = inputs["members"]
    selected_predictions, selected_counts_by_fold = build_outer_selected_predictions(inputs, nested)
    central_rows = []
    central_category_rows = []
    sensitivity_rows = []
    combined_replica_frames = []
    category_replica_frames = []
    for variant in VARIANTS:
        for count in CATEGORY_COUNTS:
            values = nested["predictions"][variant][count]
            categories = values["projection_category"]
            direct_categories = values["train_category"]
            label = configuration_label(variant, count)
            summary, category_rows = evaluate_categories(
                projection, direct, categories, direct_categories,
                inputs["factor_values"], inputs["factor_stat_relative"], contract_name="nominal",
                require_category_support_for_systematic=False,
            )
            central_rows.append({"variant": variant, "category_count": count, "configuration": label, **summary})
            central_category_rows.extend(
                {"variant": variant, "category_count": count, "configuration": label, **row}
                for row in category_rows
            )
            for contract_name in SUPPORT_CONTRACTS:
                contract_summary, contract_categories = evaluate_categories(
                    projection, direct, categories, direct_categories,
                    inputs["factor_values"], inputs["factor_stat_relative"], contract_name=contract_name,
                )
                sensitivity_rows.append({
                    "variant": variant, "category_count": count, "configuration": label,
                    "contract_name": contract_name, "support_pass": contract_summary["support_pass"],
                    "support_failure_reason": contract_summary["support_failure_reason"],
                    "minimum_support_margin": contract_summary["minimum_support_margin"],
                    "categories_passing": sum(bool(row["support_pass"]) for row in contract_categories),
                    "categories_total": len(contract_categories),
                    "used_for_primary_selection": contract_name == "nominal",
                })
            combined, by_category = bootstrap_fixed_categories(
                projection, direct, categories, direct_categories, members, draws,
                inputs["factor_values"], inputs["factor_stat_relative"], label=label,
            )
            combined.insert(0, "category_count", count); combined.insert(0, "variant", variant)
            by_category.insert(0, "category_count", count); by_category.insert(0, "variant", variant)
            combined_replica_frames.append(combined); category_replica_frames.append(by_category)
            print(f"bootstrapped {label}", flush=True)
        values = selected_predictions[variant]
        categories = values["projection_category"]
        direct_categories = values["train_category"]
        label = f"{variant}_outer_selected"
        effective_count = int(max(categories.max(), direct_categories.max()) + 1)
        summary, category_rows = evaluate_categories(
            projection, direct, categories, direct_categories,
            inputs["factor_values"], inputs["factor_stat_relative"], contract_name="nominal",
            require_category_support_for_systematic=False,
        )
        central_rows.append({"variant": variant, "category_count": effective_count, "configuration": label,
                             "selection_role": "outer_selected", **summary})
        central_category_rows.extend(
            {"variant": variant, "category_count": effective_count, "configuration": label,
             "selection_role": "outer_selected", **row} for row in category_rows
        )
        for contract_name in SUPPORT_CONTRACTS:
            contract_summary, contract_categories = evaluate_categories(
                projection, direct, categories, direct_categories,
                inputs["factor_values"], inputs["factor_stat_relative"], contract_name=contract_name,
            )
            sensitivity_rows.append({
                "variant": variant, "category_count": effective_count, "configuration": label,
                "selection_role": "outer_selected", "contract_name": contract_name,
                "support_pass": contract_summary["support_pass"],
                "support_failure_reason": contract_summary["support_failure_reason"],
                "minimum_support_margin": contract_summary["minimum_support_margin"],
                "categories_passing": sum(bool(row["support_pass"]) for row in contract_categories),
                "categories_total": len(contract_categories),
                "used_for_primary_selection": contract_name == "nominal",
            })
        combined, by_category = bootstrap_fixed_categories(
            projection, direct, categories, direct_categories, members, draws,
            inputs["factor_values"], inputs["factor_stat_relative"], label=label,
        )
        combined.insert(0, "category_count", effective_count); combined.insert(0, "variant", variant)
        by_category.insert(0, "category_count", effective_count); by_category.insert(0, "variant", variant)
        combined_replica_frames.append(combined); category_replica_frames.append(by_category)
        print(f"bootstrapped {label}", flush=True)
    central = pd.DataFrame(central_rows)
    central_categories = pd.DataFrame(central_category_rows)
    combined_replicas = pd.concat(combined_replica_frames, ignore_index=True)
    category_replicas = pd.concat(category_replica_frames, ignore_index=True)
    combined_metrics = [
        "combined_nominal_asimov_ZA", "combined_systematic_aware_asimov_ZA",
        "signal_yield", "background_yield", "signal_over_background", "signal_over_sqrt_background",
        "background_neff", "signal_neff", "transferred_qcd_fraction", "direct_qcd_closure_ratio",
        "shared_qcd_relative_nonclosure", "shared_multijet_relative_nuisance", "minimum_support_margin",
    ]
    combined_summary = summarize_replicas(combined_replicas, combined_metrics, ["variant", "category_count", "label"])
    central_lookup = central.set_index("configuration")
    combined_summary["nominal_full_oof"] = [
        float(central_lookup.loc[row.label, row.metric])
        for row in combined_summary.itertuples()
    ]
    category_metrics = [
        "signal_yield", "ordinary_background_yield", "transferred_qcd_yield", "background_yield",
        "signal_over_background", "signal_over_sqrt_background", "background_neff", "signal_neff",
        "transferred_qcd_fraction", "direct_qcd_closure_yield", "direct_qcd_closure_ratio", "direct_qcd_neff",
        "minimum_support_margin",
        "nominal_asimov_ZA_contribution", "systematic_aware_ZA_standalone_contribution",
        "signal_rows", "background_rows", "background_sources", "transferred_rows", "transferred_sources",
        "direct_qcd_rows", "direct_qcd_sources",
    ]
    category_summary = summarize_replicas(
        category_replicas.rename(columns={"support_failure_reason": "failure_reason"}),
        category_metrics, ["variant", "category_count", "label", "category"],
    )
    category_lookup = central_categories.set_index(["configuration", "category"])
    category_summary["nominal_full_oof"] = [
        float(category_lookup.loc[(row.label, row.category), row.metric])
        for row in category_summary.itertuples()
    ]
    return {
        "selected_predictions": selected_predictions,
        "selected_counts_by_fold": selected_counts_by_fold,
        "central": central,
        "central_categories": central_categories,
        "support_sensitivity": pd.DataFrame(sensitivity_rows),
        "combined_replicas": combined_replicas,
        "category_replicas": category_replicas,
        "combined_summary": combined_summary,
        "category_summary": category_summary,
    }


def paired_difference_summary(values: np.ndarray, reasons: list[str]) -> dict[str, Any]:
    finite = np.isfinite(values)
    valid = values[finite]
    if len(valid):
        summary = bootstrap_quantile_summary(values, invalid_reasons=reasons)
    else:
        counts = Counter(reason or "model or reference metric undefined" for reason in reasons)
        summary = {
            "bootstrap_mean": float("nan"), "bootstrap_median": float("nan"),
            "bootstrap_standard_deviation": float("nan"), "bootstrap_p16": float("nan"),
            "bootstrap_p84": float("nan"), "bootstrap_p2p5": float("nan"),
            "bootstrap_p97p5": float("nan"), "valid_replicas": 0,
            "invalid_replicas": len(values),
            "invalid_reason_counts_json": json.dumps(dict(counts), sort_keys=True, separators=(",", ":")),
        }
    summary.update({
        "fraction_valid_difference_greater_than_zero": float(np.mean(valid > 0.0)) if len(valid) else float("nan"),
        "interval_includes_zero_68": bool(summary["bootstrap_p16"] <= 0.0 <= summary["bootstrap_p84"]) if len(valid) else False,
        "interval_includes_zero_95": bool(summary["bootstrap_p2p5"] <= 0.0 <= summary["bootstrap_p97p5"]) if len(valid) else False,
    })
    return summary


def build_paired_comparisons(evaluated: dict[str, Any]) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    replicas = evaluated["combined_replicas"]
    replica_count = int(replicas.replica_id.nunique())
    central = evaluated["central"].set_index("configuration")
    inclusive_replicas = pd.read_parquet(C7V / "bootstrap_metric_replicas.parquet")
    inclusive_replicas = inclusive_replicas[(inclusive_replicas.model == "bdt") & (inclusive_replicas.replica_id < replica_count)].sort_values("replica_id")
    require(inclusive_replicas.replica_id.tolist() == list(range(replica_count)), "inclusive registry alignment drift")
    inclusive_central = pd.read_csv(C7T / "baseline_metrics.tsv", sep="\t").set_index("baseline").loc["bdt"]
    metric_map = {
        "combined_nominal_asimov_ZA": "nominal_asimov_ZA",
        "combined_systematic_aware_asimov_ZA": "systematic_aware_asimov_ZA",
        "signal_over_background": "signal_over_background",
        "background_neff": "background_effective_events",
    }
    comparisons: list[tuple[str, str, pd.DataFrame, pd.Series, pd.DataFrame | None, pd.Series | None]] = []
    for variant in VARIANTS:
        for count in CATEGORY_COUNTS:
            label = configuration_label(variant, count)
            frame = replicas[replicas.label == label].sort_values("replica_id")
            comparisons.append((f"{label}_minus_inclusive_bdt", label, frame, central.loc[label], None, None))
        selected_label = f"{variant}_outer_selected"
        frame = replicas[replicas.label == selected_label].sort_values("replica_id")
        comparisons.append((f"{selected_label}_minus_inclusive_bdt", selected_label, frame,
                            central.loc[selected_label], None, None))
    aware_label = "mass_aware_outer_selected"
    blind_label = "mass_plane_blind_outer_selected"
    aware = replicas[replicas.label == aware_label].sort_values("replica_id")
    blind = replicas[replicas.label == blind_label].sort_values("replica_id")
    comparisons.append(("mass_plane_blind_selected_minus_mass_aware_selected", "mass_plane_blind_selected", blind,
                        central.loc[blind_label], aware, central.loc[aware_label]))
    rows = []
    replica_rows = []
    unavailable = [{
        "comparison": "categorized_BDT_minus_inclusive_BDT", "metric": "weighted_auc", "status": "not_applicable",
        "reason": "categorized result has no documented scalar category-rank score; inclusive-classifier AUC is not assigned to categories",
    }]
    for comparison, model, frame, model_central, reference_frame, reference_central in comparisons:
        require(frame.replica_id.tolist() == list(range(replica_count)), f"categorized registry drift: {comparison}")
        for category_metric, inclusive_metric in metric_map.items():
            if reference_frame is None:
                reference_values = inclusive_replicas[inclusive_metric].to_numpy(dtype=np.float64)
                reference_nominal = float(inclusive_central[
                    "asimov_ZA" if inclusive_metric == "nominal_asimov_ZA" else
                    "systematic_aware_asimov_ZA" if inclusive_metric == "systematic_aware_asimov_ZA" else
                    "background_effective_events" if inclusive_metric == "background_effective_events" else
                    inclusive_metric
                ])
                reference_name = "inclusive_bdt"
            else:
                reference_values = reference_frame[category_metric].to_numpy(dtype=np.float64)
                reference_nominal = float(reference_central[category_metric])
                reference_name = "mass_aware_selected"
            model_values = frame[category_metric].to_numpy(dtype=np.float64)
            differences = model_values - reference_values
            reasons = ["" if math.isfinite(value) else "model or reference metric undefined" for value in differences]
            summary = paired_difference_summary(differences, reasons)
            rows.append({
                "comparison": comparison, "model": model, "reference_model": reference_name,
                "metric": category_metric, "nominal_difference": float(model_central[category_metric]) - reference_nominal,
                **summary,
            })
            replica_rows.extend({
                "comparison": comparison, "model": model, "reference_model": reference_name,
                "metric": category_metric, "replica_id": replica_id,
                "difference": difference, "valid": math.isfinite(difference),
            } for replica_id, difference in enumerate(differences))
    return pd.DataFrame(rows), pd.DataFrame(replica_rows), pd.DataFrame(unavailable)


def build_prediction_tables(inputs: dict[str, Any], nested: dict[str, Any]) -> tuple[pd.DataFrame, pd.DataFrame]:
    training = inputs["training"]
    projection = inputs["projection"]
    train = training[["registry_group_id", "registry_member_index", "registry_oof_fold", "registry_training_target",
                      "registry_stratum", "transport_id", "event"]].copy()
    projected = projection[["registry_group_id", "registry_member_index", "registry_oof_fold", "registry_training_target",
                            "analysis_population_role", "transport_id", "event"]].copy()
    for variant in VARIANTS:
        for count in CATEGORY_COUNTS:
            values = nested["predictions"][variant][count]
            prefix = f"{variant}_n{count}"
            train[f"{prefix}_stage1_score"] = values["train_stage1"]
            train[f"{prefix}_stage2_score"] = values["train_stage2"]
            train[f"{prefix}_category"] = values["train_category"]
            projected[f"{prefix}_stage1_score"] = values["projection_stage1"]
            projected[f"{prefix}_stage2_score"] = values["projection_stage2"]
            projected[f"{prefix}_category"] = values["projection_category"]
    return train, projected


def standardize_source_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    result = []
    for source in rows:
        row = dict(source)
        row.setdefault("uncertainty_central", row.get("central_value", ""))
        row.setdefault("uncertainty_lower_68", "")
        row.setdefault("uncertainty_upper_68", "")
        row.setdefault("uncertainty_valid_replicas", 0)
        row.setdefault("uncertainty_support_flag", "not_applicable")
        row.setdefault("uncertainty_kind", "not_applicable_diagnostic")
        result.append(row)
    return result


def save_figure(fig: Any, figures: Path, paper: Path, stem: str, caption: str,
                rows: list[dict[str, Any]], provenance: list[dict[str, Any]]) -> None:
    rows = standardize_source_rows(rows)
    data = figures / f"{stem}.tsv"
    write_tsv(data, rows)
    for suffix in ("pdf", "png"):
        path = figures / f"{stem}.{suffix}"
        fig.savefig(path, dpi=300 if suffix == "png" else None, bbox_inches="tight")
        shutil.copy2(path, paper / "figures" / path.name)
    caption_path = figures / f"{stem}_caption.tex"
    caption_path.write_text(caption.strip() + "\n")
    shutil.copy2(caption_path, paper / "captions" / caption_path.name)
    shutil.copy2(data, paper / "source_data" / data.name)
    plt.close(fig)
    provenance.append({
        "stem": stem, "pdf": f"figures/{stem}.pdf", "png": f"figures/{stem}.png",
        "source_data": f"source_data/{stem}.tsv", "caption": f"captions/{stem}_caption.tex",
        "split": "train", "evaluation": "nested_source_group_OOF", "branding": "Delphes simulation",
    })


def paper_label(fig: Any, subtitle: str) -> None:
    fig.text(0.01, 0.99, "Delphes simulation", ha="left", va="top", fontsize=12, fontweight="bold")
    fig.text(0.99, 0.99, subtitle, ha="right", va="top", fontsize=9)


def selected_summary_row(evaluated: dict[str, Any], variant: str, metric: str) -> pd.Series:
    frame = evaluated["combined_summary"]
    label = f"{variant}_outer_selected"
    return frame[(frame.label == label) & (frame.metric == metric)].iloc[0]


def summary_source(row: pd.Series, **extra: Any) -> dict[str, Any]:
    return {
        **extra,
        "nominal_full_oof": float(row.nominal_full_oof),
        "bootstrap_mean": float(row.bootstrap_mean),
        "bootstrap_median": float(row.bootstrap_median),
        "bootstrap_standard_deviation": float(row.bootstrap_standard_deviation),
        "bootstrap_p16": float(row.bootstrap_p16), "bootstrap_p84": float(row.bootstrap_p84),
        "bootstrap_p2p5": float(row.bootstrap_p2p5), "bootstrap_p97p5": float(row.bootstrap_p97p5),
        "valid_replicas": int(row.valid_replicas), "invalid_replicas": int(row.invalid_replicas),
        "invalid_reason_counts_json": row.invalid_reason_counts_json,
        "uncertainty_central": float(row.bootstrap_median),
        "uncertainty_lower_68": float(row.bootstrap_p16),
        "uncertainty_upper_68": float(row.bootstrap_p84),
        "uncertainty_valid_replicas": int(row.valid_replicas),
        "uncertainty_support_flag": int(row.valid_replicas) > 0,
        "uncertainty_kind": "common_paired_source_member_bootstrap_p16_p84",
    }


def selected_category_summary(evaluated: dict[str, Any], variant: str, metric: str) -> pd.DataFrame:
    frame = evaluated["category_summary"]
    label = f"{variant}_outer_selected"
    return frame[(frame.label == label) & (frame.metric == metric)].sort_values("category")


def mass_sculpting_rows(inputs: dict[str, Any], categories: np.ndarray, draws: np.ndarray) -> list[dict[str, Any]]:
    projection = inputs["projection"]
    members = inputs["members"].reset_index(drop=True)
    positions = member_positions(projection, members, "categorized mass sculpting")
    background = projection.registry_training_target.to_numpy(dtype=np.int8) == 0
    weights = projection.primary_projection_physical_weight_inclusive.to_numpy(dtype=np.float64)
    variables = {
        "mbb1": (40.0, 240.0), "mbb2": (40.0, 240.0),
        "r_hh_125_120": (0.0, 160.0), "mhh": (200.0, 1400.0),
    }
    selected_categories = (0, int(categories.max()))
    rows = []
    for variable, (low, high) in variables.items():
        values = projection[variable].to_numpy(dtype=np.float64)
        edges = np.linspace(low, high, 21)
        for category in selected_categories:
            selected = background & (categories == category)
            central, _ = np.histogram(values[selected], bins=edges, weights=weights[selected])
            central = central / central.sum() if central.sum() > 0.0 else central
            replica_values = []
            for left, right in zip(edges[:-1], edges[1:]):
                in_bin = selected & (values >= left) & (values < right)
                group = group_sum(weights, in_bin, positions, len(members))
                replica_values.append(draws @ group)
            matrix = np.asarray(replica_values, dtype=np.float64).T
            totals = matrix.sum(axis=1)
            valid = totals > 0.0
            matrix[valid] /= totals[valid, None]
            matrix[~valid] = np.nan
            for index, (left, right) in enumerate(zip(edges[:-1], edges[1:])):
                column = matrix[:, index]
                finite = np.isfinite(column)
                rows.append({
                    "variable": variable, "category": category, "bin_low": left, "bin_high": right,
                    "central_value": float(central[index]),
                    "uncertainty_central": float(np.nanmedian(column)),
                    "uncertainty_lower_68": float(np.nanquantile(column, 0.16)),
                    "uncertainty_upper_68": float(np.nanquantile(column, 0.84)),
                    "uncertainty_valid_replicas": int(finite.sum()),
                    "uncertainty_support_flag": bool(finite.any()),
                    "uncertainty_kind": "source_member_bootstrap_unit_shape_p16_p84",
                })
    return rows


def make_figures(inputs: dict[str, Any], nested: dict[str, Any], evaluated: dict[str, Any],
                 paired: pd.DataFrame, draws: np.ndarray, figures: Path, paper: Path) -> list[dict[str, Any]]:
    provenance: list[dict[str, Any]] = []
    selections = nested["selections"]
    aware_values = evaluated["selected_predictions"]["mass_aware"]
    aware_count = int(max(aware_values["projection_category"].max(), aware_values["train_category"].max()) + 1)
    projection = inputs["projection"]

    # Score plane with all outer-fold selected boundaries.
    fig, ax = plt.subplots(figsize=(7.8, 6.2))
    background = projection.registry_training_target.to_numpy(dtype=np.int8) == 0
    ax.hexbin(aware_values["projection_stage1"][background], aware_values["projection_stage2"][background],
              gridsize=48, mincnt=1, bins="log", cmap="Blues")
    selection_rows = selections[(selections.variant == "mass_aware") & (selections.selection_role == "outer_selected")]
    source_rows = []
    for row in selection_rows.itertuples():
        boundaries = json.loads(row.stage2_boundaries_json)
        ax.axvline(row.stage1_threshold, color=f"C{row.outer_fold}", alpha=0.75, lw=1.2)
        for boundary in boundaries:
            ax.hlines(boundary, row.stage1_threshold, 1.0, color=f"C{row.outer_fold}", alpha=0.75, lw=1.2)
        source_rows.append({
            "outer_fold": row.outer_fold, "stage1_threshold": row.stage1_threshold,
            "stage2_boundaries_json": row.stage2_boundaries_json, "category_count": int(row.category_count),
            "uncertainty_kind": "selection_stability_not_source_bootstrap",
        })
    ax.set(xlabel="Stage-1 signal score", ylabel="Stage-2 specialized-background score", xlim=(0, 1), ylim=(0, 1))
    ax.grid(alpha=0.12); paper_label(fig, "Nested source-group OOF")
    save_figure(fig, figures, paper, "fig_c7w_01_selected_score_plane",
                "Mass-aware stage-1 versus stage-2 score plane. Colored staircases are the independently selected inner-OOF boundaries for each outer fold; no outer-held row selected a boundary.", source_rows, provenance)

    # Four score-plane densities.
    roles = [
        ("Signal", projection.registry_training_target.to_numpy(dtype=np.int8) == 1),
        ("Non-QCD background", projection.analysis_population_role.astype(str).to_numpy() == "fourb_ordinary_background"),
        ("Transferred QCD", projection.analysis_population_role.astype(str).to_numpy() == "primary_transferred_multijet_template"),
        (r"Direct $\geq4b$ QCD closure", inputs["direct"].population_kind.astype(str).to_numpy() == "hard_qcd"),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(10.5, 8.4), sharex=True, sharey=True)
    density_rows = []
    for axis, (label, mask) in zip(axes.flat, roles):
        if label.startswith("Direct"):
            x = aware_values["train_stage1"][mask]; y = aware_values["train_stage2"][mask]
        else:
            x = aware_values["projection_stage1"][mask]; y = aware_values["projection_stage2"][mask]
        counts, xedges, yedges = np.histogram2d(x, y, bins=24, range=((0, 1), (0, 1)))
        axis.pcolormesh(xedges, yedges, counts.T, shading="auto", cmap="viridis")
        axis.set_title(label); axis.set(xlabel="Stage-1 score", ylabel="Stage-2 score")
        for ix in range(len(xedges) - 1):
            for iy in range(len(yedges) - 1):
                density_rows.append({"population": label, "stage1_low": xedges[ix], "stage1_high": xedges[ix+1],
                                     "stage2_low": yedges[iy], "stage2_high": yedges[iy+1], "central_value": counts[ix, iy]})
    fig.subplots_adjust(top=0.91, hspace=0.28, wspace=0.24); paper_label(fig, "Selected mass-aware configuration")
    save_figure(fig, figures, paper, "fig_c7w_02_population_score_densities",
                r"Stage-score densities for signal, ordinary background, transferred QCD, and secondary direct-$\geq4b$ QCD closure.", density_rows, provenance)

    # Selected versus fixed-five boundary stability summary.
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.8), sharey=True)
    boundary_rows = []
    for axis, role, title in ((axes[0], "outer_selected", "Independently selected by outer fold"),
                              (axes[1], "best_supported_n5", "Fixed five-category benchmark")):
        rows = selections[(selections.variant == "mass_aware") & (selections.selection_role == role)]
        for row in rows.itertuples():
            axis.scatter([row.stage1_threshold], [row.outer_fold], color=f"C{row.outer_fold}", marker="D")
            for boundary in json.loads(row.stage2_boundaries_json):
                axis.scatter([boundary], [row.outer_fold], color=f"C{row.outer_fold}", marker="o")
            boundary_rows.append({"role": title, "outer_fold": row.outer_fold, "stage1_threshold": row.stage1_threshold,
                                  "stage2_boundaries_json": row.stage2_boundaries_json, "boundary_instability": row.boundary_instability})
        axis.set_title(title); axis.set_xlabel("Frozen score boundary"); axis.grid(axis="x", alpha=0.18)
    axes[0].set_ylabel("Outer fold"); fig.subplots_adjust(top=0.87, bottom=0.18); paper_label(fig, "Inner-OOF selection stability")
    save_figure(fig, figures, paper, "fig_c7w_03_selected_and_benchmark_boundaries",
                "Selected and fixed-five staircase boundary stability across outer folds. Diamonds denote stage-1 thresholds and circles denote stage-2 boundaries.", boundary_rows, provenance)

    # Category yields and composition.  Components remain distinct while the
    # total-background bar carries the paired source-bootstrap interval.
    signal_summary = selected_category_summary(evaluated, "mass_aware", "signal_yield")
    ordinary_summary = selected_category_summary(evaluated, "mass_aware", "ordinary_background_yield")
    qcd_summary = selected_category_summary(evaluated, "mass_aware", "transferred_qcd_yield")
    background_summary = selected_category_summary(evaluated, "mass_aware", "background_yield")
    yield_rows = []
    for metric, frame in (("signal_yield", signal_summary), ("ordinary_background_yield", ordinary_summary),
                          ("transferred_qcd_yield", qcd_summary), ("background_yield", background_summary)):
        yield_rows.extend(summary_source(row, category=int(row.category), metric=metric, variant="mass_aware")
                          for _, row in frame.iterrows())
    x = np.arange(len(signal_summary))
    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.9))
    signal_median = signal_summary.bootstrap_median.to_numpy(float)
    axes[0].errorbar(x, signal_median,
                     yerr=np.vstack([signal_median-signal_summary.bootstrap_p16.to_numpy(float),
                                     signal_summary.bootstrap_p84.to_numpy(float)-signal_median]),
                     fmt="o", capsize=4, color="#0072B2")
    axes[0].scatter(x, signal_summary.nominal_full_oof, marker="D", facecolors="none", edgecolors="#D55E00")
    ordinary_median = ordinary_summary.bootstrap_median.to_numpy(float)
    qcd_median = qcd_summary.bootstrap_median.to_numpy(float)
    total_median = background_summary.bootstrap_median.to_numpy(float)
    axes[1].bar(x, ordinary_median, color="#56B4E9", label="Non-QCD background")
    axes[1].bar(x, qcd_median, bottom=ordinary_median, color="#E69F00", label="Transferred QCD")
    axes[1].errorbar(x, total_median,
                     yerr=np.vstack([total_median-background_summary.bootstrap_p16.to_numpy(float),
                                     background_summary.bootstrap_p84.to_numpy(float)-total_median]),
                     fmt="none", ecolor="black", capsize=4, lw=1.2, label="Total 68% interval")
    for axis, ylabel in zip(axes, ("Signal yield", "Background yield")):
        axis.set_xticks(x); axis.set_xticklabels([str(int(value)) for value in signal_summary.category])
        axis.set_xlabel("Category"); axis.set_ylabel(ylabel); axis.grid(axis="y", alpha=0.18)
    axes[1].legend(fontsize=8)
    fig.subplots_adjust(top=0.88, bottom=0.18, wspace=0.3); paper_label(fig, f"{len(draws)} common source-member replicas")
    save_figure(fig, figures, paper, "fig_c7w_04_category_yields",
                "Selected-category signal yields and background composition. Asymmetric bars are source-member bootstrap 16th--84th percentile intervals; the background stack starts at zero and keeps transferred QCD explicit.",
                yield_rows, provenance)

    # Remaining category uncertainty plots.
    metric_specs = [
        ("signal_over_background", "fig_c7w_05_category_signal_over_background", r"Selected category $S/B$", r"$S/B$"),
        ("background_neff", "fig_c7w_06_category_background_neff", r"Selected category background $N_{\rm eff}$", r"Background $N_{\rm eff}$"),
        ("nominal_asimov_ZA_contribution", "fig_c7w_07_category_nominal_za", r"Category nominal $Z_A$ contribution", r"Nominal $Z_A$"),
        ("systematic_aware_ZA_standalone_contribution", "fig_c7w_08_category_systematic_za", r"Category standalone systematics-aware $Z_A$", r"Systematics-aware $Z_A$"),
    ]
    for metric, stem, caption_title, ylabel in metric_specs:
        summary = selected_category_summary(evaluated, "mass_aware", metric)
        rows = [summary_source(row, category=int(row.category), metric=metric, variant="mass_aware") for _, row in summary.iterrows()]
        x = np.arange(len(summary)); med = summary.bootstrap_median.to_numpy(float)
        lo = summary.bootstrap_p16.to_numpy(float); hi = summary.bootstrap_p84.to_numpy(float)
        fig, ax = plt.subplots(figsize=(7.4, 4.9))
        ax.errorbar(x, med, yerr=np.vstack([med-lo, hi-med]), fmt="o", capsize=4, color=COLORS["mass_aware"])
        ax.scatter(x, summary.nominal_full_oof, marker="D", facecolors="none", edgecolors="#D55E00")
        ax.set_xticks(x); ax.set_xticklabels([str(int(value)) for value in summary.category])
        ax.set_xlabel("Category"); ax.set_ylabel(ylabel); ax.grid(axis="y", alpha=0.18); paper_label(fig, f"{len(draws)} common source-member replicas")
        save_figure(fig, figures, paper, stem,
                    f"{caption_title}. Circles and asymmetric bars are bootstrap medians and 16th--84th percentile intervals; open diamonds are nominal full-OOF values.", rows, provenance)

    # Inclusive versus categorized comparison.
    c7v_summary = pd.read_csv(C7V / "bootstrap_metric_summary.tsv", sep="\t")
    comparison_rows = []
    configs = [("inclusive_bdt", "Inclusive BDT", None)] + [
        (variant, DISPLAY[variant], variant) for variant in VARIANTS
    ]
    fig, axes = plt.subplots(1, 2, figsize=(10.8, 4.8))
    for axis, (metric, inclusive_metric, ylabel) in zip(axes, [
        ("combined_nominal_asimov_ZA", "nominal_asimov_ZA", r"Nominal $Z_A$"),
        ("combined_systematic_aware_asimov_ZA", "systematic_aware_asimov_ZA", r"Systematics-aware $Z_A$"),
    ]):
        med=[];lo=[];hi=[];nom=[];labels=[]
        for key, label, variant in configs:
            if key == "inclusive_bdt":
                row = c7v_summary[(c7v_summary.model == "bdt") & (c7v_summary.metric == inclusive_metric)].iloc[0]
            else:
                row = selected_summary_row(evaluated, variant, metric)
            med.append(float(row.bootstrap_median));lo.append(float(row.bootstrap_p16));hi.append(float(row.bootstrap_p84));nom.append(float(row.nominal_full_oof));labels.append(label)
            comparison_rows.append(summary_source(row, configuration=key, metric=metric))
        x=np.arange(len(labels)); med=np.asarray(med);lo=np.asarray(lo);hi=np.asarray(hi)
        axis.errorbar(x,med,yerr=np.vstack([med-lo,hi-med]),fmt="o",capsize=4,color="#0072B2")
        axis.scatter(x,nom,marker="D",facecolors="none",edgecolors="#D55E00")
        axis.set_xticks(x);axis.set_xticklabels(labels,rotation=14,ha="right");axis.set_ylabel(ylabel);axis.grid(axis="y",alpha=.18)
    fig.subplots_adjust(top=.88,bottom=.24,wspace=.34);paper_label(fig,"Paired train-only comparison")
    save_figure(fig,figures,paper,"fig_c7w_09_inclusive_categorized_comparison",
                "Inclusive BDT and selected categorized configurations on the common source-member registry. No improvement is claimed when the paired difference interval includes zero.",comparison_rows,provenance)

    # The comparison itself is paired replica by replica; show its interval,
    # not only the marginal intervals above.
    paired_selected = paired[
        paired.comparison.isin([f"{variant}_outer_selected_minus_inclusive_bdt" for variant in VARIANTS]) &
        paired.metric.isin(["combined_nominal_asimov_ZA", "combined_systematic_aware_asimov_ZA"])
    ]
    paired_rows = []
    fig, axes = plt.subplots(1, 2, figsize=(10.8, 4.7))
    for axis, metric, ylabel in zip(
        axes,
        ("combined_nominal_asimov_ZA", "combined_systematic_aware_asimov_ZA"),
        (r"Paired $\Delta$ nominal $Z_A$", r"Paired $\Delta$ systematics-aware $Z_A$"),
    ):
        sub = paired_selected[paired_selected.metric == metric].set_index("model").loc[
            [f"{variant}_outer_selected" for variant in VARIANTS]
        ].reset_index()
        med = sub.bootstrap_median.to_numpy(float)
        lo = sub.bootstrap_p16.to_numpy(float); hi = sub.bootstrap_p84.to_numpy(float)
        x = np.arange(len(VARIANTS))
        axis.errorbar(x, med, yerr=np.vstack([med-lo, hi-med]), fmt="o", capsize=4, color="#0072B2")
        axis.scatter(x, sub.nominal_difference, marker="D", facecolors="none", edgecolors="#D55E00")
        axis.axhline(0.0, color="black", lw=1.0, ls="--")
        axis.set_xticks(x); axis.set_xticklabels([DISPLAY[variant] for variant in VARIANTS], rotation=14, ha="right")
        axis.set_ylabel(ylabel); axis.grid(axis="y", alpha=0.18)
        for row in sub.itertuples():
            paired_rows.append({
                "comparison": row.comparison, "model": row.model, "reference_model": row.reference_model,
                "metric": row.metric, "nominal_difference": row.nominal_difference,
                "bootstrap_mean": row.bootstrap_mean, "bootstrap_median": row.bootstrap_median,
                "bootstrap_standard_deviation": row.bootstrap_standard_deviation,
                "bootstrap_p16": row.bootstrap_p16, "bootstrap_p84": row.bootstrap_p84,
                "bootstrap_p2p5": row.bootstrap_p2p5, "bootstrap_p97p5": row.bootstrap_p97p5,
                "valid_replicas": row.valid_replicas, "invalid_replicas": row.invalid_replicas,
                "invalid_reason_counts_json": row.invalid_reason_counts_json,
                "fraction_valid_difference_greater_than_zero": row.fraction_valid_difference_greater_than_zero,
                "interval_includes_zero_68": row.interval_includes_zero_68,
                "uncertainty_central": row.bootstrap_median,
                "uncertainty_lower_68": row.bootstrap_p16,
                "uncertainty_upper_68": row.bootstrap_p84,
                "uncertainty_valid_replicas": row.valid_replicas,
                "uncertainty_support_flag": row.valid_replicas > 0,
                "uncertainty_kind": "paired_common_source_member_bootstrap_difference_p16_p84",
            })
    fig.subplots_adjust(top=.88,bottom=.25,wspace=.36); paper_label(fig,"Categorized minus inclusive BDT")
    save_figure(fig, figures, paper, "fig_c7w_09b_paired_differences",
                "Paired selected categorized-BDT minus inclusive-BDT differences. Circles and asymmetric bars are the paired bootstrap median and 68% interval; open diamonds are nominal full-OOF differences. An interval crossing zero is not called an improvement.",
                paired_rows, provenance)

    # Category-count scan.
    fig, axes = plt.subplots(1,2,figsize=(10.8,4.7))
    scan_rows=[]
    for axis,metric,ylabel in zip(axes,["combined_nominal_asimov_ZA","combined_systematic_aware_asimov_ZA"],[r"Nominal joint $Z_A$",r"Systematics-aware joint $Z_A$"]):
        for variant in VARIANTS:
            labels = [configuration_label(variant, count) for count in CATEGORY_COUNTS]
            sub=evaluated["combined_summary"][(evaluated["combined_summary"].label.isin(labels))&(evaluated["combined_summary"].metric==metric)].sort_values("category_count")
            x=sub.category_count.to_numpy();med=sub.bootstrap_median.to_numpy(float);lo=sub.bootstrap_p16.to_numpy(float);hi=sub.bootstrap_p84.to_numpy(float)
            axis.errorbar(x,med,yerr=np.vstack([med-lo,hi-med]),marker="o",capsize=3,color=COLORS[variant],label=DISPLAY[variant])
            scan_rows.extend(summary_source(row,variant=variant,category_count=int(row.category_count),metric=metric) for _,row in sub.iterrows())
        axis.set(xlabel="Total categories",ylabel=ylabel,xticks=CATEGORY_COUNTS);axis.grid(alpha=.18);axis.legend(fontsize=8)
    fig.subplots_adjust(top=.88,bottom=.18,wspace=.32);paper_label(fig,"Nested 2--6 category scan")
    save_figure(fig,figures,paper,"fig_c7w_10_category_count_scan",
                "Nested category-count scan with common source-member bootstrap 68% intervals. Each point uses only boundaries selected inside its outer-training partition.",scan_rows,provenance)

    # Support diagnostics with resampled row/source counts.
    support_metrics=[("background_rows","Background rows"),("transferred_rows","Transferred-QCD rows"),("direct_qcd_rows",r"Direct $\geq4b$ QCD rows")]
    fig,axes=plt.subplots(1,3,figsize=(13.2,4.5))
    support_rows=[]
    for axis,(metric,ylabel) in zip(axes,support_metrics):
        summary=selected_category_summary(evaluated,"mass_aware",metric)
        x=np.arange(len(summary));med=summary.bootstrap_median.to_numpy(float);lo=summary.bootstrap_p16.to_numpy(float);hi=summary.bootstrap_p84.to_numpy(float)
        axis.errorbar(x,med,yerr=np.vstack([med-lo,hi-med]),fmt="o",capsize=3)
        axis.set_xticks(x);axis.set_xticklabels([str(int(value)) for value in summary.category]);axis.set(xlabel="Category",ylabel=ylabel);axis.grid(axis="y",alpha=.18)
        support_rows.extend(summary_source(row,category=int(row.category),metric=metric) for _,row in summary.iterrows())
    fig.subplots_adjust(top=.86,bottom=.18,wspace=.4);paper_label(fig,"Category support diagnostics")
    save_figure(fig,figures,paper,"fig_c7w_11_category_support",
                r"Selected-category support. Bars show source-member bootstrap 68% intervals; sparse direct-$\geq4b$ QCD remains secondary closure only.",support_rows,provenance)

    # Mass-sculpting control with source-bootstrap bands.
    sculpt_rows=[]
    for variant in VARIANTS:
        values=evaluated["selected_predictions"][variant]
        local=mass_sculpting_rows(inputs,values["projection_category"],draws)
        for row in local:
            row["variant"]=variant
        sculpt_rows.extend(local)
    sculpt=pd.DataFrame(sculpt_rows)
    fig,axes=plt.subplots(2,2,figsize=(11.0,8.2))
    variable_labels={"mbb1":r"$m_{bb,1}$ [GeV]","mbb2":r"$m_{bb,2}$ [GeV]","r_hh_125_120":r"$R_{HH}(125,120)$ [GeV]","mhh":r"$m_{HH}$ [GeV]"}
    for axis,(variable,xlabel) in zip(axes.flat,variable_labels.items()):
        plot_specs = [
            ("mass_aware", 0, "#999999", "--", "Mass-aware catch-all"),
            ("mass_aware", int(evaluated["selected_predictions"]["mass_aware"]["projection_category"].max()), COLORS["mass_aware"], "-", "Mass-aware highest rank"),
            ("mass_plane_blind", int(evaluated["selected_predictions"]["mass_plane_blind"]["projection_category"].max()), COLORS["mass_plane_blind"], "-", "Mass-plane-blind highest rank"),
        ]
        for variant,category,color,linestyle,label in plot_specs:
            sub=sculpt[(sculpt.variable==variable)&(sculpt.variant==variant)&(sculpt.category==category)]
            centers=.5*(sub.bin_low.to_numpy()+sub.bin_high.to_numpy())
            axis.step(centers,sub.uncertainty_central.to_numpy(float),where="mid",color=color,linestyle=linestyle,label=label)
            axis.fill_between(centers,sub.uncertainty_lower_68.to_numpy(float),sub.uncertainty_upper_68.to_numpy(float),step="mid",color=color,alpha=.18)
        axis.set(xlabel=xlabel,ylabel="Unit-normalized background");axis.grid(alpha=.15)
    axes[0,0].legend();fig.subplots_adjust(top=.91,hspace=.34,wspace=.27);paper_label(fig,"Mass-sculpting control")
    save_figure(fig,figures,paper,"fig_c7w_12_mass_sculpting",
                r"Mass-sculpting comparison between the catch-all, the highest-rank mass-aware category, and the highest-rank mass-plane-blind control. Bands are source-member bootstrap 68% intervals.",sculpt_rows,provenance)
    return provenance


def latex_escape(value: Any) -> str:
    text = str(value)
    if "$" in text or "\\" in text:
        return text
    for old, new in (("_", r"\_"), ("%", r"\%"), ("&", r"\&"), ("#", r"\#")):
        text = text.replace(old, new)
    return text


def write_latex_table(path: Path, caption: str, label: str, headers: list[str], rows: list[list[Any]],
                      alignment: str) -> None:
    require(len(headers) == len(alignment), f"table alignment drift: {path}")
    lines = [r"\begin{table}[tb]", r"  \centering", r"  \footnotesize", f"  \\caption{{{caption}}}",
             f"  \\label{{{label}}}", f"  \\begin{{tabular}}{{{alignment}}}", r"    \toprule",
             "    " + " & ".join(headers) + r" \\", r"    \midrule"]
    lines.extend("    " + " & ".join(latex_escape(value) for value in row) + r" \\" for row in rows)
    lines += [r"    \bottomrule", r"  \end{tabular}", r"\end{table}"]
    path.write_text("\n".join(lines) + "\n")


def format_asymmetric(row: pd.Series, *, scientific: bool = False, digits: int = 4) -> str:
    median = float(row.bootstrap_median)
    if not math.isfinite(median) or int(row.get("valid_replicas", 1)) == 0:
        return r"\textit{Unsupported}"
    up = float(row.bootstrap_p84 - median)
    down = float(median - row.bootstrap_p16)
    magnitude = max(abs(median), abs(up), abs(down))
    if scientific and magnitude > 0.0:
        exponent = int(math.floor(math.log10(magnitude)))
        scale = 10.0 ** exponent
        return rf"$({median/scale:.2f}^{{+{up/scale:.2f}}}_{{-{down/scale:.2f}}})\times10^{{{exponent}}}$"
    return rf"${median:.{digits}f}^{{+{up:.{digits}f}}}_{{-{down:.{digits}f}}}$"


def make_paper_tables(paper: Path, evaluated: dict[str, Any], paired: pd.DataFrame,
                      selections: pd.DataFrame, support_sensitivity: pd.DataFrame) -> list[dict[str, Any]]:
    tables = paper / "tables"
    manifest = []
    combined = evaluated["combined_summary"]
    selected_rows = []
    for variant in VARIANTS:
        counts = evaluated["selected_counts_by_fold"][variant]
        count_display = ", ".join(f"{fold}:{count}" for fold, count in sorted(counts.items()))
        metric_rows = {metric: selected_summary_row(evaluated, variant, metric) for metric in (
            "combined_nominal_asimov_ZA", "combined_systematic_aware_asimov_ZA", "signal_over_background",
            "background_neff", "transferred_qcd_fraction",
        )}
        selected_rows.append([
            "Mass-aware" if variant == "mass_aware" else "Mass-plane-blind", count_display,
            format_asymmetric(metric_rows["combined_nominal_asimov_ZA"]),
            format_asymmetric(metric_rows["combined_systematic_aware_asimov_ZA"], scientific=True),
            format_asymmetric(metric_rows["signal_over_background"], scientific=True),
            format_asymmetric(metric_rows["background_neff"], digits=1),
            format_asymmetric(metric_rows["transferred_qcd_fraction"], digits=3),
        ])
    write_latex_table(
        tables / "tab_c7w_01_selected_configurations.tex",
        "Selected train-only nested categorized-BDT configurations with common source-member bootstrap 68\\% intervals.",
        "tab:c7w-selected", ["Variant", r"$N$ by fold", r"Nom. $Z_A$", r"Syst. $Z_A$", r"$S/B$", r"Bkg. $N_{\rm eff}$", "QCD frac."],
        selected_rows, "llrrrrr",
    )
    manifest.append({"table_id": "c7w_tab01", "path": "tables/tab_c7w_01_selected_configurations.tex", "placement": "main_text"})

    category_rows = []
    category_summary = evaluated["category_summary"]
    aware_label = "mass_aware_outer_selected"
    aware_count = int(category_summary[category_summary.label == aware_label].category.max() + 1)
    for category in range(aware_count):
        lookup = {
            metric: category_summary[(category_summary.label == aware_label) &
                                     (category_summary.category == category) &
                                     (category_summary.metric == metric)].iloc[0]
            for metric in ("signal_yield", "background_yield", "signal_over_background", "background_neff",
                           "nominal_asimov_ZA_contribution", "systematic_aware_ZA_standalone_contribution")
        }
        category_rows.append([
            category, format_asymmetric(lookup["signal_yield"], scientific=True),
            format_asymmetric(lookup["background_yield"], scientific=True),
            format_asymmetric(lookup["signal_over_background"], scientific=True),
            format_asymmetric(lookup["background_neff"], digits=1),
            format_asymmetric(lookup["nominal_asimov_ZA_contribution"]),
            format_asymmetric(lookup["systematic_aware_ZA_standalone_contribution"], scientific=True),
        ])
    write_latex_table(
        tables / "tab_c7w_02_selected_categories.tex",
        "Mass-aware selected-category yields and sensitivity diagnostics with asymmetric 68\\% source-member intervals.",
        "tab:c7w-categories", ["Category", "Signal yield", "Background yield", r"$S/B$", r"Bkg. $N_{\rm eff}$", r"Nominal $Z_A$", r"Standalone syst. $Z_A$"],
        category_rows, "lrrrrrr",
    )
    manifest.append({"table_id": "c7w_tab02", "path": "tables/tab_c7w_02_selected_categories.tex", "placement": "main_text"})

    count_rows = []
    for variant in VARIANTS:
        for count in CATEGORY_COUNTS:
            label = configuration_label(variant, count)
            nominal = combined[(combined.label == label) &
                               (combined.metric == "combined_nominal_asimov_ZA")].iloc[0]
            systematic = combined[(combined.label == label) &
                                  (combined.metric == "combined_systematic_aware_asimov_ZA")].iloc[0]
            count_rows.append([DISPLAY[variant], count, format_asymmetric(nominal),
                               format_asymmetric(systematic, scientific=True),
                               "Benchmark" if count == 5 else "Scan"])
    write_latex_table(
        tables / "tab_c7w_03_category_count_scan.tex", "Nested two-through-six category scan.", "tab:c7w-count-scan",
        ["Variant", "Categories", r"Nominal $Z_A$", r"Systematics-aware $Z_A$", "Role"], count_rows, "lrrrl",
    )
    manifest.append({"table_id": "c7w_tab03", "path": "tables/tab_c7w_03_category_count_scan.tex", "placement": "appendix"})

    paired_rows = []
    metric_labels = {
        "combined_nominal_asimov_ZA": r"Nominal $Z_A$", "combined_systematic_aware_asimov_ZA": r"Systematics-aware $Z_A$",
        "signal_over_background": r"$S/B$", "background_neff": r"Background $N_{\rm eff}$",
    }
    primary = paired[paired.comparison.str.contains("outer_selected") |
                     (paired.comparison == "mass_plane_blind_selected_minus_mass_aware_selected")]
    for row in primary.itertuples():
        scientific = row.metric in {"combined_systematic_aware_asimov_ZA", "signal_over_background"}
        fake = pd.Series({"bootstrap_median": row.bootstrap_median, "bootstrap_p16": row.bootstrap_p16, "bootstrap_p84": row.bootstrap_p84})
        model_short = "Mass-aware" if str(row.model).startswith("mass_aware") else "Mass-plane-blind"
        reference_short = "Inclusive BDT" if row.reference_model == "inclusive_bdt" else "Mass-aware"
        paired_rows.append([f"{model_short} minus {reference_short}", metric_labels[row.metric],
                            format_asymmetric(fake, scientific=scientific, digits=3),
                            "Yes" if row.interval_includes_zero_68 else "No", f"{row.fraction_valid_difference_greater_than_zero:.3f}"])
    write_latex_table(
        tables / "tab_c7w_04_paired_differences.tex",
        "Paired model differences. No improvement is claimed when the 68\\% interval includes zero.",
        "tab:c7w-paired", ["Comparison", "Metric", "Median difference (68\\%)", "Includes zero", r"$P(\Delta>0)$"],
        paired_rows, "llllr",
    )
    manifest.append({"table_id": "c7w_tab04", "path": "tables/tab_c7w_04_paired_differences.tex", "placement": "main_text"})

    stability_rows = []
    for row in selections[selections.selection_role == "outer_selected"].itertuples():
        stability_rows.append(["Mass-aware" if row.variant == "mass_aware" else "Mass-plane-blind",
                               row.outer_fold, row.category_count,
                               f"{row.stage1_target_signal_efficiency:.2f}", f"{row.stage1_threshold:.4f}",
                               f"{row.boundary_instability:.4f}"])
    write_latex_table(
        tables / "tab_c7w_05_selection_stability.tex", "Outer-fold selection-stability diagnostic; no outer-held row enters a choice.",
        "tab:c7w-stability", ["Variant", "Outer fold", "Categories", r"Stage-1 $\epsilon_S$", "Threshold", "Instability"],
        stability_rows, "lrrrrr",
    )
    manifest.append({"table_id": "c7w_tab05", "path": "tables/tab_c7w_05_selection_stability.tex", "placement": "appendix"})

    sensitivity_rows = []
    selected_support = support_sensitivity[support_sensitivity.configuration.str.endswith("_outer_selected")]
    contract_display = {"nominal": "Nominal", "strict_a": "Strict A", "strict_b": "Strict B"}
    for row in selected_support.itertuples():
        sensitivity_rows.append([DISPLAY[row.variant], contract_display[row.contract_name],
                                 f"{row.categories_passing}/{row.categories_total}",
                                 f"{row.minimum_support_margin:.2f}", "Pass" if row.support_pass else "Fail"])
    write_latex_table(
        tables / "tab_c7w_06_support_sensitivity.tex", "Predefined non-selecting support-contract sensitivity study.",
        "tab:c7w-support", ["Variant", "Contract", "Categories passing", "Minimum margin", "Status"],
        sensitivity_rows, "lllrl",
    )
    manifest.append({"table_id": "c7w_tab06", "path": "tables/tab_c7w_06_support_sensitivity.tex", "placement": "appendix"})
    write_tsv(paper / "paper_table_manifest.tsv", manifest)
    return manifest


def bootstrap_validity_summary(evaluated: dict[str, Any]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for label, group in evaluated["combined_replicas"].groupby("label", sort=False):
        valid = np.isfinite(group.combined_systematic_aware_asimov_ZA.to_numpy(float))
        reasons = Counter(group.loc[~valid, "failure_reason"].fillna("nonfinite metric").astype(str))
        category_group = evaluated["category_replicas"][evaluated["category_replicas"].label == label]
        selection_support = category_group.groupby("replica_id", sort=True).support_pass.all()
        selection_reasons = Counter(
            reason for reason in category_group.loc[~category_group.support_pass, "support_failure_reason"].astype(str)
            if reason
        )
        rows.append({
            "scope": "combined", "configuration": label, "category": "all",
            "replicas": len(group), "valid_replicas": int(valid.sum()),
            "invalid_replicas": int((~valid).sum()),
            "valid_fraction": float(valid.mean()),
            "selection_support_pass_fraction": float(selection_support.mean()),
            "zero_direct_qcd_support_fraction": float((group.shared_qcd_relative_nonclosure.isna()).mean()),
            "invalid_reason_counts_json": json.dumps(dict(sorted(reasons.items())), separators=(",", ":")),
            "selection_support_failure_counts_json": json.dumps(dict(sorted(selection_reasons.items())), separators=(",", ":")),
        })
    for (label, category), group in evaluated["category_replicas"].groupby(["label", "category"], sort=False):
        valid = np.isfinite(group.systematic_aware_ZA_standalone_contribution.to_numpy(float))
        reasons = Counter(group.loc[~valid, "support_failure_reason"].fillna("nonfinite metric").astype(str))
        rows.append({
            "scope": "category", "configuration": label, "category": int(category),
            "replicas": len(group), "valid_replicas": int(valid.sum()),
            "invalid_replicas": int((~valid).sum()), "valid_fraction": float(valid.mean()),
            "selection_support_pass_fraction": float(group.support_pass.mean()),
            "zero_direct_qcd_support_fraction": float((group.direct_qcd_rows == 0).mean()),
            "invalid_reason_counts_json": json.dumps(dict(sorted(reasons.items())), separators=(",", ":")),
            "selection_support_failure_counts_json": json.dumps(
                dict(sorted(Counter(reason for reason in group.loc[~group.support_pass, "support_failure_reason"].astype(str)
                                    if reason).items())), separators=(",", ":")
            ),
        })
    return pd.DataFrame(rows)


def validate_uncertainty_contract(evaluated: dict[str, Any], paired: pd.DataFrame,
                                  paired_replicas: pd.DataFrame, figure_manifest: list[dict[str, Any]],
                                  figures: Path, replica_count: int, *, smoke: bool) -> None:
    interval_columns = {
        "nominal_full_oof", "bootstrap_mean", "bootstrap_median", "bootstrap_standard_deviation",
        "bootstrap_p16", "bootstrap_p84", "bootstrap_p2p5", "bootstrap_p97p5",
        "valid_replicas", "invalid_replicas", "invalid_reason_counts_json",
    }
    for label, frame in (("combined", evaluated["combined_summary"]), ("category", evaluated["category_summary"])):
        require(interval_columns.issubset(frame.columns), f"{label} uncertainty summary schema incomplete")
        supported = frame.valid_replicas > 0
        require(bool((frame.loc[supported, "bootstrap_p16"] <= frame.loc[supported, "bootstrap_median"]).all() and
                     (frame.loc[supported, "bootstrap_median"] <= frame.loc[supported, "bootstrap_p84"]).all()),
                f"{label} asymmetric interval ordering failure")
    primary_labels = {f"{variant}_outer_selected" for variant in VARIANTS}
    required_primary_metrics = {
        "combined_nominal_asimov_ZA", "combined_systematic_aware_asimov_ZA", "signal_yield",
        "background_yield", "signal_over_background", "signal_over_sqrt_background", "background_neff",
        "signal_neff", "transferred_qcd_fraction", "direct_qcd_closure_ratio",
        "shared_qcd_relative_nonclosure", "minimum_support_margin",
    }
    for label in primary_labels:
        primary = evaluated["combined_summary"].loc[evaluated["combined_summary"].label == label]
        metrics = set(primary.metric)
        require(required_primary_metrics.issubset(metrics), f"primary performance uncertainty gap: {label}")
        if not smoke:
            require(bool((primary.loc[primary.metric.isin(required_primary_metrics), "valid_replicas"] > 0).all()),
                    f"primary performance metric has no valid source-bootstrap replicas: {label}")
    paired_columns = {
        "nominal_difference", "bootstrap_median", "bootstrap_p16", "bootstrap_p84", "bootstrap_p2p5",
        "bootstrap_p97p5", "fraction_valid_difference_greater_than_zero", "interval_includes_zero_68",
    }
    require(paired_columns.issubset(paired.columns), "paired-difference summary schema incomplete")
    if not smoke:
        primary_paired = paired[paired.comparison.str.contains("outer_selected")]
        require(bool((primary_paired.valid_replicas > 0).all()), "primary paired comparison without valid replicas")
    expected_metrics = {"combined_nominal_asimov_ZA", "combined_systematic_aware_asimov_ZA",
                        "signal_over_background", "background_neff"}
    for comparison, group in paired.groupby("comparison"):
        require(set(group.metric) == expected_metrics, f"paired metric coverage drift: {comparison}")
    for (comparison, metric), group in paired_replicas.groupby(["comparison", "metric"]):
        require(group.replica_id.tolist() == list(range(replica_count)),
                f"paired replica alignment failure: {comparison}/{metric}")
    source_columns = {"uncertainty_central", "uncertainty_lower_68", "uncertainty_upper_68",
                      "uncertainty_valid_replicas", "uncertainty_support_flag", "uncertainty_kind"}
    for item in figure_manifest:
        source = figures / Path(item["source_data"]).name
        frame = pd.read_csv(source, sep="\t")
        require(source_columns.issubset(frame.columns), f"figure uncertainty-source schema incomplete: {source.name}")


def paper_model_name(raw: str) -> str:
    mapping = {
        "mass_aware_outer_selected": "Mass-aware selected categorized BDT",
        "mass_plane_blind_outer_selected": "Mass-plane-blind selected categorized BDT",
        "mass_plane_blind_selected": "Mass-plane-blind selected categorized BDT",
        "mass_aware_selected": "Mass-aware selected categorized BDT",
        "inclusive_bdt": "Inclusive BDT",
    }
    return mapping.get(raw, raw.replace("_", " ").title())


def write_paper_documentation(paper: Path, evaluated: dict[str, Any], paired: pd.DataFrame,
                              figure_manifest: list[dict[str, Any]], table_manifest: list[dict[str, Any]],
                              registry_sha: str, replica_count: int) -> None:
    selected_text = []
    for variant in VARIANTS:
        nominal = selected_summary_row(evaluated, variant, "combined_nominal_asimov_ZA")
        systematic = selected_summary_row(evaluated, variant, "combined_systematic_aware_asimov_ZA")
        sb = selected_summary_row(evaluated, variant, "signal_over_background")
        counts = ", ".join(f"fold {fold}: {count}" for fold, count in sorted(evaluated["selected_counts_by_fold"][variant].items()))
        selected_text.append(
            f"{DISPLAY[variant]} selected category counts ({counts}). Its joint nominal sensitivity is "
            f"{format_asymmetric(nominal)} and its systematics-aware sensitivity is "
            f"{format_asymmetric(systematic, scientific=True)}; the selected $S/B$ is "
            f"{format_asymmetric(sb, scientific=True)}."
        )
    primary_paired = paired[
        paired.comparison.isin([f"{variant}_outer_selected_minus_inclusive_bdt" for variant in VARIANTS]) &
        (paired.metric == "combined_nominal_asimov_ZA")
    ]
    paired_text = []
    for row in primary_paired.itertuples():
        interval = pd.Series({"bootstrap_median": row.bootstrap_median,
                              "bootstrap_p16": row.bootstrap_p16, "bootstrap_p84": row.bootstrap_p84})
        conclusion = "includes zero, so no improvement is claimed" if row.interval_includes_zero_68 else "excludes zero at the 68\\% interval"
        paired_text.append(
            f"For {paper_model_name(row.model)} minus inclusive BDT, the paired nominal-$Z_A$ difference is "
            f"{format_asymmetric(interval)} and {conclusion}."
        )
    section = (
        "\\section{Train-only nested categorized BDT}\n"
        "This HIG-24-015-inspired study is based only on Delphes simulation and frozen train-source-group folds. "
        "No validation, test, observed-data, or Run-2 expected-sensitivity claim is made. The primary multijet "
        "projection is exactly $3b\\rightarrow{\\geq4b}$; direct $\\geq4b$ QCD is secondary closure only.\n\n"
        + "\n\n".join(selected_text + paired_text) + "\n\n"
        "All resampleable values use the same common source-member bootstrap registry. Primary uncertainties are "
        "asymmetric 16th--84th percentile intervals. Selection stability is reported separately from evaluation "
        "uncertainty, and a single correlated multijet nuisance is profiled jointly across exclusive categories.\n"
    )
    (paper / "sections" / "categorized_bdt.tex").write_text(section)
    (paper / "uncertainty_table.tex").write_text(
        "".join((paper / "tables" / name).read_text() for name in (
            "tab_c7w_01_selected_configurations.tex", "tab_c7w_02_selected_categories.tex",
            "tab_c7w_03_category_count_scan.tex",
        ))
    )
    (paper / "paired_improvement_table.tex").write_text(
        (paper / "tables" / "tab_c7w_04_paired_differences.tex").read_text()
    )
    (paper / "README.md").write_text(
        "# c7w train-only nested categorized BDT\n\n"
        "This package contains a Delphes-simulation, train-source-group nested-OOF categorized-BDT study. "
        "It opens no validation/test payload and no observed data. Exactly-$3b$ to $\\geq4b$ transfer is the "
        "primary QCD prediction; direct-$\\geq4b$ QCD is secondary closure only.\n\n"
        f"All model-performance values use {replica_count} paired source-member replicas from the common registry "
        f"with SHA-256 `{registry_sha}`. Figures contain asymmetric 68% intervals wherever applicable; "
        "selection-stability plots are explicitly separate diagnostics.\n"
    )
    (paper / "RESULTS_INDEX.md").write_text(
        "# c7w results index\n\n"
        f"- Figures ({len(figure_manifest)}): `figures/`, exact data in `source_data/`, captions in `captions/`.\n"
        f"- Tables ({len(table_manifest)}): `tables/`.\n"
        "- Compact uncertainty summaries: `bootstrap_metric_summary.tsv`, `category_bootstrap_summary.tsv`.\n"
        "- Paired differences: `paired_model_differences.tsv` and `paired_improvement_table.tex`.\n"
        "- Selection stability: `operating_point_stability.tsv` and `category_boundary_stability.tsv`.\n"
    )
    compile_lines = [
        r"\documentclass[11pt]{article}", r"\usepackage[margin=0.65in]{geometry}", r"\usepackage{booktabs}",
        r"\usepackage{graphicx}", r"\usepackage{amsmath}", r"\begin{document}",
        r"\input{sections/categorized_bdt.tex}",
    ]
    compile_lines.extend(rf"\input{{{row['path']}}}" for row in table_manifest)
    compile_lines.append(r"\end{document}")
    (paper / "compile_fragments.tex").write_text("\n".join(compile_lines) + "\n")


def numerical_claim_rows(evaluated: dict[str, Any], registry_sha: str) -> list[dict[str, Any]]:
    rows = []
    claim_id = 1
    units = {
        "combined_nominal_asimov_ZA": "sigma", "combined_systematic_aware_asimov_ZA": "sigma",
        "signal_over_background": "dimensionless", "background_neff": "effective events",
        "transferred_qcd_fraction": "fraction",
    }
    for variant in VARIANTS:
        for metric, unit in units.items():
            row = selected_summary_row(evaluated, variant, metric)
            rows.append({
                "claim_id": f"C7W-{claim_id:03d}", "model": DISPLAY[variant], "metric": metric,
                "nominal_full_oof": row.nominal_full_oof, "bootstrap_median": row.bootstrap_median,
                "bootstrap_p16": row.bootstrap_p16, "bootstrap_p84": row.bootstrap_p84,
                "bootstrap_p2p5": row.bootstrap_p2p5, "bootstrap_p97p5": row.bootstrap_p97p5,
                "valid_replicas": row.valid_replicas, "invalid_replicas": row.invalid_replicas,
                "units": unit, "source_file": "bootstrap_metric_summary.tsv",
                "source_row": f"label={variant}_outer_selected; metric={metric}",
                "common_bootstrap_registry_sha256": registry_sha, "train_only": True,
                "validation_used": False, "test_used": False, "observed_data_used": False,
            })
            claim_id += 1
    return rows


def run(output: Path, paper_output: Path, inspection_report: Path | None, *, smoke: bool) -> None:
    require(output != paper_output, "checkpoint and paper export paths must differ")
    require(output.is_absolute() and paper_output.is_absolute(), "outputs must be absolute")
    require(not paper_output.exists() and paper_output.parent.is_dir(), f"refusing paper-output overwrite: {paper_output}")
    if not smoke:
        require(inspection_report is not None and inspection_report.is_file(),
                "full run requires an existing visual-inspection report")
    started = time.monotonic()
    staging = prepare_staging(output)
    try:
        apply_hh4b_paper_style()
        inputs = load_inputs()
        replicates = 20 if smoke else BOOTSTRAP_REPLICATES
        draws, registry_rows = source_member_bootstrap_draws(
            inputs["members"], replicates=replicates, seed=BOOTSTRAP_SEED,
        )
        registry_path = staging / "common_source_bootstrap_registry.tsv"
        write_tsv(registry_path, registry_rows)
        registry_sha = sha256(registry_path)
        if not smoke:
            require(registry_sha == COMMON_REGISTRY_SHA, "common c7t bootstrap registry identity drift")
        write_tsv(staging / "bootstrap_registry_provenance.tsv", [{
            "registry_relative_path": registry_path.name, "registry_sha256": registry_sha,
            "bootstrap_seed": BOOTSTRAP_SEED, "bootstrap_replicates": replicates,
            "bootstrap_unit": "source_member",
            "draw_definition": "exact_c7t_numpy_rng_stratified_by_sample_class_and_process_stratum",
            "source_members": len(inputs["members"]), "validation_payload_files_opened": 0,
            "test_or_evaluation_payload_files_opened": 0,
        }])

        models = staging / "models"; models.mkdir()
        nested = nested_outer_training(inputs, models, smoke=smoke)
        evaluated = evaluate_all_configurations(inputs, nested, draws, smoke=smoke)
        paired, paired_replicas, unavailable = build_paired_comparisons(evaluated)
        validity = bootstrap_validity_summary(evaluated)
        train_predictions, projection_predictions = build_prediction_tables(inputs, nested)

        paper_output.mkdir()
        for directory in ("figures", "captions", "source_data", "tables", "sections"):
            (paper_output / directory).mkdir()
        figures = staging / "figures"; figures.mkdir()
        figure_manifest = make_figures(inputs, nested, evaluated, paired, draws, figures, paper_output)
        table_manifest = make_paper_tables(
            paper_output, evaluated, paired, nested["selections"], evaluated["support_sensitivity"],
        )
        validate_uncertainty_contract(evaluated, paired, paired_replicas, figure_manifest, figures, replicates,
                                      smoke=smoke)

        compact_frames = {
            "candidate_scan.tsv": nested["candidates"],
            "candidate_category_support.tsv": nested["candidate_categories"],
            "nested_selection_stability.tsv": nested["selections"],
            "category_boundary_stability.tsv": nested["selections"],
            "configuration_metrics.tsv": evaluated["central"],
            "category_metrics.tsv": evaluated["central_categories"],
            "bootstrap_metric_summary.tsv": evaluated["combined_summary"],
            "category_bootstrap_summary.tsv": evaluated["category_summary"],
            "paired_model_differences.tsv": paired,
            "bootstrap_validity_summary.tsv": validity,
            "operating_point_stability.tsv": nested["selections"],
            "support_sensitivity.tsv": evaluated["support_sensitivity"],
            "model_artifact_manifest.tsv": nested["model_artifacts"],
            "unavailable_metric_registry.tsv": unavailable,
        }
        for name, frame in compact_frames.items():
            write_tsv(staging / name, frame.to_dict("records"))
            if name not in {"candidate_scan.tsv", "candidate_category_support.tsv", "model_artifact_manifest.tsv"}:
                shutil.copy2(staging / name, paper_output / name)
        evaluated["combined_replicas"].to_parquet(staging / "bootstrap_metric_replicas.parquet", index=False)
        evaluated["category_replicas"].to_parquet(staging / "category_bootstrap_replicas.parquet", index=False)
        paired_replicas.to_parquet(staging / "paired_model_difference_replicas.parquet", index=False)
        train_predictions.to_parquet(staging / "train_nested_oof_predictions.parquet", index=False)
        projection_predictions.to_parquet(staging / "primary_projection_nested_oof_predictions.parquet", index=False)

        write_paper_documentation(
            paper_output, evaluated, paired, figure_manifest, table_manifest, registry_sha, replicates,
        )
        claims = numerical_claim_rows(evaluated, registry_sha)
        write_tsv(staging / "numerical_claim_registry.tsv", claims)
        write_tsv(paper_output / "numerical_claim_registry.tsv", claims)
        write_tsv(staging / "paper_figure_manifest.tsv", figure_manifest)
        write_tsv(staging / "paper_table_manifest.tsv", table_manifest)
        write_tsv(staging / "source_evidence_manifest.tsv", inputs["evidence"])
        shutil.copy2(staging / "bootstrap_registry_provenance.tsv", paper_output / "bootstrap_registry_provenance.tsv")
        if inspection_report is not None:
            shutil.copy2(inspection_report, staging / "visual_inspection_report.tsv")
            shutil.copy2(inspection_report, paper_output / "visual_inspection_report.tsv")
        else:
            pending = [{"status": "smoke_layout_review_only", "figures": len(figure_manifest)}]
            write_tsv(staging / "visual_inspection_report.tsv", pending)
            write_tsv(paper_output / "visual_inspection_report.tsv", pending)
        shutil.copytree(paper_output, staging / "paper")

        package_rows = [
            {"package": "python", "version": platform.python_version(), "interpreter": sys.executable},
            {"package": "numpy", "version": np.__version__, "interpreter": sys.executable},
            {"package": "pandas", "version": pd.__version__, "interpreter": sys.executable},
            {"package": "pyarrow", "version": pa.__version__, "interpreter": sys.executable},
            {"package": "matplotlib", "version": plt.matplotlib.__version__, "interpreter": sys.executable},
            {"package": "scikit-learn", "version": sklearn.__version__, "interpreter": sys.executable},
            {"package": "xgboost", "version": xgboost.__version__, "interpreter": sys.executable},
        ]
        write_tsv(staging / "environment_packages.tsv", package_rows)
        base_commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO, text=True).strip()
        elapsed = time.monotonic() - started
        (staging / "README.md").write_text(
            "# c7w train-only nested categorized BDT\n\n"
            "Sealed HIG-24-015-inspired two-stage categorized-BDT checkpoint. All model choices are nested "
            "inside frozen source-group outer folds. The checkpoint uses no validation/test payload or observed data.\n"
        )
        (staging / "RUN_CONTRACT.txt").write_text(
            f"command={sys.executable} {' '.join(sys.argv)}\nbase_commit={base_commit}\n"
            f"bootstrap_seed={BOOTSTRAP_SEED}\nbootstrap_replicates={replicates}\nbootstrap_unit=source_member\n"
            f"common_bootstrap_registry_sha256={registry_sha}\n"
            "validation_opened=0\ntest_opened=0\nobserved_data_opened=0\nremote_updated=0\n"
        )
        primary_summary = {}
        def json_number(value: Any) -> float | None:
            number = float(value)
            return number if math.isfinite(number) else None
        for variant in VARIANTS:
            primary_summary[variant] = {
                metric: {
                    "nominal_full_oof": json_number((row := selected_summary_row(evaluated, variant, metric)).nominal_full_oof),
                    "bootstrap_median": json_number(row.bootstrap_median), "bootstrap_p16": json_number(row.bootstrap_p16),
                    "bootstrap_p84": json_number(row.bootstrap_p84), "valid_replicas": int(row.valid_replicas),
                    "invalid_replicas": int(row.invalid_replicas),
                }
                for metric in ("combined_nominal_asimov_ZA", "combined_systematic_aware_asimov_ZA",
                               "signal_over_background", "background_neff")
            }
        summary = {
            "schema_version": 1, "status": "pn_c7w_nested_categorized_bdt_complete",
            "base_commit": base_commit, "smoke": smoke, "runtime_seconds": elapsed,
            "bootstrap_seed": BOOTSTRAP_SEED, "bootstrap_replicates": replicates,
            "bootstrap_unit": "source_member", "common_bootstrap_registry_sha256": registry_sha,
            "selected_category_counts_by_fold": evaluated["selected_counts_by_fold"],
            "primary_performance_with_uncertainty": primary_summary,
            "paired_comparisons": int(paired.comparison.nunique()), "figures": len(figure_manifest),
            "tables": len(table_manifest), "validation_payload_files_opened": 0,
            "test_or_evaluation_payload_files_opened": 0, "observed_data_opened": 0,
        }
        write_json(staging / "summary.json", summary)
        write_tsv(staging / "artifact_manifest.tsv", artifact_rows(
            staging, exclude={"artifact_manifest.tsv", "SHA256SUMS", "COMPLETE"},
        ))
        manifest_sha = seal_checkpoint(staging, output, "pn_c7w nested categorized BDT complete")
        print(json.dumps({
            "output": str(output), "paper_output": str(paper_output), "manifest_sha256": manifest_sha,
            "bootstrap_registry_sha256": registry_sha, "replicates": replicates,
            "figures": len(figure_manifest), "tables": len(table_manifest), "runtime_seconds": elapsed,
        }, sort_keys=True))
    except Exception as exc:
        failure = {"status": "failed_preserved_staging", "error_type": type(exc).__name__,
                   "error": str(exc), "traceback": traceback.format_exc()}
        if staging.exists():
            write_json(staging / "FAILURE.json", failure)
        if paper_output.exists():
            write_json(paper_output / "FAILURE.json", failure)
        raise


def main() -> None:
    args = parse_args()
    run(args.output.resolve(), args.paper_output.resolve(),
        args.inspection_report.resolve() if args.inspection_report else None, smoke=args.smoke)


if __name__ == "__main__":
    main()
