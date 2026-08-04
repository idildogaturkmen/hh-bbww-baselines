#!/usr/bin/env python3
"""Run the frozen nested train-only PN-c7x single-head SPA-Net pipeline."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import platform
import statistics
import sys
import time
from typing import Any

import numpy as np
import pandas as pd
import sklearn
from sklearn.metrics import roc_auc_score
import torch
import xgboost
from xgboost import XGBClassifier

from scripts.analysis.pn_c7_ml_common import (
    CHECKPOINTS,
    artifact_rows,
    asimov_za,
    effective_events,
    parse_sha256sums,
    prepare_staging,
    require,
    require_group_fold_integrity,
    require_train_only,
    seal_checkpoint,
    sha256,
    threshold_at_efficiency,
    verify_checkpoint,
    write_json,
    write_tsv,
)
from scripts.analysis.pn_c7x_spanet_common import PERFECT_MATCHINGS
from scripts.analysis.pn_c7x_spanet_model import (
    ATTENTION_HEADS,
    BATCH_SIZE,
    DROPOUT,
    ENCODER_LAYERS,
    EPOCHS,
    FEEDFORWARD_DIM,
    HIDDEN_DIM,
    JET_FEATURES,
    LEARNING_RATE,
    MASS_NEGATIVE_TOLERANCE_GEV,
    TORCH_ENVIRONMENT,
    WEIGHT_DECAY,
    fit_normalizer,
    invariant_features,
    parameter_count,
    train_assignment_model,
)


REPO = Path(__file__).resolve().parents[2]
C7S = CHECKPOINTS["c7s"]
C7T = CHECKPOINTS["c7t"]
TRUTH = Path(
    "/uscms_data/d3/iturkmen/hh4b_delphes/harvest_checkpoints/"
    "pn_c7x_signal_truth_labels_20260803_v1"
)
TRUTH_MANIFEST_SHA256 = "633d10a2c5ac66e1cfe6ddc1d678c67c327656adfebc1150959d563f839c38ec"
CONTRACT = REPO / "docs/paper/jhep_hh4b_ml/HIG_24_015_SINGLE_HEAD_SPANET_CONTRACT.md"
CONFIG = REPO / "configs/baselines/hh4b_expanded_cut_bdt_protocol_v2.json"
CONFIG_SHA256 = "cc56783a4a5a19619c9dd69c1c82878a47cdf6ee6b3db24c43ca11f3a60fa636"
DEFAULT_OUTPUT = Path(
    "/uscms_data/d3/iturkmen/hh4b_delphes/harvest_checkpoints/"
    "pn_c7x_single_head_spanet_training_20260803_v1"
)
BASE_SEED = 20260803
TARGET_SIGNAL_EFFICIENCY = 0.6156418377159854
PAIRING_LABELS = {
    "((0, 1), (2, 3))": 0,
    "((0, 2), (1, 3))": 1,
    "((0, 3), (1, 2))": 2,
}
IDENTITY_COLUMNS = [
    "registry_group_id",
    "registry_member_index",
    "registry_oof_fold",
    "registry_training_target",
    "registry_sample_class",
    "registry_process_or_mode",
    "registry_stratum",
    "transport_id",
    "event",
    "candidate_row_index",
]


def verify_external_checkpoint(path: Path, expected_manifest_sha: str) -> list[dict[str, Any]]:
    manifest = path / "SHA256SUMS"
    require(manifest.is_file(), f"checkpoint manifest missing: {path}")
    require(sha256(manifest) == expected_manifest_sha, f"checkpoint manifest identity drift: {path}")
    evidence = []
    for relative, expected in parse_sha256sums(manifest).items():
        member = path / relative
        require(member.is_file() and sha256(member) == expected, f"checkpoint member drift: {member}")
        evidence.append({
            "path": str(member),
            "bytes": member.stat().st_size,
            "sha256": expected,
            "role": "sealed_signal_truth_input",
        })
    return evidence


def load_inputs() -> dict[str, Any]:
    c7s_evidence = verify_checkpoint("c7s")
    truth_evidence = verify_external_checkpoint(TRUTH, TRUTH_MANIFEST_SHA256)
    require(CONTRACT.is_file(), "single-head contract missing")
    require(sha256(CONFIG) == CONFIG_SHA256, "frozen BDT configuration drift")
    training = pd.read_parquet(C7S / "tables/train_fourb_model_development.parquet")
    projection = pd.read_parquet(C7S / "tables/train_primary_physical_projection.parquet")
    direct = pd.read_parquet(C7S / "tables/train_direct_qcd_secondary_projection.parquet")
    truth = pd.read_parquet(TRUTH / "train_signal_truth_labels.parquet")
    require((len(training), len(projection), len(direct), len(truth)) == (30705, 31225, 30705, 9337), "input row-count drift")
    require_train_only(training, "registry_final_split", "development table")
    require_train_only(projection, "registry_final_split", "physical projection")
    require_group_fold_integrity(training)
    require_group_fold_integrity(projection)
    require(
        training[["registry_group_id", "event"]].reset_index(drop=True).equals(
            direct[["registry_group_id", "event"]].reset_index(drop=True)
        ),
        "direct closure row identity drift",
    )
    require(not truth.duplicated(["registry_group_id", "event"]).any(), "duplicate truth identity")
    truth_columns = [
        "registry_group_id", "event", "truth_status", "truth_partition_label",
        "assignment_loss_mask", "unique_higgs_daughter_pairs", "compatible_partitions",
        "truth_higgs_pt_low", "truth_higgs_pt_high", "candidate_jet_mask_count",
    ]
    training = training.merge(truth[truth_columns], on=["registry_group_id", "event"], how="left", validate="one_to_one")
    projection = projection.merge(truth[truth_columns], on=["registry_group_id", "event"], how="left", validate="one_to_one")
    for frame in (training, projection):
        signal = frame["registry_sample_class"].eq("signal")
        require(not frame.loc[signal, "truth_status"].isna().any(), "signal truth join incomplete")
        frame.loc[~signal, "truth_status"] = "background_masked"
        frame.loc[~signal, "truth_partition_label"] = -1
        frame.loc[~signal, "assignment_loss_mask"] = False
        frame.loc[~signal, "candidate_jet_mask_count"] = 4
        require(set(frame["pairing"].astype(str)).issubset(PAIRING_LABELS), "unknown frozen geometric pairing encoding")
        frame["frozen_geometric_partition_label"] = frame["pairing"].astype(str).map(PAIRING_LABELS).astype(np.int8)
    require(int(training["assignment_loss_mask"].sum()) == 5769, "matchable training count drift")
    require(not training.loc[training.registry_sample_class.eq("background"), "assignment_loss_mask"].astype(bool).any(), "background assignment loss unmasked")
    evidence = [
        *[
            {"path": item["checkpoint_root"] + "/" + item["relative_path"], "bytes": item["bytes"], "sha256": item["sha256"], "role": "sealed_c7s_input"}
            for item in c7s_evidence
        ],
        *truth_evidence,
        {"path": str(CONTRACT), "bytes": CONTRACT.stat().st_size, "sha256": sha256(CONTRACT), "role": "frozen_single_head_contract"},
        {"path": str(CONFIG), "bytes": CONFIG.stat().st_size, "sha256": sha256(CONFIG), "role": "frozen_downstream_classifier_parameters"},
    ]
    return {"training": training, "projection": projection, "direct": direct, "evidence": evidence}


def xgb_parameters(seed: int, smoke: bool) -> dict[str, Any]:
    parameters = json.loads(CONFIG.read_text())["models"]["global_mass_aware"]["parameters"].copy()
    require(parameters == {
        "colsample_bytree": 0.75, "gamma": 0.1, "learning_rate": 0.03,
        "max_depth": 5, "min_child_weight": 7.0, "n_estimators": 400,
        "reg_alpha": 1.0, "reg_lambda": 1.0, "subsample": 0.75,
    }, "global c7t XGBoost parameter drift")
    if smoke:
        parameters["n_estimators"] = 5
    parameters.update({
        "objective": "binary:logistic",
        "eval_metric": "logloss",
        "tree_method": "hist",
        "n_jobs": 4,
        "random_state": int(seed),
    })
    return parameters


def fit_downstream(
    features: np.ndarray,
    labels: np.ndarray,
    weights: np.ndarray,
    *,
    seed: int,
    smoke: bool,
) -> tuple[XGBClassifier, float]:
    require(features.ndim == 2 and len(features) == len(labels) == len(weights), "downstream fit shape drift")
    require(set(np.unique(labels)) == {0, 1}, "downstream classifier requires both classes")
    started = time.perf_counter()
    classifier = XGBClassifier(**xgb_parameters(seed, smoke))
    classifier.fit(features, labels, sample_weight=weights)
    return classifier, time.perf_counter() - started


def source_split_masks(frame: pd.DataFrame, held_fold: int, eligible: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
    held = frame["registry_oof_fold"].to_numpy(dtype=np.int8) == held_fold
    if eligible is not None:
        held &= eligible
    fit = ~held if eligible is None else eligible & ~held
    fit_groups = set(frame.loc[fit, "registry_group_id"])
    held_groups = set(frame.loc[held, "registry_group_id"])
    require(not (fit_groups & held_groups), f"source leakage in held fold {held_fold}")
    return fit, held


def predict_score(classifier: XGBClassifier, features: np.ndarray) -> np.ndarray:
    result = classifier.predict_proba(features)[:, 1].astype(np.float64)
    require(bool(np.isfinite(result).all()), "nonfinite downstream classifier score")
    return result


def train_nested(inputs: dict[str, Any], staging: Path, smoke: bool) -> dict[str, Any]:
    training = inputs["training"]
    projection = inputs["projection"]
    train_score = np.full(len(training), np.nan)
    projection_score = np.full(len(projection), np.nan)
    train_partition = np.full(len(training), -1, dtype=np.int8)
    projection_partition = np.full(len(projection), -1, dtype=np.int8)
    train_max_probability = np.full(len(training), np.nan, dtype=np.float32)
    projection_max_probability = np.full(len(projection), np.nan, dtype=np.float32)
    train_entropy = np.full(len(training), np.nan, dtype=np.float32)
    projection_entropy = np.full(len(projection), np.nan, dtype=np.float32)
    train_threshold = np.full(len(training), np.nan)
    projection_threshold = np.full(len(projection), np.nan)
    folds = [0] if smoke else list(range(5))
    curves: list[dict[str, Any]] = []
    fold_rows: list[dict[str, Any]] = []
    runtime_rows: list[dict[str, Any]] = []
    model_rows: list[dict[str, Any]] = []
    models_dir = staging / "models"
    models_dir.mkdir()

    labels = training["registry_training_target"].to_numpy(dtype=np.int8)
    weights = training["development_hierarchical_weight"].to_numpy(dtype=np.float64)
    matchable = training["assignment_loss_mask"].to_numpy(dtype=bool)
    truth_labels = training["truth_partition_label"].to_numpy(dtype=np.int8)

    for outer in folds:
        outer_train = training["registry_oof_fold"].to_numpy(dtype=np.int8) != outer
        outer_held = ~outer_train
        inner_oof = np.full(len(training), np.nan)
        inner_assignment_correct = []
        for inner in sorted(set(training.loc[outer_train, "registry_oof_fold"].astype(int))):
            inner_held = outer_train & training["registry_oof_fold"].eq(inner).to_numpy()
            inner_train = outer_train & ~inner_held
            require(not (set(training.loc[inner_train, "registry_group_id"]) & set(training.loc[inner_held, "registry_group_id"])), "inner source leakage")
            seed = BASE_SEED + outer * 100 + inner
            normalizer = fit_normalizer(training.loc[inner_train])
            assignment_train = inner_train & matchable
            encoder, curve, encoder_seconds = train_assignment_model(
                training.loc[assignment_train],
                truth_labels[assignment_train],
                weights[assignment_train],
                normalizer,
                seed=seed,
                smoke=smoke,
            )
            for row in curve:
                curves.append({"scope": "inner", "outer_fold": outer, "inner_fold": inner, "seed": seed, **row})
            fit_features, _, _, _ = invariant_features(encoder, training.loc[inner_train], normalizer)
            held_features, held_partition, _, _ = invariant_features(encoder, training.loc[inner_held], normalizer)
            classifier, xgb_seconds = fit_downstream(
                fit_features,
                labels[inner_train],
                weights[inner_train],
                seed=seed,
                smoke=smoke,
            )
            inner_oof[inner_held] = predict_score(classifier, held_features)
            held_matchable = matchable[inner_held]
            inner_truth = truth_labels[inner_held][held_matchable]
            inner_prediction = held_partition[held_matchable]
            inner_assignment_correct.extend((inner_truth == inner_prediction).tolist())
            runtime_rows.append({
                "scope": "inner", "outer_fold": outer, "inner_fold": inner, "seed": seed,
                "assignment_training_seconds": encoder_seconds,
                "downstream_training_seconds": xgb_seconds,
                "fit_rows": int(inner_train.sum()),
                "assignment_fit_rows": int(assignment_train.sum()),
                "held_rows": int(inner_held.sum()),
            })
        require(bool(np.isfinite(inner_oof[outer_train]).all()), f"inner OOF score incomplete for outer fold {outer}")
        threshold = threshold_at_efficiency(
            inner_oof[outer_train],
            labels[outer_train] == 1,
            weights[outer_train],
            TARGET_SIGNAL_EFFICIENCY,
        )

        seed = BASE_SEED + outer * 100 + 99
        normalizer = fit_normalizer(training.loc[outer_train])
        assignment_train = outer_train & matchable
        encoder, curve, encoder_seconds = train_assignment_model(
            training.loc[assignment_train],
            truth_labels[assignment_train],
            weights[assignment_train],
            normalizer,
            seed=seed,
            smoke=smoke,
        )
        for row in curve:
            curves.append({"scope": "outer_final", "outer_fold": outer, "inner_fold": -1, "seed": seed, **row})
        fit_features, _, _, _ = invariant_features(encoder, training.loc[outer_train], normalizer)
        held_features, held_partition, held_max, held_entropy = invariant_features(encoder, training.loc[outer_held], normalizer)
        projection_held = projection["registry_oof_fold"].to_numpy(dtype=np.int8) == outer
        projection_features, projection_part, projection_max, projection_ent = invariant_features(
            encoder, projection.loc[projection_held], normalizer
        )
        classifier, xgb_seconds = fit_downstream(
            fit_features,
            labels[outer_train],
            weights[outer_train],
            seed=seed,
            smoke=smoke,
        )
        train_score[outer_held] = predict_score(classifier, held_features)
        projection_score[projection_held] = predict_score(classifier, projection_features)
        train_partition[outer_held] = held_partition
        projection_partition[projection_held] = projection_part
        train_max_probability[outer_held] = held_max
        projection_max_probability[projection_held] = projection_max
        train_entropy[outer_held] = held_entropy
        projection_entropy[projection_held] = projection_ent
        train_threshold[outer_held] = threshold
        projection_threshold[projection_held] = threshold

        inference_times = []
        repetitions = 3 if smoke else 10
        for _ in range(repetitions):
            started = time.perf_counter()
            timing_features, _, _, _ = invariant_features(encoder, training.loc[outer_held], normalizer)
            predict_score(classifier, timing_features)
            inference_times.append(time.perf_counter() - started)
        inference_times = sorted(inference_times)
        inference_median = statistics.median(inference_times)
        inference_q1, inference_q3 = np.percentile(inference_times, [25.0, 75.0])

        encoder_path = models_dir / f"outer_fold_{outer}_assignment_state.pt"
        normalizer_path = models_dir / f"outer_fold_{outer}_normalizer.npz"
        classifier_path = models_dir / f"outer_fold_{outer}_downstream_xgboost.json"
        torch.save(encoder.state_dict(), encoder_path)
        np.savez(normalizer_path, mean=normalizer["mean"], scale=normalizer["scale"])
        classifier.save_model(classifier_path)
        for role, path in (
            ("single_head_assignment_state_dict", encoder_path),
            ("outer_train_jet_normalizer", normalizer_path),
            ("logically_distinct_downstream_xgboost", classifier_path),
        ):
            model_rows.append({
                "outer_fold": outer, "role": role, "path": path.relative_to(staging).as_posix(),
                "bytes": path.stat().st_size, "sha256": sha256(path), "seed": seed,
            })
        held_matchable_mask = matchable[outer_held]
        outer_accuracy = float(np.mean(held_partition[held_matchable_mask] == truth_labels[outer_held][held_matchable_mask]))
        inner_accuracy = float(np.mean(inner_assignment_correct))
        fold_rows.append({
            "outer_fold": outer,
            "seed": seed,
            "inner_oof_operating_threshold": threshold,
            "target_weighted_signal_efficiency": TARGET_SIGNAL_EFFICIENCY,
            "outer_train_rows": int(outer_train.sum()),
            "outer_held_rows": int(outer_held.sum()),
            "outer_assignment_train_rows": int(assignment_train.sum()),
            "inner_matchable_assignment_accuracy": inner_accuracy,
            "outer_matchable_assignment_accuracy": outer_accuracy,
            "assignment_parameter_count": parameter_count(encoder),
            "assignment_training_seconds": encoder_seconds,
            "downstream_training_seconds": xgb_seconds,
            "inference_rows": int(outer_held.sum()),
            "inference_repetitions": repetitions,
            "inference_seconds_median": inference_median,
            "inference_seconds_q1": float(inference_q1),
            "inference_seconds_q3": float(inference_q3),
            "status": "outer_fold_train_only_pass",
        })
        runtime_rows.append({
            "scope": "outer_final", "outer_fold": outer, "inner_fold": -1, "seed": seed,
            "assignment_training_seconds": encoder_seconds,
            "downstream_training_seconds": xgb_seconds,
            "fit_rows": int(outer_train.sum()),
            "assignment_fit_rows": int(assignment_train.sum()),
            "held_rows": int(outer_held.sum()),
        })
        print(f"completed outer fold {outer} threshold={threshold:.9g} assignment_accuracy={outer_accuracy:.6f}", flush=True)

    completed_train = np.isfinite(train_score)
    completed_projection = np.isfinite(projection_score)
    if smoke:
        require(completed_train.equals(training.registry_oof_fold.eq(0)) if isinstance(completed_train, pd.Series) else np.array_equal(completed_train, training.registry_oof_fold.eq(0).to_numpy()), "smoke train coverage drift")
        require(np.array_equal(completed_projection, projection.registry_oof_fold.eq(0).to_numpy()), "smoke projection coverage drift")
    else:
        require(bool(completed_train.all() and completed_projection.all()), "full OOF prediction incomplete")
    return {
        "train_score": train_score,
        "projection_score": projection_score,
        "train_partition": train_partition,
        "projection_partition": projection_partition,
        "train_max_probability": train_max_probability,
        "projection_max_probability": projection_max_probability,
        "train_entropy": train_entropy,
        "projection_entropy": projection_entropy,
        "train_threshold": train_threshold,
        "projection_threshold": projection_threshold,
        "curves": curves,
        "fold_rows": fold_rows,
        "runtime_rows": runtime_rows,
        "model_rows": model_rows,
        "completed_train": completed_train,
        "completed_projection": completed_projection,
    }


def jet_four_vectors(frame: pd.DataFrame) -> np.ndarray:
    vectors = []
    for position in range(1, 5):
        pt = frame[f"j{position}_pt"].to_numpy(dtype=np.float64)
        eta = frame[f"j{position}_eta"].to_numpy(dtype=np.float64)
        phi = frame[f"j{position}_phi"].to_numpy(dtype=np.float64)
        mass = frame[f"j{position}_mass"].to_numpy(dtype=np.float64)
        px = pt * np.cos(phi)
        py = pt * np.sin(phi)
        pz = pt * np.sinh(eta)
        energy = np.sqrt(np.maximum(0.0, px * px + py * py + pz * pz + mass * mass))
        vectors.append(np.column_stack((energy, px, py, pz)))
    return np.stack(vectors, axis=1)


def pairing_masses(frame: pd.DataFrame, labels: np.ndarray) -> np.ndarray:
    vectors = jet_four_vectors(frame)
    labels = np.asarray(labels, dtype=np.int8)
    require(set(np.unique(labels)).issubset({0, 1, 2}), "mass reconstruction label drift")
    result = np.empty((len(frame), 2), dtype=np.float64)
    for label, matching in enumerate(PERFECT_MATCHINGS):
        selected = labels == label
        for pair_index, (left, right) in enumerate(matching):
            total = vectors[selected, left] + vectors[selected, right]
            mass2 = total[:, 0] ** 2 - np.sum(total[:, 1:] ** 2, axis=1)
            result[selected, pair_index] = np.sqrt(np.maximum(0.0, mass2))
    return np.sort(result, axis=1)


def build_prediction_tables(inputs: dict[str, Any], nested: dict[str, Any], smoke: bool) -> tuple[pd.DataFrame, pd.DataFrame]:
    training = inputs["training"]
    projection = inputs["projection"]
    train = training[IDENTITY_COLUMNS + [
        "development_hierarchical_weight", "run2_candidate_physical_weight", "population_kind",
        "truth_status", "truth_partition_label", "assignment_loss_mask", "truth_higgs_pt_low",
        "truth_higgs_pt_high", "n_selected_jets", "n_extra_selected_jets", "mhh", "pairing",
        "frozen_geometric_partition_label",
    ]].copy()
    project = projection[IDENTITY_COLUMNS + [
        "analysis_population_role", "primary_projection_physical_weight_inclusive",
        "run2_candidate_physical_weight", "population_kind", "truth_status",
        "truth_partition_label", "assignment_loss_mask", "truth_higgs_pt_low", "truth_higgs_pt_high",
        "n_selected_jets", "n_extra_selected_jets", "mhh",
        "pairing", "frozen_geometric_partition_label",
    ]].copy()
    for frame, prefix in ((train, "train"), (project, "projection")):
        frame["single_head_score"] = nested[f"{prefix}_score"]
        frame["predicted_partition_label"] = nested[f"{prefix}_partition"]
        frame["assignment_max_probability"] = nested[f"{prefix}_max_probability"]
        frame["assignment_entropy"] = nested[f"{prefix}_entropy"]
        frame["nested_operating_threshold"] = nested[f"{prefix}_threshold"]
        frame["selected_at_nested_operating_point"] = frame["single_head_score"] >= frame["nested_operating_threshold"]
    completed = nested["completed_train"]
    learned_labels = nested["train_partition"][completed]
    frozen_labels = training.loc[completed, "frozen_geometric_partition_label"].to_numpy(dtype=np.int8)
    frozen_masses = pairing_masses(training.loc[completed], frozen_labels)
    learned_masses = pairing_masses(training.loc[completed], learned_labels)
    recorded_masses = np.sort(training.loc[completed, ["mbb1", "mbb2"]].to_numpy(dtype=np.float64), axis=1)
    require(bool(np.allclose(frozen_masses, recorded_masses, rtol=0.0, atol=2.0e-5)), "frozen pairing mass reconstruction drift")
    train.loc[completed, "frozen_pair_mass_low"] = frozen_masses[:, 0]
    train.loc[completed, "frozen_pair_mass_high"] = frozen_masses[:, 1]
    train.loc[completed, "learned_pair_mass_low"] = learned_masses[:, 0]
    train.loc[completed, "learned_pair_mass_high"] = learned_masses[:, 1]
    train.loc[completed, "frozen_pair_mean_absolute_mass_residual"] = np.mean(np.abs(frozen_masses - 125.0), axis=1)
    train.loc[completed, "learned_pair_mean_absolute_mass_residual"] = np.mean(np.abs(learned_masses - 125.0), axis=1)
    train.loc[completed, "frozen_pair_mean_squared_mass_residual"] = np.mean(np.square(frozen_masses - 125.0), axis=1)
    train.loc[completed, "learned_pair_mean_squared_mass_residual"] = np.mean(np.square(learned_masses - 125.0), axis=1)
    return train, project


def point_metrics(inputs: dict[str, Any], train: pd.DataFrame, project: pd.DataFrame, completed: np.ndarray) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    signal = train.registry_sample_class.eq("signal").to_numpy() & completed
    matchable = train.assignment_loss_mask.astype(bool).to_numpy() & completed
    truth = train.truth_partition_label.to_numpy(dtype=np.int8)
    learned = train.predicted_partition_label.to_numpy(dtype=np.int8)
    weights = train.development_hierarchical_weight.to_numpy(dtype=np.float64)
    learned_correct = learned[matchable] == truth[matchable]
    frozen = train.frozen_geometric_partition_label.to_numpy(dtype=np.int8)
    frozen_correct = truth[matchable] == frozen[matchable]
    assignment = [
        {"metric": "signal_rows", "value": int(signal.sum()), "denominator": int(signal.sum()), "weighting": "count"},
        {"metric": "matchable_event_efficiency", "value": float(matchable.sum() / signal.sum()), "denominator": int(signal.sum()), "weighting": "unweighted"},
        {"metric": "ambiguity_fraction", "value": float((train.truth_status.eq("ambiguous_match").to_numpy() & signal).sum() / signal.sum()), "denominator": int(signal.sum()), "weighting": "unweighted"},
        {"metric": "unmatched_fraction", "value": float((train.truth_status.eq("unmatched").to_numpy() & signal).sum() / signal.sum()), "denominator": int(signal.sum()), "weighting": "unweighted"},
        {"metric": "learned_exact_event_pairing_accuracy", "value": float(np.mean(learned_correct)), "denominator": int(matchable.sum()), "weighting": "unweighted"},
        {"metric": "learned_exact_event_pairing_accuracy", "value": float(np.average(learned_correct, weights=weights[matchable])), "denominator": int(matchable.sum()), "weighting": "development_hierarchical"},
        {"metric": "frozen_geometric_exact_event_pairing_accuracy", "value": float(np.mean(frozen_correct)), "denominator": int(matchable.sum()), "weighting": "unweighted"},
        {"metric": "frozen_geometric_exact_event_pairing_accuracy", "value": float(np.average(frozen_correct, weights=weights[matchable])), "denominator": int(matchable.sum()), "weighting": "development_hierarchical"},
        {"metric": "learned_per_higgs_pairing_accuracy", "value": float(np.mean(learned_correct)), "denominator": int(2 * matchable.sum()), "weighting": "unweighted_candidate_four_degenerate"},
        {"metric": "learned_per_jet_partner_accuracy", "value": float(np.mean(learned_correct)), "denominator": int(4 * matchable.sum()), "weighting": "unweighted_candidate_four_degenerate"},
        {"metric": "frozen_mass_residual_rms_GeV", "value": float(np.sqrt(train.loc[matchable, "frozen_pair_mean_squared_mass_residual"].mean())), "denominator": int(2 * matchable.sum()), "weighting": "unweighted"},
        {"metric": "learned_mass_residual_rms_GeV", "value": float(np.sqrt(train.loc[matchable, "learned_pair_mean_squared_mass_residual"].mean())), "denominator": int(2 * matchable.sum()), "weighting": "unweighted"},
    ]
    label = train.registry_training_target.to_numpy(dtype=np.int8)[completed]
    score = train.single_head_score.to_numpy(dtype=np.float64)[completed]
    dev_weight = weights[completed]
    selected = project.selected_at_nested_operating_point.to_numpy() & np.isfinite(project.single_head_score)
    physical_weight = project.primary_projection_physical_weight_inclusive.to_numpy(dtype=np.float64)
    signal_projection = project.registry_sample_class.eq("signal").to_numpy()
    selected_signal = selected & signal_projection
    selected_background = selected & ~signal_projection
    signal_yield = float(physical_weight[selected_signal].sum())
    background_yield = float(physical_weight[selected_background].sum())
    classification = [
        {"metric": "weighted_auc", "value": float(roc_auc_score(label, score, sample_weight=dev_weight)), "units": "unitless"},
        {"metric": "unweighted_auc", "value": float(roc_auc_score(label, score)), "units": "unitless"},
        {"metric": "selected_signal_rows", "value": int(selected_signal.sum()), "units": "rows"},
        {"metric": "selected_background_rows", "value": int(selected_background.sum()), "units": "rows"},
        {"metric": "selected_signal_yield", "value": signal_yield, "units": "Run2_weighted_events"},
        {"metric": "selected_background_yield", "value": background_yield, "units": "Run2_weighted_events"},
        {"metric": "signal_over_background", "value": signal_yield / background_yield, "units": "unitless"},
        {"metric": "signal_over_sqrt_background", "value": signal_yield / np.sqrt(background_yield), "units": "unitless"},
        {"metric": "nominal_asimov_ZA", "value": asimov_za(signal_yield, background_yield), "units": "unitless"},
        {"metric": "background_effective_events", "value": effective_events(physical_weight[selected_background]), "units": "effective_events"},
        {"metric": "signal_effective_events", "value": effective_events(physical_weight[selected_signal]), "units": "effective_events"},
        {"metric": "transferred_qcd_fraction", "value": float(physical_weight[selected & project.analysis_population_role.eq("primary_transferred_multijet_template").to_numpy()].sum() / background_yield), "units": "fraction"},
    ]
    return assignment, classification


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    require(Path(sys.executable) == Path(TORCH_ENVIRONMENT), f"run with required interpreter: {TORCH_ENVIRONMENT}")
    inputs = load_inputs()
    staging = prepare_staging(args.output)
    try:
        nested = train_nested(inputs, staging, args.smoke)
        train_predictions, projection_predictions = build_prediction_tables(inputs, nested, args.smoke)
        predictions_dir = staging / "predictions"
        predictions_dir.mkdir()
        train_predictions.to_parquet(predictions_dir / "train_fourb_single_head_oof.parquet", index=False, compression="zstd")
        projection_predictions.to_parquet(predictions_dir / "train_primary_projection_single_head_oof.parquet", index=False, compression="zstd")
        assignment, classification = point_metrics(inputs, train_predictions, projection_predictions, nested["completed_train"])
        write_tsv(staging / "assignment_metric_point_estimates.tsv", assignment)
        write_tsv(staging / "classification_metric_point_estimates.tsv", classification)
        write_tsv(staging / "assignment_loss_curves.tsv", nested["curves"])
        write_tsv(staging / "outer_fold_training_and_stability.tsv", nested["fold_rows"])
        write_tsv(staging / "runtime.tsv", nested["runtime_rows"])
        write_tsv(staging / "model_artifact_manifest.tsv", nested["model_rows"])
        write_tsv(staging / "source_evidence_manifest.tsv", inputs["evidence"])
        environment = {
            "python_executable": sys.executable,
            "python": platform.python_version(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "sklearn": sklearn.__version__,
            "xgboost": xgboost.__version__,
            "torch": torch.__version__,
            "torch_cuda_available": torch.cuda.is_available(),
            "torch_threads": torch.get_num_threads(),
            "device": "cpu",
        }
        write_json(staging / "environment.json", environment)
        summary = {
            "schema_version": 1,
            "status": "pn_c7x_single_head_spanet_smoke_pass" if args.smoke else "pn_c7x_single_head_spanet_training_pass",
            "created_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "scope": "Delphes_train_only_source_group_nested_OOF",
            "architecture": {
                "candidate_jets": 4, "jet_features": list(JET_FEATURES), "hidden_dim": HIDDEN_DIM,
                "attention_heads": ATTENTION_HEADS, "encoder_layers": ENCODER_LAYERS,
                "feedforward_dim": FEEDFORWARD_DIM, "dropout": DROPOUT,
                "epochs": 2 if args.smoke else EPOCHS, "batch_size": BATCH_SIZE,
                "learning_rate": LEARNING_RATE, "weight_decay": WEIGHT_DECAY,
                "negative_jet_mass_numerical_tolerance_GeV": MASS_NEGATIVE_TOLERANCE_GEV,
                "assignment_head": "shared_symmetric_pair_scorer_three_perfect_matchings",
                "event_classifier": "logically_distinct_frozen_parameter_XGBoost_on_invariant_representation",
            },
            "target_weighted_signal_efficiency": TARGET_SIGNAL_EFFICIENCY,
            "outer_folds_completed": [int(row["outer_fold"]) for row in nested["fold_rows"]],
            "truth_checkpoint": str(TRUTH),
            "truth_manifest_sha256": TRUTH_MANIFEST_SHA256,
            "assignment_point_metrics": {f"{row['metric']}::{row['weighting']}": row["value"] for row in assignment},
            "classification_point_metrics": {row["metric"]: row["value"] for row in classification},
            "environment": environment,
            "validation_payload_files_opened": 0,
            "test_or_evaluation_payload_files_opened": 0,
            "observed_data_opened": False,
            "full_run2_prediction": False,
            "bootstrap_evaluation_complete": False,
            "next_gate": "common source-member bootstrap evaluation, paired inclusive-BDT differences, uncertainty figures, and final single-head result checkpoint",
        }
        write_json(staging / "summary.json", summary)
        (staging / "README.md").write_text(
            "# PN-c7x train-only single-head SPA-Net training\n\nNested source-group OOF candidate-four assignment encoder and logically distinct downstream classifier. "
            "Delphes simulation only; no observed data, validation, or test access. This training checkpoint is not the final uncertainty-qualified model comparison.\n"
        )
        (staging / "RUN_CONTRACT.txt").write_text(
            "The assignment architecture and optimizer are fixed before training. Four inner source folds determine each outer fold's operating threshold at the frozen c7t cut signal efficiency. "
            "Only matchable signal contributes assignment loss; every background and unmatched/ambiguous signal row is masked. The downstream XGBoost classifier is trained separately on invariant encoder features.\n"
        )
        write_tsv(staging / "artifact_manifest.tsv", artifact_rows(staging, exclude={"artifact_manifest.tsv", "SHA256SUMS", "COMPLETE"}))
        manifest_sha = seal_checkpoint(
            staging,
            args.output,
            "pn_c7x_single_head_spanet_smoke_pass" if args.smoke else "pn_c7x_single_head_spanet_training_pass",
        )
        print(json.dumps({"checkpoint": str(args.output), "manifest_sha256": manifest_sha, **summary}, sort_keys=True), flush=True)
    except Exception:
        write_json(staging / "FAILED.json", {"status": "pn_c7x_single_head_training_failed_preserved_staging"})
        raise


if __name__ == "__main__":
    main()
