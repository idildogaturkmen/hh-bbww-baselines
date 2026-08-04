#!/usr/bin/env python3
"""Run the frozen nested train-only PN-c7y two-head SPA-Net pipeline."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import platform
import statistics
import sys
import time
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
import sklearn
import torch

from scripts.analysis.pn_c7_ml_common import (
    artifact_rows,
    prepare_staging,
    require,
    seal_checkpoint,
    sha256,
    threshold_at_efficiency,
    write_json,
    write_tsv,
)
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
)
from scripts.analysis.run_pn_c7x_single_head_spanet import (
    BASE_SEED as SINGLE_HEAD_BASE_SEED,
    IDENTITY_COLUMNS,
    TARGET_SIGNAL_EFFICIENCY,
    load_inputs,
    pairing_masses,
)
from scripts.analysis.pn_c7y_two_head_model import (
    LOSS_WEIGHTS,
    joint_selection_utility,
    loss_weight_index,
    parameter_count,
    predict_two_head,
    select_loss_weight,
    train_two_head_model,
)


REPO = Path(__file__).resolve().parents[2]
CONTRACT = REPO / "docs/paper/jhep_hh4b_ml/HIG_24_015_TWO_HEAD_SPANET_CONTRACT.md"
DEFAULT_OUTPUT = Path(
    "/uscms_data/d3/iturkmen/hh4b_delphes/harvest_checkpoints/"
    "pn_c7y_two_head_spanet_training_20260803_v1"
)
BASE_SEED = SINGLE_HEAD_BASE_SEED + 100000


def source_masks(frame: pd.DataFrame, held_fold: int) -> tuple[np.ndarray, np.ndarray]:
    held = frame["registry_oof_fold"].to_numpy(dtype=np.int8) == held_fold
    fit = ~held
    require(
        not (
            set(frame.loc[fit, "registry_group_id"])
            & set(frame.loc[held, "registry_group_id"])
        ),
        f"source leakage in held fold {held_fold}",
    )
    return fit, held


def train_nested(inputs: dict[str, Any], staging: Path, smoke: bool) -> dict[str, Any]:
    training = inputs["training"]
    projection = inputs["projection"]
    folds = [0] if smoke else list(range(5))
    fold_values = training["registry_oof_fold"].to_numpy(dtype=np.int8)
    projection_folds = projection["registry_oof_fold"].to_numpy(dtype=np.int8)
    labels = training["registry_training_target"].to_numpy(dtype=np.int8)
    weights = training["development_hierarchical_weight"].to_numpy(dtype=np.float64)
    matchable = training["assignment_loss_mask"].to_numpy(dtype=bool)
    truth = training["truth_partition_label"].to_numpy(dtype=np.int8)

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
    train_loss_weight = np.full(len(training), np.nan)
    projection_loss_weight = np.full(len(projection), np.nan)

    curves: list[dict[str, Any]] = []
    gradients: list[dict[str, Any]] = []
    candidate_rows: list[dict[str, Any]] = []
    selection_rows: list[dict[str, Any]] = []
    runtime_rows: list[dict[str, Any]] = []
    model_rows: list[dict[str, Any]] = []
    nested_training_frames: list[pd.DataFrame] = []
    nested_projection_frames: list[pd.DataFrame] = []
    models_dir = staging / "models"
    models_dir.mkdir()

    for outer in folds:
        outer_train = fold_values != outer
        outer_held = ~outer_train
        inner_folds = sorted(set(fold_values[outer_train].astype(int)))
        candidate_cache: dict[float, dict[str, np.ndarray]] = {}
        outer_candidate_rows: list[dict[str, Any]] = []
        for assignment_loss_weight in LOSS_WEIGHTS:
            weight_index = loss_weight_index(assignment_loss_weight)
            inner_score = np.full(len(training), np.nan)
            inner_partition = np.full(len(training), -1, dtype=np.int8)
            inner_projection_score = np.full(len(projection), np.nan)
            for inner in inner_folds:
                inner_held = outer_train & (fold_values == inner)
                inner_train = outer_train & ~inner_held
                projected_inner = (projection_folds == inner) & (projection_folds != outer)
                require(
                    not (
                        set(training.loc[inner_train, "registry_group_id"])
                        & set(training.loc[inner_held, "registry_group_id"])
                    ),
                    "inner source leakage",
                )
                seed = BASE_SEED + outer * 1000 + weight_index * 100 + inner
                normalizer = fit_normalizer(training.loc[inner_train])
                model, curve, diagnostic, seconds = train_two_head_model(
                    training.loc[inner_train],
                    labels[inner_train],
                    truth[inner_train],
                    matchable[inner_train],
                    weights[inner_train],
                    normalizer,
                    assignment_loss_weight=assignment_loss_weight,
                    seed=seed,
                    smoke=smoke,
                )
                held_score, held_partition, _, _ = predict_two_head(
                    model,
                    training.loc[inner_held],
                    normalizer,
                )
                projected_score, _, _, _ = predict_two_head(
                    model,
                    projection.loc[projected_inner],
                    normalizer,
                )
                inner_score[inner_held] = held_score
                inner_partition[inner_held] = held_partition
                inner_projection_score[projected_inner] = projected_score
                for row in curve:
                    curves.append({
                        "scope": "inner_candidate",
                        "outer_fold": outer,
                        "inner_fold": inner,
                        "seed": seed,
                        **row,
                    })
                gradients.append({
                    "scope": "inner_candidate",
                    "outer_fold": outer,
                    "inner_fold": inner,
                    "seed": seed,
                    **diagnostic,
                })
                runtime_rows.append({
                    "scope": "inner_candidate",
                    "outer_fold": outer,
                    "inner_fold": inner,
                    "assignment_loss_weight": assignment_loss_weight,
                    "seed": seed,
                    "training_seconds": seconds,
                    "fit_rows": int(inner_train.sum()),
                    "assignment_fit_rows": int((inner_train & matchable).sum()),
                    "held_rows": int(inner_held.sum()),
                    "inference_repetitions": "",
                    "inference_seconds_median": "",
                    "inference_seconds_q1": "",
                    "inference_seconds_q3": "",
                    "outer_weighted_auc": "",
                    "outer_exact_pairing_accuracy": "",
                    "parameter_count": "",
                })
            require(bool(np.isfinite(inner_score[outer_train]).all()),
                    f"inner score incomplete: outer {outer}, weight {assignment_loss_weight}")
            require(bool((inner_partition[outer_train] >= 0).all()),
                    f"inner assignment incomplete: outer {outer}, weight {assignment_loss_weight}")
            projection_outer_train = projection_folds != outer
            require(bool(np.isfinite(inner_projection_score[projection_outer_train]).all()),
                    f"inner projection score incomplete: outer {outer}, weight {assignment_loss_weight}")
            weighted_auc = float(
                roc_auc_score(
                    labels[outer_train],
                    inner_score[outer_train],
                    sample_weight=weights[outer_train],
                )
            )
            exact_accuracy = float(
                np.mean(inner_partition[outer_train][matchable[outer_train]] == truth[outer_train][matchable[outer_train]])
            )
            utility = joint_selection_utility(weighted_auc, exact_accuracy)
            candidate = {
                "outer_fold": outer,
                "assignment_loss_weight": assignment_loss_weight,
                "inner_weighted_auc": weighted_auc,
                "inner_exact_pairing_accuracy": exact_accuracy,
                "joint_selection_utility": utility,
                "outer_held_used_for_selection": False,
            }
            candidate_rows.append(candidate)
            outer_candidate_rows.append(candidate)
            candidate_cache[assignment_loss_weight] = {
                "training_score": inner_score,
                "training_partition": inner_partition,
                "projection_score": inner_projection_score,
            }

        selected = select_loss_weight(outer_candidate_rows)
        selected_weight = float(selected["assignment_loss_weight"])
        selected_cache = candidate_cache[selected_weight]
        threshold = threshold_at_efficiency(
            selected_cache["training_score"][outer_train],
            labels[outer_train] == 1,
            weights[outer_train],
            TARGET_SIGNAL_EFFICIENCY,
        )
        selection_rows.append({
            **selected,
            "inner_oof_operating_threshold": threshold,
            "target_weighted_signal_efficiency": TARGET_SIGNAL_EFFICIENCY,
            "candidate_weights_json": json.dumps(list(LOSS_WEIGHTS), separators=(",", ":")),
            "selection_rule": "max_half_weighted_auc_plus_half_exact_pairing_accuracy_tie_smaller_weight",
        })

        nested_train = training.loc[outer_train, IDENTITY_COLUMNS].copy()
        nested_train.insert(0, "outer_fold", outer)
        nested_train["selected_assignment_loss_weight"] = selected_weight
        nested_train["inner_oof_score"] = selected_cache["training_score"][outer_train]
        nested_train["inner_oof_partition"] = selected_cache["training_partition"][outer_train]
        nested_training_frames.append(nested_train)
        projection_outer_train = projection_folds != outer
        nested_project = projection.loc[projection_outer_train, IDENTITY_COLUMNS].copy()
        nested_project.insert(0, "outer_fold", outer)
        nested_project["selected_assignment_loss_weight"] = selected_weight
        nested_project["inner_oof_score"] = selected_cache["projection_score"][projection_outer_train]
        nested_projection_frames.append(nested_project)

        seed = BASE_SEED + outer * 1000 + 999
        normalizer = fit_normalizer(training.loc[outer_train])
        model, curve, diagnostic, seconds = train_two_head_model(
            training.loc[outer_train],
            labels[outer_train],
            truth[outer_train],
            matchable[outer_train],
            weights[outer_train],
            normalizer,
            assignment_loss_weight=selected_weight,
            seed=seed,
            smoke=smoke,
        )
        for row in curve:
            curves.append({
                "scope": "outer_final",
                "outer_fold": outer,
                "inner_fold": -1,
                "seed": seed,
                **row,
            })
        gradients.append({
            "scope": "outer_final",
            "outer_fold": outer,
            "inner_fold": -1,
            "seed": seed,
            **diagnostic,
        })
        held_score, held_partition, held_max, held_entropy = predict_two_head(
            model,
            training.loc[outer_held],
            normalizer,
        )
        projected_outer = projection_folds == outer
        projected_score, projected_partition, projected_max, projected_entropy = predict_two_head(
            model,
            projection.loc[projected_outer],
            normalizer,
        )
        train_score[outer_held] = held_score
        projection_score[projected_outer] = projected_score
        train_partition[outer_held] = held_partition
        projection_partition[projected_outer] = projected_partition
        train_max_probability[outer_held] = held_max
        projection_max_probability[projected_outer] = projected_max
        train_entropy[outer_held] = held_entropy
        projection_entropy[projected_outer] = projected_entropy
        train_threshold[outer_held] = threshold
        projection_threshold[projected_outer] = threshold
        train_loss_weight[outer_held] = selected_weight
        projection_loss_weight[projected_outer] = selected_weight

        inference_times = []
        repetitions = 3 if smoke else 10
        for _ in range(repetitions):
            started = time.perf_counter()
            predict_two_head(model, training.loc[outer_held], normalizer)
            inference_times.append(time.perf_counter() - started)
        inference_median = statistics.median(inference_times)
        inference_q1, inference_q3 = np.percentile(inference_times, [25.0, 75.0])

        model_path = models_dir / f"outer_fold_{outer}_two_head_state.pt"
        normalizer_path = models_dir / f"outer_fold_{outer}_normalizer.npz"
        torch.save(model.state_dict(), model_path)
        np.savez(normalizer_path, mean=normalizer["mean"], scale=normalizer["scale"])
        for role, path in (
            ("joint_two_head_state_dict", model_path),
            ("outer_train_jet_normalizer", normalizer_path),
        ):
            model_rows.append({
                "outer_fold": outer,
                "role": role,
                "path": path.relative_to(staging).as_posix(),
                "bytes": path.stat().st_size,
                "sha256": sha256(path),
                "seed": seed,
                "selected_assignment_loss_weight": selected_weight,
            })
        held_matchable = matchable[outer_held]
        outer_accuracy = float(
            np.mean(held_partition[held_matchable] == truth[outer_held][held_matchable])
        )
        outer_auc = float(
            roc_auc_score(
                labels[outer_held],
                held_score,
                sample_weight=weights[outer_held],
            )
        )
        runtime_rows.append({
            "scope": "outer_final",
            "outer_fold": outer,
            "inner_fold": -1,
            "assignment_loss_weight": selected_weight,
            "seed": seed,
            "training_seconds": seconds,
            "fit_rows": int(outer_train.sum()),
            "assignment_fit_rows": int((outer_train & matchable).sum()),
            "held_rows": int(outer_held.sum()),
            "inference_repetitions": repetitions,
            "inference_seconds_median": inference_median,
            "inference_seconds_q1": float(inference_q1),
            "inference_seconds_q3": float(inference_q3),
            "outer_weighted_auc": outer_auc,
            "outer_exact_pairing_accuracy": outer_accuracy,
            "parameter_count": parameter_count(model),
        })
        print(
            f"completed outer fold {outer} weight={selected_weight:g} "
            f"threshold={threshold:.9g} auc={outer_auc:.6f} pairing={outer_accuracy:.6f}",
            flush=True,
        )

    completed_train = np.isfinite(train_score)
    completed_projection = np.isfinite(projection_score)
    expected_train = np.isin(fold_values, folds)
    expected_projection = np.isin(projection_folds, folds)
    require(np.array_equal(completed_train, expected_train), "two-head train OOF coverage drift")
    require(np.array_equal(completed_projection, expected_projection), "two-head projection OOF coverage drift")
    require(bool((train_partition[completed_train] >= 0).all()), "two-head assignment coverage drift")
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
        "train_loss_weight": train_loss_weight,
        "projection_loss_weight": projection_loss_weight,
        "completed_train": completed_train,
        "completed_projection": completed_projection,
        "curves": curves,
        "gradients": gradients,
        "candidate_rows": candidate_rows,
        "selection_rows": selection_rows,
        "runtime_rows": runtime_rows,
        "model_rows": model_rows,
        "nested_training": pd.concat(nested_training_frames, ignore_index=True),
        "nested_projection": pd.concat(nested_projection_frames, ignore_index=True),
    }


def build_prediction_tables(
    inputs: dict[str, Any],
    nested: dict[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    training = inputs["training"]
    projection = inputs["projection"]
    train = training[IDENTITY_COLUMNS + [
        "development_hierarchical_weight",
        "run2_candidate_physical_weight",
        "population_kind",
        "truth_status",
        "truth_partition_label",
        "assignment_loss_mask",
        "truth_higgs_pt_low",
        "truth_higgs_pt_high",
        "n_selected_jets",
        "n_extra_selected_jets",
        "mhh",
        "pairing",
        "frozen_geometric_partition_label",
    ]].copy()
    project = projection[IDENTITY_COLUMNS + [
        "analysis_population_role",
        "primary_projection_physical_weight_inclusive",
        "run2_candidate_physical_weight",
        "population_kind",
        "truth_status",
        "truth_partition_label",
        "assignment_loss_mask",
        "truth_higgs_pt_low",
        "truth_higgs_pt_high",
        "n_selected_jets",
        "n_extra_selected_jets",
        "mhh",
        "pairing",
        "frozen_geometric_partition_label",
    ]].copy()
    for frame, prefix in ((train, "train"), (project, "projection")):
        frame["two_head_score"] = nested[f"{prefix}_score"]
        frame["predicted_partition_label"] = nested[f"{prefix}_partition"]
        frame["assignment_max_probability"] = nested[f"{prefix}_max_probability"]
        frame["assignment_entropy"] = nested[f"{prefix}_entropy"]
        frame["nested_operating_threshold"] = nested[f"{prefix}_threshold"]
        frame["selected_assignment_loss_weight"] = nested[f"{prefix}_loss_weight"]
        frame["selected_at_nested_operating_point"] = (
            frame["two_head_score"] >= frame["nested_operating_threshold"]
        )
    completed = nested["completed_train"]
    learned_labels = nested["train_partition"][completed]
    frozen_labels = training.loc[
        completed,
        "frozen_geometric_partition_label",
    ].to_numpy(dtype=np.int8)
    frozen_masses = pairing_masses(training.loc[completed], frozen_labels)
    learned_masses = pairing_masses(training.loc[completed], learned_labels)
    recorded_masses = np.sort(
        training.loc[completed, ["mbb1", "mbb2"]].to_numpy(dtype=np.float64),
        axis=1,
    )
    require(
        bool(np.allclose(frozen_masses, recorded_masses, rtol=0.0, atol=2.0e-5)),
        "frozen pairing mass reconstruction drift",
    )
    train.loc[completed, "frozen_pair_mass_low"] = frozen_masses[:, 0]
    train.loc[completed, "frozen_pair_mass_high"] = frozen_masses[:, 1]
    train.loc[completed, "learned_pair_mass_low"] = learned_masses[:, 0]
    train.loc[completed, "learned_pair_mass_high"] = learned_masses[:, 1]
    train.loc[completed, "frozen_pair_mean_absolute_mass_residual"] = np.mean(
        np.abs(frozen_masses - 125.0),
        axis=1,
    )
    train.loc[completed, "learned_pair_mean_absolute_mass_residual"] = np.mean(
        np.abs(learned_masses - 125.0),
        axis=1,
    )
    train.loc[completed, "frozen_pair_mean_squared_mass_residual"] = np.mean(
        np.square(frozen_masses - 125.0),
        axis=1,
    )
    train.loc[completed, "learned_pair_mean_squared_mass_residual"] = np.mean(
        np.square(learned_masses - 125.0),
        axis=1,
    )
    return train, project


def point_metrics(
    train: pd.DataFrame,
    project: pd.DataFrame,
    completed: np.ndarray,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    truth = train["truth_partition_label"].to_numpy(dtype=np.int8)
    predicted = train["predicted_partition_label"].to_numpy(dtype=np.int8)
    matchable = train["assignment_loss_mask"].to_numpy(dtype=bool) & completed
    frozen = train["frozen_geometric_partition_label"].to_numpy(dtype=np.int8)
    weights = train["development_hierarchical_weight"].to_numpy(dtype=np.float64)
    assignment = [
        {
            "metric": "learned_exact_event_pairing_accuracy",
            "value": float(np.mean(predicted[matchable] == truth[matchable])),
            "weighting": "unweighted",
        },
        {
            "metric": "learned_exact_event_pairing_accuracy",
            "value": float(np.average(predicted[matchable] == truth[matchable], weights=weights[matchable])),
            "weighting": "development_hierarchical",
        },
        {
            "metric": "frozen_geometric_exact_event_pairing_accuracy",
            "value": float(np.mean(frozen[matchable] == truth[matchable])),
            "weighting": "unweighted",
        },
        {
            "metric": "learned_mass_residual_rms_GeV",
            "value": float(np.sqrt(train.loc[matchable, "learned_pair_mean_squared_mass_residual"].mean())),
            "weighting": "unweighted",
        },
    ]
    label = train["registry_training_target"].to_numpy(dtype=np.int8)[completed]
    score = train["two_head_score"].to_numpy(dtype=np.float64)[completed]
    selected = (
        project["selected_at_nested_operating_point"].to_numpy(dtype=bool)
        & np.isfinite(project["two_head_score"].to_numpy(dtype=np.float64))
    )
    physical = project["primary_projection_physical_weight_inclusive"].to_numpy(dtype=np.float64)
    signal = project["registry_sample_class"].eq("signal").to_numpy()
    signal_yield = float(physical[selected & signal].sum())
    background_yield = float(physical[selected & ~signal].sum())
    classification = [
        {
            "metric": "weighted_auc",
            "value": float(roc_auc_score(label, score, sample_weight=weights[completed])),
        },
        {
            "metric": "unweighted_auc",
            "value": float(roc_auc_score(label, score)),
        },
        {"metric": "selected_signal_yield", "value": signal_yield},
        {"metric": "selected_background_yield", "value": background_yield},
        {"metric": "signal_over_background", "value": signal_yield / background_yield},
    ]
    return assignment, classification


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    require(Path(sys.executable) == Path(TORCH_ENVIRONMENT),
            f"run with required interpreter: {TORCH_ENVIRONMENT}")
    require(CONTRACT.is_file(), "frozen two-head contract missing")
    inputs = load_inputs()
    inputs["evidence"].append({
        "path": str(CONTRACT),
        "bytes": CONTRACT.stat().st_size,
        "sha256": sha256(CONTRACT),
        "role": "frozen_two_head_contract",
    })
    staging = prepare_staging(args.output)
    try:
        nested = train_nested(inputs, staging, args.smoke)
        train, projection = build_prediction_tables(inputs, nested)
        predictions = staging / "predictions"
        predictions.mkdir()
        train.to_parquet(
            predictions / "train_fourb_two_head_oof.parquet",
            index=False,
            compression="zstd",
        )
        projection.to_parquet(
            predictions / "train_primary_projection_two_head_oof.parquet",
            index=False,
            compression="zstd",
        )
        nested["nested_training"].to_parquet(
            predictions / "nested_inner_training_oof.parquet",
            index=False,
            compression="zstd",
        )
        nested["nested_projection"].to_parquet(
            predictions / "nested_inner_projection_oof.parquet",
            index=False,
            compression="zstd",
        )
        assignment, classification = point_metrics(
            train,
            projection,
            nested["completed_train"],
        )
        write_tsv(staging / "assignment_metric_point_estimates.tsv", assignment)
        write_tsv(staging / "classification_metric_point_estimates.tsv", classification)
        write_tsv(staging / "loss_curves.tsv", nested["curves"])
        write_tsv(staging / "gradient_and_loss_balance_diagnostics.tsv", nested["gradients"])
        write_tsv(staging / "inner_loss_weight_candidates.tsv", nested["candidate_rows"])
        write_tsv(staging / "loss_weight_and_operating_point_stability.tsv", nested["selection_rows"])
        write_tsv(staging / "runtime.tsv", nested["runtime_rows"])
        write_tsv(staging / "model_artifact_manifest.tsv", nested["model_rows"])
        write_tsv(staging / "source_evidence_manifest.tsv", inputs["evidence"])
        environment = {
            "python_executable": sys.executable,
            "python": platform.python_version(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "sklearn": sklearn.__version__,
            "torch": torch.__version__,
            "torch_cuda_available": torch.cuda.is_available(),
            "torch_threads": torch.get_num_threads(),
            "device": "cpu",
        }
        write_json(staging / "environment.json", environment)
        summary = {
            "schema_version": 1,
            "status": "pn_c7y_two_head_spanet_smoke_pass" if args.smoke else "pn_c7y_two_head_spanet_training_pass",
            "created_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "scope": "Delphes_train_only_source_group_nested_OOF",
            "architecture": {
                "candidate_jets": 4,
                "jet_features": list(JET_FEATURES),
                "hidden_dim": HIDDEN_DIM,
                "attention_heads": ATTENTION_HEADS,
                "encoder_layers": ENCODER_LAYERS,
                "feedforward_dim": FEEDFORWARD_DIM,
                "dropout": DROPOUT,
                "epochs": 2 if args.smoke else EPOCHS,
                "batch_size": BATCH_SIZE,
                "learning_rate": LEARNING_RATE,
                "weight_decay": WEIGHT_DECAY,
                "negative_jet_mass_numerical_tolerance_GeV": MASS_NEGATIVE_TOLERANCE_GEV,
                "assignment_loss_weights": list(LOSS_WEIGHTS),
                "assignment_head": "shared_symmetric_pair_scorer_three_perfect_matchings",
                "classification_head": "invariant_mean_max_pooling_MLP",
            },
            "target_weighted_signal_efficiency": TARGET_SIGNAL_EFFICIENCY,
            "outer_folds_completed": [int(row["outer_fold"]) for row in nested["selection_rows"]],
            "selected_loss_weights": {
                str(row["outer_fold"]): row["assignment_loss_weight"]
                for row in nested["selection_rows"]
            },
            "assignment_point_metrics": {
                f"{row['metric']}::{row['weighting']}": row["value"]
                for row in assignment
            },
            "classification_point_metrics": {
                row["metric"]: row["value"]
                for row in classification
            },
            "environment": environment,
            "validation_payload_files_opened": 0,
            "test_or_evaluation_payload_files_opened": 0,
            "observed_data_opened": False,
            "full_run2_prediction": False,
            "bootstrap_evaluation_complete": False,
            "categorized_two_head_complete": False,
            "next_gate": "common source-member bootstrap evaluation and separately labelled categorized two-head study",
        }
        write_json(staging / "summary.json", summary)
        (staging / "README.md").write_text(
            "# PN-c7y train-only two-head SPA-Net training\n\n"
            "Nested source-group OOF joint assignment and event-classification network. "
            "Loss weight and operating threshold are selected with inner folds only. "
            "Delphes simulation only; no observed data, validation, or test access.\n"
        )
        (staging / "RUN_CONTRACT.txt").write_text(
            "The two-head architecture, optimizer, loss candidates, and joint selection utility "
            "were frozen before fitting. Every supported row contributes classification loss; "
            "only uniquely matchable signal contributes assignment loss. Outer-held sources do "
            "not select loss weight or operating threshold.\n"
        )
        write_tsv(
            staging / "artifact_manifest.tsv",
            artifact_rows(staging, exclude={"artifact_manifest.tsv", "SHA256SUMS", "COMPLETE"}),
        )
        manifest_sha = seal_checkpoint(
            staging,
            args.output,
            "pn_c7y_two_head_spanet_smoke_pass"
            if args.smoke
            else "pn_c7y_two_head_spanet_training_pass",
        )
        print(json.dumps({
            "checkpoint": str(args.output),
            "manifest_sha256": manifest_sha,
            **summary,
        }, sort_keys=True), flush=True)
    except Exception:
        write_json(
            staging / "FAILED.json",
            {"status": "pn_c7y_two_head_training_failed_preserved_staging"},
        )
        raise


if __name__ == "__main__":
    main()
