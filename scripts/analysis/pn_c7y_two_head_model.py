#!/usr/bin/env python3
"""Lazy-torch joint assignment and event-classification model for PN-c7y."""

from __future__ import annotations

import math
import time
from typing import Any

import numpy as np
import pandas as pd

from scripts.analysis.pn_c7x_spanet_common import PERFECT_MATCHINGS, require
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
    TORCH_ENVIRONMENT,
    WEIGHT_DECAY,
    configure_torch,
    normalize_jets,
    require_torch,
)


LOSS_WEIGHTS = (0.25, 1.0, 4.0)


def loss_weight_index(value: float) -> int:
    require(value in LOSS_WEIGHTS, "unknown loss weight")
    return LOSS_WEIGHTS.index(value)


def select_loss_weight(rows: list[dict[str, Any]]) -> dict[str, Any]:
    require(len(rows) == len(LOSS_WEIGHTS), "loss-weight selection inventory drift")
    require(
        {float(row["assignment_loss_weight"]) for row in rows} == set(LOSS_WEIGHTS),
        "loss-weight candidate drift",
    )
    return sorted(
        rows,
        key=lambda row: (
            -float(row["joint_selection_utility"]),
            float(row["assignment_loss_weight"]),
        ),
    )[0]


def build_two_head_model(seed: int) -> Any:
    torch = require_torch()
    torch.manual_seed(int(seed))

    class CandidateFourTwoHeadNet(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.input_projection = torch.nn.Linear(len(JET_FEATURES), HIDDEN_DIM)
            encoder_layer = torch.nn.TransformerEncoderLayer(
                d_model=HIDDEN_DIM,
                nhead=ATTENTION_HEADS,
                dim_feedforward=FEEDFORWARD_DIM,
                dropout=DROPOUT,
                activation="gelu",
                batch_first=True,
                norm_first=True,
            )
            self.encoder = torch.nn.TransformerEncoder(
                encoder_layer,
                num_layers=ENCODER_LAYERS,
            )
            self.final_norm = torch.nn.LayerNorm(HIDDEN_DIM)
            self.pair_scorer = torch.nn.Sequential(
                torch.nn.Linear(3 * HIDDEN_DIM, HIDDEN_DIM),
                torch.nn.GELU(),
                torch.nn.Linear(HIDDEN_DIM, 1),
            )
            self.classification_head = torch.nn.Sequential(
                torch.nn.Linear(2 * HIDDEN_DIM, HIDDEN_DIM),
                torch.nn.GELU(),
                torch.nn.Dropout(DROPOUT),
                torch.nn.Linear(HIDDEN_DIM, 1),
            )

        def forward(self, jets: Any) -> tuple[Any, Any, Any]:
            embedding = self.final_norm(self.encoder(self.input_projection(jets)))
            pair_scores = {}
            for left in range(4):
                for right in range(left + 1, 4):
                    pair_features = torch.cat(
                        (
                            embedding[:, left] + embedding[:, right],
                            torch.abs(embedding[:, left] - embedding[:, right]),
                            embedding[:, left] * embedding[:, right],
                        ),
                        dim=1,
                    )
                    pair_scores[(left, right)] = self.pair_scorer(pair_features).squeeze(1)
            assignment_logits = torch.stack(
                [pair_scores[first] + pair_scores[second] for first, second in PERFECT_MATCHINGS],
                dim=1,
            )
            event_features = torch.cat(
                (torch.mean(embedding, dim=1), torch.amax(embedding, dim=1)),
                dim=1,
            )
            classification_logit = self.classification_head(event_features).squeeze(1)
            return assignment_logits, classification_logit, embedding

    return CandidateFourTwoHeadNet()


def _gradient_norm(torch: Any, gradients: tuple[Any, ...]) -> float:
    squared = torch.zeros((), dtype=torch.float32)
    for gradient in gradients:
        if gradient is not None:
            squared = squared + torch.sum(gradient.detach() ** 2)
    return float(torch.sqrt(squared).item())


def gradient_diagnostics(
    model: Any,
    frame: pd.DataFrame,
    class_labels: np.ndarray,
    assignment_labels: np.ndarray,
    assignment_mask: np.ndarray,
    weights: np.ndarray,
    normalizer: dict[str, np.ndarray],
) -> dict[str, float]:
    torch = require_torch()
    assignment_mask = np.asarray(assignment_mask, dtype=bool)
    supervised = np.flatnonzero(assignment_mask)
    other = np.flatnonzero(~assignment_mask)
    supervised_limit = min(1024, len(supervised))
    other_limit = min(2048 - supervised_limit, len(other))
    indices = np.concatenate((supervised[:supervised_limit], other[:other_limit]))
    require(len(indices) > 0 and supervised_limit > 0, "gradient diagnostic sample is empty")
    inputs = torch.from_numpy(normalize_jets(frame.iloc[indices], normalizer))
    class_target = torch.from_numpy(np.asarray(class_labels[indices], dtype=np.float32))
    assignment_target = torch.from_numpy(
        np.where(assignment_mask[indices], assignment_labels[indices], 0).astype(np.int64)
    )
    mask = torch.from_numpy(assignment_mask[indices])
    row_weight = torch.from_numpy(
        (
            np.asarray(weights[indices], dtype=np.float64)
            / np.mean(weights[indices])
        ).astype(np.float32)
    )
    require(bool(mask.any()), "gradient diagnostic batch has no matchable signal")
    model.eval()
    assignment_logits, class_logit, _ = model(inputs)
    class_losses = torch.nn.functional.binary_cross_entropy_with_logits(
        class_logit,
        class_target,
        reduction="none",
    )
    class_loss = torch.sum(class_losses * row_weight) / torch.sum(row_weight)
    assignment_losses = torch.nn.functional.cross_entropy(
        assignment_logits[mask],
        assignment_target[mask],
        reduction="none",
    )
    assignment_loss = (
        torch.sum(assignment_losses * row_weight[mask]) / torch.sum(row_weight[mask])
    )
    parameters = tuple(parameter for parameter in model.parameters() if parameter.requires_grad)
    class_gradients = torch.autograd.grad(
        class_loss,
        parameters,
        retain_graph=True,
        allow_unused=True,
    )
    assignment_gradients = torch.autograd.grad(
        assignment_loss,
        parameters,
        allow_unused=True,
    )
    return {
        "diagnostic_classification_loss": float(class_loss.item()),
        "diagnostic_assignment_loss": float(assignment_loss.item()),
        "classification_gradient_l2": _gradient_norm(torch, class_gradients),
        "assignment_gradient_l2": _gradient_norm(torch, assignment_gradients),
    }


def train_two_head_model(
    frame: pd.DataFrame,
    class_labels: np.ndarray,
    assignment_labels: np.ndarray,
    assignment_mask: np.ndarray,
    weights: np.ndarray,
    normalizer: dict[str, np.ndarray],
    *,
    assignment_loss_weight: float,
    seed: int,
    smoke: bool = False,
) -> tuple[Any, list[dict[str, float]], dict[str, float], float]:
    require(assignment_loss_weight in LOSS_WEIGHTS, "unfrozen assignment-loss weight")
    torch = configure_torch(seed)
    inputs = normalize_jets(frame, normalizer)
    class_labels = np.asarray(class_labels, dtype=np.float32)
    assignment_labels = np.asarray(assignment_labels, dtype=np.int64)
    assignment_mask = np.asarray(assignment_mask, dtype=bool)
    weights = np.asarray(weights, dtype=np.float64)
    require(
        len(frame)
        == len(class_labels)
        == len(assignment_labels)
        == len(assignment_mask)
        == len(weights),
        "two-head training length drift",
    )
    require(set(np.unique(class_labels)) == {0.0, 1.0}, "classification labels drift")
    require(bool(assignment_mask.any()), "two-head training has no assignment supervision")
    require(
        set(np.unique(assignment_labels[assignment_mask])).issubset({0, 1, 2}),
        "assignment label outside perfect matchings",
    )
    require(bool(np.isfinite(weights).all() and (weights > 0.0).all()), "invalid training weight")
    normalized_weights = (weights / weights.mean()).astype(np.float32)
    safe_assignment_labels = np.where(assignment_mask, assignment_labels, 0).astype(np.int64)
    dataset = torch.utils.data.TensorDataset(
        torch.from_numpy(inputs),
        torch.from_numpy(class_labels),
        torch.from_numpy(safe_assignment_labels),
        torch.from_numpy(assignment_mask),
        torch.from_numpy(normalized_weights),
    )
    generator = torch.Generator().manual_seed(int(seed))
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=min(BATCH_SIZE, len(dataset)),
        shuffle=True,
        generator=generator,
        num_workers=0,
        drop_last=False,
    )
    model = build_two_head_model(seed)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )
    epochs = 2 if smoke else EPOCHS
    curve: list[dict[str, float]] = []
    started = time.perf_counter()
    model.train()
    for epoch in range(epochs):
        class_loss_sum = 0.0
        class_weight_sum = 0.0
        assignment_loss_sum = 0.0
        assignment_weight_sum = 0.0
        class_correct_sum = 0.0
        assignment_correct_sum = 0.0
        for batch_inputs, batch_class, batch_assignment, batch_mask, batch_weights in loader:
            optimizer.zero_grad(set_to_none=True)
            assignment_logits, class_logit, _ = model(batch_inputs)
            class_losses = torch.nn.functional.binary_cross_entropy_with_logits(
                class_logit,
                batch_class,
                reduction="none",
            )
            class_loss = torch.sum(class_losses * batch_weights) / torch.sum(batch_weights)
            if bool(batch_mask.any()):
                assignment_losses = torch.nn.functional.cross_entropy(
                    assignment_logits[batch_mask],
                    batch_assignment[batch_mask],
                    reduction="none",
                )
                assignment_loss = (
                    torch.sum(assignment_losses * batch_weights[batch_mask])
                    / torch.sum(batch_weights[batch_mask])
                )
                assignment_loss_sum += float(
                    torch.sum(assignment_losses.detach() * batch_weights[batch_mask]).item()
                )
                assignment_weight_sum += float(torch.sum(batch_weights[batch_mask]).item())
                assignment_correct_sum += float(
                    torch.sum(
                        (assignment_logits.detach()[batch_mask].argmax(1) == batch_assignment[batch_mask])
                        * batch_weights[batch_mask]
                    ).item()
                )
            else:
                assignment_loss = torch.sum(assignment_logits) * 0.0
            combined_loss = class_loss + assignment_loss_weight * assignment_loss
            combined_loss.backward()
            optimizer.step()
            class_loss_sum += float(torch.sum(class_losses.detach() * batch_weights).item())
            class_weight_sum += float(torch.sum(batch_weights).item())
            class_correct_sum += float(
                torch.sum(
                    ((class_logit.detach() >= 0.0) == (batch_class >= 0.5)) * batch_weights
                ).item()
            )
        mean_class_loss = class_loss_sum / class_weight_sum
        mean_assignment_loss = assignment_loss_sum / assignment_weight_sum
        curve.append({
            "epoch": float(epoch + 1),
            "assignment_loss_weight": float(assignment_loss_weight),
            "weighted_classification_loss": mean_class_loss,
            "weighted_assignment_loss": mean_assignment_loss,
            "weighted_combined_loss": mean_class_loss + assignment_loss_weight * mean_assignment_loss,
            "weighted_classification_accuracy": class_correct_sum / class_weight_sum,
            "weighted_assignment_accuracy": assignment_correct_sum / assignment_weight_sum,
            "weighted_assignment_to_classification_loss_ratio": (
                assignment_loss_weight * mean_assignment_loss / mean_class_loss
            ),
        })
    elapsed = time.perf_counter() - started
    diagnostics = gradient_diagnostics(
        model,
        frame,
        class_labels,
        assignment_labels,
        assignment_mask,
        weights,
        normalizer,
    )
    diagnostics["assignment_loss_weight"] = float(assignment_loss_weight)
    diagnostics["weighted_assignment_to_classification_gradient_ratio"] = (
        assignment_loss_weight
        * diagnostics["assignment_gradient_l2"]
        / diagnostics["classification_gradient_l2"]
    )
    return model, curve, diagnostics, elapsed


def predict_two_head(
    model: Any,
    frame: pd.DataFrame,
    normalizer: dict[str, np.ndarray],
    *,
    batch_size: int = 2048,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    torch = require_torch()
    inputs = normalize_jets(frame, normalizer)
    scores = []
    probabilities = []
    model.eval()
    with torch.no_grad():
        for start in range(0, len(inputs), batch_size):
            assignment_logits, class_logit, _ = model(
                torch.from_numpy(inputs[start : start + batch_size])
            )
            scores.append(torch.sigmoid(class_logit).cpu().numpy())
            probabilities.append(torch.softmax(assignment_logits, dim=1).cpu().numpy())
    score = np.concatenate(scores).astype(np.float64)
    probability = np.concatenate(probabilities).astype(np.float32)
    require(score.shape == (len(frame),), "two-head score shape drift")
    require(probability.shape == (len(frame), 3), "two-head assignment probability shape drift")
    require(bool(np.isfinite(score).all() and np.isfinite(probability).all()), "nonfinite two-head prediction")
    prediction = probability.argmax(axis=1).astype(np.int8)
    maximum = probability.max(axis=1).astype(np.float32)
    entropy = (
        -probability * np.log(np.clip(probability, 1.0e-12, None))
    ).sum(axis=1).astype(np.float32)
    return score, prediction, maximum, entropy


def parameter_count(model: Any) -> int:
    return int(
        sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    )


def joint_selection_utility(weighted_auc: float, exact_pairing_accuracy: float) -> float:
    require(
        math.isfinite(weighted_auc) and math.isfinite(exact_pairing_accuracy),
        "nonfinite inner selection metric",
    )
    require(0.0 <= weighted_auc <= 1.0 and 0.0 <= exact_pairing_accuracy <= 1.0,
            "inner selection metric outside unit interval")
    return 0.5 * (weighted_auc + exact_pairing_accuracy)
