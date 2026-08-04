#!/usr/bin/env python3
"""Lazy-torch candidate-four assignment model for PN-c7x."""

from __future__ import annotations

import math
import time
from typing import Any

import numpy as np
import pandas as pd

from scripts.analysis.pn_c7x_spanet_common import PERFECT_MATCHINGS, require


TORCH_ENVIRONMENT = "/tmp/hh4b_pn_c7t_baseline_env_20260802_v1/bin/python"
JET_FEATURES = ("log1p_pt", "eta", "sin_phi", "cos_phi", "log1p_mass")
HIDDEN_DIM = 64
ATTENTION_HEADS = 4
ENCODER_LAYERS = 2
FEEDFORWARD_DIM = 256
DROPOUT = 0.05
EPOCHS = 24
BATCH_SIZE = 512
LEARNING_RATE = 1.0e-3
WEIGHT_DECAY = 1.0e-4
MASS_NEGATIVE_TOLERANCE_GEV = 2.0e-6


def require_torch():
    try:
        import torch
    except (ModuleNotFoundError, ImportError) as exc:
        raise RuntimeError(
            "PN-c7x assignment training requires torch in " + TORCH_ENVIRONMENT
        ) from exc
    return torch


def jet_feature_array(frame: pd.DataFrame) -> np.ndarray:
    values = []
    for position in range(1, 5):
        pt = frame[f"j{position}_pt"].to_numpy(dtype=np.float64)
        eta = frame[f"j{position}_eta"].to_numpy(dtype=np.float64)
        phi = frame[f"j{position}_phi"].to_numpy(dtype=np.float64)
        mass = frame[f"j{position}_mass"].to_numpy(dtype=np.float64)
        require(bool((pt >= 0.0).all()), f"negative j{position} pT")
        require(bool((mass >= -MASS_NEGATIVE_TOLERANCE_GEV).all()), f"j{position} mass below frozen numerical tolerance")
        mass = np.maximum(mass, 0.0)
        values.append(np.column_stack((np.log1p(pt), eta, np.sin(phi), np.cos(phi), np.log1p(mass))))
    result = np.stack(values, axis=1).astype(np.float32)
    require(result.shape == (len(frame), 4, 5), "jet feature shape drift")
    require(bool(np.isfinite(result).all()), "nonfinite jet input")
    return result


def fit_normalizer(frame: pd.DataFrame) -> dict[str, np.ndarray]:
    values = jet_feature_array(frame).reshape(-1, len(JET_FEATURES)).astype(np.float64)
    mean = values.mean(axis=0)
    scale = values.std(axis=0)
    require(bool(np.isfinite(mean).all() and np.isfinite(scale).all()), "nonfinite jet normalizer")
    require(bool((scale > 1.0e-8).all()), "constant jet input in normalizer")
    return {"mean": mean.astype(np.float32), "scale": scale.astype(np.float32)}


def normalize_jets(frame: pd.DataFrame, normalizer: dict[str, np.ndarray]) -> np.ndarray:
    mean = np.asarray(normalizer["mean"], dtype=np.float32)
    scale = np.asarray(normalizer["scale"], dtype=np.float32)
    require(mean.shape == (5,) and scale.shape == (5,), "jet normalizer shape drift")
    result = (jet_feature_array(frame) - mean[None, None, :]) / scale[None, None, :]
    require(bool(np.isfinite(result).all()), "nonfinite normalized jet input")
    return result.astype(np.float32)


def build_model(seed: int):
    torch = require_torch()
    torch.manual_seed(int(seed))

    class CandidateFourAssignmentNet(torch.nn.Module):
        def __init__(self):
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
            self.encoder = torch.nn.TransformerEncoder(encoder_layer, num_layers=ENCODER_LAYERS)
            self.final_norm = torch.nn.LayerNorm(HIDDEN_DIM)
            self.pair_scorer = torch.nn.Sequential(
                torch.nn.Linear(3 * HIDDEN_DIM, HIDDEN_DIM),
                torch.nn.GELU(),
                torch.nn.Linear(HIDDEN_DIM, 1),
            )

        def forward(self, jets):
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
            logits = torch.stack(
                [pair_scores[first] + pair_scores[second] for first, second in PERFECT_MATCHINGS],
                dim=1,
            )
            return logits, embedding

    return CandidateFourAssignmentNet()


def matching_label_permutation(permutation: tuple[int, int, int, int]) -> np.ndarray:
    """Map each new-index matching to its old-index matching after jet permutation."""
    require(sorted(permutation) == [0, 1, 2, 3], "invalid four-jet permutation")
    lookup = {matching: label for label, matching in enumerate(PERFECT_MATCHINGS)}
    mapping = []
    for matching in PERFECT_MATCHINGS:
        old_pairs = [tuple(sorted((permutation[left], permutation[right]))) for left, right in matching]
        old_matching = tuple(sorted(old_pairs))
        mapping.append(lookup[old_matching])
    return np.asarray(mapping, dtype=np.int64)


def configure_torch(seed: int, threads: int = 4) -> Any:
    torch = require_torch()
    torch.set_num_threads(threads)
    torch.manual_seed(int(seed))
    np.random.seed(int(seed) % (2**32 - 1))
    torch.use_deterministic_algorithms(True)
    return torch


def train_assignment_model(
    frame: pd.DataFrame,
    labels: np.ndarray,
    weights: np.ndarray,
    normalizer: dict[str, np.ndarray],
    *,
    seed: int,
    smoke: bool = False,
) -> tuple[Any, list[dict[str, float]], float]:
    torch = configure_torch(seed)
    inputs = normalize_jets(frame, normalizer)
    labels = np.asarray(labels, dtype=np.int64)
    weights = np.asarray(weights, dtype=np.float64)
    require(len(frame) > 0 and len(labels) == len(frame) and len(weights) == len(frame), "assignment training length drift")
    require(set(np.unique(labels)).issubset({0, 1, 2}), "assignment label outside perfect matchings")
    require(bool(np.isfinite(weights).all() and (weights > 0.0).all()), "invalid assignment weight")
    normalized_weights = (weights / weights.mean()).astype(np.float32)
    dataset = torch.utils.data.TensorDataset(
        torch.from_numpy(inputs),
        torch.from_numpy(labels),
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
    model = build_model(seed)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    epochs = 2 if smoke else EPOCHS
    curve = []
    started = time.perf_counter()
    model.train()
    for epoch in range(epochs):
        weighted_loss_sum = 0.0
        weight_sum = 0.0
        correct_weight = 0.0
        for batch_inputs, batch_labels, batch_weights in loader:
            optimizer.zero_grad(set_to_none=True)
            logits, _ = model(batch_inputs)
            losses = torch.nn.functional.cross_entropy(logits, batch_labels, reduction="none")
            loss = torch.sum(losses * batch_weights) / torch.sum(batch_weights)
            loss.backward()
            optimizer.step()
            weighted_loss_sum += float(torch.sum(losses.detach() * batch_weights).item())
            weight_sum += float(torch.sum(batch_weights).item())
            correct_weight += float(torch.sum((logits.detach().argmax(1) == batch_labels) * batch_weights).item())
        curve.append({
            "epoch": epoch + 1,
            "weighted_assignment_loss": weighted_loss_sum / weight_sum,
            "weighted_training_accuracy": correct_weight / weight_sum,
        })
    elapsed = time.perf_counter() - started
    return model, curve, elapsed


def invariant_features(
    model: Any,
    frame: pd.DataFrame,
    normalizer: dict[str, np.ndarray],
    *,
    batch_size: int = 2048,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    torch = require_torch()
    inputs = normalize_jets(frame, normalizer)
    features = []
    probabilities = []
    model.eval()
    with torch.no_grad():
        for start in range(0, len(inputs), batch_size):
            logits, embedding = model(torch.from_numpy(inputs[start : start + batch_size]))
            probability = torch.softmax(logits, dim=1)
            sorted_probability, _ = torch.sort(probability, dim=1)
            entropy = -torch.sum(probability * torch.log(torch.clamp(probability, min=1.0e-12)), dim=1, keepdim=True)
            event = torch.cat(
                (
                    torch.mean(embedding, dim=1),
                    torch.amax(embedding, dim=1),
                    sorted_probability,
                    torch.amax(probability, dim=1, keepdim=True),
                    entropy,
                ),
                dim=1,
            )
            features.append(event.cpu().numpy())
            probabilities.append(probability.cpu().numpy())
    feature_array = np.concatenate(features).astype(np.float32)
    probability_array = np.concatenate(probabilities).astype(np.float32)
    require(feature_array.shape == (len(frame), 2 * HIDDEN_DIM + 5), "invariant feature width drift")
    prediction = probability_array.argmax(axis=1).astype(np.int8)
    maximum = probability_array.max(axis=1).astype(np.float32)
    entropy = (-probability_array * np.log(np.clip(probability_array, 1.0e-12, None))).sum(axis=1).astype(np.float32)
    return feature_array, prediction, maximum, entropy


def parameter_count(model: Any) -> int:
    return int(sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad))
