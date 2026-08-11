#!/usr/bin/env python3
"""Small, side-effect-free contracts for the apples-to-apples BDT study."""

from __future__ import annotations

import numpy as np


def nested_masks(folds: np.ndarray, outer_fold: int, inner_fold: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return fit, inner-held, and outer-held masks from immutable fold labels."""
    folds = np.asarray(folds)
    if outer_fold == inner_fold:
        raise ValueError("outer and inner fold must differ")
    outer = folds == outer_fold
    inner = folds == inner_fold
    fit = ~(outer | inner)
    if np.any(fit & inner) or np.any(fit & outer) or np.any(inner & outer):
        raise AssertionError("nested partitions overlap")
    return fit, inner, outer


def validate_feature_contract(mass_aware: list[str], mass_plane_blind: list[str], exclusions: list[str]) -> None:
    """Enforce deterministic ordering and the exact four-variable ablation."""
    if len(mass_aware) != len(set(mass_aware)) or len(mass_plane_blind) != len(set(mass_plane_blind)):
        raise ValueError("duplicate feature")
    if mass_aware != mass_plane_blind + exclusions:
        raise ValueError("mass-aware ordering or exclusion contract changed")
    if any(name in mass_plane_blind for name in exclusions):
        raise ValueError("explicit mass-plane feature leaked into blind contract")


def validate_training_weights(weights: np.ndarray) -> None:
    """Reject signed, zero, or nonfinite learner weights."""
    values = np.asarray(weights, dtype=float)
    if values.ndim != 1 or not np.isfinite(values).all() or np.any(values <= 0.0):
        raise ValueError("training weights must be finite, one-dimensional, and positive")


def select_fixed_efficiency_threshold(scores: np.ndarray, labels: np.ndarray, weights: np.ndarray, target: float) -> float:
    """Select a deterministic signal-efficiency threshold using development rows."""
    scores = np.asarray(scores, dtype=float)
    labels = np.asarray(labels, dtype=int)
    weights = np.asarray(weights, dtype=float)
    if not (scores.shape == labels.shape == weights.shape) or not np.isfinite(scores).all():
        raise ValueError("invalid development arrays")
    validate_training_weights(weights)
    signal = labels == 1
    if not np.any(signal) or not 0.0 < target < 1.0:
        raise ValueError("invalid signal target")
    candidates = np.unique(scores[signal])
    total = weights[signal].sum()
    ranked = []
    for threshold in candidates:
        efficiency = weights[signal & (scores >= threshold)].sum() / total
        ranked.append((abs(efficiency - target), efficiency < target, -threshold, threshold))
    return float(min(ranked)[3])


def historical_rhh34_mask(r_hh_125_125: np.ndarray, universe_mask: np.ndarray) -> np.ndarray:
    """Apply the frozen strict comparator without changing its universe."""
    values = np.asarray(r_hh_125_125, dtype=float)
    universe = np.asarray(universe_mask, dtype=bool)
    if values.shape != universe.shape or not np.isfinite(values[universe]).all():
        raise ValueError("invalid comparator input")
    return universe & (values < 34.0)
