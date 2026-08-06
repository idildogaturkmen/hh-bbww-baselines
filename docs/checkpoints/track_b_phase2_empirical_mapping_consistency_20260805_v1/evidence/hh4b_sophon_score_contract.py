#!/usr/bin/env python3

from __future__ import annotations

from collections import Counter
from typing import Dict, List, Optional, Sequence

import numpy as np

from hh4b_sophon_contract import (
    OUTPUT_CLASS_COUNT,
    QCD_CLASS_INDEX,
    SIGNAL_CLASS_COUNT,
    TTBAR_CLASS_INDEX,
    class_index_to_mass_bins,
)


DEFAULT_SIGNAL_THRESHOLDS = (
    0.5,
    0.9,
    0.997,
)


def validate_score_matrix(
    scores: np.ndarray,
    *,
    name: str = "scores",
    normalization_atol: float = 1.0e-4,
) -> np.ndarray:
    """Validate one SophonHH score matrix and return float64 values."""

    array = np.asarray(
        scores,
        dtype=np.float64,
    )

    if array.ndim != 2:
        raise ValueError(
            "{} must have shape [N, {}], got {}".format(
                name,
                OUTPUT_CLASS_COUNT,
                array.shape,
            )
        )

    if array.shape[0] <= 0:
        raise ValueError(
            "{} must contain at least one event".format(
                name
            )
        )

    if array.shape[1] != OUTPUT_CLASS_COUNT:
        raise ValueError(
            "{} must have {} classes, got {}".format(
                name,
                OUTPUT_CLASS_COUNT,
                array.shape[1],
            )
        )

    if not np.all(
        np.isfinite(array)
    ):
        raise ValueError(
            "{} contains non-finite values".format(
                name
            )
        )

    if np.any(
        array < -normalization_atol
    ):
        raise ValueError(
            "{} contains probabilities below zero".format(
                name
            )
        )

    if np.any(
        array > 1.0 + normalization_atol
    ):
        raise ValueError(
            "{} contains probabilities above one".format(
                name
            )
        )

    row_sums = np.sum(
        array,
        axis=1,
    )

    if not np.allclose(
        row_sums,
        1.0,
        rtol=0.0,
        atol=normalization_atol,
    ):
        maximum_error = float(
            np.max(
                np.abs(
                    row_sums - 1.0
                )
            )
        )

        raise ValueError(
            "{} rows are not normalized; maximum absolute "
            "sum error is {:.8g}".format(
                name,
                maximum_error,
            )
        )

    return array


def ensemble_score_matrices(
    model_scores: Sequence[np.ndarray],
    *,
    normalization_atol: float = 1.0e-4,
) -> np.ndarray:
    """Validate and average score matrices from multiple models."""

    if not model_scores:
        raise ValueError(
            "model_scores must contain at least one matrix"
        )

    validated = []

    for model_index, scores in enumerate(
        model_scores
    ):
        validated.append(
            validate_score_matrix(
                scores,
                name="model_scores[{}]".format(
                    model_index
                ),
                normalization_atol=normalization_atol,
            )
        )

    reference_shape = validated[0].shape

    for model_index, scores in enumerate(
        validated[1:],
        start=1,
    ):
        if scores.shape != reference_shape:
            raise ValueError(
                "model_scores[{}] has shape {}, expected {}".format(
                    model_index,
                    scores.shape,
                    reference_shape,
                )
            )

    ensemble = np.mean(
        np.stack(
            validated,
            axis=0,
        ),
        axis=0,
    )

    return validate_score_matrix(
        ensemble,
        name="ensemble_scores",
        normalization_atol=normalization_atol,
    )


def score_components(
    scores: np.ndarray,
) -> Dict[str, np.ndarray]:
    """Return inclusive signal, QCD, and ttbar probabilities."""

    array = validate_score_matrix(
        scores
    )

    return {
        "signal": np.sum(
            array[
                :,
                :SIGNAL_CLASS_COUNT,
            ],
            axis=1,
        ),
        "qcd": array[
            :,
            QCD_CLASS_INDEX,
        ],
        "ttbar": array[
            :,
            TTBAR_CLASS_INDEX,
        ],
    }


def model_top1_agreement_fraction(
    model_scores: Sequence[np.ndarray],
) -> float:
    """Return the fraction of events with identical top-1 classes."""

    if not model_scores:
        raise ValueError(
            "model_scores must contain at least one matrix"
        )

    validated = [
        validate_score_matrix(
            scores,
            name="model_scores[{}]".format(
                model_index
            ),
        )
        for model_index, scores in enumerate(
            model_scores
        )
    ]

    reference_shape = validated[0].shape

    for model_index, scores in enumerate(
        validated[1:],
        start=1,
    ):
        if scores.shape != reference_shape:
            raise ValueError(
                "model_scores[{}] has shape {}, expected {}".format(
                    model_index,
                    scores.shape,
                    reference_shape,
                )
            )

    top1 = np.stack(
        [
            np.argmax(
                scores,
                axis=1,
            )
            for scores in validated
        ],
        axis=0,
    )

    return float(
        np.mean(
            np.all(
                top1 == top1[0:1],
                axis=0,
            )
        )
    )


def summarize_score_matrix(
    scores: np.ndarray,
    *,
    signal_thresholds: Sequence[float] = DEFAULT_SIGNAL_THRESHOLDS,
) -> Dict[str, object]:
    """Return JSON-serializable inclusive score diagnostics."""

    array = validate_score_matrix(
        scores
    )

    components = score_components(
        array
    )

    signal = components["signal"]
    qcd = components["qcd"]
    ttbar = components["ttbar"]

    threshold_summary: Dict[
        str,
        Dict[str, float],
    ] = {}

    for threshold in signal_thresholds:
        threshold_value = float(
            threshold
        )

        if (
            not np.isfinite(
                threshold_value
            )
            or threshold_value < 0.0
            or threshold_value > 1.0
        ):
            raise ValueError(
                "signal threshold must be finite and in [0, 1], "
                "got {}".format(
                    threshold
                )
            )

        selected = int(
            np.count_nonzero(
                signal > threshold_value
            )
        )

        threshold_summary[
            "{:.6g}".format(
                threshold_value
            )
        ] = {
            "count": selected,
            "fraction": float(
                selected / len(signal)
            ),
        }

    top1_counts = Counter(
        int(index)
        for index in np.argmax(
            array,
            axis=1,
        )
    )

    return {
        "n_events": int(
            len(array)
        ),
        "mean_signal_probability": float(
            np.mean(signal)
        ),
        "mean_qcd_probability": float(
            np.mean(qcd)
        ),
        "mean_ttbar_probability": float(
            np.mean(ttbar)
        ),
        "signal_probability_quantiles": {
            "q10": float(
                np.quantile(
                    signal,
                    0.10,
                )
            ),
            "median": float(
                np.quantile(
                    signal,
                    0.50,
                )
            ),
            "q90": float(
                np.quantile(
                    signal,
                    0.90,
                )
            ),
        },
        "signal_thresholds_strict_greater_than": threshold_summary,
        "top1_class_counts": {
            str(class_index): int(count)
            for class_index, count in sorted(
                top1_counts.items()
            )
        },
    }


def summarize_signal_grid(
    scores: np.ndarray,
    *,
    minimum_signal_probability: Optional[float] = None,
    top_k: int = 10,
) -> Dict[str, object]:
    """Summarize mass-grid response, optionally after a signal-score cut."""

    if top_k <= 0:
        raise ValueError(
            "top_k must be positive"
        )

    array = validate_score_matrix(
        scores
    )

    signal_scores = array[
        :,
        :SIGNAL_CLASS_COUNT,
    ]

    signal_probability = np.sum(
        signal_scores,
        axis=1,
    )

    selection = np.ones(
        len(array),
        dtype=np.bool_,
    )

    if minimum_signal_probability is not None:
        threshold = float(
            minimum_signal_probability
        )

        if (
            not np.isfinite(threshold)
            or threshold < 0.0
            or threshold > 1.0
        ):
            raise ValueError(
                "minimum_signal_probability must be finite and "
                "in [0, 1]"
            )

        selection = (
            signal_probability > threshold
        )

    selected_scores = signal_scores[
        selection
    ]

    selected_signal_probability = signal_probability[
        selection
    ]

    if len(selected_scores) == 0:
        return {
            "n_selected_events": 0,
            "minimum_signal_probability_strict_greater_than": (
                None
                if minimum_signal_probability is None
                else float(
                    minimum_signal_probability
                )
            ),
            "top_cells_by_mean_raw_probability": [],
            "top_cells_by_mean_signal_normalized_probability": [],
        }

    raw_profile = np.mean(
        selected_scores,
        axis=0,
    )

    normalized_scores = np.divide(
        selected_scores,
        selected_signal_probability[
            :,
            np.newaxis,
        ],
        out=np.zeros_like(
            selected_scores
        ),
        where=selected_signal_probability[
            :,
            np.newaxis,
        ] > 0.0,
    )

    normalized_profile = np.mean(
        normalized_scores,
        axis=0,
    )

    def top_cells(
        profile: np.ndarray,
    ) -> List[Dict[str, object]]:
        indices = np.argsort(
            profile
        )[
            ::-1
        ][
            :min(
                top_k,
                SIGNAL_CLASS_COUNT,
            )
        ]

        rows = []

        for class_index in indices:
            first_bin, second_bin = class_index_to_mass_bins(
                int(
                    class_index
                )
            )

            rows.append(
                {
                    "class_index": int(
                        class_index
                    ),
                    "first_mass_bin": [
                        float(first_bin[0]),
                        float(first_bin[1]),
                    ],
                    "second_mass_bin": [
                        float(second_bin[0]),
                        float(second_bin[1]),
                    ],
                    "mean_probability": float(
                        profile[
                            class_index
                        ]
                    ),
                }
            )

        return rows

    return {
        "n_selected_events": int(
            len(selected_scores)
        ),
        "minimum_signal_probability_strict_greater_than": (
            None
            if minimum_signal_probability is None
            else float(
                minimum_signal_probability
            )
        ),
        "top_cells_by_mean_raw_probability": top_cells(
            raw_profile
        ),
        "top_cells_by_mean_signal_normalized_probability": top_cells(
            normalized_profile
        ),
    }
