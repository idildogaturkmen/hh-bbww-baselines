#!/usr/bin/env python3

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import hashlib
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple, Union

import awkward as ak
import numpy as np
import uproot

from hh4b_sophon_contract import (
    FEATURE_NAMES,
    MAX_PARTICLES,
    OUTPUT_CLASS_COUNT,
    VECTOR_NAMES,
    build_sophon_inputs,
)

from hh4b_sophon_root_contract import (
    INPUT_PARTICLE_BRANCHES,
    INPUT_SCALAR_BRANCHES,
    INPUT_TREE_NAME,
    SELECTION_BRANCHES,
    eligible_entry_indices,
    spread_selected_entries,
    validate_input_tree,
    write_events_tree,
)

from hh4b_sophon_score_contract import (
    ensemble_score_matrices,
    model_top1_agreement_fraction,
    summarize_score_matrix,
    summarize_signal_grid,
    validate_score_matrix,
)


PathLike = Union[str, Path]

MODEL_INPUT_NAMES = (
    "pf_features",
    "pf_vectors",
    "pf_mask",
)

MODEL_OUTPUT_NAME = "softmax"
RELEASE_MODEL_COUNT = 3
EXECUTION_BACKEND_SCHEMA_VERSION = 1


@dataclass(
    frozen=True
)
class PreparedBatch:
    model_inputs: Dict[str, np.ndarray]
    selected_entries: np.ndarray
    pass_selection: np.ndarray
    pass_4j3b_selection: np.ndarray
    metadata: Dict[str, Any]


@dataclass(
    frozen=True
)
class ExecutionResult:
    selected_entries: np.ndarray
    model_scores: Tuple[np.ndarray, ...]
    ensemble_scores: np.ndarray
    receipt: Dict[str, Any]


def sha256_file(
    path: PathLike,
) -> str:
    file_path = Path(
        path
    )

    if not file_path.is_file():
        raise FileNotFoundError(
            str(
                file_path
            )
        )

    digest = hashlib.sha256()

    with file_path.open(
        "rb"
    ) as handle:
        for chunk in iter(
            lambda: handle.read(
                1024 * 1024
            ),
            b"",
        ):
            digest.update(
                chunk
            )

    return digest.hexdigest()


def validate_model_input_batch(
    model_inputs: Mapping[str, np.ndarray],
) -> int:
    actual_names = set(
        model_inputs
    )

    expected_names = set(
        MODEL_INPUT_NAMES
    )

    if actual_names != expected_names:
        missing = sorted(
            expected_names
            - actual_names
        )

        extra = sorted(
            actual_names
            - expected_names
        )

        raise KeyError(
            "model-input mismatch; missing={}, extra={}".format(
                missing,
                extra,
            )
        )

    expected_shapes = {
        "pf_features": (
            len(
                FEATURE_NAMES
            ),
            MAX_PARTICLES,
        ),
        "pf_vectors": (
            len(
                VECTOR_NAMES
            ),
            MAX_PARTICLES,
        ),
        "pf_mask": (
            1,
            MAX_PARTICLES,
        ),
    }

    event_count: Optional[int] = None

    for input_name in MODEL_INPUT_NAMES:
        array = np.asarray(
            model_inputs[
                input_name
            ]
        )

        if array.ndim != 3:
            raise ValueError(
                "{} must be three-dimensional, got {}".format(
                    input_name,
                    array.shape,
                )
            )

        expected_tail = expected_shapes[
            input_name
        ]

        if tuple(
            array.shape[
                1:
            ]
        ) != expected_tail:
            raise ValueError(
                "{} has shape {}, expected [N, {}, {}]".format(
                    input_name,
                    array.shape,
                    expected_tail[0],
                    expected_tail[1],
                )
            )

        if event_count is None:
            event_count = int(
                array.shape[
                    0
                ]
            )
        elif array.shape[0] != event_count:
            raise ValueError(
                "model inputs have inconsistent event counts"
            )

        if array.dtype != np.float32:
            raise TypeError(
                "{} must have dtype float32, got {}".format(
                    input_name,
                    array.dtype,
                )
            )

        if not np.all(
            np.isfinite(
                array
            )
        ):
            raise ValueError(
                "{} contains non-finite values".format(
                    input_name
                )
            )

    if event_count is None or event_count <= 0:
        raise ValueError(
            "model-input batch must contain at least one event"
        )

    mask = np.asarray(
        model_inputs[
            "pf_mask"
        ]
    )

    if not np.all(
        (mask == 0.0)
        | (mask == 1.0)
    ):
        raise ValueError(
            "pf_mask must contain only zero or one"
        )

    mask_counts = np.sum(
        mask,
        axis=(
            1,
            2,
        ),
    )

    if np.any(
        mask_counts <= 0.0
    ):
        raise ValueError(
            "each event must contain at least one unmasked particle"
        )

    return event_count


def prepare_batch_from_tree(
    tree,
    *,
    requested_events: int,
) -> PreparedBatch:
    if requested_events <= 0:
        raise ValueError(
            "requested_events must be positive"
        )

    tree_receipt = validate_input_tree(
        tree
    )

    selection_arrays = tree.arrays(
        list(
            SELECTION_BRANCHES
        ),
        library="np",
        how=dict,
    )

    eligible_entries = eligible_entry_indices(
        selection_arrays[
            "pass_selection"
        ],
        selection_arrays[
            "pass_4j3b_selection"
        ],
    )

    selected_entries = spread_selected_entries(
        selection_arrays[
            "pass_selection"
        ],
        selection_arrays[
            "pass_4j3b_selection"
        ],
        requested=requested_events,
    )

    if len(
        selected_entries
    ) == 0:
        raise ValueError(
            "input tree contains no eligible entries"
        )

    collected_inputs = {
        input_name: []
        for input_name in MODEL_INPUT_NAMES
    }

    selected_pass = []
    selected_fourj_threeb = []

    particle_counts = []
    used_particle_counts = []
    truncated_flags = []
    process_indices = []

    required_event_branches = list(
        INPUT_SCALAR_BRANCHES
        + INPUT_PARTICLE_BRANCHES
    )

    for selected_entry in selected_entries:
        entry = int(
            selected_entry
        )

        arrays = tree.arrays(
            required_event_branches,
            entry_start=entry,
            entry_stop=entry + 1,
            library="ak",
            how=dict,
        )

        if len(
            arrays[
                "pass_selection"
            ]
        ) != 1:
            raise RuntimeError(
                "entry read did not return exactly one event"
            )

        pass_selection = int(
            arrays[
                "pass_selection"
            ][0]
        )

        pass_4j3b = int(
            arrays[
                "pass_4j3b_selection"
            ][0]
        )

        if (
            pass_selection != 1
            or pass_4j3b != 1
        ):
            raise RuntimeError(
                "selected entry {} failed selection flags".format(
                    entry
                )
            )

        particle_variables = {}

        for branch_name in INPUT_PARTICLE_BRANCHES:
            particle_variables[
                branch_name
            ] = np.asarray(
                ak.to_numpy(
                    arrays[
                        branch_name
                    ][0]
                )
            )

        inputs = build_sophon_inputs(
            particle_variables,
            pfcand_sum_pt=float(
                arrays[
                    "pfcand_sum_pt"
                ][0]
            ),
            pfcand_sum_energy=float(
                arrays[
                    "pfcand_sum_energy"
                ][0]
            ),
        )

        for input_name in MODEL_INPUT_NAMES:
            collected_inputs[
                input_name
            ].append(
                inputs[
                    input_name
                ]
            )

        selected_pass.append(
            pass_selection
        )

        selected_fourj_threeb.append(
            pass_4j3b
        )

        particle_counts.append(
            int(
                np.asarray(
                    inputs[
                        "particle_count"
                    ]
                ).item()
            )
        )

        used_particle_counts.append(
            int(
                np.asarray(
                    inputs[
                        "used_particle_count"
                    ]
                ).item()
            )
        )

        truncated_flags.append(
            bool(
                np.asarray(
                    inputs[
                        "was_truncated"
                    ]
                ).item()
            )
        )

        process_indices.append(
            int(
                arrays[
                    "process_index"
                ][0]
            )
        )

    model_inputs = {
        input_name: np.concatenate(
            collected_inputs[
                input_name
            ],
            axis=0,
        ).astype(
            np.float32,
            copy=False,
        )
        for input_name in MODEL_INPUT_NAMES
    }

    event_count = validate_model_input_batch(
        model_inputs
    )

    if event_count != len(
        selected_entries
    ):
        raise RuntimeError(
            "prepared batch/event-selection size mismatch"
        )

    particle_array = np.asarray(
        particle_counts,
        dtype=np.int64,
    )

    used_array = np.asarray(
        used_particle_counts,
        dtype=np.int64,
    )

    truncated_array = np.asarray(
        truncated_flags,
        dtype=np.bool_,
    )

    process_counts = Counter(
        process_indices
    )

    metadata = {
        "tree": tree_receipt,
        "total_entries": int(
            tree_receipt[
                "entries"
            ]
        ),
        "eligible_entries": int(
            len(
                eligible_entries
            )
        ),
        "requested_events": int(
            requested_events
        ),
        "selected_event_count": int(
            len(
                selected_entries
            )
        ),
        "selected_entries": [
            int(
                entry
            )
            for entry in selected_entries
        ],
        "input_shapes": {
            input_name: list(
                model_inputs[
                    input_name
                ].shape
            )
            for input_name in MODEL_INPUT_NAMES
        },
        "input_dtype": "float32",
        "particle_count": {
            "minimum": int(
                np.min(
                    particle_array
                )
            ),
            "median": float(
                np.median(
                    particle_array
                )
            ),
            "maximum": int(
                np.max(
                    particle_array
                )
            ),
        },
        "used_particle_count": {
            "minimum": int(
                np.min(
                    used_array
                )
            ),
            "median": float(
                np.median(
                    used_array
                )
            ),
            "maximum": int(
                np.max(
                    used_array
                )
            ),
        },
        "truncated_event_count": int(
            np.count_nonzero(
                truncated_array
            )
        ),
        "truncated_event_fraction": float(
            np.mean(
                truncated_array
            )
        ),
        "process_index_counts": {
            str(
                index
            ): int(
                count
            )
            for index, count in sorted(
                process_counts.items()
            )
        },
    }

    return PreparedBatch(
        model_inputs=model_inputs,
        selected_entries=np.asarray(
            selected_entries,
            dtype=np.int64,
        ),
        pass_selection=np.asarray(
            selected_pass,
            dtype=np.int32,
        ),
        pass_4j3b_selection=np.asarray(
            selected_fourj_threeb,
            dtype=np.int32,
        ),
        metadata=metadata,
    )


def validate_session_interface(
    session,
    *,
    model_index: int,
) -> Dict[str, Any]:
    inputs = list(
        session.get_inputs()
    )

    outputs = list(
        session.get_outputs()
    )

    input_names = [
        item.name
        for item in inputs
    ]

    if (
        len(
            input_names
        )
        != len(
            set(
                input_names
            )
        )
    ):
        raise ValueError(
            "model {} has duplicate input names".format(
                model_index
            )
        )

    if set(
        input_names
    ) != set(
        MODEL_INPUT_NAMES
    ):
        raise ValueError(
            "model {} input names are {}, expected {}".format(
                model_index,
                sorted(
                    input_names
                ),
                sorted(
                    MODEL_INPUT_NAMES
                ),
            )
        )

    expected_channels = {
        "pf_features": len(
            FEATURE_NAMES
        ),
        "pf_vectors": len(
            VECTOR_NAMES
        ),
        "pf_mask": 1,
    }

    input_receipts = []

    for item in inputs:
        item_type = getattr(
            item,
            "type",
            None,
        )

        if item_type != "tensor(float)":
            raise TypeError(
                "model {} input {} has type {}, expected tensor(float)".format(
                    model_index,
                    item.name,
                    item_type,
                )
            )

        shape = list(
            getattr(
                item,
                "shape",
                []
            )
        )

        if len(
            shape
        ) != 3:
            raise ValueError(
                "model {} input {} has invalid shape {}".format(
                    model_index,
                    item.name,
                    shape,
                )
            )

        channel_dimension = shape[
            1
        ]

        if (
            isinstance(
                channel_dimension,
                int,
            )
            and channel_dimension
            != expected_channels[
                item.name
            ]
        ):
            raise ValueError(
                "model {} input {} has channel dimension {}, expected {}".format(
                    model_index,
                    item.name,
                    channel_dimension,
                    expected_channels[
                        item.name
                    ],
                )
            )

        input_receipts.append(
            {
                "name": item.name,
                "shape": shape,
                "type": item_type,
            }
        )

    if len(
        outputs
    ) != 1:
        raise ValueError(
            "model {} must have exactly one output".format(
                model_index
            )
        )

    output = outputs[
        0
    ]

    if output.name != MODEL_OUTPUT_NAME:
        raise ValueError(
            "model {} output is {!r}, expected {!r}".format(
                model_index,
                output.name,
                MODEL_OUTPUT_NAME,
            )
        )

    output_type = getattr(
        output,
        "type",
        None,
    )

    if output_type != "tensor(float)":
        raise TypeError(
            "model {} output type is {}, expected tensor(float)".format(
                model_index,
                output_type,
            )
        )

    output_shape = list(
        getattr(
            output,
            "shape",
            []
        )
    )

    if len(
        output_shape
    ) != 2:
        raise ValueError(
            "model {} output has invalid shape {}".format(
                model_index,
                output_shape,
            )
        )

    final_dimension = output_shape[
        -1
    ]

    if (
        isinstance(
            final_dimension,
            int,
        )
        and final_dimension
        != OUTPUT_CLASS_COUNT
    ):
        raise ValueError(
            "model {} output has {} classes, expected {}".format(
                model_index,
                final_dimension,
                OUTPUT_CLASS_COUNT,
            )
        )

    return {
        "model_index": int(
            model_index
        ),
        "inputs": input_receipts,
        "output": {
            "name": output.name,
            "shape": output_shape,
            "type": output_type,
        },
    }


def run_model_sessions(
    sessions: Sequence[object],
    model_inputs: Mapping[str, np.ndarray],
) -> Tuple[
    Tuple[np.ndarray, ...],
    Tuple[Dict[str, Any], ...],
]:
    if len(
        sessions
    ) != RELEASE_MODEL_COUNT:
        raise ValueError(
            "exactly {} model sessions are required".format(
                RELEASE_MODEL_COUNT
            )
        )

    event_count = validate_model_input_batch(
        model_inputs
    )

    score_matrices = []
    interface_receipts = []

    for model_index, session in enumerate(
        sessions
    ):
        interface_receipt = validate_session_interface(
            session,
            model_index=model_index,
        )

        raw_outputs = session.run(
            [
                MODEL_OUTPUT_NAME,
            ],
            {
                input_name: model_inputs[
                    input_name
                ]
                for input_name in MODEL_INPUT_NAMES
            },
        )

        if (
            not isinstance(
                raw_outputs,
                (
                    list,
                    tuple,
                ),
            )
            or len(
                raw_outputs
            ) != 1
        ):
            raise RuntimeError(
                "model {} did not return exactly one output".format(
                    model_index
                )
            )

        scores = validate_score_matrix(
            np.asarray(
                raw_outputs[
                    0
                ]
            ),
            name="model_scores[{}]".format(
                model_index
            ),
        )

        if scores.shape[0] != event_count:
            raise ValueError(
                "model {} returned {} events, expected {}".format(
                    model_index,
                    scores.shape[
                        0
                    ],
                    event_count,
                )
            )

        score_matrices.append(
            scores
        )

        interface_receipts.append(
            interface_receipt
        )

    return (
        tuple(
            score_matrices
        ),
        tuple(
            interface_receipts
        ),
    )


def create_onnxruntime_sessions(
    model_paths: Sequence[PathLike],
    *,
    providers: Sequence[str] = (
        "CPUExecutionProvider",
    ),
) -> Tuple[object, object, object]:
    if len(
        model_paths
    ) != RELEASE_MODEL_COUNT:
        raise ValueError(
            "exactly {} model paths are required".format(
                RELEASE_MODEL_COUNT
            )
        )

    resolved_paths = tuple(
        Path(
            path
        ).resolve(
            strict=True
        )
        for path in model_paths
    )

    if len(
        set(
            resolved_paths
        )
    ) != RELEASE_MODEL_COUNT:
        raise ValueError(
            "model paths must be distinct"
        )

    for model_path in resolved_paths:
        if not model_path.is_file():
            raise FileNotFoundError(
                str(
                    model_path
                )
            )

    try:
        import onnxruntime
    except ImportError as error:
        raise RuntimeError(
            "onnxruntime is required to create real model sessions"
        ) from error

    sessions = tuple(
        onnxruntime.InferenceSession(
            str(
                model_path
            ),
            providers=list(
                providers
            ),
        )
        for model_path in resolved_paths
    )

    if len(
        sessions
    ) != RELEASE_MODEL_COUNT:
        raise AssertionError(
            "session-count invariant failed"
        )

    return sessions


def execute_local_canary(
    root_path: PathLike,
    sessions: Sequence[object],
    *,
    requested_events: int,
    events_output_path: Optional[PathLike] = None,
) -> ExecutionResult:
    input_path = Path(
        root_path
    ).resolve(
        strict=True
    )

    if not input_path.is_file():
        raise FileNotFoundError(
            str(
                input_path
            )
        )

    with uproot.open(
        input_path
    ) as root_file:
        if INPUT_TREE_NAME not in root_file:
            raise KeyError(
                "ROOT file has no tree named {!r}".format(
                    INPUT_TREE_NAME
                )
            )

        tree = root_file[
            INPUT_TREE_NAME
        ]

        prepared = prepare_batch_from_tree(
            tree,
            requested_events=requested_events,
        )

    model_scores, interfaces = run_model_sessions(
        sessions,
        prepared.model_inputs,
    )

    ensemble_scores = ensemble_score_matrices(
        model_scores
    )

    events_receipt = None

    if events_output_path is not None:
        output_path = Path(
            events_output_path
        )

        events_receipt = write_events_tree(
            output_path,
            ensemble_scores,
            pass_selection=prepared.pass_selection,
            pass_4j3b_selection=(
                prepared.pass_4j3b_selection
            ),
        )

    receipt = {
        "schema_version": EXECUTION_BACKEND_SCHEMA_VERSION,
        "input": {
            "path": str(
                input_path
            ),
            "size_bytes": int(
                input_path.stat().st_size
            ),
            "sha256": sha256_file(
                input_path
            ),
            "tree": prepared.metadata[
                "tree"
            ],
        },
        "selection_and_preprocessing": prepared.metadata,
        "inference": {
            "released_model_count": RELEASE_MODEL_COUNT,
            "interfaces": list(
                interfaces
            ),
            "model_summaries": [
                {
                    "model_index": int(
                        model_index
                    ),
                    "score_summary": summarize_score_matrix(
                        scores
                    ),
                }
                for model_index, scores in enumerate(
                    model_scores
                )
            ],
            "model_top1_agreement_fraction": (
                model_top1_agreement_fraction(
                    model_scores
                )
            ),
            "ensemble_summary": summarize_score_matrix(
                ensemble_scores
            ),
            "ensemble_signal_grid_all": summarize_signal_grid(
                ensemble_scores
            ),
            "ensemble_signal_grid_gt_0_997": summarize_signal_grid(
                ensemble_scores,
                minimum_signal_probability=0.997,
            ),
        },
        "events_tree": events_receipt,
        "operations": {
            "local_root_opened": True,
            "input_tree_validated": True,
            "spread_selection_applied": True,
            "preprocessing_run": True,
            "model_sessions_run": True,
            "ensemble_computed": True,
            "events_tree_written": (
                events_receipt is not None
            ),
            "network_accessed": False,
        },
    }

    return ExecutionResult(
        selected_entries=prepared.selected_entries,
        model_scores=model_scores,
        ensemble_scores=ensemble_scores,
        receipt=receipt,
    )
