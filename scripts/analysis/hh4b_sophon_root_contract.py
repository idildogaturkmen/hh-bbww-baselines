#!/usr/bin/env python3

from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, Optional, Sequence, Tuple, Union

import numpy as np
import uproot

from hh4b_sophon_contract import OUTPUT_CLASS_COUNT
from hh4b_sophon_score_contract import validate_score_matrix


PathLike = Union[str, Path]

INPUT_TREE_NAME = "tree"
EVENTS_TREE_NAME = "Events"

SELECTION_BRANCHES = (
    "pass_selection",
    "pass_4j3b_selection",
)

INPUT_SCALAR_BRANCHES = (
    "pass_selection",
    "pass_4j3b_selection",
    "pfcand_sum_pt",
    "pfcand_sum_energy",
    "gen_higgs1_mass",
    "gen_higgs2_mass",
    "process_index",
)

INPUT_PARTICLE_BRANCHES = (
    "part_px",
    "part_py",
    "part_pz",
    "part_energy",
    "part_eta",
    "part_phi",
    "part_deta",
    "part_dphi",
    "part_charge",
    "part_pid",
    "part_d0val",
    "part_d0err",
    "part_dzval",
    "part_dzerr",
)

REQUIRED_INPUT_BRANCHES = (
    INPUT_SCALAR_BRANCHES
    + INPUT_PARTICLE_BRANCHES
)

SCORE_BRANCHES = tuple(
    "score_{}".format(
        class_index
    )
    for class_index in range(
        OUTPUT_CLASS_COUNT
    )
)

REQUIRED_EVENTS_BRANCHES = (
    SELECTION_BRANCHES
    + SCORE_BRANCHES
)


def _normalized_branch_names(
    branch_names: Iterable[object],
) -> Tuple[str, ...]:
    normalized = []

    for branch_name in branch_names:
        text = str(
            branch_name
        )

        # This also tolerates names supplied with a ROOT cycle suffix.
        normalized.append(
            text.split(
                ";",
                1,
            )[0]
        )

    return tuple(
        normalized
    )


def validate_required_branches(
    available_branches: Iterable[object],
    *,
    required_branches: Sequence[str],
    context: str,
) -> Tuple[str, ...]:
    """Require a branch subset while allowing unrelated extra branches."""

    available = set(
        _normalized_branch_names(
            available_branches
        )
    )

    missing = [
        branch_name
        for branch_name in required_branches
        if branch_name not in available
    ]

    if missing:
        raise KeyError(
            "{} is missing required branches: {}".format(
                context,
                missing,
            )
        )

    return tuple(
        required_branches
    )


def validate_input_tree(
    tree,
) -> Dict[str, object]:
    """Validate the released SophonHH input-tree interface."""

    classname = getattr(
        tree,
        "classname",
        None,
    )

    if classname != "TTree":
        raise TypeError(
            "input object must be a TTree, got {!r}".format(
                classname
            )
        )

    validate_required_branches(
        tree.keys(),
        required_branches=REQUIRED_INPUT_BRANCHES,
        context="input tree",
    )

    return {
        "tree_name": str(
            getattr(
                tree,
                "name",
                INPUT_TREE_NAME,
            )
        ),
        "tree_class": classname,
        "entries": int(
            tree.num_entries
        ),
        "required_branch_count": len(
            REQUIRED_INPUT_BRANCHES
        ),
    }


def validate_selection_flags(
    values,
    *,
    name: str,
) -> np.ndarray:
    """Return one-dimensional int32 binary selection flags."""

    raw = np.asarray(
        values
    )

    if raw.ndim != 1:
        raise ValueError(
            "{} must be one-dimensional, got shape {}".format(
                name,
                raw.shape,
            )
        )

    try:
        numeric = raw.astype(
            np.float64
        )
    except (TypeError, ValueError) as error:
        raise ValueError(
            "{} must contain numeric binary values".format(
                name
            )
        ) from error

    if not np.all(
        np.isfinite(
            numeric
        )
    ):
        raise ValueError(
            "{} contains non-finite values".format(
                name
            )
        )

    if not np.all(
        (numeric == 0.0)
        | (numeric == 1.0)
    ):
        raise ValueError(
            "{} must contain only 0 or 1".format(
                name
            )
        )

    return numeric.astype(
        np.int32
    )


def eligible_entry_indices(
    pass_selection,
    pass_4j3b_selection,
) -> np.ndarray:
    """Return entries satisfying both released selection flags."""

    inclusive = validate_selection_flags(
        pass_selection,
        name="pass_selection",
    )

    fourj_threeb = validate_selection_flags(
        pass_4j3b_selection,
        name="pass_4j3b_selection",
    )

    if len(inclusive) != len(
        fourj_threeb
    ):
        raise ValueError(
            "selection arrays have different lengths: {} and {}".format(
                len(inclusive),
                len(fourj_threeb),
            )
        )

    return np.flatnonzero(
        (inclusive == 1)
        & (fourj_threeb == 1)
    ).astype(
        np.int64
    )


def spread_positions(
    total: int,
    requested: int,
) -> np.ndarray:
    """Select deterministic positions across the complete candidate range."""

    if total < 0:
        raise ValueError(
            "total must be non-negative"
        )

    if requested < 0:
        raise ValueError(
            "requested must be non-negative"
        )

    if total == 0 or requested == 0:
        return np.asarray(
            [],
            dtype=np.int64,
        )

    if requested >= total:
        return np.arange(
            total,
            dtype=np.int64,
        )

    if requested == 1:
        return np.asarray(
            [
                (total - 1) // 2,
            ],
            dtype=np.int64,
        )

    positions = (
        np.arange(
            requested,
            dtype=np.int64,
        )
        * (total - 1)
        // (requested - 1)
    ).astype(
        np.int64
    )

    if len(
        np.unique(
            positions
        )
    ) != requested:
        raise AssertionError(
            "spread positions are not unique"
        )

    if (
        positions[0] != 0
        or positions[-1] != total - 1
    ):
        raise AssertionError(
            "spread positions do not span the candidate range"
        )

    return positions


def spread_selected_entries(
    pass_selection,
    pass_4j3b_selection,
    *,
    requested: int,
) -> np.ndarray:
    """Return deterministic eligible entry numbers across a file."""

    eligible = eligible_entry_indices(
        pass_selection,
        pass_4j3b_selection,
    )

    positions = spread_positions(
        len(eligible),
        requested,
    )

    return eligible[
        positions
    ]


def build_events_payload(
    scores,
    *,
    pass_selection=None,
    pass_4j3b_selection=None,
) -> Dict[str, np.ndarray]:
    """Build an official-compatible Events-tree payload."""

    validated_scores = validate_score_matrix(
        scores
    )

    event_count = len(
        validated_scores
    )

    if pass_selection is None:
        inclusive = np.ones(
            event_count,
            dtype=np.int32,
        )
    else:
        inclusive = validate_selection_flags(
            pass_selection,
            name="pass_selection",
        )

    if pass_4j3b_selection is None:
        fourj_threeb = np.ones(
            event_count,
            dtype=np.int32,
        )
    else:
        fourj_threeb = validate_selection_flags(
            pass_4j3b_selection,
            name="pass_4j3b_selection",
        )

    if len(inclusive) != event_count:
        raise ValueError(
            "pass_selection length {} does not match score rows {}".format(
                len(inclusive),
                event_count,
            )
        )

    if len(fourj_threeb) != event_count:
        raise ValueError(
            "pass_4j3b_selection length {} does not match "
            "score rows {}".format(
                len(fourj_threeb),
                event_count,
            )
        )

    payload: Dict[
        str,
        np.ndarray,
    ] = {
        "pass_selection": inclusive,
        "pass_4j3b_selection": fourj_threeb,
    }

    float_scores = validated_scores.astype(
        np.float32
    )

    for class_index, branch_name in enumerate(
        SCORE_BRANCHES
    ):
        payload[
            branch_name
        ] = float_scores[
            :,
            class_index,
        ]

    return payload


def events_branch_types() -> Dict[str, object]:
    branch_types: Dict[
        str,
        object,
    ] = {
        "pass_selection": np.int32,
        "pass_4j3b_selection": np.int32,
    }

    for branch_name in SCORE_BRANCHES:
        branch_types[
            branch_name
        ] = np.float32

    return branch_types


def validate_events_tree(
    tree,
) -> Dict[str, object]:
    """Validate the minimum official DCB-compatible Events interface."""

    classname = getattr(
        tree,
        "classname",
        None,
    )

    if classname != "TTree":
        raise TypeError(
            "Events object must be a TTree, got {!r}".format(
                classname
            )
        )

    validate_required_branches(
        tree.keys(),
        required_branches=REQUIRED_EVENTS_BRANCHES,
        context="Events tree",
    )

    return {
        "tree_name": str(
            getattr(
                tree,
                "name",
                EVENTS_TREE_NAME,
            )
        ),
        "tree_class": classname,
        "entries": int(
            tree.num_entries
        ),
        "required_branch_count": len(
            REQUIRED_EVENTS_BRANCHES
        ),
        "score_branch_count": len(
            SCORE_BRANCHES
        ),
    }


def read_events_score_matrix(
    path: PathLike,
    *,
    tree_name: str = EVENTS_TREE_NAME,
) -> np.ndarray:
    """Read and validate score_0 through score_137 from an Events TTree."""

    root_path = Path(
        path
    )

    if not root_path.is_file():
        raise FileNotFoundError(
            str(
                root_path
            )
        )

    with uproot.open(
        root_path
    ) as root_file:
        if tree_name not in root_file:
            raise KeyError(
                "ROOT file has no tree named {!r}".format(
                    tree_name
                )
            )

        tree = root_file[
            tree_name
        ]

        validate_events_tree(
            tree
        )

        scores = np.column_stack(
            [
                tree[
                    branch_name
                ].array(
                    library="np"
                )
                for branch_name in SCORE_BRANCHES
            ]
        ).astype(
            np.float32
        )

    return validate_score_matrix(
        scores
    )


def write_events_tree(
    path: PathLike,
    scores,
    *,
    pass_selection=None,
    pass_4j3b_selection=None,
    tree_name: str = EVENTS_TREE_NAME,
    overwrite: bool = False,
) -> Dict[str, object]:
    """Write and verify a DCB-compatible Events TTree."""

    root_path = Path(
        path
    )

    if root_path.exists() and not overwrite:
        raise FileExistsError(
            str(
                root_path
            )
        )

    if not root_path.parent.is_dir():
        raise FileNotFoundError(
            "output parent directory does not exist: {}".format(
                root_path.parent
            )
        )

    payload = build_events_payload(
        scores,
        pass_selection=pass_selection,
        pass_4j3b_selection=pass_4j3b_selection,
    )

    branch_types = events_branch_types()

    with uproot.recreate(
        root_path
    ) as root_file:
        tree = root_file.mktree(
            tree_name,
            branch_types,
            title=(
                "SophonHH scores for official "
                "mass-grid fitting"
            ),
        )

        tree.extend(
            payload
        )

    with uproot.open(
        root_path
    ) as root_file:
        tree = root_file[
            tree_name
        ]

        receipt = validate_events_tree(
            tree
        )

        actual_branches = set(
            _normalized_branch_names(
                tree.keys()
            )
        )

        expected_branches = set(
            REQUIRED_EVENTS_BRANCHES
        )

        if actual_branches != expected_branches:
            missing = sorted(
                expected_branches
                - actual_branches
            )

            extra = sorted(
                actual_branches
                - expected_branches
            )

            raise RuntimeError(
                "written Events branch mismatch; missing={}, extra={}".format(
                    missing,
                    extra,
                )
            )

    roundtrip_scores = read_events_score_matrix(
        root_path,
        tree_name=tree_name,
    )

    expected_scores = validate_score_matrix(
        scores
    )

    np.testing.assert_allclose(
        roundtrip_scores,
        expected_scores,
        rtol=0.0,
        atol=1.0e-7,
    )

    receipt.update(
        {
            "path": str(
                root_path
            ),
            "file_size_bytes": int(
                root_path.stat().st_size
            ),
        }
    )

    return receipt
