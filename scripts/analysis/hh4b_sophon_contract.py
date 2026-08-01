#!/usr/bin/env python3

from __future__ import annotations

from typing import Dict, Mapping, Optional, Sequence, Tuple, Union

import numpy as np


ArrayLike = Union[Sequence[float], np.ndarray]

MAX_PARTICLES = 256

SIGNAL_CLASS_COUNT = 136
QCD_CLASS_INDEX = 136
TTBAR_CLASS_INDEX = 137
OUTPUT_CLASS_COUNT = 138

FEATURE_NAMES = (
    "part_pt_log",
    "part_e_log",
    "part_logptrel",
    "part_logerel",
    "part_deltaR",
    "part_charge",
    "part_isChargedHadron",
    "part_isNeutralHadron",
    "part_isPhoton",
    "part_isElectron",
    "part_isMuon",
    "part_d0",
    "part_d0err",
    "part_dz",
    "part_dzerr",
    "part_deta",
    "part_dphi",
    "part_eta",
    "part_phi",
)

VECTOR_NAMES = (
    "part_px",
    "part_py",
    "part_pz",
    "part_energy",
)

REQUIRED_PARTICLE_FIELDS = (
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

FEATURE_TRANSFORMS = (
    (1.7, 0.7, -5.0, 5.0),
    (2.0, 0.7, -5.0, 5.0),
    (-4.7, 0.7, -5.0, 5.0),
    (-4.7, 0.7, -5.0, 5.0),
    (0.2, 4.0, -5.0, 5.0),
    (0.0, 1.0, -1.0e8, 1.0e8),
    (0.0, 1.0, -1.0e8, 1.0e8),
    (0.0, 1.0, -1.0e8, 1.0e8),
    (0.0, 1.0, -1.0e8, 1.0e8),
    (0.0, 1.0, -1.0e8, 1.0e8),
    (0.0, 1.0, -1.0e8, 1.0e8),
    (0.0, 1.0, -1.0e8, 1.0e8),
    (0.0, 1.0, 0.0, 1.0),
    (0.0, 1.0, -1.0e8, 1.0e8),
    (0.0, 1.0, 0.0, 1.0),
    (0.0, 1.0, -1.0e8, 1.0e8),
    (0.0, 1.0, -1.0e8, 1.0e8),
    (0.0, 1.0, -1.0e8, 1.0e8),
    (0.0, 1.0, -1.0e8, 1.0e8),
)

MASS_BINS = tuple(
    (float(low), float(low + 10))
    for low in range(40, 200, 10)
)


def _as_1d_array(
    values: ArrayLike,
    *,
    name: str,
    dtype: np.dtype,
) -> np.ndarray:
    array = np.asarray(
        values,
        dtype=dtype,
    )

    if array.ndim != 1:
        raise ValueError(
            "{} must be one-dimensional, got shape {}".format(
                name,
                array.shape,
            )
        )

    return array


def _validated_particle_arrays(
    particle_vars: Mapping[str, ArrayLike],
) -> Dict[str, np.ndarray]:
    missing = [
        name
        for name in REQUIRED_PARTICLE_FIELDS
        if name not in particle_vars
    ]

    if missing:
        raise KeyError(
            "missing particle fields: {}".format(
                missing
            )
        )

    arrays: Dict[str, np.ndarray] = {}

    for name in REQUIRED_PARTICLE_FIELDS:
        dtype = (
            np.int64
            if name == "part_pid"
            else np.float64
        )

        arrays[name] = _as_1d_array(
            particle_vars[name],
            name=name,
            dtype=dtype,
        )

    lengths = {
        name: len(array)
        for name, array in arrays.items()
    }

    if len(set(lengths.values())) != 1:
        raise ValueError(
            "particle fields have inconsistent lengths: {}".format(
                lengths
            )
        )

    for name, array in arrays.items():
        if not np.all(
            np.isfinite(array)
        ):
            raise ValueError(
                "{} contains non-finite values".format(
                    name
                )
            )

    return arrays


def _transform_channel(
    values: np.ndarray,
    transform: Tuple[
        float,
        float,
        float,
        float,
    ],
) -> np.ndarray:
    subtract, multiply, clip_min, clip_max = transform

    return np.clip(
        (values - subtract) * multiply,
        clip_min,
        clip_max,
    )


def build_sophon_inputs(
    particle_vars: Mapping[str, ArrayLike],
    *,
    pfcand_sum_pt: float,
    pfcand_sum_energy: float,
    max_particles: int = MAX_PARTICLES,
) -> Dict[str, np.ndarray]:
    """Build the released SophonHH ONNX inputs for one event."""

    if max_particles <= 0:
        raise ValueError(
            "max_particles must be positive"
        )

    if (
        not np.isfinite(pfcand_sum_pt)
        or pfcand_sum_pt <= 0.0
    ):
        raise ValueError(
            "pfcand_sum_pt must be finite and positive"
        )

    if (
        not np.isfinite(pfcand_sum_energy)
        or pfcand_sum_energy <= 0.0
    ):
        raise ValueError(
            "pfcand_sum_energy must be finite and positive"
        )

    arrays = _validated_particle_arrays(
        particle_vars
    )

    particle_count = len(
        arrays["part_px"]
    )

    used = min(
        particle_count,
        max_particles,
    )

    px = arrays["part_px"][:used]
    py = arrays["part_py"][:used]
    pz = arrays["part_pz"][:used]
    energy = arrays["part_energy"][:used]

    eta = arrays["part_eta"][:used]
    phi = arrays["part_phi"][:used]
    deta = arrays["part_deta"][:used]
    dphi = arrays["part_dphi"][:used]

    charge = arrays["part_charge"][:used]
    pid = arrays["part_pid"][:used]

    d0val = arrays["part_d0val"][:used]
    d0err = arrays["part_d0err"][:used]
    dzval = arrays["part_dzval"][:used]
    dzerr = arrays["part_dzerr"][:used]

    pt = np.hypot(
        px,
        py,
    )

    if np.any(
        pt <= 0.0
    ):
        raise ValueError(
            "particle transverse momentum must be positive"
        )

    if np.any(
        energy <= 0.0
    ):
        raise ValueError(
            "particle energy must be positive"
        )

    is_electron = np.isin(
        pid,
        (-11, 11),
    ).astype(np.float64)

    is_muon = np.isin(
        pid,
        (-13, 13),
    ).astype(np.float64)

    is_photon = (
        pid == 22
    ).astype(np.float64)

    is_charged_hadron = (
        (charge != 0.0)
        & (is_electron == 0.0)
        & (is_muon == 0.0)
    ).astype(np.float64)

    is_neutral_hadron = (
        (charge == 0.0)
        & (is_photon == 0.0)
    ).astype(np.float64)

    raw_features = (
        np.log(pt),
        np.log(energy),
        np.log(
            pt / pfcand_sum_pt
        ),
        np.log(
            energy / pfcand_sum_energy
        ),
        np.hypot(
            deta,
            dphi,
        ),
        charge,
        is_charged_hadron,
        is_neutral_hadron,
        is_photon,
        is_electron,
        is_muon,
        np.tanh(d0val),
        d0err,
        np.tanh(dzval),
        dzerr,
        deta,
        dphi,
        eta,
        phi,
    )

    features = np.zeros(
        (
            1,
            len(FEATURE_NAMES),
            max_particles,
        ),
        dtype=np.float32,
    )

    for channel, pair in enumerate(
        zip(
            raw_features,
            FEATURE_TRANSFORMS,
        )
    ):
        values, transform = pair

        features[
            0,
            channel,
            :used,
        ] = _transform_channel(
            values,
            transform,
        ).astype(np.float32)

    vectors = np.zeros(
        (
            1,
            len(VECTOR_NAMES),
            max_particles,
        ),
        dtype=np.float32,
    )

    for channel, values in enumerate(
        (
            px,
            py,
            pz,
            energy,
        )
    ):
        vectors[
            0,
            channel,
            :used,
        ] = values.astype(
            np.float32
        )

    mask = np.zeros(
        (
            1,
            1,
            max_particles,
        ),
        dtype=np.float32,
    )

    mask[
        0,
        0,
        :used,
    ] = 1.0

    return {
        "pf_features": features,
        "pf_vectors": vectors,
        "pf_mask": mask,
        "particle_count": np.asarray(
            particle_count,
            dtype=np.int64,
        ),
        "used_particle_count": np.asarray(
            used,
            dtype=np.int64,
        ),
        "was_truncated": np.asarray(
            particle_count > max_particles,
            dtype=np.bool_,
        ),
    }


def _mass_bin_index(
    mass: float,
) -> Optional[int]:
    if (
        not np.isfinite(mass)
        or mass < 40.0
        or mass > 200.0
    ):
        return None

    result: Optional[int] = None

    # The public implementation uses inclusive boundaries and does
    # not stop after a match. An exact shared edge therefore maps
    # to the higher of the two neighboring bins.
    for index, bounds in enumerate(
        MASS_BINS
    ):
        low, high = bounds

        if low <= mass <= high:
            result = index

    return result


def calculate_class_index(
    gen_higgs1_mass: float,
    gen_higgs2_mass: float,
    process_index: int,
) -> int:
    """Reproduce calculateClsIndex from the pinned public release."""

    mass1 = min(
        float(gen_higgs1_mass),
        float(gen_higgs2_mass),
    )

    mass2 = max(
        float(gen_higgs1_mass),
        float(gen_higgs2_mass),
    )

    bin1 = _mass_bin_index(
        mass1
    )

    bin2 = _mass_bin_index(
        mass2
    )

    if (
        bin1 is None
        or bin2 is None
    ):
        return (
            QCD_CLASS_INDEX
            if process_index == 0
            else TTBAR_CLASS_INDEX
        )

    index = 0

    for first in range(
        bin1 + 1
    ):
        for second in range(
            first,
            len(MASS_BINS),
        ):
            if (
                first == bin1
                and second == bin2
            ):
                return index

            index += 1

    return (
        QCD_CLASS_INDEX
        if process_index == 0
        else TTBAR_CLASS_INDEX
    )


def class_index_to_mass_bins(
    class_index: int,
) -> Tuple[
    Tuple[float, float],
    Tuple[float, float],
]:
    if (
        class_index < 0
        or class_index >= SIGNAL_CLASS_COUNT
    ):
        raise ValueError(
            "signal class index must be in [0, 135], got {}".format(
                class_index
            )
        )

    index = 0

    for first in range(
        len(MASS_BINS)
    ):
        for second in range(
            first,
            len(MASS_BINS),
        ):
            if index == class_index:
                return (
                    MASS_BINS[first],
                    MASS_BINS[second],
                )

            index += 1

    raise AssertionError(
        "unreachable signal class index"
    )
