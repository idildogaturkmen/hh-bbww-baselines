#!/usr/bin/env python3
"""Shared CMS-inspired HH4b BDT-v2 category and feature contracts.

This module contains no model-training code.  It preserves the exact BDT-v1
mass-aware feature order, assigns the a-priori 450 GeV mHH categories, and
derives detector-level angular features from the four candidate-jet
four-vectors.

The Higgs pairing is reconstructed without using the stored pairing metadata:
the three disjoint four-jet pairings are ranked by

  (sum |mjj - 125 GeV|, |mjj(a) - mjj(b)|, -(pT(a) + pT(b))).

The higher-pT dijet is H1 (the first pairing wins an exact tie).  Within each
Higgs candidate, the higher-pT jet is the leading jet (the lower jet index wins
an exact tie).

Rest-frame conventions use active Lorentz boosts with beta = p_parent/E_parent:

* theta* is the H1 angle in the HH rest frame relative to the HH laboratory
  flight direction (the helicity axis);
* each jet decay angle is the leading-jet angle in its Higgs rest frame
  relative to that Higgs candidate's laboratory flight direction.

The +z laboratory beam direction is the deterministic fallback only when a
parent has exactly zero three-momentum.  Cosines are not clipped.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import csv
import json
import math
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd

from hh4b_bdt_v1_common import (
    MASS_AWARE_FEATURES as _V1_MASS_AWARE_FEATURES,
    SOURCE_COLUMNS as _V1_SOURCE_COLUMNS,
    add_derived_features as add_v1_derived_features,
    forbidden_feature_reason,
)


MHH_BOUNDARY_GEV = 450.0
CATEGORY_NAMES: tuple[str, ...] = ("low_mhh", "high_mhh")

GLOBAL_V1_MASS_AWARE_REFERENCE: tuple[str, ...] = tuple(
    _V1_MASS_AWARE_FEATURES
)

PAIRWISE_DR_FEATURES: tuple[str, ...] = (
    "dr_j1_j2",
    "dr_j1_j3",
    "dr_j1_j4",
    "dr_j2_j3",
    "dr_j2_j4",
    "dr_j3_j4",
)

CMS_INSPIRED_DERIVED_FEATURES: tuple[str, ...] = (
    *PAIRWISE_DR_FEATURES,
    "min_candidate_pair_dr",
    "max_candidate_pair_dr",
    "mean_candidate_pair_dr",
    "min_cross_higgs_pair_dr",
    "max_cross_higgs_pair_dr",
    "candidate_vector_pt_sum",
    "cos_theta_star_h1_in_hh_rest",
    "abs_cos_theta_star_h1_in_hh_rest",
    "cos_theta_j1_in_h1_rest",
    "abs_cos_theta_j1_in_h1_rest",
    "cos_theta_j_h2_leading_in_h2_rest",
    "abs_cos_theta_j_h2_leading_in_h2_rest",
)

CATEGORIZED_MASS_AWARE_FEATURES: tuple[str, ...] = (
    GLOBAL_V1_MASS_AWARE_REFERENCE + CMS_INSPIRED_DERIVED_FEATURES
)

EXPLICIT_DIJET_MASS_PLANE_FEATURES: frozenset[str] = frozenset(
    {"mbb1", "mbb2", "delta_mbb", "r_hh_125_125"}
)

CATEGORIZED_EXPLICIT_DIJET_MASS_PLANE_BLIND_FEATURES: tuple[str, ...] = tuple(
    feature
    for feature in CATEGORIZED_MASS_AWARE_FEATURES
    if feature not in EXPLICIT_DIJET_MASS_PLANE_FEATURES
)

FEATURE_VARIANTS: Mapping[str, tuple[str, ...]] = {
    "global_v1_mass_aware_reference": GLOBAL_V1_MASS_AWARE_REFERENCE,
    "low_mhh_cms_inspired_mass_aware": CATEGORIZED_MASS_AWARE_FEATURES,
    "high_mhh_cms_inspired_mass_aware": CATEGORIZED_MASS_AWARE_FEATURES,
    "categorized_explicit_dijet_mass_plane_blind": (
        CATEGORIZED_EXPLICIT_DIJET_MASS_PLANE_BLIND_FEATURES
    ),
}

JET_FOUR_VECTOR_COLUMNS: tuple[str, ...] = tuple(
    f"j{jet}_{field}"
    for jet in range(1, 5)
    for field in ("pt", "eta", "phi", "mass")
)

SOURCE_COLUMNS: tuple[str, ...] = tuple(
    dict.fromkeys((*_V1_SOURCE_COLUMNS, *JET_FOUR_VECTOR_COLUMNS))
)

PAIRINGS: tuple[tuple[tuple[int, int], tuple[int, int]], ...] = (
    ((0, 1), (2, 3)),
    ((0, 2), (1, 3)),
    ((0, 3), (1, 2)),
)

DERIVED_FEATURE_DEFINITIONS: Mapping[str, str] = {
    "dr_j1_j2": "DeltaR(j1,j2) with wrapped DeltaPhi",
    "dr_j1_j3": "DeltaR(j1,j3) with wrapped DeltaPhi",
    "dr_j1_j4": "DeltaR(j1,j4) with wrapped DeltaPhi",
    "dr_j2_j3": "DeltaR(j2,j3) with wrapped DeltaPhi",
    "dr_j2_j4": "DeltaR(j2,j4) with wrapped DeltaPhi",
    "dr_j3_j4": "DeltaR(j3,j4) with wrapped DeltaPhi",
    "min_candidate_pair_dr": "minimum of all six candidate-jet DeltaR values",
    "max_candidate_pair_dr": "maximum of all six candidate-jet DeltaR values",
    "mean_candidate_pair_dr": "arithmetic mean of all six candidate-jet DeltaR values",
    "min_cross_higgs_pair_dr": "minimum DeltaR among the four cross-Higgs jet pairs",
    "max_cross_higgs_pair_dr": "maximum DeltaR among the four cross-Higgs jet pairs",
    "candidate_vector_pt_sum": "|vector sum of four candidate-jet transverse momenta|",
    "cos_theta_star_h1_in_hh_rest": (
        "cosine between H1 in the HH rest frame and the HH lab flight direction"
    ),
    "abs_cos_theta_star_h1_in_hh_rest": (
        "absolute value of cos_theta_star_h1_in_hh_rest"
    ),
    "cos_theta_j1_in_h1_rest": (
        "cosine between leading-pT H1 jet in the H1 rest frame and the H1 lab "
        "flight direction"
    ),
    "abs_cos_theta_j1_in_h1_rest": (
        "absolute value of cos_theta_j1_in_h1_rest"
    ),
    "cos_theta_j_h2_leading_in_h2_rest": (
        "cosine between leading-pT H2 jet in the H2 rest frame and the H2 lab "
        "flight direction"
    ),
    "abs_cos_theta_j_h2_leading_in_h2_rest": (
        "absolute value of cos_theta_j_h2_leading_in_h2_rest"
    ),
}


@dataclass(frozen=True)
class LorentzVector:
    """Minimal (E, px, py, pz) Lorentz vector with metric (+,-,-,-)."""

    e: float
    px: float
    py: float
    pz: float

    @classmethod
    def from_pt_eta_phi_mass(
        cls,
        pt: float,
        eta: float,
        phi: float,
        mass: float,
    ) -> "LorentzVector":
        values = np.asarray([pt, eta, phi, mass], dtype=np.float64)
        if not np.all(np.isfinite(values)):
            raise ValueError("four-vector inputs must be finite")
        if pt < 0.0:
            raise ValueError("four-vector pt must be nonnegative")
        px = float(pt * math.cos(phi))
        py = float(pt * math.sin(phi))
        pz = float(pt * math.sinh(eta))
        # The frozen Delphes schema has a handful of O(1e-6 GeV) negative
        # stored jet masses from upstream floating-point reconstruction.
        # A four-vector depends on m^2, so preserve the signed input and use
        # it quadratically; do not clip or silently replace the source value.
        energy2 = px * px + py * py + pz * pz + mass * mass
        return cls(float(math.sqrt(energy2)), px, py, pz)

    def __add__(self, other: "LorentzVector") -> "LorentzVector":
        return LorentzVector(
            self.e + other.e,
            self.px + other.px,
            self.py + other.py,
            self.pz + other.pz,
        )

    @property
    def spatial(self) -> np.ndarray:
        return np.asarray([self.px, self.py, self.pz], dtype=np.float64)

    @property
    def pt(self) -> float:
        return float(math.hypot(self.px, self.py))

    @property
    def momentum(self) -> float:
        return float(np.linalg.norm(self.spatial))

    @property
    def mass(self) -> float:
        mass2 = self.e * self.e - float(np.dot(self.spatial, self.spatial))
        if mass2 < -1.0e-8 * max(self.e * self.e, 1.0):
            raise ValueError(f"unphysical negative invariant mass squared: {mass2}")
        return float(math.sqrt(max(mass2, 0.0)))

    @property
    def beta(self) -> np.ndarray:
        if self.e <= 0.0:
            raise ValueError("parent energy must be positive")
        beta = self.spatial / self.e
        beta2 = float(np.dot(beta, beta))
        if beta2 >= 1.0:
            raise ValueError(f"parent boost is not subluminal: beta^2={beta2}")
        return beta

    def boost(self, beta: Sequence[float]) -> "LorentzVector":
        """Transform into a frame moving with velocity ``beta``.

        For beta equal to a timelike parent's p/E this sends the parent to its
        rest frame.
        """

        beta_array = np.asarray(beta, dtype=np.float64)
        if beta_array.shape != (3,) or not np.all(np.isfinite(beta_array)):
            raise ValueError("boost beta must be a finite three-vector")
        beta2 = float(np.dot(beta_array, beta_array))
        if beta2 < 0.0 or beta2 >= 1.0:
            raise ValueError("boost beta magnitude must be smaller than one")
        if beta2 == 0.0:
            return self
        gamma = 1.0 / math.sqrt(1.0 - beta2)
        momentum = self.spatial
        beta_dot_p = float(np.dot(beta_array, momentum))
        factor = ((gamma - 1.0) * beta_dot_p / beta2) - gamma * self.e
        boosted_p = momentum + factor * beta_array
        boosted_e = gamma * (self.e - beta_dot_p)
        return LorentzVector(
            float(boosted_e),
            float(boosted_p[0]),
            float(boosted_p[1]),
            float(boosted_p[2]),
        )


def wrapped_delta_phi(phi1: float, phi2: float) -> float:
    """Return phi1-phi2 in [-pi, pi] with deterministic atan2 wrapping."""

    if not math.isfinite(phi1) or not math.isfinite(phi2):
        raise ValueError("azimuths must be finite")
    return float(math.atan2(math.sin(phi1 - phi2), math.cos(phi1 - phi2)))


def delta_r(
    eta1: float,
    phi1: float,
    eta2: float,
    phi2: float,
) -> float:
    """Return detector-space DeltaR using a wrapped azimuthal difference."""

    values = (eta1, phi1, eta2, phi2)
    if not all(math.isfinite(value) for value in values):
        raise ValueError("DeltaR inputs must be finite")
    return float(math.hypot(eta1 - eta2, wrapped_delta_phi(phi1, phi2)))


def pairwise_delta_rs(
    etas: Sequence[float],
    phis: Sequence[float],
) -> dict[tuple[int, int], float]:
    """Calculate all six pairwise DeltaR values for four candidate jets."""

    if len(etas) != 4 or len(phis) != 4:
        raise ValueError("exactly four eta and four phi values are required")
    return {
        (first, second): delta_r(
            float(etas[first]),
            float(phis[first]),
            float(etas[second]),
            float(phis[second]),
        )
        for first in range(4)
        for second in range(first + 1, 4)
    }


def pairwise_delta_r_summary(
    values: Mapping[tuple[int, int], float],
) -> tuple[float, float, float]:
    """Return minimum, maximum, and arithmetic mean of six DeltaR values."""

    if set(values) != {
        (0, 1),
        (0, 2),
        (0, 3),
        (1, 2),
        (1, 3),
        (2, 3),
    }:
        raise ValueError("the complete six-pair DeltaR mapping is required")
    array = np.asarray(list(values.values()), dtype=np.float64)
    if not np.all(np.isfinite(array)) or np.any(array < 0.0):
        raise ValueError("pairwise DeltaR values must be finite and nonnegative")
    return float(np.min(array)), float(np.max(array)), float(np.mean(array))


def reconstruct_higgs_pairs(
    jets: Sequence[LorentzVector],
) -> tuple[tuple[int, int], tuple[int, int]]:
    """Reconstruct ordered H1/H2 jet pairs from four-vectors alone."""

    if len(jets) != 4:
        raise ValueError("exactly four candidate jets are required")
    ranked: list[
        tuple[tuple[float, float, float], tuple[int, int], tuple[int, int]]
    ] = []
    for pair_a, pair_b in PAIRINGS:
        vector_a = jets[pair_a[0]] + jets[pair_a[1]]
        vector_b = jets[pair_b[0]] + jets[pair_b[1]]
        score = (
            abs(vector_a.mass - 125.0) + abs(vector_b.mass - 125.0),
            abs(vector_a.mass - vector_b.mass),
            -(vector_a.pt + vector_b.pt),
        )
        ranked.append((score, pair_a, pair_b))
    _, pair_a, pair_b = min(ranked, key=lambda item: item[0])
    vector_a = jets[pair_a[0]] + jets[pair_a[1]]
    vector_b = jets[pair_b[0]] + jets[pair_b[1]]
    if vector_b.pt > vector_a.pt:
        return pair_b, pair_a
    return pair_a, pair_b


def cross_higgs_pair_indices(
    h1_pair: Sequence[int],
    h2_pair: Sequence[int],
) -> tuple[tuple[int, int], ...]:
    """Return the four sorted cross-Higgs jet pairs."""

    if len(h1_pair) != 2 or len(h2_pair) != 2:
        raise ValueError("each Higgs pair must contain two jet indices")
    if set(h1_pair) | set(h2_pair) != {0, 1, 2, 3}:
        raise ValueError("Higgs pairs must partition the four jet indices")
    if set(h1_pair) & set(h2_pair):
        raise ValueError("Higgs pairs must be disjoint")
    return tuple(
        sorted(
            tuple(sorted((int(first), int(second))))
            for first in h1_pair
            for second in h2_pair
        )
    )


def leading_jet_index(
    pair: Sequence[int],
    jets: Sequence[LorentzVector],
) -> int:
    """Return the higher-pT jet index, breaking exact ties by lower index."""

    if len(pair) != 2:
        raise ValueError("a two-jet pair is required")
    return min((int(pair[0]), int(pair[1])), key=lambda index: (-jets[index].pt, index))


def _axis_from_parent(parent: LorentzVector) -> np.ndarray:
    momentum = parent.spatial
    magnitude = float(np.linalg.norm(momentum))
    if magnitude == 0.0:
        return np.asarray([0.0, 0.0, 1.0], dtype=np.float64)
    return momentum / magnitude


def rest_frame_cosine(
    child: LorentzVector,
    parent: LorentzVector,
    axis: Sequence[float] | None = None,
) -> float:
    """Cosine of a child in the parent rest frame relative to a lab axis."""

    reference = (
        _axis_from_parent(parent)
        if axis is None
        else np.asarray(axis, dtype=np.float64)
    )
    if reference.shape != (3,) or not np.all(np.isfinite(reference)):
        raise ValueError("rest-frame reference axis must be finite")
    reference_norm = float(np.linalg.norm(reference))
    if reference_norm == 0.0:
        raise ValueError("rest-frame reference axis must be nonzero")
    child_rest = child.boost(parent.beta)
    child_momentum = child_rest.spatial
    child_norm = float(np.linalg.norm(child_momentum))
    if child_norm == 0.0:
        raise ValueError("rest-frame child momentum must be nonzero")
    return float(
        np.dot(child_momentum, reference)
        / (child_norm * reference_norm)
    )


def assign_mhh_category(mhh: float) -> str:
    """Assign exactly one frozen category, with 450 GeV in high_mhh."""

    value = float(mhh)
    if not math.isfinite(value):
        raise ValueError("mHH must be finite")
    return "low_mhh" if value < MHH_BOUNDARY_GEV else "high_mhh"


def validate_feature_names(features: Sequence[str]) -> None:
    """Reject duplicate or forbidden nominal feature names."""

    duplicates = sorted(
        feature
        for feature, count in Counter(features).items()
        if count > 1
    )
    if duplicates:
        raise ValueError(f"duplicate selected features: {duplicates}")
    forbidden = {
        feature: reason
        for feature in features
        if (reason := forbidden_feature_reason(feature)) is not None
    }
    if forbidden:
        raise ValueError(f"forbidden selected features: {forbidden}")


def require_finite_features(
    frame: pd.DataFrame,
    features: Sequence[str],
) -> None:
    """Fail on missing columns, missing values, or nonfinite values."""

    validate_feature_names(features)
    missing_columns = sorted(set(features) - set(frame.columns))
    if missing_columns:
        raise ValueError(f"missing selected feature columns: {missing_columns}")
    failures: dict[str, dict[str, int]] = {}
    for feature in features:
        series = frame[feature]
        missing = int(series.isna().sum())
        numeric = pd.to_numeric(series, errors="coerce").to_numpy(
            dtype=np.float64,
            copy=False,
        )
        nonfinite = int(np.count_nonzero(~np.isfinite(numeric)))
        if missing or nonfinite:
            failures[feature] = {
                "missing": missing,
                "nonfinite_including_missing": nonfinite,
            }
    if failures:
        raise ValueError(f"selected feature values are invalid: {failures}")


def derive_cms_inspired_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Return v1 plus deterministic CMS-inspired four-jet features."""

    missing = sorted(set(SOURCE_COLUMNS) - set(frame.columns))
    if missing:
        raise ValueError(f"missing source columns: {missing}")
    result = add_v1_derived_features(frame)
    derived_rows: list[dict[str, float]] = []
    jet_fields = [
        (f"j{jet}_pt", f"j{jet}_eta", f"j{jet}_phi", f"j{jet}_mass")
        for jet in range(1, 5)
    ]
    for row in frame.itertuples(index=False):
        values = row._asdict()
        jets = [
            LorentzVector.from_pt_eta_phi_mass(
                float(values[pt]),
                float(values[eta]),
                float(values[phi]),
                float(values[mass]),
            )
            for pt, eta, phi, mass in jet_fields
        ]
        etas = [float(values[eta]) for _, eta, _, _ in jet_fields]
        phis = [float(values[phi]) for _, _, phi, _ in jet_fields]
        dr_values = pairwise_delta_rs(etas, phis)
        minimum, maximum, mean = pairwise_delta_r_summary(dr_values)
        h1_pair, h2_pair = reconstruct_higgs_pairs(jets)
        cross_pairs = cross_higgs_pair_indices(h1_pair, h2_pair)
        cross_values = np.asarray(
            [dr_values[pair] for pair in cross_pairs],
            dtype=np.float64,
        )
        h1 = jets[h1_pair[0]] + jets[h1_pair[1]]
        h2 = jets[h2_pair[0]] + jets[h2_pair[1]]
        hh = h1 + h2
        h1_leading = jets[leading_jet_index(h1_pair, jets)]
        h2_leading = jets[leading_jet_index(h2_pair, jets)]
        cos_star = rest_frame_cosine(h1, hh)
        cos_h1 = rest_frame_cosine(h1_leading, h1)
        cos_h2 = rest_frame_cosine(h2_leading, h2)
        feature_row = {
            "dr_j1_j2": dr_values[(0, 1)],
            "dr_j1_j3": dr_values[(0, 2)],
            "dr_j1_j4": dr_values[(0, 3)],
            "dr_j2_j3": dr_values[(1, 2)],
            "dr_j2_j4": dr_values[(1, 3)],
            "dr_j3_j4": dr_values[(2, 3)],
            "min_candidate_pair_dr": minimum,
            "max_candidate_pair_dr": maximum,
            "mean_candidate_pair_dr": mean,
            "min_cross_higgs_pair_dr": float(np.min(cross_values)),
            "max_cross_higgs_pair_dr": float(np.max(cross_values)),
            "candidate_vector_pt_sum": hh.pt,
            "cos_theta_star_h1_in_hh_rest": cos_star,
            "abs_cos_theta_star_h1_in_hh_rest": abs(cos_star),
            "cos_theta_j1_in_h1_rest": cos_h1,
            "abs_cos_theta_j1_in_h1_rest": abs(cos_h1),
            "cos_theta_j_h2_leading_in_h2_rest": cos_h2,
            "abs_cos_theta_j_h2_leading_in_h2_rest": abs(cos_h2),
        }
        derived_rows.append(feature_row)
    derived = pd.DataFrame.from_records(
        derived_rows,
        columns=list(CMS_INSPIRED_DERIVED_FEATURES),
        index=frame.index,
    )
    for feature in CMS_INSPIRED_DERIVED_FEATURES:
        result[feature] = derived[feature]
    require_finite_features(result, CATEGORIZED_MASS_AWARE_FEATURES)
    return result


def feature_quality_rows(
    frame: pd.DataFrame,
    features: Sequence[str],
    *,
    category: str,
) -> list[dict[str, Any]]:
    """Audit selected features without changing or clipping values."""

    validate_feature_names(features)
    missing_columns = sorted(set(features) - set(frame.columns))
    if missing_columns:
        raise ValueError(f"missing selected feature columns: {missing_columns}")
    rows: list[dict[str, Any]] = []
    for order, feature in enumerate(features, start=1):
        series = frame[feature]
        numeric = pd.to_numeric(series, errors="coerce")
        values = numeric.to_numpy(dtype=np.float64, copy=False)
        missing_mask = series.isna().to_numpy()
        finite_mask = np.isfinite(values)
        finite_values = values[finite_mask]
        missing = int(np.count_nonzero(missing_mask))
        nonfinite = int(np.count_nonzero(~finite_mask & ~missing_mask))
        unique = int(numeric.nunique(dropna=False))
        constant = unique <= 1
        rows.append(
            {
                "category": category,
                "feature_order": order,
                "feature": feature,
                "feature_origin": (
                    "cms_inspired_derived"
                    if feature in CMS_INSPIRED_DERIVED_FEATURES
                    else "frozen_bdt_v1"
                ),
                "rows": len(frame),
                "finite_rows": int(np.count_nonzero(finite_mask)),
                "nonfinite_rows": nonfinite,
                "missing_rows": missing,
                "minimum": (
                    float(np.min(finite_values)) if finite_values.size else ""
                ),
                "maximum": (
                    float(np.max(finite_values)) if finite_values.size else ""
                ),
                "mean": (
                    float(np.mean(finite_values)) if finite_values.size else ""
                ),
                "standard_deviation": (
                    float(np.std(finite_values, ddof=0))
                    if finite_values.size
                    else ""
                ),
                "unique_values": unique,
                "constant": constant,
                "status": (
                    "pass"
                    if missing == 0 and nonfinite == 0 and not constant
                    else "fail"
                ),
            }
        )
    return rows


def _plain_value(value: Any) -> str:
    if isinstance(value, (dict, list, tuple)):
        return json.dumps(value, sort_keys=True, separators=(",", ":"))
    if isinstance(value, (np.bool_, bool)):
        return "true" if bool(value) else "false"
    if isinstance(value, (np.floating, float)):
        return f"{float(value):.12g}"
    if isinstance(value, (np.integer, int)):
        return str(int(value))
    return "" if value is None else str(value)


def latex_escape(value: Any) -> str:
    """Escape a plain-text table cell for LaTeX."""

    text = _plain_value(value)
    replacements = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
        "→": r"$\to$",
        "≥": r"$\geq$",
        "≤": r"$\leq$",
        "γ": r"$\gamma$",
    }
    return "".join(replacements.get(character, character) for character in text)


def write_table_bundle(
    directory: Path,
    name: str,
    rows: Iterable[Mapping[str, Any]],
    fields: Sequence[str],
    *,
    caption: str,
    label: str,
    publication_fields: Sequence[str] | None = None,
    latex_raw_fields: Sequence[str] = (),
) -> tuple[Path, Path, Path]:
    """Write TSV, Markdown, and booktabs-compatible LaTeX table forms."""

    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    materialized = [dict(row) for row in rows]
    fields = tuple(fields)
    if not fields:
        raise ValueError("table fields must be nonempty")
    for row in materialized:
        extras = set(row) - set(fields)
        if extras:
            raise ValueError(f"table row has unexpected fields: {sorted(extras)}")

    tsv_path = directory / f"{name}.tsv"
    with tsv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=list(fields),
            delimiter="\t",
            lineterminator="\n",
            extrasaction="raise",
        )
        writer.writeheader()
        for row in materialized:
            writer.writerow({field: _plain_value(row.get(field, "")) for field in fields})

    shown = tuple(publication_fields or fields)
    if not shown or not set(shown).issubset(fields):
        raise ValueError("publication fields must be a nonempty subset of fields")
    latex_raw = frozenset(latex_raw_fields)
    if not latex_raw.issubset(shown):
        raise ValueError("raw LaTeX fields must be publication fields")

    markdown_path = directory / f"{name}.md"
    markdown_lines = [
        "| " + " | ".join(field.replace("_", " ") for field in shown) + " |",
        "|" + "|".join("---" for _ in shown) + "|",
    ]
    for row in materialized:
        cells = [
            _plain_value(row.get(field, ""))
            .replace("\\", r"\\")
            .replace("|", r"\|")
            .replace("\n", " ")
            for field in shown
        ]
        markdown_lines.append("| " + " | ".join(cells) + " |")
    markdown_path.write_text("\n".join(markdown_lines) + "\n", encoding="utf-8")

    latex_path = directory / f"{name}.tex"
    align = "l" + "r" * (len(shown) - 1)
    wide = len(shown) >= 6
    table_environment = "table*" if wide else "table"
    latex_lines = [
        rf"\begin{{{table_environment}}}[htbp]",
        r"\centering",
        r"\small" if not wide else r"\scriptsize",
        rf"\caption{{{latex_escape(caption)}}}",
        rf"\label{{{latex_escape(label)}}}",
    ]
    if wide:
        latex_lines.extend(
            [r"\setlength{\tabcolsep}{3pt}", r"\resizebox{\textwidth}{!}{%"]
        )
    latex_lines.extend(
        [
            rf"\begin{{tabular}}{{{align}}}",
            r"\toprule",
            " & ".join(
                latex_escape(field.replace("_", " ")) for field in shown
            )
            + r" \\",
            r"\midrule",
        ]
    )
    for row in materialized:
        latex_lines.append(
            " & ".join(
                (
                    _plain_value(row.get(field, ""))
                    if field in latex_raw
                    else latex_escape(row.get(field, ""))
                )
                for field in shown
            )
            + r" \\"
        )
    latex_lines.extend([r"\bottomrule", r"\end{tabular}"])
    if wide:
        latex_lines.append(r"}")
    latex_lines.extend([rf"\end{{{table_environment}}}", ""])
    latex_path.write_text("\n".join(latex_lines), encoding="utf-8")
    return tsv_path, markdown_path, latex_path
