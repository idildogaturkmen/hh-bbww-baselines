#!/usr/bin/env python3
"""Shared, torch-free contracts for the PN-c7x single-head SPA-Net study."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from pathlib import PurePosixPath, Path
import zlib

import numpy as np


PERFECT_MATCHINGS: tuple[tuple[tuple[int, int], tuple[int, int]], ...] = (
    ((0, 1), (2, 3)),
    ((0, 2), (1, 3)),
    ((0, 3), (1, 2)),
)
MATCHING_TO_LABEL = {matching: label for label, matching in enumerate(PERFECT_MATCHINGS)}


@dataclass(frozen=True)
class TruthPartitionResult:
    status: str
    label: int
    unique_higgs_daughter_pairs: int
    compatible_partitions: int
    daughter_pairs: tuple[tuple[int, int], ...]


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def sha256(path: Path, chunk_bytes: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_bytes), b""):
            digest.update(chunk)
    return digest.hexdigest()


def adler32(path: Path, chunk_bytes: int = 8 * 1024 * 1024) -> str:
    value = 1
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_bytes), b""):
            value = zlib.adler32(chunk, value)
    return f"{value & 0xFFFFFFFF:08x}"


def safe_archive_member(name: str) -> str:
    """Validate a registry-bound POSIX archive member without normalizing it."""
    require(bool(name), "empty archive member")
    member = PurePosixPath(name)
    require(not member.is_absolute(), f"absolute archive member rejected: {name}")
    require(".." not in member.parts, f"parent traversal archive member rejected: {name}")
    require("" not in member.parts, f"empty archive component rejected: {name}")
    return name


def delta_phi(left: float, right: float) -> float:
    return float((left - right + np.pi) % (2.0 * np.pi) - np.pi)


def delta_r(eta_left: float, phi_left: float, eta_right: float, phi_right: float) -> float:
    return float(np.hypot(eta_left - eta_right, delta_phi(phi_left, phi_right)))


def _children(d1: np.ndarray, d2: np.ndarray, index: int) -> tuple[int, ...]:
    size = len(d1)
    if index < 0 or index >= size:
        return ()
    first = int(d1[index])
    last = int(d2[index])
    if first < 0 or last < first or first >= size:
        return ()
    return tuple(range(first, min(last, size - 1) + 1))


def b_descendants(
    pid: np.ndarray,
    d1: np.ndarray,
    d2: np.ndarray,
    higgs_index: int,
    max_depth: int = 8,
) -> tuple[int, ...]:
    """Return every unique first b descendant reached from one Higgs copy."""
    found: list[int] = []
    seen: set[int] = set()

    def walk(index: int, depth: int) -> None:
        if depth > max_depth or index in seen:
            return
        seen.add(index)
        for child in _children(d1, d2, index):
            if abs(int(pid[child])) == 5:
                if child not in found:
                    found.append(child)
            else:
                walk(child, depth + 1)

    walk(higgs_index, 0)
    return tuple(found)


def unique_higgs_daughter_pairs(
    pid: np.ndarray,
    d1: np.ndarray,
    d2: np.ndarray,
) -> tuple[tuple[tuple[int, int], ...], bool]:
    """Collapse duplicate Higgs copies; flag any Higgs with more than two b descendants."""
    pairs: list[tuple[int, int]] = []
    ambiguous_decay = False
    for higgs_index in np.flatnonzero(np.asarray(pid) == 25):
        descendants = b_descendants(pid, d1, d2, int(higgs_index))
        if len(descendants) > 2:
            ambiguous_decay = True
            continue
        if len(descendants) == 2:
            pair = tuple(sorted((int(descendants[0]), int(descendants[1]))))
            if pair not in pairs:
                pairs.append(pair)
    return tuple(pairs), ambiguous_decay


def compatible_jet_pairs(
    daughter_pair: tuple[int, int],
    candidate_raw_indices: tuple[int, int, int, int],
    particle_eta: np.ndarray,
    particle_phi: np.ndarray,
    jet_eta: np.ndarray,
    jet_phi: np.ndarray,
    dr_max: float,
) -> tuple[tuple[int, int], ...]:
    compatible: list[tuple[int, int]] = []
    b0, b1 = daughter_pair
    for local_left in range(4):
        for local_right in range(local_left + 1, 4):
            jet_left = candidate_raw_indices[local_left]
            jet_right = candidate_raw_indices[local_right]
            direct = (
                delta_r(particle_eta[b0], particle_phi[b0], jet_eta[jet_left], jet_phi[jet_left]) < dr_max
                and delta_r(particle_eta[b1], particle_phi[b1], jet_eta[jet_right], jet_phi[jet_right]) < dr_max
            )
            swapped = (
                delta_r(particle_eta[b0], particle_phi[b0], jet_eta[jet_right], jet_phi[jet_right]) < dr_max
                and delta_r(particle_eta[b1], particle_phi[b1], jet_eta[jet_left], jet_phi[jet_left]) < dr_max
            )
            if direct or swapped:
                compatible.append((local_left, local_right))
    return tuple(compatible)


def resolve_truth_partition(
    pid: np.ndarray,
    d1: np.ndarray,
    d2: np.ndarray,
    particle_eta: np.ndarray,
    particle_phi: np.ndarray,
    jet_eta: np.ndarray,
    jet_phi: np.ndarray,
    candidate_raw_indices: tuple[int, int, int, int],
    dr_max: float = 0.4,
) -> TruthPartitionResult:
    require(len(candidate_raw_indices) == 4, "truth partition requires exactly four candidate jets")
    require(len(set(candidate_raw_indices)) == 4, "candidate raw jet indices are not unique")
    require(all(0 <= index < len(jet_eta) for index in candidate_raw_indices), "candidate raw jet index out of range")
    daughter_pairs, ambiguous_decay = unique_higgs_daughter_pairs(pid, d1, d2)
    if ambiguous_decay or len(daughter_pairs) > 2:
        return TruthPartitionResult("ambiguous_truth", -1, len(daughter_pairs), 0, daughter_pairs)
    if len(daughter_pairs) < 2:
        return TruthPartitionResult("truth_unavailable", -1, len(daughter_pairs), 0, daughter_pairs)

    pair_options = [
        compatible_jet_pairs(
            pair,
            candidate_raw_indices,
            particle_eta,
            particle_phi,
            jet_eta,
            jet_phi,
            dr_max,
        )
        for pair in daughter_pairs
    ]
    partitions: set[tuple[tuple[int, int], tuple[int, int]]] = set()
    for first in pair_options[0]:
        for second in pair_options[1]:
            if set(first).isdisjoint(second):
                partitions.add(tuple(sorted((first, second))))
    if len(partitions) == 0:
        return TruthPartitionResult("unmatched", -1, 2, 0, daughter_pairs)
    if len(partitions) > 1:
        return TruthPartitionResult("ambiguous_match", -1, 2, len(partitions), daughter_pairs)
    matching = next(iter(partitions))
    require(matching in MATCHING_TO_LABEL, f"unexpected perfect matching {matching}")
    return TruthPartitionResult("matchable", MATCHING_TO_LABEL[matching], 2, 1, daughter_pairs)


def massless_pair_pt(
    daughter_pair: tuple[int, int],
    particle_pt: np.ndarray,
    particle_phi: np.ndarray,
) -> float:
    """Vector-sum pT of the two truth b daughters; independent of Higgs-copy choice."""
    first, second = daughter_pair
    px = particle_pt[first] * np.cos(particle_phi[first]) + particle_pt[second] * np.cos(particle_phi[second])
    py = particle_pt[first] * np.sin(particle_phi[first]) + particle_pt[second] * np.sin(particle_phi[second])
    return float(np.hypot(px, py))
