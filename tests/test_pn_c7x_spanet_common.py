from __future__ import annotations

import numpy as np
import pytest

from scripts.analysis.pn_c7x_spanet_common import (
    MATCHING_TO_LABEL,
    PERFECT_MATCHINGS,
    resolve_truth_partition,
    safe_archive_member,
    unique_higgs_daughter_pairs,
)


def truth_event():
    # Two H->bb decays plus an upstream duplicate Higgs copy of the first decay.
    pid = np.array([25, 25, 5, -5, 5, -5, 25], dtype=np.int32)
    d1 = np.array([2, 4, -1, -1, -1, -1, 0], dtype=np.int32)
    d2 = np.array([3, 5, -1, -1, -1, -1, 0], dtype=np.int32)
    particle_eta = np.array([0.0, 0.0, 0.10, -0.10, 1.00, 1.20, 0.0])
    particle_phi = np.array([0.0, 0.0, 0.20, -0.20, 1.00, 1.20, 0.0])
    jet_eta = np.array([0.11, 1.01, -0.11, 1.19])
    jet_phi = np.array([0.19, 1.01, -0.19, 1.19])
    return pid, d1, d2, particle_eta, particle_phi, jet_eta, jet_phi


def test_perfect_matching_registry_is_complete_and_symmetric():
    assert len(PERFECT_MATCHINGS) == 3
    assert set(MATCHING_TO_LABEL.values()) == {0, 1, 2}
    for matching in PERFECT_MATCHINGS:
        assert sorted(vertex for pair in matching for vertex in pair) == [0, 1, 2, 3]
        assert all(left < right for left, right in matching)
        assert matching[0] < matching[1]


def test_duplicate_higgs_copies_collapse_by_daughter_identity():
    pid, d1, d2, *_ = truth_event()
    pairs, ambiguous = unique_higgs_daughter_pairs(pid, d1, d2)
    assert pairs == ((2, 3), (4, 5))
    assert ambiguous is False


def test_unique_truth_partition_collapses_b_and_higgs_interchange():
    result = resolve_truth_partition(*truth_event(), (0, 2, 1, 3), dr_max=0.4)
    assert result.status == "matchable"
    assert result.label == 0
    assert result.compatible_partitions == 1
    assert result.daughter_pairs == ((2, 3), (4, 5))


def test_unmatched_truth_is_not_given_a_label():
    values = list(truth_event())
    values[-2] = np.array([4.0, 4.1, 4.2, 4.3])
    result = resolve_truth_partition(*values, (0, 1, 2, 3), dr_max=0.4)
    assert result.status == "unmatched"
    assert result.label == -1
    assert result.compatible_partitions == 0


def test_multiple_compatible_partitions_fail_ambiguous():
    pid, d1, d2, particle_eta, particle_phi, _, _ = truth_event()
    jet_eta = np.full(4, 0.5)
    jet_phi = np.full(4, 0.5)
    particle_eta[2:6] = 0.5
    particle_phi[2:6] = 0.5
    result = resolve_truth_partition(
        pid, d1, d2, particle_eta, particle_phi, jet_eta, jet_phi, (0, 1, 2, 3), dr_max=0.4
    )
    assert result.status == "ambiguous_match"
    assert result.label == -1
    assert result.compatible_partitions == 3


def test_higgs_with_more_than_two_b_descendants_fails_ambiguous():
    pid = np.array([25, 25, 5, -5, 5, 5, -5], dtype=np.int32)
    d1 = np.array([2, 5, -1, -1, -1, -1, -1], dtype=np.int32)
    d2 = np.array([4, 6, -1, -1, -1, -1, -1], dtype=np.int32)
    coordinates = np.zeros(7)
    jets = np.zeros(4)
    result = resolve_truth_partition(
        pid, d1, d2, coordinates, coordinates, jets, jets, (0, 1, 2, 3), dr_max=0.4
    )
    assert result.status == "ambiguous_truth"
    assert result.label == -1


def test_candidate_indices_must_be_four_unique_in_range():
    with pytest.raises(RuntimeError, match="not unique"):
        resolve_truth_partition(*truth_event(), (0, 0, 1, 2))
    with pytest.raises(RuntimeError, match="out of range"):
        resolve_truth_partition(*truth_event(), (0, 1, 2, 99))


@pytest.mark.parametrize("member", ["root/a.root", "./products/reconstruction.tar.gz"])
def test_safe_archive_members(member):
    assert safe_archive_member(member) == member


@pytest.mark.parametrize("member", ["", "/root/a.root", "../a.root", "root/../../a.root"])
def test_unsafe_archive_members_fail_closed(member):
    with pytest.raises(RuntimeError):
        safe_archive_member(member)
