from __future__ import annotations

from pathlib import Path
import sys

SCRIPT_DIR = Path(__file__).resolve().parents[1] / "scripts" / "analysis"
sys.path.insert(0, str(SCRIPT_DIR))

from aggregate_hh4b_initial200_selection_stability import (  # noqa: E402
    AggregationError,
    Candidate,
    EXPECTED_STRUCTURE_IDS,
    TARGET_SIGNAL_EFFICIENCY,
    coordinatewise_median,
    linear_quantile,
    optional_variables,
    rank_fold_candidates,
    reduce_replica_category,
    selection_ranking_key,
)


def candidate(
    structure_id: str,
    *,
    fold: int = 0,
    feasible: bool = True,
    signal: float = TARGET_SIGNAL_EFFICIENCY + 0.001,
    background: float = 0.4,
    cuts: int = 1,
    thresholds: dict[str, float] | None = None,
) -> Candidate:
    return Candidate(
        job_index=fold,
        replica=0,
        outer_fold=fold,
        category_id="exact3tag",
        payload_key="replica_0000__exact3tag",
        structure_id=structure_id,
        category_family_id=f"exact3tag::{structure_id}",
        cut_count=cuts,
        signal_efficiency=signal,
        background_efficiency=background,
        feasible=feasible,
        all_inner_support_pass=True,
        pooled_support_pass=True,
        refit_thresholds=thresholds or {"r_hh_125_125": 30.0 + fold},
        canonical_payload_sha256="a" * 64,
        structure_result_sha256="b" * 64,
        inner_crossfit_sha256="c" * 64,
        execution_provenance_sha256="d" * 64,
        bundle_sha256="e" * 64,
    )


def full_candidate_set(overrides: dict[str, Candidate]) -> list[Candidate]:
    return [overrides.get(structure, candidate(structure)) for structure in EXPECTED_STRUCTURE_IDS]


def test_feasible_ranking_prioritizes_background_before_overshoot() -> None:
    first = EXPECTED_STRUCTURE_IDS[0]
    second = EXPECTED_STRUCTURE_IDS[1]
    values = full_candidate_set({
        first: candidate(first, background=0.1, signal=TARGET_SIGNAL_EFFICIENCY + 0.01, cuts=3),
        second: candidate(second, background=0.2, signal=TARGET_SIGNAL_EFFICIENCY + 0.00001, cuts=1),
    })
    winner = rank_fold_candidates(values)[0].candidate
    assert winner.structure_id == first


def test_any_feasible_candidate_beats_every_infeasible_candidate() -> None:
    feasible_structure = EXPECTED_STRUCTURE_IDS[-1]
    values = [
        candidate(
            structure,
            feasible=structure == feasible_structure,
            signal=(TARGET_SIGNAL_EFFICIENCY if structure == feasible_structure else TARGET_SIGNAL_EFFICIENCY - 1e-8),
            background=0.99 if structure == feasible_structure else 0.01,
        )
        for structure in EXPECTED_STRUCTURE_IDS
    ]
    assert rank_fold_candidates(values)[0].candidate.structure_id == feasible_structure


def test_infeasible_ranking_maximizes_signal_then_minimizes_background() -> None:
    first = EXPECTED_STRUCTURE_IDS[0]
    second = EXPECTED_STRUCTURE_IDS[1]
    first_candidate = candidate(first, feasible=False, signal=0.58, background=0.2)
    second_candidate = candidate(second, feasible=False, signal=0.57, background=0.01)
    assert selection_ranking_key(first_candidate) < selection_ranking_key(second_candidate)


def test_replica_tie_uses_lexicographic_structure_and_only_supporting_folds() -> None:
    lexicographic = EXPECTED_STRUCTURE_IDS[0]
    other = EXPECTED_STRUCTURE_IDS[9]
    final = EXPECTED_STRUCTURE_IDS[18]
    winners = [
        candidate(lexicographic, fold=0, thresholds={"x": 1.0}),
        candidate(other, fold=1, thresholds={"x": 100.0}),
        candidate(lexicographic, fold=2, thresholds={"x": 5.0}),
        candidate(other, fold=3, thresholds={"x": 200.0}),
        candidate(final, fold=4, thresholds={"x": 300.0}),
    ]
    result = reduce_replica_category(winners)
    assert result.tie_resolution_applied is True
    assert result.unique_modal_family is False
    assert result.tied_structure_ids == (lexicographic, other)
    assert result.selected_structure_id == lexicographic
    assert result.supporting_outer_folds == (0, 2)
    assert result.median_thresholds == {"x": 3.0}


def test_coordinatewise_median_requires_identical_coordinates() -> None:
    assert coordinatewise_median([{"x": 1.0, "y": 8.0}, {"x": 3.0, "y": 2.0}]) == {
        "x": 2.0,
        "y": 5.0,
    }
    try:
        coordinatewise_median([{"x": 1.0}, {"y": 2.0}])
    except AggregationError as exc:
        assert "coordinate mismatch" in str(exc)
    else:
        raise AssertionError("mismatched threshold coordinates were accepted")


def test_optional_variable_decoder_is_exact() -> None:
    assert optional_variables("radial_mass__category__mass_only") == frozenset()
    assert optional_variables("radial_mass__category__plus_mhh") == frozenset({"mhh"})
    assert optional_variables(
        "radial_mass__category__plus_ht_candidate_jets_and_max_drbb"
    ) == frozenset({"ht_candidate_jets", "max_drbb"})


def test_linear_quantile_uses_declared_linear_interpolation() -> None:
    values = [0.0, 10.0, 20.0, 30.0]
    assert linear_quantile(values, 0.0) == 0.0
    assert linear_quantile(values, 0.5) == 15.0
    assert linear_quantile(values, 1.0) == 30.0


if __name__ == "__main__":
    tests = sorted(
        (name, value)
        for name, value in globals().items()
        if name.startswith("test_") and callable(value)
    )
    for name, function in tests:
        function()
        print(f"{name}=PASS")
    print(f"TEST_COUNT={len(tests)}")
