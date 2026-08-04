from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from scripts.analysis.pn_c7_ml_common import source_member_bootstrap_draws
from scripts.analysis.pn_c7x_spanet_evaluation import (
    ASSIGNMENT_METRICS,
    assignment_paired_differences,
    evaluate_assignment_bootstrap,
    evaluate_binned_pairing,
    summarize_assignment_replicas,
    summarize_binned_pairing,
)


C7S = Path("/uscms_data/d3/iturkmen/hh4b_delphes/harvest_checkpoints/pn_c7s_final_train_baseline_inputs_20260802_v1")
TRAINING = Path("/uscms_data/d3/iturkmen/hh4b_delphes/harvest_checkpoints/pn_c7x_single_head_spanet_training_20260803_v1")


def inputs():
    members = pd.read_csv(C7S / "member_fold_registry.tsv", sep="\t")
    predictions = pd.read_parquet(TRAINING / "predictions/train_fourb_single_head_oof.parquet")
    draws, _ = source_member_bootstrap_draws(members, replicates=20, seed=20260802)
    return members, predictions, draws


def test_assignment_bootstrap_resamples_complete_source_members_and_is_deterministic():
    members, predictions, draws = inputs()
    evaluated = evaluate_assignment_bootstrap(predictions, members, draws)
    repeated = evaluate_assignment_bootstrap(predictions, members, draws)
    pd.testing.assert_frame_equal(evaluated, repeated)
    assert len(evaluated) == 21
    assert evaluated.replica_id.tolist() == [-1] + list(range(20))
    assert set(evaluated.matchable_rows) != {int(predictions.assignment_loss_mask.sum())}
    assert np.isfinite(evaluated[list(ASSIGNMENT_METRICS)].to_numpy()).all()


def test_assignment_summaries_and_paired_differences_align_all_replicas():
    members, predictions, draws = inputs()
    evaluated = evaluate_assignment_bootstrap(predictions, members, draws)
    summary = summarize_assignment_replicas(evaluated)
    paired, paired_replicas = assignment_paired_differences(evaluated)
    assert set(summary.metric) == set(ASSIGNMENT_METRICS)
    assert summary.valid_replicas.eq(20).all()
    assert paired.valid_replicas.eq(20).all()
    assert paired_replicas.groupby("metric").replica_id.nunique().eq(20).all()
    accuracy = paired[paired.metric.eq("exact_event_pairing_accuracy")].iloc[0]
    assert accuracy.nominal_difference > 0


def test_every_required_pairing_bin_has_source_bootstrap_interval_and_paired_difference():
    members, predictions, draws = inputs()
    evaluated = evaluate_binned_pairing(predictions, members, draws)
    summary, paired = summarize_binned_pairing(evaluated)
    assert set(summary.variable) == {
        "mHH_GeV", "truth_Higgs_pT_GeV", "selected_jet_multiplicity", "extra_jet_activity"
    }
    assert set(summary.model) == {"single_head_learned", "frozen_geometric"}
    assert summary.valid_replicas.eq(20).all()
    assert paired.valid_replicas.eq(20).all()
    expected_bins = summary[["variable", "bin_index"]].drop_duplicates()
    assert len(paired) == len(expected_bins)
