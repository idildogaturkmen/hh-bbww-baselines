"""Single source of truth for every frozen input path this package reads.

Read-only. No path in this file is ever opened for writing by any script in
this package. holdout_B, holdout_Q, inference_qcd_1, inference_qcd_2 are not
referenced anywhere in this file or any script that imports it.
"""
from pathlib import Path

TRACK_A_ROOT = Path("/uscms_data/d3/iturkmen/hh4b_delphes/track_a_mechanism_benchmark_20260814_v1")
TRACK_B_ROOT = Path("/uscms_data/d3/iturkmen/hh4b_delphes/track_b_phase4_preflight_20260812")

# Track-A: post_training_evaluation_20260815_v1 is explicitly NOT authoritative
# and is never referenced below.
TRACK_A_EVAL_V2 = TRACK_A_ROOT / "post_training_evaluation_20260815_v2"
TRACK_A_MECH_INTERP = TRACK_A_ROOT / "mechanism_interpretation_20260817_v1"

TRACK_A_EVALUATION_RESULTS_JSON = TRACK_A_EVAL_V2 / "EVALUATION_RESULTS.json"
TRACK_A_SHA256SUMS = TRACK_A_EVAL_V2 / "SHA256SUMS"
TRACK_A_SUPPORTED_REJECTION_RANGE_TSV = TRACK_A_MECH_INTERP / "TRACKA_SUPPORTED_REJECTION_RANGE.tsv"
TRACK_A_MECH_INTERP_SHA256SUMS = TRACK_A_MECH_INTERP / "SHA256SUMS"

PHASE4L = TRACK_B_ROOT / "phase4L_event_transformer_development_training_20260813_v1"
PHASE4M = TRACK_B_ROOT / "phase4M_holdout_a_adjudication_20260813_v1"
PHASE4O = TRACK_B_ROOT / "phase4O_trackb_publication_transfer_readiness_20260814_v1"
PHASE4W = TRACK_B_ROOT / "phase4W_historical_reconciliation_20260817_v1"

TRACK_B_HOLDOUT_A_WORKING_POINTS_TSV = PHASE4M / "HOLDOUT_A_WORKING_POINTS.tsv"
TRACK_B_HOLDOUT_A_SEED_STABILITY_TSV = PHASE4M / "HOLDOUT_A_SEED_STABILITY.tsv"
TRACK_B_HOLDOUT_A_ADJUDICATION_RESULT_MD = PHASE4M / "HOLDOUT_A_ADJUDICATION_RESULT.md"
TRACK_B_PHASE4M_SHA256SUMS = PHASE4M / "SHA256SUMS"
TRACK_B_PHASE4O_SHA256SUMS = PHASE4O / "SHA256SUMS"
TRACK_B_PHASE4W_SHA256SUMS = PHASE4W / "SHA256SUMS"

# Explicitly forbidden -- never opened by this package. Listed here only so a
# grep for these names in scripts/ finds this comment, not a live code path.
FORBIDDEN_TOKENS = (
    "holdout_B", "holdout_Q", "inference_qcd_1", "inference_qcd_2",
)

ALL_SHA256SUMS_DIRS = [
    TRACK_A_EVAL_V2,
    TRACK_A_MECH_INTERP,
    PHASE4L,
    PHASE4M,
    PHASE4O,
    PHASE4W,
]
