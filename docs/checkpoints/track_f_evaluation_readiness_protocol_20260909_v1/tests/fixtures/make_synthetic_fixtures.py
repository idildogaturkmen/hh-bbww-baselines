#!/usr/bin/env python3
"""Generates SYNTHETIC_TEST_FIXTURE_*.npz / .json files -- random numbers
with a controllable injected effect, NOT real model output, used ONLY to
prove code/evaluate_multi_model.py, code/decide_go_no_go.py,
code/build_tables.py, and code/build_figures.py run end-to-end and produce
schema-conformant, sensible output. Every filename this script writes
contains SYNTHETIC_TEST_FIXTURE; every array is drawn from np.random with
a fixed seed. No number here should ever be read as a real physics result.

CORRECTED 2026-09-10 (PREREGISTRATION.md Amendment 2026-09-10): pred_b1..b4/
truth_b1..b4 replace the pre-2026-09-09 higgs1/higgs2/assignment_correct
booleans, per the H1<->H2 permutation-symmetry fix. Critically, this
generator does NOT use a consistent H1-vs-H2 slot convention or a
consistent inner-jet order on EITHER the predicted or the truth side -- for
every event, independently, it randomly decides (a) which physical Higgs
pair is written into the "pair 1" (b1,b2) vs "pair 2" (b3,b4) slots, and
(b) which jet within each pair is written first. This exactly mirrors the
real permutation-symmetric SPA-Net output (event_config/trackb_hh4b.yaml's
PERMUTATIONS: EVENT: [h1,h2], h1: [b1,b2], h2: [b3,b4]) and is a deliberate,
non-negotiable integration-level regression guard: if match_higgs_pairs()
silently relied on a consistent slot convention (the exact bug this
amendment fixes), this generator would expose it immediately, since no
such convention exists in the data it produces.

Three scenarios, matching PREREGISTRATION.md's story definitions so the
smoke test can check the decision machinery reaches the RIGHT conclusion
on data engineered to demonstrate each one:
  - "storyA": TEST's score distribution AND per-Higgs-pair correctness
    probability are both given a real advantage over CONTROL.
  - "storyC": TEST is statistically indistinguishable from CONTROL.
  - "harm": TEST is engineered WORSE than CONTROL on both axes.
"""
import argparse
import json
import os

import numpy as np

N_SIGNAL, N_QCD, N_TTBAR = 2000, 2000, 2000
N_JET_CHOICES = [4, 4, 4, 5, 5, 6]  # weighted toward 4, some 5/6 -- all >=4

# Fixed jet-index roles used internally before random relabeling (see module
# docstring): role A = jets (0,1), role B = jets (2,3). A "wrong" prediction
# for a role uses (4,5)/(6,7) -- placeholder indices chosen to be mutually
# jet-disjoint from ROLE_A, ROLE_B, and each other in EVERY combination of
# role_a_correct/role_b_correct, so a "wrong" prediction for one role can
# never accidentally reuse a jet from the OTHER role's prediction (that
# would produce a spurious "degenerate pair" -- a repeated jet across
# pred_b1..pred_b4 -- as an artifact of fixture construction, not of
# anything under audit here). These indices may exceed a given event's own
# n_jets; nothing in evaluate_multi_model.py currently checks pred_b* <
# n_jets, and this fixture's only job is to exercise the statistical/
# symmetry logic, not physical jet-count consistency.
ROLE_A = (0, 1)
ROLE_B = (2, 3)
WRONG_A = (4, 5)
WRONG_B = (6, 7)


def _apply_inner_swap(pair0, pair1, swap_mask):
    out0 = np.where(swap_mask, pair1, pair0)
    out1 = np.where(swap_mask, pair0, pair1)
    return out0, out1


def make_arm(rng, effect_size):
    n = N_SIGNAL + N_QCD + N_TTBAR
    process = np.concatenate([np.zeros(N_SIGNAL, dtype=np.int8),
                               np.ones(N_QCD, dtype=np.int8),
                               np.full(N_TTBAR, 2, dtype=np.int8)])
    event_id = np.arange(n, dtype=np.int64)

    is_signal = process == 0
    base = rng.normal(0.0, 1.0, size=n)
    score = np.where(is_signal, base + 0.6 + effect_size, base)
    score = score.astype(np.float32)

    n_jets = rng.choice(N_JET_CHOICES, size=n).astype(np.int8)

    # ---- truth: role A = (0,1), role B = (2,3), for signal events only ----
    assignment_defined = is_signal.copy()
    swap_truth_h1_inner = rng.random(n) < 0.5
    swap_truth_h2_inner = rng.random(n) < 0.5
    swap_truth_outer = rng.random(n) < 0.5

    roleA0, roleA1 = _apply_inner_swap(np.full(n, ROLE_A[0]), np.full(n, ROLE_A[1]), swap_truth_h1_inner)
    roleB0, roleB1 = _apply_inner_swap(np.full(n, ROLE_B[0]), np.full(n, ROLE_B[1]), swap_truth_h2_inner)

    truth_b1 = np.where(swap_truth_outer, roleB0, roleA0)
    truth_b2 = np.where(swap_truth_outer, roleB1, roleA1)
    truth_b3 = np.where(swap_truth_outer, roleA0, roleB0)
    truth_b4 = np.where(swap_truth_outer, roleA1, roleB1)
    truth_b1 = np.where(assignment_defined, truth_b1, -1).astype(np.int64)
    truth_b2 = np.where(assignment_defined, truth_b2, -1).astype(np.int64)
    truth_b3 = np.where(assignment_defined, truth_b3, -1).astype(np.int64)
    truth_b4 = np.where(assignment_defined, truth_b4, -1).astype(np.int64)

    # ---- predicted: independent per-role correctness probability, independent
    # outer/inner relabeling from truth's own (proves no shared convention is
    # required for correct scoring) ----
    p_correct = np.clip(0.45 + effect_size * 0.5, 0.01, 0.99)
    role_a_correct = rng.random(n) < p_correct
    role_b_correct = rng.random(n) < p_correct

    predA0 = np.where(role_a_correct, ROLE_A[0], WRONG_A[0])
    predA1 = np.where(role_a_correct, ROLE_A[1], WRONG_A[1])
    predB0 = np.where(role_b_correct, ROLE_B[0], WRONG_B[0])
    predB1 = np.where(role_b_correct, ROLE_B[1], WRONG_B[1])

    swap_pred_h1_inner = rng.random(n) < 0.5
    swap_pred_h2_inner = rng.random(n) < 0.5
    swap_pred_outer = rng.random(n) < 0.5

    predA0, predA1 = _apply_inner_swap(predA0, predA1, swap_pred_h1_inner)
    predB0, predB1 = _apply_inner_swap(predB0, predB1, swap_pred_h2_inner)

    pred_b1 = np.where(swap_pred_outer, predB0, predA0).astype(np.int64)
    pred_b2 = np.where(swap_pred_outer, predB1, predA1).astype(np.int64)
    pred_b3 = np.where(swap_pred_outer, predA0, predB0).astype(np.int64)
    pred_b4 = np.where(swap_pred_outer, predA1, predB1).astype(np.int64)

    higgs_mass_1 = np.where(is_signal, rng.normal(125.0, 15.0, size=n), rng.normal(90.0, 30.0, size=n)).astype(np.float32)
    higgs_mass_2 = np.where(is_signal, rng.normal(125.0, 15.0, size=n), rng.normal(90.0, 30.0, size=n)).astype(np.float32)

    return dict(event_id=event_id, process=process, score=score,
                assignment_defined=assignment_defined,
                pred_b1=pred_b1, pred_b2=pred_b2, pred_b3=pred_b3, pred_b4=pred_b4,
                truth_b1=truth_b1, truth_b2=truth_b2, truth_b3=truth_b3, truth_b4=truth_b4,
                higgs_mass_1=higgs_mass_1, higgs_mass_2=higgs_mass_2, n_jets=n_jets)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    rng_control = np.random.default_rng(1000 + args.seed)
    control = make_arm(rng_control, effect_size=0.0)
    np.savez(os.path.join(args.out_dir, f"SYNTHETIC_TEST_FIXTURE_control_seed{args.seed}.npz"), **control)

    rng_a = np.random.default_rng(2000 + args.seed)
    test_a = make_arm(rng_a, effect_size=0.35)
    test_a["event_id"] = control["event_id"]
    test_a["process"] = control["process"]
    np.savez(os.path.join(args.out_dir, f"SYNTHETIC_TEST_FIXTURE_test_storyA_seed{args.seed}.npz"), **test_a)

    rng_c = np.random.default_rng(3000 + args.seed)
    test_c = make_arm(rng_c, effect_size=0.0)
    test_c["event_id"] = control["event_id"]
    test_c["process"] = control["process"]
    np.savez(os.path.join(args.out_dir, f"SYNTHETIC_TEST_FIXTURE_test_storyC_seed{args.seed}.npz"), **test_c)

    rng_h = np.random.default_rng(4000 + args.seed)
    test_harm = make_arm(rng_h, effect_size=-0.5)
    test_harm["event_id"] = control["event_id"]
    test_harm["process"] = control["process"]
    np.savez(os.path.join(args.out_dir, f"SYNTHETIC_TEST_FIXTURE_test_harm_seed{args.seed}.npz"), **test_harm)

    native10m_agg = {
        "_synthetic_test_fixture": True,
        "checkpoint_sha256": "0" * 64,
        "population": {"n_scored": N_SIGNAL + N_QCD + N_TTBAR, "n_signal": N_SIGNAL, "n_qcd": N_QCD, "n_ttbar": N_TTBAR},
        "roc_auc": {"signal_vs_all_background": 0.83, "signal_vs_qcd": 0.85, "signal_vs_ttbar": 0.80},
        "working_points": [
            {"target_signal_efficiency": eff, "threshold": 0.0, "achieved_signal_efficiency": eff,
             "n_signal_pass": int(eff * N_SIGNAL), "n_signal_total": N_SIGNAL,
             "all_background_rejection": 20.0, "all_background_efficiency": 0.05,
             "n_all_background_pass": 100, "n_all_background_total": N_QCD + N_TTBAR,
             "qcd_rejection": 25.0, "qcd_efficiency": 0.04, "n_qcd_pass": 80, "n_qcd_total": N_QCD,
             "ttbar_rejection": 15.0, "ttbar_efficiency": 0.067, "n_ttbar_pass": 20, "n_ttbar_total": N_TTBAR}
            for eff in [0.10, 0.075, 0.05, 0.04, 0.03]
        ],
    }
    with open(os.path.join(args.out_dir, "SYNTHETIC_TEST_FIXTURE_native10m_aggregate.json"), "w") as f:
        json.dump(native10m_agg, f, indent=2)

    print(f"Wrote synthetic fixtures to {args.out_dir}")


if __name__ == "__main__":
    main()
