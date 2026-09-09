#!/usr/bin/env python3
"""Unit tests proving code/evaluate_multi_model.py's match_higgs_pairs() is
genuinely invariant under every symmetry declared in this project's own
SPA-Net event topology (event_config/trackb_hh4b.yaml):
    PERMUTATIONS:
        EVENT: [ h1, h2 ]     -- the two Higgs candidates are interchangeable
        h1: [ b1, b2 ]        -- h1's two jets are interchangeable
        h2: [ b3, b4 ]        -- h2's two jets are interchangeable

Written as part of PREREGISTRATION.md Amendment 2026-09-10, BEFORE any
ParT2M result exists -- this is exactly the audit the amendment records.

Pure unit tests against match_higgs_pairs() directly (no .npz, no CLI) --
small, hand-constructed integer arrays only. Every property required by
the task is its own test function so a failure names the specific broken
invariant, not a generic "something's wrong".
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "code"))
from evaluate_multi_model import match_higgs_pairs  # noqa: E402

PASS = []
FAIL = []


def check(desc, cond):
    if cond:
        PASS.append(desc)
        print(f"  PASS: {desc}")
    else:
        FAIL.append(desc)
        print(f"  FAIL: {desc}")


def arr(*vals):
    return np.array(vals, dtype=np.int64)


def run_case(pred_b1, pred_b2, pred_b3, pred_b4, truth_b1, truth_b2, truth_b3, truth_b4):
    n_correct, exact = match_higgs_pairs(arr(pred_b1), arr(pred_b2), arr(pred_b3), arr(pred_b4),
                                          arr(truth_b1), arr(truth_b2), arr(truth_b3), arr(truth_b4))
    return int(n_correct[0]), bool(exact[0])


def main():
    print("=== test_symmetry_invariance ===")

    # ---- Base case: predicted pairs exactly match truth pairs, no swap. ----
    n_correct_base, exact_base = run_case(0, 1, 2, 3, 0, 1, 2, 3)
    check("base case (both pairs correct, direct order): n_correct_higgs == 2", n_correct_base == 2)
    check("base case: exact_event_correct == True", exact_base is True)

    # ---- Property 1: swapping PREDICTED H1/H2 does not change either metric. ----
    n_correct_pred_swapped, exact_pred_swapped = run_case(2, 3, 0, 1, 0, 1, 2, 3)
    check("swapping predicted H1/H2 (pred pairs reordered): n_correct_higgs unchanged",
          n_correct_pred_swapped == n_correct_base)
    check("swapping predicted H1/H2: exact_event_correct unchanged", exact_pred_swapped == exact_base)

    # ---- Property 2: swapping TRUTH H1/H2 does not change either metric. ----
    n_correct_truth_swapped, exact_truth_swapped = run_case(0, 1, 2, 3, 2, 3, 0, 1)
    check("swapping truth H1/H2 (truth pairs reordered): n_correct_higgs unchanged",
          n_correct_truth_swapped == n_correct_base)
    check("swapping truth H1/H2: exact_event_correct unchanged", exact_truth_swapped == exact_base)

    # ---- Property 3: swapping the two jets INSIDE any pair does not change either metric. ----
    # Swap inside predicted pair 1 (0,1 -> 1,0), predicted pair 2 (2,3 -> 3,2),
    # AND inside truth pair 1 and truth pair 2, all at once -- if this still
    # matches, inner-pair ordering genuinely never matters, on both sides at once.
    n_correct_inner_swapped, exact_inner_swapped = run_case(1, 0, 3, 2, 1, 0, 3, 2)
    check("swapping jets inside every pair (both pred and truth): n_correct_higgs unchanged",
          n_correct_inner_swapped == n_correct_base)
    check("swapping jets inside every pair: exact_event_correct unchanged", exact_inner_swapped == exact_base)

    # Also check inner-pair swap on ONLY the predicted side (truth left in direct order).
    n_correct_inner_pred_only, exact_inner_pred_only = run_case(1, 0, 3, 2, 0, 1, 2, 3)
    check("swapping jets inside predicted pairs only: n_correct_higgs unchanged",
          n_correct_inner_pred_only == n_correct_base)
    check("swapping jets inside predicted pairs only: exact_event_correct unchanged",
          exact_inner_pred_only == exact_base)

    # ---- Property 4: exactly one correct Higgs pair -> 0.5 per-Higgs accuracy, exact_event False. ----
    # Predicted pair 1 = truth pair 1 (correct); predicted pair 2 = jets {4,5}, matches NEITHER truth pair.
    n_correct_one, exact_one = run_case(0, 1, 4, 5, 0, 1, 2, 3)
    check("one correct Higgs pair: n_correct_higgs == 1", n_correct_one == 1)
    check("one correct Higgs pair: per-Higgs accuracy (n_correct_higgs/2) == 0.5", n_correct_one / 2.0 == 0.5)
    check("one correct Higgs pair: exact_event_correct == False", exact_one is False)

    # ---- Property 5: both correct (already covered by base case, restated explicitly). ----
    check("both correct: per-Higgs accuracy == 1.0", n_correct_base / 2.0 == 1.0)
    check("both correct: exact_event_correct == True (restated)", exact_base is True)

    # ---- Property 6: neither correct -> 0. ----
    n_correct_none, exact_none = run_case(4, 5, 6, 7, 0, 1, 2, 3)
    check("neither correct: n_correct_higgs == 0", n_correct_none == 0)
    check("neither correct: per-Higgs accuracy == 0.0", n_correct_none / 2.0 == 0.0)
    check("neither correct: exact_event_correct == False", exact_none is False)

    # ---- Vectorized consistency: running all cases together in one call must give
    # the same per-row results as running them one at a time (no cross-row leakage). ----
    pred_b1 = arr(0, 2, 1, 0, 4)
    pred_b2 = arr(1, 3, 0, 1, 5)
    pred_b3 = arr(2, 0, 3, 4, 6)
    pred_b4 = arr(3, 1, 2, 5, 7)
    truth_b1 = arr(0, 0, 1, 0, 0)
    truth_b2 = arr(1, 1, 0, 1, 1)
    truth_b3 = arr(2, 2, 3, 2, 2)
    truth_b4 = arr(3, 3, 2, 3, 3)
    n_correct_batch, exact_batch = match_higgs_pairs(pred_b1, pred_b2, pred_b3, pred_b4,
                                                        truth_b1, truth_b2, truth_b3, truth_b4)
    expected_n_correct = [2, 2, 2, 1, 0]
    expected_exact = [True, True, True, False, False]
    check("vectorized batch matches expected n_correct_higgs for every row",
          list(n_correct_batch) == expected_n_correct)
    check("vectorized batch matches expected exact_event_correct for every row",
          list(exact_batch) == expected_exact)

    print(f"\n=== test_symmetry_invariance SUMMARY: {len(PASS)} passed, {len(FAIL)} failed (of {len(PASS) + len(FAIL)}) ===")
    if FAIL:
        for f in FAIL:
            print(f"  - {f}")
    return 0 if not FAIL else 1


if __name__ == "__main__":
    sys.exit(main())
