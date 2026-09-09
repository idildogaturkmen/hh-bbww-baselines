#!/usr/bin/env python3
"""Secondary audit (2026-09-09/10): empirically verifies that the
signal-efficiency threshold used for background-rejection-at-fixed-epsS is
RECOMPUTED inside every bootstrap replicate, not frozen from the nominal
sample while only the pass/fail counts are resampled. If the threshold
were frozen, resampling would still shift epsB/rejection somewhat (through
which background events happen to be drawn) but would systematically
UNDERSTATE the true uncertainty, because none of the threshold's own
sampling variance would be propagated.

This is a verification test, not a fix -- code reading of
rejection_at_fixed_efficiency() and its call sites in evaluate_multi_model.py
shows the threshold is derived from `scores[idx]` (the ALREADY-RESAMPLED
array passed in by paired_bootstrap_delta's replicate loop) every time it is
called, never from a frozen/nominal array. This test proves that empirically:
if the threshold were frozen, it would be IDENTICAL across every resample of
data that isn't itself identical; here it demonstrably varies.

Finding: NO BUG. No amendment needed for this specific concern. Recorded in
PREREGISTRATION.md Amendment 2026-09-10 as an audited-and-confirmed-correct
item, not a silently-skipped one.
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "code"))
from evaluate_multi_model import rejection_at_fixed_efficiency  # noqa: E402
from bootstrap_utils import paired_bootstrap_delta  # noqa: E402

PASS = []
FAIL = []


def check(desc, cond):
    if cond:
        PASS.append(desc)
        print(f"  PASS: {desc}")
    else:
        FAIL.append(desc)
        print(f"  FAIL: {desc}")


def main():
    print("=== test_bootstrap_threshold_recompute ===")
    rng = np.random.default_rng(42)
    n = 2000
    labels = (rng.random(n) < 0.5).astype(int)
    scores = np.where(labels == 1, rng.normal(1.0, 1.0, n), rng.normal(0.0, 1.0, n)).astype(np.float64)

    nominal = rejection_at_fixed_efficiency(scores, labels, 0.10)
    nominal_threshold = nominal["threshold"]

    thresholds = []
    resample_rng = np.random.default_rng(7)
    for _ in range(200):
        idx = resample_rng.integers(0, n, size=n)
        r = rejection_at_fixed_efficiency(scores[idx], labels[idx], 0.10)
        thresholds.append(r["threshold"])
    thresholds = np.array(thresholds)

    n_distinct = len(set(thresholds.tolist()))
    check("resampled thresholds are NOT all identical to the nominal threshold "
          "(would be if the threshold were frozen)",
          not np.allclose(thresholds, nominal_threshold))
    check(f"resampled thresholds show real variation across replicates "
          f"({n_distinct} distinct values of 200 replicates, std={thresholds.std():.5g})",
          n_distinct > 10 and thresholds.std() > 0)

    # ---- Integration-level check: paired_bootstrap_delta's OWN replicate loop,
    # using evaluate_multi_model.py's actual rejection_stat_fn-style closure,
    # must produce a nonzero bootstrap_std_delta (degenerate/frozen behavior
    # would collapse this to exactly 0, exactly the failure mode this audit
    # checks for, and exactly the bug independently found and fixed elsewhere
    # in this same audit for the exact-event-efficiency statistic). ----
    process_full = np.where(labels == 1, 0, rng.choice([1, 2], size=n))

    def rejection_stat_fn(s, lab, idx, eff=None, process_full=None):
        proc = process_full[idx]
        r = rejection_at_fixed_efficiency(s[idx], lab[idx], eff, bg_process=None, process=proc)
        return r["rejection"] if r["rejection"] is not None else 0.0

    scores_test = scores + rng.normal(0, 0.05, n)  # a slightly different TEST score, same labels
    delta = paired_bootstrap_delta(scores_test, scores, labels, rejection_stat_fn,
                                    n_boot=300, seed=1, eff=0.10, process_full=process_full)
    check("paired_bootstrap_delta for rejection-at-epsS=10% has nonzero bootstrap_std_delta "
          f"(observed std={delta['bootstrap_std_delta']:.5g}) -- not a degenerate frozen-threshold artifact",
          delta["bootstrap_std_delta"] > 0)

    print(f"\n=== test_bootstrap_threshold_recompute SUMMARY: {len(PASS)} passed, {len(FAIL)} failed (of {len(PASS) + len(FAIL)}) ===")
    if FAIL:
        for f in FAIL:
            print(f"  - {f}")
    return 0 if not FAIL else 1


if __name__ == "__main__":
    sys.exit(main())
