#!/usr/bin/env python3
"""Statistics helpers shared by evaluate_matched_2m.py: paired bootstrap,
McNemar's test, and exact finite-count intervals for sparse tails.

No new statistical formula invented here beyond what is standard --
per project convention (SPA2M_PART_EVALUATION_PLAN.md: "no new statistical
formula" for the established asimov_z/b95 machinery; the same discipline
is applied here for AUC/rejection bootstrap and McNemar).
"""
import math

import numpy as np
# scipy is imported lazily inside mcnemar_test()/exact_poisson_count_interval()
# only -- roc_auc()/paired_bootstrap_delta() have zero non-numpy dependencies
# so they remain usable even in an environment with a broken/absent scipy.


def roc_auc(scores, labels_binary):
    """Dependency-free AUC (Mann-Whitney U / rank-sum equivalent), so this
    package does not hard-require sklearn in whatever environment actually
    runs the evaluation. Ties are averaged (standard mid-rank treatment).
    labels_binary: 1=signal/positive, 0=background/negative.
    """
    scores = np.asarray(scores, dtype=np.float64)
    labels_binary = np.asarray(labels_binary, dtype=np.int64)
    n_pos = int(labels_binary.sum())
    n_neg = len(labels_binary) - n_pos
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    order = np.argsort(scores, kind="mergesort")
    ranks = np.empty(len(scores), dtype=np.float64)
    sorted_scores = scores[order]
    i = 0
    r = 1
    while i < len(sorted_scores):
        j = i
        while j + 1 < len(sorted_scores) and sorted_scores[j + 1] == sorted_scores[i]:
            j += 1
        avg_rank = (r + (r + (j - i))) / 2.0
        ranks[order[i:j + 1]] = avg_rank
        r += (j - i + 1)
        i = j + 1
    sum_ranks_pos = ranks[labels_binary == 1].sum()
    auc = (sum_ranks_pos - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)
    return float(auc)


def paired_bootstrap_delta(scores_test, scores_control, labels, statistic_fn, n_boot=2000, seed=0, ci=0.95, **kwargs):
    """scores_test, scores_control, labels: arrays over the SAME events,
    same order (e.g. TEST and CONTROL model scores plus shared labels on
    the identical val cohort). statistic_fn(scores, labels, idx, **kwargs)
    -> scalar (e.g. AUC, or rejection at a fixed efficiency), called with
    the SAME resampled index array for both test and control each
    replicate -- this pairing (not two independent bootstraps) is what
    controls for shared per-event variance between the two models.

    Returns dict with observed delta (test - control), bootstrap mean/std,
    and a percentile CI on the delta.
    """
    rng = np.random.default_rng(seed)
    n = len(scores_test)
    assert len(scores_control) == n == len(labels), "paired bootstrap requires equal-length, same-order arrays"

    idx_all = np.arange(n)
    observed = statistic_fn(scores_test, labels, idx_all, **kwargs) - statistic_fn(scores_control, labels, idx_all, **kwargs)

    deltas = np.empty(n_boot)
    for i in range(n_boot):
        idx = rng.integers(0, n, size=n)
        deltas[i] = statistic_fn(scores_test, labels, idx, **kwargs) - statistic_fn(scores_control, labels, idx, **kwargs)

    alpha = 1 - ci
    lo, hi = np.quantile(deltas, [alpha / 2, 1 - alpha / 2])
    return {
        "observed_delta_test_minus_control": float(observed),
        "bootstrap_mean_delta": float(deltas.mean()),
        "bootstrap_std_delta": float(deltas.std()),
        "ci_level": ci,
        "ci_low": float(lo),
        "ci_high": float(hi),
        "n_boot": n_boot,
        "excludes_zero": bool((lo > 0) or (hi < 0)),
    }


def mcnemar_test(control_correct, test_correct):
    """Paired event-level correctness arrays (bool, same events, same
    order). Returns discordant counts, exact binomial McNemar p-value
    (used whenever the discordant total is small, which is the regime a
    reconstruction-accuracy comparison at ~0.49 average jet accuracy is
    likely to be in for any single event-level definition of 'correct'),
    plus the continuity-corrected chi-square statistic/p-value for
    cross-reference. Pure stdlib (math.comb / math.erfc) -- no scipy
    dependency, so this never degrades regardless of environment health.
    """
    control_correct = np.asarray(control_correct, dtype=bool)
    test_correct = np.asarray(test_correct, dtype=bool)
    assert control_correct.shape == test_correct.shape

    n_01 = int(np.sum(~control_correct & test_correct))   # control wrong, test right
    n_10 = int(np.sum(control_correct & ~test_correct))   # control right, test wrong
    n_00 = int(np.sum(~control_correct & ~test_correct))
    n_11 = int(np.sum(control_correct & test_correct))
    n_discordant = n_01 + n_10

    if n_discordant == 0:
        exact_p = 1.0
    else:
        k = min(n_01, n_10)
        # exact two-sided binomial(n_discordant, 0.5) test: P(X<=k) via math.comb, doubled.
        cdf_k = sum(math.comb(n_discordant, i) for i in range(k + 1)) / (2 ** n_discordant)
        exact_p = min(1.0, 2 * cdf_k)

    if n_discordant > 0:
        chi2_stat = (abs(n_01 - n_10) - 1) ** 2 / n_discordant
        # chi-square(df=1) survival function has a closed form via erfc: P(X>x) = erfc(sqrt(x/2))
        chi2_p = math.erfc(math.sqrt(chi2_stat / 2))
    else:
        chi2_stat, chi2_p = 0.0, 1.0

    return {
        "contingency": {"control_wrong_test_right": n_01, "control_right_test_wrong": n_10,
                         "both_wrong": n_00, "both_right": n_11},
        "n_discordant": n_discordant,
        "exact_binomial_p_value": float(exact_p),
        "continuity_corrected_chi2_statistic": float(chi2_stat),
        "continuity_corrected_chi2_p_value": float(chi2_p),
        "test_better_than_control": n_01 > n_10,
    }


def exact_poisson_count_interval(n_pass, ci=0.95):
    """Exact (Garwood) Poisson confidence interval for a raw survivor
    count -- used instead of an asymptotic Wald/normal interval whenever a
    working point has few survivors (the sparse-tail regime this pipeline
    explicitly must not silently under-report). Falls back to a Wilson-type
    normal approximation (flagged explicitly in the output) only if scipy
    is unavailable/broken in the running environment -- the exact method
    is always preferred and used whenever scipy imports successfully."""
    alpha = 1 - ci
    try:
        from scipy import stats
        lo = 0.0 if n_pass == 0 else 0.5 * stats.chi2.ppf(alpha / 2, 2 * n_pass)
        hi = 0.5 * stats.chi2.ppf(1 - alpha / 2, 2 * (n_pass + 1))
        method = "exact_garwood_chi2"
    except Exception as e:  # noqa: BLE001 -- degrade, do not crash the whole evaluation over an environment issue
        z = 1.959963984540054 if abs(ci - 0.95) < 1e-9 else 1.6448536269514722  # 95%/90% normal quantiles only
        lo = max(0.0, n_pass - z * math.sqrt(max(n_pass, 1)))
        hi = n_pass + z * math.sqrt(max(n_pass, 1))
        method = f"FALLBACK_normal_approx_scipy_unavailable ({e!r})"
    return {"n_pass": int(n_pass), "ci_level": ci, "count_ci_low": float(lo), "count_ci_high": float(hi), "method": method}
