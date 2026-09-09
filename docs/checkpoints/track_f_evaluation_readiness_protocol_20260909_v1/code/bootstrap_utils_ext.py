#!/usr/bin/env python3
"""Additive extension of bootstrap_utils.py (copied forward, unmodified, in
this same directory). Does NOT edit bootstrap_utils.py -- imports from it.

Adds exactly one new statistical function this project did not already
have: an exact (Clopper-Pearson) binomial confidence interval for a RATE
(k successes of n trials), complementing bootstrap_utils.exact_poisson_
count_interval()'s interval for a raw COUNT. PREREGISTRATION.md section 6
requires both: the Poisson interval answers "how uncertain is this raw
survivor count", the binomial interval answers "how uncertain is the
achieved efficiency/rejection rate itself" -- different question, standard
different formula, not a duplicate.
"""
import math

from bootstrap_utils import roc_auc, paired_bootstrap_delta, mcnemar_test, exact_poisson_count_interval  # noqa: F401


def exact_binomial_interval(k, n, ci=0.95):
    """Clopper-Pearson exact interval for a binomial rate k/n (e.g. an
    achieved epsB, or any pass/total efficiency). Standard construction via
    Beta-distribution quantiles: lower = Beta(k, n-k+1).ppf(alpha/2),
    upper = Beta(k+1, n-k).ppf(1-alpha/2), with the usual k=0 / k=n
    boundary conventions (lower=0 / upper=1 respectively). Falls back to a
    flagged normal (Wald) approximation only if scipy is unavailable --
    the exact method is always preferred and used whenever scipy imports
    successfully, matching exact_poisson_count_interval()'s own pattern.
    """
    if n <= 0:
        return {"k": int(k), "n": int(n), "ci_level": ci, "rate_ci_low": None, "rate_ci_high": None,
                "method": "undefined_zero_trials"}
    k = int(k)
    n = int(n)
    alpha = 1 - ci
    rate = k / n
    try:
        from scipy import stats
        lo = 0.0 if k == 0 else stats.beta.ppf(alpha / 2, k, n - k + 1)
        hi = 1.0 if k == n else stats.beta.ppf(1 - alpha / 2, k + 1, n - k)
        method = "exact_clopper_pearson_beta"
    except Exception as e:  # noqa: BLE001 -- degrade, do not crash the whole evaluation over an environment issue
        z = 1.959963984540054 if abs(ci - 0.95) < 1e-9 else 1.6448536269514722  # 95%/90% normal quantiles only
        se = math.sqrt(max(rate * (1 - rate), 0.0) / n)
        lo = max(0.0, rate - z * se)
        hi = min(1.0, rate + z * se)
        method = f"FALLBACK_normal_wald_approx_scipy_unavailable ({e!r})"
    return {"k": k, "n": n, "ci_level": ci, "rate": float(rate),
            "rate_ci_low": float(lo), "rate_ci_high": float(hi), "method": method}
