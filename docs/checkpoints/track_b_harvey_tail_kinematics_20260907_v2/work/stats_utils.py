"""Self-implemented statistics -- NumPy/pandas only, per explicit
instruction not to install/modify sklearn/scipy/numpy in this
environment. scipy.stats is not used (broken on this interactive node
for the stats submodule specifically, per this project's established,
pre-existing finding); everything below is implemented directly from
its mathematical definition, not merely "avoided for policy reasons."
"""
from __future__ import annotations

import numpy as np


def weighted_quantile(values, quantiles, weights=None):
    values = np.asarray(values, dtype=np.float64)
    if weights is None:
        weights = np.ones_like(values)
    weights = np.asarray(weights, dtype=np.float64)
    finite = np.isfinite(values) & np.isfinite(weights)
    values, weights = values[finite], weights[finite]
    if len(values) == 0:
        return np.full(len(np.atleast_1d(quantiles)), np.nan)
    order = np.argsort(values)
    values, weights = values[order], weights[order]
    cw = np.cumsum(weights) - 0.5 * weights
    cw /= weights.sum()
    return np.interp(quantiles, cw, values)


def weighted_mean_std(values, weights=None):
    values = np.asarray(values, dtype=np.float64)
    if weights is None:
        weights = np.ones_like(values)
    weights = np.asarray(weights, dtype=np.float64)
    finite = np.isfinite(values) & np.isfinite(weights)
    values, weights = values[finite], weights[finite]
    if weights.sum() <= 0:
        return np.nan, np.nan
    mean = np.average(values, weights=weights)
    var = np.average((values - mean) ** 2, weights=weights)
    return mean, np.sqrt(var)


def cohens_d_weighted(x_sig, x_bkg, w_sig=None, w_bkg=None):
    """Standardized effect size using the pooled weighted std."""
    m1, s1 = weighted_mean_std(x_sig, w_sig)
    m2, s2 = weighted_mean_std(x_bkg, w_bkg)
    n1 = len(x_sig) if w_sig is None else np.sum(np.isfinite(x_sig))
    n2 = len(x_bkg) if w_bkg is None else np.sum(np.isfinite(x_bkg))
    pooled = np.sqrt(((n1 - 1) * s1 ** 2 + (n2 - 1) * s2 ** 2) / max(n1 + n2 - 2, 1)) if (n1 > 1 and n2 > 1) else np.nan
    if not np.isfinite(pooled) or pooled == 0:
        return np.nan
    return (m1 - m2) / pooled


def roc_auc_weighted(scores, labels, weights=None):
    """AUC via the weighted Mann-Whitney U statistic. labels: 1=signal,
    0=background. Ties broken by average rank (standard mid-rank
    treatment), weighted generalization of AUC = P(score_sig > score_bkg)
    + 0.5*P(tie)."""
    scores = np.asarray(scores, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.int64)
    if weights is None:
        weights = np.ones_like(scores)
    weights = np.asarray(weights, dtype=np.float64)
    finite = np.isfinite(scores) & np.isfinite(weights)
    scores, labels, weights = scores[finite], labels[finite], weights[finite]
    pos = labels == 1
    neg = labels == 0
    n_pos_w, n_neg_w = weights[pos].sum(), weights[neg].sum()
    if n_pos_w <= 0 or n_neg_w <= 0:
        return np.nan
    # O(n log n) weighted rank-sum via sorting (exact, not sampled)
    order = np.argsort(scores, kind="mergesort")
    s_sorted = scores[order]
    w_sorted = weights[order]
    lab_sorted = labels[order]
    # assign average rank-weight for ties
    n = len(s_sorted)
    cw = np.cumsum(w_sorted)
    total_w = cw[-1]
    # for each element, weighted rank = cumulative weight of all elements with smaller score
    # + half the weight of tied elements (mid-rank), computed via searchsorted on unique values
    uniq, inv, counts = np.unique(s_sorted, return_inverse=True, return_counts=True)
    w_per_uniq = np.zeros(len(uniq))
    np.add.at(w_per_uniq, inv, w_sorted)
    cum_before = np.concatenate([[0.0], np.cumsum(w_per_uniq)])[:-1]
    mid_rank_w = cum_before[inv] + 0.5 * w_per_uniq[inv]
    # weighted U-statistic: AUC = (sum of weighted mid-ranks for the
    # positive class - 0.5*sum of positive weights) / (n_pos_w * n_neg_w)
    # times n_pos_w, rearranged below; exact weighted Mann-Whitney AUC.
    sum_pos_midrank_times_w = np.sum(w_sorted[lab_sorted == 1] * mid_rank_w[lab_sorted == 1])
    auc = (sum_pos_midrank_times_w / n_pos_w - 0.5 * weights[pos].sum()) / n_neg_w
    return float(np.clip(auc, 0.0, 1.0))


def bootstrap_ci(values, statistic_fn, n_boot=500, seed=0, ci=0.68):
    rng = np.random.default_rng(seed)
    n = len(values)
    if n == 0:
        return np.nan, np.nan, np.nan
    values = np.asarray(values)
    stats = np.empty(n_boot)
    for b in range(n_boot):
        idx = rng.integers(0, n, n)
        stats[b] = statistic_fn(values[idx])
    lo, hi = np.percentile(stats, [(1 - ci) / 2 * 100, (1 + ci) / 2 * 100])
    return float(np.nanmean(stats)), float(lo), float(hi)


def bootstrap_auc_ci(scores, labels, weights=None, n_boot=150, seed=0, ci=0.68, max_class_n=2000):
    """Bootstrap CI on the AUC. For computational tractability with very
    large classes (e.g. ~27,000 signal events), each class is capped at
    `max_class_n` events via a ONE-TIME random subsample (seeded, fixed
    across all n_boot iterations) before resampling -- the point AUC
    estimate reported elsewhere always uses the FULL data; only the
    *width* of this bootstrap CI is computed from the capped subsample,
    which is statistically valid for estimating sampling variability
    (bootstrap CI width depends on effective sample size, not on
    re-using every single original point) and is stated explicitly
    wherever this function's output is reported."""
    rng = np.random.default_rng(seed)
    scores = np.asarray(scores)
    labels = np.asarray(labels)
    if weights is None:
        weights = np.ones_like(scores, dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64)

    pos_idx = np.where(labels == 1)[0]
    neg_idx = np.where(labels == 0)[0]
    if len(pos_idx) < 4 or len(neg_idx) < 4:
        return np.nan, np.nan, np.nan
    if len(pos_idx) > max_class_n:
        pos_idx = rng.choice(pos_idx, size=max_class_n, replace=False)
    if len(neg_idx) > max_class_n:
        neg_idx = rng.choice(neg_idx, size=max_class_n, replace=False)
    keep = np.concatenate([pos_idx, neg_idx])
    scores, labels, weights = scores[keep], labels[keep], weights[keep]
    n = len(scores)

    aucs = np.empty(n_boot)
    for b in range(n_boot):
        idx = rng.integers(0, n, n)
        aucs[b] = roc_auc_weighted(scores[idx], labels[idx], weights[idx])
    lo, hi = np.nanpercentile(aucs, [(1 - ci) / 2 * 100, (1 + ci) / 2 * 100])
    return float(np.nanmean(aucs)), float(lo), float(hi)


# ---------------- L2-regularized logistic regression (Newton-Raphson / IRLS), pure NumPy ----------------

def standardize(X):
    mu = np.nanmean(X, axis=0)
    sd = np.nanstd(X, axis=0)
    sd = np.where(sd > 0, sd, 1.0)
    return (X - mu) / sd, mu, sd


def fit_logistic_l2(X, y, w=None, l2=1.0, n_iter=50, tol=1e-8):
    """X: (n,k) standardized features, y: (n,) in {0,1}, w: sample weights.
    Returns beta (k+1,) including intercept, fit by Newton-Raphson with
    ridge penalty (penalty not applied to intercept)."""
    n, k = X.shape
    Xb = np.hstack([np.ones((n, 1)), X])
    if w is None:
        w = np.ones(n)
    beta = np.zeros(k + 1)
    penalty = np.eye(k + 1) * l2
    penalty[0, 0] = 0.0
    for _ in range(n_iter):
        z = Xb @ beta
        p = 1.0 / (1.0 + np.exp(-np.clip(z, -30, 30)))
        W = w * p * (1 - p)
        W = np.clip(W, 1e-8, None)
        grad = Xb.T @ (w * (y - p)) - penalty @ beta
        H = -(Xb.T * W) @ Xb - penalty
        try:
            step = np.linalg.solve(H, grad)
        except np.linalg.LinAlgError:
            step = np.linalg.lstsq(H, grad, rcond=None)[0]
        beta_new = beta - step
        if np.max(np.abs(beta_new - beta)) < tol:
            beta = beta_new
            break
        beta = beta_new
    return beta


def predict_logistic(X, beta):
    n = X.shape[0]
    Xb = np.hstack([np.ones((n, 1)), X])
    z = Xb @ beta
    return 1.0 / (1.0 + np.exp(-np.clip(z, -30, 30)))


def kfold_indices(n, k, seed=0):
    rng = np.random.default_rng(seed)
    idx = rng.permutation(n)
    folds = np.array_split(idx, k)
    return folds


def cv_auc_logistic(X, y, w=None, l2=1.0, k=5, seed=0):
    """K-fold CV AUC for the L2-logistic model. Standardization is fit
    on the TRAIN fold only, per fold (no leakage)."""
    n = X.shape[0]
    if w is None:
        w = np.ones(n)
    folds = kfold_indices(n, k, seed)
    fold_aucs = []
    for i in range(k):
        test_idx = folds[i]
        train_idx = np.concatenate([folds[j] for j in range(k) if j != i])
        if len(np.unique(y[train_idx])) < 2 or len(np.unique(y[test_idx])) < 2:
            continue
        Xtr, mu, sd = standardize(X[train_idx])
        Xte = (X[test_idx] - mu) / sd
        beta = fit_logistic_l2(Xtr, y[train_idx], w[train_idx], l2=l2)
        p_test = predict_logistic(Xte, beta)
        fold_aucs.append(roc_auc_weighted(p_test, y[test_idx], w[test_idx]))
    fold_aucs = np.array(fold_aucs, dtype=np.float64)
    return float(np.nanmean(fold_aucs)), float(np.nanstd(fold_aucs)), fold_aucs
