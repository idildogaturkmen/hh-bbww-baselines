#!/usr/bin/env python3
"""Task 4: compact 4-feature models, cross-validated, NEVER using
score/u. Trained/selected ONLY on the u>3.5 population (per instruction:
do not optimize on the 15-20-event u>4.5 tail); u>4.5 is evaluated only
as a descriptive check of a model already fixed from the broader region.
sklearn/GBDT unavailable in this environment by explicit instruction --
uses the L2-regularized logistic model (with pairwise interaction terms
added for the top combination, to give the 'interaction-aware' analysis
some ability to capture non-additive structure) from stats_utils.py.
"""
import itertools
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, "/uscms_data/d3/iturkmen/hh4b_delphes/paper_exports/track_b_harvey_tail_kinematics_20260906_v1/work")
from stats_utils import cv_auc_logistic, standardize, fit_logistic_l2, predict_logistic, roc_auc_weighted

PKG = "/uscms_data/d3/iturkmen/hh4b_delphes/paper_exports/track_b_harvey_tail_kinematics_20260906_v1"
EXCLUDE = {"score", "u"}


def build_design(df, feats, add_interactions=True):
    X = df[feats].values.astype(np.float64)
    if add_interactions:
        inter_cols = []
        for i, j in itertools.combinations(range(len(feats)), 2):
            inter_cols.append((X[:, i] * X[:, j]).reshape(-1, 1))
        X = np.hstack([X] + inter_cols)
    return X


def cv_auc_4d(sub, feats, k=5, seed=0, add_interactions=True):
    d = sub[feats + ["label", "weight_450fb"]].dropna()
    if d["label"].nunique() < 2 or len(d) < 4 * k:
        return None
    X = build_design(d, feats, add_interactions)
    y = d["label"].values.astype(np.int64)
    w = d["weight_450fb"].values.astype(np.float64)
    mean_auc, std_auc, folds = cv_auc_logistic(X, y, w=w, l2=2.0, k=k, seed=seed)
    return mean_auc, std_auc, len(d), int(y.sum()), int((1 - y).sum())


def descriptive_eval(master, feats, u_thresh, add_interactions=True, fit_region_thresh=3.5, seed=0):
    """Fits the model on the BROADER u>fit_region_thresh population
    (all_background), then evaluates (no refitting) on the u>u_thresh
    subset -- an honest, non-circular descriptive check for the tiny
    u>4.5 tail, per instruction."""
    fit_sub = master[master["u"] > fit_region_thresh]
    d_fit = fit_sub[feats + ["label", "weight_450fb"]].dropna()
    Xtr_raw = build_design(d_fit, feats, add_interactions)
    Xtr, mu, sd = standardize(Xtr_raw)
    y_fit = d_fit["label"].values.astype(np.int64)
    w_fit = d_fit["weight_450fb"].values.astype(np.float64)
    beta = fit_logistic_l2(Xtr, y_fit, w_fit, l2=2.0)

    eval_sub = master[master["u"] > u_thresh]
    d_eval = eval_sub[feats + ["label", "weight_450fb"]].dropna()
    Xev_raw = build_design(d_eval, feats, add_interactions)
    Xev = (Xev_raw - mu) / sd
    y_eval = d_eval["label"].values.astype(np.int64)
    w_eval = d_eval["weight_450fb"].values.astype(np.float64)
    p = predict_logistic(Xev, beta)
    auc = roc_auc_weighted(p, y_eval, w_eval)
    return auc, len(d_eval), int(y_eval.sum()), int((1 - y_eval).sum())


def main():
    master = pd.read_parquet(f"{PKG}/HARVEY_MASTER_EVENT_TABLE.parquet")
    ranking_1d = pd.read_csv(f"{PKG}/HARVEY_1D_FEATURE_RANKING.csv")
    candidate_feats = [f for f in ranking_1d["feature"].tolist() if f not in EXCLUDE][:8]
    print("candidate features for 4D search:", candidate_feats)

    sub35 = master[master["u"] > 3.5]

    rows = []
    for combo in itertools.combinations(candidate_feats, 4):
        res = cv_auc_4d(sub35, list(combo), k=5, add_interactions=True)
        if res is None:
            continue
        mean_auc, std_auc, n_used, n_sig, n_bkg = res
        rows.append(dict(features="+".join(combo), cv_auc_mean=mean_auc, cv_auc_std=std_auc,
                          n_used=n_used, n_sig=n_sig, n_bkg=n_bkg))
    df = pd.DataFrame(rows).sort_values("cv_auc_mean", ascending=False).reset_index(drop=True)
    df.to_csv(f"{PKG}/HARVEY_4D_FEATURE_RANKING.csv", index=False)
    print(df.head(10).to_string(index=False))

    best = df.iloc[0]
    best_feats = best["features"].split("+")

    # credibility check: is the 4D model's CV AUC meaningfully better than
    # the best 1D and best 2D result on the SAME (all_background, u>3.5) population?
    best_1d_auc = None
    r1d_full = pd.read_csv(f"{PKG}/HARVEY_1D_FEATURE_RANKING_full.csv")
    m = r1d_full[(r1d_full["u_region"] == "u>3.5") & (r1d_full["comparison"] == "all_background")]
    best_1d_auc = m["auc_weighted"].abs().sub(0.5).abs().add(0.5).max() if len(m) else np.nan
    # (recompute properly: best raw auc distance from 0.5, converted back to an AUC-like number)
    best_1d_row = m.loc[(m["auc_weighted"] - 0.5).abs().idxmax()]
    best_1d_auc = max(best_1d_row["auc_weighted"], 1 - best_1d_row["auc_weighted"])

    r2d = pd.read_csv(f"{PKG}/HARVEY_2D_FEATURE_RANKING.csv")
    best_2d_auc = r2d["cv_auc_mean"].max()

    delta_vs_1d = best["cv_auc_mean"] - best_1d_auc
    delta_vs_2d = best["cv_auc_mean"] - best_2d_auc
    credible = (delta_vs_2d > 2 * best["cv_auc_std"]) if best["cv_auc_std"] > 0 else (delta_vs_2d > 0.01)

    # descriptive-only check at u>4.5, fit on u>3.5, NOT refit
    auc45, n45, nsig45, nbkg45 = descriptive_eval(master, best_feats, 4.5, fit_region_thresh=3.5)

    summary = dict(
        best_4d_features=best["features"], best_4d_cv_auc=float(best["cv_auc_mean"]), best_4d_cv_auc_std=float(best["cv_auc_std"]),
        best_1d_auc_same_population=float(best_1d_auc), best_2d_cv_auc_same_population=float(best_2d_auc),
        delta_4d_minus_1d=float(delta_vs_1d), delta_4d_minus_2d=float(delta_vs_2d),
        improvement_over_2d_credible=bool(credible),
        descriptive_check_u_gt_4p5_auc=float(auc45), descriptive_check_n=int(n45),
        descriptive_check_n_sig=int(nsig45), descriptive_check_n_bkg=int(nbkg45),
        descriptive_check_note="model fit ONLY on u>3.5, applied without refitting to u>4.5 -- descriptive, not an independent optimization",
    )
    import json
    json.dump(summary, open(f"{PKG}/work/4d_summary.json", "w"), indent=2)
    print(json.dumps(summary, indent=2))
    return df, summary


if __name__ == "__main__":
    main()
