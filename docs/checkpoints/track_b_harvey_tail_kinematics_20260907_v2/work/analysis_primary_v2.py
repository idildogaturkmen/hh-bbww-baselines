#!/usr/bin/env python3
"""v2 CORRECTION 1: primary all-background 1D/2D/4D analysis, met_pt
EXCLUDED (QCD has met_pt=NaN for all 68 rows; including it silently
restricts every downstream model to the 42-event non-QCD-only
population). Every primary feature below is verified to have ZERO NaN
across all 110 u>3.5 background rows (68 QCD + 27 ttbar + 15 other),
so 1D/2D/4D all use the EXACT SAME 27,449-row (110-background) table.

Also implements a nested-CV selection-bias check for the chosen best
2D pair and 4D set: the naive 'best of a K-candidate search' CV AUC is
optimistic because the search itself used the same finite background
sample the CV is evaluated on. Nested CV re-does the search
independently within each outer fold's training data only.
"""
import itertools
import sys
import json

import numpy as np
import pandas as pd

sys.path.insert(0, "/uscms_data/d3/iturkmen/hh4b_delphes/paper_exports/track_b_harvey_tail_kinematics_20260907_v2/work")
from stats_utils import (weighted_quantile, weighted_mean_std, cohens_d_weighted, roc_auc_weighted,
                          bootstrap_auc_ci, cv_auc_logistic, standardize, fit_logistic_l2, predict_logistic,
                          kfold_indices)

PKG = "/uscms_data/d3/iturkmen/hh4b_delphes/paper_exports/track_b_harvey_tail_kinematics_20260907_v2"

# met_pt explicitly excluded -- see correction note above and in the executive summary
PRIMARY_FEATURES = ["n_selected_jets", "HT", "mHH", "pT_HH", "eta_HH", "RHH", "n_btag_loose",
                     "mH1", "mH2", "pT_H1", "pT_H2", "DeltaR_bb_H1", "DeltaR_bb_H2",
                     "DeltaR_HH", "DeltaEta_HH", "DeltaPhi_HH", "mass_asym", "pt_asym",
                     "jet1_pt", "jet2_pt", "jet3_pt", "jet4_pt", "pt5", "eta5", "probB5", "probC5", "probL5",
                     "min_dR_j5_leading4", "pt5_over_pt4", "pt5_over_HT"]

COMPARISONS = [("all_background", None), ("QCD", "QCD"), ("ttbar", "ttbar")]


def summarize_feature(df_sig, df_bkg, feature, min_n_for_ci=20):
    x_sig, w_sig = df_sig[feature].values, df_sig["weight_450fb"].values
    x_bkg, w_bkg = df_bkg[feature].values, df_bkg["weight_450fb"].values
    n_sig_raw, n_bkg_raw = np.isfinite(x_sig).sum(), np.isfinite(x_bkg).sum()
    if n_sig_raw < 2 or n_bkg_raw < 2:
        return None
    q_sig = weighted_quantile(x_sig, [0.25, 0.5, 0.75], w_sig)
    q_bkg = weighted_quantile(x_bkg, [0.25, 0.5, 0.75], w_bkg)
    m_sig, s_sig = weighted_mean_std(x_sig, w_sig)
    m_bkg, s_bkg = weighted_mean_std(x_bkg, w_bkg)
    d = cohens_d_weighted(x_sig, x_bkg, w_sig, w_bkg)
    scores = np.concatenate([x_sig, x_bkg])
    labels = np.concatenate([np.ones(len(x_sig)), np.zeros(len(x_bkg))])
    weights_ = np.concatenate([w_sig, w_bkg])
    auc_w = roc_auc_weighted(scores, labels, weights_)
    auc_raw = roc_auc_weighted(scores, labels, None)
    do_boot = (n_bkg_raw >= min_n_for_ci) and (n_sig_raw >= min_n_for_ci)
    auc_mean, auc_lo, auc_hi = bootstrap_auc_ci(scores, labels, weights_, n_boot=150) if do_boot else (auc_w, np.nan, np.nan)
    return dict(feature=feature, n_sig_raw=int(n_sig_raw), n_bkg_raw=int(n_bkg_raw),
                median_sig=q_sig[1], iqr_sig=q_sig[2] - q_sig[0], median_bkg=q_bkg[1], iqr_bkg=q_bkg[2] - q_bkg[0],
                mean_sig_w=m_sig, mean_bkg_w=m_bkg, std_sig_w=s_sig, std_bkg_w=s_bkg,
                cohens_d=d, auc_weighted=auc_w, auc_raw_count=auc_raw,
                auc_boot_mean=auc_mean, auc_boot_lo68=auc_lo, auc_boot_hi68=auc_hi,
                ci_computed=do_boot, separation_score=abs(auc_w - 0.5))


def run_1d(master, u_thresh):
    sub = master[master["u"] > u_thresh]
    sig = sub[sub["label"] == 1]
    rows = []
    for comp_name, class_filter in COMPARISONS:
        bkg = sub[sub["label"] == 0] if class_filter is None else sub[sub["background_class"] == class_filter]
        for feat in PRIMARY_FEATURES:
            r = summarize_feature(sig, bkg, feat, min_n_for_ci=20)
            if r is not None:
                r["comparison"] = comp_name
                r["u_region"] = f"u>{u_thresh}"
                rows.append(r)
    return pd.DataFrame(rows)


def build_design(df, feats, add_interactions=True):
    X = df[feats].values.astype(np.float64)
    if add_interactions:
        for i, j in itertools.combinations(range(len(feats)), 2):
            X = np.hstack([X, (X[:, i] * X[:, j]).reshape(-1, 1)])
    return X


def search_2d(sub, feats, k=5, seed=0):
    rows = []
    d_all = sub[feats + ["label", "weight_450fb"]].dropna()
    for f1, f2 in itertools.combinations(feats, 2):
        d = d_all[[f1, f2, "label", "weight_450fb"]]
        X = d[[f1, f2]].values.astype(np.float64)
        y = d["label"].values.astype(np.int64)
        w = d["weight_450fb"].values.astype(np.float64)
        mean_auc, std_auc, _ = cv_auc_logistic(X, y, w=w, l2=1.0, k=k, seed=seed)
        rows.append(dict(feature_1=f1, feature_2=f2, cv_auc_mean=mean_auc, cv_auc_std=std_auc,
                          n_used=len(d), n_sig=int(y.sum()), n_bkg=int((1 - y).sum())))
    return pd.DataFrame(rows).sort_values("cv_auc_mean", ascending=False).reset_index(drop=True)


def search_4d(sub, feats, k=5, seed=0):
    rows = []
    d_all = sub[feats + ["label", "weight_450fb"]].dropna()
    for combo in itertools.combinations(feats, 4):
        cols = list(combo)
        d = d_all[cols + ["label", "weight_450fb"]]
        X = build_design(d, cols, add_interactions=True)
        y = d["label"].values.astype(np.int64)
        w = d["weight_450fb"].values.astype(np.float64)
        mean_auc, std_auc, _ = cv_auc_logistic(X, y, w=w, l2=2.0, k=k, seed=seed)
        rows.append(dict(features="+".join(cols), cv_auc_mean=mean_auc, cv_auc_std=std_auc,
                          n_used=len(d), n_sig=int(y.sum()), n_bkg=int((1 - y).sum())))
    return pd.DataFrame(rows).sort_values("cv_auc_mean", ascending=False).reset_index(drop=True)


def nested_cv_2d(sub, feats, outer_k=5, inner_k=3, seed=0):
    """Selection-bias-corrected AUC for the 2D pair search: within each
    outer fold, the best pair is re-selected using ONLY that fold's
    training data (via inner CV), then evaluated on the untouched outer
    test fold. Returns (mean outer-test AUC, std, list of per-fold chosen pairs)."""
    d_all = sub[feats + ["label", "weight_450fb"]].dropna().reset_index(drop=True)
    y_all = d_all["label"].values.astype(np.int64)
    n = len(d_all)
    outer_folds = kfold_indices(n, outer_k, seed=seed)
    outer_aucs = []
    chosen_pairs = []
    for i in range(outer_k):
        test_idx = outer_folds[i]
        train_idx = np.concatenate([outer_folds[j] for j in range(outer_k) if j != i])
        if len(np.unique(y_all[train_idx])) < 2 or len(np.unique(y_all[test_idx])) < 2:
            continue
        d_train = d_all.iloc[train_idx]
        best_pair, best_inner_auc = None, -1
        for f1, f2 in itertools.combinations(feats, 2):
            X = d_train[[f1, f2]].values.astype(np.float64)
            y = d_train["label"].values.astype(np.int64)
            w = d_train["weight_450fb"].values.astype(np.float64)
            mean_auc, _, _ = cv_auc_logistic(X, y, w=w, l2=1.0, k=inner_k, seed=seed)
            if np.isfinite(mean_auc) and mean_auc > best_inner_auc:
                best_inner_auc, best_pair = mean_auc, (f1, f2)
        chosen_pairs.append(best_pair)
        f1, f2 = best_pair
        Xtr_raw = d_train[[f1, f2]].values.astype(np.float64)
        Xtr, mu, sd = standardize(Xtr_raw)
        ytr = d_train["label"].values.astype(np.int64)
        wtr = d_train["weight_450fb"].values.astype(np.float64)
        beta = fit_logistic_l2(Xtr, ytr, wtr, l2=1.0)
        d_test = d_all.iloc[test_idx]
        Xte = (d_test[[f1, f2]].values.astype(np.float64) - mu) / sd
        p_test = predict_logistic(Xte, beta)
        auc_test = roc_auc_weighted(p_test, d_test["label"].values, d_test["weight_450fb"].values)
        outer_aucs.append(auc_test)
    return float(np.nanmean(outer_aucs)), float(np.nanstd(outer_aucs)), chosen_pairs


def nested_cv_4d(sub, feats, outer_k=5, inner_k=3, seed=0, top_n_candidates=None):
    d_all = sub[feats + ["label", "weight_450fb"]].dropna().reset_index(drop=True)
    y_all = d_all["label"].values.astype(np.int64)
    n = len(d_all)
    outer_folds = kfold_indices(n, outer_k, seed=seed)
    outer_aucs = []
    chosen_sets = []
    combos = list(itertools.combinations(feats, 4))
    for i in range(outer_k):
        test_idx = outer_folds[i]
        train_idx = np.concatenate([outer_folds[j] for j in range(outer_k) if j != i])
        if len(np.unique(y_all[train_idx])) < 2 or len(np.unique(y_all[test_idx])) < 2:
            continue
        d_train = d_all.iloc[train_idx]
        best_combo, best_inner_auc = None, -1
        for combo in combos:
            cols = list(combo)
            X = build_design(d_train, cols, add_interactions=True)
            y = d_train["label"].values.astype(np.int64)
            w = d_train["weight_450fb"].values.astype(np.float64)
            mean_auc, _, _ = cv_auc_logistic(X, y, w=w, l2=2.0, k=inner_k, seed=seed)
            if np.isfinite(mean_auc) and mean_auc > best_inner_auc:
                best_inner_auc, best_combo = mean_auc, cols
        chosen_sets.append(best_combo)
        Xtr_raw = build_design(d_train, best_combo, add_interactions=True)
        Xtr, mu, sd = standardize(Xtr_raw)
        ytr = d_train["label"].values.astype(np.int64)
        wtr = d_train["weight_450fb"].values.astype(np.float64)
        beta = fit_logistic_l2(Xtr, ytr, wtr, l2=2.0)
        d_test = d_all.iloc[test_idx]
        Xte_raw = build_design(d_test, best_combo, add_interactions=True)
        Xte = (Xte_raw - mu) / sd
        p_test = predict_logistic(Xte, beta)
        auc_test = roc_auc_weighted(p_test, d_test["label"].values, d_test["weight_450fb"].values)
        outer_aucs.append(auc_test)
    return float(np.nanmean(outer_aucs)), float(np.nanstd(outer_aucs)), chosen_sets


def main():
    master = pd.read_parquet(f"{PKG}/HARVEY_MASTER_EVENT_TABLE.parquet")

    res35 = run_1d(master, 3.5)
    res45 = run_1d(master, 4.5)
    res45["descriptive_only_tiny_N"] = True
    res35["descriptive_only_tiny_N"] = False
    full = pd.concat([res35, res45], ignore_index=True)

    rank_basis = res35[res35["comparison"].isin(["all_background", "QCD", "ttbar"])]
    ranking = (rank_basis.groupby("feature")["separation_score"].agg(["mean", "std", "count"])
               .reset_index().sort_values("mean", ascending=False))
    ranking.columns = ["feature", "mean_abs_auc_minus_half_u3p5", "std_across_comparisons", "n_comparisons"]

    full.to_csv(f"{PKG}/HARVEY_1D_FEATURE_RANKING_full.csv", index=False)
    ranking.to_csv(f"{PKG}/HARVEY_1D_FEATURE_RANKING.csv", index=False)
    print("=== PRIMARY 1D ranking (met_pt excluded, n_bkg=110 for all rows) ===")
    print(ranking.head(10).to_string(index=False))

    sub35 = master[master["u"] > 3.5]
    n_bkg_check = (sub35["label"] == 0).sum()
    assert n_bkg_check == 110, n_bkg_check

    top8 = ranking["feature"].tolist()[:8]
    df_2d = search_2d(sub35, top8, k=5)
    df_2d.to_csv(f"{PKG}/HARVEY_2D_FEATURE_RANKING_full.csv", index=False)
    df_2d.to_csv(f"{PKG}/HARVEY_2D_FEATURE_RANKING.csv", index=False)
    print("\n=== PRIMARY 2D ranking (n_bkg=110) ===")
    print(df_2d.head(6).to_string(index=False))

    top6 = ranking["feature"].tolist()[:6]
    df_4d = search_4d(sub35, top6, k=5)
    df_4d.to_csv(f"{PKG}/HARVEY_4D_FEATURE_RANKING.csv", index=False)
    print("\n=== PRIMARY 4D ranking (n_bkg=110) ===")
    print(df_4d.head(6).to_string(index=False))

    best_2d = df_2d.iloc[0]
    best_4d = df_4d.iloc[0]

    print("\n=== nested CV selection-bias check, 2D ===")
    nested_2d_auc, nested_2d_std, chosen_2d = nested_cv_2d(sub35, top8, outer_k=5, inner_k=3)
    print(f"nested-CV outer-test AUC = {nested_2d_auc:.4f} +/- {nested_2d_std:.4f}; "
          f"naive best-of-search CV AUC = {best_2d['cv_auc_mean']:.4f}; chosen pairs per outer fold: {chosen_2d}")

    print("\n=== nested CV selection-bias check, 4D ===")
    nested_4d_auc, nested_4d_std, chosen_4d = nested_cv_4d(sub35, top6, outer_k=5, inner_k=3)
    print(f"nested-CV outer-test AUC = {nested_4d_auc:.4f} +/- {nested_4d_std:.4f}; "
          f"naive best-of-search CV AUC = {best_4d['cv_auc_mean']:.4f}; chosen sets per outer fold: {chosen_4d}")

    summary = dict(
        n_bkg_primary=110, n_sig_primary=27339,
        best_1d=ranking.iloc[0]["feature"], best_1d_score=float(ranking.iloc[0]["mean_abs_auc_minus_half_u3p5"]),
        best_2d=[best_2d["feature_1"], best_2d["feature_2"]], best_2d_cv_auc=float(best_2d["cv_auc_mean"]), best_2d_cv_auc_std=float(best_2d["cv_auc_std"]),
        best_4d=best_4d["features"].split("+"), best_4d_cv_auc=float(best_4d["cv_auc_mean"]), best_4d_cv_auc_std=float(best_4d["cv_auc_std"]),
        nested_cv_2d_auc=nested_2d_auc, nested_cv_2d_std=nested_2d_std, nested_cv_2d_chosen_pairs_per_fold=[list(p) for p in chosen_2d],
        nested_cv_4d_auc=nested_4d_auc, nested_cv_4d_std=nested_4d_std, nested_cv_4d_chosen_sets_per_fold=[list(p) for p in chosen_4d],
        selection_bias_note="naive best-of-search CV AUC is computed by picking the highest-CV-AUC combination among many candidates evaluated on the SAME 110-background sample the CV itself uses -- this is expected to be optimistic. Nested CV re-runs the full search independently within each outer fold's training data only, giving an honest (typically lower) estimate of out-of-sample performance. Both numbers are reported; the nested number is the more defensible one.",
    )
    json.dump(summary, open(f"{PKG}/work/primary_v2_summary.json", "w"), indent=2)
    print("\n", json.dumps(summary, indent=2, default=str))
    return summary


if __name__ == "__main__":
    main()
