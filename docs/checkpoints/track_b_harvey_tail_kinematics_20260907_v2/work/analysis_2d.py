#!/usr/bin/env python3
"""Task 3: 2D correlations. For the top-ranked 1D features, pairwise
2-feature classifiers (L2-logistic, cross-validated) -- NEVER using
score/u. Ranks feature PAIRS by CV AUC at u>3.5 (all-background)."""
import itertools
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, "/uscms_data/d3/iturkmen/hh4b_delphes/paper_exports/track_b_harvey_tail_kinematics_20260906_v1/work")
from stats_utils import cv_auc_logistic, standardize

PKG = "/uscms_data/d3/iturkmen/hh4b_delphes/paper_exports/track_b_harvey_tail_kinematics_20260906_v1"

EXCLUDE = {"score", "u"}


def top_n_features(ranking_csv, n=8):
    r = pd.read_csv(ranking_csv)
    return r["feature"].tolist()[:n]


def pair_cv_auc(sub, f1, f2, k=5, seed=0):
    cols = [f1, f2]
    d = sub[cols + ["label", "weight_450fb"]].dropna()
    if d["label"].nunique() < 2 or len(d) < 4 * k:
        return None
    X = d[cols].values.astype(np.float64)
    y = d["label"].values.astype(np.int64)
    w = d["weight_450fb"].values.astype(np.float64)
    mean_auc, std_auc, _ = cv_auc_logistic(X, y, w=w, l2=1.0, k=k, seed=seed)
    return mean_auc, std_auc, len(d), int(y.sum()), int((1 - y).sum())


def main():
    master = pd.read_parquet(f"{PKG}/HARVEY_MASTER_EVENT_TABLE.parquet")
    top_feats = top_n_features(f"{PKG}/HARVEY_1D_FEATURE_RANKING.csv", n=8)
    top_feats = [f for f in top_feats if f not in EXCLUDE]
    print("top features for 2D pairing:", top_feats)

    rows = []
    for u_thresh in [3.5, 4.5]:
        sub_all = master[master["u"] > u_thresh]
        for comp_name, class_filter in [("all_background", None), ("QCD", "QCD"), ("ttbar", "ttbar")]:
            sig = sub_all[sub_all["label"] == 1]
            bkg = sub_all[sub_all["background_class"] == class_filter] if class_filter else sub_all[sub_all["label"] == 0]
            sub = pd.concat([sig, bkg])
            for f1, f2 in itertools.combinations(top_feats, 2):
                res = pair_cv_auc(sub, f1, f2, k=5 if u_thresh == 3.5 else 3)
                if res is None:
                    continue
                mean_auc, std_auc, n_used, n_sig, n_bkg = res
                rows.append(dict(u_region=f"u>{u_thresh}", comparison=comp_name,
                                  feature_1=f1, feature_2=f2, cv_auc_mean=mean_auc, cv_auc_std=std_auc,
                                  n_used=n_used, n_sig=n_sig, n_bkg=n_bkg,
                                  descriptive_only_tiny_N=(u_thresh == 4.5)))
    df = pd.DataFrame(rows)
    df.to_csv(f"{PKG}/HARVEY_2D_FEATURE_RANKING_full.csv", index=False)

    ranking = (df[(df["u_region"] == "u>3.5") & (df["comparison"] == "all_background")]
               .sort_values("cv_auc_mean", ascending=False)
               .reset_index(drop=True))
    ranking.to_csv(f"{PKG}/HARVEY_2D_FEATURE_RANKING.csv", index=False)
    print(ranking.head(10).to_string(index=False))
    return df, ranking


if __name__ == "__main__":
    main()
