#!/usr/bin/env python3
"""Task 2: comprehensive 1D comparison. signal vs {all_background, QCD,
ttbar}, separately for u>3.5 and u>4.5. Weighted + raw-count summaries,
weighted quantiles, standardized effect size, univariate ROC AUC,
bootstrap uncertainty. Ranks features by robust separation (mean of
|AUC-0.5| across the three background comparisons at u>3.5, since the
u>4.5 background sample is explicitly too small to rank on alone)."""
import sys
import numpy as np
import pandas as pd

sys.path.insert(0, "/uscms_data/d3/iturkmen/hh4b_delphes/paper_exports/track_b_harvey_tail_kinematics_20260906_v1/work")
from stats_utils import weighted_quantile, weighted_mean_std, cohens_d_weighted, roc_auc_weighted, bootstrap_auc_ci

WORK = "/uscms_data/d3/iturkmen/hh4b_delphes/paper_exports/track_b_harvey_tail_kinematics_20260906_v1/work"
PKG = "/uscms_data/d3/iturkmen/hh4b_delphes/paper_exports/track_b_harvey_tail_kinematics_20260906_v1"

FEATURES = ["n_selected_jets", "HT", "mHH", "pT_HH", "eta_HH", "RHH", "met_pt", "n_btag_loose",
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
    if do_boot:
        auc_mean, auc_lo, auc_hi = bootstrap_auc_ci(scores, labels, weights_, n_boot=150)
    else:
        auc_mean, auc_lo, auc_hi = auc_w, np.nan, np.nan

    return dict(
        feature=feature, n_sig_raw=int(n_sig_raw), n_bkg_raw=int(n_bkg_raw),
        median_sig=q_sig[1], iqr_sig=q_sig[2] - q_sig[0], median_bkg=q_bkg[1], iqr_bkg=q_bkg[2] - q_bkg[0],
        mean_sig_w=m_sig, mean_bkg_w=m_bkg, std_sig_w=s_sig, std_bkg_w=s_bkg,
        cohens_d=d, auc_weighted=auc_w, auc_raw_count=auc_raw,
        auc_boot_mean=auc_mean, auc_boot_lo68=auc_lo, auc_boot_hi68=auc_hi,
        ci_computed=do_boot,
        separation_score=abs(auc_w - 0.5),
    )


def run_region(master, u_thresh, min_n_for_ci):
    sub = master[master["u"] > u_thresh]
    sig = sub[sub["label"] == 1]
    rows = []
    for comp_name, class_filter in COMPARISONS:
        if class_filter is None:
            bkg = sub[sub["label"] == 0]
        else:
            bkg = sub[sub["background_class"] == class_filter]
        for feat in FEATURES:
            r = summarize_feature(sig, bkg, feat, min_n_for_ci)
            if r is not None:
                r["comparison"] = comp_name
                r["u_region"] = f"u>{u_thresh}"
                rows.append(r)
    return pd.DataFrame(rows)


def main():
    master = pd.read_parquet(f"{PKG}/HARVEY_MASTER_EVENT_TABLE.parquet")
    res35 = run_region(master, 3.5, min_n_for_ci=20)
    res45 = run_region(master, 4.5, min_n_for_ci=20)  # will mostly not get CIs, tiny N -- descriptive only
    res45["descriptive_only_tiny_N"] = True
    res35["descriptive_only_tiny_N"] = False
    full = pd.concat([res35, res45], ignore_index=True)

    # ranking: mean |AUC-0.5| across the 3 comparisons at u>3.5 ONLY (per instruction:
    # do not over-interpret u>4.5 due to tiny background)
    rank_basis = res35[res35["comparison"].isin(["all_background", "QCD", "ttbar"])]
    ranking = (rank_basis.groupby("feature")["separation_score"]
               .agg(["mean", "std", "count"]).reset_index()
               .sort_values("mean", ascending=False))
    ranking.columns = ["feature", "mean_abs_auc_minus_half_u3p5", "std_across_comparisons", "n_comparisons"]

    full.to_csv(f"{PKG}/HARVEY_1D_FEATURE_RANKING_full.csv", index=False)
    ranking.to_csv(f"{PKG}/HARVEY_1D_FEATURE_RANKING.csv", index=False)
    print(ranking.head(10).to_string(index=False))
    return full, ranking


if __name__ == "__main__":
    main()
