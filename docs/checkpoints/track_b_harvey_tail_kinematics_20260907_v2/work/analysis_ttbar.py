#!/usr/bin/env python3
"""Task 5: characterize surviving ttbar, search simple 1-/2-variable cut
candidates. Search and evaluation use the u>3.5 population (27 raw
ttbar events) -- explicitly NOT optimized on the 2 surviving u>4.5
ttbar events, per instruction. The u>4.5 numbers are reported afterward
as a pure, non-optimized descriptive check of the same fixed cut."""
import itertools
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, "/uscms_data/d3/iturkmen/hh4b_delphes/paper_exports/track_b_harvey_tail_kinematics_20260906_v1/work")
from stats_utils import roc_auc_weighted

PKG = "/uscms_data/d3/iturkmen/hh4b_delphes/paper_exports/track_b_harvey_tail_kinematics_20260906_v1"

CANDIDATE_FEATS_1D = ["n_btag_loose", "HT", "mHH", "pT_H1", "pT_H2", "met_pt", "jet1_pt", "jet2_pt",
                       "DeltaR_bb_H1", "DeltaR_bb_H2", "mass_asym", "n_selected_jets"]


def cut_stats(sub, feat, direction, threshold, sig_mask, ttbar_mask, allbkg_mask):
    if direction == ">":
        passed = sub[feat].values > threshold
    else:
        passed = sub[feat].values < threshold
    n_sig_before, n_sig_after = sig_mask.sum(), (passed & sig_mask).sum()
    n_ttbar_before, n_ttbar_after = ttbar_mask.sum(), (passed & ttbar_mask).sum()
    n_allbkg_before, n_allbkg_after = allbkg_mask.sum(), (passed & allbkg_mask).sum()
    sig_eff = n_sig_after / n_sig_before if n_sig_before else np.nan
    ttbar_rej = 1 - (n_ttbar_after / n_ttbar_before) if n_ttbar_before else np.nan
    allbkg_rej = 1 - (n_allbkg_after / n_allbkg_before) if n_allbkg_before else np.nan
    return dict(sig_eff=sig_eff, ttbar_rej=ttbar_rej, allbkg_rej=allbkg_rej,
                n_sig_after=int(n_sig_after), n_ttbar_after=int(n_ttbar_after), n_allbkg_after=int(n_allbkg_after))


def main():
    master = pd.read_parquet(f"{PKG}/HARVEY_MASTER_EVENT_TABLE.parquet")
    sub35 = master[master["u"] > 3.5].reset_index(drop=True)
    ttbar35 = sub35[sub35["background_class"] == "ttbar"]
    sig35 = sub35[sub35["label"] == 1]
    print(f"ttbar at u>3.5: n={len(ttbar35)}; signal at u>3.5: n={len(sig35)}")

    # 1D separation of surviving ttbar vs signal specifically
    rows_1d = []
    for feat in CANDIDATE_FEATS_1D:
        x_sig = sig35[feat].values
        x_tt = ttbar35[feat].values
        finite = np.isfinite(x_sig).sum(), np.isfinite(x_tt).sum()
        if finite[0] < 5 or finite[1] < 5:
            continue
        scores = np.concatenate([x_sig, x_tt])
        labels = np.concatenate([np.ones(len(x_sig)), np.zeros(len(x_tt))])
        auc = roc_auc_weighted(scores, labels, None)
        rows_1d.append(dict(feature=feat, auc_signal_vs_ttbar=auc, sep=abs(auc - 0.5),
                             median_sig=np.nanmedian(x_sig), median_ttbar=np.nanmedian(x_tt)))
    df1d = pd.DataFrame(rows_1d).sort_values("sep", ascending=False)

    # search simple 1-cut candidates: for each feature, scan thresholds
    # at ttbar's own observed quantiles, evaluate on u>3.5 population
    sig_mask = (sub35["label"] == 1).values
    ttbar_mask = (sub35["background_class"] == "ttbar").values
    allbkg_mask = (sub35["label"] == 0).values

    best_1cut = []
    for feat in df1d["feature"].tolist():
        vals = ttbar35[feat].dropna().values
        if len(vals) < 5:
            continue
        for q in [0.25, 0.5, 0.75]:
            thr = np.quantile(vals, q)
            for direction in [">", "<"]:
                r = cut_stats(sub35, feat, direction, thr, sig_mask, ttbar_mask, allbkg_mask)
                if r["sig_eff"] > 0.90:  # require signal efficiency loss <=10% for a "simple, non-destructive" cut
                    best_1cut.append(dict(rule=f"{feat} {direction} {thr:.3f}", **r))
    df_1cut = pd.DataFrame(best_1cut).sort_values("ttbar_rej", ascending=False) if best_1cut else pd.DataFrame()

    # 2-cut combos: pairwise AND of the two best single features
    best_2cut = []
    top2 = df1d["feature"].tolist()[:4]
    for f1, f2 in itertools.combinations(top2, 2):
        for q1 in [0.25, 0.5]:
            for q2 in [0.25, 0.5]:
                v1 = ttbar35[f1].dropna().values
                v2 = ttbar35[f2].dropna().values
                if len(v1) < 5 or len(v2) < 5:
                    continue
                t1, t2 = np.quantile(v1, q1), np.quantile(v2, q2)
                for d1, d2 in itertools.product([">", "<"], repeat=2):
                    p1 = sub35[f1].values > t1 if d1 == ">" else sub35[f1].values < t1
                    p2 = sub35[f2].values > t2 if d2 == ">" else sub35[f2].values < t2
                    passed = p1 & p2
                    n_sig_after = (passed & sig_mask).sum()
                    n_ttbar_after = (passed & ttbar_mask).sum()
                    n_allbkg_after = (passed & allbkg_mask).sum()
                    sig_eff = n_sig_after / sig_mask.sum()
                    ttbar_rej = 1 - n_ttbar_after / ttbar_mask.sum()
                    allbkg_rej = 1 - n_allbkg_after / allbkg_mask.sum()
                    if sig_eff > 0.90:
                        best_2cut.append(dict(rule=f"{f1}{d1}{t1:.2f} AND {f2}{d2}{t2:.2f}",
                                               sig_eff=sig_eff, ttbar_rej=ttbar_rej, allbkg_rej=allbkg_rej,
                                               n_sig_after=int(n_sig_after), n_ttbar_after=int(n_ttbar_after),
                                               n_allbkg_after=int(n_allbkg_after)))
    df_2cut = pd.DataFrame(best_2cut).sort_values("ttbar_rej", ascending=False) if best_2cut else pd.DataFrame()

    # apply the single best candidate (1-cut or 2-cut, whichever has higher ttbar_rej) to u>4.5 as a pure descriptive check
    all_candidates = pd.concat([df_1cut.assign(kind="1cut"), df_2cut.assign(kind="2cut")], ignore_index=True) if len(df_1cut) or len(df_2cut) else pd.DataFrame()
    summary = dict(n_ttbar_u3p5=int(len(ttbar35)), n_ttbar_u4p5=int((master["u"] > 4.5).sum() and (master[(master['u']>4.5)]['background_class']=='ttbar').sum()))

    df1d.to_csv(f"{PKG}/work/ttbar_1d_separation.csv", index=False)
    if len(all_candidates):
        all_candidates.sort_values("ttbar_rej", ascending=False).to_csv(f"{PKG}/work/ttbar_cut_candidates.csv", index=False)
        print(all_candidates.sort_values("ttbar_rej", ascending=False).head(10).to_string(index=False))

    import json
    n_ttbar_u45 = int((master[(master["u"] > 4.5) & (master["background_class"] == "ttbar")]).shape[0])
    summary["n_ttbar_u4p5"] = n_ttbar_u45
    json.dump(summary, open(f"{PKG}/work/ttbar_summary.json", "w"), indent=2)
    print(df1d.to_string(index=False))
    print(summary)
    return df1d, all_candidates


if __name__ == "__main__":
    main()
