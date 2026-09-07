"""Track D, Task 4 -- validate the exploratory pT_H1 < 406.115 GeV ttbar
candidate cut using the two distinct frozen ttbar production lanes
(process labels `inference_ttbar_1`, `inference_ttbar_2` in
HARVEY_MASTER_EVENT_TABLE.parquet), since the original candidate was
chosen AND evaluated on the same 27-event pooled ttbar sample.

Read-only. No new inference, no new cut optimization beyond the simple,
predeclared leave-one-lane-out design below.
"""
import json
import os

import numpy as np
import pandas as pd

OUT_DIR = "/uscms_data/d3/iturkmen/hh4b_delphes/paper_exports/track_b_harvey_final_question_closure_20260907_v1"
WORK_DIR = os.path.join(OUT_DIR, "work")
FIG_DIR = os.path.join(OUT_DIR, "figures")

MASTER_PARQUET = ("/uscms_data/d3/iturkmen/repos/hh-bbww-baselines-bdt/docs/checkpoints/"
                   "track_b_harvey_tail_kinematics_20260907_v2/HARVEY_MASTER_EVENT_TABLE.parquet")

FROZEN_CUT = 406.1153676240428
FEATURES_TO_CHECK = ["pT_H1", "jet1_pt", "HT", "pT_H2", "jet2_pt"]


def wilson_ci(k, n, z=1.0):
    if n == 0:
        return (np.nan, np.nan)
    phat = k / n
    denom = 1 + z ** 2 / n
    center = (phat + z ** 2 / (2 * n)) / denom
    half = z * np.sqrt(phat * (1 - phat) / n + z ** 2 / (4 * n ** 2)) / denom
    return center - half, center + half


def evaluate_cut(threshold, ttbar_df, signal_df, lane_name):
    n_before = len(ttbar_df)
    n_after = int((ttbar_df["pT_H1"] < threshold).sum())
    rej = 1 - n_after / n_before if n_before else np.nan
    ci_lo, ci_hi = wilson_ci(n_before - n_after, n_before) if n_before else (np.nan, np.nan)
    n_sig_before = len(signal_df)
    n_sig_after = int((signal_df["pT_H1"] < threshold).sum())
    eff = n_sig_after / n_sig_before if n_sig_before else np.nan
    return dict(lane=lane_name, threshold=threshold, n_ttbar_before=n_before, n_ttbar_after=n_after,
                ttbar_rejection=rej, ttbar_rejection_ci68_lo=ci_lo, ttbar_rejection_ci68_hi=ci_hi,
                n_signal_before=n_sig_before, n_signal_after=n_sig_after, signal_efficiency=eff)


def main():
    df = pd.read_parquet(MASTER_PARQUET)
    lane1 = df[df["process"] == "inference_ttbar_1"].copy()
    lane2 = df[df["process"] == "inference_ttbar_2"].copy()
    signal = df[df["process"] == "signal"].copy()
    assert len(lane1) > 0 and len(lane2) > 0, "one or both ttbar lanes are empty in the master table"
    distinct_lanes = (set(lane1["event_uid"]) & set(lane2["event_uid"])) == set()
    assert distinct_lanes, "ttbar lanes are not disjoint by event_uid"

    rows = []
    rows.append(evaluate_cut(FROZEN_CUT, lane1, signal, "inference_ttbar_1"))
    rows.append(evaluate_cut(FROZEN_CUT, lane2, signal, "inference_ttbar_2"))
    rows.append(evaluate_cut(FROZEN_CUT, pd.concat([lane1, lane2]), signal, "pooled (both lanes)"))

    # ---- leave-one-lane-out: derive threshold (median of that lane's own pT_H1) on
    # one lane only, evaluate on the OTHER lane + the full frozen signal population ----
    thr_from_lane1 = float(lane1["pT_H1"].median())
    thr_from_lane2 = float(lane2["pT_H1"].median())
    loo_rows = []
    loo_rows.append(dict(derived_on="inference_ttbar_1", threshold=thr_from_lane1,
                          **{f"eval_{k}": v for k, v in evaluate_cut(thr_from_lane1, lane2, signal, "inference_ttbar_2").items()}))
    loo_rows.append(dict(derived_on="inference_ttbar_2", threshold=thr_from_lane2,
                          **{f"eval_{k}": v for k, v in evaluate_cut(thr_from_lane2, lane1, signal, "inference_ttbar_1").items()}))
    threshold_ratio = max(thr_from_lane1, thr_from_lane2) / min(thr_from_lane1, thr_from_lane2)
    thresholds_stable = threshold_ratio < 1.15

    # ---- qualitative "hard/boosted ttbar" check across both lanes independently ----
    qual_rows = []
    for feat in FEATURES_TO_CHECK:
        med_sig = float(signal[feat].median())
        med_l1 = float(lane1[feat].median())
        med_l2 = float(lane2[feat].median())
        qual_rows.append(dict(feature=feat, signal_median=med_sig,
                               lane1_median=med_l1, lane1_ratio_to_signal=med_l1 / med_sig,
                               lane2_median=med_l2, lane2_ratio_to_signal=med_l2 / med_sig,
                               both_lanes_boosted_gt1p5x=bool(med_l1 / med_sig > 1.5 and med_l2 / med_sig > 1.5)))

    n_qual_confirmed = sum(r["both_lanes_boosted_gt1p5x"] for r in qual_rows)
    verdict_qualitative_reproduced = n_qual_confirmed >= 4  # out of 5 features

    n1, n2 = len(lane1), len(lane2)
    if n1 < 10 or n2 < 10:
        overall_verdict = "SAMPLE_TOO_SMALL"
    elif thresholds_stable and verdict_qualitative_reproduced:
        overall_verdict = "REPRODUCIBLE_INDICATIVE"
    elif not thresholds_stable:
        overall_verdict = "NOT_REPRODUCED"
    else:
        overall_verdict = "SAMPLE_TOO_SMALL"

    out_csv = pd.DataFrame(rows)
    out_csv.to_csv(os.path.join(OUT_DIR, "TTBAR_LANE_VALIDATION.csv"), index=False)
    pd.DataFrame(loo_rows).to_csv(os.path.join(WORK_DIR, "ttbar_leave_one_lane_out.csv"), index=False)
    pd.DataFrame(qual_rows).to_csv(os.path.join(WORK_DIR, "ttbar_qualitative_feature_check.csv"), index=False)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(7, 5))
    bins = np.linspace(0, max(signal["pT_H1"].quantile(0.99), lane1["pT_H1"].max(), lane2["pT_H1"].max()), 30)
    ax.hist(signal["pT_H1"].to_numpy(dtype=float), bins=bins, density=True, histtype="step", color="gray", label=f"signal (n={len(signal)})")
    ax.hist(lane1["pT_H1"].to_numpy(dtype=float), bins=bins, density=True, histtype="step", color="C0", label=f"inference_ttbar_1 (n={n1})")
    ax.hist(lane2["pT_H1"].to_numpy(dtype=float), bins=bins, density=True, histtype="step", color="C1", label=f"inference_ttbar_2 (n={n2})")
    ax.axvline(FROZEN_CUT, color="red", ls="--", label=f"frozen candidate cut ({FROZEN_CUT:.1f} GeV)")
    ax.set_xlabel("pT_H1 [GeV]"); ax.set_ylabel("density")
    ax.set_title("pT_H1: signal vs. two independent ttbar lanes (u>3.5)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "ttbar_lane_validation_pTH1.png"), dpi=150)
    plt.close(fig)

    result = dict(
        frozen_candidate_cut_GeV=FROZEN_CUT,
        n_ttbar_lane1=n1, n_ttbar_lane2=n2, lanes_disjoint_by_event_uid=distinct_lanes,
        per_lane_results=rows,
        leave_one_lane_out=loo_rows,
        threshold_from_lane1_median=thr_from_lane1, threshold_from_lane2_median=thr_from_lane2,
        threshold_ratio=threshold_ratio, thresholds_stable_lt_1p15=thresholds_stable,
        qualitative_feature_check=qual_rows, n_features_both_lanes_boosted_gt1p5x=n_qual_confirmed,
        TTBAR_SUPPRESSION=overall_verdict,
    )
    with open(os.path.join(WORK_DIR, "ttbar_lane_validation_result.json"), "w") as f:
        json.dump(result, f, indent=2, default=str)
    print(json.dumps(result, indent=2, default=str))


if __name__ == "__main__":
    main()
