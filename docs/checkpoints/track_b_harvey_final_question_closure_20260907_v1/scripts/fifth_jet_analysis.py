"""Track D, Task 1 (part 2) -- post-processing analysis of the executed
leading-four-only counterfactual (FIFTH_JET_COUNTERFACTUAL_EVENTS_ALL400K.parquet).
Pure post-processing: no model, checkpoint, or HDF5 touched.
"""
import json
import os

import numpy as np
import pandas as pd

OUT_DIR = "/uscms_data/d3/iturkmen/hh4b_delphes/paper_exports/track_b_harvey_final_question_closure_20260907_v1"
WORK_DIR = os.path.join(OUT_DIR, "work")
FIG_DIR = os.path.join(OUT_DIR, "figures")

N_BOOT = 2000
RNG = np.random.default_rng(0)


def bootstrap_ci(x, stat_fn, n_boot=N_BOOT):
    x = np.asarray(x, dtype=np.float64)
    if len(x) == 0:
        return (np.nan, np.nan)
    idx = RNG.integers(0, len(x), size=(n_boot, len(x)))
    vals = stat_fn(x[idx], axis=1)
    return float(np.percentile(vals, 16)), float(np.percentile(vals, 84))


def main():
    df = pd.read_parquet(os.path.join(OUT_DIR, "FIFTH_JET_COUNTERFACTUAL_EVENTS_ALL400K.parquet"))
    sig5 = df[(df["process_label"] == "signal") & (df["n_selected_jets"] >= 5)].copy()
    n_sig5 = len(sig5)

    du = sig5["delta_u_logit"].to_numpy(dtype=float)
    median_du = float(np.median(du))
    mean_du = float(np.mean(du))
    ci_median = bootstrap_ci(du, np.median)
    ci_mean = bootstrap_ci(du, np.mean)

    frac_more_signal_like = float((du > 0).mean())  # u_logit_B > u_logit_A: removing jets>=5 RAISES score
    frac_less_signal_like = float((du < 0).mean())
    frac_unchanged = float((du == 0).mean())

    uA = sig5["u_logit_A"].to_numpy(dtype=float)
    uB = sig5["u_logit_B"].to_numpy(dtype=float)

    def crossing_counts(thr):
        a_pass = uA > thr
        b_pass = uB > thr
        gained = int((~a_pass & b_pass).sum())   # crosses INTO the tail after masking (B>thr, A<=thr)
        lost = int((a_pass & ~b_pass).sum())      # falls OUT of the tail after masking
        both = int((a_pass & b_pass).sum())
        neither = int((~a_pass & ~b_pass).sum())
        return dict(threshold=thr, n_pass_A=int(a_pass.sum()), n_pass_B=int(b_pass.sum()),
                    gained_B_only=gained, lost_A_only=lost, both=both, neither=neither)

    crossings_35 = crossing_counts(3.5)
    crossings_45 = crossing_counts(4.5)

    # ---- stratifications ----
    def strat_by_bins(col, bins, labels):
        cats = pd.cut(sig5[col], bins=bins, labels=labels, include_lowest=True)
        g = sig5.groupby(cats, observed=True)["delta_u_logit"]
        return {str(k): dict(n=int(v.shape[0]), median_delta_u_logit=float(v.median()), mean_delta_u_logit=float(v.mean()))
                for k, v in g if v.shape[0] > 0}

    strat_pt5 = strat_by_bins("pt5", [0, 30, 50, 75, 100, 150, np.inf], ["0-30", "30-50", "50-75", "75-100", "100-150", "150+"])
    strat_mindr = strat_by_bins("min_dR_j5_leading4", [0, 0.5, 1.0, 1.5, 2.0, np.inf], ["0-0.5", "0.5-1.0", "1.0-1.5", "1.5-2.0", "2.0+"])
    strat_njets = sig5.groupby("n_selected_jets")["delta_u_logit"].agg(["count", "median", "mean"]).reset_index()

    result = dict(
        n_signal_ge5jets=n_sig5,
        n_signal_total=int((df["process_label"] == "signal").sum()),
        median_delta_u_logit=median_du, median_delta_u_logit_ci68=ci_median,
        mean_delta_u_logit=mean_du, mean_delta_u_logit_ci68=ci_mean,
        frac_more_signal_like_after_masking=frac_more_signal_like,
        frac_less_signal_like_after_masking=frac_less_signal_like,
        frac_exactly_unchanged=frac_unchanged,
        threshold_crossings_u3p5=crossings_35,
        threshold_crossings_u4p5=crossings_45,
        stratification_by_pt5=strat_pt5,
        stratification_by_min_dR_j5_leading4=strat_mindr,
        hh_origin_truth_match_stratification="NOT ATTEMPTED -- would require joining this 400k development-cohort "
            "event identity (production_2M_val.h5 row index) to the physical master-table truth-matching convention "
            "(signal_holdout_A row identity), which is a DIFFERENT signal sample/identity; no proven join key exists "
            "between the two, so this stratification is not performed rather than assumed.",
        spanet_assignment_output_recovery="NOT ATTEMPTED in this pass. A legitimate ground-truth assignment target "
            "DOES exist for this population (TARGETS/h1/{b1,b2}, TARGETS/h2/{b3,b4} in production_2M_val.h5), so this "
            "would be technically feasible in a follow-up, but decoding the model's assignment head (not just the "
            "EVENT/signal classification head already captured here) is a separate, more expensive computation not "
            "run in this pass to keep it tractable. Not invented/estimated here.",
    )

    with open(os.path.join(WORK_DIR, "fifth_jet_analysis_result.json"), "w") as f:
        json.dump(result, f, indent=2, default=str)

    # per-event output table (only signal, n_selected_jets>=5, the causal population)
    sig5_out = sig5[["row_index", "n_selected_jets", "pt5", "min_dR_j5_leading4",
                      "u_logit_A", "u_logit_B", "delta_u_logit", "delta_logit_margin"]]
    sig5_out.to_parquet(os.path.join(OUT_DIR, "FIFTH_JET_COUNTERFACTUAL_EVENTS.parquet"))

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.hist(du, bins=80, color="C0")
    ax.axvline(0, color="k", lw=1)
    ax.axvline(median_du, color="red", ls="--", label=f"median={median_du:.3f}")
    ax.set_xlabel("Δu_logit = u_logit(leading-4-only) - u_logit(native)")
    ax.set_ylabel("events")
    ax.set_title(f"Fifth-jet counterfactual: score shift under masking (n={n_sig5} signal, ≥5 jets)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "delta_u_logit_distribution.png"), dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.scatter(sig5["pt5"].to_numpy(dtype=float), du, s=3, alpha=0.15)
    ax.axhline(0, color="k", lw=1)
    ax.set_xlabel("pT5 [GeV]"); ax.set_ylabel("Δu_logit")
    ax.set_title("Δu_logit vs. pT5")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "delta_u_logit_vs_pt5.png"), dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.scatter(sig5["min_dR_j5_leading4"].to_numpy(dtype=float), du, s=3, alpha=0.15, color="C2")
    ax.axhline(0, color="k", lw=1)
    ax.set_xlabel("min ΔR(j5, leading four)"); ax.set_ylabel("Δu_logit")
    ax.set_title("Δu_logit vs. min ΔR(j5, leading four)")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "delta_u_logit_vs_minDR5.png"), dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(1, 2, figsize=(11, 5))
    for a, (crossings, thr) in zip(ax, [(crossings_35, 3.5), (crossings_45, 4.5)]):
        cats = ["stayed\nbelow", "gained\n(A≤thr→B>thr)", "lost\n(A>thr→B≤thr)", "stayed\nabove"]
        vals = [crossings["neither"], crossings["gained_B_only"], crossings["lost_A_only"], crossings["both"]]
        a.bar(cats, vals, color=["gray", "C2", "C3", "C0"])
        a.set_title(f"u_logit threshold {thr}")
        a.set_ylabel("n events")
        for i, v in enumerate(vals):
            a.text(i, v, str(v), ha="center", va="bottom", fontsize=9)
    fig.suptitle("Threshold crossings under the leading-four-only counterfactual (signal, ≥5 jets)")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "threshold_crossings_counterfactual.png"), dpi=150)
    plt.close(fig)

    print(json.dumps(result, indent=2, default=str))


if __name__ == "__main__":
    main()
