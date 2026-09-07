"""Track D, Task 3 -- are the FP32 spike bins (k=2, u~6.9237; k=4, u~6.6227)
kinematically special, in the physical signal sample?

Track A already established these are float32-probability quantization
artifacts of the softmax, not discrete classifier modes -- this task does
NOT reopen that. It asks a different, purely kinematic question: within
the physical signal population (HARVEY_MASTER_EVENT_TABLE.parquet, u>3.5,
27,339 signal events, a DIFFERENT sample from Track A's 400k development
cohort -- identity NOT assumed equivalent, per instruction), do events
that happen to land in the k=2/k=4 float32 bins look kinematically
different from events in neighboring populated float32 bins, or does the
kinematics vary smoothly through the float32 comb?

Read-only. No new inference. No frozen file modified.
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

SPACING = 2.0 ** -24
FEATURES = ["n_selected_jets", "n_btag_loose", "HT", "mHH", "RHH", "pT_H1", "pT_H2",
            "jet1_pt", "jet2_pt", "DeltaR_bb_H1", "DeltaR_bb_H2", "pt5", "min_dR_j5_leading4"]

N_BOOT = 5000
RNG = np.random.default_rng(0)


def bootstrap_median_ci(x, n_boot=N_BOOT):
    x = np.asarray(x, dtype=np.float64)
    x = x[~np.isnan(x)]
    if len(x) == 0:
        return dict(median=np.nan, ci_lo=np.nan, ci_hi=np.nan, n=0)
    idx = RNG.integers(0, len(x), size=(n_boot, len(x)))
    meds = np.median(x[idx], axis=1)
    return dict(median=float(np.median(x)), ci_lo=float(np.percentile(meds, 16)),
                ci_hi=float(np.percentile(meds, 84)), n=int(len(x)))


def cliffs_delta(a, b):
    a = np.asarray(a, dtype=np.float64); a = a[~np.isnan(a)]
    b = np.asarray(b, dtype=np.float64); b = b[~np.isnan(b)]
    if len(a) == 0 or len(b) == 0:
        return np.nan
    gt = sum((ai > b).sum() for ai in a)
    lt = sum((ai < b).sum() for ai in a)
    return (gt - lt) / (len(a) * len(b))


def main():
    df = pd.read_parquet(MASTER_PARQUET)
    sig = df[df["process"] == "signal"].copy()
    assert len(sig) == 27339, f"unexpected signal count in master table: {len(sig)}"

    below_one = sig["score"] < 1.0
    k = np.full(len(sig), -1, dtype=np.int64)
    k[below_one.values] = np.round((1.0 - sig.loc[below_one, "score"].values) / SPACING).astype(np.int64)
    sig = sig.assign(k_lattice=k)

    counts = sig["k_lattice"].value_counts().sort_index()
    counts_table = counts[counts.index >= 0].reset_index()
    counts_table.columns = ["k_lattice", "n_events"]
    counts_table.to_csv(os.path.join(WORK_DIR, "physical_signal_k_lattice_counts.csv"), index=False)

    target_ks = [2, 4]
    populated_ks = sorted(int(x) for x in counts_table["k_lattice"] if counts_table.loc[counts_table.k_lattice == x, "n_events"].iloc[0] >= 5)

    def neighbors_of(k0, n_side=2):
        idx = populated_ks.index(k0) if k0 in populated_ks else None
        if idx is None:
            return []
        lo = max(0, idx - n_side)
        hi = min(len(populated_ks), idx + n_side + 1)
        return [kk for kk in populated_ks[lo:hi] if kk != k0]

    rows = []
    effect_rows = []
    for k0 in target_ks:
        target = sig[sig["k_lattice"] == k0]
        neigh_ks = neighbors_of(k0)
        neigh = sig[sig["k_lattice"].isin(neigh_ks)]
        rows.append(dict(k_lattice=k0, role="TARGET", n_events=len(target), neighbor_ks=neigh_ks))
        for feat in FEATURES:
            tgt_stat = bootstrap_median_ci(target[feat].values)
            nb_stat = bootstrap_median_ci(neigh[feat].values)
            delta = cliffs_delta(target[feat].values, neigh[feat].values)
            effect_rows.append(dict(
                k_lattice=k0, feature=feat,
                target_n=tgt_stat["n"], target_median=tgt_stat["median"],
                target_ci_lo=tgt_stat["ci_lo"], target_ci_hi=tgt_stat["ci_hi"],
                neighbor_n=nb_stat["n"], neighbor_median=nb_stat["median"],
                neighbor_ci_lo=nb_stat["ci_lo"], neighbor_ci_hi=nb_stat["ci_hi"],
                cliffs_delta=delta,
            ))

    effect_df = pd.DataFrame(effect_rows)
    effect_df.to_csv(os.path.join(OUT_DIR, "FP32_SPIKE_KINEMATICS_TABLE.csv"), index=False)

    # global smooth-through-comb check: median of each feature vs k, over all
    # populated k in [populated_ks[0], populated_ks[-1]] with n>=5
    smooth_check = {}
    for feat in FEATURES:
        meds = sig.groupby("k_lattice")[feat].median()
        meds = meds[meds.index.isin(populated_ks)].sort_index()
        smooth_check[feat] = {int(kk): (float(v) if pd.notna(v) else None) for kk, v in meds.items()}

    n_large_effect = int((effect_df["cliffs_delta"].abs() >= 0.33).sum())
    n_features_tested = len(effect_df)
    if n_large_effect == 0:
        verdict = "NO_CLEAR_SPECIAL_MODE"
    elif n_large_effect / n_features_tested < 0.15:
        verdict = "INCONCLUSIVE"
    else:
        verdict = "DISTINCT"

    os.makedirs(FIG_DIR, exist_ok=True)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    plot_feats = ["HT", "mHH", "RHH", "pT_H1", "n_btag_loose", "n_selected_jets"]
    for ax, feat in zip(axes.ravel(), plot_feats):
        meds = sig.groupby("k_lattice")[feat].median()
        meds = meds[meds.index.isin(populated_ks)].sort_index()
        meds_idx = np.asarray(meds.index, dtype=np.float64)
        meds_val = np.asarray(meds.values, dtype=np.float64)
        ax.plot(meds_idx, meds_val, "o-", color="gray", ms=3, lw=0.8, label="all populated k")
        for k0, color in zip(target_ks, ["C1", "C2"]):
            if k0 in meds.index:
                ax.plot(float(k0), float(meds.loc[k0]), "*", color=color, ms=16, label=f"k={k0}")
        ax.set_xlabel("k (float32 lattice steps from 1.0)")
        ax.set_ylabel(f"median {feat}")
    axes.ravel()[0].legend(fontsize=8)
    fig.suptitle("Median kinematics vs. FP32 lattice bin k (physical signal, u>3.5)")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "fp32_spike_feature_comparison.png"), dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 5.5))
    ax.scatter(sig["HT"].to_numpy(), sig["mHH"].to_numpy(), s=4, alpha=0.15, color="gray", label="all signal, u>3.5")
    for k0, color in zip(target_ks, ["C1", "C2"]):
        sub = sig[sig["k_lattice"] == k0]
        ax.scatter(sub["HT"].to_numpy(), sub["mHH"].to_numpy(), s=18, color=color, label=f"k={k0} (n={len(sub)})")
    ax.set_xlabel("HT [GeV]"); ax.set_ylabel("mHH [GeV]")
    ax.legend()
    ax.set_title("HT vs mHH: spike-bin events vs. full u>3.5 signal population")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "fp32_spike_ht_mhh.png"), dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 5))
    fracs, labels = [], []
    for k0 in target_ks:
        sub = sig[sig["k_lattice"] == k0]
        fracs.append(float((sub["n_selected_jets"] >= 5).mean()) if len(sub) else np.nan)
        labels.append(f"k={k0}\n(n={len(sub)})")
    baseline = float((sig["n_selected_jets"] >= 5).mean())
    ax.bar(labels, fracs, color=["C1", "C2"])
    ax.axhline(baseline, color="k", ls="--", label=f"all u>3.5 signal ({baseline:.3f})")
    ax.set_ylabel("fraction with >=5 selected jets")
    ax.set_title("Fifth-jet presence: spike bins vs. full population")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "fp32_spike_fifth_jet_fraction.png"), dpi=150)
    plt.close(fig)

    result = dict(
        master_parquet=MASTER_PARQUET, n_signal_u35=int(len(sig)),
        note="physical signal sample (holdout_A-derived, u>3.5) -- NOT the same event "
             "identity as Track A's 400k development cohort; only the float32 SCORE BIN "
             "definition (k-lattice) is shared/reused.",
        target_ks=target_ks,
        k2_n_events=int((sig["k_lattice"] == 2).sum()), k4_n_events=int((sig["k_lattice"] == 4).sum()),
        populated_ks_ge5events=populated_ks,
        neighbors_used={k0: neighbors_of(k0) for k0 in target_ks},
        n_features_tested=n_features_tested, n_large_effect_cliffs_ge_0p33=n_large_effect,
        smooth_through_comb_medians=smooth_check,
        verdict="SPIKE_KINEMATICS = " + verdict,
    )
    with open(os.path.join(WORK_DIR, "fp32_spike_kinematics_result.json"), "w") as f:
        json.dump(result, f, indent=2, default=str)
    print(json.dumps(result, indent=2, default=str))


if __name__ == "__main__":
    main()
