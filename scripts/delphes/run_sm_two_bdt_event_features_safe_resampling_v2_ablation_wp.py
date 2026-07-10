'''
Resampling with Poisson bootstrap to estimate MC-statistical uncertainty in the two-BDT working points.
This is complementary to multiseed training stability. Multiseed variation probes training/split stability
while Poisson bootstrap resampling probes finite-MC statistical fluctuations in the selected regions.
'''

import importlib.util
import os
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split


REPO = Path(os.environ["HH4B_REPO"])
BASE_SCRIPT = REPO / "scripts/delphes/train_sm_normalized_hh4b_two_bdt_event_features_safe_v2.py"

OUTDIR = REPO / "outputs/tables/sm_normalized_hh4b_two_bdt_event_features_safe_resampling_v2_ablation_wp_2026_07_10"
OUTDIR.mkdir(parents=True, exist_ok=True)

LUMI_PB = 450.0 * 1000.0
TEST_SIZE = 0.35

SEEDS = list(range(10, 30))
N_BOOTSTRAPS = int(os.environ.get("N_BOOTSTRAPS", "200"))

# Main fixed working points to stress-test.
WORKING_POINTS = [
    (0.775, 0.500, "previous_v2_loose_mid"),
    (0.800, 0.500, "previous_v2_loose_best"),
    (0.825, 0.500, "ablation_best_median"),
    (0.850, 0.500, "ablation_tighter_qcd_same_top"),
    (0.825, 0.475, "ablation_qcd825_looser_top"),
    (0.850, 0.475, "qcd850_looser_top"),
    (0.825, 0.525, "previous_nominal_balanced"),
]




spec = importlib.util.spec_from_file_location("safe_event_bdt", BASE_SCRIPT)
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)

df, feature_cols = mod.load_all()


def class_balanced_weights(y):
    y = np.asarray(y).astype(int)
    n_pos = max((y == 1).sum(), 1)
    n_neg = max((y == 0).sum(), 1)
    return np.where(y == 1, 0.5 / n_pos, 0.5 / n_neg)


def neff(weights):
    w = np.asarray(weights, dtype=float)
    if len(w) == 0 or np.sum(w * w) <= 0:
        return 0.0
    return float((np.sum(w) ** 2) / np.sum(w * w))


def train_task(train_df, test_df, mask_col, seed):
    task_train = train_df[train_df["is_signal"] | train_df[mask_col]].copy()
    task_test = test_df[test_df["is_signal"] | test_df[mask_col]].copy()

    y_train = task_train["is_signal"].astype(int)
    y_test = task_test["is_signal"].astype(int)

    clf = GradientBoostingClassifier(
        n_estimators=250,
        learning_rate=0.04,
        max_depth=3,
        subsample=0.8,
        random_state=seed,
    )

    clf.fit(
        task_train[feature_cols],
        y_train,
        sample_weight=class_balanced_weights(y_train),
    )

    task_test = task_test.copy()
    task_test["score"] = clf.predict_proba(task_test[feature_cols])[:, 1]

    auc = roc_auc_score(y_test, task_test["score"])
    wauc = roc_auc_score(
        y_test,
        task_test["score"],
        sample_weight=task_test["weight_pb_scaled_to_full"],
    )

    all_scores = clf.predict_proba(test_df[feature_cols])[:, 1]

    return all_scores, auc, wauc


nominal_rows = []
bootstrap_rows = []
auc_rows = []
composition_rows = []

for seed in SEEDS:
    rng = np.random.default_rng(seed + 100000)

    train_df, test_df = train_test_split(
        df,
        test_size=TEST_SIZE,
        random_state=seed,
        stratify=df["target"],
    )

    total_counts = df.groupby("analysis_sample").size()
    test_counts = test_df.groupby("analysis_sample").size()

    test_df = test_df.copy()
    test_df["weight_pb_scaled_to_full"] = test_df["weight_pb"] * test_df["analysis_sample"].map(
        lambda s: total_counts.loc[s] / test_counts.loc[s]
    )

    qcd_scores, qcd_auc, qcd_wauc = train_task(train_df, test_df, "is_qcd", seed)
    top_scores, top_auc, top_wauc = train_task(train_df, test_df, "is_top", seed)

    test_df["bdt_qcd_score"] = qcd_scores
    test_df["bdt_top_score"] = top_scores

    auc_rows.append({
        "seed": seed,
        "qcd_auc": qcd_auc,
        "qcd_weighted_auc": qcd_wauc,
        "top_auc": top_auc,
        "top_weighted_auc": top_wauc,
    })

    for tq, tt, wp_name in WORKING_POINTS:
        sel = (test_df["bdt_qcd_score"] >= tq) & (test_df["bdt_top_score"] >= tt)
        selected = test_df.loc[sel].copy()

        sig = selected[selected["is_signal"]]
        bkg = selected[~selected["is_signal"]]

        s_pb = sig["weight_pb_scaled_to_full"].sum()
        b_pb = bkg["weight_pb_scaled_to_full"].sum()
        s_ev = s_pb * LUMI_PB
        b_ev = b_pb * LUMI_PB

        nominal_rows.append({
            "seed": seed,
            "working_point": wp_name,
            "qcd_threshold": tq,
            "top_threshold": tt,
            "signal_events_450fb": s_ev,
            "background_events_450fb": b_ev,
            "S_over_B": s_ev / b_ev if b_ev > 0 else np.nan,
            "S_over_sqrtB": s_ev / np.sqrt(b_ev) if b_ev > 0 else np.nan,
            "S_over_10pctB": s_ev / (0.10 * b_ev) if b_ev > 0 else np.nan,
            "n_signal_test_rows": len(sig),
            "n_background_test_rows": len(bkg),
            "neff_signal": neff(sig["weight_pb_scaled_to_full"]),
            "neff_background": neff(bkg["weight_pb_scaled_to_full"]),
        })

        # Per-sample nominal composition for this seed and WP.
        for (sample, group), sub in selected.groupby(["analysis_sample", "group"]):
            xsec_pb = sub["weight_pb_scaled_to_full"].sum()
            composition_rows.append({
                "seed": seed,
                "working_point": wp_name,
                "qcd_threshold": tq,
                "top_threshold": tt,
                "sample": sample,
                "group": group,
                "rows": len(sub),
                "xsec_pb": xsec_pb,
                "expected_events_450fb": xsec_pb * LUMI_PB,
            })

        # Poisson bootstrap / resampling of selected rows.
        if len(selected) == 0:
            continue

        w = selected["weight_pb_scaled_to_full"].to_numpy()
        is_sig = selected["is_signal"].to_numpy(dtype=bool)

        for iboot in range(N_BOOTSTRAPS):
            k = rng.poisson(1.0, size=len(selected))

            s_boot_pb = np.sum(k[is_sig] * w[is_sig])
            b_boot_pb = np.sum(k[~is_sig] * w[~is_sig])

            s_boot_ev = s_boot_pb * LUMI_PB
            b_boot_ev = b_boot_pb * LUMI_PB

            bootstrap_rows.append({
                "seed": seed,
                "bootstrap": iboot,
                "working_point": wp_name,
                "qcd_threshold": tq,
                "top_threshold": tt,
                "signal_events_450fb": s_boot_ev,
                "background_events_450fb": b_boot_ev,
                "S_over_B": s_boot_ev / b_boot_ev if b_boot_ev > 0 else np.nan,
                "S_over_sqrtB": s_boot_ev / np.sqrt(b_boot_ev) if b_boot_ev > 0 else np.nan,
                "S_over_10pctB": s_boot_ev / (0.10 * b_boot_ev) if b_boot_ev > 0 else np.nan,
            })

nominal = pd.DataFrame(nominal_rows)
boot = pd.DataFrame(bootstrap_rows)
auc = pd.DataFrame(auc_rows)
composition = pd.DataFrame(composition_rows)

nominal_summary = (
    nominal.groupby(["working_point", "qcd_threshold", "top_threshold"], as_index=False)
    .agg(
        n_seeds=("seed", "count"),
        S_over_sqrtB_mean=("S_over_sqrtB", "mean"),
        S_over_sqrtB_std_seed=("S_over_sqrtB", "std"),
        S_over_B_mean=("S_over_B", "mean"),
        S_over_B_std_seed=("S_over_B", "std"),
        median_background_test_rows=("n_background_test_rows", "median"),
        min_background_test_rows=("n_background_test_rows", "min"),
        median_neff_background=("neff_background", "median"),
        min_neff_background=("neff_background", "min"),
        median_signal_test_rows=("n_signal_test_rows", "median"),
        min_signal_test_rows=("n_signal_test_rows", "min"),
    )
    .sort_values("S_over_sqrtB_mean", ascending=False)
)

boot_summary = (
    boot.groupby(["working_point", "qcd_threshold", "top_threshold"], as_index=False)
    .agg(
        n_boot=("bootstrap", "count"),
        S_over_sqrtB_mean=("S_over_sqrtB", "mean"),
        S_over_sqrtB_std_bootstrap=("S_over_sqrtB", "std"),
        S_over_sqrtB_p16=("S_over_sqrtB", lambda x: float(np.nanpercentile(x, 16))),
        S_over_sqrtB_p50=("S_over_sqrtB", lambda x: float(np.nanpercentile(x, 50))),
        S_over_sqrtB_p84=("S_over_sqrtB", lambda x: float(np.nanpercentile(x, 84))),
        S_over_B_mean=("S_over_B", "mean"),
        S_over_B_std_bootstrap=("S_over_B", "std"),
        B_events_p16=("background_events_450fb", lambda x: float(np.nanpercentile(x, 16))),
        B_events_p50=("background_events_450fb", lambda x: float(np.nanpercentile(x, 50))),
        B_events_p84=("background_events_450fb", lambda x: float(np.nanpercentile(x, 84))),
    )
    .sort_values("S_over_sqrtB_mean", ascending=False)
)

auc_summary = auc.agg({
    "qcd_auc": ["mean", "std"],
    "qcd_weighted_auc": ["mean", "std"],
    "top_auc": ["mean", "std"],
    "top_weighted_auc": ["mean", "std"],
})

composition_summary = (
    composition.groupby(["working_point", "sample", "group"], as_index=False)
    .agg(
        mean_expected_events_450fb=("expected_events_450fb", "mean"),
        std_expected_events_450fb=("expected_events_450fb", "std"),
        median_rows=("rows", "median"),
    )
)

composition_summary["fraction_within_group"] = (
    composition_summary["mean_expected_events_450fb"]
    / composition_summary.groupby(["working_point", "group"])["mean_expected_events_450fb"].transform("sum")
)

nominal.to_csv(OUTDIR / "resampling_nominal_by_seed.csv", index=False)
boot.to_csv(OUTDIR / "resampling_poisson_bootstrap_detail.csv", index=False)
auc.to_csv(OUTDIR / "resampling_auc_by_seed.csv", index=False)
composition.to_csv(OUTDIR / "resampling_composition_by_seed.csv", index=False)

nominal_summary.to_csv(OUTDIR / "resampling_nominal_seed_summary.csv", index=False)
boot_summary.to_csv(OUTDIR / "resampling_poisson_bootstrap_summary.csv", index=False)
composition_summary.to_csv(OUTDIR / "resampling_composition_summary.csv", index=False)

(OUTDIR / "resampling_nominal_seed_summary.md").write_text(nominal_summary.to_markdown(index=False) + "\n")
(OUTDIR / "resampling_poisson_bootstrap_summary.md").write_text(boot_summary.to_markdown(index=False) + "\n")
(OUTDIR / "resampling_auc_summary.md").write_text(auc_summary.to_markdown() + "\n")
(OUTDIR / "resampling_composition_summary.md").write_text(composition_summary.to_markdown(index=False) + "\n")

readme = f"""# Safe event-feature two-BDT resampling study

This study adds Poisson bootstrap resampling of selected test events to estimate MC-statistical uncertainty in the two-BDT working points.

Number of train/test seeds: {len(SEEDS)}
Poisson bootstraps per seed per working point: {N_BOOTSTRAPS}

Working points:
{WORKING_POINTS}

This is complementary to multiseed training stability. Multiseed variation probes training/split stability, while Poisson bootstrap resampling probes finite-MC statistical fluctuations in the selected regions.
"""
(OUTDIR / "README.md").write_text(readme)

print("\n=== Resampling AUC summary ===")
print(auc_summary.to_string())

print("\n=== Nominal seed summary ===")
print(nominal_summary.to_string(index=False))

print("\n=== Poisson bootstrap resampling summary ===")
print(boot_summary.to_string(index=False))

print("\n=== Composition summary ===")
print(composition_summary.to_string(index=False))

print(f"\nWrote outputs to: {OUTDIR}")
