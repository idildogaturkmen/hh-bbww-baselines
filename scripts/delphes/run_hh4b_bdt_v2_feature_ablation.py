#!/usr/bin/env python3

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
OUTDIR = REPO / "outputs/tables/hh4b_bdt_v2_feature_ablation_2026_07_10"
OUTDIR.mkdir(parents=True, exist_ok=True)

LUMI_PB = 450_000.0
SEEDS = list(range(5))

spec = importlib.util.spec_from_file_location("bdt_v2_loader", BASE_SCRIPT)
loader = importlib.util.module_from_spec(spec)
spec.loader.exec_module(loader)


def class_balanced_weights(y):
    y = np.asarray(y).astype(int)
    n0 = max((y == 0).sum(), 1)
    n1 = max((y == 1).sum(), 1)
    w = np.ones(len(y), dtype=float)
    w[y == 0] = len(y) / (2.0 * n0)
    w[y == 1] = len(y) / (2.0 * n1)
    return w


def neff(weights):
    w = np.asarray(weights, dtype=float)
    denom = np.sum(w * w)
    return float(w.sum() * w.sum() / denom) if denom > 0 else 0.0


def make_clf(seed):
    return GradientBoostingClassifier(
        n_estimators=180,
        learning_rate=0.04,
        max_depth=3,
        subsample=0.8,
        random_state=seed,
    )


def train_task(train_df, test_df, features, mask_col, seed):
    task_train = train_df[train_df["is_signal"] | train_df[mask_col]].copy()
    task_test = test_df[test_df["is_signal"] | test_df[mask_col]].copy()

    y_train = task_train["target"].to_numpy().astype(int)
    y_test = task_test["target"].to_numpy().astype(int)

    clf = make_clf(seed)
    clf.fit(
        task_train[features],
        y_train,
        sample_weight=class_balanced_weights(y_train),
    )

    task_scores = clf.predict_proba(task_test[features])[:, 1]
    all_scores = clf.predict_proba(test_df[features])[:, 1]

    auc = roc_auc_score(y_test, task_scores)
    wauc = roc_auc_score(
        y_test,
        task_scores,
        sample_weight=task_test["weight_pb_scaled_to_full"],
    )

    return all_scores, auc, wauc


def scan(test_df):
    rows = []
    for tq in np.round(np.arange(0.50, 0.951, 0.025), 3):
        for tt in np.round(np.arange(0.50, 0.951, 0.025), 3):
            sel = (test_df["qcd_score"] >= tq) & (test_df["top_score"] >= tt)
            selected = test_df[sel]
            sig = selected[selected["is_signal"]]
            bkg = selected[~selected["is_signal"]]

            s_pb = sig["weight_pb_scaled_to_full"].sum()
            b_pb = bkg["weight_pb_scaled_to_full"].sum()
            s_ev = s_pb * LUMI_PB
            b_ev = b_pb * LUMI_PB

            rows.append({
                "qcd_threshold": tq,
                "top_threshold": tt,
                "signal_events_450fb": s_ev,
                "background_events_450fb": b_ev,
                "S_over_B": s_ev / b_ev if b_ev > 0 else np.nan,
                "S_over_sqrtB": s_ev / np.sqrt(b_ev) if b_ev > 0 else np.nan,
                "n_signal_test_rows": len(sig),
                "n_background_test_rows": len(bkg),
                "neff_signal": neff(sig["weight_pb_scaled_to_full"]),
                "neff_background": neff(bkg["weight_pb_scaled_to_full"]),
            })
    return pd.DataFrame(rows)


def run():
    df, all_features = loader.load_all()

    requested_groups = {
        "mass_core": [
            "mbb1", "mbb2", "avg_mbb", "delta_mbb",
            "r_hh", "r_hh_125_125", "r_hh_125_120", "mhh",
        ],
        "angular_topology": [
            "drbb1", "drbb2", "h_delta_eta", "h_delta_phi",
            "h_delta_r", "h_pt_balance", "avg_drbb", "max_drbb", "min_drbb",
        ],
        "event_activity": [
            "n_selected_jets", "n_selected_bjets", "n_extra_selected_jets",
            "n_extra_selected_bjets", "ht_selected_jets", "ht_selected_bjets",
            "ht_candidate_jets", "ht_over_mhh",
        ],
        "jet_kinematics": [
            "j1_pt", "j2_pt", "j3_pt", "j4_pt",
            "j1_eta", "j2_eta", "j3_eta", "j4_eta",
            "j1_mass", "j2_mass", "j3_mass", "j4_mass",
            "pt_sum4", "pt_asym_12", "pt_asym_34",
            "hh_pt", "hh_eta", "h1_pt", "h1_eta", "h2_pt", "h2_eta",
        ],
        "all_v2_features": list(all_features),
    }

    feature_groups = {}
    for name, feats in requested_groups.items():
        keep = [f for f in feats if f in df.columns and f in all_features]
        feature_groups[name] = keep
        print(f"{name}: {len(keep)} features")

    auc_rows = []
    best_rows = []

    for group_name, features in feature_groups.items():
        if len(features) == 0:
            continue

        for seed in SEEDS:
            print(f"\n=== {group_name}, seed {seed} ===")

            train_df, test_df = train_test_split(
                df,
                test_size=0.35,
                random_state=seed,
                stratify=df["target"],
            )
            train_df = train_df.copy()
            test_df = test_df.copy()

            total_counts = df.groupby("analysis_sample").size()
            test_counts = test_df.groupby("analysis_sample").size()
            scale_to_full = (total_counts / test_counts).replace([np.inf, -np.inf], np.nan).fillna(1.0)
            test_df["weight_pb_scaled_to_full"] = test_df["weight_pb"] * test_df["analysis_sample"].map(scale_to_full)

            qcd_scores, qcd_auc, qcd_wauc = train_task(train_df, test_df, features, "is_qcd", seed + 101)
            top_scores, top_auc, top_wauc = train_task(train_df, test_df, features, "is_top", seed + 202)

            test_df["qcd_score"] = qcd_scores
            test_df["top_score"] = top_scores

            auc_rows.append({
                "feature_group": group_name,
                "seed": seed,
                "n_features": len(features),
                "qcd_auc": qcd_auc,
                "qcd_weighted_auc": qcd_wauc,
                "top_auc": top_auc,
                "top_weighted_auc": top_wauc,
            })

            sc = scan(test_df)
            stable = sc[
                (sc["n_background_test_rows"] >= 50)
                & (sc["neff_background"] >= 20)
                & (sc["n_signal_test_rows"] >= 30)
            ].copy()

            if len(stable):
                best = stable.sort_values("S_over_sqrtB", ascending=False).iloc[0].to_dict()
                best["feature_group"] = group_name
                best["seed"] = seed
                best["n_features"] = len(features)
                best_rows.append(best)

    auc = pd.DataFrame(auc_rows)
    best = pd.DataFrame(best_rows)

    auc_summary = (
        auc.groupby("feature_group", as_index=False)
        .agg(
            n_features=("n_features", "first"),
            qcd_weighted_auc_mean=("qcd_weighted_auc", "mean"),
            qcd_weighted_auc_std=("qcd_weighted_auc", "std"),
            top_weighted_auc_mean=("top_weighted_auc", "mean"),
            top_weighted_auc_std=("top_weighted_auc", "std"),
        )
        .sort_values("qcd_weighted_auc_mean", ascending=False)
    )

    best_summary = (
        best.groupby("feature_group", as_index=False)
        .agg(
            n_features=("n_features", "first"),
            S_over_sqrtB_mean=("S_over_sqrtB", "mean"),
            S_over_sqrtB_std=("S_over_sqrtB", "std"),
            S_over_B_mean=("S_over_B", "mean"),
            qcd_threshold_median=("qcd_threshold", "median"),
            top_threshold_median=("top_threshold", "median"),
            median_neff_background=("neff_background", "median"),
            median_background_test_rows=("n_background_test_rows", "median"),
        )
        .sort_values("S_over_sqrtB_mean", ascending=False)
    )

    auc.to_csv(OUTDIR / "feature_ablation_auc_by_seed.csv", index=False)
    best.to_csv(OUTDIR / "feature_ablation_best_by_seed.csv", index=False)
    auc_summary.to_csv(OUTDIR / "feature_ablation_auc_summary.csv", index=False)
    best_summary.to_csv(OUTDIR / "feature_ablation_best_summary.csv", index=False)

    (OUTDIR / "feature_ablation_auc_summary.md").write_text(auc_summary.to_markdown(index=False) + "\n")
    (OUTDIR / "feature_ablation_best_summary.md").write_text(best_summary.to_markdown(index=False) + "\n")

    readme = """# HH4b BDT-v2 feature ablation

Compares BDT-v2 performance using different groups of reconstructed v2 features.

Feature groups:
- mass_core
- angular_topology
- event_activity
- jet_kinematics
- all_v2_features

Purpose:
Identify which reconstructed quantities drive QCD/top separation and whether a more complex model is likely to add information beyond the current tabular BDT.
"""
    (OUTDIR / "README.md").write_text(readme)

    print("\n=== AUC summary ===")
    print(auc_summary.to_string(index=False))

    print("\n=== Best stable significance summary ===")
    print(best_summary.to_string(index=False))

    print("\nWrote:", OUTDIR)


if __name__ == "__main__":
    run()
