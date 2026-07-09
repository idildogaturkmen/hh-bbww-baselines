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
SCRIPT = REPO / "scripts/delphes/train_sm_normalized_hh4b_bdt.py"
OUTDIR = REPO / "outputs/tables/sm_normalized_hh4b_bdt_multiseed_2026_07_09"
OUTDIR.mkdir(parents=True, exist_ok=True)

spec = importlib.util.spec_from_file_location("bdtmod", SCRIPT)
bdtmod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(bdtmod)

df, feature_cols = bdtmod.load_all()

LUMI_PB = 450.0 * 1000.0
THRESHOLDS = [0.84, 0.85, 0.86, 0.87, 0.885, 0.91]
SEEDS = list(range(10, 30))

rows = []

for seed in SEEDS:
    train_df, test_df = train_test_split(
        df,
        test_size=0.35,
        random_state=seed,
        stratify=df["target"],
    )

    X_train = train_df[feature_cols]
    y_train = train_df["target"].astype(int)
    X_test = test_df[feature_cols]
    y_test = test_df["target"].astype(int)

    n_sig_train = max((y_train == 1).sum(), 1)
    n_bkg_train = max((y_train == 0).sum(), 1)
    train_weight = np.where(y_train == 1, 0.5 / n_sig_train, 0.5 / n_bkg_train)

    clf = GradientBoostingClassifier(
        n_estimators=250,
        learning_rate=0.04,
        max_depth=3,
        subsample=0.8,
        random_state=seed,
    )

    clf.fit(X_train, y_train, sample_weight=train_weight)

    test_df = test_df.copy()
    test_df["bdt_score"] = clf.predict_proba(X_test)[:, 1]

    total_counts = df.groupby("sample").size()
    test_counts = test_df.groupby("sample").size()
    test_df["weight_pb_scaled_to_full"] = test_df["weight_pb"] * test_df["sample"].map(
        lambda s: total_counts.loc[s] / test_counts.loc[s]
    )

    auc_unweighted = roc_auc_score(y_test, test_df["bdt_score"])
    auc_weighted = roc_auc_score(
        y_test,
        test_df["bdt_score"],
        sample_weight=test_df["weight_pb_scaled_to_full"],
    )

    for thr in THRESHOLDS:
        sel = test_df["bdt_score"] >= thr

        s_pb = test_df.loc[sel & (test_df["target"] == 1), "weight_pb_scaled_to_full"].sum()
        b_pb = test_df.loc[sel & (test_df["target"] == 0), "weight_pb_scaled_to_full"].sum()

        s_ev = s_pb * LUMI_PB
        b_ev = b_pb * LUMI_PB

        rows.append({
            "seed": seed,
            "threshold": thr,
            "unweighted_auc": auc_unweighted,
            "physics_weighted_auc": auc_weighted,
            "signal_events_450fb": s_ev,
            "background_events_450fb": b_ev,
            "S_over_B": s_ev / b_ev if b_ev > 0 else np.nan,
            "S_over_sqrtB": s_ev / np.sqrt(b_ev) if b_ev > 0 else np.nan,
            "S_over_10pctB": s_ev / (0.10 * b_ev) if b_ev > 0 else np.nan,
            "n_signal_test_rows": int((sel & (test_df["target"] == 1)).sum()),
            "n_background_test_rows": int((sel & (test_df["target"] == 0)).sum()),
        })

detail = pd.DataFrame(rows)

summary = (
    detail.groupby("threshold", as_index=False)
    .agg(
        n_seeds=("seed", "count"),
        auc_mean=("physics_weighted_auc", "mean"),
        auc_std=("physics_weighted_auc", "std"),
        S_over_sqrtB_mean=("S_over_sqrtB", "mean"),
        S_over_sqrtB_std=("S_over_sqrtB", "std"),
        S_over_B_mean=("S_over_B", "mean"),
        S_over_B_std=("S_over_B", "std"),
        median_background_test_rows=("n_background_test_rows", "median"),
        min_background_test_rows=("n_background_test_rows", "min"),
        median_signal_test_rows=("n_signal_test_rows", "median"),
        min_signal_test_rows=("n_signal_test_rows", "min"),
    )
)

detail.to_csv(OUTDIR / "bdt_multiseed_threshold_detail.csv", index=False)
summary.to_csv(OUTDIR / "bdt_multiseed_threshold_summary.csv", index=False)

(OUTDIR / "bdt_multiseed_threshold_detail.md").write_text(detail.to_markdown(index=False) + "\n")
(OUTDIR / "bdt_multiseed_threshold_summary.md").write_text(summary.to_markdown(index=False) + "\n")

print("\n=== Multi-seed BDT stability summary ===")
print(summary.to_string(index=False))
print(f"\nWrote outputs to: {OUTDIR}")
