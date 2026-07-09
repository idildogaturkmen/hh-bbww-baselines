'''
Multi-BDT study for SM-normalized HH4b analysis.
This script trains two dedicated BDTs:
- BDT_QCD: trained to separate SM-normalized HH signal from QCD bbbb HT-sliced backgrounds.
- BDT_top: trained to separate SM-normalized HH signal from ttbar/top-like backgrounds.
The goal is to test a CMS-inspired two-discriminant category strategy rather than relying on a single global BDT. The outputs include:
- AUC for each BDT.
- A 2D rectangular threshold scan.
- Stable threshold scan requiring minimum background statistics.
- Fixed non-overlapping category yields and background composition.
Signal normalization follows SM HH cross sections times BR(H->bb)^2.
Backgrounds use generator cross-section weights.
This script is intended to be run in the HH4b analysis environment with the appropriate dependencies installed and the HH4b_REPO environment variable set to the root of the repository.

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
BASE_BDT_SCRIPT = REPO / "scripts/delphes/train_sm_normalized_hh4b_bdt.py"

OUTDIR = REPO / "outputs/tables/sm_normalized_hh4b_two_bdt_multiseed_2026_07_09"
OUTDIR.mkdir(parents=True, exist_ok=True)

LUMI_PB = 450.0 * 1000.0
TEST_SIZE = 0.35
SEEDS = list(range(10, 30))

spec = importlib.util.spec_from_file_location("bdtmod", BASE_BDT_SCRIPT)
bdtmod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(bdtmod)

df, feature_cols = bdtmod.load_all()

df = df.copy()
df["is_signal"] = df["group"].eq("signal")
df["is_qcd"] = df["sample"].str.startswith("qcd_")
df["is_top"] = df["sample"].str.contains("ttbar|ttbb|tt_", case=False, regex=True)


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
    return clf, all_scores, auc, wauc


# Candidate thresholds motivated by the first two-BDT run.
fixed_pairs = [
    (0.875, 0.650),
    (0.875, 0.600),
    (0.875, 0.700),
    (0.850, 0.600),
    (0.850, 0.650),
    (0.850, 0.750),
    (0.825, 0.650),
    (0.800, 0.750),
]

scan_qcd = np.round(np.arange(0.800, 0.901, 0.025), 3)
scan_top = np.round(np.arange(0.500, 0.751, 0.025), 3)

rows = []
best_rows = []
auc_rows = []

for seed in SEEDS:
    train_df, test_df = train_test_split(
        df,
        test_size=TEST_SIZE,
        random_state=seed,
        stratify=df["target"],
    )

    total_counts = df.groupby("sample").size()
    test_counts = test_df.groupby("sample").size()

    test_df = test_df.copy()
    test_df["weight_pb_scaled_to_full"] = test_df["weight_pb"] * test_df["sample"].map(
        lambda s: total_counts.loc[s] / test_counts.loc[s]
    )

    _, qcd_scores, qcd_auc, qcd_wauc = train_task(train_df, test_df, "is_qcd", seed)
    _, top_scores, top_auc, top_wauc = train_task(train_df, test_df, "is_top", seed)

    test_df["bdt_qcd_score"] = qcd_scores
    test_df["bdt_top_score"] = top_scores

    auc_rows.append({
        "seed": seed,
        "qcd_auc": qcd_auc,
        "qcd_weighted_auc": qcd_wauc,
        "top_auc": top_auc,
        "top_weighted_auc": top_wauc,
    })

    # Fixed selected pairs.
    for tq, tt in fixed_pairs:
        sel = (test_df["bdt_qcd_score"] >= tq) & (test_df["bdt_top_score"] >= tt)
        sig = test_df[sel & test_df["is_signal"]]
        bkg = test_df[sel & ~test_df["is_signal"]]

        s_pb = sig["weight_pb_scaled_to_full"].sum()
        b_pb = bkg["weight_pb_scaled_to_full"].sum()

        s_ev = s_pb * LUMI_PB
        b_ev = b_pb * LUMI_PB

        rows.append({
            "seed": seed,
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

    # Per-seed stable scan to find best stable rectangle.
    scan_rows = []
    for tq in scan_qcd:
        for tt in scan_top:
            sel = (test_df["bdt_qcd_score"] >= tq) & (test_df["bdt_top_score"] >= tt)
            sig = test_df[sel & test_df["is_signal"]]
            bkg = test_df[sel & ~test_df["is_signal"]]

            s_pb = sig["weight_pb_scaled_to_full"].sum()
            b_pb = bkg["weight_pb_scaled_to_full"].sum()

            s_ev = s_pb * LUMI_PB
            b_ev = b_pb * LUMI_PB

            scan_rows.append({
                "seed": seed,
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

    scan_df = pd.DataFrame(scan_rows)
    stable = scan_df[
        (scan_df["n_background_test_rows"] >= 20)
        & (scan_df["neff_background"] >= 5)
        & (scan_df["n_signal_test_rows"] >= 20)
    ].copy()

    if len(stable):
        best = stable.sort_values("S_over_sqrtB", ascending=False).iloc[0].to_dict()
        best_rows.append(best)

detail = pd.DataFrame(rows)
best = pd.DataFrame(best_rows)
auc = pd.DataFrame(auc_rows)

fixed_summary = (
    detail.groupby(["qcd_threshold", "top_threshold"], as_index=False)
    .agg(
        n_seeds=("seed", "count"),
        S_over_sqrtB_mean=("S_over_sqrtB", "mean"),
        S_over_sqrtB_std=("S_over_sqrtB", "std"),
        S_over_B_mean=("S_over_B", "mean"),
        S_over_B_std=("S_over_B", "std"),
        median_background_test_rows=("n_background_test_rows", "median"),
        min_background_test_rows=("n_background_test_rows", "min"),
        median_neff_background=("neff_background", "median"),
        min_neff_background=("neff_background", "min"),
        median_signal_test_rows=("n_signal_test_rows", "median"),
        min_signal_test_rows=("n_signal_test_rows", "min"),
    )
    .sort_values("S_over_sqrtB_mean", ascending=False)
)

best_summary = (
    best.agg(
        {
            "qcd_threshold": ["median", "min", "max"],
            "top_threshold": ["median", "min", "max"],
            "S_over_sqrtB": ["mean", "std", "median", "min", "max"],
            "S_over_B": ["mean", "std", "median"],
            "n_background_test_rows": ["median", "min"],
            "neff_background": ["median", "min"],
            "n_signal_test_rows": ["median", "min"],
        }
    )
    if len(best)
    else pd.DataFrame()
)

auc_summary = auc.agg({
    "qcd_auc": ["mean", "std"],
    "qcd_weighted_auc": ["mean", "std"],
    "top_auc": ["mean", "std"],
    "top_weighted_auc": ["mean", "std"],
})

detail.to_csv(OUTDIR / "two_bdt_multiseed_fixed_pairs_detail.csv", index=False)
fixed_summary.to_csv(OUTDIR / "two_bdt_multiseed_fixed_pairs_summary.csv", index=False)
best.to_csv(OUTDIR / "two_bdt_multiseed_best_stable_per_seed.csv", index=False)
auc.to_csv(OUTDIR / "two_bdt_multiseed_auc_detail.csv", index=False)

(OUTDIR / "two_bdt_multiseed_fixed_pairs_summary.md").write_text(fixed_summary.to_markdown(index=False) + "\n")
(OUTDIR / "two_bdt_multiseed_best_stable_per_seed.md").write_text(best.to_markdown(index=False) + "\n")
(OUTDIR / "two_bdt_multiseed_best_stable_summary.md").write_text(best_summary.to_markdown() + "\n")
(OUTDIR / "two_bdt_multiseed_auc_summary.md").write_text(auc_summary.to_markdown() + "\n")

print("\n=== Two-BDT multiseed AUC summary ===")
print(auc_summary.to_string())

print("\n=== Fixed pair multiseed summary ===")
print(fixed_summary.head(15).to_string(index=False))

print("\n=== Best stable rectangle per seed summary ===")
print(best_summary.to_string())

print(f"\nWrote outputs to: {OUTDIR}")
