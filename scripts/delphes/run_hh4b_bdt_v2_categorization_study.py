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

OUTDIR = REPO / "outputs/tables/hh4b_bdt_v2_categorization_2026_07_10"
OUTDIR.mkdir(parents=True, exist_ok=True)

LUMI_PB = 450_000.0
SEEDS = list(range(20))

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
    den = np.sum(w * w)
    return float(w.sum() * w.sum() / den) if den > 0 else 0.0


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
    clf.fit(task_train[features], y_train, sample_weight=class_balanced_weights(y_train))

    task_scores = clf.predict_proba(task_test[features])[:, 1]
    all_scores = clf.predict_proba(test_df[features])[:, 1]

    auc = roc_auc_score(y_test, task_scores)
    wauc = roc_auc_score(
        y_test,
        task_scores,
        sample_weight=task_test["weight_pb_scaled_to_full"],
    )

    return all_scores, auc, wauc


def assign_categories(df):
    q = df["qcd_score"]
    t = df["top_score"]

    cat = np.full(len(df), "outside", dtype=object)

    # Mutually exclusive, ordered from purest to loosest.
    cat[(q >= 0.875) & (t >= 0.550)] = "CAT0_very_high_QCD_high_top"
    cat[(cat == "outside") & (q >= 0.825) & (t >= 0.500)] = "CAT1_high_QCD_medium_top"
    cat[(cat == "outside") & (q >= 0.775) & (t >= 0.500)] = "CAT2_nominal_loose_mid"
    cat[(cat == "outside") & (q >= 0.725) & (t >= 0.500)] = "CAT3_lower_QCD_medium_top"
    cat[(cat == "outside") & (q >= 0.700) & (t >= 0.450)] = "CAT4_loose_low_purity"

    return cat


def summarize_categories(test_df, seed):
    rows = []
    cats = [
        "CAT0_very_high_QCD_high_top",
        "CAT1_high_QCD_medium_top",
        "CAT2_nominal_loose_mid",
        "CAT3_lower_QCD_medium_top",
        "CAT4_loose_low_purity",
    ]

    for c in cats:
        d = test_df[test_df["category"].eq(c)]
        sig = d[d["is_signal"]]
        bkg = d[~d["is_signal"]]

        S = sig["weight_pb_scaled_to_full"].sum() * LUMI_PB
        B = bkg["weight_pb_scaled_to_full"].sum() * LUMI_PB

        rows.append({
            "seed": seed,
            "category": c,
            "signal_events_450fb": S,
            "background_events_450fb": B,
            "S_over_B": S / B if B > 0 else np.nan,
            "S_over_sqrtB": S / np.sqrt(B) if B > 0 else np.nan,
            "n_signal_rows": len(sig),
            "n_background_rows": len(bkg),
            "neff_background": neff(bkg["weight_pb_scaled_to_full"]),
        })

    # Combined category significance approximation.
    rows_df = pd.DataFrame(rows)
    valid = rows_df["background_events_450fb"] > 0
    z_comb = np.sqrt(np.sum(
        rows_df.loc[valid, "signal_events_450fb"]**2
        / rows_df.loc[valid, "background_events_450fb"]
    ))

    selected = test_df[test_df["category"].ne("outside")]
    sig = selected[selected["is_signal"]]
    bkg = selected[~selected["is_signal"]]

    S = sig["weight_pb_scaled_to_full"].sum() * LUMI_PB
    B = bkg["weight_pb_scaled_to_full"].sum() * LUMI_PB

    inclusive_row = {
        "seed": seed,
        "category": "INCLUSIVE_union_of_categories",
        "signal_events_450fb": S,
        "background_events_450fb": B,
        "S_over_B": S / B if B > 0 else np.nan,
        "S_over_sqrtB": S / np.sqrt(B) if B > 0 else np.nan,
        "n_signal_rows": len(sig),
        "n_background_rows": len(bkg),
        "neff_background": neff(bkg["weight_pb_scaled_to_full"]),
        "combined_category_Z": z_comb,
    }

    rows_df["combined_category_Z"] = np.nan
    rows_df = pd.concat([rows_df, pd.DataFrame([inclusive_row])], ignore_index=True)

    return rows_df


def run():
    df, features = loader.load_all()

    all_rows = []
    auc_rows = []

    for seed in SEEDS:
        print(f"=== seed {seed} ===")

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

        qcd_scores, qcd_auc, qcd_wauc = train_task(train_df, test_df, features, "is_qcd", seed + 1001)
        top_scores, top_auc, top_wauc = train_task(train_df, test_df, features, "is_top", seed + 2001)

        test_df["qcd_score"] = qcd_scores
        test_df["top_score"] = top_scores
        test_df["category"] = assign_categories(test_df)

        auc_rows.append({
            "seed": seed,
            "qcd_weighted_auc": qcd_wauc,
            "top_weighted_auc": top_wauc,
        })

        all_rows.append(summarize_categories(test_df, seed))

    cat = pd.concat(all_rows, ignore_index=True)
    auc = pd.DataFrame(auc_rows)

    category_summary = (
        cat.groupby("category", as_index=False)
        .agg(
            S_mean=("signal_events_450fb", "mean"),
            B_mean=("background_events_450fb", "mean"),
            S_over_B_mean=("S_over_B", "mean"),
            S_over_sqrtB_mean=("S_over_sqrtB", "mean"),
            S_over_sqrtB_std=("S_over_sqrtB", "std"),
            median_signal_rows=("n_signal_rows", "median"),
            median_background_rows=("n_background_rows", "median"),
            median_neff_background=("neff_background", "median"),
            combined_category_Z_mean=("combined_category_Z", "mean"),
            combined_category_Z_std=("combined_category_Z", "std"),
        )
        .sort_values("category")
    )

    auc_summary = pd.DataFrame([{
        "qcd_weighted_auc_mean": auc["qcd_weighted_auc"].mean(),
        "qcd_weighted_auc_std": auc["qcd_weighted_auc"].std(),
        "top_weighted_auc_mean": auc["top_weighted_auc"].mean(),
        "top_weighted_auc_std": auc["top_weighted_auc"].std(),
    }])

    cat.to_csv(OUTDIR / "categorization_by_seed.csv", index=False)
    category_summary.to_csv(OUTDIR / "categorization_summary.csv", index=False)
    auc.to_csv(OUTDIR / "categorization_auc_by_seed.csv", index=False)
    auc_summary.to_csv(OUTDIR / "categorization_auc_summary.csv", index=False)

    (OUTDIR / "categorization_summary.md").write_text(category_summary.to_markdown(index=False) + "\n")
    (OUTDIR / "categorization_auc_summary.md").write_text(auc_summary.to_markdown(index=False) + "\n")

    readme = """# HH4b BDT-v2 categorization study

This is a first HHH-inspired categorization study using the two BDT-v2 scores:
- BDT_QCD analogous to BDTnonres
- BDT_top analogous to BDTres / top-background suppressor

Categories are mutually exclusive rectangles in the BDT_QCD vs BDT_top plane.
The combined-category significance is approximated as sqrt(sum_i S_i^2 / B_i).
"""
    (OUTDIR / "README.md").write_text(readme)

    print("\n=== AUC summary ===")
    print(auc_summary.to_string(index=False))

    print("\n=== Category summary ===")
    print(category_summary.to_string(index=False))

    print("\nWrote:", OUTDIR)


if __name__ == "__main__":
    run()
