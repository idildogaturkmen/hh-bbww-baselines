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

OUTDIR = REPO / "outputs/tables/hh4b_bdt_v2_full_categorization_lumi_2026_07_10"
OUTDIR.mkdir(parents=True, exist_ok=True)

LUMIS_FB = [138.0, 350.0, 450.0, 4000.0]
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


def safe_auc(y, score, weight=None):
    if len(np.unique(y)) < 2:
        return np.nan
    return roc_auc_score(y, score, sample_weight=weight)


def train_background_specific_bdt(train_df, test_df, features, mask_col, seed):
    """
    Train one BDT specifically against one background family.
    mask_col is usually is_qcd or is_top.
    """
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

    return {
        "all_scores": all_scores,
        "task_auc_unweighted": safe_auc(y_test, task_scores),
        "task_auc_weighted": safe_auc(
            y_test,
            task_scores,
            task_test["weight_pb_scaled_to_full"].to_numpy(),
        ),
    }


def assign_categories(df):
    """
    Mutually exclusive HHH-inspired categories in the BDT_QCD vs BDT_top plane.
    Ordered from highest-purity to lower-purity.
    """
    q = df["bdt_qcd_score"]
    t = df["bdt_top_score"]

    cat = np.full(len(df), "outside", dtype=object)

    cat[(q >= 0.875) & (t >= 0.550)] = "CAT0_very_high_QCD_high_top"
    cat[(cat == "outside") & (q >= 0.825) & (t >= 0.500)] = "CAT1_high_QCD_medium_top"
    cat[(cat == "outside") & (q >= 0.775) & (t >= 0.500)] = "CAT2_nominal_loose_mid"
    cat[(cat == "outside") & (q >= 0.725) & (t >= 0.500)] = "CAT3_lower_QCD_medium_top"
    cat[(cat == "outside") & (q >= 0.700) & (t >= 0.450)] = "CAT4_loose_low_purity"

    return cat


def asimov_z(S, B):
    if S <= 0 or B <= 0:
        return 0.0
    return float(np.sqrt(2.0 * ((S + B) * np.log(1.0 + S / B) - S)))


def summarize_for_lumi(test_df, seed, lumi_fb):
    lumi_pb = lumi_fb * 1000.0

    cats = [
        "CAT0_very_high_QCD_high_top",
        "CAT1_high_QCD_medium_top",
        "CAT2_nominal_loose_mid",
        "CAT3_lower_QCD_medium_top",
        "CAT4_loose_low_purity",
    ]

    rows = []

    for c in cats:
        d = test_df[test_df["category"].eq(c)]
        sig = d[d["is_signal"]]
        bkg = d[~d["is_signal"]]

        S = sig["weight_pb_scaled_to_full"].sum() * lumi_pb
        B = bkg["weight_pb_scaled_to_full"].sum() * lumi_pb

        rows.append({
            "seed": seed,
            "luminosity_fb": lumi_fb,
            "category": c,
            "signal_events": S,
            "background_events": B,
            "S_over_B": S / B if B > 0 else np.nan,
            "S_over_sqrtB": S / np.sqrt(B) if B > 0 else np.nan,
            "asimov_Z_stat_only": asimov_z(S, B),
            "S_over_sqrt_B_plus_10pctB_syst": S / np.sqrt(B + (0.10 * B)**2) if B > 0 else np.nan,
            "n_signal_rows": len(sig),
            "n_background_rows": len(bkg),
            "neff_background": neff(bkg["weight_pb_scaled_to_full"]),
        })

    cat_df = pd.DataFrame(rows)

    valid = cat_df["background_events"] > 0
    combined_s_over_sqrtb = np.sqrt(np.sum(
        cat_df.loc[valid, "signal_events"]**2 / cat_df.loc[valid, "background_events"]
    ))
    combined_asimov = np.sqrt(np.sum(cat_df.loc[valid, "asimov_Z_stat_only"]**2))
    combined_syst10 = np.sqrt(np.sum(cat_df.loc[valid, "S_over_sqrt_B_plus_10pctB_syst"]**2))

    selected = test_df[test_df["category"].ne("outside")]
    sig = selected[selected["is_signal"]]
    bkg = selected[~selected["is_signal"]]

    S = sig["weight_pb_scaled_to_full"].sum() * lumi_pb
    B = bkg["weight_pb_scaled_to_full"].sum() * lumi_pb

    inclusive = {
        "seed": seed,
        "luminosity_fb": lumi_fb,
        "category": "INCLUSIVE_union_of_categories",
        "signal_events": S,
        "background_events": B,
        "S_over_B": S / B if B > 0 else np.nan,
        "S_over_sqrtB": S / np.sqrt(B) if B > 0 else np.nan,
        "asimov_Z_stat_only": asimov_z(S, B),
        "S_over_sqrt_B_plus_10pctB_syst": S / np.sqrt(B + (0.10 * B)**2) if B > 0 else np.nan,
        "n_signal_rows": len(sig),
        "n_background_rows": len(bkg),
        "neff_background": neff(bkg["weight_pb_scaled_to_full"]),
        "combined_category_S_over_sqrtB": combined_s_over_sqrtb,
        "combined_category_asimov_Z": combined_asimov,
        "combined_category_syst10_Z": combined_syst10,
    }

    cat_df["combined_category_S_over_sqrtB"] = np.nan
    cat_df["combined_category_asimov_Z"] = np.nan
    cat_df["combined_category_syst10_Z"] = np.nan

    return pd.concat([cat_df, pd.DataFrame([inclusive])], ignore_index=True)


def summarize_composition(test_df, seed, lumi_fb):
    lumi_pb = lumi_fb * 1000.0
    rows = []

    for category in sorted([c for c in test_df["category"].unique() if c != "outside"]):
        dcat = test_df[test_df["category"].eq(category)]
        total_sig = dcat[dcat["is_signal"]]["weight_pb_scaled_to_full"].sum() * lumi_pb
        total_bkg = dcat[~dcat["is_signal"]]["weight_pb_scaled_to_full"].sum() * lumi_pb

        for sample, ds in dcat.groupby("analysis_sample"):
            group = "signal" if ds["is_signal"].iloc[0] else "background"
            expected = ds["weight_pb_scaled_to_full"].sum() * lumi_pb
            denom = total_sig if group == "signal" else total_bkg

            rows.append({
                "seed": seed,
                "luminosity_fb": lumi_fb,
                "category": category,
                "sample": sample,
                "group": group,
                "expected_events": expected,
                "rows": len(ds),
                "fraction_within_group": expected / denom if denom > 0 else np.nan,
            })

    return pd.DataFrame(rows)


def run():
    df, features = loader.load_all()

    all_cat_rows = []
    all_comp_rows = []
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

        qcd = train_background_specific_bdt(train_df, test_df, features, "is_qcd", seed + 1001)
        top = train_background_specific_bdt(train_df, test_df, features, "is_top", seed + 2001)

        test_df["bdt_qcd_score"] = qcd["all_scores"]
        test_df["bdt_top_score"] = top["all_scores"]
        test_df["category"] = assign_categories(test_df)

        auc_rows.append({
            "seed": seed,
            "qcd_bdt_weighted_auc": qcd["task_auc_weighted"],
            "top_bdt_weighted_auc": top["task_auc_weighted"],
            "qcd_bdt_unweighted_auc": qcd["task_auc_unweighted"],
            "top_bdt_unweighted_auc": top["task_auc_unweighted"],
        })

        for lumi_fb in LUMIS_FB:
            all_cat_rows.append(summarize_for_lumi(test_df, seed, lumi_fb))
            all_comp_rows.append(summarize_composition(test_df, seed, lumi_fb))

    cat = pd.concat(all_cat_rows, ignore_index=True)
    comp = pd.concat(all_comp_rows, ignore_index=True)
    auc = pd.DataFrame(auc_rows)

    summary = (
        cat.groupby(["luminosity_fb", "category"], as_index=False)
        .agg(
            S_mean=("signal_events", "mean"),
            B_mean=("background_events", "mean"),
            S_over_B_mean=("S_over_B", "mean"),
            S_over_sqrtB_mean=("S_over_sqrtB", "mean"),
            S_over_sqrtB_std=("S_over_sqrtB", "std"),
            asimov_Z_mean=("asimov_Z_stat_only", "mean"),
            syst10_Z_mean=("S_over_sqrt_B_plus_10pctB_syst", "mean"),
            median_signal_rows=("n_signal_rows", "median"),
            median_background_rows=("n_background_rows", "median"),
            median_neff_background=("neff_background", "median"),
            combined_category_S_over_sqrtB_mean=("combined_category_S_over_sqrtB", "mean"),
            combined_category_S_over_sqrtB_std=("combined_category_S_over_sqrtB", "std"),
            combined_category_asimov_Z_mean=("combined_category_asimov_Z", "mean"),
            combined_category_syst10_Z_mean=("combined_category_syst10_Z", "mean"),
        )
        .sort_values(["luminosity_fb", "category"])
    )

    auc_summary = pd.DataFrame([{
        "qcd_bdt_weighted_auc_mean": auc["qcd_bdt_weighted_auc"].mean(),
        "qcd_bdt_weighted_auc_std": auc["qcd_bdt_weighted_auc"].std(),
        "top_bdt_weighted_auc_mean": auc["top_bdt_weighted_auc"].mean(),
        "top_bdt_weighted_auc_std": auc["top_bdt_weighted_auc"].std(),
    }])

    comp_summary = (
        comp.groupby(["luminosity_fb", "category", "sample", "group"], as_index=False)
        .agg(
            expected_events_mean=("expected_events", "mean"),
            rows_median=("rows", "median"),
            fraction_within_group_mean=("fraction_within_group", "mean"),
        )
        .sort_values(["luminosity_fb", "category", "group", "expected_events_mean"], ascending=[True, True, True, False])
    )

    cat.to_csv(OUTDIR / "full_categorization_by_seed_lumi.csv", index=False)
    summary.to_csv(OUTDIR / "full_categorization_summary_by_lumi.csv", index=False)
    comp.to_csv(OUTDIR / "full_categorization_composition_by_seed_lumi.csv", index=False)
    comp_summary.to_csv(OUTDIR / "full_categorization_composition_summary_by_lumi.csv", index=False)
    auc.to_csv(OUTDIR / "full_categorization_auc_by_seed.csv", index=False)
    auc_summary.to_csv(OUTDIR / "full_categorization_auc_summary.csv", index=False)

    (OUTDIR / "full_categorization_summary_by_lumi.md").write_text(summary.to_markdown(index=False) + "\n")
    (OUTDIR / "full_categorization_composition_summary_by_lumi.md").write_text(comp_summary.to_markdown(index=False) + "\n")
    (OUTDIR / "full_categorization_auc_summary.md").write_text(auc_summary.to_markdown(index=False) + "\n")

    readme = """# HH4b full BDT-v2 categorization luminosity study

Date: 2026-07-10

This study retrains the two background-specific BDT-v2 classifiers seed-by-seed:
- BDT_QCD: signal vs QCD-enriched backgrounds
- BDT_top: signal vs ttbar/top background

Events are assigned to mutually exclusive HHH-inspired categories in the BDT_QCD vs BDT_top plane.
The category yields and approximate significances are recomputed at 138/fb, 350/fb, 450/fb, and 4000/fb.

Metrics include:
- S/B
- S/sqrt(B)
- Asimov stat-only Z
- approximate S/sqrt(B + (0.1B)^2) orientation metric
- combined-category sqrt(sum_i S_i^2/B_i)
- category composition by sample
"""
    (OUTDIR / "README.md").write_text(readme)

    print("\n=== AUC summary ===")
    print(auc_summary.to_string(index=False))

    print("\n=== Full categorization summary by luminosity ===")
    print(summary.to_string(index=False))

    print("\nWrote:", OUTDIR)


if __name__ == "__main__":
    run()
