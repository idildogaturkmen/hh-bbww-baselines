#!/usr/bin/env python3

"""
BDT-v3 qcdplus topology-only categorized audit.

Purpose:
- Reproduce the frozen BDT-v3 qcdplus training and test split.
- Save the scored test dataframe with bdt_qcd_score and bdt_top_score.
- Define mutually exclusive HHH-style categories in the two-BDT score plane.
- Produce category yields, background composition, and mass-summary tables.

This script does not replace the frozen baseline. It is the next analysis layer.
"""

import importlib.util
import os
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

REPO = Path(os.environ["HH4B_REPO"])
STORE = Path(os.environ["HH4B_STORE"])

V3_SCRIPT = REPO / "scripts/delphes/train_sm_normalized_hh4b_two_bdt_event_features_safe_v3_qcdplus_topology_only.py"

OUTDIR = REPO / "outputs/tables/bdt_v3_qcdplus_topology_only_categories_2026_07_13"
OUTDIR.mkdir(parents=True, exist_ok=True)

spec = importlib.util.spec_from_file_location("bdt_v3_qcdplus", V3_SCRIPT)
v3 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(v3)

base = v3.base
LUMI_PB = base.LUMI_PB
RANDOM_STATE = base.RANDOM_STATE
TEST_SIZE = base.TEST_SIZE


def neff(weights):
    weights = np.asarray(weights, dtype=float)
    if len(weights) == 0 or np.sum(weights * weights) <= 0:
        return 0.0
    return float((np.sum(weights) ** 2) / np.sum(weights * weights))


def assign_category(df):
    q = df["bdt_qcd_score"]
    t = df["bdt_top_score"]

    cat = np.full(len(df), "UNSELECTED", dtype=object)

    cat[(q >= 0.850) & (t >= 0.850)] = "CAT0_high_purity_diagnostic"
    cat[(q >= 0.850) & (t >= 0.500) & (t < 0.850)] = "CAT1_tight"
    cat[(q >= 0.800) & (q < 0.850) & (t >= 0.500)] = "CAT2_medium_tight"
    cat[(q >= 0.700) & (q < 0.800) & (t >= 0.500)] = "CAT3_medium"
    cat[(q >= 0.500) & (q < 0.700) & (t >= 0.500)] = "CAT4_loose"

    return cat


def weighted_mean(x, w):
    x = np.asarray(x, dtype=float)
    w = np.asarray(w, dtype=float)
    if len(x) == 0 or np.sum(w) <= 0:
        return np.nan
    return float(np.sum(x * w) / np.sum(w))


def main():
    df, feature_cols = v3.load_all()

    train_df, test_df = train_test_split(
        df,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=df["target"],
    )

    total_counts = df.groupby("analysis_sample").size()
    test_counts = test_df.groupby("analysis_sample").size()

    test_df = test_df.copy()
    test_df["weight_pb_scaled_to_full"] = test_df["weight_pb"] * test_df["analysis_sample"].map(
        lambda s: total_counts.loc[s] / test_counts.loc[s]
    )
    test_df["weight_events_450fb"] = test_df["weight_pb_scaled_to_full"] * LUMI_PB

    qcd_scores, qcd_auc, qcd_wauc, qcd_imp, qcd_ntrain, qcd_ntest = base.train_task(
        train_df, test_df, feature_cols, "is_qcd", "qcd"
    )
    top_scores, top_auc, top_wauc, top_imp, top_ntrain, top_ntest = base.train_task(
        train_df, test_df, feature_cols, "is_top", "top"
    )

    test_df["bdt_qcd_score"] = qcd_scores
    test_df["bdt_top_score"] = top_scores
    test_df["category"] = assign_category(test_df)

    scored_path = OUTDIR / "scored_test_events_bdt_v3_qcdplus_topology_only.parquet"
    test_df.to_parquet(scored_path, index=False)

    auc_summary = pd.DataFrame([
        {
            "classifier": "BDT_QCD_event_features_safe_v3_qcdplus_topology_only",
            "negative_class": "QCD bbbb HT slices",
            "unweighted_auc": qcd_auc,
            "physics_weighted_auc": qcd_wauc,
            "n_train_task": qcd_ntrain,
            "n_test_task": qcd_ntest,
        },
        {
            "classifier": "BDT_top_event_features_safe_v3_qcdplus_topology_only",
            "negative_class": "ttbar / top-like backgrounds",
            "unweighted_auc": top_auc,
            "physics_weighted_auc": top_wauc,
            "n_train_task": top_ntrain,
            "n_test_task": top_ntest,
        },
    ])
    auc_summary.to_csv(OUTDIR / "category_audit_auc_summary.csv", index=False)
    (OUTDIR / "category_audit_auc_summary.md").write_text(
        auc_summary.to_markdown(index=False) + "\n"
    )

    selected_cats = [
        "CAT0_high_purity_diagnostic",
        "CAT1_tight",
        "CAT2_medium_tight",
        "CAT3_medium",
        "CAT4_loose",
    ]

    yield_rows = []
    for cat in selected_cats:
        sub = test_df[test_df["category"].eq(cat)]
        sig = sub[sub["is_signal"]]
        bkg = sub[~sub["is_signal"]]

        s = sig["weight_events_450fb"].sum()
        b = bkg["weight_events_450fb"].sum()

        yield_rows.append({
            "category": cat,
            "n_signal_test_rows": len(sig),
            "n_background_test_rows": len(bkg),
            "signal_events_450fb": s,
            "background_events_450fb": b,
            "S_over_B": s / b if b > 0 else np.nan,
            "S_over_sqrtB": s / np.sqrt(b) if b > 0 else np.nan,
            "S_over_10pctB": s / (0.10 * b) if b > 0 else np.nan,
            "neff_signal": neff(sig["weight_events_450fb"]),
            "neff_background": neff(bkg["weight_events_450fb"]),
        })

    yields = pd.DataFrame(yield_rows)
    yields.to_csv(OUTDIR / "category_yields.csv", index=False)
    (OUTDIR / "category_yields.md").write_text(yields.to_markdown(index=False) + "\n")

    comp = (
        test_df[test_df["category"].isin(selected_cats) & ~test_df["is_signal"]]
        .groupby(["category", "analysis_sample"], as_index=False)
        .agg(
            n_test_rows=("event", "size"),
            background_events_450fb=("weight_events_450fb", "sum"),
            neff_sample=("weight_events_450fb", neff),
            mean_qcd_score=("bdt_qcd_score", "mean"),
            mean_top_score=("bdt_top_score", "mean"),
        )
    )

    total_bkg_by_cat = comp.groupby("category")["background_events_450fb"].transform("sum")
    comp["fraction_of_category_background"] = comp["background_events_450fb"] / total_bkg_by_cat
    comp = comp.sort_values(["category", "background_events_450fb"], ascending=[True, False])

    comp.to_csv(OUTDIR / "category_background_composition.csv", index=False)
    (OUTDIR / "category_background_composition.md").write_text(
        comp.to_markdown(index=False) + "\n"
    )

    mass_vars = ["mbb1", "mbb2", "avg_mbb", "delta_mbb", "mhh", "r_hh", "h_delta_r"]
    mass_rows = []

    for cat in selected_cats:
        for label, mask in [
            ("signal", test_df["is_signal"]),
            ("background", ~test_df["is_signal"]),
        ]:
            sub = test_df[test_df["category"].eq(cat) & mask].copy()
            if len(sub) == 0:
                continue

            row = {
                "category": cat,
                "group": label,
                "n_test_rows": len(sub),
                "yield_450fb": sub["weight_events_450fb"].sum(),
                "neff": neff(sub["weight_events_450fb"]),
            }

            for var in mass_vars:
                row[f"{var}_median"] = float(sub[var].median())
                row[f"{var}_weighted_mean"] = weighted_mean(sub[var], sub["weight_events_450fb"])
                row[f"{var}_p16"] = float(sub[var].quantile(0.16))
                row[f"{var}_p84"] = float(sub[var].quantile(0.84))

            mass_rows.append(row)

    mass = pd.DataFrame(mass_rows)
    mass.to_csv(OUTDIR / "category_mass_summary.csv", index=False)
    (OUTDIR / "category_mass_summary.md").write_text(mass.to_markdown(index=False) + "\n")

    category_definitions = pd.DataFrame([
        {
            "category": "CAT0_high_purity_diagnostic",
            "definition": "bdt_qcd_score >= 0.850 and bdt_top_score >= 0.850",
            "intended_use": "Diagnostic only unless effective background statistics are adequate",
        },
        {
            "category": "CAT1_tight",
            "definition": "bdt_qcd_score >= 0.850 and 0.500 <= bdt_top_score < 0.850",
            "intended_use": "Tight high-score category",
        },
        {
            "category": "CAT2_medium_tight",
            "definition": "0.800 <= bdt_qcd_score < 0.850 and bdt_top_score >= 0.500",
            "intended_use": "Medium-tight category around best supported qcd threshold",
        },
        {
            "category": "CAT3_medium",
            "definition": "0.700 <= bdt_qcd_score < 0.800 and bdt_top_score >= 0.500",
            "intended_use": "Medium category",
        },
        {
            "category": "CAT4_loose",
            "definition": "0.500 <= bdt_qcd_score < 0.700 and bdt_top_score >= 0.500",
            "intended_use": "Loose/control-like selected category",
        },
    ])

    category_definitions.to_csv(OUTDIR / "category_definitions.csv", index=False)
    (OUTDIR / "category_definitions.md").write_text(
        category_definitions.to_markdown(index=False) + "\n"
    )

    readme = OUTDIR / "README.md"
    readme.write_text(
        "# BDT-v3 qcdplus topology-only categorized audit\n\n"
        "This directory contains HHH-style mutually exclusive categories built from the two topology-only BDT-v3 qcdplus scores.\n\n"
        "The script reproduces the frozen BDT-v3 qcdplus train/test split and saves the scored test dataframe.\n\n"
        "Files:\n"
        "- scored_test_events_bdt_v3_qcdplus_topology_only.parquet\n"
        "- category_yields.csv and .md\n"
        "- category_background_composition.csv and .md\n"
        "- category_mass_summary.csv and .md\n"
        "- category_definitions.csv and .md\n"
        "- category_audit_auc_summary.csv and .md\n\n"
        "CAT0 is diagnostic unless effective background statistics are sufficient.\n"
    )

    print("\n=== AUC summary ===")
    print(auc_summary.to_string(index=False))

    print("\n=== Category yields ===")
    print(yields.to_string(index=False))

    print("\n=== Top background components by category ===")
    print(comp.groupby("category").head(5).to_string(index=False))

    print("\nWrote:", OUTDIR)
    print("Scored test events:", scored_path)


if __name__ == "__main__":
    main()
