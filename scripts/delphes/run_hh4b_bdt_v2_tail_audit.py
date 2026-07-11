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

OUTDIR = REPO / "outputs/tables/hh4b_bdt_v2_tail_audit_2026_07_11"
OUTDIR.mkdir(parents=True, exist_ok=True)

SEEDS = list(range(20))
LUMI_450_PB = 450000.0

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


def asimov_z(S, B):
    if S <= 0 or B <= 0:
        return 0.0
    return float(np.sqrt(2.0 * ((S + B) * np.log(1.0 + S / B) - S)))


def train_background_specific_bdt(train_df, test_df, features, mask_col, seed):
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
    q = df["bdt_qcd_score"]
    t = df["bdt_top_score"]

    cat = np.full(len(df), "outside", dtype=object)

    cat[(q >= 0.875) & (t >= 0.550)] = "CAT0_very_high_QCD_high_top"
    cat[(cat == "outside") & (q >= 0.825) & (t >= 0.500)] = "CAT1_high_QCD_medium_top"
    cat[(cat == "outside") & (q >= 0.775) & (t >= 0.500)] = "CAT2_nominal_loose_mid"
    cat[(cat == "outside") & (q >= 0.725) & (t >= 0.500)] = "CAT3_lower_QCD_medium_top"
    cat[(cat == "outside") & (q >= 0.700) & (t >= 0.450)] = "CAT4_loose_low_purity"

    return cat


def make_event_table(test_df, features, seed):
    d = test_df.copy()
    d["seed"] = seed
    d["group"] = np.where(d["is_signal"].astype(bool), "signal", "background")
    d["tail_score"] = d["bdt_qcd_score"] * d["bdt_top_score"]
    d["expected_events_450fb"] = d["weight_pb_scaled_to_full"] * LUMI_450_PB

    base_cols = [
        "seed",
        "analysis_sample",
        "group",
        "target",
        "is_signal",
        "is_qcd",
        "is_top",
        "weight_pb",
        "weight_pb_scaled_to_full",
        "expected_events_450fb",
        "bdt_qcd_score",
        "bdt_top_score",
        "tail_score",
        "category",
    ]

    id_cols = [
        "event",
        "event_id",
        "source_root",
        "source_root_index",
        "root_file",
        "file_index",
    ]

    physics_cols = [
        "mbb1",
        "mbb2",
        "avg_mbb",
        "delta_mbb",
        "r_hh",
        "r_hh_125_125",
        "r_hh_125_120",
        "mhh",
        "hh_pt",
        "hh_eta",
        "h1_pt",
        "h1_eta",
        "h2_pt",
        "h2_eta",
        "h_delta_eta",
        "h_delta_phi",
        "h_delta_r",
        "h_pt_balance",
        "drbb1",
        "drbb2",
        "n_selected_jets",
        "n_selected_bjets",
        "n_extra_selected_jets",
        "n_extra_selected_bjets",
        "ht_selected_jets",
        "ht_selected_bjets",
        "ht_candidate_jets",
        "j1_pt",
        "j2_pt",
        "j3_pt",
        "j4_pt",
        "j1_eta",
        "j2_eta",
        "j3_eta",
        "j4_eta",
        "j1_mass",
        "j2_mass",
        "j3_mass",
        "j4_mass",
        "pt_sum4",
        "ht_over_mhh",
        "pt_asym_12",
        "pt_asym_34",
        "avg_drbb",
        "max_drbb",
        "min_drbb",
    ]

    cols = []
    for block in [base_cols, id_cols, physics_cols, list(features)]:
        for c in block:
            if c in d.columns and c not in cols:
                cols.append(c)

    return d[cols].copy()


def summarize_tail_regions(event_df):
    frac_list = [0.001, 0.002, 0.005, 0.010, 0.020, 0.050]

    summary_rows = []
    comp_rows = []

    for seed, dseed in event_df.groupby("seed"):
        bg_all = dseed[dseed["group"].eq("background")]
        if len(bg_all) == 0:
            continue

        for frac in frac_list:
            cut = bg_all["tail_score"].quantile(1.0 - frac)
            selected = dseed[dseed["tail_score"] >= cut].copy()

            sig = selected[selected["group"].eq("signal")]
            bkg = selected[selected["group"].eq("background")]

            S = sig["expected_events_450fb"].sum()
            B = bkg["expected_events_450fb"].sum()
            w = bkg["expected_events_450fb"].to_numpy(dtype=float)

            tail_region = f"top_{100.0 * frac:.1f}pct_background_tail"

            summary_rows.append({
                "seed": seed,
                "tail_region": tail_region,
                "background_tail_fraction": frac,
                "tail_score_cut": cut,
                "signal_rows": len(sig),
                "background_rows": len(bkg),
                "S_450fb": S,
                "B_450fb": B,
                "S_over_B": S / B if B > 0 else np.nan,
                "S_over_sqrtB": S / np.sqrt(B) if B > 0 else np.nan,
                "asimov_Z_stat_only": asimov_z(S, B),
                "background_neff": neff(w) if len(w) else 0.0,
                "largest_background_weight_fraction": float(w.max() / w.sum()) if len(w) and w.sum() > 0 else 0.0,
            })

            total_bkg = B
            for sample, ds in bkg.groupby("analysis_sample"):
                y = ds["expected_events_450fb"].sum()
                comp_rows.append({
                    "seed": seed,
                    "tail_region": tail_region,
                    "analysis_sample": sample,
                    "rows": len(ds),
                    "expected_events_450fb": y,
                    "fraction_of_tail_background": y / total_bkg if total_bkg > 0 else np.nan,
                })

    return pd.DataFrame(summary_rows), pd.DataFrame(comp_rows)


def summarize_tail_features(event_df, features):
    feature_candidates = [
        "mbb1",
        "mbb2",
        "avg_mbb",
        "delta_mbb",
        "r_hh",
        "mhh",
        "hh_pt",
        "h1_pt",
        "h2_pt",
        "h_delta_r",
        "h_pt_balance",
        "drbb1",
        "drbb2",
        "n_selected_jets",
        "n_selected_bjets",
        "n_extra_selected_jets",
        "n_extra_selected_bjets",
        "ht_selected_jets",
        "ht_selected_bjets",
        "ht_candidate_jets",
        "j1_pt",
        "j2_pt",
        "j3_pt",
        "j4_pt",
        "pt_sum4",
        "ht_over_mhh",
        "pt_asym_12",
        "pt_asym_34",
        "avg_drbb",
        "max_drbb",
        "min_drbb",
    ]

    for c in features:
        if c not in feature_candidates:
            feature_candidates.append(c)

    feature_cols = [
        c for c in feature_candidates
        if c in event_df.columns and pd.api.types.is_numeric_dtype(event_df[c])
    ]

    rows = []

    for seed, dseed in event_df.groupby("seed"):
        bg = dseed[dseed["group"].eq("background")]
        if len(bg) == 0:
            continue

        tight_cut = bg["tail_score"].quantile(0.999)

        regions = {
            "CAT0": dseed[dseed["category"].eq("CAT0_very_high_QCD_high_top")],
            "top_0.1pct_background_tail": dseed[dseed["tail_score"] >= tight_cut],
        }

        for region_name, dreg in regions.items():
            if len(dreg) == 0:
                continue

            for (group, sample), ds in dreg.groupby(["group", "analysis_sample"]):
                row = {
                    "seed": seed,
                    "region": region_name,
                    "group": group,
                    "analysis_sample": sample,
                    "rows": len(ds),
                    "expected_events_450fb": ds["expected_events_450fb"].sum(),
                    "median_bdt_qcd_score": ds["bdt_qcd_score"].median(),
                    "median_bdt_top_score": ds["bdt_top_score"].median(),
                    "median_tail_score": ds["tail_score"].median(),
                }
                for c in feature_cols:
                    row[f"median_{c}"] = ds[c].median()
                rows.append(row)

    return pd.DataFrame(rows)


def run():
    df, features = loader.load_all()

    all_event_rows = []
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

        test_df["weight_pb_scaled_to_full"] = (
            test_df["weight_pb"] * test_df["analysis_sample"].map(scale_to_full)
        )

        qcd = train_background_specific_bdt(train_df, test_df, features, "is_qcd", seed + 1001)
        top = train_background_specific_bdt(train_df, test_df, features, "is_top", seed + 2001)

        test_df["bdt_qcd_score"] = qcd["all_scores"]
        test_df["bdt_top_score"] = top["all_scores"]
        test_df["category"] = assign_categories(test_df)

        all_event_rows.append(make_event_table(test_df, features, seed))

        auc_rows.append({
            "seed": seed,
            "qcd_bdt_weighted_auc": qcd["task_auc_weighted"],
            "top_bdt_weighted_auc": top["task_auc_weighted"],
            "qcd_bdt_unweighted_auc": qcd["task_auc_unweighted"],
            "top_bdt_unweighted_auc": top["task_auc_unweighted"],
        })

    OUTDIR.mkdir(parents=True, exist_ok=True)

    event_df = pd.concat(all_event_rows, ignore_index=True)
    auc = pd.DataFrame(auc_rows)

    event_df.to_csv(OUTDIR / "event_level_bdt_scores_by_seed.csv", index=False)

    top_bkg = (
        event_df[event_df["group"].eq("background")]
        .sort_values(["seed", "tail_score"], ascending=[True, False])
        .groupby("seed", group_keys=False)
        .head(200)
    )
    top_sig = (
        event_df[event_df["group"].eq("signal")]
        .sort_values(["seed", "tail_score"], ascending=[True, False])
        .groupby("seed", group_keys=False)
        .head(200)
    )

    top_bkg.to_csv(OUTDIR / "top200_signal_like_background_events_by_seed.csv", index=False)
    top_sig.to_csv(OUTDIR / "top200_signal_like_signal_events_by_seed.csv", index=False)

    tail_by_seed, comp_by_seed = summarize_tail_regions(event_df)
    tail_by_seed.to_csv(OUTDIR / "tail_region_summary_by_seed.csv", index=False)
    comp_by_seed.to_csv(OUTDIR / "tail_background_composition_by_seed.csv", index=False)

    tail_summary = (
        tail_by_seed.groupby("tail_region", as_index=False)
        .agg(
            background_tail_fraction=("background_tail_fraction", "median"),
            tail_score_cut_median=("tail_score_cut", "median"),
            signal_rows_median=("signal_rows", "median"),
            background_rows_median=("background_rows", "median"),
            S_450fb_mean=("S_450fb", "mean"),
            B_450fb_mean=("B_450fb", "mean"),
            S_over_B_mean=("S_over_B", "mean"),
            S_over_sqrtB_mean=("S_over_sqrtB", "mean"),
            S_over_sqrtB_std=("S_over_sqrtB", "std"),
            asimov_Z_mean=("asimov_Z_stat_only", "mean"),
            background_neff_median=("background_neff", "median"),
            largest_background_weight_fraction_median=("largest_background_weight_fraction", "median"),
        )
        .sort_values("background_tail_fraction")
    )

    comp_summary = (
        comp_by_seed.groupby(["tail_region", "analysis_sample"], as_index=False)
        .agg(
            rows_median=("rows", "median"),
            expected_events_450fb_mean=("expected_events_450fb", "mean"),
            fraction_of_tail_background_mean=("fraction_of_tail_background", "mean"),
        )
        .sort_values(["tail_region", "expected_events_450fb_mean"], ascending=[True, False])
    )

    feature_summary = summarize_tail_features(event_df, features)

    auc.to_csv(OUTDIR / "tail_audit_auc_by_seed.csv", index=False)
    tail_summary.to_csv(OUTDIR / "tail_region_summary.csv", index=False)
    comp_summary.to_csv(OUTDIR / "tail_background_composition_summary.csv", index=False)
    feature_summary.to_csv(OUTDIR / "tail_feature_medians_by_seed_sample.csv", index=False)

    tight_region = "top_0.1pct_background_tail"
    tight_comp = comp_summary[comp_summary["tail_region"].eq(tight_region)].copy()

    auc_summary = pd.DataFrame([{
        "qcd_bdt_weighted_auc_mean": auc["qcd_bdt_weighted_auc"].mean(),
        "qcd_bdt_weighted_auc_std": auc["qcd_bdt_weighted_auc"].std(),
        "top_bdt_weighted_auc_mean": auc["top_bdt_weighted_auc"].mean(),
        "top_bdt_weighted_auc_std": auc["top_bdt_weighted_auc"].std(),
    }])
    auc_summary.to_csv(OUTDIR / "tail_audit_auc_summary.csv", index=False)

    md = []
    md.append("# HH4b BDT-v2 tail audit\n")
    md.append("Date: 2026-07-11\n")
    md.append("This audit retrains the two BDT-v2 classifiers seed-by-seed, saves event-level BDT scores, and studies the most signal-like tails.\n")
    md.append("The goal is to identify which few high-score events dominate the highest-purity region and which backgrounds need more MC statistics.\n")
    md.append("## AUC summary\n")
    md.append(auc_summary.to_markdown(index=False))
    md.append("\n## Tail region summary\n")
    md.append(tail_summary.to_markdown(index=False))
    md.append("\n## Tightest background tail composition\n")
    if len(tight_comp):
        md.append(tight_comp.to_markdown(index=False))
    else:
        md.append("No tight-tail composition rows found.")
    md.append("\n## Interpretation checklist\n")
    md.append("- Low `background_neff_median` means the tail is MC-statistics limited.\n")
    md.append("- Large `largest_background_weight_fraction_median` means a few weighted events dominate the estimate.\n")
    md.append("- The next generated background should be whichever sample dominates the tightest tail.\n")
    md.append("- This is diagnostic and does not change the training inputs.\n")

    (OUTDIR / "tail_audit_summary.md").write_text("\n\n".join(md) + "\n")

    print("\n=== AUC summary ===")
    print(auc_summary.to_string(index=False))

    print("\n=== Tail region summary ===")
    print(tail_summary.to_string(index=False))

    print("\n=== Tightest tail composition ===")
    print(tight_comp.to_string(index=False))

    print("\nWrote:", OUTDIR)


if __name__ == "__main__":
    run()
