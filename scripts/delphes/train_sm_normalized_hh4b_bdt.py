#!/usr/bin/env python3

import os
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split


STORE = Path(os.environ["HH4B_STORE"])
REPO = Path(os.environ["HH4B_REPO"])

OUTDIR = REPO / "outputs/tables/sm_normalized_hh4b_bdt_2026_07_09"
OUTDIR.mkdir(parents=True, exist_ok=True)

LUMI_FB = 450.0
LUMI_PB = LUMI_FB * 1000.0
TEST_SIZE = 0.35
RANDOM_STATE = 42

FEATURES_BASE = [
    "mbb1", "mbb2", "avg_mbb", "delta_mbb", "mhh",
    "drbb1", "drbb2",
    "j1_pt", "j2_pt", "j3_pt", "j4_pt",
]


def find_one(patterns):
    matches = []
    for pat in patterns:
        matches.extend(sorted((STORE / "parquet").glob(pat)))
    matches = [p for p in matches if "test" not in p.name.lower()]
    if not matches:
        raise FileNotFoundError(f"No file matched: {patterns}")
    return matches[-1]


def add_features(df):
    df = df.copy()
    df["r_hh"] = np.sqrt((df["mbb1"] - 125.0) ** 2 + (df["mbb2"] - 125.0) ** 2)
    df["pt_sum4"] = df["j1_pt"] + df["j2_pt"] + df["j3_pt"] + df["j4_pt"]
    df["pt_asym_12"] = (df["j1_pt"] - df["j2_pt"]) / (df["j1_pt"] + df["j2_pt"] + 1e-9)
    df["pt_asym_34"] = (df["j3_pt"] - df["j4_pt"]) / (df["j3_pt"] + df["j4_pt"] + 1e-9)
    return df


def load_one(sample, group, path, n_generated, xsec_pb):
    df = pd.read_parquet(path)
    df = add_features(df)

    df["sample"] = sample
    df["group"] = group
    df["target"] = 1 if group == "signal" else 0
    df["n_generated"] = n_generated
    df["xsec_pb"] = xsec_pb
    df["weight_pb"] = xsec_pb / n_generated

    return df


def load_all():
    frames = []

    # Provisional external SM HH xsec × BR(H→bb)^2 normalization.
    frames.append(load_one(
        "ggF_HH4b_SMnorm",
        "signal",
        find_one(["HH4b_ggf_hh4b_10k*merged*hh4b_candidates.parquet"]),
        10000,
        10.55 / 1000.0,
    ))

    frames.append(load_one(
        "VBF_HH4b_SMnorm",
        "signal",
        find_one(["HH4b_vbf_hh4b_10k*merged*hh4b_candidates.parquet"]),
        10000,
        0.587 / 1000.0,
    ))

    frames.append(load_one(
        "ttbar_200k",
        "background",
        find_one(["ttbar_200k_merged_hh4b_candidates.parquet"]),
        200000,
        512.2746276855469,
    ))

    frames.append(load_one(
        "Zbbbb_100k",
        "background",
        find_one(["zbbbb*100k*merged*hh4b_candidates.parquet", "Zbbbb*100k*merged*hh4b_candidates.parquet"]),
        100000,
        6.912992,
    ))

    qcd_meta = pd.read_csv(Path(os.environ.get("QCD_META", STORE / "metadata/qcd_bbbb_iht_slice_scan_20000.csv")))
    for _, row in qcd_meta.iterrows():
        tag = str(row["tag"])
        frames.append(load_one(
            tag,
            "background",
            find_one([f"{tag}*merged*hh4b_candidates.parquet", f"{tag}*hh4b_candidates.parquet"]),
            int(row["n_generated"]),
            float(row["xsec_pb"]),
        ))

    df = pd.concat(frames, ignore_index=True)

    feature_cols = FEATURES_BASE + ["r_hh", "pt_sum4", "pt_asym_12", "pt_asym_34"]
    missing = [c for c in feature_cols if c not in df.columns]
    if missing:
        raise RuntimeError(f"Missing features: {missing}")

    df = df.replace([np.inf, -np.inf], np.nan).dropna(subset=feature_cols)
    return df, feature_cols


def main():
    df, feature_cols = load_all()

    train_df, test_df = train_test_split(
        df,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=df["target"],
    )

    X_train = train_df[feature_cols]
    y_train = train_df["target"].astype(int)

    X_test = test_df[feature_cols]
    y_test = test_df["target"].astype(int)

    # Shape-discrimination training weights: balance signal/background classes.
    n_sig_train = max((y_train == 1).sum(), 1)
    n_bkg_train = max((y_train == 0).sum(), 1)
    train_weight = np.where(y_train == 1, 0.5 / n_sig_train, 0.5 / n_bkg_train)

    clf = GradientBoostingClassifier(
        n_estimators=250,
        learning_rate=0.04,
        max_depth=3,
        subsample=0.8,
        random_state=RANDOM_STATE,
    )

    clf.fit(X_train, y_train, sample_weight=train_weight)

    test_df = test_df.copy()
    test_df["bdt_score"] = clf.predict_proba(X_test)[:, 1]

    # Scale test subset back to full sample statistics sample-by-sample.
    total_counts = df.groupby("sample").size()
    test_counts = test_df.groupby("sample").size()

    def scale_to_full(sample):
        return total_counts.loc[sample] / test_counts.loc[sample]

    test_df["weight_pb_scaled_to_full"] = test_df["weight_pb"] * test_df["sample"].map(scale_to_full)

    unweighted_auc = roc_auc_score(y_test, test_df["bdt_score"])
    physics_weighted_auc = roc_auc_score(
        y_test,
        test_df["bdt_score"],
        sample_weight=test_df["weight_pb_scaled_to_full"],
    )

    auc_summary = pd.DataFrame([
        {
            "training": "class-balanced GradientBoostingClassifier",
            "normalization": "provisional SM-normalized signal, MG5-normalized backgrounds",
            "test_size": TEST_SIZE,
            "unweighted_auc": unweighted_auc,
            "physics_weighted_auc": physics_weighted_auc,
            "n_train": len(train_df),
            "n_test": len(test_df),
        }
    ])

    thresholds = np.linspace(0.0, 0.995, 200)
    scan_rows = []

    for thr in thresholds:
        sel = test_df["bdt_score"] >= thr

        s_pb = test_df.loc[sel & (test_df["target"] == 1), "weight_pb_scaled_to_full"].sum()
        b_pb = test_df.loc[sel & (test_df["target"] == 0), "weight_pb_scaled_to_full"].sum()

        s_ev = s_pb * LUMI_PB
        b_ev = b_pb * LUMI_PB

        scan_rows.append({
            "threshold": thr,
            "signal_xsec_pb": s_pb,
            "background_xsec_pb": b_pb,
            "signal_events_450fb": s_ev,
            "background_events_450fb": b_ev,
            "S_over_B": s_ev / b_ev if b_ev > 0 else np.nan,
            "S_over_sqrtB": s_ev / np.sqrt(b_ev) if b_ev > 0 else np.nan,
            "S_over_10pctB": s_ev / (0.10 * b_ev) if b_ev > 0 else np.nan,
            "n_signal_test_rows": int((sel & (test_df["target"] == 1)).sum()),
            "n_background_test_rows": int((sel & (test_df["target"] == 0)).sum()),
        })

    scan = pd.DataFrame(scan_rows)

    best_sqrtb = scan.sort_values("S_over_sqrtB", ascending=False).head(10)
    best_s_over_b = scan.sort_values("S_over_B", ascending=False).head(10)
    best_syst = scan.sort_values("S_over_10pctB", ascending=False).head(10)

    importances = pd.DataFrame({
        "feature": feature_cols,
        "importance": clf.feature_importances_,
    }).sort_values("importance", ascending=False)


    # Composition of selected events at important BDT thresholds.
    comp_thresholds = [0.84, 0.865, 0.87, 0.885, 0.91]
    comp_rows = []

    for thr in comp_thresholds:
        sel = test_df["bdt_score"] >= thr
        selected = test_df.loc[sel].copy()

        for (sample, group), sub in selected.groupby(["sample", "group"]):
            xsec_pb = sub["weight_pb_scaled_to_full"].sum()
            expected_events = xsec_pb * LUMI_PB

            comp_rows.append({
                "threshold": thr,
                "sample": sample,
                "group": group,
                "selected_test_rows": len(sub),
                "xsec_pb": xsec_pb,
                "expected_events_450fb": expected_events,
            })

    comp = pd.DataFrame(comp_rows)

    if len(comp):
        comp["fraction_within_group_at_threshold"] = (
            comp["expected_events_450fb"]
            / comp.groupby(["threshold", "group"])["expected_events_450fb"].transform("sum")
        )

        comp.to_csv(OUTDIR / "bdt_threshold_composition_by_sample.csv", index=False)
        (OUTDIR / "bdt_threshold_composition_by_sample.md").write_text(comp.to_markdown(index=False) + "\n")

    by_sample = (
        test_df.assign(expected_events_450fb=lambda d: d["weight_pb_scaled_to_full"] * LUMI_PB)
        .groupby(["sample", "group"], as_index=False)
        .agg(
            test_rows=("sample", "size"),
            xsec_pb=("weight_pb_scaled_to_full", "sum"),
            expected_events_450fb=("expected_events_450fb", "sum"),
            median_bdt_score=("bdt_score", "median"),
            q90_bdt_score=("bdt_score", lambda x: float(np.quantile(x, 0.90))),
            q99_bdt_score=("bdt_score", lambda x: float(np.quantile(x, 0.99))),
        )
    )

    auc_summary.to_csv(OUTDIR / "bdt_auc_summary.csv", index=False)
    scan.to_csv(OUTDIR / "bdt_threshold_scan.csv", index=False)
    best_sqrtb.to_csv(OUTDIR / "bdt_best_thresholds_by_sqrtB.csv", index=False)
    best_s_over_b.to_csv(OUTDIR / "bdt_best_thresholds_by_S_over_B.csv", index=False)
    best_syst.to_csv(OUTDIR / "bdt_best_thresholds_by_10pctB.csv", index=False)
    importances.to_csv(OUTDIR / "bdt_feature_importances.csv", index=False)
    by_sample.to_csv(OUTDIR / "bdt_testset_by_sample.csv", index=False)

    (OUTDIR / "bdt_auc_summary.md").write_text(auc_summary.to_markdown(index=False) + "\n")
    (OUTDIR / "bdt_best_thresholds_by_sqrtB.md").write_text(best_sqrtb.to_markdown(index=False) + "\n")
    (OUTDIR / "bdt_best_thresholds_by_S_over_B.md").write_text(best_s_over_b.to_markdown(index=False) + "\n")
    (OUTDIR / "bdt_best_thresholds_by_10pctB.md").write_text(best_syst.to_markdown(index=False) + "\n")
    (OUTDIR / "bdt_feature_importances.md").write_text(importances.to_markdown(index=False) + "\n")
    (OUTDIR / "bdt_testset_by_sample.md").write_text(by_sample.to_markdown(index=False) + "\n")

    readme = """# Provisional SM-normalized HH4b BDT

This uses:
- SM-normalized ggF HH→4b = 10.55 fb
- SM-normalized VBF HH→4b = 0.587 fb
- QCD bbbb HT-sliced 20k/slice
- ttbar 200k
- Zbbbb 100k

The signal normalization is provisional until Harvey confirms the convention.
"""
    (OUTDIR / "README.md").write_text(readme)

    print("\n=== BDT AUC summary ===")
    print(auc_summary.to_string(index=False))

    print("\n=== Feature importances ===")
    print(importances.to_string(index=False))

    print("\n=== Best thresholds by S/sqrt(B) ===")
    print(best_sqrtb.to_string(index=False))

    print(f"\nWrote outputs to: {OUTDIR}")


if __name__ == "__main__":
    main()
