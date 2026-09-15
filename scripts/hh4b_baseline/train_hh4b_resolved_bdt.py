#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.inspection import permutation_importance
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import train_test_split


def asimov_z(s: float, b: float) -> float:
    if s <= 0 or b <= 0:
        return 0.0
    return math.sqrt(max(0.0, 2.0 * ((s + b) * math.log(1.0 + s / b) - s)))


def neff(w) -> float:
    w = np.asarray(w, dtype=float)
    w = w[np.isfinite(w)]
    if len(w) == 0 or np.sum(w * w) <= 0:
        return 0.0
    return float(np.sum(w) ** 2 / np.sum(w * w))


def summarize_selection(df: pd.DataFrame, mask, name: str) -> dict:
    sub = df.loc[mask].copy()
    sig = sub["is_signal"].astype(bool)
    bkg = ~sig

    finite_w = np.isfinite(sub["physics_weight"])

    sw = sub.loc[sig & finite_w, "physics_weight"].sum()
    bw = sub.loc[bkg & finite_w, "physics_weight"].sum()
    bkg_w = sub.loc[bkg & finite_w, "physics_weight"].to_numpy()

    return {
        "selection": name,
        "raw_signal": int(sig.sum()),
        "raw_background": int(bkg.sum()),
        "raw_background_missing_weight": int((bkg & ~finite_w).sum()),
        "S_w": float(sw),
        "B_w": float(bw),
        "S_over_B": float(sw / bw) if bw > 0 else 0.0,
        "S_over_SplusB": float(sw / (sw + bw)) if (sw + bw) > 0 else 0.0,
        "S_over_sqrtB": float(sw / math.sqrt(bw)) if bw > 0 else 0.0,
        "S_over_sqrtSplusB": float(sw / math.sqrt(sw + bw)) if (sw + bw) > 0 else 0.0,
        "asimov_Z_A": asimov_z(sw, bw),
        "N_eff_bkg": neff(bkg_w),
    }


def make_balanced_training_weights(y, physics_w):
    y = np.asarray(y).astype(int)
    w = np.asarray(physics_w).astype(float)

    out = np.zeros_like(w, dtype=float)

    for cls in [0, 1]:
        mask = y == cls
        total = np.sum(w[mask])
        if total > 0:
            out[mask] = 0.5 * w[mask] / total

    nonzero = out[out > 0]
    if len(nonzero) > 0:
        out = out / np.mean(nonzero)

    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--features",
        default="outputs/hh4b_baseline/full_features/hh4b_resolved_features.parquet",
    )
    parser.add_argument(
        "--outdir",
        default="outputs/hh4b_baseline/bdt_v1",
    )
    parser.add_argument("--random-state", type=int, default=12345)
    args = parser.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    df = pd.read_parquet(args.features)

    # Exclude samples with missing/non-finite proxy weights from weighted BDT training/evaluation.
    # They remain documented in the cut baseline.
    df_trainable = df[np.isfinite(df["physics_weight"])].copy().reset_index(drop=True)

    print("[INFO] Loaded feature table:", args.features)
    print("[INFO] Total rows:", len(df))
    print("[INFO] Trainable finite-weight rows:", len(df_trainable))
    print("[INFO] Excluded missing-weight rows:", len(df) - len(df_trainable))
    print("[INFO] Signal/background in trainable table:")
    print(df_trainable["is_signal"].value_counts().rename(index={0: "background", 1: "signal"}).to_string())

    if df_trainable["is_signal"].nunique() < 2:
        raise RuntimeError("Need both signal and background in trainable sample.")

    feature_cols = [
        "n_ak4",
        "n_btag",
        "n_ak8_200",
        "n_ak8_300",
        "ht",
        "vbf_mjj",
        "vbf_deta",
        "b1_pt",
        "b1_eta",
        "b1_mass",
        "b1_btag",
        "b2_pt",
        "b2_eta",
        "b2_mass",
        "b2_btag",
        "b3_pt",
        "b3_eta",
        "b3_mass",
        "b3_btag",
        "b4_pt",
        "b4_eta",
        "b4_mass",
        "b4_btag",
        "h1_mass",
        "h2_mass",
        "h1_pt",
        "h2_pt",
        "hh_mass",
        "hh_pt",
        "dr_h1_bb",
        "dr_h2_bb",
        "pairing_score",
        "h_mass_avg",
        "h_mass_diff",
        "h1_abs_m125",
        "h2_abs_m125",
        "category_hmass_70_190",
        "category_hmass_90_160",
        "category_vbf_like",
        "category_boosted_1ak8_2b",
    ]

    feature_cols = [c for c in feature_cols if c in df_trainable.columns]

    X = df_trainable[feature_cols].astype(float)
    X = X.replace([np.inf, -np.inf], np.nan).fillna(-999.0)

    y = df_trainable["is_signal"].astype(int)
    w_phys = df_trainable["physics_weight"].astype(float)

    idx_all = np.arange(len(df_trainable))

    idx_train, idx_temp = train_test_split(
        idx_all,
        test_size=0.40,
        random_state=args.random_state,
        stratify=y,
    )

    idx_val, idx_test = train_test_split(
        idx_temp,
        test_size=0.50,
        random_state=args.random_state,
        stratify=y.iloc[idx_temp],
    )

    X_train = X.iloc[idx_train]
    y_train = y.iloc[idx_train]
    w_train_phys = w_phys.iloc[idx_train]

    X_val = X.iloc[idx_val]
    y_val = y.iloc[idx_val]
    w_val_phys = w_phys.iloc[idx_val]

    X_test = X.iloc[idx_test]
    y_test = y.iloc[idx_test]
    w_test_phys = w_phys.iloc[idx_test]

    w_train_balanced = make_balanced_training_weights(y_train, w_train_phys)

    model = HistGradientBoostingClassifier(
        max_iter=400,
        learning_rate=0.04,
        max_leaf_nodes=31,
        min_samples_leaf=30,
        l2_regularization=1.0,
        early_stopping=True,
        validation_fraction=0.15,
        random_state=args.random_state,
    )

    model.fit(X_train, y_train, sample_weight=w_train_balanced)

    val_score = model.predict_proba(X_val)[:, 1]
    test_score = model.predict_proba(X_test)[:, 1]

    df_val = df_trainable.iloc[idx_val].copy()
    df_val["bdt_score"] = val_score

    df_test = df_trainable.iloc[idx_test].copy()
    df_test["bdt_score"] = test_score

    metrics = {
        "n_total_rows_original": int(len(df)),
        "n_total_trainable_finite_weight": int(len(df_trainable)),
        "n_excluded_missing_weight": int(len(df) - len(df_trainable)),
        "n_train": int(len(idx_train)),
        "n_val": int(len(idx_val)),
        "n_test": int(len(idx_test)),
        "n_signal_trainable": int(y.sum()),
        "n_background_trainable": int((1 - y).sum()),
        "feature_cols": feature_cols,
        "raw_auc_val": float(roc_auc_score(y_val, val_score)),
        "raw_auc_test": float(roc_auc_score(y_test, test_score)),
        "weighted_auc_val": float(roc_auc_score(y_val, val_score, sample_weight=w_val_phys)),
        "weighted_auc_test": float(roc_auc_score(y_test, test_score, sample_weight=w_test_phys)),
        "raw_ap_val": float(average_precision_score(y_val, val_score)),
        "raw_ap_test": float(average_precision_score(y_test, test_score)),
        "weighted_ap_val": float(average_precision_score(y_val, val_score, sample_weight=w_val_phys)),
        "weighted_ap_test": float(average_precision_score(y_test, test_score, sample_weight=w_test_phys)),
    }

    # Validation threshold scan.
    thresholds = sorted(set(np.quantile(val_score, np.linspace(0.50, 0.995, 120)).tolist()))

    scan_rows = []
    for thr in thresholds:
        row = summarize_selection(df_val, df_val["bdt_score"] >= thr, f"val_score_ge_{thr:.6f}")
        row["threshold"] = float(thr)
        scan_rows.append(row)

    scan = pd.DataFrame(scan_rows)
    scan = scan.sort_values(["asimov_Z_A", "S_over_sqrtSplusB", "S_over_B"], ascending=False)

    # Choose best validation threshold with at least modest effective background statistics.
    stable = scan[scan["N_eff_bkg"] >= 10].copy()
    if len(stable) > 0:
        best_thr = float(stable.iloc[0]["threshold"])
        best_policy = "best_val_ZA_with_Neff_ge_10"
    else:
        best_thr = float(scan.iloc[0]["threshold"])
        best_policy = "best_val_ZA_no_Neff_requirement"

    # Test categories.
    category_rows = []

    category_rows.append(
        summarize_selection(
            df_test,
            np.ones(len(df_test), dtype=bool),
            "test_all_resolved_4b",
        )
    )

    category_rows.append(
        summarize_selection(
            df_test,
            df_test["category_hmass_70_190"].astype(bool),
            "test_cut_hmass_70_190",
        )
    )

    category_rows.append(
        summarize_selection(
            df_test,
            df_test["category_hmass_90_160"].astype(bool),
            "test_cut_hmass_90_160",
        )
    )

    category_rows.append(
        summarize_selection(
            df_test,
            df_test["category_vbf_like"].astype(bool),
            "test_cut_vbf_like",
        )
    )

    category_rows.append(
        summarize_selection(
            df_test,
            df_test["category_boosted_1ak8_2b"].astype(bool),
            "test_cut_boosted_1AK8_2b",
        )
    )

    category_rows.append(
        summarize_selection(
            df_test,
            df_test["bdt_score"] >= best_thr,
            f"test_bdt_best_val_threshold_{best_thr:.6f}",
        )
    )

    for thr in [0.50, 0.70, 0.80, 0.90, 0.95, 0.98, 0.99]:
        category_rows.append(
            summarize_selection(
                df_test,
                df_test["bdt_score"] >= thr,
                f"test_bdt_score_ge_{thr:.2f}",
            )
        )

    # Quantile-based high-score regions.
    for q in [0.80, 0.90, 0.95, 0.98, 0.99]:
        thr = float(np.quantile(test_score, q))
        category_rows.append(
            summarize_selection(
                df_test,
                df_test["bdt_score"] >= thr,
                f"test_top_{int(round((1-q)*100))}pct_score_ge_{thr:.6f}",
            )
        )

    cats = pd.DataFrame(category_rows)

    # Dominant backgrounds in best BDT region.
    best_mask = df_test["bdt_score"] >= best_thr
    best = df_test.loc[
        best_mask
        & (df_test["is_signal"] == 0)
        & np.isfinite(df_test["physics_weight"])
    ].copy()

    if len(best) > 0:
        bkg = (
            best.groupby("process_group")
            .agg(
                raw_background=("physics_weight", "size"),
                B_w=("physics_weight", "sum"),
                B_w2=("physics_weight", lambda x: float(np.sum(np.asarray(x, dtype=float) ** 2))),
            )
            .reset_index()
        )
        total_b = bkg["B_w"].sum()
        bkg["fraction_of_B_w"] = bkg["B_w"] / total_b if total_b > 0 else 0.0
        bkg["N_eff_group"] = bkg["B_w"] ** 2 / bkg["B_w2"]
        bkg = bkg.sort_values("B_w", ascending=False)
    else:
        bkg = pd.DataFrame(
            columns=[
                "process_group",
                "raw_background",
                "B_w",
                "B_w2",
                "fraction_of_B_w",
                "N_eff_group",
            ]
        )

    # Permutation importance on a small test subset.
    n_perm = min(5000, len(X_test))
    X_perm = X_test.iloc[:n_perm]
    y_perm = y_test.iloc[:n_perm]

    try:
        perm = permutation_importance(
            model,
            X_perm,
            y_perm,
            n_repeats=5,
            random_state=args.random_state,
            scoring="roc_auc",
        )

        imp = pd.DataFrame(
            {
                "feature": feature_cols,
                "importance_mean": perm.importances_mean,
                "importance_std": perm.importances_std,
            }
        ).sort_values("importance_mean", ascending=False)
    except Exception as exc:
        imp = pd.DataFrame({"error": [str(exc)]})

    # Save outputs.
    joblib.dump(model, outdir / "hh4b_resolved_bdt_model.joblib")

    metrics_with_threshold = metrics | {
        "best_threshold": best_thr,
        "best_threshold_policy": best_policy,
    }

    with open(outdir / "hh4b_resolved_bdt_metrics.json", "w") as f:
        json.dump(metrics_with_threshold, f, indent=2)

    scan.to_csv(outdir / "hh4b_resolved_bdt_validation_threshold_scan.csv", index=False)
    cats.to_csv(outdir / "hh4b_resolved_bdt_test_categories.csv", index=False)
    bkg.to_csv(outdir / "hh4b_resolved_bdt_best_region_backgrounds.csv", index=False)
    imp.to_csv(outdir / "hh4b_resolved_bdt_permutation_importance.csv", index=False)

    df_test[
        [
            "event_key",
            "sample",
            "process_group",
            "is_signal",
            "physics_weight",
            "bdt_score",
            "category_hmass_70_190",
            "category_hmass_90_160",
            "category_vbf_like",
            "category_boosted_1ak8_2b",
        ]
    ].to_csv(outdir / "hh4b_resolved_bdt_test_scores.csv", index=False)

    print("\n[INFO] Metrics:")
    print(json.dumps(metrics_with_threshold, indent=2))

    print("\n[INFO] Test category yields:")
    print(cats.to_string(index=False))

    print("\n[INFO] Dominant backgrounds in best BDT region:")
    print(bkg.head(30).to_string(index=False))

    print("\n[INFO] Top permutation importances:")
    print(imp.head(30).to_string(index=False))

    print("\n[INFO] Wrote outputs to:", outdir)


if __name__ == "__main__":
    main()
