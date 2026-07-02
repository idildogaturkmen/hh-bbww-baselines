'''
Trains a new BDT for HH->4b resolved analysis, version 2.
This version removes binary cut-category flags from the training inputs and uses them only for evaluation.
It also scans validation-defined BDT score bins and applies the selected bins to the held-out test

BDT v2 with:
Training preselection:
  >=4 b-tag AK4 jets

Signal:
  HH_4b

Background:
  all finite-weight non-HH4b backgrounds

Features:
  same kinematics and mass-pairing features,
  but remove binary category flags from training:
    category_hmass_70_190
    category_hmass_90_160
    category_vbf_like
    category_boosted_1ak8_2b

Evaluation:
  test_all_resolved_4b
  test_hmass_70_190
  test_hmass_90_160
  BDT bin 1: moderate score
  BDT bin 2: high score
  BDT bin 3: very high score only if N_eff is acceptable
  VBF-like category separately
  '''

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


def two_b_over_s(s: float, b: float) -> float:
    if s <= 0:
        return float("inf")
    return float(2.0 * b / s)


def summarize_selection(
    df: pd.DataFrame,
    mask,
    selection: str,
    category_type: str = "",
    score_low: float | None = None,
    score_high: float | None = None,
) -> dict:
    sub = df.loc[mask].copy()

    if len(sub) == 0:
        return {
            "selection": selection,
            "category_type": category_type,
            "score_low": score_low,
            "score_high": score_high,
            "raw_signal": 0,
            "raw_background": 0,
            "raw_background_missing_weight": 0,
            "S_w": 0.0,
            "B_w": 0.0,
            "S_over_B": 0.0,
            "S_over_SplusB": 0.0,
            "S_over_sqrtB": 0.0,
            "S_over_sqrtSplusB": 0.0,
            "asimov_Z_A": 0.0,
            "N_eff_bkg": 0.0,
            "twoB_over_S": float("inf"),
        }

    sig = sub["is_signal"].to_numpy(dtype=bool)
    bkg = ~sig

    w = sub["physics_weight"].to_numpy(dtype=float)
    finite_w = np.isfinite(w)

    sw = float(np.sum(w[sig & finite_w]))
    bw = float(np.sum(w[bkg & finite_w]))
    bkg_w = w[bkg & finite_w]

    return {
        "selection": selection,
        "category_type": category_type,
        "score_low": score_low,
        "score_high": score_high,
        "raw_signal": int(np.sum(sig)),
        "raw_background": int(np.sum(bkg)),
        "raw_background_missing_weight": int(np.sum(bkg & ~finite_w)),
        "S_w": sw,
        "B_w": bw,
        "S_over_B": float(sw / bw) if bw > 0 else 0.0,
        "S_over_SplusB": float(sw / (sw + bw)) if (sw + bw) > 0 else 0.0,
        "S_over_sqrtB": float(sw / math.sqrt(bw)) if bw > 0 else 0.0,
        "S_over_sqrtSplusB": float(sw / math.sqrt(sw + bw)) if (sw + bw) > 0 else 0.0,
        "asimov_Z_A": asimov_z(sw, bw),
        "N_eff_bkg": neff(bkg_w),
        "twoB_over_S": two_b_over_s(sw, bw),
    }


def summarize_combined_category_set(rows: list[dict], selection: str, category_type: str) -> dict:
    sw = float(sum(r["S_w"] for r in rows))
    bw = float(sum(r["B_w"] for r in rows))
    z_quad = float(math.sqrt(sum(r["asimov_Z_A"] ** 2 for r in rows)))

    return {
        "selection": selection,
        "category_type": category_type,
        "score_low": None,
        "score_high": None,
        "raw_signal": int(sum(r["raw_signal"] for r in rows)),
        "raw_background": int(sum(r["raw_background"] for r in rows)),
        "raw_background_missing_weight": int(sum(r["raw_background_missing_weight"] for r in rows)),
        "S_w": sw,
        "B_w": bw,
        "S_over_B": float(sw / bw) if bw > 0 else 0.0,
        "S_over_SplusB": float(sw / (sw + bw)) if (sw + bw) > 0 else 0.0,
        "S_over_sqrtB": float(sw / math.sqrt(bw)) if bw > 0 else 0.0,
        "S_over_sqrtSplusB": float(sw / math.sqrt(sw + bw)) if (sw + bw) > 0 else 0.0,
        "asimov_Z_A": z_quad,
        "N_eff_bkg": float("nan"),
        "twoB_over_S": two_b_over_s(sw, bw),
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


def mask_bin(scores: np.ndarray, lo: float, hi: float | None):
    if hi is None:
        return scores >= lo
    return (scores >= lo) & (scores < hi)


def scan_cumulative_thresholds(df_val: pd.DataFrame, val_score: np.ndarray) -> pd.DataFrame:
    thresholds = sorted(set(np.quantile(val_score, np.linspace(0.20, 0.995, 160)).tolist()))

    rows = []
    for thr in thresholds:
        mask = df_val["bdt_score"].to_numpy() >= thr
        row = summarize_selection(
            df_val,
            mask,
            selection=f"val_bdt_score_ge_{thr:.6f}",
            category_type="validation_cumulative",
            score_low=float(thr),
            score_high=None,
        )
        rows.append(row)

    out = pd.DataFrame(rows)
    out = out.sort_values(
        ["asimov_Z_A", "N_eff_bkg", "S_over_B"],
        ascending=[False, False, False],
    )
    return out


def optimize_bdt_bins(
    df_val: pd.DataFrame,
    val_score: np.ndarray,
    min_neff: float,
    min_raw_bkg: int,
    min_raw_sig: int,
) -> tuple[pd.DataFrame, dict, list[tuple[float, float | None]]]:
    # Candidate thresholds are score quantiles. This avoids hard-coding score values.
    q_grid = np.linspace(0.35, 0.95, 25)
    thresholds = sorted(set(float(x) for x in np.quantile(val_score, q_grid)))

    rows = []

    # Try 3-bin configurations: [t1,t2), [t2,t3), [t3,inf)
    for i, t1 in enumerate(thresholds):
        for j, t2 in enumerate(thresholds):
            for k, t3 in enumerate(thresholds):
                if not (t1 < t2 < t3):
                    continue

                bin_defs = [(t1, t2), (t2, t3), (t3, None)]
                bin_rows = []

                for bidx, (lo, hi) in enumerate(bin_defs, start=1):
                    mask = mask_bin(val_score, lo, hi)
                    r = summarize_selection(
                        df_val,
                        mask,
                        selection=f"val_bin{bidx}_{lo:.6f}_{'inf' if hi is None else f'{hi:.6f}'}",
                        category_type="validation_optimized_bin_candidate",
                        score_low=lo,
                        score_high=hi,
                    )
                    r["bin_index"] = bidx
                    r["passes_stability"] = bool(
                        r["S_w"] > 0
                        and r["B_w"] > 0
                        and r["N_eff_bkg"] >= min_neff
                        and r["raw_background"] >= min_raw_bkg
                        and r["raw_signal"] >= min_raw_sig
                    )
                    bin_rows.append(r)

                stable_rows = [r for r in bin_rows if r["passes_stability"]]

                if len(stable_rows) == 0:
                    continue

                combined_z_quad = math.sqrt(sum(r["asimov_Z_A"] ** 2 for r in stable_rows))
                total_s = sum(r["S_w"] for r in stable_rows)
                total_b = sum(r["B_w"] for r in stable_rows)
                total_raw_sig = sum(r["raw_signal"] for r in stable_rows)
                total_raw_bkg = sum(r["raw_background"] for r in stable_rows)
                min_bin_neff = min(r["N_eff_bkg"] for r in stable_rows)

                rows.append(
                    {
                        "t1": t1,
                        "t2": t2,
                        "t3": t3,
                        "n_stable_bins": len(stable_rows),
                        "combined_Z_quad": combined_z_quad,
                        "total_S_w": total_s,
                        "total_B_w": total_b,
                        "total_S_over_B": total_s / total_b if total_b > 0 else 0.0,
                        "total_S_over_sqrtSplusB": total_s / math.sqrt(total_s + total_b)
                        if (total_s + total_b) > 0
                        else 0.0,
                        "total_twoB_over_S": two_b_over_s(total_s, total_b),
                        "total_raw_signal": total_raw_sig,
                        "total_raw_background": total_raw_bkg,
                        "min_stable_bin_N_eff_bkg": min_bin_neff,
                    }
                )

    scan = pd.DataFrame(rows)

    if len(scan) == 0:
        # Fallback: one stable cumulative threshold.
        cumulative = scan_cumulative_thresholds(df_val, val_score)
        stable = cumulative[
            (cumulative["N_eff_bkg"] >= min_neff)
            & (cumulative["raw_background"] >= min_raw_bkg)
            & (cumulative["raw_signal"] >= min_raw_sig)
        ].copy()

        if len(stable) == 0:
            best = cumulative.iloc[0].to_dict()
            lo = float(best["score_low"])
            selected_bins = [(lo, None)]
            best_config = {
                "mode": "fallback_best_cumulative_no_stability",
                "t1": lo,
                "t2": None,
                "t3": None,
                "n_stable_bins": 1,
                "combined_Z_quad": float(best["asimov_Z_A"]),
            }
        else:
            stable = stable.sort_values(
                ["asimov_Z_A", "N_eff_bkg", "S_over_B"],
                ascending=[False, False, False],
            )
            best = stable.iloc[0].to_dict()
            lo = float(best["score_low"])
            selected_bins = [(lo, None)]
            best_config = {
                "mode": "fallback_best_cumulative_with_stability",
                "t1": lo,
                "t2": None,
                "t3": None,
                "n_stable_bins": 1,
                "combined_Z_quad": float(best["asimov_Z_A"]),
            }

        return scan, best_config, selected_bins

    # Prefer more stable bins if combined significance is close.
    scan = scan.sort_values(
        ["combined_Z_quad", "n_stable_bins", "min_stable_bin_N_eff_bkg", "total_S_over_B"],
        ascending=[False, False, False, False],
    ).reset_index(drop=True)

    best = scan.iloc[0].to_dict()
    t1, t2, t3 = float(best["t1"]), float(best["t2"]), float(best["t3"])

    all_bins = [(t1, t2), (t2, t3), (t3, None)]
    selected_bins = []

    for lo, hi in all_bins:
        r = summarize_selection(
            df_val,
            mask_bin(val_score, lo, hi),
            selection="stability_check",
            category_type="validation_best_bin_stability_check",
            score_low=lo,
            score_high=hi,
        )
        if (
            r["S_w"] > 0
            and r["B_w"] > 0
            and r["N_eff_bkg"] >= min_neff
            and r["raw_background"] >= min_raw_bkg
            and r["raw_signal"] >= min_raw_sig
        ):
            selected_bins.append((lo, hi))

    best_config = {
        "mode": "optimized_three_bin_scan",
        "t1": t1,
        "t2": t2,
        "t3": t3,
        "n_stable_bins": int(best["n_stable_bins"]),
        "combined_Z_quad": float(best["combined_Z_quad"]),
        "total_S_w": float(best["total_S_w"]),
        "total_B_w": float(best["total_B_w"]),
        "total_S_over_B": float(best["total_S_over_B"]),
        "total_S_over_sqrtSplusB": float(best["total_S_over_sqrtSplusB"]),
        "total_twoB_over_S": float(best["total_twoB_over_S"]),
        "min_stable_bin_N_eff_bkg": float(best["min_stable_bin_N_eff_bkg"]),
    }

    return scan, best_config, selected_bins


def background_tables_for_categories(
    df: pd.DataFrame,
    category_masks: list[tuple[str, np.ndarray]],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    group_rows = []
    sample_rows = []

    for cat_name, mask in category_masks:
        sub = df.loc[mask].copy()
        sub = sub[(sub["is_signal"] == 0) & np.isfinite(sub["physics_weight"])].copy()

        if len(sub) == 0:
            continue

        g = (
            sub.groupby("process_group")
            .agg(
                raw_background=("physics_weight", "size"),
                B_w=("physics_weight", "sum"),
                B_w2=("physics_weight", lambda x: float(np.sum(np.asarray(x, dtype=float) ** 2))),
            )
            .reset_index()
        )
        total_b = g["B_w"].sum()
        g["fraction_of_B_w"] = g["B_w"] / total_b if total_b > 0 else 0.0
        g["N_eff_group"] = g["B_w"] ** 2 / g["B_w2"]
        g["category"] = cat_name
        group_rows.append(g)

        s = (
            sub.groupby(["sample", "process_group"])
            .agg(
                raw_background=("physics_weight", "size"),
                B_w=("physics_weight", "sum"),
                B_w2=("physics_weight", lambda x: float(np.sum(np.asarray(x, dtype=float) ** 2))),
            )
            .reset_index()
        )
        total_b = s["B_w"].sum()
        s["fraction_of_B_w"] = s["B_w"] / total_b if total_b > 0 else 0.0
        s["N_eff_sample"] = s["B_w"] ** 2 / s["B_w2"]
        s["category"] = cat_name
        sample_rows.append(s)

    group_out = pd.concat(group_rows, ignore_index=True) if group_rows else pd.DataFrame()
    sample_out = pd.concat(sample_rows, ignore_index=True) if sample_rows else pd.DataFrame()

    if len(group_out) > 0:
        group_out = group_out.sort_values(["category", "B_w"], ascending=[True, False])
    if len(sample_out) > 0:
        sample_out = sample_out.sort_values(["category", "B_w"], ascending=[True, False])

    return group_out, sample_out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--features",
        default="outputs/hh4b_baseline/full_features/hh4b_resolved_features.parquet",
    )
    parser.add_argument(
        "--outdir",
        default="outputs/hh4b_baseline/bdt_v2",
    )
    parser.add_argument("--random-state", type=int, default=12345)
    parser.add_argument("--min-neff-bin", type=float, default=5.0)
    parser.add_argument("--min-raw-bkg-bin", type=int, default=20)
    parser.add_argument("--min-raw-sig-bin", type=int, default=5)
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

    # V2 intentionally removes binary cut-category flags from training.
    # We evaluate those categories later, but we do not let the BDT memorize them directly.
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
        max_iter=500,
        learning_rate=0.035,
        max_leaf_nodes=31,
        min_samples_leaf=35,
        l2_regularization=2.0,
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
        "removed_from_training_but_used_for_evaluation": [
            "category_hmass_70_190",
            "category_hmass_90_160",
            "category_vbf_like",
            "category_boosted_1ak8_2b",
        ],
        "raw_auc_val": float(roc_auc_score(y_val, val_score)),
        "raw_auc_test": float(roc_auc_score(y_test, test_score)),
        "weighted_auc_val": float(roc_auc_score(y_val, val_score, sample_weight=w_val_phys)),
        "weighted_auc_test": float(roc_auc_score(y_test, test_score, sample_weight=w_test_phys)),
        "raw_ap_val": float(average_precision_score(y_val, val_score)),
        "raw_ap_test": float(average_precision_score(y_test, test_score)),
        "weighted_ap_val": float(average_precision_score(y_val, val_score, sample_weight=w_val_phys)),
        "weighted_ap_test": float(average_precision_score(y_test, test_score, sample_weight=w_test_phys)),
        "score_min_val": float(np.min(val_score)),
        "score_max_val": float(np.max(val_score)),
        "score_min_test": float(np.min(test_score)),
        "score_max_test": float(np.max(test_score)),
    }

    # Validation cumulative threshold scan.
    val_cumulative_scan = scan_cumulative_thresholds(df_val, val_score)

    # Validation non-overlapping BDT-bin optimization.
    bin_scan, best_bin_config, selected_bins = optimize_bdt_bins(
        df_val=df_val,
        val_score=val_score,
        min_neff=args.min_neff_bin,
        min_raw_bkg=args.min_raw_bkg_bin,
        min_raw_sig=args.min_raw_sig_bin,
    )

    metrics["best_bin_config"] = best_bin_config
    metrics["selected_bins"] = [
        {"low": float(lo), "high": None if hi is None else float(hi)}
        for lo, hi in selected_bins
    ]

    # Test cut baselines.
    test_rows = []
    test_rows.append(
        summarize_selection(
            df_test,
            np.ones(len(df_test), dtype=bool),
            selection="test_cut_all_resolved_4b",
            category_type="cut_baseline",
        )
    )
    test_rows.append(
        summarize_selection(
            df_test,
            df_test["category_hmass_70_190"].astype(bool),
            selection="test_cut_hmass_70_190",
            category_type="cut_baseline",
        )
    )
    test_rows.append(
        summarize_selection(
            df_test,
            df_test["category_hmass_90_160"].astype(bool),
            selection="test_cut_hmass_90_160",
            category_type="cut_baseline",
        )
    )
    test_rows.append(
        summarize_selection(
            df_test,
            df_test["category_vbf_like"].astype(bool),
            selection="test_cut_vbf_like",
            category_type="cut_baseline",
        )
    )
    test_rows.append(
        summarize_selection(
            df_test,
            df_test["category_boosted_1ak8_2b"].astype(bool),
            selection="test_cut_boosted_1AK8_2b",
            category_type="cut_baseline",
        )
    )

    # Fixed cumulative BDT thresholds.
    fixed_thresholds = [0.30, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.72, 0.74, 0.76]
    for thr in fixed_thresholds:
        test_rows.append(
            summarize_selection(
                df_test,
                df_test["bdt_score"].to_numpy() >= thr,
                selection=f"test_bdt_score_ge_{thr:.2f}",
                category_type="bdt_fixed_cumulative",
                score_low=thr,
                score_high=None,
            )
        )

    # Validation-optimized non-overlapping bins applied to test.
    optimized_test_bin_rows = []
    category_masks_for_backgrounds = []

    for idx, (lo, hi) in enumerate(selected_bins, start=1):
        mask = mask_bin(test_score, lo, hi)
        hi_label = "inf" if hi is None else f"{hi:.6f}"
        name = f"test_bdt_opt_bin{idx}_{lo:.6f}_{hi_label}"
        row = summarize_selection(
            df_test,
            mask,
            selection=name,
            category_type="bdt_v2_validation_optimized_bin",
            score_low=lo,
            score_high=hi,
        )
        row["bin_index"] = idx
        optimized_test_bin_rows.append(row)
        test_rows.append(row)
        category_masks_for_backgrounds.append((name, mask))

    if len(optimized_test_bin_rows) > 0:
        combined = summarize_combined_category_set(
            optimized_test_bin_rows,
            selection="test_bdt_v2_combined_validation_optimized_bins",
            category_type="bdt_v2_combined",
        )
        test_rows.append(combined)

    # Also create manual non-overlapping bins around the useful v1 range.
    manual_bins = [
        (0.40, 0.50),
        (0.50, 0.60),
        (0.60, 0.68),
        (0.68, 0.74),
        (0.74, None),
    ]

    manual_rows = []
    for idx, (lo, hi) in enumerate(manual_bins, start=1):
        mask = mask_bin(test_score, lo, hi)
        hi_label = "inf" if hi is None else f"{hi:.2f}"
        name = f"test_bdt_manual_bin{idx}_{lo:.2f}_{hi_label}"
        row = summarize_selection(
            df_test,
            mask,
            selection=name,
            category_type="bdt_manual_nonoverlap_bin",
            score_low=lo,
            score_high=hi,
        )
        row["bin_index"] = idx
        manual_rows.append(row)
        test_rows.append(row)
        category_masks_for_backgrounds.append((name, mask))

    stable_manual_rows = [
        r for r in manual_rows
        if r["S_w"] > 0
        and r["B_w"] > 0
        and r["N_eff_bkg"] >= args.min_neff_bin
        and r["raw_background"] >= args.min_raw_bkg_bin
        and r["raw_signal"] >= args.min_raw_sig_bin
    ]

    if len(stable_manual_rows) > 0:
        test_rows.append(
            summarize_combined_category_set(
                stable_manual_rows,
                selection="test_bdt_manual_combined_stable_bins",
                category_type="bdt_manual_combined",
            )
        )

    test_categories = pd.DataFrame(test_rows)

    # Background tables for BDT bins.
    group_bkg, sample_bkg = background_tables_for_categories(
        df_test,
        category_masks_for_backgrounds,
    )

    # Permutation importance.
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

    # Save artifacts.
    joblib.dump(model, outdir / "hh4b_resolved_bdt_v2_model.joblib")

    with open(outdir / "hh4b_resolved_bdt_v2_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    df_val[
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
    ].to_csv(outdir / "hh4b_resolved_bdt_v2_validation_scores.csv", index=False)

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
    ].to_csv(outdir / "hh4b_resolved_bdt_v2_test_scores.csv", index=False)

    val_cumulative_scan.to_csv(
        outdir / "hh4b_resolved_bdt_v2_validation_cumulative_scan.csv",
        index=False,
    )

    bin_scan.to_csv(
        outdir / "hh4b_resolved_bdt_v2_validation_bin_optimization_scan.csv",
        index=False,
    )

    test_categories.to_csv(
        outdir / "hh4b_resolved_bdt_v2_test_categories.csv",
        index=False,
    )

    group_bkg.to_csv(
        outdir / "hh4b_resolved_bdt_v2_backgrounds_by_process_group.csv",
        index=False,
    )

    sample_bkg.to_csv(
        outdir / "hh4b_resolved_bdt_v2_backgrounds_by_sample.csv",
        index=False,
    )

    imp.to_csv(
        outdir / "hh4b_resolved_bdt_v2_permutation_importance.csv",
        index=False,
    )

    # Write a short interpretation note.
    note = []
    note.append("# HH4b resolved BDT v2\n")
    note.append("This version removes binary cut-category flags from the training inputs and uses them only for evaluation.\n")
    note.append("It also scans validation-defined BDT score bins and applies the selected bins to the held-out test set.\n")
    note.append("\n## Main caveat\n")
    note.append("Rows with missing/non-finite proxy weights are excluded from training and weighted evaluation.\n")
    note.append("QCD normalization remains incomplete until sample-specific QCD metadata are available.\n")
    note.append("\n## Best validation bin configuration\n")
    note.append("```json\n")
    note.append(json.dumps(best_bin_config, indent=2))
    note.append("\n```\n")
    note.append("\n## Selected bins\n")
    note.append("```json\n")
    note.append(json.dumps(metrics["selected_bins"], indent=2))
    note.append("\n```\n")
    note.append("\n## Metrics\n")
    note.append("```json\n")
    note.append(json.dumps(metrics, indent=2))
    note.append("\n```\n")

    (outdir / "hh4b_resolved_bdt_v2_interpretation.md").write_text(
        "".join(note),
        encoding="utf-8",
    )

    print("\n[INFO] Metrics:")
    print(json.dumps(metrics, indent=2))

    print("\n[INFO] Validation cumulative scan top rows:")
    print(val_cumulative_scan.head(20).to_string(index=False))

    print("\n[INFO] Validation bin optimization top rows:")
    if len(bin_scan) > 0:
        print(bin_scan.head(20).to_string(index=False))
    else:
        print("[INFO] No multi-bin stable configurations found; used fallback cumulative threshold.")

    print("\n[INFO] Test categories:")
    cols = [
        "selection",
        "category_type",
        "score_low",
        "score_high",
        "raw_signal",
        "raw_background",
        "S_w",
        "B_w",
        "S_over_B",
        "S_over_sqrtSplusB",
        "asimov_Z_A",
        "N_eff_bkg",
        "twoB_over_S",
    ]
    print(test_categories[cols].to_string(index=False))

    print("\n[INFO] Backgrounds by process group:")
    if len(group_bkg) > 0:
        print(group_bkg.head(60).to_string(index=False))
    else:
        print("[INFO] No background table rows.")

    print("\n[INFO] Top permutation importances:")
    print(imp.head(30).to_string(index=False))

    print("\n[INFO] Wrote outputs to:", outdir)


if __name__ == "__main__":
    main()
