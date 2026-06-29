'''
Builds a BDT classifier with rough-weighted training, and compares to cut-based baselines.
Uses HistGradientBoostingClassifier from sklearn, which is a fast implementation of gradient boosting on decision trees.

1. Train rough-weighted BDT.
2. Make ROC curves: raw and rough-weighted.
3. Overlay cut-based baseline points on the ROC curve.
4. Compare cut baselines for uncorrected/global/pT-binned m_bb.
5. Save a CSV table with raw and weighted S, B, S/B, S/(S+B), Asimov Z.
6. Still exclude QCD/gamma/minbias/upsilon because they lack safe metadata.
'''
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import (
    average_precision_score,
    auc,
    roc_auc_score,
    roc_curve,
)


DROP_FEATURES = {
    "n_leptons",
    "b1_btag",
    "b2_btag",
    "n_electrons",
}


MASS_VARIANTS = [
    "mbb_uncorrected",
    "mbb_global_median_scaled",
    "mbb_pt_binned_scaled",
]


MASS_WINDOWS = [
    (70, 160),
    (80, 150),
    (90, 140),
    (95, 145),
    (100, 140),
    (100, 150),
    (105, 145),
]


def asimov_z(s: float, b: float) -> float:
    if s <= 0 or b <= 0:
        return 0.0
    return float(np.sqrt(2.0 * ((s + b) * np.log(1.0 + s / b) - s)))


def class_balanced_training_weights(y: np.ndarray) -> np.ndarray:
    n_sig = np.sum(y == 1)
    n_bkg = np.sum(y == 0)
    w = np.ones_like(y, dtype=float)
    w[y == 1] = 0.5 * len(y) / max(n_sig, 1)
    w[y == 0] = 0.5 * len(y) / max(n_bkg, 1)
    return w / np.mean(w)


def weighted_neff(w: pd.Series) -> float:
    w = np.asarray(w, dtype=float)
    if len(w) == 0 or np.sum(w * w) <= 0:
        return 0.0
    return float(np.sum(w) ** 2 / np.sum(w * w))


def weighted_counts(df: pd.DataFrame, mask: pd.Series | np.ndarray) -> dict:
    sub = df[mask]
    sig = sub[sub["target"] == 1]
    bkg = sub[sub["target"] == 0]

    s_raw = int(len(sig))
    b_raw = int(len(bkg))
    s_w = float(sig["physics_weight_nominal"].sum())
    b_w = float(bkg["physics_weight_nominal"].sum())

    total_sig_raw = int((df["target"] == 1).sum())
    total_bkg_raw = int((df["target"] == 0).sum())
    total_sig_w = float(df.loc[df["target"] == 1, "physics_weight_nominal"].sum())
    total_bkg_w = float(df.loc[df["target"] == 0, "physics_weight_nominal"].sum())

    return {
        "S_raw": s_raw,
        "B_raw": b_raw,
        "S_weighted": s_w,
        "B_weighted": b_w,
        "S_over_B_raw": s_raw / b_raw if b_raw > 0 else np.nan,
        "S_over_SplusB_raw": s_raw / (s_raw + b_raw) if (s_raw + b_raw) > 0 else np.nan,
        "asimovZ_raw": asimov_z(s_raw, b_raw),
        "S_over_B_weighted": s_w / b_w if b_w > 0 else np.nan,
        "S_over_SplusB_weighted": s_w / (s_w + b_w) if (s_w + b_w) > 0 else np.nan,
        "asimovZ_weighted": asimov_z(s_w, b_w),
        "signal_eff_raw": s_raw / max(total_sig_raw, 1),
        "background_eff_raw": b_raw / max(total_bkg_raw, 1),
        "signal_eff_weighted": s_w / max(total_sig_w, 1e-12),
        "background_eff_weighted": b_w / max(total_bkg_w, 1e-12),
        "background_neff": weighted_neff(bkg["physics_weight_nominal"]),
    }


def make_cut_baseline_table(df: pd.DataFrame, outdir: Path) -> pd.DataFrame:
    rows = []

    for split_name in ["val", "test", "valtest", "all"]:
        if split_name == "valtest":
            sub = df[df["split"].isin(["val", "test"])].copy()
        elif split_name == "all":
            sub = df.copy()
        else:
            sub = df[df["split"] == split_name].copy()

        for mass_col in MASS_VARIANTS:
            if mass_col not in sub.columns:
                continue

            for low, high in MASS_WINDOWS:
                mask = (
                    (sub["ht"] >= 100)
                    & (sub[mass_col] >= low)
                    & (sub[mass_col] <= high)
                )

                row = {
                    "split": split_name,
                    "selection": "HT>=100",
                    "mass_variant": mass_col,
                    "window_low": low,
                    "window_high": high,
                }
                row.update(weighted_counts(sub, mask))
                rows.append(row)

    tab = pd.DataFrame(rows)
    tab.to_csv(outdir / "cut_baseline_mass_variant_comparison.csv", index=False)

    # Best validation choice per mass variant, then report nearest same cut on test.
    best_val = (
        tab[tab["split"] == "val"]
        .sort_values("asimovZ_weighted", ascending=False)
        .groupby("mass_variant", as_index=False)
        .head(1)
        .copy()
    )
    best_val.to_csv(outdir / "cut_baseline_best_validation_by_mass_variant.csv", index=False)

    return tab


def scan_thresholds(df: pd.DataFrame, split: str) -> pd.DataFrame:
    sub = df[df["split"] == split].copy()

    thresholds = np.unique(
        np.concatenate(
            [
                np.linspace(0, 1, 201),
                np.quantile(sub["bdt_score"], np.linspace(0, 1, 201)),
            ]
        )
    )

    rows = []
    for thr in thresholds:
        mask = sub["bdt_score"] >= thr
        row = {"split": split, "threshold": float(thr)}
        row.update(weighted_counts(sub, mask))
        rows.append(row)

    return pd.DataFrame(rows)


def score_bin_table(df: pd.DataFrame) -> pd.DataFrame:
    bins = np.linspace(0, 1, 21)
    rows = []

    for lo, hi in zip(bins[:-1], bins[1:]):
        if hi == 1:
            mask = (df["bdt_score"] >= lo) & (df["bdt_score"] <= hi)
        else:
            mask = (df["bdt_score"] >= lo) & (df["bdt_score"] < hi)

        row = {"score_low": lo, "score_high": hi}
        row.update(weighted_counts(df, mask))
        rows.append(row)

    return pd.DataFrame(rows)


def make_roc_outputs(df: pd.DataFrame, cut_tab: pd.DataFrame, outdir: Path) -> None:
    valtest = df[df["split"].isin(["val", "test"])].copy()
    y = valtest["target"].to_numpy()
    score = valtest["bdt_score"].to_numpy()
    w = valtest["physics_weight_nominal"].to_numpy()

    fpr_raw, tpr_raw, _ = roc_curve(y, score)
    fpr_w, tpr_w, _ = roc_curve(y, score, sample_weight=w)

    roc_raw_auc = auc(fpr_raw, tpr_raw)
    roc_w_auc = auc(fpr_w, tpr_w)

    pd.DataFrame(
        {"fpr_raw": fpr_raw, "tpr_raw": tpr_raw}
    ).to_csv(outdir / "roc_curve_raw_valtest.csv", index=False)

    pd.DataFrame(
        {"fpr_weighted": fpr_w, "tpr_weighted": tpr_w}
    ).to_csv(outdir / "roc_curve_weighted_valtest.csv", index=False)

    # Plot raw ROC with cut baseline points.
    plt.figure(figsize=(7.2, 5.4))
    plt.plot(fpr_raw, tpr_raw, label=f"BDT raw ROC, AUC={roc_raw_auc:.3f}")

    plt.xlabel("Background efficiency")
    plt.ylabel("Signal efficiency")
    plt.title("Raw BDT ROC, validation+test")
    plt.legend()
    plt.tight_layout()
    plt.savefig(outdir / "roc_raw_clean_valtest.png", dpi=200)
    plt.close()

    # Plot weighted ROC with cut baseline points.
    plt.figure(figsize=(7.2, 5.4))
    plt.plot(fpr_w, tpr_w, label=f"BDT weighted ROC, AUC={roc_w_auc:.3f}")

    plt.xlabel("Weighted background efficiency")
    plt.ylabel("Weighted signal efficiency")
    plt.title("Rough-weighted BDT ROC, validation+test")
    plt.legend()
    plt.tight_layout()
    plt.savefig(outdir / "roc_weighted_clean_valtest.png", dpi=200)
    plt.close()


def plot_score_shapes(df: pd.DataFrame, outdir: Path) -> None:
    bins = np.linspace(0, 1, 41)

    plt.figure(figsize=(7.2, 5.2))
    plt.hist(df.loc[df["target"] == 1, "bdt_score"], bins=bins, histtype="step", density=True, linewidth=2, label="Signal")
    plt.hist(df.loc[df["target"] == 0, "bdt_score"], bins=bins, histtype="step", density=True, linewidth=2, label="Background")
    plt.xlabel("BDT score")
    plt.ylabel("Unit-normalized events")
    plt.title("BDT score shape, validation+test")
    plt.legend()
    plt.tight_layout()
    plt.savefig(outdir / "bdt_score_normalized_shape_valtest.png", dpi=200)
    plt.close()

    groups = [g for g in sorted(df["process_group"].unique()) if g != "signal"]

    plt.figure(figsize=(8.5, 5.5))
    values = [df.loc[df["process_group"] == g, "bdt_score"] for g in groups]
    weights = [df.loc[df["process_group"] == g, "physics_weight_nominal"] for g in groups]
    plt.hist(values, bins=bins, weights=weights, stacked=True, label=groups)

    sig = df[df["target"] == 1]
    plt.hist(
        sig["bdt_score"],
        bins=bins,
        weights=sig["physics_weight_nominal"],
        histtype="step",
        linewidth=2,
        label="signal",
    )

    plt.xlabel("BDT score")
    plt.ylabel("Expected events, rough weighted")
    plt.title("Rough-weighted BDT score distribution")
    plt.legend(fontsize=7, ncol=2)
    plt.tight_layout()
    plt.savefig(outdir / "bdt_score_rough_weighted_stack_valtest.png", dpi=200)
    plt.close()


def plot_scans(scan: pd.DataFrame, outdir: Path) -> None:
    plt.figure(figsize=(7.2, 5.2))
    for split in ["val", "test"]:
        sub = scan[scan["split"] == split]
        plt.plot(sub["threshold"], sub["asimovZ_weighted"], label=split)
    plt.xlabel("BDT score threshold")
    plt.ylabel("Rough-weighted Asimov-Z proxy")
    plt.title("Rough-weighted Z proxy vs BDT threshold")
    plt.legend()
    plt.tight_layout()
    plt.savefig(outdir / "bdt_threshold_scan_weighted_asimovZ.png", dpi=200)
    plt.close()

    plt.figure(figsize=(7.2, 5.2))
    for split in ["val", "test"]:
        sub = scan[scan["split"] == split]
        plt.plot(sub["threshold"], sub["S_over_SplusB_weighted"], label=split)
    plt.xlabel("BDT score threshold")
    plt.ylabel("Rough-weighted S/(S+B)")
    plt.title("Rough-weighted S/(S+B) vs BDT threshold")
    plt.legend()
    plt.tight_layout()
    plt.savefig(outdir / "bdt_threshold_scan_weighted_SoverSplusB.png", dpi=200)
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="outputs/physics_weights_lepemu_rough/onelep_pre_mbb_classifier_input_with_weights.parquet")
    parser.add_argument("--features", default="outputs/classifier_input_lepemu/feature_columns.json")
    parser.add_argument("--outdir", default="outputs/bdt_weighted_baseline_lepemu_rough")
    args = parser.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    df0 = pd.read_parquet(args.input)

    df = df0[df0["has_physics_weight"] & np.isfinite(df0["physics_weight_nominal"])].copy()

    print("Input events:", len(df0))
    print("Events used:", len(df))
    print("Excluded missing/unsafe weights:", len(df0) - len(df))

    with open(args.features) as f:
        features = json.load(f)

    features = [
        c for c in features
        if c in df.columns
        and c not in DROP_FEATURES
        and df[c].nunique(dropna=False) > 1
        and pd.api.types.is_numeric_dtype(df[c])
    ]

    with open(outdir / "bdt_feature_columns.json", "w") as f:
        json.dump(features, f, indent=2)

    train = df[df["split"] == "train"].copy()
    val = df[df["split"] == "val"].copy()
    test = df[df["split"] == "test"].copy()

    model = HistGradientBoostingClassifier(
        max_iter=300,
        learning_rate=0.05,
        max_leaf_nodes=31,
        l2_regularization=0.01,
        random_state=12345,
    )

    X_train = train[features]
    y_train = train["target"].to_numpy()

    model.fit(X_train, y_train, sample_weight=class_balanced_training_weights(y_train))

    metrics = []
    for name, sub in [("train", train), ("val", val), ("test", test)]:
        score = model.predict_proba(sub[features])[:, 1]
        df.loc[sub.index, "bdt_score"] = score

        auc_raw = roc_auc_score(sub["target"], score)
        ap_raw = average_precision_score(sub["target"], score)

        auc_weighted = roc_auc_score(
            sub["target"],
            score,
            sample_weight=sub["physics_weight_nominal"],
        )

        metrics.append(
            {
                "split": name,
                "roc_auc_raw": auc_raw,
                "roc_auc_weighted": auc_weighted,
                "average_precision_raw": ap_raw,
            }
        )

        print(
            f"{name:5s}: raw ROC AUC = {auc_raw:.4f}, "
            f"weighted ROC AUC = {auc_weighted:.4f}, "
            f"average precision = {ap_raw:.4f}"
        )

    pd.DataFrame(metrics).to_csv(outdir / "bdt_classification_metrics.csv", index=False)
    df.to_parquet(outdir / "onelep_pre_mbb_with_bdt_score_rough_weighted.parquet", index=False)

    cut_tab = make_cut_baseline_table(df, outdir)

    scan = pd.concat([scan_thresholds(df, "val"), scan_thresholds(df, "test")], ignore_index=True)
    scan.to_csv(outdir / "bdt_threshold_scan_rough_weighted.csv", index=False)

    valtest = df[df["split"].isin(["val", "test"])].copy()
    bins = score_bin_table(valtest)
    bins.to_csv(outdir / "bdt_score_bin_table_rough_weighted_valtest.csv", index=False)

    process_summary = (
        valtest.groupby("process_group")
        .agg(
            raw_events=("target", "size"),
            weighted_events=("physics_weight_nominal", "sum"),
            neff=("physics_weight_nominal", weighted_neff),
        )
        .reset_index()
        .sort_values("weighted_events", ascending=False)
    )
    process_summary.to_csv(outdir / "weighted_process_summary_valtest.csv", index=False)

    make_roc_outputs(df, cut_tab, outdir)
    plot_score_shapes(valtest, outdir)
    plot_scans(scan, outdir)

    best_val = scan[scan["split"] == "val"].sort_values("asimovZ_weighted", ascending=False).iloc[0]
    test_scan = scan[scan["split"] == "test"].copy()
    nearest_idx = (test_scan["threshold"] - best_val["threshold"]).abs().idxmin()
    test_at_best = test_scan.loc[nearest_idx]

    summary = {
        "n_input_events": int(len(df0)),
        "n_used_events": int(len(df)),
        "n_excluded_missing_or_unsafe_weight": int(len(df0) - len(df)),
        "features": features,
        "best_validation_threshold": float(best_val["threshold"]),
        "best_validation_row": best_val.to_dict(),
        "test_nearest_best_validation_threshold": test_at_best.to_dict(),
        "caveat": "Rough weighted result only. QCD/gamma/minbias/upsilon excluded pending metadata. Background xsecs are approximate proxies.",
    }

    with open(outdir / "bdt_rough_weighted_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print("\n=== Cut baseline comparison, validation+test, 80-150 ===")
    print(
        cut_tab[
            (cut_tab["split"] == "valtest")
            & (cut_tab["window_low"] == 80)
            & (cut_tab["window_high"] == 150)
        ][
            [
                "mass_variant",
                "S_raw",
                "B_raw",
                "S_weighted",
                "B_weighted",
                "S_over_B_weighted",
                "S_over_SplusB_weighted",
                "asimovZ_weighted",
                "background_neff",
            ]
        ].to_string(index=False)
    )

    print("\n=== Weighted process summary, validation+test ===")
    print(process_summary.to_string(index=False))

    print("\n=== Best validation threshold, rough weighted ===")
    print(best_val.to_string())

    print("\n=== Test row nearest best validation threshold ===")
    print(test_at_best.to_string())

    print("\nWrote outputs to:", outdir)


if __name__ == "__main__":
    main()