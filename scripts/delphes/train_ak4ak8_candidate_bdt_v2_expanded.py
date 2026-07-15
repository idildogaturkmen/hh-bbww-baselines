#!/usr/bin/env python3

from pathlib import Path
import os
import json
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline

import matplotlib.pyplot as plt

STORE = Path(os.environ["HH4B_STORE"])
REPO = Path(os.environ["HH4B_REPO"])

OUTDIR = REPO / "outputs/tables/ak4ak8_candidate_bdt_v2_expanded_2026_07_15"
PLOTDIR = REPO / "outputs/plots/ak4ak8_candidate_bdt_v2_expanded_2026_07_15"
OUTDIR.mkdir(parents=True, exist_ok=True)
PLOTDIR.mkdir(parents=True, exist_ok=True)

SAMPLES = [
    {
        "sample": "ggF_HH_ak4ak8_10k",
        "group": "signal",
        "label": 1,
        "paths": [STORE / "parquet/ggf_hh4b_ak4ak8_10k_hh4b_candidates.parquet"],
    },
    {
        "sample": "VBF_HH_ak4ak8_10k",
        "group": "signal",
        "label": 1,
        "paths": [STORE / "parquet/vbf_hh4b_ak4ak8_10k_hh4b_candidates.parquet"],
    },
    {
        "sample": "ttbar_ak4ak8_extra50k",
        "group": "ttbar",
        "label": 0,
        "paths": sorted((STORE / "parquet").glob("ttbar_extra50k_ak4ak8_v1_shard*_hh4b_candidates.parquet")),
    },
    {
        "sample": "qcd_bbbb_ak4ak8_extra50k",
        "group": "qcd_bbbb",
        "label": 0,
        "paths": sorted((STORE / "parquet").glob("qcd_bbbb_ak4ak8_extra50k_v1_shard*_hh4b_candidates.parquet")),
    },
    {
        "sample": "zbbbb_ak4ak8_extra50k",
        "group": "zbbbb",
        "label": 0,
        "paths": sorted((STORE / "parquet").glob("zbbbb_ak4ak8_extra50k_v1_shard*_hh4b_candidates.parquet")),
    },
]

for spec in SAMPLES:
    if "paths" not in spec:
        raise KeyError(f"Malformed sample spec without paths: {spec}")

frames = []
manifest_rows = []

for spec in SAMPLES:
    paths = [p for p in spec["paths"] if p.exists()]
    n_rows = 0

    for p in paths:
        df = pd.read_parquet(p).copy()
        df["analysis_sample"] = spec["sample"]
        df["analysis_group"] = spec["group"]
        df["label"] = spec["label"]
        df["source_file"] = p.name
        frames.append(df)
        n_rows += len(df)

    manifest_rows.append({
        "sample": spec["sample"],
        "group": spec["group"],
        "label": spec["label"],
        "n_files_found": len(paths),
        "n_candidate_rows": n_rows,
        "files": ";".join(str(p) for p in paths),
    })

manifest = pd.DataFrame(manifest_rows)
manifest.to_csv(OUTDIR / "ak4ak8_candidate_dataset_manifest_v2.csv", index=False)
(OUTDIR / "ak4ak8_candidate_dataset_manifest_v2.md").write_text(manifest.to_markdown(index=False) + "\n")

if not frames:
    raise RuntimeError("No candidate parquet files found.")

data = pd.concat(frames, ignore_index=True)

exclude = {
    "event",
    "label",
    "weight",
    "xsec_pb",
    "source_file",
}
numeric_cols = []
for c in data.columns:
    if c in exclude:
        continue
    if pd.api.types.is_numeric_dtype(data[c]):
        numeric_cols.append(c)

good_cols = []
for c in numeric_cols:
    s = data[c]
    if s.notna().mean() < 0.5:
        continue
    if s.nunique(dropna=True) <= 1:
        continue
    good_cols.append(c)

X = data[good_cols]
y = data["label"].astype(int).values
groups = data["analysis_group"].values
samples = data["analysis_sample"].values

n_sig = max(int(np.sum(y == 1)), 1)
n_bkg = max(int(np.sum(y == 0)), 1)

w = np.where(y == 1, 0.5 / n_sig, 0.5 / n_bkg)

bkg_groups = sorted(set(groups[y == 0]))
for g in bkg_groups:
    mask = (y == 0) & (groups == g)
    if mask.sum() > 0:
        w[mask] *= (n_bkg / mask.sum()) / len(bkg_groups)

dataset_summary = {
    "n_rows": int(len(data)),
    "n_signal_rows": int(np.sum(y == 1)),
    "n_background_rows": int(np.sum(y == 0)),
    "n_ttbar_rows": int(np.sum(groups == "ttbar")),
    "n_qcd_bbbb_rows": int(np.sum(groups == "qcd_bbbb")),
    "n_zbbbb_rows": int(np.sum(groups == "zbbbb")),
    "n_features": len(good_cols),
    "features": good_cols,
}
(OUTDIR / "ak4ak8_candidate_dataset_summary_v2.json").write_text(json.dumps(dataset_summary, indent=2))

summary_rows = []
pred_seed0 = None
feature_importance_seed0 = None

seeds = list(range(150))

for seed in seeds:
    idx = np.arange(len(data))
    train_idx, test_idx = train_test_split(
        idx,
        test_size=0.35,
        random_state=seed,
        stratify=y,
    )

    model = make_pipeline(
        SimpleImputer(strategy="median"),
        GradientBoostingClassifier(
            n_estimators=450,
            learning_rate=0.025,
            max_depth=3,
            subsample=0.8,
            random_state=seed,
        ),
    )

    model.fit(
        X.iloc[train_idx],
        y[train_idx],
        gradientboostingclassifier__sample_weight=w[train_idx],
    )

    score = model.predict_proba(X.iloc[test_idx])[:, 1]
    auc_all = roc_auc_score(y[test_idx], score)

    def auc_vs_group(group_name):
        mask = (y[test_idx] == 1) | ((y[test_idx] == 0) & (groups[test_idx] == group_name))
        if len(np.unique(y[test_idx][mask])) < 2:
            return np.nan
        return roc_auc_score(y[test_idx][mask], score[mask])

    row = {
        "seed": seed,
        "auc_all": float(auc_all),
        "auc_vs_ttbar": float(auc_vs_group("ttbar")),
        "auc_vs_qcd_bbbb": float(auc_vs_group("qcd_bbbb")),
        "auc_vs_zbbbb": float(auc_vs_group("zbbbb")),
    }

    best = None
    for thr in np.linspace(0.0, 1.0, 201):
        sel = score >= thr
        s = int(np.sum(sel & (y[test_idx] == 1)))
        b = int(np.sum(sel & (y[test_idx] == 0)))
        b_tt = int(np.sum(sel & (y[test_idx] == 0) & (groups[test_idx] == "ttbar")))
        b_qcd = int(np.sum(sel & (y[test_idx] == 0) & (groups[test_idx] == "qcd_bbbb")))
        b_zbb = int(np.sum(sel & (y[test_idx] == 0) & (groups[test_idx] == "zbbbb")))

        if b < 25:
            continue

        cand = {
            "threshold": float(thr),
            "selected_signal_rows": s,
            "selected_background_rows": b,
            "selected_ttbar_rows": b_tt,
            "selected_qcd_bbbb_rows": b_qcd,
            "selected_zbbbb_rows": b_zbb,
            "S_over_sqrtB_rows": float(s / np.sqrt(b)) if b > 0 else np.nan,
            "S_over_B_rows": float(s / b) if b > 0 else np.nan,
        }

        if best is None or cand["S_over_sqrtB_rows"] > best["S_over_sqrtB_rows"]:
            best = cand

    if best:
        row.update(best)

    summary_rows.append(row)

    if seed == 0:
        pred_seed0 = pd.DataFrame({
            "row_index": test_idx,
            "label": y[test_idx],
            "score": score,
            "analysis_sample": samples[test_idx],
            "analysis_group": groups[test_idx],
        })

        gb = model.named_steps["gradientboostingclassifier"]
        feature_importance_seed0 = pd.DataFrame({
            "feature": good_cols,
            "importance": gb.feature_importances_,
        }).sort_values("importance", ascending=False)

        fpr, tpr, _ = roc_curve(y[test_idx], score)
        plt.figure(figsize=(5, 5))
        plt.plot(fpr, tpr, label=f"AUC = {auc_all:.3f}")
        plt.plot([0, 1], [0, 1], linestyle="--")
        plt.xlabel("False positive rate")
        plt.ylabel("True positive rate")
        plt.title("AK4/AK8 expanded candidate BDT v2")
        plt.legend()
        plt.tight_layout()
        plt.savefig(PLOTDIR / "ak4ak8_candidate_bdt_v2_roc_seed0.png", dpi=200)
        plt.savefig(PLOTDIR / "ak4ak8_candidate_bdt_v2_roc_seed0.pdf")
        plt.close()

summary = pd.DataFrame(summary_rows)
summary.to_csv(OUTDIR / "ak4ak8_candidate_bdt_v2_multiseed_summary.csv", index=False)
(OUTDIR / "ak4ak8_candidate_bdt_v2_multiseed_summary.md").write_text(summary.to_markdown(index=False) + "\n")

agg = summary.agg(["mean", "std", "min", "max"]).reset_index().rename(columns={"index": "stat"})
agg.to_csv(OUTDIR / "ak4ak8_candidate_bdt_v2_multiseed_aggregate.csv", index=False)
(OUTDIR / "ak4ak8_candidate_bdt_v2_multiseed_aggregate.md").write_text(agg.to_markdown(index=False) + "\n")

if feature_importance_seed0 is not None:
    feature_importance_seed0.to_csv(OUTDIR / "ak4ak8_candidate_bdt_v2_feature_importance_seed0.csv", index=False)
    (OUTDIR / "ak4ak8_candidate_bdt_v2_feature_importance_seed0.md").write_text(
        feature_importance_seed0.head(30).to_markdown(index=False) + "\n"
    )

if pred_seed0 is not None:
    seed0 = summary.loc[summary["seed"] == 0].iloc[0]
    thr = float(seed0["threshold"])
    pred_seed0["selected_at_seed0_best"] = pred_seed0["score"] >= thr
    pred_seed0.to_csv(OUTDIR / "ak4ak8_candidate_bdt_v2_seed0_predictions.csv", index=False)

    score_rows = []
    for key in ["analysis_group", "analysis_sample"]:
        g = (
            pred_seed0.groupby(key)
            .agg(
                n_rows=("score", "count"),
                n_selected=("selected_at_seed0_best", "sum"),
                mean_score=("score", "mean"),
                median_score=("score", "median"),
                p90_score=("score", lambda x: x.quantile(0.90)),
                p99_score=("score", lambda x: x.quantile(0.99)),
            )
            .reset_index()
        )
        g["selection_fraction"] = g["n_selected"] / g["n_rows"]
        g.insert(0, "grouping", key)
        g = g.rename(columns={key: "category"})
        score_rows.append(g)

    inspection = pd.concat(score_rows, ignore_index=True)
    inspection.to_csv(OUTDIR / "ak4ak8_candidate_bdt_v2_seed0_score_inspection.csv", index=False)
    (OUTDIR / "ak4ak8_candidate_bdt_v2_seed0_score_inspection.md").write_text(inspection.to_markdown(index=False) + "\n")

    selected = pred_seed0[pred_seed0["selected_at_seed0_best"]]
    comp = selected.groupby(["analysis_group", "analysis_sample"]).size().reset_index(name="n_selected")
    comp.to_csv(OUTDIR / "ak4ak8_candidate_bdt_v2_seed0_selected_composition.csv", index=False)
    (OUTDIR / "ak4ak8_candidate_bdt_v2_seed0_selected_composition.md").write_text(comp.to_markdown(index=False) + "\n")

    plt.figure(figsize=(7, 5))
    for group, df in pred_seed0.groupby("analysis_group"):
        plt.hist(df["score"], bins=40, histtype="step", density=True, label=group)
    plt.axvline(thr, linestyle="--", label=f"seed0 threshold = {thr:.3f}")
    plt.xlabel("BDT score")
    plt.ylabel("Normalized density")
    plt.title("AK4/AK8 expanded candidate BDT v2: score by group")
    plt.legend()
    plt.tight_layout()
    plt.savefig(PLOTDIR / "ak4ak8_candidate_bdt_v2_score_by_group_seed0.png", dpi=200)
    plt.savefig(PLOTDIR / "ak4ak8_candidate_bdt_v2_score_by_group_seed0.pdf")
    plt.close()

print("=== Manifest ===")
print(manifest.to_string(index=False))

print("\n=== Dataset summary ===")
print(json.dumps(dataset_summary, indent=2))

print("\n=== Multi-seed aggregate ===")
print(agg.to_string(index=False))

if pred_seed0 is not None:
    print("\n=== Seed 0 score inspection ===")
    print(inspection.to_string(index=False))
    print("\n=== Seed 0 selected composition ===")
    print(comp.to_string(index=False))

if feature_importance_seed0 is not None:
    print("\n=== Top feature importances ===")
    print(feature_importance_seed0.head(30).to_string(index=False))

print("\nWrote:", OUTDIR)
print("Wrote:", PLOTDIR)
