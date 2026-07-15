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

OUTDIR = REPO / "outputs/tables/ak4ak8_candidate_bdt_v1_2026_07_14"
PLOTDIR = REPO / "outputs/plots/ak4ak8_candidate_bdt_v1_2026_07_14"
OUTDIR.mkdir(parents=True, exist_ok=True)
PLOTDIR.mkdir(parents=True, exist_ok=True)

SAMPLES = [
    {
        "sample": "ggF_HH_ak4ak8_10k",
        "label": 1,
        "group": "signal",
        "paths": [STORE / "parquet/ggf_hh4b_ak4ak8_10k_hh4b_candidates.parquet"],
    },
    {
        "sample": "VBF_HH_ak4ak8_10k",
        "label": 1,
        "group": "signal",
        "paths": [STORE / "parquet/vbf_hh4b_ak4ak8_10k_hh4b_candidates.parquet"],
    },
    {
        "sample": "ttbar_ak4ak8_extra50k",
        "label": 0,
        "group": "ttbar",
        "paths": sorted(STORE.glob("parquet/ttbar_extra50k_ak4ak8_v1_shard*_hh4b_candidates.parquet")),
    },
    {
        "sample": "qcd_bbbb_ak4ak8_smoke10k",
        "label": 0,
        "group": "qcdlike",
        "paths": [STORE / "parquet/qcd_bbbb_ak4ak8_smoke10k_hh4b_candidates.parquet"],
    },
    {
        "sample": "zbbbb_ak4ak8_smoke10k",
        "label": 0,
        "group": "qcdlike",
        "paths": [STORE / "parquet/zbbbb_ak4ak8_smoke10k_hh4b_candidates.parquet"],
    },
]

frames = []
manifest_rows = []

for spec in SAMPLES:
    paths = [p for p in spec["paths"] if p.exists()]
    n_rows = 0

    for p in paths:
        df = pd.read_parquet(p)
        df = df.copy()
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
manifest.to_csv(OUTDIR / "ak4ak8_candidate_dataset_manifest.csv", index=False)
(OUTDIR / "ak4ak8_candidate_dataset_manifest.md").write_text(manifest.to_markdown(index=False) + "\n")

if not frames:
    raise RuntimeError("No candidate parquet files found.")

data = pd.concat(frames, ignore_index=True)

# Use numeric candidate-level features only. Exclude identifiers and labels.
exclude = {
    "event", "label", "weight", "xsec_pb", "source_file",
}
numeric_cols = []
for c in data.columns:
    if c in exclude:
        continue
    if pd.api.types.is_numeric_dtype(data[c]):
        numeric_cols.append(c)

# Remove columns that are constant or almost entirely missing.
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

# Balance signal/background in training by row count.
n_sig = max(int(np.sum(y == 1)), 1)
n_bkg = max(int(np.sum(y == 0)), 1)
base_weight = np.where(y == 1, 0.5 / n_sig, 0.5 / n_bkg)

# Also keep qcdlike and ttbar from being completely dominated by whichever has more rows.
for g in np.unique(groups[y == 0]):
    mask = (y == 0) & (groups == g)
    if mask.sum() > 0:
        base_weight[mask] *= (n_bkg / mask.sum()) / len(np.unique(groups[y == 0]))

dataset_summary = {
    "n_rows": int(len(data)),
    "n_signal_rows": int(np.sum(y == 1)),
    "n_background_rows": int(np.sum(y == 0)),
    "features": good_cols,
    "n_features": len(good_cols),
}
(OUTDIR / "ak4ak8_candidate_dataset_summary.json").write_text(json.dumps(dataset_summary, indent=2))

summary_rows = []
pred_frames = []

seeds = list(range(30))

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
            n_estimators=350,
            learning_rate=0.03,
            max_depth=3,
            subsample=0.8,
            random_state=seed,
        ),
    )

    model.fit(X.iloc[train_idx], y[train_idx], gradientboostingclassifier__sample_weight=base_weight[train_idx])

    score = model.predict_proba(X.iloc[test_idx])[:, 1]

    auc_all = roc_auc_score(y[test_idx], score)

    def auc_vs_group(group_name):
        mask = (y[test_idx] == 1) | ((y[test_idx] == 0) & (groups[test_idx] == group_name))
        if len(np.unique(y[test_idx][mask])) < 2:
            return np.nan
        return roc_auc_score(y[test_idx][mask], score[mask])

    auc_ttbar = auc_vs_group("ttbar")
    auc_qcdlike = auc_vs_group("qcdlike")

    # Stable threshold scan using unweighted row counts.
    best = None
    for thr in np.linspace(0.0, 1.0, 201):
        sel = score >= thr
        s = int(np.sum(sel & (y[test_idx] == 1)))
        b = int(np.sum(sel & (y[test_idx] == 0)))
        if b < 10:
            continue
        z = s / np.sqrt(b) if b > 0 else np.nan
        sb = s / b if b > 0 else np.nan
        row = {
            "seed": seed,
            "threshold": float(thr),
            "selected_signal_rows": s,
            "selected_background_rows": b,
            "S_over_sqrtB_rows": float(z),
            "S_over_B_rows": float(sb),
        }
        if best is None or row["S_over_sqrtB_rows"] > best["S_over_sqrtB_rows"]:
            best = row

    summary = {
        "seed": seed,
        "auc_all": float(auc_all),
        "auc_vs_ttbar": float(auc_ttbar),
        "auc_vs_qcdlike": float(auc_qcdlike),
    }
    if best:
        summary.update(best)

    summary_rows.append(summary)

    if seed == 0:
        pred = pd.DataFrame({
            "row_index": test_idx,
            "label": y[test_idx],
            "score": score,
            "analysis_sample": samples[test_idx],
            "analysis_group": groups[test_idx],
        })
        pred_frames.append(pred)

        fpr, tpr, _ = roc_curve(y[test_idx], score)
        plt.figure(figsize=(5, 5))
        plt.plot(fpr, tpr, label=f"AUC = {auc_all:.3f}")
        plt.plot([0, 1], [0, 1], linestyle="--")
        plt.xlabel("False positive rate")
        plt.ylabel("True positive rate")
        plt.title("AK4/AK8 candidate BDT pilot")
        plt.legend()
        plt.tight_layout()
        plt.savefig(PLOTDIR / "ak4ak8_candidate_bdt_roc_seed0.png", dpi=200)
        plt.savefig(PLOTDIR / "ak4ak8_candidate_bdt_roc_seed0.pdf")
        plt.close()

summary = pd.DataFrame(summary_rows)
summary.to_csv(OUTDIR / "ak4ak8_candidate_bdt_multiseed_summary.csv", index=False)
(OUTDIR / "ak4ak8_candidate_bdt_multiseed_summary.md").write_text(summary.to_markdown(index=False) + "\n")

agg = summary.agg(["mean", "std", "min", "max"]).reset_index().rename(columns={"index": "stat"})
agg.to_csv(OUTDIR / "ak4ak8_candidate_bdt_multiseed_aggregate.csv", index=False)
(OUTDIR / "ak4ak8_candidate_bdt_multiseed_aggregate.md").write_text(agg.to_markdown(index=False) + "\n")

if pred_frames:
    pred = pd.concat(pred_frames, ignore_index=True)
    pred.to_csv(OUTDIR / "ak4ak8_candidate_bdt_seed0_predictions.csv", index=False)

print("=== Manifest ===")
print(manifest.to_string(index=False))

print("\n=== Dataset summary ===")
print(json.dumps(dataset_summary, indent=2))

print("\n=== Multi-seed aggregate ===")
print(agg.to_string(index=False))

print("\nWrote:", OUTDIR)
print("Wrote:", PLOTDIR)
