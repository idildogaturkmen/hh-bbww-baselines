'''
Adds event-level features from two BDTs trained to separate signal from QCD and top backgrounds, respectively. The BDTs are trained on the same training set, and then applied to the test set. The resulting scores are used to scan over a grid of thresholds to find the best stable rectangle in the 2D score space.
The script runs the training and evaluation for multiple random seeds, and saves the results to CSV and Markdown files in the outputs directory.

Features added:
- `bdt_qcd_score`: score from the BDT trained to separate signal from QCD background
- `bdt_top_score`: score from the BDT trained to separate signal from top background
- `S_over_B`: signal over background ratio for the selected events
- `S_over_sqrtB`: signal over square root of background for the selected events
The script requires the following environment variables to be set:
- `HH4B_REPO`: path to the hh-bbww-baselines repository
'''

import importlib.util
import os
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split


STORE = Path(os.environ["HH4B_STORE"])
REPO = Path(os.environ["HH4B_REPO"])
BASE_BDT_SCRIPT = REPO / "scripts/delphes/train_sm_normalized_hh4b_bdt.py"

OUTDIR = REPO / "outputs/tables/sm_normalized_hh4b_two_bdt_event_features_2026_07_09"
OUTDIR.mkdir(parents=True, exist_ok=True)

LUMI_PB = 450.0 * 1000.0
TEST_SIZE = 0.35
RANDOM_STATE = 42

spec = importlib.util.spec_from_file_location("bdtmod", BASE_BDT_SCRIPT)
bdtmod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(bdtmod)

df, base_features = bdtmod.load_all()
df = df.copy()


def find_one(patterns):
    matches = []
    for pat in patterns:
        matches.extend(sorted((STORE / "parquet").glob(pat)))
    matches = [p for p in matches if "test" not in p.name.lower()]
    if not matches:
        raise FileNotFoundError(f"No match for {patterns}")
    return matches[-1]


def event_path_for_sample(sample):
    if sample == "ggF_HH4b_SMnorm":
        return find_one(["HH4b_ggf_hh4b_10k*merged*event_summary.parquet"])
    if sample == "VBF_HH4b_SMnorm":
        return find_one(["HH4b_vbf_hh4b_10k*merged*event_summary.parquet"])
    if sample == "ttbar_200k":
        return find_one(["ttbar_200k_merged_event_summary.parquet"])
    if sample == "Zbbbb_100k":
        return find_one(["zbbbb*100k*merged_event_summary.parquet", "Zbbbb*100k*merged_event_summary.parquet"])
    if sample.startswith("qcd_"):
        if sample.endswith("combined120k"):
            return find_one([f"{sample}_merged_event_summary.parquet"])
        return find_one([f"{sample}_event_summary.parquet", f"{sample}_merged_event_summary.parquet"])
    raise ValueError(f"Unknown sample {sample}")


event_frames = []
for sample in sorted(df["sample"].unique()):
    p = event_path_for_sample(sample)
    ev = pd.read_parquet(p)
    keep = ["event"]

    for c in [
        "n_jet_pt30_eta25",
        "n_bjet_pt30_eta25",
        "ht_pt30_eta25",
        "event_cross_section_pb",
    ]:
        if c in ev.columns:
            keep.append(c)

    ev = ev[keep].copy()
    ev["sample"] = sample
    event_frames.append(ev)

event_meta = pd.concat(event_frames, ignore_index=True)

df = df.merge(event_meta, on=["sample", "event"], how="left")

# Robust fallbacks if a column is missing in an old sample.
if "n_jet_pt30_eta25" not in df.columns:
    df["n_jet_pt30_eta25"] = 4.0
if "n_bjet_pt30_eta25" not in df.columns:
    df["n_bjet_pt30_eta25"] = df.get("n_selected_bjets", 4.0)
if "ht_pt30_eta25" not in df.columns:
    df["ht_pt30_eta25"] = df["pt_sum4"]

df["n_extra_jets_pt30_eta25"] = df["n_jet_pt30_eta25"] - 4.0
df["n_extra_bjets_pt30_eta25"] = df["n_bjet_pt30_eta25"] - 4.0
df["ht_over_mhh"] = df["ht_pt30_eta25"] / (df["mhh"] + 1e-9)

event_features = [
    "n_jet_pt30_eta25",
    "n_bjet_pt30_eta25",
    "n_extra_jets_pt30_eta25",
    "n_extra_bjets_pt30_eta25",
    "ht_pt30_eta25",
    "ht_over_mhh",
]

feature_cols = base_features + event_features

df = df.replace([np.inf, -np.inf], np.nan).dropna(subset=feature_cols)

df["is_signal"] = df["group"].eq("signal")
df["is_qcd"] = df["sample"].str.startswith("qcd_")
df["is_top"] = df["sample"].str.contains("ttbar|ttbb|tt_", case=False, regex=True)

train_df, test_df = train_test_split(
    df,
    test_size=TEST_SIZE,
    random_state=RANDOM_STATE,
    stratify=df["target"],
)

total_counts = df.groupby("sample").size()
test_counts = test_df.groupby("sample").size()

test_df = test_df.copy()
test_df["weight_pb_scaled_to_full"] = test_df["weight_pb"] * test_df["sample"].map(
    lambda s: total_counts.loc[s] / test_counts.loc[s]
)


def class_balanced_weights(y):
    y = np.asarray(y).astype(int)
    n_pos = max((y == 1).sum(), 1)
    n_neg = max((y == 0).sum(), 1)
    return np.where(y == 1, 0.5 / n_pos, 0.5 / n_neg)


def neff(weights):
    w = np.asarray(weights, dtype=float)
    if len(w) == 0 or np.sum(w * w) <= 0:
        return 0.0
    return float((np.sum(w) ** 2) / np.sum(w * w))


def train_task(mask_col, label):
    task_train = train_df[train_df["is_signal"] | train_df[mask_col]].copy()
    task_test = test_df[test_df["is_signal"] | test_df[mask_col]].copy()

    y_train = task_train["is_signal"].astype(int)
    y_test = task_test["is_signal"].astype(int)

    clf = GradientBoostingClassifier(
        n_estimators=250,
        learning_rate=0.04,
        max_depth=3,
        subsample=0.8,
        random_state=RANDOM_STATE,
    )

    clf.fit(
        task_train[feature_cols],
        y_train,
        sample_weight=class_balanced_weights(y_train),
    )

    task_test = task_test.copy()
    task_test["score"] = clf.predict_proba(task_test[feature_cols])[:, 1]

    auc = roc_auc_score(y_test, task_test["score"])
    wauc = roc_auc_score(
        y_test,
        task_test["score"],
        sample_weight=task_test["weight_pb_scaled_to_full"],
    )

    all_scores = clf.predict_proba(test_df[feature_cols])[:, 1]

    imp = pd.DataFrame({
        "feature": feature_cols,
        f"importance_{label}": clf.feature_importances_,
    }).sort_values(f"importance_{label}", ascending=False)

    return clf, all_scores, auc, wauc, imp, len(task_train), len(task_test)


qcd_clf, qcd_scores, qcd_auc, qcd_wauc, qcd_imp, qcd_ntrain, qcd_ntest = train_task("is_qcd", "qcd")
top_clf, top_scores, top_auc, top_wauc, top_imp, top_ntrain, top_ntest = train_task("is_top", "top")

test_df["bdt_qcd_score"] = qcd_scores
test_df["bdt_top_score"] = top_scores

auc_summary = pd.DataFrame([
    {
        "classifier": "BDT_QCD_event_features",
        "negative_class": "QCD bbbb HT slices",
        "unweighted_auc": qcd_auc,
        "physics_weighted_auc": qcd_wauc,
        "n_train_task": qcd_ntrain,
        "n_test_task": qcd_ntest,
    },
    {
        "classifier": "BDT_top_event_features",
        "negative_class": "ttbar / top-like backgrounds",
        "unweighted_auc": top_auc,
        "physics_weighted_auc": top_wauc,
        "n_train_task": top_ntrain,
        "n_test_task": top_ntest,
    },
])

thresholds_qcd = np.round(np.arange(0.50, 0.951, 0.025), 3)
thresholds_top = np.round(np.arange(0.50, 0.951, 0.025), 3)

scan_rows = []

for tq in thresholds_qcd:
    for tt in thresholds_top:
        sel = (test_df["bdt_qcd_score"] >= tq) & (test_df["bdt_top_score"] >= tt)
        sig = test_df[sel & test_df["is_signal"]]
        bkg = test_df[sel & ~test_df["is_signal"]]

        s_pb = sig["weight_pb_scaled_to_full"].sum()
        b_pb = bkg["weight_pb_scaled_to_full"].sum()

        s_ev = s_pb * LUMI_PB
        b_ev = b_pb * LUMI_PB

        scan_rows.append({
            "qcd_threshold": tq,
            "top_threshold": tt,
            "signal_events_450fb": s_ev,
            "background_events_450fb": b_ev,
            "S_over_B": s_ev / b_ev if b_ev > 0 else np.nan,
            "S_over_sqrtB": s_ev / np.sqrt(b_ev) if b_ev > 0 else np.nan,
            "S_over_10pctB": s_ev / (0.10 * b_ev) if b_ev > 0 else np.nan,
            "n_signal_test_rows": len(sig),
            "n_background_test_rows": len(bkg),
            "neff_signal": neff(sig["weight_pb_scaled_to_full"]),
            "neff_background": neff(bkg["weight_pb_scaled_to_full"]),
        })

scan = pd.DataFrame(scan_rows)

stable = scan[
    (scan["n_background_test_rows"] >= 20)
    & (scan["neff_background"] >= 5)
    & (scan["n_signal_test_rows"] >= 20)
].copy()

best_stable = stable.sort_values("S_over_sqrtB", ascending=False).head(20)

score_summary = (
    test_df.assign(expected_events_450fb=lambda d: d["weight_pb_scaled_to_full"] * LUMI_PB)
    .groupby(["sample", "group"], as_index=False)
    .agg(
        test_rows=("sample", "size"),
        xsec_pb=("weight_pb_scaled_to_full", "sum"),
        expected_events_450fb=("expected_events_450fb", "sum"),
        median_bdt_qcd=("bdt_qcd_score", "median"),
        q90_bdt_qcd=("bdt_qcd_score", lambda x: float(np.quantile(x, 0.90))),
        q99_bdt_qcd=("bdt_qcd_score", lambda x: float(np.quantile(x, 0.99))),
        median_bdt_top=("bdt_top_score", "median"),
        q90_bdt_top=("bdt_top_score", lambda x: float(np.quantile(x, 0.90))),
        q99_bdt_top=("bdt_top_score", lambda x: float(np.quantile(x, 0.99))),
    )
)

auc_summary.to_csv(OUTDIR / "two_bdt_event_features_auc_summary.csv", index=False)
qcd_imp.to_csv(OUTDIR / "bdt_qcd_event_feature_importances.csv", index=False)
top_imp.to_csv(OUTDIR / "bdt_top_event_feature_importances.csv", index=False)
scan.to_csv(OUTDIR / "two_bdt_event_features_rectangle_scan.csv", index=False)
best_stable.to_csv(OUTDIR / "two_bdt_event_features_best_rectangles_stable.csv", index=False)
score_summary.to_csv(OUTDIR / "two_bdt_event_features_score_summary_by_sample.csv", index=False)

(OUTDIR / "two_bdt_event_features_auc_summary.md").write_text(auc_summary.to_markdown(index=False) + "\n")
(OUTDIR / "bdt_qcd_event_feature_importances.md").write_text(qcd_imp.to_markdown(index=False) + "\n")
(OUTDIR / "bdt_top_event_feature_importances.md").write_text(top_imp.to_markdown(index=False) + "\n")
(OUTDIR / "two_bdt_event_features_best_rectangles_stable.md").write_text(best_stable.to_markdown(index=False) + "\n")
(OUTDIR / "two_bdt_event_features_score_summary_by_sample.md").write_text(score_summary.to_markdown(index=False) + "\n")

readme = """# Two-BDT event-feature HH4b baseline

This extends the candidate-level two-BDT baseline by merging event-level variables from the event_summary parquet files:
- n_jet_pt30_eta25
- n_bjet_pt30_eta25
- n_extra_jets_pt30_eta25
- n_extra_bjets_pt30_eta25
- ht_pt30_eta25
- ht_over_mhh

The goal is to test whether event-level/top-sensitive information improves the weak BDT_top baseline.
"""
(OUTDIR / "README.md").write_text(readme)

print("\n=== Two-BDT event-feature AUC summary ===")
print(auc_summary.to_string(index=False))

print("\n=== Best stable event-feature 2D rectangles ===")
print(best_stable.head(15).to_string(index=False))

print("\n=== QCD feature importances ===")
print(qcd_imp.head(20).to_string(index=False))

print("\n=== Top feature importances ===")
print(top_imp.head(20).to_string(index=False))

print(f"\nWrote outputs to: {OUTDIR}")
