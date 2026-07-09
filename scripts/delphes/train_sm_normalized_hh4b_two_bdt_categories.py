'''
Trains the SM-normalized HH4b signal against two separate background families using two dedicated BDTs.
The first BDT is trained against QCD bbbb HT-sliced backgrounds.
The second BDT is trained against ttbar / top-like backgrounds.
The outputs include:
- AUC for each BDT.
- A 2D rectangular threshold scan.
- Stable threshold scan requiring minimum background statistics.
- Fixed non-overlapping category yields and background composition.
Signal normalization follows SM HH cross sections times BR(H->bb)^2.
Backgrounds use generator cross-section weights.
'''

import importlib.util
import os
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split


REPO = Path(os.environ["HH4B_REPO"])
BASE_BDT_SCRIPT = REPO / "scripts/delphes/train_sm_normalized_hh4b_bdt.py"

OUTDIR = REPO / "outputs/tables/sm_normalized_hh4b_two_bdt_2026_07_09"
OUTDIR.mkdir(parents=True, exist_ok=True)

LUMI_FB = 450.0
LUMI_PB = LUMI_FB * 1000.0
TEST_SIZE = 0.35
RANDOM_STATE = 42

# Load the same data-loading function as the one-BDT baseline.
spec = importlib.util.spec_from_file_location("bdtmod", BASE_BDT_SCRIPT)
bdtmod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(bdtmod)

df, feature_cols = bdtmod.load_all()

# Define background groups.
df = df.copy()
df["is_signal"] = df["group"].eq("signal")
df["is_qcd"] = df["sample"].str.startswith("qcd_")
df["is_top"] = df["sample"].str.contains("ttbar|ttbb|tt_", case=False, regex=True)

# Common train/test split for all downstream comparisons.
train_df, test_df = train_test_split(
    df,
    test_size=TEST_SIZE,
    random_state=RANDOM_STATE,
    stratify=df["target"],
)

# Scale test subset back to full sample statistics, sample by sample.
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


def train_binary_bdt(train_subset, test_subset, background_mask_name, label):
    """
    Train signal-vs-specific-background BDT.

    Positive class = signal.
    Negative class = requested background family.
    """
    task_train = train_subset[train_subset["is_signal"] | train_subset[background_mask_name]].copy()
    task_test = test_subset[test_subset["is_signal"] | test_subset[background_mask_name]].copy()

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

    test_scores_task = clf.predict_proba(task_test[feature_cols])[:, 1]
    auc_unweighted = roc_auc_score(y_test, test_scores_task)

    # Weighted AUC only within the task test subset.
    task_test = task_test.copy()
    task_test["score"] = test_scores_task
    auc_weighted = roc_auc_score(
        y_test,
        task_test["score"],
        sample_weight=task_test["weight_pb_scaled_to_full"],
    )

    # Apply model to every test event, not only the task subset.
    all_scores = clf.predict_proba(test_subset[feature_cols])[:, 1]

    importance = pd.DataFrame({
        "feature": feature_cols,
        f"importance_{label}": clf.feature_importances_,
    }).sort_values(f"importance_{label}", ascending=False)

    return clf, all_scores, auc_unweighted, auc_weighted, importance, len(task_train), len(task_test)


def neff(weights):
    w = np.asarray(weights, dtype=float)
    if len(w) == 0 or np.sum(w * w) <= 0:
        return 0.0
    return float((np.sum(w) ** 2) / np.sum(w * w))


# Train two dedicated BDTs.
qcd_clf, qcd_scores, qcd_auc, qcd_wauc, qcd_imp, qcd_ntrain, qcd_ntest = train_binary_bdt(
    train_df,
    test_df,
    "is_qcd",
    "qcd",
)

top_clf, top_scores, top_auc, top_wauc, top_imp, top_ntrain, top_ntest = train_binary_bdt(
    train_df,
    test_df,
    "is_top",
    "top",
)

test_df["bdt_qcd_score"] = qcd_scores
test_df["bdt_top_score"] = top_scores

auc_summary = pd.DataFrame([
    {
        "classifier": "BDT_QCD",
        "positive_class": "SM-normalized HH signal",
        "negative_class": "QCD bbbb HT slices",
        "unweighted_auc": qcd_auc,
        "physics_weighted_auc": qcd_wauc,
        "n_train_task": qcd_ntrain,
        "n_test_task": qcd_ntest,
    },
    {
        "classifier": "BDT_top",
        "positive_class": "SM-normalized HH signal",
        "negative_class": "ttbar / top-like backgrounds",
        "unweighted_auc": top_auc,
        "physics_weighted_auc": top_wauc,
        "n_train_task": top_ntrain,
        "n_test_task": top_ntest,
    },
])

# 2D rectangular scan: require both BDT_QCD and BDT_top above thresholds.
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
            "signal_xsec_pb": s_pb,
            "background_xsec_pb": b_pb,
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

# Stable selections require enough raw and effective background statistics.
stable_scan = scan[
    (scan["n_background_test_rows"] >= 20)
    & (scan["neff_background"] >= 5)
    & (scan["n_signal_test_rows"] >= 20)
].copy()

best_all = scan.sort_values("S_over_sqrtB", ascending=False).head(20)
best_stable = stable_scan.sort_values("S_over_sqrtB", ascending=False).head(20)

# Fixed non-overlapping categories inspired by CMS-style 2D categorization.
# These are intentionally conservative starting categories, not final optimized categories.
cat_defs = [
    ("CAT0_tight", 0.875, 0.875),
    ("CAT1_qcdtight_topmedium", 0.850, 0.800),
    ("CAT2_qcdmedium_toptight", 0.800, 0.850),
    ("CAT3_medium", 0.800, 0.750),
    ("CAT4_loose", 0.700, 0.700),
]

assigned = pd.Series(False, index=test_df.index)
cat_rows = []
cat_comp_rows = []

for cat_name, tq, tt in cat_defs:
    sel = (
        (test_df["bdt_qcd_score"] >= tq)
        & (test_df["bdt_top_score"] >= tt)
        & (~assigned)
    )

    assigned.loc[sel] = True

    cat_df = test_df.loc[sel].copy()
    sig = cat_df[cat_df["is_signal"]]
    bkg = cat_df[~cat_df["is_signal"]]

    s_pb = sig["weight_pb_scaled_to_full"].sum()
    b_pb = bkg["weight_pb_scaled_to_full"].sum()

    s_ev = s_pb * LUMI_PB
    b_ev = b_pb * LUMI_PB

    cat_rows.append({
        "category": cat_name,
        "qcd_threshold": tq,
        "top_threshold": tt,
        "signal_xsec_pb": s_pb,
        "background_xsec_pb": b_pb,
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

    for (sample, group), sub in cat_df.groupby(["sample", "group"]):
        xsec_pb = sub["weight_pb_scaled_to_full"].sum()
        cat_comp_rows.append({
            "category": cat_name,
            "sample": sample,
            "group": group,
            "selected_test_rows": len(sub),
            "xsec_pb": xsec_pb,
            "expected_events_450fb": xsec_pb * LUMI_PB,
        })

cat_summary = pd.DataFrame(cat_rows)
cat_comp = pd.DataFrame(cat_comp_rows)

if len(cat_comp):
    cat_comp["fraction_within_category_group"] = (
        cat_comp["expected_events_450fb"]
        / cat_comp.groupby(["category", "group"])["expected_events_450fb"].transform("sum")
    )

# Per-sample score summaries.
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

# Save outputs.
auc_summary.to_csv(OUTDIR / "two_bdt_auc_summary.csv", index=False)
qcd_imp.to_csv(OUTDIR / "bdt_qcd_feature_importances.csv", index=False)
top_imp.to_csv(OUTDIR / "bdt_top_feature_importances.csv", index=False)
scan.to_csv(OUTDIR / "two_bdt_rectangle_scan.csv", index=False)
best_all.to_csv(OUTDIR / "two_bdt_best_rectangles_all.csv", index=False)
best_stable.to_csv(OUTDIR / "two_bdt_best_rectangles_stable.csv", index=False)
cat_summary.to_csv(OUTDIR / "two_bdt_fixed_category_summary.csv", index=False)
cat_comp.to_csv(OUTDIR / "two_bdt_fixed_category_composition.csv", index=False)
score_summary.to_csv(OUTDIR / "two_bdt_score_summary_by_sample.csv", index=False)

(OUTDIR / "two_bdt_auc_summary.md").write_text(auc_summary.to_markdown(index=False) + "\n")
(OUTDIR / "bdt_qcd_feature_importances.md").write_text(qcd_imp.to_markdown(index=False) + "\n")
(OUTDIR / "bdt_top_feature_importances.md").write_text(top_imp.to_markdown(index=False) + "\n")
(OUTDIR / "two_bdt_best_rectangles_all.md").write_text(best_all.to_markdown(index=False) + "\n")
(OUTDIR / "two_bdt_best_rectangles_stable.md").write_text(best_stable.to_markdown(index=False) + "\n")
(OUTDIR / "two_bdt_fixed_category_summary.md").write_text(cat_summary.to_markdown(index=False) + "\n")
(OUTDIR / "two_bdt_fixed_category_composition.md").write_text(cat_comp.to_markdown(index=False) + "\n")
(OUTDIR / "two_bdt_score_summary_by_sample.md").write_text(score_summary.to_markdown(index=False) + "\n")

readme = """# Two-BDT SM-normalized HH4b category baseline

This study uses two dedicated BDTs:

- BDT_QCD: trained to separate SM-normalized HH signal from QCD bbbb HT-sliced backgrounds.
- BDT_top: trained to separate SM-normalized HH signal from ttbar/top-like backgrounds.

The goal is to test a CMS-inspired two-discriminant category strategy rather than relying on a single global BDT. The outputs include:
- AUC for each BDT.
- A 2D rectangular threshold scan.
- Stable threshold scan requiring minimum background statistics.
- Fixed non-overlapping category yields and background composition.

Signal normalization follows SM HH cross sections times BR(H->bb)^2.
Backgrounds use generator cross-section weights.
"""
(OUTDIR / "README.md").write_text(readme)

print("\n=== Two-BDT AUC summary ===")
print(auc_summary.to_string(index=False))

print("\n=== Best stable 2D rectangles ===")
print(best_stable.head(15).to_string(index=False))

print("\n=== Fixed category summary ===")
print(cat_summary.to_string(index=False))

print("\n=== Score summary by sample ===")
print(score_summary.to_string(index=False))

print(f"\nWrote outputs to: {OUTDIR}")
