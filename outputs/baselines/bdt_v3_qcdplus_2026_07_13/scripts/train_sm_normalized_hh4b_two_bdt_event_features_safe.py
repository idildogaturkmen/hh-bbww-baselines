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

OUTDIR = REPO / "outputs/tables/sm_normalized_hh4b_two_bdt_event_features_safe_2026_07_09"
OUTDIR.mkdir(parents=True, exist_ok=True)

LUMI_PB = 450.0 * 1000.0
TEST_SIZE = 0.35
RANDOM_STATE = 42

QCD_META = Path(os.environ.get("QCD_META", STORE / "metadata/qcd_bbbb_iht_slice_scan_20000.csv"))

SIGNALS = {
    "ggF_HH4b_SMnorm": {
        "group": "signal",
        "cand_patterns": ["HH4b_ggf_hh4b_10k*merged*hh4b_candidates.parquet"],
        "event_patterns": ["HH4b_ggf_hh4b_10k*merged*event_summary.parquet"],
        "n_generated": 10000,
        "xsec_pb": 10.55 / 1000.0,
    },
    "VBF_HH4b_SMnorm": {
        "group": "signal",
        "cand_patterns": ["HH4b_vbf_hh4b_10k*merged*hh4b_candidates.parquet"],
        "event_patterns": ["HH4b_vbf_hh4b_10k*merged*event_summary.parquet"],
        "n_generated": 10000,
        "xsec_pb": 0.587 / 1000.0,
    },
}

BACKGROUNDS = {
    "ttbar_200k": {
        "group": "background",
        "cand_patterns": ["ttbar_200k_merged_hh4b_candidates.parquet"],
        "event_patterns": ["ttbar_200k_merged_event_summary.parquet"],
        "n_generated": 200000,
        "xsec_pb": 512.2746276855469,
    },
    "Zbbbb_100k": {
        "group": "background",
        "cand_patterns": ["zbbbb*100k*merged*hh4b_candidates.parquet", "Zbbbb*100k*merged*hh4b_candidates.parquet"],
        "event_patterns": ["zbbbb*100k*merged_event_summary.parquet", "Zbbbb*100k*merged_event_summary.parquet"],
        "n_generated": 100000,
        "xsec_pb": 6.912992,
    },
}

BASE_FEATURES = [
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
        raise FileNotFoundError(f"No match for patterns: {patterns}")
    return matches[-1]


def add_candidate_features(df):
    df = df.copy()
    df["r_hh"] = np.sqrt((df["mbb1"] - 125.0) ** 2 + (df["mbb2"] - 125.0) ** 2)
    df["pt_sum4"] = df["j1_pt"] + df["j2_pt"] + df["j3_pt"] + df["j4_pt"]
    df["pt_asym_12"] = (df["j1_pt"] - df["j2_pt"]) / (df["j1_pt"] + df["j2_pt"] + 1e-9)
    df["pt_asym_34"] = (df["j3_pt"] - df["j4_pt"]) / (df["j3_pt"] + df["j4_pt"] + 1e-9)
    return df


def choose_merge_keys(cand, ev, sample_name):
    # Prefer original per-shard sample + event if available and unique.
    if "sample" in cand.columns and "sample" in ev.columns:
        keys = ["sample", "event"]
        if ev.duplicated(keys).sum() == 0:
            return keys

    # Then try source_campaign + sample + event.
    if all(c in cand.columns for c in ["source_campaign", "sample"]) and all(c in ev.columns for c in ["source_campaign", "sample"]):
        keys = ["source_campaign", "sample", "event"]
        if ev.duplicated(keys).sum() == 0:
            return keys

    # Then try source_campaign + event.
    if "source_campaign" in cand.columns and "source_campaign" in ev.columns:
        keys = ["source_campaign", "event"]
        if ev.duplicated(keys).sum() == 0:
            return keys

    # Last resort: event only, only if unique.
    keys = ["event"]
    if ev.duplicated(keys).sum() == 0:
        return keys

    dup = ev[ev.duplicated(["event"], keep=False)].head()
    raise RuntimeError(
        f"Cannot find unique event-merge keys for {sample_name}. "
        f"event IDs are duplicated and no usable sample/source key is available.\n"
        f"Example duplicated event rows:\n{dup}"
    )


def load_sample(sample_name, cfg):
    cand_path = find_one(cfg["cand_patterns"])
    event_path = find_one(cfg["event_patterns"])

    cand = pd.read_parquet(cand_path)
    ev = pd.read_parquet(event_path)

    n_before = len(cand)

    # Preserve original source identifiers before assigning the analysis sample label.
    cand = cand.copy()
    ev = ev.copy()

    merge_keys = choose_merge_keys(cand, ev, sample_name)

    event_cols = list(merge_keys)
    for c in [
        "n_jet_pt30_eta25",
        "n_bjet_pt30_eta25",
        "ht_pt30_eta25",
        "event_cross_section_pb",
    ]:
        if c in ev.columns and c not in event_cols:
            event_cols.append(c)

    ev_small = ev[event_cols].copy()

    merged = cand.merge(ev_small, on=merge_keys, how="left", validate="many_to_one")

    n_after = len(merged)
    if n_after != n_before:
        raise RuntimeError(
            f"Row count changed after merge for {sample_name}: before={n_before}, after={n_after}. "
            f"Merge keys were {merge_keys}. This indicates duplication."
        )

    missing_event_info = merged["n_jet_pt30_eta25"].isna().sum() if "n_jet_pt30_eta25" in merged.columns else len(merged)
    if missing_event_info:
        print(f"WARNING: {sample_name} has {missing_event_info} rows missing event-level info.")

    merged = add_candidate_features(merged)

    # Event-level derived features.
    if "n_jet_pt30_eta25" not in merged.columns:
        merged["n_jet_pt30_eta25"] = 4.0
    if "n_bjet_pt30_eta25" not in merged.columns:
        merged["n_bjet_pt30_eta25"] = merged.get("n_selected_bjets", 4.0)
    if "ht_pt30_eta25" not in merged.columns:
        merged["ht_pt30_eta25"] = merged["pt_sum4"]

    merged["n_extra_jets_pt30_eta25"] = merged["n_jet_pt30_eta25"] - 4.0
    merged["n_extra_bjets_pt30_eta25"] = merged["n_bjet_pt30_eta25"] - 4.0
    merged["ht_over_mhh"] = merged["ht_pt30_eta25"] / (merged["mhh"] + 1e-9)

    # Analysis labels and weights.
    merged["analysis_sample"] = sample_name
    merged["group"] = cfg["group"]
    merged["target"] = 1 if cfg["group"] == "signal" else 0
    merged["n_generated"] = cfg["n_generated"]
    merged["xsec_pb"] = cfg["xsec_pb"]
    merged["weight_pb"] = cfg["xsec_pb"] / cfg["n_generated"]

    print(
        f"Loaded {sample_name}: candidates={len(merged)}, "
        f"event_path={event_path.name}, cand_path={cand_path.name}, merge_keys={merge_keys}"
    )

    return merged


def load_all():
    frames = []

    for name, cfg in SIGNALS.items():
        frames.append(load_sample(name, cfg))

    for name, cfg in BACKGROUNDS.items():
        frames.append(load_sample(name, cfg))

    qcd = pd.read_csv(QCD_META)
    for _, row in qcd.iterrows():
        tag = str(row["tag"])
        cfg = {
            "group": "background",
            "cand_patterns": [f"{tag}_hh4b_candidates.parquet", f"{tag}_merged_hh4b_candidates.parquet"],
            "event_patterns": [f"{tag}_event_summary.parquet", f"{tag}_merged_event_summary.parquet"],
            "n_generated": int(row["n_generated"]),
            "xsec_pb": float(row["xsec_pb"]),
        }
        frames.append(load_sample(tag, cfg))

    out = pd.concat(frames, ignore_index=True)
    out["is_signal"] = out["group"].eq("signal")
    out["is_qcd"] = out["analysis_sample"].str.startswith("qcd_")
    out["is_top"] = out["analysis_sample"].str.contains("ttbar|ttbb|tt_", case=False, regex=True)

    feature_cols = BASE_FEATURES + [
        "r_hh", "pt_sum4", "pt_asym_12", "pt_asym_34",
        "n_jet_pt30_eta25", "n_bjet_pt30_eta25",
        "n_extra_jets_pt30_eta25", "n_extra_bjets_pt30_eta25",
        "ht_pt30_eta25", "ht_over_mhh",
    ]

    out = out.replace([np.inf, -np.inf], np.nan).dropna(subset=feature_cols)

    return out, feature_cols


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


def train_task(train_df, test_df, feature_cols, mask_col, label):
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

    clf.fit(task_train[feature_cols], y_train, sample_weight=class_balanced_weights(y_train))

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

    return all_scores, auc, wauc, imp, len(task_train), len(task_test)


def main():
    df, feature_cols = load_all()

    row_counts = (
        df.groupby(["analysis_sample", "group"], as_index=False)
        .agg(
            rows=("event", "size"),
            n_generated=("n_generated", "first"),
            xsec_pb=("xsec_pb", "first"),
        )
    )
    row_counts.to_csv(OUTDIR / "safe_merge_row_counts.csv", index=False)
    (OUTDIR / "safe_merge_row_counts.md").write_text(row_counts.to_markdown(index=False) + "\n")

    # Sanity: all candidate signal yield before BDT.
    sig_all_pb = df.loc[df["is_signal"], "weight_pb"].sum()
    sig_all_events = sig_all_pb * LUMI_PB

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

    qcd_scores, qcd_auc, qcd_wauc, qcd_imp, qcd_ntrain, qcd_ntest = train_task(
        train_df, test_df, feature_cols, "is_qcd", "qcd"
    )
    top_scores, top_auc, top_wauc, top_imp, top_ntrain, top_ntest = train_task(
        train_df, test_df, feature_cols, "is_top", "top"
    )

    test_df["bdt_qcd_score"] = qcd_scores
    test_df["bdt_top_score"] = top_scores

    auc_summary = pd.DataFrame([
        {
            "classifier": "BDT_QCD_event_features_safe",
            "negative_class": "QCD bbbb HT slices",
            "unweighted_auc": qcd_auc,
            "physics_weighted_auc": qcd_wauc,
            "n_train_task": qcd_ntrain,
            "n_test_task": qcd_ntest,
        },
        {
            "classifier": "BDT_top_event_features_safe",
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

            # Hard sanity check: selected signal cannot exceed all candidate-level signal.
            if s_ev > sig_all_events * 1.001:
                raise RuntimeError(
                    f"Selected signal {s_ev} exceeds all candidate signal {sig_all_events}. "
                    "This indicates duplicated rows or scaling error."
                )

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

    qcd_imp.to_csv(OUTDIR / "bdt_qcd_event_feature_importances_safe.csv", index=False)
    top_imp.to_csv(OUTDIR / "bdt_top_event_feature_importances_safe.csv", index=False)
    auc_summary.to_csv(OUTDIR / "two_bdt_event_features_auc_summary_safe.csv", index=False)
    scan.to_csv(OUTDIR / "two_bdt_event_features_rectangle_scan_safe.csv", index=False)
    best_stable.to_csv(OUTDIR / "two_bdt_event_features_best_rectangles_stable_safe.csv", index=False)

    (OUTDIR / "bdt_qcd_event_feature_importances_safe.md").write_text(qcd_imp.to_markdown(index=False) + "\n")
    (OUTDIR / "bdt_top_event_feature_importances_safe.md").write_text(top_imp.to_markdown(index=False) + "\n")
    (OUTDIR / "two_bdt_event_features_auc_summary_safe.md").write_text(auc_summary.to_markdown(index=False) + "\n")
    (OUTDIR / "two_bdt_event_features_best_rectangles_stable_safe.md").write_text(best_stable.to_markdown(index=False) + "\n")

    readme = f"""# Safe event-feature two-BDT HH4b baseline

This is the corrected event-feature version. It verifies that merging event-level features does not change the number of candidate rows.

All-candidate SM-normalized signal yield at 450/fb: {sig_all_events:.6f} events.

The previous non-safe event-feature output is invalid because merging on non-unique event IDs duplicated rows.
"""
    (OUTDIR / "README.md").write_text(readme)

    print("\n=== Safe merge row counts ===")
    print(row_counts.to_string(index=False))

    print("\nAll-candidate signal events at 450/fb:", sig_all_events)

    print("\n=== Safe event-feature AUC summary ===")
    print(auc_summary.to_string(index=False))

    print("\n=== Best stable safe event-feature rectangles ===")
    print(best_stable.head(15).to_string(index=False))

    print("\n=== Top feature importances ===")
    print(top_imp.head(20).to_string(index=False))

    print(f"\nWrote outputs to: {OUTDIR}")


if __name__ == "__main__":
    main()
