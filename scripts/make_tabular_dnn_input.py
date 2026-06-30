"""
Prepare tabular DNN inputs from the rough-normalized HH->bbWW feature table.

This creates a reproducible NPZ file with:
- standardized numeric features
- binary signal/background labels
- CMS-inspired single-lepton multiclass labels
- physics weights
- split labels
- event metadata for debugging

The standardization is fit on the train split only.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


DEFAULT_DROP_FEATURES = {
    # labels / weights / constants / non-useful variables
    "target",
    "weight",
    "unit_weight",
    "physics_weight_nominal",
    "has_physics_weight",
    "n_leptons",

    # constants in your current one-lepton table
    "b1_btag",
    "b2_btag",

    # electron/muon duplicated channel indicator; keep n_muons or lepton_type if useful,
    # but drop n_electrons to avoid exact redundancy if n_muons already encodes it.
    "n_electrons",
}


def assign_cms_singlelep_class(row: pd.Series) -> str:
    """
    Approximate the CMS single-lepton nonresonant category grouping.

    CMS single-lepton categories:
      - HH(ggF)
      - HH(VBF)
      - Top + Higgs
      - W+jets + Other

    COLLIDE currently has HH_bbWW but no explicit VBF-HH folder in your local table,
    so HH_bbWW is treated as HH_ggF_like.
    """
    sample = str(row.get("sample", ""))
    process_group = str(row.get("process_group", "")).lower()
    target = int(row.get("target", 0))

    if target == 1 or sample == "HH_bbWW":
        return "HH_ggF_like"

    if process_group in {"ttbar", "single_higgs", "ttv_tth_tttt"}:
        return "Top_Higgs"

    return "WJets_Other"


def safe_numeric_features(df: pd.DataFrame, requested_features: list[str]) -> list[str]:
    features = []
    for c in requested_features:
        if c not in df.columns:
            continue
        if c in DEFAULT_DROP_FEATURES:
            continue
        if pd.api.types.is_numeric_dtype(df[c]):
            features.append(c)
    return features


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        default="outputs/topology_features_lepemu_rough/onelep_pre_mbb_classifier_input_with_topology_features.parquet",
        help="Input feature table parquet.",
    )
    parser.add_argument(
        "--features",
        default="outputs/topology_features_lepemu_rough/feature_columns_topology.json",
        help="JSON list of feature columns.",
    )
    parser.add_argument(
        "--outdir",
        default="outputs/dnn_inputs_lepemu_rough_topology",
        help="Output directory.",
    )
    parser.add_argument(
        "--drop-constant-features",
        action="store_true",
        default=True,
        help="Drop features that are constant after filtering.",
    )
    args = parser.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    print("Reading:", args.input)
    df0 = pd.read_parquet(args.input)
    print("Input rows:", len(df0))

    df = df0[
        df0["has_physics_weight"]
        & np.isfinite(df0["physics_weight_nominal"])
    ].copy()

    print("Rows with safe rough physics weights:", len(df))
    print("Excluded rows:", len(df0) - len(df))

    with open(args.features) as f:
        requested_features = json.load(f)

    features = safe_numeric_features(df, requested_features)

    if args.drop_constant_features:
        nonconstant = []
        for c in features:
            x = df[c].replace([np.inf, -np.inf], np.nan)
            if x.nunique(dropna=True) > 1:
                nonconstant.append(c)
        features = nonconstant

    if len(features) == 0:
        raise RuntimeError("No usable numeric features found.")

    print("Number of DNN features:", len(features))

    X = df[features].replace([np.inf, -np.inf], np.nan)

    train_mask = df["split"].astype(str).eq("train").to_numpy()
    val_mask = df["split"].astype(str).eq("val").to_numpy()
    test_mask = df["split"].astype(str).eq("test").to_numpy()

    if train_mask.sum() == 0:
        raise RuntimeError("No train rows found. Check the split column.")

    # Impute using train medians only.
    medians = X.loc[train_mask].median(numeric_only=True)
    X = X.fillna(medians)

    # Any feature that was all-NaN in train still has NaNs after median fill.
    # Fill those with zero and record them.
    still_nan_cols = X.columns[X.isna().any()].tolist()
    if still_nan_cols:
        print("Warning: columns still NaN after median fill; filling with zero:")
        for c in still_nan_cols:
            print(" ", c)
        X = X.fillna(0.0)

    scaler = StandardScaler()
    X_scaled = np.zeros((len(df), len(features)), dtype=np.float32)
    X_scaled[train_mask] = scaler.fit_transform(X.loc[train_mask]).astype(np.float32)
    X_scaled[~train_mask] = scaler.transform(X.loc[~train_mask]).astype(np.float32)

    y_binary = df["target"].astype(np.int64).to_numpy()

    class_names = ["HH_ggF_like", "Top_Higgs", "WJets_Other"]
    class_to_id = {name: i for i, name in enumerate(class_names)}
    cms_class_name = df.apply(assign_cms_singlelep_class, axis=1)
    y_multiclass = cms_class_name.map(class_to_id).astype(np.int64).to_numpy()

    physics_weight = df["physics_weight_nominal"].astype(np.float32).to_numpy()
    split = df["split"].astype(str).to_numpy()

    # Useful class-balanced training weights, separate from physics weights.
    # These are normalized to mean ~1 on the training split.
    train_weight_binary = np.ones(len(df), dtype=np.float32)
    for cls in [0, 1]:
        m = train_mask & (y_binary == cls)
        if m.sum() > 0:
            train_weight_binary[m] = train_mask.sum() / (2.0 * m.sum())

    train_weight_multiclass = np.ones(len(df), dtype=np.float32)
    for cls in range(len(class_names)):
        m = train_mask & (y_multiclass == cls)
        if m.sum() > 0:
            train_weight_multiclass[m] = train_mask.sum() / (len(class_names) * m.sum())

    npz_path = outdir / "tabular_dnn_inputs.npz"
    np.savez_compressed(
        npz_path,
        X=X_scaled,
        y_binary=y_binary,
        y_multiclass=y_multiclass,
        physics_weight=physics_weight,
        train_weight_binary=train_weight_binary,
        train_weight_multiclass=train_weight_multiclass,
        split=split,
        feature_names=np.asarray(features),
        class_names=np.asarray(class_names),
    )

    pd.DataFrame({
        "feature": features,
        "median_train": medians.reindex(features).values,
        "scaler_mean": scaler.mean_,
        "scaler_scale": scaler.scale_,
    }).to_csv(outdir / "feature_preprocessing.csv", index=False)

    with open(outdir / "feature_columns.json", "w") as f:
        json.dump(features, f, indent=2)

    with open(outdir / "class_map.json", "w") as f:
        json.dump(class_to_id, f, indent=2)

    meta_cols = [
        "split", "target", "sample", "process_group", "file", "event_in_file",
        "physics_weight_nominal",
    ]
    meta_cols = [c for c in meta_cols if c in df.columns]
    meta = df[meta_cols].copy()
    meta["cms_singlelep_class"] = cms_class_name.values
    meta["cms_singlelep_class_id"] = y_multiclass
    meta.to_parquet(outdir / "event_metadata.parquet", index=False)

    rows = []
    for s in ["train", "val", "test"]:
        m = split == s
        rows.append({
            "split": s,
            "raw_events": int(m.sum()),
            "signal_raw": int(((y_binary == 1) & m).sum()),
            "background_raw": int(((y_binary == 0) & m).sum()),
            "signal_weighted": float(physics_weight[(y_binary == 1) & m].sum()),
            "background_weighted": float(physics_weight[(y_binary == 0) & m].sum()),
        })

    pd.DataFrame(rows).to_csv(outdir / "split_summary.csv", index=False)

    class_summary = (
        pd.DataFrame({
            "split": split,
            "class": cms_class_name.values,
            "physics_weight": physics_weight,
        })
        .groupby(["split", "class"])
        .agg(raw_events=("class", "size"), weighted_events=("physics_weight", "sum"))
        .reset_index()
    )
    class_summary.to_csv(outdir / "multiclass_summary.csv", index=False)

    print("\nWrote:", npz_path)
    print("Wrote:", outdir / "feature_preprocessing.csv")
    print("Wrote:", outdir / "event_metadata.parquet")
    print("Wrote:", outdir / "split_summary.csv")
    print("Wrote:", outdir / "multiclass_summary.csv")

    print("\nSplit summary:")
    print(pd.DataFrame(rows).to_string(index=False))

    print("\nMulticlass summary:")
    print(class_summary.to_string(index=False))


if __name__ == "__main__":
    main()
