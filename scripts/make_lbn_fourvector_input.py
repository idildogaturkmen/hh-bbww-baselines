"""
Build a first LBN-style four-vector input for HH->bbWW. 
The Lorentz Boost Network (LBN) is a specialized neural network layer used in particle physics. 
It takes the raw four-vectors of final-state particles (their energy and 3D momentum) and autonomously calculates physics-motivated features by transforming them into different frames of reference

v0 object set:
  0: b1
  1: b2
  2: leading lepton
  3: MET pseudo-object

Object features:
  log_pt, eta, sin_phi, cos_phi, log_mass, present

Also builds:
  - pairwise manual-LBN features from object four-vectors
  - auxiliary event-level features
  - labels/splits/weights matching the existing rough-weighted recoMET setup
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


OBJECT_NAMES = ["b1", "b2", "lepton", "met"]
OBJECT_FEATURES = ["log_pt", "eta", "sin_phi", "cos_phi", "log_mass", "present"]


AUX_CANDIDATES = [
    "n_jets",
    "n_jets_raw",
    "n_leptons",
    "n_electrons",
    "n_muons",
    "ht",
    "met",
    "mbb_uncorrected",
    "mbb_global_median_scaled",
    "mbb_pt_binned_scaled",
    "b1_btag",
    "b2_btag",
    "b1_btag_phys",
    "b2_btag_phys",
    "dr_bb_top2_btag",
    "abs_b1_eta",
    "abs_b2_eta",
    "deta_bb",
    "dphi_bb",
    "b2_over_b1_pt",
    "met_over_ht",
    "mbb_pt_binned_over_ht",
    "leading_lepton_pt",
    "leading_lepton_eta",
    "leading_lepton_iso",
    "leading_lepton_reliso",
    "leading_lepton_is_muon",
    "leading_lepton_is_electron",
    "n_iso_leptons_reliso_lt_0p1",
    "n_iso_leptons_reliso_lt_0p2",
    "dphi_lep_met",
    "mt_lep_met",
    "met_over_lepton_pt",
    "pt_bb",
    "eta_bb_reco",
    "dphi_bb_met",
    "dphi_lep_bb",
    "dr_lep_b1",
    "dr_lep_b2",
    "min_dr_lep_b",
    "m_lep_b1",
    "m_lep_b2",
    "min_m_lep_b",
    "dphi_met_b1",
    "dphi_met_b2",
    "min_dphi_met_b",
    "mt_bb_met",
    "pt_bb_over_ht",
    "met_over_pt_bb",
    "n_selected_ak4_jets",
    "n_btagged_jets",
    "n_nonb_jets",
    "third_jet_pt",
    "fourth_jet_pt",
    "n_nonb_jet_pairs",
    "hadronic_W_candidate_mass",
    "abs_mjj_minus_W",
    "hadronic_W_candidate_dr",
    "hadronic_W_candidate_pt",
    "hadronic_W_candidate_deta",
    "hadronic_W_candidate_dphi",
    "n_bjj_top_candidates",
    "top_candidate_mass_closest",
    "min_abs_m_bjj_minus_top",
    "min_top_candidate_mass",
]


def safe_array(df: pd.DataFrame, col: str, default: float = 0.0) -> np.ndarray:
    if col in df.columns:
        return pd.to_numeric(df[col], errors="coerce").to_numpy(dtype=np.float64)
    return np.full(len(df), default, dtype=np.float64)


def safe_bool(df: pd.DataFrame, col: str, default: bool = False) -> np.ndarray:
    if col in df.columns:
        return df[col].fillna(default).astype(bool).to_numpy()
    return np.full(len(df), default, dtype=bool)


def finite_positive(x: np.ndarray) -> np.ndarray:
    return np.isfinite(x) & (x > 0)


def phi_to_sincos(phi: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    phi = np.where(np.isfinite(phi), phi, 0.0)
    return np.sin(phi), np.cos(phi)


def pt_eta_phi_m_to_cartesian(pt, eta, phi, mass):
    pt = np.where(np.isfinite(pt), pt, 0.0)
    eta = np.where(np.isfinite(eta), eta, 0.0)
    phi = np.where(np.isfinite(phi), phi, 0.0)
    mass = np.where(np.isfinite(mass), mass, 0.0)

    px = pt * np.cos(phi)
    py = pt * np.sin(phi)
    pz = pt * np.sinh(eta)
    e = np.sqrt(np.maximum(px * px + py * py + pz * pz + mass * mass, 0.0))
    return px, py, pz, e


def delta_phi(phi1, phi2):
    d = phi1 - phi2
    return np.arctan2(np.sin(d), np.cos(d))


def invariant_mass(px, py, pz, e):
    m2 = e * e - px * px - py * py - pz * pz
    return np.sqrt(np.maximum(m2, 0.0))


def build_pair_features(raw_obj: dict[str, dict[str, np.ndarray]], object_mask: np.ndarray):
    pairs = [
        ("b1", "b2"),
        ("b1", "lepton"),
        ("b2", "lepton"),
        ("lepton", "met"),
        ("b1", "met"),
        ("b2", "met"),
    ]

    features = []
    names = []

    for a, b in pairs:
        ai = OBJECT_NAMES.index(a)
        bi = OBJECT_NAMES.index(b)

        pt1 = raw_obj[a]["pt"]
        eta1 = raw_obj[a]["eta"]
        phi1 = raw_obj[a]["phi"]
        m1 = raw_obj[a]["mass"]

        pt2 = raw_obj[b]["pt"]
        eta2 = raw_obj[b]["eta"]
        phi2 = raw_obj[b]["phi"]
        m2 = raw_obj[b]["mass"]

        px1, py1, pz1, e1 = pt_eta_phi_m_to_cartesian(pt1, eta1, phi1, m1)
        px2, py2, pz2, e2 = pt_eta_phi_m_to_cartesian(pt2, eta2, phi2, m2)

        pair_px = px1 + px2
        pair_py = py1 + py2
        pair_pz = pz1 + pz2
        pair_e = e1 + e2

        pair_m = invariant_mass(pair_px, pair_py, pair_pz, pair_e)
        pair_pt = np.sqrt(pair_px * pair_px + pair_py * pair_py)
        dphi = np.abs(delta_phi(phi1, phi2))
        deta = np.abs(eta1 - eta2)
        dr = np.sqrt(deta * deta + dphi * dphi)

        present = object_mask[:, ai] * object_mask[:, bi]

        for arr, suffix in [
            (np.log1p(pair_m), "log_mass"),
            (np.log1p(pair_pt), "log_pt"),
            (deta, "abs_deta"),
            (dphi, "abs_dphi"),
            (dr, "dr"),
            (present, "present"),
        ]:
            features.append(arr)
            names.append(f"{a}_{b}_{suffix}")

    X_pair = np.vstack(features).T.astype(np.float32)
    return X_pair, names


def robust_standardize_matrix(X, train_mask, feature_names):
    X = X.astype(np.float64)
    out = np.zeros_like(X, dtype=np.float32)
    rows = []

    for j, name in enumerate(feature_names):
        x = X[:, j]
        train_x = x[train_mask]
        train_x = train_x[np.isfinite(train_x)]

        if len(train_x) == 0:
            center = 0.0
            scale = 1.0
        else:
            center = float(np.median(train_x))
            q25, q75 = np.percentile(train_x, [25, 75])
            scale = float(q75 - q25)

            if not np.isfinite(scale) or scale <= 1e-8:
                scale = float(np.std(train_x))

            if not np.isfinite(scale) or scale <= 1e-8:
                scale = 1.0

        z = (np.where(np.isfinite(x), x, center) - center) / scale
        z = np.clip(z, -20.0, 20.0)
        out[:, j] = z.astype(np.float32)

        rows.append(
            {
                "feature": name,
                "center_median_train": center,
                "scale_train": scale,
            }
        )

    return out, pd.DataFrame(rows)


def standardize_object_features(X_obj, object_mask, split):
    X = X_obj.copy().astype(np.float64)
    train_mask = split == "train"

    rows = []
    present_idx = OBJECT_FEATURES.index("present")

    for ifeat, feat in enumerate(OBJECT_FEATURES):
        if ifeat == present_idx:
            continue

        vals = X[train_mask, :, ifeat]
        masks = object_mask[train_mask, :] > 0.5
        vals = vals[masks]
        vals = vals[np.isfinite(vals)]

        if len(vals) == 0:
            center = 0.0
            scale = 1.0
        else:
            center = float(np.mean(vals))
            scale = float(np.std(vals))

            if not np.isfinite(scale) or scale <= 1e-8:
                scale = 1.0

        x = X[:, :, ifeat]
        x = np.where(np.isfinite(x), x, center)
        z = (x - center) / scale
        z = np.clip(z, -20.0, 20.0)
        z = np.where(object_mask > 0.5, z, 0.0)

        X[:, :, ifeat] = z

        rows.append(
            {
                "object_feature": feat,
                "center_mean_train_present": center,
                "scale_std_train_present": scale,
            }
        )

    X[:, :, present_idx] = object_mask
    return X.astype(np.float32), pd.DataFrame(rows)


def infer_binary_labels(df: pd.DataFrame) -> np.ndarray:
    if "target" in df.columns:
        return df["target"].astype(int).to_numpy()

    if "y_binary" in df.columns:
        return df["y_binary"].astype(int).to_numpy()

    pg = df["process_group"].astype(str).str.lower()
    sample = df["sample"].astype(str).str.lower()

    is_signal = (
        pg.str.contains("signal")
        | pg.str.contains("hh")
        | sample.str.contains("hh")
        | sample.str.contains("gluglu")
    )

    return is_signal.astype(int).to_numpy()


def infer_multiclass_labels(df: pd.DataFrame):
    if "cms_singlelep_class_id" in df.columns and "cms_singlelep_class" in df.columns:
        y = df["cms_singlelep_class_id"].astype(int).to_numpy()

        names = (
            df[["cms_singlelep_class_id", "cms_singlelep_class"]]
            .drop_duplicates()
            .sort_values("cms_singlelep_class_id")["cms_singlelep_class"]
            .astype(str)
            .tolist()
        )

        return y, names

    pg = df["process_group"].astype(str).str.lower()
    sample = df["sample"].astype(str).str.lower()

    is_hh = infer_binary_labels(df).astype(bool)

    is_top_higgs = (
        pg.str.contains("ttbar")
        | pg.str.contains("single_higgs")
        | pg.str.contains("ttv")
        | pg.str.contains("tth")
        | pg.str.contains("tttt")
        | sample.str.contains("tt")
        | sample.str.contains("tth")
        | sample.str.contains("higgs")
    )

    y = np.full(len(df), 2, dtype=np.int64)
    y[is_top_higgs.to_numpy()] = 1
    y[is_hh] = 0

    return y, ["HH_ggF_like", "Top_Higgs", "WJets_Other"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        default="outputs/hadronic_w_top_features_lepemu_rough_recoMET/onelep_pre_mbb_classifier_input_with_hadronic_w_top_features.parquet",
    )
    parser.add_argument(
        "--outdir",
        default="outputs/lbn_inputs_lepemu_rough_recoMET_v0",
    )
    args = parser.parse_args()

    input_path = Path(args.input)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    print("Reading:", input_path)
    df0 = pd.read_parquet(input_path)

    print("Input rows:", len(df0))
    print("Input columns:", len(df0.columns))

    if "physics_weight_nominal" not in df0.columns:
        raise RuntimeError("Expected physics_weight_nominal in input table.")

    w = pd.to_numeric(df0["physics_weight_nominal"], errors="coerce").to_numpy(dtype=np.float64)

    safe = np.isfinite(w) & (w > 0)

    if "has_physics_weight" in df0.columns:
        safe &= df0["has_physics_weight"].fillna(False).astype(bool).to_numpy()

    df = df0.loc[safe].reset_index(drop=True)

    print("Rows with safe rough physics weights:", len(df))
    print("Excluded rows:", len(df0) - len(df))

    split = df["split"].astype(str).to_numpy()
    physics_weight = df["physics_weight_nominal"].astype(float).to_numpy(dtype=np.float32)

    y_binary = infer_binary_labels(df).astype(np.int64)
    y_multiclass, class_names = infer_multiclass_labels(df)
    y_multiclass = y_multiclass.astype(np.int64)

    print("Binary signal raw:", int(y_binary.sum()))
    print("Binary background raw:", int((y_binary == 0).sum()))

    print("Class names:", class_names)

    for i, name in enumerate(class_names):
        print(f"  {i} {name}: raw={(y_multiclass == i).sum()}")

    n = len(df)
    n_obj = len(OBJECT_NAMES)
    n_feat = len(OBJECT_FEATURES)

    X_obj = np.zeros((n, n_obj, n_feat), dtype=np.float32)
    object_mask = np.zeros((n, n_obj), dtype=np.float32)

    b1_pt = safe_array(df, "b1_pt")
    b1_eta = safe_array(df, "b1_eta")
    b1_phi = safe_array(df, "b1_phi")
    b1_mass = safe_array(df, "b1_mass")

    b2_pt = safe_array(df, "b2_pt")
    b2_eta = safe_array(df, "b2_eta")
    b2_phi = safe_array(df, "b2_phi")
    b2_mass = safe_array(df, "b2_mass")

    lep_pt = safe_array(df, "leading_lepton_pt")
    lep_eta = safe_array(df, "leading_lepton_eta")
    lep_phi = safe_array(df, "leading_lepton_phi")

    is_muon = safe_bool(df, "leading_lepton_is_muon")
    is_electron = safe_bool(df, "leading_lepton_is_electron")
    lep_mass = np.where(is_muon, 0.105658, np.where(is_electron, 0.000511, 0.0))

    met_pt = safe_array(df, "met")
    met_phi = safe_array(df, "met_phi")
    met_eta = safe_array(df, "met_eta", default=0.0)
    met_mass = np.zeros(n, dtype=np.float64)

    raw_obj = {
        "b1": {"pt": b1_pt, "eta": b1_eta, "phi": b1_phi, "mass": b1_mass},
        "b2": {"pt": b2_pt, "eta": b2_eta, "phi": b2_phi, "mass": b2_mass},
        "lepton": {"pt": lep_pt, "eta": lep_eta, "phi": lep_phi, "mass": lep_mass},
        "met": {"pt": met_pt, "eta": met_eta, "phi": met_phi, "mass": met_mass},
    }

    for i, name in enumerate(OBJECT_NAMES):
        pt = raw_obj[name]["pt"]
        eta = raw_obj[name]["eta"]
        phi = raw_obj[name]["phi"]
        mass = raw_obj[name]["mass"]

        present = finite_positive(pt).astype(np.float32)
        object_mask[:, i] = present

        sin_phi, cos_phi = phi_to_sincos(phi)

        X_obj[:, i, OBJECT_FEATURES.index("log_pt")] = np.log1p(np.where(finite_positive(pt), pt, 0.0))
        X_obj[:, i, OBJECT_FEATURES.index("eta")] = np.where(np.isfinite(eta), eta, 0.0)
        X_obj[:, i, OBJECT_FEATURES.index("sin_phi")] = sin_phi
        X_obj[:, i, OBJECT_FEATURES.index("cos_phi")] = cos_phi
        X_obj[:, i, OBJECT_FEATURES.index("log_mass")] = np.log1p(np.where(finite_positive(mass), mass, 0.0))
        X_obj[:, i, OBJECT_FEATURES.index("present")] = present

    X_obj, obj_preproc = standardize_object_features(X_obj, object_mask, split)

    X_pair_raw, pair_feature_names = build_pair_features(raw_obj, object_mask)

    X_pair, pair_preproc = robust_standardize_matrix(
        X_pair_raw,
        train_mask=(split == "train"),
        feature_names=pair_feature_names,
    )

    aux_cols = [c for c in AUX_CANDIDATES if c in df.columns]

    print("Aux features:", len(aux_cols))

    for c in aux_cols:
        print(" ", c)

    X_aux_raw = np.vstack(
        [pd.to_numeric(df[c], errors="coerce").to_numpy(dtype=np.float64) for c in aux_cols]
    ).T

    X_aux, aux_preproc = robust_standardize_matrix(
        X_aux_raw,
        train_mask=(split == "train"),
        feature_names=aux_cols,
    )

    metadata_cols = [
        c for c in [
            "split",
            "target",
            "sample",
            "process_group",
            "file",
            "file_index",
            "event_in_file",
            "is_calib",
            "physics_weight_nominal",
            "has_physics_weight",
        ]
        if c in df.columns
    ]

    meta = df[metadata_cols].copy()
    meta["y_binary"] = y_binary
    meta["y_multiclass"] = y_multiclass
    meta["cms_singlelep_class"] = [class_names[i] for i in y_multiclass]
    meta["cms_singlelep_class_id"] = y_multiclass

    npz_path = outdir / "lbn_fourvector_inputs_v0.npz"
    meta_path = outdir / "event_metadata.parquet"

    np.savez_compressed(
        npz_path,
        X_obj=X_obj,
        object_mask=object_mask,
        X_pair=X_pair,
        X_aux=X_aux,
        y_binary=y_binary,
        y_multiclass=y_multiclass,
        physics_weight=physics_weight,
        split=split,
        object_names=np.array(OBJECT_NAMES),
        object_features=np.array(OBJECT_FEATURES),
        pair_feature_names=np.array(pair_feature_names),
        aux_feature_names=np.array(aux_cols),
        class_names=np.array(class_names),
    )

    meta.to_parquet(meta_path, index=False)

    obj_preproc.to_csv(outdir / "object_preprocessing.csv", index=False)
    pair_preproc.to_csv(outdir / "pair_preprocessing.csv", index=False)
    aux_preproc.to_csv(outdir / "aux_preprocessing.csv", index=False)

    with open(outdir / "object_feature_names.json", "w") as f:
        json.dump(
            {
                "object_names": OBJECT_NAMES,
                "object_features": OBJECT_FEATURES,
                "pair_feature_names": pair_feature_names,
                "aux_feature_names": aux_cols,
                "class_names": class_names,
            },
            f,
            indent=2,
        )

    split_summary = []

    for s in ["train", "val", "test"]:
        m = split == s

        split_summary.append(
            {
                "split": s,
                "raw_events": int(m.sum()),
                "signal_raw": int(y_binary[m].sum()),
                "background_raw": int((y_binary[m] == 0).sum()),
                "signal_weighted": float(physics_weight[m & (y_binary == 1)].sum()),
                "background_weighted": float(physics_weight[m & (y_binary == 0)].sum()),
            }
        )

    split_summary = pd.DataFrame(split_summary)
    split_summary.to_csv(outdir / "split_summary.csv", index=False)

    print("\nWrote:", npz_path)
    print("Wrote:", meta_path)

    print("\nShapes:")
    print("  X_obj:", X_obj.shape)
    print("  object_mask:", object_mask.shape)
    print("  X_pair:", X_pair.shape)
    print("  X_aux:", X_aux.shape)

    print("\nSplit summary:")
    print(split_summary.to_string(index=False))


if __name__ == "__main__":
    main()
