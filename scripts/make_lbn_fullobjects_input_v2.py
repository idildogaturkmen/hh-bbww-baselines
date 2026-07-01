'''
Makes a new LBN input file with full object features, including b-tagging and lepton information.
This script reads the v0 LBN input file and the event metadata, processes the events to
extract the relevant object features, and saves the new LBN input file along with metadata and preprocessing information.
The new LBN input file will contain:
- Object features for b-jets, leptons, MET, and non-b jets.
- Pairwise features for all object pairs.
The script also generates summary statistics and saves them in a JSON file.
Usage:
    python scripts/make_lbn_fullobjects_input_v2.py \
        --v0-input <path_to_v0_input.npz> \
        --metadata <path_to_event_metadata.parquet> \
        --outdir <output_directory> \
        --jet-pt-min <minimum_jet_pt> \
        --jet-eta-max <maximum_jet_eta>
'''

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq


OBJECT_NAMES = ["b1", "b2", "lepton", "met", "j1", "j2", "j3", "j4"]

OBJECT_FEATURES = [
    "log_pt",
    "eta",
    "sin_phi",
    "cos_phi",
    "log_mass",
    "btag",
    "btagphys",
    "charge",
    "is_bslot",
    "is_nonbjet",
    "is_lepton",
    "is_met",
    "present",
]

PAIR_FEATURES_PER_PAIR = [
    "log_mass",
    "log_pt",
    "abs_deta",
    "abs_dphi",
    "dr",
    "present",
]

SOURCE_COLUMNS = [
    "FullReco_JetAK4_PT",
    "FullReco_JetAK4_Eta",
    "FullReco_JetAK4_Phi",
    "FullReco_JetAK4_Mass",
    "FullReco_JetAK4_BTag",
    "FullReco_JetAK4_BTagPhys",
    "FullReco_JetAK4_Charge",
    "FullReco_Electron_PT",
    "FullReco_Electron_Eta",
    "FullReco_Electron_Phi",
    "FullReco_MuonTight_PT",
    "FullReco_MuonTight_Eta",
    "FullReco_MuonTight_Phi",
    "FullReco_MET_MET",
    "FullReco_MET_Phi",
    "FullReco_PUPPIMET_MET",
    "FullReco_PUPPIMET_Phi",
]


def as_list(x):
    if x is None:
        return []
    if isinstance(x, list):
        return x
    if hasattr(x, "tolist"):
        y = x.tolist()
        return y if isinstance(y, list) else [y]
    return list(x)


def get_list(data, col, idx):
    if col not in data:
        return []
    return as_list(data[col][idx])


def safe_float(x, default=0.0):
    try:
        y = float(x)
        if np.isfinite(y):
            return y
    except Exception:
        pass
    return default


def delta_phi(phi1, phi2):
    d = phi1 - phi2
    return (d + np.pi) % (2 * np.pi) - np.pi


def fourvec(pt, eta, phi, mass):
    pt = max(float(pt), 0.0)
    eta = float(eta)
    phi = float(phi)
    mass = max(float(mass), 0.0)

    px = pt * np.cos(phi)
    py = pt * np.sin(phi)
    pz = pt * np.sinh(eta)
    e = np.sqrt(max(px * px + py * py + pz * pz + mass * mass, 0.0))
    return e, px, py, pz


def empty_obj():
    return {
        "pt": 0.0,
        "eta": 0.0,
        "phi": 0.0,
        "mass": 0.0,
        "btag": 0.0,
        "btagphys": 0.0,
        "charge": 0.0,
        "is_bslot": 0.0,
        "is_nonbjet": 0.0,
        "is_lepton": 0.0,
        "is_met": 0.0,
        "present": False,
    }


def make_object_features(obj):
    if not obj["present"]:
        return [0.0] * len(OBJECT_FEATURES)

    pt = max(safe_float(obj.get("pt", 0.0)), 0.0)
    eta = safe_float(obj.get("eta", 0.0))
    phi = safe_float(obj.get("phi", 0.0))
    mass = max(safe_float(obj.get("mass", 0.0)), 0.0)

    return [
        np.log1p(pt),
        eta,
        np.sin(phi),
        np.cos(phi),
        np.log1p(mass),
        safe_float(obj.get("btag", 0.0)),
        safe_float(obj.get("btagphys", 0.0)),
        safe_float(obj.get("charge", 0.0)),
        safe_float(obj.get("is_bslot", 0.0)),
        safe_float(obj.get("is_nonbjet", 0.0)),
        safe_float(obj.get("is_lepton", 0.0)),
        safe_float(obj.get("is_met", 0.0)),
        1.0,
    ]


def pair_features(obj_a, obj_b):
    if not obj_a["present"] or not obj_b["present"]:
        return [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

    e1, px1, py1, pz1 = fourvec(obj_a["pt"], obj_a["eta"], obj_a["phi"], obj_a["mass"])
    e2, px2, py2, pz2 = fourvec(obj_b["pt"], obj_b["eta"], obj_b["phi"], obj_b["mass"])

    e = e1 + e2
    px = px1 + px2
    py = py1 + py2
    pz = pz1 + pz2

    m2 = e * e - px * px - py * py - pz * pz
    mass = np.sqrt(max(m2, 0.0))
    pt = np.sqrt(px * px + py * py)

    deta = abs(obj_a["eta"] - obj_b["eta"])
    dphi = abs(delta_phi(obj_a["phi"], obj_b["phi"]))
    dr = np.sqrt(deta * deta + dphi * dphi)

    return [
        np.log1p(mass),
        np.log1p(pt),
        deta,
        dphi,
        dr,
        1.0,
    ]


def build_event_objects(data, idx, jet_pt_min=20.0, jet_eta_max=5.0):
    jet_pt = get_list(data, "FullReco_JetAK4_PT", idx)
    jet_eta = get_list(data, "FullReco_JetAK4_Eta", idx)
    jet_phi = get_list(data, "FullReco_JetAK4_Phi", idx)
    jet_mass = get_list(data, "FullReco_JetAK4_Mass", idx)
    jet_btag = get_list(data, "FullReco_JetAK4_BTag", idx)
    jet_btagphys = get_list(data, "FullReco_JetAK4_BTagPhys", idx)
    jet_charge = get_list(data, "FullReco_JetAK4_Charge", idx)

    njet = min(len(jet_pt), len(jet_eta), len(jet_phi), len(jet_mass))
    jets = []

    for j in range(njet):
        pt = safe_float(jet_pt[j])
        eta = safe_float(jet_eta[j])
        phi = safe_float(jet_phi[j])
        mass = safe_float(jet_mass[j])

        if pt < jet_pt_min:
            continue
        if abs(eta) > jet_eta_max:
            continue

        btag = safe_float(jet_btag[j]) if j < len(jet_btag) else 0.0
        btagphys = safe_float(jet_btagphys[j]) if j < len(jet_btagphys) else btag
        charge = safe_float(jet_charge[j]) if j < len(jet_charge) else 0.0

        jets.append(
            {
                "idx": j,
                "pt": pt,
                "eta": eta,
                "phi": phi,
                "mass": mass,
                "btag": btag,
                "btagphys": btagphys,
                "charge": charge,
                "is_bslot": 0.0,
                "is_nonbjet": 0.0,
                "is_lepton": 0.0,
                "is_met": 0.0,
                "present": True,
            }
        )

    jets_for_b = sorted(
        jets,
        key=lambda x: (x["btagphys"], x["btag"], x["pt"]),
        reverse=True,
    )

    bjets = []
    used_indices = set()

    for slot in range(2):
        if slot < len(jets_for_b):
            obj = dict(jets_for_b[slot])
            obj["is_bslot"] = 1.0
            bjets.append(obj)
            used_indices.add(obj["idx"])
        else:
            bjets.append(empty_obj())

    nonb = [j for j in jets if j["idx"] not in used_indices]
    nonb = sorted(nonb, key=lambda x: x["pt"], reverse=True)

    nonb_objs = []
    for k in range(4):
        if k < len(nonb):
            obj = dict(nonb[k])
            obj["is_nonbjet"] = 1.0
            nonb_objs.append(obj)
        else:
            nonb_objs.append(empty_obj())

    ele_pt = get_list(data, "FullReco_Electron_PT", idx)
    ele_eta = get_list(data, "FullReco_Electron_Eta", idx)
    ele_phi = get_list(data, "FullReco_Electron_Phi", idx)

    mu_pt = get_list(data, "FullReco_MuonTight_PT", idx)
    mu_eta = get_list(data, "FullReco_MuonTight_Eta", idx)
    mu_phi = get_list(data, "FullReco_MuonTight_Phi", idx)

    leptons = []

    ne = min(len(ele_pt), len(ele_eta), len(ele_phi))
    for i in range(ne):
        leptons.append(
            {
                "pt": safe_float(ele_pt[i]),
                "eta": safe_float(ele_eta[i]),
                "phi": safe_float(ele_phi[i]),
                "mass": 0.000511,
                "btag": 0.0,
                "btagphys": 0.0,
                "charge": 0.0,
                "is_bslot": 0.0,
                "is_nonbjet": 0.0,
                "is_lepton": 1.0,
                "is_met": 0.0,
                "present": True,
            }
        )

    nm = min(len(mu_pt), len(mu_eta), len(mu_phi))
    for i in range(nm):
        leptons.append(
            {
                "pt": safe_float(mu_pt[i]),
                "eta": safe_float(mu_eta[i]),
                "phi": safe_float(mu_phi[i]),
                "mass": 0.105658,
                "btag": 0.0,
                "btagphys": 0.0,
                "charge": 0.0,
                "is_bslot": 0.0,
                "is_nonbjet": 0.0,
                "is_lepton": 1.0,
                "is_met": 0.0,
                "present": True,
            }
        )

    leptons = sorted(leptons, key=lambda x: x["pt"], reverse=True)
    lepton = leptons[0] if leptons else empty_obj()

    met_vals = get_list(data, "FullReco_MET_MET", idx)
    met_phis = get_list(data, "FullReco_MET_Phi", idx)

    if len(met_vals) == 0 or len(met_phis) == 0:
        met_vals = get_list(data, "FullReco_PUPPIMET_MET", idx)
        met_phis = get_list(data, "FullReco_PUPPIMET_Phi", idx)

    if len(met_vals) > 0 and len(met_phis) > 0:
        met = {
            "pt": safe_float(met_vals[0]),
            "eta": 0.0,
            "phi": safe_float(met_phis[0]),
            "mass": 0.0,
            "btag": 0.0,
            "btagphys": 0.0,
            "charge": 0.0,
            "is_bslot": 0.0,
            "is_nonbjet": 0.0,
            "is_lepton": 0.0,
            "is_met": 1.0,
            "present": True,
        }
    else:
        met = empty_obj()

    return [bjets[0], bjets[1], lepton, met] + nonb_objs


def standardize_object_features(X_obj, split):
    X = X_obj.copy()
    scale_indices = [0, 1, 4]
    train_mask = split == "train"

    rows = []
    for i in np.where(train_mask)[0]:
        for j in range(X.shape[1]):
            if X[i, j, -1] > 0.5:
                rows.append(X[i, j, scale_indices])

    if not rows:
        return X, pd.DataFrame()

    arr = np.asarray(rows, dtype=np.float32)
    med = np.nanmedian(arr, axis=0)
    std = np.nanstd(arr, axis=0)
    std = np.where(std > 1e-6, std, 1.0)

    for k, feat_idx in enumerate(scale_indices):
        X[:, :, feat_idx] = (X[:, :, feat_idx] - med[k]) / std[k]
        X[:, :, feat_idx] *= X[:, :, -1]

    prep = pd.DataFrame(
        {
            "feature": [OBJECT_FEATURES[i] for i in scale_indices],
            "median_train_present": med,
            "std_train_present": std,
        }
    )
    return X, prep


def standardize_pair_features(X_pair, split):
    X = X_pair.copy()
    train_mask = split == "train"

    rows = []
    for i in np.where(train_mask)[0]:
        for start in range(0, X.shape[1], len(PAIR_FEATURES_PER_PAIR)):
            if X[i, start + 5] > 0.5:
                rows.append(X[i, start : start + 5])

    if not rows:
        return X, pd.DataFrame()

    arr = np.asarray(rows, dtype=np.float32)
    med = np.nanmedian(arr, axis=0)
    std = np.nanstd(arr, axis=0)
    std = np.where(std > 1e-6, std, 1.0)

    for start in range(0, X.shape[1], len(PAIR_FEATURES_PER_PAIR)):
        present = X[:, start + 5]
        for k in range(5):
            X[:, start + k] = (X[:, start + k] - med[k]) / std[k]
            X[:, start + k] *= present

    prep = pd.DataFrame(
        {
            "feature": PAIR_FEATURES_PER_PAIR[:5],
            "median_train_present": med,
            "std_train_present": std,
        }
    )
    return X, prep


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--v0-input",
        default="outputs/lbn_inputs_lepemu_rough_recoMET_v0/lbn_fourvector_inputs_v0.npz",
    )
    parser.add_argument(
        "--metadata",
        default="outputs/lbn_inputs_lepemu_rough_recoMET_v0/event_metadata.parquet",
    )
    parser.add_argument(
        "--outdir",
        default="outputs/lbn_inputs_lepemu_fullobjects_v2",
    )
    parser.add_argument("--jet-pt-min", type=float, default=20.0)
    parser.add_argument("--jet-eta-max", type=float, default=5.0)
    args = parser.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    print("Reading v0 input:", args.v0_input)
    z0 = np.load(args.v0_input, allow_pickle=True)
    payload = {k: z0[k] for k in z0.files}

    print("Reading metadata:", args.metadata)
    meta = pd.read_parquet(args.metadata).reset_index(drop=True)

    n = len(meta)
    print("Rows:", n)

    if "file" not in meta.columns:
        raise ValueError("metadata must contain a 'file' column")
    if "event_in_file" not in meta.columns:
        raise ValueError("metadata must contain an 'event_in_file' column")

    if "split" in meta.columns:
        split = meta["split"].astype(str).to_numpy()
    elif "split" in payload:
        split = payload["split"].astype(str)
    else:
        raise ValueError("Could not find split in metadata or v0 npz")

    X_obj_raw = np.zeros((n, len(OBJECT_NAMES), len(OBJECT_FEATURES)), dtype=np.float32)
    object_mask = np.zeros((n, len(OBJECT_NAMES)), dtype=np.float32)

    pair_names = []
    for i in range(len(OBJECT_NAMES)):
        for j in range(i + 1, len(OBJECT_NAMES)):
            for f in PAIR_FEATURES_PER_PAIR:
                pair_names.append(f"{OBJECT_NAMES[i]}_{OBJECT_NAMES[j]}_{f}")

    X_pair_raw = np.zeros((n, len(pair_names)), dtype=np.float32)

    failures = []
    object_count_rows = []

    grouped = meta.groupby("file", sort=False).indices
    print("Number of source files:", len(grouped))

    for file_i, (file_path, row_indices) in enumerate(grouped.items(), start=1):
        p = Path(file_path)
        if not p.exists():
            failures.append((file_path, "missing_file", len(row_indices)))
            continue

        try:
            schema_cols = set(pq.read_schema(p).names)
            cols = [c for c in SOURCE_COLUMNS if c in schema_cols]
            table = pq.read_table(p, columns=cols)
            data = {c: table[c].to_pylist() for c in cols}
        except Exception as e:
            failures.append((file_path, f"read_error: {e}", len(row_indices)))
            continue

        if file_i % 25 == 0 or file_i == 1:
            print(f"  processed source file {file_i}/{len(grouped)}: {p.name}")

        for out_row in row_indices:
            event_idx = int(meta.loc[out_row, "event_in_file"])
            if event_idx < 0:
                failures.append((file_path, "negative_event_index", 1))
                continue

            any_col = cols[0] if cols else None
            if any_col is None or event_idx >= len(data[any_col]):
                failures.append((file_path, "event_index_out_of_range", 1))
                continue

            objs_raw = build_event_objects(
                data,
                event_idx,
                jet_pt_min=args.jet_pt_min,
                jet_eta_max=args.jet_eta_max,
            )

            for j, obj in enumerate(objs_raw):
                X_obj_raw[out_row, j, :] = np.asarray(make_object_features(obj), dtype=np.float32)
                object_mask[out_row, j] = 1.0 if obj["present"] else 0.0

            pair_values = []
            for a in range(len(OBJECT_NAMES)):
                for b in range(a + 1, len(OBJECT_NAMES)):
                    pair_values.extend(pair_features(objs_raw[a], objs_raw[b]))

            X_pair_raw[out_row, :] = np.asarray(pair_values, dtype=np.float32)

            object_count_rows.append(
                {
                    "row": out_row,
                    "n_present_objects": int(object_mask[out_row].sum()),
                    "has_b1": int(object_mask[out_row, 0]),
                    "has_b2": int(object_mask[out_row, 1]),
                    "has_lepton": int(object_mask[out_row, 2]),
                    "has_met": int(object_mask[out_row, 3]),
                    "n_nonb_present": int(object_mask[out_row, 4:].sum()),
                }
            )

    print("Standardizing object features...")
    X_obj, obj_prep = standardize_object_features(X_obj_raw, split)

    print("Standardizing pair features...")
    X_pair, pair_prep = standardize_pair_features(X_pair_raw, split)

    payload["X_obj"] = X_obj.astype(np.float32)
    payload["object_mask"] = object_mask.astype(np.float32)
    payload["X_pair"] = X_pair.astype(np.float32)
    payload["object_names"] = np.asarray(OBJECT_NAMES)
    payload["object_feature_names"] = np.asarray(OBJECT_FEATURES)
    payload["pair_feature_names"] = np.asarray(pair_names)

    out_npz = outdir / "lbn_fullobjects_inputs_v2.npz"
    np.savez_compressed(out_npz, **payload)

    meta_out = outdir / "event_metadata.parquet"
    meta.to_parquet(meta_out, index=False)

    obj_prep.to_csv(outdir / "object_preprocessing.csv", index=False)
    pair_prep.to_csv(outdir / "pair_preprocessing.csv", index=False)

    pd.DataFrame(object_count_rows).to_csv(
        outdir / "object_presence_summary_per_event.csv",
        index=False,
    )

    if failures:
        pd.DataFrame(failures, columns=["file", "reason", "n_rows"]).to_csv(
            outdir / "v2_build_failures.csv",
            index=False,
        )
    else:
        pd.DataFrame(columns=["file", "reason", "n_rows"]).to_csv(
            outdir / "v2_build_failures.csv",
            index=False,
        )

    summary = {
        "n_rows": int(n),
        "n_source_files": int(len(grouped)),
        "X_obj_shape": list(X_obj.shape),
        "X_pair_shape": list(X_pair.shape),
        "object_names": OBJECT_NAMES,
        "object_features": OBJECT_FEATURES,
        "n_pair_features": int(X_pair.shape[1]),
        "jet_pt_min": args.jet_pt_min,
        "jet_eta_max": args.jet_eta_max,
        "n_failures": int(len(failures)),
        "mean_present_objects": float(object_mask.sum(axis=1).mean()),
        "mean_nonb_present": float(object_mask[:, 4:].sum(axis=1).mean()),
        "frac_has_b1": float(object_mask[:, 0].mean()),
        "frac_has_b2": float(object_mask[:, 1].mean()),
        "frac_has_lepton": float(object_mask[:, 2].mean()),
        "frac_has_met": float(object_mask[:, 3].mean()),
    }

    with open(outdir / "v2_input_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print("\n=== v2 input summary ===")
    for k, v in summary.items():
        print(f"{k}: {v}")

    print("\nWrote:")
    print(" ", out_npz)
    print(" ", meta_out)
    print(" ", outdir / "v2_input_summary.json")
    print(" ", outdir / "object_presence_summary_per_event.csv")
    print(" ", outdir / "v2_build_failures.csv")


if __name__ == "__main__":
    main()
