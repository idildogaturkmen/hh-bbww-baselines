'''
Adds hadronic W and top candidate features to parquet files.
Features added:
- n_selected_ak4_jets: number of selected AK4 jets
- n_btagged_jets: number of b-tagged jets
- n_nonb_jets: number of non-b-tagged jets 
- third_jet_pt: pt of the third jet (if present)
- fourth_jet_pt: pt of the fourth jet (if present)
- n_nonb_jet_pairs: number of non-b jet pairs
- hadronic_W_candidate_mass: mass of the best hadronic W candidate (from non-b jet pairs)
- abs_mjj_minus_W: absolute difference between hadronic W candidate mass and W mass
- hadronic_W_candidate_dr: delta R between the two jets forming the best hadronic W candidate
- hadronic_W_candidate_pt: pt of the best hadronic W candidate
- hadronic_W_candidate_deta: delta eta between the two jets forming the best hadronic W candidate
- hadronic_W_candidate_dphi: delta phi between the two jets forming the best hadronic W candidate
- n_bjj_top_candidates: number of top candidates formed from one b-tagged jet and two non-b jets
- top_candidate_mass_closest: mass of the top candidate closest to the top mass
- min_abs_m_bjj_minus_top: minimum absolute difference between top candidate mass and top mass
- min_top_candidate_mass: minimum mass of the top candidates

'''
from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq


W_MASS_GEV = 80.4
TOP_MASS_GEV = 172.5

JET_BRANCHES = {
    "pt": "FullReco_JetAK4_PT",
    "eta": "FullReco_JetAK4_Eta",
    "phi": "FullReco_JetAK4_Phi",
    "mass": "FullReco_JetAK4_Mass",
    "btag": "FullReco_JetAK4_BTag",
    "btag_phys": "FullReco_JetAK4_BTagPhys",
}

NEW_FEATURES = [
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


def list_or_empty(x):
    if x is None:
        return []
    if isinstance(x, float) and np.isnan(x):
        return []
    if isinstance(x, np.ndarray):
        return x.tolist()
    if isinstance(x, (list, tuple)):
        return list(x)
    return [x]


def dphi(phi1, phi2):
    return float(np.arctan2(np.sin(phi1 - phi2), np.cos(phi1 - phi2)))


def delta_r(j1, j2):
    return float(np.sqrt((j1["eta"] - j2["eta"]) ** 2 + dphi(j1["phi"], j2["phi"]) ** 2))


def fourvec(pt, eta, phi, mass):
    px = pt * np.cos(phi)
    py = pt * np.sin(phi)
    pz = pt * np.sinh(eta)
    e = np.sqrt(max(0.0, px * px + py * py + pz * pz + mass * mass))
    return px, py, pz, e


def add_fourvec(vectors):
    px = sum(v[0] for v in vectors)
    py = sum(v[1] for v in vectors)
    pz = sum(v[2] for v in vectors)
    e = sum(v[3] for v in vectors)
    return px, py, pz, e


def inv_mass_from_vec(vec):
    px, py, pz, e = vec
    m2 = e * e - px * px - py * py - pz * pz
    return float(np.sqrt(max(0.0, m2)))


def pt_from_vec(vec):
    px, py, _, _ = vec
    return float(np.sqrt(px * px + py * py))


def event_features(row) -> dict:
    pts = list_or_empty(row[JET_BRANCHES["pt"]])
    etas = list_or_empty(row[JET_BRANCHES["eta"]])
    phis = list_or_empty(row[JET_BRANCHES["phi"]])
    masses = list_or_empty(row[JET_BRANCHES["mass"]])
    btags = list_or_empty(row[JET_BRANCHES["btag"]])

    jets = []
    for i, pt in enumerate(pts):
        try:
            pt = float(pt)
            eta = float(etas[i])
            phi = float(phis[i])
            mass = float(masses[i])
            btag = float(btags[i])
        except Exception:
            continue

        if not np.isfinite(pt + eta + phi + mass + btag):
            continue

        # Match your earlier AK4 jet selection.
        if pt > 25.0 and abs(eta) < 2.4:
            vec = fourvec(pt, eta, phi, mass)
            jets.append(
                {
                    "pt": pt,
                    "eta": eta,
                    "phi": phi,
                    "mass": mass,
                    "btag": btag,
                    "vec": vec,
                }
            )

    jets = sorted(jets, key=lambda j: j["pt"], reverse=True)
    bjets = [j for j in jets if j["btag"] > 0.5]
    nonb_jets = [j for j in jets if j["btag"] <= 0.5]

    out = {
        "n_selected_ak4_jets": len(jets),
        "n_btagged_jets": len(bjets),
        "n_nonb_jets": len(nonb_jets),
        "third_jet_pt": jets[2]["pt"] if len(jets) >= 3 else np.nan,
        "fourth_jet_pt": jets[3]["pt"] if len(jets) >= 4 else np.nan,
        "n_nonb_jet_pairs": 0,
        "hadronic_W_candidate_mass": np.nan,
        "abs_mjj_minus_W": np.nan,
        "hadronic_W_candidate_dr": np.nan,
        "hadronic_W_candidate_pt": np.nan,
        "hadronic_W_candidate_deta": np.nan,
        "hadronic_W_candidate_dphi": np.nan,
        "n_bjj_top_candidates": 0,
        "top_candidate_mass_closest": np.nan,
        "min_abs_m_bjj_minus_top": np.nan,
        "min_top_candidate_mass": np.nan,
    }

    # Best W candidate from non-b jet pairs: choose mjj closest to W mass.
    nonb_pairs = list(itertools.combinations(nonb_jets, 2))
    out["n_nonb_jet_pairs"] = len(nonb_pairs)

    best_w = None
    best_w_diff = np.inf

    for j1, j2 in nonb_pairs:
        vec = add_fourvec([j1["vec"], j2["vec"]])
        mjj = inv_mass_from_vec(vec)
        diff = abs(mjj - W_MASS_GEV)

        if diff < best_w_diff:
            best_w_diff = diff
            best_w = (j1, j2, vec, mjj)

    if best_w is not None:
        j1, j2, w_vec, mjj = best_w
        out["hadronic_W_candidate_mass"] = mjj
        out["abs_mjj_minus_W"] = abs(mjj - W_MASS_GEV)
        out["hadronic_W_candidate_dr"] = delta_r(j1, j2)
        out["hadronic_W_candidate_pt"] = pt_from_vec(w_vec)
        out["hadronic_W_candidate_deta"] = abs(j1["eta"] - j2["eta"])
        out["hadronic_W_candidate_dphi"] = abs(dphi(j1["phi"], j2["phi"]))

    # Top candidates from one b-tagged jet + two non-b jets.
    top_masses = []
    for b in bjets:
        for j1, j2 in nonb_pairs:
            vec = add_fourvec([b["vec"], j1["vec"], j2["vec"]])
            top_masses.append(inv_mass_from_vec(vec))

    out["n_bjj_top_candidates"] = len(top_masses)

    if top_masses:
        top_masses = np.asarray(top_masses, dtype=float)
        diffs = np.abs(top_masses - TOP_MASS_GEV)
        best_idx = int(np.argmin(diffs))
        out["top_candidate_mass_closest"] = float(top_masses[best_idx])
        out["min_abs_m_bjj_minus_top"] = float(diffs[best_idx])
        out["min_top_candidate_mass"] = float(np.min(top_masses))

    return out


def extract_features_for_file(file_path: str, event_indices: list[int]) -> pd.DataFrame:
    columns = list(JET_BRANCHES.values())
    table = pq.read_table(file_path, columns=columns).to_pandas()

    rows = []
    for idx in event_indices:
        idx = int(idx)
        feats = event_features(table.iloc[idx])
        feats["file"] = file_path
        feats["event_in_file"] = idx
        rows.append(feats)

    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        default="outputs/topology_features_lepemu_rough/onelep_pre_mbb_classifier_input_with_topology_features.parquet",
    )
    parser.add_argument(
        "--features",
        default="outputs/topology_features_lepemu_rough/feature_columns_topology.json",
    )
    parser.add_argument(
        "--outdir",
        default="outputs/hadronic_w_top_features_lepemu_rough",
    )
    args = parser.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    df = pd.read_parquet(args.input)

    all_extra = []
    for file_path, group in df.groupby("file"):
        event_indices = sorted(group["event_in_file"].astype(int).unique().tolist())
        print(f"Processing {file_path} with {len(event_indices)} selected events")
        extra = extract_features_for_file(file_path, event_indices)
        all_extra.append(extra)

    extra_df = pd.concat(all_extra, ignore_index=True)

    merged = df.merge(extra_df, on=["file", "event_in_file"], how="left")

    for c in NEW_FEATURES:
        merged[c] = merged[c].replace([np.inf, -np.inf], np.nan)

    out_table = outdir / "onelep_pre_mbb_classifier_input_with_hadronic_w_top_features.parquet"
    merged.to_parquet(out_table, index=False)
    merged.head(500).to_csv(outdir / "hadronic_w_top_feature_preview.csv", index=False)

    with open(args.features) as f:
        features = json.load(f)

    updated_features = list(features)
    for c in NEW_FEATURES:
        if c not in updated_features:
            updated_features.append(c)

    out_features = outdir / "feature_columns_hadronic_w_top.json"
    with open(out_features, "w") as f:
        json.dump(updated_features, f, indent=2)

    print("\nWrote:", out_table)
    print("Wrote:", out_features)
    print("\nAdded features:")
    for c in NEW_FEATURES:
        print(" ", c)


if __name__ == "__main__":
    main()
