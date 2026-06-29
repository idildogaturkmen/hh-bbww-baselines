'''
Adds topology features to a parquet table of events, and saves the augmented table and feature list.
Features added:
- pt_bb: transverse momentum of the bb system
- eta_bb_reco: pseudorapidity of the bb system
- phi_bb_reco: azimuthal angle of the bb system
- dphi_bb_met: delta phi between the bb system and MET
- dphi_lep_bb: delta phi between the leading lepton and the bb system
- dr_lep_b1: delta R between the leading lepton and b1
- dr_lep_b2: delta R between the leading lepton and b2
- min_dr_lep_b: minimum delta R between the leading lepton and the two b jets
- dphi_met_b1: delta phi between MET and b1
- dphi_met_b2: delta phi between MET and b2
- min_dphi_met_b: minimum delta phi between MET and the two b jets
- m_lep_b1: invariant mass of the leading lepton and b1
- m_lep_b2: invariant mass of the leading lepton and b2
- min_m_lep_b: minimum invariant mass of the leading lepton and the two b jets
- mt_bb_met: transverse mass of the bb system and MET
- pt_bb_over_ht: ratio of pt_bb to HT
- met_over_pt_bb: ratio of MET to pt_bb     
'''
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


NEW_FEATURES = [
    "pt_bb",
    "eta_bb_reco",
    "phi_bb_reco",
    "dphi_bb_met",
    "dphi_lep_bb",
    "dr_lep_b1",
    "dr_lep_b2",
    "min_dr_lep_b",
    "dphi_met_b1",
    "dphi_met_b2",
    "min_dphi_met_b",
    "m_lep_b1",
    "m_lep_b2",
    "min_m_lep_b",
    "mt_bb_met",
    "pt_bb_over_ht",
    "met_over_pt_bb",
]


def dphi(a, b):
    x = a - b
    return np.arctan2(np.sin(x), np.cos(x))


def delta_r(eta1, phi1, eta2, phi2):
    return np.sqrt((eta1 - eta2) ** 2 + dphi(phi1, phi2) ** 2)


def fourvec(pt, eta, phi, mass):
    px = pt * np.cos(phi)
    py = pt * np.sin(phi)
    pz = pt * np.sinh(eta)
    e = np.sqrt(np.maximum(0.0, px**2 + py**2 + pz**2 + mass**2))
    return px, py, pz, e


def inv_mass(px, py, pz, e):
    m2 = e**2 - px**2 - py**2 - pz**2
    return np.sqrt(np.maximum(0.0, m2))


def eta_from_p(px, py, pz):
    p = np.sqrt(px**2 + py**2 + pz**2)
    denom = np.maximum(1e-12, p - pz)
    return 0.5 * np.log(np.maximum(1e-12, (p + pz) / denom))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        default="outputs/lepton_met_features_lepemu_rough/onelep_pre_mbb_classifier_input_with_lepton_met_features.parquet",
    )
    parser.add_argument(
        "--features",
        default="outputs/lepton_met_features_lepemu_rough/feature_columns_augmented.json",
    )
    parser.add_argument(
        "--outdir",
        default="outputs/topology_features_lepemu_rough",
    )
    args = parser.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    df = pd.read_parquet(args.input)

    # b-jet four-vectors
    b1_px, b1_py, b1_pz, b1_e = fourvec(df["b1_pt"], df["b1_eta"], df["b1_phi"], df["b1_mass"])
    b2_px, b2_py, b2_pz, b2_e = fourvec(df["b2_pt"], df["b2_eta"], df["b2_phi"], df["b2_mass"])

    bb_px = b1_px + b2_px
    bb_py = b1_py + b2_py
    bb_pz = b1_pz + b2_pz
    bb_e = b1_e + b2_e

    df["pt_bb"] = np.sqrt(bb_px**2 + bb_py**2)
    df["phi_bb_reco"] = np.arctan2(bb_py, bb_px)
    df["eta_bb_reco"] = eta_from_p(bb_px, bb_py, bb_pz)

    # Lepton as approximately massless
    lep_px, lep_py, lep_pz, lep_e = fourvec(
        df["leading_lepton_pt"],
        df["leading_lepton_eta"],
        df["leading_lepton_phi"],
        0.0,
    )

    df["dr_lep_b1"] = delta_r(df["leading_lepton_eta"], df["leading_lepton_phi"], df["b1_eta"], df["b1_phi"])
    df["dr_lep_b2"] = delta_r(df["leading_lepton_eta"], df["leading_lepton_phi"], df["b2_eta"], df["b2_phi"])
    df["min_dr_lep_b"] = np.minimum(df["dr_lep_b1"], df["dr_lep_b2"])

    df["m_lep_b1"] = inv_mass(lep_px + b1_px, lep_py + b1_py, lep_pz + b1_pz, lep_e + b1_e)
    df["m_lep_b2"] = inv_mass(lep_px + b2_px, lep_py + b2_py, lep_pz + b2_pz, lep_e + b2_e)
    df["min_m_lep_b"] = np.minimum(df["m_lep_b1"], df["m_lep_b2"])

    df["dphi_bb_met"] = np.abs(dphi(df["phi_bb_reco"], df["met_phi"]))
    df["dphi_lep_bb"] = np.abs(dphi(df["leading_lepton_phi"], df["phi_bb_reco"]))

    df["dphi_met_b1"] = np.abs(dphi(df["met_phi"], df["b1_phi"]))
    df["dphi_met_b2"] = np.abs(dphi(df["met_phi"], df["b2_phi"]))
    df["min_dphi_met_b"] = np.minimum(df["dphi_met_b1"], df["dphi_met_b2"])

    # Transverse mass of bb system + MET
    mbb = df["mbb_pt_binned_scaled"].fillna(df["mbb_uncorrected"])
    et_bb = np.sqrt(np.maximum(0.0, mbb**2 + df["pt_bb"]**2))
    met_px = df["met"] * np.cos(df["met_phi"])
    met_py = df["met"] * np.sin(df["met_phi"])
    mt2 = (et_bb + df["met"])**2 - (bb_px + met_px)**2 - (bb_py + met_py)**2
    df["mt_bb_met"] = np.sqrt(np.maximum(0.0, mt2))

    df["pt_bb_over_ht"] = df["pt_bb"] / df["ht"].replace(0, np.nan)
    df["met_over_pt_bb"] = df["met"] / df["pt_bb"].replace(0, np.nan)

    # Clean infinities
    for c in NEW_FEATURES:
        df[c] = df[c].replace([np.inf, -np.inf], np.nan)

    out_table = outdir / "onelep_pre_mbb_classifier_input_with_topology_features.parquet"
    df.to_parquet(out_table, index=False)
    df.head(500).to_csv(outdir / "topology_feature_preview.csv", index=False)

    with open(args.features) as f:
        features = json.load(f)

    augmented = list(features)
    for c in NEW_FEATURES:
        if c not in augmented:
            augmented.append(c)

    out_features = outdir / "feature_columns_topology.json"
    with open(out_features, "w") as f:
        json.dump(augmented, f, indent=2)

    print("Wrote:", out_table)
    print("Wrote:", out_features)
    print("\nNew features:")
    for c in NEW_FEATURES:
        print(" ", c)


if __name__ == "__main__":
    main()
