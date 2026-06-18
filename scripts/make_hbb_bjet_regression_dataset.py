"""
Build a CMS-inspired b-jet response regression dataset for HH -> bbWW.

Purpose
-------
Create an ML-ready dataset for correcting the pT response of truth-matched
H -> bb reconstructed AK4 b jets.

Each row is one matched H->bb b jet.

Matching chain
--------------
status-23 H->bb gen b quark
        -> nearest reconstructed FullReco AK4 jet with DeltaR < 0.4
        -> nearest FullReco GenJetAK4 with DeltaR < 0.4

Target
------
target = log(GenJetAK4 pT / RecoJetAK4 pT)

Later, a model prediction y_pred can be applied as:
corrected pT = reco pT * exp(y_pred)

Important caveat
----------------
This is simulation-only and CMS-inspired, not a full CMS b-jet regression
reproduction. COLLIDE-1M provides PF candidates and jet constituents, but not
explicit secondary-vertex variables.
"""

import argparse
import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyarrow.parquet as pq


SNAPSHOT_DIR = Path("outputs/dataset_cache/collide_1m/datasets--fastmachinelearning--collide-1m/snapshots")
OUTDIR = Path("outputs/hbb_bjet_regression_dataset")
PLOT_DIR = Path("outputs/plots/hbb_bjet_regression_dataset")
SUMMARY_MD = Path("RESULTS_HBB_BJET_REGRESSION_DATASET.md")

DEFAULT_MAX_RECO_JETS = -1
DR_GENB_RECO = 0.4
DR_RECO_GENJET = 0.4
EPS = 1e-6

COLUMNS = [
    # Reco AK4 jets
    "FullReco_JetAK4_PT",
    "FullReco_JetAK4_Eta",
    "FullReco_JetAK4_Phi",
    "FullReco_JetAK4_Mass",
    "FullReco_JetAK4_BTag",
    "FullReco_JetAK4_BTagPhys",
    "FullReco_JetAK4_Charge",
    "FullReco_JetAK4_Constituents",

    # Gen AK4 jets
    "FullReco_GenJetAK4_PT",
    "FullReco_GenJetAK4_Eta",
    "FullReco_GenJetAK4_Phi",
    "FullReco_GenJetAK4_Mass",

    # Gen particles for H->bb truth matching
    "FullReco_GenPart_PT",
    "FullReco_GenPart_Eta",
    "FullReco_GenPart_Phi",
    "FullReco_GenPart_PID",
    "FullReco_GenPart_Status",

    # PF candidates
    "FullReco_PFCand_PT",
    "FullReco_PFCand_Eta",
    "FullReco_PFCand_Phi",
    "FullReco_PFCand_PID",
    "FullReco_PFCand_Charge",
    "FullReco_PFCand_Mass",
    "FullReco_PFCand_D0",
    "FullReco_PFCand_DZ",
    "FullReco_PFCand_ErrorD0",
    "FullReco_PFCand_ErrorDZ",
    "FullReco_PFCand_fUniqueID",
    "FullReco_PFCand_PuppiW",

    # Tight muons and electrons
    "FullReco_MuonTight_PT",
    "FullReco_MuonTight_Eta",
    "FullReco_MuonTight_Phi",
    "FullReco_MuonTight_IsolationVarRhoCorr",

    "FullReco_Electron_PT",
    "FullReco_Electron_Eta",
    "FullReco_Electron_Phi",
    "FullReco_Electron_EhadOverEem",
    "FullReco_Electron_IsolationVarRhoCorr",

    # Event-level information
    "FullReco_MET_MET",
    "FullReco_MET_Phi",
    "FullReco_PUPPIMET_MET",
    "FullReco_PUPPIMET_Phi",
    "FullReco_PrimaryVertex_Z",
    "FullReco_PrimaryVertex_SumPT2",
]


FEATURE_NAMES = [
    # Basic jet features
    "log_reco_pt",
    "reco_eta",
    "abs_reco_eta",
    "sin_reco_phi",
    "cos_reco_phi",
    "log_reco_mass",
    "reco_btag",
    "reco_btagPhys",
    "reco_charge",
    "jet_rank_by_pt",
    "jet_rank_by_btag",
    "n_reco_jets_retained",
    "ht_retained",
    "log_ht_retained",
    "met",
    "log_met",
    "puppimet",
    "log_puppimet",
    "pv_z",
    "log_pv_sumpt2",

    # PF constituent features
    "n_constituents",
    "log_n_constituents",
    "sum_constituent_pt",
    "constituent_pt_sum_over_reco_pt",
    "leading_constituent_pt_frac",
    "second_constituent_pt_frac",
    "third_constituent_pt_frac",
    "ptD",
    "jet_width",
    "mean_constituent_deltaR",
    "max_constituent_deltaR",
    "charged_pt_frac",
    "neutral_pt_frac",
    "charged_multiplicity",
    "neutral_multiplicity",
    "charged_multiplicity_frac",
    "photon_pt_frac",
    "electron_pt_frac",
    "muon_pt_frac",
    "charged_hadron_pt_frac",
    "neutral_hadron_pt_frac",
    "mean_puppi_weight",
    "min_puppi_weight",
    "max_puppi_weight",
    "mean_abs_d0_charged",
    "max_abs_d0_charged",
    "mean_abs_dz_charged",
    "max_abs_dz_charged",
    "mean_abs_d0sig_charged",
    "mean_abs_dzsig_charged",

    # Lepton-in-jet features
    "has_pf_muon",
    "n_pf_muons",
    "pf_muon_pt_frac",
    "has_pf_electron",
    "n_pf_electrons",
    "pf_electron_pt_frac",
    "has_tight_muon_dr04",
    "tight_muon_pt_frac",
    "min_tight_muon_deltaR",
    "has_reco_electron_dr04",
    "reco_electron_pt_frac",
    "min_reco_electron_deltaR",
]


def ensure_dirs():
    OUTDIR.mkdir(parents=True, exist_ok=True)
    PLOT_DIR.mkdir(parents=True, exist_ok=True)


def find_hh_files(n_files=None):
    files = sorted(SNAPSHOT_DIR.glob("*/HH_bbWW/*.parquet"))
    if not files:
        raise FileNotFoundError(
            f"Could not find HH_bbWW parquet files under {SNAPSHOT_DIR}. "
            "Run the dataset download/cache step first."
        )
    if n_files is not None:
        files = files[:n_files]
    return files


def delta_phi(phi1, phi2):
    dphi = float(phi1) - float(phi2)
    while dphi > math.pi:
        dphi -= 2.0 * math.pi
    while dphi <= -math.pi:
        dphi += 2.0 * math.pi
    return dphi


def delta_r(eta1, phi1, eta2, phi2):
    return math.sqrt((float(eta1) - float(eta2)) ** 2 + delta_phi(phi1, phi2) ** 2)


def invariant_mass_from_jet_rows(j0, j1):
    pt1, eta1, phi1, m1 = j0
    pt2, eta2, phi2, m2 = j1

    px1 = pt1 * np.cos(phi1)
    py1 = pt1 * np.sin(phi1)
    pz1 = pt1 * np.sinh(eta1)
    e1 = np.sqrt(px1**2 + py1**2 + pz1**2 + m1**2)

    px2 = pt2 * np.cos(phi2)
    py2 = pt2 * np.sin(phi2)
    pz2 = pt2 * np.sinh(eta2)
    e2 = np.sqrt(px2**2 + py2**2 + pz2**2 + m2**2)

    e = e1 + e2
    px = px1 + px2
    py = py1 + py2
    pz = pz1 + pz2

    m2_tot = e**2 - px**2 - py**2 - pz**2
    return float(np.sqrt(max(m2_tot, 0.0)))


def event_from_batch(columns, row):
    return {name: columns[name][row] for name in COLUMNS}


def get_scalar(event, key, default=0.0):
    value = event.get(key, default)
    if isinstance(value, list):
        if len(value) == 0:
            return default
        return float(value[0])
    try:
        return float(value)
    except Exception:
        return default


def build_reco_jets(
    event,
    max_reco_jets=DEFAULT_MAX_RECO_JETS,
    min_reco_pt=0.0,
    max_abs_eta=-1.0):

    n_raw = len(event["FullReco_JetAK4_PT"])
    jets = []

    for j in range(n_raw):
        pt = float(event["FullReco_JetAK4_PT"][j])
        eta = float(event["FullReco_JetAK4_Eta"][j])

        if pt < min_reco_pt:
            continue

        if max_abs_eta is not None and max_abs_eta >= 0.0 and abs(eta) > max_abs_eta:
            continue

        jets.append(
            {
                "idx": j,
                "pt": pt,
                "eta": eta,
                "phi": float(event["FullReco_JetAK4_Phi"][j]),
                "mass": float(event["FullReco_JetAK4_Mass"][j]),
                "btag": float(event["FullReco_JetAK4_BTag"][j]),
                "btagPhys": float(event["FullReco_JetAK4_BTagPhys"][j]),
                "charge": float(event["FullReco_JetAK4_Charge"][j]),
                "constituents": event["FullReco_JetAK4_Constituents"][j],
            }
        )

        if max_reco_jets is not None and max_reco_jets > 0 and len(jets) >= max_reco_jets:
            break

    return jets, n_raw


def build_genjets(event):
    jets = []
    n = len(event["FullReco_GenJetAK4_PT"])
    for j in range(n):
        jets.append(
            {
                "idx": j,
                "pt": float(event["FullReco_GenJetAK4_PT"][j]),
                "eta": float(event["FullReco_GenJetAK4_Eta"][j]),
                "phi": float(event["FullReco_GenJetAK4_Phi"][j]),
                "mass": float(event["FullReco_GenJetAK4_Mass"][j]),
            }
        )
    return jets


def find_hbb_bquarks(event):
    pids = event["FullReco_GenPart_PID"]
    statuses = event["FullReco_GenPart_Status"]
    pts = event["FullReco_GenPart_PT"]

    candidates = []
    for i, pid in enumerate(pids):
        if abs(int(pid)) == 5 and int(statuses[i]) == 23:
            candidates.append(i)

    candidates = sorted(candidates, key=lambda i: float(pts[i]), reverse=True)
    return candidates[:2]


def match_bquarks_to_reco_jets(event, b_indices, reco_jets):
    matches = []
    for bidx in b_indices:
        b_eta = float(event["FullReco_GenPart_Eta"][bidx])
        b_phi = float(event["FullReco_GenPart_Phi"][bidx])

        best = None
        best_dr = float("inf")
        for jet in reco_jets:
            dr = delta_r(b_eta, b_phi, jet["eta"], jet["phi"])
            if dr < best_dr:
                best = jet
                best_dr = dr

        matches.append(
            {
                "bidx": int(bidx),
                "jet": best,
                "dr": float(best_dr),
                "matched": bool(best is not None and best_dr < DR_GENB_RECO),
            }
        )
    return matches


def match_reco_jet_to_genjet(reco_jet, genjets):
    best = None
    best_dr = float("inf")
    for gj in genjets:
        dr = delta_r(reco_jet["eta"], reco_jet["phi"], gj["eta"], gj["phi"])
        if dr < best_dr:
            best = gj
            best_dr = dr

    if best is None or best_dr >= DR_RECO_GENJET:
        return None, float(best_dr)
    return best, float(best_dr)


def pf_id_to_row_mapping(event):
    ids = event["FullReco_PFCand_fUniqueID"]
    return {int(uid): i for i, uid in enumerate(ids)}


def safe_fraction(num, denom):
    if denom <= 0:
        return 0.0
    return float(num / denom)


def constituent_features(event, reco_jet, id_to_row):
    constituent_ids = reco_jet["constituents"]
    rows = []
    for cid in constituent_ids:
        idx = id_to_row.get(int(cid))
        if idx is not None:
            rows.append(idx)

    reco_pt = max(float(reco_jet["pt"]), EPS)

    if len(rows) == 0:
        return {
            "n_constituents": 0.0,
            "log_n_constituents": 0.0,
            "sum_constituent_pt": 0.0,
            "constituent_pt_sum_over_reco_pt": 0.0,
            "leading_constituent_pt_frac": 0.0,
            "second_constituent_pt_frac": 0.0,
            "third_constituent_pt_frac": 0.0,
            "ptD": 0.0,
            "jet_width": 0.0,
            "mean_constituent_deltaR": 0.0,
            "max_constituent_deltaR": 0.0,
            "charged_pt_frac": 0.0,
            "neutral_pt_frac": 0.0,
            "charged_multiplicity": 0.0,
            "neutral_multiplicity": 0.0,
            "charged_multiplicity_frac": 0.0,
            "photon_pt_frac": 0.0,
            "electron_pt_frac": 0.0,
            "muon_pt_frac": 0.0,
            "charged_hadron_pt_frac": 0.0,
            "neutral_hadron_pt_frac": 0.0,
            "mean_puppi_weight": 0.0,
            "min_puppi_weight": 0.0,
            "max_puppi_weight": 0.0,
            "mean_abs_d0_charged": 0.0,
            "max_abs_d0_charged": 0.0,
            "mean_abs_dz_charged": 0.0,
            "max_abs_dz_charged": 0.0,
            "mean_abs_d0sig_charged": 0.0,
            "mean_abs_dzsig_charged": 0.0,
            "has_pf_muon": 0.0,
            "n_pf_muons": 0.0,
            "pf_muon_pt_frac": 0.0,
            "has_pf_electron": 0.0,
            "n_pf_electrons": 0.0,
            "pf_electron_pt_frac": 0.0,
        }

    pt = np.array([float(event["FullReco_PFCand_PT"][i]) for i in rows], dtype=float)
    eta = np.array([float(event["FullReco_PFCand_Eta"][i]) for i in rows], dtype=float)
    phi = np.array([float(event["FullReco_PFCand_Phi"][i]) for i in rows], dtype=float)
    pid = np.array([int(event["FullReco_PFCand_PID"][i]) for i in rows], dtype=int)
    charge = np.array([float(event["FullReco_PFCand_Charge"][i]) for i in rows], dtype=float)
    d0 = np.array([float(event["FullReco_PFCand_D0"][i]) for i in rows], dtype=float)
    dz = np.array([float(event["FullReco_PFCand_DZ"][i]) for i in rows], dtype=float)
    err_d0 = np.array([float(event["FullReco_PFCand_ErrorD0"][i]) for i in rows], dtype=float)
    err_dz = np.array([float(event["FullReco_PFCand_ErrorDZ"][i]) for i in rows], dtype=float)
    puppi = np.array([float(event["FullReco_PFCand_PuppiW"][i]) for i in rows], dtype=float)

    sum_pt = float(np.sum(pt))
    denom = max(sum_pt, EPS)

    drs = np.array(
        [delta_r(eta[i], phi[i], reco_jet["eta"], reco_jet["phi"]) for i in range(len(rows))],
        dtype=float,
    )

    sorted_pt = np.sort(pt)[::-1]
    leading_frac = safe_fraction(sorted_pt[0], denom) if len(sorted_pt) > 0 else 0.0
    second_frac = safe_fraction(sorted_pt[1], denom) if len(sorted_pt) > 1 else 0.0
    third_frac = safe_fraction(sorted_pt[2], denom) if len(sorted_pt) > 2 else 0.0

    charged = np.abs(charge) > 0
    neutral = ~charged
    abs_pid = np.abs(pid)

    is_photon = abs_pid == 22
    is_electron = abs_pid == 11
    is_muon = abs_pid == 13
    is_lepton = is_electron | is_muon
    is_charged_hadron = charged & (~is_lepton)
    is_neutral_hadron = neutral & (~is_photon)

    charged_d0 = np.abs(d0[charged])
    charged_dz = np.abs(dz[charged])

    d0sig = np.abs(d0[charged]) / np.maximum(np.abs(err_d0[charged]), EPS)
    dzsig = np.abs(dz[charged]) / np.maximum(np.abs(err_dz[charged]), EPS)

    return {
        "n_constituents": float(len(rows)),
        "log_n_constituents": float(np.log1p(len(rows))),
        "sum_constituent_pt": sum_pt,
        "constituent_pt_sum_over_reco_pt": safe_fraction(sum_pt, reco_pt),
        "leading_constituent_pt_frac": leading_frac,
        "second_constituent_pt_frac": second_frac,
        "third_constituent_pt_frac": third_frac,
        "ptD": safe_fraction(np.sqrt(np.sum(pt**2)), denom),
        "jet_width": safe_fraction(np.sum(pt * drs), denom),
        "mean_constituent_deltaR": float(np.mean(drs)),
        "max_constituent_deltaR": float(np.max(drs)),
        "charged_pt_frac": safe_fraction(np.sum(pt[charged]), denom),
        "neutral_pt_frac": safe_fraction(np.sum(pt[neutral]), denom),
        "charged_multiplicity": float(np.sum(charged)),
        "neutral_multiplicity": float(np.sum(neutral)),
        "charged_multiplicity_frac": safe_fraction(float(np.sum(charged)), float(len(rows))),
        "photon_pt_frac": safe_fraction(np.sum(pt[is_photon]), denom),
        "electron_pt_frac": safe_fraction(np.sum(pt[is_electron]), denom),
        "muon_pt_frac": safe_fraction(np.sum(pt[is_muon]), denom),
        "charged_hadron_pt_frac": safe_fraction(np.sum(pt[is_charged_hadron]), denom),
        "neutral_hadron_pt_frac": safe_fraction(np.sum(pt[is_neutral_hadron]), denom),
        "mean_puppi_weight": float(np.mean(puppi)),
        "min_puppi_weight": float(np.min(puppi)),
        "max_puppi_weight": float(np.max(puppi)),
        "mean_abs_d0_charged": float(np.mean(charged_d0)) if len(charged_d0) else 0.0,
        "max_abs_d0_charged": float(np.max(charged_d0)) if len(charged_d0) else 0.0,
        "mean_abs_dz_charged": float(np.mean(charged_dz)) if len(charged_dz) else 0.0,
        "max_abs_dz_charged": float(np.max(charged_dz)) if len(charged_dz) else 0.0,
        "mean_abs_d0sig_charged": float(np.mean(d0sig)) if len(d0sig) else 0.0,
        "mean_abs_dzsig_charged": float(np.mean(dzsig)) if len(dzsig) else 0.0,
        "has_pf_muon": float(np.any(is_muon)),
        "n_pf_muons": float(np.sum(is_muon)),
        "pf_muon_pt_frac": safe_fraction(np.sum(pt[is_muon]), denom),
        "has_pf_electron": float(np.any(is_electron)),
        "n_pf_electrons": float(np.sum(is_electron)),
        "pf_electron_pt_frac": safe_fraction(np.sum(pt[is_electron]), denom),
    }


def lepton_dr_features(event, reco_jet):
    mu_pts = event["FullReco_MuonTight_PT"]
    mu_etas = event["FullReco_MuonTight_Eta"]
    mu_phis = event["FullReco_MuonTight_Phi"]

    el_pts = event["FullReco_Electron_PT"]
    el_etas = event["FullReco_Electron_Eta"]
    el_phis = event["FullReco_Electron_Phi"]

    reco_pt = max(float(reco_jet["pt"]), EPS)

    mu_dr = []
    mu_pt_in = []
    for pt, eta, phi in zip(mu_pts, mu_etas, mu_phis):
        dr = delta_r(float(eta), float(phi), reco_jet["eta"], reco_jet["phi"])
        if dr < 0.4:
            mu_dr.append(dr)
            mu_pt_in.append(float(pt))

    el_dr = []
    el_pt_in = []
    for pt, eta, phi in zip(el_pts, el_etas, el_phis):
        dr = delta_r(float(eta), float(phi), reco_jet["eta"], reco_jet["phi"])
        if dr < 0.4:
            el_dr.append(dr)
            el_pt_in.append(float(pt))

    return {
        "has_tight_muon_dr04": float(len(mu_dr) > 0),
        "tight_muon_pt_frac": safe_fraction(np.sum(mu_pt_in), reco_pt),
        "min_tight_muon_deltaR": float(np.min(mu_dr)) if len(mu_dr) else 9.0,
        "has_reco_electron_dr04": float(len(el_dr) > 0),
        "reco_electron_pt_frac": safe_fraction(np.sum(el_pt_in), reco_pt),
        "min_reco_electron_deltaR": float(np.min(el_dr)) if len(el_dr) else 9.0,
    }


def jet_ranks(reco_jets):
    pt_order = sorted(reco_jets, key=lambda j: j["pt"], reverse=True)
    btag_order = sorted(reco_jets, key=lambda j: j["btag"], reverse=True)

    pt_rank = {j["idx"]: rank for rank, j in enumerate(pt_order)}
    btag_rank = {j["idx"]: rank for rank, j in enumerate(btag_order)}
    return pt_rank, btag_rank


def build_row(event, global_event, file_label, pair_slot, bidx, reco_jet, genjet, dr_genb_reco, dr_reco_genjet, matched_mbb, pt_rank, btag_rank, id_to_row, max_reco_jets, min_reco_pt, max_abs_eta):
    reco_pt = max(float(reco_jet["pt"]), EPS)
    genjet_pt = max(float(genjet["pt"]), EPS)

    reco_jets, n_raw = build_reco_jets(
    event,
    max_reco_jets=max_reco_jets,
    min_reco_pt=min_reco_pt,
    max_abs_eta=max_abs_eta,
    )
    ht = float(np.sum([j["pt"] for j in reco_jets]))

    features = {
        "log_reco_pt": float(np.log(reco_pt)),
        "reco_eta": float(reco_jet["eta"]),
        "abs_reco_eta": float(abs(reco_jet["eta"])),
        "sin_reco_phi": float(np.sin(reco_jet["phi"])),
        "cos_reco_phi": float(np.cos(reco_jet["phi"])),
        "log_reco_mass": float(np.log(max(reco_jet["mass"], 0.0) + 1.0)),
        "reco_btag": float(reco_jet["btag"]),
        "reco_btagPhys": float(reco_jet["btagPhys"]),
        "reco_charge": float(reco_jet["charge"]),
        "jet_rank_by_pt": float(pt_rank[reco_jet["idx"]]),
        "jet_rank_by_btag": float(btag_rank[reco_jet["idx"]]),
        "n_reco_jets_retained": float(len(reco_jets)),
        "ht_retained": ht,
        "log_ht_retained": float(np.log(max(ht, 0.0) + 1.0)),
        "met": get_scalar(event, "FullReco_MET_MET", 0.0),
        "log_met": float(np.log(get_scalar(event, "FullReco_MET_MET", 0.0) + 1.0)),
        "puppimet": get_scalar(event, "FullReco_PUPPIMET_MET", 0.0),
        "log_puppimet": float(np.log(get_scalar(event, "FullReco_PUPPIMET_MET", 0.0) + 1.0)),
        "pv_z": get_scalar(event, "FullReco_PrimaryVertex_Z", 0.0),
        "log_pv_sumpt2": float(np.log(get_scalar(event, "FullReco_PrimaryVertex_SumPT2", 0.0) + 1.0)),
    }

    features.update(constituent_features(event, reco_jet, id_to_row))
    features.update(lepton_dr_features(event, reco_jet))

    feature_values = [float(features[name]) for name in FEATURE_NAMES]

    target = float(np.log(genjet_pt / reco_pt))
    response = float(genjet_pt / reco_pt)

    return {
        "file": file_label,
        "event_id": int(global_event),
        "pair_slot": int(pair_slot),
        "gen_b_idx": int(bidx),
        "reco_jet_idx": int(reco_jet["idx"]),
        "genjet_idx": int(genjet["idx"]),
        "dr_genb_reco": float(dr_genb_reco),
        "dr_reco_genjet": float(dr_reco_genjet),

        "reco_pt": float(reco_jet["pt"]),
        "reco_eta": float(reco_jet["eta"]),
        "reco_phi": float(reco_jet["phi"]),
        "reco_mass": float(reco_jet["mass"]),

        "genjet_pt": float(genjet["pt"]),
        "genjet_eta": float(genjet["eta"]),
        "genjet_phi": float(genjet["phi"]),
        "genjet_mass": float(genjet["mass"]),

        "genb_pt": float(event["FullReco_GenPart_PT"][bidx]),
        "genb_eta": float(event["FullReco_GenPart_Eta"][bidx]),
        "genb_phi": float(event["FullReco_GenPart_Phi"][bidx]),

        "target_log_genjet_pt_over_reco_pt": target,
        "response_genjet_pt_over_reco_pt": response,
        "matched_mbb_uncorrected": float(matched_mbb),

        **{name: float(value) for name, value in zip(FEATURE_NAMES, feature_values)},
    }


def process_event(event, global_event, file_label, max_reco_jets, min_reco_pt, max_abs_eta):
    '''
    Process one event and return a list of rows (one per matched b jet) and a status string.
    '''
    reco_jets, n_raw = build_reco_jets(
    event,
    max_reco_jets=max_reco_jets,
    min_reco_pt=min_reco_pt,
    max_abs_eta=max_abs_eta,
    )
    genjets = build_genjets(event)

    if len(reco_jets) < 2 or len(genjets) == 0:
        return [], "too_few_jets"

    b_indices = find_hbb_bquarks(event)
    if len(b_indices) != 2:
        return [], "no_two_status23_bquarks"

    b_matches = match_bquarks_to_reco_jets(event, b_indices, reco_jets)
    if not all(m["matched"] for m in b_matches):
        return [], "unmatched_genb_to_reco"

    reco_indices = [m["jet"]["idx"] for m in b_matches]
    if reco_indices[0] == reco_indices[1]:
        return [], "same_reco_jet"

    genjet_matches = []
    for m in b_matches:
        genjet, dr = match_reco_jet_to_genjet(m["jet"], genjets)
        if genjet is None:
            return [], "unmatched_reco_to_genjet"
        genjet_matches.append((genjet, dr))

    genjet_indices = [gm[0]["idx"] for gm in genjet_matches]
    if genjet_indices[0] == genjet_indices[1]:
        return [], "same_genjet"

    j0 = b_matches[0]["jet"]
    j1 = b_matches[1]["jet"]
    matched_mbb = invariant_mass_from_jet_rows(
        [j0["pt"], j0["eta"], j0["phi"], j0["mass"]],
        [j1["pt"], j1["eta"], j1["phi"], j1["mass"]],
    )

    pt_rank, btag_rank = jet_ranks(reco_jets)
    id_to_row = pf_id_to_row_mapping(event)

    rows = []
    for slot, m in enumerate(b_matches):
        genjet, dr_reco_genjet = genjet_matches[slot]
        rows.append(
            build_row(
                event=event,
                global_event=global_event,
                file_label=file_label,
                pair_slot=slot,
                bidx=m["bidx"],
                reco_jet=m["jet"],
                genjet=genjet,
                dr_genb_reco=m["dr"],
                dr_reco_genjet=dr_reco_genjet,
                matched_mbb=matched_mbb,
                pt_rank=pt_rank,
                btag_rank=btag_rank,
                id_to_row=id_to_row,
                max_reco_jets=max_reco_jets,
                min_reco_pt=min_reco_pt,
                max_abs_eta=max_abs_eta,
            )
        )

    return rows, "usable"


def scan_events(args):
    files = find_hh_files(args.n_files)
    rows = []

    counts = {
        "scanned": 0,
        "usable_events": 0,
        "usable_jet_rows": 0,
        "too_few_jets": 0,
        "no_two_status23_bquarks": 0,
        "unmatched_genb_to_reco": 0,
        "same_reco_jet": 0,
        "unmatched_reco_to_genjet": 0,
        "same_genjet": 0,
    }

    global_event = 0

    for file_path in files:
        file_label = str(file_path.relative_to(SNAPSHOT_DIR))
        print(f"Scanning {file_label}", flush=True)

        parquet_file = pq.ParquetFile(file_path)
        for batch in parquet_file.iter_batches(batch_size=args.batch_size, columns=COLUMNS):
            columns = batch.to_pydict()

            for i in range(batch.num_rows):
                if args.max_events is not None and counts["scanned"] >= args.max_events:
                    return pd.DataFrame(rows), counts

                event = event_from_batch(columns, i)
                event_rows, status = process_event(
                    event,
                    global_event,
                    file_label,
                    max_reco_jets=args.max_reco_jets,
                    min_reco_pt=args.min_reco_pt,
                    max_abs_eta=args.max_abs_eta,
                )

                counts["scanned"] += 1
                if status == "usable":
                    rows.extend(event_rows)
                    counts["usable_events"] += 1
                    counts["usable_jet_rows"] += len(event_rows)
                else:
                    counts[status] = counts.get(status, 0) + 1

                if counts["scanned"] % args.progress_every == 0:
                    print(
                        f"scanned={counts['scanned']} "
                        f"usable_events={counts['usable_events']} "
                        f"jet_rows={counts['usable_jet_rows']}",
                        flush=True,
                    )

                global_event += 1

    return pd.DataFrame(rows), counts


def split_by_event(df, seed, train_frac=0.80, val_frac=0.10):
    '''
    Split the dataset by event ID into train, validation, and test sets.
    '''
    rng = np.random.default_rng(seed)
    event_ids = np.array(sorted(df["event_id"].unique()))
    rng.shuffle(event_ids)

    n = len(event_ids)
    n_train = int(train_frac * n)
    n_val = int(val_frac * n)

    train_events = set(event_ids[:n_train])
    val_events = set(event_ids[n_train:n_train + n_val])
    test_events = set(event_ids[n_train + n_val:])

    split = np.empty(len(df), dtype=object)
    for i, eid in enumerate(df["event_id"].to_numpy()):
        if eid in train_events:
            split[i] = "train"
        elif eid in val_events:
            split[i] = "val"
        else:
            split[i] = "test"

    df = df.copy()
    df["split"] = split
    return df


def save_npz_for_split(df, split_name):
    part = df[df["split"] == split_name].copy()

    X = part[FEATURE_NAMES].to_numpy(dtype=np.float32)
    y = part["target_log_genjet_pt_over_reco_pt"].to_numpy(dtype=np.float32)

    out = OUTDIR / f"{split_name}.npz"
    np.savez_compressed(
        out,
        X=X,
        y=y,
        event_id=part["event_id"].to_numpy(dtype=np.int64),
        pair_slot=part["pair_slot"].to_numpy(dtype=np.int64),
        reco_jet_idx=part["reco_jet_idx"].to_numpy(dtype=np.int64),
        genjet_idx=part["genjet_idx"].to_numpy(dtype=np.int64),
        reco_pt=part["reco_pt"].to_numpy(dtype=np.float32),
        reco_eta=part["reco_eta"].to_numpy(dtype=np.float32),
        reco_phi=part["reco_phi"].to_numpy(dtype=np.float32),
        reco_mass=part["reco_mass"].to_numpy(dtype=np.float32),
        genjet_pt=part["genjet_pt"].to_numpy(dtype=np.float32),
        genjet_eta=part["genjet_eta"].to_numpy(dtype=np.float32),
        genjet_phi=part["genjet_phi"].to_numpy(dtype=np.float32),
        genjet_mass=part["genjet_mass"].to_numpy(dtype=np.float32),
        genb_pt=part["genb_pt"].to_numpy(dtype=np.float32),
        genb_eta=part["genb_eta"].to_numpy(dtype=np.float32),
        genb_phi=part["genb_phi"].to_numpy(dtype=np.float32),
        response_genjet_pt_over_reco_pt=part["response_genjet_pt_over_reco_pt"].to_numpy(dtype=np.float32),
        matched_mbb_uncorrected=part["matched_mbb_uncorrected"].to_numpy(dtype=np.float32),
    )
    return out, len(part), part["event_id"].nunique()


def summarize_series(values):
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    if len(arr) == 0:
        return {"n": 0}
    return {
        "n": int(len(arr)),
        "mean": float(np.mean(arr)),
        "median": float(np.median(arr)),
        "std": float(np.std(arr)),
        "p16": float(np.percentile(arr, 16)),
        "p84": float(np.percentile(arr, 84)),
    }


def make_plots(df):
    plt.figure(figsize=(7.5, 5.0))
    plt.hist(df["target_log_genjet_pt_over_reco_pt"], bins=80, histtype="step", linewidth=1.5)
    plt.axvline(0.0, linestyle="--", linewidth=1.0)
    plt.xlabel("target = log(GenJetAK4 pT / Reco AK4 pT)")
    plt.ylabel("Jet rows")
    plt.title("B-jet regression target distribution")
    plt.tight_layout()
    plt.savefig(PLOT_DIR / "target_log_response_distribution.png", dpi=200)
    plt.close()

    plt.figure(figsize=(7.5, 5.0))
    plt.scatter(df["reco_pt"], df["response_genjet_pt_over_reco_pt"], s=4, alpha=0.25)
    plt.axhline(1.0, linestyle="--", linewidth=1.0)
    plt.xlabel("Reco AK4 jet pT [GeV]")
    plt.ylabel("GenJetAK4 pT / Reco AK4 pT")
    plt.title("Response target vs reco pT")
    plt.tight_layout()
    plt.savefig(PLOT_DIR / "response_vs_reco_pt_scatter.png", dpi=200)
    plt.close()

    bins = [20, 30, 40, 50, 65, 80, 100, 140, 200, 400, 1000]
    rows = []
    for lo, hi in zip(bins[:-1], bins[1:]):
        part = df[(df["reco_pt"] >= lo) & (df["reco_pt"] < hi)]
        if len(part) == 0:
            continue
        rows.append(
            {
                "center": 0.5 * (lo + hi),
                "n": len(part),
                "median": np.median(part["response_genjet_pt_over_reco_pt"]),
                "p16": np.percentile(part["response_genjet_pt_over_reco_pt"], 16),
                "p84": np.percentile(part["response_genjet_pt_over_reco_pt"], 84),
            }
        )
    prof = pd.DataFrame(rows)
    prof.to_csv(OUTDIR / "response_profile_vs_reco_pt.csv", index=False)

    plt.figure(figsize=(7.5, 5.0))
    plt.plot(prof["center"], prof["median"], marker="o", label="median")
    plt.fill_between(prof["center"], prof["p16"], prof["p84"], alpha=0.2, label="16-84%")
    plt.axhline(1.0, linestyle="--", linewidth=1.0)
    plt.xlabel("Reco AK4 jet pT [GeV]")
    plt.ylabel("GenJetAK4 pT / Reco AK4 pT")
    plt.title("Response profile vs reco pT")
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(PLOT_DIR / "response_profile_vs_reco_pt.png", dpi=200)
    plt.close()

    plt.figure(figsize=(7.5, 5.0))
    plt.hist(df["n_constituents"], bins=60, histtype="step", linewidth=1.5)
    plt.xlabel("Number of PF constituents in matched b jet")
    plt.ylabel("Jet rows")
    plt.title("PF constituent multiplicity")
    plt.tight_layout()
    plt.savefig(PLOT_DIR / "n_constituents_distribution.png", dpi=200)
    plt.close()


def write_summary(df, counts, split_info, args):
    target_summary = summarize_series(df["target_log_genjet_pt_over_reco_pt"])
    response_summary = summarize_series(df["response_genjet_pt_over_reco_pt"])
    mbb_summary = summarize_series(df.drop_duplicates("event_id")["matched_mbb_uncorrected"])

    lines = [
        "# H->bb B-Jet Regression Dataset",
        "",
        "This dataset is built for a CMS-inspired b-jet response regression study on COLLIDE-1M HH->bbWW.",
        "",
        "Each row is one truth-matched H->bb reconstructed AK4 b jet. The correction target is the matched GenJetAK4 response:",
        "",
        "`target = log(GenJetAK4 pT / RecoJetAK4 pT)`",
        "",
        "The corrected pT can later be computed as:",
        "",
        "`corrected pT = reco pT * exp(model prediction)`",
        "",
        "## Important caveat",
        "",
        "This is simulation-only and CMS-inspired, not a full CMS b-jet regression reproduction. The dataset includes PF-candidate and jet-constituent information, but does not include explicit secondary-vertex variables.",
        "",
        "## Reco AK4 jet object selection",
        "",
        f"- max reco jets: {args.max_reco_jets} (`-1` means all selected reco AK4 jets)",
        f"- minimum reco jet pT: {args.min_reco_pt} GeV",
        f"- maximum |eta|: {args.max_abs_eta} (`-1` means no eta cut)",
        "",
        "## Event and row counts",
        "",
        f"- raw HH events scanned: {counts['scanned']}",
        f"- usable events with two matched reco jets and two matched GenJetAK4 targets: {counts['usable_events']}",
        f"- usable b-jet rows: {counts['usable_jet_rows']}",
        f"- unmatched gen b -> reco AK4: {counts['unmatched_genb_to_reco']}",
        f"- unmatched reco AK4 -> GenJetAK4: {counts['unmatched_reco_to_genjet']}",
        f"- same reco jet matched to both gen b quarks: {counts['same_reco_jet']}",
        f"- same GenJetAK4 matched to both reco jets: {counts['same_genjet']}",
        "",
        "## Split counts",
        "",
        "| Split | Jet rows | Unique events |",
        "|---|---:|---:|",
    ]

    for split_name, info in split_info.items():
        lines.append(f"| {split_name} | {info['n_rows']} | {info['n_events']} |")

    lines.extend(
        [
            "",
            "## Target summary",
            "",
            "| Quantity | n | Mean | Median | Std | 16% | 84% |",
            "|---|---:|---:|---:|---:|---:|---:|",
            f"| log response target | {target_summary['n']} | {target_summary['mean']:.4f} | {target_summary['median']:.4f} | {target_summary['std']:.4f} | {target_summary['p16']:.4f} | {target_summary['p84']:.4f} |",
            f"| GenJet/reco response | {response_summary['n']} | {response_summary['mean']:.4f} | {response_summary['median']:.4f} | {response_summary['std']:.4f} | {response_summary['p16']:.4f} | {response_summary['p84']:.4f} |",
            "",
            "## Uncorrected event-level m_bb summary",
            "",
            "| n events | Mean [GeV] | Median [GeV] | Std [GeV] | 16% [GeV] | 84% [GeV] |",
            "|---:|---:|---:|---:|---:|---:|",
            f"| {mbb_summary['n']} | {mbb_summary['mean']:.2f} | {mbb_summary['median']:.2f} | {mbb_summary['std']:.2f} | {mbb_summary['p16']:.2f} | {mbb_summary['p84']:.2f} |",
            "",
            "## Feature groups",
            "",
            "- Reco AK4 jet kinematics and b-tag quantities",
            "- Event-level HT, MET, PUPPIMET, and primary vertex quantities",
            "- PF-candidate constituent composition and shape features",
            "- PF muon/electron-in-jet features",
            "- Tight muon/electron proximity features",
            "",
            "## Outputs",
            "",
            "- `outputs/hbb_bjet_regression_dataset/train.npz`",
            "- `outputs/hbb_bjet_regression_dataset/val.npz`",
            "- `outputs/hbb_bjet_regression_dataset/test.npz`",
            "- `outputs/hbb_bjet_regression_dataset/jets_all.csv`",
            "- `outputs/hbb_bjet_regression_dataset/feature_names.json`",
            "- `outputs/hbb_bjet_regression_dataset/metadata.json`",
            "- `outputs/hbb_bjet_regression_dataset/response_profile_vs_reco_pt.csv`",
            "- `outputs/plots/hbb_bjet_regression_dataset/*.png`",
        ]
    )

    SUMMARY_MD.write_text("\n".join(lines) + "\n")


def parse_args():
    parser = argparse.ArgumentParser(description="Build H->bb b-jet regression dataset.")
    parser.add_argument("--n-files", type=int, default=2)
    parser.add_argument("--max-events", type=int, default=-1)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--progress-every", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=20260618)
    parser.add_argument(
        "--max-reco-jets",
        type=int,
        default=DEFAULT_MAX_RECO_JETS,
        help="Maximum number of reco AK4 jets to consider. Use -1 for all reco AK4 jets.",
    )
    parser.add_argument(
    "--min-reco-pt",
    type=float,
    default=0.0,
    help="Minimum reconstructed AK4 jet pT to consider for matching/features.",
    )
    parser.add_argument(
        "--max-abs-eta",
        type=float,
        default=-1.0,
        help="Maximum |eta| for reconstructed AK4 jets. Use -1 for no eta cut.",
    )
    args = parser.parse_args()
    if args.max_events < 0:
        args.max_events = None
    return args


def main():
    args = parse_args()
    ensure_dirs()

    df, counts = scan_events(args)
    if df.empty:
        raise RuntimeError("No usable b-jet regression rows were produced.")

    df = split_by_event(df, args.seed)

    jets_csv = OUTDIR / "jets_all.csv"
    df.to_csv(jets_csv, index=False)

    with open(OUTDIR / "feature_names.json", "w") as f:
        json.dump(FEATURE_NAMES, f, indent=2)

    split_info = {}
    for split_name in ["train", "val", "test"]:
        out, n_rows, n_events = save_npz_for_split(df, split_name)
        split_info[split_name] = {
            "path": str(out),
            "n_rows": int(n_rows),
            "n_events": int(n_events),
        }

    metadata = {
        "args": vars(args),
        "counts": counts,
        "feature_names": FEATURE_NAMES,
        "n_features": len(FEATURE_NAMES),
        "split_info": split_info,
        "target": "log(FullReco_GenJetAK4_PT / FullReco_JetAK4_PT)",
        "matching": {
            "gen_b_to_reco_ak4_dr": DR_GENB_RECO,
            "reco_ak4_to_genjet_ak4_dr": DR_RECO_GENJET,
            "max_reco_jets": args.max_reco_jets,
            "min_reco_pt": args.min_reco_pt,
            "max_abs_eta": args.max_abs_eta,
        },
        "caveat": "Simulation-only CMS-inspired dataset; no explicit secondary-vertex variables available.",
    }

    with open(OUTDIR / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    make_plots(df)
    write_summary(df, counts, split_info, args)

    print("\nDataset complete.")
    print(f"Rows: {len(df)}")
    print(f"Unique events: {df['event_id'].nunique()}")
    print(f"Features: {len(FEATURE_NAMES)}")
    print("\nCounts:")
    print(json.dumps(counts, indent=2))
    print("\nSplit info:")
    print(json.dumps(split_info, indent=2))
    print(f"\nSaved summary: {SUMMARY_MD}")
    print(f"Saved outputs: {OUTDIR}")
    print(f"Saved plots: {PLOT_DIR}")


if __name__ == "__main__":
    main()