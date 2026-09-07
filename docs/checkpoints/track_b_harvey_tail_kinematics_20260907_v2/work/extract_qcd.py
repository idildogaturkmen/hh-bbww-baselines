#!/usr/bin/env python3
"""QCD tail extraction, purely from the already-frozen, local
FULL_QCD_SURVIVOR_SIDECAR.h5 -- no live ROOT read needed for the core
jet kinematics (already present in the sidecar). Genuine SPA-Net
assignment indices are reused directly (QCD is the one population in
this project with a frozen assignment sidecar)."""
import sys
import numpy as np
import pandas as pd
import h5py

sys.path.insert(0, "/uscms_data/d3/iturkmen/hh4b_delphes/paper_exports/track_b_harvey_tail_kinematics_20260906_v1/work")
from kinematics import leading_four_topology, delta_r, BTAG_LOOSE_PROBB

SIDECAR = "/uscms_data/d3/iturkmen/hh4b_delphes/paper_exports/track_b_harvey_tail_characterization_20260825_v1/work/FULL_QCD_SURVIVOR_SIDECAR.h5"
U_MIN_KEEP = 3.5


def u_of(score):
    score = np.asarray(score, dtype=np.float64)
    out = np.full(score.shape, np.inf)
    m = score < 1.0
    out[m] = -np.log10(1.0 - score[m])
    return out


def main():
    f = h5py.File(SIDECAR, "r")
    score = f["score_spanet_10m"][:].astype(np.float64)
    u = u_of(score)
    keep = u > U_MIN_KEEP
    print(f"QCD sidecar rows total={len(score)}, u>{U_MIN_KEEP}: {keep.sum()}")

    idx = np.where(keep)[0]
    jet_pt = f["jet_pt"][:][idx].astype(np.float64)
    jet_eta = f["jet_eta"][:][idx].astype(np.float64)
    jet_phi = f["jet_phi"][:][idx].astype(np.float64)
    jet_mass = f["jet_mass"][:][idx].astype(np.float64)
    jet_probB = f["jet_probB"][:][idx].astype(np.float64)
    jet_probC = f["jet_probC"][:][idx].astype(np.float64)
    jet_probL = f["jet_probL"][:][idx].astype(np.float64)
    n_sel = f["n_selected_jets"][:][idx].astype(np.int64)
    HT = f["HT"][:][idx].astype(np.float64)
    source_file_index = f["source_file_index"][:][idx]
    raw_entry_index = f["raw_entry_index"][:][idx]
    source_file = f["source_file"][:][idx]
    assign_idx = f["spanet_10m_assignment_indices"][:][idx]

    topo = leading_four_topology(jet_pt, jet_eta, jet_phi, jet_mass)
    n_btag_loose = (jet_probB > BTAG_LOOSE_PROBB).sum(axis=1).astype(np.int64)

    has5 = n_sel >= 5
    pt5, eta5, phi5 = jet_pt[:, 4], jet_eta[:, 4], jet_phi[:, 4]
    probB5, probC5, probL5 = jet_probB[:, 4], jet_probC[:, 4], jet_probL[:, 4]
    dR_j5_to_leading4 = np.stack([delta_r(eta5, phi5, jet_eta[:, k], jet_phi[:, k]) for k in range(4)], axis=1)
    min_dR5 = dR_j5_to_leading4.min(axis=1)
    nearest_of_four = dR_j5_to_leading4.argmin(axis=1)
    pt5_over_pt4 = np.divide(pt5, jet_pt[:, 3], out=np.full_like(pt5, np.nan), where=jet_pt[:, 3] > 0)
    pt5_over_HT = np.divide(pt5, HT, out=np.full_like(pt5, np.nan), where=HT > 0)

    # genuine SPA-Net assignment: does the assignment set equal the leading-four {0,1,2,3}?
    assign_sorted = np.sort(assign_idx, axis=1)
    leading4_ref = np.tile(np.array([0, 1, 2, 3]), (len(idx), 1))
    assignment_eq_leading4 = np.all(assign_sorted == leading4_ref, axis=1)
    j5_in_assignment = np.any(assign_idx == 4, axis=1)
    n_unassigned = np.array([len(set(range(int(n))) - set(assign_idx[i].tolist())) for i, n in enumerate(n_sel)])

    df = pd.DataFrame(dict(
        process="QCD", source_file=[s.decode() if isinstance(s, bytes) else s for s in source_file],
        file_index=source_file_index.astype(np.int64), entry_index=raw_entry_index.astype(np.int64),
        score=score[idx], u=u[idx],
        n_selected_jets=n_sel, HT=HT, met_pt=np.nan, met_phi=np.nan, n_btag_loose=n_btag_loose,
        mHH=topo["mHH_leading_four"], RHH=topo["RHH"], pT_HH=topo["pT_HH"], eta_HH=topo["eta_HH"],
        mH1=topo["mH1"], mH2=topo["mH2"], pT_H1=topo["ptH1"], pT_H2=topo["ptH2"],
        DeltaR_bb_H1=topo["dRbb1"], DeltaR_bb_H2=topo["dRbb2"],
        DeltaR_HH=topo["deltaR_HH"], DeltaEta_HH=topo["deltaEta_HH"], DeltaPhi_HH=topo["deltaPhi_HH"],
        mass_asym=topo["mass_asym"], pt_asym=topo["pt_asym"],
        pt5=pt5, eta5=eta5, probB5=probB5, probC5=probC5, probL5=probL5,
        min_dR_j5_leading4=min_dR5, nearest_of_four_idx=nearest_of_four,
        pt5_over_pt4=pt5_over_pt4, pt5_over_HT=pt5_over_HT, has_5th_jet=has5,
        has_genuine_assignment=True,
        assignment_eq_leading4=assignment_eq_leading4, j5_in_assignment=j5_in_assignment,
        n_unassigned_selected_jets=n_unassigned,
        n_truth_bhadrons_fromhh=np.nan, j5_is_truth_matched_to_HH_b=np.nan,
        n_truth_matched_in_leading4=np.nan, truth_all4_in_leading4=np.nan,
    ))
    for k in range(6):
        df[f"jet{k+1}_pt"] = jet_pt[:, k]
        df[f"jet{k+1}_eta"] = jet_eta[:, k]
        df[f"jet{k+1}_phi"] = jet_phi[:, k]
        df[f"jet{k+1}_mass"] = jet_mass[:, k]
        df[f"jet{k+1}_probB"] = jet_probB[:, k]
        df[f"jet{k+1}_probC"] = jet_probC[:, k]
        df[f"jet{k+1}_probL"] = jet_probL[:, k]

    out = "/uscms_data/d3/iturkmen/hh4b_delphes/paper_exports/track_b_harvey_tail_kinematics_20260906_v1/work/qcd_tail.parquet"
    df.to_parquet(out, index=False)
    print(f"wrote {out}, n={len(df)}, u>4.5: {(df['u']>4.5).sum()}")


if __name__ == "__main__":
    main()
