#!/usr/bin/env python3
"""Targeted read-only re-extraction of the u>3.5 non-QCD (ttbar/ZJetsToQQ/
SingleTop/TTbarW/TW/TTbarZ/ttH) tail: 42 events across 36 distinct source
files, identified from the already-frozen job_summary survivor score
lists joined to nonqcd_tail_topology.json's source_path identities.
Re-extracts the FULL 10-jet-slot kinematics (this project's existing
nonqcd_tail_topology.json only stored pt/eta/probB, not phi/mass/probC/
probL) so the master table has identical per-jet fields for every
process, and adds MET / 5th-jet fields / b-tag multiplicity, none of
which existed in the prior extraction. No genuine SPA-Net assignment
exists for these processes (unchanged finding from every prior package)
-- assignment-related fields are geometric-leading-4 only, labeled as
such.
"""
from __future__ import annotations

import json
import math
import sys

import numpy as np
import pandas as pd
import uproot

sys.path.insert(0, "/uscms_data/d3/iturkmen/hh4b_delphes/paper_exports/track_b_harvey_tail_kinematics_20260906_v1/work")
from kinematics import leading_four_topology, native_select_and_order, delta_r, BTAG_LOOSE_PROBB

XROOTD_PREFIX = "root://cceos.ihep.ac.cn:1094/"
BRANCHES = ["jet_pt", "jet_eta", "jet_phi", "jet_mass", "jet_sophonAK4_probB", "jet_sophonAK4_probC", "jet_sophonAK4_probL",
            "HT", "met_pt", "met_phi"]


def u_of(score):
    return float("inf") if score >= 1.0 else -math.log10(1.0 - score)


def build_matched_list():
    topo = json.load(open("/uscms_data/d3/iturkmen/hh4b_delphes/paper_exports/track_b_harvey_followup_finescan_likelihood_20260829_v1/work/nonqcd_tail_topology.json"))
    nonqcd_files = ["job_summary_small.json", "job_summary_medium.json", "job_summary_zjetstoqq.json", "job_summary_ttbar_PRESERVED.json"]
    base = "/uscms_data/d3/iturkmen/hh4b_delphes/paper_exports/track_b_harvey_complete_3to5pct_study_20260825_v1/work"
    LANE_TO_JOB_KEY = {("TTbar", "inference_ttbar_1"): "inference_ttbar_1", ("TTbar", "inference_ttbar_2"): "inference_ttbar_2"}

    survivors_by_key = {}
    for fn in nonqcd_files:
        d = json.load(open(f"{base}/{fn}"))
        for proc, pdata in d["processes"].items():
            survivors_by_key[proc] = {(s[0], s[1]): s[2] for s in pdata["spanet_10m"]["survivors"]}

    def job_key_for(process, lane):
        return LANE_TO_JOB_KEY.get((process, lane), process)

    matched = []
    for e in topo:
        jk = job_key_for(e["process"], e.get("lane"))
        key = (e["file_index"], e["entry_index"])
        if jk in survivors_by_key and key in survivors_by_key[jk]:
            score = survivors_by_key[jk][key]
            u = u_of(score)
            if u > 3.5:
                matched.append(dict(process=jk, source_path=e["source_path"],
                                     file_index=e["file_index"], entry_index=e["entry_index"],
                                     score=score, u=u))
    return matched


def main():
    matched = build_matched_list()
    print(f"n_events_to_extract={len(matched)}")

    by_file = {}
    for m in matched:
        by_file.setdefault(m["source_path"], []).append(m)
    print(f"n_distinct_files={len(by_file)}")

    rows = []
    for path, events in by_file.items():
        url = XROOTD_PREFIX + path
        entry_indices = sorted(set(e["entry_index"] for e in events))
        with uproot.open(url, timeout=60) as rf:
            tree = rf["tree"]
            arrs = tree.arrays(BRANCHES, entry_start=min(entry_indices), entry_stop=max(entry_indices) + 1, library="ak")
        offset = min(entry_indices)

        for e in events:
            local_i = e["entry_index"] - offset
            row_arrs = {k: arrs[k][local_i:local_i + 1] for k in BRANCHES}
            jet_pt, jet_eta, jet_phi, jet_mass, jet_probB, jet_probC, jet_probL, n_sel = native_select_and_order(
                row_arrs["jet_pt"], row_arrs["jet_eta"], row_arrs["jet_phi"], row_arrs["jet_mass"],
                row_arrs["jet_sophonAK4_probB"], row_arrs["jet_sophonAK4_probC"], row_arrs["jet_sophonAK4_probL"])
            topo = leading_four_topology(jet_pt, jet_eta, jet_phi, jet_mass)
            HT = float(row_arrs["HT"][0])
            met_pt = float(row_arrs["met_pt"][0])
            met_phi = float(row_arrs["met_phi"][0])
            n_btag_loose = int((jet_probB[0] > BTAG_LOOSE_PROBB).sum())

            has5 = n_sel[0] >= 5
            pt5, eta5, phi5 = jet_pt[0, 4], jet_eta[0, 4], jet_phi[0, 4]
            probB5, probC5, probL5 = jet_probB[0, 4], jet_probC[0, 4], jet_probL[0, 4]
            drs = np.array([delta_r(np.array([eta5]), np.array([phi5]), jet_eta[0:1, k], jet_phi[0:1, k])[0] for k in range(4)])
            min_dR5 = drs.min()
            nearest_of_four = int(drs.argmin())
            pt5_over_pt4 = pt5 / jet_pt[0, 3] if jet_pt[0, 3] > 0 else np.nan
            pt5_over_HT = pt5 / HT if HT > 0 else np.nan

            row = dict(
                process=e["process"], source_file=path, file_index=e["file_index"], entry_index=e["entry_index"],
                score=e["score"], u=e["u"],
                n_selected_jets=int(n_sel[0]), HT=HT, met_pt=met_pt, met_phi=met_phi, n_btag_loose=n_btag_loose,
                mHH=topo["mHH_leading_four"][0], RHH=topo["RHH"][0], pT_HH=topo["pT_HH"][0], eta_HH=topo["eta_HH"][0],
                mH1=topo["mH1"][0], mH2=topo["mH2"][0], pT_H1=topo["ptH1"][0], pT_H2=topo["ptH2"][0],
                DeltaR_bb_H1=topo["dRbb1"][0], DeltaR_bb_H2=topo["dRbb2"][0],
                DeltaR_HH=topo["deltaR_HH"][0], DeltaEta_HH=topo["deltaEta_HH"][0], DeltaPhi_HH=topo["deltaPhi_HH"][0],
                mass_asym=topo["mass_asym"][0], pt_asym=topo["pt_asym"][0],
                pt5=pt5, eta5=eta5, probB5=probB5, probC5=probC5, probL5=probL5,
                min_dR_j5_leading4=min_dR5, nearest_of_four_idx=nearest_of_four,
                pt5_over_pt4=pt5_over_pt4, pt5_over_HT=pt5_over_HT, has_5th_jet=bool(has5),
                has_genuine_assignment=False,
                assignment_eq_leading4=np.nan, j5_in_assignment=np.nan, n_unassigned_selected_jets=np.nan,
                n_truth_bhadrons_fromhh=np.nan, j5_is_truth_matched_to_HH_b=np.nan,
                n_truth_matched_in_leading4=np.nan, truth_all4_in_leading4=np.nan,
            )
            for k in range(6):
                row[f"jet{k+1}_pt"] = jet_pt[0, k]
                row[f"jet{k+1}_eta"] = jet_eta[0, k]
                row[f"jet{k+1}_phi"] = jet_phi[0, k]
                row[f"jet{k+1}_mass"] = jet_mass[0, k]
                row[f"jet{k+1}_probB"] = jet_probB[0, k]
                row[f"jet{k+1}_probC"] = jet_probC[0, k]
                row[f"jet{k+1}_probL"] = jet_probL[0, k]
            rows.append(row)

    df = pd.DataFrame(rows)
    out = "/uscms_data/d3/iturkmen/hh4b_delphes/paper_exports/track_b_harvey_tail_kinematics_20260906_v1/work/nonqcd_tail.parquet"
    df.to_parquet(out, index=False)
    print(f"wrote {out}, n={len(df)}, u>4.5: {(df['u']>4.5).sum()}")
    print(df["process"].value_counts())


if __name__ == "__main__":
    main()
