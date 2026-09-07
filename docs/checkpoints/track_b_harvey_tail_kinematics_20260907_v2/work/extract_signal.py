#!/usr/bin/env python3
"""Read-only extraction of signal (holdout_A) kinematics for the u>3.5
tail. No SPA-Net training, no inference, no holdout_B/Stage-C access,
no MC generation. Reads the same 200 frozen signal ROOT files already
used for the frozen SPA10M score archive (signal_holdout_A.tsv), joins
on (file_index, entry_index) exactly as SPANET_SIGNAL_SCORES_WITH_EVENT_ID.npz
was itself built.

Truth-matching convention (this analysis's own, stated explicitly, not
assumed identical to whatever convention -- if any -- was used at
SPA-Net H5-build time): a selected jet is "truth-matched to an HH b" if
it is the nearest selected jet (min DeltaR) to a gen_bhadron with
gen_bhadron_fromhh==True, and that DeltaR < 0.4 (AK4 cone).
"""
from __future__ import annotations

import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import awkward as ak
import numpy as np
import pandas as pd
import uproot

sys.path.insert(0, "/uscms_data/d3/iturkmen/hh4b_delphes/paper_exports/track_b_harvey_tail_kinematics_20260906_v1/work")
from kinematics import leading_four_topology, native_select_and_order, delta_r, BTAG_LOOSE_PROBB

SIGNAL_LIST = "/uscms_data/d3/iturkmen/hh4b_delphes/track_b_phase4_preflight_20260812/phase4M_holdout_a_adjudication_20260813_v1/work/split_lists/signal_holdout_A.tsv"
SCORE_NPZ = "/uscms_data/d3/iturkmen/hh4b_delphes/paper_exports/track_b_common_actual_sm_efficiency_scan_20260822_v1/work/SPANET_SIGNAL_SCORES_WITH_EVENT_ID.npz"
XROOTD_PREFIX = "root://cceos.ihep.ac.cn:1094/"
U_MIN_KEEP = 3.5  # anything below this is dropped immediately to bound memory; u>4.5 is a strict subset
N_WORKERS = 4

BRANCHES = ["jet_pt", "jet_eta", "jet_phi", "jet_mass", "jet_sophonAK4_probB", "jet_sophonAK4_probC", "jet_sophonAK4_probL",
            "HT", "met_pt", "met_phi",
            "gen_bhadron_eta", "gen_bhadron_phi", "gen_bhadron_pt", "gen_bhadron_fromhh"]


def u_of(score):
    score = np.asarray(score, dtype=np.float64)
    out = np.full(score.shape, np.inf)
    m = score < 1.0
    out[m] = -np.log10(1.0 - score[m])
    return out


def process_file(file_index: int, remote_path: str) -> pd.DataFrame:
    url = XROOTD_PREFIX + remote_path
    with uproot.open(url, timeout=60) as rf:
        tree = rf["tree"]
        available = set(tree.keys())
        present = [b for b in BRANCHES if b in available]
        missing = [b for b in BRANCHES if b not in available]
        arrs = tree.arrays(present, library="ak")
        n_file = len(arrs)
        if missing:
            print(f"[{remote_path}] missing branches, filled NaN/False: {missing}", file=sys.stderr)
            for b in missing:
                if b == "gen_bhadron_fromhh":
                    arrs[b] = ak.Array([[] for _ in range(n_file)])
                elif b in ("gen_bhadron_eta", "gen_bhadron_phi", "gen_bhadron_pt"):
                    arrs[b] = ak.Array([[] for _ in range(n_file)])
                else:
                    arrs[b] = ak.Array(np.full(n_file, np.nan))

    jet_pt, jet_eta, jet_phi, jet_mass, jet_probB, jet_probC, jet_probL, n_sel = native_select_and_order(
        arrs["jet_pt"], arrs["jet_eta"], arrs["jet_phi"], arrs["jet_mass"],
        arrs["jet_sophonAK4_probB"], arrs["jet_sophonAK4_probC"], arrs["jet_sophonAK4_probL"])

    topo = leading_four_topology(jet_pt, jet_eta, jet_phi, jet_mass)

    HT = ak.to_numpy(arrs["HT"]).astype(np.float64)
    met_pt = ak.to_numpy(arrs["met_pt"]).astype(np.float64)
    met_phi = ak.to_numpy(arrs["met_phi"]).astype(np.float64)
    n_btag_loose = (jet_probB[:, :10] > BTAG_LOOSE_PROBB).sum(axis=1).astype(np.int64)
    # only count among the n_sel real slots -- padded slots have probB=0.0, already below threshold, so this is safe

    # ---- 5th-jet fields (rank index 4, 0-indexed) ----
    has5 = n_sel >= 5
    pt5 = jet_pt[:, 4]
    eta5 = jet_eta[:, 4]
    phi5 = jet_phi[:, 4]
    probB5, probC5, probL5 = jet_probB[:, 4], jet_probC[:, 4], jet_probL[:, 4]
    dR_j5_to_leading4 = np.stack([delta_r(eta5, phi5, jet_eta[:, k], jet_phi[:, k]) for k in range(4)], axis=1)
    min_dR5 = dR_j5_to_leading4.min(axis=1)
    nearest_of_four = dR_j5_to_leading4.argmin(axis=1)
    pt5_over_pt4 = np.divide(pt5, jet_pt[:, 3], out=np.full_like(pt5, np.nan), where=jet_pt[:, 3] > 0)
    pt5_over_HT = np.divide(pt5, HT, out=np.full_like(pt5, np.nan), where=HT > 0)

    # ---- truth matching (gen_bhadron_fromhh, this analysis's own DeltaR<0.4 convention) ----
    gbh_eta, gbh_phi, gbh_pt = arrs["gen_bhadron_eta"], arrs["gen_bhadron_phi"], arrs["gen_bhadron_pt"]
    gbh_fromhh = arrs["gen_bhadron_fromhh"]
    n_events = n_file
    truth_matched_slot = np.full((n_events, 10), False)  # which of the 10 selected-jet slots is truth-matched
    n_truth_bhadrons_fromhh = np.zeros(n_events, dtype=np.int64)
    for i in range(n_events):
        mask = np.asarray(gbh_fromhh[i], dtype=bool)
        n_truth_bhadrons_fromhh[i] = int(mask.sum())
        if not mask.any() or n_sel[i] == 0:
            continue
        teta = np.asarray(gbh_eta[i])[mask]
        tphi = np.asarray(gbh_phi[i])[mask]
        nslots = min(int(n_sel[i]), 10)
        for te, tp in zip(teta, tphi):
            drs = delta_r(np.full(nslots, te), np.full(nslots, tp), jet_eta[i, :nslots], jet_phi[i, :nslots])
            j = int(np.argmin(drs))
            if drs[j] < 0.4:
                truth_matched_slot[i, j] = True

    j5_is_truth_matched = truth_matched_slot[:, 4]
    n_truth_matched_in_leading4 = truth_matched_slot[:, :4].sum(axis=1)
    truth_all4_in_leading4 = (n_truth_bhadrons_fromhh == 4) & (n_truth_matched_in_leading4 == 4)

    df = pd.DataFrame(dict(
        process="signal", source_file=remote_path, file_index=file_index,
        entry_index=np.arange(n_events, dtype=np.int64),
        n_selected_jets=n_sel.astype(np.int64), HT=HT, met_pt=met_pt, met_phi=met_phi,
        n_btag_loose=n_btag_loose,
        mHH=topo["mHH_leading_four"], RHH=topo["RHH"], pT_HH=topo["pT_HH"], eta_HH=topo["eta_HH"],
        mH1=topo["mH1"], mH2=topo["mH2"], pT_H1=topo["ptH1"], pT_H2=topo["ptH2"],
        DeltaR_bb_H1=topo["dRbb1"], DeltaR_bb_H2=topo["dRbb2"],
        DeltaR_HH=topo["deltaR_HH"], DeltaEta_HH=topo["deltaEta_HH"], DeltaPhi_HH=topo["deltaPhi_HH"],
        mass_asym=topo["mass_asym"], pt_asym=topo["pt_asym"],
        pt5=pt5, eta5=eta5, probB5=probB5, probC5=probC5, probL5=probL5,
        min_dR_j5_leading4=min_dR5, nearest_of_four_idx=nearest_of_four,
        pt5_over_pt4=pt5_over_pt4, pt5_over_HT=pt5_over_HT, has_5th_jet=has5,
        n_truth_bhadrons_fromhh=n_truth_bhadrons_fromhh,
        j5_is_truth_matched_to_HH_b=j5_is_truth_matched,
        n_truth_matched_in_leading4=n_truth_matched_in_leading4,
        truth_all4_in_leading4=truth_all4_in_leading4,
    ))
    for k in range(6):
        df[f"jet{k+1}_pt"] = jet_pt[:, k]
        df[f"jet{k+1}_eta"] = jet_eta[:, k]
        df[f"jet{k+1}_phi"] = jet_phi[:, k]
        df[f"jet{k+1}_mass"] = jet_mass[:, k]
        df[f"jet{k+1}_probB"] = jet_probB[:, k]
        df[f"jet{k+1}_probC"] = jet_probC[:, k]
        df[f"jet{k+1}_probL"] = jet_probL[:, k]
    return df


def main():
    files = []
    with open(SIGNAL_LIST) as f:
        for line in f:
            sha, path = line.rstrip("\n").split("\t")
            files.append(path)
    assert len(files) == 200

    scores = np.load(SCORE_NPZ)
    score_arr = scores["score_spanet_10M"].astype(np.float64)
    score_file_idx = scores["file_index"]
    score_entry_idx = scores["source_entry_index"]
    u_arr = u_of(score_arr)
    score_df = pd.DataFrame(dict(file_index=score_file_idx.astype(np.int64),
                                  entry_index=score_entry_idx.astype(np.int64),
                                  score=score_arr, u=u_arr))

    t0 = time.time()
    frames = []
    n_done = 0
    with ThreadPoolExecutor(max_workers=N_WORKERS) as ex:
        futs = {ex.submit(process_file, i, p): i for i, p in enumerate(files)}
        for fut in as_completed(futs):
            i = futs[fut]
            try:
                df = fut.result()
            except Exception as e:
                print(f"FILE {i} FAILED: {e!r}", file=sys.stderr)
                raise
            frames.append(df)
            n_done += 1
            if n_done % 20 == 0 or n_done == len(files):
                print(f"{n_done}/{len(files)} files done, elapsed={time.time()-t0:.1f}s", file=sys.stderr)

    full = pd.concat(frames, ignore_index=True)
    full = full.merge(score_df, on=["file_index", "entry_index"], how="left")
    n_before = len(full)
    n_missing_score = full["score"].isna().sum()
    full = full[full["u"] > U_MIN_KEEP].reset_index(drop=True)

    out_path = "/uscms_data/d3/iturkmen/hh4b_delphes/paper_exports/track_b_harvey_tail_kinematics_20260906_v1/work/signal_tail.parquet"
    full.to_parquet(out_path, index=False)
    print(f"n_events_all_files={n_before} n_missing_score_join={n_missing_score} "
          f"n_kept_u_gt_{U_MIN_KEEP}={len(full)} wrote={out_path} total_time={time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
