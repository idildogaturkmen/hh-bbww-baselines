#!/usr/bin/env python3
"""Correction 3 (v2): truth-matching audit + bijective rematch.

This performs a READ-ONLY re-extraction of the SAME 200 already-authorized
signal_holdout_A ROOT files used by v1's extract_signal.py -- no new
holdout, no new SPA-Net inference, no training, no new MC. It is required
because the raw truth b-hadron (eta,phi,pt,fromhh) positions were NOT
retained in signal_tail.parquet / the master table -- only v1's derived
boolean flags were kept, which is insufficient to redo the matching.

Only the u>3.5 tail events already identified in the v1/v2 master table
are processed (27,339 signal events); files containing none of them are
skipped entirely.

v1 (CURRENT) algorithm, exact pseudocode (from extract_signal.py):

    for each event i:
        truth_objects = [ t for t in gen_bhadron if gen_bhadron_fromhh[t] == True ]
        n_truth_bhadrons_fromhh[i] = len(truth_objects)
        for t in truth_objects:                      # loop order: per TRUTH object
            dR[j] = deltaR(truth_eta[t], truth_phi[t], jet_eta[i,j], jet_phi[i,j])  for j in selected jet slots [0, n_sel)
            j_star = argmin_j dR[j]
            if dR[j_star] < 0.4:
                truth_matched_slot[i, j_star] = True  # sets a boolean; does NOT remove j_star from the pool

This is NOT one-to-one (not bijective):
  - One truth object -> at most ONE jet (each truth loop iteration picks a single argmin). So no truth object
    claims multiple jets.
  - One jet CAN be the nearest match for MULTIPLE truth objects (nothing removes j_star from later iterations'
    candidate pool). When that happens, truth_matched_slot[i,j_star] is simply set True once -- the collision is
    invisible in the output (silently undercounts distinct matched truth objects in that event) and a jet that
    should optimally have gone to a different, more distant truth object is not reassigned.

CORRECTED (v2) algorithm -- proper one-to-one (bijective) minimum-DeltaR matching, brute-force
(<=8 truth objects, <=10 jet slots in this dataset -- fully tractable without any package install):

    candidates = [ (dR(t,j), t, j) for all truth t, jet j with dR(t,j) < 0.4 ]
    sort candidates ascending by dR
    used_truth, used_jet = {}, {}
    for (dR, t, j) in candidates:
        if t not in used_truth and j not in used_jet:
            assign t -> j
            used_truth.add(t); used_jet.add(j)
    # remaining truth objects / jets are unmatched

This is the standard greedy approximation to the assignment problem (exact for this sparse,
small-cardinality regime); each truth object and each jet slot is used at most once.

Coverage caveat: 1,085 of 27,339 signal events (4.0%) have n_selected_jets > 6; for those events
jets beyond slot 6 are becoming available again here (same native_select_and_order() ordering as
v1, re-run from the raw ROOT branches -- not limited to the 6 leading jets saved in signal_tail.parquet),
so this audit actually has MORE jet-slot coverage than the parquet-only fields would allow, up to n_sel
capped at 10, matching v1's own capping.
"""
from __future__ import annotations

import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import awkward as ak
import numpy as np
import pandas as pd
import uproot

V1_WORK = "/uscms_data/d3/iturkmen/hh4b_delphes/paper_exports/track_b_harvey_tail_kinematics_20260906_v1/work"
sys.path.insert(0, V1_WORK)
from kinematics import native_select_and_order, delta_r  # noqa: E402

XROOTD_PREFIX = "root://cceos.ihep.ac.cn:1094/"
N_WORKERS = 4
BRANCHES = ["jet_pt", "jet_eta", "jet_phi", "jet_mass", "jet_sophonAK4_probB", "jet_sophonAK4_probC", "jet_sophonAK4_probL",
            "gen_bhadron_eta", "gen_bhadron_phi", "gen_bhadron_pt", "gen_bhadron_fromhh"]

MASTER = "../HARVEY_MASTER_EVENT_TABLE.parquet"
SIGNAL_TAIL = "signal_tail.parquet"
OUT_PARQUET = "truth_rematch_audit_correction3.parquet"


def bijective_match(teta, tphi, jeta, jphi, nslots, dr_max=0.4):
    """Global greedy min-DeltaR one-to-one matching. Returns:
    matched_jet_for_truth: array of length len(teta), -1 if unmatched
    matched_slot_bool: bool array of length nslots (True if that jet slot got a truth match)
    """
    n_t = len(teta)
    matched_jet_for_truth = np.full(n_t, -1, dtype=np.int64)
    matched_slot = np.zeros(nslots, dtype=bool)
    if n_t == 0 or nslots == 0:
        return matched_jet_for_truth, matched_slot
    dR = np.stack([delta_r(teta, tphi, np.full(n_t, jeta[j]), np.full(n_t, jphi[j])) for j in range(nslots)], axis=1)
    cands = []
    for t in range(n_t):
        for j in range(nslots):
            if dR[t, j] < dr_max:
                cands.append((dR[t, j], t, j))
    cands.sort(key=lambda x: x[0])
    used_t, used_j = set(), set()
    for d, t, j in cands:
        if t in used_t or j in used_j:
            continue
        used_t.add(t)
        used_j.add(j)
        matched_jet_for_truth[t] = j
        matched_slot[j] = True
    return matched_jet_for_truth, matched_slot


def process_file(file_index, remote_path, needed_entries):
    """needed_entries: set of entry_index values (within this file) to process."""
    url = XROOTD_PREFIX + remote_path
    with uproot.open(url, timeout=60) as rf:
        tree = rf["tree"]
        available = set(tree.keys())
        present = [b for b in BRANCHES if b in available]
        missing = [b for b in BRANCHES if b not in available]
        arrs = tree.arrays(present, library="ak")
        n_file = len(arrs)
        if missing:
            print(f"[{remote_path}] missing branches, filled empty: {missing}", file=sys.stderr)
            for b in missing:
                arrs[b] = ak.Array([[] for _ in range(n_file)])

    jet_pt, jet_eta, jet_phi, jet_mass, jet_probB, jet_probC, jet_probL, n_sel = native_select_and_order(
        arrs["jet_pt"], arrs["jet_eta"], arrs["jet_phi"], arrs["jet_mass"],
        arrs["jet_sophonAK4_probB"], arrs["jet_sophonAK4_probC"], arrs["jet_sophonAK4_probL"])

    gbh_eta, gbh_phi, gbh_fromhh = arrs["gen_bhadron_eta"], arrs["gen_bhadron_phi"], arrs["gen_bhadron_fromhh"]

    rows = []
    for i in sorted(needed_entries):
        mask = np.asarray(gbh_fromhh[i], dtype=bool)
        n_truth = int(mask.sum())
        nslots = min(int(n_sel[i]), 10)
        if n_truth == 0 or nslots == 0:
            j5_matched_new = False
            n_matched_leading4_new = 0
            all4_leading4_new = False
        else:
            teta = np.asarray(gbh_eta[i])[mask]
            tphi = np.asarray(gbh_phi[i])[mask]
            jeta = jet_eta[i, :nslots]
            jphi = jet_phi[i, :nslots]
            _, matched_slot = bijective_match(teta, tphi, jeta, jphi, nslots)
            j5_matched_new = bool(matched_slot[4]) if nslots > 4 else False
            n_matched_leading4_new = int(matched_slot[:4].sum())
            all4_leading4_new = (n_truth == 4) and (n_matched_leading4_new == 4)
        rows.append(dict(
            file_index=file_index, entry_index=i,
            n_truth_bhadrons_fromhh_reverify=n_truth,
            j5_is_truth_matched_to_HH_b_CORRECTED=j5_matched_new,
            n_truth_matched_in_leading4_CORRECTED=n_matched_leading4_new,
            truth_all4_in_leading4_CORRECTED=all4_leading4_new,
        ))
    return pd.DataFrame(rows)


def main():
    t0 = time.time()
    tail = pd.read_parquet(SIGNAL_TAIL, columns=["file_index", "entry_index", "source_file"])
    file_map = tail.drop_duplicates("file_index").set_index("file_index")["source_file"].to_dict()
    needed_by_file = tail.groupby("file_index")["entry_index"].apply(set).to_dict()
    print(f"files to re-read: {len(needed_by_file)}; total events to rematch: {len(tail)}")

    results = []
    with ThreadPoolExecutor(max_workers=N_WORKERS) as ex:
        futs = {ex.submit(process_file, fi, file_map[fi], needed_by_file[fi]): fi for fi in needed_by_file}
        done = 0
        for fut in as_completed(futs):
            fi = futs[fut]
            try:
                df = fut.result()
                results.append(df)
            except Exception as e:
                print(f"FILE {fi} ({file_map[fi]}) FAILED: {e}", file=sys.stderr)
            done += 1
            if done % 20 == 0:
                print(f"[{done}/{len(needed_by_file)}] files done, {time.time()-t0:.0f}s elapsed")

    out = pd.concat(results, ignore_index=True)
    out.to_parquet(OUT_PARQUET, index=False)
    print(f"wrote {OUT_PARQUET}: {len(out)} rows, {time.time()-t0:.0f}s total")


if __name__ == "__main__":
    main()
