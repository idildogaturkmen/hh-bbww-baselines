#!/usr/bin/env python3
"""Correction 3a: EXACT maximum-cardinality, minimum-total-DeltaR bipartite
truth<->jet matching, compared event-by-event against the GREEDY
edge-sorted one-to-one matching computed in
audit_and_rematch_truth_correction3.py (which is one-to-one but was not
proven globally minimal in total DeltaR -- user-flagged gap).

Second read-only pass over the SAME 200 authorized signal_holdout_A files
(no new holdout, no new inference, no training, no new MC) -- required
because raw truth (eta,phi) and jet (eta,phi) per-event coordinates were
not retained on disk after the first pass. Only the u>3.5 tail events
already identified in the master table are processed (27,339 signal
events); per-file cost is unchanged from the first pass (~108s for 200
files), so this is expected to complete in a similar time.
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
from exact_bipartite_match import exact_match, greedy_match  # noqa: E402

XROOTD_PREFIX = "root://cceos.ihep.ac.cn:1094/"
N_WORKERS = 4
BRANCHES = ["jet_pt", "jet_eta", "jet_phi", "jet_mass", "jet_sophonAK4_probB", "jet_sophonAK4_probC", "jet_sophonAK4_probL",
            "gen_bhadron_eta", "gen_bhadron_phi", "gen_bhadron_pt", "gen_bhadron_fromhh"]

SIGNAL_TAIL = "signal_tail.parquet"
OUT_PARQUET = "truth_rematch_exact_vs_greedy_correction3a.parquet"


def per_event_result(teta, tphi, jeta, jphi, nslots, n_truth):
    dR = np.stack([delta_r(teta, tphi, np.full(n_truth, jeta[j]), np.full(n_truth, jphi[j])) for j in range(nslots)], axis=1) \
        if (n_truth > 0 and nslots > 0) else np.zeros((n_truth, nslots))
    a_ex, n_ex, tot_ex = exact_match(dR, n_truth, nslots) if n_truth > 0 and nslots > 0 else (np.full(n_truth, -1, dtype=np.int64), 0, 0.0)
    a_gr, n_gr, tot_gr = greedy_match(dR, n_truth, nslots) if n_truth > 0 and nslots > 0 else (np.full(n_truth, -1, dtype=np.int64), 0, 0.0)
    return a_ex, n_ex, tot_ex, a_gr, n_gr, tot_gr


def process_file(file_index, remote_path, needed_entries):
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
            j5_ex = j5_gr = False
            n4_ex = n4_gr = 0
            all4_ex = all4_gr = False
            assign_ex = assign_gr = ()
        else:
            teta = np.asarray(gbh_eta[i])[mask]
            tphi = np.asarray(gbh_phi[i])[mask]
            jeta = jet_eta[i, :nslots]
            jphi = jet_phi[i, :nslots]
            a_ex, n_ex, tot_ex, a_gr, n_gr, tot_gr = per_event_result(teta, tphi, jeta, jphi, nslots, n_truth)
            j5_ex = bool(np.any(a_ex == 4)) if nslots > 4 else False
            j5_gr = bool(np.any(a_gr == 4)) if nslots > 4 else False
            n4_ex = int(np.sum((a_ex >= 0) & (a_ex < 4)))
            n4_gr = int(np.sum((a_gr >= 0) & (a_gr < 4)))
            all4_ex = (n_truth == 4) and (n4_ex == 4)
            all4_gr = (n_truth == 4) and (n4_gr == 4)
            assign_ex = tuple(int(x) for x in a_ex)
            assign_gr = tuple(int(x) for x in a_gr)

        rows.append(dict(
            file_index=file_index, entry_index=i,
            n_truth_bhadrons_fromhh=n_truth, nslots=nslots,
            assignment_exact=assign_ex, assignment_greedy=assign_gr,
            j5_matched_exact=j5_ex, j5_matched_greedy=j5_gr,
            n_matched_leading4_exact=n4_ex, n_matched_leading4_greedy=n4_gr,
            all4_leading4_exact=all4_ex, all4_leading4_greedy=all4_gr,
        ))
    return pd.DataFrame(rows)


def main():
    t0 = time.time()
    tail = pd.read_parquet(SIGNAL_TAIL, columns=["file_index", "entry_index", "source_file"])
    file_map = tail.drop_duplicates("file_index").set_index("file_index")["source_file"].to_dict()
    needed_by_file = tail.groupby("file_index")["entry_index"].apply(set).to_dict()
    print(f"files to re-read: {len(needed_by_file)}; total events: {len(tail)}")

    results = []
    with ThreadPoolExecutor(max_workers=N_WORKERS) as ex:
        futs = {ex.submit(process_file, fi, file_map[fi], needed_by_file[fi]): fi for fi in needed_by_file}
        done = 0
        for fut in as_completed(futs):
            fi = futs[fut]
            try:
                results.append(fut.result())
            except Exception as e:
                print(f"FILE {fi} ({file_map[fi]}) FAILED: {e}", file=sys.stderr)
            done += 1
            if done % 20 == 0:
                print(f"[{done}/{len(needed_by_file)}] files done, {time.time()-t0:.0f}s elapsed")

    out = pd.concat(results, ignore_index=True)
    out["assignment_exact"] = out["assignment_exact"].apply(lambda x: str(x))
    out["assignment_greedy"] = out["assignment_greedy"].apply(lambda x: str(x))
    out.to_parquet(OUT_PARQUET, index=False)
    print(f"wrote {OUT_PARQUET}: {len(out)} rows, {time.time()-t0:.0f}s total")

    n_assign_differ = int((out["assignment_exact"] != out["assignment_greedy"]).sum())
    n_j5_differ = int((out["j5_matched_exact"] != out["j5_matched_greedy"]).sum())
    n_n4_differ = int((out["n_matched_leading4_exact"] != out["n_matched_leading4_greedy"]).sum())
    n_all4_differ = int((out["all4_leading4_exact"] != out["all4_leading4_greedy"]).sum())
    print(f"assignment differs: {n_assign_differ}")
    print(f"j5 flag differs: {n_j5_differ}")
    print(f"n_matched_leading4 differs: {n_n4_differ}")
    print(f"all4_leading4 differs: {n_all4_differ}")
    print(f"EXACT j5 match fraction (of all {len(out)}): {out['j5_matched_exact'].mean():.6f}")
    print(f"EXACT all4_leading4 fraction (of all {len(out)}): {out['all4_leading4_exact'].mean():.6f}")


if __name__ == "__main__":
    main()
