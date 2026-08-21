#!/usr/bin/env python3
"""
Tasks A, C, D, E for the residual-background diagnostic. Reads the
scored sidecar (scored_matched_400k.npz) written by
score_matched_cohort.py. All four-jet topology observables use the
column mapping proven in FEATURE_SEMANTICS_FREEZE.md (Task B) - not
re-derived here, just applied.

Pairing rule (predeclared, applied identically to every population):
of the 3 ways to split 4 leading jets into two dijet pairs -
(01)(23), (02)(13), (03)(12) - the "descriptive pairing" is the one
minimizing R_HH = sqrt((m1-125)^2 + (m2-125)^2). Within that chosen
pairing, "pair1" is defined as the dijet with the HIGHER total pT
(deterministic tie-break, applied uniformly) - never assigned by which
mass is bigger/smaller, to avoid a mass-sorting artifact in the m1-vs-m2
plane. m_HH (four-jet invariant mass) is reported separately as
PAIRING-INDEPENDENT (sum of all 4 four-vectors - identical regardless
of how they're paired).

DEVELOPMENT ONLY. No training, no Condor, no inference/test data.
"""
import json, os

import numpy as np
import pandas as pd

OUT = "/uscms_data/d3/iturkmen/hh4b_delphes/paper_exports/track_b_development_residual_backgrounds_20260821_v1"
RES = f"{OUT}/results"
EPS_S_LIST = [0.60, 0.50, 0.40, 0.25, 0.20, 0.10]
PRIMARY_WPS = [0.50, 0.40]
LIMITED_N = 100
EXTREME_LIMITED_N = 20

# feature_extraction.py column slices (verified in FEATURE_SEMANTICS_FREEZE.md)
PT_SLICE, ETA_SLICE, PHI_SLICE, MASS_SLICE = slice(0, 10), slice(10, 20), slice(20, 30), slice(30, 40)
MASK_SLICE, NSEL_COL, HT_COL = slice(40, 50), 50, 51
PROBB_SLICE, PROBC_SLICE, PROBL_SLICE = slice(52, 62), slice(62, 72), slice(72, 82)


def four_vec(pt, eta, phi, mass):
    px = pt * np.cos(phi)
    py = pt * np.sin(phi)
    pz = pt * np.sinh(eta)
    E = np.sqrt(np.maximum(px**2 + py**2 + pz**2 + mass**2, 0.0))
    return np.stack([E, px, py, pz], axis=-1)  # (..., 4)


def inv_mass(p4):
    E, px, py, pz = p4[..., 0], p4[..., 1], p4[..., 2], p4[..., 3]
    m2 = E**2 - px**2 - py**2 - pz**2
    return np.sqrt(np.maximum(m2, 0.0))


def pt_of(p4):
    return np.sqrt(p4[..., 1]**2 + p4[..., 2]**2)


def delta_r(eta1, phi1, eta2, phi2):
    deta = eta1 - eta2
    dphi = np.abs(phi1 - phi2)
    dphi = np.where(dphi > np.pi, 2 * np.pi - dphi, dphi)
    return np.sqrt(deta**2 + dphi**2)


def compute_four_jet_topology(X):
    """Vectorized over ALL events. Returns a dict of per-event arrays (NaN where
    invalid4 is False) plus the valid4 boolean mask."""
    n = X.shape[0]
    pt = X[:, PT_SLICE][:, :4]
    eta = X[:, ETA_SLICE][:, :4]
    phi = X[:, PHI_SLICE][:, :4]
    mass = X[:, MASS_SLICE][:, :4]
    mask4 = X[:, MASK_SLICE][:, :4]
    valid4 = (mask4 == 1).all(axis=1)

    p4 = four_vec(pt, eta, phi, mass)  # (n, 4jets, 4)

    pairings = [((0, 1), (2, 3)), ((0, 2), (1, 3)), ((0, 3), (1, 2))]
    pairing_results = []  # list of dicts, one per pairing, each with m1,m2,R_HH arrays (n,)
    for (i1, i2), (j1, j2) in pairings:
        p_a = p4[:, i1] + p4[:, i2]
        p_b = p4[:, j1] + p4[:, j2]
        m_a, m_b = inv_mass(p_a), inv_mass(p_b)
        R_HH = np.sqrt((m_a - 125.0) ** 2 + (m_b - 125.0) ** 2)
        pairing_results.append({"m_a": m_a, "m_b": m_b, "R_HH": R_HH, "p_a": p_a, "p_b": p_b,
                                 "idx_a": (i1, i2), "idx_b": (j1, j2)})

    R_stack = np.stack([pr["R_HH"] for pr in pairing_results], axis=0)  # (3, n)
    best_pairing_idx = np.argmin(R_stack, axis=0)  # (n,) which of the 3 pairings minimizes R_HH

    m1 = np.full(n, np.nan); m2 = np.full(n, np.nan); R_HH_best = np.full(n, np.nan)
    pt1 = np.full(n, np.nan); pt2 = np.full(n, np.nan)
    dR1 = np.full(n, np.nan); dR2 = np.full(n, np.nan)
    for k, pr in enumerate(pairing_results):
        sel = (best_pairing_idx == k) & valid4
        p_a, p_b, m_a, m_b = pr["p_a"][sel], pr["p_b"][sel], pr["m_a"][sel], pr["m_b"][sel]
        pt_a, pt_b = pt_of(p_a), pt_of(p_b)
        # pair1 := higher-pT dijet (deterministic tie-break convention, applied uniformly)
        a_is_pair1 = pt_a >= pt_b
        m1[sel] = np.where(a_is_pair1, m_a, m_b)
        m2[sel] = np.where(a_is_pair1, m_b, m_a)
        pt1[sel] = np.where(a_is_pair1, pt_a, pt_b)
        pt2[sel] = np.where(a_is_pair1, pt_b, pt_a)
        R_HH_best[sel] = pr["R_HH"][sel]
        i1, i2 = pr["idx_a"]; j1, j2 = pr["idx_b"]
        dR_a = delta_r(eta[sel, i1], phi[sel, i1], eta[sel, i2], phi[sel, i2])
        dR_b = delta_r(eta[sel, j1], phi[sel, j1], eta[sel, j2], phi[sel, j2])
        dR1[sel] = np.where(a_is_pair1, dR_a, dR_b)
        dR2[sel] = np.where(a_is_pair1, dR_b, dR_a)

    mass_asym = np.abs(m1 - m2) / (m1 + m2)
    dR_max = np.maximum(dR1, dR2)
    dR_min = np.minimum(dR1, dR2)

    p4_sum = p4.sum(axis=1)  # pairing-independent: sum of all 4 jets
    m_HH = np.where(valid4, inv_mass(p4_sum), np.nan)

    result = {
        "valid4": valid4,
        "n_selected_jets": X[:, NSEL_COL],
        "HT": X[:, HT_COL],
        "leading_jet_pt": pt[:, 0],
        "probB_leading4_mean": np.where(valid4, X[:, PROBB_SLICE][:, :4].mean(axis=1), np.nan),
        "probC_leading4_mean": np.where(valid4, X[:, PROBC_SLICE][:, :4].mean(axis=1), np.nan),
        "probL_leading4_mean": np.where(valid4, X[:, PROBL_SLICE][:, :4].mean(axis=1), np.nan),
        "n_real_jet_slots": X[:, MASK_SLICE].sum(axis=1),
        "m_bb1": np.where(valid4, m1, np.nan), "m_bb2": np.where(valid4, m2, np.nan),
        "R_HH": np.where(valid4, R_HH_best, np.nan),
        "mass_asymmetry": np.where(valid4, mass_asym, np.nan),
        "deltaR_bb1": np.where(valid4, dR1, np.nan), "deltaR_bb2": np.where(valid4, dR2, np.nan),
        "deltaR_bb_max": np.where(valid4, dR_max, np.nan), "deltaR_bb_min": np.where(valid4, dR_min, np.nan),
        "pT_pair1": np.where(valid4, pt1, np.nan), "pT_pair2": np.where(valid4, pt2, np.nan),
        "m_HH": m_HH,
        # all-three-pairings pairing-independent summary set (item 1 of Task C)
        "all_pairings_m_a": np.stack([pr["m_a"] for pr in pairing_results], axis=1),  # (n,3)
        "all_pairings_m_b": np.stack([pr["m_b"] for pr in pairing_results], axis=1),
        "all_pairings_R_HH": np.stack([pr["R_HH"] for pr in pairing_results], axis=1),
    }
    return result


def n_flag(n):
    if n < EXTREME_LIMITED_N:
        return "EXTREMELY_LIMITED"
    if n < LIMITED_N:
        return "LIMITED"
    return "ok"


def summarize_population(topo, sel_mask, label):
    """sel_mask: boolean array over the full 400k. Returns a dict of summary stats
    for every Task C observable, restricted further to valid4 within sel_mask, with N and flag."""
    valid = topo["valid4"] & sel_mask
    n_total_sel = int(sel_mask.sum())
    n_valid4 = int(valid.sum())
    n_excluded_lt4jets = n_total_sel - n_valid4

    def stat(key, mask=valid):
        arr = topo[key][mask]
        arr = arr[~np.isnan(arr)]
        if len(arr) == 0:
            return {"n": 0, "mean": None, "median": None, "std": None, "min": None, "max": None}
        return {"n": int(len(arr)), "mean": float(np.mean(arr)), "median": float(np.median(arr)),
                "std": float(np.std(arr)), "min": float(np.min(arr)), "max": float(np.max(arr))}

    out = {
        "label": label, "n_selected_total": n_total_sel, "n_valid4": n_valid4,
        "n_excluded_fewer_than_4_real_jets": n_excluded_lt4jets,
        "n_flag": n_flag(n_valid4),
        "n_selected_jets": stat("n_selected_jets", sel_mask),
        "HT": stat("HT", sel_mask),
        "leading_jet_pt": stat("leading_jet_pt", sel_mask),
        "probB_leading4_mean": stat("probB_leading4_mean"),
        "probC_leading4_mean": stat("probC_leading4_mean"),
        "probL_leading4_mean": stat("probL_leading4_mean"),
        "n_real_jet_slots": stat("n_real_jet_slots", sel_mask),
        "m_bb1": stat("m_bb1"), "m_bb2": stat("m_bb2"), "R_HH": stat("R_HH"),
        "mass_asymmetry": stat("mass_asymmetry"),
        "deltaR_bb1": stat("deltaR_bb1"), "deltaR_bb2": stat("deltaR_bb2"),
        "deltaR_bb_max": stat("deltaR_bb_max"), "deltaR_bb_min": stat("deltaR_bb_min"),
        "pT_pair1": stat("pT_pair1"), "pT_pair2": stat("pT_pair2"),
        "m_HH": stat("m_HH"),
    }
    return out


def main():
    os.makedirs(RES, exist_ok=True)
    d = np.load(f"{RES}/scored_matched_400k.npz", allow_pickle=True)
    X, y, process = d["X"], d["y"], d["process"]
    score = {"K": d["score_K"], "KF": d["score_KF"], "SPANET2M": d["score_SPANET2M"]}
    with open(f"{RES}/thresholds_frozen.json") as f:
        thresholds = json.load(f)

    is_qcd = process == "qcd"
    is_ttbar = process == "ttbar"
    is_bg = is_qcd | is_ttbar
    n_qcd_total, n_ttbar_total = int(is_qcd.sum()), int(is_ttbar.sum())
    n_bg_total = int(is_bg.sum())

    print("computing four-jet topology for all 400,000 events (vectorized, once)...")
    topo = compute_four_jet_topology(X)
    print(f"valid4 (>=4 real jet slots): {int(topo['valid4'].sum())} / 400000")

    # ---------------- Task A: composition table ----------------
    comp_rows = []
    for arm in ["K", "KF", "SPANET2M"]:
        for epsS in EPS_S_LIST:
            thr = thresholds[arm][str(epsS)] if str(epsS) in thresholds[arm] else thresholds[arm][epsS]
            passmask = score[arm] >= thr
            n_qcd_pass = int((passmask & is_qcd).sum())
            n_ttbar_pass = int((passmask & is_ttbar).sum())
            n_bg_pass = n_qcd_pass + n_ttbar_pass
            comp_rows.append({
                "model": arm, "epsS": epsS, "threshold": thr,
                "role": "PRIMARY" if epsS in PRIMARY_WPS else ("cross-check" if epsS == 0.60 else "TAIL_DIAGNOSTIC"),
                "n_qcd_survive": n_qcd_pass, "n_ttbar_survive": n_ttbar_pass, "n_bg_survive_total": n_bg_pass,
                "raw_qcd_fraction_of_survivors": (n_qcd_pass / n_bg_pass) if n_bg_pass > 0 else None,
                "raw_ttbar_fraction_of_survivors": (n_ttbar_pass / n_bg_pass) if n_bg_pass > 0 else None,
                "preselection_qcd_fraction": n_qcd_total / n_bg_total,
                "preselection_ttbar_fraction": n_ttbar_total / n_bg_total,
                "n_flag": n_flag(n_bg_pass),
                "NOTE": "RAW DEVELOPMENT-COHORT FRACTIONS, NOT PHYSICALLY NORMALIZED BACKGROUND COMPOSITION",
            })
    # SPA-Net-10M: frozen SUMMARY counts only (no event-level topology - separate governance)
    spanet10m_path = ("/uscms_data/d3/iturkmen/hh4b_delphes/track_b_phase4_preflight_20260812/"
                       "phase4AI_spanet_10M_scaling_seed0_20260821_v1/classification_evaluation_10M/"
                       "work/classification_evaluation_10M_result.json")
    with open(spanet10m_path) as f:
        spanet10m = json.load(f)
    for wp in spanet10m["working_points"]:
        epsS = wp["target_signal_efficiency"]
        n_qcd_pass, n_ttbar_pass = wp["n_qcd_pass"], wp["n_ttbar_pass"]
        n_bg_pass = n_qcd_pass + n_ttbar_pass
        comp_rows.append({
            "model": "SPANET10M_SUMMARY_ONLY", "epsS": epsS, "threshold": wp["threshold"],
            "role": "PRIMARY" if epsS in PRIMARY_WPS else ("cross-check" if epsS == 0.60 else "TAIL_DIAGNOSTIC"),
            "n_qcd_survive": n_qcd_pass, "n_ttbar_survive": n_ttbar_pass, "n_bg_survive_total": n_bg_pass,
            "raw_qcd_fraction_of_survivors": (n_qcd_pass / n_bg_pass) if n_bg_pass > 0 else None,
            "raw_ttbar_fraction_of_survivors": (n_ttbar_pass / n_bg_pass) if n_bg_pass > 0 else None,
            "preselection_qcd_fraction": n_qcd_total / n_bg_total,
            "preselection_ttbar_fraction": n_ttbar_total / n_bg_total,
            "n_flag": n_flag(n_bg_pass),
            "NOTE": "SUMMARY COUNTS ONLY (frozen phase4AI classification_evaluation_10M, checkpoint "
                    "epoch=47) - event-level topology UNAVAILABLE for SPA-Net-10M in this phase; "
                    "requires a separately governed frozen-checkpoint rescoring pass",
        })
    comp_df = pd.DataFrame(comp_rows)
    comp_df.to_csv(f"{OUT}/RESIDUAL_BACKGROUND_COMPOSITION.csv", index=False)
    print(f"wrote {OUT}/RESIDUAL_BACKGROUND_COMPOSITION.csv ({len(comp_df)} rows)")

    # ---------------- Task C: topology summaries ----------------
    topo_rows = []
    # pre-selection populations (no epsS)
    for proc_label, proc_mask in [("qcd", is_qcd), ("ttbar", is_ttbar), ("combined_background", is_bg)]:
        s = summarize_population(topo, proc_mask, f"preselection_{proc_label}")
        s["population_type"] = "preselection"; s["process"] = proc_label; s["model"] = None; s["epsS"] = None
        topo_rows.append(s)
    # model survivors at epsS 0.50, 0.40
    for arm in ["K", "KF", "SPANET2M"]:
        for epsS in PRIMARY_WPS:
            thr = thresholds[arm][str(epsS)] if str(epsS) in thresholds[arm] else thresholds[arm][epsS]
            passmask = score[arm] >= thr
            for proc_label, proc_mask in [("qcd", is_qcd), ("ttbar", is_ttbar), ("combined_background", is_bg)]:
                sel = passmask & proc_mask
                s = summarize_population(topo, sel, f"{arm}_epsS{epsS}_{proc_label}")
                s["population_type"] = "model_survivors"; s["process"] = proc_label; s["model"] = arm; s["epsS"] = epsS
                topo_rows.append(s)

    # flatten nested stat dicts into columns for CSV
    def flatten(row):
        flat = {k: v for k, v in row.items() if not isinstance(v, dict)}
        for key in ["n_selected_jets", "HT", "leading_jet_pt", "probB_leading4_mean", "probC_leading4_mean",
                    "probL_leading4_mean", "n_real_jet_slots", "m_bb1", "m_bb2", "R_HH", "mass_asymmetry",
                    "deltaR_bb1", "deltaR_bb2", "deltaR_bb_max", "deltaR_bb_min", "pT_pair1", "pT_pair2", "m_HH"]:
            for stat_name, stat_val in row[key].items():
                flat[f"{key}_{stat_name}"] = stat_val
        return flat

    topo_flat = pd.DataFrame([flatten(r) for r in topo_rows])
    topo_flat.to_csv(f"{OUT}/RESIDUAL_BACKGROUND_TOPOLOGY_SUMMARY.csv", index=False)
    print(f"wrote {OUT}/RESIDUAL_BACKGROUND_TOPOLOGY_SUMMARY.csv ({len(topo_flat)} rows)")

    # ---------------- Task D: disagreement ----------------
    disagree_rows = []
    kf_only_masks, spa_only_masks = {}, {}
    for epsS in PRIMARY_WPS:
        thr_kf = thresholds["KF"][str(epsS)] if str(epsS) in thresholds["KF"] else thresholds["KF"][epsS]
        thr_spa = thresholds["SPANET2M"][str(epsS)] if str(epsS) in thresholds["SPANET2M"] else thresholds["SPANET2M"][epsS]
        kf_pass = score["KF"] >= thr_kf
        spa_pass = score["SPANET2M"] >= thr_spa
        for proc_label, proc_mask in [("qcd", is_qcd), ("ttbar", is_ttbar)]:
            both = kf_pass & spa_pass & proc_mask
            kf_only = kf_pass & (~spa_pass) & proc_mask
            spa_only = (~kf_pass) & spa_pass & proc_mask
            neither = (~kf_pass) & (~spa_pass) & proc_mask
            disagree_rows.append({
                "epsS": epsS, "process": proc_label,
                "KF_pass_SPA_pass": int(both.sum()), "KF_pass_SPA_fail": int(kf_only.sum()),
                "KF_fail_SPA_pass": int(spa_only.sum()), "KF_fail_SPA_fail": int(neither.sum()),
                "n_flag_KF_pass_SPA_fail": n_flag(int(kf_only.sum())),
                "n_flag_KF_fail_SPA_pass": n_flag(int(spa_only.sum())),
            })
        # combined-background KF-only / SPA-only for the topology comparison (Task D second half)
        kf_only_masks[epsS] = kf_pass & (~spa_pass) & is_bg
        spa_only_masks[epsS] = (~kf_pass) & spa_pass & is_bg

    disagree_df = pd.DataFrame(disagree_rows)
    disagree_df.to_csv(f"{OUT}/MODEL_DISAGREEMENT_COUNTS.csv", index=False)
    print(f"wrote {OUT}/MODEL_DISAGREEMENT_COUNTS.csv ({len(disagree_df)} rows)")

    disagree_topo_rows = []
    for epsS in PRIMARY_WPS:
        for label, mask in [("KF_only_survivors", kf_only_masks[epsS]), ("SPA_only_survivors", spa_only_masks[epsS])]:
            s = summarize_population(topo, mask, f"{label}_epsS{epsS}_combined_background")
            s["population_type"] = "disagreement"; s["process"] = "combined_background"
            s["model"] = label; s["epsS"] = epsS
            disagree_topo_rows.append(s)
    disagree_topo_flat = pd.DataFrame([flatten(r) for r in disagree_topo_rows])
    disagree_topo_flat.to_csv(f"{RES}/disagreement_topology_summary.csv", index=False)
    print(f"wrote {RES}/disagreement_topology_summary.csv ({len(disagree_topo_flat)} rows)")

    # save raw per-event arrays needed by the plotting script (avoid recompute)
    np.savez(f"{RES}/topology_full.npz", **{k: v for k, v in topo.items() if isinstance(v, np.ndarray)},
             is_qcd=is_qcd, is_ttbar=is_ttbar, is_bg=is_bg,
             score_K=score["K"], score_KF=score["KF"], score_SPANET2M=score["SPANET2M"])
    print(f"wrote {RES}/topology_full.npz")

    with open(f"{RES}/task_summary.json", "w") as f:
        json.dump({
            "n_qcd_total": n_qcd_total, "n_ttbar_total": n_ttbar_total, "n_bg_total": n_bg_total,
            "n_valid4_total": int(topo["valid4"].sum()),
            "n_excluded_fewer_than_4_jets_total": int((~topo["valid4"]).sum()),
        }, f, indent=2)
    print("Task A/C/D computation complete.")


if __name__ == "__main__":
    main()
