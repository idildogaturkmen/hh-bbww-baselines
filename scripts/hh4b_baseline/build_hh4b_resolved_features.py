'''
Build HH4b resolved features.
Features in this script are based on the "resolved" analysis strategy, which uses four resolved b-jets to reconstruct two Higgs bosons.
The script reads in Parquet files containing selected events, extracts relevant features, and saves the results to a new Parquet file along with cutflow summaries and feature summaries.

Features:
    - Event-level features: number of AK4 jets, number of b-tagged jets, number of AK8 jets with pt > 200 GeV and 300 GeV, HT, VBF-like features (mjj and deta).
    - Higgs reconstruction features: best pairing of four b-jets to form two Higgs candidates, their masses, transverse momenta, and delta R between the b-jets.
The script also summarizes the cutflow and feature distributions for different event categories.
Usage:
    python build_hh4b_resolved_features.py --input-glob "outputs/collide_selected_backgrounds/**/*.parquet" --metadata "config/collide_sample_metadata_rough.csv" --outdir "outputs/hh4b_baseline" --lumi-fb 138    
    
'''

from __future__ import annotations

import argparse
import importlib.util
import math
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq


def load_scout_module():
    path = Path("scripts/channel_scouting/scout_hh_channels.py")
    spec = importlib.util.spec_from_file_location("scout_hh_channels", path)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


scout = load_scout_module()


def safe_arr(row, col):
    if col is None or col not in row.index:
        return []
    return scout.as_list(row[col])


def vec_pt_eta_phi_m(pt, eta, phi, mass):
    px = pt * math.cos(phi)
    py = pt * math.sin(phi)
    pz = pt * math.sinh(eta)
    e = math.sqrt(max(px * px + py * py + pz * pz + mass * mass, 0.0))
    return e, px, py, pz


def add_vec(vs):
    e = sum(v[0] for v in vs)
    px = sum(v[1] for v in vs)
    py = sum(v[2] for v in vs)
    pz = sum(v[3] for v in vs)
    pt = math.sqrt(px * px + py * py)
    mass = math.sqrt(max(e * e - px * px - py * py - pz * pz, 0.0))
    eta = 0.0
    if e != abs(pz):
        try:
            eta = 0.5 * math.log((e + pz) / (e - pz))
        except Exception:
            eta = 0.0
    phi = math.atan2(py, px)
    return {"e": e, "px": px, "py": py, "pz": pz, "pt": pt, "eta": eta, "phi": phi, "mass": mass}


def best_hh4b_pairing(row, m, bjets):
    pt = safe_arr(row, m["ak4_pt"])
    eta = safe_arr(row, m["ak4_eta"])
    phi = safe_arr(row, m["ak4_phi"])
    mass = safe_arr(row, m["ak4_mass"])

    idx = [b["idx"] for b in bjets[:4]]
    if len(idx) < 4:
        return {}

    def jet4(i):
        return vec_pt_eta_phi_m(
            float(pt[i]),
            float(eta[i]) if i < len(eta) else 0.0,
            float(phi[i]) if i < len(phi) else 0.0,
            float(mass[i]) if i < len(mass) else 0.0,
        )

    pairings = [
        ((idx[0], idx[1]), (idx[2], idx[3])),
        ((idx[0], idx[2]), (idx[1], idx[3])),
        ((idx[0], idx[3]), (idx[1], idx[2])),
    ]

    best = None
    for p1, p2 in pairings:
        h1 = add_vec([jet4(p1[0]), jet4(p1[1])])
        h2 = add_vec([jet4(p2[0]), jet4(p2[1])])
        hh = add_vec([jet4(p1[0]), jet4(p1[1]), jet4(p2[0]), jet4(p2[1])])

        eta1a = float(eta[p1[0]]) if p1[0] < len(eta) else 0.0
        eta1b = float(eta[p1[1]]) if p1[1] < len(eta) else 0.0
        phi1a = float(phi[p1[0]]) if p1[0] < len(phi) else 0.0
        phi1b = float(phi[p1[1]]) if p1[1] < len(phi) else 0.0

        eta2a = float(eta[p2[0]]) if p2[0] < len(eta) else 0.0
        eta2b = float(eta[p2[1]]) if p2[1] < len(eta) else 0.0
        phi2a = float(phi[p2[0]]) if p2[0] < len(phi) else 0.0
        phi2b = float(phi[p2[1]]) if p2[1] < len(phi) else 0.0

        dr1 = scout.delta_r(eta1a, phi1a, eta1b, phi1b)
        dr2 = scout.delta_r(eta2a, phi2a, eta2b, phi2b)

        score = abs(h1["mass"] - 125.0) + abs(h2["mass"] - 125.0)

        candidate = {
            "h1_mass": h1["mass"],
            "h2_mass": h2["mass"],
            "h1_pt": h1["pt"],
            "h2_pt": h2["pt"],
            "hh_mass": hh["mass"],
            "hh_pt": hh["pt"],
            "dr_h1_bb": dr1,
            "dr_h2_bb": dr2,
            "pairing_score": score,
            "h_mass_avg": 0.5 * (h1["mass"] + h2["mass"]),
            "h_mass_diff": abs(h1["mass"] - h2["mass"]),
            "h1_abs_m125": abs(h1["mass"] - 125.0),
            "h2_abs_m125": abs(h2["mass"] - 125.0),
            "pair_a0": p1[0],
            "pair_a1": p1[1],
            "pair_b0": p2[0],
            "pair_b1": p2[1],
        }

        if best is None or candidate["pairing_score"] < best["pairing_score"]:
            best = candidate

    return best or {}


def get_bjets(row, m, ak4_pt_min, ak4_eta_max, btag_threshold):
    pt = safe_arr(row, m["ak4_pt"])
    eta = safe_arr(row, m["ak4_eta"])
    phi = safe_arr(row, m["ak4_phi"])
    mass = safe_arr(row, m["ak4_mass"])

    binfo = scout.btag_mask_and_scores(row, m, ak4_pt_min, ak4_eta_max, btag_threshold)
    bjets = []

    for idx, is_b, score, p in binfo:
        if not is_b:
            continue

        bjets.append({
            "idx": idx,
            "pt": float(pt[idx]) if idx < len(pt) else 0.0,
            "eta": float(eta[idx]) if idx < len(eta) else 0.0,
            "phi": float(phi[idx]) if idx < len(phi) else 0.0,
            "mass": float(mass[idx]) if idx < len(mass) else 0.0,
            "btag": float(score),
        })

    bjets = sorted(bjets, key=lambda x: (x["btag"], x["pt"]), reverse=True)
    return bjets


def get_basic_counts(row, m, ak4_pt_min, ak4_eta_max):
    ak4_pt = safe_arr(row, m["ak4_pt"])
    ak4_eta = safe_arr(row, m["ak4_eta"])
    ak8_pt = safe_arr(row, m["ak8_pt"])
    ak8_eta = safe_arr(row, m["ak8_eta"])

    n_ak4 = scout.count_pt_eta(ak4_pt, ak4_eta, ak4_pt_min, ak4_eta_max)
    n_ak8_200 = scout.count_pt_eta(ak8_pt, ak8_eta, 200.0, 2.5)
    n_ak8_300 = scout.count_pt_eta(ak8_pt, ak8_eta, 300.0, 2.5)
    ht = sum(float(p) for i, p in enumerate(ak4_pt) if p >= ak4_pt_min and (i >= len(ak4_eta) or abs(float(ak4_eta[i])) <= ak4_eta_max))

    return n_ak4, n_ak8_200, n_ak8_300, ht


def asimov_z(s, b):
    if s <= 0 or b <= 0:
        return 0.0
    return math.sqrt(max(0.0, 2.0 * ((s + b) * math.log(1.0 + s / b) - s)))


def neff(w):
    w = np.asarray(w, dtype=float)
    w = w[np.isfinite(w)]
    if len(w) == 0 or np.sum(w * w) <= 0:
        return 0.0
    return float(np.sum(w) ** 2 / np.sum(w * w))


def summarize_region(df, mask, name):
    sub = df.loc[mask].copy()
    sig = sub["is_signal"].astype(bool)
    bkg = ~sig

    sw = sub.loc[sig & np.isfinite(sub["physics_weight"]), "physics_weight"].sum()
    bw = sub.loc[bkg & np.isfinite(sub["physics_weight"]), "physics_weight"].sum()
    bkg_w = sub.loc[bkg & np.isfinite(sub["physics_weight"]), "physics_weight"].to_numpy()

    raw_sig_missing = int((sig & ~np.isfinite(sub["physics_weight"])).sum())
    raw_bkg_missing = int((bkg & ~np.isfinite(sub["physics_weight"])).sum())

    return {
        "region": name,
        "raw_signal": int(sig.sum()),
        "raw_background": int(bkg.sum()),
        "raw_signal_missing_weight": raw_sig_missing,
        "raw_background_missing_weight": raw_bkg_missing,
        "S_w": float(sw),
        "B_w": float(bw),
        "S_over_B": float(sw / bw) if bw > 0 else 0.0,
        "S_over_SplusB": float(sw / (sw + bw)) if (sw + bw) > 0 else 0.0,
        "S_over_sqrtB": float(sw / math.sqrt(bw)) if bw > 0 else 0.0,
        "S_over_sqrtSplusB": float(sw / math.sqrt(sw + bw)) if (sw + bw) > 0 else 0.0,
        "asimov_Z_A": asimov_z(sw, bw),
        "N_eff_bkg": neff(bkg_w),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-glob", default="outputs/collide_selected_backgrounds/**/*.parquet")
    parser.add_argument("--metadata", default="config/collide_sample_metadata_rough.csv")
    parser.add_argument("--outdir", default="outputs/hh4b_baseline")
    parser.add_argument("--lumi-fb", type=float, default=138.0)
    parser.add_argument("--ak4-pt-min", type=float, default=30.0)
    parser.add_argument("--ak4-eta-max", type=float, default=2.5)
    parser.add_argument("--btag-threshold", type=float, default=0.5)
    parser.add_argument("--max-files", type=int, default=None)
    parser.add_argument("--max-events-per-file", type=int, default=None)
    args = parser.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    files = sorted(Path(".").glob(args.input_glob))
    if args.max_files:
        files = files[:args.max_files]

    metadata = scout.load_metadata(args.metadata)
    sample_info = scout.build_sample_info(files, metadata, args.lumi_fb)

    rows = []
    per_file_rows = []

    for file_idx, f in enumerate(files, start=1):
        sample = scout.sample_from_path(f)
        sinfo = sample_info[sample]
        group = str(sinfo["process_group"])
        weight = float(sinfo["event_weight"]) if sinfo["has_weight"] else float("nan")

        pf = pq.ParquetFile(f)
        cols = set(pf.schema_arrow.names)
        m = scout.discover_columns(cols)
        needed = sorted({c for c in m.values() if c is not None})

        if not needed:
            print(f"[WARN] No useful columns in {f}")
            continue

        df = pq.read_table(f, columns=needed).to_pandas()
        if args.max_events_per_file:
            df = df.iloc[:args.max_events_per_file]

        n_read = len(df)
        n_pre = 0

        print(f"[INFO] [{file_idx}/{len(files)}] {sample}: {n_read} rows")

        for local_idx, row in df.iterrows():
            bjets = get_bjets(row, m, args.ak4_pt_min, args.ak4_eta_max, args.btag_threshold)
            if len(bjets) < 4:
                continue

            n_pre += 1
            n_ak4, n_ak8_200, n_ak8_300, ht = get_basic_counts(row, m, args.ak4_pt_min, args.ak4_eta_max)
            vbf_mjj, vbf_deta = scout.vbf_pair_features(row, m, 30.0, 5.0, args.btag_threshold)
            pair = best_hh4b_pairing(row, m, bjets)

            is_sig = sample == "HH_4b"
            is_hh_any = scout.is_signal(sample, group, "hh_any")

            out = {
                "event_key": f"{file_idx}:{local_idx}",
                "file": str(f),
                "sample": sample,
                "process_group": group,
                "is_signal": int(is_sig),
                "is_hh_any": int(is_hh_any),
                "has_weight": bool(sinfo["has_weight"]),
                "physics_weight": weight,
                "n_ak4": n_ak4,
                "n_btag": len(bjets),
                "n_ak8_200": n_ak8_200,
                "n_ak8_300": n_ak8_300,
                "ht": ht,
                "vbf_mjj": vbf_mjj,
                "vbf_deta": vbf_deta,
            }

            for i in range(4):
                bj = bjets[i]
                out[f"b{i+1}_pt"] = bj["pt"]
                out[f"b{i+1}_eta"] = bj["eta"]
                out[f"b{i+1}_phi"] = bj["phi"]
                out[f"b{i+1}_mass"] = bj["mass"]
                out[f"b{i+1}_btag"] = bj["btag"]

            out.update(pair)

            out["category_hmass_70_190"] = int(
                70.0 <= out.get("h1_mass", -999.0) <= 190.0
                and 70.0 <= out.get("h2_mass", -999.0) <= 190.0
            )
            out["category_hmass_90_160"] = int(
                90.0 <= out.get("h1_mass", -999.0) <= 160.0
                and 90.0 <= out.get("h2_mass", -999.0) <= 160.0
            )
            out["category_vbf_like"] = int(vbf_mjj >= 500.0 and vbf_deta >= 3.0)
            out["category_boosted_1ak8_2b"] = int(n_ak8_300 >= 1 and len(bjets) >= 2)

            rows.append(out)

        per_file_rows.append({
            "file": str(f),
            "sample": sample,
            "process_group": group,
            "raw_read": n_read,
            "raw_ge4b": n_pre,
            "has_weight": bool(sinfo["has_weight"]),
            "event_weight": weight,
            "xsec_proxy": sinfo["xsec_proxy"],
        })

    feat = pd.DataFrame(rows)
    per_file = pd.DataFrame(per_file_rows)

    feature_path = outdir / "hh4b_resolved_features.parquet"
    feat.to_parquet(feature_path, index=False)

    per_file.to_csv(outdir / "hh4b_resolved_file_cutflow.csv", index=False)

    summaries = []
    summaries.append(summarize_region(feat, np.ones(len(feat), dtype=bool), "resolved_4b_basic"))
    summaries.append(summarize_region(feat, feat["category_hmass_70_190"].astype(bool), "resolved_4b_Hmass_70_190"))
    summaries.append(summarize_region(feat, feat["category_hmass_90_160"].astype(bool), "resolved_4b_Hmass_90_160"))
    summaries.append(summarize_region(feat, feat["category_vbf_like"].astype(bool), "resolved_4b_VBF_like"))
    summaries.append(summarize_region(feat, feat["category_boosted_1ak8_2b"].astype(bool), "resolved_4b_boosted_1AK8_2b"))

    cut_summary = pd.DataFrame(summaries)
    cut_summary.to_csv(outdir / "hh4b_resolved_cut_baseline_summary.csv", index=False)

    feature_cols = [
        c for c in feat.columns
        if c not in {
            "event_key", "file", "sample", "process_group",
            "is_signal", "is_hh_any", "has_weight", "physics_weight",
        }
    ]
    feature_summary = feat[feature_cols].describe().T
    feature_summary.to_csv(outdir / "hh4b_resolved_feature_summary.csv")

    print("\n[INFO] Wrote:")
    print(" -", feature_path)
    print(" -", outdir / "hh4b_resolved_file_cutflow.csv")
    print(" -", outdir / "hh4b_resolved_cut_baseline_summary.csv")
    print(" -", outdir / "hh4b_resolved_feature_summary.csv")

    print("\n[INFO] Cut baseline summary:")
    print(cut_summary.to_string(index=False))


if __name__ == "__main__":
    main()
