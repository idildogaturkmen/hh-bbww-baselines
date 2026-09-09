"""
Small (O(100-1000) jet) inference canary for the pinned JP-JEPA Mini
pretrained encoder, against real Yang-Li resolved-AK4 constituents.

This is an INDEPENDENT implementation (not a copy of
part_jpjepa_dual_production_bundle_20260825_v2/bundle/code/extract_shard.py)
of the feature construction verified in FEATURE_COMPATIBILITY_MATRIX.md
and PREPROCESSING_SPEC.md, written to cross-check that prior pipeline
rather than blindly reuse its code. It reads a small, bounded number of
events directly from three real ROOT files (QCD, HH4b signal, ttbar) via
XRootD, builds native-order pf_features/pf_vectors/pf_mask exactly as
verified against upstream JP-JEPA (jetparticle-jepa @
c68509eead1866c2c86714147023f5e8312634c4), reorders features the way
JP-JEPA's own inference.py does, runs the pinned pretrained Mini encoder,
and extracts jet_embedding = all_layer_outputs[-1][0][0] exactly as
verified in EXTRACTION_POINT_PROOF.md.

Canary scale only -- NOT the 2M production path. No SPA-Net training.
No write to any shared/production location; all output stays under
this package's own work/ directory.
"""
import argparse
import json
import math
import sys
import time

import awkward as ak
import numpy as np
import uproot


MAXLEN = 128
N_JETS_PER_EVENT_CAP = 10  # matches native SPA-Net contract's 10-jet-slot cap; not essential for a canary but kept for realism

NATURAL_ORDER = ["pt_log", "e_log", "logptrel", "logerel", "deltaR", "charge",
                 "isCHad", "isNHad", "isPhoton", "isElectron", "isMuon",
                 "d0", "d0err", "dz", "dzerr", "deta", "dphi"]

JET_BRANCHES = ["jet_pt", "jet_eta", "jet_phi", "jet_energy"]
PART_BRANCHES = ["part_px", "part_py", "part_pz", "part_energy", "part_eta", "part_phi",
                 "part_charge", "part_pid", "part_d0val", "part_d0err", "part_dzval",
                 "part_dzerr", "part_label"]


def std_pt_log(x): return np.clip((x - 1.7) * 0.7, -5.0, 5.0)
def std_e_log(x): return np.clip((x - 2.0) * 0.7, -5.0, 5.0)
def std_logptrel(x): return np.clip((x - (-4.7)) * 0.7, -5.0, 5.0)
def std_logerel(x): return np.clip((x - (-4.7)) * 0.7, -5.0, 5.0)
def std_deltar(x): return np.clip((x - 0.2) * 4.0, -5.0, 5.0)
def wrap_phi(x): return (x + math.pi) % (2 * math.pi) - math.pi


def pid_category(pid, charge):
    apid = np.abs(pid)
    isElectron = (apid == 11)
    isMuon = (apid == 13)
    isPhoton = (pid == 22)
    remaining = ~(isElectron | isMuon | isPhoton)
    isCHad = remaining & (charge != 0)
    isNHad = remaining & (charge == 0)
    return isCHad, isNHad, isPhoton, isElectron, isMuon


def extract_jets_from_file(url, process_name, n_events, max_jets, endpoint_timeout=30):
    """Independent re-implementation of the verified extraction (see
    FEATURE_COMPATIBILITY_MATRIX.md section 1-4): AK4 jet selection
    (pt>30, |eta|<2.5), pt-descending order, top N_JETS_PER_EVENT_CAP,
    constituents assigned via part_label, deta/dphi DERIVED from
    part_eta/part_phi minus the assigned jet's eta/phi (native
    part_deta/part_dphi are verified NOT usable -- see
    FEATURE_COMPATIBILITY_MATRIX.md section 2 -- and are not read here
    at all)."""
    with uproot.open(url, timeout=endpoint_timeout) as f:
        tree = f["tree"]
        a = tree.arrays(JET_BRANCHES + PART_BRANCHES, entry_start=0, entry_stop=n_events, library="ak")

    pt, eta, phi, jetE = a["jet_pt"], a["jet_eta"], a["jet_phi"], a["jet_energy"]
    orig_idx = ak.local_index(pt, axis=1)
    sel_mask = (pt > 30.0) & (abs(eta) < 2.5)
    sel_pt, sel_eta, sel_phi, sel_jetE = pt[sel_mask], eta[sel_mask], phi[sel_mask], jetE[sel_mask]
    sel_idx = orig_idx[sel_mask]
    order = ak.argsort(sel_pt, axis=1, ascending=False)
    sel_pt, sel_eta, sel_phi, sel_jetE, sel_idx = (x[order] for x in (sel_pt, sel_eta, sel_phi, sel_jetE, sel_idx))

    labels_all = a["part_label"]
    px_all, py_all, pz_all, e_all = a["part_px"], a["part_py"], a["part_pz"], a["part_energy"]
    peta_all, pphi_all = a["part_eta"], a["part_phi"]
    charge_all, pid_all = a["part_charge"], a["part_pid"]
    d0val_all, d0err_all = a["part_d0val"], a["part_d0err"]
    dzval_all, dzerr_all = a["part_dzval"], a["part_dzerr"]

    feats, lvs, masks, meta = [], [], [], []
    for ev in range(len(pt)):
        if len(feats) >= max_jets:
            break
        n_sel = min(len(sel_pt[ev]), N_JETS_PER_EVENT_CAP)
        if n_sel == 0:
            continue
        labels_ev = np.asarray(labels_all[ev], dtype=np.int64)
        if len(labels_ev) == 0:
            continue
        px_ev = np.asarray(px_all[ev]); py_ev = np.asarray(py_all[ev]); pz_ev = np.asarray(pz_all[ev])
        e_ev = np.asarray(e_all[ev])
        eta_ev = np.asarray(peta_all[ev]); phi_ev = np.asarray(pphi_all[ev])
        charge_ev = np.asarray(charge_all[ev], dtype=np.int64)
        pid_ev = np.asarray(pid_all[ev], dtype=np.int64)
        d0val_ev = np.asarray(d0val_all[ev]); d0err_ev = np.asarray(d0err_all[ev])
        dzval_ev = np.asarray(dzval_all[ev]); dzerr_ev = np.asarray(dzerr_all[ev])
        isCHad_ev, isNHad_ev, isPhoton_ev, isElectron_ev, isMuon_ev = pid_category(pid_ev, charge_ev)
        pt_ev = np.hypot(px_ev, py_ev)
        d0_ev = np.tanh(d0val_ev); dz_ev = np.tanh(dzval_ev)
        d0err_c_ev = np.clip(d0err_ev, 0.0, 1.0); dzerr_c_ev = np.clip(dzerr_ev, 0.0, 1.0)

        for slot in range(n_sel):
            if len(feats) >= max_jets:
                break
            oidx = int(ak.to_numpy(sel_idx[ev])[slot])
            sel = labels_ev == oidx
            n_real = int(sel.sum())
            if n_real == 0:
                continue
            n_cap = min(n_real, MAXLEN)

            jet_eta_v = float(ak.to_numpy(sel_eta[ev])[slot])
            jet_phi_v = float(ak.to_numpy(sel_phi[ev])[slot])
            jet_pt_v = float(ak.to_numpy(sel_pt[ev])[slot])
            jet_e_v = float(ak.to_numpy(sel_jetE[ev])[slot])

            pt_c = pt_ev[sel][:n_cap]; e_c = e_ev[sel][:n_cap]
            deta_c = eta_ev[sel][:n_cap] - jet_eta_v
            dphi_c = wrap_phi(phi_ev[sel][:n_cap] - jet_phi_v)
            deltar_c = std_deltar(np.hypot(deta_c, dphi_c))

            feat = np.zeros((17, MAXLEN), dtype=np.float32)
            feat[0, :n_cap] = std_pt_log(np.log(np.clip(pt_c, 1e-9, None)))
            feat[1, :n_cap] = std_e_log(np.log(np.clip(e_c, 1e-9, None)))
            feat[2, :n_cap] = std_logptrel(np.log(np.clip(pt_c, 1e-9, None) / max(jet_pt_v, 1e-9)))
            feat[3, :n_cap] = std_logerel(np.log(np.clip(e_c, 1e-9, None) / max(jet_e_v, 1e-9)))
            feat[4, :n_cap] = deltar_c
            feat[5, :n_cap] = charge_ev[sel][:n_cap]
            feat[6, :n_cap] = isCHad_ev[sel][:n_cap]
            feat[7, :n_cap] = isNHad_ev[sel][:n_cap]
            feat[8, :n_cap] = isPhoton_ev[sel][:n_cap]
            feat[9, :n_cap] = isElectron_ev[sel][:n_cap]
            feat[10, :n_cap] = isMuon_ev[sel][:n_cap]
            feat[11, :n_cap] = d0_ev[sel][:n_cap]
            feat[12, :n_cap] = d0err_c_ev[sel][:n_cap]
            feat[13, :n_cap] = dz_ev[sel][:n_cap]
            feat[14, :n_cap] = dzerr_c_ev[sel][:n_cap]
            feat[15, :n_cap] = deta_c
            feat[16, :n_cap] = dphi_c

            lv = np.zeros((4, MAXLEN), dtype=np.float32)
            lv[0, :n_cap] = px_ev[sel][:n_cap]; lv[1, :n_cap] = py_ev[sel][:n_cap]
            lv[2, :n_cap] = pz_ev[sel][:n_cap]; lv[3, :n_cap] = e_ev[sel][:n_cap]

            m = np.zeros((1, MAXLEN), dtype=np.float32)
            m[0, :n_cap] = 1.0

            feats.append(feat); lvs.append(lv); masks.append(m)
            meta.append({"process": process_name, "event": ev, "slot": slot,
                         "jet_pt": jet_pt_v, "jet_eta": jet_eta_v, "n_constituents": n_cap})

    return (np.stack(feats, axis=0), np.stack(lvs, axis=0), np.stack(masks, axis=0), meta)


def build_jpjepa_reordered(feat_natural):
    """[deta,dphi]-first reorder, exactly as JP-JEPA's own inference.py
    applies it (verified in PREPROCESSING_SPEC.md section 5)."""
    return np.concatenate([feat_natural[:, -2:, :], feat_natural[:, :-2, :]], axis=1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jpjepa-src-dir", required=True, help="path to the pinned jetparticle-jepa clone (contains particle_transformer.py)")
    ap.add_argument("--checkpoint", required=True, help="path to jpjepa_mini_pretrained.ckpt")
    ap.add_argument("--n-events-per-file", type=int, default=150)
    ap.add_argument("--max-jets-per-process", type=int, default=350)
    ap.add_argument("--out-json", required=True)
    ap.add_argument("--endpoint", default="root://cceos.ihep.ac.cn/")
    args = ap.parse_args()

    sys.path.insert(0, args.jpjepa_src_dir)
    import torch  # noqa: E402
    from particle_transformer import part_mini  # noqa: E402

    files = {
        "qcd": args.endpoint.rstrip("/") + "//eos/ihep/cms/store/user/coli/datasets/hh4b/training/QCD_DelphesHH4JTrig_merged_ntuple/QCD_DelphesHH4JTrig_ntuple_mergeid166.root",
        "signal_hh4b": args.endpoint.rstrip("/") + "//eos/ihep/cms/store/user/coli/datasets/hh4b/training/HH4b_2HDM_H3VAR_H1H2_40to200_merged_ntuple/HH4b_2HDM_H3VAR_H1H2_40to200_ntuple_id0-9.root",
        "ttbar": args.endpoint.rstrip("/") + "//eos/ihep/cms/store/user/coli/datasets/hh4b/training/TTbar_ntuple/selected_ntuples_0000.root",
    }

    all_feat, all_lv, all_mask, all_meta = [], [], [], []
    t_extract0 = time.time()
    for proc, url in files.items():
        feat, lv, mask, meta = extract_jets_from_file(url, proc, args.n_events_per_file, args.max_jets_per_process, endpoint_timeout=30)
        print(f"[extract] {proc}: {feat.shape[0]} jets")
        all_feat.append(feat); all_lv.append(lv); all_mask.append(mask); all_meta.extend(meta)
    feat_natural = np.concatenate(all_feat, axis=0)
    lv = np.concatenate(all_lv, axis=0)
    mask = np.concatenate(all_mask, axis=0)
    extract_time = time.time() - t_extract0

    feat_jpjepa = build_jpjepa_reordered(feat_natural)

    model = part_mini(pretrained_weights=args.checkpoint, num_classes=None)
    model.eval()

    def run(feat_in, lv_in, mask_in):
        with torch.no_grad():
            f = torch.from_numpy(feat_in)
            v = torch.from_numpy(lv_in)
            m = torch.from_numpy(mask_in)
            _, outs = model.forward(f, v, m)
            return outs[-1][0][0].numpy()

    t0 = time.time()
    emb1 = run(feat_jpjepa, lv, mask)
    infer_time = time.time() - t0
    emb2 = run(feat_jpjepa, lv, mask)  # determinism check

    # padding/mask sensitivity: corrupt the padded region's feature values (keep mask=0 there) and re-run
    feat_corrupted = feat_jpjepa.copy()
    pad_positions = (mask[:, 0, :] == 0.0)
    rng = np.random.default_rng(0)
    for i in range(feat_corrupted.shape[0]):
        n_pad = int(pad_positions[i].sum())
        if n_pad > 0:
            feat_corrupted[i, :, pad_positions[i]] = (rng.normal(size=(n_pad, feat_corrupted.shape[1])).astype(np.float32) * 5.0)
    lv_corrupted = lv.copy()
    for i in range(lv_corrupted.shape[0]):
        n_pad = int(pad_positions[i].sum())
        if n_pad > 0:
            lv_corrupted[i, :, pad_positions[i]] = (rng.normal(size=(n_pad, lv_corrupted.shape[1])).astype(np.float32) * 50.0)
    emb_padcorrupt = run(feat_corrupted, lv_corrupted, mask)

    n = emb1.shape[0]
    dim = emb1.shape[1]
    norms = np.linalg.norm(emb1, axis=1)
    per_dim_mean = emb1.mean(axis=0)
    per_dim_std = emb1.std(axis=0)
    near_constant_thresh = 1e-4
    n_near_constant = int(np.sum(per_dim_std < near_constant_thresh))

    jet_pt_arr = np.array([m["jet_pt"] for m in all_meta])
    jet_eta_arr = np.array([m["jet_eta"] for m in all_meta])
    n_const_arr = np.array([m["n_constituents"] for m in all_meta])
    proc_arr = np.array([m["process"] for m in all_meta])

    def corr(x, y):
        if np.std(x) < 1e-12 or np.std(y) < 1e-12:
            return None
        return float(np.corrcoef(x, y)[0, 1])

    per_process_stats = {}
    for proc in files.keys():
        m = proc_arr == proc
        if m.sum() == 0:
            continue
        per_process_stats[proc] = {
            "n_jets": int(m.sum()),
            "mean_norm": float(norms[m].mean()),
            "std_norm": float(norms[m].std()),
            "mean_embedding_l2_from_global_mean": float(np.linalg.norm(emb1[m].mean(axis=0) - emb1.mean(axis=0))),
        }

    report = {
        "n_jets_total": int(n),
        "embedding_dim": int(dim),
        "extract_time_s": extract_time,
        "infer_time_s": infer_time,
        "all_finite": bool(np.isfinite(emb1).all()),
        "norm_stats": {"mean": float(norms.mean()), "std": float(norms.std()),
                        "min": float(norms.min()), "max": float(norms.max())},
        "per_dim_mean_stats": {"mean": float(per_dim_mean.mean()), "std_of_means": float(per_dim_mean.std())},
        "per_dim_std_stats": {"mean": float(per_dim_std.mean()), "min": float(per_dim_std.min()), "max": float(per_dim_std.max())},
        "n_near_constant_dims_std_lt_1e-4": n_near_constant,
        "near_constant_dim_indices": np.where(per_dim_std < near_constant_thresh)[0].tolist(),
        "determinism_check": {
            "max_abs_diff_repeat_run": float(np.max(np.abs(emb1 - emb2))),
            "bitwise_identical": bool(np.array_equal(emb1, emb2)),
        },
        "padding_mask_sensitivity_check": {
            "description": "corrupted the zero-padded (mask=0) feature and lorentz-vector slots with large random noise, re-ran; if masking is correctly implemented end-to-end, output must be unchanged",
            "max_abs_diff_vs_uncorrupted": float(np.max(np.abs(emb1 - emb_padcorrupt))),
            "mask_correctly_ignored_padding": bool(np.allclose(emb1, emb_padcorrupt, atol=1e-4)),
        },
        "dependence_on_kinematics": {
            "corr_norm_vs_jet_pt": corr(norms, jet_pt_arr),
            "corr_norm_vs_jet_eta": corr(norms, jet_eta_arr),
            "corr_norm_vs_n_constituents": corr(norms, n_const_arr),
        },
        "per_process_stats": per_process_stats,
        "jet_pt_stats": {"min": float(jet_pt_arr.min()), "max": float(jet_pt_arr.max()), "mean": float(jet_pt_arr.mean())},
        "n_constituents_stats": {"min": int(n_const_arr.min()), "max": int(n_const_arr.max()), "mean": float(n_const_arr.mean())},
        "torch_version": torch.__version__,
        "checkpoint_path": args.checkpoint,
    }

    with open(args.out_json, "w") as fh:
        json.dump(report, fh, indent=2)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
