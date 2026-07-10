#!/usr/bin/env python3

import argparse
import os
from pathlib import Path

import awkward as ak
import numpy as np
import pandas as pd
import uproot


def delta_phi(a, b):
    return (a - b + np.pi) % (2 * np.pi) - np.pi


def delta_r(eta1, phi1, eta2, phi2):
    return float(np.hypot(eta1 - eta2, delta_phi(phi1, phi2)))


def find_branch(tree, options):
    keys = set(tree.keys())
    for opt in options:
        if opt in keys:
            return opt
    # fallback: match by final component
    final_opts = {o.split("/")[-1].lower(): o for o in options}
    for k in tree.keys():
        kk = k.split("/")[-1].lower()
        if kk in final_opts:
            return k
    raise KeyError(f"Could not find any of {options}")


def get_branches(tree):
    return {
        "jet_pt": find_branch(tree, ["Jet.PT", "Jet/Jet.PT"]),
        "jet_eta": find_branch(tree, ["Jet.Eta", "Jet/Jet.Eta"]),
        "jet_phi": find_branch(tree, ["Jet.Phi", "Jet/Jet.Phi"]),
        "jet_mass": find_branch(tree, ["Jet.Mass", "Jet/Jet.Mass"]),
        "jet_btag": find_branch(tree, ["Jet.BTag", "Jet/Jet.BTag"]),
        "particle_pid": find_branch(tree, ["Particle.PID", "Particle/Particle.PID"]),
        "particle_status": find_branch(tree, ["Particle.Status", "Particle/Particle.Status"]),
        "particle_m1": find_branch(tree, ["Particle.M1", "Particle/Particle.M1"]),
        "particle_m2": find_branch(tree, ["Particle.M2", "Particle/Particle.M2"]),
        "particle_d1": find_branch(tree, ["Particle.D1", "Particle/Particle.D1"]),
        "particle_d2": find_branch(tree, ["Particle.D2", "Particle/Particle.D2"]),
        "particle_pt": find_branch(tree, ["Particle.PT", "Particle/Particle.PT"]),
        "particle_eta": find_branch(tree, ["Particle.Eta", "Particle/Particle.Eta"]),
        "particle_phi": find_branch(tree, ["Particle.Phi", "Particle/Particle.Phi"]),
    }


def direct_or_recursive_b_daughters(pid, d1, d2, higgs_idx, max_depth=8):
    n = len(pid)

    def valid_child_range(i):
        if i < 0 or i >= n:
            return []
        a = int(d1[i])
        b = int(d2[i])
        if a < 0 or b < 0 or a >= n:
            return []
        b = min(b, n - 1)
        return list(range(a, b + 1))

    direct = [j for j in valid_child_range(higgs_idx) if abs(int(pid[j])) == 5]
    if len(direct) >= 2:
        return direct[:2]

    found = []
    seen = set()

    def walk(i, depth):
        if depth > max_depth or i in seen:
            return
        seen.add(i)
        for j in valid_child_range(i):
            if abs(int(pid[j])) == 5:
                found.append(j)
            else:
                walk(j, depth + 1)

    walk(higgs_idx, 0)

    unique = []
    for j in found:
        if j not in unique:
            unique.append(j)
    return unique[:2]


def find_higgs_to_bb_truth(pid, d1, d2, pt):
    higgs_indices = [i for i, p in enumerate(pid) if int(p) == 25]
    candidates = []

    for h in higgs_indices:
        bs = direct_or_recursive_b_daughters(pid, d1, d2, h)
        if len(bs) >= 2:
            score = float(pt[h]) if h < len(pt) else 0.0
            candidates.append((score, h, bs[:2]))

    # Prefer two highest-pT Higgs candidates with bb daughters.
    candidates = sorted(candidates, key=lambda x: x[0], reverse=True)
    if len(candidates) < 2:
        return None

    # Avoid duplicate identical b sets.
    chosen = []
    used_b_sets = set()
    for score, h, bs in candidates:
        key = tuple(sorted(bs))
        if key not in used_b_sets:
            chosen.append((h, bs))
            used_b_sets.add(key)
        if len(chosen) == 2:
            break

    if len(chosen) < 2:
        return None

    return chosen


def match_truth_to_selected_jets(truth, selected_global_indices, jet_eta, jet_phi, p_eta, p_phi, dr_max):
    # Return assignment[2,2] in local selected-jet indices, or -1.
    assignment = np.full((2, 2), -1, dtype=np.int64)

    pairs = []
    for h_idx, (_, b_indices) in enumerate(truth):
        for bslot, bidx in enumerate(b_indices[:2]):
            for local_j, global_j in enumerate(selected_global_indices):
                dr = delta_r(float(p_eta[bidx]), float(p_phi[bidx]), float(jet_eta[global_j]), float(jet_phi[global_j]))
                if dr < dr_max:
                    pairs.append((dr, h_idx, bslot, local_j))

    pairs = sorted(pairs, key=lambda x: x[0])

    used_truth = set()
    used_jets = set()

    for dr, h_idx, bslot, local_j in pairs:
        truth_key = (h_idx, bslot)
        if truth_key in used_truth or local_j in used_jets:
            continue
        assignment[h_idx, bslot] = local_j
        used_truth.add(truth_key)
        used_jets.add(local_j)

    return assignment


def sample_configs(store):
    return [
        {
            "sample": "ggF_HH4b_SMnorm",
            "group": "signal",
            "xsec_pb": 0.01055,
            "n_generated": 10000,
            "patterns": ["HH4b_ggf_hh4b_10k_transfer_cluster_84730223_shard_*_pythia8_delphes.root"],
        },
        {
            "sample": "VBF_HH4b_SMnorm",
            "group": "signal",
            "xsec_pb": 0.000587,
            "n_generated": 10000,
            "patterns": ["HH4b_vbf_hh4b_10k_transfer_cluster_3473183_shard_*_pythia8_delphes.root"],
        },
        {
            "sample": "ttbar_200k",
            "group": "background",
            "xsec_pb": 512.2746276855469,
            "n_generated": 200000,
            "patterns": ["ttbar_200k_shard*_pythia8_delphes.root"],
        },
        {
            "sample": "Zbbbb_100k",
            "group": "background",
            "xsec_pb": 6.912992,
            "n_generated": 100000,
            "patterns": ["zbbbb_presel_100k_shard*_pythia8_delphes.root"],
        },
        {
            "sample": "qcd_bbbb_iht100to200_20000",
            "group": "background",
            "xsec_pb": 206.2399139404297,
            "n_generated": 20000,
            "patterns": ["qcd_bbbb_iht100to200_20000*delphes.root"],
        },
        {
            "sample": "qcd_bbbb_iht200to400_combined120k",
            "group": "background",
            "xsec_pb": 126.35226440429688,
            "n_generated": 120000,
            "patterns": [
                "qcd_bbbb_iht200to400_20000*delphes.root",
                "qcd_bbbb_iht200to400_extra100k*delphes.root",
            ],
        },
        {
            "sample": "qcd_bbbb_iht400to600_20000",
            "group": "background",
            "xsec_pb": 9.774867057800293,
            "n_generated": 20000,
            "patterns": ["qcd_bbbb_iht400to600_20000*delphes.root"],
        },
        {
            "sample": "qcd_bbbb_iht600plus_20000",
            "group": "background",
            "xsec_pb": 1.7876876592636108,
            "n_generated": 20000,
            "patterns": ["qcd_bbbb_iht600plus_20000*delphes.root"],
        },
    ]


def collect_files(root_dir, patterns):
    files = []
    for pat in patterns:
        files.extend(sorted(root_dir.glob(pat)))
    # de-duplicate while preserving order
    seen = set()
    out = []
    for f in files:
        if f not in seen:
            out.append(f)
            seen.add(f)
    return out


def build(args):
    repo = Path(os.environ["HH4B_REPO"])
    store = Path(os.environ["HH4B_STORE"])
    root_dir = store / "root"

    outdir = store / "spanet_npz" / args.tag
    outdir.mkdir(parents=True, exist_ok=True)

    table_dir = repo / "outputs/tables" / f"hh4b_spanet_dataset_{args.tag}"
    table_dir.mkdir(parents=True, exist_ok=True)

    sample_rows = []
    event_rows = []

    X_list = []
    mask_list = []
    y_list = []
    assignment_list = []
    assignment_mask_list = []
    weight_list = []
    sample_id_list = []
    event_id_list = []
    source_root_list = []
    n_selected_list = []
    n_btag_list = []

    configs = sample_configs(store)
    sample_to_id = {cfg["sample"]: i for i, cfg in enumerate(configs)}

    total_kept = 0

    for cfg in configs:
        sample = cfg["sample"]
        group = cfg["group"]
        is_signal = group == "signal"
        y = 1 if is_signal else 0
        weight_pb = float(cfg["xsec_pb"]) / float(cfg["n_generated"])

        files = collect_files(root_dir, cfg["patterns"])
        if args.max_files_per_sample is not None:
            files = files[:args.max_files_per_sample]

        print(f"\n=== {sample}: {len(files)} ROOT files ===")

        sample_seen = 0
        sample_kept = 0
        sample_truth_complete = 0
        sample_truth_available = 0

        for fidx, path in enumerate(files):
            if args.max_events_per_sample is not None and sample_seen >= args.max_events_per_sample:
                break

            print("Reading", path.name)

            with uproot.open(path) as f:
                tree = f["Delphes"]
                branches = get_branches(tree)

                remaining = None
                if args.max_events_per_sample is not None:
                    remaining = max(args.max_events_per_sample - sample_seen, 0)
                    if remaining <= 0:
                        break

                stop = remaining if remaining is not None else None
                names = list(branches.values())

                arr = tree.arrays(names, entry_stop=stop, library="ak")

            n_events = len(arr[branches["jet_pt"]])
            sample_seen += n_events

            for iev in range(n_events):
                jet_pt = np.asarray(arr[branches["jet_pt"]][iev], dtype=float)
                jet_eta = np.asarray(arr[branches["jet_eta"]][iev], dtype=float)
                jet_phi = np.asarray(arr[branches["jet_phi"]][iev], dtype=float)
                jet_mass = np.asarray(arr[branches["jet_mass"]][iev], dtype=float)
                jet_btag = np.asarray(arr[branches["jet_btag"]][iev], dtype=float)

                if len(jet_pt) == 0:
                    continue

                selected = np.where((jet_pt > args.jet_pt_min) & (np.abs(jet_eta) < args.jet_eta_max))[0]
                if len(selected) < args.min_jets:
                    continue

                selected = selected[np.argsort(jet_pt[selected])[::-1]]
                selected = selected[:args.max_jets]

                n_sel = len(selected)
                n_btag = int(np.sum(jet_btag[selected] > args.btag_min))

                X = np.zeros((args.max_jets, 5), dtype=np.float32)
                jet_mask = np.zeros(args.max_jets, dtype=np.bool_)

                for local_j, global_j in enumerate(selected):
                    X[local_j, 0] = jet_pt[global_j]
                    X[local_j, 1] = jet_eta[global_j]
                    X[local_j, 2] = jet_phi[global_j]
                    X[local_j, 3] = jet_mass[global_j]
                    X[local_j, 4] = jet_btag[global_j]
                    jet_mask[local_j] = True

                assignment = np.full((2, 2), -1, dtype=np.int64)
                assignment_mask = 0
                truth_available = 0

                if is_signal:
                    pid = np.asarray(arr[branches["particle_pid"]][iev], dtype=int)
                    d1 = np.asarray(arr[branches["particle_d1"]][iev], dtype=int)
                    d2 = np.asarray(arr[branches["particle_d2"]][iev], dtype=int)
                    p_pt = np.asarray(arr[branches["particle_pt"]][iev], dtype=float)
                    p_eta = np.asarray(arr[branches["particle_eta"]][iev], dtype=float)
                    p_phi = np.asarray(arr[branches["particle_phi"]][iev], dtype=float)

                    truth = find_higgs_to_bb_truth(pid, d1, d2, p_pt)
                    if truth is not None:
                        truth_available = 1
                        sample_truth_available += 1
                        assignment = match_truth_to_selected_jets(
                            truth,
                            selected,
                            jet_eta,
                            jet_phi,
                            p_eta,
                            p_phi,
                            args.dr_match,
                        )
                        complete = np.all(assignment >= 0) and len(set(assignment.reshape(-1).tolist())) == 4
                        if complete:
                            assignment_mask = 1
                            sample_truth_complete += 1

                X_list.append(X)
                mask_list.append(jet_mask)
                y_list.append(y)
                assignment_list.append(assignment)
                assignment_mask_list.append(assignment_mask)
                weight_list.append(weight_pb)
                sample_id_list.append(sample_to_id[sample])
                event_id_list.append(iev)
                source_root_list.append(path.name)
                n_selected_list.append(n_sel)
                n_btag_list.append(n_btag)

                event_rows.append({
                    "sample": sample,
                    "group": group,
                    "source_root": path.name,
                    "event_in_file": iev,
                    "n_selected_jets": n_sel,
                    "n_selected_btags": n_btag,
                    "is_signal": y,
                    "assignment_mask": assignment_mask,
                    "truth_available": truth_available,
                    "weight_pb": weight_pb,
                })

                sample_kept += 1
                total_kept += 1

        sample_rows.append({
            "sample": sample,
            "group": group,
            "root_files": len(files),
            "events_seen": sample_seen,
            "events_kept": sample_kept,
            "keep_efficiency": sample_kept / sample_seen if sample_seen else 0.0,
            "truth_available_events": sample_truth_available,
            "truth_complete_assignment_events": sample_truth_complete,
            "truth_complete_fraction_of_kept": sample_truth_complete / sample_kept if sample_kept else 0.0,
            "xsec_pb": cfg["xsec_pb"],
            "n_generated_for_weight": cfg["n_generated"],
            "weight_pb": weight_pb,
        })

    if total_kept == 0:
        raise RuntimeError("No events kept.")

    X = np.stack(X_list)
    jet_mask = np.stack(mask_list)
    y = np.asarray(y_list, dtype=np.int64)
    assignment = np.stack(assignment_list)
    assignment_mask = np.asarray(assignment_mask_list, dtype=np.int64)
    weight_pb = np.asarray(weight_list, dtype=np.float64)
    sample_id = np.asarray(sample_id_list, dtype=np.int64)
    event_id = np.asarray(event_id_list, dtype=np.int64)
    n_selected = np.asarray(n_selected_list, dtype=np.int64)
    n_btag = np.asarray(n_btag_list, dtype=np.int64)
    source_root = np.asarray(source_root_list)

    rng = np.random.default_rng(args.seed)
    split = np.full(len(y), 0, dtype=np.int64)  # 0 train, 1 val, 2 test
    for sid in np.unique(sample_id):
        idx = np.where(sample_id == sid)[0]
        rng.shuffle(idx)
        n = len(idx)
        n_train = int(args.train_frac * n)
        n_val = int(args.val_frac * n)
        split[idx[:n_train]] = 0
        split[idx[n_train:n_train + n_val]] = 1
        split[idx[n_train + n_val:]] = 2

    feature_names = np.asarray(["pt", "eta", "phi", "mass", "btag"])
    sample_names = np.asarray([cfg["sample"] for cfg in configs])
    sample_groups = np.asarray([cfg["group"] for cfg in configs])

    for split_name, split_id in [("train", 0), ("val", 1), ("test", 2), ("all", -1)]:
        if split_id >= 0:
            idx = np.where(split == split_id)[0]
        else:
            idx = np.arange(len(y))

        out = outdir / f"hh4b_spanet_leading{args.max_jets}_{split_name}.npz"
        np.savez_compressed(
            out,
            X_jets=X[idx],
            jet_mask=jet_mask[idx],
            y=y[idx],
            assignment=assignment[idx],
            assignment_mask=assignment_mask[idx],
            weight_pb=weight_pb[idx],
            sample_id=sample_id[idx],
            event_id=event_id[idx],
            n_selected_jets=n_selected[idx],
            n_selected_btags=n_btag[idx],
            feature_names=feature_names,
            sample_names=sample_names,
            sample_groups=sample_groups,
        )
        print("Wrote", out, "events=", len(idx))

    sample_summary = pd.DataFrame(sample_rows)
    event_summary = pd.DataFrame(event_rows)

    sample_summary.to_csv(table_dir / "spanet_dataset_sample_summary.csv", index=False)
    event_summary.to_csv(table_dir / "spanet_dataset_event_summary.csv", index=False)
    (table_dir / "spanet_dataset_sample_summary.md").write_text(sample_summary.to_markdown(index=False) + "\n")

    readme = f"""# HH4b SPA-Net leading-{args.max_jets} dataset

Dataset tag: `{args.tag}`

Inputs:
- X_jets: shape (events, {args.max_jets}, 5), features = pt, eta, phi, mass, btag
- jet_mask: real/padded jet mask
- y: signal/background label
- assignment: shape (events, 2, 2), jet indices for H1/H2 daughters, -1 if missing
- assignment_mask: 1 if all four H→bb truth daughters are matched to selected jets
- weight_pb: event physics weight = xsec / generated events

Selection:
- jets with pt > {args.jet_pt_min} GeV and |eta| < {args.jet_eta_max}
- at least {args.min_jets} selected jets
- keep up to leading {args.max_jets} selected jets by pt
- truth matching ΔR < {args.dr_match}

Notes:
- Jet flavor is not included as an input feature.
- Background events have assignment labels set to -1 and assignment_mask = 0.
- Signal events without complete truth assignment are kept for classification but masked for assignment loss.
"""
    (table_dir / "README.md").write_text(readme)

    print("\n=== Sample summary ===")
    print(sample_summary.to_string(index=False))

    print("\nOutput directory:", outdir)
    print("Table directory:", table_dir)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="nominal_v0")
    ap.add_argument("--max-jets", type=int, default=8)
    ap.add_argument("--min-jets", type=int, default=4)
    ap.add_argument("--jet-pt-min", type=float, default=30.0)
    ap.add_argument("--jet-eta-max", type=float, default=2.5)
    ap.add_argument("--btag-min", type=float, default=0.0)
    ap.add_argument("--dr-match", type=float, default=0.4)
    ap.add_argument("--max-files-per-sample", type=int, default=None)
    ap.add_argument("--max-events-per-sample", type=int, default=None)
    ap.add_argument("--seed", type=int, default=12345)
    ap.add_argument("--train-frac", type=float, default=0.70)
    ap.add_argument("--val-frac", type=float, default=0.15)
    args = ap.parse_args()

    build(args)


if __name__ == "__main__":
    main()
