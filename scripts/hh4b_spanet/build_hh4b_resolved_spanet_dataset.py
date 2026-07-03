#!/usr/bin/env python3
"""
Build a resolved HH->4b SPA-Net-style assignment dataset.

This script uses only signal HH_4b parquet files. It selects AK4 reco jets,
finds two truth H->bb decays from GenPart, matches the four truth b quarks to
selected reco jets, and writes:

  - intermediate parquet with event-level information
  - arrays npz with X, mask, targets, split, event_id
  - summary.csv
  - event_index.csv
  - feature_names.json
  - RESULTS_HH4B_RESOLVED_SPANET_DATASET.md

Important interpretation:
  If the number of fully matchable events is small, do not train SPA-Net yet.
  First debug truth matching / GenPart tracing.
"""

from __future__ import annotations

import argparse
import glob
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


FEATURE_NAMES = [
    "log_pt",
    "eta",
    "sin_phi",
    "cos_phi",
    "log_mass",
    "btag",
    "btagphys",
]


def as_array(x: Any, dtype=float) -> np.ndarray:
    if x is None:
        return np.asarray([], dtype=dtype)
    if isinstance(x, np.ndarray):
        return x.astype(dtype, copy=False)
    try:
        return np.asarray(list(x), dtype=dtype)
    except TypeError:
        return np.asarray([], dtype=dtype)


def delta_phi(phi1: float, phi2: float) -> float:
    dphi = phi1 - phi2
    while dphi > math.pi:
        dphi -= 2.0 * math.pi
    while dphi <= -math.pi:
        dphi += 2.0 * math.pi
    return dphi


def delta_r(eta1: float, phi1: float, eta2: float, phi2: float) -> float:
    return math.sqrt((eta1 - eta2) ** 2 + delta_phi(phi1, phi2) ** 2)


def descendants_of(idx: int, d1: np.ndarray, d2: np.ndarray, n: int) -> set[int]:
    """Return recursive descendants using Delphes-style D1/D2 daughter ranges."""
    seen: set[int] = set()
    stack = [idx]

    while stack:
        i = stack.pop()
        if i < 0 or i >= n:
            continue

        a = int(d1[i])
        b = int(d2[i])

        if a < 0 and b < 0:
            continue

        if a >= 0 and b >= a:
            children = range(a, min(b + 1, n))
        elif a >= 0:
            children = [a]
        elif b >= 0:
            children = [b]
        else:
            children = []

        for c in children:
            if c not in seen and c != idx:
                seen.add(c)
                stack.append(c)

    return seen


def find_higgs_bb_candidates(
    pid: np.ndarray,
    d1: np.ndarray,
    d2: np.ndarray,
) -> list[list[int]]:
    """
    Find Higgs candidates with descendant b quarks.

    Returns a list of candidates. Each candidate is a list of GenPart indices
    for b/bbar descendants. The caller later chooses two candidates with at
    least two b quarks each.
    """
    n = len(pid)
    higgs_indices = [int(i) for i in np.where(pid == 25)[0]]

    candidates: list[list[int]] = []
    for hidx in higgs_indices:
        desc = descendants_of(hidx, d1, d2, n)
        b_desc = sorted([i for i in desc if abs(int(pid[i])) == 5])
        if len(b_desc) >= 2:
            candidates.append(b_desc)

    # Remove exact duplicates caused by Higgs copies.
    unique: list[list[int]] = []
    seen: set[tuple[int, ...]] = set()
    for cand in candidates:
        key = tuple(cand)
        if key not in seen:
            seen.add(key)
            unique.append(cand)

    return unique


def choose_two_higgs_b_pairs(
    pid: np.ndarray,
    pt: np.ndarray,
    eta: np.ndarray,
    phi: np.ndarray,
    d1: np.ndarray,
    d2: np.ndarray,
) -> list[tuple[int, int]]:
    """
    Choose two truth H->bb pairs.

    Conservative version:
      - find Higgs candidates by descendant tracing
      - require at least two candidates
      - for each candidate, take two highest-pt b descendants
      - require the final four b indices to be unique

    This is intentionally strict; if it gives too few events, run the GenPart
    diagnostic and improve the truth tracing.
    """
    candidates = find_higgs_bb_candidates(pid, d1, d2)

    if len(candidates) < 2:
        return []

    # Prefer candidates with high total b pt.
    scored = []
    for cand in candidates:
        bs = sorted(cand, key=lambda i: float(pt[i]) if i < len(pt) else 0.0, reverse=True)
        if len(bs) >= 2:
            score = sum(float(pt[i]) for i in bs[:2])
            scored.append((score, bs[:2]))

    scored.sort(reverse=True, key=lambda x: x[0])

    pairs: list[tuple[int, int]] = []
    used: set[int] = set()
    for _, pair in scored:
        if pair[0] in used or pair[1] in used:
            continue
        pairs.append((int(pair[0]), int(pair[1])))
        used.update(pair)
        if len(pairs) == 2:
            break

    if len(pairs) != 2:
        return []

    if len(set([pairs[0][0], pairs[0][1], pairs[1][0], pairs[1][1]])) != 4:
        return []

    return pairs


def select_reco_jets(row: pd.Series, max_jets: int, min_pt: float, max_abs_eta: float) -> list[dict[str, float]]:
    pt = as_array(row.get("FullReco_JetAK4_PT"), float)
    eta = as_array(row.get("FullReco_JetAK4_Eta"), float)
    phi = as_array(row.get("FullReco_JetAK4_Phi"), float)
    mass = as_array(row.get("FullReco_JetAK4_Mass"), float)
    btag = as_array(row.get("FullReco_JetAK4_BTag"), float)
    btagphys = as_array(row.get("FullReco_JetAK4_BTagPhys"), float)
    charge = as_array(row.get("FullReco_JetAK4_Charge"), float)

    n = min(len(pt), len(eta), len(phi), len(mass))
    jets = []
    for i in range(n):
        if not np.isfinite(pt[i]) or not np.isfinite(eta[i]) or not np.isfinite(phi[i]):
            continue
        if pt[i] < min_pt:
            continue
        if abs(eta[i]) > max_abs_eta:
            continue

        jets.append(
            {
                "orig_index": int(i),
                "pt": float(pt[i]),
                "eta": float(eta[i]),
                "phi": float(phi[i]),
                "mass": float(mass[i]),
                "btag": float(btag[i]) if i < len(btag) else 0.0,
                "btagphys": float(btagphys[i]) if i < len(btagphys) else 0.0,
                "charge": float(charge[i]) if i < len(charge) else 0.0,
            }
        )

    # Rank by btag, then btagphys, then pt.
    jets.sort(key=lambda j: (j["btag"], j["btagphys"], j["pt"]), reverse=True)
    return jets[:max_jets]


def match_truth_bs_to_jets(
    truth_pairs: list[tuple[int, int]],
    gen_eta: np.ndarray,
    gen_phi: np.ndarray,
    jets: list[dict[str, float]],
    match_dr: float,
) -> tuple[list[int], list[float]]:
    targets = []
    drs = []

    for pair in truth_pairs:
        for bidx in pair:
            best_j = -1
            best_dr = 999.0
            for jidx, jet in enumerate(jets):
                dr = delta_r(float(gen_eta[bidx]), float(gen_phi[bidx]), jet["eta"], jet["phi"])
                if dr < best_dr:
                    best_dr = dr
                    best_j = jidx

            if best_dr < match_dr:
                targets.append(best_j)
                drs.append(best_dr)
            else:
                targets.append(-1)
                drs.append(best_dr)

    return targets, drs


def make_feature_array(jets: list[dict[str, float]], max_jets: int) -> tuple[np.ndarray, np.ndarray]:
    X = np.zeros((max_jets, len(FEATURE_NAMES)), dtype=np.float32)
    mask = np.zeros((max_jets,), dtype=np.bool_)

    for i, jet in enumerate(jets[:max_jets]):
        X[i, :] = np.asarray(
            [
                math.log(max(jet["pt"], 1e-6)),
                jet["eta"],
                math.sin(jet["phi"]),
                math.cos(jet["phi"]),
                math.log(max(jet["mass"], 1e-6)),
                jet["btag"],
                jet["btagphys"],
            ],
            dtype=np.float32,
        )
        mask[i] = True

    return X, mask


def split_indices(n: int, seed: int, train_frac: float, val_frac: float) -> np.ndarray:
    rng = np.random.default_rng(seed)
    order = rng.permutation(n)
    split = np.empty(n, dtype=object)

    n_train = int(round(train_frac * n))
    n_val = int(round(val_frac * n))

    split[order[:n_train]] = "train"
    split[order[n_train : n_train + n_val]] = "val"
    split[order[n_train + n_val :]] = "test"
    return split


def write_summary(outdir: Path, metrics: dict[str, Any]) -> None:
    rows = [{"metric": k, "value": v} for k, v in metrics.items()]
    pd.DataFrame(rows).to_csv(outdir / "summary.csv", index=False)

    lines = [
        "# Resolved HH4b SPA-Net dataset summary",
        "",
        "## Purpose",
        "",
        "This dataset builder creates a resolved HH→bbbb assignment dataset using AK4 jets.",
        "The target is two H→bb pairs, represented as four jet indices:",
        "",
        "`[H1_b1, H1_b2, H2_b1, H2_b2]`",
        "",
        "## Summary metrics",
        "",
        "| Metric | Value |",
        "|---|---:|",
    ]
    for k, v in metrics.items():
        lines.append(f"| {k} | {v} |")

    lines += [
        "",
        "## Interpretation",
        "",
        "If the number of fully matchable events is small, do not train SPA-Net yet.",
        "The next step should be to debug the GenPart truth definition and matching efficiency.",
        "",
    ]

    (outdir / "RESULTS_HH4B_RESOLVED_SPANET_DATASET.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-glob", default="outputs/collide_selected_backgrounds/HH_4b/*.parquet")
    parser.add_argument("--outdir", default="outputs/hh4b_spanet/resolved_v1")
    parser.add_argument("--max-jets", type=int, default=10)
    parser.add_argument("--min-jet-pt", type=float, default=20.0)
    parser.add_argument("--max-abs-eta", type=float, default=2.5)
    parser.add_argument("--match-dr", type=float, default=0.4)
    parser.add_argument("--seed", type=int, default=12345)
    parser.add_argument("--train-frac", type=float, default=0.70)
    parser.add_argument("--val-frac", type=float, default=0.15)
    parser.add_argument("--max-files", type=int, default=None)
    parser.add_argument("--max-events", type=int, default=None)
    args = parser.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    files = sorted(glob.glob(args.input_glob))
    if args.max_files is not None:
        files = files[: args.max_files]

    if not files:
        raise FileNotFoundError(
            f"No input files matched {args.input_glob}. "
            "This is expected in the temp pod if the COLLIDE data is not mounted."
        )

    columns = [
        "FullReco_JetAK4_PT",
        "FullReco_JetAK4_Eta",
        "FullReco_JetAK4_Phi",
        "FullReco_JetAK4_Mass",
        "FullReco_JetAK4_BTag",
        "FullReco_JetAK4_BTagPhys",
        "FullReco_JetAK4_Charge",
        "FullReco_GenPart_PT",
        "FullReco_GenPart_Eta",
        "FullReco_GenPart_Phi",
        "FullReco_GenPart_PID",
        "FullReco_GenPart_M1",
        "FullReco_GenPart_M2",
        "FullReco_GenPart_D1",
        "FullReco_GenPart_D2",
        "FullReco_GenPart_Status",
    ]

    X_list = []
    mask_list = []
    target_list = []
    event_rows = []
    intermediate_rows = []

    total_events = 0
    events_with_two_higgs_bb = 0
    events_with_at_least_4_selected_jets = 0
    fully_matchable = 0

    global_event_id = 0

    for path in files:
        df = pd.read_parquet(path, columns=columns)

        for local_idx, row in df.iterrows():
            if args.max_events is not None and total_events >= args.max_events:
                break

            total_events += 1

            gen_pid = as_array(row.get("FullReco_GenPart_PID"), int)
            gen_pt = as_array(row.get("FullReco_GenPart_PT"), float)
            gen_eta = as_array(row.get("FullReco_GenPart_Eta"), float)
            gen_phi = as_array(row.get("FullReco_GenPart_Phi"), float)
            gen_d1 = as_array(row.get("FullReco_GenPart_D1"), int)
            gen_d2 = as_array(row.get("FullReco_GenPart_D2"), int)

            truth_pairs = choose_two_higgs_b_pairs(gen_pid, gen_pt, gen_eta, gen_phi, gen_d1, gen_d2)
            has_two_higgs_bb = len(truth_pairs) == 2
            if has_two_higgs_bb:
                events_with_two_higgs_bb += 1

            jets = select_reco_jets(row, args.max_jets, args.min_jet_pt, args.max_abs_eta)
            if len(jets) >= 4:
                events_with_at_least_4_selected_jets += 1

            targets = [-1, -1, -1, -1]
            drs = [999.0, 999.0, 999.0, 999.0]
            is_fully_matchable = False

            if has_two_higgs_bb and len(jets) >= 4:
                targets, drs = match_truth_bs_to_jets(truth_pairs, gen_eta, gen_phi, jets, args.match_dr)

                # Require four matched targets and no duplicate reco jet.
                if all(t >= 0 for t in targets) and len(set(targets)) == 4:
                    is_fully_matchable = True

            if is_fully_matchable:
                fully_matchable += 1
                X, mask = make_feature_array(jets, args.max_jets)

                X_list.append(X)
                mask_list.append(mask)
                target_list.append(np.asarray(targets, dtype=np.int64))

                event_rows.append(
                    {
                        "event_id": global_event_id,
                        "source_file": path,
                        "local_index": int(local_idx),
                    }
                )

            intermediate_rows.append(
                {
                    "event_id": global_event_id,
                    "source_file": path,
                    "local_index": int(local_idx),
                    "n_selected_jets": len(jets),
                    "has_two_higgs_bb": has_two_higgs_bb,
                    "fully_matchable": is_fully_matchable,
                    "target_0": targets[0],
                    "target_1": targets[1],
                    "target_2": targets[2],
                    "target_3": targets[3],
                    "match_dr_0": drs[0],
                    "match_dr_1": drs[1],
                    "match_dr_2": drs[2],
                    "match_dr_3": drs[3],
                }
            )

            global_event_id += 1

        if args.max_events is not None and total_events >= args.max_events:
            break

    if X_list:
        X_arr = np.stack(X_list)
        mask_arr = np.stack(mask_list)
        target_arr = np.stack(target_list)
        split_arr = split_indices(len(X_arr), args.seed, args.train_frac, args.val_frac)
        event_id_arr = np.asarray([r["event_id"] for r in event_rows], dtype=np.int64)
    else:
        X_arr = np.zeros((0, args.max_jets, len(FEATURE_NAMES)), dtype=np.float32)
        mask_arr = np.zeros((0, args.max_jets), dtype=np.bool_)
        target_arr = np.zeros((0, 4), dtype=np.int64)
        split_arr = np.asarray([], dtype=object)
        event_id_arr = np.asarray([], dtype=np.int64)

    np.savez_compressed(
        outdir / "hh4b_resolved_spanet_arrays.npz",
        X=X_arr,
        mask=mask_arr,
        targets=target_arr,
        split=split_arr,
        event_id=event_id_arr,
    )

    pd.DataFrame(event_rows).to_csv(outdir / "event_index.csv", index=False)
    pd.DataFrame(intermediate_rows).to_parquet(outdir / "hh4b_resolved_spanet_intermediate.parquet", index=False)

    with open(outdir / "feature_names.json", "w") as f:
        json.dump(FEATURE_NAMES, f, indent=2)

    split_counts = dict(pd.Series(split_arr).value_counts()) if len(split_arr) else {}

    # Simple pairing baseline: compare three possible pairings among the four highest-btag jets.
    # For matchable events, target full-event pairing is counted as correct only if both pairs match,
    # ignoring order within each pair and order of the two Higgs candidates.
    baseline_full_correct = 0
    baseline_per_higgs_correct = 0
    total_higgs_pairs = 2 * len(target_arr)

    for targets in target_arr:
        true_pairs = {frozenset([int(targets[0]), int(targets[1])]), frozenset([int(targets[2]), int(targets[3])])}
        pred_pairs = {frozenset([0, 1]), frozenset([2, 3])}
        n_pair_correct = len(true_pairs & pred_pairs)
        if n_pair_correct == 2:
            baseline_full_correct += 1
        baseline_per_higgs_correct += n_pair_correct

    baseline_full_acc = baseline_full_correct / len(target_arr) if len(target_arr) else 0.0
    baseline_per_higgs_acc = baseline_per_higgs_correct / total_higgs_pairs if total_higgs_pairs else 0.0

    metrics = {
        "total_events_scanned": int(total_events),
        "events_with_two_higgs_bb_candidates": int(events_with_two_higgs_bb),
        "events_with_at_least_4_selected_jets": int(events_with_at_least_4_selected_jets),
        "fully_matchable_events_in_topN_selected_jets": int(fully_matchable),
        "fully_matchable_fraction_of_total": float(fully_matchable / total_events) if total_events else 0.0,
        "fully_matchable_fraction_of_two_higgs_bb": float(fully_matchable / events_with_two_higgs_bb)
        if events_with_two_higgs_bb
        else 0.0,
        "train_events": int(split_counts.get("train", 0)),
        "val_events": int(split_counts.get("val", 0)),
        "test_events": int(split_counts.get("test", 0)),
        "baseline_full_event_pairing_accuracy_on_matchable": float(baseline_full_acc),
        "baseline_per_higgs_pairing_accuracy_on_matchable": float(baseline_per_higgs_acc),
        "max_jets": int(args.max_jets),
        "min_jet_pt": float(args.min_jet_pt),
        "max_abs_eta": float(args.max_abs_eta),
        "match_dr": float(args.match_dr),
        "n_input_files": int(len(files)),
    }

    write_summary(outdir, metrics)

    print(json.dumps(metrics, indent=2))
    print(f"Wrote outputs to {outdir}")


if __name__ == "__main__":
    main()
