"""
Diagnose GenPart truth structure for HH->4b signal samples.

Purpose:
  The recovered resolved HH4b SPA-Net builder found only a small number of
  fully matchable events. This diagnostic checks whether that is due to:
    - strict Higgs descendant tracing,
    - Higgs status-copy structure,
    - b-quark mother/daughter links,
    - reco-jet acceptance/matching,
    - or a real acceptance limitation.

Outputs:
  - summary.csv
  - per_event_diagnostics.csv
  - pid_status_counts.csv
  - example_events.md
"""

from __future__ import annotations

import argparse
import glob
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


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


def daughter_indices(i: int, d1: np.ndarray, d2: np.ndarray, n: int) -> list[int]:
    if i < 0 or i >= n:
        return []
    a = int(d1[i]) if i < len(d1) else -1
    b = int(d2[i]) if i < len(d2) else -1

    if a >= 0 and b >= a:
        return [j for j in range(a, min(b + 1, n)) if j != i]
    if a >= 0:
        return [a] if a < n and a != i else []
    if b >= 0:
        return [b] if b < n and b != i else []
    return []


def descendants_of(i: int, d1: np.ndarray, d2: np.ndarray, n: int) -> set[int]:
    seen: set[int] = set()
    stack = [i]
    while stack:
        cur = stack.pop()
        for child in daughter_indices(cur, d1, d2, n):
            if child not in seen:
                seen.add(child)
                stack.append(child)
    return seen


def mother_indices(i: int, m1: np.ndarray, m2: np.ndarray, n: int) -> list[int]:
    out = []
    if i < len(m1) and 0 <= int(m1[i]) < n:
        out.append(int(m1[i]))
    if i < len(m2) and 0 <= int(m2[i]) < n and int(m2[i]) not in out:
        out.append(int(m2[i]))
    return out


def ancestors_of(i: int, m1: np.ndarray, m2: np.ndarray, n: int) -> set[int]:
    seen: set[int] = set()
    stack = [i]
    while stack:
        cur = stack.pop()
        for mom in mother_indices(cur, m1, m2, n):
            if mom not in seen:
                seen.add(mom)
                stack.append(mom)
    return seen


def select_reco_jets(row: pd.Series, min_pt: float, max_abs_eta: float) -> list[dict[str, float]]:
    pt = as_array(row.get("FullReco_JetAK4_PT"), float)
    eta = as_array(row.get("FullReco_JetAK4_Eta"), float)
    phi = as_array(row.get("FullReco_JetAK4_Phi"), float)
    btag = as_array(row.get("FullReco_JetAK4_BTag"), float)
    btagphys = as_array(row.get("FullReco_JetAK4_BTagPhys"), float)

    n = min(len(pt), len(eta), len(phi))
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
                "idx": int(i),
                "pt": float(pt[i]),
                "eta": float(eta[i]),
                "phi": float(phi[i]),
                "btag": float(btag[i]) if i < len(btag) else 0.0,
                "btagphys": float(btagphys[i]) if i < len(btagphys) else 0.0,
            }
        )

    jets.sort(key=lambda j: (j["btag"], j["btagphys"], j["pt"]), reverse=True)
    return jets


def count_truth_structures(pid, pt, eta, phi, m1, m2, d1, d2, status, jets, match_dr, max_jets):
    n = len(pid)

    hidx = [int(i) for i in np.where(pid == 25)[0]]
    bidx = [int(i) for i in np.where(np.abs(pid) == 5)[0]]

    # Method A: Higgs -> descendant b quarks.
    higgs_desc_b_counts = []
    higgs_desc_b_sets = []
    for h in hidx:
        desc = descendants_of(h, d1, d2, n)
        bs = sorted([i for i in desc if abs(int(pid[i])) == 5])
        higgs_desc_b_counts.append(len(bs))
        if len(bs) >= 2:
            higgs_desc_b_sets.append(tuple(bs))

    unique_higgs_desc_b_sets = sorted(set(higgs_desc_b_sets))
    n_higgs_with_ge2_desc_b = len(unique_higgs_desc_b_sets)

    # Method B: b quarks whose ancestor chain contains a Higgs.
    b_with_h_ancestor = []
    b_direct_h_mother = []
    for b in bidx:
        moms = mother_indices(b, m1, m2, n)
        if any(pid[m] == 25 for m in moms):
            b_direct_h_mother.append(b)
        ancestors = ancestors_of(b, m1, m2, n)
        if any(pid[a] == 25 for a in ancestors):
            b_with_h_ancestor.append(b)

    # Dedupe rough final b candidates by eta/phi/status not attempted here.
    # This diagnostic intentionally counts raw GenPart entries first.
    n_b_with_h_ancestor = len(b_with_h_ancestor)
    n_b_direct_h_mother = len(b_direct_h_mother)

    # Match b-with-H-ancestor to selected reco jets.
    top_jets = jets[:max_jets]
    matched_b_ancestor = 0
    unique_matched_jets = set()

    for b in b_with_h_ancestor:
        best_j = -1
        best_dr = 999.0
        for jidx, jet in enumerate(top_jets):
            dr = delta_r(float(eta[b]), float(phi[b]), jet["eta"], jet["phi"])
            if dr < best_dr:
                best_dr = dr
                best_j = jidx
        if best_dr < match_dr:
            matched_b_ancestor += 1
            unique_matched_jets.add(best_j)

    return {
        "n_genpart": int(n),
        "n_higgs_pid25": int(len(hidx)),
        "n_b_pid_abs5": int(len(bidx)),
        "n_higgs_with_ge2_desc_b": int(n_higgs_with_ge2_desc_b),
        "n_b_direct_h_mother": int(n_b_direct_h_mother),
        "n_b_with_h_ancestor": int(n_b_with_h_ancestor),
        "n_selected_jets": int(len(jets)),
        "n_top_jets": int(len(top_jets)),
        "n_b_with_h_ancestor_matched_to_top_jets": int(matched_b_ancestor),
        "n_unique_matched_top_jets_from_h_ancestor_b": int(len(unique_matched_jets)),
        "has_two_higgs_by_descendants": bool(n_higgs_with_ge2_desc_b >= 2),
        "has_four_b_with_h_ancestor": bool(n_b_with_h_ancestor >= 4),
        "has_four_selected_jets": bool(len(jets) >= 4),
        "has_four_unique_matched_top_jets_from_h_ancestor_b": bool(len(unique_matched_jets) >= 4),
    }


def make_example_text(event_examples: list[dict]) -> str:
    lines = ["# HH4b GenPart truth diagnostic examples", ""]
    for ex in event_examples:
        lines.append(f"## Example event {ex['event_id']}")
        lines.append("")
        lines.append(f"- source_file: `{ex['source_file']}`")
        lines.append(f"- local_index: {ex['local_index']}")
        lines.append("")
        lines.append("### Summary")
        lines.append("")
        for k, v in ex["summary"].items():
            lines.append(f"- {k}: {v}")
        lines.append("")
        lines.append("### Higgs entries")
        lines.append("")
        lines.append("| idx | status | pt | eta | phi | mothers | daughters |")
        lines.append("|---:|---:|---:|---:|---:|---|---|")
        for row in ex["higgs_rows"]:
            lines.append(
                f"| {row['idx']} | {row['status']} | {row['pt']:.3f} | {row['eta']:.3f} | {row['phi']:.3f} | "
                f"{row['mothers']} | {row['daughters']} |"
            )
        lines.append("")
        lines.append("### b-quark entries")
        lines.append("")
        lines.append("| idx | pid | status | pt | eta | phi | mothers | has_H_ancestor |")
        lines.append("|---:|---:|---:|---:|---:|---:|---|---|")
        for row in ex["b_rows"]:
            lines.append(
                f"| {row['idx']} | {row['pid']} | {row['status']} | {row['pt']:.3f} | {row['eta']:.3f} | {row['phi']:.3f} | "
                f"{row['mothers']} | {row['has_h_ancestor']} |"
            )
        lines.append("")
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-glob", default="outputs/collide_selected_backgrounds/HH_4b/*.parquet")
    parser.add_argument("--outdir", default="outputs/hh4b_spanet/genpart_truth_diagnostic")
    parser.add_argument("--max-files", type=int, default=None)
    parser.add_argument("--max-events", type=int, default=2000)
    parser.add_argument("--min-jet-pt", type=float, default=20.0)
    parser.add_argument("--max-abs-eta", type=float, default=2.5)
    parser.add_argument("--max-jets", type=int, default=10)
    parser.add_argument("--match-dr", type=float, default=0.4)
    parser.add_argument("--n-examples", type=int, default=10)
    args = parser.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    files = sorted(glob.glob(args.input_glob))
    if args.max_files is not None:
        files = files[: args.max_files]

    if not files:
        raise FileNotFoundError(
            f"No files matched {args.input_glob}. "
            "This is expected in the temporary pod if the COLLIDE data is not mounted."
        )

    columns = [
        "FullReco_GenPart_PID",
        "FullReco_GenPart_PT",
        "FullReco_GenPart_Eta",
        "FullReco_GenPart_Phi",
        "FullReco_GenPart_M1",
        "FullReco_GenPart_M2",
        "FullReco_GenPart_D1",
        "FullReco_GenPart_D2",
        "FullReco_GenPart_Status",
        "FullReco_JetAK4_PT",
        "FullReco_JetAK4_Eta",
        "FullReco_JetAK4_Phi",
        "FullReco_JetAK4_BTag",
        "FullReco_JetAK4_BTagPhys",
    ]

    per_event = []
    pid_status_counter = Counter()
    event_examples = []
    total = 0

    for path in files:
        df = pd.read_parquet(path, columns=columns)

        for local_idx, row in df.iterrows():
            if args.max_events is not None and total >= args.max_events:
                break

            pid = as_array(row["FullReco_GenPart_PID"], int)
            pt = as_array(row["FullReco_GenPart_PT"], float)
            eta = as_array(row["FullReco_GenPart_Eta"], float)
            phi = as_array(row["FullReco_GenPart_Phi"], float)
            m1 = as_array(row["FullReco_GenPart_M1"], int)
            m2 = as_array(row["FullReco_GenPart_M2"], int)
            d1 = as_array(row["FullReco_GenPart_D1"], int)
            d2 = as_array(row["FullReco_GenPart_D2"], int)
            status = as_array(row["FullReco_GenPart_Status"], int)

            jets = select_reco_jets(row, args.min_jet_pt, args.max_abs_eta)

            for p, st in zip(pid, status):
                if abs(int(p)) in {5, 25}:
                    pid_status_counter[(int(p), int(st))] += 1

            diag = count_truth_structures(
                pid, pt, eta, phi, m1, m2, d1, d2, status,
                jets, args.match_dr, args.max_jets
            )
            diag["event_id"] = total
            diag["source_file"] = path
            diag["local_index"] = int(local_idx)
            per_event.append(diag)

            if len(event_examples) < args.n_examples:
                n = len(pid)
                h_indices = [int(i) for i in np.where(pid == 25)[0]]
                b_indices = [int(i) for i in np.where(np.abs(pid) == 5)[0]]

                higgs_rows = []
                for h in h_indices[:20]:
                    higgs_rows.append(
                        {
                            "idx": h,
                            "status": int(status[h]) if h < len(status) else -999,
                            "pt": float(pt[h]) if h < len(pt) else float("nan"),
                            "eta": float(eta[h]) if h < len(eta) else float("nan"),
                            "phi": float(phi[h]) if h < len(phi) else float("nan"),
                            "mothers": mother_indices(h, m1, m2, n),
                            "daughters": daughter_indices(h, d1, d2, n),
                        }
                    )

                b_rows = []
                for b in b_indices[:40]:
                    ancestors = ancestors_of(b, m1, m2, n)
                    b_rows.append(
                        {
                            "idx": b,
                            "pid": int(pid[b]),
                            "status": int(status[b]) if b < len(status) else -999,
                            "pt": float(pt[b]) if b < len(pt) else float("nan"),
                            "eta": float(eta[b]) if b < len(eta) else float("nan"),
                            "phi": float(phi[b]) if b < len(phi) else float("nan"),
                            "mothers": mother_indices(b, m1, m2, n),
                            "has_h_ancestor": any(pid[a] == 25 for a in ancestors),
                        }
                    )

                event_examples.append(
                    {
                        "event_id": total,
                        "source_file": path,
                        "local_index": int(local_idx),
                        "summary": diag,
                        "higgs_rows": higgs_rows,
                        "b_rows": b_rows,
                    }
                )

            total += 1

        if args.max_events is not None and total >= args.max_events:
            break

    per_event_df = pd.DataFrame(per_event)
    per_event_df.to_csv(outdir / "per_event_diagnostics.csv", index=False)

    summary_rows = []
    summary_rows.append(("events_scanned", len(per_event_df)))

    bool_cols = [
        "has_two_higgs_by_descendants",
        "has_four_b_with_h_ancestor",
        "has_four_selected_jets",
        "has_four_unique_matched_top_jets_from_h_ancestor_b",
    ]
    for c in bool_cols:
        n_pass = int(per_event_df[c].sum()) if c in per_event_df else 0
        frac = n_pass / len(per_event_df) if len(per_event_df) else 0.0
        summary_rows.append((c + "_count", n_pass))
        summary_rows.append((c + "_fraction", frac))

    numeric_cols = [
        "n_higgs_pid25",
        "n_b_pid_abs5",
        "n_higgs_with_ge2_desc_b",
        "n_b_direct_h_mother",
        "n_b_with_h_ancestor",
        "n_selected_jets",
        "n_b_with_h_ancestor_matched_to_top_jets",
        "n_unique_matched_top_jets_from_h_ancestor_b",
    ]
    for c in numeric_cols:
        if c in per_event_df:
            summary_rows.append((c + "_mean", float(per_event_df[c].mean())))
            summary_rows.append((c + "_median", float(per_event_df[c].median())))
            summary_rows.append((c + "_max", float(per_event_df[c].max())))

    pd.DataFrame(summary_rows, columns=["metric", "value"]).to_csv(outdir / "summary.csv", index=False)

    pid_status_rows = [
        {"pid": pid_val, "status": status_val, "count": count}
        for (pid_val, status_val), count in sorted(pid_status_counter.items())
    ]
    pd.DataFrame(pid_status_rows).to_csv(outdir / "pid_status_counts.csv", index=False)

    (outdir / "example_events.md").write_text(make_example_text(event_examples), encoding="utf-8")

    print("Wrote:", outdir)
    print(pd.DataFrame(summary_rows, columns=["metric", "value"]).to_string(index=False))


if __name__ == "__main__":
    main()
