#!/usr/bin/env python3

import argparse
from pathlib import Path
import sys

import pyarrow.parquet as pq

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

from make_hbb_bjet_regression_dataset import (
    COLUMNS,
    find_hh_files,
    event_from_batch,
)


def as_int(x, default=-999):
    try:
        return int(x)
    except Exception:
        return default


def arr(event, name):
    x = event.get(name, [])
    return [] if x is None else x


def valid_idx(i, n):
    return isinstance(i, int) and 0 <= i < n


def daughters_by_d1d2(event, idx):
    pids = arr(event, "FullReco_GenPart_PID")
    d1s = arr(event, "FullReco_GenPart_D1")
    d2s = arr(event, "FullReco_GenPart_D2")
    n = len(pids)

    if len(d1s) != n or len(d2s) != n:
        return []

    d1 = as_int(d1s[idx])
    d2 = as_int(d2s[idx])

    if valid_idx(d1, n) and valid_idx(d2, n) and d2 >= d1:
        return list(range(d1, d2 + 1))
    return []


def daughters_by_parent(event, idx):
    pids = arr(event, "FullReco_GenPart_PID")
    m1s = arr(event, "FullReco_GenPart_M1")
    m2s = arr(event, "FullReco_GenPart_M2")
    n = len(pids)

    if len(m1s) != n or len(m2s) != n:
        return []

    out = []
    for j in range(n):
        if as_int(m1s[j]) == idx or as_int(m2s[j]) == idx:
            out.append(j)
    return out


def describe_particle(event, idx):
    pids = arr(event, "FullReco_GenPart_PID")
    pts = arr(event, "FullReco_GenPart_PT")
    etas = arr(event, "FullReco_GenPart_Eta")
    phis = arr(event, "FullReco_GenPart_Phi")
    statuses = arr(event, "FullReco_GenPart_Status")
    m1s = arr(event, "FullReco_GenPart_M1")
    m2s = arr(event, "FullReco_GenPart_M2")

    n = len(pids)
    pid = as_int(pids[idx])
    status = as_int(statuses[idx]) if len(statuses) == n else -999
    pt = float(pts[idx]) if len(pts) == n else -1.0
    eta = float(etas[idx]) if len(etas) == n else -999.0
    phi = float(phis[idx]) if len(phis) == n else -999.0
    m1 = as_int(m1s[idx]) if len(m1s) == n else -999
    m2 = as_int(m2s[idx]) if len(m2s) == n else -999

    parent_pids = []
    for m in [m1, m2]:
        if valid_idx(m, n):
            parent_pids.append(as_int(pids[m]))

    return f"idx={idx:4d} pid={pid:5d} status={status:4d} pt={pt:8.2f} eta={eta:7.3f} phi={phi:7.3f} m1={m1:4d} m2={m2:4d} parent_pids={parent_pids}"


def print_tree(event, idx, depth=0, max_depth=4, visited=None):
    if visited is None:
        visited = set()
    if idx in visited or depth > max_depth:
        return
    visited.add(idx)

    indent = "  " * depth
    print(indent + describe_particle(event, idx))

    d_d1d2 = daughters_by_d1d2(event, idx)
    d_parent = daughters_by_parent(event, idx)

    daughters = d_parent if d_parent else d_d1d2

    if daughters:
        print(indent + f"  daughters_by_parent={d_parent}")
        print(indent + f"  daughters_by_d1d2={d_d1d2}")

    for d in daughters:
        print_tree(event, d, depth + 1, max_depth=max_depth, visited=visited)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-files", type=int, default=1)
    parser.add_argument("--n-events", type=int, default=3)
    parser.add_argument("--max-depth", type=int, default=5)
    args = parser.parse_args()

    files = find_hh_files(args.n_files)
    shown = 0

    for file_path in files:
        print(f"\nFILE: {file_path}")
        pf = pq.ParquetFile(file_path)

        for batch in pf.iter_batches(columns=COLUMNS, batch_size=20):
            cols = batch.to_pydict()
            n_rows = len(next(iter(cols.values())))

            for row_idx in range(n_rows):
                event = event_from_batch(cols, row_idx)

                pids = arr(event, "FullReco_GenPart_PID")
                statuses = arr(event, "FullReco_GenPart_Status")
                n = len(pids)

                w_indices = [i for i, pid in enumerate(pids) if abs(as_int(pid)) == 24]
                h_indices = [i for i, pid in enumerate(pids) if abs(as_int(pid)) == 25]

                print("\n" + "=" * 100)
                print(f"EVENT {shown}")
                print(f"n_genparts={n}")
                print(f"H indices: {h_indices}")
                print(f"W indices: {w_indices}")

                print("\nHiggs particles:")
                for h in h_indices:
                    print_tree(event, h, depth=0, max_depth=args.max_depth)

                print("\nW particles:")
                for w in w_indices:
                    print("\nW tree:")
                    print_tree(event, w, depth=0, max_depth=args.max_depth)

                print("\nAll status-23 quarks/leptons/neutrinos:")
                for i, pid in enumerate(pids):
                    apid = abs(as_int(pid))
                    status = as_int(statuses[i]) if len(statuses) == n else -999
                    if status == 23 or apid in [1,2,3,4,5,6,11,12,13,14,15,16]:
                        print(describe_particle(event, i))

                shown += 1
                if shown >= args.n_events:
                    return


if __name__ == "__main__":
    main()
