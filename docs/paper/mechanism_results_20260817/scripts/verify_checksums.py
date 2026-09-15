#!/usr/bin/env python3
"""Verify SHA256SUMS for every frozen source directory before anything reads
their contents. Read-only. Exits non-zero (and refuses) if any manifest is
missing or any listed file fails its checksum -- per the governing
instruction, a mismatch means STOP, not silent repair.
"""
from __future__ import annotations

import hashlib
import sys
from pathlib import Path

from frozen_paths import ALL_SHA256SUMS_DIRS


def verify_dir(d: Path) -> tuple[int, int, list[str]]:
    manifest = d / "SHA256SUMS"
    if not manifest.is_file():
        return 0, 0, [f"NO SHA256SUMS in {d}"]
    n_ok, n_total, problems = 0, 0, []
    for line in manifest.read_text().splitlines():
        line = line.rstrip("\n")
        if not line.strip():
            continue
        expected, _, relpath = line.partition("  ")
        if not relpath:
            expected, _, relpath = line.partition(" ")
        relpath = relpath.strip()
        target = d / relpath
        n_total += 1
        if not target.is_file():
            problems.append(f"MISSING FILE listed in manifest: {target}")
            continue
        actual = hashlib.sha256(target.read_bytes()).hexdigest()
        if actual != expected.strip():
            problems.append(f"HASH MISMATCH: {target} expected={expected.strip()} actual={actual}")
        else:
            n_ok += 1
    return n_ok, n_total, problems


def main() -> int:
    all_problems = []
    print(f"{'directory':<90} {'ok/total':>10}")
    for d in ALL_SHA256SUMS_DIRS:
        n_ok, n_total, problems = verify_dir(d)
        print(f"{str(d):<90} {n_ok:>4}/{n_total:<5}")
        all_problems.extend(problems)
    if all_problems:
        print("\nSTOP -- checksum verification failed:", file=sys.stderr)
        for p in all_problems:
            print(f"  {p}", file=sys.stderr)
        return 1
    print("\nAll frozen source directories: 100% checksum-clean. Safe to read.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
