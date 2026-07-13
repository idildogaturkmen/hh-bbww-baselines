#!/usr/bin/env python3

"""
BDT-v3 qcdplus topology-only / mass-blinded baseline.

This reuses the frozen BDT-v3 qcdplus samples and training logic, but removes
direct Higgs-candidate mass variables from FEATURE_COLS.

Purpose:
Compare against the mass-aware BDT-v3 qcdplus baseline to quantify mass sculpting.
"""

import importlib.util
import os
from pathlib import Path

REPO = Path(os.environ["HH4B_REPO"])

SOURCE_SCRIPT = REPO / "scripts/delphes/train_sm_normalized_hh4b_two_bdt_event_features_safe_v3_qcdplus.py"
OUTDIR = REPO / "outputs/tables/sm_normalized_hh4b_two_bdt_event_features_safe_v3_qcdplus_topology_only_2026_07_13"
OUTDIR.mkdir(parents=True, exist_ok=True)

spec = importlib.util.spec_from_file_location("bdt_v3_qcdplus_source", SOURCE_SCRIPT)
source = importlib.util.module_from_spec(spec)
spec.loader.exec_module(source)

DROP_FEATURES = {
    # Direct candidate mass / Higgs-mass targeting variables
    "mbb1",
    "mbb2",
    "avg_mbb",
    "delta_mbb",
    "r_hh",
    "r_hh_125_125",
    "r_hh_125_120",

    # Global invariant mass and direct jet masses
    "mhh",
    "j1_mass",
    "j2_mass",
    "j3_mass",
    "j4_mass",

    # Derived variable containing mhh
    "ht_over_mhh",
}

FEATURE_COLS = [c for c in source.FEATURE_COLS if c not in DROP_FEATURES]

# Make source.load_all use the reduced feature list.
source.FEATURE_COLS = FEATURE_COLS

# Expose these so category scripts can import this file like the normal v3 wrapper.
base = source.base
load_all = source.load_all

base.OUTDIR = OUTDIR
base.load_all = load_all

if __name__ == "__main__":
    base.main()

    readme = OUTDIR / "README.md"
    old = readme.read_text() if readme.exists() else ""

    prefix = """# BDT-v3 qcdplus topology-only / mass-blinded baseline

This output uses the same samples, weights, train/test split, and BDT hyperparameters as the frozen BDT-v3 qcdplus baseline.

Direct Higgs-candidate mass variables are removed from the BDT inputs:
- mbb1, mbb2, avg_mbb, delta_mbb
- r_hh, r_hh_125_125, r_hh_125_120
- mhh
- j1_mass, j2_mass, j3_mass, j4_mass
- ht_over_mhh

This is intended as a mass-sculpting validation, not as a replacement for the mass-aware baseline.

Remaining features:
"""

    feature_block = "\n".join(f"- {c}" for c in FEATURE_COLS) + "\n\n"
    readme.write_text(prefix + feature_block + old)
