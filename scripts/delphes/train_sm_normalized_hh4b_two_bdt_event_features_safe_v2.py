"""
V2 safe event-feature two-BDT HH4b baseline.

This script reuses the validated safe two-BDT training/scanning logic from
train_sm_normalized_hh4b_two_bdt_event_features_safe.py, but replaces load_all()
so that it reads the v2 reconstruction parquets from:

  $HH4B_STORE/parquet_v2_nominal

It intentionally excludes truth/identifier columns such as j*_flavor,
event, sample, source_root, raw_index, and selected_index from BDT inputs.
"""

import importlib.util
import os
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(os.environ["HH4B_REPO"])
STORE = Path(os.environ["HH4B_STORE"])
V2_DIR = STORE / "parquet_v2_nominal"

BASE_SCRIPT = REPO / "scripts/delphes/train_sm_normalized_hh4b_two_bdt_event_features_safe.py"
OUTDIR = REPO / "outputs/tables/sm_normalized_hh4b_two_bdt_event_features_safe_v2_2026_07_10"
OUTDIR.mkdir(parents=True, exist_ok=True)

spec = importlib.util.spec_from_file_location("safe_event_bdt_v1", BASE_SCRIPT)
base = importlib.util.module_from_spec(spec)
spec.loader.exec_module(base)

SAMPLES = [
    {
        "analysis_sample": "ggF_HH4b_SMnorm",
        "group": "signal",
        "n_generated": 10000,
        "xsec_pb": 0.01055,
        "path": V2_DIR / "ggF_HH4b_SMnorm_v2_hh4b_candidates.parquet",
    },
    {
        "analysis_sample": "VBF_HH4b_SMnorm",
        "group": "signal",
        "n_generated": 10000,
        "xsec_pb": 0.000587,
        "path": V2_DIR / "VBF_HH4b_SMnorm_v2_hh4b_candidates.parquet",
    },
    {
        "analysis_sample": "ttbar_200k",
        "group": "background",
        "n_generated": 200000,
        "xsec_pb": 512.2746276855469,
        "path": V2_DIR / "ttbar_200k_v2_hh4b_candidates.parquet",
    },
    {
        "analysis_sample": "Zbbbb_100k",
        "group": "background",
        "n_generated": 100000,
        "xsec_pb": 6.912992,
        "path": V2_DIR / "Zbbbb_100k_v2_hh4b_candidates.parquet",
    },
    {
        "analysis_sample": "qcd_bbbb_iht100to200_20000",
        "group": "background",
        "n_generated": 20000,
        "xsec_pb": 206.2399139404297,
        "path": V2_DIR / "qcd_bbbb_iht100to200_20000_v2_hh4b_candidates.parquet",
    },
    {
        "analysis_sample": "qcd_bbbb_iht200to400_combined120k",
        "group": "background",
        "n_generated": 120000,
        "xsec_pb": 126.35226440429688,
        "path": V2_DIR / "qcd_bbbb_iht200to400_combined120k_v2_hh4b_candidates.parquet",
    },
    {
        "analysis_sample": "qcd_bbbb_iht400to600_20000",
        "group": "background",
        "n_generated": 20000,
        "xsec_pb": 9.774867057800293,
        "path": V2_DIR / "qcd_bbbb_iht400to600_20000_v2_hh4b_candidates.parquet",
    },
    {
        "analysis_sample": "qcd_bbbb_iht600plus_20000",
        "group": "background",
        "n_generated": 20000,
        "xsec_pb": 1.7876876592636108,
        "path": V2_DIR / "qcd_bbbb_iht600plus_20000_v2_hh4b_candidates.parquet",
    },
]

FEATURE_COLS = [
    # Candidate mass / HH geometry
    "mbb1", "mbb2", "avg_mbb", "delta_mbb",
    "r_hh", "r_hh_125_125", "r_hh_125_120",
    "mhh", "hh_pt", "hh_eta",

    # Higgs candidate kinematics
    "h1_pt", "h1_eta",
    "h2_pt", "h2_eta",
    "h_delta_eta", "h_delta_phi", "h_delta_r",
    "h_pt_balance",

    # Angular structure
    "drbb1", "drbb2",

    # Event-level v2 features
    "n_selected_jets", "n_selected_bjets",
    "n_extra_selected_jets", "n_extra_selected_bjets",
    "ht_selected_jets", "ht_selected_bjets", "ht_candidate_jets",

    # Candidate jet kinematics
    "j1_pt", "j2_pt", "j3_pt", "j4_pt",
    "j1_eta", "j2_eta", "j3_eta", "j4_eta",
    "j1_mass", "j2_mass", "j3_mass", "j4_mass",

    # Derived features added below
    "pt_sum4", "ht_over_mhh",
    "pt_asym_12", "pt_asym_34",
    "avg_drbb", "max_drbb", "min_drbb",
]


def add_derived_features(df):
    out = df.copy()

    eps = 1.0e-9

    out["pt_sum4"] = out[["j1_pt", "j2_pt", "j3_pt", "j4_pt"]].sum(axis=1)
    out["ht_over_mhh"] = out["ht_selected_jets"] / (out["mhh"].abs() + eps)

    out["pt_asym_12"] = (out["j1_pt"] - out["j2_pt"]).abs() / (out["j1_pt"] + out["j2_pt"] + eps)
    out["pt_asym_34"] = (out["j3_pt"] - out["j4_pt"]).abs() / (out["j3_pt"] + out["j4_pt"] + eps)

    out["avg_drbb"] = 0.5 * (out["drbb1"] + out["drbb2"])
    out["max_drbb"] = out[["drbb1", "drbb2"]].max(axis=1)
    out["min_drbb"] = out[["drbb1", "drbb2"]].min(axis=1)

    return out


def load_all():
    frames = []

    for cfg in SAMPLES:
        path = cfg["path"]
        if not path.exists():
            raise FileNotFoundError(f"Missing v2 candidate parquet for {cfg['analysis_sample']}: {path}")

        df = pd.read_parquet(path)
        df = add_derived_features(df)

        df["analysis_sample"] = cfg["analysis_sample"]
        df["group"] = cfg["group"]
        df["n_generated"] = cfg["n_generated"]
        df["xsec_pb"] = cfg["xsec_pb"]
        df["weight_pb"] = cfg["xsec_pb"] / cfg["n_generated"]

        frames.append(df)

        print(
            f"Loaded {cfg['analysis_sample']}: rows={len(df)}, "
            f"xsec_pb={cfg['xsec_pb']}, path={path.name}"
        )

    out = pd.concat(frames, ignore_index=True)

    out["is_signal"] = out["group"].eq("signal")
    out["target"] = out["is_signal"].astype(int)
    out["is_qcd"] = out["analysis_sample"].str.startswith("qcd_")
    out["is_top"] = out["analysis_sample"].str.contains("ttbar|ttbb|tt_", case=False, regex=True)

    missing = [c for c in FEATURE_COLS if c not in out.columns]
    if missing:
        raise RuntimeError(f"Missing required v2 feature columns: {missing}")

    # Explicitly protect against accidentally using truth / identifiers.
    forbidden = [
        "sample", "analysis_sample", "source_root", "source_root_index", "event",
        "j1_flavor", "j2_flavor", "j3_flavor", "j4_flavor",
        "j1_raw_index", "j2_raw_index", "j3_raw_index", "j4_raw_index",
        "j1_selected_index", "j2_selected_index", "j3_selected_index", "j4_selected_index",
    ]
    bad = [c for c in FEATURE_COLS if c in forbidden]
    if bad:
        raise RuntimeError(f"Forbidden non-detector/truth/id columns in FEATURE_COLS: {bad}")

    out = out.replace([np.inf, -np.inf], np.nan).dropna(subset=FEATURE_COLS).copy()

    print("\n=== V2 input row counts ===")
    print(
        out.groupby(["analysis_sample", "group"], as_index=False)
        .agg(
            rows=("event", "size"),
            n_generated=("n_generated", "first"),
            xsec_pb=("xsec_pb", "first"),
        )
        .to_string(index=False)
    )

    print("\n=== V2 BDT feature columns ===")
    for c in FEATURE_COLS:
        print(" ", c)

    return out, FEATURE_COLS


# Reuse validated v1 training/scanning code, but replace loader and output dir.
base.OUTDIR = OUTDIR
base.load_all = load_all

if __name__ == "__main__":
    base.main()

    readme = OUTDIR / "README.md"
    old = readme.read_text() if readme.exists() else ""
    prefix = """# V2 safe event-feature two-BDT HH4b baseline

This output uses v2 reconstructed HH4b candidates from `$HH4B_STORE/parquet_v2_nominal`.

It reuses the validated safe two-BDT event-feature training/scanning logic, but loads v2 candidate parquets directly.
No truth-level jet flavor columns, source identifiers, raw indices, selected indices, or event IDs are used as BDT features.

"""
    readme.write_text(prefix + old)
