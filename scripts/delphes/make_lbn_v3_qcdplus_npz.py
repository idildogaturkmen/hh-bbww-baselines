#!/usr/bin/env python3

"""
Build an LBN-DNN-ready dataset from the BDT-v3 qcdplus candidate parquets.

Output:
  $HH4B_STORE/lbn_npz/v3_qcdplus/lbn_v3_qcdplus_candidates.npz

Arrays:
  X_p4:              (N, 4, 4), candidate jet four-vectors [E, px, py, pz]
  X_aux_topology:    scalar topology-only auxiliary features
  X_aux_mass_aware:  scalar mass-aware auxiliary features
  y:                 signal label
  is_signal
  is_qcd
  is_top
  weight_pb
  weight_events_450fb
  train_mask
  test_mask
  sample_id
  sample_names
  topology_feature_names
  mass_aware_feature_names
"""

import importlib.util
import os
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

REPO = Path(os.environ["HH4B_REPO"])
STORE = Path(os.environ["HH4B_STORE"])

MASS_SCRIPT = REPO / "scripts/delphes/train_sm_normalized_hh4b_two_bdt_event_features_safe_v3_qcdplus.py"
TOPO_SCRIPT = REPO / "scripts/delphes/train_sm_normalized_hh4b_two_bdt_event_features_safe_v3_qcdplus_topology_only.py"

OUTDIR = STORE / "lbn_npz/v3_qcdplus"
OUTDIR.mkdir(parents=True, exist_ok=True)
OUT = OUTDIR / "lbn_v3_qcdplus_candidates.npz"

LUMI_PB = 450_000.0
RANDOM_STATE = 12345
TEST_SIZE = 0.35


def import_wrapper(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def pt_eta_phi_m_to_epxpypz(pt, eta, phi, mass):
    pt = np.asarray(pt, dtype=np.float64)
    eta = np.asarray(eta, dtype=np.float64)
    phi = np.asarray(phi, dtype=np.float64)
    mass = np.asarray(mass, dtype=np.float64)

    mass = np.maximum(mass, 0.0)

    px = pt * np.cos(phi)
    py = pt * np.sin(phi)
    pz = pt * np.sinh(eta)
    e = np.sqrt(np.maximum(px * px + py * py + pz * pz + mass * mass, 0.0))

    return np.stack([e, px, py, pz], axis=-1)


def main():
    mass_mod = import_wrapper(MASS_SCRIPT, "bdt_v3_mass")
    topo_mod = import_wrapper(TOPO_SCRIPT, "bdt_v3_topology")

    df, mass_features = mass_mod.load_all()
    _, topology_features = topo_mod.load_all()

    required = []
    for i in range(1, 5):
        required += [f"j{i}_pt", f"j{i}_eta", f"j{i}_phi", f"j{i}_mass"]

    missing = [c for c in required if c not in df.columns]
    if missing:
        similar = [c for c in df.columns if any(k in c.lower() for k in ["phi", "pt", "eta", "mass", "jet"])]
        raise RuntimeError(
            "Missing columns needed for LBN four-vectors:\n"
            + "\n".join(missing)
            + "\n\nSimilar available columns:\n"
            + "\n".join(similar)
            + "\n\nIf j*_phi is missing, patch the candidate-maker to save selected jet phi and regenerate the candidate parquets."
        )

    # Build candidate jet four-vectors.
    p4s = []
    for i in range(1, 5):
        p4 = pt_eta_phi_m_to_epxpypz(
            df[f"j{i}_pt"].values,
            df[f"j{i}_eta"].values,
            df[f"j{i}_phi"].values,
            df[f"j{i}_mass"].values,
        )
        p4s.append(p4)

    X_p4 = np.stack(p4s, axis=1).astype("float32")  # (N, 4 jets, 4-vector)

    # Auxiliary scalar features.
    X_aux_topology = df[topology_features].values.astype("float32")
    X_aux_mass_aware = df[mass_features].values.astype("float32")

    btag_features = ["j1_btag", "j2_btag", "j3_btag", "j4_btag"]
    missing_btag = [c for c in btag_features if c not in df.columns]
    if missing_btag:
        raise RuntimeError(f"Missing btag columns for LBN auxiliary input: {missing_btag}")
    X_btag = df[btag_features].values.astype("float32")

    y = df["is_signal"].astype("int64").values
    is_signal = df["is_signal"].astype(bool).values
    is_qcd = df["is_qcd"].astype(bool).values
    is_top = df["is_top"].astype(bool).values

    weight_pb = df["weight_pb"].astype("float64").values
    weight_events_450fb = weight_pb * LUMI_PB

    sample_cat = pd.Categorical(df["analysis_sample"])
    sample_id = sample_cat.codes.astype("int32")
    sample_names = np.asarray(sample_cat.categories.astype(str), dtype="U")

    indices = np.arange(len(df))
    train_idx, test_idx = train_test_split(
        indices,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=df["target"],
    )

    train_mask = np.zeros(len(df), dtype=bool)
    test_mask = np.zeros(len(df), dtype=bool)
    train_mask[train_idx] = True
    test_mask[test_idx] = True

    # Scale test weights to full sample, matching BDT/DNN logic.
    total_counts = df.groupby("analysis_sample").size()
    test_counts = df.iloc[test_idx].groupby("analysis_sample").size()

    weight_pb_scaled_to_full = weight_pb.copy()
    for sample in total_counts.index:
        mask = (df["analysis_sample"].values == sample) & test_mask
        if sample in test_counts.index and test_counts.loc[sample] > 0:
            weight_pb_scaled_to_full[mask] = weight_pb[mask] * total_counts.loc[sample] / test_counts.loc[sample]

    weight_events_450fb_scaled_to_full = weight_pb_scaled_to_full * LUMI_PB

    np.savez_compressed(
        OUT,
        X_p4=X_p4,
        X_aux_topology=X_aux_topology,
        X_aux_mass_aware=X_aux_mass_aware,
        X_btag=X_btag,
        y=y,
        is_signal=is_signal,
        is_qcd=is_qcd,
        is_top=is_top,
        weight_pb=weight_pb,
        weight_events_450fb=weight_events_450fb,
        weight_pb_scaled_to_full=weight_pb_scaled_to_full,
        weight_events_450fb_scaled_to_full=weight_events_450fb_scaled_to_full,
        train_mask=train_mask,
        test_mask=test_mask,
        sample_id=sample_id,
        sample_names=sample_names,
        topology_feature_names=np.asarray(topology_features, dtype="U"),
        mass_aware_feature_names=np.asarray(mass_features, dtype="U"),
        btag_feature_names=np.asarray(btag_features, dtype="U"),
        analysis_sample=df["analysis_sample"].astype(str).values.astype("U"),
        group=df["group"].astype(str).values.astype("U"),
        target=df["target"].astype(str).values.astype("U"),
    )

    summary_rows = [
        {
            "quantity": "n_events",
            "value": len(df),
        },
        {
            "quantity": "n_train",
            "value": int(train_mask.sum()),
        },
        {
            "quantity": "n_test",
            "value": int(test_mask.sum()),
        },
        {
            "quantity": "n_signal",
            "value": int(is_signal.sum()),
        },
        {
            "quantity": "n_background",
            "value": int((~is_signal).sum()),
        },
        {
            "quantity": "n_topology_features",
            "value": len(topology_features),
        },
        {
            "quantity": "n_mass_aware_features",
            "value": len(mass_features),
        },
        {
            "quantity": "n_btag_features",
            "value": len(btag_features),
        },
    ]

    summary = pd.DataFrame(summary_rows)
    summary.to_csv(OUTDIR / "lbn_v3_qcdplus_dataset_summary.csv", index=False)
    (OUTDIR / "lbn_v3_qcdplus_dataset_summary.md").write_text(
        summary.to_markdown(index=False) + "\n"
    )

    by_sample = (
        df.groupby(["analysis_sample", "group"], as_index=False)
        .agg(
            rows=("target", "size"),
            n_generated=("n_generated", "first"),
            xsec_pb=("xsec_pb", "first"),
            weight_pb=("weight_pb", "first"),
        )
    )
    by_sample.to_csv(OUTDIR / "lbn_v3_qcdplus_by_sample.csv", index=False)
    (OUTDIR / "lbn_v3_qcdplus_by_sample.md").write_text(
        by_sample.to_markdown(index=False) + "\n"
    )

    readme = OUTDIR / "README.md"
    readme.write_text(
        "# LBN-v3 qcdplus dataset\n\n"
        "This dataset is built from the same BDT-v3 qcdplus candidate parquets used for the BDT and DNN baselines.\n\n"
        "Main array:\n"
        "- `X_p4`: shape `(N, 4, 4)`, containing the four selected candidate jets in `[E, px, py, pz]` format.\n\n"
        "Auxiliary arrays:\n"
        "- `X_aux_topology`: topology-only scalar features.\n"
        "- `X_aux_mass_aware`: mass-aware scalar features.\n"
        "- `X_btag`: four candidate jet b-tag scores.\n\n"
        "Labels and masks:\n"
        "- `y`: signal label.\n"
        "- `is_qcd`: QCD-background mask.\n"
        "- `is_top`: top-background mask.\n"
        "- `train_mask` and `test_mask`: deterministic split using the same random seed, test fraction, and target stratification as the BDT/DNN studies.\n\n"
        "Interpretation:\n"
        "This dataset is intended for an LBN-DNN baseline. Since LBN uses four-vectors, it should be treated as physics-structured and mass-aware by construction.\n"
    )

    print("Wrote:", OUT)
    print("X_p4:", X_p4.shape)
    print("X_aux_topology:", X_aux_topology.shape)
    print("X_aux_mass_aware:", X_aux_mass_aware.shape)
    print("X_btag:", X_btag.shape)
    print("n_train:", train_mask.sum())
    print("n_test:", test_mask.sum())


if __name__ == "__main__":
    main()
