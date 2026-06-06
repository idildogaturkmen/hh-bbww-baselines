import json
import math
from pathlib import Path

import numpy as np
from datasets import load_dataset


HH_NPZ_DIR = Path("outputs/hbb_npz")
OUTDIR = Path("outputs/ttbar_classifier")
OUTDIR.mkdir(parents=True, exist_ok=True)

RANDOM_SEED = 42

FEATURE_NAMES = [
    "n_jets",
    "leading_jet_pt",
    "subleading_jet_pt",
    "highest_btag",
    "second_highest_btag",
    "mbb_top2_btag",
    "deltaR_top2_btag",
    "HT",
]

TTBAR_FILES = {
    "train": [
        "tt0123j_5f_ckm_LO_MLM_semiLeptonic/"
        "tt0123j_5f_ckm_LO_MLM_semiLeptonic-NEVENT10000-RS30000001.parquet",
        "tt0123j_5f_ckm_LO_MLM_semiLeptonic/"
        "tt0123j_5f_ckm_LO_MLM_semiLeptonic-NEVENT10000-RS30000002.parquet",
    ]
}


def delta_phi(phi1, phi2):
    dphi = phi1 - phi2
    while dphi > math.pi:
        dphi -= 2 * math.pi
    while dphi <= -math.pi:
        dphi += 2 * math.pi
    return dphi


def delta_r(j1, j2):
    return math.sqrt((j1[1] - j2[1]) ** 2 + delta_phi(j1[2], j2[2]) ** 2)


def invariant_mass(j1, j2):
    pt1, eta1, phi1, m1 = j1[:4]
    pt2, eta2, phi2, m2 = j2[:4]

    px1 = pt1 * np.cos(phi1)
    py1 = pt1 * np.sin(phi1)
    pz1 = pt1 * np.sinh(eta1)
    e1 = np.sqrt(px1**2 + py1**2 + pz1**2 + m1**2)

    px2 = pt2 * np.cos(phi2)
    py2 = pt2 * np.sin(phi2)
    pz2 = pt2 * np.sinh(eta2)
    e2 = np.sqrt(px2**2 + py2**2 + pz2**2 + m2**2)

    e = e1 + e2
    px = px1 + px2
    py = py1 + py2
    pz = pz1 + pz2

    m2_tot = e**2 - px**2 - py**2 - pz**2
    return math.sqrt(max(float(m2_tot), 0.0))


def event_features_from_jets(jets):
    """
    jets shape: (n_jets, 6)
    features: pt, eta, phi, mass, btag, btagPhys
    """
    if len(jets) < 2:
        return None

    pts = jets[:, 0]
    btags = jets[:, 4]

    order_pt = np.argsort(pts)[::-1]
    leading_pt = pts[order_pt[0]]
    subleading_pt = pts[order_pt[1]]

    order_btag = np.argsort(btags)[::-1]
    j1 = jets[order_btag[0]]
    j2 = jets[order_btag[1]]

    highest_btag = btags[order_btag[0]]
    second_highest_btag = btags[order_btag[1]]
    mbb = invariant_mass(j1, j2)
    drbb = delta_r(j1, j2)
    ht = float(np.sum(pts))

    return np.array(
        [
            len(jets),
            leading_pt,
            subleading_pt,
            highest_btag,
            second_highest_btag,
            mbb,
            drbb,
            ht,
        ],
        dtype=np.float32,
    )


def load_hh_split(split):
    data = np.load(HH_NPZ_DIR / f"{split}.npz")
    jets_all = data["jets"]
    mask_all = data["mask"]

    xs = []
    for jets, mask in zip(jets_all, mask_all):
        real = mask > 0
        feats = event_features_from_jets(jets[real])
        if feats is not None:
            xs.append(feats)

    return np.stack(xs)


def load_ttbar_features(target_n):
    ds = load_dataset(
        "fastmachinelearning/collide-1m",
        data_files=TTBAR_FILES,
        split="train",
        streaming=True,
    )

    xs = []
    n_scanned = 0
    n_lt2jets = 0

    for event in ds:
        n_scanned += 1

        pt = event["FullReco_JetAK4_PT"]
        eta = event["FullReco_JetAK4_Eta"]
        phi = event["FullReco_JetAK4_Phi"]
        mass = event["FullReco_JetAK4_Mass"]
        btag = event["FullReco_JetAK4_BTag"]
        btag_phys = event["FullReco_JetAK4_BTagPhys"]

        n_jets = len(pt)
        if n_jets < 2:
            n_lt2jets += 1
            continue

        jets = np.zeros((n_jets, 6), dtype=np.float32)
        jets[:, 0] = pt
        jets[:, 1] = eta
        jets[:, 2] = phi
        jets[:, 3] = mass
        jets[:, 4] = btag
        jets[:, 5] = btag_phys

        feats = event_features_from_jets(jets)
        if feats is not None:
            xs.append(feats)

        if len(xs) % 1000 == 0 and len(xs) > 0:
            print(
                f"ttbar usable events: {len(xs)}; scanned: {n_scanned}",
                flush=True,
            )

        if len(xs) >= target_n:
            break

    if len(xs) < target_n:
        raise RuntimeError(
            f"Only found {len(xs)} ttbar events, but target_n={target_n}."
        )

    summary = {
        "ttbar_scanned_events": n_scanned,
        "ttbar_lt2jets_rejected": n_lt2jets,
        "ttbar_usable_events": len(xs),
    }

    return np.stack(xs), summary


def save_split(split, X_hh, X_tt, rng):
    n = min(len(X_hh), len(X_tt))
    X_hh = X_hh[:n]
    X_tt = X_tt[:n]

    y_hh = np.ones(n, dtype=np.int64)
    y_tt = np.zeros(n, dtype=np.int64)

    X = np.concatenate([X_hh, X_tt], axis=0)
    y = np.concatenate([y_hh, y_tt], axis=0)

    sample = np.array(["HH"] * n + ["ttbar_semileptonic"] * n)

    idx = rng.permutation(len(y))
    X = X[idx]
    y = y[idx]
    sample = sample[idx]

    np.savez(
        OUTDIR / f"features_{split}.npz",
        X=X,
        y=y,
        sample=sample,
        feature_names=np.array(FEATURE_NAMES),
    )

    return {
        f"{split}_hh_events": int(n),
        f"{split}_ttbar_events": int(n),
        f"{split}_total_events": int(len(y)),
    }


def main():
    rng = np.random.default_rng(RANDOM_SEED)

    print("Loading HH features...", flush=True)
    X_hh_train = load_hh_split("train")
    X_hh_val = load_hh_split("val")
    X_hh_test = load_hh_split("test")

    hh_counts = {
        "train": len(X_hh_train),
        "val": len(X_hh_val),
        "test": len(X_hh_test),
    }

    total_hh = sum(hh_counts.values())
    print(f"HH counts: {hh_counts}, total={total_hh}", flush=True)

    print("Loading semileptonic ttbar features...", flush=True)
    X_tt_all, ttbar_summary = load_ttbar_features(total_hh)

    perm = rng.permutation(len(X_tt_all))
    X_tt_all = X_tt_all[perm]

    n_train = len(X_hh_train)
    n_val = len(X_hh_val)
    n_test = len(X_hh_test)

    X_tt_train = X_tt_all[:n_train]
    X_tt_val = X_tt_all[n_train : n_train + n_val]
    X_tt_test = X_tt_all[n_train + n_val : n_train + n_val + n_test]

    summary = {}
    summary.update(ttbar_summary)
    summary.update(
        {
            "feature_names": FEATURE_NAMES,
            "random_seed": RANDOM_SEED,
            "hh_train_events": int(n_train),
            "hh_val_events": int(n_val),
            "hh_test_events": int(n_test),
            "ttbar_source_files": TTBAR_FILES["train"],
        }
    )

    summary.update(save_split("train", X_hh_train, X_tt_train, rng))
    summary.update(save_split("val", X_hh_val, X_tt_val, rng))
    summary.update(save_split("test", X_hh_test, X_tt_test, rng))

    with open(OUTDIR / "dataset_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(json.dumps(summary, indent=2))
    print(f"\nSaved classifier datasets to {OUTDIR}")


if __name__ == "__main__":
    main()
