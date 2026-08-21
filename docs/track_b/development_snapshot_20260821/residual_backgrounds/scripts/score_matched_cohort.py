#!/usr/bin/env python3
"""
step2n-equivalent (paper_exports residual-background study), Task A prep.

READ-ONLY against every frozen artifact. Verifies the matched-400k
sidecar SHA256 before touching it. Re-scores K/KF using the FROZEN
step2j models with the SAME iteration_range-from-best_iteration
convention as the already-frozen score_k_kf_spanet_matched_400k.py
(step2j/matched_400k/scripts/) - not reinvented, same logic. Loads
SPA-Net-2M score directly from the sidecar (spa_score field) - never
recomputed. Loads the ALREADY-FROZEN per-classifier thresholds from
step2j's K_KF_SPANET_MATCHED_400K_COMPARISON.json - never recomputed
or reoptimized.

No training. No Condor. No inference/test data (same sidecar step2j's
own matched-400k comparison used).
"""
import hashlib, json, os

import numpy as np
import xgboost as xgb

STUDY = "/uscms_data/d3/iturkmen/hh4b_delphes/track_b_harvey_bdt_working_points_20260818_v1"
J = f"{STUDY}/step2j_interim_10m_convergence_20260820_v1"
OUT = "/uscms_data/d3/iturkmen/hh4b_delphes/paper_exports/track_b_development_residual_backgrounds_20260821_v1"

NPZ_PATH = ("/eos/uscms/store/user/iturkmen/hh4b_delphes/track_b_harvey_bdt_working_points_20260818_v1/"
            "step2j_matched_400k_feature_sidecar_20260820_v1/matched_400k_features.npz")
EXPECTED_NPZ_SHA256 = "9588d0fe79e32d455467c719eaaa57541fcbead9835b8c32bebb9d68b96a2d30"

K_MODEL = f"{J}/results/K_seed0/stage2j_K_seed0_interim_convergence.json"
K_RESULT = f"{J}/results/K_seed0/stage2j_K_seed0_result.json"
KF_MODEL = f"{J}/results/KF_seed0/stage2j_KF_seed0_interim_convergence.json"
KF_RESULT = f"{J}/results/KF_seed0/stage2j_KF_seed0_result.json"
THRESHOLDS_SOURCE = f"{J}/matched_400k/results/K_KF_SPANET_MATCHED_400K_COMPARISON.json"

K_WIDTH, KF_WIDTH = 52, 82
EPS_S_LIST = [0.60, 0.50, 0.40, 0.25, 0.20, 0.10]


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    os.makedirs(f"{OUT}/results", exist_ok=True)

    npz_sha256 = sha256_file(NPZ_PATH)
    assert npz_sha256 == EXPECTED_NPZ_SHA256, f"SIDECAR HASH MISMATCH: {npz_sha256} != {EXPECTED_NPZ_SHA256}"
    print(f"sidecar SHA256 verified: {npz_sha256}")

    data = np.load(NPZ_PATH, allow_pickle=True)
    X = data["X"]
    y = data["y"].astype(int)
    process = data["process"]
    spa_score = data["spa_score"].astype(float)
    assert X.shape == (400000, KF_WIDTH), X.shape

    with open(K_RESULT) as f:
        k_res = json.load(f)
    with open(KF_RESULT) as f:
        kf_res = json.load(f)

    k_model_sha = sha256_file(K_MODEL)
    kf_model_sha = sha256_file(KF_MODEL)
    assert k_model_sha == k_res["model_sha256"], "K model SHA256 does not match its own result.json"
    assert kf_model_sha == kf_res["model_sha256"], "KF model SHA256 does not match its own result.json"
    print(f"K model verified: {k_model_sha}  (best_iteration={k_res['best_iteration']})")
    print(f"KF model verified: {kf_model_sha}  (best_iteration={kf_res['best_iteration']})")

    booster_k = xgb.Booster(); booster_k.load_model(K_MODEL)
    booster_kf = xgb.Booster(); booster_kf.load_model(KF_MODEL)

    score_k = booster_k.predict(xgb.DMatrix(X[:, :K_WIDTH]), iteration_range=(0, k_res["best_iteration"] + 1))
    score_kf = booster_kf.predict(xgb.DMatrix(X[:, :KF_WIDTH]), iteration_range=(0, kf_res["best_iteration"] + 1))

    with open(THRESHOLDS_SOURCE) as f:
        thresholds_source = json.load(f)
    thresholds = {}
    for arm_key, arm_label in [("K", "K"), ("KF", "KF"), ("SPANET", "SPANET2M")]:
        thresholds[arm_label] = {
            wp["nominal_epsS"]: wp["threshold"] for wp in thresholds_source["working_points"][arm_key]
        }
    print("thresholds loaded (already-frozen, not recomputed):")
    for label, tmap in thresholds.items():
        print(f"  {label}: {tmap}")

    np.savez(
        f"{OUT}/results/scored_matched_400k.npz",
        y=y, process=process, X=X,
        score_K=score_k, score_KF=score_kf, score_SPANET2M=spa_score,
    )
    with open(f"{OUT}/results/thresholds_frozen.json", "w") as f:
        json.dump(thresholds, f, indent=2)
    with open(f"{OUT}/results/scoring_provenance.json", "w") as f:
        json.dump({
            "npz_path": NPZ_PATH, "npz_sha256": npz_sha256,
            "k_model_path": K_MODEL, "k_model_sha256": k_model_sha, "k_best_iteration": k_res["best_iteration"],
            "kf_model_path": KF_MODEL, "kf_model_sha256": kf_model_sha, "kf_best_iteration": kf_res["best_iteration"],
            "thresholds_source": THRESHOLDS_SOURCE,
            "spa2m_score_source": "sidecar 'spa_score' field, not recomputed",
        }, f, indent=2)
    print(f"wrote {OUT}/results/scored_matched_400k.npz, thresholds_frozen.json, scoring_provenance.json")


if __name__ == "__main__":
    main()
