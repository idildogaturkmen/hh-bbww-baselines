#!/usr/bin/env python3
"""Extract every number this package plots/tabulates from the frozen source
artifacts, and write a single MECHANISM_RESULTS_DATA.json.

This script only reads. It never opens holdout_A (already consumed by
Phase-4M, not reopened here), holdout_B, holdout_Q, or inference_qcd_1/2.
It does not train, retrain, or evaluate any model -- every number below is a
transcription or a simple, explicitly-labeled arithmetic derivation
(ratios, tier classification) of numbers that already exist in a frozen,
checksum-verified artifact.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

from frozen_paths import (
    TRACK_A_EVALUATION_RESULTS_JSON,
    TRACK_A_SUPPORTED_REJECTION_RANGE_TSV,
    TRACK_B_HOLDOUT_A_WORKING_POINTS_TSV,
    TRACK_B_HOLDOUT_A_SEED_STABILITY_TSV,
)

OUT_PATH = Path(__file__).resolve().parent.parent / "MECHANISM_RESULTS_DATA.json"

EPS_S_GRID_B = [0.6, 0.5, 0.4, 0.25, 0.2, 0.1]
ARMS_B = ["E0", "E1", "E2"]


def tier_from_raw_qcd(raw_qcd: float) -> str:
    if raw_qcd >= 100:
        return "well_supported"
    if raw_qcd >= 10:
        return "statistically_limited_but_reportable"
    return "exploratory_not_primary_quantitative_claim"


# ---------------------------------------------------------------------------
# Track B: phase4M holdout_A (already opened once, already frozen; read only)
# ---------------------------------------------------------------------------

def load_trackb() -> dict:
    with open(TRACK_B_HOLDOUT_A_SEED_STABILITY_TSV) as f:
        seed_stab_rows = list(csv.DictReader(f, delimiter="\t"))
    with open(TRACK_B_HOLDOUT_A_WORKING_POINTS_TSV) as f:
        per_seed_rows = list(csv.DictReader(f, delimiter="\t"))

    # aggregate table, keyed (arm, eps_S)
    agg = {}
    for r in seed_stab_rows:
        key = (r["arm"], float(r["eps_S_target"]))
        agg[key] = {
            "R_B_mean": float(r["R_B_mean"]),
            "R_B_std": float(r["R_B_std"]),
            "raw_qcd_mean": float(r["raw_qcd_mean"]),
            "auc_mean": float(r["auc_mean"]),
            "n_seeds": int(r["n_seeds"]),
        }

    # per-seed table, keyed (arm, eps_S) -> list of (seed, R_B, raw_qcd, support_label)
    per_seed = {}
    for r in per_seed_rows:
        key = (r["arm"], float(r["eps_S_target"]))
        per_seed.setdefault(key, []).append({
            "seed": int(r["seed"]),
            "R_B": float(r["R_B"]),
            "raw_surviving_qcd": int(r["raw_surviving_qcd"]),
            "raw_surviving_signal": int(r["raw_surviving_signal"]),
            "support_label": r["support_label"],
        })

    working_points = []
    for eps_s in EPS_S_GRID_B:
        for arm in ARMS_B:
            key = (arm, eps_s)
            a = agg[key]
            seeds = sorted(per_seed[key], key=lambda x: x["seed"])
            assert len(seeds) == 3, f"expected 3 seeds for {key}, got {len(seeds)}"
            tier_mean = tier_from_raw_qcd(a["raw_qcd_mean"])
            per_seed_tiers = {tier_from_raw_qcd(s["raw_surviving_qcd"]) for s in seeds}
            working_points.append({
                "arm": arm,
                "eps_S": eps_s,
                "R_B_mean": a["R_B_mean"],
                "R_B_std": a["R_B_std"],
                "raw_qcd_mean": a["raw_qcd_mean"],
                "auc_mean": a["auc_mean"],
                "n_seeds": a["n_seeds"],
                "support_tier_of_mean": tier_mean,
                "per_seed_support_labels_agree_with_mean_tier": (
                    per_seed_tiers == {tier_mean}
                ),
                "per_seed": seeds,
            })

    # mechanism gain ratios: paired by seed index, E1/E0 and E2/E1
    gain = []
    for eps_s in EPS_S_GRID_B:
        row = {"eps_S": eps_s}
        for num_arm, den_arm, label in [("E1", "E0", "E1_over_E0"), ("E2", "E1", "E2_over_E1")]:
            num_seeds = {s["seed"]: s["R_B"] for s in per_seed[(num_arm, eps_s)]}
            den_seeds = {s["seed"]: s["R_B"] for s in per_seed[(den_arm, eps_s)]}
            per_seed_ratios = [num_seeds[i] / den_seeds[i] for i in sorted(num_seeds)]
            ratio_of_means = agg[(num_arm, eps_s)]["R_B_mean"] / agg[(den_arm, eps_s)]["R_B_mean"]
            mean_of_ratios = sum(per_seed_ratios) / len(per_seed_ratios)
            std_of_ratios = (
                sum((x - mean_of_ratios) ** 2 for x in per_seed_ratios) / (len(per_seed_ratios) - 1)
            ) ** 0.5
            row[label] = {
                "ratio_of_seed_means": ratio_of_means,
                "mean_of_per_seed_ratios": mean_of_ratios,
                "std_of_per_seed_ratios": std_of_ratios,
                "per_seed_ratios": per_seed_ratios,
                "pct_change_ratio_of_means": (ratio_of_means - 1.0) * 100.0,
                "min_raw_qcd_mean_of_the_two_arms": min(
                    agg[(num_arm, eps_s)]["raw_qcd_mean"], agg[(den_arm, eps_s)]["raw_qcd_mean"]
                ),
            }
            row[label]["support_tier"] = tier_from_raw_qcd(row[label]["min_raw_qcd_mean_of_the_two_arms"])
        gain.append(row)

    return {"working_points": working_points, "mechanism_gain": gain}


# ---------------------------------------------------------------------------
# Track A: post_training_evaluation_20260815_v2 + mechanism_interpretation
# ---------------------------------------------------------------------------

def load_tracka() -> dict:
    with open(TRACK_A_EVALUATION_RESULTS_JSON) as f:
        ev = json.load(f)
    with open(TRACK_A_SUPPORTED_REJECTION_RANGE_TSV) as f:
        support_rows = list(csv.DictReader(f, delimiter="\t"))

    eps_s_grid = ev["efficiency_grid"]["epsilon_S"]

    def model_curve(model_key: str) -> list[dict]:
        stv = ev["models"][model_key]["seed_to_seed_variation"]
        out = []
        for eps_s in eps_s_grid:
            gk = f"rb_eps_S={eps_s}"
            out.append({
                "eps_S": eps_s,
                "R_B_mean": stv[f"{gk}_mean"],
                "R_B_std": stv[f"{gk}_std"],
                "per_seed": stv[f"{gk}_per_seed"],
            })
        return out

    dnn_curve = model_curve("dnn")
    tf_curve = model_curve("cmstransformer")

    # per-seed support flags straight from EVALUATION_RESULTS.json (all 3 seeds,
    # not just the primary seed) for the fixed-epsilon_S grid actually plotted.
    def support_by_eps_s(model_key: str) -> dict:
        out = {}
        for seed, seed_block in ev["models"][model_key]["per_seed"].items():
            for gk, row in seed_block["rb_grid"].items():
                eps_s = float(gk.split("=")[1])
                out.setdefault(eps_s, {})[seed] = {
                    "raw_background_rows": row["raw_background_rows"],
                    "qcd_raw_rows": row["qcd_raw_rows"],
                    "qcd_Neff": row["qcd_Neff"],
                    "scientifically_supportable": row["scientifically_supportable"],
                }
        return out

    dnn_support = support_by_eps_s("dnn")
    tf_support = support_by_eps_s("cmstransformer")

    # fixed-epsilon_B tail rows (primary seed 20260811 only) -- separate slicing,
    # used for the support-limits table and the figure-4 tail annotation, never
    # merged into the fixed-epsilon_S curve itself.
    eps_b_tail = {"dnn": [], "cmstransformer": []}
    for r in support_rows:
        if r["working_point_type"] != "fixed_epsilon_B":
            continue
        model = r["model"]
        eps_b_tail[model].append({
            "seed": int(r["seed"]),
            "target_epsilon_B": float(r["target_epsilon_B"]),
            "realized_epsilon_S": float(r["realized_epsilon_S"]),
            "realized_epsilon_B": float(r["realized_epsilon_B"]),
            "R_B": float(r["R_B_rejection"]),
            "raw_background_rows": int(r["raw_background_rows"]),
            "qcd_raw_rows": int(r["qcd_raw_rows"]),
            "qcd_Neff": float(r["qcd_Neff"]),
            "scientifically_supportable": r["scientifically_supportable"] == "True",
        })

    refs = ev["references"]

    fig3_models = [
        {
            "label": "Cut baseline\n" + r"($R_{HH}<34$)",
            "short": "cut_baseline",
            "R_B": refs["CUT_BASELINE_historical_RHH_lt_34"]["R_B_rejection_factor_1_over_epsilon_B"],
            "R_B_err": None,
            "kind": "frozen_reference_single_operating_point",
        },
        {
            "label": r"BDT CONTROL$_0$" + "\n(34-feat.)",
            "short": "bdt_control0",
            "R_B": refs["BDT_CONTROL_0"]["rb_eps_s_0.40_recomputed_here"],
            "R_B_err": None,
            "kind": "frozen_reference",
        },
        {
            "label": "Dense DNN",
            "short": "dnn",
            "R_B": ev["models"]["dnn"]["seed_to_seed_variation"]["rb_eps_S=0.4_mean"],
            "R_B_err": ev["models"]["dnn"]["seed_to_seed_variation"]["rb_eps_S=0.4_std"],
            "bootstrap_ci68": [
                ev["models"]["dnn"]["bootstrap_rb_eps_s_0.40_primary_seed"]["ci68_lo"],
                ev["models"]["dnn"]["bootstrap_rb_eps_s_0.40_primary_seed"]["ci68_hi"],
            ],
            "kind": "measured_this_benchmark",
        },
        {
            "label": "CMS-inspired\nfive-jet pairwise\nevent Transformer",
            "short": "cmstransformer",
            "R_B": ev["models"]["cmstransformer"]["seed_to_seed_variation"]["rb_eps_S=0.4_mean"],
            "R_B_err": ev["models"]["cmstransformer"]["seed_to_seed_variation"]["rb_eps_S=0.4_std"],
            "bootstrap_ci68": [
                ev["models"]["cmstransformer"]["bootstrap_rb_eps_s_0.40_primary_seed"]["ci68_lo"],
                ev["models"]["cmstransformer"]["bootstrap_rb_eps_s_0.40_primary_seed"]["ci68_hi"],
            ],
            "kind": "measured_this_benchmark",
        },
        {
            "label": "FiveJetParTNet",
            "short": "fivejetpartnet",
            "R_B": refs["FiveJetParTNet"]["rb_eps_s_0.40_frozen_reference"],
            "R_B_err": None,
            "kind": "frozen_reference_cited_not_recomputed",
        },
        {
            "label": r"BDT NEW$_C$" + "\n(55-feat.)",
            "short": "bdt_newc",
            "R_B": refs["BDT_NEW_C"]["rb_eps_s_0.40_recomputed_here"],
            "R_B_err": None,
            "kind": "frozen_reference",
        },
    ]

    return {
        "dnn_curve": dnn_curve,
        "cmstransformer_curve": tf_curve,
        "dnn_support_by_eps_s": {str(k): v for k, v in dnn_support.items()},
        "cmstransformer_support_by_eps_s": {str(k): v for k, v in tf_support.items()},
        "eps_b_tail": eps_b_tail,
        "paired_bootstrap": ev["paired_dnn_vs_transformer_bootstrap"],
        "fig3_models": fig3_models,
        "references_raw": refs,
    }


def main() -> None:
    data = {
        "provenance": {
            "track_a_eval_v2": str(TRACK_A_EVALUATION_RESULTS_JSON),
            "track_a_mechanism_interp": str(TRACK_A_SUPPORTED_REJECTION_RANGE_TSV),
            "track_b_working_points": str(TRACK_B_HOLDOUT_A_WORKING_POINTS_TSV),
            "track_b_seed_stability": str(TRACK_B_HOLDOUT_A_SEED_STABILITY_TSV),
            "note": (
                "Every field below is either a direct transcription of an "
                "already-frozen number, or a simple documented arithmetic "
                "derivation (ratio, tier classification) of already-frozen "
                "numbers. No model was trained, retrained, or evaluated to "
                "produce this file. holdout_A was not reopened; holdout_B, "
                "holdout_Q, inference_qcd_1, inference_qcd_2 were not opened."
            ),
        },
        "track_b": load_trackb(),
        "track_a": load_tracka(),
    }
    OUT_PATH.write_text(json.dumps(data, indent=2, default=str))
    print(f"wrote {OUT_PATH}")


if __name__ == "__main__":
    main()
