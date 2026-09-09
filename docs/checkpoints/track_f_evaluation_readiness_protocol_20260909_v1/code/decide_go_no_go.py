#!/usr/bin/env python3
"""Applies PREREGISTRATION.md section 7's frozen GO/NO-GO rule, mechanically,
to one or more comparison_result.json files (one per TEST seed, produced by
evaluate_multi_model.py). Never hand-edit a decision this script would
produce -- if the rule needs to change, amend PREREGISTRATION.md section 7
(dated, in the Amendments section) and this script together, then rerun.

Usage:
    python3 decide_go_no_go.py --seed-result seed0=comparison_result_seed0.json \\
        [--seed-result seed1=comparison_result_seed1.json ...] \\
        --out-json go_no_go_decision.json

The first --seed-result given is treated as "the original seed" for
PREREGISTRATION.md section 7.1's initial decision. Order of subsequent
--seed-result flags does not otherwise matter.
"""
import argparse
import json

MORE_SEEDS_EFFECT_SIZE_FLOOR = 0.005  # PREREGISTRATION.md section 7.2


def load_seed(label, path):
    with open(path) as f:
        r = json.load(f)
    flags = r["story_evaluation"]["single_seed_flags"]
    delta = r["paired_bootstrap_delta_auc"]["all_background"]
    return {
        "label": label, "path": path,
        "sig_class": flags["sig_class"], "sig_reco": flags["sig_reco"],
        "harm_class": flags["harm_class"], "harm_reco": flags["harm_reco"],
        "underpowered": flags["underpowered"],
        "delta_auc_all_background_point_estimate": delta["observed_delta_test_minus_control"],
        "delta_auc_all_background_ci": [delta["ci_low"], delta["ci_high"]],
    }


def more_seeds_decision(original):
    """PREREGISTRATION.md section 7.1. Evaluated from the ORIGINAL (first) seed only."""
    if original["harm_class"] or original["harm_reco"]:
        return {"decision": "NO_GO_INVESTIGATE_HARM",
                "reason": f"original seed shows statistically supported harm "
                          f"(harm_class={original['harm_class']}, harm_reco={original['harm_reco']}); "
                          "investigate before running any further seed, per section 7.1."}
    if original["sig_class"] or original["sig_reco"]:
        return {"decision": "GO_RUN_2_ADDITIONAL_SEEDS",
                "reason": f"original seed shows an improvement signal with no harm "
                          f"(sig_class={original['sig_class']}, sig_reco={original['sig_reco']}); "
                          "a single seed cannot establish robustness -- replicate before further commitment."}
    if original["underpowered"]:
        return {"decision": "INCONCLUSIVE_RUN_1_ADDITIONAL_SEED",
                "reason": "original seed resolved neither a significant effect nor a well-powered null "
                          "(CI half-width > 2x observed |delta|); run exactly one additional seed and "
                          "re-apply this rule to the 2-seed pooled estimate, per section 7.1's bounded response."}
    return {"decision": "NO_GO_STORY_C",
            "reason": "original seed shows a well-powered null (neither sig_class/sig_reco nor underpowered) "
                      "-- Story C, controlled negative result; do not run further ParT2M seeds chasing a flip."}


def additional8m_decision(seeds):
    """PREREGISTRATION.md section 7.2. Reachable only once >=3 total seeds are evaluated
    (the original seed plus the 2 additional seeds section 7.1's GO branch calls for)."""
    if len(seeds) < 3:
        return {"reachable": False, "decision": "NOT_REACHABLE_YET", "n_agree": None,
                "median_delta_auc_all_background": None,
                "reason": f"only {len(seeds)} seed(s) evaluated; the additional8M decision requires "
                          "the 3-seed replication step (section 7.1's GO branch) to have completed first."}

    original_sign = seeds[0]["delta_auc_all_background_point_estimate"] >= 0
    n_agree = 0
    for s in seeds:
        clean_signal = (s["sig_class"] or s["sig_reco"]) and not (s["harm_class"] or s["harm_reco"])
        same_direction = (s["delta_auc_all_background_point_estimate"] >= 0) == original_sign
        if clean_signal and same_direction:
            n_agree += 1

    any_harm = any(s["harm_class"] or s["harm_reco"] for s in seeds)
    deltas = sorted(s["delta_auc_all_background_point_estimate"] for s in seeds)
    median_delta = deltas[len(deltas) // 2] if len(deltas) % 2 == 1 else \
        (deltas[len(deltas) // 2 - 1] + deltas[len(deltas) // 2]) / 2.0

    if any_harm:
        return {"reachable": True, "decision": "DO_NOT_AUTHORIZE", "n_agree": n_agree,
                "median_delta_auc_all_background": median_delta,
                "reason": "at least one evaluated seed shows statistically supported harm -- do not authorize "
                          "regardless of agreement/magnitude among the others."}
    if n_agree >= 2 and median_delta >= MORE_SEEDS_EFFECT_SIZE_FLOOR:
        return {"reachable": True, "decision": "AUTHORIZE", "n_agree": n_agree,
                "median_delta_auc_all_background": median_delta,
                "reason": f"n_agree={n_agree}/{len(seeds)} seeds show a clean same-direction improvement signal "
                          f"and median delta-AUC={median_delta:.5f} >= {MORE_SEEDS_EFFECT_SIZE_FLOOR} "
                          "(the predeclared floor, ~2.5x this project's own measured native 2M->10M "
                          "scaling noise ceiling of ~0.002 AUC)."}
    return {"reachable": True, "decision": "DO_NOT_AUTHORIZE", "n_agree": n_agree,
            "median_delta_auc_all_background": median_delta,
            "reason": f"n_agree={n_agree}/{len(seeds)} (need >=2) and/or median delta-AUC={median_delta:.5f} "
                      f"< {MORE_SEEDS_EFFECT_SIZE_FLOOR} floor -- resource-aware bar not cleared; "
                      "do not authorize the ~9.7-day additional8M production."}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed-result", action="append", required=True,
                     help="label=path.json, repeatable; first occurrence is the 'original' seed")
    ap.add_argument("--out-json", required=True)
    args = ap.parse_args()

    seeds = []
    for entry in args.seed_result:
        label, _, path = entry.partition("=")
        if not path:
            raise SystemExit(f"--seed-result must be label=path.json, got: {entry!r}")
        seeds.append(load_seed(label, path))

    decision = {
        "_label": "GO_NO_GO_DECISION -- mechanically derived from PREREGISTRATION.md section 7, do not hand-edit",
        "n_seeds_evaluated": len(seeds),
        "per_seed": [
            {"seed": i, **{k: v for k, v in s.items() if k not in ("label", "path")}}
            for i, s in enumerate(seeds)
        ],
        "more_seeds_decision": more_seeds_decision(seeds[0]),
        "additional8m_decision": additional8m_decision(seeds),
    }

    with open(args.out_json, "w") as f:
        json.dump(decision, f, indent=2)
    print(json.dumps(decision, indent=2))
    print(f"Wrote {args.out_json}")


if __name__ == "__main__":
    main()
