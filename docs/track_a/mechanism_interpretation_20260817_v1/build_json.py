#!/usr/bin/env python3
"""Builds TRACKA_MECHANISM_INTERPRETATION_20260817.json as a machine-readable twin of the .md,
read-only against post_training_evaluation_20260815_v2's already-validated artifacts. No new
computation of any score/metric -- values are transcribed (and spot-checked) from that package."""
import json
from pathlib import Path

V2 = Path("/uscms_data/d3/iturkmen/hh4b_delphes/track_a_mechanism_benchmark_20260814_v1/post_training_evaluation_20260815_v2")
OUT = Path(__file__).parent

eval_results = json.loads((V2 / "EVALUATION_RESULTS.json").read_text())

doc = {
    "naming_discipline": {
        "custom_model_correct_name": "CMS-inspired five-jet pairwise event Transformer",
        "custom_model_source": "model_literature_freeze_20260813_v1/models/cms_inspired_fivejet_pairwise_event_transformer.py",
        "class_name": "CMSInspiredFiveJetPairwiseEventTransformer",
        "forbidden_names": ["SPA-Net", "canonical ParT", "official CMS reproduction"],
        "documented_cms_facts_matched": [
            "five leading b-tagged jets as tokens",
            "pairwise jet-pair features modify self-attention",
            "eight Transformer blocks",
            "separate 3-hidden-layer event-feature encoder",
            "common 3-hidden-layer decoder",
        ],
        "undocumented_hyperparameters_tagged": "PROJECT_CHOICE_NOT_CMS in model source",
    },
    "cms_reference_not_a_target": {
        "cms_hig24010_public_claim": "approximately two orders of magnitude additional background "
            "rejection at ~40% signal efficiency, three orders at ~10%",
        "commensurable_with_this_benchmark": False,
        "differences": ["simulation", "tagger", "background_model", "sample_support",
                          "model_implementation", "physical_normalization"],
    },
    "measured_comparison_eps_S_0.40": {
        "cut_baseline_historical_RHH_lt_34": {
            "status": "measured_single_fixed_point",
            "R_B": eval_results["references"]["CUT_BASELINE_historical_RHH_lt_34"]["R_B_rejection_factor_1_over_epsilon_B"],
        },
        "BDT_CONTROL_0": {"status": "measured", "auc": eval_results["references"]["BDT_CONTROL_0"]["auc"],
                            "R_B": eval_results["references"]["BDT_CONTROL_0"]["rb_eps_s_0.40_recomputed_here"]},
        "dense_DNN_this_benchmark": {
            "status": "measured_5fold_oof_3seed",
            "auc_seed_mean": eval_results["models"]["dnn"]["seed_to_seed_variation"]["auc_mean"],
            "auc_seed_std": eval_results["models"]["dnn"]["seed_to_seed_variation"]["auc_std"],
            "R_B_seed_mean": eval_results["models"]["dnn"]["seed_to_seed_variation"]["rb_eps_S=0.4_mean"],
            "R_B_seed_std": eval_results["models"]["dnn"]["seed_to_seed_variation"]["rb_eps_S=0.4_std"],
        },
        "CMS_inspired_transformer_this_benchmark": {
            "status": "measured_5fold_oof_3seed",
            "auc_seed_mean": eval_results["models"]["cmstransformer"]["seed_to_seed_variation"]["auc_mean"],
            "auc_seed_std": eval_results["models"]["cmstransformer"]["seed_to_seed_variation"]["auc_std"],
            "R_B_seed_mean": eval_results["models"]["cmstransformer"]["seed_to_seed_variation"]["rb_eps_S=0.4_mean"],
            "R_B_seed_std": eval_results["models"]["cmstransformer"]["seed_to_seed_variation"]["rb_eps_S=0.4_std"],
        },
        "FiveJetParTNet": {"status": "cited_frozen_reference_not_retrained",
                             "auc": eval_results["references"]["FiveJetParTNet"]["auc_frozen_reference"],
                             "R_B": eval_results["references"]["FiveJetParTNet"]["rb_eps_s_0.40_frozen_reference"]},
        "BDT_NEW_C": {"status": "measured", "auc": eval_results["references"]["BDT_NEW_C"]["auc"],
                       "R_B": eval_results["references"]["BDT_NEW_C"]["rb_eps_s_0.40_recomputed_here"]},
        "categorized_BDT": {"status": "UNAVAILABLE"},
        "LBN": {"status": "UNAVAILABLE"},
        "point_estimate_ordering": "cut(1.9496) << CONTROL_0(13.206) < DNN(13.293) < "
            "transformer(14.004) < FiveJetParTNet(14.154) < NEW_C(14.707)",
        "only_dnn_and_transformer_have_computed_confidence_intervals": True,
        "no_comparison_to_control0_newc_partnet_outside_eps_s_0.40": True,
    },
    "unavailable_comparisons": {
        "categorized_BDT": {
            "status": "UNAVAILABLE_AS_COMPARABLE",
            "candidates_audited": 3,
            "reason": "two contractually specified but never trained; one trained on a "
                      "structurally different ~31,225-row physical-significance-optimization "
                      "population, not this benchmark's 1,042,397-row TRAIN population",
        },
        "LBN": {
            "status": "UNAVAILABLE",
            "reason": "contractually specified (lbn_dnn, hh4b_train_fold_and_baseline_benchmark_"
                      "contract_20260805_v1) intending this same population/folds/weights, but "
                      "never trained -- no checkpoint or output artifact exists",
        },
    },
    "three_separable_limitations": {
        "classifier_capacity": "dense DNN (14,849 params, no pairwise/5th-jet input) underperforms "
            "the transformer (426,697 params, pairwise-attention-bias) and NEW_C (engineered "
            "pairwise features) at eps_S=0.40 -- consistent with, not proof of, a capacity/"
            "inductive-bias gap",
        "finite_qcd_support": "both newly-trained models hit the identical qcd_Neff<10 floor at the "
            "identical working point (eps_B=0.001) -- a property of the shared QCD Monte Carlo "
            "sample's finite statistics, not of either classifier",
        "representation_richness": "Track A's binary Delphes b-tag and single-inclusive-QCD "
            "background model cannot test whether continuous flavor information or a richer "
            "background model would add further separation on top of the pairwise-attention "
            "mechanism already measured here -- this is the question handed to Track B",
    },
    "max_supportable_rejection": {
        "support_gate": "raw_background_rows>=100 AND qcd_raw_rows>=10 AND qcd_Neff>=10",
        "deepest_fixed_epsilon_B_supported": {"epsilon_B": 0.01, "R_B_approx": 100.0,
            "qcd_Neff_dnn": 41.693368, "qcd_Neff_transformer": 50.920341},
        "deepest_fixed_epsilon_S_supported": {"epsilon_S": 0.10,
            "R_B_dnn": 172.701688, "R_B_transformer": 167.151212,
            "qcd_Neff_dnn": 19.796310, "qcd_Neff_transformer": 28.302882},
        "first_unsupported_point": {"epsilon_B": 0.001, "R_B_approx": 1000.0,
            "qcd_Neff_dnn": 2.190317, "qcd_Neff_transformer": 6.558152,
            "scientifically_supportable": False},
        "no_tail_score_converted_to_sensitivity_claim_beyond_support_floor": True,
    },
    "paired_dnn_vs_transformer_bootstrap": eval_results["paired_dnn_vs_transformer_bootstrap"],
    "cms_sensitivity_caveat": "Track A has never computed a CMS-comparable expected-signal-strength "
        "limit (blocked on 4b SR Monte Carlo statistics, unrelated to classifier choice). Even if "
        "attempted, falling short of CMS's public expected limit would not by itself diagnose a "
        "classifier failure, since CMS's full sensitivity includes trigger efficiency, flavor "
        "tagging, mass regression, a validated data-driven background model, signal-region "
        "categorization, and the full systematic treatment -- none of which this classifier-only "
        "mechanism study attempts to reproduce.",
    "resource_summary": {
        "dnn_total_wallclock_hours": 0.9282501254091201,
        "cmstransformer_total_wallclock_hours": 25.166869984959185,
        "dnn_parameters": 14849,
        "cmstransformer_parameters": 426697,
        "speed_ratio_transformer_slower_than_dnn": 25.166869984959185 / 0.9282501254091201,
        "param_ratio_transformer_vs_dnn": 426697 / 14849,
    },
    "scope_discipline": {
        "no_new_training": True, "no_track_b_events": True, "no_cms_collision_data": True,
        "no_threshold_tuning": True, "no_architecture_change": True,
        "no_commit_or_push": True,
    },
    "source_package": "post_training_evaluation_20260815_v2 (validated, 25/25 checks, 0 mismatches; "
        "supersedes v1 per that package's own CORRECTIONS_FROM_V1.md)",
}

(OUT / "TRACKA_MECHANISM_INTERPRETATION_20260817.json").write_text(json.dumps(doc, indent=2, default=str))
print("wrote TRACKA_MECHANISM_INTERPRETATION_20260817.json")
