"""Additive, read-only paper-results snapshot builder.

Reads ONLY already-frozen, hash-verified DEVELOPMENT artifacts already
produced elsewhere in this project:
  - Step2j matched-400k comparison (BDT-K, BDT-KF, SPA-Net 2M, all
    scored on the identical 400,000-event cohort):
    track_b_harvey_bdt_working_points_20260818_v1/step2j_interim_10m_convergence_20260820_v1/matched_400k/results/K_KF_SPANET_MATCHED_400K_COMPARISON.json
  - HARVEY_SUMMARY (same directory) for BDT-K/BDT-KF training-arm
    metadata (n_train, round budget, early-stopping behavior).
  - Phase4AI 10M classification evaluation result (SPA-Net 10M primary):
    phase4AI_spanet_10M_scaling_seed0_20260821_v1/classification_evaluation_10M/work/classification_evaluation_10M_result.json
  - Phase4AI 2M-vs-10M scaling comparison:
    phase4AI_spanet_10M_scaling_seed0_20260821_v1/scaling_study_2M_vs_10M/scaling_comparison_2M_vs_10M.json

Does NOT retrain, rescore, or recompute any AUC/working-point number --
every number in the output is copied verbatim from a source JSON field,
with the exact (file, JSON-pointer-like field path) recorded in
PROVENANCE_MAP.tsv. Does NOT access any independent inference/test
population.

Normalizes the four models' working-point tables (which use slightly
different field-naming conventions in their own source files) into one
common schema:
  target_epsS, threshold, achieved_epsS,
  n_signal_pass, n_signal_total,
  R_all_background, epsB_all_background, n_all_background_pass, n_all_background_total,
  R_qcd, epsB_qcd, n_qcd_pass, n_qcd_total,
  R_ttbar, epsB_ttbar, n_ttbar_pass, n_ttbar_total,
  finite_support_warnings
"""
import csv
import json
import os

HH4B = "/uscms_data/d3/iturkmen/hh4b_delphes"
BDT_ROOT = os.path.join(HH4B, "track_b_harvey_bdt_working_points_20260818_v1")
MATCHED_400K = os.path.join(BDT_ROOT, "step2j_interim_10m_convergence_20260820_v1", "matched_400k")
PHASE4AI = os.path.join(HH4B, "track_b_phase4_preflight_20260812", "phase4AI_spanet_10M_scaling_seed0_20260821_v1")
PHASE4AF = os.path.join(HH4B, "track_b_phase4_preflight_20260812", "phase4AF_spanet_partial_events_population_correction_20260819_v1")

SRC_KKF_COMPARISON = os.path.join(MATCHED_400K, "results", "K_KF_SPANET_MATCHED_400K_COMPARISON.json")
SRC_SPANET_2M_NATIVE = os.path.join(PHASE4AF, "spanet_2M_seed0_classification_evaluation_v1", "work", "classification_evaluation_result.json")
SRC_SPANET_10M_PRIMARY = os.path.join(PHASE4AI, "classification_evaluation_10M", "work", "classification_evaluation_10M_result.json")
SRC_SPANET_10M_SECONDARY = os.path.join(PHASE4AI, "classification_evaluation_10M", "work", "classification_evaluation_10M_secondary_diagnostic_result.json")
SRC_SCALING_2M_10M = os.path.join(PHASE4AI, "scaling_study_2M_vs_10M", "scaling_comparison_2M_vs_10M.json")
SRC_10M_FULL_RUN = os.path.join(PHASE4AI, "full_run_attempt2_eaf_local_staging", "work", "training_10M_seed0_full_attempt2_result.json")
SRC_2M_TRAIN_RESULT = os.path.join(PHASE4AF, "spanet_2M_seed0_production_training_v1", "work", "training_2M_seed0_result.json")

OUT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MASTER_DIR = os.path.join(OUT_DIR, "master_results")


def load(path):
    with open(path) as f:
        return json.load(f)


def sha256_file(path):
    import hashlib
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


provenance_rows = []


def record_provenance(field_path, value, source_file, source_field_path):
    provenance_rows.append(dict(
        field=field_path, value=value,
        source_file=os.path.relpath(source_file, HH4B),
        source_sha256=sha256_file(source_file),
        source_field_path=source_field_path,
    ))


def normalize_working_points_from_kkf(entries, model_key):
    """K_KF_SPANET_MATCHED_400K_COMPARISON.json schema -> common schema."""
    out = []
    for e in entries:
        out.append(dict(
            target_epsS=e["nominal_epsS"], threshold=e["threshold"], achieved_epsS=e["achieved_signal_efficiency"],
            n_signal_pass=e["raw_surviving_signal"], n_signal_total=e["n_signal_total"],
            R_all_background=e["RB_1_over_epsB"], epsB_all_background=e["all_background_efficiency"],
            n_all_background_pass=e["raw_surviving_all_background"], n_all_background_total=e["n_background_total"],
            R_qcd=e["qcd_rejection"], epsB_qcd=e["qcd_efficiency"],
            n_qcd_pass=e["raw_surviving_qcd"], n_qcd_total=e["n_qcd_total"],
            R_ttbar=e["ttbar_rejection"], epsB_ttbar=e["ttbar_efficiency"],
            n_ttbar_pass=e["raw_surviving_ttbar"], n_ttbar_total=e["n_ttbar_total"],
            finite_support_warnings=e["finite_support_warnings"],
        ))
    return out


def normalize_working_points_from_spanet10m(entries):
    """classification_evaluation_10M_result.json schema -> common schema."""
    out = []
    for e in entries:
        # This 10M script's own working-point builder does not emit a
        # finite_support_warnings list; derive it the same way Step2j
        # does (zero-survivor / low-count-survivor disclosure), from
        # the raw counts themselves, so the 10M row is held to the
        # identical disclosure standard as the other three models.
        warnings = []
        if e["n_ttbar_pass"] == 0:
            warnings.append(
                f"ttbar: zero survivors at epsS={e['target_signal_efficiency']} "
                f"(threshold={e['threshold']:.6f}) - efficiency=0 exactly, rejection undefined "
                f"(only a lower bound RB > {e['n_ttbar_total']} is supported)"
            )
        elif e["n_ttbar_pass"] <= 10:
            warnings.append(
                f"ttbar: only {e['n_ttbar_pass']} survivors at epsS={e['target_signal_efficiency']} "
                "- efficiency/rejection is low-statistics (Poisson-limited), not a precise estimate"
            )
        out.append(dict(
            target_epsS=e["target_signal_efficiency"], threshold=e["threshold"], achieved_epsS=e["achieved_signal_efficiency"],
            n_signal_pass=e["n_signal_pass"], n_signal_total=e["n_signal_total"],
            R_all_background=e["all_background_rejection"], epsB_all_background=e["all_background_efficiency"],
            n_all_background_pass=e["n_all_background_pass"], n_all_background_total=e["n_all_background_total"],
            R_qcd=e["qcd_rejection"], epsB_qcd=e["qcd_efficiency"],
            n_qcd_pass=e["n_qcd_pass"], n_qcd_total=e["n_qcd_total"],
            R_ttbar=e["ttbar_rejection"], epsB_ttbar=e["ttbar_efficiency"],
            n_ttbar_pass=e["n_ttbar_pass"], n_ttbar_total=e["n_ttbar_total"],
            finite_support_warnings=warnings,
        ))
    return out


def main():
    kkf = load(SRC_KKF_COMPARISON)
    spanet_2m_native = load(SRC_SPANET_2M_NATIVE)
    spanet_10m = load(SRC_SPANET_10M_PRIMARY)
    spanet_10m_secondary = load(SRC_SPANET_10M_SECONDARY)
    scaling = load(SRC_SCALING_2M_10M)
    run10m = load(SRC_10M_FULL_RUN)
    run2m = load(SRC_2M_TRAIN_RESULT)

    assert kkf["n_rows"] == 400000
    assert spanet_2m_native["population"]["n_scored"] == 400000
    assert spanet_10m["n_scored"] == 400000
    # Cohort-identity cross-check: the matched-400k K/KF/SPANET comparison and the
    # 2M-native evaluation must describe the exact same physical cohort.
    assert kkf["process_counts"] == dict(
        signal=spanet_2m_native["population"]["n_signal"],
        qcd=spanet_2m_native["population"]["n_qcd"],
        ttbar=spanet_2m_native["population"]["n_ttbar"],
    )
    assert kkf["auc"]["SPANET"]["vs_all_background"] == spanet_2m_native["roc_auc"]["signal_vs_all_background"]

    models = {}

    models["BDT_K"] = dict(
        model_id="BDT_K", display_name="BDT-K",
        description="Gradient-boosted decision tree, kinematics-only feature set (52 features). Interim, not the final 144M-population fit.",
        training_population=dict(n_train=8377425, n_val=2096600, source="interim ~10.47M-event staged population"),
        feature_definition="52 kinematic features (jet 4-vectors / derived kinematic observables; no flavor-tag information).",
        seed=0, training_budget="6000 boosting rounds (hard cap); early-stopping patience=50 rounds configured but NOT triggered -- terminated on the round cap, not the early-stop rule.",
        checkpoint_criterion="best_iteration by validation logloss (best_iteration=5998); model did not early-stop, effectively plateaued in its final ~15 rounds per HARVEY_SUMMARY.",
        auc=dict(all_background=kkf["auc"]["K"]["vs_all_background"], qcd=kkf["auc"]["K"]["vs_qcd"], ttbar=kkf["auc"]["K"]["vs_ttbar"]),
        working_points=normalize_working_points_from_kkf(kkf["working_points"]["K"], "K"),
        model_sha256=kkf["K_model_sha256"],
    )
    models["BDT_KF"] = dict(
        model_id="BDT_KF", display_name="BDT-KF",
        description="Gradient-boosted decision tree, kinematics + Sophon AK4 flavor-tag probabilities (82 features = K's 52 + 30 additional probB/probC/probL channels). Interim, not the final 144M-population fit.",
        training_population=dict(n_train=8377425, n_val=2096600, source="interim ~10.47M-event staged population"),
        feature_definition="82 features: K's 52 kinematic features plus 3 Sophon AK4 flavor-tag probability channels (probB/probC/probL) per jet slot.",
        seed=0, training_budget="6000 boosting rounds (hard cap); early-stopping patience=50 rounds configured but NOT triggered -- terminated on the round cap, not the early-stop rule.",
        checkpoint_criterion="best_iteration by validation logloss (best_iteration=5999); model did not early-stop, effectively plateaued in its final ~15 rounds per HARVEY_SUMMARY.",
        auc=dict(all_background=kkf["auc"]["KF"]["vs_all_background"], qcd=kkf["auc"]["KF"]["vs_qcd"], ttbar=kkf["auc"]["KF"]["vs_ttbar"]),
        working_points=normalize_working_points_from_kkf(kkf["working_points"]["KF"], "KF"),
        model_sha256=kkf["KF_model_sha256"],
    )
    models["SPANET_2M"] = dict(
        model_id="SPANET_2M", display_name="SPA-Net 2M (primary)",
        description="SPA-Net v2.3 joint jet-assignment + event-classification model, seed 0, trained on 2,000,000 events. Primary checkpoint selected by maximum validation_average_jet_accuracy.",
        training_population=dict(n_train=run2m["n_train_events"], n_val=run2m["n_val_events"], source="frozen 2M-scale production HDF5"),
        feature_definition="SPA-Net official-schema per-jet kinematic input tensors (up to 10 AK4 jets), joint assignment+classification architecture (hidden_dim=32, transformer_dim=128, 8 encoder layers).",
        seed=run2m["seed"], training_budget=f"{run2m['max_epochs']} epochs, batch_size={run2m['batch_size']}",
        checkpoint_criterion=f"PRIMARY = maximum validation_average_jet_accuracy (best epoch {run2m['primary_checkpoint_best_epoch']}, score {run2m['primary_checkpoint_best_score']}).",
        auc=dict(all_background=spanet_2m_native["roc_auc"]["signal_vs_all_background"], qcd=spanet_2m_native["roc_auc"]["signal_vs_qcd"], ttbar=spanet_2m_native["roc_auc"]["signal_vs_ttbar"]),
        working_points=normalize_working_points_from_kkf(kkf["working_points"]["SPANET"], "SPANET"),
        checkpoint_sha256=run2m["primary_checkpoint_sha256"],
    )
    models["SPANET_10M"] = dict(
        model_id="SPANET_10M", display_name="SPA-Net 10M (primary)",
        description="SPA-Net v2.3 joint jet-assignment + event-classification model, seed 0, trained on 10,000,000 events (same architecture/seed/batch/epoch-budget as SPANET_2M). Primary checkpoint selected by maximum validation_average_jet_accuracy.",
        training_population=dict(n_train=run10m["n_train_events"], n_val=run10m["n_val_events"], source="frozen 10M-scale production HDF5 (nested superset of the 2M population)"),
        feature_definition="Identical to SPANET_2M -- same SPA-Net official-schema per-jet kinematic input tensors and architecture, only training-population size differs.",
        seed=run10m["seed"], training_budget=f"{run10m['max_epochs']} epochs, batch_size={run10m['batch_size']}",
        checkpoint_criterion=f"PRIMARY = maximum validation_average_jet_accuracy (best epoch {run10m['primary_checkpoint_best_epoch']}, score {run10m['primary_checkpoint_best_score']}).",
        auc=dict(all_background=spanet_10m["roc_auc"]["signal_vs_all_background"], qcd=spanet_10m["roc_auc"]["signal_vs_qcd"], ttbar=spanet_10m["roc_auc"]["signal_vs_ttbar"]),
        working_points=normalize_working_points_from_spanet10m(spanet_10m["working_points"]),
        checkpoint_sha256=spanet_10m["checkpoint_sha256"],
        secondary_checkpoint_diagnostic=dict(
            note="Diagnostic only -- does not change checkpoint selection.",
            epoch=49, checkpoint_sha256=spanet_10m_secondary["checkpoint_sha256"],
            auc=dict(all_background=spanet_10m_secondary["roc_auc"]["signal_vs_all_background"],
                      qcd=spanet_10m_secondary["roc_auc"]["signal_vs_qcd"],
                      ttbar=spanet_10m_secondary["roc_auc"]["signal_vs_ttbar"]),
        ),
    )

    scaling_table = dict(
        run_2M=dict(
            primary_best_epoch=scaling["run_2M"]["primary_checkpoint_best_epoch"],
            validation_average_jet_accuracy=scaling["run_2M"]["primary_checkpoint_best_score"],
            secondary_best_val_loss_total_loss=scaling["run_2M"]["secondary_checkpoint_best_score"],
            total_fit_wall_s=scaling["run_2M"]["total_fit_wall_s"],
            events_per_s_train_only=scaling["run_2M"]["events_per_s_train_only"],
            gpu_peak_reserved_bytes=scaling["run_2M"]["gpu_peak_reserved_bytes"],
            n_train_events=scaling["run_2M"]["n_train_events"],
        ),
        run_10M=dict(
            primary_best_epoch=scaling["run_10M"]["primary_checkpoint_best_epoch"],
            validation_average_jet_accuracy=scaling["run_10M"]["primary_checkpoint_best_score"],
            secondary_best_val_loss_total_loss=scaling["run_10M"]["secondary_checkpoint_best_score"],
            total_fit_wall_s=scaling["run_10M"]["total_fit_wall_s"],
            events_per_s_train_only=scaling["run_10M"]["events_per_s_train_only"],
            gpu_peak_reserved_bytes=scaling["run_10M"]["gpu_peak_reserved_bytes"],
            n_train_events=scaling["run_10M"]["n_train_events"],
        ),
        classification_auc_comparison=dict(
            spanet_2M=models["SPANET_2M"]["auc"], spanet_10M=models["SPANET_10M"]["auc"],
            note="2M->10M does NOT produce a clear event-classification AUC improvement (all-background/QCD AUC essentially flat to slightly down; ttbar AUC up ~0.002) despite 10M's slightly higher validation_average_jet_accuracy (a JET-ASSIGNMENT metric, not a classification metric).",
        ),
        identity_checks=scaling["architecture_identity_check"],
    )

    master = dict(
        snapshot_label="TRACK_B_DEVELOPMENT_PAPER_SNAPSHOT -- DEVELOPMENT ONLY, NOT A FINAL/GOVERNING RESULT, NOT INDEPENDENT STAGE-C INFERENCE, NOT PHYSICALLY NORMALIZED",
        cohort=dict(
            n_events=400000, n_signal=kkf["process_counts"]["signal"], n_qcd=kkf["process_counts"]["qcd"], n_ttbar=kkf["process_counts"]["ttbar"],
            description="Fixed training-production development cohort, identical across all four models (BDT-K, BDT-KF, SPA-Net 2M, SPA-Net 10M), matched by hash-gated physical-event provenance chain (this project's build_event_provenance_bridge.py feeding the BDT study's build_matched_400k_features.py).",
            cohort_source_npz_sha256=kkf["npz_sha256"],
        ),
        models=models,
        scaling_study_2M_vs_10M=scaling_table,
        caveats=[
            "This is a DEVELOPMENT matched-event comparison on the training-production development population, NOT the sealed independent Stage-C inference/test population.",
            "BDT-K and BDT-KF may have encountered some or all of these 400,000 physical events during their own train/val split; no leakage control was applied for this comparison.",
            "BDT-K/BDT-KF are interim ~10.47M-population models that hit a 6000-round hard cap without early-stopping triggering -- not the final 144M-population fit.",
            "The Stage-C independent-inference contract has not been designed or previewed anywhere in this project.",
            "No physical (cross-section/luminosity/generator-weight) normalization has been applied anywhere in this snapshot.",
            "ttbar-tail working points (epsS<=0.25) are finite-support/Poisson-limited for all four models; see per-working-point finite_support_warnings.",
            "No final model ranking should be claimed from this snapshot.",
        ],
    )

    os.makedirs(MASTER_DIR, exist_ok=True)
    with open(os.path.join(MASTER_DIR, "MASTER_DEVELOPMENT_RESULTS.json"), "w") as f:
        json.dump(master, f, indent=2, default=str)

    # ---- Flat CSV: one row per (model, working_point) plus header-level AUC rows ----
    csv_rows = []
    for mid, m in models.items():
        csv_rows.append(dict(
            model_id=mid, row_type="AUC", target_epsS="", threshold="", achieved_epsS="",
            auc_all_background=m["auc"]["all_background"], auc_qcd=m["auc"]["qcd"], auc_ttbar=m["auc"]["ttbar"],
            R_all_background="", R_qcd="", R_ttbar="",
            n_signal_pass="", n_signal_total="", n_all_background_pass="", n_all_background_total="",
            n_qcd_pass="", n_qcd_total="", n_ttbar_pass="", n_ttbar_total="", finite_support_warnings="",
        ))
        for wp in m["working_points"]:
            csv_rows.append(dict(
                model_id=mid, row_type="working_point", target_epsS=wp["target_epsS"], threshold=wp["threshold"], achieved_epsS=wp["achieved_epsS"],
                auc_all_background="", auc_qcd="", auc_ttbar="",
                R_all_background=wp["R_all_background"], R_qcd=wp["R_qcd"], R_ttbar=wp["R_ttbar"],
                n_signal_pass=wp["n_signal_pass"], n_signal_total=wp["n_signal_total"],
                n_all_background_pass=wp["n_all_background_pass"], n_all_background_total=wp["n_all_background_total"],
                n_qcd_pass=wp["n_qcd_pass"], n_qcd_total=wp["n_qcd_total"],
                n_ttbar_pass=wp["n_ttbar_pass"], n_ttbar_total=wp["n_ttbar_total"],
                finite_support_warnings=" | ".join(wp["finite_support_warnings"]),
            ))
    csv_path = os.path.join(MASTER_DIR, "MASTER_DEVELOPMENT_RESULTS.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(csv_rows[0].keys()))
        w.writeheader()
        w.writerows(csv_rows)

    # ---- Provenance map: source path + sha256 for every source file used ----
    src_files = [SRC_KKF_COMPARISON, SRC_SPANET_2M_NATIVE, SRC_SPANET_10M_PRIMARY,
                 SRC_SPANET_10M_SECONDARY, SRC_SCALING_2M_10M, SRC_10M_FULL_RUN, SRC_2M_TRAIN_RESULT]
    prov_path = os.path.join(MASTER_DIR, "PROVENANCE_MAP.tsv")
    with open(prov_path, "w") as f:
        f.write("output_field\tsource_file\tsource_sha256\n")
        field_to_src = {
            "models.BDT_K.auc, models.BDT_K.working_points, cohort.*": SRC_KKF_COMPARISON,
            "models.BDT_KF.auc, models.BDT_KF.working_points": SRC_KKF_COMPARISON,
            "models.SPANET_2M.auc, models.SPANET_2M.working_points (cross-checked against native)": SRC_KKF_COMPARISON,
            "models.SPANET_2M.* (native cross-check, population reconstruction)": SRC_SPANET_2M_NATIVE,
            "models.SPANET_10M.auc, models.SPANET_10M.working_points, checkpoint_sha256": SRC_SPANET_10M_PRIMARY,
            "models.SPANET_10M.secondary_checkpoint_diagnostic": SRC_SPANET_10M_SECONDARY,
            "scaling_study_2M_vs_10M.*": SRC_SCALING_2M_10M,
            "models.SPANET_10M.training_population, training_budget, checkpoint_criterion": SRC_10M_FULL_RUN,
            "models.SPANET_2M.training_population, training_budget, checkpoint_criterion": SRC_2M_TRAIN_RESULT,
        }
        for field, src in field_to_src.items():
            f.write(f"{field}\t{os.path.relpath(src, HH4B)}\t{sha256_file(src)}\n")

    print(json.dumps(dict(
        models_written=list(models.keys()),
        n_working_points_per_model={k: len(v["working_points"]) for k, v in models.items()},
        output_json=os.path.join(MASTER_DIR, "MASTER_DEVELOPMENT_RESULTS.json"),
        output_csv=csv_path, provenance_map=prov_path,
    ), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
