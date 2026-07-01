from pathlib import Path
import pandas as pd

outdir = Path("outputs/final_summaries")
outdir.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------
# 1. Corrected BDT summary
# ---------------------------------------------------------------------

bdt_results = pd.DataFrame([
    {
        "model": "Lepton/MET BDT",
        "selection_type": "best validation threshold",
        "threshold": 0.440,
        "test_raw_auc": 0.706,
        "test_weighted_auc": 0.676,
        "test_average_precision": 0.076,
        "S_w": 9.445,
        "B_w": 1.983e6,
        "S_over_B": 4.76e-6,
        "asimov_Z_A": 0.00671,
        "N_eff_bkg": 30.07,
        "comment": "Broad, not high-purity; useful diagnostic but not strongest nominal baseline.",
    },
    {
        "model": "Simple-topology BDT",
        "selection_type": "best validation threshold",
        "threshold": 0.655,
        "test_raw_auc": 0.700,
        "test_weighted_auc": 0.672,
        "test_average_precision": 0.076,
        "S_w": 1.793,
        "B_w": 1.639e5,
        "S_over_B": 1.09e-5,
        "asimov_Z_A": 0.00443,
        "N_eff_bkg": 6.18,
        "comment": "Best-validation threshold is unstable on test; use score categories/stable thresholds for interpretation.",
    },
    {
        "model": "Hadronic-W/top BDT",
        "selection_type": "best validation threshold",
        "threshold": 0.740,
        "test_raw_auc": 0.711,
        "test_weighted_auc": 0.691,
        "test_average_precision": 0.081,
        "S_w": 0.0598,
        "B_w": 961.7,
        "S_over_B": 6.22e-5,
        "asimov_Z_A": 0.00193,
        "N_eff_bkg": 3.14,
        "comment": "High apparent purity but too statistically unstable in the high-score tail.",
    },
    {
        "model": "Simple-topology BDT",
        "selection_type": "stable broad threshold",
        "threshold": 0.275,
        "test_raw_auc": 0.700,
        "test_weighted_auc": 0.672,
        "test_average_precision": 0.076,
        "S_w": 13.092,
        "B_w": 3.753e6,
        "S_over_B": 3.49e-6,
        "asimov_Z_A": 0.00676,
        "N_eff_bkg": 58.68,
        "comment": "Stable broad corrected-recoMET reference region.",
    },
    {
        "model": "Simple-topology BDT",
        "selection_type": "stable high-purity threshold",
        "threshold": 0.685,
        "test_raw_auc": 0.700,
        "test_weighted_auc": 0.672,
        "test_average_precision": 0.076,
        "S_w": 0.897,
        "B_w": 1.924e4,
        "S_over_B": 4.66e-5,
        "asimov_Z_A": 0.00646,
        "N_eff_bkg": 25.96,
        "comment": "Higher-purity stable-ish region; lower acceptance than broad threshold.",
    },
])

bdt_results.to_csv(outdir / "bbww_final_bdt_corrected_results.csv", index=False)


# ---------------------------------------------------------------------
# 2. BDT score categories
# ---------------------------------------------------------------------

bdt_categories = pd.DataFrame([
    {
        "bdt_category": "<0.625",
        "S_w": 21.64,
        "B_w": 1.061e7,
        "S_over_B": 2.04e-6,
        "asimov_Z_A": 0.00664,
        "N_eff_bkg": 104.81,
        "comment": "Background-like region; large background and very low purity.",
    },
    {
        "bdt_category": "0.625-0.70",
        "S_w": 2.39,
        "B_w": 1.516e5,
        "S_over_B": 1.58e-5,
        "asimov_Z_A": 0.00614,
        "N_eff_bkg": 20.23,
        "comment": "Mildly signal-like region; S/B improves but effective background statistics are modest.",
    },
    {
        "bdt_category": "0.70-0.75",
        "S_w": 2.63,
        "B_w": 5.242e4,
        "S_over_B": 5.02e-5,
        "asimov_Z_A": 0.01149,
        "N_eff_bkg": 65.68,
        "comment": "Signal-like region with better purity and stable effective background statistics.",
    },
    {
        "bdt_category": ">=0.75",
        "S_w": 3.89,
        "B_w": 3.609e4,
        "S_over_B": 1.08e-4,
        "asimov_Z_A": 0.02045,
        "N_eff_bkg": 47.77,
        "comment": "Most signal-enriched BDT category; best S/B and best approximate Z_A.",
    },
    {
        "bdt_category": "Inclusive >0.625",
        "S_w": 8.91,
        "B_w": 2.401e5,
        "S_over_B": 3.70e-5,
        "asimov_Z_A": 0.01817,
        "N_eff_bkg": 47.69,
        "comment": "Broader signal-like region; more acceptance but lower purity than the >=0.75 bin.",
    },
])

bdt_categories.to_csv(outdir / "bbww_final_bdt_categories.csv", index=False)


# ---------------------------------------------------------------------
# 3. DNN baseline summary
# ---------------------------------------------------------------------

dnn_baselines = pd.DataFrame([
    {
        "model": "Binary tabular DNN",
        "input_type": "flat engineered event variables",
        "test_raw_auc": 0.655,
        "test_weighted_auc": 0.609,
        "test_average_precision": None,
        "test_Z_A": 0.00583,
        "test_N_eff_bkg": 37.68,
        "comment": "Learns some ranking but weaker than corrected BDT.",
    },
    {
        "model": "Small-regularized DNN",
        "input_type": "flat engineered event variables, stronger regularization",
        "test_raw_auc": 0.658,
        "test_weighted_auc": 0.583,
        "test_average_precision": None,
        "test_Z_A": 0.00271,
        "test_N_eff_bkg": 5.81,
        "comment": "Unstable high-score tail; not useful as nominal model.",
    },
    {
        "model": "Linear DNN",
        "input_type": "flat engineered event variables, nearly linear classifier",
        "test_raw_auc": 0.665,
        "test_weighted_auc": 0.621,
        "test_average_precision": None,
        "test_Z_A": 0.00665,
        "test_N_eff_bkg": 73.68,
        "comment": "Stable but weak separation; useful sanity check.",
    },
    {
        "model": "Multiclass tabular DNN",
        "input_type": "flat engineered event variables, 3-class output",
        "test_raw_auc": 0.647,
        "test_weighted_auc": 0.611,
        "test_average_precision": None,
        "test_Z_A": 0.00553,
        "test_N_eff_bkg": 13.25,
        "comment": "CMS-style categories, but no gain over BDT.",
    },
    {
        "model": "LBN-style v1 full",
        "input_type": "reduced objects b1,b2,lepton,MET + pair + aux",
        "test_raw_auc": 0.664,
        "test_weighted_auc": 0.625,
        "test_average_precision": None,
        "test_Z_A": 0.00571,
        "test_N_eff_bkg": 28.37,
        "comment": "Object input helps slightly but does not beat corrected BDT.",
    },
])

dnn_baselines.to_csv(outdir / "bbww_final_dnn_baseline_summary.csv", index=False)


# ---------------------------------------------------------------------
# 4. v2 full-object DNN ablation summary
# ---------------------------------------------------------------------

v2_dnn = pd.DataFrame([
    {
        "model": "v2 full: obj+pair+aux",
        "object_input": True,
        "pair_input": True,
        "aux_input": True,
        "test_raw_auc": 0.662,
        "test_weighted_auc": 0.645,
        "test_average_precision": 0.0656,
        "test_argmax_HH_Z_A": 0.00592,
        "test_argmax_HH_N_eff_bkg": 22.02,
        "test_global_Z_A": 0.00657,
        "comment": "Best v2 weighted AUC; improves ranking but not final HH-like region.",
    },
    {
        "model": "v2 obj+aux, no pair",
        "object_input": True,
        "pair_input": False,
        "aux_input": True,
        "test_raw_auc": 0.676,
        "test_weighted_auc": 0.601,
        "test_average_precision": 0.0694,
        "test_argmax_HH_Z_A": 0.00591,
        "test_argmax_HH_N_eff_bkg": 29.70,
        "test_global_Z_A": 0.00608,
        "comment": "Best raw AUC/AP among v2 models, but weaker weighted AUC.",
    },
    {
        "model": "v2 obj+pair only",
        "object_input": True,
        "pair_input": True,
        "aux_input": False,
        "test_raw_auc": 0.642,
        "test_weighted_auc": 0.574,
        "test_average_precision": 0.0615,
        "test_argmax_HH_Z_A": 0.00541,
        "test_argmax_HH_N_eff_bkg": 31.40,
        "test_global_Z_A": 0.00599,
        "comment": "Object and pair inputs alone do not replace engineered auxiliary physics features.",
    },
])

v2_dnn.to_csv(outdir / "bbww_final_dnn_ablation_summary.csv", index=False)


# ---------------------------------------------------------------------
# 5. Compact final model summary
# ---------------------------------------------------------------------

final_summary = pd.DataFrame([
    {
        "model_or_region": "Corrected simple-topology BDT, broad stable threshold",
        "best_test_Z_A": 0.00676,
        "S_over_B": 3.49e-6,
        "N_eff_bkg": 58.68,
        "test_weighted_auc": 0.672,
        "interpretation": "Safest broad corrected-recoMET BDT reference.",
    },
    {
        "model_or_region": "Corrected BDT >=0.75 score category",
        "best_test_Z_A": 0.02045,
        "S_over_B": 1.08e-4,
        "N_eff_bkg": 47.77,
        "test_weighted_auc": None,
        "interpretation": "Most signal-enriched BDT bin; best approximate Z_A.",
    },
    {
        "model_or_region": "v2 full DNN, obj+pair+aux",
        "best_test_Z_A": 0.00657,
        "S_over_B": None,
        "N_eff_bkg": 22.02,
        "test_weighted_auc": 0.645,
        "interpretation": "Best v2 weighted AUC, but does not beat corrected BDT final region.",
    },
    {
        "model_or_region": "v2 obj+aux DNN",
        "best_test_Z_A": 0.00608,
        "S_over_B": None,
        "N_eff_bkg": 29.70,
        "test_weighted_auc": 0.601,
        "interpretation": "No clear gain over BDT; pair branch not obviously essential.",
    },
    {
        "model_or_region": "v2 obj+pair-only DNN",
        "best_test_Z_A": 0.00599,
        "S_over_B": None,
        "N_eff_bkg": 31.40,
        "test_weighted_auc": 0.574,
        "interpretation": "Weakest v2 ablation; auxiliary engineered features remain important.",
    },
])

final_summary.to_csv(outdir / "bbww_final_model_summary.csv", index=False)


# ---------------------------------------------------------------------
# 6. Markdown summary for easy reading
# ---------------------------------------------------------------------

md = []
md.append("# Compact bbWW final summary tables\n")
md.append("These tables summarize the corrected `HH→bbWW*` one-lepton baseline study.\n")
md.append("The quoted yields and significances use rough proxy MC weights and should be interpreted as diagnostic, not final CMS significances.\n")

tables = [
    ("Corrected BDT results", bdt_results),
    ("BDT score categories", bdt_categories),
    ("DNN baseline summary", dnn_baselines),
    ("v2 DNN ablation summary", v2_dnn),
    ("Final compact model summary", final_summary),
]

for title, df in tables:
    md.append(f"\n## {title}\n")
    md.append(df.to_markdown(index=False))

md.append(
    """

## Final interpretation

After correcting recoMET usage, the simple-topology BDT remains the strongest practical baseline.
The BDT score categories show physically meaningful signal enrichment, with the `>=0.75` bin giving the best approximate significance and S/B.

The DNN studies show that object-level representations help somewhat, especially in the v2 full-object model, but none of the DNN variants improves the final HH-like region over the corrected BDT baseline.

The `HH→bbWW*` one-lepton region therefore remains useful as a completed baseline and diagnostic study, but the low absolute significance motivates pivoting toward a more promising HH channel or boosted kinematic region.
"""
)

(outdir / "bbww_final_summary_tables.md").write_text("\n".join(md))

print("Wrote final summary tables to:", outdir)
for path in sorted(outdir.glob("bbww_final_*")):
    print(" -", path)
