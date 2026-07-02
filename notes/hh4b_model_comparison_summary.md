# Resolved HH4b model comparison summary

## Purpose

This note compares the current resolved HH4b baselines under the same held-out test split and the same Poisson-bootstrap stability criteria.

The compared approaches are:

1. Cut baseline
2. BDT v2
3. Dense DNN v1
4. LBN-inspired DNN v2

The main metric is not only AUC. The main decision criterion is whether the model produces a stable analysis category with competitive S/B, Asimov Z_A, bootstrap-median Z_A, background effective statistics, and limited single-event/background-sample dominance.

## Headline result

The best stable baseline remains:

- BDT v2
- Working point: BDT score >= 0.55
- Nominal Z_A: 0.0386
- Bootstrap median Z_A: 0.0391
- Bootstrap 16-84% Z_A range: 0.0321-0.0496
- S/B: 1.003e-04
- N_eff_bkg: 6.51

## Paper-style comparison table

| Model | Working point | Stable? | Raw AUC | Weighted AUC | Raw S | Raw B | S_w | B_w | S/B | Nominal Z_A | Boot median Z_A | Boot p16 Z_A | Boot p84 Z_A | N_eff bkg | Max single bkg frac | Top sample frac |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Cut baseline | Resolved 4b preselection | yes |  |  | 262 | 3025 | 19.0547 | 3.259e+05 | 5.846e-05 | 0.0334 | 0.0334 | 0.0294 | 0.0383 | 17.2527 | 0.1240 | 0.3720 |
| BDT v2 | BDT score >= 0.55 | yes | 0.7785 | 0.6648 | 204 | 1110 | 14.8365 | 1.478e+05 | 1.003e-04 | 0.0386 | 0.0391 | 0.0321 | 0.0496 | 6.5142 | 0.2734 | 0.5468 |
| Dense DNN v1 | DNN score >= 0.35 | yes | 0.7804 | 0.6479 | 243 | 1846 | 17.6729 | 2.296e+05 | 7.696e-05 | 0.0369 | 0.0377 | 0.0318 | 0.0449 | 10.4628 | 0.1760 | 0.5280 |
| LBN-DNN v2 | LBN-DNN score >= 0.65 | yes | 0.7637 | 0.6884 | 188 | 998 | 13.6728 | 1.351e+05 | 1.012e-04 | 0.0372 | 0.0382 | 0.0305 | 0.0493 | 5.4790 | 0.2991 | 0.5981 |

## Interpretation

The cut baseline gives the lowest sensitivity, with nominal Z_A = 0.0334 and bootstrap median Z_A = 0.0334.

The BDT v2 is currently the preferred baseline. Its stable working point, BDT score >= 0.55, gives nominal Z_A = 0.0386, bootstrap median Z_A = 0.0391, and S/B = 1.003e-04.

The dense DNN v1 improves over the cut baseline and passes stability, but it does not beat BDT v2. Its stable working point, DNN score >= 0.35, gives nominal Z_A = 0.0369, bootstrap median Z_A = 0.0377, and S/B = 7.696e-05.

The LBN-DNN v2 is a useful physics-aware model. It gives better weighted ranking behavior than the dense DNN and reaches BDT-like S/B. However, under the same stability criteria, its best stable working point, LBN-DNN score >= 0.65, gives nominal Z_A = 0.0372 and bootstrap median Z_A = 0.0382, slightly below the BDT baseline.

The LBN-DNN has higher-score regions with larger apparent S/B and nominal Z_A, but those regions fail stability because they have low N_eff_bkg and are too sensitive to high-weight Z+bb events. Therefore, they should be treated as promising diagnostics rather than final categories.

## Recommended plots

Main plots for paper or presentation:

- outputs/hh4b_baseline/model_comparison/plots/hh4b_model_comparison_ZA_bootstrap.png
- outputs/hh4b_baseline/model_comparison/plots/hh4b_model_comparison_signal_purity.png
- outputs/hh4b_baseline/model_comparison/plots/hh4b_model_comparison_auc.png

Useful backup plot:

- outputs/hh4b_baseline/model_comparison/plots/hh4b_model_comparison_background_dominance.png

## Suggested wording

A concise result statement is:

> In the resolved HH4b baseline study, a BDT, dense DNN, and LBN-inspired DNN were compared under the same held-out test split and Poisson-bootstrap stability criteria. The dense DNN and LBN-DNN both improve over the cut baseline, but neither clearly outperforms the stable BDT working point. The LBN-DNN achieves BDT-like purity and improved weighted ranking, but its most aggressive high-score categories fail stability due to high-weight Z+bb background dominance. The BDT v2 therefore remains the preferred resolved HH4b classifier baseline before moving to assignment-aware SPA-Net reconstruction.


## Future-work idea kept for later

If time remains after the rigorous baseline comparison and SPA-Net setup, a more novel end-of-project direction is:

A hybrid boosted/resolved HH4b reconstruction network that combines SPA-Net assignment with GN2X-style boosted Hbb tagging, bJR-style mass regression, and RINO/DINO-style pretraining or regularization for QCD robustness.

Possible inputs:

- AK4 jets
- AK8 jets
- optional jet constituents/subjets/tracks if available
- global event features

Possible outputs:

- resolved H1 assignment: two AK4 jets
- resolved H2 assignment: two AK4 jets
- boosted H candidates: one AK8 jet each
- event topology class: resolved-resolved, resolved-boosted, boosted-boosted
- H candidate mass/pT regression
- event-level HH vs QCD/ttbar/Z+bb discriminant

Possible combined loss:

- assignment loss
- H mass regression loss
- topology classification loss
- event classification loss
- mass-decorrelation or anti-sculpting penalty
- optional DINO/RINO-style consistency loss across clustered views

This should remain future work until the resolved BDT/DNN/LBN-DNN/SPA-Net baselines are complete and the specific failure modes are documented.

