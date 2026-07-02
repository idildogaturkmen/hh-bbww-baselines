# HH4b resolved BDT v2 stability and resampling summary

## Purpose

This note documents the stability study for the resolved HH4b BDT v2 baseline. The goal is to test whether the BDT improvement over the cut baseline is robust under finite-MC fluctuations, rather than being driven by a small number of high-weight background events.

All finite-weight backgrounds are kept in the nominal evaluation. This is not a background-removal study. It is a robustness study using Poisson bootstrap resampling of the held-out test split.

## Inputs and important output files

Core input:

- outputs/hh4b_baseline/bdt_v2/hh4b_resolved_bdt_v2_test_scores.csv

Main stability outputs:

- outputs/hh4b_baseline/bdt_v2_stability/bdt_v2_stability_ranked_categories.csv
- outputs/hh4b_baseline/bdt_v2_stability/bdt_v2_stability_recommended_categories.csv
- outputs/hh4b_baseline/bdt_v2_stability/bdt_v2_stability_summary.csv
- outputs/hh4b_baseline/bdt_v2_stability/bdt_v2_stability_top_influential_background_events.csv
- outputs/hh4b_baseline/bdt_v2_stability/bdt_v2_stability_bootstrap_summary.csv

Paper-ready derived files from this documentation script:

- outputs/hh4b_baseline/bdt_v2_stability/bdt_v2_stability_paper_table.csv
- outputs/hh4b_baseline/bdt_v2_stability/bdt_v2_stability_paper_table.md
- outputs/hh4b_baseline/bdt_v2_stability/plots/bdt_v2_stability_ZA_bootstrap.png
- outputs/hh4b_baseline/bdt_v2_stability/plots/bdt_v2_stability_signal_purity.png
- outputs/hh4b_baseline/bdt_v2_stability/plots/bdt_v2_stability_background_dominance.png

The full bootstrap replicate table is useful for provenance, but it is probably too large for the paper body:

- outputs/hh4b_baseline/bdt_v2_stability/bdt_v2_stability_bootstrap_replicates.csv

## Stability criteria

A category is marked stable if it passes all of the following criteria:

- raw signal events >= 5
- raw background events >= 20
- background effective statistics N_eff_bkg >= 5
- largest single background event contributes <= 30% of total B_w
- largest background sample contributes <= 70% of total B_w
- bootstrap relative half-width of Z_A <= 60%

These criteria are intentionally practical and diagnostic. They are not a replacement for a full CMS likelihood with systematic uncertainties, but they prevent overclaiming unstable BDT tails.

## Main result

The best stable BDT working point is BDT score >= 0.55.

- Resolved 4b cut baseline: Z_A = 0.0334, bootstrap median Z_A = 0.0334, S/B = 5.846e-05, N_eff_bkg = 17.25
- Conservative BDT working point, score >= 0.40: Z_A = 0.0353, bootstrap median Z_A = 0.0355, S/B = 7.040e-05, N_eff_bkg = 12.43
- Best stable BDT working point, score >= 0.55: Z_A = 0.0386, bootstrap median Z_A = 0.0391, S/B = 1.003e-04, N_eff_bkg = 6.51

Thus, the BDT gives a modest but stable improvement over the inclusive resolved 4b cut baseline. The improvement is not large, but it survives Poisson bootstrap resampling at moderate BDT score thresholds.

## Key table for paper or internal note

| Selection | Type | Stable? | Raw S | Raw B | S_w | B_w | S/B | Nominal Z_A | Boot Z_A p16 | Boot Z_A median | Boot Z_A p84 | Boot rel. width | N_eff bkg | Max single bkg frac | Top bkg sample | Top sample frac |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Resolved 4b cut | cut_baseline | yes | 262 | 3025 | 19.0547 | 3.259e+05 | 5.846e-05 | 0.0334 | 0.0294 | 0.0334 | 0.0383 | 0.1334 | 17.2527 | 0.1240 | ZJetsTobb_13TeV-madgraphMLM-pythia8 | 0.3720 |
| H-mass 70-190 | cut_baseline | yes | 202 | 1650 | 14.6910 | 2.113e+05 | 6.954e-05 | 0.0320 | 0.0273 | 0.0326 | 0.0399 | 0.1938 | 8.8905 | 0.1913 | ZJetsTobb_13TeV-madgraphMLM-pythia8 | 0.5740 |
| H-mass 90-160 | cut_baseline | no | 131 | 873 | 9.5274 | 9.890e+04 | 9.634e-05 | 0.0303 | 0.0251 | 0.0312 | 0.0402 | 0.2412 | 5.7097 | 0.4087 | ZJetsTobb_13TeV-madgraphMLM-pythia8 | 0.4087 |
| BDT >= 0.40 | bdt_cumulative | yes | 243 | 2010 | 17.6729 | 2.510e+05 | 7.040e-05 | 0.0353 | 0.0306 | 0.0355 | 0.0417 | 0.1557 | 12.4270 | 0.1610 | ZJetsTobb_13TeV-madgraphMLM-pythia8 | 0.4830 |
| BDT >= 0.50 | bdt_cumulative | yes | 218 | 1408 | 15.8547 | 2.164e+05 | 7.327e-05 | 0.0341 | 0.0292 | 0.0346 | 0.0415 | 0.1770 | 9.3171 | 0.1868 | ZJetsTobb_13TeV-madgraphMLM-pythia8 | 0.5603 |
| BDT >= 0.55 | bdt_cumulative | yes | 204 | 1110 | 14.8365 | 1.478e+05 | 1.003e-04 | 0.0386 | 0.0321 | 0.0391 | 0.0496 | 0.2248 | 6.5142 | 0.2734 | ZJetsTobb_13TeV-madgraphMLM-pythia8 | 0.5468 |
| BDT >= 0.60 | bdt_cumulative | yes | 180 | 861 | 13.0910 | 1.388e+05 | 9.434e-05 | 0.0351 | 0.0293 | 0.0363 | 0.0469 | 0.2418 | 5.7574 | 0.2913 | ZJetsTobb_13TeV-madgraphMLM-pythia8 | 0.5825 |
| BDT >= 0.65 | bdt_cumulative | no | 164 | 594 | 11.9274 | 1.242e+05 | 9.601e-05 | 0.0338 | 0.0275 | 0.0346 | 0.0459 | 0.2667 | 4.6384 | 0.3253 | ZJetsTobb_13TeV-madgraphMLM-pythia8 | 0.6507 |
| BDT >= 0.72 | bdt_cumulative | no | 108 | 216 | 7.8546 | 5.554e+04 | 1.414e-04 | 0.0333 | 0.0251 | 0.0348 | 0.0676 | 0.6105 | 1.8653 | 0.7277 | ZJetsTobb_13TeV-madgraphMLM-pythia8 | 0.7277 |
| BDT >= 0.74 | bdt_cumulative | no | 85 | 132 | 6.1819 | 4.837e+04 | 1.278e-04 | 0.0281 | 0.0206 | 0.0295 | 0.0737 | 0.9000 | 1.4235 | 0.8357 | ZJetsTobb_13TeV-madgraphMLM-pythia8 | 0.8357 |

## Recommended plots for paper or presentation

Recommended main plot:

- plots/bdt_v2_stability_ZA_bootstrap.png

This plot compares the nominal Z_A and bootstrap 16-84% interval for the cut baseline and BDT working points. It is the best single figure to show the resampling result.

Useful supporting plot:

- plots/bdt_v2_stability_signal_purity.png

This plot shows that the BDT improves S/B relative to the inclusive resolved 4b cut.

Best appendix or backup plot:

- plots/bdt_v2_stability_background_dominance.png

This plot shows why high-score BDT categories are not used as the primary result: the largest single event and largest background sample start to dominate the weighted background.

## Interpretation

The BDT v2 learns meaningful HH4b-like structure and provides a stable, modest improvement over the cut baseline at moderate score thresholds. The best current working point is score >= 0.55, while score >= 0.40 is a more conservative alternative.

Very high BDT-score regions should not be used as the main result. Although they can have higher apparent purity, they fail stability criteria because they have low effective background statistics and are dominated by individual high-weight Z+bb events.

The dominant limitation is Z+bb background modeling and limited effective MC statistics in the signal-like tail. This does not mean Z+bb should be removed from evaluation. It means that final category optimization needs either more robust background modeling, more MC statistics, data-driven control regions, or a full likelihood treatment with uncertainties.

## Suggested paper wording

A concise way to describe this result is:

> The resolved HH4b BDT baseline gives a modest but stable improvement over the inclusive resolved 4b cut. Using Poisson bootstrap resampling of the held-out test sample, the working point BDT score >= 0.55 improves the Asimov significance from approximately 0.033 for the inclusive resolved 4b selection to approximately 0.039, while increasing S/B from 5.8e-5 to about 1.0e-4. Higher-score categories are not used as the primary result because they are dominated by a small number of high-weight Z+bb events and fail stability requirements.

## Next step

The next fair model comparison is a simple dense DNN trained on the same resolved HH4b features and evaluated with the same stability/resampling criteria. The DNN should be compared against the stable BDT working point score >= 0.55, not against unstable high-score BDT tails.
