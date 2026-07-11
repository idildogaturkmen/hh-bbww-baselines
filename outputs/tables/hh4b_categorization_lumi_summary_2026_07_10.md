# HH→4b BDT-v2 categorization and luminosity summary — 2026-07-10

## Categorization setup

A full seed-by-seed BDT-v2 categorization study was performed using two background-specific classifiers:

- BDT_QCD: signal vs QCD-enriched backgrounds
- BDT_top: signal vs ttbar/top background

Events were assigned to mutually exclusive categories in the 2D plane of BDT_QCD vs BDT_top, following the same general strategy as the HHH→4b2γ categorization approach.

## Main result at 450/fb

Single-cut BDT-v2 baseline:
- S/sqrt(B) ≈ 0.181
- S/B ≈ 3.35e-4

Categorized BDT-v2 result:
- Inclusive union: S/sqrt(B) ≈ 0.185
- Combined category approximation: sqrt(sum_i S_i^2/B_i) ≈ 0.221 ± 0.020

This corresponds to an approximate 22% improvement in stat-only combined significance relative to the single-cut BDT-v2 baseline.

## Highest-purity category

CAT0:
- S ≈ 44.6 events at 450/fb
- B ≈ 70,147 events at 450/fb
- S/B ≈ 6.47e-4
- S/sqrt(B) ≈ 0.169
- median background rows ≈ 58
- median background neff ≈ 27

CAT0 nearly doubles S/B relative to the previous single-cut BDT-v2 working point, but it is still limited by MC statistics.

## Luminosity dependence

Combined category stat-only significance:
- 138/fb: ≈ 0.122
- 350/fb: ≈ 0.195
- 450/fb: ≈ 0.221
- 4000/fb: ≈ 0.659

The simple 10% background-systematic orientation metric remains approximately flat near 0.008, showing that background control would dominate if uncertainties are at that scale.

## Background composition at 450/fb

CAT0 is dominated by:
- ttbar: about 54%
- QCD bbbb iHT200–400: about 31%
- QCD bbbb iHT100–200: about 16%, but with very low raw statistics
- smaller contributions from QCD iHT400–600, Zbbbb, and QCD iHT600+

## Interpretation

Categorization is promising and should be kept as part of the analysis path. However, the high-purity categories are limited by MC statistics, especially in QCD iHT200–400, ttbar, and high-weight low-statistics QCD iHT100–200 contributions.

## Next steps

1. Increase QCD bbbb iHT200–400 statistics.
2. Increase ttbar statistics.
3. Add more QCD bbbb iHT100–200 statistics to stabilize high-weight low-row contributions.
4. Re-run BDT-v2 and the full categorization study with enlarged backgrounds.
5. Compare the categorized BDT-v2 result to SPA-Net once the SPA-Net training finishes.
