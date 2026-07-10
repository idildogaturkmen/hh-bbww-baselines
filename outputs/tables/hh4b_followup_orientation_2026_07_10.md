# HH→4b analysis orientation summary — 2026-07-10

## Reference BDT-v2 baseline

The current BDT-v2 studies use 450/fb as the reference luminosity.

Current best single-cut BDT-v2 working point:
- BDT_QCD > 0.775
- BDT_top > 0.500
- S/B ≈ 3.35e-4
- S/sqrt(B) ≈ 0.181 at 450/fb

## Luminosity projection

For a fixed model and fixed selection/category definition:
- S scales linearly with luminosity.
- B scales linearly with luminosity.
- S/B remains constant.
- Stat-only S/sqrt(B) scales approximately as sqrt(L).

Reference luminosities considered:
- 138/fb: Run 2 scale
- 350/fb: Run 2 + Run 3 scale
- 450/fb: current reference value used in the baseline tables
- 4000/fb: end-of-LHC + HL-LHC scale

The simple projection is an orientation check only. It does not replace a full retraining/evaluation study.

## HHH-inspired categorization

Motivated by the HHH→4b2γ strategy, I tested a first categorization in the 2D plane of:
- BDT_QCD, analogous to a nonresonant-background suppressor
- BDT_top, analogous to a top/resonant-background suppressor

The categories are mutually exclusive rectangles in the BDT_QCD vs BDT_top plane.

First categorization result at 450/fb:
- Inclusive union of categories: S/sqrt(B) ≈ 0.185
- Approximate combined category significance: sqrt(sum_i S_i^2/B_i) ≈ 0.221 ± 0.020
- Highest-purity category: S/B ≈ 6.47e-4

## Interpretation

Categorization appears promising. The approximate combined-category significance improves relative to the single-cut BDT-v2 baseline, and the highest-purity category nearly doubles S/B. However, the high-purity categories are still MC-statistics limited, especially in the dominant QCD iHT200–400 and ttbar residual backgrounds.

## Next steps

1. Run a more complete categorization rerun with seed-by-seed BDT retraining and per-luminosity evaluation at 138/fb, 350/fb, 450/fb, and 4000/fb.
2. Increase targeted background statistics, especially QCD bbbb iHT200–400 and ttbar.
3. Re-run BDT-v2 and categorization after background expansion.
4. Compare S/B and S/sqrt(B) scales to CMS HH→4b analysis-level results.
5. Continue SPA-Net training and compare against the categorized BDT-v2 baseline.
