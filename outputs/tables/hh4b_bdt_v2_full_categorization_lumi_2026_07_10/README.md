# HH4b full BDT-v2 categorization luminosity study

Date: 2026-07-10

This study retrains the two background-specific BDT-v2 classifiers seed-by-seed:
- BDT_QCD: signal vs QCD-enriched backgrounds
- BDT_top: signal vs ttbar/top background

Events are assigned to mutually exclusive HHH-inspired categories in the BDT_QCD vs BDT_top plane.
The category yields and approximate significances are recomputed at 138/fb, 350/fb, 450/fb, and 4000/fb.

Metrics include:
- S/B
- S/sqrt(B)
- Asimov stat-only Z
- approximate S/sqrt(B + (0.1B)^2) orientation metric
- combined-category sqrt(sum_i S_i^2/B_i)
- category composition by sample
