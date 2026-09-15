# Initial HH→4b CMS sensitivity-gap brief for Harvey

This is an initial **Delphes simulation expected-limit diagnostic**, not an official CMS result. It uses train-only samples, 13 TeV, 138 fb⁻¹, and no invented experimental systematics. Validation and test payloads opened: **0 and 0**.

## Measured/computed facts

The closest apples-to-apples public result is the updated CMS HIG-24-010 Run-2 resolved analysis: expected μHH95 = **5.9** and observed = 10.0. The combined Run-2+Run-3 value 2.8 is not the target.

| fixed RHH<34 likelihood | μ95 stat-only | ratio to CMS 5.9 | μ95 with per-bin finite-MC constraints | gain |
|---|---:|---:|---:|---:|
| pooled one bin | 74.02 | 12.54 | 68,085 | — |
| exact3tag + ge4tag | 44.33 | 7.51 | 32,938 | ×1.67 stat-only |
| categories + fixed RHH bins [0,10,20,30,34] | 31.96 | 5.42 | 545.7 | ×1.39 beyond categories stat-only |

The fixed selection has S=184.72 and B=48,653,343. Its selected background has Neff=57.49 and a 13.19% weighted-MC relative uncertainty. QCD importance-sampled sources contribute 95.48% of B (46.45 million) but only Neff≈52.4; ttbar contributes 4.24% (2.06 million) with Neff≈18,288. Thus the absolute background is internally closed to the frozen source table, but its most important tail is poorly supported.

## Diagnostic inferences

1. Pooling categories was a real, quantified inference loss: recovering category identity improves stat-only μ95 by 40% (a ×1.67 sensitivity gain).
2. Deterministic RHH shape information is valuable (another ×1.39 stat-only gain), but the finite-MC shape result is not yet publication-defensible. Its dramatic improvement depends on sparsely supported weighted QCD bins and independent background constraints.
3. The three largest quantified bottlenecks are: enormous selected QCD yield/S:B≈3.8×10⁻⁶; finite weighted QCD support; and discarded category/shape information in the historical count. The first two dominate.
4. Existing exact3/≥4 tables cannot quantify 0–2-tag migration or truth pairing. Consequently b tagging and pairing/mass resolution remain unquantified hypotheses, not established causes.

The next highest-information improvement is a source-stratified QCD closure/tail-support audit followed by a simulation-only lower-tag transfer prototype. More classifier optimization cannot make an unsupported weighted tail trustworthy.

## Future hypotheses and irreducible differences

- GLOBAL BDT: pending the separately frozen train-only study; no number is imported or fabricated.
- Categorized BDT and SPA-Net: future tests. SPA-Net is motivated only if a train-only truth study establishes a material pairing ceiling.
- CMS's collision-data multijet transfer, control/validation regions, trigger calibration, detector/b-tag calibrations, nuisance correlations, and classifier/category likelihood cannot be reproduced exactly with Delphes simulation. These are irreducible limitations of this comparison.
