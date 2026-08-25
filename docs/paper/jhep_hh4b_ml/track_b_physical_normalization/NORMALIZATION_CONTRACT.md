# Normalization Contract

These model-independent constants and equations are FINAL for the exact four model artifacts in `MODEL_FREEZE.tsv`. They do not vary by model; only thresholds and pass counts vary.

- SM holdout_A threshold population: `508168`
- full SM normalization population: `2538851`
- SM weight at 450/fb: `0.000473265`
- physical QCD inference population: `87623306`
- QCD1 / QCD2: `27683689` / `59939617`
- QCD weight at 450/fb: `7.3112875413134075`

Definitions:

```text
epsilon_SM = N_SM_pass / 508168
S(L) = epsilon_SM * 2538851 * 0.000473265 * (L/450)
epsilon_QCD = N_QCD_pass / 87623306
R_QCD = 87623306 / N_QCD_pass                         [N_QCD_pass > 0]
B_QCD(L) = N_QCD_pass * 7.3112875413134075 * (L/450)
S_over_B = S / B
S_over_sqrtB = S / sqrt(B)
Z_A = sqrt(2*((S+B)*ln(1+S/B)-S))
```

All 4000/fb quantities use exactly the same contract with scale `4000/450`.

## Threshold semantics

For each target, `k = ceil(target * 508168)`. The threshold is the kth-largest finite float32 SM score. An event passes iff `score >= threshold`; the complete threshold tie block passes. The tables report target and actual achieved efficiency separately, together with `N_SM_pass`, the strict-above count, and tie-block size. No target is silently substituted for an achieved efficiency.
