# Rough-normalized one-lepton HH→bbWW BDT status

## Current nominal baseline
The simple-topology BDT is the current nominal baseline.

Reasons:
- Best robust test weighted AUC among tested BDT variants.
- Validation/test behavior is reasonably consistent.
- Background effective statistics are more stable than the W/top variants.
- High-score categories improve S/B relative to the cut baseline.

## W/top feature study
Hadronic-W/top variables are physically motivated and improve purity in some regions.

However:
- Explicit W/top cuts remove too much signal to improve the rough weighted Z proxy.
- The unregularized W/top BDT showed high-score instability.
- The regularized W/top BDT is more stable and higher purity, but the high-score region still has limited background effective statistics.
- In val+test, the regularized high-score region is dominated by ttbar and WW. The earlier high-weight Z+bb event appears in the training split, not the evaluation splits.

## Main caveats
- Rough normalization only.
- QCD/gamma/minbias/upsilon are excluded pending metadata.
- No systematic uncertainties included yet.
- Some high-score regions have limited effective background statistics.

## Likely next directions
1. Improve normalization/background metadata.
2. Build a simple BDT-category likelihood or counting model.
3. Study ttbar/WW control regions and top-rejection handles.
4. Move to DNN only after the baseline is stable.
