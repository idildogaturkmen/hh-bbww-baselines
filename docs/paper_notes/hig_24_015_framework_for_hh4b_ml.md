# HIG-24-015 framework lessons for HH4b ML

## Source analysis

CMS HIG-24-015 searches for HHH production in the 4b2gamma final state using Run 2 CMS data.
Its structure is useful for this HH4b ML study because it handles an extremely rare Higgs signal,
large reducible/nonresonant backgrounds, multiple MVAs, categories, and final statistical extraction.

## Key framework elements to adapt

### 1. Use simulation for shapes/optimization but validate background modeling

HIG-24-015 does not trust simulation alone for fake-photon nonresonant backgrounds.
It builds a Low Score Region (LSR), morphs photon-ID scores using a fake-photon PDF, and normalizes
the morphed background in sidebands.

HH4b analogue:
- Use QCD HT-sliced simulation for baseline shapes.
- Build sideband/closure tests using R_HH sidebands, 3b control regions, or low-BDT regions.
- Treat QCD generator normalization as a baseline, not a final data-driven estimate.

### 2. Use two MVAs, not just one

HIG-24-015 uses:
- BDTnonres to suppress nonresonant backgrounds.
- BDTres to suppress resonant H/HH backgrounds, especially ttH.

HH4b analogue:
- BDT_QCD: HH vs QCD multijet.
- BDT_top: HH vs ttbar/ttbb/top-like backgrounds.
- Final categorization in the 2D plane of BDT_QCD and BDT_top.

### 3. Optimize categories with minimum background statistics

HIG-24-015 defines categories in the 2D BDT plane and requires enough sideband events per category.

HH4b analogue:
- Define non-overlapping BDT categories.
- Require minimum effective background statistics per category.
- Report category-wise S, B, S/B, S/sqrt(B), and MC-stat uncertainty.

### 4. Avoid sculpting the final fit variable

HIG-24-015 excludes m_gg from BDTnonres because m_gg is used for final signal extraction.

HH4b analogue:
- If final fit variable is mHH or R_HH, train classifiers both with and without that variable.
- Check that the classifier does not sculpt artificial mass peaks in QCD.
- Provide decorrelated or mass-excluded baselines.

### 5. Include systematic and stability studies

Minimum HH4b ML systematics:
- multiseed training stability
- MC statistical uncertainty
- QCD slice normalization uncertainty
- ttbar normalization uncertainty
- b-tag efficiency/mistag variations
- jet energy scale/resolution-like variations
