# Simulation methods draft

All Monte Carlo samples in this study are generated at leading order using MadGraph5_aMC, showered and hadronized with Pythia8, and passed through Delphes fast detector simulation.

The study focuses on reconstruction, background modeling, importance sampling, and machine-learning baseline comparisons rather than precision cross-section predictions. No NLO, NNLO, or N3LO K-factors are applied unless explicitly stated.

## Main generated samples

- QCD bbbb: generated as `p p > b b~ b b~ QED=0`, tree-level LO QCD, nominally O(alpha_s^4).
- ttbar: generated as `p p > t t~`, tree-level LO QCD, nominally O(alpha_s^2).
- Zbbbb: generated as `p p > z b b~, z > b b~`, tree-level LO Z + heavy flavor.
- ggF HEFT HH4b: LO in an effective ggHH approximation.
- VBF HH4b: LO electroweak VBF-like HH generation.

## Detector simulation

Detector effects are modeled using Delphes fast simulation with a CMS-like detector card. Reconstructed jets and b-tagging information are taken from Delphes output objects.

## Stored dataset

The released dataset is an analysis-level Delphes-derived dataset. It contains compact event-summary parquet files, HH4b-candidate parquet files, manifests, summaries, cards, checksums, and script snapshots. It does not include all full Delphes ROOT, HepMC, or LHE files.
