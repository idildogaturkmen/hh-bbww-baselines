# Simulation setup

Repository commit: e8ba98ac3b17b25cf4aad8e0d58741d4570a48b5

## Generator setup

All samples in this release are generated at leading order using MadGraph5_aMC, then showered and hadronized with Pythia8, and passed through Delphes fast detector simulation.

These samples are not NLO, NNLO, or N3LO samples. No higher-order K-factors are applied unless explicitly stated elsewhere.

Approximate perturbative order of the main generated samples:

- QCD bbbb: tree-level LO QCD, generated as p p > b b~ b b~ QED=0, nominally O(alpha_s^4)
- ttbar: tree-level LO QCD, generated as p p > t t~, nominally O(alpha_s^2)
- Zbbbb: tree-level LO Z + heavy flavor, generated as p p > z b b~, z > b b~
- ggF HEFT HH4b: LO in the HEFT/effective ggHH approximation
- VBF HH4b: LO electroweak VBF-like HH generation

## Detector simulation

Detector effects are modeled using Delphes fast simulation with a CMS-like detector card. Jets and b-tagging are taken from Delphes reconstructed objects.

The exact Delphes card used should be stored under:

cards/delphes/

## Stored dataset format

This release stores analysis-level Delphes-derived parquet files, not full raw Delphes ROOT files for every shard.

Included:
- merged event-summary parquet files
- merged HH4b-candidate parquet files
- campaign manifests
- summary tables
- MG5 cards
- Delphes card
- snapshots of key scripts

Not included:
- full HepMC files
- full LHE files
- all Delphes ROOT shards

The full production can be regenerated from the cards, scripts, random seeds in the manifests, and the repository commit above.
