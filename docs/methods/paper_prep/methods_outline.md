# Methods outline: HH4b Delphes ML benchmark

## Simulation
- MG5_aMC event generation
- Pythia8 showering/hadronization
- Delphes CMS-like detector simulation
- Compact parquet analysis format

## Signal normalization
- ggF and VBF HH samples used for shapes and efficiencies
- Signal yields normalized externally to SM HH cross sections times BR(H->bb)^2
- MG5 signal cross-section sanity checks documented separately

## Background model
- QCD bbbb generated in non-overlapping inclusive-HT slices
- QCD slice weights from generator cross sections
- ttbar and Zbbbb generator-weighted
- QCD multijet normalization treated as simulation-level baseline pending closure/systematic tests

## Reconstruction
- Four b-tagged jet HH candidate reconstruction
- Pairing chosen by Higgs mass compatibility
- v1 candidate-level features
- v2 event-level features planned

## Baselines
- Simple mass-region cuts
- BDT baseline
- Multiseed BDT stability
- BDT-tail background composition

## Next improvements
- Targeted QCD tail statistics
- v2 features
- systematic variations
- binned likelihood / Asimov significance
- representation-learning models
