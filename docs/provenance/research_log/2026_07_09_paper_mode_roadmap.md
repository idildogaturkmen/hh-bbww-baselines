# HH4b Delphes ML paper-mode roadmap

Target: JHEP/PRD-style ML/phenomenology paper, not a CMS experimental measurement.

## Baseline claim

Construct a reproducible Delphes-level benchmark for HH->4b classification with:
- SM-normalized nonresonant HH signal.
- Generator-weighted, phase-space-sliced QCD backgrounds.
- Multiseed training stability.
- Targeted background-statistics improvement in BDT-tail regions.
- Future representation-learning comparison.

## Required layers before paper-quality claims

1. Frozen sample manifests and normalization configs.
2. Cut baseline with SM-normalized signal and generator-weighted backgrounds.
3. BDT baseline with multiseed stability.
4. BDT-tail composition and targeted background generation.
5. Correct combination of extra QCD slices without double-counting.
6. MC statistical uncertainty / bootstrap studies.
7. Systematic variations:
   - b-tag efficiency/mistag
   - jet energy scale/resolution
   - QCD slice normalization
   - signal normalization
   - top normalization
8. Sideband/closure tests for QCD modeling.
9. Stronger ML baselines:
   - BDT v1
   - BDT v2 event-level features
   - DNN/MLP
   - DeepSets/ParticleNet/Particle Transformer style model if feasible
   - later SSL/foundation embeddings inspired by RINO/J-JEPA/JP-JEPA
10. Statistical model:
   - Asimov significance with background uncertainty
   - simple binned likelihood over BDT or mass variables
   - no discovery/sensitivity claim without systematic uncertainties

## Current robust statement

The provisional SM-normalized BDT gives a modest but stable improvement over simple cut regions. High-score BDT tails remain MC-stat limited, motivating targeted QCD HT-slice production.
