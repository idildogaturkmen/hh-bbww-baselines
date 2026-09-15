# Track-B Literature Resources

This file records the primary literature and software-methodology sources
used to motivate and define the Track-B HH -> 4b model-comparison study.

The roles below are intentionally distinguished: the dataset/framework
source is not the same as the primary SPA-Net architecture/training source,
and the newest SPA-Net paper is not automatically the primary
hyperparameter source.

## 1. Chiang et al. — primary task-matched native SPA-Net reference

**Cheng-Wei Chiang, Feng-Yang Hsieh, Shih-Chieh Hsu, Ian Low**

**Deep learning to improve the sensitivity of Di-Higgs searches in the 4b channel**

- Journal: JHEP 09 (2024) 139
- arXiv: 2401.14198
- DOI: 10.1007/JHEP09(2024)139

Role in this study:

- Primary task-level literature anchor for the native SPA-Net HH -> 4b arm.
- Directly studies di-Higgs production with H -> bb and uses SPA-Net for
  symmetry-aware jet assignment together with event discrimination.
- The nonresonant configuration is the source of the primary
  literature-aligned SPA-Net hyperparameter configuration used in the
  Track-B native SPA-Net baseline.
- Particularly relevant because the study combines assignment and
  signal/background classification rather than treating SPA-Net only as
  a reconstruction algorithm.

## 2. Yang & Li — Track-B dataset/framework source

**Tianyi Yang, Congqiao Li**

**Potential of di-Higgs observation via a calibratable jet-free HH -> 4b framework**

- arXiv: 2508.15048

Role in this study:

- Source framework and high-statistics Delphes dataset underlying Track B.
- Provides the training/inference ntuples and the all-particle,
  jet-free HH -> 4b framework against which this reconstructed-jet
  representation study is motivated.
- The present study does NOT claim to reproduce the Yang-Li architecture.
  It uses the shared dataset to ask a different controlled question:
  how much discrimination is obtained from reconstructed kinematics,
  continuous flavor information, symmetry-aware assignment, and learned
  constituent-level representations.
- Dataset normalization and sample-composition handling should follow the
  authors' documented/source-clarified conventions.

## 3. Haoyang Li et al. — newer SPA-Net methodology reference

**Haoyang Li et al.**

**Reconstruction of boosted and resolved multi-Higgs-boson events with symmetry-preserving attention networks**

- Journal: JHEP 11 (2025) 119
- arXiv: 2412.03819
- DOI: 10.1007/JHEP11(2025)119

Role in this study:

- Newer methodology reference for SPA-Net applied to resolved and boosted
  multi-Higgs reconstruction.
- Supports the relevance of partial-event/missing-target-aware SPA-Net
  training for multi-Higgs reconstruction.
- Useful reference for the SPA-Net v2.3 / modern multi-Higgs methodology
  lineage.
- NOT used as the primary numerical hyperparameter source for this study:
  its reconstruction task and architecture choices differ from the
  Chiang HH -> 4b joint classification task.

## Study-level interpretation

The primary native SPA-Net baseline is therefore best described as:

> A literature-aligned SPA-Net v2.3 implementation for the Yang-Li
> high-statistics HH -> 4b benchmark, with the primary joint
> assignment/classification configuration anchored to Chiang et al.,
> while Haoyang Li et al. provides a newer multi-Higgs SPA-Net methodology
> reference.

These references should remain distinct in the paper:

1. **Yang & Li** — dataset/framework source.
2. **Chiang et al.** — closest task-matched native SPA-Net analysis and
   primary configuration anchor.
3. **Haoyang Li et al.** — newer resolved/boosted multi-Higgs SPA-Net
   methodology reference.

