# HH4b model comparison and reporting contract

## Purpose

This document freezes the common reporting contract for all HH→4b baseline and machine-learning models in this repository. Every model must be trained, selected, evaluated, and compared on the same data partitions, reconstruction definitions, physical-normalization registry, multijet treatment, category definitions, nuisance model, and inference procedure.

The goal is to separate two questions:

1. **Machine-learning performance:** how well a method ranks, classifies, reconstructs, and calibrates events.
2. **Physics performance:** how much expected HH sensitivity the method provides under the same physical and statistical assumptions.

No model may receive a more favorable sample definition, weight convention, threshold-selection rule, category definition, or nuisance treatment than another model.

## Scope

The contract applies to cut-based baselines, BDTs, dense DNNs, LBN models, SPA-Net, Particle Transformer or other transformer-based models, representation-learning or embedding-enhanced models, and future architectures.

## Prerequisite gates

Final physics comparisons remain unauthorized until the following are frozen:

- full-campaign generator denominators;
- signed-weight conventions and sidecars where required;
- hard-QCD exclusive-bin stitching and extension-merging rules;
- process definitions, beam energy, forced-decay scope, overlap rules, and filter conventions;
- external reference cross sections, k-factors, branching factors, filter efficiencies, and uncertainties;
- luminosity scenario;
- train, validation, and final-evaluation membership;
- primary lower-b-tag pseudo-data multijet strategy;
- direct stitched hard-QCD secondary closure/projection role;
- common likelihood and nuisance model.

Validation and final-evaluation candidate content must remain sealed until the corresponding authorization gates pass.

## Common data and optimization rules

Every comparison must use identical train, validation, and final-evaluation membership; identical process and category definitions; identical physical-weight sidecars; identical quality requirements; train-only hyperparameter and threshold selection; the same luminosity, cross-section registry, and nuisance assumptions; and checksum-frozen model and output artifacts.

Training weights and physical inference weights remain distinct. Architecture-level discrimination metrics may use unit weights or another explicitly nonnegative convention. Signed physical weights must be retained for yields, templates, likelihoods, significances, and limits.

## Required model identity record

For every trained model, record:

- model name and version;
- repository commit;
- training script and configuration paths;
- train, validation, and final-evaluation manifest checksums;
- input schema and preprocessing configuration;
- random seed;
- parameter and trainable-parameter counts;
- optimizer, learning-rate schedule, batch size, epochs, and early-stopping rule;
- loss definition and weighting conventions;
- hardware type;
- training wall time;
- peak memory where available;
- inference throughput in events per second;
- model checkpoint path and SHA-256;
- threshold-selection objective and threshold;
- threshold source partition, which must be train only.

## Classification metrics

For every classifier, record:

- ROC AUC under the frozen nonnegative evaluation convention;
- precision-recall AUC;
- ROC and precision-recall curves;
- signal efficiency at fixed background efficiencies;
- background rejection, `1 / epsilon_B`, at signal efficiencies of 25%, 50%, and the cut-baseline-matched signal efficiency;
- significance-improvement characteristic, `epsilon_S / sqrt(epsilon_B)`, as a curve;
- maximum train-selected significance-improvement value;
- bootstrap confidence intervals for AUC and key working points;
- per-process and per-category discrimination;
- score distributions before and after physical weighting;
- calibration diagnostics, including Brier score and expected calibration error when probabilities are interpreted probabilistically.

Signed physical weights must not be passed blindly into ROC implementations that assume nonnegative sample weights. Unit-weight or explicitly nonnegative ranking metrics are architecture diagnostics; physical sensitivity is evaluated separately with signed weights in the statistical model.

## Reconstruction metrics

For assignment or reconstruction models, record:

- reconstructable-event fraction;
- exact full-event assignment efficiency;
- at-least-one-Higgs-correct efficiency;
- per-Higgs pairing efficiency and purity;
- assignment performance versus jet multiplicity, Higgs transverse momentum, and `m_HH`;
- resolved, boosted, and b-tag-category breakdowns;
- category confusion matrix;
- assignment efficiency conditional on true and predicted category;
- rejected or unassigned fraction;
- Higgs-mass and di-Higgs-mass bias and resolution;
- downstream classifier and inference improvement attributable to reconstruction.

## Physics-yield and inference metrics

At every frozen operating point, record:

- signal yield `S`;
- total background yield `B`;
- yield by process family and category;
- signed sum of weights;
- sum of squared weights;
- effective event count, `(sum w)^2 / sum(w^2)`;
- negative-weight contribution and cancellation fraction;
- missing, duplicate, nonfinite, and failed-join counts;
- `S / B`;
- `S / sqrt(B)` as a supporting diagnostic;
- statistical-only Asimov significance;
- systematic-aware Asimov or likelihood significance;
- expected discovery significance from the common likelihood;
- expected 95% CL upper limit on `mu_HH`;
- expected cross-section upper limit in physical units and as a multiple of the SM prediction;
- nuisance-parameter pulls, constraints, and impacts where applicable.

The expected 95% CL upper limit on `mu_HH` is the primary model-ranking metric once the common statistical model is frozen. AUC, `S/B`, `S/sqrt(B)`, and Asimov significance are supporting diagnostics.

## Multijet-specific reporting

The primary multijet estimate is the lower-b-tag simulation pseudo-data transfer. Direct stitched hard-QCD is secondary closure/projection only.

Record source and target regions, transfer-factor definition, train-only derivation inputs, closure by category and kinematic region, closure uncertainty, the separate direct-QCD comparison, overlap exclusions, and multijet-nuisance impacts on significance and `mu_HH`.

Direct stitched hard-QCD must never be summed with overlapping `qcd_bbbb_general` or `qcd_bbbb_iht400to600` samples as though they were disjoint.

## Robustness and ablations

For each model, record when practical:

- mean and standard deviation across at least three seeds;
- bootstrap confidence intervals;
- learning curves and overtraining checks;
- performance by process, category, jet multiplicity, Higgs transverse momentum, and `m_HH`;
- detector and modeling variation sensitivity;
- input and architecture ablations;
- calibration before and after calibration procedures;
- inference-time and memory tradeoffs;
- failure modes and out-of-domain behavior.

## Standard comparison artifacts

Every completed comparison should produce:

1. a machine-readable model registry row;
2. a machine-readable metric table;
3. ROC, precision-recall, SIC, and calibration plots;
4. reconstruction-efficiency and purity plots where applicable;
5. physically normalized discriminant templates;
6. process and category yield tables;
7. effective-count and signed-weight diagnostics;
8. expected-limit and significance summaries;
9. seed and bootstrap uncertainty summaries;
10. configuration, checkpoint, and output checksums.

## Minimum common comparison table

The final headline table must include at least:

- model;
- parameter count;
- ROC AUC;
- PR AUC;
- background rejection at 25% and 50% signal efficiency;
- assignment efficiency and purity where applicable;
- `S` and `B`;
- `S/B`;
- `S/sqrt(B)`;
- statistical-only Asimov significance;
- systematic-aware significance;
- expected 95% CL `mu_HH` upper limit;
- effective signal and background counts;
- inference throughput;
- training-seed spread.

## Energy-scenario rule

Current generated samples are 13 TeV samples. A physical Run-2 interpretation must use 13 TeV cross sections and acceptances. Run-3 and HL-LHC results require separate 13.6 TeV and 14 TeV cross-section and acceptance treatment.

A result obtained by changing luminosity while retaining 13 TeV kinematics and acceptances must be labeled **fixed-13-TeV-kinematics luminosity projection**, not a complete Run-2-plus-Run-3 or HL-LHC prediction.

## Scientific interpretation rule

A model is better only when its improvement survives the common-data, common-normalization, common-threshold, and common-likelihood contract. Improvements must include statistical uncertainty, weighted-sample stability, and relevant systematic effects. No claim may rely only on a higher AUC when expected physical sensitivity is unchanged or degraded.
