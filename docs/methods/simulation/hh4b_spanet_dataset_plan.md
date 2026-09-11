# HH→4b SPA-Net Dataset Plan

## Motivation

The current BDT-v2 baseline uses reconstructed v2 candidate/event features and is the strongest tabular-feature model so far. A simple DNN-v2 trained on the same tabular features underperforms the BDT-v2. Feature ablations show that the full v2 feature set performs best, with angular topology helping QCD rejection and event activity helping top rejection.

This motivates moving toward an assignment-aware model such as SPA-Net, where the model learns the HH→4b jet assignment and classification jointly rather than relying only on hand-engineered candidate features.

## Current v2 parquet status

The current v2 candidate parquets contain:
- four selected candidate jets: j1, j2, j3, j4
- jet kinematics: pt, eta, phi, mass
- b-tag/rank/flavor diagnostic columns
- event-level features: n jets, n b-tags, extra jets/b-tags, HT
- pairing diagnostics from the closest-mass reconstruction

These files are sufficient for a toy 4-jet SPA-Net-like smoke test, but not for a full SPA-Net training setup.

## Limitation of current v2 candidate parquets

The v2 parquets only store the final four candidate jets selected by the reconstruction algorithm. A full SPA-Net HH→4b setup should instead use a larger leading-N jet collection, for example up to 8 selected jets, with masks.

The full dataset should allow the model to learn which jets should be assigned to the two Higgs bosons, rather than only evaluating the four jets already chosen by the baseline reconstruction.

## Required full SPA-Net inputs

For each event:

### Jet inputs
For up to N selected jets, for example N = 8:
- jet_pt
- jet_eta
- jet_phi
- jet_mass
- jet_btag
- optional: jet_rank, jet_btag_rank

### Masks
- jet_mask indicating which padded jets are real

### Event-level features
Optional global features:
- n_selected_jets
- n_selected_bjets
- n_extra_selected_jets
- n_extra_selected_bjets
- ht_selected_jets
- ht_selected_bjets

### Labels for signal
For HH→4b signal events:
- index labels for the two jets assigned to H1
- index labels for the two jets assigned to H2
- permutation symmetry handled so H1/H2 and b/b within each Higgs are exchangeable

### Labels for background
For background events:
- no valid Higgs assignment
- classification label background = 0
- assignment labels masked or set to ignore index

## Truth-labeling strategy

Use generator-level H→bb daughters from the ROOT Particle branch:
- identify Higgs bosons
- identify b daughters from each Higgs
- match reconstructed selected jets to truth b quarks using ΔR
- assign jets to H1/H2 if matched within a chosen ΔR threshold
- require four matched b jets for assignment-supervised signal subset
- keep all events for classification, with assignment loss masked when labels are incomplete

## Initial model stages

### Stage 1: dataset builder
Build ROOT → parquet/npz dataset with leading-N jets, masks, weights, process labels, and truth assignment labels.

### Stage 2: toy assignment-aware model
Train a small permutation-aware model on leading-N jets to predict:
- HH signal/background classification
- H1/H2 jet assignment for signal events with complete truth labels

### Stage 3: SPA-Net
Move to a SPA-Net-style architecture after the dataset and labels are validated.

## Evaluation metrics

Compare against BDT-v2:
- QCD weighted AUC
- top weighted AUC
- signal/background classifier AUC
- S/sqrt(B) at 450/fb
- S/B
- selected background composition
- assignment accuracy on signal:
  - all-four matched fraction
  - correct pairing over all signal candidates
  - correct pairing given all-four matched

## Immediate next steps

1. Build a ROOT-to-SPA-Net dataset writer for ggF/VBF signal first.
2. Validate truth assignment labels on signal.
3. Extend dataset writer to backgrounds with classification labels and masked assignment labels.
4. Train toy leading-N jet model.
5. Compare toy model to BDT-v2.
