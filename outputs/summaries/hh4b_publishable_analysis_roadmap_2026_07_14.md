# HH→4b AK4/AK8 SPA-Net analysis roadmap

## Main goal

Develop a CMS-style Delphes HH→4b analysis using AK4/AK8 objects, compare baseline ML methods, and validate the Higgs-pair reconstruction using ZZ/ZH→4b standard-candle samples.

## Core paper story

The target paper should not be a model zoo. The central story should be:

1. Build a consistent AK4/AK8 Delphes HH→4b dataset.
2. Compare cut baseline, BDT, DNN, SPA-Net, and two-head SPA-Net.
3. Use SPA-Net for permutation-aware Higgs assignment.
4. Use separate QCD-like and ttbar heads to address the two dominant background topologies.
5. Use ZZ→4b and ZH→4b as validation samples for bb mass reconstruction and pairing.
6. Evaluate working points with MC-statistical stability requirements, not only raw best S/sqrt(B).

## Current status

- Original full qcdplus ML results use small-R anti-kT R=0.5 jets.
- New AK4/AK8 Delphes card is available:
  - small-R jets: anti-kT R=0.4
  - FatJets: anti-kT R=0.8
- AK4/AK8 smoke samples were produced from existing HepMC files.
- R=0.5 vs AK4/AK8 smoke comparisons were made for jet multiplicity, HT, FatJets, and bb mass-plane reconstruction.
- Deep two-head SPA-Net is the best neural-network model so far, but the raw tight working point is MC-stat limited.
- AK4/AK8 ttbar extra 50k production is in progress.

## Immediate production plan

### Step 1: Finish AK4/AK8 ttbar extra 50k

Purpose:
- Improve top-background statistics.
- Stabilize the two-head SPA-Net top-rejection working point.
- Keep ROOT files for future AK4/AK8 dataset building.

### Step 2: Produce ZZ/ZH→4b AK4/AK8 pilots

Purpose:
- Validate mass reconstruction and jet pairing.
- Check whether the pipeline reconstructs:
  - ZZ near (91, 91) GeV
  - ZH near (91, 125) GeV
  - HH near (125, 125) GeV

Start with small pilot samples before large production.

### Step 3: Build consistent AK4/AK8 qcdplus v1

Needed samples:
- ggF HH→4b
- VBF HH→4b
- ttbar
- Z+bbbb
- QCD bbbb in HT bins, especially high HT
- ZZ→4b validation
- ZH→4b validation

### Step 4: Retrain models only after the dataset is consistent

Do not mix old R=0.5 samples with AK4/AK8 samples in final model comparisons.

Models to include:
- cut baseline
- mass-aware BDT
- DNN
- first SPA-Net
- two-head SPA-Net

Optional later:
- Sophon/ParT comparison if dataset becomes available.

## Publishability criteria

The result becomes much stronger if it includes:

1. A consistent AK4/AK8 dataset.
2. Stable background estimates with enough MC statistics.
3. Validation on ZZ/ZH→4b mass peaks.
4. Clear comparisons to simple baselines.
5. Control-region logic for QCD-like and ttbar-like backgrounds.
6. Honest treatment of limitations:
   - Delphes, not full CMS simulation.
   - b tagging is a Delphes proxy, not real DeepJet/ParticleNet.
   - final sensitivity projections are illustrative unless backgrounds are sufficiently large.

## Near-term decision

After ttbar extra 50k finishes:
- Verify all ROOT/parquet/metadata outputs.
- Summarize the ttbar production.
- Then start ZZ/ZH→4b AK4/AK8 pilot generation.
