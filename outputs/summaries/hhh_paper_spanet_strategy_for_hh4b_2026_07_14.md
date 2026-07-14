# HHH → 6b SPA-Net paper strategy and adaptation to HH → 4b

Reference: CMS HHH → 6b paper, CMS-HIG-24-012 / arXiv:2607.05145.

## What the paper does

The HHH analysis uses a symmetry-preserving attention network, SPA-Net, to separate HHH → 6b from QCD multijet, ttbar, and HH → 4b. The same network also helps assign/cluster jets into Higgs candidates and define topology-enriched categories.

Main ingredients:
- small-radius AK4 jets and large-radius AK8 jets
- anti-kT clustering with R=0.4 and R=0.8
- ParticleNet b-tag and bb-tag information
- jet inputs: pT, eta, cos(phi), sin(phi), mass
- up to ten small-radius jets and three large-radius jets
- pairwise angular separations and dijet masses
- event-level variables such as HT and MET
- two-stage classifier:
  1. inclusive classifier: HHH, HH, QCD, ttbar
  2. second classifier on signal-enriched events
- topology categorizer:
  categories based on number of merged and resolved Higgs candidates
- signal extraction:
  binned fit in high ProbMultiH = ProbHH + ProbHHH region, especially ProbMultiH > 0.8
- dedicated ttbar control region and QCD background modeling/validation

## What maps directly to this HH → 4b Delphes analysis

Current implementation:
- resolved HH → 4b only
- leading-8 AK4-like Delphes jets
- jet features: pt, eta, phi/mass/btag currently; Deep SPA-Net uses sin(phi), cos(phi)
- assignment head for H1/H2 pairing
- event classification with all selected events

Implemented next:
- Deep/two-head SPA-Net:
  - qcd_score: HH vs QCD-like/Zbbbb
  - top_score: HH vs ttbar
  - assignment head: H1/H2 jet pairing
  - event-level features and pair summaries
  - best-checkpoint selection
  - two-score rectangle scan, analogous to two-BDT baseline

## What should come after Deep/two-head SPA-Net

1. Evaluate Deep/two-head SPA-Net:
   - all-background ROC
   - signal-vs-QCD ROC
   - signal-vs-ttbar ROC
   - qcd_score/top_score plane
   - best rectangle scan
   - category yields
   - comparison to BDT, DNN, LBN, and first SPA-Net

2. Implement a stage-2 refiner:
   - train only on events passing loose stage-1 selection
   - same target heads/classes
   - analogous to the HHH paper's second classifier

3. Add category/control-region logic:
   - tight signal region: high qcd_score and high top_score
   - ttbar control region: high top_score-background-like or low top_score signal-like rejection region
   - QCD-like control/sideband regions
   - bins in combined score or ProbMultiH-like score

4. Add boosted/merged extension:
   - AK8 jets with pT > 300 GeV
   - Hbb tag proxy or ParticleNet/Sophon score if available
   - resolved/semi-merged/merged HH categories

5. Add official-CMS realism later:
   - POWHEG ttbar and HH samples
   - NNLO normalization
   - pileup and PUPPI
   - GEANT4 or official centrally produced CMS samples
   - ParticleNet/DeepJet scores from NanoAOD or official reconstruction

## Near-term scientific message

The first SPA-Net baseline improved over the dense DNN but remained below the mass-aware BDT. Its weakness is ttbar rejection. The HHH paper supports a more structured approach: multi-class or multi-head SPA-Net, staged refinement, topology categories, and control regions. Therefore, the Deep/two-head SPA-Net is the right next step.
