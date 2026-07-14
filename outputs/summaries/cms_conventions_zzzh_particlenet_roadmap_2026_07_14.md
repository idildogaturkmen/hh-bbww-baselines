# CMS-convention roadmap: AK4/AK8, ParticleNet-style inputs, and ZZ/ZH validation

## Current Delphes status

The current Delphes card uses:
- small-R jets: anti-kT, R = 0.5
- FatJets: anti-kT, R = 0.8
- FatJet substructure enabled: trimming, pruning, SoftDrop, N-subjettiness

Therefore the current resolved HH→4b results should be described as using small-R anti-kT R=0.5 Delphes jets, not CMS AK4 jets.

## CMS-convention target

For a more CMS-like Delphes workflow:
- small-R jets should be anti-kT R=0.4, AK4-like
- large-R jets should remain anti-kT R=0.8, AK8-like
- AK8 pileup mitigation and ParticleNet cannot be faithfully reproduced from the current Delphes card alone
- Delphes b-tag and FatJet tag variables should be described as proxies, not CMS DeepJet or ParticleNet

## Immediate action

Create a new Delphes card:
- `cards/delphes/delphes_card_CMS_lpc_ak4ak8.tcl`
- set `FastJetFinder ParameterR = 0.4`
- set `GenJetFinder ParameterR = 0.4`
- keep `FatJetFinder ParameterR = 0.8`

Do not overwrite the old card. Current R=0.5 results remain useful as Delphes-v1. New AK4/AK8 samples should use a separate production tag.

## ParticleNet-style path

Near-term:
- use Delphes `Jet.BTag` as a b-tag proxy
- use FatJet pT, eta, mass, SoftDrop mass, N-subjettiness, and groomed variables as boosted-H proxy inputs
- do not call these ParticleNet

CMS-realistic:
- move to CMS NanoAOD / official samples
- use official DeepJet and ParticleNet/ParticleNet-MD scores
- apply official scale factors and systematics
- evaluate template shapes under jet energy/mass/tagging variations

## ZZ/ZH incorporation

The ZZ/ZH paper motivates these processes as standard-candle benchmarks for HH-like analyses because they share the same experimental reconstruction challenges but have larger cross sections.

For HH→4b, use:
- ZZ→4b as a validation peak near (91, 91)
- ZH→4b as a validation peak near (91, 125) or (125, 91)
- HH→4b as the target peak near (125, 125)

Near-term plan:
1. Generate Delphes ZZ→4b and ZH→4b samples.
2. Run the same small-R reconstruction and SPA-Net inference.
3. Produce `mbb1` vs `mbb2` mass-plane plots.
4. Define ZZ, ZH, and HH ellipses/windows.
5. Use ZZ/ZH to validate pairing and background modeling.
6. Later, train a multiclass model: HH, ZZ, ZH, QCD-like, ttbar.

## Relation to current SPA-Net work

The current Deep/two-head SPA-Net is the right immediate next step:
- qcd_score: HH vs QCD-like/Zbbbb
- top_score: HH vs ttbar
- assignment head: H1/H2 jet pairing

After this:
1. diagnose two-head output
2. build stage-2 refiner
3. add AK4/AK8 Delphes-v2 production
4. add ZZ/ZH validation
5. move toward official CMS NanoAOD/CMSSW workflow
