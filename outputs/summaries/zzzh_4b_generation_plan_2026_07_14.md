# ZZ/ZH → 4b validation sample plan

Goal: use ZZ→4b and ZH→4b as standard-candle validation samples for HH→4b reconstruction, inspired by the ZZ/ZH analysis strategy.

## Physics role

- ZZ→4b should populate a bb mass plane near (91, 91) GeV.
- ZH→4b should populate a bb mass plane near (91, 125) or (125, 91) GeV.
- HH→4b should populate a bb mass plane near (125, 125) GeV.

These samples can validate:
- jet pairing,
- mass-plane reconstruction,
- SPA-Net assignment behavior,
- background-modeling/control-region strategy.

## Delphes convention

Future ZZ/ZH production should use:
- `cards/delphes/delphes_card_CMS_lpc_ak4ak8.tcl`
- small jets: anti-kT R=0.4
- FatJets: anti-kT R=0.8

Do not mix these with the previous R=0.5 qcdplus results.

## Proposed initial sample sizes

Start with smoke samples:
- ZZ→4b: 10k generated events
- ZH→4b: 10k generated events

Then, if reconstruction behaves well:
- ZZ→4b: 100k
- ZH→4b: 100k

## Analysis outputs

For each sample:
- selected event count
- leading small-R jet multiplicity
- b-tag multiplicity
- best-pair mbb1 vs mbb2
- SPA-Net-assigned mbb1 vs mbb2
- truth-matched mass plane when available
- ZZ/ZH/HH mass-region efficiency

## Future ML extension

After validation:
- multiclass classifier: HH, ZZ, ZH, QCD-like, ttbar
- or use ZZ/ZH as validation-only standard candles while keeping HH-vs-background training.
