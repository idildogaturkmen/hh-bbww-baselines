# HH4b reference-baseline checkpoint — 2026-07-24

This checkpoint preserves the compact canonical artifacts establishing the
first CMS-inspired resolved HH→bbbb development baseline.

## Frozen analysis definition

- Detector domain: Run-2-like 13 TeV frozen-v2 Delphes.
- Candidate requirement inherited from reconstruction: at least four
  selected b-tagged jets.
- All four candidate jets: pT > 40 GeV and |eta| < 2.4.
- Analysis region: r_hh_125_120 < 55.
- Signal region: r_hh_125_120 < 30.
- Control region: 30 <= r_hh_125_120 < 55.
- Low/high-mHH boundary: 450 GeV.

## Data policy

- Signal modes remain separate: ggF HH and VBF HH.
- Background families remain separate.
- Development cutflow uses train and validation only.
- Signal and background test samples remain sealed.
- No cut optimization was performed while constructing the reference
  cutflow.

## Established checkpoint

- Signal development generated events: 197,000.
- Signal development candidate rows: 12,221.
- Background development generated events: 4,545,457.
- Background development candidate rows: 57,850.
- The joint unweighted reference cutflow is frozen.
- Run-2 signal normalization is frozen.
- Run-3 and HL-LHC signal outputs are labeled projections using the current
  Run-2-like acceptance.

## Remaining limitations

This checkpoint is not yet a complete physics-weighted background
prediction.

- QCD HardQCD pThat-stratum cross sections and sampling corrections remain
  unresolved.
- QCD sample overlap and exclusivity remain to be frozen.
- Filtered triboson effective-normalization semantics remain unresolved.
- Run-3 and HL-LHC detector acceptances have not been simulated.
- Full physical S/B and significance are not yet authorized.
- BDT model selection must use train and validation only.

## Artifact layout

Canonical repository files are copied beneath:

- `artifacts/repository/`
- `artifacts/generated/`

The `generated/` name intentionally replaces the source repository's
ignored `outputs/` directory component.
