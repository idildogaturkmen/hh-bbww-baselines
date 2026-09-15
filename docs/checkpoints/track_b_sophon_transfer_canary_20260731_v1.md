# Track B Sophon transfer canary checkpoint

Date: 2026-07-31

## Purpose

Test whether the released variable-mass, full-event HH4b Sophon
ensemble can consume fixed-mass SM HH4b Track A Delphes events.

## Repository state

- Base branch: `delphes-hh4b-production`
- Base commit: `5d2cb189db2359cb216a3167efbbbc0d585f3bf9`
- Checkpoint branch: `track-b-sophon-transfer`

## Released model contract

- Public repository: `pku-hep-group/jetfree-hh4b`
- Pinned commit:
  `e8387f7e49a7f27c65b2f4a1f75cacf01cd45392`
- Models:
  - `model0.onnx`
  - `model1.onnx`
  - `model2.onnx`
- Inputs:
  - `pf_features [N,19,256]`
  - `pf_vectors [N,4,256]`
  - `pf_mask [N,1,256]`
- Output:
  - 136 signal mass-grid classes
  - class 136: QCD
  - class 137: ttbar

## Track A sources

- Signal:
  `ggf_hh4b_ak4ak8_10k_pythia8_delphes.root`
- QCD:
  `qcd_bbbb_iht600plus_20000_pythia8_delphes.root`
- Top:
  `ttbar_200k_shard000_pythia8_delphes.root`

The adapter merges collections in the order:

1. `EFlowTrack`
2. `EFlowPhoton`
3. `EFlowNeutralHadron`

Global particle sums are computed before truncating to 256 particles.

## Vertex and DZ convention

The three Track A source files do not contain
`Vertex/Vertex.Z`.

They do contain:

- `EFlowTrack.D0`
- `EFlowTrack.ErrorD0`
- `EFlowTrack.DZ`
- `EFlowTrack.ErrorDZ`

Therefore the canary uses the stored `EFlowTrack.DZ` value directly:

- `primary_vertex_z_available = false`
- `track_dz_convention = raw EFlowTrack.DZ`

This is a documented domain difference to study.

## Event selection and sampling

The official four-jet preselection was applied:

- leading jet \(p_T > 75\) GeV
- second jet \(p_T > 60\) GeV
- third jet \(p_T > 45\) GeV
- fourth jet \(p_T > 40\) GeV
- selected-jet \(H_T > 330\) GeV
- jet \(p_T > 30\) GeV and \(|\eta| < 2.5\)

Spread scan:

| Process | Scanned | Passing | Retained |
|---|---:|---:|---:|
| Signal | 4096 | 1069 (26.099%) | 64 |
| QCD | 4096 | 3090 (75.439%) | 64 |
| Top | 4096 | 1297 (31.665%) | 64 |

## Particle multiplicity

| Process | Median | p95 | Maximum | Truncated |
|---|---:|---:|---:|---:|
| Signal | 213.5 | 294.0 | 355 | 14/64 |
| QCD | 209.0 | 317.3 | 349 | 15/64 |
| Top | 198.0 | 293.5 | 340 | 12/64 |

All 192 events used the raw-DZ fallback.

## Technical validation

Constructed inputs:

- `pf_features: (192, 19, 256)`
- `pf_vectors: (192, 4, 256)`
- `pf_mask: (192, 1, 256)`

All three models produced output shape `(192,138)`.

Maximum softmax-sum errors:

- model 0: `1.192e-07`
- model 1: `1.192e-07`
- model 2: `1.788e-07`
- ensemble: `1.788e-07`

Result:

`TRACK_A_THREE_MODEL_TRANSFER_CANARY_PASS`

## Initial descriptive response

| Process | Mean P(signal) | Mean P(QCD) | Mean P(ttbar) |
|---|---:|---:|---:|
| Signal | 0.485417 | 0.132269 | 0.382314 |
| QCD | 0.417303 | 0.192551 | 0.390146 |
| Top | 0.165994 | 0.127321 | 0.706685 |

Three-model individual-class top-1 agreement:

- Signal: 62.500%
- QCD: 59.375%
- Top: 82.812%

These are unweighted compatibility results from 64 events per
process. They are not a final performance benchmark.

## Interpretation

- End-to-end technical transfer passed.
- Top classification transfers strongly.
- Signal has a larger mean aggregate signal probability than QCD,
  but the signal-QCD margin is modest.
- Heavy-flavor high-HT QCD is substantially out of the released
  training domain or intrinsically signal-like.
- The largest individual 138-class output is not the correct
  three-category decision rule because signal probability is split
  across 136 mass-grid classes.
- Particle truncation and raw-DZ handling require controlled
  domain-shift studies.

## Next gates

1. Compare aggregate category winners:
   `sum(signal classes)` vs QCD vs ttbar.
2. Run a same-event raw-DZ versus zero-DZ ablation.
3. Audit fixed-mass HH signal-grid localization near
   \((125,125)\) GeV.
4. Expand to a larger unweighted sample.
5. Only afterward construct weighted Track A performance metrics.
6. Retry IHEP access after the July 31-August 1 maintenance period.

## External IHEP status

The IHEP XRootD endpoint timed out from FNAL, CERN, and a local
network during annual IHEP maintenance. Congqiao indicated that
service should hopefully return after August 1.
