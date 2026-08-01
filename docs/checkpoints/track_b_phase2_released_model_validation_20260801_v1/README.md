# Track B Phase 2 released-model validation checkpoint

## Status

**Released-model reproduction and all currently accessible control-sample
validations are complete. Independent HH evaluation is pending one external
dataset path.**

- Freeze time: `2026-08-01T18:11:57Z`
- Local branch: `delphes-hh4b-production`
- Local commit: `6d14d5b6d8244e408a44c0cb4c21e187a535b9d7`
- Local origin: `https://github.com/idildogaturkmen/hh-bbww-baselines.git`
- Public release: `https://github.com/pku-hep-group/jetfree-hh4b.git`
- Public release commit: `e8387f7e49a7f27c65b2f4a1f75cacf01cd45392`
- IHEP endpoint: `root://cceos.ihep.ac.cn:1094`

No ROOT files, ONNX models, NumPy arrays, or DCB outputs are stored in this
checkpoint. All large validation inputs and outputs were processed under
`/tmp` and removed after each run.

## Scientific purpose

Track B tests whether the released variable-mass, full-event HH4b model and
its particle-level representation can be reproduced and later transferred to
the fixed-mass CMS-style Track A analysis.

The released model produces 138 softmax outputs:

- indices 0–135: the 136 ordered two-mass signal-grid classes;
- index 136: QCD;
- index 137: ttbar.

The nominal inclusive signal discriminant is the sum of indices 0–135.

## Completed validation gates

### 1. Released preprocessing and inference contract

The exact 19-feature transformation, 256-particle truncation, zero padding,
four-vector input, mask input, and three-model ensemble were reproduced at
public commit `e8387f7e49a7f27c65b2f4a1f75cacf01cd45392`.

All three models ran in isolated CPU processes with bounded memory. Model
outputs were finite and normalized to unity.

### 2. Released-model canaries

| Sample | Events | Principal result |
|---|---:|---|
| Variable-mass signal | 64 | Mean signal probability 0.970627 |
| Training QCD | 64 | Mean QCD probability 0.931263; QCD top-1 100% |
| Training ttbar | 64 | Mean ttbar probability 0.665297; ttbar top-1 79.688% |

The tiny signal canary is a contract test, not an unbiased efficiency or
resolution measurement.

### 3. Signal mass-grid coverage

A spread sample of 36,864 signal events from 12 files covered all 136 signal
classes. Individual files contained broad mixtures of classes rather than
single fixed-mass points.

This established that future sampling must use spread or randomized event
selection. Taking the first N events is not acceptable.

### 4. Background domain behavior

Training, inference, and extension samples for QCD and ttbar produced
consistent class behavior. The extension datasets are additional samples,
not new target classes.

### 5. Minor-background response map

Fourteen additional processes were audited. Top-associated backgrounds were
predominantly ttbar-like. Most non-top processes were predominantly QCD-like.
Processes containing genuine H or Z resonances showed elevated signal-grid
response, especially ZH and ZZ.

These results are descriptive and unweighted because the two `gen_weight`
entries remain undefined.

### 6. Raw ZH and ZZ resonance localization

For high-score events, the 136-class signal-grid response localized near the
physical resonance pairs:

- ZH: leading cells near `(85–105, 115–135)` GeV;
- ZZ: leading cells near `(75–105, 85–115)` GeV.

The ZH localization was stronger and narrower than ZZ, as expected from the
observed score maps.

### 7. Official ZH DCB validation

From 512 uniformly sampled selected ZH events:

- 253 passed 4j3b;
- 35 passed 4j3b and `P_signal > 0.997`;
- 33/35 primary DCB fits succeeded;
- 31/33 successful primary peaks were inside the physical 40–200 GeV range;
- 28 events had a non-sentinel secondary peak;
- 24/28 such secondary peaks were physically inside 40–200 GeV;
- among successful fits, 21/33 primary peaks were in the coarse ZH region;
- among successful fits, 24/33 had either peak in the coarse ZH region.

The coarse ZH region uses sorted fitted masses within 15 GeV of `(mZ, mH)`.

### 8. Official ZZ DCB validation

Across two ZZ files and 1,024 uniformly sampled selected events:

- 539 passed 4j3b;
- 41 passed 4j3b and `P_signal > 0.997`;
- 41/41 primary DCB fits succeeded;
- 36/41 primary peaks were physically inside 40–200 GeV;
- 29/41 had a non-sentinel secondary peak;
- 27/41 had a physical secondary peak;
- 15/41 primary peaks were in the coarse ZZ region;
- 21/41 had either physical peak in the coarse ZZ region.

The coarse ZZ region requires both fitted masses to be within 15 GeV of
`mZ`.

### 9. IHEP namespace discovery

The endpoint `root://cceos.ihep.ac.cn:1094` is operational and accessible with the CMS proxy.

A recursive search of all 8,833 entries visible under

`/eos/ihep/cms/store/user/coli/datasets`

found only the top-level directories `JetClassII` and `hh4b`.

No exact or broader candidate was found for:

- `ggHH`
- `ggHHkl0`
- `ggHHkl5`
- `qqHH`
- `sm_incl_derived_4j3bor2b`

Therefore, the independent HH evaluation samples are not visible under the
currently shared namespace.

## Unresolved items

### Independent HH evaluation sample

The one major Phase 2 blocker is an exact XRootD path for an independent
ggF or VBF HH sample.

Two emails have already been sent to the dataset contact, who said they
would respond. No additional message should be sent yet.

### `gen_weight` definitions

Audited files contain two weight entries. The second was consistently the
first multiplied by `1e-4`, but the official producer stores the values
without semantic labels.

No physically normalized yield or weighted ROC result should use these
entries until their definitions are confirmed.

## Storage constraints

- Track A at `/uscms_data/d3/iturkmen/hh4b_delphes` must remain intact.
- Large Track B processing remains temporary under `/tmp`.
- A full persistent dataset transfer must not begin until `eosquota`
  visibly reports 5 TB.
- No certificate files or `~/.globus` state may be modified.

## Next technical gate

When an independent HH file path is received:

1. Run a read-only path and schema check.
2. Run a 64-event preprocessing and three-model inference canary.
3. Verify particle multiplicity, truncation, selections, and score
   normalization.
4. Run a spread 1,000-event pilot.
5. Measure unweighted HH efficiency and signal-grid localization.
6. Run the official DCB reconstruction.
7. Evaluate ROC curves against independent QCD and ttbar.
8. Add physical weighting only after `gen_weight` semantics are confirmed.

Machine-readable values are stored in `summary.json`.
