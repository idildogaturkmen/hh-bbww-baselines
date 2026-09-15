# Track B raw-EFlow transfer characterization V4

Date: 2026-08-01

## Machine-readable status

    RESULT=TRACK_B_RAW_EFLOW_TRANSFER_CHARACTERIZATION_V4_DIRECT_PROBABILITIES_PASS
    SECOND_SOFTMAX_APPLIED=0
    DOMAIN_LABEL=raw-EFlow-to-PUPPI-model
    NEXT_MILESTONE=frozen-score transfer baseline

## Purpose

This checkpoint records the first scientifically valid,
deterministic zero-shot transfer characterization of the
released IHEP jet-free HH4b ensemble on Track A raw-EFlow
events.

The released ONNX output node is already a softmax and
returns 138 normalized probabilities. No second softmax was
applied.

All quantitative probabilities, AUC values, bootstrap
intervals, and posterior-localization values reported by V3
are superseded by V4.

## Scientific scope

The diagnostic used:

- 256 fixed-mass SM ggF HH to 4b events;
- 256 QCD bbbb events;
- 256 top-pair events;
- 768 events in total;
- three released ONNX models;
- direct normalized 138-class outputs;
- deterministic selection and tensor receipts;
- unweighted diagnostic metrics.

This result is:

- a valid zero-shot domain-transfer baseline;
- a test of whether the released representation retains
  information on Track A;
- a reproducible starting point for transfer learning.

This result is not:

- an exact IHEP PUPPI reproduction;
- a complete released 4j3b reproduction;
- a full-production Track A benchmark;
- a physics-normalized significance result;
- evidence that the released model fails in its intended
  detector domain.

## Released-model contract

Pinned public repository:

`pku-hep-group/jetfree-hh4b`

Pinned commit:

`e8387f7e49a7f27c65b2f4a1f75cacf01cd45392`

Model checksums:

| Model | SHA-256 |
|---|---|
| model0 | `7b7de7ba1bfbbf00117b251661320afabe166c74c8687e3673d97590a3745786` |
| model1 | `5514875d359fd4d6516e3367eb3264e6b4fea21f82755fc45851d06f788d429f` |
| model2 | `2d02df9702bfa13f8c20b467087121abab2556fa654e12084b73cee968f3566e` |

ONNX inputs:

- `pf_features`: `[N,19,256]`;
- `pf_vectors`: `[N,4,256]`;
- `pf_mask`: `[N,1,256]`.

ONNX output:

- node name: `softmax`;
- shape: `[N,138]`;
- indices 0 through 135: signal mass classes;
- index 136: QCD;
- index 137: top-pair.

## Track A input receipts

| Process | SHA-256 |
|---|---|
| Signal | `10f7c11b558482a945d3b566b44190ffd6a78e7bfb50aef44f45ca5e88c1571c` |
| QCD | `c89dc72a52be4f47def38534b7dcd63cb3c024f372bf85e59190371b941e3459` |
| Top | `c74ad8091105a4d122c02fe92b385451b445449f7fd203dc974b08127988adeb` |

## Selection contract

The first 4096 events per process were scanned.

Kinematic four-jet selection:

- jet pT greater than 30 GeV;
- absolute jet eta less than 2.5;
- leading thresholds greater than 75, 60, 45, and 40 GeV;
- selected-jet HT greater than 330 GeV.

Passing-event receipts:

| Process | Passing | Passing-index SHA-256 |
|---|---:|---|
| Signal | 1111 | `90acb426589d2beae386f1ee0966b065b3df4a2913e1388baca992bc94a2c531` |
| QCD | 3041 | `8848e022ddd61e6115ce8e4d371337826d5081b8789aa360a74eab0bf60b191a` |
| Top | 1323 | `6f0b202dda52d78ee701bbdeda870118c4a35add161e23beee8ab098c130a9c5` |

Retained-event receipts:

| Process | Retained | Retained-index SHA-256 |
|---|---:|---|
| Signal | 256 | `c3fe93ffc17a564ced7d46c0c09394d73c0fa2005941f3fd1eb5e2c106b50d2a` |
| QCD | 256 | `aca73bf867ad005064575ecbf8f677a977771dbca6cab0505b78a220fe348216` |
| Top | 256 | `4d7ab3ef23cce2e66b15f03cf3241fe44d3eb7f2e0aa656cfe2499396ec90fcc` |

## Combined tensor receipts

| Tensor | Shape | SHA-256 |
|---|---|---|
| `pf_features` | `(768,19,256)` | `b7af6be3198a315a8927c99708eb4f24c8302224027733aba08ec8873e5aee61` |
| `pf_vectors` | `(768,4,256)` | `c1dddee3b0676985dd81d0dec38c87085e4d3a5c713f2481443820ecc7355b55` |
| `pf_mask` | `(768,1,256)` | `5117a64545738e402b6f3d0fa03b54a08ed6ec3d37a5c8a5908c0d662cad2c55` |

V3 and V4 input tensors were byte-identical.

## Direct model-output receipts

All direct ONNX outputs were:

- finite;
- nonnegative;
- bounded by one;
- normalized to one within approximately `2e-7`.

| Model | Direct probability SHA-256 |
|---|---|
| model0 | `e6fc0ac2a1b9c560e9763b0bbe92d42736a50c69ecda73672243c5db863aae8a` |
| model1 | `79b3024e1ab84b124530cc0cebea6731b5bab2565d45e152bf46e5a6c4563d6f` |
| model2 | `aef36366e9065453d4efa6cb827152796c50b311726e115efa8bdc550f830c15` |

Ensemble direct probability SHA-256:

`8047777f5ce4b75d7f3859f374f431c11cfc3290055fb528add8ff5928498fdf`

## Process-level response

| Process | Mean Psignal | Mean PQCD | Mean Pttbar |
|---|---:|---:|---:|
| Signal | 0.487152 | 0.105985 | 0.406863 |
| QCD | 0.392709 | 0.188176 | 0.419115 |
| Top | 0.170488 | 0.111112 | 0.718399 |

Aggregate winners:

| Input process | Signal winner | QCD winner | Top winner |
|---|---:|---:|---:|
| Signal | 131 | 12 | 113 |
| QCD | 97 | 37 | 122 |
| Top | 20 | 16 | 220 |

Top-domain transfer is substantially stronger than
signal-versus-QCD transfer.

## Signal-versus-QCD discrimination

Individual direct Psignal AUC values:

| Model | AUC |
|---|---:|
| model0 | 0.599869 |
| model1 | 0.529968 |
| model2 | 0.624741 |

Ensemble direct Psignal AUC:

`0.596954`

Bootstrap summary:

- mean: `0.597053`;
- standard deviation: `0.025420`;
- 95 percent interval: `[0.547724,0.645194]`.

Conditional discriminant:

`D(signal,QCD) = Psignal / (Psignal + PQCD)`

Conditional-discriminant AUC:

`0.655045`

Bootstrap summary:

- mean: `0.655246`;
- standard deviation: `0.024703`;
- 95 percent interval: `[0.605251,0.703328]`.

## Top-domain discrimination

Pttbar top-versus-nontop AUC:

`0.792793`

Bootstrap summary:

- mean: `0.793384`;
- standard deviation: `0.016845`;
- 95 percent interval: `[0.759590,0.825756]`.

## Ensemble disagreement

| Process | Mean Psignal standard deviation | p95 | Maximum |
|---|---:|---:|---:|
| Signal | 0.146927 | 0.321074 | 0.405194 |
| QCD | 0.118132 | 0.291047 | 0.400425 |
| Top | 0.083919 | 0.286996 | 0.394458 |

The released ensemble is unstable under the Track A
raw-EFlow domain, particularly for signal.

## Particle truncation

| Process | Median multiplicity | p90 | Events above 256 |
|---|---:|---:|---:|
| Signal | 211.0 | 286.0 | 69/256 |
| QCD | 225.0 | 299.5 | 75/256 |
| Top | 206.0 | 277.0 | 42/256 |

Psignal by truncation:

| Process | Nontruncated | Truncated | Difference |
|---|---:|---:|---:|
| Signal | 0.532359 | 0.364633 | -0.167726 |
| QCD | 0.411291 | 0.347864 | -0.063427 |
| Top | 0.171186 | 0.166932 | -0.004254 |

Truncation is an important classification-domain effect for
signal and, more weakly, QCD.

## Fixed-mass signal localization

For the 256 fixed-mass signal events:

- class-100 `(125,125)` mean probability: `0.001191`;
- 115-135 GeV-region mean probability: `0.010890`;
- conditional Psignal fraction in that region: `0.023308`;
- signal-class argmax in that region: `3/256`;
- mean conditional posterior masses:
  `(128.109,172.646)` GeV;
- conditional fraction with both masses at least 155 GeV:
  `0.365520`.

Dominant signal class:

- class 134, `(185,195)` GeV: `108/256`.

The 115-135 GeV conditional fraction is:

- nontruncated: `0.022047`;
- truncated: `0.026726`.

The localization shift is not explained by 256-particle
truncation alone.

## Scientific interpretation

V4 establishes a valid zero-shot transfer baseline.

The released variable-mass model retains transferable
information:

- top discrimination is substantial;
- signal-versus-QCD ranking is above random;
- the conditional signal-QCD discriminant is informative.

However:

- direct raw-EFlow inference is not well calibrated;
- signal-versus-QCD separation is modest;
- ensemble disagreement is large;
- signal classification is truncation-sensitive;
- fixed-mass localization is strongly shifted.

The leading unresolved cause is detector and representation
mismatch between:

- Track A raw EFlow candidates;
- official IHEP PUPPI ParticleFlowCandidate inputs;
- the IHEP detector card;
- the IHEP pileup model;
- the released trigger and b-tag domain.

## Research decision

Do not scale the uncalibrated zero-shot classifier directly
to all Track A production events.

The next unblocked scientific milestone is the frozen-score
transfer baseline:

1. construct a bounded, process-representative Track A
   development sample;
2. obtain the direct 138 outputs from all three released
   models;
3. freeze those outputs as pretrained features;
4. train a small calibration classifier;
5. use grouped train, validation, and held-out test splits;
6. compare against Track A baselines on identical events;
7. evaluate both unweighted discrimination and the existing
   physical-normalization protocol.

The decisive detector-domain milestone remains a
same-generator-event signal comparison using the official
IHEP PUPPI card once the authoritative pileup artifact is
available.
