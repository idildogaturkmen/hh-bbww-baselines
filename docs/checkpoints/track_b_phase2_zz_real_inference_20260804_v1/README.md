# Track B Phase 2: real ZZ released-model inference

## Status

The three released SophonHH ONNX models were evaluated sequentially
on 64 deterministic real ZZ control events. Their arithmetic ensemble,
source-entry mapping, score archive, and official 140-branch Events tree
were validated.

## Sample selection

- Original entries: 200,000
- Eligible entries: 8,674
- Selected events: 64
- First original entry: 3
- Last original entry: 199990
- Process index: 12 for all selected events
- Particle truncation: 0 events

## Ensemble diagnostics

- Mean cumulative signal probability: 0.5334414761
- Mean QCD probability: 0.3906304873
- Mean ttbar probability: 0.0759280360
- Events with signal probability > 0.5: 34/64
- Events with signal probability > 0.9: 24/64
- Events with signal probability > 0.997: 7/64
- Three-model top-1 agreement: 0.734375

The leading signal-grid cell is 80--90 GeV by 90--100 GeV.
The high-purity events also concentrate around the expected ZZ mass
region.

## Classification interpretation

The cumulative signal probability sums 136 mass-grid classes. QCD is
one class. Therefore cumulative signal probability may exceed the QCD
probability even when QCD is the highest individual class.

## Memory strategy

The earlier all-in-one ZZ attempt was OOM-killed. The successful path
used a deterministic 64-event source skim and loaded only one ONNX
session at a time. The successful Python process reached approximately
2.70 GiB maximum resident memory and caused no new OOM kill.

## Scope limitation

This is an unweighted 64-event control canary. It is not a
full-statistics signal-discrimination result.

## Next controls

QCD and inclusive `TTbar_forInfer2` are next, using the same
memory-bounded workflow.
