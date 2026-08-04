# Track B Phase 2: real inclusive-ttbar released-model inference

## Status

The three released SophonHH ONNX models were evaluated sequentially
on 64 deterministic real inclusive-ttbar control events. The ensemble,
source-entry mapping, official Events tree, and score archive were
validated.

## Sample

- Namespace: `TTbar_forInfer2`
- Original entries: 200,000
- Eligible entries: 89,948
- Selected events: 64
- Original-entry span: 0 through 199999
- Source `process_index`: -1 for all selected events
- Dedicated model ttbar output: class 137
- Particle truncation: 0 events

## Classification

- ttbar top-1: 50/64
- QCD top-1: 13/64
- Signal-grid top-1: 1/64
- Mean ttbar probability: 0.6571873053
- Median ttbar probability: 0.8107165992
- Mean QCD probability: 0.1687427122
- Mean cumulative signal probability: 0.1740699879
- Three-model top-1 agreement: 0.9375

## Ttbar probability thresholds

- ttbar probability > 0.5: 46/64
- ttbar probability > 0.9: 26/64
- ttbar probability > 0.997: 12/64

## Signal false-positive tail

- Signal probability > 0.5: 9/64
- Signal probability > 0.9: 1/64
- Signal probability > 0.997: 0/64

## Memory strategy

Only one ONNX model session was resident at a time. Maximum Python
resident memory was approximately 2.73 GiB, and no new cgroup OOM kill
was observed.

## Scope

This is an unweighted 64-event control canary. It is not a precise
measurement of ttbar rejection or signal mistag probability.

## Next step

After publication, construct the combined ZH, ZZ, QCD, and inclusive
ttbar 64-event control comparison.
