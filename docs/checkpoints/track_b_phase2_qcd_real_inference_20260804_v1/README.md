# Track B Phase 2: real QCD released-model inference

## Status

The three released SophonHH ONNX models were evaluated sequentially
on 64 deterministic real QCD control events. The arithmetic ensemble,
source-entry mapping, official Events tree, and model-score archive
were validated.

The inference was executed at commit
`463e34e5b06da33193bec3c1572a712681e73130`. The checkpoint is being
committed on a later descendant after verifying that the execution
commit remains in branch history and that the backend and all artifacts
remain byte-identical.

## Sample

- Original entries: 157,503
- Eligible entries: 86,099
- Selected events: 64
- First original entry: 2
- Last original entry: 157502
- Process index: 0 for all selected events
- Particle truncation: 0 events

## QCD classification

- QCD top-1: 62/64
- ttbar top-1: 2/64
- Signal-grid top-1: 0/64
- Mean QCD probability: 0.8885440354
- Median QCD probability: 0.9830515385
- Mean cumulative signal probability: 0.0894953092
- Median cumulative signal probability: 0.0167533548
- Three-model top-1 agreement: 0.96875

## Signal false-positive tail

- Signal probability > 0.5: 3/64
- Signal probability > 0.9: 1/64
- Signal probability > 0.997: 0/64

The residual signal-grid response is diffuse. No strict
signal-probability event was observed.

## Memory strategy

Only one ONNX model session was resident at a time. Maximum Python
resident memory was approximately 2.70 GiB, and no new cgroup OOM kill
was observed.

## Scope

This is an unweighted 64-event control canary. It is not a precise
measurement of QCD rejection or the extreme score tail.

## Next control

The next released-model control is inclusive `TTbar_forInfer2`.
