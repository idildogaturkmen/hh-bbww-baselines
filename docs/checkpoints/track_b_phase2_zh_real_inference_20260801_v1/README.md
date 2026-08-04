# Track B Phase 2: real ZH released-model inference

## Status

The three released SophonHH ONNX models and their arithmetic
ensemble were validated on 64 deterministic selected events from
the real IHEP ZH control sample.

The inference was run at commit `9a666d62fab9dd4dee4f1d18edb59bcf3bf967b6`. This
checkpoint is being added later on the same linear branch history
at parent commit `0711c8a3c500fbd6c5c38a6036e2086f6e5641e8`.

## Principal diagnostics

- Mean signal probability: `0.749217948499`
- Mean QCD probability: `0.159643788461`
- Mean ttbar probability: `0.0911382584912`
- Model top-1 agreement fraction: `0.703125`
- Signal probability strictly greater than 0.997: `15/64`
- Leading mass-grid class: `62`
- Leading first mass bin: `[80.0, 90.0]`
- Leading second mass bin: `[120.0, 130.0]`

The leading response lies near the intended Z/H mass combination,
providing the expected qualitative resonance-control behavior.

## Validations

- Deterministic Step 20 event selection reproduced exactly.
- All three model score matrices are normalized.
- The arithmetic ensemble is normalized.
- The official 140-branch Events tree round-tripped exactly.
- No particle truncation occurred.
- No network access was used during inference.
- No persistent binary artifact was written.

## Scope limitation

This is an unweighted 64-event control canary. It does not replace
the pending independent-HH validation.

## Next controls

The next samples are ZZ, QCD, and inclusive `TTbar_forInfer2`.
