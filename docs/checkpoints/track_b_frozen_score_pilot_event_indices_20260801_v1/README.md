# Track B frozen-score pilot event indices

Date: 2026-08-01

## Status

    RESULT=TRACK_B_FROZEN_SCORE_PILOT_EVENT_INDICES_VALID

This checkpoint freezes exact event identities for all 53
members in the previously committed Track B frozen-score pilot.

## Alignment prerequisite

The eight-member source-access and event-key alignment canary
passed for train and validation signal, QCD, ttbar, and
additional-background categories:

- 8 of 8 members passed;
- 40 candidate-to-ROOT rows were checked;
- all reported jet pT, eta, phi, and mass differences were zero;
- zero sealed-test members were opened;
- no ONNX inference was executed.

The canary shell later encountered an extraneous literal `BASH`
command after the scientific success marker. The committed
receipt records that post-success wrapper artifact separately;
it does not invalidate the completed canary.

## Event selection

For each pilot member, all validated candidate event identities
were read from the already-frozen candidate Parquet cache.
Candidate rows were ranked deterministically by a SHA256 digest
of the selection seed, member ID, ROOT event entry, and candidate
row index. At most the frozen per-member selected-event cap was
retained.

This step opened no ROOT files, staged no EOS bundles, ran no
ONNX models, and used no physical event weights.

## Cross-check against Yang and Li

Reference: Tianyi Yang and Congqiao Li,
arXiv:2508.15048v2.

The authors' model is a 138-class particle-level classifier using
all pileup-mitigated reconstructed particles, with 136 unordered
variable-mass signal classes plus QCD and tt classes. This
event-index checkpoint is infrastructure for applying the
released model to Track A raw EFlow. It is not claimed to
reproduce the authors' particle representation, training
statistics, detector setup, or calibration workflow.

## Next gate

Run a bounded direct-probability ONNX scoring canary on the
frozen event identities. The canary must use all three pinned
models, apply no external softmax, retain the full 138-class
outputs, and write only to `/tmp`.
