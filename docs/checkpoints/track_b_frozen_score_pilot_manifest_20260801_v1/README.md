# Track B frozen-score pilot manifest

Date: 2026-08-01

## Status

    RESULT=TRACK_B_FROZEN_SCORE_PILOT_MANIFEST_VALID

This checkpoint freezes a bounded member-level train/validation pilot for a domain-transfer study of the released variable-mass HH4b ensemble on Track A.

No ROOT events or ONNX models were opened for inference while creating the pilot manifest. The only ROOT reads were provenance checks for the 27 supplemental ttbar mappings.

## Frozen development provenance

- 367 validated nonempty members;
- 340 primary candidate-path/checksum mappings;
- 27 supplemental unique ttbar-shard mappings;
- 218 synthesized empty members excluded;
- zero test or sealed-test members included;
- zero train/validation group overlap.

## Pilot limits

- deterministic member selection seed: `track-b-frozen-score-pilot-20260801-v1`;
- raw-event scan cap: 1024 per member;
- accepted-event cap: 256 per member;
- event indices are not yet frozen;
- physical weights are not used in this manifest-freeze step.

## Cross-check against Yang and Li

Reference: Tianyi Yang and Congqiao Li, *Potential of di-Higgs observation via a calibratable jet-free HH to 4b framework*, arXiv:2508.15048v2.

Released-model conventions retained exactly:

- 138 normalized output probabilities;
- 136 unordered variable-mass signal classes plus QCD and tt;
- event discriminant equal to the sum of the 136 signal probabilities;
- arithmetic mean of the three model discriminants for the ensemble discriminant.

Known Track B deviations:

- Track A raw EFlow is not the authors' pileup-mitigated reconstructed-particle input;
- fixed-mass SM HH is not the authors' flat variable-mass signal prior;
- exact JetClass-II detector, pileup, SophonAK4 trigger, and 4j3b parity is unavailable;
- the authors' event-level mass fit and correction are not yet applied.

The deterministic pilot quotas are a Track B transfer-study design. They are not claimed to reproduce the authors' training mixture or training statistics.

## Next gate

Open only the selected pilot sources, resolve each source locator, prove raw-ROOT-entry to candidate-cache event-key alignment, and freeze event indices before ONNX score production.
