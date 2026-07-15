# AK4/AK8 b-tag information audit

This audit determines whether the current Delphes files contain only binary
tag decisions, multiple working-point bits, or a continuous discriminator.

Interpretation:

- Two unique values, usually 0 and 1: binary working-point decision.
- Several integer values or powers of two: possible working-point bit mask.
- Many continuously distributed values: continuous tag-score candidate.

The final production card must be frozen before large-scale sample production.
