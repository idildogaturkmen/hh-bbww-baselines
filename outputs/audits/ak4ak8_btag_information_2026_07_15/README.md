# AK4/AK8 b-tag information audit

This audit searches recursively through Delphes split-branch paths.

Interpretation:

- Values {0, 1}: one binary tagging decision.
- Several integer values: possible Delphes working-point bit mask.
- Many continuously distributed values: possible score-like variable.

Delphes BTag fields must not be described as DeepJet discriminators unless
a calibrated continuous discriminator has explicitly been implemented.
