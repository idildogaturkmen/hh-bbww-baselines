# SPA-Net readiness notes

This audit checks whether the current v2 candidate parquets are enough for a toy SPA-Net-style model.

Interpretation:
- If j1/j2/j3/j4 pt/eta/phi/mass are present, a toy 4-candidate-jet assignment/classification model is possible.
- A full SPA-Net model should use leading-N selected jets, not only the four already selected candidate jets.
- Full SPA-Net also needs truth assignment labels mapping reconstructed jets to H1/H2/background.
- If only candidate jets are stored, the next step is to build a ROOT-to-SPA-Net dataset writer using Jet branches and generator-level H→bb truth daughters.
