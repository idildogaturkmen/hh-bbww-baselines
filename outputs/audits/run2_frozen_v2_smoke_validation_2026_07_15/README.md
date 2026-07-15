# Frozen-v2 Delphes-card smoke validation

The old and new ROOT files were produced from identical HepMC events.

Required validation conditions:

1. Jet multiplicities and kinematics should be identical.
2. Differences should be confined to flavor association and b tagging.
3. Observed tagging efficiencies should be statistically compatible with
   the formulas encoded in the Delphes card.
4. Candidate-count changes must be documented before the card is frozen.

The old card uses JetFlavorAssociation DeltaR=0.5.
The new candidate card uses DeltaR=0.4.
