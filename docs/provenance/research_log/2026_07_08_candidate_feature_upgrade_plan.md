# Candidate feature upgrade plan

The first BDT baseline shows that the all-feature model is mostly driven by reconstructed mass-plane variables:
- R_HH
- mHH
- mbb1, mbb2
- avg_mbb, delta_mbb

Topology-only BDT still has nontrivial separation, so the next useful improvement is to enrich the candidate parquet with more event and candidate features.

Candidate-level features to add:
- pT of each reconstructed Higgs candidate
- eta/phi of each reconstructed Higgs candidate
- deltaR, deltaEta, deltaPhi between the two Higgs candidates
- scalar HT of selected jets
- selected jet multiplicity
- selected b-tagged jet multiplicity
- extra selected jet count beyond the four b candidates
- extra b-tagged jet count beyond the four b candidates
- leading/subleading dijet pT
- pT balance between the two Higgs candidates
- min/max/mean jet eta
- b-tag values or b-tag scores if available from Delphes objects

Purpose:
- Test whether BDT/DNN performance improves beyond simple mass-window variables.
- Reduce over-reliance on R_HH and mbb variables.
- Better separate QCD bbbb from ttbar/top-heavy-flavor topologies.
