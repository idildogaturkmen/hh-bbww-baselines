# HH4b reconstruction v2 validation

Tested `reconstruct_hh4b_candidates_v2.py` on the kept ttbb_50k shard000 Delphes ROOT file.

Result:
- v1 rows: 637
- v2 rows: 637
- common events: 637
- events only in v1: 0
- events only in v2: 0

This means v2 preserves the event-level requirement of at least four selected b-tagged jets on this shard.

Candidate-level variables change for a subset of events:
- mbb1 changed in about 34% of events
- mbb2 changed in about 35% of events
- avg_mbb, delta_mbb, and mHH changed in about 13% of events

This is expected because v2 adds stable h1/h2 ordering, additional pairing diagnostics, and richer event/candidate-level variables. v2 should be treated as a new reconstruction schema and not mixed with v1 outputs in the same nominal analysis without clear labeling.

Useful new features include:
- n_selected_jets
- n_selected_bjets
- n_extra_selected_jets
- n_extra_selected_bjets
- HT variables
- h1/h2 pT, eta, phi
- h_delta_eta, h_delta_phi, h_delta_r
- h_pt_balance
- jet eta/phi/mass/btag/flavor diagnostics
