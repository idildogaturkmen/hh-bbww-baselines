# BDT-v3 qcdplus topology-only / mass-blinded baseline

This output uses the same samples, weights, train/test split, and BDT hyperparameters as the frozen BDT-v3 qcdplus baseline.

Direct Higgs-candidate mass variables are removed from the BDT inputs:
- mbb1, mbb2, avg_mbb, delta_mbb
- r_hh, r_hh_125_125, r_hh_125_120
- mhh
- j1_mass, j2_mass, j3_mass, j4_mass
- ht_over_mhh

This is intended as a mass-sculpting validation, not as a replacement for the mass-aware baseline.

Remaining features:
- hh_pt
- hh_eta
- h1_pt
- h1_eta
- h2_pt
- h2_eta
- h_delta_eta
- h_delta_phi
- h_delta_r
- h_pt_balance
- drbb1
- drbb2
- n_selected_jets
- n_selected_bjets
- n_extra_selected_jets
- n_extra_selected_bjets
- ht_selected_jets
- ht_selected_bjets
- ht_candidate_jets
- j1_pt
- j2_pt
- j3_pt
- j4_pt
- j1_eta
- j2_eta
- j3_eta
- j4_eta
- pt_sum4
- pt_asym_12
- pt_asym_34
- avg_drbb
- max_drbb
- min_drbb

# Safe event-feature two-BDT HH4b baseline

This is the corrected event-feature version. It verifies that merging event-level features does not change the number of candidate rows.

All-candidate SM-normalized signal yield at 450/fb: 322.646760 events.

The previous non-safe event-feature output is invalid because merging on non-unique event IDs duplicated rows.
