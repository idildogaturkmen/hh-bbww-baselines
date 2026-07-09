# Two-BDT event-feature HH4b baseline

This extends the candidate-level two-BDT baseline by merging event-level variables from the event_summary parquet files:
- n_jet_pt30_eta25
- n_bjet_pt30_eta25
- n_extra_jets_pt30_eta25
- n_extra_bjets_pt30_eta25
- ht_pt30_eta25
- ht_over_mhh

The goal is to test whether event-level/top-sensitive information improves the weak BDT_top baseline.
