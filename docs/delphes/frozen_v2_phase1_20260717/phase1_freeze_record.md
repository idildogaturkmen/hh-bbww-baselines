# Frozen-v2 Phase 1 production record

Campaign: frozen_v2_phase1_reprocess_20260717

## Validated production

- Shards: 25
- Generated events: 250,000
- Delphes ROOT events: 250,000
- Candidate rows: 19,002
- Candidate columns: 80
- Unique candidate event IDs: 19,002
- Detector card:
  1b2041245162de8360defdc502404e0696496c50773d4aa7f043798651a9517c

## Samples

- qcd_bbbb_general: 50,000 generated; 1,799 candidates
- qcd_bbbb_iht400to600: 100,000 generated; 15,554 candidates
- ttbar: 50,000 generated; 171 candidates
- zbbbb: 50,000 generated; 1,478 candidates

## Scientific status

This campaign validates production, detector simulation, reconstruction,
provenance, schema and dataset splitting.

The targeted QCD-bbbb and Zbbbb samples are training-enrichment samples
unless a later process-overlap audit establishes an exclusive physical
normalization.

No final cuts, BDT, DNN or SPA-Net result should be quoted from this
campaign as the final physical sensitivity estimate.
