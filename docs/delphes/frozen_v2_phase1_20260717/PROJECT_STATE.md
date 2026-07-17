# Frozen-v2 HH→4b production state

Last updated: 2026-07-17

## Repository branch

`delphes-hh4b-production`

## Frozen detector configuration

Card:

`cards/delphes/delphes_card_CMS_lpc_ak4ak8_run2_frozen_v2.tcl`

SHA256:

`1b2041245162de8360defdc502404e0696496c50773d4aa7f043798651a9517c`

## Completed Phase 1 production

Campaign:

`frozen_v2_phase1_reprocess_20260717`

Condor clusters:

- smoke: 84901349 on lpcschedd6.fnal.gov
- remaining production: 59823035 on lpcschedd5.fnal.gov

Validated totals:

- 25 shards
- 250,000 generated events
- 250,000 Delphes ROOT events
- 19,002 candidate rows
- 80 candidate columns
- 19,002 unique candidate event IDs

Processes:

- qcd_bbbb_general: 50,000 generated, 1,799 candidates
- qcd_bbbb_iht400to600: 100,000 generated, 15,554 candidates
- ttbar: 50,000 generated, 171 candidates
- zbbbb: 50,000 generated, 1,478 candidates

Phase 1 final audit status:

`PHASE1_FINAL_AUDIT_VALID`

## Storage locations

EOS inputs:

`/store/user/iturkmen/hh4b_delphes/run2_13tev/frozen_v2/phase1_inputs_20260717`

EOS outputs:

`/store/user/iturkmen/hh4b_delphes/run2_13tev/frozen_v2/phase1_reprocess_20260717`

EOS freeze archive:

`/store/user/iturkmen/hh4b_delphes/run2_13tev/frozen_v2/phase1_reprocess_20260717/provenance/freeze_record_20260717.tar.gz`

Freeze archive SHA256:

`3ccf63ee6abb912ce591c0cb979843b644511e3764dc508616ca2500262ee3a5`

## Gate 1 process discovery

Discovered:

- 12 MG5 process directories
- 43 retained HepMC files
- 25 Phase 1 HepMC files mapped exactly by path

Process families:

- ggf_hh4b: 3 process directories
- vbf_hh4b: 1
- qcd_bbbb: 2
- ttbar: 1
- ttbb: 1
- zbbbb: 1
- zh4b: 1
- zz4b: 1
- unresolved Zbb: 1

## Important unresolved issues

1. Current QCD MG5 card snapshots do not prove the exact generation
   settings of the retained `qcd_bbbb_iht400to600` files.
2. Exact per-run banners and generation metadata must be audited before
   treating any sample as a physically normalized evaluation sample.
3. Targeted QCD-bbbb, Zbbbb, and ttbb remain enrichment-only until
   overlap rules are proven.
4. Signal truth ancestry and four-jet matching must be validated before
   large ggF/VBF production.
5. Exact Delphes jet constituent references are not yet validated for
   constituent-based ML.
6. No final cut, BDT, DNN, or SPA-Net significance has been authorized.
7. No 500K or 5M production has been authorized.

## Current next gate

Gate 1B: exact HepMC generation-lineage audit using immutable MG5 run
banners, logs, card snapshots, and stored generation metadata.

No new event generation should be submitted until Gate 1B is reviewed.

## Gate 1C exact generation lineage

Status:

`GATE1C_EXACT_LINEAGE_VALID`

Validated:

- 25 exact HepMC-to-MG5-run mappings
- 25 unique run banners
- 25 unique HepMC hashes
- 25 unique seeds
- banner and propagated generator cross sections mutually consistent
  to a maximum relative difference of 5.49e-08

The general targeted QCD-bbbb campaign contains the IHT 400–600
region and is not additive with the targeted IHT 400–600 campaign.

## Current next gate

Audit and freeze the existing adaptive 500K physical-QCD importance
campaign before authorizing any duplicate QCD generation.

The separate ML-tail campaign must remain explicitly classified as
training enrichment and must not replace the physical QCD estimate.

## Physical-QCD adaptive 500K audit

Campaign:

`qcd_hardqcd_importance_adaptive500k_phys_20260717_0022`

Production status:

`VALID`

Statistical status:

`SIGNAL_LIKE_TAIL_NOT_CONVERGED`

Validated:

- 54 completed jobs
- 500,000 generated events
- 54 verified canonical bundles
- 4,266 cross-layer checks passed

Signal-like statistics:

- 27 HH-like rows, ESS 3.188
- 11 rHH<80 rows, ESS 3.169
- 5 rHH<50 rows, ESS 1.991
- maximum event fractions between 41% and 59%
- bootstrap uncertainties between 51% and 67%

The sealed test split contains only pTHat strata 0, 2, and 3.

The next approved action is preparation and review of one additional
adaptive physical-QCD wave with sealed test shards declared for missing
strata 1, 4, 5, 6, and 7.

No 5M campaign and no final ML inference are authorized.
