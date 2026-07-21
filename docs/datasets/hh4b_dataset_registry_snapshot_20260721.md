# HH→4b Dataset Registry Snapshot

Snapshot date: 2026-07-21

## Accounting summary

- Receipt-backed successful events: 1,340,400
- Native counted events: 40,000
- Promoted legacy counted events: 750,000
- Counted completed events after overlay: 790,000
- Submitted events pending validation: 400,000
- Projected counted events after successful scale-out validation: 1,190,000
- Provisional remaining events to 5M: 3,810,000

Submitted events remain pending and are not classified as completed in this snapshot.

## Sample-family registry

| Family | Display name | Role | Group | Existing successful | Counted completed | Submitted pending | Projected counted | Candidates | Status | Training | Physics yields |
|---|---|---|---:|---:|---:|---:|---:|---:|---|---|---|
| bbh_hbb_4fs | bbh_hbb_4fs | background | single_higgs | 0 | 0 | 0 | 0 | 0 | planned_template_or_pilot_pending | False | False |
| ggf_hh4b_sm | ggF_HH4b_SMnorm | signal | hh_signal | 0 | 0 | 0 | 0 | 0 | legacy_existing_lineage_audit_required | REVIEW_REQUIRED | False |
| ggh_hbb | ggh_hbb | background | single_higgs | 0 | 0 | 0 | 0 | 0 | planned_template_or_pilot_pending | False | False |
| qcd_bbbb_general | qcd_bbbb_general | background | qcd | 150,000 | 0 | 0 | 0 | 5,416 | receipt_backed_existing | False | False |
| qcd_bbbb_iht400to600 | qcd_bbbb_iht400to600 | background | qcd | 150,000 | 0 | 0 | 0 | 23,100 | receipt_backed_existing | False | False |
| schannel_single_top | schannel_single_top | background | single_top | 0 | 0 | 0 | 0 | 0 | planned_template_or_pilot_pending | False | False |
| tchannel_antitop | tchannel_antitop | background | single_top | 0 | 0 | 0 | 0 | 0 | planned_template_or_pilot_pending | False | False |
| tchannel_top | tchannel_top | background | single_top | 0 | 0 | 0 | 0 | 0 | planned_template_or_pilot_pending | False | False |
| triboson_zbb | triboson_zbb | background | diboson_triboson | 0 | 0 | 0 | 0 | 0 | planned_template_or_pilot_pending | False | False |
| ttbar_inclusive | ttbar_inclusive | background | top | 750,000 | 750,000 | 0 | 750,000 | 2,439 | receipt_backed_existing | True | False |
| ttbb_diagnostic | ttbb diagnostic | diagnostic_background | top_heavy_flavor | 0 | 0 | 0 | 0 | 0 | legacy_existing_lineage_audit_required | REVIEW_REQUIRED | False |
| tth_hbb | tth_hbb | background | rare_top | 10,100 | 10,000 | 100,000 | 110,000 | 1,201 | receipt_backed_existing_plus_submitted_pending_receipt | True | False |
| tttt | tttt | background | rare_top | 10,100 | 10,000 | 100,000 | 110,000 | 1,877 | receipt_backed_existing_plus_submitted_pending_receipt | True | False |
| ttw | ttw | background | rare_top | 0 | 0 | 0 | 0 | 0 | planned_template_or_pilot_pending | False | False |
| ttz_zbb | ttz_zbb | background | rare_top | 10,100 | 10,000 | 100,000 | 110,000 | 1,111 | receipt_backed_existing_plus_submitted_pending_receipt | True | False |
| tw_antitop | tw_antitop | background | single_top | 0 | 0 | 0 | 0 | 0 | planned_template_or_pilot_pending | False | False |
| tw_top | tw_top | background | single_top | 0 | 0 | 0 | 0 | 0 | planned_template_or_pilot_pending | False | False |
| vbf_hbb | vbf_hbb | background | single_higgs | 10,100 | 10,000 | 100,000 | 110,000 | 6 | receipt_backed_existing_plus_submitted_pending_receipt | True | False |
| vbf_hh4b_sm | VBF_HH4b_SMnorm | signal | hh_signal | 0 | 0 | 0 | 0 | 0 | legacy_existing_lineage_audit_required | REVIEW_REQUIRED | False |
| wh_hbb | wh_hbb | background | single_higgs | 0 | 0 | 0 | 0 | 0 | planned_template_or_pilot_pending | False | False |
| ww | ww | background | diboson_triboson | 0 | 0 | 0 | 0 | 0 | planned_template_or_pilot_pending | False | False |
| wz_zbb | wz_zbb | background | diboson_triboson | 0 | 0 | 0 | 0 | 0 | planned_template_or_pilot_pending | False | False |
| zbbbb | zbbbb | background | zjets_heavy_flavor | 250,000 | 0 | 0 | 0 | 7,284 | receipt_backed_existing | True | False |
| zh4b_forced | zh4b_forced | background | single_higgs | 0 | 0 | 0 | 0 | 0 | failed_or_incomplete_only | False | False |
| zz4b_forced | zz4b_forced | background | diboson | 0 | 0 | 0 | 0 | 0 | failed_or_incomplete_only | False | False |

## Interpretation rules

- `receipt_backed_existing` requires exact LHE, HepMC, ROOT, receipt, and EOS lineage.
- `submitted_pending_receipt` is planned or running production and is not yet completed.
- Training authorization does not imply physical-yield authorization.
- `REVIEW_REQUIRED` entries must not be used silently in final training or inference.
- Final-test production remains unauthorized.

## Provenance

- Inventory: `unified_background_production_inventory_20260721.tsv`
- Contract: `unified_background_phase1_contract_20260721.tsv`
- Promotion overlay: `unified_background_phase2_ttbar_promotion_overlay_20260721.tsv`
- Scale manifest: `unified_background_5m_waveb_priority_scale400k_20260721_v1_manifest.csv`
- Wave-B plan: `unified_background_5m_waveb_importance_plan_20260721.tsv`
- Manual catalog: `hh4b_manual_sample_catalog_20260721.tsv`
