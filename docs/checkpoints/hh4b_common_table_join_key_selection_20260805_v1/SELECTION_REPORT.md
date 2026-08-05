# Common-table population and join-key selection

## Train accounting

- all train event rows: 3,799,873;
- resolved model rows: 1,171,072;
- non-broad train rows retained for accounting: 2,628,801;
- validation and test remain sealed.

The 1,171,072 rows are not the full train split. They are the resolved-model view of the complete 3,799,873-row train table.

## Top coefficient-table candidates

- score 36: `hh4b_physical_normalization_run2_coefficient_analytical_closure_freeze_20260731_v2/ordinary_campaign_run2_allocation_registry.tsv` (run2_yield_coefficient,cross_section_coefficient,effective_reference_factor_pb,campaign_sumw)
- score 32: `hh4b_physical_normalization_run2_coefficient_analytical_closure_freeze_20260731_v2/hard_qcd_shard_run2_coefficient_registry.tsv` (run2_yield_coefficient,cross_section_coefficient,pthat,shard)
- score 29: `hh4b_physical_normalization_run2_coefficient_analytical_closure_freeze_20260731_v2/ordinary_process_run2_coefficient_registry.tsv` (run2_yield_coefficient,cross_section_coefficient,effective_reference_factor_pb)
- score 26: `hh4b_physical_normalization_run2_coefficient_analytical_closure_freeze_20260731_v2/nominal_zero_coefficient_exclusion_registry.tsv` (run2_yield_coefficient,cross_section_coefficient,shard)
- score 23: `hh4b_physical_normalization_hard_qcd_global_weight_contract_freeze_20260731_v1/hard_qcd_shard_weight_contract.tsv` (run2_yield_coefficient,pthat,shard)
- score 23: `hh4b_physical_normalization_hard_qcd_global_weight_contract_freeze_20260731_v1/hard_qcd_testfill_exclusion_contract.tsv` (run2_yield_coefficient,pthat,shard)
- score 13: `hh4b_physical_normalization_hard_qcd_global_weight_contract_freeze_20260731_v1/hard_qcd_exclusive_bin_weight_contract.tsv` (pthat,shard)
- score 13: `hh4b_physical_normalization_run2_coefficient_analytical_closure_freeze_20260731_v2/hard_qcd_bin_run2_analytical_closure.tsv` (pthat,shard)
- score 13: `hh4b_physical_normalization_train_candidate_physical_weight_sidecar_materialization_20260731_v1/train_candidate_physical_weight_sidecar_registry.tsv` (transport_id,population_kind)
- score 10: `hh4b_physical_normalization_global_ordinary_registry_freeze_20260731_v1/global_ordinary_process_registry.tsv` (effective_reference_factor_pb)
- score 6: `hh4b_physical_normalization_train_candidate_physical_weight_sidecar_materialization_20260731_v1/train_candidate_physical_weight_sign_closure.tsv` (population_kind)
- score 2: `hh4b_physical_normalization_global_ordinary_registry_freeze_20260731_v1/batch_A_reconstructed_process_denominator_binding.tsv` ()
- score 2: `hh4b_physical_normalization_global_ordinary_registry_freeze_20260731_v1/full_inventory_event_reconciliation.tsv` ()
- score 2: `hh4b_physical_normalization_global_ordinary_registry_freeze_20260731_v1/global_ordinary_campaign_denominator_binding.tsv` ()
- score 2: `hh4b_physical_normalization_global_ordinary_registry_freeze_20260731_v1/ordinary_process_classification.tsv` ()
- score 2: `hh4b_physical_normalization_global_ordinary_registry_freeze_20260731_v1/ordinary_vs_hard_qcd_disjointness.tsv` ()
- score 2: `hh4b_physical_normalization_global_ordinary_registry_freeze_20260731_v1/source_evidence_manifest.tsv` ()
- score 2: `hh4b_physical_normalization_global_ordinary_registry_freeze_20260731_v1/unresolved_inventory_discrepancy.tsv` ()
- score 2: `hh4b_physical_normalization_hard_qcd_global_weight_contract_freeze_20260731_v1/hard_qcd_campaign_extension_binding.tsv` ()
- score 2: `hh4b_physical_normalization_hard_qcd_global_weight_contract_freeze_20260731_v1/hard_qcd_overlap_exclusion_policy.tsv` ()

No event payload rows were read. No model was trained.
