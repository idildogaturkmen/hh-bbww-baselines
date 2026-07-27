| level | level_name | validation | required | comparison | pass_condition | failure_action | byte_identity_claimed |
|---|---|---|---|---|---|---|---|
| 1 | exact_configuration_and_seed_reproduction | generated_event_count | True | integer_exact | exactly 10000 | fail_member_block_scaleout_no_publish | False |
| 1 | exact_configuration_and_seed_reproduction | requested_seed | True | integer_exact | receipt and log equal frozen seed | fail_member_block_scaleout_no_publish | False |
| 1 | exact_configuration_and_seed_reproduction | generator_completion | True | log_and_exit | exit zero and completion marker | fail_member_block_scaleout_no_publish | False |
| 1 | exact_configuration_and_seed_reproduction | pythia_completion | True | log_and_exit | exit zero and Pythia 8.312 marker | fail_member_block_scaleout_no_publish | False |
| 1 | exact_configuration_and_seed_reproduction | delphes_completion | True | log_and_exit | exit zero and Delphes marker | fail_member_block_scaleout_no_publish | False |
| 1 | exact_configuration_and_seed_reproduction | root_entry_count | True | integer_exact | Delphes entries exactly 10000 | fail_member_block_scaleout_no_publish | False |
| 1 | exact_configuration_and_seed_reproduction | root_readability | True | structural | ROOT and Delphes tree readable | fail_member_block_scaleout_no_publish | False |
| 1 | exact_configuration_and_seed_reproduction | canonical72_schema | True | schema_exact | 72 columns in frozen order | fail_member_block_scaleout_no_publish | False |
| 1 | exact_configuration_and_seed_reproduction | required_finite_values | True | numeric | all required values finite | fail_member_block_scaleout_no_publish | False |
| 1 | exact_configuration_and_seed_reproduction | unique_event_keys | True | key_exact | no duplicate sample/event keys | fail_member_block_scaleout_no_publish | False |
| 1 | exact_configuration_and_seed_reproduction | source_member_identity | True | categorical_exact | member identity exact | fail_member_block_scaleout_no_publish | False |
| 1 | exact_configuration_and_seed_reproduction | output_checksums | True | sha256 | all artifacts in receipt | fail_member_block_scaleout_no_publish | False |
| 2 | generator_event_identity_where_stable_comparison_available | generator_event_identity | conditional_on_stable_comparator | stable_event_comparator | event identity where stable comparison is available | fail_member_block_scaleout_no_publish | False |
| 3 | delphes_branch_level_event_identity_where_stable_comparison_available | delphes_branch_event_identity | conditional_on_stable_comparator | stable_branch_comparator | branch identity where stable comparison is available | fail_member_block_scaleout_no_publish | False |
| 4 | canonical_candidate_agreement_on_15_legacy_common_columns | legacy_common_columns | True | predeclared_15_column_contract | row/key/exact fields pass and differences are explained | fail_member_block_scaleout_no_publish | False |
| 4 | canonical_candidate_agreement_on_15_legacy_common_columns | cross_member_duplicate_keys | True | aggregate_key_exact | zero duplicates across regenerated members | fail_member_block_scaleout_no_publish | False |
