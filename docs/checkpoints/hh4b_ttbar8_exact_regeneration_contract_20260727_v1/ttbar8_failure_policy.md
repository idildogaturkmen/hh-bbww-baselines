| failure_class | detection | job_action | retry_policy | seed_policy | publication_policy | scaleout_effect |
|---|---|---|---|---|---|---|
| input_checksum_mismatch | preflight SHA256 | exit before generation | no_automatic_retry | seed_never_changes | no_canonical_publication_on_failure | block_scaleout_if_canary_or_unexplained_physics |
| existing_output | exclusive target preflight | exit before generation | no_automatic_retry | seed_never_changes | no_canonical_publication_on_failure | block_scaleout_if_canary_or_unexplained_physics |
| generator_failure | nonzero exit or missing completion | hold for review | no_automatic_retry | seed_never_changes | no_canonical_publication_on_failure | block_scaleout_if_canary_or_unexplained_physics |
| event_count_mismatch | LHE event count | hold for review | no_automatic_retry | seed_never_changes | no_canonical_publication_on_failure | block_scaleout_if_canary_or_unexplained_physics |
| pythia_failure | nonzero exit or missing marker | hold for review | no_automatic_retry | seed_never_changes | no_canonical_publication_on_failure | block_scaleout_if_canary_or_unexplained_physics |
| delphes_failure | nonzero exit or missing marker | hold for review | no_automatic_retry | seed_never_changes | no_canonical_publication_on_failure | block_scaleout_if_canary_or_unexplained_physics |
| root_validation_failure | readability or entries | hold for review | no_automatic_retry | seed_never_changes | no_canonical_publication_on_failure | block_scaleout_if_canary_or_unexplained_physics |
| reconstruction_failure | nonzero exit or schema/content check | hold | no_automatic_retry | seed_never_changes | no_canonical_publication_on_failure | block_scaleout_if_canary_or_unexplained_physics |
| identity_disagreement | exactness levels 1-4 | hold and adjudicate | no_automatic_retry | seed_never_changes | no_canonical_publication_on_failure | block_scaleout_if_canary_or_unexplained_physics |
| transfer_failure | size/SHA256 mismatch | hold and retain partial logs | no_automatic_retry | seed_never_changes | no_canonical_publication_on_failure | block_scaleout_if_canary_or_unexplained_physics |
| condor_held_job | HoldReason/ClassAd | manual inspection only | no_automatic_retry | seed_never_changes | no_canonical_publication_on_failure | block_scaleout_if_canary_or_unexplained_physics |
