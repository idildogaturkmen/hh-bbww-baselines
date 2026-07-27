| check | expected | observed | passed | evidence |
| --- | --- | --- | --- | --- |
| starting_commit | 582ab0a1f15cc8f990ac30aac31b17669975e842 | 582ab0a1f15cc8f990ac30aac31b17669975e842 | True | required four-command starting-state gate |
| contract_status | hh4b_ttbar8_exact_regeneration_contract_frozen | hh4b_ttbar8_exact_regeneration_contract_frozen | True | /uscms_data/d3/iturkmen/repos/hh-bbww-baselines/docs/checkpoints/hh4b_ttbar8_exact_regeneration_contract_20260727_v1/summary.json |
| protected_artifacts_verified | 17 | 17 | True | fresh contract audit |
| canary_member_seed_events | ttbar_100k_shard003\|105003\|10000 | ttbar_100k_shard003\|105003\|10000 | True | frozen deterministic selection |
| canary_jobs_prepared | 1 | 1 | True | /uscms_data/d3/iturkmen/hh4b_delphes/condor_submit/hh4b_ttbar8_exact_regeneration_canary_submission_20260727_v1/hh4b_ttbar8_exact_regeneration_canary.sub |
| scaleout_jobs_prepared | 7 | 7 | True | /uscms_data/d3/iturkmen/repos/hh-bbww-baselines/docs/checkpoints/hh4b_ttbar8_exact_regeneration_contract_20260727_v1/hh4b_ttbar8_scaleout.sub |
| previously_submitted_canary_jobs | 0 | 0 | True | queue and bounded history audit on LPC schedds 4, 5, and 6 |
| previously_submitted_scaleout_jobs | 0 | 0 | True | queue and bounded history audit on LPC schedds 4, 5, and 6 |
| unique_output_receipt_log_and_temporary_paths | True | True | True | local and EOS no-overwrite checks before directory creation |
| canary_submit_queue_processes | 1 | 1 | True | /uscms_data/d3/iturkmen/hh4b_delphes/condor_submit/hh4b_ttbar8_exact_regeneration_canary_submission_20260727_v1/hh4b_ttbar8_exact_regeneration_canary.sub |
| scaleout_submit_remains_unsubmitted | True | True | True | /uscms_data/d3/iturkmen/repos/hh-bbww-baselines/docs/checkpoints/hh4b_ttbar8_exact_regeneration_contract_20260727_v1/hh4b_ttbar8_scaleout.sub |
| condor_dry_run_classads | 1 | 1 | True | /uscms_data/d3/iturkmen/repos/hh-bbww-baselines/outputs/agent_runs/hh4b_ttbar8_exact_regeneration_canary_submission_20260727_v1/canary.dryrun.ads |
| focused_unit_tests_passed | 12 | 12 | True | tests/test_prepare_hh4b_ttbar8_exact_regeneration.py |
| git_diff_check | pass | pass | True | git diff --check before submission |
