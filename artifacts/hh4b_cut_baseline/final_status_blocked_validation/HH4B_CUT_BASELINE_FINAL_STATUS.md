# Resolved HH→4b cut-baseline final status

Status: **BLOCKED AT VALIDATION BY PRE-ACCESS INFRASTRUCTURE FAILURE**

- Train-only cut-baseline study: **COMPLETE**.
- Train-only methodology and results: **FROZEN**.
- Validation attempt: infrastructure failure before event access.
- Validation scientific result: **NOT AVAILABLE**.
- Validation payloads opened: **0**.
- Test payloads opened: **0**.
- Corrected validation retry authorized: **FALSE**.

This is not a scientific failure of the cut. No scheduler failure is interpreted as a physics observation, no partial validation performance is derived, and no threshold, variable, family, search budget, nominal rule, methodology, or reporting choice was changed.

## Frozen nominal deployment candidate

- `exact3tag`: `r_hh_125_125 < 36.40814019639858` and `HT_candidate_jets > 176.5458068847656 GeV`.
- `ge4tag`: `r_hh_125_125 < 33.92808917804956`, `mHH > 164.73708096689654 GeV`, and `|Delta eta_HH| < 6.904302164473993`.

The historical primary simple cut reference remains `R_HH(125,125) < 34`, including for subsequent apples-to-apples model comparisons.

## Train-only performance

Primary generalization estimate: pooled five-fold nested outer-OOF. Values are a Run-2 expected-yield projection from Delphes simulation at `sqrt(s)=13 TeV`, `138 fb^-1` equivalent. These are not official CMS results and include no systematic uncertainties.

| Selection/scope | epsS | epsB | background rejection | S/B | stat-only ZA | signal Neff | background Neff |
|---|---:|---:|---:|---:|---:|---:|---:|
| nested outer-OOF / exact3tag | 0.61278496 | 0.29251071 | 3.4186783 | 2.9434585e-06 | 0.020527839 | 10873.547 | 60.813009 |
| nested outer-OOF / ge4tag | 0.60335384 | 0.15965908 | 6.2633458 | 3.2505947e-05 | 0.039421226 | 3643.0642 | 2.9036817 |
| nested outer-OOF / combined | 0.61039638 | 0.28553696 | 3.5021735 | 3.8111617e-06 | 0.026978118 | 14516.582 | 63.333592 |
| historical R_HH<34 / combined | 0.59043194 | 0.27724642 | 3.6068996 | 3.7967471e-06 | 0.026483034 | 14029.326 | 57.486605 |

The frozen optimized deployment cut has no material stat-only significance improvement over the historical simple cut. In the frozen 2,000-draw paired source-group bootstrap, the combined fixed-cut-minus-historical `ZA` difference has median `3.968605699912128e-05` and 95% interval `[-0.0008187977698690706, 0.0006179069891563046]`, spanning zero.

## All-1000 selection stability

The full study contains 270,000 ranked structure evaluations, 10,000 fold winners, and 2,000 replica/category reductions. The detailed optimized structures are not especially stable under resampling; this did not change the frozen nominal cut.

| Category | modal structure | modal frequency | nominal recovery | top-two gap | tie fraction | infeasible fold-winner fraction |
|---|---|---:|---:|---:|---:|---:|
| exact3tag | asymmetric_rectangular_mass__category__plus_abs_h_delta_eta | 0.185 | 0.119 | 0.051 | 0.380 | 0.5552 |
| ge4tag | asymmetric_rectangular_mass__category__mass_only | 0.207 | 0.034 | 0.082 | 0.380 | 0.1572 |

## Validation and test boundary

Exactly one campaign submission was accepted: cluster `3795859` on `lpcschedd4.fnal.gov`. All 116 jobs subsequently exited during the frozen runtime probe because the runner exported `$SCRATCH/runtime/site-packages`, while the authorized runtime archive stores `awkward` and the other packages under `lib/python3.9/site-packages`. The common exception was `ModuleNotFoundError: No module named 'awkward'`.

The audit proves queue=0, history=116, runtime-probe failures=116, durable markers=0, scientific workers=0, validation sources opened=0, validation evaluation cycles=0, validation payloads opened=0, and test payloads opened=0. Consequently no validation aggregation or validation-performance figure is scientifically allowed.

**Figure 17 / validation-comparison products were intentionally not produced because no validation event payload was opened and no valid validation performance measurement exists.** The completed publication set contains 16 train-only figures, each with PDF, PNG, and machine-readable sidecar.

Cluster 3795859 is permanently NEVER RESUBMIT. A corrected replacement validation campaign is not authorized by this report or this session.

## Immutable production registry

| Cluster | Role | Schedd | Never resubmit |
|---:|---|---|---|
| 3754344 | bounded_six_job_transfer_pilot | `lpcschedd4.fnal.gov` | YES |
| 3755882 | original_270_scan | `lpcschedd4.fnal.gov` | YES |
| 3768139 | four_job_transfer_pilot | `lpcschedd4.fnal.gov` | YES |
| 30002685 | initial200_stability | `lpcschedd5.fnal.gov` | YES |
| 30020809 | escalation800_stability | `lpcschedd5.fnal.gov` | YES |
| 3795859 | one_time_validation_preaccess_failure | `lpcschedd4.fnal.gov` | YES |

## Checkpoint closure

| Checkpoint | SHA256SUMS SHA256 | Last checkpoint commit |
|---|---|---|
| `docs/checkpoints/hh4b_cut_baseline_master_train_only_freeze_20260810_v1` | `c6adf047567cbb5f6eab4c4e25a6d6e1a40a541f1ab24f5af7c60ef04631b587` | `ba536089a86282437ab07f7a837fbffb1086204f` |
| `docs/checkpoints/hh4b_cut_baseline_master_train_only_freeze_20260811_v2` | `0d6a7b26bdfb4c086f123fa1973bc9fd2c7010de42d0f1db50651c2a4b418e9b` | `27c14307885a7ad9214360ac5be61183a4b5ed70` |
| `docs/checkpoints/hh4b_cut_baseline_master_train_only_freeze_20260811_v3` | `d3350869882317ea41bc2a1386b71aadd5802ba060048939bcb9a26f00a4d159` | `d69be420d01dd2b731aac17ed4ff33cc0691722e` |
| `docs/checkpoints/hh4b_cut_baseline_validation_authorization_20260810_v1` | `1b4332b36643bbb1689311ca80c7e41f0a34f801e22d2b7f7c40ed8aca43fe21` | `62726d179652d5b5681486533e27ccf7cdd45b53` |
| `docs/checkpoints/hh4b_cut_baseline_validation_authorization_20260811_v2` | `0762cde01035e052c8cdf3b85210a398a592e840191ee18ff1eca6905f26cfaf` | `70e5b4d8eeb5cba9686010f9730b5e5ab73c2b02` |
| `docs/checkpoints/hh4b_cut_baseline_validation_authorization_20260811_v3` | `4f9d6639e696a8abd7ef1e2bd4a53b29686d7f7e0e22e7c0a6a0eac765eece51` | `3e54a57a65aa572786a4816588c604898adfe40f` |
| `docs/checkpoints/hh4b_cut_baseline_validation_campaign_presubmission_20260811_v1` | `5194bcdd76dd6afb5661f83a2a65e3827285e65b2c11bd27c476b11ed728be46` | `b1b4e2cdd34a3c28a82112952f6104698ff38c5b` |
| `docs/checkpoints/hh4b_cut_baseline_validation_metadata_20260810_v1` | `ebfb27711dc61f7281915127beefc8d3f9491962cba0b4fd9a0498e0c9cf3c45` | `96c5733487462e6b7448d7efb7390e0ffc1e6e85` |
| `docs/checkpoints/hh4b_cut_baseline_validation_preaccess_failure_audit_20260811_v1` | `dcc10d5bfc0b6fdfed414a64d9942cc10d19d1c17c7a6a5e5b90999f989bf85f` | `24dc2c0751c5be88a55c7c7221e14a062270cff8` |
| `docs/checkpoints/hh4b_cut_baseline_validation_submission_20260811_v1` | `47657c1b9ad5b3980c606da65b6bb53cc34792b53e86f1a74506817912244f0f` | `b890468b0a64c2219955968c0250376593580237` |
| `docs/checkpoints/hh4b_cut_baseline_validation_transport_recovery_20260811_v1` | `1f6ab77eaef51bba3bd034add3c3faa3b2108b9db8e148b4ca97e7aeb64e96e6` | `6fd18aab8e6ddab5ce4045af54e20926cadbbccc` |
| `docs/checkpoints/hh4b_train_cut_baseline_performance_and_comparator_20260810_v1` | `89c98a7a80f2fd1f842c470a50f3e55c190460f8a80cfc5ac4eb7b9b48f1edb0` | `4f5893d4d828172170284484d6a1e1675ea4f372` |
| `docs/checkpoints/hh4b_train_cut_baseline_publication_figures_20260810_v1` | `4e270d328a96e418bb5b21c22736aa273d1d1176a6524731e164e478b7bd4345` | `1d4ff11b1706ecc1d7918115233ac45af5650071` |
| `docs/checkpoints/hh4b_train_historical_rhh125125_lt34_freeze_20260806_v1` | `b32fd988b5d29711a446a30296df609f0f3f96e2da49e05270158fcfa503b102` | `e715af2b1a976f8b5af6b76251e0998f42787ae1` |
| `docs/checkpoints/hh4b_train_multivariate_cut_and_figure_contract_20260806_v1` | `a0616ebc10f54b059bbbf0dc291ac69a0e2f1ffe61e13c217310349a737971f5` | `e99d0a09e038cc2699c777036c8540d57555ab6f` |
| `docs/checkpoints/hh4b_train_multivariate_cut_bounded_six_job_transfer_pilot_20260807_v1` | `7b034a2d9eb0716a8324322d953b543c47c6447e10002f82ba8c1dd4de51d80b` | `bac9ed51ada51d2d51e1593bfa8606b2cc9139f1` |
| `docs/checkpoints/hh4b_train_multivariate_cut_canary_acceptance_repair_20260806_v1` | `0144f3431070bac85afcc8bed73c07d4cc156cdfc12823710d3148312d6e8857` | `5b96183b489e206b0d22b75da370d71610de55ab` |
| `docs/checkpoints/hh4b_train_multivariate_cut_canary_only_configuration_20260806_v1` | `859df59b4206feeccb4e4df408ed6d31ec308a8c276bbe0847794ee5dc1a91f9` | `956dbd3febaf279ef98b3f6325e1c7e724e78945` |
| `docs/checkpoints/hh4b_train_multivariate_cut_category_endpoint_amendment_20260806_v1` | `9c7343054bb2fcb8b3175687e525d4e1a71955c0c023cba76c32dc4e733a0ae7` | `1cd21990ae07eeb503b858c1e57e7121b6a0e875` |
| `docs/checkpoints/hh4b_train_multivariate_cut_full_270_integrity_audit_20260807_v1` | `cc6e79d027749d4fb78f0950c4408b3e598aca97de3a963bd7f9b70a7e5ab5ae` | `23701eb96e850ee2e9174e5b5ec89cbb22246602` |
| `docs/checkpoints/hh4b_train_multivariate_cut_implementation_clarification_20260806_v1` | `9bdf5d420143e280736bc8008c47863027358e1cfdd57b8ba4f46b7f1550b20c` | `d830a84fdb94dc5c0c34c5c28978cbdc453dc9e9` |
| `docs/checkpoints/hh4b_train_multivariate_cut_passing_committed_canary_freeze_20260806_v1` | `e131aabf41559a19db7a31016d1de9cedcd41311fc1263370feed7bdd6a7b471` | `8aac74a38649e81e8cc0061a11aac8baed188e60` |
| `docs/checkpoints/hh4b_train_multivariate_cut_post_publication_transfer_provenance_canary_20260806_v1` | `c5c5b8cbcde554db7bd6928ed9eb58eb03ee6fd3313369b290da0edc97c2a89a` | `dfccc037be7c64dd4345d4bdb4418f41004d7d60` |
| `docs/checkpoints/hh4b_train_multivariate_cut_repo_native_optimizer_20260806_v1` | `1f1dce536c331c8af99697c2faf46f39c3f81cc465fe5c39b04858192b365120` | `11f8bc445983bf47bcca33827e138ca87a7e5a39` |
| `docs/checkpoints/hh4b_train_multivariate_cut_selection_stability_aggregation_protocol_20260808_v1` | `0cfc828ad073684080a79e45c71d84b103ade6c6ad7bde42bfa3ea78e645b8c1` | `b317d2c3270166edce86c70a3972e6db3b75c186` |
| `docs/checkpoints/hh4b_train_multivariate_cut_selection_stability_all1000_aggregation_20260810_v1` | `8a86e1dadc8c861a8d7da8461d4e95bd7b69275dc04d96997261471c058f7725` | `1f2b598230f5dbac82ce483ff593d31d46afbda4` |
| `docs/checkpoints/hh4b_train_multivariate_cut_selection_stability_draw_registry_20260808_v1` | `f148f6b470c0f41165a92fec3ee0163286087ae47089b38f85dc1078142b4119` | `1e52f208f5f4564b929cda1b6fe47b7795424273` |
| `docs/checkpoints/hh4b_train_multivariate_cut_selection_stability_escalation800_cluster_in_place_transport_recovery_20260810_v1` | `a540847d4624fe73d34a6863800fb273a079e5f7df84e323c522196f92b42cf7` | `fe995b16eda400a3e822a7f5117ee1b3e53c46a1` |
| `docs/checkpoints/hh4b_train_multivariate_cut_selection_stability_escalation800_complete_return_audit_20260810_v1` | `a423453bbe3554d0fa1bd0ec52e77dcca8ddeb51b6e039c423d061c41d73d0f9` | `9845fc97d4d35b3d532d4c892acb0f7095ba7e33` |
| `docs/checkpoints/hh4b_train_multivariate_cut_selection_stability_escalation800_cross_filesystem_publication_repair_20260810_v1` | `5dfecde94246f5986bbb3baaf86f4c7598e6202c9fd9cdb42445b0c96089dd3d` | `d11cca4909fb4fcf1052cc9ec357b748efc0a616` |
| `docs/checkpoints/hh4b_train_multivariate_cut_selection_stability_escalation800_eos_recovery_canary_20260810_v1` | `05ce4218e74ab79d3ce1e0f9c97372cbea4df4079d2d6777025ce06ed12880cb` | `75898d4bd05158ddedd95fbc15bff53b54a35d72` |
| `docs/checkpoints/hh4b_train_multivariate_cut_selection_stability_escalation800_eos_transfer_recovery_20260810_v1` | `eac4ace8a9ece7e66e0c8573a95160a36a96480d4a10344aa81e3f9f42efe6cb` | `515002fdb1ce01f4de3ca3a066f8cc92e07d81bd` |
| `docs/checkpoints/hh4b_train_multivariate_cut_selection_stability_escalation800_presubmission_package_20260810_v1` | `cc03fd11f7a66f55ecf32b958c7a616edbe04c8c1cc0bb4aa334b0e191028e90` | `5d730256593f346a9f47914980d9bb4796cbc23a` |
| `docs/checkpoints/hh4b_train_multivariate_cut_selection_stability_escalation800_storage_amendment_20260809_v1` | `f5b71b7aad8a6fe82c75a34a1cd8160f9f04509ebbd196b6c90ad04cf0cc1c8a` | `191c2c0737c6a60459ad95e62af41e33665d60d8` |
| `docs/checkpoints/hh4b_train_multivariate_cut_selection_stability_escalation800_submission_20260810_v1` | `bcc0129528f801774c8e7ccb5ff381b9fc5477149e58a03200fd478024d0f169` | `ca9064bef431491dba8490ca3b95dba0cc9d0ac7` |
| `docs/checkpoints/hh4b_train_multivariate_cut_selection_stability_escalation_200_999_authorization_20260809_v1` | `2bd620dff4fbc2d38f89f4f55da090f5d4183ac71925b0b40a891d77702dc9bf` | `f5bd2abc69e0ed3b91c6f9ecfb1fce128aa1f8b3` |
| `docs/checkpoints/hh4b_train_multivariate_cut_selection_stability_full_budget_pilot_and_granularity_20260808_v1` | `ed48ad3e16a32abcddf86a25f414e0e961a7af9fefdc87ca486f878dce32b520` | `5b8447897cc98f7df00d460274b857ec882266b8` |
| `docs/checkpoints/hh4b_train_multivariate_cut_selection_stability_initial200_aggregation_20260809_v1` | `3ca7c99ccd775313590e8dc09e2fe9044a30d8740575f63929af891e0fb333a4` | `18a4ff9b2723c0eae7a550ae5071c74001188a49` |
| `docs/checkpoints/hh4b_train_multivariate_cut_selection_stability_initial200_complete_return_audit_20260809_v1` | `d5a66de4dedfff24b44123203cd272ee471c256cd9dd2cd00274979448967800` | `fd2557fffee1c3f1a5a5d903c7fd624cad5c92c9` |
| `docs/checkpoints/hh4b_train_multivariate_cut_selection_stability_initial200_presubmission_package_20260808_v1` | `7854483f491456a6985bdd56ed136be21e5520fe4f4cab12086170c2f4f3fd03` | `d8e60031225005017c613908fba0671fd82e36d2` |
| `docs/checkpoints/hh4b_train_multivariate_cut_selection_stability_initial200_schedd_provenance_correction_20260808_v1` | `a7726b5d79597b2582922ce3711959e0516fe1c662fd5e88ff06ecaa973bf2e0` | `269449d2b0ac04ba1e4caee3b2a838e1e5ee21ef` |
| `docs/checkpoints/hh4b_train_multivariate_cut_selection_stability_initial200_submission_20260808_v1` | `0c75aab9d0e0930f69327b56ac3c983d075476eff4c15a0dcac4856ade7c3ebb` | `e2e8ff6032af921b6639e7c26359aad233a8c47f` |
| `docs/checkpoints/hh4b_train_multivariate_cut_selection_stability_reoptimization_canary_20260808_v1` | `5b78602a40044946d095cd5d7a1b1cb4a0aa72e2a5d550cf7739ca3d7088ca1a` | `86954fb70686a8c135caa6a431788e6541819075` |
| `docs/checkpoints/hh4b_train_multivariate_cut_selection_stability_transfer_pilot_and_initial200_authorization_20260808_v2` | `8247c46d2fb5808dc09b35a78764c46d1c5efa4f442e1b1b45553575b3f5a413` | `eb474c715cd38a3eb0193213ea69959e66331b08` |
| `docs/checkpoints/hh4b_train_multivariate_cut_train_only_deployment_candidate_20260807_v1` | `ba8710ba3c682042cef20a04f2c77f37aebcc7f50a918c160743b2ff4de3f256` | `4ec362d4832bd07600188ef29ade55282af3a9d0` |
| `docs/checkpoints/hh4b_train_multivariate_cut_train_only_nested_oof_aggregation_20260807_v1` | `02a64ccc2959d64ddfcb66b252b158b1305f560eba8e7310f078f1466077647b` | `8f94b553ec2eb81dbdc0087da062be24ac179b0e` |

The pre-access failure checkpoint is `docs/checkpoints/hh4b_cut_baseline_validation_preaccess_failure_audit_20260811_v1` with SHA256SUMS hash `dcc10d5bfc0b6fdfed414a64d9942cc10d19d1c17c7a6a5e5b90999f989bf85f`. The frozen runner hash is `20c41377c2aaadf5d2de26c3698bac570022e81c81105a30889af6c50e3e59c1`; the runtime bundle hash is `4a9c4061e1d975f276326022454e578c5f0de3c094b9b56345f192cccc654112`.

## Remaining decision

`NEXT_DECISION_REQUIRED=determine separately whether a replacement validation campaign is scientifically/procedurally permissible given that the failed campaign never opened validation payloads`

No cross-model final test evaluation may occur before that separate decision and any newly authorized, provenance-complete validation procedure. `VALIDATION_SCIENTIFIC_RESULT_AVAILABLE=FALSE`, `VALIDATION_PAYLOADS_OPENED=0`, `TEST_PAYLOADS_OPENED=0`, and `CORRECTED_VALIDATION_RETRY_AUTHORIZED=FALSE`.
