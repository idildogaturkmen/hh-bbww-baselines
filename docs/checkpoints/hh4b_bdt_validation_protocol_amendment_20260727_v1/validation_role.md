| scope | activity | authorization | frequency | constraint | status |
|---|---|---|---|---|---|
| validation_split | split_role | frozen | exactly_once_in_next_gate | model_selection_and_generalization_split | predeclared |
| model_fitting | use_validation_during_fitting | prohibited | never | fit_only_on_complete_frozen_train_population | frozen |
| validation_split | compare_global_v1_with_categorized_v2 | permitted | exactly_once_in_next_gate | predeclared_protocol_only | frozen |
| validation_split | select_final_nominal_bdt | permitted | exactly_once_in_next_gate | predeclared_protocol_only | frozen |
| validation_split | check_optimized_cut_baseline | permitted | exactly_once_in_next_gate | predeclared_protocol_only | frozen |
| validation_split | evaluate_predeclared_mass_plane_blind_diagnostic | permitted | exactly_once_in_next_gate | predeclared_protocol_only | frozen |
| validation_split | new_feature_engineering | prohibited | never | no_post_inspection_reoptimization | frozen |
| validation_split | changing_450_GeV_boundary | prohibited | never | no_post_inspection_reoptimization | frozen |
| validation_split | adding_categories | prohibited | never | no_post_inspection_reoptimization | frozen |
| validation_split | changing_model_hyperparameters | prohibited | never | no_post_inspection_reoptimization | frozen |
| validation_split | retraining_hyperparameter_searches | prohibited | never | no_post_inspection_reoptimization | frozen |
| validation_split | changing_training_weights | prohibited | never | no_post_inspection_reoptimization | frozen |
| validation_split | choosing_new_background_families | prohibited | never | no_post_inspection_reoptimization | frozen |
| validation_split | changing_reconstruction | prohibited | never | no_post_inspection_reoptimization | frozen |
| validation_split | reopening_validation_after_inspecting_results | prohibited | never | no_post_inspection_reoptimization | frozen |
