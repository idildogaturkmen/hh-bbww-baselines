| model or method | decision class | role | nominal | validation predeclared | selection basis | status |
|---|---|---|---|---|---|---|
| global_v1_mass_aware | primary_nominal_bdt | canonical_nominal_model | true | true | strong cut-matched rejection; simpler global strategy; v2 gain negligible and bootstrap-unstable | frozen |
| categorized_cms_inspired_mass_aware | secondary_categorized_bdt | advanced CMS-inspired categorized alternative | false | true | predeclared categorized alternative; not validation-selectable | frozen |
| global_v1_explicit_dijet_mass_plane_blind | diagnostic_ablation | global explicit dijet-mass-plane-blind diagnostic | false | true | diagnostic only | retained_non_nominal |
| categorized_v1_features_mass_aware_ablation | diagnostic_ablation | category-split-only diagnostic | false | false | train-only diagnostic only | retained_non_nominal |
| categorized_cms_inspired_explicit_dijet_mass_plane_blind | diagnostic_ablation | categorized explicit dijet-mass-plane-blind diagnostic | false | false | train-only diagnostic only | retained_non_nominal |
| optimized_cut | cut_baseline | optimized R_HH cut baseline | false | true | frozen benchmark baseline | retained |
