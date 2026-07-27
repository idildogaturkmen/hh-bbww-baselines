| order | condition | observed | operator | required | passed | failure action |
|---|---|---|---|---|---|---|
| 1 | lower_point_estimate_background_efficiency | 0.0377709065254 | < | 0 | false | global_v1_mass_aware |
| 2 | minimum_relative_background_efficiency_reduction | -0.176610479384 | >= | 0.02 | false | global_v1_mass_aware |
| 3 | bootstrap_fraction_favoring_categorized_v2 | 0.027 | >= | 0.84 | false | global_v1_mass_aware |
| 4 | signal_mode_equalization | 0.0306839044859 | <= | 0.05 | true | global_v1_mass_aware |
| 5 | integrity_category_and_model_application | 0 | == | 0 | true | global_v1_mass_aware |
| 6 | frozen_rule_outcome | global_v1_mass_aware | selected_by | all_five_conditions_or_global_fallback | true | none |
