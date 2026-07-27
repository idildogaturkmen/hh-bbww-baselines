| method | decision class | weighted oof auc | low mhh weighted oof auc | high mhh weighted oof auc | cut matched weighted signal efficiency | cut matched weighted background efficiency | interpretation |
|---|---|---|---|---|---|---|---|
| global_v1_mass_aware | primary_nominal | 0.7737142628735474 |  |  | 0.586048406198 | 0.184305471151 | frozen primary; substantially better rejection than cut |
| categorized_cms_inspired_mass_aware | secondary_categorized |  | 0.753744498261 | 0.805324916529 | 0.586061481041 | 0.183893493127 | predeclared secondary; negligible nominal-point gain with bootstrap interval spanning zero |
| global_v1_explicit_dijet_mass_plane_blind | diagnostic_ablation | 0.7317255338237391 |  |  | 0.585979511964 | 0.228052621132 | non-nominal diagnostic |
| categorized_v1_features_mass_aware_ablation | diagnostic_ablation |  | 0.732298600605 | 0.80113168093 | 0.586086077713 | 0.196387017332 | non-nominal diagnostic |
| categorized_cms_inspired_explicit_dijet_mass_plane_blind | diagnostic_ablation |  | 0.720446872512 | 0.768686433991 | 0.586060436882 | 0.235698239994 | non-nominal diagnostic |
| optimized_cut | cut_baseline |  |  |  | 0.585957314769 | 0.236726985161 | optimized R_HH < 34 cut baseline |
