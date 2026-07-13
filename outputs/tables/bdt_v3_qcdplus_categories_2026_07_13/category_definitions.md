| category                    | definition                                                | intended_use                                                        |
|:----------------------------|:----------------------------------------------------------|:--------------------------------------------------------------------|
| CAT0_high_purity_diagnostic | bdt_qcd_score >= 0.850 and bdt_top_score >= 0.850         | Diagnostic only unless effective background statistics are adequate |
| CAT1_tight                  | bdt_qcd_score >= 0.850 and 0.500 <= bdt_top_score < 0.850 | Tight high-score category                                           |
| CAT2_medium_tight           | 0.800 <= bdt_qcd_score < 0.850 and bdt_top_score >= 0.500 | Medium-tight category around best supported qcd threshold           |
| CAT3_medium                 | 0.700 <= bdt_qcd_score < 0.800 and bdt_top_score >= 0.500 | Medium category                                                     |
| CAT4_loose                  | 0.500 <= bdt_qcd_score < 0.700 and bdt_top_score >= 0.500 | Loose/control-like selected category                                |
