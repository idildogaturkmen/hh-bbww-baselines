STARTED_UTC = 2026-09-06T00:00:00Z
PACKAGE = track_b_harvey_tail_kinematics_20260907_v2 (methodological corrections to v1, v1 preserved untouched)

CORRECTION_1_MET_PRIMARY_SPLIT = COMPLETE (PRIMARY n_bkg=110, met_pt excluded; AUXILIARY_NONQCD_MET n=42, clearly labeled, not compared numerically to PRIMARY)
CORRECTION_2_TTBAR_ARITHMETIC_AND_TABLE = COMPLETE (14/27=51.85% rejected, process-by-process table, exploratory Z_A/B95)
CORRECTION_3_TRUTH_MATCH_AUDIT = COMPLETE (algorithm audited as many-to-one/non-bijective; exact maximum-cardinality/minimum-total-DeltaR bipartite matcher implemented, unit-tested, and run over the full 27,339-event population; heatmap regenerated with EXACT matching)
CORRECTION_3_RESOLUTION = IHEP VOMS/CMS proxy renewed by user 2026-09-07 (voms-proxy-info -timeleft confirmed positive lifetime, 43154s, before resuming); work/audit_correction3a_exact_vs_greedy.py ran to completion (200/200 files, 27339 events, 107s, zero file failures)
CORRECTION_4_COUNTERFACTUAL_PLAN = COMPLETE (plan only, not executed -- "leading-four-only counterfactual"; u_logit formula corrected to the exact softplus(Delta)/ln(10) definition, consistent with Track A, replacing an earlier draft's large-Delta approximation)

LAST_COMPLETED_STEP = All four corrections complete and verified; executive summary, fifth-jet summary, ttbar summary, 1D/2D/4D tables, heatmap figure, receipt.json, STATUS.md, SHA256SUMS all regenerated and consistent
CURRENT_STEP = COMPLETE

NOT_DONE_BY_DESIGN = leading-four-only counterfactual inference (Correction 4) intentionally NOT executed per instruction; no new SPA-Net inference, no training, no holdout_B/Stage-C access, no new MC generation
