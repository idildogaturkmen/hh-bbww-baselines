# STATUS — track_b_harvey_final_question_closure_20260907_v1 (Track D)

**COMPLETE / READY_FOR_HARVEY.** Purely additive package. No Track A/B/C
package, frozen checkpoint, or normalization contract was modified. No git
command was run and the git repository was not touched by this task.

## What this package is

The final Harvey-question closure pass. Section 0 (`HARVEY_QUESTION_
CLOSURE_MATRIX.md`) inventoried every substantive Harvey point against the
existing Track A/B/C record and found 11 of 16 already fully resolved.
Only the 5 genuinely remaining items were executed:

| Task | Question | Verdict |
|---|---|---|
| 1 | Does the 5th/extra jet causally perturb classification? | **Yes — causal perturbation confirmed** (real A/B counterfactual, executed for the first time) |
| 2 | Is the QCD tail rate stable as existing statistics increase? | u>3.5: **STABLE_WITHIN_CURRENT_MC**; u>4.5: **TOO_FEW_EVENTS_TO_JUDGE** |
| 3 | Are the FP32 spike bins kinematically special? | **NO_CLEAR_SPECIAL_MODE** |
| 4 | Does the exploratory ttbar cut validate on independent lanes? | **NOT_REPRODUCED** |
| 5 | Preselection: three-stage reconciliation | Compiled into one table; `pTHat` bias remains the only generation-cost lever, canary not run |

## Safety / receipt summary

- Governing checkpoint SHA256 verified: `fd9ea100825ecc0412a2666f01fd19d1a91b25a50109b958c307102fa6943de5` ✓
- Validation HDF5 SHA256 verified: `3c94bf8300d1dc3324e2cf25f9b6c618dffb94819cb3a297a43ac06320d5c2a2` ✓
- One new-inference run performed (Task 1 only): CPU-only, 400,000 events, 2 forward passes/event, model.eval()+torch.no_grad(), no training/gradient/optimizer step. Sanity check (masking correctness on the 232,662 events where it must be a no-op) passed exactly (max diff = 0.0).
- No training, no new MC generation, no Condor submission, no ParT, no package installs, no holdout_B/Stage-C access, no normalization-contract changes.
- SHA256SUMS generated and verified (`sha256sum -c SHA256SUMS` — all OK).

## Final printed summary

```
HARVEY_QUESTIONS_TOTAL = 16
ANSWERED_WITH_RESULT = 4
ANSWERED_WITH_LIMITATION = 1
REQUIRES_NEW_MC = 0
FIFTH_JET_COUNTERFACTUAL_RUN = YES
FIFTH_JET_COUNTERFACTUAL_RESULT = CAUSAL_PERTURBATION_CONFIRMED
QCD_EXISTING_STATS_STABILITY_RESULT_U35 = STABLE_WITHIN_CURRENT_MC
QCD_EXISTING_STATS_STABILITY_RESULT_U45 = TOO_FEW_EVENTS_TO_JUDGE
SPIKE_KINEMATICS_RESULT = NO_CLEAR_SPECIAL_MODE
TTBAR_SUPPRESSION_RESULT = NOT_REPRODUCED
NEW_TRAINING_LAUNCHED = NO
NEW_MC_GENERATION_LAUNCHED = NO
HOLDOUT_B_OR_STAGEC_ACCESSED = NO
FROZEN_PACKAGES_MODIFIED = NO
CHECKSUMS_PASS = YES
READY_FOR_HARVEY = YES
```
