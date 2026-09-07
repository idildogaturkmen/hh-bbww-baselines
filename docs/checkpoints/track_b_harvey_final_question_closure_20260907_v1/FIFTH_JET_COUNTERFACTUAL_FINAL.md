# Fifth-jet / extra-jet counterfactual — executed (highest-priority new result)

Harvey asked directly whether the fifth jet actually "perturbs" a signal
event's classification. Every prior package answered this only
observationally (events with a 5th jet score lower, on average, than
events without one) and explicitly flagged that as association, not a
tested causal effect. **This task executes the causal test for the first
time.**

## Setup, exactly as specified and verified

- **(A) Original input**: the native SPA10M input tensor, unchanged mask,
  scored by the unmodified frozen governing checkpoint.
- **(B) Leading-four-only counterfactual**: identical event, identical
  jet ordering/kinematics for slots 0–3, with `mask[:, 4:] = False`. Jet
  *data* values are never touched — only the boolean mask.
- Both scored through the identical `model.forward()` code path, same
  frozen checkpoint, same 400,000-event `production_2M_val.h5` cohort and
  row order used by every prior frozen evaluation in this project.
- **Checkpoint SHA256 verified**: `fd9ea100825ecc0412a2666f01fd19d1a91b25a5
  0109b958c307102fa6943de5` ✓ exact match.
- **HDF5 SHA256 verified**: `3c94bf8300d1dc3324e2cf25f9b6c618dffb94819cb3a
  297a43ac06320d5c2a2` ✓ exact match.
- **Sanity check** (built into the script, not a post-hoc claim): for every
  one of the 232,662 events with `n_selected_jets<=4` (where masking slot
  ≥4 changes nothing, since those slots were already empty), `max|Δu_logit|
  = 0.0` exactly — the masking implementation is verified correct, not
  merely assumed.
- CPU-only, `model.eval()` + `torch.no_grad()` throughout. No optimizer, no
  gradient, no `Trainer.fit()`. No Stage-C/holdout_B file referenced. No
  ParT code touched. No package installed. Forward pass (both A and B):
  157.0s wall for 400,000 events (2548 events/s, both passes combined).
- `u_logit = softplus(Δ)/ln(10)`, `Δ = z1 - z0` — the exact, non-saturating
  definition specified in the (previously unexecuted) plan, not the
  FP32-saturating stored `u`.
- **This is an "all-extra-jets-removed" / leading-four-only counterfactual,
  not a jet-5-only isolation.** The optional direct jet-5-only test (C) was
  evaluated against the instruction's precondition — proving from the
  architecture that arbitrary mask patterns preserve intended set semantics
  — and that proof was not established with confidence in the time
  available for this pass, so **C was not run**. B tests the effect of the
  full extra-jet set (ranks ≥5, up to 6 additional jets for the largest
  `n_selected_jets=10` events), not jet 5 in isolation.

## Primary result: yes, the extra jets causally move the score

Population: 88,854 signal events with `n_selected_jets>=5` (of 193,358
total signal events in the cohort).

| quantity | value |
|---|---:|
| median Δu_logit (B−A) | **−0.0902** (68% bootstrap CI [−0.0928, −0.0879]) |
| mean Δu_logit (B−A) | **−0.1484** (68% bootstrap CI [−0.1512, −0.1458]) |
| fraction more signal-like after masking (Δ>0) | 41.9% |
| fraction less signal-like after masking (Δ<0) | 58.1% |
| fraction exactly unchanged | 0.0% |

**Every single one of the 88,854 events changes score under masking — zero
are unaffected.** The bulk effect is small and slightly negative on
average (removing the extra jets slightly *lowers* the score for most
events, i.e. the extra jet set on average contributes real, if modest,
signal-supporting information) — but a substantial minority (42%) move the
other way.

## Threshold crossings (the practical question: does this change tail membership?)

| threshold | n pass, native (A) | n pass, masked (B) | gained (A≤thr→B>thr) | lost (A>thr→B≤thr) |
|---|---:|---:|---:|---:|
| u_logit > 3.5 | 5,374 | 7,622 | **2,578** | 330 |
| u_logit > 4.5 | 3,334 | 6,165 | **3,441** | 610 |

At both working points, far more events **gain** tail membership when the
extra jets are masked than **lose** it (2,578 vs. 330 at u>3.5; 3,441 vs.
610 at u>4.5) — even though the bulk median/mean effect above is slightly
negative. This is not a contradiction: it means the events sitting *near*
the classifier's decision boundary are disproportionately drawn from the
41.9% subpopulation for whom the extra jet(s) were mildly confusing the
classifier, while deep in the bulk (far from any threshold) the extra
jet(s) more often help. Figure: `figures/threshold_crossings_counterfactual.png`.
Full distribution: `figures/delta_u_logit_distribution.png`.

## Stratification by pT5

| pT5 range | n | median Δu_logit |
|---|---:|---:|
| 30–50 GeV | 59,111 | −0.061 |
| 50–75 GeV | 23,455 | −0.192 |
| 75–100 GeV | 4,683 | −0.297 |
| 100–150 GeV | 1,462 | −0.326 |
| ≥150 GeV | 143 | −0.563 |

**Monotonic**: the harder the extra jet, the more its removal hurts the
score (more negative Δ) — a physically sensible, and now causally
demonstrated, dependence. Figure: `figures/delta_u_logit_vs_pt5.png`.

## Stratification by min ΔR(j5, leading four)

| min ΔR range | n | median Δu_logit |
|---|---:|---:|
| 0.0–0.5 | 4,999 | −0.021 |
| 0.5–1.0 | 36,534 | −0.141 |
| 1.0–1.5 | 26,843 | **−0.232** |
| 1.5–2.0 | 12,809 | −0.031 |
| ≥2.0 | 7,669 | **+0.159** |

**Non-monotonic, and genuinely new**: jets at intermediate separation
(ΔR 1.0–1.5 from the leading four) contribute the *most* positive
information (their removal hurts the most); very isolated extra jets
(ΔR≥2.0) actually *hurt* the classification when included (their removal
*helps*, Δ>0). Track B's v2 package explicitly removed an unsupported
ΔR-bin physical-origin interpretation (no truth-origin branch exists to
attribute bins to ISR/FSR/pileup). **This result does not reopen that
interpretation** — no origin-truth information was used here either — but
it does establish, for the first time, that the ΔR dependence is not
merely observational/correlative: masking causally reproduces the same
qualitative pattern. Figure: `figures/delta_u_logit_vs_minDR5.png`.

## What was not attempted, and why (disclosed, not silently skipped)

- **HH-origin truth-match stratification**: would require joining this
  400k development-cohort event identity (`production_2M_val.h5` row
  index) to the physical master table's truth-matching convention
  (`signal_holdout_A` row identity) — a different signal sample. No proven
  join key exists between the two. Not attempted rather than assumed.
- **SPA-Net assignment-output recovery** (does the predicted 4-jet
  assignment change; was original rank-5 in the assignment): a legitimate
  ground-truth target *does* exist for this population
  (`TARGETS/h1/{b1,b2}`, `TARGETS/h2/{b3,b4}` in `production_2M_val.h5`),
  so this is technically feasible in a follow-up — but decoding the
  model's assignment head is a separate, more expensive computation not
  run in this pass. Not invented or estimated here.
- **(C) direct jet-5-only isolation**: not run — see "Setup" above.

## Bottom line

**Yes — the fifth (and any further extra) jet causally perturbs SPA-Net's
signal classification for the same event.** Every affected event's score
changes under masking (never zero), the bulk effect is a modest net
positive contribution from the extra jets (median/mean Δu_logit negative
under removal), and the effect depends systematically and non-trivially on
both the extra jet's pT (monotonic) and its angular separation from the
leading four (non-monotonic, peaking at intermediate ΔR). This upgrades
Harvey's original observational finding to a demonstrated causal one.
