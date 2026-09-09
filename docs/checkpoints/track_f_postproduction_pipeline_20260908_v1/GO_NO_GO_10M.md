# GO / NO-GO rule: ParT 2M → ParT 10M

**Status: rule prepared, not evaluated. No 10M ParT production authorized or
launched by this package.** This document exists so that, once the 2M
matched comparison (EVALUATION_2M_PART_RUNBOOK.md) has actually run, the
decision to scale to 10M is made against a predeclared rubric fixed *before*
seeing the 2M result -- not by eyeballing which number is larger afterward.

## Why a predeclared rule, and why it must be more than "the number went up"

1. **Scaling alone buys little here.** The project's own native-feature-only
   scaling study (2M → 10M, identical architecture/seed/optimizer, per
   `configs/FROZEN_FACTS.json:native_10m_baseline_for_go_nogo_reference`)
   moved all-background AUC by **less than 0.001** (0.96928 → 0.9693),
   QCD AUC by -0.0004, and ttbar AUC by +0.002. Five times more native-only
   training data bought essentially nothing on these headline numbers. Any
   ParT effect that is smaller than this *null* 5x-data effect is not
   obviously worth a ~5x-larger, much slower production.
2. **10M ParT production is expensive and operationally risky.** The
   project's own estimate for the 8M *new-category* embeddings needed on
   top of the existing 2M is **~232h single-worker wall time**
   (`SPANET_PART_COMPARISON_CONTRACT_v2.md`), and the exact_2M production
   has already required multiple credential-expiry firefighting rounds
   (IHEP X.509/VOMS proxy, dangling `BEARER_TOKEN_FILE`) to reach even its
   current 747/880 + 7/220 state. Scaling 5x multiplies this operational
   risk, not just the compute.
3. **The ParT/AK4 domain-mismatch question is still open.** The frozen ParT
   checkpoint was trained on JetClass large-R jets (R=0.8, 500-1000 GeV) and
   is applied here to AK4 jets (R=0.4, pT>30 GeV). The project's own
   carried-forward verdict is `REVIEW_NEEDED`/`DOMAIN_SHIFT_REQUIRES_VALIDATION`,
   **never resolved to PASS**, at any scale. Scaling up before this is even
   discussed would compound an unresolved validity question, not test it.

## GO requires ALL of the following (predeclared, evaluated against the
matched 2M EVALUATION_RESULT.json)

| # | Gate | Predeclared threshold | Where computed |
|---|---|---|---|
| G1 | Paired bootstrap CI for Δ(all-background AUC), TEST−CONTROL | **excludes zero AND `ci_low > +0.001`** (must clear the null 5x-data-scaling effect measured natively, not just be "positive") | `paired_bootstrap_delta_auc.all_background` |
| G2 | Same, QCD-only and ttbar-only | at least one of the two also has `ci_low > 0` (need not both clear +0.001, but must not be negative) | `paired_bootstrap_delta_auc.{qcd,ttbar}` |
| G3 | Reconstruction: McNemar paired test | **NOT** (`continuity_corrected_chi2_p_value < 0.05` AND `test_better_than_control == False`) -- i.e. TEST must not be *significantly worse* at event-level reconstruction correctness | `mcnemar_reconstruction_correctness` |
| G4 | Rejection at tight working points (4%, 3%) | TEST's `rejection` point estimate at each working point is not below CONTROL's `exact_poisson_count_interval.count_ci_low`-implied rejection (i.e. any apparent TEST degradation at a sparse working point must not exceed what Poisson noise on CONTROL's own count already allows) | `rejection_at_fixed_efficiency.{control,test}.epsS_0.04/0.03` |
| G5 | Stratification robustness | the AUC improvement (or at minimum non-degradation, `auc_test >= auc_control`) holds in the **`ge5`** jet-multiplicity stratum specifically, not only in `eq4` -- a signal-topology-driven effect should not vanish exactly where extra jets (and therefore extra ParT-embedded objects) matter most | `jet_multiplicity_stratified_auc.ge5` |
| G6 | Pipeline integrity | the 2M postflight/join/preprocess pipeline that produced this result reports zero identity/mask/checksum failures (`OVERALL_POSTFLIGHT_PASS`, `OVERALL_JOIN_VERIFICATION_PASS`, `all_finite` all `true` throughout) -- a result built on a pipeline that needed exceptions is not a basis for a scale-up decision | postflight/join/build receipts |
| G7 | Operational readiness | the exact_2M gap-fill resume completed within a bounded number of credential-refresh interventions (no open-ended firefighting) -- a documented, human judgment call recorded in this file at decision time, not a computed number | production STATUS.md at decision time |

**GO** = G1 AND G2 AND G3 AND G4 AND G5 AND G6 hold, **and** G7 is judged
acceptable by whoever makes the call. A GO decision must still carry the
open ParT/AK4 domain-mismatch caveat forward explicitly into any 10M
proposal -- it does not resolve that question, it only says the 2M evidence
is strong enough to justify testing it at 10x the AK4-domain-shift risk.

## NO-GO if ANY of the following hold

- G1's CI includes zero or is negative (no reliable classification
  improvement at 2M -- scaling representation effects up rarely creates
  signal that wasn't there at smaller N).
- G3 shows CONTROL significantly better at reconstruction (TEST actively
  hurts assignment quality).
- G4 shows a tight-working-point rejection collapse beyond Poisson noise
  (a sparse-tail regression that would only get worse, not better, at
  larger scale if the underlying cause is a genuine feature-quality issue).
- G6 fails for any reason (a pipeline that isn't clean at 2M should not be
  trusted to build a correct 10M input either -- fix it at 2M first).
- Non-finite embeddings, an identity/mask mismatch, or a checkpoint-hash
  mismatch is found anywhere in the 2M pipeline at decision time, even if
  the numeric result otherwise looks favorable.

## What GO actually authorizes (and does not)

A GO verdict from this rubric authorizes, at most, **preparing** an 8M
new-category ParT embedding production plan (mirroring exact_2M's own
manifest/shard/postflight design) and a matched SPA-Net 10M-native vs
10M+ParT training plan -- consistent with `SPANET_PART_COMPARISON_CONTRACT_v2.md`'s
own "secondary, conditional" framing. It does **not** authorize:
- launching that 8M production (a separate, explicit decision + this
  project's own standing prerequisite: "must not be launched until the 2M
  producer fix has passed its own canary" -- already satisfied per
  `track_b_part_producer_v3_ihep_auth_fix_20260908_v1`, but re-verify at
  decision time),
- full144 in any form,
- touching `holdout_B`/Stage C/JP-JEPA.

## How to evaluate this rule once the 2M result exists

```bash
python3 - <<'PYEOF'
import json
r = json.load(open("EVALUATION_RESULT.json"))  # from EVALUATION_2M_PART_RUNBOOK.md

g1 = r["paired_bootstrap_delta_auc"]["all_background"]
G1 = g1["excludes_zero"] and g1["ci_low"] > 0.001

g2q = r["paired_bootstrap_delta_auc"]["qcd"]["ci_low"] > 0
g2t = r["paired_bootstrap_delta_auc"]["ttbar"]["ci_low"] > 0
G2 = g2q or g2t

mc = r["mcnemar_reconstruction_correctness"]
G3 = not (mc["continuity_corrected_chi2_p_value"] < 0.05 and not mc["test_better_than_control"])

def wp(model, eff):
    return r["rejection_at_fixed_efficiency"][model][f"epsS_{eff}"]["all_background"]
G4 = all(
    (wp("test", eff)["rejection"] or 0) >= (wp("control", eff).get("exact_poisson_count_interval", {}).get("count_ci_low", 0))
    for eff in (0.04, 0.03)
)

strat = r["jet_multiplicity_stratified_auc"]["ge5"]
G5 = strat.get("auc_test", 0) >= strat.get("auc_control", 1)

print(f"G1={G1} G2={G2} G3={G3} G4={G4} G5={G5}")
print("MECHANICAL_GATES_ALL_PASS (G1-G5):", all([G1, G2, G3, G4, G5]))
print("Still requires G6 (pipeline integrity receipts) and G7 (human operational judgment) before an actual GO.")
PYEOF
```
