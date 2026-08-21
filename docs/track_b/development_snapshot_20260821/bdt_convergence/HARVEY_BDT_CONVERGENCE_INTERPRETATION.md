# Interpretation — Harvey's round-budget question

## "What is the early stopping criterion, specifically?"

Monitor: validation-set logloss, minimize, patience 50 rounds, `save_best=True`. Verified directly from the frozen `convergence_gate_step2f.py`/`j_train_convergence.py` code used to produce these exact runs — not from memory or documentation.

**Neither K nor KF triggered it.** Both ran the full 6000-round cap. This was already known from step2j/step2k, but this diagnostic adds the *quantitative* reason why: at no point in either 6000-round run did the model go more than 13 consecutive rounds without at least a marginal new best validation logloss (K: max gap 8 rounds; KF: max gap 13 rounds) — nowhere close to the 50-round threshold. The curve *looks* flat because the remaining improvements are tiny in *magnitude*, but they are not tiny in *frequency* — they kept arriving often enough to keep resetting the patience counter the entire way to round 6000.

## "Change in loss between 200, 1000, 4000, and 6000 rounds — how flat does the plot become?"

**BDT-K**: val logloss falls from 0.528945 (round 200) to 0.505537 (1000) to 0.496148 (4000) to 0.494990 (6000). The 200→1000 interval (800 rounds) contributes 0.023407 (+4.425% in magnitude) of improvement; the 4000→6000 interval (2000 rounds — 2.5x more rounds) contributes only 0.001157 (+0.233% in magnitude) — roughly 20.2x less improvement from a longer interval. From round 5950 to 6000 (the very tail), val logloss moves by only -1.43e-05 — i.e. the curve is flat to 4-5 decimal places by the end, exactly consistent with the near-overlap of the raw and smoothed curves visible in the figures.

**BDT-KF**: val logloss falls from 0.288252 (round 200) to 0.238045 (1000) to 0.226978 (4000) to 0.225868 (6000). The 200→1000 interval (800 rounds) contributes 0.050207 (+17.418% in magnitude) of improvement; the 4000→6000 interval (2000 rounds — 2.5x more rounds) contributes only 0.001110 (+0.489% in magnitude) — roughly 45.2x less improvement from a longer interval. From round 5950 to 6000 (the very tail), val logloss moves by only -2.09e-05 — i.e. the curve is flat to 4-5 decimal places by the end, exactly consistent with the near-overlap of the raw and smoothed curves visible in the figures.

## Is the 4000→6000 gain substantial, small-but-measurable, or negligible relative to the extra 50% round cost?

**BDT-K**: -0.233% relative change in val logloss (**small but measurable**) for +50% training rounds (projected 52.4h → 78.6h at full 144M scale, +26.2h). The improvement is real, monotonic, and consistently in the same direction — it is not noise — but it is quantitatively small compared to the extra compute cost.

**BDT-KF**: -0.489% relative change in val logloss (**small but measurable**) for +50% training rounds (projected 48.33h → 72.5h at full 144M scale, +24.2h). The improvement is real, monotonic, and consistently in the same direction — it is not noise — but it is quantitatively small compared to the extra compute cost.

This diagnostic reports the objective numbers only. Selecting a final round budget for the 144M fit is explicitly **not** done here, per instruction — that remains a separate decision for Harvey/the team, informed by (but not made by) this diagnostic.

