# Track-B: physical normalization of four HH→4b development models

## QCD-ONLY BACKGROUND — STATISTICAL-ONLY — NOT FINAL TOTAL EXPECTED SENSITIVITY

This directory archives a physically-normalized comparison of four
frozen HH→4b development models, translating classifier output scores
into expected signal and QCD-background event yields at common
Standard-Model (SM) signal efficiencies.

## The four models

- **BDT-K** — gradient-boosted decision tree using reconstructed-jet
  kinematics only (52 features: per-jet pT/eta/phi/mass over up to 10
  jets, jet count, HT).
- **BDT-KF** — the same kinematic features plus continuous per-jet
  flavor-tag probabilities (probB/probC/probL), 82 features total.
- **SPA-Net 2M** — a symmetry-aware neural network that jointly
  performs jet assignment and event classification, trained on 2
  million events, using the same seven per-jet physical inputs
  (pT, eta, phi, mass, probB, probC, probL) as BDT-KF.
- **SPA-Net 10M** — the identical architecture, trained on 5x the
  event population (10 million events).

All four are **development** artifacts: the BDT models are seed-0
convergence runs (a fixed 6000-round training budget was reached
without early stopping firing), not the final full-population fit; the
SPA-Net models are single-seed development checkpoints. See
`MODEL_FREEZE.md` for exact architecture, training population, and
checkpoint-selection details, and the model SHA256 hashes that
identify the exact frozen artifacts used.

## Purpose

Earlier comparisons of these four models were reported mainly in terms
of classifier metrics (ROC AUC, background rejection at a target
efficiency). This archive instead translates each model's raw score
into a **physically normalized** expected yield: for a chosen SM
signal efficiency, how many signal and QCD-background events are
actually expected at a given integrated luminosity, and what
significance does that imply.

## Normalization

| Quantity | Value |
|---|---:|
| SM efficiency-measurement population | 508,168 |
| Full stored SM population | 2,538,851 |
| SM fixed weight at 450 fb$^{-1}$ | 0.000473265 |
| QCD inference population | 87,623,306 |
| QCD fixed weight at 450 fb$^{-1}$ | 7.3112875413134075 |

```
epsilon_SM = N_SM_pass / 508168
S(L)       = epsilon_SM * 2538851 * 0.000473265 * (L/450)

epsilon_QCD = N_QCD_pass / 87623306
R_QCD       = 87623306 / N_QCD_pass
B_QCD(L)    = N_QCD_pass * 7.3112875413134075 * (L/450)

S/B, S/sqrt(B)
Z_A = sqrt(2*((S+B)*ln(1+S/B) - S))
```

`epsilon_SM` is measured on a fixed 508,168-event SM sample; `S(L)`
extrapolates that measured efficiency to the full stored SM population
of 2,538,851 events (the complete set of SM events retained after the
dataset's own ntuple-level selection — not the total number of
generated events, and not a population that was itself directly
scored). QCD always uses the complete 87,623,306-event stored QCD
population. Full definitions, thresholding rule, and constants are in
`NORMALIZATION_CONTRACT.md` and `NORMALIZATION_CONSTANTS.json`.

## Finite QCD Monte Carlo

At tight signal-efficiency working points, very few simulated QCD
events pass the selection — sometimes zero. Zero simulated QCD events
does **not** mean zero physical background, and reporting an
undefined-but-technically-computable "infinite" rejection or
significance in that regime would be misleading. This archive instead
reports, alongside the central (best-estimate) value, an exact
one-sided 95% Poisson upper bound on the QCD background
($B_{QCD,95}$) and the resulting conservative significance
$Z_A(B_{95})$. For observed QCD counts below 20, $Z_A(B_{95})$ is the
recommended primary quantity and the central value is a diagnostic
only; at exactly zero observed QCD events, only $Z_A(B_{95})$ is
reported (central background, rejection, S/B, S/sqrt(B), and Z_A are
undefined). See `FINITE_MC_REPORTING_POLICY.md` for the exact rule and
its statistical basis.

## Observed pattern (development comparison — not a final ranking)

- **Continuous flavor information gives the dominant improvement**
  between the two BDT models: adding probB/probC/probL to the
  kinematic feature set (K → KF) produces, by a wide margin, the
  largest single gain in QCD rejection seen in this comparison.
- **SPA-Net gives an additional, statistically robust gain over
  BDT-KF** in the region with adequate raw QCD Monte Carlo support
  (roughly SM efficiency 20% down to 7.5%).
- **The 2M → 10M training-population increase does not improve the
  model uniformly.** It is mildly worse than 2M at looser working
  points, and only becomes a clear, still-adequately-supported
  improvement at tighter working points within the statistically
  robust range.
- **The tightest working points are limited by available QCD Monte
  Carlo statistics**, not by the underlying physics — see
  `FINITE_MC_REPORTING_POLICY.md` and the `support` label attached to
  every row of the result tables.

## Files

- `NORMALIZATION_CONTRACT.md` — exact equations, constants, and
  threshold-definition rule.
- `NORMALIZATION_CONSTANTS.json` — the same constants in machine-readable form.
- `FINITE_MC_REPORTING_POLICY.md` — the finite-QCD-count reporting rule.
- `MODEL_FREEZE.md` — the four models' architecture, training
  population, and checkpoint-selection rule, with SHA256 identity.
- `CANONICAL_FOUR_MODEL_NORMALIZED_RESULTS.csv` / `.md` — the full
  frozen result set: all 17 target signal efficiencies x 4 models (68
  rows), with raw counts, derived physical quantities, and support
  labels.
- `MAIN_MODEL_COMPARISON.csv` / `.md` — a compact subset of the above
  at the working points most useful for a quick comparison.
- `FULL_COMMON_GRID_TABLE.csv` / `.md` — the complete 17-efficiency x
  4-model grid in one flat table, including both originally-scanned
  and supplementally-completed rows (see provenance labels).
- `YANG_LI_REFERENCE_AND_BACKGROUND_BUDGET.csv` / `.md` — comparison
  against an external published stat-only reference significance (see
  below).
- `NORMALIZATION_MAPPING_CAVEAT.md` — explicit caveats on how this
  local normalization does and does not map onto external references.
- `SOURCE_PROVENANCE.md` — how these numbers were produced and
  cross-checked.
- `SHA256SUMS` — checksums for every file in this directory.
- `za_b95_vs_actual_sm_efficiency.pdf` / `.png` — main figure:
  $Z_A(B_{95})$ vs. actual achieved SM efficiency, all four models.
- `qcd_rejection_vs_actual_sm_efficiency.pdf` / `.png` — backup
  figure: QCD rejection vs. actual SM efficiency (log scale; included
  because a log-scale rejection plot can visually overstate the
  significance of the least-supported, tightest working points — see
  the main figure for the recommended comparison).

## Yang-Li external reference

A published stat-only significance ($S=20.5$, $B=32.7$,
$Z_A=3.283763859$) is included **only as an external reference point**,
not as a target this analysis claims to match or exceed. This
analysis's local signal/QCD normalization and event selection do not
map one-to-one onto the reference's background composition — no ad hoc
correction was applied to force numerical agreement. See
`NORMALIZATION_MAPPING_CAVEAT.md` and
`YANG_LI_REFERENCE_AND_BACKGROUND_BUDGET.md` for the full statement of
this caveat.

## Scope

QCD-only background. Statistical uncertainty only (no systematics).
Not a final total-background sensitivity estimate. No event-level data
or model/checkpoint binaries are included in this archive.
