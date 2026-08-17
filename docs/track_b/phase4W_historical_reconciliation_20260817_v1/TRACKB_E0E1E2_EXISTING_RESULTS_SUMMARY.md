# E0/E1/E2 Existing Results: Summary (Read-Only, No Recomputation)

All numbers below are transcribed from already-frozen, checksum-verified
Phase-4L (`internal_val`, development) and Phase-4M (`holdout_A`, one-time)
artifacts, and were additionally independently re-derived from the raw
per-seed `eval_*.json` files by Phase-4O
(`VERIFICATION_REPORT.md`) -- both derivations agree exactly except for
the one corrected prose transcription noted below. Nothing here was
recomputed by this phase.

## What was evaluated

- Models: E0 (five-jet kinematics only), E1 (E0 + continuous SophonAK4
  `probB/C/L`), E2 (E1 + deterministic pairwise `[m_ij, |deta_ij|,
  |dphi_ij|, dR_ij]` as an additive pre-softmax attention bias).
- Seeds: 0, 1, 2 for every arm (9 checkpoints total, all trained once, in
  Phase-4L, never retrained since).
- Partitions: `internal_val` (development, Phase-4L) and `holdout_A`
  (one-time, Phase-4M). `holdout_B`, `inference_qcd_1`, `inference_qcd_2`
  were not used for any number below.

## R_B (= 1/eps_B) at fixed signal efficiency, seed mean +/- std

### Development (`internal_val`, Phase-4L)

| eps_S | E0 | E1 | E2 |
|---:|---:|---:|---:|
| 0.40 | 283.09 +/- 27.19 | 370.61 +/- 26.80 | 216.31 +/- 17.96 |
| 0.10 | 3133.8 +/- 0.0 (exploratory) | 6615.7 +/- 4207.2 (exploratory) | 7660.3 +/- 3550.9 (exploratory) |

Full 6-point grid (0.60/0.50/0.40/0.25/0.20/0.10) is in
`phase4L_event_transformer_development_training_20260813_v1/DEVELOPMENT_WORKING_POINTS_FIXED_EPSB.tsv`
and `DEVELOPMENT_SEED_STABILITY.tsv`; not re-transcribed row-by-row here to
avoid a second manual-transcription error site (see the erratum note
below).

### Holdout_A (one-time, Phase-4M)

| eps_S | E0 | E1 | E2 |
|---:|---:|---:|---:|
| 0.60 | 105.65 +/- 4.09 | 130.19 +/- 7.36 | 88.88 +/- 8.01 |
| 0.50 | 173.74 +/- 9.36 | 237.58 +/- 5.11 | 144.42 +/- 12.26 |
| 0.40 | 285.66 +/- 43.88 | 439.52 +/- 38.39 | 244.59 +/- 19.30 |
| 0.25 | 736.37 +/- 54.39 | 1193.18 +/- 202.93 | 759.82 +/- 124.63 |
| 0.20 | 1094.20 +/- 133.80 | 1740.96 +/- 240.23 | 1370.08 +/- 501.83 |
| 0.10 (exploratory, raw QCD<10) | 4441.88 +/- 483.21 | 7972.61 +/- 1610.71 | 10250.50 +/- 7248.20 |

Source: `phase4M_holdout_a_adjudication_20260813_v1/HOLDOUT_A_ADJUDICATION_RESULT.md`,
`HOLDOUT_A_WORKING_POINTS_FIXED_EPSB.tsv`, `HOLDOUT_A_SEED_STABILITY.tsv`.

## AUC (secondary), mean of 3 seeds

| Sample | E0 | E1 | E2 |
|---|---:|---:|---:|
| Development (`internal_val`) | 0.9723 | 0.9764 | 0.9740 |
| Holdout_A | 0.9730 (rounded from 0.97295) | 0.9769 | 0.9739 |

## E0 -> E1: effect of continuous Sophon flavor information

**Consistent, statistically supported improvement, reproduced on
holdout_A.** Development: R_B(E1)/R_B(E0) - 1 ranges from **+22%**
(`eps_S=0.60`) to **+31%** (`eps_S=0.40`, corrected -- see erratum below)
to **+61%** (`eps_S=0.25`). All 3 seeds agree in direction at every
working point with `eps_S>=0.25`. On `holdout_A`: fully reproduces, zero
seed-level sign flips at any in-region working point (`eps_S=0.60` down to
`0.20`), with a similar magnitude (+23% to +62%, per
`HOLDOUT_A_ADJUDICATION_RESULT.md`).

**Erratum note:** the development-stage `eps_S=0.40` figure was originally
mis-transcribed as "+33%" in
`phase4L.../MECHANISM_ABLATION_RESULT.md`. Independently re-derived twice
(once in the original document's own numbers, once by Phase-4O reading
raw per-seed JSONs directly), the correct figure is **+31%** (+30.9%
exactly). This is a prose/arithmetic correction only -- no underlying
`R_B` value, checkpoint, or conclusion changed. See
`phase4O_trackb_publication_transfer_readiness_20260814_v1/PHASE4L_MECHANISM_ABLATION_ERRATUM.md`.
The `+22%`/`+61%` endpoints were independently confirmed correct in both
passes.

## E1 -> E2: additional effect of pairwise attention bias

**Consistent degradation on development, not an improvement.** At every
working point with `eps_S>=0.25`, all 3 seeds show `R_B(E2) < R_B(E1)`, no
exceptions: -31% (`eps_S=0.60`) to -42% (`eps_S=0.40`) to -38%
(`eps_S=0.25`) (all three figures independently re-confirmed correct by
Phase-4O, no erratum needed). This is reported as a genuine mechanism
result, not a bug: E2 in fact has the *smallest* train/val loss gap of the
three arms (no overfitting pathology), and the project's own governing
constraints ("no per-arm retuning after seeing performance") explicitly
forbid patching E2 to look better.

**On holdout_A: partially reproduces.** At `eps_S in {0.60, 0.50, 0.40,
0.25}`, all 3 seeds still agree with the development finding
(`E1 > E2`), exactly matching `internal_val`. At `eps_S=0.20`, **seed 0
flips**: `R_B(E1)=1464.4 < R_B(E2)=2050.1`. This is the single instance of
the mean-favors-E1-but-one-seed-disagrees pattern in the whole study, and
it lands exactly at E2's raw-surviving-QCD count of 10 -- the boundary of
the "statistically limited but reportable" support tier. Seeds 1 and 2 at
the same working point still favor E1. Per the pre-declared "all 3 seeds
must agree at an in-region point to call it full reproduction" rule, this
is recorded as **partial reproduction**, not full reproduction and not
non-reproduction -- and it was reported as found, not adjusted, excluded,
or investigated further with additional sealed data.

## Statistically supported rejection range

Using the development `eps_B` grid
(`DEVELOPMENT_WORKING_POINTS_FIXED_EPSB.tsv`): `eps_B=1e-2` (`R_B=100`) is
the deepest point that is **well-supported** (raw QCD >= 100) for every
arm and every seed. `eps_B=1e-3` (`R_B=1000`) remains
**statistically limited but reportable** (raw QCD = 13 for every arm/seed
on the 12,535-event development QCD sample). `eps_B=1e-4` is
**exploratory only** (raw QCD = 1) and is not a primary claim for any arm.
On `holdout_A`, the E1>E0 finding is well-supported through `eps_S=0.20`
(raw QCD counts documented directly in `HOLDOUT_A_WORKING_POINTS.tsv`);
`eps_S=0.10` on `holdout_A` is explicitly marked exploratory (raw QCD<10)
in the source document and is not treated as a primary quantitative claim
there either.

## Bottom line (as already adjudicated, not re-derived here)

`phase4M_holdout_a_adjudication_20260813_v1/HOLDOUT_A_ADJUDICATION_RESULT.md`'s
own summary table:

| Comparison | Development (4L) | Holdout_A (4M) | Reproduced? |
|---|---|---|---|
| E1 > E0 | consistent, all seeds, `eps_S>=0.25` | consistent, all seeds, `eps_S>=0.20` | **Yes, fully** |
| E1 > E2 | consistent, all seeds, `eps_S>=0.25` | consistent for `eps_S>=0.25`; 1 seed flips at `eps_S=0.20` | **Partially** -- one discrepancy reported |

`holdout_A` is a spent resource for E0/E1/E2 arm adjudication; it may not
be reopened or reused to further tune, re-rank, or re-select among these
three arms.
