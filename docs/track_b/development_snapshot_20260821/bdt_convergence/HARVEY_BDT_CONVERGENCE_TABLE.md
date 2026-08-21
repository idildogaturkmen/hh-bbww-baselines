# Harvey BDT Convergence Table

Source: step2j interim-10.47M-event per-round training histories (seed 0), **hash-verified** against `step2k/evidence/STEP2J_EVIDENCE_FREEZE.json` before any number below was extracted. Read-only diagnostic - no retraining, no Condor submission, no inference/test data.

## Task A — early-stopping contract (verified from actual frozen code, not memory)

- **Monitored dataset**: `val`
- **Monitored metric**: `logloss`
- **Direction**: minimize (maximize=False)
- **Patience**: 50 rounds
- **save_best**: True
- Source: convergence_gate_step2f.py lines 112-117 (EARLY_STOPPING_ROUNDS=50, EARLY_STOPPING_METRIC='logloss', EARLY_STOPPING_DATA='val', EARLY_STOPPING_MAXIMIZE=False); j_train_convergence.py lines 128-130 (xgb.callback.EarlyStopping(rounds=orig.EARLY_STOPPING_ROUNDS, metric_name=orig.EARLY_STOPPING_METRIC, data_name=orig.EARLY_STOPPING_DATA, maximize=orig.EARLY_STOPPING_MAXIMIZE, save_best=True))

| Arm | Fired? | best_iteration (0-idx) | best round (1-idx) | matches history argmin | longest gap w/o improvement (whole run) | that gap's round | gap remaining at round 6000 |
|---|---|---|---|---|---|---|---|
| K | False | 5998 | 5999 | True | 8 | 5120 | 1 |
| KF | False | 5999 | 6000 | True | 13 | 5578 | 0 |

**Why the 50-round rule never fired despite visual flatness**: neither arm ever went more than 13 consecutive rounds without a new (even if tiny) validation-logloss minimum — K's longest gap was 8 rounds (near round 5120), KF's was 13 (near round 5578), both far short of the 50-round patience. Tiny improvements kept resetting the patience counter throughout the run, all the way to the very end.

## Task B — exact round-budget table

("after N rounds" = history row index N-1, verified explicitly, not array index N)

| Arm | Round | Train logloss | Val logloss | Train AUC | Val AUC |
|---|---|---|---|---|---|
| K | 200 | 0.528796 | 0.528945 | 0.812015 | 0.811872 |
| K | 1000 | 0.503388 | 0.505537 | 0.829006 | 0.827332 |
| K | 4000 | 0.487141 | 0.496148 | 0.841820 | 0.834710 |
| K | 6000 | 0.481711 | 0.494990 | 0.846089 | 0.835623 |
| KF | 200 | 0.287987 | 0.288252 | 0.949696 | 0.949626 |
| KF | 1000 | 0.235969 | 0.238045 | 0.965592 | 0.964926 |
| KF | 4000 | 0.218000 | 0.226978 | 0.970518 | 0.967877 |
| KF | 6000 | 0.212409 | 0.225868 | 0.972089 | 0.968185 |

### Marginal changes (validation set)

| Arm | Interval | Δ val logloss | % Δ val logloss | Δ val AUC | % Δ val AUC |
|---|---|---|---|---|---|
| K | 200->1000 | -0.023407 | -4.425% | 0.015460 | +1.904% |
| K | 1000->4000 | -0.009390 | -1.857% | 0.007378 | +0.892% |
| K | 4000->6000 | -0.001157 | -0.233% | 0.000912 | +0.109% |
| KF | 200->1000 | -0.050207 | -17.418% | 0.015300 | +1.611% |
| KF | 1000->4000 | -0.011067 | -4.649% | 0.002951 | +0.306% |
| KF | 4000->6000 | -0.001110 | -0.489% | 0.000308 | +0.032% |

### Late-training detail and cumulative bests

| Arm | val logloss @5950 | val logloss @6000 | Δ 5950→6000 |
|---|---|---|---|
| K | 0.495005 | 0.494990 | -0.000014 |
| KF | 0.225889 | 0.225868 | -0.000021 |

| Arm | Best through 200 | round | Best through 1000 | round | Best through 4000 | round | Best through 6000 | round |
|---|---|---|---|---|---|---|---|---|
| K | 0.528945 | 200 | 0.505537 | 1000 | 0.496148 | 4000 | 0.494990 | 5999 |
| KF | 0.288252 | 200 | 0.238045 | 1000 | 0.226978 | 3999 | 0.225868 | 6000 |

## Task D — compute-budget scaling (approximate, linear in round count)

*APPROXIMATE LINEAR scaling of training wall-time only, proportional to round count relative to the measured/projected 6000-round figure from step2k's RUNTIME_PROJECTION.md. Excludes fixed DMatrix construction/shard-staging overhead (~34 min combined per arm at full 144M scale, small relative to the multi-day training times below, but NOT zero).*

| Arm | 1000 rounds | 2000 rounds | 3000 rounds | 4000 rounds | 6000 rounds (reference) |
|---|---|---|---|---|---|
| K | 13.1h | 26.2h | 39.3h | 52.4h | 78.6h |
| KF | 12.08h | 24.17h | 36.25h | 48.33h | 72.5h |

**4000→6000 cost increase: +50% boosting rounds (2000 more rounds).**

| Arm | Val logloss gain (abs) | Val logloss gain (%) | Val AUC gain (abs) |
|---|---|---|---|
| K | -0.001157 | -0.233% | 0.000912 |
| KF | -0.001110 | -0.489% | 0.000308 |
