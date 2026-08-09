# Initial-200 selection-stability aggregation

This checkpoint freezes the predeclared train-only aggregation of selection-
stability replicas 0--199. Its sole scientific input is the normalized 54,000-
row production inventory from the preceding complete-return audit checkpoint.
Transfer-pilot results are excluded.

The implementation selected exactly one structure for every replica, category,
and outer fold using pooled true inner-OOF metrics only. Feasible candidates
were ordered by minimum background efficiency, minimum signal-efficiency
overshoot above the frozen target, fewer cuts, and lexicographic structure ID.
When all candidates were infeasible, the implementation used maximum signal
efficiency, minimum background efficiency, fewer cuts, and lexicographic
structure ID. Outer-fold physics evaluation was never used for selection.

The audit covers all 54,000 ranking rows, 2,000 fold winners, and 400
replica/category reductions. A second complete run reproduced every original
output byte-for-byte.

## Initial-200 result

| Category | Modal replica-selected structure | Frequency | Second frequency | Gap | Nominal recovery | Non-unique five-fold modes | Infeasible fold winners |
|---|---|---:|---:|---:|---:|---:|---:|
| `exact3tag` | `asymmetric_rectangular_mass__category__plus_abs_h_delta_eta` | 0.185 | 0.145 | 0.040 | 0.120 | 79/200 | 538/1000 |
| `ge4tag` | `asymmetric_rectangular_mass__category__mass_only` | 0.260 | 0.130 | 0.130 | 0.025 | 72/200 | 156/1000 |

Replica-level `ht_candidate_jets` inclusion lies in the predeclared ambiguous
interval in both categories: 0.360 for `exact3tag` and 0.335 for `ge4tag`.
Every fold winner passes support. Infeasible fold winners are retained as valid
scientific outcomes and are not execution failures.

These are stability diagnostics only. They do not change the already-frozen
nominal deployment candidate. The initial-200 result cannot cancel the
escalation to the predeclared 1,000 replicas, which was triggered before this
production campaign by the nominal five-fold result.

## Frozen boundaries

- Validation payloads opened: **0**.
- Test payloads opened: **0**.
- Pilot results used: **no**.
- Nominal deployment candidate changed: **no**.
- Replicas 200--999 authorized by this checkpoint: **no**.
- Clusters `3755882`, `3768139`, and `30002685` must never be resubmitted.

The next state transition is a separate explicit authorization checkpoint for
replicas 200--999. This checkpoint itself performs no authorization or Condor
submission.

`evidence/aggregation/` contains the complete ranking ledger, winner and
replica tables, diagnostics, raw conditional thresholds, plot-data sidecars,
summary, manifests, independent audit, and checksums. `implementation/`
contains byte-identical copies of the aggregator, auditor, and protocol tests.
`initial200_aggregation_freeze.json` is the machine-readable freeze contract.
