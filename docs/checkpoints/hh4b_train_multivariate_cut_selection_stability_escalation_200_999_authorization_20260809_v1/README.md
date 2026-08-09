# Selection-stability replicas 200--999 authorization

This checkpoint explicitly authorizes the already-pretriggered production
escalation from 200 to 1,000 selection-stability replicas. It authorizes exactly
replicas 200--999: 800 replicas, two categories, and five outer folds, for 8,000
Condor jobs and 216,000 structure evaluations.

The authorization is not a new decision based on the observed initial-200
frequencies. The escalation was triggered before the initial-200 campaign by
the frozen nominal five-fold stability result, and the predeclared aggregation
protocol states that replicas 0--199 cannot cancel it. The completed initial-
200 aggregation is nevertheless a required provenance gate and is frozen at
Git commit `18a4ff9b2723c0eae7a550ae5071c74001188a49`.

The scientific architecture remains
`replica × category × outer-fold × all 27 structures sequentially`. The search
budget is `63_128_16_8`, the signal-efficiency target is `0.585957`, and the
source-group bootstrap seed is `20260806`. Source-fold assignments, original
event weights, literal row duplication, bootstrap UID suffixes, and the full
441-source resampling universe remain unchanged.

The existing draw registry already freezes all 1,000 replicas. Its 800-replica
escalation block was rehashed at authorization time:

- path: `/uscms_data/d3/iturkmen/hh4b_delphes/baselines/multivariate_cut_selection_stability_draw_registry_v1_20260808/selection_stability_draw_counts_escalation_800.parquet`
- rows: 226,901
- SHA-256: `091c2a401cb6ab74d9219449c117fc99832737cc2951e812dc7765540615a279`

The draws must not be regenerated. Transfer-pilot results remain excluded from
all stability aggregation, and the nominal deployment candidate is unchanged.
Validation and test remain sealed.

This checkpoint performs no package build and no submission. Its next gate is
to build and audit the immutable transfer-safe escalation package, freeze the
pre-submission evidence, and only then submit exactly once. Known clusters
`3755882`, `3768139`, and `30002685` must never be resubmitted.
