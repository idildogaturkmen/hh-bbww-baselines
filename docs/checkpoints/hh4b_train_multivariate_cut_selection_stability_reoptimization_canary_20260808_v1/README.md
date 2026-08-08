# Selection-stability full-reoptimization canary freeze

This checkpoint freezes the successful implementation canary for source-group
bootstrap reoptimization.

The canary exercised two frozen stability replicas, both b-tag categories, all
three mass parameterizations, multiple outer folds, six nested-OOF structure
jobs, and an identical deterministic rerun.

Bootstrap source multiplicity was represented by literal source-row copies.
Per-event weights were not scaled by multiplicity. This preserves source
bootstrap sumw, sumw2, raw-row, and effective-count semantics while allowing the
existing frozen worker and optimizer to run without changing their statistical
implementation. Source-fold assignments were not reassigned.

The canary used the predeclared bounded implementation budget 7/16/4/2.
Therefore its measured runtime is implementation evidence only and is not a
valid estimate of the full 63/128/16/8 production runtime.

Scientific feasibility/support values from the canary are not physics results
and are not infrastructure acceptance gates. Validation and test remained
sealed.

Production submission is NOT authorized by this checkpoint. The next gate is a
small bootstrap reoptimization pilot using the full production search budget,
followed by a production design/runtime freeze before replicas 0--199 may be
submitted.
