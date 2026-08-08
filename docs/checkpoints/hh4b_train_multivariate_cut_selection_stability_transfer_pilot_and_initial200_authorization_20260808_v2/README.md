# Selection-stability category-fold transfer pilot and initial-200 authorization

The bounded category-fold transfer pilot completed successfully on Condor
cluster 3768139. Four category-fold jobs returned 108 of 108 expected
full-budget nested-OOF structure results. All four Condor jobs completed with
zero exit codes and no signal termination. Returned structure checksums,
execution provenance, exact 27-structure membership, inner-fold coverage, and
outer-fold isolation all passed. Validation and test remained sealed.

The initial v1 audit stopped only because the history parser assumed numeric
values for `ExitBySignal` and `RemoteWallClock`; the LPC history output used the
valid representations `false` and `undefined`. The v2 audit repaired only this
audit-side parsing issue and then passed the complete returned-artifact audit.
No Condor work was resubmitted.

All 108 pilot structures passed pooled support. Three were feasible at the
predeclared target signal efficiency and 105 were infeasible. Feasibility is a
scientific per-bootstrap-structure outcome and was not used as an execution or
infrastructure acceptance gate.

This checkpoint authorizes only frozen selection-stability replicas 0--199:
200 bootstrap replicas, two categories, five outer folds, 27 structures per
category-fold job, for 2000 Condor jobs and 54000 structure evaluations at the
unchanged 63/128/16/8 search budget. The selected architecture is one bootstrap
replica, one category, one outer fold, and all 27 structures executed
sequentially.

Replicas 200--999 are not authorized by this checkpoint. The nominal deployment
candidate is not changed by the stability bootstrap. Validation and test remain
sealed. A complete returned-output audit is required before aggregation of the
initial 200 replicas.
