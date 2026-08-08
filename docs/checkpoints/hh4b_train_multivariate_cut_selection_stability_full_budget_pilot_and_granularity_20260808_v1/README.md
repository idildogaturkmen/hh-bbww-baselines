# Selection-stability full-production-budget pilot and granularity freeze

This checkpoint freezes the bounded source-bootstrap reoptimization pilot run
with the full production optimizer budget 63/128/16/8.

The pilot exercised both b-tag categories, the two frozen nominal families, and
1-, 2-, 3-, and 4-continuous-cut structures. An identical full-budget rerun
reproduced both the canonical structure payload and the inner-crossfit output.
The pilot is runtime/resource/production-path evidence only; its physics values
must not enter the 200- or 1000-replica stability aggregation.

Production granularity is selected using computational evidence only. For each
cut dimensionality, the maximum measured full-budget pilot runtime is combined
with the exact dimensionality composition of the 27 frozen structures in one
category. The selected unit is one bootstrap replica, one category, one outer
fold, and all 27 structures executed sequentially.

This gives 2000 Condor jobs for the initial 200 replicas and 10000 jobs for the
full predeclared 1000 replicas, while preserving exactly the same 270 structure
evaluations per replica. No scientific selection rule, source resampling rule,
outer-fold isolation rule, or nominal deployment candidate is changed.

This checkpoint does not authorize production submission. The next gate is to
build a transfer-safe category-fold runner/package and pass a bounded Condor
transfer pilot. Validation and test remain sealed.
