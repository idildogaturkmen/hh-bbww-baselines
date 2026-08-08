# Multivariate-cut selection-stability source-group draw registry

This checkpoint freezes the resampling registry used for the independent
selection-reoptimization stability diagnostic.

The stability diagnostic is distinct from the 2000-replica paired metric
bootstrap. The paired metric bootstrap retains its historical seed 20260727
and frozen draw parquet. The selection-stability diagnostic instead uses the
predeclared seed 20260806.

A total of 1000 stability replicas are frozen before any stability
reoptimization is executed. Replicas 0--199 form the initial 200-replica
diagnostic, while replicas 200--999 are the already-predeclared escalation
block.

The source universe contains the same 441 physical source groups and 23
(sample_class, process_or_mode) strata as the frozen historical source-group
bootstrap. Resampling is nonparametric with replacement within each stratum.
Source-fold assignments are never resampled or reassigned; bootstrap
multiplicities are joined to the already-frozen fold-table rows using group_id.

Ninety source groups have zero rows in the exact3tag/ge4tag endpoint tables.
They remain in the whole-source resampling universe and naturally contribute
no category rows when drawn.

This checkpoint does not run the optimizer, change the nominal deployment
candidate, open validation, or open test.
