# Selection-stability replica aggregation protocol

This checkpoint freezes the replica-level reduction rule before any initial-200
selection-stability production results are generated.

For each bootstrap replica and b-tag category, the five outer-fold structure
winners are obtained using the already-frozen nested inner-OOF selection rule.
The structure selected most frequently across the five folds is the replica-level
representative structure. If the maximum count is tied, the tied structures
already have equal inner-OOF-derived selection frequency, so the frozen
deployment-candidate fallback is completed by lexicographic `structure_id`.

Replica-level thresholds are the coordinate-wise medians of the fold-specific
development-refit thresholds among only those folds selecting the replica-level
representative structure. These thresholds are stability diagnostics and never
retune or replace the frozen nominal deployment candidate.

The transfer-qualification pilot results do not enter the stability aggregation.
Replicas 0--199 enter the initial stability aggregation. The escalation to the
pre-frozen replicas 200--999 was already triggered by the nominal five-fold
stability result (modal frequency 0.40 in both categories), so observations from
the first 200 replicas cannot cancel that escalation.

Validation and test remain sealed.
