# Initial-200 selection-stability complete-return audit

This checkpoint freezes the completed return audit for the initial 200
selection-stability production replicas before any scientific aggregation.

## Frozen outcome

- Production cluster `30002685` on the authoritative schedd
  `lpcschedd5.fnal.gov` completed cleanly: **2000/2000 jobs** (procedures
  0--1999).
- The return audit passed: **54000/54000 structure results**, with the exact
  frozen 27-structure set in every job, verified internal and canonical-payload
  SHA-256 identities, complete inner-crossfit fold coverage, matching execution
  provenance, and `outer_fold_used_for_selection=false` everywhere.
- The final queue snapshot has zero rows. The final history snapshot has exactly
  2000 unique clean completions with exit code zero and no signal exits.
- Validation payloads opened: **0**. Test payloads opened: **0**.

The normalized `structure_result_inventory.tsv` is committed directly. It is
40,920,127 bytes, below GitHub's 100,000,000-byte per-file limit, and is the
deterministic production input for the later predeclared aggregation.

## Scientific interpretation and frozen boundary

The pooled inner-OOF diagnostics contain 6,561 feasible and 47,439 infeasible
structure results. All 54,000 structure results pass pooled support, with zero
pooled support failures. Infeasible structures are valid scientific outcomes,
not failed executions; support gates remain separate from feasibility.

At this checkpoint:

- no initial-200 aggregation has occurred;
- no fold-level winners have been selected;
- no replica-level stability statistics have been computed;
- transfer-qualification pilot results remain excluded from aggregation;
- the nominal deployment candidate is unchanged; and
- replicas 200--999 remain not production-authorized.

Cluster `30002685` must never be resubmitted. The original submission receipt is
preserved byte-for-byte in its earlier checkpoint, including its known incorrect
schedd field; the later frozen provenance correction establishes
`lpcschedd5.fnal.gov` as authoritative.

The next step is the already-predeclared initial-200 aggregation protocol in
`../hh4b_train_multivariate_cut_selection_stability_aggregation_protocol_20260808_v1/`.
That later task must use the frozen production inventory, exclude pilot results,
and keep validation and test sealed.

## Provenance

- Documentation/provenance parent HEAD:
  `269449d2b0ac04ba1e4caee3b2a838e1e5ee21ef`
- Scientific execution/package HEAD embedded in returned results:
  `b317d2c3270166edce86c70a3972e6db3b75c186`
- Aggregation protocol SHA-256:
  `a856d9027cc1d2e11a9d60e72b4104b5b8c24ce61acf43def0d4d487ab867eea`
- Frozen worker SHA-256:
  `739147c33d6597018f4dce8c483278c64ff8b41533de1f1b213749dfb39332ed`
- Frozen optimizer SHA-256:
  `7760ee7ae309bec6cd47eb1c73ed04ec12bb805f133527498fcb75aca8088a62`

`complete_return_audit_freeze.json` records the machine-readable contract.
`SHA256SUMS` covers every other file in this checkpoint. The copied audit and
scheduler evidence retain their original bytes.
