# HH4b 3b Delphes branch audit

## Purpose

This checkpoint establishes that the Delphes ROOT inputs needed for a future
three-b-tag control-candidate builder expose the six branches used by the
frozen four-b-tag builder. It is a representative ROOT metadata audit only.

## Result

- Status: `hh4b_3b_delphes_branch_audit_pass`
- Development members assigned to strata: 581
- Primary schema strata: 52
- Representatives audited: 54
- Representatives failed: 0
- Required branch checks: 324
- Missing checks: 0
- Incompatible checks: 0
- Entry-count mismatches: 0
- Temporary files remaining: 0

Representatives were selected deterministically from the complete committed
source map. Outer bundles were processed sequentially. Only the requested ROOT
member, or the requested nested archive and ROOT member for ggF, was extracted.
Temporary materializations were deleted after each metadata audit.

## Safety record

- ROOT event arrays read: 0
- Candidate files read: 0
- Test members considered: 0
- Reconstruction performed: 0
- Physics yields calculated: 0
- Static array-read safety review: `pass`

## Reproducibility

- Source commit: `9e32ae1b3159e30953fa13dac5a25fecf8edf654`
- Source-map SHA-256: `e797ab8cc878ea1fbb84c05cd52fb8919c6e67bb56bab07bef46701615a29f60`
- Configuration SHA-256: `e6bb735bd6d975056448de20e06770460df1390726931dda460e9431b0b0f778`

## Next gate

`implement_hh4b_3b_control_candidate_builder_with_4b_parity_mode`
