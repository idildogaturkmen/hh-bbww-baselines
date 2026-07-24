# HH4b 3b ROOT source map

## Purpose

This checkpoint freezes one Delphes ROOT archive locator for every member of
the canonical HH4b train/validation development membership. It is a
provenance and source-access artifact; it does not reconstruct three-b-tag
candidates or inspect ROOT event data.

## Result

- Status: `hh4b_3b_root_source_map_pass`
- Development members: 581
- Resolved members: 581
- Train / validation members: 458 / 123
- Signal / background members: 107 / 474
- Zero-row background members: 220
- Unique remote bundles checked: 581
- Failed remote metadata checks: 0
- Cross-split locator groups: 0

The background mapping uses exact destination paths before checksum fallback.
VBF sources use the canonical signal registry and lower materialized registry.
ggF sources use only the three candidate provenance columns and locate ROOT
files inside the nested reconstruction archive carried by each outer bundle.

## Safety record

- Candidate physics columns read: 0
- Test members considered: 0
- ROOT files extracted: 0
- ROOT files opened: 0
- ROOT event arrays read: 0
- Event generation performed: 0
- Physics yields calculated: 0

## Reproducibility

- Source commit: `a231664d59c5c5a84e1f7aea08c7a7906e9a0321`
- Development manifest SHA-256: `b3e42a0af56594445e297e2347ccfd3bd799fd107a578af844c9733bfd2ebe6a`
- Configuration SHA-256: `19f72133c6d6082cdc742f926c79dad4cff24fb3dd3f1900f44bbee21c339595`

## Next gate

`audit_required_delphes_branches_for_3b_reconstruction`
