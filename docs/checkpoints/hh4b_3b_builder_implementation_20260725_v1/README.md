# HH4b exactly-3b control builder implementation

## Purpose

This checkpoint records the implementation and synthetic-only unit validation
of an HH4b exactly-three-tag control-candidate builder. The new builder has
explicit `threeb-control` and `fourb-parity` modes and reads only the six
Delphes jet branches authorized by the prerequisite metadata audit.

## Result

- Status: `hh4b_3b_builder_implementation_pass`
- Unit tests collected: 23
- Unit tests passed: 23
- Synthetic parity mismatches: 0
- Four-b parity columns: 72
- Three-b control columns: 83
- Real development, validation, and test ROOT files opened: 0
- Physics yields, significances, normalizations, and transfer factors: 0
- Temporary test files remaining: 0

The synthetic parity fixture covers events with fewer than four, exactly four,
more than four, and more than eight tagged jets; competing four-jet
combinations; pairing ties; and a zero-candidate fixture. It is not evidence of
parity on real Delphes samples.

## Frozen references

- Builder:
  `scripts/delphes/reconstruct_hh4b_candidates_v2.py`
  (`4d7eb8e3400ec50c1c254744e6f3a2b11bd033a4525c13ec22e85247ecf59c57`)
- Policy:
  `configs/production/hh4b_rich_v2_reconstruction_policy_v1.yaml`
  (`4b2a951dba7ac8bd0e2e982e3447e998e07b86d63390c248600976b3f4912b68`)

Both references remained unchanged.

## Authorization

The implementation is unit tested but is not authorized for production
reconstruction, validation reconstruction, test access, physics yields,
normalization, or transfer-factor calculation.

## Next gate

`prove_hh4b_4b_parity_on_real_delphes_canaries`
