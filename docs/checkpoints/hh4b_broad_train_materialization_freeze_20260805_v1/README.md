# Complete broad-train materialization freeze

**Checkpoint date:** 2026-08-05 UTC
**Branch:** `delphes-hh4b-production`
**Implementation head before this evidence commit:** `306d6fe33be54add161e72adba7bc9bcc5298e90`
**Audited worker head:** `1d552cd51b8285f7df277ab77fd7e966f25ce3db`

## Closed production gate

The complete broad-ML training partition was materialized and audited:

- 464 / 464 sources;
- 3,799,873 / 3,799,873 events;
- 441 primary physical sources containing 3,569,873 events;
- 23 auxiliary `qcd_bbbb` sources containing 230,000 events;
- 1,171,072 broad-event-eligible rows;
- 30,397 assignment-matchable rows;
- 3,799,873 globally unique event identifiers;
- 459 / 459 scale-out Condor jobs completed with exit code zero;
- five portable pilot products were reused after byte-identity validation.

## Access and generator-weight transport closure

Access modes:

- 442 remote bundle sources;
- 22 local-direct-ROOT sources staged through the portable protocol.

Generator-weight transport classes:

- 241 `uniform_member_constant`;
- 14 `signed_exact_event_sidecar`;
- 59 `qcd_uniform_constant`;
- 127 `qcd_variable_frozen_fragment`;
- 23 `not_applicable_auxiliary_qcd`.

The 23 auxiliary-QCD sources remain classification-only. They do not receive
physical generator weights and are not authorized for physical-yield
evaluation.

## Sealing state

- physical luminosity-weight application authorized: **false**;
- validation payloads opened: **0**;
- test payloads opened: **0**;
- models trained during materialization: **0**.

## Frozen evidence

- `full_train_materialization_inventory.tsv` records the 464 immutable
  Parquet/receipt products, byte sizes, SHA-256 digests, event counts, access
  modes, transport classes, and supervision counts.
- `full_train_materialization_contract.json` records the global closure and
  production contract.
- `audit_evidence/` preserves the full Condor history, source audit, global
  summary, audit log, and original audit checksums.

## Next gate

Build train-only common baseline tables from `broad_event_eligible` rows while
preserving source-group identity, immutable population roles, generator-weight
provenance, and the separation between classification supervision and
signal-only assignment supervision. Validation and test remain sealed.
