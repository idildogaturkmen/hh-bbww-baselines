# Common train-table HTCondor scale-out pilot freeze

This checkpoint freezes the exact single-source worker and evidence from the
passed seven-source production-style pilot.

## Passed pilot

- campaign:
  `hh4b_train_common_table_scaleout_pilot_v2_20260805T213031Z`;
- HTCondor cluster: `85042207`;
- submitted processes: exactly `0` through `6`;
- sources audited: 7 / 7;
- outer-fold coverage: 0, 1, 2, 3, 4;
- accounting rows: 61,000 / 61,000;
- resolved rows: 34,979 / 34,979;
- held jobs: 0;
- byte-identical internal worker reruns: 7 / 7.

The pilot demonstrated global event-UID uniqueness, resolved-table subset
closure, fold-local comparison-weight joins, historical reconstruction parity,
portable runtime execution, headerless queue construction, and transfer from
checksum-validated schedd-readable staged inputs.

## Frozen implementation

The exact passed worker is committed at:

`../../../../scripts/analysis/hh4b_train_common_table_single_source_worker_v1.py`

The checkpoint contract records its SHA-256 digest and requires it to be
byte-identical to the worker used in the passed pilot.

## Still sealed

- physical luminosity-weight application authorized: false;
- validation payloads opened: 0;
- test payloads opened: 0;
- models trained: 0.

## Next gate

Construct the full 464-source train common-table production manifest, validate
all frozen joins, reuse the seven byte-audited pilot products, and submit only
the remaining 457 sources. Full physical-weight authorization still requires
the subsequent 464-source global closure audit.
