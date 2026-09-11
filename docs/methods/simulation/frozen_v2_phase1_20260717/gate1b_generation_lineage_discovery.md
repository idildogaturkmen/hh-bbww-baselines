# Gate 1B: generation-lineage discovery

Date: 2026-07-17

## Result

Status:

`GATE1B_LINEAGE_DISCOVERY_COMPLETE`

Discovered:

- 42 immutable MG5 run banners
- 25 Phase 1 manifest rows
- QCD-general run-banner evidence
- QCD IHT 400–600 run-banner evidence
- ttbar run-banner evidence
- Zbbbb run-banner evidence

## QCD evidence

The retained general QCD-bbbb campaign has run banners with:

- `ihtmin = 0`
- `ihtmax = -1`

The retained QCD-bbbb IHT 400–600 campaign has ten run banners
with:

- `ihtmin = 400`
- `ihtmax = 600`

The general and IHT 400–600 samples therefore overlap in physical
phase space. They must not be summed directly as independent physical
background components.

## Interpretation

This gate establishes that matching immutable run banners exist.

It does not yet establish exact one-to-one HepMC-to-banner lineage.
Family matching remains candidate evidence until the exact run name,
input SHA256, Phase 1 metadata SHA256, shard number, event count, seed,
and generation cuts are checked together.

The broad evidence-file search produced 1,211 matches because internal
MG5 subprocess logs repeat common search terms. That table is
diagnostic only and is intentionally not tracked as canonical metadata.

## Next gate

Gate 1C: exact one-to-one lineage resolution for all 25 Phase 1 HepMC
files.

No new production is authorized by Gate 1B.
