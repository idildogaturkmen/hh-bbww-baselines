# QCD bbbb iHT200-400 extra 100k v3

Production date: 2026-07-11

Purpose: BDT-v2-ready targeted extra statistics for the dominant QCD bbbb iHT200-400 component in the highest-score HH→4b BDT tail.

## Sample definition

- Process: QCD bbbb
- Phase space: 200 <= iHT < 400 GeV
- Generated events: 100,000
- Shards: 10
- Events per shard: 10,000
- Candidate rows: 6,842
- Events with >=4 selected b-tagged jets: 6,842
- Median generated cross section: about 126.40 pb
- Seed base: 652000

## BDT-v2 integration status

This sample was produced using the v2 candidate reconstruction path, so it is intended to be usable in the BDT-v2 / tail-audit pipeline.

For BDT-v2 studies, combine this sample with the existing v2-ready iHT200-400 combined120k sample.

Do not include `qcd_bbbb_iht200to400_extra100k_v2` in the BDT-v2 merge unless its v2 features are reconstructed, because that sample has the older compact candidate schema.

## Weighting note

When combining iHT200-400 QCD samples, use the iHT200-400 slice cross section once and divide by the total generated events in the combined v2-ready iHT200-400 sample.

For the current BDT-v2-ready combination:

- existing iHT200-400 combined120k: 120,000 generated events
- extra100k_v3: 100,000 generated events
- total BDT-v2-ready iHT200-400 generated events: 220,000
