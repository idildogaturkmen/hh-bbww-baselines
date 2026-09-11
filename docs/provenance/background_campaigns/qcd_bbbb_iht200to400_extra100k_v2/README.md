# QCD bbbb iHT200-400 extra 100k v2

Production date: 2026-07-11

Purpose: targeted extra statistics for the dominant QCD bbbb iHT200-400 component in the highest-score HH→4b BDT tail.

## Sample definition

- Process: QCD bbbb
- Phase space: 200 <= iHT < 400 GeV
- Generated events: 100,000
- Shards: 10
- Events per shard: 10,000
- Candidate rows: 7,039
- Events with >=4 selected b-tagged jets: 7,039
- Median generated cross section: about 126.69 pb
- Seed base: 452000

## Important weighting note

When combining with the existing iHT200-400 QCD samples, do not double-count the iHT200-400 slice cross section.

The combined iHT200-400 sample should use the iHT200-400 cross section once, divided by the total number of generated events in the combined iHT200-400 slice.

Current intended combined count after including this sample:

- original iHT200-400 nominal: 20,000
- extra100k: 100,000
- extra100k_v2: 100,000
- total iHT200-400 generated events: 220,000

## Available artifacts

The parquet-level outputs are available for all 10 shards and include merged candidate outputs.

Only the ROOT files listed in `root_file_manifest.txt` are currently retained in storage.

## BDT-v2 integration status

The merged candidate parquet currently contains the older compact candidate schema with 15 columns. It is not directly ready for the current BDT-v2 / tail-audit pipeline because several v2 event-topology and activity features are missing.

This sample should not be merged into the BDT-v2 training set until v2 candidate features are reconstructed or the sample is regenerated with v2 feature production enabled.
