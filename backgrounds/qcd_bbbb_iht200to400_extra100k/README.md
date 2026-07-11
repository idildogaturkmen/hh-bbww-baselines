# QCD bbbb iHT200-400 extra 100k

Production date: 2026-07-11

Purpose: increase MC statistics for the dominant QCD bbbb iHT200-400 residual background in the categorized BDT-v2 HH→4b study.

## Sample definition

- Process: QCD bbbb
- Phase space: 200 <= iHT < 400 GeV
- Generated events: 100,000
- Shards: 10
- Events per shard: 10,000
- Candidate rows: 6,865
- Median generated cross section: about 126.32 pb

## Important weighting note

When combining this sample with the existing iHT200-400 QCD sample, do not double-count the iHT200-400 slice cross section.

Use the iHT200-400 cross section once, with the total number of generated events in the combined iHT200-400 slice.

## Available artifacts

The derived parquet outputs are available for all 10 shards and include:
- event summaries
- HH4b candidate rows
- merged event summary
- merged HH4b candidates
- v2 candidate parquets

Only the ROOT files currently listed in `root_file_manifest.txt` are retained in the storage area. At the time of this manifest, only one ROOT file was found for this extra sample, while the full 100k-event parquet-level outputs are present.

## Files in this folder

- `metadata_summary.txt`
- `root_file_manifest.txt`
- `parquet_file_manifest.txt`
- `parquet_v2_file_manifest.txt`
- `metadata_file_manifest.txt`
- `log_file_manifest.txt`
- `production_script.sh`, if available
