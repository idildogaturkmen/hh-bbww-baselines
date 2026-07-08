# HH4b Delphes analysis-level data release plan

This repository does not store large raw Delphes ROOT, HepMC, or LHE files.

The reproducible release consists of:
- MG5 process cards
- Delphes/Pythia pipeline scripts
- campaign manifests with seeds
- summary tables
- compact merged event-summary parquet files
- compact merged HH4b-candidate parquet files
- plots and validation summaries

The full raw simulation can be regenerated from the cards, seeds, and scripts. For selected validation samples, one Delphes ROOT shard may be kept separately as a validation artifact.
