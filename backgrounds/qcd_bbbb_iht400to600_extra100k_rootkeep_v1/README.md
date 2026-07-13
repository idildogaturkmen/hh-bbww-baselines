# QCD bbbb iHT400-600 extra 100k root-kept v1

Purpose: targeted extra QCD statistics for the high-score HH→4b BDT tail, while preserving Delphes ROOT files for future SPA-Net or particle-level studies.

Sample:
- Process: QCD bbbb
- Phase space: 400 <= iHT < 600 GeV
- Generated events: 100,000
- Shards: 10
- Events per shard: 10,000
- ROOT retention: yes
- Intended use: BDT-v2, DNN-v2, future SPA-Net/particle-level studies

Weighting note:
When this sample is combined with the nominal iHT400-600 sample, use the iHT400-600 cross section once and divide by the total generated events in the combined iHT400-600 sample. Do not treat the nominal and extra samples as independent physics cross-section components.
