# HH4b Delphes ROOT recovery status

As of 2026-07-10, the nominal HH4b Delphes dataset uses compact parquet files for the original v1 BDT baseline, but full ROOT files are required for v2 reconstruction and truth/reconstruction follow-up studies.

The original storage-safe campaigns intentionally deleted most large intermediate files:
- Delphes ROOT was kept only for shard000 in several multi-shard campaigns.
- HepMC and LHE files were deleted after parquet production.
- QCD iHT slice scans explicitly deleted ROOT and HepMC after creating parquet outputs.

Recovered/available ROOT inputs:
- ggF HH4b: complete.
- VBF HH4b: complete.
- ttbar_200k: recovered to complete ROOT availability.
- QCD iHT nominal samples: recovered to complete ROOT availability.
- Zbbbb_100k: recovery in progress.

Going forward, Delphes ROOT files should be kept by default. ROOT files should not be deleted unless cleanup is explicitly requested.
