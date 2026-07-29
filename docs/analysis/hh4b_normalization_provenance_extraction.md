# HH4b normalization-provenance extraction

The full archive-header inventory is converted into a frozen extraction policy before any
production metadata content is read.

The policy permits only regular files that:

1. have a safe relative archive path;
2. are at most 5 MiB;
3. are not ROOT, Parquet, HepMC, LHE, ZIP, TAR, or nested archives;
4. have a provenance classification such as a generator banner, process card, run card,
   generator log, shower card, Delphes log, provenance JSON, or QCD importance record.

Exactly 547 files across all 53 EOS process-campaign bundles satisfy the frozen policy.
The operation is resumable and retains one temporary production bundle at a time. Each
bundle is verified against its previously frozen byte size and SHA-256 digest.

The extracted raw bytes are consolidated into a deterministic provenance TAR.GZ archive
with a complete per-file SHA-256 manifest. Event payloads, candidate Parquet files,
validation content, and final-evaluation content remain closed.

This gate does not parse or assign cross sections, branching fractions, filter
efficiencies, sums of generator weights, luminosities, physical yields, or thresholds.

The local legacy ttbar candidate campaign remains a separate blocking provenance-recovery
task before the normalization registry can be frozen.
