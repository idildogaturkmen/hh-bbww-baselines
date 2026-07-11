# HH→4b Delphes background samples

This folder contains the lightweight metadata needed to reproduce the Delphes-level HH→4b background samples used in the analysis.

Large ROOT/Parquet files are **not committed to git**. They should be distributed separately through an external data location. This folder records the sample definitions, cross sections, number of generated events, file patterns, production scripts, and manifests.

## Background samples

The main sample table is:

- `background_samples.csv`

The main backgrounds are:

- `ttbar_200k`
- `Zbbbb_100k`
- `qcd_bbbb_iht100to200_20000`
- `qcd_bbbb_iht200to400_combined120k`
- `qcd_bbbb_iht400to600_20000`
- `qcd_bbbb_iht600plus_20000`

An additional high-statistics sample was produced on 2026-07-11:

- `qcd_bbbb_iht200to400_extra100k`

This extra sample is intended to reduce MC-statistical uncertainty in the dominant residual QCD background. When combining with the existing iHT200-400 QCD sample, the slice cross section should not be double-counted; the combined sample should use the same iHT200-400 cross section divided by the total number of generated events in that slice.

## Generator and detector setup

All samples are generated at leading order with MadGraph5_aMC, showered/hadronized with Pythia8, and passed through Delphes using the CMS-like card in this repository.

These samples are intended for Delphes-level method development and ML comparisons, not as final CMS-level background estimates.

## Reproducibility notes

For each sample, the analysis should preserve:

- generator process/card information
- run settings
- Delphes card
- production script
- generated ROOT file manifest
- metadata summary
- cross section
- number of generated events
- event weight convention

The event weight used in the current studies is:

```text
weight_pb = xsec_pb / n_generated
expected_events = weight_pb * luminosity_pb

Now add the new QCD sample-specific folder:

```bash
cat > backgrounds/qcd_bbbb_iht200to400_extra100k/README.md <<'MD'
# QCD bbbb iHT200-400 extra 100k

Production date: 2026-07-11

Purpose: increase MC statistics for the dominant QCD bbbb iHT200-400 residual background in the categorized BDT-v2 HH→4b study.

## Sample definition

- Process: QCD bbbb
- Phase space: 200 <= iHT < 400 GeV
- Generated events: 100,000
- Cross section: 126.35226440429688 pb
- Per-event standalone weight: 126.35226440429688 / 100000 pb
- Intended use: combine with the existing iHT200-400 QCD sample for improved statistics

Important: when combining with existing iHT200-400 events, do not double-count the slice cross section. Use the iHT200-400 slice cross section once and divide by the total number of generated events in the combined slice.

## Files

- `metadata_summary.txt`: production summary copied from the LPC metadata area
- `root_file_manifest.txt`: ROOT files produced for this sample
- `log_file_manifest.txt`: production logs
- `production_script.sh`: script used for production, if available
