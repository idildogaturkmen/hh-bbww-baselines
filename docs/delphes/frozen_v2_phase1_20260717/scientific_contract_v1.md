# Frozen-v2 production scientific contract

## Detector and reconstruction

- Final detector card:
  delphes_card_CMS_lpc_ak4ak8_run2_frozen_v2.tcl
- Required detector-card SHA256:
  1b2041245162de8360defdc502404e0696496c50773d4aa7f043798651a9517c
- Old-card ROOT and Parquet files are development/domain-shift samples only.
- All final samples must use the same frozen reconstruction schema.

## Physical evaluation versus training enrichment

- Physical and training-enrichment datasets must be recorded separately.
- Targeted QCD-bbbb samples are enrichment-only unless an exclusive
  phase-space and normalization prescription is proven.
- Targeted Zbbbb is enrichment-only unless overlap with inclusive Z+jets
  is resolved.
- Targeted ttbb is enrichment/diagnostic-only unless an explicit veto or
  exclusive truth-level definition separates it from inclusive ttbar.
- Inclusive ttbar is the candidate physical top-background source.
- Physical QCD slices must be mutually exclusive and collectively cover
  the intended phase space.
- Overlapping inclusive and targeted samples must never be added directly.

## Signal

- ggF HH and VBF HH require a process-definition and normalization audit.
- HH truth ancestry and four-jet assignment must pass validation before
  scaling signal production.
- Signal pilots are not sufficient for final sensitivity estimates.

## Dataset splitting

- Splits occur by independent generation shard.
- No event or shard may appear in more than one split.
- The final test sample remains untouched until all choices are frozen.

## Statistical expansion rule

Production expands by strata, beginning with an authorized 500K wave.

For every important final score/category region:

- raw test events: at least 20, preferably 50;
- effective sample size: at least 25, preferably 50;
- relative weighted MC uncertainty: below 20%, preferably below 10%;
- largest single-event yield fraction: below 20%, preferably below 10%;
- result stable under bootstrap resampling and repeated model seeds.

Five million events is a maximum planning target, not an automatic
submission size.
