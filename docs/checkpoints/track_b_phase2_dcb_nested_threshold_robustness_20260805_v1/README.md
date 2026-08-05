# Track B Phase 2 DCB nested-threshold robustness checkpoint

## Status

**The released DCB control validation and exhaustive nested-threshold robustness study are complete for the deterministic 1,000-event ZH and ZZ released-model products. No additional ZH/ZZ DCB scale-out is required for Phase 2.**

- Checkpoint staged from repository head: `3be9b921c117e60c9c68e1c6002423aaa07ce79b`
- Samples: ZH and ZZ control processes
- Base selection: `pass_selection == 1` and `pass_4j3b_selection == 1`
- Score thresholds: `0.8`, `0.9`, `0.95`, `0.99`, and `0.997`
- Interpretation: unweighted released-model control validation

No ROOT or NPZ binaries are stored in this checkpoint. The preserved event tables, summaries, contracts, manifests, and receipts are sufficient to reproduce every reported count and localization fraction from the accepted fit outputs.

## Design

- The accepted `p_signal > 0.997` fits were reused.
- Only the disjoint band `0.8 < p_signal <= 0.997` was newly fitted.
- New fits: 358 ZH events and 238 ZZ events.
- Combined exhaustive populations: 545 ZH events and 305 ZZ events.
- No event was fitted twice.
- Every nested threshold was derived from one combined event table.

## Principal threshold results

### ZH

| `p_signal` threshold | Events | Fit success | Physical primary / successful | Either peak in ZH region / successful |
|---:|---:|---:|---:|---:|
| > 0.8 | 545 | 527/545 (96.70%) | 467/527 (88.61%) | 247/527 (46.87%) |
| > 0.9 | 470 | 455/470 (96.81%) | 411/455 (90.33%) | 220/455 (48.35%) |
| > 0.95 | 419 | 405/419 (96.66%) | 367/405 (90.62%) | 204/405 (50.37%) |
| > 0.99 | 279 | 273/279 (97.85%) | 256/273 (93.77%) | 160/273 (58.61%) |
| > 0.997 | 187 | 183/187 (97.86%) | 176/183 (96.17%) | 122/183 (66.67%) |

### ZZ

| `p_signal` threshold | Events | Fit success | Physical primary / successful | Either peak in ZZ region / successful |
|---:|---:|---:|---:|---:|
| > 0.8 | 305 | 299/305 (98.03%) | 248/299 (82.94%) | 120/299 (40.13%) |
| > 0.9 | 236 | 231/236 (97.88%) | 194/231 (83.98%) | 104/231 (45.02%) |
| > 0.95 | 196 | 192/196 (97.96%) | 163/192 (84.90%) | 93/192 (48.44%) |
| > 0.99 | 110 | 108/110 (98.18%) | 98/108 (90.74%) | 59/108 (54.63%) |
| > 0.997 | 67 | 65/67 (97.01%) | 61/65 (93.85%) | 40/65 (61.54%) |

## Scientific interpretation

Primary optimizer success is stable across thresholds at approximately 97–98%. The physical-primary and target-region fractions improve as the released signal score is tightened.

- ZH either-peak localization rises from 46.87% at `p_signal > 0.8` to 66.67% at `p_signal > 0.997`.
- ZZ either-peak localization rises from 40.13% at `p_signal > 0.8` to 61.54% at `p_signal > 0.997`.

This monotonic behavior supports a real association between the released-model signal score and resonance-like localization in the 136-cell score plane. It does not establish a calibrated mass estimator or an unbiased signal mass resolution.

## Fit-quality findings

- ZH: 4 primary optimizer failures; 7 successful but nonphysical primary solutions; 29 invalid or sentinel secondary solutions; and 11 valid but nonphysical secondary solutions.
- ZZ: 2 primary optimizer failures; 4 successful but nonphysical primary solutions; 11 invalid or sentinel secondary solutions; and 9 valid but nonphysical secondary solutions.
- Parameter-bound contact is retained as a nuisance diagnostic and is not an automatic event exclusion.
- ZH has no event with `bg_amp > 1` in the full `p_signal > 0.8` population.
- ZZ has two events with `bg_amp > 1` at `p_signal > 0.8`. Removing those two events in the predeclared sensitivity view changes the either-peak localization fraction only from 40.13% to 40.40%.

Therefore the catastrophic background-amplitude outliers do not drive the principal localization trend.

## Interpretation boundary

These results are unweighted control-sample diagnostics. They are not:

- cross-section-weighted efficiencies;
- calibrated mistag rates;
- signal mass-resolution measurements;
- expected yields or significances;
- a substitute for evaluation on an independent HH signal sample.

The two `gen_weight` entries remain semantically undefined and must not be used for physical normalization until their producer definitions are confirmed.

## Phase 2 decision

The ZH/ZZ DCB control and threshold-robustness gates are complete. The next internal gate is authoritative provenance for the mapping between signal output indices 0–135 and the released mass-grid labels.

The major external Phase 2 blockers remain:

1. an exact path to an independent ggF or VBF HH evaluation sample;
2. authoritative definitions of the two `gen_weight` entries.

No additional ZH/ZZ DCB fitting should be launched unless a future provenance or implementation audit identifies a concrete defect.

## Git-safe TSV serialization

Several original TSV artifacts produced by Python `csv.writer` use CRLF record terminators. Git reports the carriage return in each CRLF terminator as trailing whitespace. The checkpoint therefore stores Git-safe LF representations.

`tsv_git_serialization_receipt.json` records every normalized TSV, its immutable source SHA-256 when applicable, its checkpoint SHA-256, line-ending counts, header, column count, row count, and proofs that header, row order, row widths, and every field value are unchanged. The combined event table additionally closes at 850 unique identities: 545 ZH and 305 ZZ. No scientific content was changed.

The original source artifacts remain preserved under their recorded `/tmp` paths and hashes in `source_artifact_manifest.tsv`.

## Evidence map

- `evidence/official_threshold/`: accepted exhaustive `p_signal > 0.997` results and post-hoc acceptance receipt.
- `evidence/pathology_audit/`: failed, nonphysical, bound-contact, and background-amplitude diagnostics.
- `evidence/nested_threshold/`: exhaustive event table, threshold summary, predeclared quality contract, and incremental-band receipt.
- `source_manifests/`: original `/tmp` artifact manifests for all three source runs.
- `source_artifact_manifest.tsv`: source paths, source hashes, copied checkpoint paths, and provenance roles.
