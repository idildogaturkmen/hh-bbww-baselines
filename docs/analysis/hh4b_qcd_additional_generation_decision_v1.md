# HH4b additional QCD generation decision v1

## Evidence

The six physical-QCD events in the ge4 `R_HH<34` region occupy four authoritative pTHat strata: 75–100, 100–200, 200–300, and 500–700 GeV. Every populated campaign×pTHat×RHH cell has exactly one row and Neff=1; the ge4 `[30,34)` bin has zero QCD rows. The most important missing support is therefore not inclusive QCD normalization but fourth-tag, Higgs-mass-like tails within those strata.

The total selected QCD is dominated by 75–100 (39.49%), 100–200 (32.22%), 50–75 (24.24%), with smaller higher-pTHat contributions. The 50–75 stratum has no ge4 event, so it also requires explicit tail coverage even though it does not appear in the six populated critical cells.

For a positive weighted estimate, relative finite-MC precision is approximately `1/sqrt(Neff)`. Targets of 20%, 10%, and 5% correspond to Neff 25, 100, and 400. Starting from Neff≈1 in a critical populated cell requires naive factors 25, 100, and 400 in effective support. For the four currently populated critical pTHat strata, whose existing production totals 789,355 events, naive unfiltered scaling implies approximately:

| target | additional generated events | 10k-event jobs | ROOT/bundle storage at frozen 166 kB/event |
|---|---:|---:|---:|
| 20% (Neff 25) | 18.94 million | 1,895 | 3.15 TB |
| 10% (Neff 100) | 78.15 million | 7,815 | 13.0 TB |
| 5% (Neff 400) | 314.95 million | 31,496 | 52.4 TB |

These are lower-information naive estimates: they do not solve the empty 50–75 or ge4 `[30,34)` cells and assume effective support scales linearly. CPU cannot be responsibly quoted before a resource pilot freezes generator throughput; the job counts and storage are the defensible planning quantities today.

## Why brute force is inefficient

Only 6 of 1,811,373 generated physical-QCD events populate ge4 `R_HH<34`. Inclusive scaling spends nearly all CPU/storage outside the needed fourth-tag mass tail. A targeted campaign should stratify authoritative pTHat and enrich heavy-flavor/tag-like topologies or generator-level four-b phase space, while retaining an unbiased estimator through explicit filter efficiencies and disjoint/overlap-aware strata.

Candidate sampling variables include pTHat bins, generator heavy-flavor multiplicity, b-quark pT/eta acceptance, and generator-level four-b mass/pair-mass regions. None may be used as an undocumented veto. Every stratum must freeze: generator process/card and version, pTHat/filter definitions, generated and accepted counts, filter efficiency with uncertainty, cross section and units, signed sumw/sumw2 denominator, per-event weight convention, random seeds, parent/extension relationship, overlap-removal/stitching rule, and exact event-weight transport.

Validation requires statistically supported overlap samples between the new enriched strata and the current hard-QCD estimator, plus an unfiltered or less-filtered control sample. Closure must be checked by pTHat, tag multiplicity, mass plane, RHH, mHH, candidate kinematics, and source groups before the new estimator enters a likelihood.

## Decision

The existing sample cannot validate any tested transfer to ge4 and cannot support the nominal shape bins. Targeted generation is likely required, but the sampling/filter design and overlap-control sample are not yet frozen, and no CPU pilot exists.

**GENERATION_LIKELY_REQUIRED_BUT_DESIGN_NOT_READY**
