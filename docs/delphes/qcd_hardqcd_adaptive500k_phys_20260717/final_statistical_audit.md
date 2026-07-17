# Inclusive-QCD adaptive 500K statistical audit

Campaign: `qcd_hardqcd_importance_adaptive500k_phys_20260717_0022`  
Cluster: `84891534` on `lpcschedd6.fnal.gov`  
Production payload commit: `ce9964783b56a0d9eca84a6b2a5304a06c7b978f`  
Audit implementation commit: `05882d4`

## Terminal and cross-layer status

- All 54 jobs reached Condor terminal state 4 with exit code 0 and
  `ExitBySignal=false`.
- All 54 returned receipts have stage `complete_copied_and_verified`.
- EOS contains exactly 54 canonical bundles and no partial bundle.
- Requested, HepMC, ROOT, and event-summary counts are all exactly 500,000.
- All 54 receipt/EOS/downloaded Adler-32 triplets agree.
- All 4,266 recorded shard-level archive, metadata, hash, count, pTHat, seed,
  split, and provenance checks pass.
- Candidate Parquets contain 27 total rows. Forty-four shard candidate
  Parquets are valid readable zero-row products.
- Streaming validation retained no downloaded bundle, HepMC, ROOT, or
  extracted tree. The measured canonical EOS payload is 83,620,859,853 bytes
  (0.076052728994 TiB).

## Per-bin selection statistics

| pTHat [GeV] | generated | exactly 2b | exactly 3b | >=4b / HH-like | rHH<80 | rHH<50 | HH-like efficiency or exact one-sided 95% upper bound |
|---|---:|---:|---:|---:|---:|---:|---:|
| 50-75 | 309,163 | 1,081 | 12 | 0 | 0 | 0 | <9.68976804972e-6 |
| 75-100 | 77,388 | 562 | 22 | 1 | 1 | 1 | 1.29219000362e-5 |
| 100-200 | 48,521 | 657 | 24 | 2 | 2 | 1 | 4.12192658849e-5 |
| 200-300 | 14,543 | 310 | 29 | 1 | 1 | 0 | 6.87616035206e-5 |
| 300-500 | 12,855 | 340 | 40 | 2 | 2 | 2 | 1.55581485803e-4 |
| 500-700 | 12,526 | 412 | 47 | 8 | 2 | 1 | 6.38671563149e-4 |
| 700-1000 | 12,504 | 420 | 38 | 6 | 3 | 0 | 4.79846449136e-4 |
| 1000-inf | 12,500 | 363 | 59 | 7 | 0 | 0 | 5.6e-4 |

The other exact zero-count bounds are:

- rHH<80: 50-75, `9.68976804972e-6`; 1000-inf,
  `2.39629866060e-4`.
- rHH<50: 50-75, `9.68976804972e-6`; 200-300,
  `2.05970140086e-4`; 700-1000, `2.39553218216e-4`; 1000-inf,
  `2.39629866060e-4`.

The zero-candidate 50-75 stratum remains part of the physical estimator. Its
zero count is represented by the bound above, not by assigning zero physical
cross section.

## Physical estimator

Each pTHat stratum uses one event-count-weighted aggregate cross-section
estimate. An event is weighted as
`sigma_stratum * raw_event_weight / sum(raw_event_weight)`, where the
denominator covers every accepted generated event across all shards of that
stratum in the dataset being evaluated. No shard receives an independent
`sigma/N_shard` normalization.

| region | count | physical yield [pb] | conditional variance [pb^2] | ESS | maximum event fraction |
|---|---:|---:|---:|---:|---:|
| exactly 2b | 4,145 | 104,123.156498 | 5,042,170.33005 | 2,139.399 | 5.6723e-4 |
| exactly 3b | 271 | 2,323.780314 | 92,271.861789 | 58.510 | 0.025345 |
| >=4b / HH-like | 27 | 96.1748466 | 2,901.259920 | 3.188 | 0.411098 |
| rHH<80 | 11 | 95.8931497 | 2,901.248893 | 3.169 | 0.412305 |
| rHH<50 | 5 | 66.5716000 | 2,226.431375 | 1.991 | 0.593906 |

For HH-like/rHH<80, bins 75-100 and 100-200 contribute 53.88% and 45.67%
of the conditional variance, respectively. The loose-control variance is
instead led by the low bins: 50-75 contributes 74.11% of exactly-2b variance
and 45.11% of exactly-3b variance.

Compressed storage is 167,241.72 bytes/generated event,
3.097068883e9 bytes/HH-like event, and 7.601896350e9 bytes/rHH<80 event.

## Bootstrap stability

One thousand deterministic whole-shard bootstrap replicas were drawn within
each pTHat stratum.

| metric | nominal [pb] | bootstrap p16-p84 [pb] | relative bootstrap std |
|---|---:|---:|---:|
| exactly 2b | 104,123.156 | 102,220.369-106,088.085 | 1.83% |
| exactly 3b | 2,323.780 | 1,939.682-2,704.947 | 16.62% |
| HH-like | 96.1748 | 53.4625-147.9217 | 50.55% |
| rHH<80 | 95.8931 | 53.1599-147.6822 | 50.70% |
| rHH<50 | 66.5716 | 27.0265-109.1291 | 66.76% |

## Comparison with the 80K pilot

| metric | 80K pilot | adaptive 500K |
|---|---:|---:|
| HH-like rows | 23 | 27 |
| rHH<80 rows | 10 | 11 |
| rHH<50 rows | 2 | 5 |
| HH-like physical yield [pb] | 24.8237 | 96.1748 |
| HH-like conditional variance [pb^2] | 108.364 | 2,901.260 |
| HH-like ESS | 5.685 | 3.188 |
| HH-like maximum event fraction | 20.65% | 41.11% |
| rHH<80 physical yield [pb] | 22.9532 | 95.8931 |
| rHH<80 ESS | 4.921 | 3.169 |
| rHH<50 physical yield [pb] | 0.81894 | 66.5716 |
| rHH<50 ESS | 1.021 | 1.991 |
| bytes/generated event | 184,220.80 | 167,241.72 |

The 500K wave sharply reduced loose-control variance (exactly-2b by 96.6%
and exactly-3b by 76.6%), but it exposed previously unobserved, high-physical-
weight HH-like events in 75-100 and 100-200 GeV. The resulting HH-like and
rHH estimators are not converged even though campaign validation passes.

## Split audit

The immutable whole-shard assignment is unchanged: train 410,453 events,
validation 55,004, and test 34,543. Test was not used for allocation,
threshold selection, or tuning.

The current test subset covers only pTHat bins 50-75, 100-200, and 200-300.
It contains one HH-like/rHH<80 event and no rHH<50 event. It is therefore not
an eight-stratum final physical-evaluation sample. Any future approved wave
should predeclare sealed test shards for the missing 75-100, 300-500,
500-700, 700-1000, and 1000-inf strata before outcomes are examined.

## Review recommendation

Recommend **another targeted adaptive inclusive-HardQCD wave**, not 5M and
not final ML/inference yet.

This recommendation uses train+validation only. The updated control-aware
diagnostic shares, including the existing 2.5% per-bin floor, are:

| pTHat [GeV] | diagnostic share | diagnostic 500K-equivalent events |
|---|---:|---:|
| 50-75 | 60.2424% | 301,212 |
| 75-100 | 16.5617% | 82,809 |
| 100-200 | 10.2104% | 51,052 |
| 200-300 | 2.90635% | 14,532 |
| 300-500 | 2.57280% | 12,864 |
| 500-700 | 2.50535% | 12,527 |
| 700-1000 | 2.50085% | 12,504 |
| 1000-inf | 2.50010% | 12,500 |

This table is diagnostic evidence, not an authorized or prepared campaign.
Any next wave requires a new human-approved total and identity. Inclusive
pTHat-stratified HardQCD remains the physical inference sample; QCD_bbbb,
Zbbbb, ttbb, and other targeted heavy-flavor samples remain explicitly
ML-enrichment only.

No 5M campaign was prepared or submitted.
