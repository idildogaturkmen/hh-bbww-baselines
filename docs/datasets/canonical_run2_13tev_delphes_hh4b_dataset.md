# Canonical Run-2 13 TeV Delphes HH→4b Dataset

## Scope

This dataset card documents the immutable Delphes samples used for the HH→4b machine-learning baseline study.

The canonical background contains exactly 5,000,000 generated events in 520 frozen members. Signal samples are counted separately and are not included in the 5M background.

The dataset is registry-driven and remains partitioned by shard. The ROOT files must not be combined into one giant merged ROOT file.

## Canonical background status

- Generated background events: **5,000,000**
- Canonical background members: **520**
- Train events: **3,791,373**
- Validation events: **1,024,084**
- Test events: **184,543**
- Membership: **validated and frozen**
- Classifier-source manifest: **validated**
- Classifier-table materialization: **authorized**
- Physics weights: **not yet frozen**
- Physics-yield normalization: **not yet authorized**

Generated-event counts and candidate-row counts are different quantities. The 5,000,000 total always refers to generated canonical events.

## Background composition by canonical component

| Component | Description | Events | Members | Train | Validation | Test | Candidate rows |
| --- | --- | --- | --- | --- | --- | --- | --- |
| qcd | Adaptive HardQCD importance-sampled production | 2,510,000 | 261 | 1,811,373 | 514,084 | 184,543 | 81 |
| ttbar | Canonical subset of legacy inclusive ttbar production | 270,000 | 27 | 220,000 | 50,000 | 0 | 998 |
| wavea | QCD bbbb, inclusive ttbar and Zbbbb Wave-A production | 1,300,000 | 130 | 1,040,000 | 260,000 | 0 | 38,239 |
| waveb | ttH(bb), ttZ(bb), tttt and VBF H(bb) Wave-B subset | 170,000 | 17 | 140,000 | 30,000 | 0 | 16,930 |
| wavec1 | ttW, WH(bb), WW and WZ(bb) Wave-C1 production | 220,000 | 22 | 180,000 | 40,000 | 0 | 393 |
| single_top | Single-top pilot and scale-out production | 330,000 | 33 | 240,000 | 90,000 | 0 | 267 |
| hbb | ggH(bb) and bbH(bb) background production | 150,000 | 15 | 120,000 | 30,000 | 0 | 1,758 |
| triboson | WWZ, WZZ and ZZZ with direct-Z→bb truth filtering | 50,000 | 15 | 40,000 | 10,000 | 0 | 201 |

## Detailed background composition by process family

| Component | Process family | Events | Members | Train | Validation | Test |
| --- | --- | --- | --- | --- | --- | --- |
| qcd | qcd_hardqcd | 2,510,000 | 261 | 1,811,373 | 514,084 | 184,543 |
| ttbar | ttbar_inclusive | 270,000 | 27 | 220,000 | 50,000 | 0 |
| wavea | qcd_bbbb_general | 150,000 | 15 | 120,000 | 30,000 | 0 |
| wavea | qcd_bbbb_iht400to600 | 150,000 | 15 | 120,000 | 30,000 | 0 |
| wavea | ttbar_inclusive | 750,000 | 75 | 600,000 | 150,000 | 0 |
| wavea | zbbbb | 250,000 | 25 | 200,000 | 50,000 | 0 |
| waveb | tth_hbb | 40,000 | 4 | 30,000 | 10,000 | 0 |
| waveb | tttt | 40,000 | 4 | 40,000 | 0 | 0 |
| waveb | ttz_zbb | 40,000 | 4 | 30,000 | 10,000 | 0 |
| waveb | vbf_hbb | 50,000 | 5 | 40,000 | 10,000 | 0 |
| wavec1 | ttw | 60,000 | 6 | 50,000 | 10,000 | 0 |
| wavec1 | wh_hbb | 60,000 | 6 | 50,000 | 10,000 | 0 |
| wavec1 | ww | 40,000 | 4 | 30,000 | 10,000 | 0 |
| wavec1 | wz_zbb | 60,000 | 6 | 50,000 | 10,000 | 0 |
| single_top | schannel_single_top | 30,000 | 3 | 20,000 | 10,000 | 0 |
| single_top | tchannel_antitop | 80,000 | 8 | 60,000 | 20,000 | 0 |
| single_top | tchannel_top | 80,000 | 8 | 60,000 | 20,000 | 0 |
| single_top | tw_antitop | 70,000 | 7 | 50,000 | 20,000 | 0 |
| single_top | tw_top | 70,000 | 7 | 50,000 | 20,000 | 0 |
| hbb | bbh_hbb_4fs | 50,000 | 5 | 40,000 | 10,000 | 0 |
| hbb | ggh_hbb | 100,000 | 10 | 80,000 | 20,000 | 0 |
| triboson | wwz_zbb | 26,000 | 5 | 20,800 | 5,200 | 0 |
| triboson | wzz_zbb | 16,500 | 5 | 13,200 | 3,300 | 0 |
| triboson | zzz_zbb | 7,500 | 5 | 6,000 | 1,500 | 0 |

## Candidate-source availability

The classifier-source audit found 27 canonical members with an existing local candidate Parquet and 493 members whose candidate Parquet must be extracted from an audited EOS bundle.

| Component | Local candidate Parquets | EOS bundle extractions | Candidate rows |
| --- | --- | --- | --- |
| qcd | 0 | 261 | 81 |
| ttbar | 27 | 0 | 998 |
| wavea | 0 | 130 | 38,239 |
| waveb | 0 | 17 | 16,930 |
| wavec1 | 0 | 22 | 393 |
| single_top | 0 | 33 | 267 |
| hbb | 0 | 15 | 1,758 |
| triboson | 0 | 15 | 201 |

## Signal samples

| Signal process | Canonical events | Train | Validation | Test | Status |
| --- | --- | --- | --- | --- | --- |
| VBF HH→bbbb | 100,000 | 80,000 | 20,000 | 0 | Canonical membership validated and frozen; physics-yield normalization is not yet authorized. |
| ggF HH→bbbb | Not yet frozen | — | — | — | Production artifacts exist, but canonical ggF membership and duplicate handling require a separate audit. |

The VBF signal count is supported by the passing canonical 100k registry audit. No publishable ggF count is assigned until its canonical registry is reconciled and frozen.

## Canonical background accounting

```text
Adaptive HardQCD importance production       2,510,000
Legacy inclusive ttbar canonical subset        270,000
Wave-A audited production                    1,300,000
Wave-B deterministic canonical subset          170,000
Wave-C1 audited production                     220,000
Single-top pilot and scale-out                  330,000
H(bb) backgrounds                              150,000
Canonical triboson                              50,000
                                              ---------
Total                                        5,000,000
```

## Required accounting and reuse conventions

1. QA canaries, smoke tests, malformed jobs and failed jobs contribute zero canonical events.
2. Frozen train, validation and test assignments must be preserved. The QCD test split must not be used for training, hyperparameter selection or threshold optimization.
3. Candidate rows are selected analysis candidates and must not be reported as generated-event counts.
4. `physics_yield_authorized=false` means final normalization has not yet been frozen; it does not invalidate the events.
5. Importance-sampled, enriched and truth-filtered samples require process-specific normalization.
6. Triboson normalization must apply the direct-Z→bb branching/filter factor exactly once.
7. Final HH signal normalization must use the separately frozen official SM production cross sections and branching fractions, not raw generator estimates.

## Authoritative artifacts

- Repository commit at generation: `fe94f0043c510537d9670a8c6a295f814b9db3ad`
- Unified registry: `/uscms_data/d3/iturkmen/repos/hh-bbww-baselines/outputs/agent_runs/final_background_unified5m_registry_audit_20260723_v2/unified_background_5m_registry.tsv`
- Unified registry SHA-256: `6bbb256a326147cd679fbbb512574d845b2310eb5a09a71758a7bb210402353c`
- Unified registry audit: `/uscms_data/d3/iturkmen/repos/hh-bbww-baselines/outputs/agent_runs/final_background_unified5m_registry_audit_20260723_v2/unified_background_5m_registry.json`
- Unified registry audit SHA-256: `f629c6f567e1b95d754275c947a3701baba51647511f882fe6cb0ff7b13ceb7f`
- Classifier-source manifest: `/uscms_data/d3/iturkmen/repos/hh-bbww-baselines/outputs/agent_runs/final_background_5m_classifier_source_audit_20260723_v1/background_5m_classifier_source_manifest.tsv`
- Classifier-source manifest SHA-256: `65bc318444ca29150a314efc6232a533522761d31eb0cc531630285f35d7d582`
- Classifier-source audit: `/uscms_data/d3/iturkmen/repos/hh-bbww-baselines/outputs/agent_runs/final_background_5m_classifier_source_audit_20260723_v1/background_5m_classifier_source_audit.json`
- Classifier-source audit SHA-256: `54de5c3005d0d16f8f273f125ffaeb0fec115b139731e8ac58540a65d567bb30`
- VBF canonical audit: `/uscms_data/d3/iturkmen/repos/hh-bbww-baselines/outputs/agent_runs/vbf_hh4b_sm_canonical100k_registry_audit_20260723_v1/vbf_hh4b_canonical100k_registry.json`
- VBF canonical audit SHA-256: `6b9044ba3fdf30003a60a70acac7c7b574074bd6d145fef4383b53158c8847e0`

## Reproducibility

Classifier-table materialization must retain the canonical component, process family, source campaign, target tag, shard identity, seed and frozen dataset split for every member.

This card documents membership and event accounting. Final physics weights and yield normalization constants belong in a separate, versioned normalization freeze.

Dataset card generated at: `2026-07-23T18:39:07.693773+00:00`
