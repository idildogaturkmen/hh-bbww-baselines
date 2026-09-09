# Wall-clock / CPU / storage estimate for doubling the QCD tail statistics

**Scope.** Planning estimate only. No Condor/EAF job launched, no generation
started. `NEW_JOB_LAUNCHED = NO`. All rates below are **actual measured**
throughput from existing receipts in this project; every number is cited to
its source file. Where no real measurement exists, this is stated explicitly
rather than invented (per instruction).

## 1. The production chain and what is/isn't actually measured

MG/Pythia -> HepMC -> Delphes -> ROOT/Parquet/scoring, per stage:

| stage | measured? | rate | source |
|---|---|---|---|
| Generation (MG5 matrix element) | **not applicable** | -- | The governing QCD sample is confirmed **direct Pythia8** (`HardQCD`-style multijet, single inclusive `pTHat>75 GeV` slice), **not MG5/LHE** (`CURRENT_QCD_GENERATION_AUDIT.md` Sec.2-3, verified from `gen_weight`=1.0 and no LHE multi-weight structure). There is no MG5 stage in this chain. |
| Pythia8 shower + Delphes + ROOT reconstruction (bundled) | **yes, measured** | 1.3517 allocated core-hours (Usr+Sys busy: 1.2461 h) for 50,000 events, 6 concurrent Condor jobs, `request_cpus=1` each | `track_a_targeted_qcd_pilot_arm1_200to500_50k_20260814_v1/PILOT_JOB_SUMMARY.tsv` + raw Condor `Job terminated` ClassAds. This is the **same pilot** already used by this project's own `TENFOLD_BACKGROUND_FEASIBILITY_MEMO.md` to project QCD generation cost, and is independently re-confirmed here. |
| Delphes-only, isolated | not measured at Condor/production scale | ~115 events/s (proxy, single core, local non-Condor smoke run, mtime-derived) | `hepmc/qcd_bbbb_presel_10k_pythia8.hepmc` timestamps vs `root/qcd_bbbb_presel_10k_pythia8_delphes.root` |
| ROOT/Parquet conversion | not isolated in production | ~700 events/s (same local smoke-run proxy) | as above |
| BDT-KF scoring (CPU) | **yes, measured (implied)** | 365,097,108 rows/hour aggregate (back-derived from "2.16 h for 788.6M QCD rows") | `TENFOLD_BACKGROUND_FEASIBILITY_MEMO.md` Route 3; CPU-only, "No GPU is required for this BDT scoring step" |
| SPA-Net 10M inference-only throughput | **NOT MEASURED anywhere in this project** | -- | `CPU_GPU_RESOURCE_TABLE.csv` explicitly flags this as an unmeasured gap; only SPA-Net **training** throughput exists (16,848 events/s, one A100 MIG 4g.40gb slice, 8.81h for 10M mixed events, `ADJUDICATION_REPORT_10M_SEED0_50EPOCH.md`) |

**Caveat carried forward from the project's own prior estimate** (stated,
not hidden): the 1.35 CPU-hr/50,000-event pilot rate is for a **local**
Pythia8+Delphes pilot at a different `pTHat` slice (200-500 GeV) than the
governing sample's inclusive `pTHat>75 GeV` production, and the actual
IHEP/author production chain's real cost could differ materially. This is
used here, as it was in `TENFOLD_BACKGROUND_FEASIBILITY_MEMO.md`, as "a
deliberately rough scale proxy -- not an author-chain benchmark," because it
is the only real, measured Pythia8+Delphes QCD generation rate anywhere in
this project.

## 2. CPU-hours required to double the QCD sample (either tail)

Per `HARVEY_MC_DOUBLING_ESTIMATE.md` Sec.1, doubling u>3.5 and u>4.5 both
require the **same** additional generated sample: +278,360,000,000 events
(a full second production the size of the current one).

```
rate (allocated-wall basis) = 1.3517 CPU-hr / 50,000 events = 2.7034e-5 CPU-hr/event
rate (CPU-busy basis)       = 1.2461 CPU-hr / 50,000 events = 2.4922e-5 CPU-hr/event

CPU-hours (allocated-wall) = 278,360,000,000 x 2.7034e-5 = 7,525,184 CPU-slot-hours
CPU-hours (CPU-busy)       = 278,360,000,000 x 2.4922e-5 = 6,937,288 CPU-slot-hours
```

**Cross-check:** scaling the project's own already-published 10x-literal-
regeneration figure (67.6M CPU-hours for +2.505e12 events,
`TENFOLD_BACKGROUND_FEASIBILITY_MEMO.md`) down by the ratio
`278.36e9 / 2.505e12 = 1/9.0` gives **7,511,831 CPU-hours** — agrees with
the direct calculation above to within 0.2%, as expected since both use the
same underlying pilot rate.

**These are single-CPU-*slot*-hours, not independently verified physical-
core-hours** (the same caveat the project's own audits carry: whether one
HTCondor slot maps to one physical core or one SMT thread was never
independently verified at LPC).

**Headline number: doubling the QCD sample behind either tail costs
approximately 6.9-7.5 million CPU-slot-hours.**

## 3. Expected wall time at representative Condor parallelism

| concurrent slots | wall time (allocated-wall CPU-hour basis) |
|---:|---|
| 150 (the largest QCD-specific Condor submission actually observed in this project, `qcd_hardqcd_importance_adaptive1500k_phys_wave3_20260718`, 150 jobs queued) | ~50,168 hours = **~5.7 years** |
| 1,000 | ~7,525 hours = **~314 days** |
| 10,000 | ~753 hours = **~31.4 days** |
| 64,000 (the illustrative scale used in the project's own prior Harvey CPU/GPU audit) | ~118 hours = **~4.9 days** |
| 100,000 (the scale used in `TENFOLD_BACKGROUND_FEASIBILITY_MEMO.md`'s own Route 4 illustration) | ~75 hours = **~3.1 days** |

None of the 1,000+-slot rows represents a concurrency level actually
sustained by a real QCD campaign in this project (the largest observed QCD-
generation-specific submission is 150 jobs; 459 concurrently-running slots
was observed once, for an unrelated training-shard build, not QCD
generation) — they are presented, as in the project's own prior audits,
as **hypothetical allocation scenarios**, not resource commitments.

## 4. Storage

Doubling the QCD sample requires storing a second copy of the current
selected/inference-level ntuples:

- Current QCD selected-ntuple remote footprint: 540.589 GB (`TENFOLD_
  BACKGROUND_FEASIBILITY_MEMO.md` Sec.1, `QCD_DelphesHH4JTrig_forInfer(+2)_
  merged_ntuple`, 87,623,306 rows).
- **Additional storage for doubling: ~540.6 GB** (selected-ntuple level
  only; this excludes transient HepMC/Delphes-full-event intermediate
  storage upstream of selection, which is far larger per event and was
  never measured for the governing IHEP-side chain -- `CURRENT_QCD_
  GENERATION_AUDIT.md` Sec.4).

## 5. Scoring cost of the additional sample

- **BDT-KF (CPU, measured rate):** 87,623,306 additional rows / 365,097,108
  rows/hour = **0.24 hours (~14.4 minutes)**.
- **SPA-Net 10M (the model these tail regions are actually defined on):**
  **no inference-only throughput measurement exists anywhere in this
  project.** The only related number is SPA-Net **training** throughput
  (16,848 events/s on one A100 MIG slice); if inference is assumed to be at
  least as fast as training per event (a common but unverified assumption,
  since inference has no backward pass), a rough **upper bound** would be
  87,623,306 / 16,848 = ~5,201 s = **~1.4 hours** on one such GPU slice.
  **This is explicitly not a measurement and should not be quoted as one.**

## 6. Bottleneck stage

**Generation (Pythia8 + Delphes) dominates by roughly 5-6 orders of
magnitude.** ~7.1-7.5 million CPU-hours for generation vs. ~14 minutes for
BDT-KF scoring and (order) ~1.4 hours for a training-rate-bounded SPA-Net
inference guess. Storage (~541 GB) and transfer are negligible next to the
compute cost. **Any effort to reduce total cost must reduce generation
cost, not scoring/storage cost** -- this is the direct motivation for the
targeted-generation designs in `HARVEY_PRESELECTION_CANDIDATES.md` Part 5-6
and the existing `track_b_qcd_tail_generation_strategy_20260902_v1` package.

## 7. Harvey-ready answer

> At current acceptance and measured throughput, doubling the expected raw
> support above **u>3.5** (currently 68 raw QCD events, N_eff=89.3 combined,
> 10.6% relative MC uncertainty) would require approximately **2.78e11
> additional generated QCD events** (a full second production the size of
> the current sample), costing approximately **7.5 million CPU-slot-hours**
> (measured at 1.35 CPU-hr/50,000 events on the project's own Pythia8+
> Delphes pilot) -- roughly **3-5 days at a hypothetical 64,000-100,000
> concurrent-slot allocation**, or **multiple years at the largest
> concurrency (150 slots) actually used for a QCD campaign in this project
> to date**. Doubling **u>4.5** (currently 12 raw QCD events) requires the
> identical additional generation, since ordinary (non-importance-sampled)
> generation cannot differentially target one tail region. Storage grows by
> ~541 GB; scoring cost is negligible (minutes) next to generation cost,
> which is the overwhelming bottleneck.

`NEW_JOB_LAUNCHED = NO`
