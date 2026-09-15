# Full-144M ParT + SPA-Net scale feasibility audit (2026-08-26)

Planning/audit only. No embedding production launched, no file modified inside
`track_b_part_full_embedding_production_exact_2M_20260824_v1` (frozen,
authoritative, untouched), no training run, no `holdout_A`/`holdout_B` file
opened, no interaction with the live BDT-K Condor job (cluster `85248687.0`,
`lpcschedd6.fnal.gov`, still `RUNNING` at audit time — not queried, not
touched), no expensive inference. All facts below come from reading existing
frozen manifests/reports plus a handful of cheap local metadata checks
(`du`, `wc -l`, JSON/TSV parsing, one `grep`) run directly by the auditor to
verify subagent claims — no ROOT files were opened and no XRootD reads were
performed in this task.

**Methodology note.** Three parallel research passes were run, then the
auditor independently re-verified the highest-stakes claims against primary
sources. One subagent claim was found to be **wrong** and is corrected here:
it read the frozen 2M cohort's binary classification label name
(`label 1 = "ggHH"`, a code-level shorthand in `build_hdf5_2M.py`) as if it
named the physical signal *process*, and concluded the 2M cohort's signal was
`gghh_kl1` (the holdout process). Direct inspection of
`manifest_train_2M.tsv` and `SHARD_MANIFEST.json` shows the actual source
files for the 2M cohort's "signal" bucket are
`HH4b_2HDM_H3VAR_H1H2_40to200_merged_ntuple/*.root` — the **same** H3VAR
family used by the full-144M training population, not `gghh_kl1`. This
correction materially changes the Task 1 and Task 6 conclusions below in a
more favorable direction than the initial subagent read suggested.

---

## Task 1 — the exact 144M population

**Authoritative manifest:** `FULL144M_FILE_MANIFEST.json` (4 byte-identical
copies under `track_b_harvey_bdt_working_points_20260818_v1/step2k_*`,
`step2n_*`, `step2o_*`).

| Quantity | Value |
|---|---:|
| Total training events (`grand_total_rows`) | **144,291,951** |
| signal (H3VAR) | 69,750,093 (500 files) |
| QCD | 62,941,859 (400 files) |
| ttbar | 11,599,999 (200 files) |
| Total source files | 1,100 |
| `keep_fraction` (final BDT fit) | 1.0 |
| `train_val_split` (final BDT fit) | `None` — deliberately pooled, round budget fixed at 4000 instead of early-stopping |

**Process composition.** Three macro-buckets, one physical process each:
- **signal** = `HH4b_2HDM_H3VAR_H1H2_40to200_merged_ntuple` (a 2HDM H3VAR H1↔H2
  sample) — **not** ggHH+VBFHH combined, and **not** `gghh_kl1`.
- **QCD** = `QCD_DelphesHH4JTrig_merged_ntuple`.
- **ttbar** = `TTbar_ntuple`.

One row = one event (candidate jets are zero-padded to 10 slots *within* the
row; rows are never exploded per jet — confirmed from
`stage3_materialize_full144m.py`/`stage1_materialize_canary.py`).

**Source ROOT files.** IHEP EOS, owner account `coli`, path prefix
`/eos/ihep/cms/store/user/coli/datasets/hh4b/training/`, accessed via XRootD
redirector `root://cceos.ihep.ac.cn:1094/`. Provenance chain
(`phase4O_trackb_publication_transfer_readiness_20260814_v1/VERIFICATION_REPORT.md`,
`YANG_LI_V2_PHYSICAL_REFERENCE.md`) confirms `coli` = Congqiao Li; dataset
traced to arXiv:2508.15048v2 (Tianyi Yang & Congqiao Li, Peking University,
`pku-hep-group/jetfree-hh4b`) — this **is** the "Congqiao/Yang-Li ~144M"
population the task refers to. A flattened-feature copy (52/82-column BDT
K/KF vectors, not raw constituents) is staged at FNAL EOS,
`step2k_full144m_staged_data_20260821_v1/`, 49,926,361,446 bytes
(46.498 GiB), independently checksum- and re-extraction-verified.

**Split contract.** The final BDT fit pools everything, but a governing
80/20 **file-level** split exists and is fully specified/reusable
(`step2f_full_source_governing_convergence_20260819_v1/scripts/convergence_gate_step2f.py`):

| Process | Train files / rows | Val files / rows |
|---|---:|---:|
| signal | 400 / 55,798,332 | 100 / 13,951,761 |
| QCD | 320 / 50,355,392 | 80 / 12,586,467 |
| ttbar | 160 / 9,279,999 | 40 / 2,320,000 |
| **Total** | **880 / 115,433,723** | **220 / 28,858,228** |

**Event identity convention.** `(source_file_id, source_entry_index)` —
`source_entry_index` is the 0-based row position within the source TTree in
native read order; `source_file_id` resolves to a full remote path via
`PROVENANCE_FILE_ID_MANIFEST.json`. Stable and joinable.

**A separate, distinct population exists and must not be confused with the
above:** `FROZEN_REMOTE_EVALUATION_MANIFEST.md`
(`step3_scoring/remote_evaluation_inventory_20260819_v1/`) is a **held-out
evaluation** population — 16 background processes / 718 files (different
selection cuts, `forInfer`/`forInfer2` directories) plus 9 signal benchmark
points (`gghh_kl0/1/5`, 6 VBF cv/c2v/kl variants; 4,800 files). `holdout_A`/
`holdout_B` = `gghh_kl1`, and are confirmed **excluded** from the 144M
training build (`TASK_D_EVALUATION_BOUNDARY_FREEZE.md`: "No file from
`gghh_kl1` ... was opened, read, or referenced" in the 144M build).

**Is the existing 2M SPA-Net cohort an exact subset?** **Yes, at the
file-population level; no, at the row level.** Direct verification
(`manifest_train_2M.tsv`, `SHARD_MANIFEST.json`):
- Its train split touches **exactly** the same 400 signal + 320 QCD + 160
  ttbar governing train files as the 144M split above (880 files, byte-exact
  match on file lists and even on per-file `(offset, quota)` windows between
  the SPA-Net HDF5 manifest and the ParT shard manifest).
- Its signal bucket is drawn from the same `HH4b_2HDM_H3VAR_H1H2` family as
  full144's signal bucket (see correction above) — not a different process.
- But each file is read via a **windowed subsample**
  (`cyclic_contiguous_window(offset, length, mod)`, e.g. 2,720 of 156,976
  rows from one QCD file), not `keep_fraction=1.0`. So the 2M cohort's rows
  are a **file-complete, proportionally-stratified windowed subsample** of
  the 144M population — every governing file is represented, but only a
  minority of each file's rows are included. It is a subset of the
  *population*, not a literal subset of the *row set* you'd get by reading
  every row.
- Process mix matches to 2 decimal places: 48.34% signal / 43.62% QCD /
  8.04% ttbar in both the 2M train split and the full 144M population — this
  is by design, not coincidence.

**Is the existing ~10M SPA-Net cohort an exact subset?** **Likely yes by the
same file-population logic**, though not independently row-verified in this
audit. `phase4AI_spanet_10M_scaling_seed0_20260821_v1`: 10,000,000 train
events, signal=4,833,956 (48.34%) / qcd=4,362,118 (43.62%) / ttbar=803,926
(8.04%) — identical process-mix percentages to the 2M cohort and to full144,
strongly suggesting the same governing-file, windowed-quota design at a
larger per-file quota. This is **distinct from** the BDT's own
`step2j_interim_10m_convergence_20260820_v1` cohort (10,474,025 rows,
8,377,425 train / 2,096,600 val), which is a strict **file-order prefix**
subset (first N files, `keep_fraction=1.0`) of the same governing list — a
different construction, same underlying governing population. Neither
10M cohort's row-level identity was cross-checked against
`PROVENANCE_FILE_ID_MANIFEST.json` in this audit; recommended as a first,
cheap follow-up (metadata-only join, no ROOT reads).

---

## Task 2 — constituent availability matrix

No new ROOT inspection was performed; this reuses two already-frozen
schema audits (`phase4J_development_data_contract_20260813_v1/TRACKB_CONSTITUENT_SCHEMA.tsv`,
`track_b_part_embedding_preflight_20260824_v1/TRACKB_CONSTITUENT_COMPATIBILITY.md`),
sourced from the actual ntuple-writer C++
(`makeNtuplesHH4bAllObjectsOptionalSel.C`). Those audits verified branch
content **only** on local dev signal (`gghh`) and local dev QCD samples —
**not** on the actual full144 buckets (H3VAR signal, IHEP QCD, IHEP ttbar)
and **not** on any of the 16 remote evaluation backgrounds (that manifest is
metadata-only; "no ROOT tree opened").

| Field | Local dev signal/QCD (verified) | Full144 QCD (`QCD_DelphesHH4JTrig`) | Full144 ttbar (`TTbar_ntuple`) | Full144 signal (H3VAR) | 16 remote backgrounds |
|---|---|---|---|---|---|
| Constituent 4-vectors | READY | DERIVABLE (same file family as ParT's own QCD source — `QCD_DelphesHH4JTrig_ntuple_mergeid166.root` is literally in both manifests) | UNKNOWN — not checked | UNKNOWN — not checked | UNKNOWN — not checked |
| PID/type | READY | DERIVABLE | UNKNOWN | UNKNOWN | UNKNOWN |
| Charge | READY | DERIVABLE | UNKNOWN | UNKNOWN | UNKNOWN |
| d0/dz + uncertainties | DERIVABLE (unit convention: resolved to **millimeters** per `track_b_dual_encoder_canary_20260824_v1`, cross-checked against the Delphes tracking-volume card) | DERIVABLE | UNKNOWN | UNKNOWN | UNKNOWN |
| Constituent→jet association | READY | DERIVABLE | UNKNOWN | UNKNOWN | UNKNOWN |
| Jet axis / per-jet deta,dphi | SEMANTIC_MISMATCH→DERIVABLE (stored `part_deta`/`part_dphi` are event-PF-sum-relative, not per-jet; per-jet recomputation is arithmetically trivial and was actually built+verified in the dual-encoder canary) | same | same | same | same |
| **ParT checkpoint↔jet-definition match** | **SEMANTIC_MISMATCH (uniform across all processes)** | — | — | — | — |

**Checkpoint-level SEMANTIC_MISMATCH, not process-specific:** the frozen
ParT checkpoint (`ParT_full.pt`,
sha256 `61e752f80d7c237d4b18b97705df416a8518dd9e3d5a78a8bbdeebadd787fec0`,
independently re-verified on disk this session) was trained on JetClass
large-R jets (R=0.8, pT 500–1000 GeV). Track-B applies it to AK4 jets
(R=0.4, pT>30 GeV). `PART_CHECKPOINT_INVENTORY.md` records this exact
checkpoint's Track-B verdict as a **carried-forward BLOCKED** finding,
later downgraded only to `REVIEW_NEEDED`/`DOMAIN_SHIFT_REQUIRES_VALIDATION`
— it has never been resolved to a PASS. This applies identically to every
process and every scale (2M, 10M, 144M); it is a scientific-validity
question, not a data-availability one, and it gates the whole study
independent of how far it is scaled.

**Bottom line:** engineering-level constituent availability is *plausible*
project-wide (same upstream `jetfree-hh4b` writer code, and the one file
family that was cross-checked — QCD — matches exactly), but only actually
**verified** for a small local dev slice. Before any full144 (or even 10M)
production, the cheapest closing action is a 1–2-file branch-listing peek on
one real H3VAR signal file and one real `TTbar_ntuple` file (not done here,
per this task's read-only/no-expensive-inference constraint) to turn
"DERIVABLE" into "READY" for the buckets that actually matter.

---

## Task 3 — exact population scale

| Quantity | Value | Basis |
|---|---:|---|
| Full144 events | **144,291,951** | exact, `FULL144M_FILE_MANIFEST.json` |
| Full144 real jets requiring ParT embeddings | **≈652.5M (point estimate)** — **not exact** | extrapolated |
| Governing train/val split (reusable) | **115,433,723 / 28,858,228** | exact, step2f |
| Expected embedding shards | **1,100** (one per governing source file, unchanged count) | structural, matches both `FULL144M_FILE_MANIFEST.json`'s shard order and the ParT package's own `SHARD_MANIFEST.json` |

**Real-jet estimate, and why it's better-grounded than a naive
extrapolation.** The exact 2.4M-event cohort (2.0M train + 0.4M val) has
**10,853,109** real (mask=1) jets — verified by direct `h5py` sum
(`SECTION_A_B_EXACT_COHORT_AND_INPUT_DIMENSION.md`), giving a measured ratio
of **4.5221 real jets/event**. Because this cohort is (Task 1) a
file-complete, process-proportional windowed subsample of the *identical*
governing population underlying full144 — not an independent or biased
sample — this ratio is a low-bias estimator, not a blind guess:

```
144,291,951 events × 4.52212875 jets/event ≈ 652,506,780 real jets
```

This is still labeled an **estimate**: jet multiplicity could differ
slightly between a file's sampled window and its full row set (pileup/HT
correlations across a file are not required to be flat), and no formal
error bar was computed in this audit.

**How to get the exact count without reading 144M raw events (recommended,
not executed here):** the already-staged, already-checksummed FULL144M
BDT feature arrays at
`/eos/uscms/store/user/iturkmen/.../step2k_full144m_staged_data_20260821_v1/`
(46.498 GiB, on FNAL EOS already) contain, per the K-feature contract itself,
a per-event `n_selected_jets` field (part of `K_WIDTH=52`) using the
**identical** candidate definition (`pt>30 GeV & |eta|<2.5`, top-10) that
defines "real jet" for the ParT/SPA-Net study. Summing that column requires
zero new XRootD/ROOT reads — it's a pure array reduction over data that is
already local-to-FNAL and already integrity-verified. This was **not**
executed in this audit (out of scope for a planning-only task) but is the
concrete next step for an exact count.

---

## Task 4 — storage / compute plan

All numbers below use **measured** figures only; where no measurement
exists (GPU throughput), that is stated explicitly rather than estimated,
per this task's instruction.

**Storage (128-d float32 embeddings, using the ≈652.5M-real-jet estimate):**

| | ParT-only (512 B/jet) | Both encoders as actually built — ParT + JP-JEPA Mini (1,024 B/jet) |
|---|---:|---:|
| Raw embedding bytes | ≈334.1 GB (≈311.2 GiB) | ≈668.2 GB (≈622.4 GiB) |
| + join-key overhead (provenance columns: `source_file_id`, `source_entry_index`, `jet_slot`, `process`; ~20–30 B/jet, matching the frozen package's own key schema) | + ≈13–20 GB | + ≈13–20 GB |
| **Realistic HDF5/EOS footprint** | **≈325–355 GiB** | **≈635–665 GiB** |

Cross-check: the frozen 2M package's own Section E storage estimate for its
exact 10,853,109-jet population is 10.35 GiB (both encoders); scaling by the
exact 60.12× event-count ratio (144,291,951 / 2,400,000) gives 622.4 GiB —
matches the direct full144 estimate above.

**Source I/O (raw ROOT input volume):** **not measured** in this audit.
Only flat feature-array staging size (46.5 GiB) is known; that excludes the
constituent-level branches ParT actually needs, which are far heavier per
event. Recommended next step: a metadata-only `xrdfs stat` sweep over the
1,100 governing files (no event reads) to get real file sizes.

**GPU inference time: cannot be stated — no measurement exists anywhere in
this project.** `README_EAF_LAUNCH.md`/`gpu_benchmark.py` were written but
never executed (`nvidia-smi` unavailable in every session that attempted a
benchmark; `torch.cuda.is_available()` → `False` every time). Per this
task's explicit instruction, no GPU number is invented here. **This is
itself a load-bearing prerequisite gap**, independent of the 2M/10M/144M
scale question — see Task 8.

**CPU throughput (measured, the only real throughput data that exists):**

| Measurement | Scale | Rate | Source |
|---|---|---:|---|
| Both encoders, combined, batch=512, 8 threads | 44,761 jets (production-scale pilot) | **43.09 jets/sec** | `SECTION_F_PRODUCTION_PILOT.md` (supersedes an earlier 100-jet extrapolation) |
| ParT_full only, isolated | 100 jets (small-N canary) | ≈215.1 jets/sec (4.65 ms/jet) | `PRIORITY_7_RUNTIME_STORAGE_BENCHMARK.md`, `track_b_dual_encoder_canary_20260824_v1` |

Single-CPU-process wall-time extrapolation to full144 (≈652.5M jets), for
context only — **not** a GPU estimate:

| Basis | Wall time (1 CPU process) |
|---|---|
| 43.09 jets/sec (both encoders, production-scale-measured) | ≈4,206 h ≈ 175 days |
| 215.1 jets/sec (ParT-only, small-N-measured) | ≈843 h ≈ 35 days |

Both require the project's established Condor per-file parallelization
pattern to be tractable; parallel wall time was not measured (Section F: "not
measured here, stated as a direction only").

**Scaling relative to the 2.4M-event production:** full144 is **60.12×**
the 2.4M-event cohort in events, real jets, and (by construction) storage.
Note that "the 2.4M-event production" has itself **not actually run** — see
Task 5.

---

## Task 5 — reuse, do not duplicate

**Critical framing correction:** the frozen "exact 2M cohort" ParT embedding
**production has not been executed**. `SHARD_MANIFEST.json`'s own bookkeeping
marks **every** shard, including `train_0000`, as `status: "pending"`; only
one shard (`train_0000.h5`, 2,720 of 2,400,000 events, 12,089 real jets,
6,971,840 bytes) exists on disk, produced as a CPU validation canary,
outside the tracked production state. Total package footprint on disk:
**15,358,844 bytes (~14.6 MiB)** — i.e., ~0.005% of the eventual ≈325–355 GiB
(ParT-only) full144 footprint. What is "frozen" today is a **validated,
byte-exact production plan and pipeline**, not completed embeddings.

**What is genuinely reusable right now:**
- The identity-key convention: `(source_file_id, source_entry_index,
  jet_slot)`, already shared and cross-consistent across the BDT population
  manifest, the SPA-Net-native HDF5 build, and the ParT shard manifest — no
  redesign needed.
- The governing 1,100-file / 80-20 split, already reused unmodified across
  the BDT (144M), the BDT-interim (10.47M, strict prefix), the SPA-Net-native
  10M, and the SPA-Net-native+ParT 2M cohorts.
- The one produced canary shard and the fully validated `extract_shard.py`
  pipeline (sha256 `395ace2915aec9708b3408fafd7731ab9eeae4d510ac319c4f4e60e299381f22`).

**What is missing, and should be added before scaling:** there is **no
single dedicated preprocessing-contract-SHA artifact** today — preprocessing
is anchored only informally, by `extract_shard.py`'s own file hash plus
narrative closure docs. A named contract file (hashing the exact feature
definition: candidate cuts, d0/dz unit convention, per-jet deta/dphi
recomputation, PID mapping) should be created and hash-pinned before any
larger production, so future shards can be validated against a single
version identifier rather than an implicit code hash.

**Parent/child manifest design (recommended, not built in this audit):**

```
PARENT  = FULL144M_FILE_MANIFEST.json (already exists)
          key: (source_file_id, source_entry_index)
          fields already present: process, remote_file, rank, n_events_file

CHILD   = part_embedding_manifest_v2.json (extends today's SHARD_MANIFEST.json)
          key: (source_file_id, source_entry_index, jet_slot)
          fields:
            source_file            (== parent remote_file / file_id)
            source_entry_index     (== parent join key)
            native_event_identity  (native_hdf5_row_start/end within shard)
            jet_slot                0-9, matches SPA-Net MASK ordering
            checkpoint_sha          61e752f8...787fec0  (ParT_full.pt)
            preprocessing_contract_sha   (NEW — to be created, see above)
            coverage_window         {root_offset, root_quota, wrapped}  — needed
                                     because existing/planned shards are WINDOWED
                                     subsamples, not full-file reads
            status                  pending | complete | verified
```

Because today's shards are **windowed** (not full-file) subsamples, a future
full144 (or 10M) pass is not simply "add the delta files" — it must also
decide whether to (a) widen the existing windows in place (risking
re-embedding already-embedded rows unless windows are constructed to nest),
or (b) treat any already-completed 2M shard's window as a fixed, permanently
excluded range and only embed the complement. Option (b) is the only one
that guarantees zero duplicate computation and should be the design target;
it requires the coverage_window field above to be authoritative and checked
before scheduling any new shard.

---

## Task 6 — full-stat SPA-Net requirements

**The scientifically fair ladder:**

| Rung | Native SPA-Net | SPA-Net+ParT | Status |
|---|---|---|---|
| 2M | **EXISTS** (`production_2M_train.h5`/`production_2M_val.h5`, frozen, sha256-verified) | **Pipeline validated, production not executed** (Task 5) | comparison not yet runnable — child side missing |
| 10M | **EXISTS, trained, adjudicated PASS** (`phase4AI_spanet_10M_scaling_seed0_20260821_v1`) | **Does not exist** — no ParT embedding work has been done at 10M scale | comparison not yet runnable — child side missing |
| full144 | **Does not exist** — explicitly disclaimed (`README.md:89`, "No 144M/full-statistics dataset built") | **Does not exist** | neither side exists |

**Why `full144 SPA+ParT` vs `native SPA 2M` alone cannot establish a
representation advantage:** it conflates two independent variables —
representation (native 7-feature jet kinematics vs. ParT constituent-level
embeddings) and statistics (2M vs. 144M events, a 60× difference). Any
measured performance delta could be entirely attributable to the 60× larger
training set and nothing about ParT. A valid causal claim requires holding
N fixed and varying only representation — i.e., the row-matched pairs in
the table above. Only after each matched pair is measured is it valid to
additionally ask how the representation effect changes with scale
(representation×scale interaction) — that is a legitimate second-order
question, but it needs the matched rungs first, not a diagonal shortcut.

**Does a native full-stat SPA-Net pipeline exist?** **No.** What would need
to be built before full144 ParT extraction is actually useful:
1. A `build_hdf5_144M.py` analog to the existing `build_hdf5_2M.py`/
   `build_hdf5_10M.py`, applying the identical 7-feature/10-slot/mask
   contract to the full144 governing population (reusing the exact same
   governing file lists and, ideally, the step2f 80/20 split rather than the
   BDT's pooled/no-split final-fit convention).
2. Chunked/streaming training-time data loading — a naive full-144M
   in-memory approach is not viable at any stage of this pipeline (Task 7).
3. A resolved decision on which val convention to use (reuse the governing
   28.86M-event val split, since the final BDT fit's `train_val_split=None`
   choice was specific to fixed-round-budget XGBoost training and doesn't
   transfer to a network that needs early-stopping/model-selection
   signal).
4. No existing full144 SPA-Net compute-time benchmark exists — this needs
   its own resource estimate once (1)–(3) exist, and should not be assumed
   proportional to BDT training time (different architecture, different
   hardware).

**Recommended concrete next step within this ladder:** produce SPA-Net+ParT
at 10M scale. The native 10M baseline already exists and is
adjudicated — this is the cheapest rung that (a) yields a real, matched
scientific comparison against an existing baseline, and (b) is exactly the
scale needed to retire the "no production-scale GPU/throughput measurement"
gap (Task 4) before committing to a 60×-larger full144 run.

---

## Task 7 — streaming architecture

The RAM finding needs a correction relative to this task's framing: the
"~10 GiB, not viable" figure in the prompt appears to conflate two different
numbers from `track_b_exact_cohort_scale_closure_20260824_v1`. The **10.35
GiB** figure (`SECTION_E_TRUE_RESOURCE_SCALE.md`) is a **disk storage**
estimate for the finished embeddings, explicitly called "small, entirely
unremarkable." The actual **RAM-not-viable** finding is separate
(`SECTION_F_PRODUCTION_PILOT.md`): a 44,761-jet in-memory pilot peaked at
**5.89 GB RSS**; naively scaling the same all-in-memory approach to the true
10,853,109-jet 2.4M-event population implies **on the order of 1+ TB of
RAM** — and that is only the 2.4M-event scale, before any full144
extension (which would be another ~60× on top of that). This is a hard
architectural constraint already recognized as such by the frozen preflight
documents, not a new finding of this audit.

**Recommended design** (not built in this audit; consistent with what the
frozen documents already recommend and with the project's existing
file-sharded structure):

- **Shard-local lazy reads.** Process and write results per source file
  (already how both `FULL144M_FILE_MANIFEST.json` and the ParT
  `SHARD_MANIFEST.json` are organized — 1,100 shards, one per governing
  file). No architectural change needed here, just executing what the
  manifests already imply.
- **Embedding sidecars, not a monolithic joined HDF5.** Keep ParT embeddings
  in their own per-shard HDF5 files, keyed by `(source_file_id,
  source_entry_index, jet_slot)`, entirely separate from the native
  SPA-Net kinematic HDF5. Never materialize a merged, multi-hundred-GB file.
- **Deterministic joins at read time**, using the identity key both systems
  already share — either via an explicit per-shard row-order guarantee
  (if native and ParT shards are built to iterate the same file in the same
  order, a positional join needs no lookup table) or a lightweight per-shard
  index if row order can't be guaranteed identical.
- **Training-time streaming, if SPA-Net's data loader permits it.** Whether
  SPA-Net's current training code can accept a pluggable/side-loaded feature
  source at `Dataset.__getitem__` time was **not verified** in this audit —
  this is the one place a small, bounded interface check (not a full build)
  is warranted before committing to a no-monolithic-file design: confirm
  SPA-Net's loader can open two shard files per batch and align rows by the
  shared key, rather than requiring one pre-merged input file. If it
  cannot, a per-shard "attach" step (chunked read + write, never a full
  in-memory join) is the fallback, not a monolithic merge.

---

## Task 8 — decision

**FULL144_PART_SPANET_FEASIBLE_WITH_GPU_THROUGHPUT_MEASUREMENT_AND_10M_VALIDATION_RUN**

This is not a hard structural block: the governing 144M population, its
identity-key convention, its file-level split, and its process composition
are all exactly known, immutable, and already shared — verified directly in
this audit, not merely assumed — by the BDT population, the SPA-Net-native
2M and 10M cohorts, and the ParT embedding shard plan. The earlier concern
that the ParT/SPA-Net study's "signal" was a different physical process
(`gghh_kl1`, the holdout process) than full144's training signal (H3VAR) was
checked directly against the frozen manifests and found to be **incorrect**
— both use H3VAR. Storage at full144 scale (≈325–665 GiB depending on
one- vs two-encoder output) is unremarkable for this project's existing EOS
footprint.

What actually gates progress, in priority order:
1. **The ParT checkpoint's own domain-mismatch verdict is still open**
   (`REVIEW_NEEDED`/`DOMAIN_SHIFT_REQUIRES_VALIDATION`, never resolved to
   PASS) — this gates the study's scientific validity at *any* scale,
   including 2M, and should not be deferred until after a large production
   run.
2. **No GPU throughput has ever been measured** for ParT inference in this
   project — a hard blocker for any credible full144 (or even 10M) compute
   budget, independent of everything else in this report.
3. **The "frozen 2M cohort" is a validated plan, not completed data** — 1 of
   1,100 shards exists. There is no production-scale experience yet to
   extrapolate from beyond a CPU-only pilot.
4. **No native full144 SPA-Net pipeline exists** — building one is a
   prerequisite for the fair-ladder comparison Task 6 requires, independent
   of ParT.
5. **The full144-scale streaming/embedding-sidecar architecture is designed
   here but not built or tested** — the existing shard structure is a good
   foundation, but the training-time dual-source read path has not been
   verified against SPA-Net's actual data loader.

**Recommendation: B — scale next to 10M.**

Not A: stopping at 2M would forgo a well-founded, already substantially
de-risked path — the file-population match, identity-key compatibility, and
manageable storage footprint are all more favorable than the task's framing
worried about, and a native 10M SPA-Net baseline already exists and is
adjudicated, ready to be paired with a ParT-embedded counterpart at no
additional infrastructure cost.

Not C: jumping straight to full144 would spend a ~60×-larger, effectively
unbudgeted compute request (no GPU number exists to budget against) on a
scale where the representation itself hasn't been validated as scientifically
sound at any scale yet, and where the necessary native full144 SPA-Net
pipeline doesn't exist. This would violate the requirement that scale
decisions rest on measured resource requirements, not "more data is
better."

B is the recommended path because producing SPA-Net+ParT at 10M scale
simultaneously (a) yields the first real matched-N scientific comparison
against an already-existing, already-adjudicated native baseline, (b)
produces the first production-scale GPU throughput measurement needed to
credibly plan full144's resource budget, and (c) forces resolution of the
checkpoint domain-mismatch question at a scale cheap enough to redo if the
answer is unfavorable — all before committing to the full144 infrastructure
build (native pipeline + streaming join + 1,100-shard full-row production)
that a "prepare full144 immediately" decision would require up front.
