# Harvey follow-up: FP32 score-precision audit of the u~6.6 / u~6.9 spikes

Status: DIAGNOSTIC ONLY -- no model trained, no frozen checkpoint modified,
no Stage-C/holdout_B file opened, no large inference job launched, no
interference with ParT production. `NEW_JOB_LAUNCHED = NO`.

Governing model referenced throughout: **frozen SPA-Net 10M, seed 0,
50-epoch baseline, primary checkpoint (epoch 47)**,
`sha256=fd9ea100825ecc0412a2666f01fd19d1a91b25a50109b958c307102fa6943de5`,
on disk at
`.../phase4AI_spanet_10M_scaling_seed0_20260821_v1/full_run_attempt2_eaf_local_staging/spanet_10M_seed0/version_0/checkpoints/primary-epoch=47-step=234336-validation_average_jet_accuracy=0.4928.ckpt`
(byte-identical to `last.ckpt` in the same directory; hash independently
re-verified here with `sha256sum`, matches the SHA256SUMS entry in
`docs/track_b/development_snapshot_20260821/spanet_10m_scaling/SHA256SUMS`
and the frozen contract at
`docs/analysis_contracts/SPANET_PART_RESOURCE_AWARE_COMPARISON.md.save`).

**Headline finding, stated up front:** the mechanism Harvey suspected is
real and is confirmed exactly by pure arithmetic (Task 2) and by real,
on-disk per-event float32 score data (Task 3) -- but the per-event
archive that survives for the *governing* 10M checkpoint contains only
summary statistics, not individual scores. The exact per-event census
below is therefore run on the real, frozen per-event archive of the
**2M seed-0 sibling checkpoint** (same architecture, same evaluation
code path, same 400,000-event cohort, AUC within 1-4e-4 of the 10M
governing checkpoint), used as the closest available faithful stand-in.
This substitution is disclosed everywhere it matters below; nothing
about the governing checkpoint's own per-event scores is asserted
beyond what its frozen summary JSON actually contains.

---

## Task 1 -- exact score computation and dtype (traced from real code, not assumed)

The actual scoring code was located outside this git worktree, on disk
at `/uscms_data/d3/iturkmen/hh4b_delphes/track_b_phase4_preflight_20260812/`
(it is *not* present anywhere inside the `hh-bbww-baselines*` git
repositories -- only its SHA256 and JSON output were copied into the
frozen docs snapshot at
`docs/track_b/development_snapshot_20260821/spanet_10m_scaling/SHA256SUMS`).
Two files matter:

- `phase4AI_spanet_10M_scaling_seed0_20260821_v1/classification_evaluation_10M/evaluate_classification_10M.py`
  (the governing-checkpoint evaluation; hash `715c1c34...` matches SHA256SUMS)
- `phase4AF_spanet_partial_events_population_correction_20260819_v1/spanet_2M_seed0_classification_evaluation_v1/evaluate_classification.py`
  (the 2M sibling evaluation; `evaluate_classification_10M.py` line 40
  does `import evaluate_classification as base` and reuses its functions
  **verbatim, unmodified** -- so both checkpoints are scored by the
  identical code path)

Traced facts, by line number, in `evaluate_classification.py`:

| Question | Answer | Evidence |
|---|---|---|
| Training precision | `precision="32-true"` (true FP32, no AMP) | `launch_10M_seed0_full_50epoch_attempt2.py:513` (2M launcher diffed byte-identical per the project's own diff-proof) |
| Any `.half()`/`.float16`/`autocast` anywhere in the eval path | **No** -- grepped both files, zero hits (only unrelated `dtype=torch.int64`/`dtype=float` on integer assignment targets/weights, line 107/124) | direct `grep` of both files |
| Forward pass | CPU only (`options.num_gpu=0`, line 267), `model.eval()` + `torch.no_grad()` (lines 293/305) | `score_full_population()`, lines 283-322 |
| Classification logits | `outputs.classifications["EVENT/signal"]`, shape `[B, 2]`, **float32** (model loaded from an FP32 checkpoint via `torch.load(..., map_location="cpu")` with no dtype cast, line 275) | line 308 |
| Softmax call | `torch.softmax(logits, dim=-1)[:, 1]` -- **float32 in, float32 out** (no `.double()`, no dtype arg) | line 311 |
| Stored classifier score dtype | **float32** -- `probs.numpy()` on a float32 torch tensor yields a numpy `float32` array (line 312); concatenated with `np.concatenate` which preserves dtype (line 319) | line 312, 319; independently re-verified below (`probs.dtype == float32`) |
| Output-file dtype | **Two different answers depending on checkpoint:** <br>-- 2M sibling: `np.save(..., probs)` at line 556 writes the **raw float32 array unchanged** to `val_signal_probs_400k.npy`. <br>-- **Governing 10M checkpoint: no per-event array is ever written.** `evaluate_classification_10M.py` has no `np.save` call at all; it only calls `json.dump(result, ...)` (lines 165-166) on a `dict` built from `base.describe()`, whose `mean/std/min/max/pXX` fields are each wrapped in Python `float()` (`describe()`, lines 325-336) -- i.e. float64 **summary statistics**, not per-event data. | `evaluate_classification.py:555-556`; `evaluate_classification_10M.py:163-166` |
| Cast to float32 before/after softmax | Before: input logits already float32 (no upcast to float64 anywhere in the forward/softmax path). After: `.numpy()` preserves float32; the *only* place a cast to float64 happens is inside `describe()`'s Python `float()` wrapper, applied only to aggregate statistics (mean, std, min, max, percentiles), never to individual per-event scores. | as above |
| Was the plotted `u=-log10(1-score)` computed from the stored float32 probability | **For the governing 10M checkpoint: cannot be determined from any frozen artifact**, because no per-event score array for it exists anywhere found on disk (confirmed by an exhaustive `find` across the entire `hh4b_delphes` work tree, the git repo, and EOS -- see Task 3). Wherever Harvey's u-histogram spike plot was actually produced, it was **not** built from a script or archive currently discoverable in this project's frozen record for the 10M checkpoint. It **is** directly reproducible from the 2M sibling's `val_signal_probs_400k.npy`, and doing so (Task 3, below) reproduces spikes at exactly the reported locations. | exhaustive search log below |

**Exhaustive search performed** (not assumed absent): `find` across
every `hh-bbww-baselines*` git worktree, `docs/checkpoints/`,
`docs/paper/`, `docs/track_b/`, `/eos/uscms/store/user/iturkmen/`, and
the full `/uscms_data/d3/iturkmen/hh4b_delphes/` tree for
`*classification_evaluation_10M*`, `evaluate_classification*.py`, and
any `*.npz/.npy/.parquet/.csv` file with "score"/"prob" in its name.
Result: exactly one pair of scripts and one `work/` directory per
checkpoint (2M, 10M), both located above; the 10M `work/` directory
contains only the two JSON summary files, confirmed by direct `ls`.
`docs/paper/jhep_hh4b_ml/track_b_physical_normalization/SOURCE_PROVENANCE.md`
independently and explicitly states: "No event-level data is included
in this archive... any large per-event score array" -- consistent with
the code-level finding.

---

## Task 2 -- FP32 lattice near score=1 (pure arithmetic, no data needed)

For normalized float32 values in `[0.5, 1)` the exponent is fixed at
`-1`, so the unit-in-last-place is exactly `2^-24`. Representable
values immediately below 1.0 are `score_k = 1 - k * 2^-24` for integer
`k = 1, 2, 3, ...`, and `u_k = -log10(k * 2^-24)`.
Full table (k=1..32): [`FP32_LATTICE_TABLE.csv`](FP32_LATTICE_TABLE.csv),
generated by [`scripts/build_fp32_lattice_table.py`](scripts/build_fp32_lattice_table.py)
(pure `numpy`/`struct` arithmetic; no model, no checkpoint, no data file
touched).

| k | 1-score | u_k = -log10(k*2^-24) |
|--:|---:|---:|
| 1 | 2^-24 = 5.9604645e-08 | **7.224720** |
| **2** | 2*2^-24 = 1.1920929e-07 | **6.923690** |
| 3 | 3*2^-24 = 1.7881393e-07 | 6.747599 |
| **4** | 4*2^-24 = 2.3841858e-07 | **6.622660** |
| 5 | 5*2^-24 = 2.9802322e-07 | 6.525750 |
| 8 | 8*2^-24 = 4.7683716e-07 | 6.321630 |
| 16 | 16*2^-24 = 9.5367432e-07 | 6.020600 |
| 32 | 32*2^-24 = 1.9073486e-06 | 5.719570 |

**Match to Harvey's reported spike locations:**

- **u ~ 6.9** -> nearest lattice point **k=2, u_2 = 6.92369**. Difference
  from "6.9" is 0.024 -- well inside any plausible histogram bin width
  (typical bin widths 0.02-0.1 in this range).
- **u ~ 6.6** -> nearest lattice point **k=4, u_4 = 6.62266**. Difference
  from "6.6" is 0.023.

This is an exact, parameter-free arithmetic identity -- it holds
regardless of which SPA-Net checkpoint, training run, or physics
process is involved. It is evidence for *where a pile-up could land*,
not yet evidence that a pile-up actually exists at those points in real
data. Task 3 supplies that with real, on-disk data.

---

## Task 3 -- exact unique-value census (real, frozen, on-disk per-event data)

**Data used:** `val_signal_probs_400k.npy` (400,000 float32 values) and
`val_process_labels_400k.npy` (400,000 process-label strings), the
**real**, already-existing, read-only output of the SPA-Net **2M**
seed-0 evaluation (checkpoint sha256
`dc39cf76f0d8e40d07228f240fe58b1179e8134cd4f96c5c786a7d528d31ea8d`),
at `phase4AF_.../spanet_2M_seed0_classification_evaluation_v1/work/`.
**This is not the governing 10M checkpoint** -- see Task 1's finding
that no per-event array exists for it. Nothing was recomputed, rerun,
or re-inferred to produce this census; the two `.npy` files were only
loaded and counted. Script:
[`scripts/census_fp32_scores.py`](scripts/census_fp32_scores.py). Full
output: [`FP32_UNIQUE_SCORE_CENSUS.csv`](FP32_UNIQUE_SCORE_CENSUS.csv)
(27 rows, one per unique float32 value found at u>5.5), machine-readable
summary: `fp32_census_summary.json`.

Cohort composition (identical to the governing 10M evaluation's own
cohort, since both share the same `production_2M_val.h5`): n_signal =
193,358; n_qcd = 174,485; n_ttbar = 32,157; n_all_background = 206,642.
There is no "other background" bucket in this cohort -- only QCD and
ttbar exist as background processes (`other_background` row below is
included per Harvey's request and is correctly empty by construction,
not an omission).

### Exact counts, u > 5.5

| Population | n_total | n(u>5.5) | n unique float32 values (u>5.5) | n(score==1.0 exactly) | n(score==nextafter(1,0)) |
|---|---:|---:|---:|---:|---:|
| signal | 193,358 | **1,256** | 27 | **128** | 0 |
| all_background | 206,642 | **0** | 0 | 0 | 0 |
| qcd | 174,485 | **0** | 0 | 0 | 0 |
| ttbar | 32,157 | **0** | 0 | 0 | 0 |
| other_background | 0 | 0 | 0 | 0 | 0 |

`nextafter(float32(1.0), 0.0) = 0.99999994039535522` (k=1) has **zero**
occurrences in any population -- see the "only even k" observation
below.

### The two spikes Harvey asked about, exact counts at the matching lattice points

| Lattice point | u_k | signal count | all-background count | QCD count | ttbar count |
|---|---:|---:|---:|---:|---:|
| k=2 (nearest to "u~6.9") | 6.923690 | **80** | 0 | 0 | 0 |
| k=4 (nearest to "u~6.6") | 6.622660 | **53** | 0 | 0 | 0 |

Full row-by-row detail (score, exact decimal, float32 hex, k, u, raw
count) for all 27 unique tail values is in
[`FP32_UNIQUE_SCORE_CENSUS.csv`](FP32_UNIQUE_SCORE_CENSUS.csv); the
first several rows:

```
population,score_f32,score_f32_exact_decimal,score_f32_hex_be,k_round,u,raw_count
signal,1.0,1,3f800000,0,inf,128
signal,0.9999998807907104,0.99999988079071044922,3f7ffffe,2,6.923689900271567,80
signal,0.9999997615814209,0.99999976158142089844,3f7ffffc,4,6.622659904607587,53
signal,0.9999996423721313,0.99999964237213134766,3f7ffffa,6,6.446568645551905,35
signal,0.9999995231628418,0.99999952316284179688,3f7ffff8,8,6.321629908943605,39
```

**Weighted background yield:** not computed -- explicitly N/A. This
development-cohort per-event archive carries no per-event
cross-section/luminosity weight field (raw event counts only); a
weighted yield exists only in the separate aggregate physical-
normalization freeze
(`docs/paper/jhep_hh4b_ml/track_b_physical_normalization/CANONICAL_FOUR_MODEL_NORMALIZED_RESULTS.csv`),
which carries no per-event scores and cannot be joined to this census.
Not fabricated. In any case the exact background count at both spike
locations is **0**, so a weighted background yield at these two points
is trivially 0 regardless of the per-event weight.

**Observed pattern, disclosed but not over-interpreted:** all 27 unique
values (and both exact 1.0 and every k found) have **even** `k` only
(2,4,6,8,...,52) -- no odd-`k` value (nextafter(1,0), 3*2^-24,
5*2^-24, ...) appears anywhere in this 400,000-event cohort. This is
consistent with `torch.softmax`'s internal division-and-round-to-
nearest-even arithmetic for a 2-class input systematically favoring
even mantissa bit patterns near 1.0; it has **not** been independently
verified bit-for-bit here (would require instrumenting the CPU softmax
kernel itself, out of scope for this diagnostic) and is reported as an
observation, not a proven mechanism.

### Answering Harvey's exact question

> ARE THERE HUNDREDS OF EVENTS IN THE SPIKES AROUND u=6.6 AND u=6.9?

**No, not hundreds -- tens.** In the real per-event archive available
(2M sibling checkpoint): **53 events** at the u~6.6 lattice point
(k=4) and **80 events** at the u~6.9 lattice point (k=2), **all in the
signal population, zero background events in this 400,000-event
development cohort**. The single largest pile-up bin in this tail is
the exact-1.0 bin itself, at **128 signal events** -- also short of
"hundreds," and also entirely signal, zero background events in this
cohort. Summed over the entire u>5.5 tail (all 27 unique values plus
the exact-1.0 bin), there are 1,256 signal events and 0 background
events **in this development cohort**. This is a statement about this
specific 400,000-event archive, not a physical background-yield
determination -- it does not replace the separate, much larger
full physical-background tail study (which pools the full QCD/ttbar/
minor-background statistics and can and does contain rare survivors at
tight working points; see e.g. the eps_S=0.10/0.20 finite-support
caveats already documented for this same checkpoint family in
`COMPARISON_TABLE_400K_DEVELOPMENT.md`). If Harvey's original histogram
showed background events in these bins, that is inconsistent with this
real per-event archive and would need to be re-examined against
whatever produced that plot (which, per Task 1, could not be located
for the governing 10M checkpoint).

Figure: [`figures/u_probability_fp32_tail.png`](figures/u_probability_fp32_tail.png)
(fine-binned histogram of the real tail, k-lattice grid lines overlaid;
the two spikes land exactly on the k=2/k=4 predicted lines).

---

## Task 4 -- do raw logits already exist?

**No.** Searched every frozen inference/output artifact reachable for
`z0`, `z1`, `classification_logits`, "logit margin", or any pre-softmax
classifier output:

- `evaluate_classification.py`'s `score_full_population()` computes
  `logits` locally (line 308) and immediately reduces it to `probs`
  (line 311) -- `logits` itself is never returned, printed, or saved.
  Only `probs` and `true` leave the function (line 322).
- `evaluate_classification_10M.py` follows the identical pattern via
  the imported function; its own JSON output additionally only stores
  aggregate `roc_auc`/`score_distributions`/`working_points`, never
  logits.
- No `.pt`/`.npz`/`.npy` file anywhere in the searched trees is named
  or contains anything resembling `z0`/`z1`/logit arrays for either
  checkpoint.

`RAW_LOGITS_AVAILABLE = NO`.

Since no logit archive exists, `u_logit`, the `u_prob32` vs. `u_logit`
scatter, the `u_logit` histogram, and the float64
`sigmoid(Delta)` diagnostic **cannot be produced from existing frozen
artifacts** -- producing them requires new inference (Task 5). Per
instruction, no such inference was run; `figures/u_logit_tail.png` and
`figures/u_prob_vs_u_logit.png` are correctly **absent** from this
package rather than fabricated. Only the real, existing
`figures/u_probability_fp32_tail.png` (Task 3, probability-only) is
included.

---

## Task 5 -- tail-only rerun plan (prepared, NOT launched)

Because both the 2M and 10M evaluations score the **identical**
`production_2M_val.h5` file in the **identical** row order (verified:
`evaluate_classification_10M.py` inherits `base.HDF5_VAL` unchanged --
it overrides only `PRIMARY_CKPT`/`SECONDARY_CKPT`), the row index of
every event in the 2M archive is a valid, exact event identity in the
10M archive too. This lets the rerun be restricted to exactly the
events already implicated by the real 2M tail census above, instead of
rescoring all 400,000 events.

**Event identities (real, extracted from the frozen 2M archive, not
invented):** [`TAIL_ONLY_RERUN_EVENT_IDENTITIES.csv`](TAIL_ONLY_RERUN_EVENT_IDENTITIES.csv)
-- **1,256 row indices** (all falling in the signal block of the
cohort, `174,485 <= row_index < 367,843`, consistent with the
qcd/signal/ttbar concatenation order `evaluate_classification.py`
documents), each with its 2M reference score and `u_2M`.

**Prepared script (guarded, refuses to execute its `main()` without a
manual code edit):**
[`scripts/tail_only_rerun_logits_PREPARED_NOT_LAUNCHED.py`](scripts/tail_only_rerun_logits_PREPARED_NOT_LAUNCHED.py).
It would:

1. Re-verify the governing checkpoint's SHA256 before touching it.
2. Build the identical `model`/`validation_dataset` the existing
   evaluation scripts build (`base.build_model_and_dataset()`,
   unmodified import, no duplication).
3. Load the governing 10M primary checkpoint (`strict=True`, same as
   production).
4. Build a `torch.utils.data.Subset` over **only the 1,256 listed row
   indices** (not all 400,000) and a small `DataLoader` over that
   subset.
5. Forward-pass in `torch.no_grad()` (no gradient, no optimizer, no
   training) and, per event, record: `z0`, `z1`,
   `Delta = z1 - z0` (signal minus background, both float64-cast for
   the record), the production `softmax` probability (float32, same
   code path as today), `sigmoid(Delta)` recomputed independently in
   float64, and the event identity (row index + process label + 2M
   reference score).
6. Write everything to `work_tail_rerun/tail_only_rerun_result.json`.

**Exact command that would launch it** (not run by this session):

```bash
# On the host with the pinned pytorch_lightning/spanet environment
# (the same one evaluate_classification_10M.py itself requires --
# this LPC session's python3 has no `torch`/`spanet` module, confirmed
# directly: ModuleNotFoundError: No module named 'torch').
cd docs/checkpoints/track_b_harvey_fp32_score_precision_20260906_v1/scripts
python3 -c "
import tail_only_rerun_logits_PREPARED_NOT_LAUNCHED as t
t.main()
"
```

**Number of events that would need rerunning:** 1,256 (0.314% of the
400,000-event cohort) -- not the full population.

**Estimated cost:** the existing, already-completed 2M evaluation
scored the full 400,000-event cohort, CPU-only, in 110.0 s wall time
(`classification_forward_pass_wall_s` in
`.../spanet_2M_seed0_classification_evaluation_v1/work/classification_evaluation_result.json`),
i.e. ~3,636 events/s single-process CPU throughput. At that rate the
forward pass itself for 1,256 events is ~0.35 s; total wall time is
dominated by fixed overhead (model construction, checkpoint
deserialization, HDF5 open), which the same evaluation script already
demonstrates takes well under two minutes end-to-end. This is a
CPU-only job (`options.num_gpu=0`) -- **no GPU/EAF allocation is
required**, only the pinned Python environment. This is far below the
threshold of "a large inference job."

`NEW_JOB_LAUNCHED = NO` -- nothing above was executed by this session.

---

## Task 6 -- output quantization vs. model precision vs. model pathology

To be precise, as instructed:

- **What Tasks 2-3 establish:** a pile-up of stored probabilities at a
  small number of discrete float32 values immediately below 1.0 is
  exactly what IEEE-754 float32 rounding produces whenever many
  distinct, very-high-confidence real-valued softmax outputs are
  quantized onto a `2^-24`-spaced lattice near 1. This **is** evidence
  of finite-precision quantization of the **post-softmax
  representation** -- confirmed both arithmetically (Task 2) and with
  real per-event data landing exactly on the predicted lattice points
  (Task 3).
- **What it does NOT establish, per explicit instruction:**
  - It is **not** evidence the classifier "learned something
    pathological." A very confident, well-separated classifier is
    expected to produce many events with true probability
    indistinguishable from 1 at float32 resolution; this is a
    predictable consequence of high discrimination power (both
    checkpoints have AUC ~0.968-0.978 against every background
    definition), not a symptom of a broken model.
  - It is **not** evidence that "FP32 training is invalid." Training
    was explicitly `precision="32-true"` (Task 1); the quantization
    observed here is a property of representing a probability in
    32 bits at inference/storage time, unrelated to whether FP32 is an
    adequate training precision.
  - It is **not**, by itself, evidence that "the underlying logit
    ordering is lost before softmax" -- softmax followed by float32
    rounding can map many different (but all very large) logit margins
    onto the same rounded probability while the underlying logit
    values themselves remain perfectly ordered and smoothly
    distributed. **This specific claim cannot be checked from any
    currently frozen artifact** (Task 4: no logits are stored for
    either checkpoint) and is exactly what Task 5's prepared (not
    launched) rerun is designed to resolve.
- **Do not call the model calibrated.** Nothing in this diagnostic
  measures calibration (agreement between stated probability and true
  frequency); AUC and float32 saturation are both silent on
  calibration, and no calibration claim is made anywhere in this
  report.

`LOGIT_BASED_TAIL_RESOLVED = NOT_YET` -- resolving whether the logit
margin stays smooth through the FP32 probability spikes requires the
Task 5 rerun, which was prepared but intentionally not launched in
this diagnostic-only pass.

---

## Files in this package

- `HARVEY_FP32_SCORE_PRECISION_REPORT.md` -- this file
- `HARVEY_FP32_TWO_PARAGRAPH_SUMMARY.md` -- condensed summary
- `FP32_LATTICE_TABLE.csv` -- Task 2 output (k=1..32)
- `FP32_UNIQUE_SCORE_CENSUS.csv` -- Task 3 output (27 rows, u>5.5, real data)
- `fp32_census_summary.json` -- Task 3 machine-readable summary incl. spike-focus counts
- `TAIL_ONLY_RERUN_EVENT_IDENTITIES.csv` -- Task 5 input (1,256 real row indices)
- `figures/u_probability_fp32_tail.png` -- Task 3 figure
- `scripts/build_fp32_lattice_table.py`, `scripts/census_fp32_scores.py`,
  `scripts/build_figures.py` -- executed, read-only/pure-arithmetic scripts
- `scripts/tail_only_rerun_logits_PREPARED_NOT_LAUNCHED.py` -- Task 5, guarded, NOT run
- `receipt.json`, `SHA256SUMS`

## Explicitly not done

No model trained. No frozen checkpoint (2M or 10M) opened for writing
or modified in any way -- only `sha256sum`-verified read access to the
governing checkpoint's bytes (to confirm identity) and ordinary
`torch.load(..., map_location="cpu")` reads inside the *prepared, not
executed* Task 5 script. No Stage-C/holdout_B file opened (this entire
diagnostic uses only the frozen 400,000-event development cohort). No
inference job launched (the only "new" numbers in this package are
pure arithmetic on a closed-form lattice, and counts/censuses over two
pre-existing `.npy` files that were already on disk before this task
began). No ParT production artifact touched.
