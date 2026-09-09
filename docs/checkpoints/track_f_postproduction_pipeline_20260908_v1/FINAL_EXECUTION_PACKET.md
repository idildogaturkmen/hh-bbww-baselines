# FINAL EXECUTION PACKET — postproduction → join → preprocessing → SPA2M+ParT training

**Built:** 2026-09-09, this session, by re-auditing every script in this
package for actual executability (not just reading docstrings) and by
dry-running every step that can be dry-run without touching the live
exact_2M production or launching training. **The live EAF worker1
process was never inspected, touched, or interfered with.** No join, no
preprocessing, no SPA-Net training, no additional-8M/10M ParT production,
no JP-JEPA, no holdout_B, no Stage C was run this session.

**Verdict: `READY_TO_EXECUTE_AFTER_880_220`**, with exactly one concrete,
non-production-blocking gap (Step 9, matched evaluation — see below). Steps
1–8 (everything through both training launches) have no known blocker
beyond production reaching 880/880 train + 220/220 val.

---

## 0. What was actually verified this session (not just read)

| # | Finding | How verified |
|---|---|---|
| F1 | `verify_shard_checksums.py`'s core hash-check logic works end-to-end against **real** production data | Ran `check_one_shard()` directly against `train_0000` on EOS: receipt+H5 fetched via `xrdcp`, hashes matched, `pass=true` |
| F2 | `aggregate_embedding_audit.py`'s window-disjointness proof + H5-schema checks (`JOIN_KEY/native_hdf5_row_index`, `JOIN_KEY/jet_slot`, `EMBEDDINGS/part_full` shape `(n,128)` float32) work against real shards | Ran with `--partial-ok --max-shards 3` against real `train_0000..0002`: `AGGREGATE_PASS=true`, all schema/window checks passed |
| F3 | Every hardcoded SHA256 in every script (ParT checkpoint, native train/val H5, manifest, native-control checkpoint) matches the actual file on disk | Independently re-hashed all of them this session; all match exactly |
| F4 | `SPANET_REPO` pinned-commit + clean-working-tree preflight checks in `launch_spa2m_part.py` pass right now | `git rev-parse HEAD` = `debbdc999...`, `git status --porcelain` empty |
| F5 | `launch_spa2m_part.py` reproduces `launch_2M_seed0_training.py`'s hyperparameters **exactly** (seed, batch size, epochs, architecture dict, optimizer dict, loss scales, balance flags, `partial_events`, `dataset_limit`, dataloader workers, trainer construction args, checkpoint monitor/mode/save policy, the weight-loading monkeypatch, the secondary-val-loss subclass) except the two documented, intentional differences (HDF5 paths, event YAML) | Line-by-line diff against the real native launcher (`launch_2M_seed0_training.py`), not the docstring's claim |
| F6 | ParT is genuinely frozen/precomputed in the training path | `launch_spa2m_part.py` never loads a ParT model or runs a ParT forward pass — it only ever reads pre-built, standardized static embedding columns from the augmented H5 |
| F7 | The join is identity-based and fail-closed | Read `streaming_native_part_join.py`: asserts every `(row,slot)` falls in its own shard's declared window, asserts no within-shard duplicate keys, hard-fails if a ParT embedding lands on a native-MASK "not real" slot; writes into the output at the *identity-derived* `[row_lo:row_hi)` slice, never by list order. **Then proved this executes correctly** by building a synthetic 12-event/2-shard population with the exact real schema and running the join for real — 100% coverage, correct output shape `(12,10,135)` |
| F8 | `verify_join_coverage.py`'s 100%-coverage / re-opened-mask / `mask⇒covered` gate works | Ran for real against the synthetic joined output: all 10 checks `true`, `OVERALL_JOIN_VERIFICATION_PASS=true` |
| F9 | Train-only statistics cannot leak validation | `compute_train_embedding_stats.py` takes only `--joined-train-h5` — it never opens, references, or accepts a val path (plus a filename guard). `build_augmented_input.py` applies the **same** frozen train-derived means/stds unchanged to both `--split train` and `--split val` — there is no separate val-stats code path anywhere in the package. Structurally impossible to leak, not just "shouldn't happen." |
| F10 | The active-dimension threshold logic is numerically correct | Built a synthetic population with a known ground truth (30 near-constant dims, 98 real-variance dims). `compute_train_embedding_stats.py`'s data-driven gap threshold (0.00105) and its `1e-3` threshold scan both recovered **exactly** the true 98 active dims (`active_dims == list(range(30,128))` → `True`) |
| F11 | `build_augmented_input.py` produces correct output for both variants | Ran both `--variant active` (→ 105-wide = 7+98) and `--variant all128` (→135-wide) on the synthetic joined H5: correct shapes, `all_finite=true`, masked slots exactly `0.0`, standardized real-jet columns z-scored correctly, `TARGETS`/`CLASSIFICATIONS` copied verbatim from the native H5, named `(N,10)` per-key datasets matching the real `trackb_hh4b.yaml` template's `INPUTS/SEQUENTIAL/Source` structure |
| F12 | `bootstrap_utils.py`'s `roc_auc`/`paired_bootstrap_delta`/`mcnemar_test`/`exact_poisson_count_interval` run correctly | Smoke-tested all four with synthetic arrays; `exact_poisson_count_interval` correctly fell back to its documented normal-approximation path (see gap G3 below) |
| F13 | Native H5 checksums hardcoded in `run_full_join.sh` are still correct | Re-hashed both `production_2M_{train,val}.h5` (676 MB / 135 MB) — exact match |
| F14 | All 8 Python scripts `py_compile` clean; all 3 shell scripts `bash -n` clean | Ran this session, including the new wrapper (§0, gap G4) |

## Gaps / caveats found (none block Steps 1–8; G3 doesn't block anything; only G5 blocks Step 9)

- **G1 — "routine" postflight is not actually fast.** `run_postflight_gate.sh`'s Step 2 (`aggregate_embedding_audit.py`) is invoked **without** `--partial-ok`/`--max-shards`, so it unconditionally re-downloads, re-hashes, and re-opens **every one of the 1,100 manifest shards** (≈7–8 GB, ~2,200 `xrdcp` calls) regardless of the `CHECKSUM_SAMPLE_N` setting that only affects Step 4. `POSTFLIGHT_RUNBOOK.md`'s description of the no-arg call as "fast" is stale/misleading — both the "routine" and "`CHECKSUM_SAMPLE_N=full`" invocations pay this same full-corpus cost via Step 2; they only differ in Step 4's own sample size. Budget **30–90+ minutes** for either call, not "fast." Not a defect — the design intentionally makes Step 2 a full completion proof — just a documentation correction.
- **G2 — `JOIN_AND_PREPROCESS_RUNBOOK.md` cites a `receipt.json`'s `dry_run_verification` section that does not exist anywhere in this package.** The prior session's synthetic-H5 dry-run evidence was apparently never written to disk (or was lost). **This session regenerated that evidence for real** (F7–F11 above); the runbook's narrative claim is now actually substantiated, but the missing file itself was not recreated (no script in this package writes one — the claim lived only in prose). Not a blocker, but worth knowing the original citation was dangling.
- **G3 — `scipy` is broken in this LPC shell's default `/usr/bin/python3`** (`numpy 1.23.5` from `/usr/lib64` is missing a working `numpy._typing`, and the user-local `scipy 1.13.1` needs it → `ModuleNotFoundError` on `from scipy import stats`). `bootstrap_utils.py`'s `exact_poisson_count_interval()` already degrades gracefully to its documented normal-approximation fallback in this environment (confirmed, F12) — not fatal, nothing else in this package imports scipy. Only matters for Step 9: if you want the *exact* Garwood CI on sparse-tail rejection working points (not the fallback), run that step in an environment with working scipy/numpy, or fix this one first (e.g. `pip install --user -U numpy scipy` in a scratch venv).
- **G4 — No EAF GPU wrapper existed for `launch_spa2m_part.py`** (unlike the native launcher, which has `run_2M_seed0_training.sh`). **Built this session**: `scripts/train/run_spa2m_part_training.sh`, a direct generalization of the native wrapper's proven pixi-env resolution and `nvidia-smi -L` GPU/MIG discovery-and-validation logic (never a stale UUID, never a placeholder, refuses to guess among multiple visible devices), parameterized for `--variant`. `bash -n` clean; **not run** (no GPU on this node).
- **G5 (the one real blocker, Step 9 only) — no inference/scoring script exists anywhere in the project that produces the `.npz` files `evaluate_matched_2m.py` requires.** That script's own docstring says so explicitly ("this script does not launch inference"). The closest precedent, `evaluate_classification.py` (native-control classification eval, already run and on disk), only extracts the classification signal score + reconstructed process label via a CPU forward pass — it does **not** extract per-event jet-pair assignment correctness, reconstructed Higgs masses, or `event_id`, all of which `evaluate_matched_2m.py`'s documented input schema requires for both CONTROL and TEST. **A new script must be written** (reusing `evaluate_classification.py`'s CPU-forward-pass / `DataLoader(drop_last=False)` / process-label-reconstruction pattern as a starting point, extended to also pull `outputs.assignments`, compute `higgs_mass_{1,2}` from the predicted pairing, and tag `event_id=native_hdf5_row_index`) before Step 9 can actually run. This does not block Steps 1–8; there is naturally time to write it while training runs.
- **Minor/cosmetic:** `NATIVE_CONTROL_CHECKPOINT_SHA256` is defined in `launch_spa2m_part.py` but never referenced anywhere in that file (dead constant; harmless — the real check happens implicitly since the CONTROL checkpoint itself is never touched by this launcher). Also: `launch_spa2m_part.py` does not replicate the native launcher's post-fit checkpoint-reload/`state_dict`-key integrity check (`reload_ok`, `missing_keys`, `unexpected_keys`) — a reduced-rigor omission, not a hyperparameter difference, and not required by the "reproduce hyperparameters exactly" instruction; flagging for awareness only.
- **Resources:** disk is not a concern (17 TB free on `/uscms_data/d3`; the full pipeline through both training-input builds needs on the order of 45–50 GB of scratch: ~10 GB local shard sync + ~13 GB joined H5s + ~23 GB for both augmented-input variants × both splits). RAM: `compute_train_embedding_stats.py` peaks at **~4.6–6 GB RSS** at full 2M-train scale (the single `(9,044,239, 128)` float32 real-jet array is the dominant allocation) — this LPC worktree shell currently reports only ~8.5 GB free; run Step 5 on a node/batch slot with more comfortable headroom, not a busy shared interactive shell. Training itself needs the EAF GPU/pixi environment (`torch`/`pytorch_lightning`/the pinned `spanet` clone are **not** installed in this shell's system `python3` at all — expected, training was never meant to run here).

---

## Ordered execution commands

All paths below are absolute. `PKG` is this package's root. Run Steps 1–3
from any LPC/EAF login node with `xrdfs`/`xrdcp`/Kerberos+X.509 access.
Run Steps 4–6 from a node with real disk/RAM headroom (not necessarily
GPU). Run Steps 7–8 **only** from an EAF GPU session. Step 9 is currently
blocked on G5 above.

```bash
PKG=/uscms_data/d3/iturkmen/hh4b_delphes/track_f_postproduction_pipeline_20260908_v1
```

### Step 1 — Final census (cheap, repeat as often as you like while waiting)

```bash
CENSUS_SCRIPT=/uscms_data/d3/iturkmen/hh4b_delphes/track_b_part_postflight_2m_20260907_v1/census/census_2m.py
EOS_ROOT=root://cmseos.fnal.gov//store/user/iturkmen/hh4b_delphes/track_b_part_full_embedding_production_exact_2M_20260824_v1/shards
CKPT_SHA256=61e752f80d7c237d4b18b97705df416a8518dd9e3d5a78a8bbdeebadd787fec0

TS=$(date -u +%Y%m%dT%H%M%SZ)
OUT_DIR="$PKG/logs/census_${TS}"
mkdir -p "$OUT_DIR"
xrdfs cmseos.fnal.gov ls "$EOS_ROOT/train" > "$OUT_DIR/train_listing.txt"
xrdfs cmseos.fnal.gov ls "$EOS_ROOT/val"   > "$OUT_DIR/val_listing.txt"
python3 "$CENSUS_SCRIPT" "$OUT_DIR/train_listing.txt" "$OUT_DIR/val_listing.txt" \
    "$CKPT_SHA256" "$OUT_DIR/census.json"
```

**Expected output:** `TRAIN: 880/880 ... gate_pass=True`, `VAL: 220/220 ...
gate_pass=True`, `CHECKPOINT: pass=True`, `OVERALL_CENSUS_PASS = True`,
exit code 0.

**STOP condition:** if `OVERALL_CENSUS_PASS` is not exactly `True` (any
`h5_only`/`receipt_only`/`missing_both`/`duplicate_or_collision_ids`/
`unexpected_ids_outside_expected_range` nonzero, or `complete_pairs` <
expected, or checkpoint hash mismatch) — **stop, do not proceed to Step
2**, wait and re-run Step 1 later. This step is read-only and safe to
re-run unlimited times.

### Step 2 — Routine postflight gate

```bash
bash "$PKG/scripts/postflight/run_postflight_gate.sh"
```

**Expected output:** a fresh `$PKG/logs/postflight_<UTC>/` directory
containing `train_listing.txt`, `val_listing.txt`, `census.json`,
`aggregate_audit.json`, `manifest_cross_check.json`, `checksum_train.json`,
`checksum_val.json`, `GATE_RESULT.json`, `GATE_RESULT.txt`. Per **G1
above, this is not fast** — budget 30–90+ minutes; Step 2/4 of the gate
(`aggregate_embedding_audit.py`) always re-downloads and re-verifies all
1,100 shards' content regardless of this being the "routine" call.

**STOP condition:** proceed only if the command's own exit code is `0`
**and** `GATE_RESULT.json` contains `"OVERALL_POSTFLIGHT_PASS": true`.
Anything else (including 3-of-4 sub-checks passing) — stop, investigate,
do not join. Record this run's `$PKG/logs/postflight_<UTC>/` path.

### Step 3 — Full checksum postflight (immediately before the join, not routinely)

```bash
CHECKSUM_SAMPLE_N=full bash "$PKG/scripts/postflight/run_postflight_gate.sh"
```

**Expected output:** same artifact set as Step 2, but `checksum_train.json`
/`checksum_val.json` cover **every** shard (not a 20-sample spot check) —
same real cost as Step 2 either way (G1). This is the rigorous,
immediately-pre-freeze run; use **this** run's output directory (not
Step 2's) as the gate input to Step 4.

**STOP condition:** identical to Step 2 — `OVERALL_POSTFLIGHT_PASS: true`
and exit 0, or stop.

```bash
GATE_DIR="<the $PKG/logs/postflight_<UTC>/ path this Step 3 run printed>"
```

### Step 4 — Identity-safe join

```bash
JOIN_TS=$(date -u +%Y%m%dT%H%M%SZ)
bash "$PKG/scripts/join/run_full_join.sh" "$GATE_DIR" "$PKG/logs/join_${JOIN_TS}"
JOIN_DIR="$PKG/logs/join_${JOIN_TS}"
```

**Expected output:** `$JOIN_DIR/joined_train.h5` (`joined_features` shape
`(2000000,10,135)` float32 = 7 native + 128 ParT, plus `mask`,
`part_coverage_mask`), `$JOIN_DIR/joined_val.h5` (`(400000,10,135)`),
`JOIN_RECEIPT_train.json`, `JOIN_RECEIPT_val.json`,
`JOIN_RECEIPT_COMBINED.json`. Refuses to run at all if `$GATE_DIR` didn't
pass (hard `exit 11`) or if either native H5's SHA256 has drifted (hard
`exit 13`). Needs ~10 GB (local shard sync) + ~13 GB (joined outputs) of
scratch disk; not RAM-heavy (shard-streaming design, proven at 5/20/50/150
bounded scale already, and now also proven correct end-to-end on a
synthetic population this session, F7–F8).

**STOP condition:** proceed only if the command's own exit code is `0`
**and** `JOIN_RECEIPT_COMBINED.json` has `"OVERALL_JOIN_PASS": true`. If
any per-split `OVERALL_JOIN_VERIFICATION_PASS` is false — stop; do not
compute statistics or train on a partially- or incorrectly-joined file.

### Step 5 — TRAIN-ONLY embedding statistics (never val)

```bash
python3 "$PKG/scripts/preprocess/compute_train_embedding_stats.py" \
    --joined-train-h5 "$JOIN_DIR/joined_train.h5" \
    --out "$JOIN_DIR/PART_TRAIN_STATS_FROZEN.json"
```

**Expected output:** `n_train_events=2000000 n_real_jets_train=9044239
all_finite=True`, a `data_driven_gap_threshold_recommended_for_review`,
and an explicit scan across `{1e-1,1e-2,5e-3,1e-3,1e-4,1e-5}`. Peak RSS
~4.6–6 GB (see resource note above — pick a node with headroom).

**STOP condition — do not skip this judgment call:** read the printed
threshold scan and the data-driven gap threshold. **Only if** the full
2,000,000-train-event population's gap sits close to the prior
provisional `1e-3`/20-active-dim result should you reuse `1e-3` in Step
6 — if the full-population gap lands somewhere materially different,
justify and use *that* value instead. Silently copying `1e-3` without
looking at this file's output is exactly the "silently canonize" failure
mode this step exists to prevent. Record your chosen threshold as
`ACTIVE_THRESHOLD` below.

```bash
ACTIVE_THRESHOLD="<value you justified from Step 5's scan, e.g. 1e-3>"
```

### Step 6 — Build both model-input variants (train + val, both from the SAME frozen train stats)

```bash
YAML_TEMPLATE=/uscms_data/d3/iturkmen/hh4b_delphes/track_b_phase4_preflight_20260812/phase4S_official_spanet_integration_canary_20260814_v1/event_config/trackb_hh4b.yaml
NATIVE_H5_TRAIN=/uscms_data/d3/iturkmen/hh4b_delphes/track_b_phase4_preflight_20260812/phase4AF_spanet_partial_events_population_correction_20260819_v1/production_hdf5_build_v1/work/production_2M_train.h5
NATIVE_H5_VAL=/uscms_data/d3/iturkmen/hh4b_delphes/track_b_phase4_preflight_20260812/phase4AF_spanet_partial_events_population_correction_20260819_v1/production_hdf5_build_v1/work/production_2M_val.h5

# Variant A: primary resource-aware active-dimension input
for SPLIT in train val; do
  NATIVE_VAR="NATIVE_H5_${SPLIT^^}"
  python3 "$PKG/scripts/preprocess/build_augmented_input.py" \
      --joined-h5 "$JOIN_DIR/joined_${SPLIT}.h5" \
      --native-h5 "${!NATIVE_VAR}" \
      --split "${SPLIT}" \
      --train-stats "$JOIN_DIR/PART_TRAIN_STATS_FROZEN.json" \
      --variant active --active-threshold "$ACTIVE_THRESHOLD" \
      --event-yaml-template "$YAML_TEMPLATE" \
      --out-h5 "$JOIN_DIR/spa2m_part_active_${SPLIT}.h5" \
      --out-yaml "$JOIN_DIR/part_augmented_active_hh4b.yaml" \
      --out-receipt "$JOIN_DIR/BUILD_RECEIPT_active_${SPLIT}.json"
done

# Variant B: all-128 sensitivity/control input
for SPLIT in train val; do
  NATIVE_VAR="NATIVE_H5_${SPLIT^^}"
  python3 "$PKG/scripts/preprocess/build_augmented_input.py" \
      --joined-h5 "$JOIN_DIR/joined_${SPLIT}.h5" \
      --native-h5 "${!NATIVE_VAR}" \
      --split "${SPLIT}" \
      --train-stats "$JOIN_DIR/PART_TRAIN_STATS_FROZEN.json" \
      --variant all128 --eps 1e-3 \
      --event-yaml-template "$YAML_TEMPLATE" \
      --out-h5 "$JOIN_DIR/spa2m_part_all128_${SPLIT}.h5" \
      --out-yaml "$JOIN_DIR/part_augmented_all128_hh4b.yaml" \
      --out-receipt "$JOIN_DIR/BUILD_RECEIPT_all128_${SPLIT}.json"
done
```

**Expected output:** 4 H5 files (`spa2m_part_{active,all128}_{train,val}.h5`),
2 event YAMLs, 4 build receipts. `active` variant width = `7 +
n_retained` (verify `n_retained_part_dims` in the receipt matches what
Step 5 justified); `all128` width = `135` exactly. ~23 GB additional
scratch disk across all 4 files.

**STOP condition:** open each `BUILD_RECEIPT_*.json` and confirm
`"all_finite": true` and the expected `n_retained_part_dims` **before**
launching any training. A non-finite value or an unexpected dimension
count here means stop and re-derive Step 5, do not train on it.

### Step 7 — Primary SPA2M+ParT training (active variant)

**Manual gate, by design:** `launch_spa2m_part.py` hardcodes
`AUTHORIZED_TO_RUN = False`. A human must open that file and change it to
`True` after personally reviewing this packet and the Step 6 receipts —
this session does not and will not do that edit, and this command will
refuse with exit code 2 until it's done. Run only on an **EAF GPU
session**, never LPC.

```bash
bash "$PKG/scripts/train/run_spa2m_part_training.sh" active \
    "$JOIN_DIR/spa2m_part_active_train.h5" \
    "$JOIN_DIR/spa2m_part_active_val.h5" \
    "$JOIN_DIR/part_augmented_active_hh4b.yaml" \
    "$JOIN_DIR/BUILD_RECEIPT_active_train.json" \
    spanet_2M_part_active_seed0
```

**Expected output:** `$PKG/scripts/train/work/training_spanet_2M_part_active_seed0_result.json`
plus TensorBoard logs and primary/secondary/`last` checkpoints under
`$PKG/scripts/train/spanet_2M_part_active_seed0/`. Expect wall time on
the order of the native run's measured `total_fit_wall_s≈8577s` (~2.4h)
at the same batch size/epoch budget, modulo the wider input.

**STOP condition:** confirm `exit_status: "COMPLETED"` (not `"FAILED"`),
`n_train_events: 2000000`, `n_val_events: 400000`,
`len(epoch_history) == 50`, and `primary_checkpoint_sha256` is non-null
before treating this as a usable checkpoint for Step 9.

### Step 8 — All-128 sensitivity/control training

Same manual `AUTHORIZED_TO_RUN` gate as Step 7 (already flipped once
Step 7 ran).

```bash
bash "$PKG/scripts/train/run_spa2m_part_training.sh" all128 \
    "$JOIN_DIR/spa2m_part_all128_train.h5" \
    "$JOIN_DIR/spa2m_part_all128_val.h5" \
    "$JOIN_DIR/part_augmented_all128_hh4b.yaml" \
    "$JOIN_DIR/BUILD_RECEIPT_all128_train.json" \
    spanet_2M_part_all128_seed0
```

**Expected output / STOP condition:** identical structure and checks to
Step 7, under `run_name=spanet_2M_part_all128_seed0`.

### Step 9 — Matched evaluation (**BLOCKED on G5 above — read before attempting**)

`evaluate_matched_2m.py` is ready and correct (F12; its own docstring is
explicit that it does not perform inference). It requires two pre-built
`.npz` files in its documented schema
(`event_id, process, score, assignment_correct, assignment_defined,
higgs_mass_1, higgs_mass_2, n_jets`, + optional ParT-norm/pT/eta columns)
for the **identical** val cohort, one for the existing CONTROL checkpoint
(`configs/FROZEN_FACTS.json:native_spa2m_control.checkpoint_path`,
verified present and hash-correct this session) and one for the Step 7
TEST checkpoint. **No script producing that schema exists yet anywhere in
the project** (G5) — write one first, adapting
`.../spanet_2M_seed0_classification_evaluation_v1/evaluate_classification.py`'s
CPU-forward-pass pattern to also extract predicted jet-pair assignments,
`higgs_mass_{1,2}`, and `event_id=native_hdf5_row_index`, run it once
against CONTROL and once against TEST, then:

```bash
cd "$PKG/scripts/eval"
python3 evaluate_matched_2m.py \
    --control-npz <path/to/control_2m_native_eval.npz> \
    --test-npz    <path/to/test_2m_part_active_eval.npz> \
    --control-training-result /uscms_data/d3/iturkmen/hh4b_delphes/track_b_phase4_preflight_20260812/phase4AF_spanet_partial_events_population_correction_20260819_v1/spanet_2M_seed0_production_training_v1/work/training_2M_seed0_result.json \
    --test-training-result "$PKG/scripts/train/work/training_spanet_2M_part_active_seed0_result.json" \
    --build-receipt "$JOIN_DIR/BUILD_RECEIPT_active_val.json" \
    --out-json "$JOIN_DIR/EVALUATION_RESULT_active.json"
```

**Expected output:** `EVALUATION_RESULT_active.json` with AUC (all-bg/QCD/
ttbar), paired-bootstrap ΔAUC, rejection-at-fixed-efficiency, McNemar
reconstruction comparison, jet-multiplicity-stratified AUC, resource
comparison. Every result is labeled
`MATCHED_DEVELOPMENT_VALIDATION_RESULT_NOT_FINAL_TEST_PERFORMANCE` by the
script itself — repeat this labeling in any downstream write-up (per
`FROZEN_FACTS.json:data_policy.labeling_rule`).

**STOP condition:** the script itself hard-fails (exit 2/3) if the
control/test `event_id` or `process` arrays don't match after sorting —
treat that as a real identity bug, not a warning, and do not report
anything from that run. If G3 (scipy) is unresolved, note in the writeup
that sparse-tail rejection CIs used the normal-approximation fallback,
not the exact Garwood method.

---

## Final verdict

**`READY_TO_EXECUTE_AFTER_880_220`** for Steps 1–8 (postproduction census
→ postflight → identity-safe join → train-only statistics → both
augmented-input variants → both training launches). Nothing in that path
depends on anything except production reaching 880/880 train + 220/220
val — every script involved was read for actual executability against the
real schema, every hardcoded hash/path was independently re-verified
against real files this session, and every step that could be dry-run
without touching production or launching training was dry-run for real
(not just reasoned about) and passed, including an exact ground-truth
match on the active-dimension detection logic.

**One concrete blocker exists for Step 9 only** (G5): the CONTROL/TEST
`.npz` inference/scoring script `evaluate_matched_2m.py` depends on does
not exist yet anywhere in the project and must be written — this does not
block starting or completing Steps 1–8, and there is natural wall-clock
time to write it while Step 7/8 training runs.
